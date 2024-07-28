import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from scripts.data_ingestion import load_weather_data, load_terrain_data, merge_data
from models.model import train_gbm, UNet


def retrain_gbm(weather_path, flood_path, model_path='models/gbm_model.pkl'):
    weather_df = load_weather_data(weather_path)
    flood_df = pd.read_csv(flood_path)
    X = weather_df[['rainfall', 'temperature']].values
    y = flood_df['flood_occurred'].values
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42)
    gbm = train_gbm(X_train, y_train)
    y_pred = gbm.predict(X_val)
    from sklearn.metrics import accuracy_score, classification_report
    acc = accuracy_score(y_val, y_pred)
    print(f"GBM Validation Accuracy: {acc:.2f}")
    print("Classification Report:\n", classification_report(y_val, y_pred))
    joblib.dump(gbm, model_path)
    print("GBM model retrained and saved.")


def retrain_unet(terrain_path, flood_mask_path, model_path='models/unet_model.pth'):
    terrain, _ = load_terrain_data(terrain_path)
    flood_mask, _ = load_terrain_data(flood_mask_path)
    terrain = (terrain - terrain.mean()) / (terrain.std() + 1e-8)
    patches_X, patches_y = [], []
    patch_size = 32
    for i in range(0, terrain.shape[0] - patch_size, patch_size//2):
        for j in range(0, terrain.shape[1] - patch_size, patch_size//2):
            patch_terrain = terrain[i:i+patch_size, j:j+patch_size]
            patch_mask = flood_mask[i:i+patch_size, j:j+patch_size]
            patches_X.append(patch_terrain)
            patches_y.append(patch_mask)
    X = torch.stack([torch.tensor(p, dtype=torch.float32).unsqueeze(0)
                    for p in patches_X])
    y = torch.stack([torch.tensor(p, dtype=torch.float32).unsqueeze(0)
                    for p in patches_y])
    n_train = int(0.8 * len(X))
    X_train, X_val = X[:n_train], X[n_train:]
    y_train, y_val = y[:n_train], y[n_train:]
    model = UNet(in_channels=1, out_channels=1)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()
    best_val_acc = 0
    patience = 3
    patience_counter = 0
    for epoch in range(20):
        model.train()
        train_loss = 0
        for i in range(len(X_train)):
            optimizer.zero_grad()
            output = model(X_train[i:i+1])
            loss = criterion(output, y_train[i:i+1])
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for i in range(len(X_val)):
                val_output = model(X_val[i:i+1])
                val_loss += criterion(val_output, y_val[i:i+1]).item()
                val_pred = (torch.sigmoid(val_output) > 0.5).float()
                correct += (val_pred == y_val[i:i+1]).sum().item()
                total += y_val[i:i+1].numel()
        val_acc = correct / total
        print(f"Epoch {epoch+1}, Train Loss: {train_loss/len(X_train):.4f}, Val Loss: {val_loss/len(X_val):.4f}, Val Acc: {val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), model_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    print(
        f"U-Net training completed. Best validation accuracy: {best_val_acc:.4f}")
    print("U-Net model retrained and saved.")


if __name__ == "__main__":
    retrain_gbm('data/weather.csv', 'data/flood.csv')
    retrain_unet('data/terrain.tif', 'data/flood_mask.tif')
