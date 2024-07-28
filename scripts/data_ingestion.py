import pandas as pd
import numpy as np
import rasterio


def load_weather_data(csv_path):
    df = pd.read_csv(csv_path)
    required_cols = ["time", "rainfall", "temperature"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")
    return df


def load_terrain_data(tif_path):
    with rasterio.open(tif_path) as src:
        terrain = src.read(1)
        profile = src.profile
    return terrain, profile


def merge_data(weather_df, flood_df, terrain_features):
    merged = weather_df.merge(flood_df, on="time", how="left")
    for key, value in terrain_features.items():
        merged[key] = value
    return merged
