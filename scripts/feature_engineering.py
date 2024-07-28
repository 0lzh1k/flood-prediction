import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer


def handle_outliers(df, columns, method='iqr'):
    df_clean = df.copy()
    for col in columns:
        if method == 'iqr':
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df_clean[col] = df_clean[col].clip(lower_bound, upper_bound)
    return df_clean


def normalize_features(df, columns):
    scaler = StandardScaler()
    df_norm = df.copy()
    df_norm[columns] = scaler.fit_transform(df[columns])
    return df_norm, scaler


def encode_categorical(df, categorical_columns):
    df_encoded = df.copy()
    encoders = {}
    for col in categorical_columns:
        encoder = LabelEncoder()
        df_encoded[col] = encoder.fit_transform(df[col].astype(str))
        encoders[col] = encoder
    return df_encoded, encoders


def aggregate_time_series(df, time_col='time', agg_window='1D'):
    df[time_col] = pd.to_datetime(df[time_col])
    df_agg = df.set_index(time_col).resample(agg_window).agg(
        {'rainfall': 'sum', 'temperature': 'mean'}).reset_index()
    return df_agg


def handle_missing_values(df, strategy='mean'):
    imputer = SimpleImputer(strategy=strategy)
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    df_imputed = df.copy()
    df_imputed[numerical_cols] = imputer.fit_transform(df[numerical_cols])
    return df_imputed, imputer


def preprocess_features(weather_df, flood_df, terrain_features=None):
    weather_df, _ = handle_missing_values(weather_df)
    flood_df, _ = handle_missing_values(flood_df)
    weather_df = handle_outliers(weather_df, ['rainfall', 'temperature'])
    weather_df = aggregate_time_series(weather_df)
    merged_df = weather_df.merge(flood_df, on='time', how='left')
    if terrain_features:
        for key, value in terrain_features.items():
            merged_df[key] = value
    feature_cols = ['rainfall', 'temperature']
    merged_df, scaler = normalize_features(merged_df, feature_cols)
    return merged_df, scaler


if __name__ == "__main__":
    print("Feature engineering module loaded.")
