import pandas as pd
import numpy as np
import rasterio
from rasterio.transform import from_bounds
import os


def generate_sample_weather_data(output_path='data/sample_weather.csv'):
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=365, freq='D')
    rainfall = np.random.exponential(5, 365)
    temperature = 20 + 10 * \
        np.sin(2 * np.pi * np.arange(365) / 365) + np.random.normal(0, 5, 365)
    df = pd.DataFrame({'time': dates, 'rainfall': rainfall,
                      'temperature': temperature})
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path


def generate_sample_flood_data(output_path='data/sample_flood.csv'):
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=365, freq='D')
    flood_occurred = np.random.choice([0, 1], 365, p=[0.9, 0.1])
    df = pd.DataFrame({'time': dates, 'flood_occurred': flood_occurred})
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path


def generate_sample_terrain_data(output_path='data/sample_terrain.tif'):
    np.random.seed(42)
    x = np.linspace(71.4, 71.5, 100)
    y = np.linspace(51.1, 51.2, 100)
    X, Y = np.meshgrid(x, y)
    elevation = 300 + 50 * \
        np.exp(-((X - 71.45)**2 + (Y - 51.15)**2) / 0.01) + \
        np.random.normal(0, 5, (100, 100))
    transform = from_bounds(71.4, 51.1, 71.5, 51.2, 100, 100)
    profile = {'driver': 'GTiff', 'dtype': 'float32', 'width': 100,
               'height': 100, 'count': 1, 'crs': 'EPSG:4326', 'transform': transform}
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(elevation.astype('float32'), 1)
    return output_path


def generate_sample_flood_mask(output_path='data/sample_flood_mask.tif'):
    np.random.seed(42)
    x = np.linspace(71.4, 71.5, 100)
    y = np.linspace(51.1, 51.2, 100)
    X, Y = np.meshgrid(x, y)
    base_flood = np.exp(-((X - 71.42)**2 + (Y - 51.14)**2) / 0.005)
    random_flood = np.random.random((100, 100)) > 0.85
    flood_mask = ((base_flood > 0.3) | random_flood).astype(np.float32)
    transform = from_bounds(71.4, 51.1, 71.5, 51.2, 100, 100)
    profile = {'driver': 'GTiff', 'dtype': 'float32', 'width': 100,
               'height': 100, 'count': 1, 'crs': 'EPSG:4326', 'transform': transform}
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(flood_mask, 1)
    return output_path


if __name__ == "__main__":
    generate_sample_weather_data()
    generate_sample_flood_data()
    generate_sample_terrain_data()
    generate_sample_flood_mask()
    print("Sample data generated!")
