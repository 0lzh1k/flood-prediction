import pandas as pd
import numpy as np
from pathlib import Path

from scripts.generate_sample_data import (
    generate_sample_weather_data,
    generate_sample_flood_data,
    generate_sample_terrain_data,
    generate_sample_flood_mask,
)
from scripts.data_ingestion import load_weather_data, load_terrain_data, merge_data


def test_load_weather_data_and_schema(tmp_path: Path):
    weather_path = tmp_path / "weather.csv"
    generate_sample_weather_data(str(weather_path))
    df = load_weather_data(str(weather_path))
    assert set(["time", "rainfall", "temperature"]).issubset(df.columns)
    assert len(df) == 365


def test_load_terrain_and_mask(tmp_path: Path):
    terrain_path = tmp_path / "terrain.tif"
    mask_path = tmp_path / "mask.tif"
    generate_sample_terrain_data(str(terrain_path))
    generate_sample_flood_mask(str(mask_path))

    terrain, t_profile = load_terrain_data(str(terrain_path))
    mask, m_profile = load_terrain_data(str(mask_path))

    assert terrain.shape == (100, 100)
    assert mask.shape == (100, 100)
    for prof in (t_profile, m_profile):
        assert "transform" in prof and "crs" in prof and "width" in prof and "height" in prof

    assert float(mask.min()) >= 0.0 and float(mask.max()) <= 1.0


def test_merge_data_adds_features(tmp_path: Path):
    # Minimal weather/flood with matching time keys
    dates = pd.date_range("2023-01-01", periods=5, freq="D")
    weather_df = pd.DataFrame({"time": dates, "rainfall": np.arange(5), "temperature": np.arange(10, 15)})
    flood_df = pd.DataFrame({"time": dates, "flood_occurred": [0, 1, 0, 0, 1]})

    merged = merge_data(weather_df, flood_df, {"elev_mean": 250.0, "slope": 0.02})

    assert len(merged) == len(weather_df)  # left join by weather_df
    assert set(["flood_occurred", "elev_mean", "slope"]).issubset(merged.columns)
