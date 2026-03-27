"""Data cleaning and validation for raw ingested data."""

from __future__ import annotations

import logging

import pandas as pd

from src.config import DATA_DIR

logger = logging.getLogger(__name__)


def clean_weather(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and validate weather data."""
    df = df.copy()

    # Drop fully-null rows
    non_key_cols = [c for c in df.columns if c not in ("lake_key", "datetime")]
    df.dropna(how="all", subset=non_key_cols, inplace=True)

    # Cap physically impossible values
    if "temperature_2m" in df.columns:
        df.loc[df["temperature_2m"] < -50, "temperature_2m"] = pd.NA
        df.loc[df["temperature_2m"] > 55, "temperature_2m"] = pd.NA

    if "surface_pressure" in df.columns:
        df.loc[df["surface_pressure"] < 870, "surface_pressure"] = pd.NA
        df.loc[df["surface_pressure"] > 1085, "surface_pressure"] = pd.NA

    if "relative_humidity_2m" in df.columns:
        df["relative_humidity_2m"] = df["relative_humidity_2m"].clip(0, 100)

    if "wind_speed_10m" in df.columns:
        df.loc[df["wind_speed_10m"] < 0, "wind_speed_10m"] = pd.NA

    # Interpolate short gaps (up to 6 hours)
    numeric_cols = df.select_dtypes(include="number").columns
    for col in numeric_cols:
        if col != "lake_key":
            df[col] = df[col].interpolate(method="linear", limit=6)

    logger.info(f"Cleaned weather data: {len(df)} rows, {df.isna().sum().sum()} remaining NaNs")
    return df


def clean_water(df: pd.DataFrame) -> pd.DataFrame:
    """Clean water temperature and level data."""
    df = df.copy()

    if "water_temp_c" in df.columns:
        # Water temp bounds: -2°C to 40°C for Indiana freshwater
        df.loc[df["water_temp_c"] < -2, "water_temp_c"] = pd.NA
        df.loc[df["water_temp_c"] > 40, "water_temp_c"] = pd.NA
        df["water_temp_f"] = df["water_temp_c"] * 9 / 5 + 32

    if "gage_height_ft" in df.columns:
        # Remove obvious outliers (negative or >100ft for Indiana lakes)
        df.loc[df["gage_height_ft"] < 0, "gage_height_ft"] = pd.NA
        df.loc[df["gage_height_ft"] > 100, "gage_height_ft"] = pd.NA

    # Forward-fill gaps up to 7 days for water data (changes slowly)
    numeric_cols = df.select_dtypes(include="number").columns
    for col in numeric_cols:
        df[col] = df[col].interpolate(method="linear", limit=7)

    logger.info(f"Cleaned water data: {len(df)} rows")
    return df


def clean_catches(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and validate tournament catch data."""
    df = df.copy()

    # Remove rows with no date or lake
    df.dropna(subset=["date", "lake_key"], inplace=True)

    # Weight validation — Indiana state record largemouth is 14.12 lbs
    # World record is 22.25 lbs. Cap at 16 lbs as reasonable max for Indiana.
    if "weight_lbs" in df.columns:
        df.loc[df["weight_lbs"] <= 0, "weight_lbs"] = pd.NA
        df.loc[df["weight_lbs"] > 16, "weight_lbs"] = pd.NA

    # Length validation
    if "length_in" in df.columns:
        df.loc[df["length_in"] <= 0, "length_in"] = pd.NA
        df.loc[df["length_in"] > 30, "length_in"] = pd.NA

    # Ensure date column is datetime
    df["date"] = pd.to_datetime(df["date"])

    logger.info(f"Cleaned catches: {len(df)} rows, {df['lake_key'].nunique()} lakes")
    return df


def load_and_clean_all() -> dict[str, pd.DataFrame]:
    """Load all raw data and return cleaned DataFrames."""
    result = {}

    # Weather
    weather_dir = DATA_DIR / "raw" / "weather"
    if weather_dir.exists():
        dfs = []
        for f in weather_dir.glob("*.parquet"):
            df = pd.read_parquet(f)
            # Ensure datetime is a column, not just the index
            if df.index.name == "datetime":
                df = df.reset_index()
            dfs.append(df)
        if dfs:
            result["weather"] = clean_weather(pd.concat(dfs, ignore_index=True))

    # Water
    water_dir = DATA_DIR / "raw" / "water"
    if water_dir.exists():
        dfs = []
        for f in water_dir.glob("*.parquet"):
            dfs.append(pd.read_parquet(f))
        if dfs:
            result["water"] = clean_water(pd.concat(dfs, ignore_index=True))

    # Astro (no cleaning needed — computed, not measured)
    astro_dir = DATA_DIR / "raw" / "astro"
    if astro_dir.exists():
        dfs = []
        for f in astro_dir.glob("*.parquet"):
            dfs.append(pd.read_parquet(f))
        if dfs:
            result["astro"] = pd.concat(dfs, ignore_index=True)

    # Catches
    catches_path = DATA_DIR / "processed" / "catches.parquet"
    if catches_path.exists():
        result["catches"] = clean_catches(pd.read_parquet(catches_path))

    return result
