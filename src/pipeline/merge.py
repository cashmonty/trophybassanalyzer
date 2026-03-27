"""Merge all data sources into a single analysis-ready dataset."""

from __future__ import annotations

import logging

import pandas as pd

from src.config import DATA_DIR
from src.pipeline.clean import load_and_clean_all
from src.pipeline.features import engineer_all_features

logger = logging.getLogger(__name__)


def merge_conditions(weather: pd.DataFrame, water: pd.DataFrame | None,
                     astro: pd.DataFrame | None) -> pd.DataFrame:
    """Merge weather, water, and astro data into a unified conditions DataFrame."""
    # Start with weather as the base (hourly)
    df = weather.copy()

    # Ensure datetime column exists (weather may use it as index)
    if "datetime" not in df.columns:
        if df.index.name == "datetime":
            df = df.reset_index()
        elif "time" in df.columns:
            df = df.rename(columns={"time": "datetime"})
        else:
            # Try to use the index regardless of its name
            df = df.reset_index()
            if df.columns[0] != "datetime":
                df = df.rename(columns={df.columns[0]: "datetime"})
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["date"] = df["datetime"].dt.date

    # Merge water data (daily granularity)
    if water is not None and len(water) > 0:
        water = water.copy()
        water["date"] = pd.to_datetime(water["date"]).dt.date
        water_cols = ["date", "lake_key", "water_temp_c", "water_temp_f",
                      "gage_height_ft", "discharge_cfs"]
        water_cols = [c for c in water_cols if c in water.columns]
        df = df.merge(water[water_cols], on=["date", "lake_key"], how="left")

    # Merge astro data (daily granularity)
    if astro is not None and len(astro) > 0:
        astro = astro.copy()
        astro["date"] = pd.to_datetime(astro["date"]).dt.date
        df = df.merge(
            astro, on=["date", "lake_key"], how="left", suffixes=("", "_astro")
        )

    logger.info(f"Merged conditions: {len(df)} rows, {len(df.columns)} columns")
    return df


def merge_catches_with_conditions(conditions: pd.DataFrame,
                                  catches: pd.DataFrame) -> pd.DataFrame:
    """Create the final analysis dataset by joining catches with conditions.

    For each catch event, find the closest hourly conditions record.
    Also creates a binary 'trophy_caught' column for all condition rows.
    """
    conditions = conditions.copy()
    catches = catches.copy()

    conditions["datetime"] = pd.to_datetime(conditions["datetime"])
    catches["date"] = pd.to_datetime(catches["date"])

    # Daily aggregation of catches per lake
    agg_dict = {
        "catch_count": ("weight_lbs", "count"),
        "max_weight": ("weight_lbs", "max"),
        "avg_weight": ("weight_lbs", "mean"),
        "trophy_count": ("is_trophy", "sum"),
    }
    if "is_super_trophy" in catches.columns:
        agg_dict["super_trophy_count"] = ("is_super_trophy", "sum")

    daily_catches = catches.groupby(["date", "lake_key"]).agg(**agg_dict).reset_index()

    # Normalize both sides to date objects for join
    daily_catches["date"] = pd.to_datetime(daily_catches["date"]).dt.date
    conditions["date"] = pd.to_datetime(conditions["datetime"]).dt.date

    # Merge daily catch stats onto conditions
    df = conditions.merge(daily_catches, on=["date", "lake_key"], how="left")

    # Fill days with no catches
    df["catch_count"] = df["catch_count"].fillna(0).astype(int)
    df["trophy_count"] = df["trophy_count"].fillna(0).astype(int)
    df["trophy_caught"] = (df["trophy_count"] > 0).astype(int)

    logger.info(
        f"Final dataset: {len(df)} rows, "
        f"{df['trophy_caught'].sum()} trophy-hours, "
        f"{df['catch_count'].sum()} total catches"
    )
    return df


def build_merged_dataset() -> pd.DataFrame:
    """Full pipeline: load, clean, merge, engineer features, save."""
    logger.info("Loading and cleaning all data sources...")
    data = load_and_clean_all()

    if "weather" not in data:
        raise RuntimeError("No weather data found. Run data ingestion first.")

    # Merge conditions
    conditions = merge_conditions(
        weather=data["weather"],
        water=data.get("water"),
        astro=data.get("astro"),
    )

    # Engineer features
    conditions = engineer_all_features(conditions)

    # Merge with catches if available
    if "catches" in data:
        df = merge_catches_with_conditions(conditions, data["catches"])
    else:
        logger.warning("No catch data found — building conditions-only dataset")
        df = conditions
        df["catch_count"] = 0
        df["trophy_count"] = 0
        df["trophy_caught"] = 0

    # Save
    output_path = DATA_DIR / "processed" / "merged.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.info(f"Saved merged dataset to {output_path} ({len(df)} rows)")

    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    df = build_merged_dataset()
    print(f"\nMerged dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nSample:\n{df.head()}")
