"""Seasonal and temporal pattern analysis for trophy bass catches."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def seasonal_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Compute catch rates and trophy rates by month."""
    if "trophy_caught" not in df.columns:
        return pd.DataFrame()

    # Get daily-level data (avoid hourly duplication)
    daily = df.groupby(["date", "lake_key"]).agg(
        catch_count=("catch_count", "first"),
        trophy_count=("trophy_count", "first"),
    ).reset_index()

    daily["date"] = pd.to_datetime(daily["date"])
    daily["month"] = daily["date"].dt.month
    daily["month_name"] = daily["date"].dt.strftime("%B")

    monthly = daily.groupby(["month", "month_name"]).agg(
        total_days=("date", "count"),
        total_catches=("catch_count", "sum"),
        total_trophies=("trophy_count", "sum"),
    ).reset_index()

    monthly["catches_per_day"] = monthly["total_catches"] / monthly["total_days"]
    monthly["trophy_per_day"] = monthly["total_trophies"] / monthly["total_days"]
    monthly["trophy_rate"] = np.where(
        monthly["total_catches"] > 0,
        monthly["total_trophies"] / monthly["total_catches"],
        0,
    )

    return monthly.sort_values("month")


def hourly_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Compute catch rates by hour of day."""
    if "hour" not in df.columns or "catch_count" not in df.columns:
        return pd.DataFrame()

    hourly = df.groupby("hour").agg(
        total_rows=("catch_count", "count"),
        total_catches=("catch_count", "sum"),
        total_trophies=("trophy_count", "sum"),
    ).reset_index()

    hourly["catch_rate"] = hourly["total_catches"] / hourly["total_rows"]
    hourly["trophy_rate"] = hourly["total_trophies"] / hourly["total_rows"]

    return hourly


def lake_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """Compare catch statistics across lakes."""
    if "lake_key" not in df.columns:
        return pd.DataFrame()

    daily = df.groupby(["date", "lake_key"]).agg(
        catch_count=("catch_count", "first"),
        trophy_count=("trophy_count", "first"),
    ).reset_index()

    comparison = daily.groupby("lake_key").agg(
        total_days=("date", "nunique"),
        total_catches=("catch_count", "sum"),
        total_trophies=("trophy_count", "sum"),
    ).reset_index()

    comparison["catches_per_day"] = comparison["total_catches"] / comparison["total_days"]
    comparison["trophy_rate"] = np.where(
        comparison["total_catches"] > 0,
        comparison["total_trophies"] / comparison["total_catches"],
        0,
    )

    return comparison.sort_values("total_trophies", ascending=False)


def spawn_phase_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze catch rates by bass spawn phase."""
    if "spawn_phase" not in df.columns:
        return pd.DataFrame()

    wt_col = "water_temp_estimated" if "water_temp_estimated" in df.columns else "temperature_2m"
    phase_stats = df.groupby("spawn_phase").agg(
        total_hours=("catch_count", "count"),
        total_catches=("catch_count", "sum"),
        total_trophies=("trophy_count", "sum"),
        avg_temp=("temperature_2m", "mean"),
        avg_water_temp=(wt_col, "mean"),
    ).reset_index()

    phase_stats["catch_rate"] = phase_stats["total_catches"] / phase_stats["total_hours"]
    phase_stats["trophy_rate"] = phase_stats["total_trophies"] / phase_stats["total_hours"]

    phase_order = ["WINTER", "PRE_SPAWN", "SPAWN", "POST_SPAWN", "SUMMER", "FALL", "TURNOVER"]
    phase_stats["phase_order"] = phase_stats["spawn_phase"].map(
        {p: i for i, p in enumerate(phase_order)}
    )
    return phase_stats.sort_values("phase_order")


def pressure_pattern_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze catch rates by pressure trend."""
    if "pressure_trend_class" not in df.columns:
        return pd.DataFrame()

    stats = df.groupby("pressure_trend_class").agg(
        total_hours=("catch_count", "count"),
        total_catches=("catch_count", "sum"),
        total_trophies=("trophy_count", "sum"),
        avg_pressure=("pressure_msl", "mean"),
    ).reset_index()

    stats["catch_rate"] = stats["total_catches"] / stats["total_hours"]
    stats["trophy_rate"] = stats["total_trophies"] / stats["total_hours"]

    return stats


def front_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze catch rates relative to weather fronts."""
    if "front_type" not in df.columns:
        return pd.DataFrame()

    stats = df.groupby("front_type").agg(
        total_hours=("catch_count", "count"),
        total_catches=("catch_count", "sum"),
        total_trophies=("trophy_count", "sum"),
    ).reset_index()

    stats["catch_rate"] = stats["total_catches"] / stats["total_hours"]
    stats["trophy_rate"] = stats["total_trophies"] / stats["total_hours"]

    return stats
