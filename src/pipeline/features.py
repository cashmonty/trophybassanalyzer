"""Feature engineering for trophy bass prediction."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def add_pressure_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add barometric pressure trend features."""
    if "pressure_msl" not in df.columns:
        return df

    df = df.sort_values(["lake_key", "datetime"])

    for hours in [3, 6]:
        col = f"pressure_trend_{hours}h"
        df[col] = df.groupby("lake_key")["pressure_msl"].diff(periods=hours)

    # Classify pressure trend
    def classify_trend(val):
        if pd.isna(val):
            return "unknown"
        if val > 1.5:
            return "rising"
        elif val < -1.5:
            return "falling"
        return "stable"

    df["pressure_trend_class"] = df["pressure_trend_3h"].apply(classify_trend)

    return df


def add_front_detection(df: pd.DataFrame) -> pd.DataFrame:
    """Detect weather fronts from pressure, wind, and cloud changes."""
    if not {"pressure_msl", "wind_speed_10m", "wind_direction_10m"}.issubset(df.columns):
        return df

    df = df.sort_values(["lake_key", "datetime"])

    # 6-hour pressure change
    p_change = df.groupby("lake_key")["pressure_msl"].diff(6)
    # Wind direction change
    wd_change = df.groupby("lake_key")["wind_direction_10m"].diff(6).abs()
    wd_change = wd_change.where(wd_change <= 180, 360 - wd_change)

    conditions = [
        (p_change < -3) & (wd_change > 30),  # pre-frontal
        (p_change > 3) & (wd_change > 30),    # post-frontal
    ]
    choices = ["pre_frontal", "post_frontal"]
    df["front_type"] = np.select(conditions, choices, default="stable")

    # Days since last front
    is_front = df["front_type"] != "stable"
    df["days_since_last_front"] = (
        is_front.groupby(df["lake_key"]).cumsum()
        .groupby(df["lake_key"])
        .transform(lambda x: x.groupby(x).cumcount())
    ) / 24.0  # Convert hours to days

    return df


def add_spawn_phase(df: pd.DataFrame) -> pd.DataFrame:
    """Classify bass seasonal phase based on water/air temperature.

    Indiana largemouth bass phases (based on water temperature in °C):
    - WINTER: below 7°C (44°F) — bass are lethargic, deep, slow metabolism
    - PRE_SPAWN: 7-14°C (44-58°F) spring — staging on secondary points, creek channels
      This is THE trophy window. Big females feed aggressively before bedding.
    - SPAWN: 14-20°C (58-68°F) spring — on beds, shallow flats, hard bottom
    - POST_SPAWN: 20-23°C (68-74°F) early summer — recovering, transitioning
    - SUMMER: above 23°C or 20°C+ July-Aug — deep structure, thermocline, night feeding
    - FALL: 14-23°C (58-74°F) Sept-Nov — shad migrations, aggressive feeding
    - TURNOVER: 7-14°C (44-58°F) fall — lake mixing, tough bite, fish relocating
    """
    temp_col = "water_temp_estimated" if "water_temp_estimated" in df.columns else "temperature_2m"

    if temp_col not in df.columns:
        return df

    # Use 7-day rolling average for phase stability
    temp_smooth = df.groupby("lake_key")[temp_col].transform(
        lambda x: x.rolling(7 * 24, min_periods=24).mean()
    )

    month = pd.to_datetime(df["datetime"]).dt.month

    phase_conditions = [
        temp_smooth < 7,                                                    # WINTER
        (temp_smooth >= 7) & (temp_smooth < 14) & (month <= 6),           # PRE_SPAWN
        (temp_smooth >= 14) & (temp_smooth < 20) & (month <= 6),          # SPAWN
        (temp_smooth >= 20) & (temp_smooth < 23) & (month <= 7),          # POST_SPAWN
        (temp_smooth >= 23) | ((temp_smooth >= 20) & (month.between(7, 8))),  # SUMMER
        (temp_smooth >= 14) & (temp_smooth < 23) & (month >= 9),          # FALL
        (temp_smooth >= 7) & (temp_smooth < 14) & (month >= 9),           # TURNOVER
    ]
    phase_choices = [
        "WINTER", "PRE_SPAWN", "SPAWN", "POST_SPAWN",
        "SUMMER", "FALL", "TURNOVER"
    ]
    df["spawn_phase"] = np.select(phase_conditions, phase_choices, default="UNKNOWN")

    return df


def add_temp_stability(df: pd.DataFrame) -> pd.DataFrame:
    """Add temperature stability metric (3-day rolling std dev)."""
    temp_col = "temperature_2m"
    if temp_col not in df.columns:
        return df

    df["temp_stability_3day"] = df.groupby("lake_key")[temp_col].transform(
        lambda x: x.rolling(72, min_periods=12).std()
    )
    return df


def estimate_water_temp(df: pd.DataFrame) -> pd.DataFrame:
    """Estimate water temperature from air temp using exponential smoothing with lag."""
    if "temperature_2m" not in df.columns:
        return df

    # If we already have real water temp, use it and fill gaps
    if "water_temp_c" in df.columns:
        df["water_temp_estimated"] = df["water_temp_c"]
    else:
        df["water_temp_estimated"] = pd.NA

    # Fill gaps with exponential smoothing of air temp (water lags air by ~24-48h)
    alpha = 0.02  # Slow response — water changes slowly
    for lake in df["lake_key"].unique():
        mask = df["lake_key"] == lake
        air_temp = df.loc[mask, "temperature_2m"].values
        water_est = np.empty_like(air_temp)
        water_est[0] = air_temp[0] if not np.isnan(air_temp[0]) else 10.0

        for i in range(1, len(air_temp)):
            if np.isnan(air_temp[i]):
                water_est[i] = water_est[i - 1]
            else:
                water_est[i] = alpha * air_temp[i] + (1 - alpha) * water_est[i - 1]

        # Only fill where we don't have real data
        lake_water = df.loc[mask, "water_temp_estimated"].values
        nan_mask = pd.isna(lake_water)
        lake_water[nan_mask] = water_est[nan_mask]
        df.loc[mask, "water_temp_estimated"] = lake_water

    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add time-based features."""
    dt = pd.to_datetime(df["datetime"])
    df["hour"] = dt.dt.hour
    df["day_of_week"] = dt.dt.dayofweek
    df["month"] = dt.dt.month
    df["day_of_year"] = dt.dt.dayofyear
    df["is_weekend"] = dt.dt.dayofweek >= 5
    return df


def add_wind_direction_class(df: pd.DataFrame) -> pd.DataFrame:
    """Classify wind direction into angler-relevant categories.

    For Indiana bass fishing:
    - South/SW winds push warm air, activate shallow fish, best pre-spawn trigger
    - North/NW winds = cold front aftermath, fish go deep and lock down
    - East winds historically poor fishing ("wind from the east, fish bite least")
    - West winds = generally stable, decent fishing
    """
    if "wind_direction_10m" not in df.columns:
        return df

    wd = df["wind_direction_10m"]
    conditions = [
        (wd >= 157.5) & (wd < 247.5),   # S, SW, W-ish
        (wd >= 247.5) & (wd < 337.5),   # W, NW, N-ish
        (wd >= 337.5) | (wd < 22.5),    # N
        (wd >= 22.5) & (wd < 67.5),     # NE
        (wd >= 67.5) & (wd < 157.5),    # E, SE
    ]
    choices = ["south_warm", "northwest_cold", "north", "northeast", "east_poor"]
    df["wind_class"] = np.select(conditions, choices, default="variable")

    return df


def add_warming_trend(df: pd.DataFrame) -> pd.DataFrame:
    """Track consecutive warming days — the #1 pre-spawn trophy trigger.

    3+ consecutive days of rising water temp = big bass move shallow.
    Also tracks the 3-day air temp trend for forecast use.
    """
    temp_col = "water_temp_estimated" if "water_temp_estimated" in df.columns else "temperature_2m"
    if temp_col not in df.columns:
        return df

    df = df.sort_values(["lake_key", "datetime"])

    # Daily average temp per lake (for trend detection)
    df["_daily_temp"] = df.groupby(["lake_key", pd.to_datetime(df["datetime"]).dt.date])[temp_col].transform("mean")

    # 3-day temperature change
    df["temp_change_3day"] = df.groupby("lake_key")["_daily_temp"].diff(periods=72)  # 72 hours

    # Is the water warming? (positive trend over 48+ hours)
    df["is_warming_trend"] = (df["temp_change_3day"] > 0.5).astype(int)

    # Consecutive warm days (approximate via rolling sum of warming hours)
    df["warming_streak"] = df.groupby("lake_key")["is_warming_trend"].transform(
        lambda x: x.rolling(120, min_periods=24).sum()  # 5-day window
    )

    df.drop(columns=["_daily_temp"], inplace=True)
    return df


def add_prefrontal_window(df: pd.DataFrame) -> pd.DataFrame:
    """Identify the 12-24 hour pre-frontal feeding window.

    This is THE money window for trophy bass. As pressure begins to drop
    and warm south winds push in ahead of a cold front, big bass feed
    aggressively. The window typically starts 18-24 hours before the front
    arrives and ends when pressure bottoms out.
    """
    if "pressure_trend_3h" not in df.columns or "front_type" not in df.columns:
        return df

    df = df.sort_values(["lake_key", "datetime"])

    # Look ahead: is a front coming in the next 24 hours?
    # Use a forward-looking rolling window on front_type
    is_front = (df["front_type"] == "pre_frontal").astype(int)

    # Mark the 24 hours leading up to pre-frontal detection
    df["hours_to_front"] = is_front.groupby(df["lake_key"]).transform(
        lambda x: x.iloc[::-1].rolling(24, min_periods=1).max().iloc[::-1]
    )

    # The feeding window: pressure dropping but not crashed, warm wind
    pressure_dropping = df["pressure_trend_3h"] < -0.3
    has_wind = df.get("wind_class", pd.Series(dtype=str)).isin(["south_warm", "variable"])

    df["prefrontal_feed_window"] = (
        (df["hours_to_front"] > 0) &
        pressure_dropping &
        (has_wind if "wind_class" in df.columns else True)
    ).astype(int)

    return df


def add_water_level_trend(df: pd.DataFrame) -> pd.DataFrame:
    """Track water level changes — rising water triggers feeding.

    Rising water pushes baitfish into new cover, activates bass.
    Falling water pushes fish to main-lake structure.
    Stable water = predictable patterns.
    """
    if "gage_height_ft" not in df.columns:
        return df

    df = df.sort_values(["lake_key", "datetime"])

    # Daily gage height change
    df["water_level_change_1d"] = df.groupby("lake_key")["gage_height_ft"].diff(periods=24)

    # Classify
    def classify_level(val):
        if pd.isna(val):
            return "unknown"
        if val > 0.1:
            return "rising"
        elif val < -0.1:
            return "falling"
        return "stable"

    df["water_level_trend"] = df["water_level_change_1d"].apply(classify_level)

    return df


def engineer_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all feature engineering steps."""
    logger.info(f"Engineering features on {len(df)} rows...")

    df = add_time_features(df)
    df = estimate_water_temp(df)
    df = add_pressure_features(df)
    df = add_front_detection(df)
    df = add_temp_stability(df)
    df = add_wind_direction_class(df)
    df = add_warming_trend(df)
    df = add_water_level_trend(df)
    df = add_spawn_phase(df)
    df = add_prefrontal_window(df)

    logger.info(f"Feature engineering complete. Columns: {list(df.columns)}")
    return df
