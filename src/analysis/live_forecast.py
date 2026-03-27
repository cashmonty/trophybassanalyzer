"""Live 7-day trophy bass forecast using ALL available data sources.

Data sources:
  - Open-Meteo Forecast API: real 7-day hourly weather
  - USGS Water Services: live water temp + gage height
  - ephem: real astronomical/solunar data
  - Feature pipeline: pressure trends, front detection, spawn phase,
    warming trends, water level trends, pre-frontal windows
  - LightGBM model: trained on 15 years of historical data

Usage:
    python -m src.analysis.live_forecast
"""

from __future__ import annotations

import asyncio
import logging
from datetime import date, datetime, timedelta

import httpx
import numpy as np
import pandas as pd

from src.config import DATA_DIR, load_lakes
from src.ingest.astro import compute_astro_for_lake
from src.analysis.forecast import (
    _score_water_temp,
    _score_season,
    _score_solunar,
    _score_pressure,
    _score_wind,
    _score_cloud_cover,
)

logger = logging.getLogger(__name__)

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
USGS_IV_URL = "https://waterservices.usgs.gov/nwis/iv/"  # Instantaneous values (live)

HOURLY_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "apparent_temperature",
    "precipitation",
    "rain",
    "snowfall",
    "surface_pressure",
    "pressure_msl",
    "cloud_cover",
    "cloud_cover_low",
    "cloud_cover_mid",
    "cloud_cover_high",
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_gusts_10m",
]

# USGS parameter codes
PARAM_WATER_TEMP = "00010"
PARAM_GAGE_HEIGHT = "00065"


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

def fetch_7day_weather(lat: float, lon: float) -> pd.DataFrame:
    """Fetch 7-day hourly weather forecast from Open-Meteo."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(HOURLY_VARIABLES),
        "timezone": "America/Indiana/Indianapolis",
        "forecast_days": 7,
    }
    resp = httpx.get(FORECAST_URL, params=params, timeout=30.0)
    resp.raise_for_status()
    data = resp.json()

    hourly = data.get("hourly", {})
    if not hourly or "time" not in hourly:
        return pd.DataFrame()

    df = pd.DataFrame(hourly)
    df["datetime"] = pd.to_datetime(df["time"])
    df.drop(columns=["time"], inplace=True)
    return df


def fetch_live_usgs(station_id: str, lake_key: str) -> dict:
    """Fetch recent USGS instantaneous values for water temp and gage height.

    Returns dict with keys: water_temp_c, water_temp_f, gage_height_ft, gage_history.
    gage_history is a list of (date, value) for the last 7 days for trend detection.
    """
    result = {
        "water_temp_c": None,
        "water_temp_f": None,
        "gage_height_ft": None,
        "gage_history": [],
    }

    try:
        # Fetch last 7 days of instantaneous values for trend detection
        params = {
            "format": "json",
            "sites": station_id,
            "period": "P7D",
            "parameterCd": f"{PARAM_WATER_TEMP},{PARAM_GAGE_HEIGHT}",
            "siteStatus": "all",
        }
        resp = httpx.get(USGS_IV_URL, params=params, timeout=30.0)
        resp.raise_for_status()
        data = resp.json()

        time_series = data.get("value", {}).get("timeSeries", [])
        for ts in time_series:
            var_codes = ts.get("variable", {}).get("variableCode", [])
            if not var_codes:
                continue
            param_code = var_codes[0].get("value", "")

            values = []
            for value_set in ts.get("values", []):
                for entry in value_set.get("value", []):
                    raw_val = entry.get("value")
                    dt_str = entry.get("dateTime", "")
                    try:
                        val = float(raw_val)
                        if val == -999999:
                            continue
                        values.append((dt_str, val))
                    except (TypeError, ValueError):
                        continue

            if not values:
                continue

            # Most recent value
            latest_dt, latest_val = values[-1]

            if param_code == PARAM_WATER_TEMP:
                result["water_temp_c"] = latest_val
                result["water_temp_f"] = round(latest_val * 9 / 5 + 32, 1)
                logger.info(f"  Live USGS water temp for {lake_key}: {latest_val:.1f}C / {result['water_temp_f']:.1f}F")

            elif param_code == PARAM_GAGE_HEIGHT:
                result["gage_height_ft"] = latest_val
                # Build daily history for trend detection
                daily_vals = {}
                for dt_str, val in values:
                    d = dt_str[:10]
                    daily_vals[d] = val  # last reading of each day
                result["gage_history"] = sorted(daily_vals.items())
                logger.info(f"  Live USGS gage height for {lake_key}: {latest_val:.2f} ft ({len(daily_vals)} days history)")

    except Exception as e:
        logger.warning(f"  Could not fetch live USGS for {lake_key}: {e}")

    return result


def get_recent_water_temp(lake_key: str) -> float | None:
    """Fallback: get recent water temp from historical parquet data."""
    water_path = DATA_DIR / "raw" / "water" / f"{lake_key}.parquet"
    if not water_path.exists():
        return None
    try:
        df = pd.read_parquet(water_path)
        if "water_temp_c" in df.columns:
            recent = df.dropna(subset=["water_temp_c"]).tail(1)
            if len(recent) > 0:
                return float(recent["water_temp_c"].iloc[0])
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Feature engineering (applied to forecast data)
# ---------------------------------------------------------------------------

def apply_full_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the full feature engineering pipeline to forecast data.

    Reuses the same logic from src.pipeline.features but adapted for
    a single-lake 7-day forecast DataFrame.
    """
    df = df.sort_values("datetime").copy()

    # Time features
    dt = pd.to_datetime(df["datetime"])
    df["hour"] = dt.dt.hour
    df["month"] = dt.dt.month
    df["day_of_year"] = dt.dt.dayofyear

    # Pressure trends (3h and 6h)
    if "pressure_msl" in df.columns:
        for hours in [3, 6]:
            df[f"pressure_trend_{hours}h"] = df["pressure_msl"].diff(periods=hours)

        def classify_trend(val):
            if pd.isna(val):
                return "unknown"
            if val > 1.5:
                return "rising"
            elif val < -1.5:
                return "falling"
            return "stable"

        df["pressure_trend_class"] = df["pressure_trend_3h"].apply(classify_trend)

    # Wind class
    if "wind_direction_10m" in df.columns:
        wd = df["wind_direction_10m"]
        conditions = [
            (wd >= 157.5) & (wd < 247.5),
            (wd >= 247.5) & (wd < 337.5),
            (wd >= 337.5) | (wd < 22.5),
            (wd >= 22.5) & (wd < 67.5),
            (wd >= 67.5) & (wd < 157.5),
        ]
        choices = ["south_warm", "northwest_cold", "north", "northeast", "east_poor"]
        df["wind_class"] = np.select(conditions, choices, default="variable")

    # Front detection
    if {"pressure_msl", "wind_speed_10m", "wind_direction_10m"}.issubset(df.columns):
        p_change = df["pressure_msl"].diff(6)
        wd_change = df["wind_direction_10m"].diff(6).abs()
        wd_change = wd_change.where(wd_change <= 180, 360 - wd_change)

        front_conditions = [
            (p_change < -3) & (wd_change > 30),
            (p_change > 3) & (wd_change > 30),
        ]
        df["front_type"] = np.select(front_conditions, ["pre_frontal", "post_frontal"], default="stable")

        # Days since last front
        is_front = df["front_type"] != "stable"
        cumsum = is_front.cumsum()
        df["days_since_last_front"] = cumsum.groupby(cumsum).cumcount() / 24.0

    # Temperature stability (3-day rolling std)
    if "temperature_2m" in df.columns:
        df["temp_stability_3day"] = df["temperature_2m"].rolling(72, min_periods=12).std()

    # Warming trend
    temp_col = "water_temp_estimated" if "water_temp_estimated" in df.columns else "temperature_2m"
    if temp_col in df.columns:
        daily_temp = df.groupby(pd.to_datetime(df["datetime"]).dt.date)[temp_col].transform("mean")
        df["temp_change_3day"] = daily_temp.diff(periods=72)
        df["is_warming_trend"] = (df["temp_change_3day"] > 0.5).astype(int)
        df["warming_streak"] = df["is_warming_trend"].rolling(120, min_periods=24).sum()

    # Water level trend
    if "gage_height_ft" in df.columns and df["gage_height_ft"].notna().any():
        df["water_level_change_1d"] = df["gage_height_ft"].diff(periods=24)

        def classify_level(val):
            if pd.isna(val):
                return "unknown"
            if val > 0.1:
                return "rising"
            elif val < -0.1:
                return "falling"
            return "stable"

        df["water_level_trend"] = df["water_level_change_1d"].apply(classify_level)
    else:
        df["water_level_change_1d"] = np.nan
        df["water_level_trend"] = "unknown"

    # Spawn phase
    if temp_col in df.columns:
        temp_smooth = df[temp_col].rolling(7 * 24, min_periods=24).mean()
        month = pd.to_datetime(df["datetime"]).dt.month

        phase_conditions = [
            temp_smooth < 7,
            (temp_smooth >= 7) & (temp_smooth < 14) & (month <= 6),
            (temp_smooth >= 14) & (temp_smooth < 20) & (month <= 6),
            (temp_smooth >= 20) & (temp_smooth < 23) & (month <= 7),
            (temp_smooth >= 23) | ((temp_smooth >= 20) & (month.between(7, 8))),
            (temp_smooth >= 14) & (temp_smooth < 23) & (month >= 9),
            (temp_smooth >= 7) & (temp_smooth < 14) & (month >= 9),
        ]
        phase_choices = ["WINTER", "PRE_SPAWN", "SPAWN", "POST_SPAWN", "SUMMER", "FALL", "TURNOVER"]
        df["spawn_phase"] = np.select(phase_conditions, phase_choices, default="UNKNOWN")

    # Pre-frontal feeding window
    if "pressure_trend_3h" in df.columns and "front_type" in df.columns:
        is_front = (df["front_type"] == "pre_frontal").astype(int)
        df["hours_to_front"] = is_front.iloc[::-1].rolling(24, min_periods=1).max().iloc[::-1]

        pressure_dropping = df["pressure_trend_3h"] < -0.3
        has_wind = df.get("wind_class", pd.Series(dtype=str)).isin(["south_warm", "variable"])

        df["prefrontal_feed_window"] = (
            (df["hours_to_front"] > 0) &
            pressure_dropping &
            (has_wind if "wind_class" in df.columns else True)
        ).astype(int)

    return df


def apply_water_temp_estimation(df: pd.DataFrame, live_water_temp_c: float | None,
                                fallback_water_temp_c: float | None) -> pd.DataFrame:
    """Estimate water temperature with best available starting point."""
    if "temperature_2m" not in df.columns:
        return df

    # Determine starting water temp
    start_temp = live_water_temp_c or fallback_water_temp_c
    if start_temp is None:
        # Estimate from air temp
        start_temp = df["temperature_2m"].iloc[0]
        if not np.isnan(start_temp):
            month = df["datetime"].iloc[0].month
            if month in (3, 4, 5):
                start_temp *= 0.75
            elif month in (9, 10, 11):
                start_temp *= 1.1
        else:
            start_temp = 10.0

    alpha = 0.04
    air_temp = df["temperature_2m"].values
    water_est = np.empty_like(air_temp, dtype=float)
    water_est[0] = start_temp

    for i in range(1, len(air_temp)):
        if np.isnan(air_temp[i]):
            water_est[i] = water_est[i - 1]
        else:
            water_est[i] = alpha * air_temp[i] + (1 - alpha) * water_est[i - 1]

    df["water_temp_estimated"] = water_est
    df["water_temp_f_est"] = water_est * 9 / 5 + 32
    return df


def apply_gage_height(df: pd.DataFrame, current_gage: float | None,
                      gage_history: list[tuple[str, float]]) -> pd.DataFrame:
    """Apply gage height data — use live value and project forward."""
    if current_gage is None:
        return df

    # Set current gage height for all forecast rows
    # If we have history, compute a daily trend and project
    if gage_history and len(gage_history) >= 2:
        vals = [v for _, v in gage_history]
        daily_change = (vals[-1] - vals[0]) / max(len(vals) - 1, 1)

        # Project gage height forward from current
        hours = np.arange(len(df))
        df["gage_height_ft"] = current_gage + (hours / 24.0) * daily_change
    else:
        df["gage_height_ft"] = current_gage

    return df


# ---------------------------------------------------------------------------
# ML model integration
# ---------------------------------------------------------------------------

def get_ml_predictions(df: pd.DataFrame) -> pd.Series | None:
    """Run the trained LightGBM model on forecast data, if available."""
    try:
        from src.analysis.model import load_model, load_feature_cols, prepare_features, align_features

        model = load_model()
        training_cols = load_feature_cols()

        X, _ = prepare_features(df)
        X = align_features(X, training_cols)

        predictions = model.predict(X)
        logger.info(f"  ML model predictions: min={predictions.min():.4f}, max={predictions.max():.4f}, mean={predictions.mean():.4f}")
        return pd.Series(predictions, index=df.index)

    except FileNotFoundError:
        logger.warning("  No trained model found — skipping ML predictions")
        return None
    except Exception as e:
        logger.warning(f"  ML prediction failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Historical pattern analysis
# ---------------------------------------------------------------------------

def load_historical_profiles() -> dict:
    """Load historical data and compute per-lake trophy profiles.

    Returns a dict with:
        lake_trophy_rates: {lake_key: trophies_per_year}
        lake_trophy_counts: {lake_key: total_trophies}
        optimal_water_temp_f: (min, peak, max) from historical trophy catches
        best_moon_phases: list of moon phases ranked by trophy frequency
        best_spawn_phases: list of spawn phases ranked by trophy frequency
        best_wind_classes: list of wind classes ranked by trophy frequency
    """
    profiles = {
        "lake_trophy_rates": {},
        "lake_trophy_counts": {},
        "optimal_water_temp_f": (48, 62, 78),  # defaults
        "best_moon_phases": [],
        "best_spawn_phases": [],
        "best_wind_classes": [],
    }

    try:
        catches_path = DATA_DIR / "processed" / "catches.parquet"
        merged_path = DATA_DIR / "processed" / "merged.parquet"

        if not catches_path.exists() or not merged_path.exists():
            return profiles

        catches = pd.read_parquet(catches_path)
        catches["date"] = pd.to_datetime(catches["date"])
        trophy_catches = catches[catches["weight_lbs"] >= 7.0]

        # Per-lake trophy counts
        lake_counts = trophy_catches.groupby("lake_key").size().to_dict()
        profiles["lake_trophy_counts"] = lake_counts

        # Trophy rate: trophies per year per lake
        if len(catches) > 0:
            years = (catches["date"].max() - catches["date"].min()).days / 365.25
            years = max(years, 1)
            for lake_key, count in lake_counts.items():
                profiles["lake_trophy_rates"][lake_key] = count / years

        # Load merged for condition analysis
        merged = pd.read_parquet(merged_path, columns=[
            "lake_key", "max_weight", "water_temp_estimated",
            "moon_phase_name", "spawn_phase", "wind_class",
            "pressure_trend_class",
        ])
        trophy_hours = merged[merged["max_weight"] >= 7.0]

        if len(trophy_hours) == 0:
            return profiles

        # Optimal water temp range from actual trophy catches
        if "water_temp_estimated" in trophy_hours.columns:
            wt = trophy_hours["water_temp_estimated"].dropna()
            if len(wt) > 0:
                wt_f = wt * 9 / 5 + 32
                profiles["optimal_water_temp_f"] = (
                    float(wt_f.quantile(0.1)),
                    float(wt_f.median()),
                    float(wt_f.quantile(0.9)),
                )

        # Best moon phases (ranked by frequency)
        if "moon_phase_name" in trophy_hours.columns:
            moon_counts = trophy_hours["moon_phase_name"].value_counts()
            profiles["best_moon_phases"] = moon_counts.index.tolist()

        # Best spawn phases
        if "spawn_phase" in trophy_hours.columns:
            phase_counts = trophy_hours["spawn_phase"].value_counts()
            profiles["best_spawn_phases"] = phase_counts.index.tolist()

        # Best wind classes
        if "wind_class" in trophy_hours.columns:
            wind_counts = trophy_hours["wind_class"].value_counts()
            profiles["best_wind_classes"] = wind_counts.index.tolist()

        logger.info(f"  Historical profiles loaded: {sum(lake_counts.values())} total trophies across {len(lake_counts)} lakes")
        logger.info(f"  Optimal water temp: {profiles['optimal_water_temp_f'][0]:.0f}-{profiles['optimal_water_temp_f'][2]:.0f}F (peak {profiles['optimal_water_temp_f'][1]:.0f}F)")

    except Exception as e:
        logger.warning(f"  Could not load historical profiles: {e}")

    return profiles


def _score_historical_match(row: pd.Series, profiles: dict) -> float:
    """Score based on how well current conditions match historical trophy patterns (0-10 points).

    Compares forecast conditions against the actual conditions when trophies
    were caught historically at each lake.
    """
    score = 0.0

    # Lake trophy track record (0-3 pts)
    lake_key = row.get("lake_key", "")
    trophy_count = profiles.get("lake_trophy_counts", {}).get(lake_key, 0)
    if trophy_count >= 15:
        score += 3.0  # Proven trophy producer
    elif trophy_count >= 8:
        score += 2.0
    elif trophy_count >= 3:
        score += 1.0

    # Water temp in historical optimal zone (0-3 pts)
    temp_f = row.get("water_temp_f_est")
    if not pd.isna(temp_f):
        opt_min, opt_peak, opt_max = profiles.get("optimal_water_temp_f", (48, 62, 78))
        if opt_min <= temp_f <= opt_max:
            # Closer to peak = higher score
            dist_from_peak = abs(temp_f - opt_peak)
            half_range = (opt_max - opt_min) / 2
            score += 3.0 * max(0, 1 - dist_from_peak / half_range)

    # Moon phase match (0-2 pts)
    moon = row.get("moon_phase_name", "")
    best_moons = profiles.get("best_moon_phases", [])
    if moon in best_moons[:3]:
        score += 2.0
    elif moon in best_moons[:5]:
        score += 1.0

    # Spawn phase match (0-2 pts)
    phase = row.get("spawn_phase", "")
    best_phases = profiles.get("best_spawn_phases", [])
    if phase in best_phases[:2]:
        score += 2.0
    elif phase in best_phases[:4]:
        score += 1.0

    return min(score, 10.0)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _score_time_of_day(hour: int) -> float:
    """Score time of day for bass fishing (0-10 points)."""
    if 5 <= hour <= 8:
        return 10.0
    elif 17 <= hour <= 20:
        return 9.0
    elif 9 <= hour <= 11:
        return 6.0
    elif 15 <= hour <= 16:
        return 6.0
    elif 12 <= hour <= 14:
        return 4.0
    elif 21 <= hour <= 22:
        return 3.0
    else:
        return 1.0


def _score_spawn_phase(phase: str) -> float:
    """Score based on bass seasonal phase (0-8 points)."""
    scores = {
        "PRE_SPAWN": 8.0,   # Prime trophy window
        "SPAWN": 6.0,       # Big fish on beds
        "FALL": 5.0,        # Fall feed-up
        "POST_SPAWN": 4.0,  # Transitioning
        "SUMMER": 3.0,      # Tough but doable
        "TURNOVER": 2.0,    # Difficult
        "WINTER": 1.0,      # Very tough
    }
    return scores.get(str(phase), 3.0)


def _score_warming_trend(is_warming: float, warming_streak: float) -> float:
    """Score warming trend (0-5 points). 3+ warm days = trophy trigger."""
    score = 0.0
    if not pd.isna(is_warming) and is_warming > 0:
        score += 2.0
    if not pd.isna(warming_streak):
        if warming_streak >= 72:  # 3+ days
            score += 3.0
        elif warming_streak >= 48:
            score += 2.0
        elif warming_streak >= 24:
            score += 1.0
    return score


def _score_frontal_activity(front_type: str, prefrontal: float, hours_to_front: float) -> float:
    """Score frontal activity (0-7 points). Pre-frontal = money window."""
    if not pd.isna(prefrontal) and prefrontal > 0:
        return 7.0  # Pre-frontal feeding window — best
    if str(front_type) == "pre_frontal":
        return 5.0
    if not pd.isna(hours_to_front) and hours_to_front > 0:
        return 4.0  # Front approaching
    if str(front_type) == "post_frontal":
        return 1.0  # Post-frontal lockjaw
    return 3.0  # Stable — neutral


def _score_water_level(trend: str, change: float) -> float:
    """Score water level trend (0-5 points). Rising = feeding trigger."""
    if str(trend) == "rising":
        return 5.0
    elif str(trend) == "stable":
        return 3.0
    elif str(trend) == "falling":
        return 2.0
    return 3.0  # unknown


def compute_trophy_score_full(row: pd.Series, ml_pred: float | None = None,
                              profiles: dict | None = None) -> float:
    """Compute composite trophy bass score (0-100) using ALL available data.

    Scoring breakdown (max 100):
        Water temp:        0-18 pts  (pre-spawn sweet spot)
        Season:            0-10 pts  (timing)
        Spawn phase:       0-8 pts   (biological state)
        Solunar:           0-8 pts   (moon/tidal)
        Pressure trend:    0-7 pts   (frontal systems)
        Frontal activity:  0-7 pts   (pre-frontal = money)
        Wind:              0-7 pts   (direction + speed)
        Cloud cover:       0-5 pts   (overcast better)
        Time of day:       0-8 pts   (dawn/dusk prime)
        Warming trend:     0-5 pts   (consecutive warm days)
        Water level:       0-4 pts   (rising = feeding trigger)
        Historical match:  0-10 pts  (how well conditions match actual trophy history)
        ML model boost:    0-3 pts   (trained model pattern match)
    """
    temp_f = row.get("water_temp_f_est")
    if pd.isna(temp_f) and "water_temp_estimated" in row.index:
        temp_c = row.get("water_temp_estimated")
        temp_f = temp_c * 9 / 5 + 32 if not pd.isna(temp_c) else None

    score = 0.0
    score += _score_water_temp(temp_f) * (18 / 25)                               # 0-18
    score += _score_season(row.get("month", 1), row.get("day_of_year", 1)) * 0.5  # 0-10
    score += _score_spawn_phase(row.get("spawn_phase", "UNKNOWN"))                 # 0-8
    score += _score_solunar(row.get("solunar_base_score")) * 0.4                   # 0-8
    score += _score_pressure(
        row.get("pressure_msl"), row.get("pressure_trend_3h")) * (7 / 15)         # 0-7
    score += _score_frontal_activity(
        row.get("front_type", "stable"),
        row.get("prefrontal_feed_window", 0),
        row.get("hours_to_front", 0))                                               # 0-7
    score += _score_wind(
        row.get("wind_speed_10m"), row.get("wind_class", "unknown")) * 0.7         # 0-7
    score += _score_cloud_cover(row.get("cloud_cover")) * 0.5                      # 0-5
    score += _score_time_of_day(int(row.get("hour", 12))) * 0.8                   # 0-8
    score += _score_warming_trend(
        row.get("is_warming_trend", 0), row.get("warming_streak", 0))             # 0-5
    score += _score_water_level(
        row.get("water_level_trend", "unknown"),
        row.get("water_level_change_1d", 0))                                       # 0-4

    # Historical pattern match (up to 10 points from actual trophy data)
    if profiles:
        score += _score_historical_match(row, profiles)                            # 0-10

    score = min(score, 97.0)  # Cap heuristic at 97

    # ML model boost (up to 3 additional points)
    if ml_pred is not None and not np.isnan(ml_pred):
        score += min(ml_pred * 30, 3.0)

    return min(score, 100.0)


# ---------------------------------------------------------------------------
# Main forecast generator
# ---------------------------------------------------------------------------

def generate_live_forecast() -> pd.DataFrame:
    """Generate a 7-day live trophy bass forecast using all available data."""
    lakes = load_lakes()
    today = date.today()
    end_date = today + timedelta(days=6)

    logger.info(f"Generating live 7-day forecast: {today} to {end_date}")
    logger.info("Data sources: Open-Meteo forecast, USGS live water, ephem solunar, feature pipeline, historical patterns, LightGBM model")

    # Load historical trophy profiles from all dashboard data
    logger.info("Loading historical trophy profiles from 15 years of data...")
    profiles = load_historical_profiles()

    all_lake_forecasts = []

    for lake in lakes:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {lake.key} ({lake.name})...")

        # 1. Fetch real weather forecast
        logger.info(f"  Fetching 7-day weather forecast...")
        weather_df = fetch_7day_weather(lake.lat, lake.lon)
        if weather_df.empty:
            logger.warning(f"  No weather forecast for {lake.key}, skipping")
            continue
        weather_df["lake_key"] = lake.key

        # 2. Fetch live USGS water data
        live_water_temp_c = None
        current_gage = None
        gage_history = []

        if lake.usgs_station:
            logger.info(f"  Fetching live USGS data (station {lake.usgs_station})...")
            usgs_data = fetch_live_usgs(lake.usgs_station, lake.key)
            live_water_temp_c = usgs_data["water_temp_c"]
            current_gage = usgs_data["gage_height_ft"]
            gage_history = usgs_data["gage_history"]
        else:
            logger.info(f"  No USGS station for {lake.key}, using fallback water temp")

        # Fallback to historical parquet if no live data
        fallback_water_temp = None
        if live_water_temp_c is None:
            fallback_water_temp = get_recent_water_temp(lake.key)
            if fallback_water_temp:
                logger.info(f"  Using historical water temp: {fallback_water_temp:.1f}C")

        # 3. Compute real astro/solunar data
        logger.info(f"  Computing solunar data...")
        astro_df = compute_astro_for_lake(lake.key, lake.lat, lake.lon, today, end_date)
        astro_df["date"] = pd.to_datetime(astro_df["date"]).dt.date

        # 4. Merge astro onto weather
        weather_df["date"] = weather_df["datetime"].dt.date
        astro_cols = ["date", "lake_key", "moon_illumination", "moon_phase_name",
                      "solunar_base_score", "sunrise", "sunset",
                      "major_period_1_start", "major_period_1_end",
                      "major_period_2_start", "major_period_2_end",
                      "minor_period_1_start", "minor_period_1_end",
                      "minor_period_2_start", "minor_period_2_end"]
        weather_df = weather_df.merge(
            astro_df[astro_cols], on=["date", "lake_key"], how="left"
        )

        # 5. Apply water temp estimation with best available data
        weather_df = apply_water_temp_estimation(weather_df, live_water_temp_c, fallback_water_temp)

        # 6. Apply gage height data
        weather_df = apply_gage_height(weather_df, current_gage, gage_history)

        # 7. Run FULL feature engineering pipeline
        logger.info(f"  Running full feature pipeline...")
        weather_df = apply_full_features(weather_df)

        # 8. Get ML model predictions
        logger.info(f"  Running ML model...")
        ml_preds = get_ml_predictions(weather_df)

        # 9. Score every hour with all data
        logger.info(f"  Computing trophy scores...")
        if ml_preds is not None:
            weather_df["ml_prediction"] = ml_preds.values
            weather_df["trophy_score"] = weather_df.apply(
                lambda row: compute_trophy_score_full(row, row.get("ml_prediction"), profiles), axis=1
            )
        else:
            weather_df["trophy_score"] = weather_df.apply(
                lambda row: compute_trophy_score_full(row, profiles=profiles), axis=1
            )

        all_lake_forecasts.append(weather_df)
        logger.info(f"  Done. Score range: {weather_df['trophy_score'].min():.0f}-{weather_df['trophy_score'].max():.0f}")

    if not all_lake_forecasts:
        logger.error("No forecasts generated for any lake")
        return pd.DataFrame()

    hourly = pd.concat(all_lake_forecasts, ignore_index=True)

    # Save hourly
    hourly_path = DATA_DIR / "processed" / "live_forecast_hourly.parquet"
    hourly_path.parent.mkdir(parents=True, exist_ok=True)
    hourly.to_parquet(hourly_path, index=False)

    # Aggregate to daily
    daily = hourly.groupby(["date", "lake_key"]).agg(
        max_score=("trophy_score", "max"),
        mean_score=("trophy_score", "mean"),
        best_hour_idx=("trophy_score", "idxmax"),
        water_temp_f_avg=("water_temp_f_est", "mean"),
        air_temp_high=("temperature_2m", "max"),
        air_temp_low=("temperature_2m", "min"),
        wind_avg=("wind_speed_10m", "mean"),
        wind_max=("wind_gusts_10m", "max"),
        cloud_avg=("cloud_cover", "mean"),
        precip_total=("precipitation", "sum"),
        pressure_avg=("pressure_msl", "mean"),
    ).reset_index()

    # Get best hour as actual hour value
    daily["best_hour"] = hourly.loc[daily["best_hour_idx"].values, "hour"].values
    daily.drop(columns=["best_hour_idx"], inplace=True)

    # Get dominant wind, spawn phase for each day
    for col_name, src_col in [("dominant_wind", "wind_class"), ("spawn_phase", "spawn_phase")]:
        if src_col in hourly.columns:
            day_vals = []
            for _, row in daily.iterrows():
                mask = (hourly["date"] == row["date"]) & (hourly["lake_key"] == row["lake_key"])
                lake_hrs = hourly.loc[mask, src_col]
                if len(lake_hrs) > 0:
                    mode = lake_hrs.mode()
                    day_vals.append(mode.iloc[0] if len(mode) > 0 else "unknown")
                else:
                    day_vals.append("unknown")
            daily[col_name] = day_vals

    # Moon phase
    if "moon_phase_name" in hourly.columns:
        moon_daily = hourly.groupby(["date", "lake_key"])["moon_phase_name"].first().reset_index()
        daily = daily.merge(moon_daily, on=["date", "lake_key"], how="left")
    if "moon_illumination" in hourly.columns:
        moon_illum = hourly.groupby(["date", "lake_key"])["moon_illumination"].first().reset_index()
        daily = daily.merge(moon_illum, on=["date", "lake_key"], how="left")

    # Rating
    daily["rating"] = pd.cut(
        daily["max_score"],
        bins=[0, 35, 50, 65, 80, 100],
        labels=["Poor", "Fair", "Good", "Great", "Excellent"],
    )

    # Save
    daily_path = DATA_DIR / "processed" / "live_forecast_daily.parquet"
    daily.to_parquet(daily_path, index=False)

    logger.info(f"\nSaved live forecast: {len(daily)} daily records, {len(hourly)} hourly records")
    return daily


# ---------------------------------------------------------------------------
# CLI display
# ---------------------------------------------------------------------------

def _fmt_hour(h):
    """Format hour as 12h AM/PM string."""
    if pd.isna(h):
        return "  --  "
    h = int(h)
    ampm = "AM" if h < 12 else "PM"
    h12 = h % 12 or 12
    return f"{h12}{ampm}"


def _fmt_time(dt_val):
    """Format a datetime to HH:MM AM/PM."""
    if pd.isna(dt_val) or dt_val is None:
        return "--:--"
    if isinstance(dt_val, str):
        try:
            dt_val = pd.Timestamp(dt_val)
        except Exception:
            return "--:--"
    try:
        return dt_val.strftime("%-I:%M%p").lower()
    except ValueError:
        return dt_val.strftime("%I:%M%p").lower().lstrip("0")


def print_forecast(daily: pd.DataFrame, hourly: pd.DataFrame | None = None) -> None:
    """Pretty-print the 7-day forecast with solunar periods."""
    lakes = load_lakes()
    lake_names = {l.key: l.name for l in lakes}

    print("\n" + "=" * 95)
    print("  7-DAY TROPHY BASS FORECAST - INDIANA")
    print(f"  Generated: {datetime.now().strftime('%Y-%m-%d %I:%M %p')}")
    print(f"  Data: Open-Meteo forecast + USGS live water + solunar + feature pipeline + LightGBM model")
    print("=" * 95)

    for d in sorted(daily["date"].unique()):
        day_data = daily[daily["date"] == d].sort_values("max_score", ascending=False)
        day_name = pd.Timestamp(d).strftime("%A %b %d")
        moon = day_data.iloc[0].get("moon_phase_name", "")
        moon_illum = day_data.iloc[0].get("moon_illumination", 0)
        if pd.isna(moon_illum):
            moon_illum = 0
        phase = day_data.iloc[0].get("spawn_phase", "")

        solunar_str = ""
        if hourly is not None:
            day_hours = hourly[hourly["date"] == d]
            if len(day_hours) > 0:
                first_row = day_hours.iloc[0]
                majors = []
                for prefix in ["major_period_1", "major_period_2"]:
                    start = first_row.get(f"{prefix}_start")
                    end = first_row.get(f"{prefix}_end")
                    if start is not None and not pd.isna(start):
                        majors.append(f"{_fmt_time(start)}-{_fmt_time(end)}")
                minors = []
                for prefix in ["minor_period_1", "minor_period_2"]:
                    start = first_row.get(f"{prefix}_start")
                    end = first_row.get(f"{prefix}_end")
                    if start is not None and not pd.isna(start):
                        minors.append(f"{_fmt_time(start)}-{_fmt_time(end)}")
                if majors:
                    solunar_str += f"  Major: {', '.join(majors)}"
                if minors:
                    solunar_str += f"  Minor: {', '.join(minors)}"

        print(f"\n{'-' * 95}")
        print(f"  {day_name}  |  Moon: {moon} ({moon_illum*100:.0f}%)  |  Phase: {phase}")
        if solunar_str:
            print(f"  Solunar:{solunar_str}")
        print(f"{'-' * 95}")
        print(f"  {'Lake':<18} {'Score':>5} {'Rating':<9} {'Best Hr':>7} "
              f"{'WaterF':>6} {'Hi/Lo':>9} {'Wind':>5} {'Gust':>5} {'Cloud':>5} {'Rain':>6} {'Wind Dir':<12}")

        for _, row in day_data.iterrows():
            lake_name = lake_names.get(row["lake_key"], row["lake_key"])
            air_hi_f = row["air_temp_high"] * 9 / 5 + 32
            air_lo_f = row["air_temp_low"] * 9 / 5 + 32
            wind_mph = row["wind_avg"] * 0.621371
            gust_mph = row["wind_max"] * 0.621371
            precip_in = row["precip_total"] / 25.4

            print(f"  {lake_name:<18} {row['max_score']:>5.0f} {str(row['rating']):<9} "
                  f"{_fmt_hour(row['best_hour']):>7} "
                  f"{row['water_temp_f_avg']:>5.0f}F {air_hi_f:>3.0f}/{air_lo_f:>3.0f}F "
                  f"{wind_mph:>4.0f}  {gust_mph:>4.0f}  {row['cloud_avg']:>4.0f}% "
                  f"{precip_in:>5.2f}\" {row['dominant_wind']:<12}")

    top3 = daily.nlargest(3, "max_score")
    print(f"\n{'=' * 95}")
    print("  TOP 3 FISHING WINDOWS THIS WEEK:")
    print(f"{'=' * 95}")
    for i, (_, row) in enumerate(top3.iterrows(), 1):
        lake_name = lake_names.get(row["lake_key"], row["lake_key"])
        day_name = pd.Timestamp(row["date"]).strftime("%A %b %d")
        phase = row.get("spawn_phase", "")
        print(f"  #{i}: {lake_name} on {day_name} - Score {row['max_score']:.0f} "
              f"({row['rating']}) - Best at {_fmt_hour(row['best_hour'])} [{phase}]")

    print()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )
    daily = generate_live_forecast()
    if len(daily) > 0:
        hourly_path = DATA_DIR / "processed" / "live_forecast_hourly.parquet"
        hourly = pd.read_parquet(hourly_path) if hourly_path.exists() else None
        print_forecast(daily, hourly)
