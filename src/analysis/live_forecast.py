"""Live 7-day trophy bass forecast using real weather forecast + solunar data.

Fetches actual forecast weather from Open-Meteo Forecast API (not archive),
computes real astronomical/solunar data, derives all features from the real
data, and scores each hour for trophy bass probability.

Usage:
    python -m src.analysis.live_forecast
"""

from __future__ import annotations

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


def compute_wind_class(wind_direction: pd.Series) -> pd.Series:
    """Classify wind direction into fishing-relevant categories."""
    wd = wind_direction
    conditions = [
        (wd >= 157.5) & (wd < 247.5),
        (wd >= 247.5) & (wd < 337.5),
        (wd >= 337.5) | (wd < 22.5),
        (wd >= 22.5) & (wd < 67.5),
        (wd >= 67.5) & (wd < 157.5),
    ]
    choices = ["south_warm", "northwest_cold", "north", "northeast", "east_poor"]
    return pd.Series(
        np.select(conditions, choices, default="variable"),
        index=wind_direction.index,
    )


def compute_pressure_trends(df: pd.DataFrame) -> pd.DataFrame:
    """Compute pressure trends from actual forecast pressure data."""
    if "pressure_msl" not in df.columns:
        return df
    df = df.sort_values("datetime")
    for hours in [3, 6]:
        col = f"pressure_trend_{hours}h"
        df[col] = df["pressure_msl"].diff(periods=hours)
    return df


def estimate_water_temp_from_forecast(df: pd.DataFrame) -> pd.DataFrame:
    """Estimate water temp from air temp using exponential smoothing.

    Uses a slower alpha since water lags air temp by 24-48h.
    Initializes from recent historical water temp if available.
    """
    if "temperature_2m" not in df.columns:
        return df

    # Alpha = 0.04 — water responds to air temp but slowly (24-48h lag)
    # Higher than historical (0.02) because we only have 7 days to show change
    alpha = 0.04
    air_temp = df["temperature_2m"].values
    water_est = np.empty_like(air_temp, dtype=float)

    # Use first air temp as starting point, adjusted for season
    start_temp = air_temp[0] if not np.isnan(air_temp[0]) else 10.0
    month = df["datetime"].iloc[0].month
    if month in (3, 4, 5):
        start_temp = start_temp * 0.75  # water lags spring warming
    elif month in (9, 10, 11):
        start_temp = start_temp * 1.1  # water retains summer heat

    water_est[0] = start_temp
    for i in range(1, len(air_temp)):
        if np.isnan(air_temp[i]):
            water_est[i] = water_est[i - 1]
        else:
            water_est[i] = alpha * air_temp[i] + (1 - alpha) * water_est[i - 1]

    df["water_temp_estimated"] = water_est
    df["water_temp_f_est"] = water_est * 9 / 5 + 32
    return df


def _score_time_of_day(hour: int, sunrise_dt=None, sunset_dt=None) -> float:
    """Score time of day for bass fishing (0-10 points).

    Dawn and dusk are prime. Midday is worst. Night is poor.
    First/last light windows are the absolute best.
    """
    # Default sunrise/sunset for Indiana if not provided
    if 5 <= hour <= 8:
        return 10.0  # Dawn/early morning — prime
    elif 17 <= hour <= 20:
        return 9.0   # Dusk — prime
    elif 9 <= hour <= 11:
        return 6.0   # Late morning — decent
    elif 15 <= hour <= 16:
        return 6.0   # Afternoon — warming up
    elif 12 <= hour <= 14:
        return 4.0   # Midday — slowest
    elif 21 <= hour <= 22:
        return 3.0   # Early night — some activity
    else:
        return 1.0   # Middle of night — very low


def compute_trophy_score_live(row: pd.Series) -> float:
    """Compute composite trophy bass score (0-100) for a single hourly record.

    Scoring breakdown (max 100):
        Water temp:    0-25 pts
        Season:        0-20 pts
        Solunar:       0-15 pts (reduced from 20 — moon matters but isn't everything)
        Pressure:      0-10 pts (reduced — trends matter most)
        Wind:          0-10 pts
        Cloud cover:   0-10 pts
        Time of day:   0-10 pts (NEW — dawn/dusk are prime, midnight is not)
    """
    temp_f = row.get("water_temp_f_est")
    if pd.isna(temp_f) and "water_temp_estimated" in row.index:
        temp_c = row.get("water_temp_estimated")
        temp_f = temp_c * 9 / 5 + 32 if not pd.isna(temp_c) else None

    score = 0.0
    score += _score_water_temp(temp_f)                                          # 0-25
    score += _score_season(row.get("month", 1), row.get("day_of_year", 1))     # 0-20
    # Scale solunar down to 0-15 (from 0-20) — moon helps but isn't dominant
    score += _score_solunar(row.get("solunar_base_score")) * 0.75              # 0-15
    # Scale pressure to 0-10 (from 0-15)
    score += _score_pressure(row.get("pressure_msl"), row.get("pressure_trend_3h")) * (10/15)  # 0-10
    score += _score_wind(row.get("wind_speed_10m"), row.get("wind_class", "unknown"))  # 0-10
    score += _score_cloud_cover(row.get("cloud_cover"))                        # 0-10
    score += _score_time_of_day(int(row.get("hour", 12)))                      # 0-10

    return min(score, 100.0)


def get_recent_water_temp(lake_key: str) -> float | None:
    """Try to get recent actual water temp from USGS data for better initialization."""
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


def generate_live_forecast() -> pd.DataFrame:
    """Generate a 7-day live trophy bass forecast for all lakes."""
    lakes = load_lakes()
    today = date.today()
    end_date = today + timedelta(days=6)

    logger.info(f"Generating live 7-day forecast: {today} to {end_date}")

    all_lake_forecasts = []

    for lake in lakes:
        logger.info(f"Fetching weather forecast for {lake.key} ({lake.name})...")

        # 1. Fetch real weather forecast
        weather_df = fetch_7day_weather(lake.lat, lake.lon)
        if weather_df.empty:
            logger.warning(f"No weather forecast for {lake.key}, skipping")
            continue

        weather_df["lake_key"] = lake.key

        # 2. Compute real astro/solunar data
        logger.info(f"Computing solunar data for {lake.key}...")
        astro_df = compute_astro_for_lake(
            lake.key, lake.lat, lake.lon, today, end_date
        )
        astro_df["date"] = pd.to_datetime(astro_df["date"]).dt.date

        # 3. Merge astro onto weather
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

        # 4. Derive features from REAL data
        weather_df = compute_pressure_trends(weather_df)

        # Wind class from real wind direction
        if "wind_direction_10m" in weather_df.columns:
            weather_df["wind_class"] = compute_wind_class(weather_df["wind_direction_10m"])

        # Water temp estimation
        recent_water = get_recent_water_temp(lake.key)
        if recent_water is not None:
            logger.info(f"  Using recent water temp {recent_water:.1f}°C for {lake.key}")
            # Override the starting temp with real data
            weather_df = estimate_water_temp_from_forecast(weather_df)
            # Re-initialize with actual water temp
            alpha = 0.04
            air_temp = weather_df["temperature_2m"].values
            water_est = np.empty_like(air_temp, dtype=float)
            water_est[0] = recent_water
            for i in range(1, len(air_temp)):
                if np.isnan(air_temp[i]):
                    water_est[i] = water_est[i - 1]
                else:
                    water_est[i] = alpha * air_temp[i] + (1 - alpha) * water_est[i - 1]
            weather_df["water_temp_estimated"] = water_est
            weather_df["water_temp_f_est"] = water_est * 9 / 5 + 32
        else:
            weather_df = estimate_water_temp_from_forecast(weather_df)

        # Time features
        weather_df["hour"] = weather_df["datetime"].dt.hour
        weather_df["month"] = weather_df["datetime"].dt.month
        weather_df["day_of_year"] = weather_df["datetime"].dt.dayofyear

        # 5. Score every hour
        weather_df["trophy_score"] = weather_df.apply(compute_trophy_score_live, axis=1)

        all_lake_forecasts.append(weather_df)

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

    # Get wind class for best hours
    best_wind = []
    for _, row in daily.iterrows():
        lake_hours = hourly[
            (hourly["date"] == row["date"]) & (hourly["lake_key"] == row["lake_key"])
        ]
        if len(lake_hours) > 0 and "wind_class" in lake_hours.columns:
            best_wind.append(lake_hours["wind_class"].mode().iloc[0] if len(lake_hours) > 0 else "unknown")
        else:
            best_wind.append("unknown")
    daily["dominant_wind"] = best_wind

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

    logger.info(f"Saved live forecast: {len(daily)} daily records, {len(hourly)} hourly records")
    return daily


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
        # Windows strftime doesn't support %-I
        return dt_val.strftime("%I:%M%p").lower().lstrip("0")


def print_forecast(daily: pd.DataFrame, hourly: pd.DataFrame | None = None) -> None:
    """Pretty-print the 7-day forecast with solunar periods."""
    lakes = load_lakes()
    lake_names = {l.key: l.name for l in lakes}

    print("\n" + "=" * 90)
    print("  7-DAY TROPHY BASS FORECAST - INDIANA")
    print(f"  Generated: {datetime.now().strftime('%Y-%m-%d %I:%M %p')}")
    print("=" * 90)

    for d in sorted(daily["date"].unique()):
        day_data = daily[daily["date"] == d].sort_values("max_score", ascending=False)
        day_name = pd.Timestamp(d).strftime("%A %b %d")
        moon = day_data.iloc[0].get("moon_phase_name", "")
        moon_illum = day_data.iloc[0].get("moon_illumination", 0)

        # Get solunar periods from hourly data (same for all lakes on a given day, roughly)
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

        print(f"\n{'-' * 90}")
        print(f"  {day_name}  |  Moon: {moon} ({moon_illum*100:.0f}%)")
        if solunar_str:
            print(f"  Solunar:{solunar_str}")
        print(f"{'-' * 90}")
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

    # Top 3 overall picks
    top3 = daily.nlargest(3, "max_score")
    print(f"\n{'=' * 90}")
    print("  TOP 3 FISHING WINDOWS THIS WEEK:")
    print(f"{'=' * 90}")
    for i, (_, row) in enumerate(top3.iterrows(), 1):
        lake_name = lake_names.get(row["lake_key"], row["lake_key"])
        day_name = pd.Timestamp(row["date"]).strftime("%A %b %d")
        print(f"  #{i}: {lake_name} on {day_name} - Score {row['max_score']:.0f} "
              f"({row['rating']}) - Best at {_fmt_hour(row['best_hour'])}")

    print()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )
    daily = generate_live_forecast()
    if len(daily) > 0:
        # Load hourly for solunar period display
        hourly_path = DATA_DIR / "processed" / "live_forecast_hourly.parquet"
        hourly = pd.read_parquet(hourly_path) if hourly_path.exists() else None
        print_forecast(daily, hourly)
