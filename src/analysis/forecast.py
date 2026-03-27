"""Generate 2026 trophy bass predictions using condition scoring + astronomical data.

Instead of relying solely on a sparse ML model (45 labeled trophies isn't enough),
we use a composite scoring system based on known trophy bass science:
  - Water temperature sweet spot (pre-spawn 48-62°F is prime)
  - Solunar/moon phase activity
  - Barometric pressure trend
  - Seasonal timing
  - Warming trends
  - Wind direction favorability

The ML model supplements this when available, but the scoring system produces
meaningful, actionable predictions on its own.
"""

from __future__ import annotations

import logging
import math
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

from src.config import DATA_DIR, load_lakes

logger = logging.getLogger(__name__)


def get_climatological_weather(df_historical: pd.DataFrame, year: int = 2026) -> pd.DataFrame:
    """Generate climatological averages for the forecast year."""
    df = df_historical.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["day_of_year"] = df["datetime"].dt.dayofyear
    df["hour"] = df["datetime"].dt.hour

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    groupby_keys = ["lake_key", "day_of_year", "hour"]
    exclude = [
        "catch_count", "trophy_count", "trophy_caught",
        "max_weight", "avg_weight", "year", "super_trophy_count",
    ]
    numeric_cols = [c for c in numeric_cols if c not in exclude and c not in groupby_keys]

    clim = df.groupby(["lake_key", "day_of_year", "hour"])[numeric_cols].mean().reset_index()

    rows = []
    start = date(year, 1, 1)
    end = date(year, 12, 31)
    current = start
    while current <= end:
        doy = current.timetuple().tm_yday
        for hour in range(24):
            rows.append({
                "date": current,
                "datetime": datetime(year, current.month, current.day, hour),
                "day_of_year": doy,
                "hour": hour,
            })
        current += timedelta(days=1)

    forecast_base = pd.DataFrame(rows)

    lakes = load_lakes()
    all_forecasts = []
    for lake in lakes:
        lake_clim = clim[clim["lake_key"] == lake.key].copy()
        if len(lake_clim) == 0:
            continue
        merged = forecast_base.merge(
            lake_clim, on=["day_of_year", "hour"], how="left", suffixes=("", "_clim")
        )
        merged["lake_key"] = lake.key
        all_forecasts.append(merged)

    if not all_forecasts:
        return pd.DataFrame()

    result = pd.concat(all_forecasts, ignore_index=True)
    result["month"] = pd.to_datetime(result["datetime"]).dt.month
    return result


def _compute_2026_astro() -> pd.DataFrame:
    """Compute actual 2026 astronomical data for all lakes."""
    from src.ingest.astro import compute_astro_for_lake

    lakes = load_lakes()
    start = date(2026, 1, 1)
    end = date(2026, 12, 31)

    frames = []
    for lake in lakes:
        logger.info(f"Computing 2026 astro for {lake.key}...")
        df = compute_astro_for_lake(lake.key, lake.lat, lake.lon, start, end)
        frames.append(df)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _score_water_temp(temp_f: float) -> float:
    """Score water temperature for trophy bass (0-25 points).

    Sweet spot: 48-62°F (pre-spawn staging through early spawn).
    Good: 62-72°F (spawn/post-spawn).
    Acceptable: 72-82°F (summer, dawn/dusk bite).
    Poor: below 44 or above 85.
    """
    if pd.isna(temp_f):
        return 10.0  # neutral
    if 52 <= temp_f <= 62:
        return 25.0  # Prime pre-spawn/spawn
    elif 48 <= temp_f < 52:
        return 20.0  # Early pre-spawn
    elif 62 < temp_f <= 68:
        return 20.0  # Spawn/post-spawn
    elif 68 < temp_f <= 75:
        return 15.0  # Early summer
    elif 44 <= temp_f < 48:
        return 12.0  # Late winter movement
    elif 75 < temp_f <= 82:
        return 10.0  # Summer — still catchable
    elif temp_f > 82:
        return 5.0   # Hot — tough bite
    else:
        return 3.0   # Cold — very tough


def _score_season(month: int, day_of_year: int) -> float:
    """Score seasonal timing for trophy bass (0-20 points).

    April-May = peak pre-spawn/spawn (biggest fish of the year).
    September-October = fall feed-up (second best window).
    """
    if month == 4:
        return 20.0
    elif month == 5:
        return 18.0
    elif month == 3 and day_of_year > 75:  # Late March
        return 15.0
    elif month == 10:
        return 15.0
    elif month == 9:
        return 14.0
    elif month == 6:
        return 12.0
    elif month in (7, 8):
        return 8.0
    elif month == 11:
        return 6.0
    else:
        return 3.0  # Dec-Feb


def _score_solunar(solunar_score: float) -> float:
    """Score solunar activity (0-20 points)."""
    if pd.isna(solunar_score):
        return 10.0
    # Solunar score is 0-100, map to 0-20
    return min(solunar_score / 100.0 * 20.0, 20.0)


def _score_pressure(pressure_hpa: float, pressure_trend: float) -> float:
    """Score barometric pressure conditions (0-15 points).

    Falling pressure (pre-frontal) = best.
    Stable high = good.
    Rising rapidly (post-frontal) = worst.
    """
    score = 7.5  # baseline

    if pd.isna(pressure_trend):
        return score

    # Slowly falling pressure = pre-frontal feeding window
    if -2.0 < pressure_trend < -0.3:
        score += 7.5  # Best — pre-frontal feed
    elif -0.3 <= pressure_trend <= 0.3:
        score += 4.0  # Stable — decent
    elif 0.3 < pressure_trend < 1.5:
        score += 1.0  # Slowly rising — OK
    elif pressure_trend <= -2.0:
        score += 3.0  # Rapidly falling — front arriving, still good
    else:
        score -= 2.0  # Rapidly rising — post-frontal lockjaw

    return max(0, min(score, 15.0))


def _score_wind(wind_speed_kmh: float, wind_class: str = "unknown") -> float:
    """Score wind conditions (0-10 points).

    Light-moderate south/SW wind = best for trophy bass.
    Calm = decent (sight fishing conditions).
    Strong NW = post-frontal, poor.
    """
    if pd.isna(wind_speed_kmh):
        return 5.0

    wind_mph = wind_speed_kmh * 0.621371

    # Speed scoring
    if 5 <= wind_mph <= 15:
        speed_score = 5.0  # Perfect chop
    elif wind_mph < 5:
        speed_score = 3.5  # Calm — OK for sight fishing
    elif 15 < wind_mph <= 25:
        speed_score = 3.0  # Windy but fishable
    else:
        speed_score = 1.0  # Too windy

    # Direction scoring
    if wind_class == "south_warm":
        dir_score = 5.0
    elif wind_class in ("variable", "unknown"):
        dir_score = 3.0
    elif wind_class == "northwest_cold":
        dir_score = 1.0
    elif wind_class == "east_poor":
        dir_score = 1.5
    else:
        dir_score = 2.5

    return min(speed_score + dir_score, 10.0)


def _score_cloud_cover(cloud_pct: float) -> float:
    """Score cloud cover (0-10 points). Overcast = better for trophies."""
    if pd.isna(cloud_pct):
        return 5.0
    if cloud_pct >= 70:
        return 10.0  # Overcast — bass roam and feed shallow
    elif cloud_pct >= 40:
        return 7.0   # Partly cloudy — good
    else:
        return 4.0   # Sunny — fish tight to cover


def compute_trophy_score(row: pd.Series) -> float:
    """Compute composite trophy bass score (0-100) for a single hourly record."""
    temp_f = row.get("water_temp_f_est")
    if pd.isna(temp_f) and "water_temp_estimated" in row.index:
        temp_c = row.get("water_temp_estimated")
        temp_f = temp_c * 9 / 5 + 32 if not pd.isna(temp_c) else None

    score = 0.0
    score += _score_water_temp(temp_f)
    score += _score_season(row.get("month", 1), row.get("day_of_year", 1))
    score += _score_solunar(row.get("solunar_base_score"))
    score += _score_pressure(row.get("pressure_msl"), row.get("pressure_trend_3h"))
    score += _score_wind(row.get("wind_speed_10m"), row.get("wind_class", "unknown"))
    score += _score_cloud_cover(row.get("cloud_cover"))

    return min(score, 100.0)


def generate_2026_predictions(df_historical: pd.DataFrame) -> pd.DataFrame:
    """Generate trophy probability predictions for all of 2026."""
    logger.info("Generating 2026 forecast base from climatological averages...")
    forecast_df = get_climatological_weather(df_historical, year=2026)

    if len(forecast_df) == 0:
        logger.error("No forecast data generated")
        return pd.DataFrame()

    # Compute and merge REAL 2026 astronomical data
    logger.info("Computing actual 2026 astronomical data...")
    astro_2026 = _compute_2026_astro()

    if len(astro_2026) > 0:
        astro_2026["date"] = pd.to_datetime(astro_2026["date"]).dt.date
        astro_cols = ["date", "lake_key", "moon_illumination", "solunar_base_score",
                      "moon_phase_name"]
        astro_merge = astro_2026[astro_cols].copy()

        # Drop averaged solunar/moon values, replace with real 2026 values
        for col in ["moon_illumination", "solunar_base_score"]:
            if col in forecast_df.columns:
                forecast_df.drop(columns=[col], inplace=True)

        forecast_df["date"] = pd.to_datetime(forecast_df["date"]).values.astype("datetime64[D]")
        astro_merge["date"] = pd.to_datetime(astro_merge["date"]).values.astype("datetime64[D]")

        forecast_df = forecast_df.merge(
            astro_merge, on=["date", "lake_key"], how="left"
        )
        logger.info("Merged real 2026 astro data into forecast")

    # Compute water temp in F for scoring
    if "water_temp_estimated" in forecast_df.columns:
        forecast_df["water_temp_f_est"] = forecast_df["water_temp_estimated"] * 9 / 5 + 32
    elif "temperature_2m" in forecast_df.columns:
        forecast_df["water_temp_f_est"] = forecast_df["temperature_2m"] * 9 / 5 + 32

    # Score each hourly record
    logger.info("Computing trophy scores for %d hourly records...", len(forecast_df))
    forecast_df["trophy_score"] = forecast_df.apply(compute_trophy_score, axis=1)

    # Aggregate to daily level
    forecast_df["date"] = pd.to_datetime(forecast_df["datetime"]).dt.date
    daily = forecast_df.groupby(["date", "lake_key"]).agg(
        max_score=("trophy_score", "max"),
        mean_score=("trophy_score", "mean"),
        best_hour=("trophy_score", "idxmax"),
        max_probability=("trophy_score", lambda x: x.max() / 100.0),  # Normalize to 0-1
        mean_probability=("trophy_score", lambda x: x.mean() / 100.0),
    ).reset_index()

    # Get best hour as actual hour value
    daily["best_hour"] = forecast_df.loc[daily["best_hour"].values, "hour"].values

    # Pull in moon phase for display
    if "moon_phase_name" in forecast_df.columns:
        moon_daily = forecast_df.groupby(["date", "lake_key"])["moon_phase_name"].first().reset_index()
        daily = daily.merge(moon_daily, on=["date", "lake_key"], how="left")

    # Rating based on score (out of 100)
    daily["rating"] = pd.cut(
        daily["max_score"],
        bins=[0, 35, 50, 65, 80, 100],
        labels=["Poor", "Fair", "Good", "Great", "Excellent"],
    )

    # Save
    output_path = DATA_DIR / "processed" / "predictions_2026.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(output_path, index=False)

    hourly_path = DATA_DIR / "processed" / "predictions_2026_hourly.parquet"
    forecast_df.to_parquet(hourly_path, index=False)

    logger.info(f"Saved 2026 predictions: {len(daily)} daily records")

    # Summary
    top_windows = daily.nlargest(20, "max_score")
    logger.info(f"\nTop 20 predicted trophy windows for 2026:")
    for _, row in top_windows.iterrows():
        moon = row.get("moon_phase_name", "")
        logger.info(
            f"  {row['date']}  {row['lake_key']:<12s}  "
            f"score={row['max_score']:.0f}  rating={row['rating']}  "
            f"best_hour={row['best_hour']}  moon={moon}"
        )

    return daily


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    merged_path = DATA_DIR / "processed" / "merged.parquet"
    if not merged_path.exists():
        print("Run merge pipeline first.")
    else:
        df = pd.read_parquet(merged_path)
        predictions = generate_2026_predictions(df)
        print(f"\nGenerated {len(predictions)} daily predictions for 2026")

        # Show score distribution
        print(f"\nScore distribution:")
        print(predictions["max_score"].describe())
        print(f"\nRating counts:")
        print(predictions["rating"].value_counts().sort_index())
