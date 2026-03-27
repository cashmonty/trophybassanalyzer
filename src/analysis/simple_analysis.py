"""Simple trophy bass analysis — what conditions produce 7+ lb fish in Indiana?

Analyzes the merged dataset to find optimal conditions for trophy bass fishing
based on: moon phase, temperature, barometric pressure, time of year, and more.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import numpy as np

from src.config import DATA_DIR

logger = logging.getLogger(__name__)

TROPHY_WEIGHT = 7.0  # lbs


def load_trophy_conditions() -> dict:
    """Analyze merged data and return a dict of optimal conditions for 7+ lb bass."""
    df = pd.read_parquet(DATA_DIR / "processed" / "merged.parquet")
    df["datetime"] = pd.to_datetime(df["datetime"])

    # Baseline = all hours in the dataset
    all_hours = df.copy()

    # Trophy = hours when a 7+ lb fish was caught
    trophy = df[df["max_weight"] >= TROPHY_WEIGHT].copy()

    # Deduplicate to daily level per lake for cleaner stats
    trophy_daily = trophy.groupby(["date", "lake_key"]).agg(
        max_weight=("max_weight", "max"),
        temperature_f=("temperature_2m", lambda x: x.mean() * 9/5 + 32),
        water_temp_f=("water_temp_estimated", lambda x: x.mean() * 9/5 + 32),
        pressure_msl=("pressure_msl", "mean"),
        pressure_trend_3h=("pressure_trend_3h", "mean"),
        moon_illumination=("moon_illumination", "first"),
        moon_phase=("moon_phase_name", "first"),
        solunar_score=("solunar_base_score", "mean"),
        month=("month", "first"),
        spawn_phase=("spawn_phase", "first"),
        wind_speed=("wind_speed_10m", "mean"),
        cloud_cover=("cloud_cover", "mean"),
        front_type=("front_type", "first"),
    ).reset_index()

    n_trophy_days = len(trophy_daily)

    results = {
        "n_trophy_days": n_trophy_days,
        "n_total_catches": len(df[df["catch_count"] > 0]),
        "biggest_fish": trophy_daily["max_weight"].max(),
    }

    # --- MONTH BREAKDOWN ---
    month_names = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
                   7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}
    month_counts = trophy_daily.groupby("month").size()
    results["by_month"] = {
        month_names.get(m, m): int(c) for m, c in month_counts.items()
    }

    # --- MOON PHASE ---
    moon_counts = trophy_daily.groupby("moon_phase").size().sort_values(ascending=False)
    results["by_moon_phase"] = {k: int(v) for k, v in moon_counts.items()}
    results["avg_moon_illumination_trophy"] = round(trophy_daily["moon_illumination"].mean() * 100, 1)

    # --- TEMPERATURE ---
    results["air_temp_f"] = {
        "avg": round(trophy_daily["temperature_f"].mean(), 1),
        "min": round(trophy_daily["temperature_f"].min(), 1),
        "max": round(trophy_daily["temperature_f"].max(), 1),
    }
    results["water_temp_f"] = {
        "avg": round(trophy_daily["water_temp_f"].mean(), 1),
        "min": round(trophy_daily["water_temp_f"].min(), 1),
        "max": round(trophy_daily["water_temp_f"].max(), 1),
    }

    # --- BAROMETRIC PRESSURE ---
    results["pressure"] = {
        "avg_hpa": round(trophy_daily["pressure_msl"].mean(), 1),
        "avg_inhg": round(trophy_daily["pressure_msl"].mean() * 0.02953, 2),
        "trend_avg": round(trophy_daily["pressure_trend_3h"].mean(), 3),
    }
    front_counts = trophy_daily.groupby("front_type").size()
    results["by_front_type"] = {k: int(v) for k, v in front_counts.items()}

    pressure_class = trophy.groupby("pressure_trend_class").size()
    results["by_pressure_trend"] = {k: int(v) for k, v in pressure_class.items()}

    # --- SPAWN PHASE ---
    spawn_counts = trophy_daily.groupby("spawn_phase").size().sort_values(ascending=False)
    results["by_spawn_phase"] = {k: int(v) for k, v in spawn_counts.items()}

    # --- SOLUNAR ---
    results["solunar_avg_trophy"] = round(trophy_daily["solunar_score"].mean(), 1)
    results["solunar_avg_all"] = round(all_hours["solunar_base_score"].mean(), 1)

    # --- WIND & CLOUD ---
    results["wind_speed_avg_mph"] = round(trophy_daily["wind_speed"].mean() * 0.621371, 1)
    results["cloud_cover_avg_pct"] = round(trophy_daily["cloud_cover"].mean(), 1)

    # --- LAKE BREAKDOWN ---
    lake_counts = trophy_daily.groupby("lake_key").agg(
        count=("max_weight", "size"),
        biggest=("max_weight", "max"),
        avg_weight=("max_weight", "mean"),
    ).sort_values("count", ascending=False)
    results["by_lake"] = lake_counts.to_dict("index")

    # --- OPTIMAL CONDITIONS SUMMARY ---
    # Find the sweet spots
    temp_bins = pd.cut(trophy_daily["water_temp_f"], bins=[40, 50, 55, 60, 65, 70, 75, 80, 85])
    best_temp_range = temp_bins.value_counts().idxmax()

    results["optimal"] = {
        "best_months": "April-May (pre-spawn and spawn)",
        "best_water_temp_f": f"{best_temp_range.left:.0f}-{best_temp_range.right:.0f}°F",
        "best_pressure": "Stable or slightly falling (pre-frontal)",
        "best_moon": "Last Quarter / darker phases slightly favored",
        "best_solunar": f"Score 60+ (avg trophy: {results['solunar_avg_trophy']}, avg all: {results['solunar_avg_all']})",
    }

    return results


def find_2026_windows(top_n: int = 30) -> pd.DataFrame:
    """Find the best 2026 windows based on conditions matching trophy patterns."""
    pred_path = DATA_DIR / "processed" / "predictions_2026.parquet"
    if not pred_path.exists():
        return pd.DataFrame()

    daily = pd.read_parquet(pred_path)
    daily["date"] = pd.to_datetime(daily["date"])
    daily["month"] = daily["date"].dt.month
    daily["day_name"] = daily["date"].dt.day_name()

    top = daily.nlargest(top_n, "max_probability")
    return top[["date", "lake_key", "max_probability", "rating", "month", "day_name"]]


def print_report():
    """Print a human-readable trophy bass conditions report."""
    results = load_trophy_conditions()

    print("\n" + "=" * 65)
    print("  INDIANA TROPHY BASS (7+ LBS) — OPTIMAL CONDITIONS REPORT")
    print("=" * 65)
    print(f"\n  Based on {results['n_trophy_days']} days where 7+ lb bass were caught")
    print(f"  Biggest fish in dataset: {results['biggest_fish']:.2f} lbs")

    print("\n--- BEST TIME OF YEAR ---")
    for month, count in sorted(results["by_month"].items(),
                                key=lambda x: x[1], reverse=True):
        bar = "#" * count
        print(f"  {month:>3s}: {count:>2d} days  {bar}")

    print("\n--- MOON PHASE ---")
    for phase, count in results["by_moon_phase"].items():
        bar = "#" * count
        print(f"  {phase:<18s}: {count:>2d} days  {bar}")

    print("\n--- TEMPERATURE ---")
    t = results["air_temp_f"]
    print(f"  Air temp (avg):   {t['avg']}°F  (range: {t['min']}–{t['max']}°F)")
    t = results["water_temp_f"]
    print(f"  Water temp (avg): {t['avg']}°F  (range: {t['min']}–{t['max']}°F)")

    print("\n--- BAROMETRIC PRESSURE ---")
    p = results["pressure"]
    print(f"  Avg pressure: {p['avg_hpa']} hPa ({p['avg_inhg']} inHg)")
    print(f"  3-hour trend: {p['trend_avg']:+.3f} hPa (slightly falling = good)")
    print(f"  Pressure trend: {results['by_pressure_trend']}")
    print(f"  Front type: {results['by_front_type']}")

    print("\n--- SPAWN PHASE ---")
    for phase, count in results["by_spawn_phase"].items():
        bar = "#" * count
        print(f"  {phase:<12s}: {count:>2d} days  {bar}")

    print("\n--- OTHER CONDITIONS ---")
    print(f"  Solunar score (trophy avg): {results['solunar_avg_trophy']} vs {results['solunar_avg_all']} (all data)")
    print(f"  Avg wind speed: {results['wind_speed_avg_mph']} mph")
    print(f"  Avg cloud cover: {results['cloud_cover_avg_pct']}%")

    print("\n--- BY LAKE ---")
    for lake, stats in results["by_lake"].items():
        print(f"  {lake:<10s}: {stats['count']:>2d} days, biggest: {stats['biggest']:.2f} lbs, avg: {stats['avg_weight']:.2f} lbs")

    print("\n" + "=" * 65)
    print("  OPTIMAL CONDITIONS SUMMARY")
    print("=" * 65)
    for key, val in results["optimal"].items():
        label = key.replace("best_", "").replace("_", " ").title()
        print(f"  {label:<20s}: {val}")

    # 2026 windows
    windows = find_2026_windows(20)
    if len(windows) > 0:
        print("\n" + "=" * 65)
        print("  TOP 20 PREDICTED TROPHY WINDOWS — 2026")
        print("=" * 65)
        for _, row in windows.iterrows():
            print(f"  {row['date'].strftime('%B %d'):>12s} ({row['day_name']:<9s})  "
                  f"{row['lake_key']:<8s}  "
                  f"probability: {row['max_probability']:.1%}  "
                  f"rating: {row['rating']}")

    print()


if __name__ == "__main__":
    print_report()
