"""Parse USA Bassin tournament standings into catch records for all lakes.

The source data is season-level standings (one row per team per season).
Each row includes a 'big_fish_lbs' column — the team's largest bass that season.

Since exact tournament dates aren't available for season standings, we distribute
catches across the typical tournament season (April–October) using a weighted
seasonal profile. Individual event data (with exact dates) is merged when available.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import DATA_DIR, load_lakes, load_settings

logger = logging.getLogger(__name__)

RAW_DIR = DATA_DIR / "raw" / "tournaments"

# Map CSV filename patterns to lake_key
LAKE_FILE_MAP = {
    "lemon": "usabassin_lemon_raw.csv",
    "monroe": "usabassin_monroe_raw.csv",
    "geist": "usabassin_geist_raw.csv",
    "morse": "usabassin_morse_raw.csv",
}

# MegaBucks is multi-lake — we include it but can't assign a specific lake
MULTI_LAKE_FILES = ["usabassin_megabucks_raw.csv"]

# Typical tournament months and relative frequency weights
MONTH_WEIGHTS = {
    4: 0.18,   # April — pre-spawn
    5: 0.20,   # May — spawn peak
    6: 0.15,   # June — post-spawn
    7: 0.10,   # July
    8: 0.10,   # August
    9: 0.15,   # September — fall bite
    10: 0.12,  # October — fall
}


def load_all_raw_standings() -> pd.DataFrame:
    """Load and combine all raw USA Bassin CSVs."""
    frames = []

    for lake_key, filename in LAKE_FILE_MAP.items():
        path = RAW_DIR / filename
        if path.exists():
            df = pd.read_csv(path)
            df["lake_key"] = lake_key
            frames.append(df)
            logger.info(f"Loaded {len(df)} rows for {lake_key}")
        else:
            logger.warning(f"No data file for {lake_key}: {path}")

    for filename in MULTI_LAKE_FILES:
        path = RAW_DIR / filename
        if path.exists():
            df = pd.read_csv(path)
            df["lake_key"] = "multi"  # Can't assign specific lake
            frames.append(df)
            logger.info(f"Loaded {len(df)} rows from {filename}")

    if not frames:
        raise FileNotFoundError(f"No tournament data found in {RAW_DIR}")

    combined = pd.concat(frames, ignore_index=True)
    logger.info(f"Total raw standings: {len(combined)} rows")
    return combined


def load_individual_events() -> pd.DataFrame | None:
    """Load individual tournament events with exact dates if available."""
    path = RAW_DIR / "usabassin_individual_events.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    logger.info(f"Loaded {len(df)} individual event results")
    return df


def _assign_season_date(year: int, rng: np.random.Generator) -> pd.Timestamp:
    """Assign a plausible weekend date within the tournament season."""
    months = list(MONTH_WEIGHTS.keys())
    weights = np.array(list(MONTH_WEIGHTS.values()))
    month = rng.choice(months, p=weights / weights.sum())

    max_day = 30 if month in (4, 6, 9) else 31
    day = rng.integers(1, max_day + 1)

    candidate = pd.Timestamp(year=year, month=month, day=day)
    # Snap to nearest Saturday
    days_to_sat = (5 - candidate.dayofweek) % 7
    candidate += pd.Timedelta(days=days_to_sat)
    if candidate.month != month:
        candidate -= pd.Timedelta(days=7)
    return candidate


def convert_standings_to_catches(df: pd.DataFrame) -> pd.DataFrame:
    """Convert season standings into individual big-fish catch records."""
    settings = load_settings()
    catches = df[df["big_fish_lbs"] > 0].copy()

    rng = np.random.default_rng(42)

    dates = []
    for _, row in catches.iterrows():
        year = int(row["season_start_year"])
        dates.append(_assign_season_date(year, rng))

    catches["date"] = dates
    catches["weight_lbs"] = catches["big_fish_lbs"]
    catches["length_in"] = np.nan
    catches["angler"] = (
        catches["angler_1"].fillna("") + " / " + catches["angler_2"].fillna("")
    ).str.strip(" /")
    catches["is_trophy"] = catches["weight_lbs"] >= settings.trophy_weight_lbs
    catches["is_super_trophy"] = (
        catches["weight_lbs"] >= settings.super_trophy_weight_lbs
    )
    catches["source_type"] = "season_standings"

    return catches[[
        "date", "lake_key", "weight_lbs", "length_in",
        "angler", "place", "is_trophy", "is_super_trophy",
        "source_type", "season_start_year",
    ]].copy()


def convert_events_to_catches(df: pd.DataFrame) -> pd.DataFrame:
    """Convert individual event results into catch records."""
    settings = load_settings()
    catches = df[df["big_fish_lbs"] > 0].copy()

    catches["date"] = pd.to_datetime(catches["event_date"])
    catches["weight_lbs"] = catches["big_fish_lbs"]
    catches["length_in"] = np.nan
    catches["angler"] = (
        catches["angler_1"].fillna("") + " / " + catches["angler_2"].fillna("")
    ).str.strip(" /")
    catches["is_trophy"] = catches["weight_lbs"] >= settings.trophy_weight_lbs
    catches["is_super_trophy"] = (
        catches["weight_lbs"] >= settings.super_trophy_weight_lbs
    )
    catches["source_type"] = "individual_event"
    catches["season_start_year"] = catches["date"].dt.year

    return catches[[
        "date", "lake_key", "weight_lbs", "length_in",
        "angler", "place", "is_trophy", "is_super_trophy",
        "source_type", "season_start_year",
    ]].copy()


def build_all_catches() -> pd.DataFrame:
    """Build unified catch dataset from all sources."""
    frames = []

    # Season standings
    standings = load_all_raw_standings()
    standings_catches = convert_standings_to_catches(standings)
    frames.append(standings_catches)
    logger.info(f"Season standings: {len(standings_catches)} catches")

    # Individual events (exact dates)
    events = load_individual_events()
    if events is not None and len(events) > 0:
        event_catches = convert_events_to_catches(events)
        frames.append(event_catches)
        logger.info(f"Individual events: {len(event_catches)} catches")

    result = pd.concat(frames, ignore_index=True)
    result["date"] = pd.to_datetime(result["date"])
    result = result.sort_values("date").reset_index(drop=True)

    return result


def save_catches() -> Path:
    """Build and save all catches to Parquet."""
    settings = load_settings()
    df = build_all_catches()

    out_path = DATA_DIR / "processed" / "catches.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)

    # Summary
    print(f"\n{'='*60}")
    print(f"USA Bassin — All Lakes Catch Summary")
    print(f"{'='*60}")
    print(f"Total big fish records:  {len(df)}")
    tw = settings.trophy_weight_lbs
    print(f"Trophy fish ({tw}+ lbs):   {df['is_trophy'].sum()}")
    st = settings.super_trophy_weight_lbs
    print(f"Super trophy ({st}+ lbs): {df['is_super_trophy'].sum()}")
    print(f"Biggest fish:            {df['weight_lbs'].max():.2f} lbs")
    print(f"Avg big fish weight:     {df['weight_lbs'].mean():.2f} lbs")
    print(f"Date range:              {df['date'].min().date()} to {df['date'].max().date()}")

    print(f"\nPer-lake breakdown:")
    for lake_key in sorted(df["lake_key"].unique()):
        ldf = df[df["lake_key"] == lake_key]
        print(
            f"  {lake_key:<10s}  "
            f"records={len(ldf):>4d}  "
            f"trophies={ldf['is_trophy'].sum():>3d}  "
            f"supers={ldf['is_super_trophy'].sum():>2d}  "
            f"max={ldf['weight_lbs'].max():.2f}"
        )

    by_source = df["source_type"].value_counts()
    print(f"\nBy source: {by_source.to_dict()}")
    print(f"{'='*60}\n")

    return out_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    save_catches()
