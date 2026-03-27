"""Parse tournament / catch-data CSVs with flexible column mapping.

Supports varied column names, normalises weights and dates, fuzzy-matches
lake names to configured lake keys, and flags trophy bass.
"""

from __future__ import annotations

import logging
import re
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import DATA_DIR, LakeConfig, load_lakes, load_settings

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Column-name detection maps  (lower-cased source -> canonical name)
# ---------------------------------------------------------------------------
_COLUMN_ALIASES: dict[str, list[str]] = {
    "date": ["date", "catch_date", "tournament_date", "event_date"],
    "weight_lbs": ["weight", "weight_lbs", "lbs", "total_weight"],
    "length_in": ["length", "length_in", "inches"],
    "lake": ["lake", "lake_name", "body_of_water", "location"],
    "angler": ["angler", "name", "angler_name", "fisherman"],
    "species": ["species", "fish_species"],
    "place": ["place", "rank", "finish", "standing"],
    "big_fish": ["big_fish", "big_bass", "lunker", "biggest"],
}

# Accepted largemouth bass labels (lowered)
_BASS_LABELS = {"largemouth", "lmb", "largemouth bass", "bass"}

# ---------------------------------------------------------------------------
# Lake-name fuzzy matching
# ---------------------------------------------------------------------------
# Maps common suffixes / prefixes so "Lake Monroe" and "Monroe Lake" both
# resolve to the configured key.

_LAKE_SUFFIXES = re.compile(
    r"\b(lake|reservoir|res|res\.)\b", re.IGNORECASE
)


def _normalise_lake_token(raw: str) -> str:
    """Strip common suffixes and collapse whitespace."""
    return _LAKE_SUFFIXES.sub("", raw).strip().lower()


def build_lake_lookup(lakes: list[LakeConfig]) -> dict[str, str]:
    """Return a dict mapping lowered variants -> lake key.

    For each configured lake we generate several candidate strings so that
    user-supplied names like "Lake Monroe", "Monroe Lake", or just "Monroe"
    all resolve correctly.
    """
    lookup: dict[str, str] = {}
    for lc in lakes:
        key = lc.key.lower()
        name_lower = lc.name.lower()
        token = _normalise_lake_token(name_lower)

        # Exact matches on key and full name
        lookup[key] = key
        lookup[name_lower] = key
        lookup[token] = key

        # "Lake <token>" and "<token> Lake" variants
        lookup[f"lake {token}"] = key
        lookup[f"{token} lake"] = key
        lookup[f"{token} reservoir"] = key
    return lookup


def match_lake(raw: str, lookup: dict[str, str]) -> str | None:
    """Try to resolve *raw* to a lake key using the lookup table.

    Falls back to contains / startswith checks when an exact hit is not found.
    """
    normed = raw.strip().lower()
    if normed in lookup:
        return lookup[normed]

    # Startswith / contains fallback
    token = _normalise_lake_token(normed)
    if token in lookup:
        return lookup[token]

    for candidate, lake_key in lookup.items():
        if token and (candidate.startswith(token) or token.startswith(candidate)):
            return lake_key

    log.warning("Could not match lake name %r to any configured lake", raw)
    return None


# ---------------------------------------------------------------------------
# Weight parsing
# ---------------------------------------------------------------------------

_OZ_RE = re.compile(r"([\d.]+)\s*oz", re.IGNORECASE)
_LBS_RE = re.compile(r"([\d.]+)\s*(lbs?|pounds?)?", re.IGNORECASE)


def parse_weight(val: object) -> float | None:
    """Convert a weight value to float pounds.  Handles numeric types, and
    strings like ``"4.5 lbs"``, ``"4 lbs 8 oz"``, ``"72 oz"``."""
    if pd.isna(val):
        return None
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip()
    if not s:
        return None

    # Pure-oz value (e.g. "72 oz" with no lbs component)
    if _OZ_RE.fullmatch(s):
        return float(_OZ_RE.fullmatch(s).group(1)) / 16.0

    total = 0.0
    lbs_match = _LBS_RE.search(s)
    if lbs_match:
        total += float(lbs_match.group(1))
    oz_match = _OZ_RE.search(s)
    if oz_match:
        total += float(oz_match.group(1)) / 16.0
    return total if total > 0 else None


# ---------------------------------------------------------------------------
# Column detection
# ---------------------------------------------------------------------------

def _detect_columns(df: pd.DataFrame) -> dict[str, str]:
    """Map canonical column names -> actual DataFrame column names."""
    mapping: dict[str, str] = {}
    cols_lower = {c.strip().lower(): c for c in df.columns}
    for canonical, aliases in _COLUMN_ALIASES.items():
        for alias in aliases:
            if alias.lower() in cols_lower:
                mapping[canonical] = cols_lower[alias.lower()]
                break
    return mapping


# ---------------------------------------------------------------------------
# Species filtering
# ---------------------------------------------------------------------------

def _is_bass(val: object) -> bool:
    if pd.isna(val):
        return True  # no species column -> assume bass-only file
    return str(val).strip().lower() in _BASS_LABELS


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_tournament_file(
    filepath: str | Path,
    trophy_threshold: float = 5.0,
    lakes: list[LakeConfig] | None = None,
) -> pd.DataFrame:
    """Parse a single CSV of tournament / catch data.

    Parameters
    ----------
    filepath : path to CSV
    trophy_threshold : weight (lbs) at or above which a fish is a trophy
    lakes : optional pre-loaded lake configs; loaded from YAML if *None*

    Returns
    -------
    pd.DataFrame with columns:
        date, lake_key, weight_lbs, length_in, angler, place,
        is_trophy, source_file
    """
    filepath = Path(filepath)
    log.info("Parsing tournament file: %s", filepath)

    if lakes is None:
        lakes = load_lakes()
    lake_lookup = build_lake_lookup(lakes)

    df = pd.read_csv(filepath)
    col_map = _detect_columns(df)
    log.debug("Detected column mapping: %s", col_map)

    # --- species filter ---
    if "species" in col_map:
        mask = df[col_map["species"]].apply(_is_bass)
        df = df.loc[mask].copy()

    # --- build output frame ---
    out: dict[str, list] = {
        "date": [],
        "lake_key": [],
        "weight_lbs": [],
        "length_in": [],
        "angler": [],
        "place": [],
        "is_trophy": [],
        "source_file": [],
    }

    for _, row in df.iterrows():
        # Date
        raw_date = row.get(col_map.get("date", ""), None)
        try:
            dt = pd.to_datetime(raw_date)
        except Exception:
            dt = pd.NaT
        out["date"].append(dt)

        # Lake
        raw_lake = row.get(col_map.get("lake", ""), "")
        out["lake_key"].append(
            match_lake(str(raw_lake), lake_lookup) if pd.notna(raw_lake) else None
        )

        # Weight
        w = parse_weight(row.get(col_map.get("weight_lbs", ""), None))
        out["weight_lbs"].append(w)
        out["is_trophy"].append(w is not None and w >= trophy_threshold)

        # Length
        raw_len = row.get(col_map.get("length_in", ""), None)
        try:
            out["length_in"].append(float(raw_len) if pd.notna(raw_len) else None)
        except (ValueError, TypeError):
            out["length_in"].append(None)

        # Angler
        out["angler"].append(
            str(row.get(col_map.get("angler", ""), "")).strip() or None
        )

        # Place
        raw_place = row.get(col_map.get("place", ""), None)
        try:
            out["place"].append(int(raw_place) if pd.notna(raw_place) else None)
        except (ValueError, TypeError):
            out["place"].append(None)

        out["source_file"].append(filepath.name)

    result = pd.DataFrame(out)
    result["date"] = pd.to_datetime(result["date"])
    log.info(
        "Parsed %d records (%d trophies) from %s",
        len(result),
        result["is_trophy"].sum(),
        filepath.name,
    )
    return result


def parse_all_tournament_files(
    directory: str | Path,
    trophy_threshold: float = 5.0,
) -> pd.DataFrame:
    """Scan *directory* for CSVs, parse each, concatenate, save as Parquet.

    The combined DataFrame is written to ``data/processed/catches.parquet``.
    """
    directory = Path(directory)
    csv_files = sorted(directory.glob("*.csv"))
    if not csv_files:
        log.warning("No CSV files found in %s", directory)
        return pd.DataFrame()

    lakes = load_lakes()
    frames = [
        parse_tournament_file(f, trophy_threshold=trophy_threshold, lakes=lakes)
        for f in csv_files
    ]
    combined = pd.concat(frames, ignore_index=True)

    # Persist
    out_dir = DATA_DIR / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "catches.parquet"
    combined.to_parquet(out_path, index=False)
    log.info("Saved %d records to %s", len(combined), out_path)

    _print_summary(combined)
    return combined


# ---------------------------------------------------------------------------
# Sample-data generator
# ---------------------------------------------------------------------------

_ANGLER_FIRST = [
    "Mike", "Steve", "Dave", "Chris", "Jeff", "Matt", "Tom", "Brian",
    "Jason", "Kevin", "Ryan", "Dan", "Mark", "Joe", "Bob", "Jim",
    "Kyle", "Scott", "Eric", "Tyler",
]
_ANGLER_LAST = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Miller", "Davis",
    "Wilson", "Anderson", "Thomas", "Martin", "Thompson", "White", "Harris",
    "Clark", "Lewis", "Walker", "Hall", "Young", "King",
]


def generate_sample_data(
    lakes: list[LakeConfig] | None = None,
    start_year: int = 2018,
    end_year: int = 2024,
    n_records: int = 500,
    seed: int = 42,
) -> pd.DataFrame:
    """Create realistic synthetic tournament catch data.

    Seasonal patterns:
    - More catches in spring (Mar-May) and fall (Sep-Nov).
    - Pre-spawn (Feb-Apr) produces heavier average weights.
    """
    rng = np.random.default_rng(seed)

    if lakes is None:
        lakes = load_lakes()
    lake_keys = [lc.key for lc in lakes]

    records: list[dict] = []
    for _ in range(n_records):
        year = rng.integers(start_year, end_year + 1)

        # Seasonal weighting: spring & fall more likely
        month_weights = np.array([
            0.03,  # Jan
            0.05,  # Feb
            0.12,  # Mar
            0.15,  # Apr
            0.13,  # May
            0.08,  # Jun
            0.06,  # Jul
            0.05,  # Aug
            0.10,  # Sep
            0.11,  # Oct
            0.08,  # Nov
            0.04,  # Dec
        ])
        month_weights /= month_weights.sum()
        month = rng.choice(np.arange(1, 13), p=month_weights)

        # Day (clamp to valid range)
        max_day = 28 if month == 2 else (30 if month in (4, 6, 9, 11) else 31)
        day = int(rng.integers(1, max_day + 1))

        catch_date = date(int(year), int(month), day)

        # Weight — base ~2.5 lbs, pre-spawn bump
        base_weight = 2.5
        if month in (2, 3, 4):
            base_weight = 3.2  # pre-spawn heavier
        weight = float(rng.lognormal(np.log(base_weight), 0.35))
        weight = round(max(0.5, min(weight, 12.0)), 2)

        # Length derived loosely from weight (W = c * L^3 approximation)
        length = round(float(7.0 + 3.0 * weight ** 0.45 + rng.normal(0, 0.5)), 1)
        length = max(8.0, min(length, 28.0))

        lake = rng.choice(lake_keys)
        first = rng.choice(_ANGLER_FIRST)
        last = rng.choice(_ANGLER_LAST)
        angler = f"{first} {last}"
        place = int(rng.integers(1, 51))

        records.append(
            {
                "date": catch_date,
                "lake_key": lake,
                "weight_lbs": weight,
                "length_in": length,
                "angler": angler,
                "place": place,
                "is_trophy": weight >= 5.0,
                "source_file": "synthetic",
            }
        )

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    log.info("Generated %d synthetic records (%d-%d)", len(df), start_year, end_year)
    return df


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _print_summary(df: pd.DataFrame) -> None:
    """Print human-readable stats for a catches DataFrame."""
    total = len(df)
    trophies = int(df["is_trophy"].sum())
    print(f"\n{'='*50}")
    print("  Tournament catch summary")
    print(f"{'='*50}")
    print(f"  Total records : {total}")
    print(f"  Trophy catches: {trophies}  ({100*trophies/max(total,1):.1f}%)")
    if "lake_key" in df.columns:
        print("\n  Per-lake breakdown:")
        breakdown = (
            df.groupby("lake_key")
            .agg(catches=("weight_lbs", "size"), trophies=("is_trophy", "sum"))
            .sort_values("catches", ascending=False)
        )
        for lake, row in breakdown.iterrows():
            c, t = int(row["catches"]), int(row["trophies"])
            print(f"    {lake:<20s}  catches={c:>4d}  trophies={t:>3d}")
    print(f"{'='*50}\n")


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    lakes = load_lakes()
    settings = load_settings()

    df = generate_sample_data(
        lakes=lakes,
        start_year=settings.start_year,
        end_year=settings.end_year,
    )

    out_dir = DATA_DIR / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "catches.parquet"
    df.to_parquet(out_path, index=False)
    print(f"Saved synthetic data to {out_path}")

    _print_summary(df)
