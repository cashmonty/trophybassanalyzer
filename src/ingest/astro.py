"""Astronomical data computation for Indiana lakes.

Computes moon phases, solunar feeding periods, and sunrise/sunset times
using the ``ephem`` library.  No external API calls required -- everything
is calculated locally from orbital mechanics.

Timezone: America/Indiana/Indianapolis
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import ephem
import pandas as pd
from tqdm import tqdm

from src.config import DATA_DIR, LakeConfig, load_lakes, load_settings

log = logging.getLogger(__name__)

TZ = ZoneInfo("America/Indiana/Indianapolis")
ASTRO_DIR = DATA_DIR / "raw" / "astro"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ephem_date(d: date, hour: float = 0.0) -> ephem.Date:
    """Convert a Python date (+ optional fractional hour) to an ephem.Date in UTC."""
    dt = datetime(d.year, d.month, d.day, tzinfo=TZ) + timedelta(hours=hour)
    utc_dt = dt.astimezone(ZoneInfo("UTC"))
    return ephem.Date(utc_dt)


def _ephem_to_local(edate: ephem.Date | None) -> datetime | None:
    """Convert an ephem.Date (UTC) to a local datetime, or None."""
    if edate is None:
        return None
    utc_dt = ephem.Date(edate).datetime().replace(tzinfo=ZoneInfo("UTC"))
    return utc_dt.astimezone(TZ)


def _moon_phase_name(illumination: float, is_waxing: bool) -> str:
    """Return a human-readable moon-phase name.

    Parameters
    ----------
    illumination : float
        0.0 -- 1.0 fraction illuminated.
    is_waxing : bool
        True when the moon's illumination is increasing day-over-day.
    """
    if illumination < 0.02:
        return "New"
    if illumination > 0.98:
        return "Full"
    if is_waxing:
        if illumination < 0.35:
            return "Waxing Crescent"
        if illumination < 0.65:
            return "First Quarter"
        return "Waxing Gibbous"
    else:
        if illumination < 0.35:
            return "Waning Crescent"
        if illumination < 0.65:
            return "Last Quarter"
        return "Waning Gibbous"


def _safe_rise_set(observer: ephem.Observer, body: ephem.Body,
                   func_name: str) -> datetime | None:
    """Call observer rising/setting helpers, returning None on circumpolar errors."""
    try:
        result = getattr(observer, func_name)(body)
        return _ephem_to_local(result)
    except (ephem.NeverUpError, ephem.AlwaysUpError):
        return None


def _clamp_period(
    center: datetime | None, half_hours: float,
) -> tuple[datetime | None, datetime | None]:
    """Return (start, end) clamped around *center* +/- *half_hours*."""
    if center is None:
        return None, None
    delta = timedelta(hours=half_hours)
    return center - delta, center + delta


def _solunar_base_score(moon_illumination: float,
                        major1_start: datetime | None,
                        major2_start: datetime | None,
                        sunrise: datetime | None,
                        sunset: datetime | None) -> int:
    """Compute a 0-100 solunar base score for bass fishing.

    Revised scoring based on actual bass tournament data patterns:

    Scoring breakdown (max 100):
        - Moon phase bonus:  0-25 pts  (peaks at New, Full, AND quarters)
        - Major period overlap with dawn/dusk: 0-20 pts each (x2 = 40 max)
        - Minor period count: 0-5 pts  (more feeding windows = better)
        - Base activity:     30 pts    (fish always feed to some degree)

    Key insight: Quarter moons (first/last quarter) produce excellent bass
    fishing because the gravitational pull creates strong tidal-like currents
    even in freshwater. New and Full remain best, but quarters shouldn't be
    penalized to near-zero like the old cosine curve did.
    """
    import math
    score = 30  # base

    # Moon-phase bonus: peaks at New (0), Full (1), with secondary peaks at quarters (0.5)
    # Use a double-frequency cosine: peaks at 0, 0.5, and 1.0
    # cos(4*pi*illum) gives peaks at 0, 0.25, 0.5, 0.75, 1.0
    # But we want New/Full strongest, quarters moderate
    primary = 15 * (0.5 * (1 + math.cos(math.pi * (2 * moon_illumination - 1))))  # New+Full
    secondary = 10 * (0.5 * (1 + math.cos(4 * math.pi * moon_illumination)))  # quarters too
    score += int(round(primary + secondary))

    # Dawn/dusk overlap bonus for each major period
    if sunrise and sunset:
        dawn_start = sunrise - timedelta(hours=1)
        dawn_end = sunrise + timedelta(hours=1)
        dusk_start = sunset - timedelta(hours=1)
        dusk_end = sunset + timedelta(hours=1)

        for mp_start in (major1_start, major2_start):
            if mp_start is None:
                continue
            mp_end = mp_start + timedelta(hours=2)
            overlaps_dawn = mp_start < dawn_end and mp_end > dawn_start
            overlaps_dusk = mp_start < dusk_end and mp_end > dusk_start
            if overlaps_dawn or overlaps_dusk:
                score += 20

    return min(score, 100)


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def _compute_day(observer: ephem.Observer, d: date) -> dict:
    """Compute all astronomical data for a single observer + date."""
    # Set observer date to local noon (converted to UTC) for consistent results
    observer.date = _ephem_date(d, hour=12.0)

    moon = ephem.Moon()
    moon.compute(observer)

    moon_illumination = moon.phase / 100.0  # ephem gives 0-100

    # Determine waxing/waning by comparing illumination with next day
    tomorrow_observer = observer.copy()
    tomorrow_observer.date = _ephem_date(d + timedelta(days=1), hour=12.0)
    moon_tomorrow = ephem.Moon()
    moon_tomorrow.compute(tomorrow_observer)
    is_waxing = (moon_tomorrow.phase / 100.0) > moon_illumination

    phase_name = _moon_phase_name(moon_illumination, is_waxing)

    # Reset to start of day for rise/set calculations
    observer.date = _ephem_date(d, hour=0.0)
    observer.horizon = "0"

    # Moonrise / moonset
    moonrise = _safe_rise_set(observer, moon, "next_rising")
    moonset = _safe_rise_set(observer, moon, "next_setting")

    # Sunrise / sunset
    sun_body = ephem.Sun()
    sunrise = _safe_rise_set(observer, sun_body, "next_rising")
    sunset = _safe_rise_set(observer, sun_body, "next_setting")

    # Civil twilight (sun center is 6 degrees below horizon)
    observer.horizon = "-6"
    sun_civ = ephem.Sun()
    civil_twilight_start = _safe_rise_set(observer, sun_civ, "next_rising")
    civil_twilight_end = _safe_rise_set(observer, sun_civ, "next_setting")
    observer.horizon = "0"  # reset

    # ----- Solunar periods -----
    # Moon transit (overhead) -- use transit method
    observer.date = _ephem_date(d, hour=0.0)
    moon_transit_body = ephem.Moon()
    try:
        transit_time = observer.next_transit(moon_transit_body)
        moon_transit_local = _ephem_to_local(transit_time)
    except Exception:
        moon_transit_local = None

    # Moon anti-transit (underfoot) -- approximately 12h offset from transit
    observer.date = _ephem_date(d, hour=0.0)
    moon_antitransit_body = ephem.Moon()
    try:
        antitransit_time = observer.next_antitransit(moon_antitransit_body)
        moon_antitransit_local = _ephem_to_local(antitransit_time)
    except Exception:
        moon_antitransit_local = None

    # Major periods: ~2 hours centered on transit and anti-transit
    major1_start, major1_end = _clamp_period(moon_transit_local, 1.0)
    major2_start, major2_end = _clamp_period(moon_antitransit_local, 1.0)

    # Minor periods: ~1 hour centered on moonrise and moonset
    minor1_start, minor1_end = _clamp_period(moonrise, 0.5)
    minor2_start, minor2_end = _clamp_period(moonset, 0.5)

    # Solunar base score
    solunar_score = _solunar_base_score(
        moon_illumination, major1_start, major2_start, sunrise, sunset
    )

    return {
        "date": d,
        "moon_illumination": round(moon_illumination, 4),
        "moon_phase_name": phase_name,
        "moonrise": moonrise,
        "moonset": moonset,
        "sunrise": sunrise,
        "sunset": sunset,
        "civil_twilight_start": civil_twilight_start,
        "civil_twilight_end": civil_twilight_end,
        "major_period_1_start": major1_start,
        "major_period_1_end": major1_end,
        "major_period_2_start": major2_start,
        "major_period_2_end": major2_end,
        "minor_period_1_start": minor1_start,
        "minor_period_1_end": minor1_end,
        "minor_period_2_start": minor2_start,
        "minor_period_2_end": minor2_end,
        "solunar_base_score": solunar_score,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_astro_for_lake(
    lake_key: str,
    lat: float,
    lon: float,
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    """Compute astronomical data for a single lake over a date range.

    Parameters
    ----------
    lake_key : str
        Short identifier for the lake (e.g. ``"patoka"``).
    lat, lon : float
        Latitude and longitude in decimal degrees.
    start_date, end_date : date
        Inclusive date range.

    Returns
    -------
    pd.DataFrame
        One row per day with all astro / solunar columns.
    """
    observer = ephem.Observer()
    observer.lat = str(lat)
    observer.lon = str(lon)
    observer.elevation = 200  # reasonable default for Indiana (metres)
    observer.pressure = 0  # disable atmospheric refraction for consistency

    num_days = (end_date - start_date).days + 1
    rows: list[dict] = []

    for i in tqdm(range(num_days), desc=f"Astro {lake_key}", leave=False):
        d = start_date + timedelta(days=i)
        row = _compute_day(observer.copy(), d)
        row["lake_key"] = lake_key
        rows.append(row)

    df = pd.DataFrame(rows)

    # Reorder columns to match spec
    col_order = [
        "date", "lake_key",
        "moon_illumination", "moon_phase_name",
        "moonrise", "moonset",
        "sunrise", "sunset",
        "civil_twilight_start", "civil_twilight_end",
        "major_period_1_start", "major_period_1_end",
        "major_period_2_start", "major_period_2_end",
        "minor_period_1_start", "minor_period_1_end",
        "minor_period_2_start", "minor_period_2_end",
        "solunar_base_score",
    ]
    df = df[col_order]

    return df


def compute_all_lakes_astro(
    lakes: list[LakeConfig],
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
    """Compute and save astronomical data for all lakes across a year range.

    Saves one Parquet file per lake under ``data/raw/astro/{lake_key}.parquet``
    and returns the combined DataFrame.

    Parameters
    ----------
    lakes : list[LakeConfig]
        Lake configurations with lat/lon.
    start_year, end_year : int
        Inclusive year range (e.g. 2015, 2025).

    Returns
    -------
    pd.DataFrame
        Combined DataFrame for all lakes.
    """
    ASTRO_DIR.mkdir(parents=True, exist_ok=True)

    start = date(start_year, 1, 1)
    end = date(end_year, 12, 31)

    all_frames: list[pd.DataFrame] = []

    for lake in tqdm(lakes, desc="Lakes (astro)"):
        log.info("Computing astro for %s (%s, %s) %d-%d",
                 lake.key, lake.lat, lake.lon, start_year, end_year)

        df = compute_astro_for_lake(lake.key, lake.lat, lake.lon, start, end)

        out_path = ASTRO_DIR / f"{lake.key}.parquet"
        df.to_parquet(out_path, index=False)
        log.info("Saved %s (%d rows)", out_path, len(df))

        all_frames.append(df)

    combined = pd.concat(all_frames, ignore_index=True)
    log.info("Total astro rows: %d", len(combined))
    return combined


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )

    lakes = load_lakes()
    settings = load_settings()

    log.info(
        "Computing astro data for %d lakes, %d-%d",
        len(lakes), settings.start_year, settings.end_year,
    )

    df = compute_all_lakes_astro(lakes, settings.start_year, settings.end_year)
    print(f"\nDone. {len(df):,} total rows across {df['lake_key'].nunique()} lakes.")
    print(df.head())
