"""Fetch water temperature and lake level data from USGS Water Services API."""

from __future__ import annotations

import asyncio
import logging
from datetime import date

import httpx
import pandas as pd
from diskcache import Cache
from tqdm import tqdm

from src.config import LakeConfig, DATA_DIR, load_lakes, load_settings

logger = logging.getLogger(__name__)

USGS_DV_URL = "https://waterservices.usgs.gov/nwis/dv/"

# USGS parameter codes
PARAM_WATER_TEMP = "00010"  # Water temperature (deg C)
PARAM_GAGE_HEIGHT = "00065"  # Gage height / lake level (feet)
PARAM_DISCHARGE = "00060"  # Discharge (cubic feet per second)

ALL_PARAMS = ",".join([PARAM_WATER_TEMP, PARAM_GAGE_HEIGHT, PARAM_DISCHARGE])

CACHE_DIR = DATA_DIR / "cache" / "water"
OUTPUT_DIR = DATA_DIR / "raw" / "water"

MAX_RETRIES = 3
BACKOFF_BASE = 2  # seconds


def _parse_usgs_json(data: dict, lake_key: str) -> pd.DataFrame:
    """Parse USGS JSON response into a flat DataFrame.

    The USGS response nests data as:
        value -> timeSeries[] -> variable + values[] -> value[]
    Each timeSeries entry corresponds to one parameter at one site.
    """
    records: list[dict] = []

    time_series_list = data.get("value", {}).get("timeSeries", [])
    if not time_series_list:
        return pd.DataFrame(columns=["date", "lake_key", "water_temp_c",
                                      "water_temp_f", "gage_height_ft",
                                      "discharge_cfs"])

    # Build a mapping: date -> {param_code: value}
    date_map: dict[str, dict[str, float | None]] = {}

    for ts in time_series_list:
        # Extract parameter code from variable.variableCode
        var_codes = ts.get("variable", {}).get("variableCode", [])
        if not var_codes:
            continue
        param_code = var_codes[0].get("value", "")

        # Walk through all value sets
        for value_set in ts.get("values", []):
            for entry in value_set.get("value", []):
                raw_val = entry.get("value")
                date_str = entry.get("dateTime", "")[:10]  # YYYY-MM-DD
                if not date_str:
                    continue

                if date_str not in date_map:
                    date_map[date_str] = {}

                # USGS uses -999999 or empty string for missing values
                try:
                    val = float(raw_val)
                    if val == -999999:
                        val = None
                except (TypeError, ValueError):
                    val = None

                date_map[date_str][param_code] = val

    for date_str, params in date_map.items():
        temp_c = params.get(PARAM_WATER_TEMP)
        temp_f = round(temp_c * 9 / 5 + 32, 2) if temp_c is not None else None

        records.append({
            "date": date_str,
            "lake_key": lake_key,
            "water_temp_c": temp_c,
            "water_temp_f": temp_f,
            "gage_height_ft": params.get(PARAM_GAGE_HEIGHT),
            "discharge_cfs": params.get(PARAM_DISCHARGE),
        })

    df = pd.DataFrame(records)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"]).dt.date
        df.sort_values("date", inplace=True)
        df.reset_index(drop=True, inplace=True)
    return df


async def _fetch_with_retry(
    client: httpx.AsyncClient,
    params: dict,
    retries: int = MAX_RETRIES,
) -> dict | None:
    """GET the USGS daily values endpoint with exponential backoff on 5xx."""
    for attempt in range(retries + 1):
        try:
            resp = await client.get(USGS_DV_URL, params=params, timeout=60)
            if resp.status_code >= 500:
                if attempt < retries:
                    wait = BACKOFF_BASE ** (attempt + 1)
                    logger.warning(
                        "USGS returned %s, retrying in %ss (attempt %d/%d)",
                        resp.status_code, wait, attempt + 1, retries,
                    )
                    await asyncio.sleep(wait)
                    continue
                logger.error("USGS returned %s after %d retries", resp.status_code, retries)
                return None
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError:
            logger.error("HTTP error fetching USGS data: %s", resp.status_code)
            return None
        except httpx.RequestError as exc:
            if attempt < retries:
                wait = BACKOFF_BASE ** (attempt + 1)
                logger.warning("Request error: %s, retrying in %ss", exc, wait)
                await asyncio.sleep(wait)
                continue
            logger.error("Request failed after %d retries: %s", retries, exc)
            return None
    return None


async def fetch_water_for_lake(
    lake_key: str,
    station_id: str,
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    """Fetch daily water data for a single lake from USGS, chunked by year.

    Uses diskcache to avoid re-fetching previously retrieved year chunks.

    Parameters
    ----------
    lake_key : str
        Identifier for the lake (e.g. ``"patoka"``).
    station_id : str
        USGS station ID (e.g. ``"03374000"``).
    start_date : date
        Earliest date to fetch.
    end_date : date
        Latest date to fetch.

    Returns
    -------
    pd.DataFrame
        Columns: date, lake_key, water_temp_c, water_temp_f,
        gage_height_ft, discharge_cfs.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = Cache(str(CACHE_DIR))

    frames: list[pd.DataFrame] = []

    async with httpx.AsyncClient() as client:
        for year in range(start_date.year, end_date.year + 1):
            cache_key = f"{lake_key}:{station_id}:{year}"

            cached = cache.get(cache_key)
            if cached is not None:
                logger.debug("Cache hit for %s year %d", lake_key, year)
                frames.append(pd.DataFrame(cached))
                continue

            chunk_start = date(year, 1, 1) if year > start_date.year else start_date
            chunk_end = date(year, 12, 31) if year < end_date.year else end_date

            params = {
                "format": "json",
                "sites": station_id,
                "startDT": chunk_start.isoformat(),
                "endDT": chunk_end.isoformat(),
                "parameterCd": ALL_PARAMS,
                "siteStatus": "all",
            }

            logger.info("Fetching USGS data for %s (%s) year %d", lake_key, station_id, year)
            data = await _fetch_with_retry(client, params)

            if data is None:
                logger.warning("No data returned for %s year %d", lake_key, year)
                continue

            df_chunk = _parse_usgs_json(data, lake_key)

            if not df_chunk.empty:
                # Cache as list-of-dicts for easy serialization
                cache.set(cache_key, df_chunk.to_dict("records"), expire=86400 * 7)
                frames.append(df_chunk)

    cache.close()

    if not frames:
        return pd.DataFrame(columns=["date", "lake_key", "water_temp_c",
                                      "water_temp_f", "gage_height_ft",
                                      "discharge_cfs"])

    result = pd.concat(frames, ignore_index=True)
    result.drop_duplicates(subset=["date", "lake_key"], inplace=True)
    result.sort_values("date", inplace=True)
    result.reset_index(drop=True, inplace=True)
    return result


async def fetch_all_lakes_water(
    lakes: list[LakeConfig],
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
    """Fetch water data for all lakes that have a USGS station.

    Parameters
    ----------
    lakes : list[LakeConfig]
        Lake configurations; lakes without ``usgs_station`` are skipped.
    start_year : int
        First year to fetch (inclusive).
    end_year : int
        Last year to fetch (inclusive).

    Returns
    -------
    pd.DataFrame
        Combined DataFrame for all lakes.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    start = date(start_year, 1, 1)
    end = date(end_year, 12, 31)

    eligible = [lk for lk in lakes if lk.usgs_station]
    if not eligible:
        logger.warning("No lakes with USGS stations found — nothing to fetch.")
        return pd.DataFrame()

    all_frames: list[pd.DataFrame] = []

    for lake in tqdm(eligible, desc="Fetching USGS water data"):
        try:
            df = await fetch_water_for_lake(
                lake_key=lake.key,
                station_id=lake.usgs_station,  # type: ignore[arg-type]
                start_date=start,
                end_date=end,
            )
        except Exception:
            logger.exception("Failed to fetch water data for %s", lake.key)
            continue

        if df.empty:
            logger.info("No water data for %s", lake.key)
            continue

        # Save per-lake Parquet
        out_path = OUTPUT_DIR / f"{lake.key}.parquet"
        df.to_parquet(out_path, index=False)
        logger.info("Saved %d rows to %s", len(df), out_path)

        all_frames.append(df)

    if not all_frames:
        return pd.DataFrame()

    combined = pd.concat(all_frames, ignore_index=True)
    return combined


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )

    lakes = load_lakes()
    settings = load_settings()

    df = asyncio.run(
        fetch_all_lakes_water(lakes, settings.start_year, settings.end_year)
    )
    print(f"\nDone. Total rows: {len(df)}")
    if not df.empty:
        print(df.head(10))
