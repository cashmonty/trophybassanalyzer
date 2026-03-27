"""Fetch 15 years of hourly weather data from the Open-Meteo Archive API for Indiana lakes."""

from __future__ import annotations

import asyncio
import logging
from datetime import date
from pathlib import Path

import httpx
import pandas as pd
from diskcache import Cache
from tqdm import tqdm

from src.config import DATA_DIR, LakeConfig, load_lakes, load_settings

logger = logging.getLogger(__name__)

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

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

DAILY_VARIABLES = [
    "temperature_2m_max",
    "temperature_2m_min",
    "sunrise",
    "sunset",
]

CACHE_DIR = DATA_DIR / "cache" / "weather"
OUTPUT_DIR = DATA_DIR / "raw" / "weather"

# Retry settings
MAX_RETRIES = 8
INITIAL_BACKOFF = 5.0  # seconds
BACKOFF_FACTOR = 2.0
REQUEST_DELAY = 1.5  # seconds between each request to avoid 429s


def _year_chunks(start_date: date, end_date: date) -> list[tuple[date, date]]:
    """Split a date range into yearly chunks (Open-Meteo ~1 year per request)."""
    chunks: list[tuple[date, date]] = []
    current_start = start_date
    while current_start <= end_date:
        chunk_end = date(current_start.year, 12, 31)
        if chunk_end > end_date:
            chunk_end = end_date
        chunks.append((current_start, chunk_end))
        current_start = date(current_start.year + 1, 1, 1)
    return chunks


def _build_params(lat: float, lon: float, start: date, end: date) -> dict:
    """Build query parameters for the Open-Meteo Archive API."""
    return {
        "latitude": lat,
        "longitude": lon,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "hourly": ",".join(HOURLY_VARIABLES),
        "daily": ",".join(DAILY_VARIABLES),
        "timezone": "America/Indiana/Indianapolis",
    }


def _cache_key(lake_key: str, start: date, end: date) -> str:
    """Generate a deterministic cache key for a request."""
    return f"weather:{lake_key}:{start.isoformat()}:{end.isoformat()}"


async def _fetch_with_retry(
    client: httpx.AsyncClient,
    params: dict,
    cache: Cache,
    cache_k: str,
) -> dict:
    """Fetch a single chunk from the API with caching and exponential backoff."""
    cached = cache.get(cache_k)
    if cached is not None:
        logger.debug("Cache hit for %s", cache_k)
        return cached

    backoff = INITIAL_BACKOFF
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = await client.get(ARCHIVE_URL, params=params, timeout=60.0)
            if resp.status_code == 429:
                wait = backoff
                logger.warning(
                    "Rate limited (429). Retrying in %.1fs (attempt %d/%d)",
                    wait,
                    attempt,
                    MAX_RETRIES,
                )
                await asyncio.sleep(wait)
                backoff *= BACKOFF_FACTOR
                continue
            resp.raise_for_status()
            data = resp.json()
            cache.set(cache_k, data, expire=86400 * 30)  # cache 30 days
            return data
        except (httpx.HTTPStatusError, httpx.RequestError) as exc:
            if attempt == MAX_RETRIES:
                logger.error("Failed after %d attempts: %s", MAX_RETRIES, exc)
                raise
            wait = backoff
            logger.warning(
                "Request error: %s. Retrying in %.1fs (attempt %d/%d)",
                exc,
                wait,
                attempt,
                MAX_RETRIES,
            )
            await asyncio.sleep(wait)
            backoff *= BACKOFF_FACTOR

    raise RuntimeError("Exhausted retries without returning")


def _parse_hourly(data: dict, lake_key: str) -> pd.DataFrame:
    """Parse hourly data from the API response into a DataFrame."""
    hourly = data.get("hourly", {})
    if not hourly or "time" not in hourly:
        return pd.DataFrame()

    df = pd.DataFrame(hourly)
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time")
    df.index.name = "datetime"
    df["lake_key"] = lake_key
    return df


def _merge_daily_into_hourly(hourly_df: pd.DataFrame, data: dict) -> pd.DataFrame:
    """Merge daily variables (sunrise, sunset, temp extremes) into the hourly frame.

    Daily values are broadcast to every hour within the same date.
    """
    daily = data.get("daily", {})
    if not daily or "time" not in daily:
        return hourly_df

    daily_df = pd.DataFrame(daily)
    daily_df["time"] = pd.to_datetime(daily_df["time"])
    daily_df = daily_df.set_index("time")
    daily_df.index.name = "date"

    # Add a date column to hourly for merging
    hourly_df["_date"] = hourly_df.index.date
    daily_df.index = daily_df.index.date  # type: ignore[assignment]

    for col in daily_df.columns:
        mapping = daily_df[col].to_dict()
        hourly_df[f"daily_{col}"] = hourly_df["_date"].map(mapping)

    hourly_df.drop(columns=["_date"], inplace=True)
    return hourly_df


async def fetch_weather_for_lake(
    lake_key: str,
    lat: float,
    lon: float,
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    """Fetch hourly + daily weather for a single lake across the full date range.

    Chunks into yearly requests to respect Open-Meteo limits.
    Returns a single DataFrame with a datetime index and 'lake_key' column.
    """
    chunks = _year_chunks(start_date, end_date)
    frames: list[pd.DataFrame] = []

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = Cache(str(CACHE_DIR))

    async with httpx.AsyncClient() as client:
        for chunk_start, chunk_end in tqdm(
            chunks,
            desc=f"  {lake_key}",
            leave=False,
            unit="yr",
        ):
            params = _build_params(lat, lon, chunk_start, chunk_end)
            cache_k = _cache_key(lake_key, chunk_start, chunk_end)
            data = await _fetch_with_retry(client, params, cache, cache_k)
            await asyncio.sleep(REQUEST_DELAY)

            df = _parse_hourly(data, lake_key)
            if df.empty:
                logger.warning(
                    "No hourly data for %s (%s to %s)",
                    lake_key,
                    chunk_start,
                    chunk_end,
                )
                continue

            df = _merge_daily_into_hourly(df, data)
            frames.append(df)

    cache.close()

    if not frames:
        logger.warning("No data at all for lake %s", lake_key)
        return pd.DataFrame()

    combined = pd.concat(frames)
    combined.sort_index(inplace=True)
    return combined


def _save_parquet(df: pd.DataFrame, lake_key: str) -> Path:
    """Save a lake's weather DataFrame to Parquet."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"{lake_key}.parquet"
    df.to_parquet(path, engine="pyarrow")
    logger.info("Saved %s (%d rows)", path, len(df))
    return path


async def fetch_all_lakes_weather(
    lakes: list[LakeConfig],
    start_year: int,
    end_year: int,
) -> dict[str, pd.DataFrame]:
    """Fetch weather data for all lakes and save each as Parquet.

    Processes lakes sequentially to stay friendly to the Open-Meteo API.
    Returns a dict mapping lake_key -> DataFrame.
    """
    start = date(start_year, 1, 1)
    end = date(end_year, 12, 31)

    # Clamp end date to yesterday if it's in the future (archive data only)
    today = date.today()
    if end >= today:
        end = date(today.year, today.month, today.day)
        # Go back one day to be safe with archive availability
        end = end.replace(day=max(end.day - 1, 1))

    results: dict[str, pd.DataFrame] = {}

    for lake in tqdm(lakes, desc="Lakes", unit="lake"):
        logger.info(
            "Fetching weather for %s (%s) [%.4f, %.4f]",
            lake.key,
            lake.name,
            lake.lat,
            lake.lon,
        )
        try:
            df = await fetch_weather_for_lake(
                lake.key, lake.lat, lake.lon, start, end
            )
            if not df.empty:
                _save_parquet(df, lake.key)
                results[lake.key] = df
        except Exception as exc:
            logger.error("Failed to fetch weather for %s: %s", lake.key, exc)
            continue

    logger.info("Completed weather fetch for %d lakes", len(results))
    return results


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    lakes = load_lakes()
    settings = load_settings()

    logger.info(
        "Fetching weather for %d lakes from %d to %d",
        len(lakes),
        settings.start_year,
        settings.end_year,
    )

    asyncio.run(fetch_all_lakes_weather(lakes, settings.start_year, settings.end_year))
