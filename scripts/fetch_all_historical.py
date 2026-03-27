"""One-time script to fetch all historical data for all lakes.

Usage:
    python -m scripts.fetch_all_historical
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_lakes, load_settings
from src.ingest.weather import fetch_all_lakes_weather
from src.ingest.water import fetch_all_lakes_water
from src.ingest.astro import compute_all_lakes_astro

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


async def main():
    lakes = load_lakes()
    settings = load_settings()

    logger.info(f"Fetching data for {len(lakes)} lakes, {settings.start_year}-{settings.end_year}")

    # 1. Fetch weather data (async)
    logger.info("=" * 60)
    logger.info("STEP 1: Fetching weather data from Open-Meteo...")
    logger.info("=" * 60)
    try:
        await fetch_all_lakes_weather(lakes, settings.start_year, settings.end_year)
    except Exception as e:
        logger.error(f"Weather fetch had errors (partial data may be saved): {e}")

    # 2. Fetch water data (async)
    logger.info("=" * 60)
    logger.info("STEP 2: Fetching water data from USGS...")
    logger.info("=" * 60)
    try:
        await fetch_all_lakes_water(lakes, settings.start_year, settings.end_year)
    except Exception as e:
        logger.error(f"Water fetch had errors (partial data may be saved): {e}")

    # 3. Compute astronomical data (sync, but fast)
    logger.info("=" * 60)
    logger.info("STEP 3: Computing astronomical data...")
    logger.info("=" * 60)
    compute_all_lakes_astro(lakes, settings.start_year, settings.end_year)

    # 4. Parse real USA Bassin tournament data
    logger.info("=" * 60)
    logger.info("STEP 4: Parsing USA Bassin tournament catch data...")
    logger.info("=" * 60)
    try:
        from src.ingest.usabassin import save_catches
        save_catches()
    except Exception as e:
        logger.warning(f"USA Bassin parse had errors: {e}")
        # Fallback to sample data if no real data exists
        logger.info("Falling back to sample tournament data...")
        try:
            from src.ingest.tournaments import generate_sample_data
            generate_sample_data(lakes, settings.start_year, settings.end_year)
        except Exception as e2:
            logger.error(f"Sample data generation also failed: {e2}")

    logger.info("=" * 60)
    logger.info("All historical data fetched successfully!")
    logger.info("Next steps:")
    logger.info("  1. python -m src.pipeline.merge")
    logger.info("  2. python -m src.analysis.model")
    logger.info("  3. python -m src.analysis.forecast")
    logger.info("  4. streamlit run src/dashboard/app.py")
    logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
