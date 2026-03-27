"""Configuration loader for lakes and settings."""

from dataclasses import dataclass
from pathlib import Path

import yaml

CONFIG_DIR = Path(__file__).parent.parent / "config"
DATA_DIR = Path(__file__).parent.parent / "data"


@dataclass
class LakeConfig:
    key: str
    name: str
    lat: float
    lon: float
    usgs_station: str | None
    state: str
    notes: str
    max_depth_ft: int | None = None
    avg_depth_ft: int | None = None


@dataclass
class Settings:
    start_year: int
    end_year: int
    timezone: str
    trophy_weight_lbs: float
    super_trophy_weight_lbs: float


def load_lakes(config_path: Path | None = None) -> list[LakeConfig]:
    """Load lake configurations from YAML."""
    path = config_path or CONFIG_DIR / "lakes.yaml"
    with open(path) as f:
        data = yaml.safe_load(f)

    lakes = []
    for key, cfg in data["lakes"].items():
        lakes.append(
            LakeConfig(
                key=key,
                name=cfg["name"],
                lat=cfg["lat"],
                lon=cfg["lon"],
                usgs_station=cfg.get("usgs_station"),
                state=cfg.get("state", "IN"),
                notes=cfg.get("notes", ""),
                max_depth_ft=cfg.get("max_depth_ft"),
                avg_depth_ft=cfg.get("avg_depth_ft"),
            )
        )
    return lakes


def load_settings(config_path: Path | None = None) -> Settings:
    """Load application settings from YAML."""
    path = config_path or CONFIG_DIR / "lakes.yaml"
    with open(path) as f:
        data = yaml.safe_load(f)

    s = data["settings"]
    return Settings(
        start_year=s["start_year"],
        end_year=s["end_year"],
        timezone=s["timezone"],
        trophy_weight_lbs=s["trophy_weight_lbs"],
        super_trophy_weight_lbs=s.get("super_trophy_weight_lbs", 7.0),
    )
