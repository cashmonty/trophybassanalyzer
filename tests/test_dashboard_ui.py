from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import plotly.graph_objects as go
import pytest

from src.analysis.correlations import compute_conditional_rates, compute_feature_importance
from src.dashboard.ui import (
    DashboardDataError,
    _optimize_dashboard_frame,
    _read_parquet_checked,
    apply_figure_style,
    lake_label,
    load_dashboard_data,
    load_dashboard_predictions,
)


def test_lake_label_prefers_config_display_name() -> None:
    lake_configs = {
        "monroe": SimpleNamespace(name="Lake Monroe"),
        "patoka": SimpleNamespace(name="Patoka Lake"),
    }

    assert lake_label("monroe", lake_configs) == "Lake Monroe"
    assert lake_label("unknown", lake_configs) == "Unknown"


def test_apply_figure_style_sets_dashboard_defaults() -> None:
    fig = go.Figure()

    styled = apply_figure_style(fig, height=420)

    assert styled.layout.height == 420
    assert styled.layout.paper_bgcolor == "rgba(0,0,0,0)"
    assert styled.layout.plot_bgcolor == "rgba(0,0,0,0)"


def test_read_parquet_checked_raises_on_missing_file(tmp_path: Path) -> None:
    with pytest.raises(DashboardDataError):
        _read_parquet_checked(tmp_path / "missing.parquet", "missing fixture")


def test_optimize_dashboard_frame_downcasts_and_categorizes() -> None:
    frame = pd.DataFrame(
        {
            "datetime": ["2026-01-01 00:00:00", "2026-01-01 01:00:00"],
            "lake_key": ["monroe", "patoka"],
            "rating": ["A", "B"],
            "weight_lbs": [7.5, 8.0],
            "catch_count": [1, 2],
            "is_trophy": [1, 0],
        }
    )

    optimized = _optimize_dashboard_frame(frame)

    assert str(optimized["datetime"].dtype).startswith("datetime64")
    assert str(optimized["lake_key"].dtype) == "category"
    assert str(optimized["rating"].dtype) == "category"
    assert str(optimized["weight_lbs"].dtype) == "float32"
    assert optimized["is_trophy"].dtype == bool


def test_feature_importance_skips_constant_inputs() -> None:
    frame = pd.DataFrame(
        {
            "temperature_2m": [55.0] * 200,
            "pressure_msl": list(range(200)),
            "trophy_caught": [0, 1] * 100,
        }
    )

    importance = compute_feature_importance(frame)

    assert "temperature_2m" not in importance["feature"].tolist()
    assert "pressure_msl" in importance["feature"].tolist()


def test_conditional_rates_returns_empty_for_constant_feature() -> None:
    frame = pd.DataFrame(
        {
            "pressure_msl": [30.0] * 50,
            "trophy_caught": [0, 1] * 25,
        }
    )

    rates = compute_conditional_rates(frame, "pressure_msl", n_bins=10)

    assert rates.empty


def test_dashboard_data_loaders_return_expected_shapes() -> None:
    load_dashboard_data.clear()
    load_dashboard_predictions.clear()

    merged, catches = load_dashboard_data()
    predictions = load_dashboard_predictions()

    assert not merged.empty
    assert {"datetime", "date", "lake_key"}.issubset(merged.columns)
    assert not catches.empty
    assert {"date", "lake_key", "weight_lbs"}.issubset(catches.columns)
    assert catches["weight_lbs"].between(0, 20, inclusive="both").all()

    if predictions is not None:
        assert not predictions.empty
        assert {"date", "lake_key", "max_probability", "rating"}.issubset(predictions.columns)
