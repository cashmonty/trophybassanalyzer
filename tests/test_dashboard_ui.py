from types import SimpleNamespace

import plotly.graph_objects as go

from src.dashboard.ui import apply_figure_style, lake_label


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
