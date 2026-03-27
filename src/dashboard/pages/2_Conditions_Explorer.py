"""Conditions Explorer -- environmental conditions vs catch outcomes."""

import sys
from pathlib import Path

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

for candidate in Path(__file__).resolve().parents:
    if (candidate / "pyproject.toml").exists() and (candidate / "src").exists():
        candidate_str = str(candidate)
        if candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)
        break

from src.dashboard.ui import bootstrap_dashboard, render_page_header, render_plotly

ctx = bootstrap_dashboard("Conditions Explorer")
COLORS = ctx.colors

render_page_header(
    "Conditions Explorer",
    "Environmental conditions with trophy overlays, distribution splits, and weight-vs-condition checks for the active lakes.",
)

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
df = ctx.merged_df

if df is None or df.empty:
    st.info("No data available. Please check that data files exist and filters are not too restrictive.")
    st.stop()

# ---------------------------------------------------------------------------
# Multi-axis time series with catch events
# ---------------------------------------------------------------------------
st.subheader("Environmental Conditions Timeline")

condition_cols = {
    "temperature_2m": "Temperature (C)",
    "pressure_msl": "Pressure (hPa)",
    "wind_speed_10m": "Wind Speed (m/s)",
    "cloud_cover": "Cloud Cover (%)",
}
available_conds = {k: v for k, v in condition_cols.items() if k in df.columns}

if available_conds and "datetime" in df.columns:
    # Subsample for performance
    plot_df = df.copy()
    if len(plot_df) > 10000:
        plot_df = plot_df.sample(10000, random_state=42).sort_values("datetime")

    n_panels = len(available_conds)
    fig = make_subplots(
        rows=n_panels, cols=1, shared_xaxes=True,
        subplot_titles=list(available_conds.values()),
        vertical_spacing=0.04,
    )

    for i, (col, label) in enumerate(available_conds.items(), 1):
        fig.add_trace(
            go.Scattergl(
                x=plot_df["datetime"], y=plot_df[col],
                mode="lines", name=label,
                line=dict(width=1, color=COLORS["water_blue"]),
                showlegend=False,
            ),
            row=i, col=1,
        )
        # Overlay trophy catches
        if "trophy_caught" in plot_df.columns:
            trophies = plot_df[plot_df["trophy_caught"] == 1]
            if not trophies.empty:
                fig.add_trace(
                    go.Scattergl(
                        x=trophies["datetime"], y=trophies[col],
                        mode="markers", name="Trophy Catch",
                        marker=dict(color=COLORS["trophy_gold"], size=5, symbol="star"),
                        showlegend=(i == 1),
                    ),
                    row=i, col=1,
                )

    fig.update_layout(title="Conditions with Trophy Catch Overlay")
    render_plotly(fig, height=250 * n_panels)
else:
    st.info("Insufficient condition columns or datetime column for timeline.")

st.markdown("---")

# ---------------------------------------------------------------------------
# Distribution plots: trophy vs non-trophy
# ---------------------------------------------------------------------------
st.subheader("Condition Distributions: Trophy vs Non-Trophy")

dist_features = [c for c in [
    "temperature_2m", "water_temp_estimated", "pressure_msl",
    "wind_speed_10m", "cloud_cover", "moon_illumination",
    "solunar_base_score", "precipitation",
] if c in df.columns]

if dist_features and "trophy_caught" in df.columns:
    selected_feature = st.selectbox("Select condition variable", dist_features)

    plot_data = df[[selected_feature, "trophy_caught"]].dropna()
    plot_data["group"] = plot_data["trophy_caught"].map({0: "No Trophy", 1: "Trophy"})

    fig_dist = px.histogram(
        plot_data,
        x=selected_feature,
        color="group",
        barmode="overlay",
        nbins=50,
        color_discrete_map={"No Trophy": COLORS["water_blue"], "Trophy": COLORS["trophy_gold"]},
        opacity=0.7,
        title=f"Distribution of {selected_feature}",
        labels={selected_feature: selected_feature, "group": ""},
    )
    render_plotly(fig_dist, height=400)

    # Stats comparison
    col1, col2 = st.columns(2)
    no_trophy = plot_data[plot_data["trophy_caught"] == 0][selected_feature]
    trophy = plot_data[plot_data["trophy_caught"] == 1][selected_feature]
    with col1:
        st.markdown("**Non-Trophy Stats**")
        st.write(no_trophy.describe().round(2))
    with col2:
        st.markdown("**Trophy Stats**")
        if not trophy.empty:
            st.write(trophy.describe().round(2))
        else:
            st.write("No trophy data in selection.")
else:
    st.info("No condition features available for distribution analysis.")

st.markdown("---")

# ---------------------------------------------------------------------------
# Scatter plots: conditions vs weight
# ---------------------------------------------------------------------------
st.subheader("Conditions vs Trophy Weight")

scatter_features = [c for c in [
    "temperature_2m", "water_temp_estimated", "pressure_msl",
    "wind_speed_10m", "cloud_cover", "moon_illumination",
    "solunar_base_score", "pressure_trend_3h",
] if c in df.columns]

if scatter_features and "max_weight" in df.columns:
    col_left, col_right = st.columns(2)

    with col_left:
        x_feat = st.selectbox("X-axis", scatter_features, key="scatter_x")
    with col_right:
        color_by = st.selectbox(
            "Color by",
            ["lake_key", "spawn_phase", "front_type", "month"],
            key="scatter_color",
        )
        color_by = color_by if color_by in df.columns else None

    scatter_df = df[df["max_weight"] > 0].dropna(subset=[x_feat, "max_weight"])

    if not scatter_df.empty:
        fig_sc = px.scatter(
            scatter_df,
            x=x_feat,
            y="max_weight",
            color=color_by,
            opacity=0.5,
            title=f"{x_feat} vs Weight",
            labels={x_feat: x_feat, "max_weight": "Max Weight (lbs)"},
            color_discrete_sequence=[COLORS["bass_green"], COLORS["water_blue"],
                                     COLORS["trophy_gold"], "#e74c3c", "#8e44ad",
                                     "#1abc9c", "#d35400"],
        )
        render_plotly(fig_sc, height=500)
    else:
        st.info("No records with valid weight for scatter plot.")
else:
    st.info("Required columns not available for scatter analysis.")
