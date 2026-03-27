"""Overview page -- high-level trophy bass statistics and trends."""

import sys
from pathlib import Path

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

for candidate in Path(__file__).resolve().parents:
    if (candidate / "pyproject.toml").exists() and (candidate / "src").exists():
        candidate_str = str(candidate)
        if candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)
        break

from src.dashboard.ui import MONTH_NAMES, bootstrap_dashboard, render_page_header, render_plotly

ctx = bootstrap_dashboard("Overview")
COLORS = ctx.colors

render_page_header(
    "Overview",
    "High-level trend view across the selected lakes: overall catch volume, trophy concentration, and seasonal timing.",
)

# ---------------------------------------------------------------------------
# Load filtered data from session state
# ---------------------------------------------------------------------------
df = ctx.merged_df
lake_configs = ctx.lake_configs

if df is None or df.empty:
    st.info("No data available. Please check that data files exist and filters are not too restrictive.")
    st.stop()

# ---------------------------------------------------------------------------
# Metric cards
# ---------------------------------------------------------------------------
total_catches = int(df["catch_count"].sum()) if "catch_count" in df.columns else 0
total_trophies = int(df["trophy_count"].sum()) if "trophy_count" in df.columns else 0
trophy_rate = (total_trophies / total_catches * 100) if total_catches > 0 else 0.0

# Top lake
top_lake = "N/A"
if "lake_key" in df.columns and "trophy_count" in df.columns:
    lake_trophies = df.groupby("lake_key", observed=True)["trophy_count"].sum()
    if not lake_trophies.empty:
        top_lake_key = lake_trophies.idxmax()
        top_lake = lake_configs[top_lake_key].name if top_lake_key in lake_configs else top_lake_key

avg_trophy_weight = 0.0
if "max_weight" in df.columns and "trophy_caught" in df.columns:
    trophy_rows = df[df["trophy_caught"] == 1]
    if not trophy_rows.empty and trophy_rows["max_weight"].notna().any():
        avg_trophy_weight = trophy_rows["max_weight"].mean()

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Catches", f"{total_catches:,}")
c2.metric("Total Trophies", f"{total_trophies:,}")
c3.metric("Trophy Rate", f"{trophy_rate:.1f}%")
c4.metric("Top Lake", top_lake)
c5.metric("Avg Trophy Weight", f"{avg_trophy_weight:.2f} lbs" if avg_trophy_weight else "N/A")

st.markdown("---")

# ---------------------------------------------------------------------------
# Trophy catch timeline
# ---------------------------------------------------------------------------
st.subheader("Trophy Catch Timeline")

if "datetime" in df.columns and "trophy_caught" in df.columns:
    trophy_events = df[df["trophy_caught"] == 1].copy()
    if not trophy_events.empty:
        size_col = "max_weight" if "max_weight" in trophy_events.columns else None
        fig = px.scatter(
            trophy_events,
            x="datetime",
            y="max_weight" if "max_weight" in trophy_events.columns else "trophy_count",
            size=size_col if size_col and trophy_events[size_col].notna().any() else None,
            color="lake_key" if "lake_key" in trophy_events.columns else None,
            color_discrete_sequence=[COLORS["bass_green"], COLORS["water_blue"],
                                     COLORS["trophy_gold"], "#e74c3c", "#8e44ad", "#1abc9c"],
            title="Trophy Catches Over Time",
            labels={"datetime": "Date", "max_weight": "Weight (lbs)", "lake_key": "Lake"},
        )
        render_plotly(fig, height=450)
    else:
        st.info("No trophy catches in the filtered data.")
else:
    st.info("Required columns not available for timeline.")

# ---------------------------------------------------------------------------
# Lake leaderboard & Monthly distribution side by side
# ---------------------------------------------------------------------------
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Lake Leaderboard")
    if {"lake_key", "catch_count", "trophy_count"}.issubset(df.columns):
        leaderboard = (
            df.groupby("lake_key", observed=True)
            .agg(
                total_catches=("catch_count", "sum"),
                total_trophies=("trophy_count", "sum"),
            )
            .reset_index()
            .sort_values("total_trophies", ascending=True)
        )
        leaderboard["lake_name"] = leaderboard["lake_key"].map(
            lambda k: lake_configs[k].name if k in lake_configs else k
        )
        fig_lb = px.bar(
            leaderboard,
            y="lake_name",
            x="total_trophies",
            orientation="h",
            color_discrete_sequence=[COLORS["trophy_gold"]],
            title="Trophies by Lake",
            labels={"total_trophies": "Trophy Count", "lake_name": ""},
        )
        render_plotly(fig_lb, height=350)
    else:
        st.info("No lake data available.")

with col_right:
    st.subheader("Monthly Distribution")
    if {"month", "catch_count", "trophy_count"}.issubset(df.columns):
        monthly = (
            df.groupby("month")
            .agg(total_catches=("catch_count", "sum"), total_trophies=("trophy_count", "sum"))
            .reset_index()
        )
        monthly["month_name"] = monthly["month"].map(MONTH_NAMES)

        fig_m = go.Figure()
        fig_m.add_trace(go.Bar(
            x=monthly["month_name"], y=monthly["total_catches"],
            name="Total Catches", marker_color=COLORS["water_blue"],
        ))
        fig_m.add_trace(go.Bar(
            x=monthly["month_name"], y=monthly["total_trophies"],
            name="Trophies", marker_color=COLORS["trophy_gold"],
        ))
        fig_m.update_layout(
            barmode="group",
            title="Catches by Month",
            xaxis_title="Month",
            yaxis_title="Count",
        )
        render_plotly(fig_m, height=350)
    else:
        st.info("No monthly data available.")

# ---------------------------------------------------------------------------
# Spawn phase and water temp breakdown
# ---------------------------------------------------------------------------
col_sp, col_wt = st.columns(2)

with col_sp:
    st.subheader("Trophy Catches by Spawn Phase")
    if {"spawn_phase", "trophy_count"}.issubset(df.columns):
        phase_data = df.groupby("spawn_phase", observed=True)["trophy_count"].sum().reset_index()
        phase_data = phase_data[phase_data["trophy_count"] > 0].sort_values("trophy_count", ascending=True)
        phase_labels = {
            "PRE_SPAWN": "Pre-spawn", "SPAWN": "Spawn", "POST_SPAWN": "Post-spawn",
            "SUMMER": "Summer", "FALL": "Fall", "TURNOVER": "Turnover", "WINTER": "Winter",
        }
        phase_data["label"] = phase_data["spawn_phase"].map(phase_labels).fillna(phase_data["spawn_phase"])
        fig_sp = px.bar(phase_data, y="label", x="trophy_count", orientation="h",
                        color_discrete_sequence=[COLORS["trophy_gold"]],
                        labels={"trophy_count": "Trophies", "label": ""})
        render_plotly(fig_sp, height=350)
    else:
        st.info("Spawn phase data not available.")

with col_wt:
    st.subheader("Water Temp at Trophy Catch")
    if {"water_temp_f", "trophy_caught"}.issubset(df.columns):
        trophy_wt = df[df["trophy_caught"] == 1]["water_temp_f"].dropna()
        if not trophy_wt.empty:
            fig_wt = go.Figure(go.Histogram(
                x=trophy_wt, nbinsx=20,
                marker_color=COLORS["water_blue"],
            ))
            fig_wt.add_vline(x=trophy_wt.median(), line_dash="dash", line_color=COLORS["trophy_gold"],
                             annotation_text=f"Median: {trophy_wt.median():.0f}F")
            fig_wt.update_layout(xaxis_title="Water Temp (F)", yaxis_title="Trophy Hours")
            render_plotly(fig_wt, height=350)
        else:
            st.info("No water temp data for trophy catches.")
    elif {"water_temp_estimated", "trophy_caught"}.issubset(df.columns):
        trophy_wt = df[df["trophy_caught"] == 1]["water_temp_estimated"].dropna() * 9 / 5 + 32
        if not trophy_wt.empty:
            fig_wt = go.Figure(go.Histogram(x=trophy_wt, nbinsx=20, marker_color=COLORS["water_blue"]))
            fig_wt.update_layout(xaxis_title="Water Temp (F)", yaxis_title="Trophy Hours")
            render_plotly(fig_wt, height=350)
    else:
        st.info("Water temp data not available.")

st.markdown("---")

# ---------------------------------------------------------------------------
# Seasonal heatmap (month x hour)
# ---------------------------------------------------------------------------
st.subheader("Seasonal Heatmap (Month x Hour)")

if all(c in df.columns for c in ["month", "hour", "catch_count"]):
    heatmap_data = (
        df.groupby(["month", "hour"])
        .agg(catch_rate=("catch_count", "mean"))
        .reset_index()
        .pivot(index="hour", columns="month", values="catch_rate")
    )
    heatmap_data.columns = [MONTH_NAMES.get(c, c) for c in heatmap_data.columns]

    fig_hm = px.imshow(
        heatmap_data,
        color_continuous_scale=["#f7fbff", COLORS["water_blue"], COLORS["bass_green"]],
        title="Average Catch Rate by Month and Hour",
        labels=dict(x="Month", y="Hour of Day", color="Avg Catches"),
        aspect="auto",
    )
    render_plotly(fig_hm, height=500)
else:
    st.info("Heatmap requires month, hour, and catch_count columns.")
