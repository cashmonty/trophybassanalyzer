"""Indiana trophy bass command center."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

for candidate in Path(__file__).resolve().parents:
    if (candidate / "pyproject.toml").exists() and (candidate / "src").exists():
        candidate_str = str(candidate)
        if candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)
        break

from src.dashboard.ui import (
    MONTH_NAMES,
    apply_figure_style,
    bootstrap_dashboard,
    lake_label,
    render_page_header,
)

TROPHY_WEIGHT = 7.0

TECHNIQUE_BY_PHASE = {
    "WINTER": {
        "title": "Cold-water structure grind",
        "baits": "Blade bait, hair jig, jigging spoon, compact swimbait",
        "approach": "Stay on channel swings, deep timber, and the first hard break near wintering holes.",
        "tip": "Favor the warmest water in the system after stable weather.",
    },
    "PRE_SPAWN": {
        "title": "Prime giant window",
        "baits": "Jerkbait, lipless crankbait, craw jig, line-through swimbait",
        "approach": "Work secondary points, transition banks, and creek-channel turns slowly and methodically.",
        "tip": "Three warm days with south wind is the cleanest green light in this dataset.",
    },
    "SPAWN": {
        "title": "Targeted bed cycle",
        "baits": "Stick worm, creature bait, compact jig, backup squarebill",
        "approach": "Look for protected hard-bottom pockets with nearby cover and slightly warmer water.",
        "tip": "Big females slide in and out quickly. Cover the edges, not just visible beds.",
    },
    "POST_SPAWN": {
        "title": "Recovery and fry-guard split",
        "baits": "Walking bait, weightless soft plastic, finesse jig, drop shot",
        "approach": "Females suspend or recover on the first cover outside the pockets while males stay shallow.",
        "tip": "Early-morning shad activity can reset the whole day.",
    },
    "SUMMER": {
        "title": "Deep structure or low-light shallow",
        "baits": "Deep crankbait, football jig, Carolina rig, frog, big worm",
        "approach": "Decide early whether the lake is on offshore timber and ledges or on shade and shallow cover.",
        "tip": "Night and first-light windows matter more than midday averages.",
    },
    "FALL": {
        "title": "Bait-driven power fishing",
        "baits": "Squarebill, spinnerbait, topwater, jerkbait, swimbait",
        "approach": "Follow bait migrations into creek arms and cover water until you find active groups.",
        "tip": "First cool nights can consolidate the best fish quickly.",
    },
    "TURNOVER": {
        "title": "Clean-water survival mode",
        "baits": "Jig, vibrating jig, slow spinnerbait, drop shot",
        "approach": "Lean on current, inflow, and the cleanest available water. Keep presentations compact.",
        "tip": "The best-looking water matters more than the textbook spot.",
    },
}


def build_trophy_daily(merged_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate trophy-catch conditions to daily lake level."""
    trophy_hours = merged_df[merged_df["max_weight"] >= TROPHY_WEIGHT].copy()
    if trophy_hours.empty:
        return pd.DataFrame()

    agg = {
        "max_weight": ("max_weight", "max"),
        "air_temp_f": ("temperature_2m", lambda x: x.mean() * 9 / 5 + 32),
        "pressure_inhg": ("pressure_msl", lambda x: x.mean() * 0.02953),
        "pressure_trend": ("pressure_trend_3h", "mean"),
        "moon_phase": ("moon_phase_name", "first"),
        "solunar_score": ("solunar_base_score", "mean"),
        "spawn_phase": ("spawn_phase", "first"),
    }
    if "water_temp_estimated" in trophy_hours.columns:
        agg["water_temp_f"] = ("water_temp_estimated", lambda x: x.mean() * 9 / 5 + 32)
    if "wind_class" in trophy_hours.columns:
        agg["wind_class"] = (
            "wind_class",
            lambda x: x.mode().iloc[0] if not x.mode().empty else "unknown",
        )

    trophy_daily = trophy_hours.groupby(["date", "lake_key"]).agg(**agg).reset_index()
    trophy_daily["date"] = pd.to_datetime(trophy_daily["date"])
    trophy_daily["month"] = trophy_daily["date"].dt.month
    return trophy_daily


def infer_current_phase(month: int) -> str:
    """Map calendar month to the dominant seasonal bass phase."""
    if month <= 2:
        return "WINTER"
    if month == 3:
        return "PRE_SPAWN"
    if month <= 5:
        return "SPAWN"
    if month == 6:
        return "POST_SPAWN"
    if month <= 8:
        return "SUMMER"
    if month <= 10:
        return "FALL"
    return "TURNOVER"


def render_pattern_note(title: str, body: str) -> None:
    """Render a single information card."""
    st.markdown(
        f"""
        <div class="tb-note">
            <strong>{title}</strong>
            <span>{body}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


ctx = bootstrap_dashboard("Indiana Trophy Bass Analyzer")
merged_df = ctx.merged_df
catches_df = ctx.catches_df
predictions_df = ctx.predictions_df

if merged_df.empty or catches_df.empty:
    st.info("No dashboard data is available for the current lake selection.")
    st.stop()

trophy_catches = catches_df[catches_df["weight_lbs"] >= TROPHY_WEIGHT].copy()
trophy_catches["month"] = trophy_catches["date"].dt.month
trophy_daily = build_trophy_daily(merged_df)

selection_label = ", ".join(lake_label(key, ctx.lake_configs) for key in ctx.selected_lakes[:3])
if len(ctx.selected_lakes) > 3:
    selection_label += f", +{len(ctx.selected_lakes) - 3} more"

render_page_header(
    "Indiana Trophy Bass Analyzer",
    (
        f"Command center for {selection_label}. Focused on when Indiana giants show up, "
        "what the water and weather look like, and which 2026 windows deserve real trip time."
    ),
    eyebrow="Audit + rebuild",
)

today = pd.Timestamp.today().normalize()
current_phase = infer_current_phase(today.month)
phase_plan = TECHNIQUE_BY_PHASE[current_phase]

best_lake = (
    trophy_catches["lake_key"].value_counts().idxmax() if not trophy_catches.empty else ctx.selected_lakes[0]
)
best_month = (
    MONTH_NAMES[trophy_catches["month"].value_counts().idxmax()]
    if not trophy_catches.empty
    else "N/A"
)
biggest_fish = trophy_catches["weight_lbs"].max() if not trophy_catches.empty else 0.0
trophy_rate = len(trophy_catches) / max(len(catches_df), 1) * 100

next_window = None
if predictions_df is not None and not predictions_df.empty:
    future_predictions = predictions_df[predictions_df["date"] >= today].copy()
    if not future_predictions.empty:
        next_window = future_predictions.nlargest(1, "max_probability").iloc[0]

metric_cols = st.columns(6)
metric_cols[0].metric("Recorded catches", f"{len(catches_df):,}")
metric_cols[1].metric("Trophy fish", f"{len(trophy_catches):,}")
metric_cols[2].metric("Biggest verified", f"{biggest_fish:.2f} lbs" if biggest_fish else "N/A")
metric_cols[3].metric("Trophy rate", f"{trophy_rate:.2f}%")
metric_cols[4].metric("Best lake", lake_label(best_lake, ctx.lake_configs))
metric_cols[5].metric("Peak month", best_month)

brief_cols = st.columns(3)
with brief_cols[0]:
    render_pattern_note(
        phase_plan["title"],
        f"{phase_plan['approach']} Throw {phase_plan['baits']}.",
    )
with brief_cols[1]:
    render_pattern_note(
        "Immediate tactical edge",
        phase_plan["tip"],
    )
with brief_cols[2]:
    if next_window is not None:
        render_pattern_note(
            "Next 2026 push",
            (
                f"{next_window['date']:%B %d, %Y} at "
                f"{lake_label(next_window['lake_key'], ctx.lake_configs)} "
                f"grades {next_window['rating']} at {next_window['max_probability']:.0%}."
            ),
        )
    else:
        render_pattern_note(
            "Prediction status",
            "Daily 2026 windows are not available yet for the current lake selection.",
        )

tab_patterns, tab_conditions, tab_predictions = st.tabs(
    ["Pattern Board", "Water and Weather", "2026 Windows"]
)

with tab_patterns:
    left, right = st.columns(2)

    with left:
        st.subheader("When the giants show")
        month_counts = (
            trophy_catches.groupby("month").size().reindex(range(1, 13), fill_value=0)
            if not trophy_catches.empty
            else pd.Series(0, index=range(1, 13))
        )
        fig_month = go.Figure(
            go.Bar(
                x=[MONTH_NAMES[m] for m in month_counts.index],
                y=month_counts.values,
                marker_color=[
                    ctx.colors["trophy_gold"] if value == month_counts.max() and value > 0 else ctx.colors["water_blue"]
                    for value in month_counts.values
                ],
                text=month_counts.values,
                textposition="outside",
            )
        )
        fig_month.update_layout(yaxis_title="7+ lb fish")
        st.plotly_chart(apply_figure_style(fig_month, height=360), use_container_width=True)

    with right:
        st.subheader("Moon phase pressure")
        phase_order = [
            "New",
            "Waxing Crescent",
            "First Quarter",
            "Waxing Gibbous",
            "Full",
            "Waning Gibbous",
            "Last Quarter",
            "Waning Crescent",
        ]
        phase_counts = (
            trophy_daily.groupby("moon_phase").size().reindex(phase_order, fill_value=0)
            if not trophy_daily.empty
            else pd.Series(0, index=phase_order)
        )
        fig_moon = go.Figure(
            go.Bar(
                x=phase_counts.index,
                y=phase_counts.values,
                marker_color=ctx.colors["bass_green"],
                text=phase_counts.values,
                textposition="outside",
            )
        )
        fig_moon.update_layout(yaxis_title="Trophy days")
        st.plotly_chart(apply_figure_style(fig_moon, height=360), use_container_width=True)

    bottom_left, bottom_right = st.columns(2)
    with bottom_left:
        st.subheader("Seasonal phase concentration")
        if not trophy_daily.empty and "spawn_phase" in trophy_daily.columns:
            phase_labels = {
                "PRE_SPAWN": "Pre-spawn",
                "SPAWN": "Spawn",
                "POST_SPAWN": "Post-spawn",
                "SUMMER": "Summer",
                "FALL": "Fall",
                "TURNOVER": "Turnover",
                "WINTER": "Winter",
            }
            phase_order = list(phase_labels)
            phase_counts = trophy_daily.groupby("spawn_phase").size().reindex(phase_order, fill_value=0)
            phase_counts = phase_counts[phase_counts > 0]
            fig_phase = go.Figure(
                go.Bar(
                    x=[phase_labels.get(phase, phase) for phase in phase_counts.index],
                    y=phase_counts.values,
                    marker_color=[
                        ctx.colors["trophy_gold"],
                        ctx.colors["clay"],
                        ctx.colors["water_blue"],
                        ctx.colors["bass_green"],
                        ctx.colors["moss"],
                        ctx.colors["fog"],
                        ctx.colors["water_blue"],
                    ][: len(phase_counts)],
                    text=phase_counts.values,
                    textposition="outside",
                )
            )
            fig_phase.update_layout(yaxis_title="Trophy days")
            st.plotly_chart(apply_figure_style(fig_phase, height=360), use_container_width=True)
        else:
            st.info("No spawn-phase data is available for the current selection.")

    with bottom_right:
        st.subheader("Lake leaderboard")
        if not trophy_catches.empty:
            lake_stats = (
                trophy_catches.groupby("lake_key")
                .agg(count=("weight_lbs", "size"), biggest=("weight_lbs", "max"), avg=("weight_lbs", "mean"))
                .sort_values("count", ascending=True)
                .reset_index()
            )
            lake_stats["lake_name"] = lake_stats["lake_key"].map(
                lambda key: lake_label(key, ctx.lake_configs)
            )
            fig_lakes = go.Figure(
                go.Bar(
                    y=lake_stats["lake_name"],
                    x=lake_stats["count"],
                    orientation="h",
                    marker_color=ctx.colors["bass_green"],
                    text=[
                        f"{count} fish | max {biggest:.1f} | avg {avg:.1f}"
                        for count, biggest, avg in zip(
                            lake_stats["count"], lake_stats["biggest"], lake_stats["avg"]
                        )
                    ],
                    textposition="outside",
                )
            )
            fig_lakes.update_layout(xaxis_title="Trophy fish")
            st.plotly_chart(apply_figure_style(fig_lakes, height=360), use_container_width=True)
        else:
            st.info("No trophy catches were found for the active lake selection.")

with tab_conditions:
    left, right = st.columns(2)
    all_with_catch = merged_df[merged_df["catch_count"] > 0].copy()
    if "water_temp_estimated" in all_with_catch.columns:
        all_with_catch["water_temp_f"] = all_with_catch["water_temp_estimated"] * 9 / 5 + 32
    all_with_catch["pressure_inhg"] = all_with_catch["pressure_msl"] * 0.02953

    with left:
        st.subheader("Water temperature bands")
        fig_temp = go.Figure()
        if "water_temp_f" in all_with_catch.columns:
            fig_temp.add_trace(
                go.Histogram(
                    x=all_with_catch["water_temp_f"].dropna(),
                    name="All catch periods",
                    marker_color=ctx.colors["water_blue"],
                    opacity=0.45,
                    nbinsx=30,
                    histnorm="probability",
                )
            )
        if not trophy_daily.empty and "water_temp_f" in trophy_daily.columns:
            fig_temp.add_trace(
                go.Histogram(
                    x=trophy_daily["water_temp_f"].dropna(),
                    name="Trophy days",
                    marker_color=ctx.colors["trophy_gold"],
                    opacity=0.8,
                    nbinsx=15,
                    histnorm="probability",
                )
            )
            fig_temp.add_vline(
                x=trophy_daily["water_temp_f"].mean(),
                line_dash="dash",
                line_color=ctx.colors["clay"],
            )
        fig_temp.update_layout(
            barmode="overlay",
            xaxis_title="Water temperature (F)",
            yaxis_title="Share of observations",
        )
        st.plotly_chart(apply_figure_style(fig_temp, height=380), use_container_width=True)

    with right:
        st.subheader("Barometric pressure")
        fig_pressure = go.Figure()
        fig_pressure.add_trace(
            go.Histogram(
                x=all_with_catch["pressure_inhg"].dropna(),
                name="All catch periods",
                marker_color=ctx.colors["fog"],
                opacity=0.45,
                nbinsx=28,
                histnorm="probability",
            )
        )
        if not trophy_daily.empty:
            fig_pressure.add_trace(
                go.Histogram(
                    x=trophy_daily["pressure_inhg"].dropna(),
                    name="Trophy days",
                    marker_color=ctx.colors["clay"],
                    opacity=0.8,
                    nbinsx=15,
                    histnorm="probability",
                )
            )
        fig_pressure.update_layout(
            barmode="overlay",
            xaxis_title="Pressure (inHg)",
            yaxis_title="Share of observations",
        )
        st.plotly_chart(apply_figure_style(fig_pressure, height=380), use_container_width=True)

    lower_left, lower_right = st.columns(2)
    with lower_left:
        st.subheader("Wind direction pattern")
        if not trophy_daily.empty and "wind_class" in trophy_daily.columns:
            wind_labels = {
                "south_warm": "South / SW",
                "northwest_cold": "NW / West",
                "north": "North",
                "northeast": "Northeast",
                "east_poor": "East / SE",
                "variable": "Variable",
                "unknown": "Unknown",
            }
            wind_counts = trophy_daily["wind_class"].value_counts()
            fig_wind = go.Figure(
                go.Bar(
                    x=[wind_labels.get(key, key) for key in wind_counts.index],
                    y=wind_counts.values,
                    marker_color=ctx.colors["bass_green"],
                    text=wind_counts.values,
                    textposition="outside",
                )
            )
            fig_wind.update_layout(yaxis_title="Trophy days")
            st.plotly_chart(apply_figure_style(fig_wind, height=360), use_container_width=True)
        else:
            st.info("Wind-class labels are not available for this slice of the data.")

    with lower_right:
        st.subheader("Weight vs. temperature")
        if not trophy_daily.empty:
            x_col = "water_temp_f" if "water_temp_f" in trophy_daily.columns else "air_temp_f"
            fig_scatter = px.scatter(
                trophy_daily,
                x=x_col,
                y="max_weight",
                color="lake_key",
                size="solunar_score" if "solunar_score" in trophy_daily.columns else None,
                hover_data=["date", "moon_phase", "pressure_inhg"],
                labels={
                    x_col: "Water temp (F)" if x_col == "water_temp_f" else "Air temp (F)",
                    "max_weight": "Best fish (lbs)",
                    "lake_key": "Lake",
                },
            )
            st.plotly_chart(apply_figure_style(fig_scatter, height=360), use_container_width=True)
        else:
            st.info("No trophy-day condition sample is available for this view.")

with tab_predictions:
    st.subheader("Upcoming windows that merit a trip")
    if predictions_df is None or predictions_df.empty:
        st.info("Run the forecast pipeline to populate 2026 windows.")
    else:
        future_predictions = predictions_df[predictions_df["date"] >= today].copy()
        if future_predictions.empty:
            st.info("The forecast file exists, but it does not contain dates after today.")
        else:
            top_future = future_predictions.nlargest(18, "max_probability").copy()
            top_future["lake_name"] = top_future["lake_key"].map(
                lambda key: lake_label(key, ctx.lake_configs)
            )
            top_future["date_label"] = top_future["date"].dt.strftime("%b %d (%a)")
            top_future["probability_pct"] = (top_future["max_probability"] * 100).round(1)

            left, right = st.columns([1.2, 1])
            with left:
                fig_windows = px.bar(
                    top_future.sort_values("max_probability", ascending=True),
                    x="max_probability",
                    y="date_label",
                    color="lake_name",
                    orientation="h",
                    labels={"max_probability": "Max trophy probability", "date_label": ""},
                    hover_data=["rating", "best_hour"],
                )
                fig_windows.update_xaxes(tickformat=".0%")
                st.plotly_chart(apply_figure_style(fig_windows, height=520), use_container_width=True)

            with right:
                upcoming_table = top_future[
                    ["date_label", "lake_name", "probability_pct", "best_hour", "rating"]
                ].copy()
                upcoming_table.columns = ["Date", "Lake", "Prob %", "Best hour", "Grade"]
                st.dataframe(upcoming_table, use_container_width=True, hide_index=True)

            monthly = future_predictions.copy()
            monthly["month"] = monthly["date"].dt.month
            monthly_summary = (
                monthly.groupby("month")
                .agg(
                    avg_probability=("max_probability", "mean"),
                    strong_days=("max_probability", lambda x: int((x >= x.quantile(0.75)).sum())),
                )
                .reset_index()
            )
            monthly_summary["month_name"] = monthly_summary["month"].map(MONTH_NAMES)
            fig_monthly = go.Figure()
            fig_monthly.add_trace(
                go.Bar(
                    x=monthly_summary["month_name"],
                    y=monthly_summary["strong_days"],
                    name="Upper-quartile days",
                    marker_color=ctx.colors["bass_green"],
                )
            )
            fig_monthly.add_trace(
                go.Scatter(
                    x=monthly_summary["month_name"],
                    y=monthly_summary["avg_probability"],
                    name="Average probability",
                    mode="lines+markers",
                    marker={"color": ctx.colors["trophy_gold"], "size": 10},
                    line={"color": ctx.colors["trophy_gold"], "width": 3},
                    yaxis="y2",
                )
            )
            fig_monthly.update_layout(
                yaxis_title="Strong days",
                yaxis2={
                    "title": "Average probability",
                    "overlaying": "y",
                    "side": "right",
                    "tickformat": ".0%",
                },
            )
            st.plotly_chart(apply_figure_style(fig_monthly, height=360), use_container_width=True)

with st.expander("Trophy ledger"):
    display_cols = ["date", "lake_key", "weight_lbs"]
    if "angler" in trophy_catches.columns:
        display_cols.append("angler")
    if "source_type" in trophy_catches.columns:
        display_cols.append("source_type")

    ledger = trophy_catches[display_cols].sort_values("weight_lbs", ascending=False).copy()
    ledger["date"] = pd.to_datetime(ledger["date"]).dt.strftime("%Y-%m-%d")
    ledger["lake_key"] = ledger["lake_key"].map(lambda key: lake_label(key, ctx.lake_configs))
    ledger.rename(
        columns={
            "date": "Date",
            "lake_key": "Lake",
            "weight_lbs": "Weight (lbs)",
            "angler": "Angler",
            "source_type": "Source",
        },
        inplace=True,
    )
    st.dataframe(ledger, use_container_width=True, hide_index=True)

st.caption(
    "Data sources: USA Bassin results, Indiana DNR records, Open-Meteo ERA5 reanalysis, USGS water services, and computed solunar timing."
)
