"""Solunar Calendar -- moon phases, solunar periods, and best fishing times."""

import sys
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import calendar

for candidate in Path(__file__).resolve().parents:
    if (candidate / "pyproject.toml").exists() and (candidate / "src").exists():
        candidate_str = str(candidate)
        if candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)
        break

from src.dashboard.ui import apply_figure_style, bootstrap_dashboard, render_page_header

ctx = bootstrap_dashboard("Solunar Calendar")
COLORS = ctx.colors

render_page_header(
    "Solunar Calendar",
    "Calendar-style solunar view for the active lakes, with moon illumination and best hourly periods for the selected month.",
)

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
df = ctx.merged_df

if df is None or df.empty:
    st.info("No data available. Please check that data files exist and filters are not too restrictive.")
    st.stop()

# ---------------------------------------------------------------------------
# Month / Year selector
# ---------------------------------------------------------------------------
col_y, col_m = st.columns(2)
available_years = sorted(df["datetime"].dt.year.unique().tolist()) if "datetime" in df.columns else [2025]
with col_y:
    selected_year = st.selectbox("Year", available_years, index=len(available_years) - 1)
with col_m:
    selected_month = st.selectbox("Month", list(range(1, 13)),
                                  format_func=lambda m: calendar.month_name[m],
                                  index=4)  # Default May (prime bass month)

# Filter to selected month
if "datetime" in df.columns:
    month_df = df[
        (df["datetime"].dt.year == selected_year) &
        (df["datetime"].dt.month == selected_month)
    ].copy()
else:
    month_df = pd.DataFrame()

# ---------------------------------------------------------------------------
# Monthly calendar grid with solunar scores
# ---------------------------------------------------------------------------
st.subheader(f"Solunar Calendar - {calendar.month_name[selected_month]} {selected_year}")

if not month_df.empty and "solunar_base_score" in month_df.columns:
    # Aggregate daily solunar scores
    daily_solunar = (
        month_df.groupby(month_df["datetime"].dt.day)
        .agg(
            avg_solunar=("solunar_base_score", "mean"),
            max_solunar=("solunar_base_score", "max"),
            moon_illum=("moon_illumination", "mean") if "moon_illumination" in month_df.columns else ("solunar_base_score", "count"),
            catches=("catch_count", "sum") if "catch_count" in month_df.columns else ("solunar_base_score", "count"),
            trophies=("trophy_count", "sum") if "trophy_count" in month_df.columns else ("solunar_base_score", "count"),
        )
        .reset_index()
    )
    daily_solunar.columns = ["day", "avg_solunar", "max_solunar", "moon_illum", "catches", "trophies"]

    # Build calendar grid
    cal = calendar.Calendar(firstweekday=6)  # Sunday start
    month_days = cal.monthdayscalendar(selected_year, selected_month)
    day_names = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]

    solunar_lookup = daily_solunar.set_index("day").to_dict("index")

    # Render calendar as HTML grid
    html = '<table style="width:100%; border-collapse:collapse; font-family:sans-serif;">'
    html += "<tr>"
    for dn in day_names:
        html += f'<th style="padding:8px; text-align:center; background:#f0f0f0; border:1px solid #ddd;">{dn}</th>'
    html += "</tr>"

    max_score = daily_solunar["max_solunar"].max() if not daily_solunar.empty else 1.0
    if max_score == 0:
        max_score = 1.0

    for week in month_days:
        html += "<tr>"
        for day in week:
            if day == 0:
                html += '<td style="border:1px solid #eee; padding:6px;">&nbsp;</td>'
            else:
                info = solunar_lookup.get(day, {})
                score = info.get("avg_solunar", 0)
                moon = info.get("moon_illum", 0)
                catches = info.get("catches", 0)
                trophies = info.get("trophies", 0)

                # Color intensity based on solunar score
                intensity = min(score / max_score, 1.0) if max_score > 0 else 0
                r = int(45 + (243 - 45) * (1 - intensity))  # from light to bass_green
                g = int(80 + (156 - 80) * intensity)
                b = int(22 + (18 - 22) * intensity)
                bg = f"rgb({r},{g},{b})"
                text_color = "white" if intensity > 0.4 else "#333"

                trophy_marker = f"<br><span style='color:{COLORS['trophy_gold']};'>T:{trophies:.0f}</span>" if trophies > 0 else ""

                html += (
                    f'<td style="border:1px solid #ddd; padding:6px; background:{bg}; '
                    f'color:{text_color}; text-align:center; vertical-align:top; min-width:80px;">'
                    f'<strong>{day}</strong><br>'
                    f'<small>Sol: {score:.1f}</small><br>'
                    f'<small>Moon: {moon:.0f}%</small>'
                    f'{trophy_marker}'
                    f'</td>'
                )
        html += "</tr>"
    html += "</table>"

    st.markdown(html, unsafe_allow_html=True)

    st.caption("Color intensity represents solunar score strength. T = trophy catches.")
else:
    st.info("No solunar data available for the selected month.")

st.markdown("---")

# ---------------------------------------------------------------------------
# Moon phase visualization (illumination gauge)
# ---------------------------------------------------------------------------
st.subheader("Moon Illumination")

if not month_df.empty and "moon_illumination" in month_df.columns:
    daily_moon = (
        month_df.groupby(month_df["datetime"].dt.day)["moon_illumination"]
        .mean()
        .reset_index()
    )
    daily_moon.columns = ["day", "illumination"]

    fig_moon = go.Figure()
    fig_moon.add_trace(go.Scatter(
        x=daily_moon["day"],
        y=daily_moon["illumination"],
        mode="lines+markers",
        line=dict(color=COLORS["trophy_gold"], width=3),
        marker=dict(
            size=daily_moon["illumination"] / 100 * 20 + 5,
            color=daily_moon["illumination"],
            colorscale=[[0, "#1a1a2e"], [0.5, COLORS["trophy_gold"]], [1, "#ffffcc"]],
            showscale=True,
            colorbar=dict(title="Illumination %"),
        ),
        fill="tozeroy",
        fillcolor="rgba(243, 156, 18, 0.1)",
        name="Moon Illumination",
    ))
    fig_moon.update_layout(
        title=f"Moon Illumination - {calendar.month_name[selected_month]} {selected_year}",
        xaxis_title="Day of Month",
        yaxis_title="Illumination (%)",
        yaxis=dict(range=[0, 105]),
    )
    st.plotly_chart(apply_figure_style(fig_moon, height=350), use_container_width=True)

    # Gauges for current or mid-month illumination
    mid_day = 15
    mid_moon = daily_moon.loc[daily_moon["day"] == mid_day, "illumination"]
    if not mid_moon.empty:
        mid_val = mid_moon.values[0]
        phase_name = (
            "New Moon" if mid_val < 5 else
            "Crescent" if mid_val < 25 else
            "First/Last Quarter" if mid_val < 60 else
            "Gibbous" if mid_val < 95 else
            "Full Moon"
        )
        st.metric(f"Mid-Month Phase (Day {mid_day})", f"{phase_name} ({mid_val:.0f}%)")
else:
    st.info("No moon illumination data available.")

st.markdown("---")

# ---------------------------------------------------------------------------
# Best fishing times by solunar score
# ---------------------------------------------------------------------------
st.subheader("Best Fishing Times by Solunar Period")

if not month_df.empty and "solunar_base_score" in month_df.columns and "hour" in month_df.columns:
    hourly_solunar = (
        month_df.groupby("hour")
        .agg(
            avg_solunar=("solunar_base_score", "mean"),
            total_catches=("catch_count", "sum") if "catch_count" in month_df.columns else ("solunar_base_score", "count"),
            total_trophies=("trophy_count", "sum") if "trophy_count" in month_df.columns else ("solunar_base_score", "count"),
        )
        .reset_index()
    )

    fig_hourly = go.Figure()
    fig_hourly.add_trace(go.Bar(
        x=hourly_solunar["hour"],
        y=hourly_solunar["avg_solunar"],
        name="Avg Solunar Score",
        marker_color=COLORS["water_blue"],
        opacity=0.7,
    ))
    fig_hourly.add_trace(go.Scatter(
        x=hourly_solunar["hour"],
        y=hourly_solunar["total_trophies"],
        name="Trophy Catches",
        mode="lines+markers",
        marker=dict(color=COLORS["trophy_gold"], size=8),
        line=dict(color=COLORS["trophy_gold"], width=2),
        yaxis="y2",
    ))
    fig_hourly.update_layout(
        title="Solunar Score and Trophy Catches by Hour",
        xaxis_title="Hour of Day",
        yaxis=dict(title="Avg Solunar Score", side="left"),
        yaxis2=dict(title="Trophy Catches", side="right", overlaying="y"),
        legend=dict(x=0.01, y=0.99),
    )
    st.plotly_chart(apply_figure_style(fig_hourly, height=400), use_container_width=True)

    # Top 5 best time windows
    top_hours = hourly_solunar.nlargest(5, "avg_solunar")
    st.markdown("**Top 5 Hours by Solunar Score:**")
    for _, row in top_hours.iterrows():
        hour = int(row["hour"])
        score = row["avg_solunar"]
        period = "AM" if hour < 12 else "PM"
        display_hour = hour if hour <= 12 else hour - 12
        if display_hour == 0:
            display_hour = 12
        st.write(f"- **{display_hour}:00 {period}** -- Solunar Score: {score:.2f}")
else:
    st.info("Solunar and hourly data not available.")

st.markdown("---")

# ---------------------------------------------------------------------------
# Solunar score vs catch rate scatter
# ---------------------------------------------------------------------------
st.subheader("Solunar Score vs Catch Outcomes")

if "solunar_base_score" in month_df.columns and "catch_count" in month_df.columns:
    fig_sol = px.scatter(
        month_df,
        x="solunar_base_score",
        y="catch_count",
        color="trophy_caught" if "trophy_caught" in month_df.columns else None,
        color_discrete_map={0: COLORS["water_blue"], 1: COLORS["trophy_gold"]},
        opacity=0.5,
        title="Solunar Score vs Catches",
        labels={"solunar_base_score": "Solunar Base Score", "catch_count": "Catch Count",
                "trophy_caught": "Trophy?"},
    )
    st.plotly_chart(apply_figure_style(fig_sol, height=400), use_container_width=True)
