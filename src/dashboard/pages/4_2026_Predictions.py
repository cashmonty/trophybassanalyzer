"""2026 Predictions -- calendar heatmap, top windows, and go/no-go indicators."""

import sys
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.express as px

for candidate in Path(__file__).resolve().parents:
    if (candidate / "pyproject.toml").exists() and (candidate / "src").exists():
        candidate_str = str(candidate)
        if candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)
        break

from src.dashboard.ui import (
    MONTH_NAMES,
    bootstrap_dashboard,
    lake_label,
    render_dataframe,
    render_page_header,
    render_plotly,
)

ctx = bootstrap_dashboard("2026 Predictions")
COLORS = ctx.colors | {"red": "#B44D34", "light_green": "#6F8A48", "gray": "#95A19A"}

render_page_header(
    "2026 Predictions",
    "Forward-looking daily trophy windows for the active lakes, filtered to dates that still matter.",
)

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
predictions = ctx.predictions_df
lake_configs = ctx.lake_configs
selected_lakes = ctx.selected_lakes

if predictions is None or predictions.empty:
    st.info("No prediction data available. Run the prediction pipeline to generate `data/processed/predictions_2026.parquet`.")
    st.stop()

# Filter to selected lakes
if selected_lakes and "lake_key" in predictions.columns:
    pred_df = predictions[predictions["lake_key"].isin(selected_lakes)].copy()
else:
    pred_df = predictions.copy()

if pred_df.empty:
    st.info("No predictions for selected lakes.")
    st.stop()

pred_df["date"] = pd.to_datetime(pred_df["date"])

# ---------------------------------------------------------------------------
# Lake selector for calendar view
# ---------------------------------------------------------------------------
available_lakes = sorted(pred_df["lake_key"].unique().tolist())
cal_lake = st.selectbox(
    "Select lake for calendar view",
    available_lakes,
    format_func=lambda key: lake_label(key, lake_configs),
)

lake_pred = pred_df[pred_df["lake_key"] == cal_lake].copy()

# ---------------------------------------------------------------------------
# Calendar heatmap
# ---------------------------------------------------------------------------
st.subheader("Daily Trophy Probability Calendar")

if "max_probability" in lake_pred.columns:
    lake_pred["month"] = lake_pred["date"].dt.month

    fig_cal = px.scatter(
        lake_pred,
        x="date",
        y="max_probability",
        color="max_probability",
        color_continuous_scale=["#f7fbff", COLORS["trophy_gold"], COLORS["bass_green"]],
        size="max_probability",
        title=f"Daily Max Trophy Probability - {lake_label(cal_lake, lake_configs)}",
        labels={"date": "Date", "max_probability": "Probability"},
        hover_data=["best_hour", "rating"] if "best_hour" in lake_pred.columns else None,
    )
    render_plotly(fig_cal, height=400)
else:
    st.info("Probability column not found in predictions.")

st.markdown("---")

# ---------------------------------------------------------------------------
# Top 20 predicted windows
# ---------------------------------------------------------------------------
st.subheader("Top 20 Predicted Windows")

top_windows = pred_df.nlargest(20, "max_probability").copy()
top_windows["lake_name"] = top_windows["lake_key"].map(
    lambda key: lake_label(key, lake_configs)
)

display_cols = ["date", "lake_name", "max_probability", "mean_probability", "best_hour", "rating"]
display_cols = [c for c in display_cols if c in top_windows.columns]

if display_cols:
    styled_df = top_windows[display_cols].copy()
    styled_df["date"] = styled_df["date"].dt.strftime("%Y-%m-%d")
    if "max_probability" in styled_df.columns:
        styled_df["max_probability"] = styled_df["max_probability"].map("{:.1%}".format)
    if "mean_probability" in styled_df.columns:
        styled_df["mean_probability"] = styled_df["mean_probability"].map("{:.1%}".format)
    render_dataframe(styled_df, hide_index=True)
else:
    st.info("Unable to display top windows -- columns missing.")

st.markdown("---")

# ---------------------------------------------------------------------------
# Go/No-Go indicator for upcoming days
# ---------------------------------------------------------------------------
st.subheader("Go / No-Go Indicator")

if "rating" in pred_df.columns:
    upcoming = lake_pred[lake_pred["date"] >= pd.Timestamp.today().normalize()].sort_values("date").head(30).copy()

    def rating_color(rating):
        rating = str(rating).strip().title()
        if rating == "Excellent":
            return COLORS["bass_green"]
        elif rating == "Great":
            return COLORS["light_green"]
        elif rating == "Good":
            return COLORS["trophy_gold"]
        elif rating == "Fair":
            return COLORS.get("red", "#e74c3c")
        else:
            return COLORS.get("gray", "#95a5a6")

    def rating_emoji(rating):
        rating = str(rating).strip().title()
        if rating in ("Excellent", "Great"):
            return "GO"
        elif rating == "Good":
            return "MAYBE"
        else:
            return "NO-GO"

    if not upcoming.empty:
        # Display as colored cards in rows of 7
        rows = [upcoming.iloc[i:i + 7] for i in range(0, len(upcoming), 7)]
        for row_data in rows:
            cols = st.columns(len(row_data))
            for col, (_, day) in zip(cols, row_data.iterrows()):
                indicator = rating_emoji(day["rating"])
                date_str = day["date"].strftime("%b %d")
                prob_str = f"{day['max_probability']:.0%}" if "max_probability" in day.index else ""
                bg_color = rating_color(day["rating"])
                col.markdown(
                    f"""
                    <div style="background-color:{bg_color}; color:white; padding:10px;
                                border-radius:8px; text-align:center; margin:2px;">
                        <strong>{date_str}</strong><br>
                        {indicator}<br>
                        <small>{prob_str}</small>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
    else:
        st.info("No upcoming predictions available.")
elif "max_probability" in pred_df.columns:
    # Fallback: use probability thresholds
    upcoming = (
        lake_pred[lake_pred["date"] >= pd.Timestamp.today().normalize()]
        .sort_values("date")
        .head(14)
        .copy()
    )
    if not upcoming.empty:
        upcoming["indicator"] = pd.cut(
            upcoming["max_probability"],
            bins=[0, 0.3, 0.6, 1.0],
            labels=["NO-GO", "MAYBE", "GO"],
        )
        cols = st.columns(min(7, len(upcoming)))
        for i, (_, day) in enumerate(upcoming.iterrows()):
            col_idx = i % 7
            date_str = day["date"].strftime("%b %d")
            ind = str(day["indicator"])
            color_map = {"GO": COLORS["bass_green"], "MAYBE": COLORS["trophy_gold"],
                         "NO-GO": COLORS.get("red", "#e74c3c")}
            bg = color_map.get(ind, COLORS.get("gray", "#95a5a6"))
            cols[col_idx].markdown(
                f"""<div style="background-color:{bg}; color:white; padding:8px;
                            border-radius:8px; text-align:center; margin:2px;">
                    <strong>{date_str}</strong><br>{ind}<br>
                    <small>{day['max_probability']:.0%}</small></div>""",
                unsafe_allow_html=True,
            )
    else:
        st.info("No upcoming predictions available.")

st.markdown("---")

# ---------------------------------------------------------------------------
# Monthly summary bar chart
# ---------------------------------------------------------------------------
st.subheader("Monthly Summary of Best Days")

if "max_probability" in pred_df.columns:
    pred_df["month"] = pred_df["date"].dt.month

    # Count "good" days per month (above median probability)
    threshold = pred_df["max_probability"].median()
    monthly_summary = (
        pred_df[pred_df["max_probability"] >= threshold]
        .groupby("month")
        .agg(good_days=("date", "nunique"), avg_probability=("max_probability", "mean"))
        .reset_index()
    )
    monthly_summary["month_name"] = monthly_summary["month"].map(MONTH_NAMES)

    fig_month = px.bar(
        monthly_summary,
        x="month_name",
        y="good_days",
        color="avg_probability",
        color_continuous_scale=["#f7fbff", COLORS["trophy_gold"], COLORS["bass_green"]],
        title="Above-Average Days per Month (All Selected Lakes)",
        labels={"good_days": "Good Days", "month_name": "Month", "avg_probability": "Avg Prob"},
    )
    render_plotly(fig_month, height=400)
else:
    st.info("Probability data not available for monthly summary.")
