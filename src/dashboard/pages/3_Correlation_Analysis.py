"""Correlation Analysis -- feature importance and conditional rates."""

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

from src.analysis.correlations import (
    compute_correlation_matrix,
    compute_feature_importance,
    compute_conditional_rates,
    generate_insights,
)
from src.dashboard.ui import bootstrap_dashboard, render_dataframe, render_page_header, render_plotly

ctx = bootstrap_dashboard("Correlation Analysis")
COLORS = ctx.colors

render_page_header(
    "Correlation Analysis",
    "Feature-importance view across the selected lakes, with conditional trophy rates and automated takeaways.",
)

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
df = ctx.merged_df

if df is None or df.empty:
    st.info("No data available. Please check that data files exist and filters are not too restrictive.")
    st.stop()

# ---------------------------------------------------------------------------
# Correlation heatmap
# ---------------------------------------------------------------------------
st.subheader("Correlation Heatmap")

try:
    corr_matrix = compute_correlation_matrix(df)
    if not corr_matrix.empty:
        fig_corr = px.imshow(
            corr_matrix,
            color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1,
            title="Feature Correlation Matrix",
            aspect="auto",
        )
        render_plotly(fig_corr, height=700)
    else:
        st.info("Unable to compute correlation matrix with available data.")
except Exception as e:
    st.warning(f"Error computing correlations: {e}")

st.markdown("---")

# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------
st.subheader("Feature Importance (vs Trophy Catches)")

try:
    importance = compute_feature_importance(df)
    if not importance.empty:
        fig_imp = px.bar(
            importance.head(15),
            x="abs_correlation",
            y="feature",
            orientation="h",
            color="pearson_r",
            color_continuous_scale=["#e74c3c", "#95a5a6", COLORS["bass_green"]],
            color_continuous_midpoint=0,
            title="Top Features by Absolute Correlation with Trophy Catches",
            labels={"abs_correlation": "|Correlation|", "feature": "", "pearson_r": "Direction"},
        )
        fig_imp.update_layout(yaxis=dict(autorange="reversed"))
        render_plotly(fig_imp, height=500)

        # Show detail table
        with st.expander("Feature Importance Details"):
            render_dataframe(
                importance[["feature", "pearson_r", "pearson_p", "pointbiserial_r",
                            "ks_statistic", "ks_pvalue", "abs_correlation"]]
                .round(4),
            )
    else:
        st.info("Not enough data to compute feature importance.")
except Exception as e:
    st.warning(f"Error computing feature importance: {e}")

st.markdown("---")

# ---------------------------------------------------------------------------
# Conditional rate plots
# ---------------------------------------------------------------------------
st.subheader("Trophy Rate by Condition Bins")

cond_features = [c for c in [
    "pressure_msl", "temperature_2m", "water_temp_estimated",
    "wind_speed_10m", "cloud_cover", "moon_illumination",
    "solunar_base_score", "pressure_trend_3h",
] if c in df.columns]

if cond_features:
    selected_cond = st.selectbox("Select feature for conditional analysis", cond_features)
    n_bins = st.slider("Number of bins", 5, 20, 10)

    try:
        rates = compute_conditional_rates(df, selected_cond, n_bins=n_bins)
        if not rates.empty:
            fig_rate = go.Figure()
            fig_rate.add_trace(go.Bar(
                x=rates["feature_mean"],
                y=rates["trophy_rate"],
                marker_color=COLORS["trophy_gold"],
                name="Trophy Rate",
                text=rates["count"],
                textposition="outside",
                texttemplate="%{text} obs",
            ))
            fig_rate.update_layout(
                title=f"Trophy Rate by {selected_cond} Bins",
                xaxis_title=selected_cond,
                yaxis_title="Trophy Rate",
                yaxis_tickformat=".1%",
            )
            render_plotly(fig_rate, height=400)
        else:
            st.info("Not enough data for conditional rate analysis.")
    except Exception as e:
        st.warning(f"Error computing conditional rates: {e}")
else:
    st.info("No numeric condition features available.")

st.markdown("---")

# ---------------------------------------------------------------------------
# Insight cards
# ---------------------------------------------------------------------------
st.subheader("Automated Insights")

try:
    insights = generate_insights(df)
    if insights:
        for insight in insights:
            if insight["type"] == "correlation":
                strength = insight.get("strength", 0)
                if strength > 0.15:
                    st.success(insight["message"])
                elif strength > 0.05:
                    st.info(insight["message"])
                else:
                    st.warning(insight["message"])
            elif insight["type"] == "optimal_range":
                st.info(insight["message"])
            else:
                st.info(insight["message"])
    else:
        st.info("Not enough data to generate insights.")
except Exception as e:
    st.warning(f"Error generating insights: {e}")
