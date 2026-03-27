"""Correlation analysis between environmental conditions and trophy catches."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

# Features to analyze against catch outcomes
ANALYSIS_FEATURES = [
    "temperature_2m", "water_temp_estimated", "pressure_msl",
    "pressure_trend_3h", "pressure_trend_6h",
    "wind_speed_10m", "wind_gusts_10m", "relative_humidity_2m",
    "cloud_cover", "precipitation", "moon_illumination",
    "solunar_base_score", "temp_stability_3day",
    "hour", "month", "day_of_year",
]


def compute_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Compute correlation matrix for numeric features vs catch metrics."""
    target_cols = ["catch_count", "trophy_count", "trophy_caught", "max_weight", "avg_weight"]
    feature_cols = [c for c in ANALYSIS_FEATURES if c in df.columns]
    target_cols = [c for c in target_cols if c in df.columns]

    all_cols = feature_cols + target_cols
    corr = df[all_cols].corr()
    return corr


def compute_feature_importance(df: pd.DataFrame, target: str = "trophy_caught") -> pd.DataFrame:
    """Rank features by mutual information and correlation with target."""
    if target not in df.columns:
        logger.warning("Target '%s' not in dataframe", target)
        return pd.DataFrame()

    feature_cols = [c for c in ANALYSIS_FEATURES if c in df.columns]
    results = []

    for col in feature_cols:
        valid = df[[col, target]].dropna()
        if len(valid) < 100:
            continue
        if valid[col].nunique() < 2 or valid[target].nunique() < 2:
            continue

        # Pearson correlation
        pearson_r, pearson_p = stats.pearsonr(valid[col], valid[target])

        # Point-biserial if target is binary
        if valid[target].nunique() == 2:
            pb_r, pb_p = stats.pointbiserialr(valid[target], valid[col])
        else:
            pb_r, pb_p = pearson_r, pearson_p

        # KS test: compare feature distribution for trophy vs non-trophy
        if valid[target].nunique() == 2:
            group_0 = valid.loc[valid[target] == 0, col]
            group_1 = valid.loc[valid[target] == 1, col]
            if len(group_1) >= 10:
                ks_stat, ks_p = stats.ks_2samp(group_0, group_1)
            else:
                ks_stat, ks_p = np.nan, np.nan
        else:
            ks_stat, ks_p = np.nan, np.nan

        results.append({
            "feature": col,
            "pearson_r": pearson_r,
            "pearson_p": pearson_p,
            "pointbiserial_r": pb_r,
            "pointbiserial_p": pb_p,
            "ks_statistic": ks_stat,
            "ks_pvalue": ks_p,
            "abs_correlation": abs(pearson_r),
        })

    if not results:
        return pd.DataFrame()

    result_df = pd.DataFrame(results).sort_values("abs_correlation", ascending=False)
    return result_df


def compute_conditional_rates(df: pd.DataFrame, feature: str,
                              target: str = "trophy_caught",
                              n_bins: int = 10) -> pd.DataFrame:
    """Compute trophy catch rate across bins of a continuous feature."""
    if feature not in df.columns or target not in df.columns:
        return pd.DataFrame()

    valid = df[[feature, target]].dropna().copy()
    if len(valid) < 2 or valid[feature].nunique() < 2:
        return pd.DataFrame()

    try:
        valid["bin"] = pd.qcut(valid[feature], q=n_bins, duplicates="drop")
    except ValueError:
        return pd.DataFrame()

    rates = valid.groupby("bin", observed=True).agg(
        count=(target, "count"),
        trophy_rate=(target, "mean"),
        feature_mean=(feature, "mean"),
    ).reset_index()

    return rates


def generate_insights(df: pd.DataFrame) -> list[dict]:
    """Generate natural-language insight cards from the data."""
    insights = []

    importance = compute_feature_importance(df)
    if len(importance) == 0:
        return insights

    # Top correlated features
    top = importance.head(5)
    for _, row in top.iterrows():
        direction = "positively" if row["pearson_r"] > 0 else "negatively"
        insights.append({
            "type": "correlation",
            "feature": row["feature"],
            "message": (
                f"{row['feature']} is {direction} correlated with trophy catches "
                f"(r={row['pearson_r']:.3f}, p={row['pearson_p']:.1e})"
            ),
            "strength": row["abs_correlation"],
        })

    # Best conditions summary
    if "trophy_caught" in df.columns:
        trophy_df = df[df["trophy_caught"] == 1]
        if len(trophy_df) > 10:
            for col in ["pressure_msl", "temperature_2m", "wind_speed_10m"]:
                if col in trophy_df.columns:
                    mean_val = trophy_df[col].mean()
                    overall_mean = df[col].mean()
                    insights.append({
                        "type": "optimal_range",
                        "feature": col,
                        "message": (
                            f"Trophy catches average {col}={mean_val:.1f} "
                            f"vs overall {overall_mean:.1f}"
                        ),
                        "trophy_mean": mean_val,
                        "overall_mean": overall_mean,
                    })

    return insights
