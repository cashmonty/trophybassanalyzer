"""LightGBM model for trophy bass catch prediction."""

from __future__ import annotations

import json
import logging

import lightgbm as lgb
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
)

from src.config import DATA_DIR

logger = logging.getLogger(__name__)

MODEL_DIR = DATA_DIR / "models"

FEATURE_COLS = [
    # Core environmental conditions (what actually matters for bass)
    "water_temp_estimated", "temperature_2m",
    "pressure_msl", "pressure_trend_3h", "pressure_trend_6h",
    "wind_speed_10m", "wind_gusts_10m",
    "relative_humidity_2m", "cloud_cover", "precipitation",
    # Astronomical
    "moon_illumination", "solunar_base_score",
    # Stability & trends
    "temp_stability_3day",
    "temp_change_3day", "is_warming_trend", "warming_streak",
    "prefrontal_feed_window", "hours_to_front",
    "water_level_change_1d", "days_since_last_front",
    # Time features — use month for seasonality but NOT day_of_week
    # (day_of_week creates artificial signal from synthetic Saturday dates)
    "month", "day_of_year",
]

CATEGORICAL_FEATURES = [
    "pressure_trend_class", "front_type", "spawn_phase",
    "wind_class", "water_level_trend",
]


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Prepare feature matrix, encoding categoricals."""
    available_numeric = [c for c in FEATURE_COLS if c in df.columns]
    available_cat = [c for c in CATEGORICAL_FEATURES if c in df.columns]

    # Encode categoricals
    encoded_dfs = []
    cat_encoded_cols = []
    for col in available_cat:
        dummies = pd.get_dummies(df[col], prefix=col, drop_first=False)
        encoded_dfs.append(dummies)
        cat_encoded_cols.extend(dummies.columns.tolist())

    feature_cols = available_numeric + cat_encoded_cols

    X = df[available_numeric].copy()
    for edf in encoded_dfs:
        X = pd.concat([X, edf], axis=1)

    return X, feature_cols


def train_model(
    df: pd.DataFrame,
    target: str = "trophy_caught",
    train_end_year: int = 2023,
    val_year: int = 2024,
    test_year: int = 2025,
) -> dict:
    """Train LightGBM model with time-based split."""
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["year"] = df["datetime"].dt.year

    # Prepare features
    X, feature_cols = prepare_features(df)
    y = df[target].astype(int)

    # Time-based split
    train_mask = df["year"] <= train_end_year
    val_mask = df["year"] == val_year
    test_mask = df["year"] == test_year

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    logger.info(
        f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}"
    )

    # Handle class imbalance
    pos_count = y_train.sum()
    neg_count = len(y_train) - pos_count
    scale_pos = neg_count / max(pos_count, 1)

    params = {
        "objective": "binary",
        "metric": ["binary_logloss", "auc"],
        "boosting_type": "gbdt",
        "num_leaves": 20,
        "max_depth": 6,
        "min_child_samples": 10,
        "learning_rate": 0.03,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "lambda_l1": 0.05,
        "lambda_l2": 0.5,
        "scale_pos_weight": min(scale_pos, 100),  # Cap to avoid extreme weighting
        "verbose": -1,
        "n_jobs": -1,
    }

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    model = lgb.train(
        params,
        train_data,
        num_boost_round=300,
        valid_sets=[train_data, val_data],
        callbacks=[
            lgb.early_stopping(30),
            lgb.log_evaluation(50),
        ],
    )

    # Evaluate
    results = {}
    for name, X_eval, y_eval in [("val", X_val, y_val), ("test", X_test, y_test)]:
        if len(X_eval) == 0:
            continue
        y_pred = model.predict(X_eval)
        auc = roc_auc_score(y_eval, y_pred) if y_eval.nunique() > 1 else 0
        ap = average_precision_score(y_eval, y_pred) if y_eval.nunique() > 1 else 0
        results[f"{name}_auc"] = auc
        results[f"{name}_ap"] = ap
        logger.info(f"{name}: AUC={auc:.4f}, AP={ap:.4f}")

    # Feature importance
    importance = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importance(importance_type="gain"),
    }).sort_values("importance", ascending=False)

    # Save model and feature columns
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(str(MODEL_DIR / "trophy_lgbm.txt"))
    importance.to_csv(MODEL_DIR / "feature_importance.csv", index=False)

    with open(MODEL_DIR / "feature_cols.json", "w") as f:
        json.dump(feature_cols, f)

    with open(MODEL_DIR / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Model saved to {MODEL_DIR}")
    logger.info(f"\nTop 10 features:\n{importance.head(10)}")

    return {
        "model": model,
        "metrics": results,
        "importance": importance,
        "feature_cols": feature_cols,
    }


def load_feature_cols() -> list[str]:
    """Load the feature columns used during training."""
    path = MODEL_DIR / "feature_cols.json"
    if not path.exists():
        raise FileNotFoundError(f"No feature columns file at {path}")
    with open(path) as f:
        return json.load(f)


def align_features(X: pd.DataFrame, training_cols: list[str]) -> pd.DataFrame:
    """Align forecast features with training features, filling missing with 0."""
    for col in training_cols:
        if col not in X.columns:
            X[col] = 0
    return X[training_cols]


def load_model() -> lgb.Booster:
    """Load saved model."""
    model_path = MODEL_DIR / "trophy_lgbm.txt"
    if not model_path.exists():
        raise FileNotFoundError(f"No model found at {model_path}. Train first.")
    return lgb.Booster(model_file=str(model_path))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    merged_path = DATA_DIR / "processed" / "merged.parquet"
    if not merged_path.exists():
        print("Run merge pipeline first: python -m src.pipeline.merge")
    else:
        df = pd.read_parquet(merged_path)
        result = train_model(df)
        print(f"\nMetrics: {result['metrics']}")
