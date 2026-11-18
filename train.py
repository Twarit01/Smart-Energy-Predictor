"""
Training pipeline for Smart Energy Predictor.

Reproduces the feature engineering and chronological split used in
Team06_MidTermProject_Final.ipynb, then tracks experiments with MLflow.
"""

import argparse
import json
from pathlib import Path

import joblib
import mlflow
from mlflow.metrics import rmse
import mlflow.sklearn
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "ML_Dataset.csv"
MODEL_DIR = ROOT / "model"
MODEL_PATH = MODEL_DIR / "model.joblib"
MLFLOW_DB = ROOT / "mlflow.db"

LAGS = [1, 2, 3, 12, 288]
ROLLING_WINDOWS = [3, 12, 288]
TARGET_COLUMN = "target"


def get_time_period(hour):
    if hour < 6:
        return "Night"
    if hour < 12:
        return "Morning"
    if hour < 18:
        return "Afternoon"
    return "Evening"


def load_and_engineer_data(csv_path=DATA_PATH):
    """Recreate the notebook's five-minute timeline and forecasting features."""
    if not Path(csv_path).exists():
        raise FileNotFoundError(
            f"Dataset not found: {csv_path}\n"
            "Place the CSV at data/ML_Dataset.csv."
        )

    df = pd.read_csv(csv_path)

    if "datetime" not in df.columns or "Power demand" not in df.columns:
        raise ValueError("Dataset must contain 'datetime' and 'Power demand'.")

    # The notebook removes the exported dataframe index column.
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"])
    df = df.sort_values("datetime").drop_duplicates("datetime", keep="first")
    df = df.set_index("datetime").asfreq("5min")
    df.index.name = "datetime"

    # Recreate calendar fields on the complete five-minute timeline.
    df["year"] = df.index.year
    df["month"] = df.index.month
    df["day"] = df.index.day
    df["hour"] = df.index.hour
    df["minute"] = df.index.minute
    df["day_of_week"] = df.index.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["time_period"] = df.index.hour.map(get_time_period)

    # Current and historical demand features, matching the notebook.
    df["moving_avg_3"] = df["Power demand"].rolling(
        window=3, min_periods=3
    ).mean()

    for lag in LAGS:
        df[f"lag_{lag}"] = df["Power demand"].shift(lag)

    for window in ROLLING_WINDOWS:
        min_periods = window if window < 288 else int(0.80 * window)
        df[f"rolling_mean_{window}"] = (
            df["Power demand"]
            .shift(1)
            .rolling(window=window, min_periods=min_periods)
            .mean()
        )

    # The label is the next five-minute demand value.
    df[TARGET_COLUMN] = df["Power demand"].shift(-1)

    required_history = [
        "Power demand",
        "moving_avg_3",
        "lag_1",
        "lag_2",
        "lag_3",
        "lag_12",
        "lag_288",
        "rolling_mean_3",
        "rolling_mean_12",
        "rolling_mean_288",
        TARGET_COLUMN,
    ]
    df = df.dropna(subset=required_history).reset_index()

    X = df.drop(columns=[TARGET_COLUMN, "datetime"])
    y = df[TARGET_COLUMN].astype(float)

    return df, X, y


def build_pipeline(model_name, numeric_features, n_estimators, max_depth, min_samples_leaf):
    """Build the notebook's impute/encode/scale/select/model workflow."""

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocess = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, ["time_period"]),
        ],
        remainder="drop",
    )

    if model_name == "linear_regression":
        estimator = LinearRegression()
    elif model_name == "random_forest":
        estimator = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            max_samples=0.5,
            random_state=42,
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("feature_selection", SelectKBest(score_func=f_regression, k=10)),
            ("model", estimator),
        ]
    )


def fit_model(model_name, X_train, y_train, X_test, y_test, args):
    numeric_features = X_train.select_dtypes(include=np.number).columns.tolist()
    pipeline = build_pipeline(
        model_name,
        numeric_features,
        args.n_estimators,
        args.max_depth,
        args.min_samples_leaf,
    )

    with mlflow.start_run(run_name=f"{model_name}_trees{args.n_estimators}_depth{args.max_depth}"):
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

        mae = mean_absolute_error(y_test, predictions)
        rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
        r2 = r2_score(y_test, predictions)

        mlflow.log_param("algorithm", model_name)
        mlflow.log_param("train_samples", len(X_train))
        mlflow.log_param("test_samples", len(X_test))
        mlflow.log_param("selected_features", 10)

        if model_name == "random_forest":
            mlflow.log_param("n_estimators", args.n_estimators)
            mlflow.log_param("max_depth", args.max_depth)
            mlflow.log_param("min_samples_leaf", args.min_samples_leaf)
            mlflow.log_param("max_samples", 0.5)
            mlflow.log_param("random_state", 42)

        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("R2", r2)
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("R2", r2)

        run_id = mlflow.active_run().info.run_id

    print(
        f"{model_name}: MAE={mae:.4f} | RMSE={rmse:.4f} | "
        f"R2={r2:.6f} | run_id={run_id}"
    )

    return {
        "model_name": model_name,
        "pipeline": pipeline,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "run_id": run_id,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and track Smart Energy Predictor models."
        )
    parser.add_argument(
        "--model",
        choices=["all", "linear_regression", "random_forest"],
        default="all",
    )
    parser.add_argument("--n_estimators", type=int, default=50)
    parser.add_argument("--max_depth", type=int, default=12)
    parser.add_argument("--min_samples_leaf", type=int, default=3)
    return parser.parse_args()


def main():
    args = parse_args()

    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB}")
    mlflow.set_experiment("Smart-Energy-Predictor")

    df, X, y = load_and_engineer_data()

    # Chronological 80:20 split with the one-row gap used in the notebook.
    split_index = int(len(X) * 0.80)
    X_train = X.iloc[: split_index - 1].copy()
    X_test = X.iloc[split_index:].copy()
    y_train = y.iloc[: split_index - 1].copy()
    y_test = y.iloc[split_index:].copy()

    print(f"Rows available after feature engineering: {len(df):,}")
    print(f"Training rows: {len(X_train):,}")
    print(f"Testing rows: {len(X_test):,}")
    print("Target: electricity demand exactly 5 minutes ahead")

    models = (
        ["linear_regression", "random_forest"]
        if args.model == "all"
        else [args.model]
    )

    results = []
    for model_name in models:
        results.append(
            fit_model(model_name, X_train, y_train, X_test, y_test, args)
        )

    # Save the lowest-MAE complete preprocessing + selection + model pipeline.
    best = min(results, key=lambda item: item["MAE"])
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best["pipeline"], MODEL_PATH)

    summary = [
        {
            "algorithm": item["model_name"],
            "MAE": round(item["MAE"], 4),
            "RMSE": round(item["RMSE"], 4),
            "R2": round(item["R2"], 6),
            "run_id": item["run_id"],
        }
        for item in results
    ]
    print("\nExperiment summary:")
    print(json.dumps(summary, indent=2))
    print(f"\nBest model: {best['model_name']}")
    print(f"Saved complete pipeline to: {MODEL_PATH}")

if __name__ == "__main__":
    main()