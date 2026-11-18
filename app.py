"""
FastAPI application for Smart Energy Predictor.

The model is loaded once at application startup. Manual predictions accept
289 consecutive five-minute demand readings (oldest first, latest last),
the timestamp of the latest reading, and weather values at that timestamp.
"""

import json
import math
import sqlite3
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Annotated

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, field_validator


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "model" / "model.joblib"
DATA_PATH = ROOT / "data" / "ML_Dataset.csv"
STATIC_DIR = ROOT / "static"
INDEX_PATH = STATIC_DIR / "index.html"
PREDICTION_LOG = ROOT / "logs" / "predictions.jsonl"

HISTORY_LENGTH = 289
WEATHER_FIELDS = ["temp", "dwpt", "rhum", "wdir", "wspd", "pres"]


def get_time_period(hour: int) -> str:
    if hour < 6:
        return "Night"
    if hour < 12:
        return "Morning"
    if hour < 18:
        return "Afternoon"
    return "Evening"


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Model not found at {MODEL_PATH}. Run train.py successfully first."
        )
    # Load the complete preprocessing + feature-selection + estimator pipeline once.
    app.state.model = joblib.load(MODEL_PATH)
    yield


app = FastAPI(
    title="Smart Energy Predictor API",
    description="Forecast electricity demand five minutes ahead.",
    version="1.0.0",
    lifespan=lifespan,
)


class ManualPredictionRequest(BaseModel):
    timestamp: datetime = Field(
        ...,
        description="Timestamp of the latest demand reading; must be on a 5-minute boundary.",
    )
    demand_history: Annotated[
        list[float],
        Field(
            min_length=HISTORY_LENGTH,
            max_length=HISTORY_LENGTH,
            description="Exactly 289 demand readings, oldest first and latest last.",
        ),
    ]
    temp: float
    dwpt: float
    rhum: Annotated[float, Field(ge=0, le=100)]
    wdir: Annotated[float, Field(ge=0, le=360)]
    wspd: Annotated[float, Field(ge=0)]
    pres: Annotated[float, Field(gt=0)]

    @field_validator("demand_history")
    @classmethod
    def validate_history_is_finite(cls, values):
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Demand history must contain only finite numbers.")
        return values

    @field_validator("timestamp")
    @classmethod
    def validate_five_minute_timestamp(cls, value):
        if value.second != 0 or value.microsecond != 0 or value.minute % 5 != 0:
            raise ValueError("Timestamp must fall exactly on a 5-minute boundary.")
        return value

    @field_validator("temp", "dwpt", "wspd", "pres")
    @classmethod
    def validate_finite_weather_values(cls, value):
        if not math.isfinite(value):
            raise ValueError("Weather values must be finite numbers.")
        return value


def build_manual_feature_row(payload: ManualPredictionRequest) -> pd.DataFrame:
    """Build the same model input columns created by train.py."""
    history = np.asarray(payload.demand_history, dtype=float)
    current = history[-1]
    previous = history[:-1]
    ts = payload.timestamp

    row = {
        "Power demand": current,
        "temp": payload.temp,
        "dwpt": payload.dwpt,
        "rhum": payload.rhum,
        "wdir": payload.wdir,
        "wspd": payload.wspd,
        "pres": payload.pres,
        "year": ts.year,
        "month": ts.month,
        "day": ts.day,
        "hour": ts.hour,
        "minute": ts.minute,
        "day_of_week": ts.weekday(),
        "is_weekend": int(ts.weekday() >= 5),
        "time_period": get_time_period(ts.hour),
        "moving_avg_3": float(np.mean(history[-3:])),
        "lag_1": float(previous[-1]),
        "lag_2": float(previous[-2]),
        "lag_3": float(previous[-3]),
        "lag_12": float(previous[-12]),
        "lag_288": float(previous[-288]),
        "rolling_mean_3": float(np.mean(previous[-3:])),
        "rolling_mean_12": float(np.mean(previous[-12:])),
        "rolling_mean_288": float(np.mean(previous[-288:])),
    }
    return pd.DataFrame([row])


def write_prediction_log(mode: str, timestamp: str, features: dict, prediction: float, model_name: str = "LinearRegression", predicted_timestamp: str | None = None):
    PREDICTION_LOG.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "logged_at": datetime.now().astimezone().isoformat(),
        "mode": mode,
        "input_timestamp": timestamp,
        "features": features,
        "predicted_demand_5min_ahead": prediction,
        "model": model_name,
        "predicted_timestamp": predicted_timestamp or (pd.Timestamp(timestamp) + pd.Timedelta(minutes=5)).isoformat(),
    }
    with PREDICTION_LOG.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, allow_nan=False) + "\n")


@app.get("/", include_in_schema=False)
def home():
    if INDEX_PATH.exists():
        return FileResponse(INDEX_PATH)
    return {
        "message": "Smart Energy Predictor API is running.",
        "docs": "/docs",
    }



@app.get("/health")
def health():
    model = getattr(app.state, "model", None)
    model_loaded = model is not None

    model_name = None
    if model_loaded:
        model_name = type(model.named_steps["model"]).__name__

    return {
        "status": "healthy" if model_loaded else "starting",
        "model_loaded": model_loaded,
        "model_name": model_name,
    }

@app.get("/dashboard-data")
def dashboard_data():
    """Return real historical demand data for the dashboard."""
    if not DATA_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="Dataset not found at data/ML_Dataset.csv."
        )

    try:
        df = pd.read_csv(DATA_PATH)

        required_columns = ["datetime", "Power demand"]
        missing = [
            col for col in required_columns if col not in df.columns
        ]

        if missing:
            raise HTTPException(
                status_code=500,
                detail=f"Dataset is missing columns: {missing}"
            )

        df = df[required_columns].copy()
        df["datetime"] = pd.to_datetime(
            df["datetime"], errors="coerce"
        )
        df["Power demand"] = pd.to_numeric(
            df["Power demand"], errors="coerce"
        )

        df = (
            df.dropna()
            .sort_values("datetime")
            .drop_duplicates(subset=["datetime"], keep="last")
        )

        if df.empty:
            raise ValueError("No valid demand records found.")

        # Last 288 readings represent approximately 24 hours.
        recent = df.tail(288)

        peak_index = recent["Power demand"].idxmax()
        latest = recent.iloc[-1]

        return {
            "total_records": int(len(df)),
            "window_records": int(len(recent)),
            "unit": "Same unit as Power demand in the dataset",
            "latest": {
                "timestamp": latest["datetime"].isoformat(),
                "demand": float(latest["Power demand"])
            },
            "statistics": {
                "average": float(recent["Power demand"].mean()),
                "minimum": float(recent["Power demand"].min()),
                "maximum": float(recent["Power demand"].max()),
                "peak_timestamp": (
                    recent.loc[peak_index, "datetime"].isoformat()
                )
            },
            "history": [
                {
                    "timestamp": row["datetime"].isoformat(),
                    "demand": float(row["Power demand"])
                }
                for _, row in recent.iterrows()
            ]
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Dashboard data error: {type(exc).__name__}: {exc}"
        ) from exc

@app.post("/predict")
def predict(payload: ManualPredictionRequest):
    model = getattr(app.state, "model", None)
    if model is None:
        raise HTTPException(status_code=503, detail="Prediction model is not loaded.")

    feature_row = build_manual_feature_row(payload)
    try:
        prediction = float(model.predict(feature_row)[0])
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {type(exc).__name__}: {exc}",
        ) from exc

    features = feature_row.iloc[0].to_dict()
    write_prediction_log(
        mode="manual",
        timestamp=payload.timestamp.isoformat(),
        features=features,
        prediction=prediction,
        model_name=type(model.named_steps["model"]).__name__,
        predicted_timestamp=(payload.timestamp + pd.Timedelta(minutes=5)).isoformat(),
    )

    return {
        "prediction": prediction,
        "unit": "same unit as the dataset's Power demand column",
        "forecast_horizon": "5 minutes",
        "input_timestamp": payload.timestamp.isoformat(),
        "predicted_timestamp": (
            payload.timestamp + pd.Timedelta(minutes=5)
        ).isoformat(),
        "model": type(model.named_steps["model"]).__name__,
    }


@app.get("/model-metrics")
def model_metrics():
    """Return the best successful run for each estimator from the local MLflow database."""
    if not (ROOT / "mlflow.db").exists():
        raise HTTPException(status_code=404, detail="MLflow database not found.")
    try:
        connection = sqlite3.connect(f"file:{ROOT / 'mlflow.db'}?mode=ro", uri=True)
        connection.row_factory = sqlite3.Row
        try:
            experiment = connection.execute(
                "SELECT experiment_id FROM experiments WHERE name = ?",
                ("Smart-Energy-Predictor",),
            ).fetchone()
            if experiment is None:
                raise HTTPException(status_code=404, detail="MLflow experiment not found.")

            runs = connection.execute(
                """SELECT run_uuid, name, start_time FROM runs
                   WHERE experiment_id = ? AND status = 'FINISHED'
                   AND lifecycle_stage = 'active'
                   ORDER BY start_time DESC""",
                (experiment["experiment_id"],),
            ).fetchall()

            grouped = {}
            for run in runs:
                run_name = run["name"] or "Unknown run"
                lower_name = run_name.lower()
                model_name = (
                    "Random Forest" if "random_forest" in lower_name else
                    "Linear Regression" if "linear_regression" in lower_name else None
                )
                if not model_name:
                    continue
                metric_rows = connection.execute(
                    """SELECT key, value FROM metrics
                       WHERE run_uuid = ?
                       AND key IN ('MAE', 'RMSE', 'R2', 'mae', 'rmse', 'r2')
                       ORDER BY timestamp DESC""",
                    (run["run_uuid"],),
                ).fetchall()
                metrics = {}
                for metric in metric_rows:
                    metrics.setdefault(metric["key"].lower(), float(metric["value"]))
                if not all(key in metrics for key in ("mae", "rmse", "r2")):
                    continue
                candidate = {
                    "model": model_name,
                    "run_name": run_name,
                    "run_id": run["run_uuid"],
                    "start_time": run["start_time"],
                    "metrics": {
                        "mae": metrics["mae"],
                        "rmse": metrics["rmse"],
                        "r2": metrics["r2"],
                    },
                }
                if model_name not in grouped or metrics["mae"] < grouped[model_name]["metrics"]["mae"]:
                    grouped[model_name] = candidate

            if not grouped:
                raise HTTPException(status_code=404, detail="No successful model runs with MAE, RMSE and R² were found.")
            selected = min(grouped.values(), key=lambda item: item["metrics"]["mae"])
            return {
                **selected,
                "comparison": sorted(grouped.values(), key=lambda item: item["metrics"]["mae"]),
                "tracking_source": "Local MLflow · mlflow.db",
            }
        finally:
            connection.close()
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"MLflow metrics error: {type(exc).__name__}: {exc}",
        ) from exc





def read_prediction_history():
    """Read persisted predictions, newest first."""
    if not PREDICTION_LOG.exists():
        return []
    records = []
    with PREDICTION_LOG.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                input_time = item["input_timestamp"]
                records.append({
                    "logged_at": item.get("logged_at"),
                    "mode": item.get("mode", "unknown"),
                    "input_timestamp": input_time,
                    "forecast_timestamp": item.get("predicted_timestamp") or (
                        pd.Timestamp(input_time) + pd.Timedelta(minutes=5)
                    ).isoformat(),
                    "prediction": float(item["predicted_demand_5min_ahead"]),
                    "model": item.get("model", "LinearRegression"),
                })
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
    records.sort(key=lambda item: item.get("logged_at") or "", reverse=True)
    return records


@app.get("/prediction-history")
def prediction_history():
    records = read_prediction_history()
    return {"count": len(records), "predictions": records}


@app.post("/demo-predict")
def demo_predict():
    """Predict from the latest valid feature row in the project's dataset."""
    model = getattr(app.state, "model", None)
    if model is None:
        raise HTTPException(status_code=503, detail="Prediction model is not loaded.")
    if not DATA_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="Dataset not found. Place it at data/ML_Dataset.csv.",
        )

    try:
        # Import the same feature-engineering function used by the training script.
        from train import load_and_engineer_data

        engineered_df, X, _ = load_and_engineer_data(DATA_PATH)
        if X.empty:
            raise ValueError("No valid rows remain after feature engineering.")

        feature_row = X.iloc[[-1]].copy()
        prediction = float(model.predict(feature_row)[0])
        input_timestamp = pd.Timestamp(engineered_df.iloc[-1]["datetime"])
        features = feature_row.iloc[0].to_dict()

        write_prediction_log(
            mode="demo",
            timestamp=input_timestamp.isoformat(),
            features=features,
            prediction=prediction,
            model_name=type(model.named_steps["model"]).__name__,
            predicted_timestamp=(input_timestamp + pd.Timedelta(minutes=5)).isoformat(),
        )

        return {
            "prediction": prediction,
            "unit": "same unit as the dataset's Power demand column",
            "forecast_horizon": "5 minutes",
            "input_timestamp": input_timestamp.isoformat(),
            "predicted_timestamp": (
                input_timestamp + pd.Timedelta(minutes=5)
            ).isoformat(),
            "model": type(model.named_steps["model"]).__name__,
            "mode": "demo using latest valid dataset row",
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Demo prediction failed: {type(exc).__name__}: {exc}",
        ) from exc
