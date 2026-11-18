import numpy as np
from fastapi.testclient import TestClient

from app import HISTORY_LENGTH, ManualPredictionRequest, app, build_manual_feature_row


def make_payload():
    return {
        "timestamp": "2024-09-12T22:35:00",
        "demand_history": [float(value) for value in range(HISTORY_LENGTH)],
        "temp": 12.7,
        "dwpt": 9.3,
        "rhum": 80.0,
        "wdir": 295.0,
        "wspd": 3.7,
        "pres": 1019.7,
    }


def test_health_endpoint_reports_loaded_model():
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["model_loaded"] is True


def test_dashboard_homepage_loads():
    with TestClient(app) as client:
        response = client.get("/")
    assert response.status_code == 200
    assert "Smart Energy Predictor" in response.text


def test_manual_feature_engineering_calculates_history_correctly():
    payload = ManualPredictionRequest(**make_payload())
    features = build_manual_feature_row(payload)

    assert features.loc[0, "Power demand"] == 288.0
    assert features.loc[0, "lag_1"] == 287.0
    assert features.loc[0, "lag_2"] == 286.0
    assert features.loc[0, "lag_3"] == 285.0
    assert features.loc[0, "lag_12"] == 276.0
    assert features.loc[0, "lag_288"] == 0.0
    assert np.isclose(features.loc[0, "moving_avg_3"], 287.0)
    assert np.isclose(features.loc[0, "rolling_mean_3"], 286.0)
    assert np.isclose(features.loc[0, "rolling_mean_12"], 281.5)
    assert np.isclose(features.loc[0, "rolling_mean_288"], 143.5)


def test_invalid_history_length_returns_422():
    payload = make_payload()
    payload["demand_history"] = payload["demand_history"][:-1]
    with TestClient(app) as client:
        response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_valid_manual_prediction_returns_forecast():
    with TestClient(app) as client:
        response = client.post("/predict", json=make_payload())
    assert response.status_code == 200
    body = response.json()
    assert isinstance(body["prediction"], (int, float))
    assert body["forecast_horizon"] == "5 minutes"
    assert body["model"] in {"LinearRegression", "RandomForestRegressor"}
    assert body["predicted_timestamp"] == "2024-09-12T22:40:00"



def test_prediction_history_endpoint_returns_saved_records():
    with TestClient(app) as client:
        response = client.get("/prediction-history")
    assert response.status_code == 200
    body = response.json()
    assert "count" in body
    assert isinstance(body["predictions"], list)


def test_model_metrics_endpoint_includes_comparison():
    with TestClient(app) as client:
        response = client.get("/model-metrics")
    assert response.status_code == 200
    body = response.json()
    assert body["model"] in {"Linear Regression", "Random Forest"}
    assert {"mae", "rmse", "r2"} <= set(body["metrics"])
    assert len(body["comparison"]) >= 1

