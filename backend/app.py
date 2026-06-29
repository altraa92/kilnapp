from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from threading import Lock
from typing import Any

import joblib
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"
METADATA_PATH = BASE_DIR / "model_metadata.json"
DATA_PATH = BASE_DIR / "synthetic_drying_data.csv"

TARGET_MOISTURE_CONTENT = 15.0
FISH_MOISTURE_CONTENT = {
    "catfish": 65.0,
    "tilapia": 80.0,
    "mackerel": 75.0,
}

FEATURE_COLUMNS = [
    "fish_type",
    "initial_weight",
    "current_weight",
    "temperature",
    "humidity",
    "elapsed_time",
    "initial_moisture_content",
    "estimated_moisture_content",
    "weight_loss",
    "percentage_weight_loss",
    "drying_progress",
]

app = Flask(__name__)
CORS(app)

_model = None
_reading_lock = Lock()
_latest_reading: dict[str, Any] | None = None

DEMO_SENSOR_TICK_SECONDS = 5.0
DEMO_SENSOR_STEP_MINUTES = 5.0
DEMO_WEIGHT_LOSS_RATE = 0.055


def ensure_model():
    global _model
    if _model is not None:
        return _model

    if not MODEL_PATH.exists():
        subprocess.run([sys.executable, "train_model.py"], cwd=BASE_DIR, check=True)

    _model = joblib.load(MODEL_PATH)
    return _model


def read_metadata() -> dict[str, Any]:
    if METADATA_PATH.exists():
        return json.loads(METADATA_PATH.read_text(encoding="utf-8"))

    return {
        "modelType": "Random Forest Regression",
        "trainingDataType": "synthetic",
        "features": FEATURE_COLUMNS,
        "metrics": None,
        "message": (
            "This model is trained on synthetic data for demonstration. "
            "It will be replaced with real experimental drying data later."
        ),
    }


def get_float(payload: dict[str, Any], key: str) -> float:
    if key not in payload:
        raise ValueError(f"Missing field: {key}")
    try:
        return float(payload[key])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be a number") from exc


def round_or_none(value: float | None, places: int = 2) -> float | None:
    if value is None:
        return None
    return round(float(value), places)


def parse_prediction_payload(payload: dict[str, Any]) -> dict[str, float | str]:
    fish_type = str(payload.get("fishType", "")).strip().lower()
    if fish_type not in FISH_MOISTURE_CONTENT:
        raise ValueError("fishType must be catfish, tilapia, or mackerel")

    initial_weight = get_float(payload, "initialWeight")
    current_weight = get_float(payload, "currentWeight")
    temperature = get_float(payload, "temperature")
    humidity = get_float(payload, "humidity")
    elapsed_time = get_float(payload, "elapsedTime")

    if initial_weight <= 0 or current_weight <= 0:
        raise ValueError("initialWeight and currentWeight must be greater than zero")
    if humidity < 0 or humidity > 100:
        raise ValueError("humidity must be between 0 and 100")
    if elapsed_time < 0:
        raise ValueError("elapsedTime cannot be negative")

    return {
        "fish_type": fish_type,
        "initial_weight": initial_weight,
        "current_weight": current_weight,
        "temperature": temperature,
        "humidity": humidity,
        "elapsed_time": elapsed_time,
    }


def build_prediction_features(
    fish_type: str,
    initial_weight: float,
    current_weight: float,
    temperature: float,
    humidity: float,
    elapsed_time: float,
) -> dict[str, float | str]:
    initial_mc = FISH_MOISTURE_CONTENT[fish_type]
    dry_matter = initial_weight * (1 - initial_mc / 100)
    estimated_mc = ((current_weight - dry_matter) / current_weight) * 100
    estimated_mc = max(0.0, min(100.0, estimated_mc))

    weight_loss = initial_weight - current_weight
    percentage_weight_loss = (weight_loss / initial_weight) * 100
    drying_progress = ((initial_mc - estimated_mc) / (initial_mc - TARGET_MOISTURE_CONTENT)) * 100
    drying_progress = max(0.0, min(100.0, drying_progress))

    return {
        "fish_type": fish_type,
        "initial_weight": initial_weight,
        "current_weight": current_weight,
        "temperature": temperature,
        "humidity": humidity,
        "elapsed_time": elapsed_time,
        "initial_moisture_content": initial_mc,
        "estimated_moisture_content": estimated_mc,
        "dry_matter": dry_matter,
        "weight_loss": weight_loss,
        "percentage_weight_loss": percentage_weight_loss,
        "drying_progress": drying_progress,
    }


def drying_status(estimated_mc: float, invalid_weight: bool) -> str:
    if invalid_weight:
        return "Invalid weight reading"
    if estimated_mc > 25:
        return "Drying in progress"
    if TARGET_MOISTURE_CONTENT < estimated_mc <= 25:
        return "Near target moisture"
    return "Drying complete"


def target_weight_for(fish_type: str, initial_weight: float) -> float:
    initial_mc = FISH_MOISTURE_CONTENT[fish_type]
    dry_matter = initial_weight * (1 - initial_mc / 100)
    return dry_matter / (1 - TARGET_MOISTURE_CONTENT / 100)


def build_prediction_response(values: dict[str, float | str]) -> dict[str, Any]:
    fish_type = str(values["fish_type"])
    initial_weight = float(values["initial_weight"])
    current_weight = float(values["current_weight"])
    temperature = float(values["temperature"])
    humidity = float(values["humidity"])
    elapsed_time = float(values["elapsed_time"])

    features = build_prediction_features(
        fish_type=fish_type,
        initial_weight=initial_weight,
        current_weight=current_weight,
        temperature=temperature,
        humidity=humidity,
        elapsed_time=elapsed_time,
    )

    invalid_weight = current_weight > initial_weight
    status = drying_status(features["estimated_moisture_content"], invalid_weight)

    alerts: list[str] = []
    if temperature > 70:
        alerts.append("High temperature warning")
    if humidity > 80:
        alerts.append("High humidity warning")
    if invalid_weight:
        alerts.append("Invalid weight reading")
    if features["estimated_moisture_content"] <= TARGET_MOISTURE_CONTENT and not invalid_weight:
        alerts.append("Drying complete")

    estimated_remaining_time: float | None
    if invalid_weight:
        estimated_remaining_time = None
    elif features["estimated_moisture_content"] <= TARGET_MOISTURE_CONTENT:
        estimated_remaining_time = 0.0
    else:
        model = ensure_model()
        model_input = pd.DataFrame([{column: features[column] for column in FEATURE_COLUMNS}])
        estimated_remaining_time = max(0.0, float(model.predict(model_input)[0]))

    return {
        "fishType": fish_type,
        "initialMoistureContent": round_or_none(features["initial_moisture_content"]),
        "targetMoistureContent": TARGET_MOISTURE_CONTENT,
        "dryMatter": round_or_none(features["dry_matter"], 3),
        "estimatedMoistureContent": round_or_none(features["estimated_moisture_content"]),
        "weightLoss": round_or_none(features["weight_loss"], 3),
        "percentageWeightLoss": round_or_none(features["percentage_weight_loss"]),
        "dryingProgress": round_or_none(features["drying_progress"]),
        "estimatedRemainingTime": round_or_none(estimated_remaining_time),
        "dryingStatus": status,
        "alerts": alerts,
    }


def reading_from_values(values: dict[str, float | str], source: str, running: bool = False) -> dict[str, Any]:
    now = time.time()
    return {
        "source": source,
        "running": running,
        "fishType": str(values["fish_type"]),
        "initialWeight": float(values["initial_weight"]),
        "currentWeight": float(values["current_weight"]),
        "temperature": float(values["temperature"]),
        "humidity": float(values["humidity"]),
        "elapsedTime": float(values["elapsed_time"]),
        "lastTickAt": now,
        "updatedAt": now,
        "tickSeconds": DEMO_SENSOR_TICK_SECONDS,
        "stepMinutes": DEMO_SENSOR_STEP_MINUTES,
    }


def values_from_reading(reading: dict[str, Any]) -> dict[str, float | str]:
    return {
        "fish_type": reading["fishType"],
        "initial_weight": reading["initialWeight"],
        "current_weight": reading["currentWeight"],
        "temperature": reading["temperature"],
        "humidity": reading["humidity"],
        "elapsed_time": reading["elapsedTime"],
    }


def serialize_reading(reading: dict[str, Any]) -> dict[str, Any]:
    public_keys = [
        "source",
        "running",
        "fishType",
        "initialWeight",
        "currentWeight",
        "temperature",
        "humidity",
        "elapsedTime",
        "updatedAt",
        "tickSeconds",
        "stepMinutes",
    ]
    return {
        key: round_or_none(reading[key], 3) if isinstance(reading.get(key), float) else reading.get(key)
        for key in public_keys
    }


def advance_demo_reading(reading: dict[str, Any]) -> None:
    if reading.get("source") != "demo" or not reading.get("running"):
        return

    now = time.time()
    ticks = int((now - float(reading["lastTickAt"])) // float(reading["tickSeconds"]))
    if ticks <= 0:
        return

    target_weight = target_weight_for(reading["fishType"], float(reading["initialWeight"]))
    current_weight = float(reading["currentWeight"])

    for _ in range(ticks):
        if current_weight <= target_weight:
            current_weight = target_weight
            reading["running"] = False
            break

        step_loss = max((current_weight - target_weight) * DEMO_WEIGHT_LOSS_RATE, 0.008)
        current_weight = max(target_weight, current_weight - step_loss)
        reading["elapsedTime"] = float(reading["elapsedTime"]) + float(reading["stepMinutes"])

    reading["currentWeight"] = round(current_weight, 3)
    reading["lastTickAt"] = float(reading["lastTickAt"]) + ticks * float(reading["tickSeconds"])
    reading["updatedAt"] = now

    features = build_prediction_features(
        fish_type=reading["fishType"],
        initial_weight=float(reading["initialWeight"]),
        current_weight=float(reading["currentWeight"]),
        temperature=float(reading["temperature"]),
        humidity=float(reading["humidity"]),
        elapsed_time=float(reading["elapsedTime"]),
    )
    if features["estimated_moisture_content"] <= TARGET_MOISTURE_CONTENT:
        reading["currentWeight"] = round(target_weight, 3)
        reading["running"] = False


def reading_payload(reading: dict[str, Any]) -> dict[str, Any]:
    values = values_from_reading(reading)
    return {
        "reading": serialize_reading(reading),
        "prediction": build_prediction_response(values),
    }


@app.get("/api/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "service": "fish-kiln-backend",
            "modelAvailable": MODEL_PATH.exists(),
        }
    )


@app.post("/api/predict")
def predict():
    payload = request.get_json(silent=True) or {}

    try:
        values = parse_prediction_payload(payload)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    return jsonify(build_prediction_response(values))


@app.post("/api/demo/start")
def start_demo_feed():
    global _latest_reading
    payload = request.get_json(silent=True) or {}

    try:
        values = parse_prediction_payload(payload)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    with _reading_lock:
        _latest_reading = reading_from_values(values, source="demo", running=True)
        return jsonify(reading_payload(_latest_reading))


@app.post("/api/demo/stop")
def stop_demo_feed():
    with _reading_lock:
        if _latest_reading is None:
            return jsonify({"reading": None, "prediction": None})
        _latest_reading["running"] = False
        _latest_reading["updatedAt"] = time.time()
        return jsonify(reading_payload(_latest_reading))


@app.get("/api/readings/latest")
def latest_reading():
    with _reading_lock:
        if _latest_reading is None:
            return jsonify({"reading": None, "prediction": None})

        advance_demo_reading(_latest_reading)
        return jsonify(reading_payload(_latest_reading))


@app.post("/api/readings")
def receive_hardware_reading():
    global _latest_reading
    payload = request.get_json(silent=True) or {}

    try:
        values = parse_prediction_payload(payload)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    with _reading_lock:
        _latest_reading = reading_from_values(values, source="hardware", running=False)
        return jsonify(reading_payload(_latest_reading))


@app.get("/api/model-info")
def model_info():
    if not MODEL_PATH.exists():
        ensure_model()

    metadata = read_metadata()
    return jsonify(metadata)


@app.get("/api/sample-data")
def sample_data():
    if not DATA_PATH.exists():
        ensure_model()

    data = pd.read_csv(DATA_PATH)
    return jsonify(data.head(8).to_dict(orient="records"))


if __name__ == "__main__":
    ensure_model()
    app.run(host="0.0.0.0", port=5000, debug=True)
