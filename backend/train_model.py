from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"
DATA_PATH = BASE_DIR / "synthetic_drying_data.csv"
METADATA_PATH = BASE_DIR / "model_metadata.json"

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


def _build_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _current_weight_from_moisture(initial_weight: float, initial_mc: float, estimated_mc: float) -> float:
    dry_matter = initial_weight * (1 - initial_mc / 100)
    return dry_matter / (1 - estimated_mc / 100)


def generate_synthetic_data(rows_per_species: int = 700, random_state: int = 42) -> pd.DataFrame:
    """Create demonstration data until real ESP32-backed drying trials are collected."""

    rng = np.random.default_rng(random_state)
    rows: list[dict[str, float | str]] = []

    species_base_minutes = {
        "catfish": 255,
        "tilapia": 310,
        "mackerel": 290,
    }
    species_factor = {
        "catfish": 0.96,
        "tilapia": 1.10,
        "mackerel": 1.04,
    }

    for fish_type, initial_mc in FISH_MOISTURE_CONTENT.items():
        for _ in range(rows_per_species):
            initial_weight = rng.uniform(0.5, 5.0)
            temperature = rng.uniform(45.0, 70.0)
            humidity = rng.uniform(25.0, 85.0)
            elapsed_time = rng.uniform(0.0, 360.0)

            expected_total_time = (
                species_base_minutes[fish_type]
                + initial_weight * 34
                + (humidity - 45) * 1.5
                + (60 - temperature) * 4.2
                + rng.normal(0, 18)
            )
            expected_total_time = float(np.clip(expected_total_time, 135, 560))

            drying_fraction = elapsed_time / expected_total_time
            drying_fraction *= rng.normal(1.0, 0.08)
            drying_fraction = float(np.clip(drying_fraction, 0, 1.08))

            end_moisture = rng.uniform(12.5, 16.5)
            estimated_mc = initial_mc - drying_fraction * (initial_mc - end_moisture)
            estimated_mc += rng.normal(0, 1.2)
            estimated_mc = float(np.clip(estimated_mc, 10.0, initial_mc))

            current_weight = _current_weight_from_moisture(
                initial_weight=initial_weight,
                initial_mc=initial_mc,
                estimated_mc=estimated_mc,
            )
            current_weight = min(current_weight, initial_weight)

            dry_matter = initial_weight * (1 - initial_mc / 100)
            estimated_mc = ((current_weight - dry_matter) / current_weight) * 100
            estimated_mc = float(np.clip(estimated_mc, 0.0, 100.0))

            weight_loss = initial_weight - current_weight
            percentage_weight_loss = (weight_loss / initial_weight) * 100
            drying_progress = ((initial_mc - estimated_mc) / (initial_mc - TARGET_MOISTURE_CONTENT)) * 100
            drying_progress = float(np.clip(drying_progress, 0.0, 100.0))

            moisture_gap = max(0.0, estimated_mc - TARGET_MOISTURE_CONTENT)
            temperature_factor = 1 + (60 - temperature) * 0.025
            humidity_factor = 1 + (humidity - 45) * 0.012
            weight_factor = 0.78 + initial_weight * 0.13

            remaining_time = (
                moisture_gap
                * 5.9
                * temperature_factor
                * humidity_factor
                * weight_factor
                * species_factor[fish_type]
                - elapsed_time * 0.018
                + rng.normal(0, 8)
            )
            if estimated_mc <= TARGET_MOISTURE_CONTENT:
                remaining_time = rng.uniform(0, 8)
            remaining_time = float(np.clip(remaining_time, 0.0, 420.0))

            rows.append(
                {
                    "fish_type": fish_type,
                    "initial_weight": round(initial_weight, 3),
                    "current_weight": round(current_weight, 3),
                    "temperature": round(temperature, 2),
                    "humidity": round(humidity, 2),
                    "elapsed_time": round(elapsed_time, 2),
                    "initial_moisture_content": round(initial_mc, 2),
                    "estimated_moisture_content": round(estimated_mc, 2),
                    "weight_loss": round(weight_loss, 3),
                    "percentage_weight_loss": round(percentage_weight_loss, 2),
                    "drying_progress": round(drying_progress, 2),
                    "remaining_drying_time": round(remaining_time, 2),
                }
            )

    data = pd.DataFrame(rows)
    return data.sample(frac=1, random_state=random_state).reset_index(drop=True)


def train_model() -> dict[str, float]:
    data = generate_synthetic_data()
    DATA_PATH.write_text(data.to_csv(index=False), encoding="utf-8")

    x = data[FEATURE_COLUMNS]
    y = data["remaining_drying_time"]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
    )

    categorical_features = ["fish_type"]
    numeric_features = [column for column in FEATURE_COLUMNS if column not in categorical_features]

    preprocessor = ColumnTransformer(
        transformers=[
            ("fish_type", _build_encoder(), categorical_features),
            ("numeric", StandardScaler(), numeric_features),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=240,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    metrics = {
        "mae": round(float(mean_absolute_error(y_test, predictions)), 3),
        "rmse": round(float(np.sqrt(mean_squared_error(y_test, predictions))), 3),
        "r2": round(float(r2_score(y_test, predictions)), 3),
    }

    joblib.dump(model, MODEL_PATH)

    metadata = {
        "modelType": "Random Forest Regression",
        "trainingDataType": "synthetic",
        "features": FEATURE_COLUMNS,
        "target": "remaining_drying_time",
        "metrics": metrics,
        "message": (
            "This model is trained on synthetic data for demonstration. "
            "It will be replaced with real experimental drying data later."
        ),
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return metrics


if __name__ == "__main__":
    evaluation = train_model()
    print("Synthetic drying data saved to:", DATA_PATH)
    print("Model saved to:", MODEL_PATH)
    print("Evaluation metrics")
    print(f"MAE: {evaluation['mae']}")
    print(f"RMSE: {evaluation['rmse']}")
    print(f"R2: {evaluation['r2']}")