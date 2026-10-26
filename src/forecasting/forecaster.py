"""Interpretable baseline and scikit-learn arrival forecasters."""

from __future__ import annotations

from pathlib import Path
import json
import os

# Keep prototype training reliable in constrained local/demo environments.
os.environ["OMP_NUM_THREADS"] = "1"

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from threadpoolctl import threadpool_limits

from src.forecasting.features import FEATURE_COLUMNS, build_features, chronological_split


DEFAULT_HORIZONS = [5, 10, 20, 30]


def _metrics(actual: pd.Series, predicted: np.ndarray) -> dict:
    return {
        "mae": round(float(mean_absolute_error(actual, predicted)), 4),
        "rmse": round(float(mean_squared_error(actual, predicted) ** .5), 4),
        "r2": round(float(r2_score(actual, predicted)), 4) if len(actual) > 1 else None,
    }


class ArrivalForecaster:
    """One regression model per forecast horizon, with a persistence baseline."""

    def __init__(self, horizons: list[int] | None = None):
        self.horizons = horizons or DEFAULT_HORIZONS
        self.models: dict[int, HistGradientBoostingRegressor] = {}
        self.metrics: dict[int, dict] = {}

    def train(self, observations: pd.DataFrame) -> dict[int, dict]:
        featured = build_features(observations, self.horizons)
        train, validation, test = chronological_split(featured)
        training = pd.concat([train, validation])
        self.metrics = {}
        for horizon in self.horizons:
            target = f"target_arrivals_{horizon}m"
            training_rows = training.dropna(subset=FEATURE_COLUMNS + [target])
            test_rows = test.dropna(subset=FEATURE_COLUMNS + [target])
            if len(training_rows) < 20 or len(test_rows) < 2:
                raise ValueError("Insufficient chronological history to train and evaluate forecasts.")
            baseline = test_rows["arrivals_rolling_mean_3"].to_numpy()
            model = HistGradientBoostingRegressor(max_iter=120, learning_rate=.08, max_leaf_nodes=15, random_state=42)
            with threadpool_limits(limits=1):
                model.fit(training_rows[FEATURE_COLUMNS], training_rows[target])
            prediction = np.clip(model.predict(test_rows[FEATURE_COLUMNS]), 0, None)
            self.models[horizon] = model
            self.metrics[horizon] = {"baseline": _metrics(test_rows[target], baseline), "ml": _metrics(test_rows[target], prediction), "test_samples": len(test_rows)}
        return self.metrics

    def forecast(self, location_id: str, observation_history: pd.DataFrame, horizons: list[int] | None = None) -> dict[int, dict]:
        requested = horizons or self.horizons
        history = observation_history[observation_history["location_id"] == location_id]
        if len(history) < 2:
            raise ValueError("At least two observations for the selected location are required.")
        row = build_features(history, requested).iloc[-1]
        baseline = max(0.0, float(row["arrivals_rolling_mean_3"]))
        result = {}
        for horizon in requested:
            if horizon in self.models:
                value = max(0.0, float(self.models[horizon].predict(pd.DataFrame([row[FEATURE_COLUMNS]]))[0]))
                method = "HistGradientBoostingRegressor"
            else:
                value, method = baseline, "rolling-arrival baseline"
            result[horizon] = {"predicted_arrivals": round(value, 3), "method": method}
        return result

    def save(self, directory: str | Path) -> None:
        destination = Path(directory)
        destination.mkdir(parents=True, exist_ok=True)
        joblib.dump({"horizons": self.horizons, "models": self.models, "metrics": self.metrics}, destination / "arrival_forecaster.joblib")
        (destination / "arrival_forecaster_metadata.json").write_text(json.dumps({"feature_columns": FEATURE_COLUMNS, "horizons": self.horizons, "metrics": self.metrics}, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory: str | Path) -> "ArrivalForecaster":
        stored = joblib.load(Path(directory) / "arrival_forecaster.joblib")
        instance = cls(stored["horizons"])
        instance.models, instance.metrics = stored["models"], stored["metrics"]
        return instance
