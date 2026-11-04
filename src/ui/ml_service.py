"""Explicit, cached synthetic evaluation lifecycle for the ML Lab."""

import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error

from src.forecasting.features import FEATURE_COLUMNS
from src.forecasting.forecaster import ArrivalForecaster
from src.synthetic.generator import generate_synthetic_queue_data

SYNTHETIC_START = "2026-01-05"
SYNTHETIC_DAYS = 8
SYNTHETIC_SEED = 9
SYNTHETIC_FREQUENCY_MINUTES = 5


@st.cache_data(show_spinner=False)
def synthetic_evaluation_data(locations):
    return generate_synthetic_queue_data(
        list(locations), SYNTHETIC_START, days=SYNTHETIC_DAYS,
        frequency_minutes=SYNTHETIC_FREQUENCY_MINUTES, seed=SYNTHETIC_SEED,
    )


@st.cache_resource(show_spinner=False)
def evaluation_bundle(locations):
    observations = synthetic_evaluation_data(locations)
    forecaster = ArrivalForecaster()
    metrics = forecaster.train(observations)
    return {"observations": observations, "forecaster": forecaster, "metrics": metrics,
            "evaluated_at": datetime.now(timezone.utc)}


@st.cache_data(show_spinner=False)
def residuals_for_horizon(locations, horizon):
    from src.evaluation.forecast_diagnostics import residual_frame

    forecaster = evaluation_bundle(locations)["forecaster"]
    return residual_frame(forecaster.test_predictions[horizon])


@st.cache_data(show_spinner=False)
def permutation_importance_for_horizon(locations, horizon):
    """Compute importance on a deterministic subset of held-out test rows."""
    forecaster = evaluation_bundle(locations)["forecaster"]
    frame = forecaster.test_predictions[horizon]
    if len(frame) < 20:
        raise ValueError("Insufficient held-out rows for permutation importance.")
    indices = np.linspace(0, len(frame) - 1, min(len(frame), 500), dtype=int)
    held_out = frame.iloc[indices]
    scoring = lambda estimator, features, actual: -mean_absolute_error(
        actual, np.clip(estimator.predict(features), 0, None)
    )
    result = permutation_importance(
        forecaster.models[horizon], held_out[FEATURE_COLUMNS], held_out.actual,
        n_repeats=3, random_state=42, n_jobs=1, scoring=scoring,
    )
    return pd.DataFrame({"feature": FEATURE_COLUMNS, "importance": result.importances_mean,
                         "std": result.importances_std}).sort_values("importance", ascending=False)
