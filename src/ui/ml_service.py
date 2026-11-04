"""Explicit, cached synthetic evaluation lifecycle for the ML Lab."""

import streamlit as st

from src.forecasting.forecaster import ArrivalForecaster
from src.synthetic.generator import generate_synthetic_queue_data


@st.cache_data(show_spinner=False)
def synthetic_evaluation_data(locations):
    return generate_synthetic_queue_data(
        list(locations), "2026-01-05", days=8, seed=9,
    )


@st.cache_resource(show_spinner=False)
def evaluation_bundle(locations):
    observations = synthetic_evaluation_data(locations)
    forecaster = ArrivalForecaster()
    metrics = forecaster.train(observations)
    return {"observations": observations, "forecaster": forecaster, "metrics": metrics}
