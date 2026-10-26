"""Simulation-first operations dashboard using QueueMind's real queue engine."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.forecasting.forecaster import ArrivalForecaster
from src.forecasting.queue_integration import forecast_queue_outcomes
from src.queueing.simulator import (
    calculate_service_rate, estimate_wait_minutes, evaluate_counter_counts,
    get_congestion_status, recommend_counter_count, simulate_queue,
)
from src.synthetic.generator import generate_synthetic_queue_data


def render_dashboard(locations: list[dict]) -> None:
    st.caption("SIMULATION MODE · All observations and projections shown here are SYNTHETIC.")
    location = st.selectbox("Location", locations, format_func=lambda item: item["display_name"])
    data = generate_synthetic_queue_data([location], "2026-01-01", days=1, seed=42)
    latest = data.iloc[-1]
    selected_counters = st.slider("Active counters", 1, 8, int(location["default_counters"]))
    horizon = 20
    projection = simulate_queue(float(latest.queue_count), float(latest.arrival_rate), float(latest.average_service_time), selected_counters, horizon)
    future_queue = projection[-1]["queue"]
    wait = estimate_wait_minutes(future_queue, selected_counters, float(latest.average_service_time))
    service_rate = calculate_service_rate(selected_counters, float(latest.average_service_time))
    congestion = get_congestion_status(future_queue, location["warning_queue_length"], location["critical_queue_length"])
    recommendation = recommend_counter_count(float(latest.queue_count), float(latest.arrival_rate), float(latest.average_service_time), selected_counters, float(location["target_wait_minutes"]), horizon, [1, 2, 3, 4, 5, 6, 7, 8])

    st.title("QueueMind Operations Dashboard")
    st.caption("System status: simulation backend online · Vision and trained forecasting model not loaded")
    metrics = st.columns(4)
    metrics[0].metric("Current Queue", int(latest.queue_count))
    metrics[1].metric("Estimated Wait", f"{estimate_wait_minutes(float(latest.queue_count), selected_counters, float(latest.average_service_time)):.1f} min")
    metrics[2].metric("Expected Queue in 20 Minutes", f"{future_queue:.1f}")
    metrics[3].metric("Active Counters", selected_counters)
    metrics = st.columns(4)
    metrics[0].metric("Arrival Rate", f"{latest.arrival_rate:.2f}/min")
    metrics[1].metric("Service Rate", f"{service_rate:.2f}/min")
    metrics[2].metric("Congestion Status", congestion.title())
    metrics[3].metric("Recommended Counters", recommendation["recommended_counters"])
    st.info(recommendation["reason"])

    history = data.tail(72).set_index("timestamp")
    st.subheader("Synthetic Queue History")
    st.line_chart(history[["queue_count"]])
    st.subheader("Future Queue Forecast (deterministic simulation)")
    st.line_chart(pd.DataFrame(projection).set_index("minute")[["queue"]])
    st.subheader("Demand Forecast → Queue Projection")
    forecast_location = {**location, "default_counters": selected_counters}
    forecast_data = data.copy()
    forecast_data.loc[:, "active_counters"] = selected_counters
    outcomes = forecast_queue_outcomes(ArrivalForecaster(), forecast_location, forecast_data)
    forecast_table = pd.DataFrame([
        {"horizon_minutes": minutes, "forecast_arrivals": value["predicted_arrivals"], "expected_queue": value["expected_queue"], "predicted_wait_minutes": value["predicted_wait_minutes"], "forecast_method": value["method"]}
        for minutes, value in outcomes.items()
    ])
    st.caption("Demand uses the rolling-arrival baseline until a trained model artifact is explicitly loaded.")
    st.dataframe(forecast_table, hide_index=True, use_container_width=True)
    st.subheader("Counter Simulator")
    scenarios = evaluate_counter_counts(float(latest.queue_count), float(latest.arrival_rate), float(latest.average_service_time), [1, 2, 3, 4, 5, 6, 7, 8], horizon)
    st.dataframe(pd.DataFrame(scenarios), hide_index=True, use_container_width=True)
