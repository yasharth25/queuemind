"""Forecast-to-queue integration using existing deterministic queue mathematics."""

from __future__ import annotations

import pandas as pd

from src.queueing.simulator import estimate_wait_minutes, recommend_counter_count, simulate_queue


def forecast_queue_outcomes(forecaster, location: dict, observation_history: pd.DataFrame, horizons: list[int] | None = None) -> dict[int, dict]:
    """Project each forecast horizon and optimise counters from its simulated state."""
    horizons = horizons or [5, 10, 20, 30]
    location_history = observation_history[observation_history["location_id"] == location["id"]].copy()
    if len(location_history) < 2:
        raise ValueError("At least two observations are required for queue forecasting.")
    location_history["timestamp"] = pd.to_datetime(location_history["timestamp"])
    location_history = location_history.sort_values("timestamp")
    interval = max(1.0, (location_history.timestamp.iloc[-1] - location_history.timestamp.iloc[-2]).total_seconds() / 60)
    latest = location_history.iloc[-1]
    forecasts = forecaster.forecast(location["id"], location_history, horizons)
    outcomes = {}
    for horizon, forecast in forecasts.items():
        arrival_rate = forecast["predicted_arrivals"] / interval
        queue = simulate_queue(float(latest.queue_count), arrival_rate, float(latest.average_service_time), int(latest.active_counters), horizon)[-1]["queue"]
        recommendation = recommend_counter_count(float(latest.queue_count), arrival_rate, float(latest.average_service_time), int(latest.active_counters), float(location["target_wait_minutes"]), horizon, list(range(1, 9)))
        outcomes[horizon] = {
            **forecast,
            "arrival_rate_per_minute": round(arrival_rate, 3),
            "expected_queue": queue,
            "predicted_wait_minutes": round(estimate_wait_minutes(queue, int(latest.active_counters), float(latest.average_service_time)), 2),
            "counter_recommendation": recommendation,
        }
    return outcomes
