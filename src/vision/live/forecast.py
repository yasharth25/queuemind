"""Measured-rate projection from fresh camera observations into queue mathematics."""

from __future__ import annotations

from datetime import datetime

from src.queueing.simulator import (
    estimate_wait_minutes, get_congestion_status, recommend_counter_count,
    simulate_queue,
)
from src.vision.live.monitor import LiveObservation

HORIZONS = (5, 10, 20, 30)


def project_live_queue(observation: LiveObservation, location: dict, *,
                       active_counters: int, service_time_minutes: float | None = None,
                       synthetic_arrival_rate: float | None = None,
                       now: datetime | None = None, freshness_s: int = 15) -> dict:
    """Project only a fresh measured rate or an explicit SYNTHETIC assumption.

    Existing ML models were trained/evaluated on synthetic facility data, not
    camera event streams. Live demand therefore uses transparent rate persistence.
    """
    if observation.location_id != location["id"]:
        raise ValueError("Observation and facility location must match.")
    if active_counters < 1:
        raise ValueError("Active counters must be positive.")
    if synthetic_arrival_rate is not None and synthetic_arrival_rate < 0:
        raise ValueError("Synthetic arrival assumption cannot be negative.")
    service_time = float(service_time_minutes or location["average_service_time_minutes"])
    if service_time <= 0:
        raise ValueError("Service time must be positive.")
    if observation.is_stale(now=now, threshold_s=freshness_s):
        return {"available": False, "reason": "Camera data is out of date",
                "last_observed_at": observation.captured_at, "points": []}
    arrival_rate = (synthetic_arrival_rate if synthetic_arrival_rate is not None
                    else observation.observed_arrival_rate)
    if arrival_rate is None:
        return {"available": False,
                "reason": "Not enough recent entry events to estimate arrival demand",
                "last_observed_at": observation.captured_at, "points": []}
    demand_source = ("SYNTHETIC_ASSUMPTION" if synthetic_arrival_rate is not None
                     else "OBSERVED_ENTRY_RATE")
    points = []
    for horizon in (0, *HORIZONS):
        queue = simulate_queue(observation.people_waiting, arrival_rate,
                               service_time, active_counters, horizon)[-1]["queue"]
        points.append({
            "minute": horizon,
            "queue": queue,
            "wait_minutes": round(estimate_wait_minutes(queue, active_counters, service_time), 2),
            "congestion": get_congestion_status(queue, location["warning_queue_length"],
                                                 location["critical_queue_length"]),
        })
    recommendation = recommend_counter_count(
        observation.people_waiting, arrival_rate, service_time, active_counters,
        location["target_wait_minutes"], 20, list(range(1, 9)),
    )
    alerts = []
    if points[3]["queue"] > points[0]["queue"]:
        alerts.append("Queue may grow over the next 20 minutes.")
    if points[3]["wait_minutes"] > location["target_wait_minutes"]:
        alerts.append("Projected wait may exceed the facility target.")
    if recommendation["recommended_counters"] > active_counters:
        alerts.append("Another counter may be needed.")
    return {
        "available": True, "last_observed_at": observation.captured_at,
        "source_type": observation.source_type.value,
        "demand_source": demand_source,
        "method": "Observed entry-rate persistence" if demand_source == "OBSERVED_ENTRY_RATE"
                  else "Explicit SYNTHETIC demand assumption",
        "arrival_rate_per_minute": arrival_rate,
        "configured_service_capacity": active_counters / service_time,
        "measured_service_boundary_rate": observation.service_boundary_rate,
        "points": points, "recommendation": recommendation,
        "alerts": tuple(alerts),
    }
