"""Immutable camera/demo snapshots and deterministic what-if queue scenarios."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from uuid import uuid4

from src.queueing.simulator import (
    estimate_wait_minutes, get_congestion_status, recommend_counter_count,
    simulate_queue,
)
from src.vision.live.camera_sources import SourceType
from src.vision.live.monitor import LiveObservation, utc_now


@dataclass(frozen=True)
class SimulationBaseline:
    snapshot_id: str
    timestamp: datetime
    location_id: str
    source_label: str
    queue_count: float
    arrival_rate_per_minute: float | None
    active_counters: int
    service_time_minutes: float
    target_wait_minutes: float
    warning_queue_length: float
    critical_queue_length: float


@dataclass(frozen=True)
class SimulationControls:
    counters: int
    arrival_multiplier: float = 1.0
    extra_arrivals: int = 0
    service_time_minutes: float = 3.0
    duration_minutes: int = 20
    playback_speed: int = 1

    def __post_init__(self):
        if self.counters < 1 or not .5 <= self.arrival_multiplier <= 2:
            raise ValueError("Counters or crowd multiplier outside supported range.")
        if not 0 <= self.extra_arrivals <= 50 or not 1 <= self.service_time_minutes <= 10:
            raise ValueError("Extra arrivals or service time outside supported range.")
        if self.duration_minutes not in (5, 10, 20, 30) or self.playback_speed not in (1, 2, 5, 10):
            raise ValueError("Choose a supported duration and playback speed.")


def baseline_from_live(observation: LiveObservation, location: dict,
                       active_counters: int, *,
                       synthetic_arrival_rate: float | None = None) -> SimulationBaseline:
    if observation.location_id != location["id"]:
        raise ValueError("Camera and facility locations do not match.")
    if active_counters < 1 or (synthetic_arrival_rate is not None and synthetic_arrival_rate < 0):
        raise ValueError("Counters and synthetic rate must be valid.")
    source = ("REPLAY VIDEO" if observation.source_type == SourceType.REPLAY_VIDEO
              else "OBSERVED CAMERA")
    if observation.is_stale():
        source += " · STALE SNAPSHOT"
    rate = observation.observed_arrival_rate
    if synthetic_arrival_rate is not None:
        rate = synthetic_arrival_rate
        source += " + SYNTHETIC DEMAND ASSUMPTION"
    return SimulationBaseline(
        observation.observation_id, observation.captured_at, location["id"], source,
        observation.people_waiting, rate, active_counters,
        float(location["average_service_time_minutes"]),
        float(location["target_wait_minutes"]),
        float(location["warning_queue_length"]),
        float(location["critical_queue_length"]),
    )


def baseline_from_demo(state: dict, location: dict) -> SimulationBaseline:
    if state["selected_location"] != location["id"]:
        raise ValueError("Demo scenario and facility locations do not match.")
    return SimulationBaseline(
        str(uuid4()), utc_now(), location["id"], "SYNTHETIC DEMO",
        float(state["current_queue"]), float(state["current_arrival_rate"]),
        int(state["active_counters"]),
        float(location["average_service_time_minutes"]),
        float(location["target_wait_minutes"]),
        float(location["warning_queue_length"]),
        float(location["critical_queue_length"]),
    )


def run_what_if(baseline: SimulationBaseline, controls: SimulationControls) -> dict:
    if baseline.arrival_rate_per_minute is None:
        raise ValueError("Choose a clearly labelled demand assumption or wait for observed arrivals.")
    if baseline.queue_count < 0 or baseline.arrival_rate_per_minute < 0:
        raise ValueError("Baseline queue and arrival demand cannot be negative.")
    current = simulate_queue(
        baseline.queue_count, baseline.arrival_rate_per_minute,
        baseline.service_time_minutes, baseline.active_counters, controls.duration_minutes,
    )
    proposed_rate = baseline.arrival_rate_per_minute * controls.arrival_multiplier
    proposed = simulate_queue(
        baseline.queue_count + controls.extra_arrivals, proposed_rate,
        controls.service_time_minutes, controls.counters, controls.duration_minutes,
    )

    def outcome(path, counters, service_time):
        queue = path[-1]["queue"]
        return {
            "counters": counters,
            "future_queue": queue,
            "predicted_wait_minutes": round(estimate_wait_minutes(queue, counters, service_time), 2),
            "risk": get_congestion_status(queue, baseline.warning_queue_length,
                                          baseline.critical_queue_length),
        }

    before = outcome(current, baseline.active_counters, baseline.service_time_minutes)
    after = outcome(proposed, controls.counters, controls.service_time_minutes)
    recommendation = recommend_counter_count(
        baseline.queue_count + controls.extra_arrivals, proposed_rate,
        controls.service_time_minutes, controls.counters, baseline.target_wait_minutes,
        controls.duration_minutes, list(range(1, 9)),
    )
    return {
        "baseline_id": baseline.snapshot_id,
        "source_label": baseline.source_label,
        "current_path": current, "proposed_path": proposed,
        "current": before, "proposed": after,
        "proposed_arrival_rate": proposed_rate,
        "recommendation": recommendation,
        "queue_reduction_pct": ((before["future_queue"] - after["future_queue"])
                                / before["future_queue"] * 100 if before["future_queue"] else None),
        "wait_reduction_pct": ((before["predicted_wait_minutes"] - after["predicted_wait_minutes"])
                               / before["predicted_wait_minutes"] * 100
                               if before["predicted_wait_minutes"] else None),
        "additional_counters": controls.counters - baseline.active_counters,
        "controls": controls,
    }
