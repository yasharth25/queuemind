"""Translate observed local-video queue state into existing queueing decisions."""

from dataclasses import dataclass

from src.queueing.simulator import (
    calculate_service_rate, estimate_wait_minutes, get_congestion_status,
    recommend_counter_count, simulate_queue,
)
from src.vision.analyzer import VisionAnalysisResult, VisionFrameResult

HORIZONS = (5, 10, 20, 30)


@dataclass(frozen=True)
class VisionQueueState:
    location_id: str
    frame_index: int
    timestamp_s: float
    current_queue: int
    observed_arrival_rate: float
    observed_service_rate: float
    active_counters: int
    state_source: str = "VISION"
    forecast_arrival_rate: float | None = None
    demand_source: str = "VIDEO_EVENTS"


def state_from_frame(result: VisionAnalysisResult, frame: VisionFrameResult,
                     location: dict, active_counters: int, *,
                     forecast_arrival_rate: float | None = None,
                     demand_source: str = "VIDEO_EVENTS") -> VisionQueueState:
    if frame not in result.frames:
        raise ValueError("Selected frame is not part of this analysis session.")
    if active_counters < 1:
        raise ValueError("Active counters must be positive.")
    if forecast_arrival_rate is not None and forecast_arrival_rate < 0:
        raise ValueError("Forecast arrival rate cannot be negative.")
    if demand_source not in {"VIDEO_EVENTS", "SYNTHETIC_ASSUMPTION"}:
        raise ValueError("Demand source must identify video events or a synthetic assumption.")
    return VisionQueueState(location["id"], frame.frame_index, frame.timestamp_s,
                            frame.queue_count, frame.arrival_rate, frame.service_rate,
                            active_counters, forecast_arrival_rate=forecast_arrival_rate,
                            demand_source=demand_source)


def project_vision_state(state: VisionQueueState, location: dict) -> dict:
    """Observed event-rate persistence → deterministic simulation → optimization.

    Short video clips are not enough to train or validate an ML demand model.
    This projection uses the measured trailing entry-event rate as a transparent
    constant-demand baseline; configured counter capacity is used for service.
    """
    if state.location_id != location["id"] or state.state_source != "VISION":
        raise ValueError("Vision state and location must match.")
    service_time = float(location["average_service_time_minutes"])
    capacity = calculate_service_rate(state.active_counters, service_time)
    arrival_rate = (state.forecast_arrival_rate if state.forecast_arrival_rate is not None
                    else state.observed_arrival_rate)
    points = []
    for horizon in (0, *HORIZONS):
        queue = simulate_queue(state.current_queue, arrival_rate,
                               service_time, state.active_counters, horizon)[-1]["queue"]
        wait = estimate_wait_minutes(queue, state.active_counters, service_time)
        points.append({
            "minute": horizon, "queue": queue, "wait": round(wait, 2),
            "congestion": get_congestion_status(
                queue, location["warning_queue_length"],
                location["critical_queue_length"],
            ),
            "forecast_arrival_rate": arrival_rate,
        })
    recommendation = recommend_counter_count(
        state.current_queue, arrival_rate, service_time,
        state.active_counters, float(location["target_wait_minutes"]), 20,
        list(range(1, 9)),
    )
    return {
        "state_source": "VISION", "demand_source": state.demand_source,
        "method": ("Observed entry-rate persistence" if state.demand_source == "VIDEO_EVENTS"
                   else "Selected SYNTHETIC demand assumption") + " + deterministic queue simulation",
        "configured_service_capacity": capacity,
        "observed_service_completion_rate": state.observed_service_rate,
        "points": points, "recommendation": recommendation,
    }
