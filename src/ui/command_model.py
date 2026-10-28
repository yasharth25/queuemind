"""Small, testable presentation calculations for the single-page Command Center."""

import pandas as pd

from src.demo.scenarios import SCENARIOS
from src.queueing.simulator import (
    calculate_service_rate,
    estimate_wait_minutes,
    evaluate_counter_counts,
    get_congestion_status,
    recommend_counter_count,
    simulate_queue,
)

HORIZONS = (5, 10, 20, 30)
TIME_SLOTS = ("08–10", "10–12", "12–14", "14–16", "16–18", "18–20")
WEEKDAYS = ("MON", "TUE", "WED", "THU", "FRI")


def risk_for(queue, wait, config):
    """Include target-wait pressure alongside configured queue thresholds."""
    status = get_congestion_status(queue, config["warning"], config["critical"])
    if status == "normal" and wait > config["target_wait"]:
        return "warning"
    return status


def snapshot(state):
    """Derive every displayed forecast and action from current simulation state."""
    config = SCENARIOS[state["selected_scenario"]]
    counters = state["active_counters"]
    queue = state["current_queue"]
    arrival_rate = config["arrival_rate"] * state["demand_intensity"]
    service_rate = calculate_service_rate(counters, config["service_time"])
    timeline = []
    for horizon in (0, *HORIZONS):
        future_queue = simulate_queue(
            queue, arrival_rate, config["service_time"], counters, horizon
        )[-1]["queue"]
        wait = estimate_wait_minutes(
            future_queue, counters, config["service_time"]
        )
        timeline.append({
            "minute": horizon,
            "queue": future_queue,
            "wait": wait,
            "risk": risk_for(future_queue, wait, config),
        })

    recommendation = recommend_counter_count(
        queue, arrival_rate, config["service_time"], counters,
        config["target_wait"], 20, list(range(1, 9)),
    )
    candidates = sorted(set(range(1, 7)) | {
        counters, recommendation["recommended_counters"]
    })
    options = evaluate_counter_counts(
        queue, arrival_rate, config["service_time"], candidates, 20,
    )
    threshold = next(
        (item["minute"] for item in timeline
         if item["wait"] > config["target_wait"]),
        None,
    )
    return {
        "config": config,
        "queue": queue,
        "wait": timeline[0]["wait"],
        "arrival_rate": arrival_rate,
        "service_rate": service_rate,
        "pressure": arrival_rate - service_rate,
        "timeline": timeline,
        "queue20": timeline[3]["queue"],
        "wait20": timeline[3]["wait"],
        "risk20": timeline[3]["risk"],
        "recommendation": recommendation,
        "options": options,
        "first_target_exceeded": threshold,
    }


def alert_items(state, view):
    """Create a ranked, bounded feed from calculated predictions and events."""
    items = []
    threshold = view["first_target_exceeded"]
    if threshold is not None:
        title = (
            "Wait target already exceeded"
            if threshold == 0 else f"Wait target crossed by +{threshold} min"
        )
        items.append({
            "level": "critical" if view["risk20"] == "critical" else "warning",
            "title": title,
            "detail": (
                f"+20 min: {view['queue20']:.0f} people · "
                f"{view['wait20']:.1f} min wait"
            ),
        })
    if view["risk20"] in {"warning", "critical"}:
        items.append({
            "level": view["risk20"],
            "title": f"{view['risk20'].title()} congestion projected",
            "detail": f"Queue reaches {view['queue20']:.0f} in 20 min.",
        })
    recommended = view["recommendation"]["recommended_counters"]
    if recommended > state["active_counters"]:
        items.append({
            "level": "warning",
            "title": f"Open counter {recommended}",
            "detail": (
                f"Simulated wait falls to "
                f"{view['recommendation']['recommended_outcome']['predicted_wait_minutes']:.1f} min."
            ),
        })
    if view["pressure"] > 0:
        items.append({
            "level": "info",
            "title": "Arrivals exceed service",
            "detail": f"Net queue pressure +{view['pressure']:.2f} people/min.",
        })
    for event in reversed(state["simulation_events"][-5:]):
        items.append({
            "level": "info",
            "title": event["text"],
            "detail": f"Simulated time +{event['minute']} min",
        })
    return items[:6]


def heatmap_data(observations, metric, service_time, counters):
    """Aggregate only synthetic observations into weekday/time-block values."""
    if metric not in {"Queue", "Wait"}:
        raise ValueError("Heatmap metric must be Queue or Wait.")
    if "data_source" not in observations or not observations["data_source"].eq("SYNTHETIC").all():
        raise ValueError("Command Center heatmap requires SYNTHETIC observations.")
    data = observations.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    data = data[data["timestamp"].dt.dayofweek < 5]
    data = data[data["timestamp"].dt.hour.between(8, 19)]
    data["day"] = data["timestamp"].dt.dayofweek.map(
        dict(enumerate(WEEKDAYS))
    )
    data["slot"] = pd.cut(
        data["timestamp"].dt.hour,
        bins=[7, 9, 11, 13, 15, 17, 19],
        labels=TIME_SLOTS,
    )
    if metric == "Wait":
        data["value"] = data["queue_count"].map(
            lambda queue: estimate_wait_minutes(
                queue, counters, service_time
            )
        )
    else:
        data["value"] = data["queue_count"]
    matrix = data.pivot_table(
        index="slot", columns="day", values="value",
        aggfunc="mean", observed=False,
    )
    return matrix.reindex(index=TIME_SLOTS, columns=WEEKDAYS)


def ml_improvement(metrics):
    """Return measured synthetic test improvement, preserving negative results."""
    return {
        horizon: (
            (values["baseline"]["rmse"] - values["ml"]["rmse"])
            / values["baseline"]["rmse"] * 100
            if values["baseline"]["rmse"] else None
        )
        for horizon, values in metrics.items()
    }
