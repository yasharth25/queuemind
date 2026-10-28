"""Deterministic, session-scoped state for the synthetic live simulation."""

from src.demo.scenarios import SCENARIOS
from src.queueing.simulator import (
    calculate_service_rate,
    estimate_wait_minutes,
    get_congestion_status,
    recommend_counter_count,
    simulate_queue,
)

MAX_SIMULATED_MINUTES = 60
SPEEDS = {"1×": 1, "2×": 2, "5×": 5, "10×": 10}


def initial_state(scenario):
    config = SCENARIOS[scenario]
    projected = simulate_queue(
        config["queue"], config["arrival_rate"], config["service_time"],
        config["counters"], 20,
    )[-1]["queue"]
    return {
        "simulation_running": False,
        "simulation_elapsed_seconds": 0,
        "simulation_elapsed_minutes": 0,
        "selected_scenario": scenario,
        "selected_location": config["location_id"],
        "active_counters": config["counters"],
        "demand_intensity": 1.0,
        "simulation_speed": "1×",
        "current_queue": float(config["queue"]),
        "current_wait": estimate_wait_minutes(
            config["queue"], config["counters"], config["service_time"]
        ),
        "current_arrival_rate": config["arrival_rate"],
        "current_service_rate": calculate_service_rate(
            config["counters"], config["service_time"]
        ),
        "simulation_events": [],
        "queue_history": [{"minute": 0, "queue": float(config["queue"])}],
        "surge_triggered": False,
        "risk": get_congestion_status(
            projected, config["warning"], config["critical"]
        ),
        "last_recommendation": None,
        "complete": False,
    }


def _event(state, kind, text):
    events = state["simulation_events"]
    if not events or (events[-1]["kind"], events[-1]["text"]) != (kind, text):
        events.append({
            "minute": state["simulation_elapsed_minutes"],
            "kind": kind,
            "text": text,
        })
        del events[:-100]


def reset(state):
    return initial_state(state["selected_scenario"])


def play(state):
    if not state["complete"]:
        state["simulation_running"] = True
        if not any(item["kind"] == "started" for item in state["simulation_events"]):
            _event(state, "started", "Simulation started")
    return state


def pause(state):
    state["simulation_running"] = False
    return state


def set_counters(state, counters):
    if counters != state["active_counters"]:
        _event(
            state, "counter_change",
            f"Counter capacity changed: {state['active_counters']} → {counters}",
        )
        state["active_counters"] = counters
        config = SCENARIOS[state["selected_scenario"]]
        state["current_service_rate"] = calculate_service_rate(
            counters, config["service_time"]
        )
        state["current_wait"] = estimate_wait_minutes(
            state["current_queue"], counters, config["service_time"]
        )
    return state


def trigger_surge(state):
    if not state["surge_triggered"]:
        state["surge_triggered"] = True
        state["demand_intensity"] = max(1.8, state["demand_intensity"])
        config = SCENARIOS[state["selected_scenario"]]
        state["current_arrival_rate"] = (
            config["arrival_rate"] * state["demand_intensity"]
        )
        _event(state, "surge", "Demand surge triggered")
    return state


def advance(state):
    if not state["simulation_running"] or state["complete"]:
        return state

    config = SCENARIOS[state["selected_scenario"]]
    minutes = min(
        SPEEDS[state["simulation_speed"]],
        MAX_SIMULATED_MINUTES - state["simulation_elapsed_minutes"],
    )
    rate = config["arrival_rate"] * state["demand_intensity"]
    state["current_queue"] = simulate_queue(
        state["current_queue"], rate, config["service_time"],
        state["active_counters"], minutes,
    )[-1]["queue"]
    state["simulation_elapsed_seconds"] += 1
    state["simulation_elapsed_minutes"] += minutes
    state["current_arrival_rate"] = rate
    state["current_service_rate"] = calculate_service_rate(
        state["active_counters"], config["service_time"]
    )
    state["current_wait"] = estimate_wait_minutes(
        state["current_queue"], state["active_counters"], config["service_time"]
    )
    state["queue_history"].append({
        "minute": state["simulation_elapsed_minutes"],
        "queue": state["current_queue"],
    })
    projection = simulate_queue(
        state["current_queue"], rate, config["service_time"],
        state["active_counters"], 20,
    )[-1]["queue"]
    risk = get_congestion_status(
        projection, config["warning"], config["critical"]
    )
    recommendation = recommend_counter_count(
        state["current_queue"], rate, config["service_time"],
        state["active_counters"], config["target_wait"], 20, list(range(1, 9)),
    )["recommended_counters"]
    if risk != state["risk"]:
        _event(state, f"risk_{risk}", f"{risk.title()} congestion predicted")
        state["risk"] = risk
    if recommendation > state["active_counters"] and (
        recommendation != state["last_recommendation"]
    ):
        _event(
            state, "recommendation",
            f"QueueMind recommends opening counter {recommendation}",
        )
    state["last_recommendation"] = recommendation
    if state["simulation_elapsed_minutes"] >= MAX_SIMULATED_MINUTES:
        state["simulation_running"] = False
        state["complete"] = True
        _event(state, "complete", "Simulation complete")
    return state
