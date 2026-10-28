"""Deterministic lightweight state engine for Scenario Lab live simulation."""
from copy import deepcopy
from src.demo.scenarios import SCENARIOS
from src.queueing.simulator import calculate_service_rate, estimate_wait_minutes, get_congestion_status, recommend_counter_count, simulate_queue

MAX_SIMULATED_MINUTES=60
SPEEDS={"1×":1,"2×":2,"5×":5,"10×":10}

def initial_state(scenario):
    cfg=SCENARIOS[scenario]
    return {"simulation_running":False,"simulation_elapsed_seconds":0,"simulation_elapsed_minutes":0,"selected_scenario":scenario,"selected_location":cfg["location_id"],"active_counters":cfg["counters"],"demand_intensity":1.0,"simulation_speed":"1×","current_queue":float(cfg["queue"]),"current_wait":estimate_wait_minutes(cfg["queue"],cfg["counters"],cfg["service_time"]),"current_arrival_rate":cfg["arrival_rate"],"current_service_rate":calculate_service_rate(cfg["counters"],cfg["service_time"]),"simulation_events":[],"surge_triggered":False,"risk":"normal","complete":False}

def _event(state, kind, text):
    if not any(item["kind"]==kind for item in state["simulation_events"]): state["simulation_events"].append({"minute":state["simulation_elapsed_minutes"],"kind":kind,"text":text})

def reset(state): return initial_state(state["selected_scenario"])
def play(state): state["simulation_running"]=True; state["complete"]=False; _event(state,"started","Simulation started"); return state
def pause(state): state["simulation_running"]=False; return state
def set_counters(state,counters):
    if counters != state["active_counters"]: _event(state,"counter_change",f"Counter capacity changed: {state['active_counters']} → {counters}"); state["active_counters"]=counters
    return state
def trigger_surge(state):
    if not state["surge_triggered"]: state["surge_triggered"]=True; state["demand_intensity"]=max(1.8,state["demand_intensity"]); _event(state,"surge","Demand surge triggered")
    return state
def advance(state):
    if not state["simulation_running"] or state["complete"]: return state
    cfg=SCENARIOS[state["selected_scenario"]]; minutes=SPEEDS[state["simulation_speed"]]; rate=cfg["arrival_rate"]*state["demand_intensity"]
    state["current_queue"]=simulate_queue(state["current_queue"],rate,cfg["service_time"],state["active_counters"],minutes)[-1]["queue"]
    state["simulation_elapsed_seconds"]+=1; state["simulation_elapsed_minutes"]+=minutes; state["current_arrival_rate"]=rate; state["current_service_rate"]=calculate_service_rate(state["active_counters"],cfg["service_time"]); state["current_wait"]=estimate_wait_minutes(state["current_queue"],state["active_counters"],cfg["service_time"])
    projection=simulate_queue(state["current_queue"],rate,cfg["service_time"],state["active_counters"],20)[-1]["queue"]; risk=get_congestion_status(projection,cfg["warning"],cfg["critical"]); recommendation=recommend_counter_count(state["current_queue"],rate,cfg["service_time"],state["active_counters"],cfg["target_wait"],20,list(range(1,9)))
    if risk!=state["risk"]: _event(state,f"risk_{risk}",f"{risk.title()} congestion predicted"); state["risk"]=risk
    if recommendation["recommended_counters"]>state["active_counters"]: _event(state,"recommendation",f"QueueMind recommends counter {recommendation['recommended_counters']}")
    if state["simulation_elapsed_minutes"]>=MAX_SIMULATED_MINUTES: state["simulation_running"]=False; state["complete"]=True; _event(state,"complete","Simulation complete")
    return state
