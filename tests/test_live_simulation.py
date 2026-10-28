from src.demo.live import SPEEDS, advance, initial_state, pause, play, reset, set_counters, trigger_surge
def test_play_pause_reset_and_speed_advancement():
 state=initial_state("Bank Rush"); play(state); state["simulation_speed"]="5×"; advance(state); assert state["simulation_elapsed_minutes"]==5
 pause(state); advance(state); assert state["simulation_elapsed_minutes"]==5
 assert reset(state)["simulation_elapsed_minutes"]==0
def test_surge_and_counter_events_are_deduplicated():
 state=initial_state("Bank Rush"); trigger_surge(state); trigger_surge(state); set_counters(state,4); set_counters(state,4)
 assert len([e for e in state["simulation_events"] if e["kind"]=="surge"])==1
 assert len([e for e in state["simulation_events"] if e["kind"]=="counter_change"])==1
def test_max_duration_stops_simulation():
 state=initial_state("Bank Rush"); play(state); state["simulation_elapsed_minutes"]=59; state["simulation_speed"]="2×"; advance(state)
 assert state["complete"] is True and state["simulation_running"] is False
