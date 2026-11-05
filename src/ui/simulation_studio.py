"""Camera-based or SYNTHETIC demo what-if studio; observations remain untouched."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.vision.live.simulation import (
    SimulationControls, baseline_from_demo, baseline_from_live, run_what_if,
)


def _baseline(locations):
    current = st.session_state.get("qm_studio_baseline")
    if current is not None:
        return current
    observation = st.session_state.get("qm_camera_observation")
    if observation is not None:
        location = next((item for item in locations if item["id"] == observation.location_id), None)
        if location is not None:
            current = baseline_from_live(observation, location,
                                         st.session_state.get("qm_camera_counters", location["default_counters"]))
    if current is None:
        state = st.session_state.qm_live
        location = next(item for item in locations if item["id"] == state["selected_location"])
        current = baseline_from_demo(state, location)
    st.session_state.qm_studio_baseline = current
    return current


def _refresh_baseline(locations):
    st.session_state.pop("qm_studio_baseline", None)
    st.session_state.pop("qm_studio_result", None)
    st.session_state.qm_studio_playing = False
    return _baseline(locations)


def _reset():
    for key in ("qm_studio_result", "qm_studio_scenarios", "qm_studio_cursor",
                "qm_studio_playing", "qm_studio_baseline", "qm_studio_counters",
                "qm_studio_crowd", "qm_studio_extra", "qm_studio_service",
                "qm_studio_duration", "qm_studio_speed"):
        st.session_state.pop(key, None)


def _choose_recommended(count):
    st.session_state.qm_studio_counters = count


@st.fragment(run_every="1s")
def _playback():
    result = st.session_state.get("qm_studio_result")
    if not result:
        return
    speed = result["controls"].playback_speed
    duration = result["controls"].duration_minutes
    cursor = st.session_state.get("qm_studio_cursor", 0)
    if st.session_state.get("qm_studio_playing") and cursor < duration:
        cursor = min(duration, cursor + speed)
        st.session_state.qm_studio_cursor = cursor
        if cursor >= duration:
            st.session_state.qm_studio_playing = False
    point = result["proposed_path"][cursor]
    waiting = round(point["queue"])
    served_capacity = min(waiting, result["controls"].counters)
    st.markdown(f"#### Simulated minute {cursor} / {duration} · {speed}× playback")
    st.progress(cursor / duration)
    first, second, third = st.columns(3)
    first.metric("People Waiting · SIMULATED", f"{point['queue']:.1f}")
    second.metric("Available Counters", result["controls"].counters)
    third.metric("Potentially in Service", served_capacity,
                 help="Illustrative capacity, not an observed transaction count.")
    shown = min(waiting, 100)
    st.markdown("<div class='qm-sim-dots'>" + "".join("<i></i>" for _ in range(shown)) + "</div>",
                unsafe_allow_html=True)
    if waiting > shown:
        st.caption(f"Showing {shown} of approximately {waiting} simulated people.")
    else:
        st.caption("Each dot represents one rounded simulated person. This is not future CCTV footage.")


def _chart(result):
    figure = go.Figure()
    for key, name, color in (("current_path", "Current setup", "#173B63"),
                             ("proposed_path", "Your proposed setup", "#F97360")):
        path = result[key]
        figure.add_trace(go.Scatter(x=[point["minute"] for point in path],
                                    y=[point["queue"] for point in path],
                                    name=name, line=dict(color=color, width=3)))
    figure.update_layout(height=285, margin=dict(l=12, r=12, t=10, b=10),
                         xaxis_title="Minutes from frozen snapshot", yaxis_title="Expected people waiting")
    st.plotly_chart(figure, width="stretch", key="qm_studio_chart")


def _comparison(result):
    before, after = result["current"], result["proposed"]
    st.markdown("### Current Setup vs Your Proposed Setup")
    left, right = st.columns(2)
    for column, title, outcome in ((left, "Current Setup", before),
                                   (right, "Your Proposed Setup", after)):
        with column:
            st.markdown(f"#### {title}")
            a, b, c = st.columns(3)
            a.metric("Counters", outcome["counters"])
            b.metric("Expected Queue", f'{outcome["future_queue"]:.1f}')
            c.metric("Estimated Wait", f'{outcome["predicted_wait_minutes"]:.1f} min')
            st.caption(f'Congestion risk: {outcome["risk"].upper()}')
    reduction = result["wait_reduction_pct"]
    if reduction is not None:
        st.metric("Simulated Wait Change", f"{reduction:+.1f}% reduction")
    st.caption(f'Additional counters: {result["additional_counters"]:+d} · '
               "Opening more counters uses more staffing slots. These are what-if estimates, not guaranteed savings.")
    st.success(result["recommendation"]["reason"])


def _scenario_comparison(result):
    scenarios = st.session_state.setdefault("qm_studio_scenarios", [])
    if st.button("Save This Scenario", disabled=len(scenarios) >= 5):
        name = f"Scenario {len(scenarios) + 1}"
        scenarios.append({"Scenario": name, "Counters": result["controls"].counters,
                          "Arrival demand / min": round(result["proposed_arrival_rate"], 2),
                          "Queue after 20 min": (result["proposed_path"][20]["queue"]
                                                 if len(result["proposed_path"]) > 20 else None),
                          "Estimated wait": result["proposed"]["predicted_wait_minutes"],
                          "Risk": result["proposed"]["risk"],
                          "Snapshot": result["baseline_id"]})
    if scenarios:
        st.markdown("### Saved What-If Scenarios")
        st.dataframe(pd.DataFrame(scenarios).drop(columns="Snapshot"), hide_index=True, width="stretch")
        for index, scenario in enumerate(tuple(scenarios)):
            if st.button(f"Delete {scenario['Scenario']}", key=f"qm_delete_scenario_{index}"):
                scenarios.pop(index)
                st.rerun()


def render_simulation_studio(locations):
    st.title("What Happens If We Change Something?")
    st.caption("Test staffing and crowd scenarios from a frozen camera observation or a SYNTHETIC demo.")
    baseline = _baseline(locations)
    st.info(f"BASELINE · {baseline.source_label} · {baseline.timestamp.isoformat()} · "
            f"{baseline.queue_count:.0f} people waiting. Changes here never alter camera observations.")
    if "SYNTHETIC" in baseline.source_label:
        st.caption("SIMULATION MODE · SYNTHETIC facility demand, not measured campus data.")
    if st.button("Refresh Baseline from Latest Observation"):
        baseline = _refresh_baseline(locations)
        st.rerun()
    if baseline.arrival_rate_per_minute is None:
        st.warning("Observed entry history is insufficient. Choose a SYNTHETIC demand assumption to simulate.")
        assumed = st.number_input("SYNTHETIC arrivals per minute", min_value=0.0,
                                  max_value=20.0, value=1.0, step=.1)
        if st.button("Use SYNTHETIC Demand Assumption"):
            observation = st.session_state.get("qm_camera_observation")
            location = next(item for item in locations if item["id"] == baseline.location_id)
            st.session_state.qm_studio_baseline = baseline_from_live(
                observation, location, baseline.active_counters, synthetic_arrival_rate=assumed)
            st.rerun()
        return
    left, right = st.columns(2)
    with left:
        counters = st.slider("How Many Counters Are Open?", 1, 8,
                             value=min(8, baseline.active_counters), key="qm_studio_counters")
        crowd = st.slider("How Busy Will It Be?", 50, 200, value=100,
                          step=10, key="qm_studio_crowd", help="Percentage of baseline arrival demand.")
        extra = st.slider("Extra People Arrive", 0, 50, value=0, key="qm_studio_extra")
    with right:
        service = st.slider("Average Service Time · minutes", 1.0, 10.0,
                            value=max(1.0, min(10.0, baseline.service_time_minutes)),
                            step=.5, key="qm_studio_service")
        duration = st.select_slider("Simulation Duration", options=[5, 10, 20, 30],
                                    value=20, key="qm_studio_duration")
        speed = st.select_slider("Simulation Speed", options=[1, 2, 5, 10],
                                 value=1, key="qm_studio_speed", format_func=lambda value: f"{value}×")
    controls = SimulationControls(counters, crowd / 100, extra, service, duration, speed)
    run_col, pause_col, reset_col = st.columns(3)
    with run_col:
        if st.button("Run Simulation", type="primary", width="stretch"):
            st.session_state.qm_studio_result = run_what_if(baseline, controls)
            st.session_state.qm_studio_cursor = 0
            st.session_state.qm_studio_playing = True
    with pause_col:
        if st.button("Pause / Resume", width="stretch") and st.session_state.get("qm_studio_result"):
            st.session_state.qm_studio_playing = not st.session_state.get("qm_studio_playing", False)
    with reset_col:
        st.button("Reset", on_click=_reset, width="stretch")
    result = st.session_state.get("qm_studio_result")
    if result and result["baseline_id"] == baseline.snapshot_id:
        _playback()
        _comparison(result)
        _chart(result)
        st.markdown("#### Best Counter Option")
        recommended = result["recommendation"]["recommended_counters"]
        st.write(f"Try **{recommended} counters** for the proposed arrival and service assumptions.")
        st.button("Try Recommended Counters", disabled=recommended == counters,
                  on_click=_choose_recommended, args=(recommended,))
        _scenario_comparison(result)
        with st.expander("How does QueueMind predict this?"):
            st.write("QueueMind uses the frozen observed rate or a labelled synthetic demand assumption, "
                     "then simulates queue evolution and compares counter counts against the configured wait target. "
                     "The existing ML models were evaluated on synthetic data and have not been validated on live CCTV streams.")
    else:
        st.caption("Run Simulation to compare current staffing with your proposal.")
