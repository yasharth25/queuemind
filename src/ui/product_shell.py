"""Stateful, lazily rendered product navigation for the QueueMind demo."""

from html import escape

import streamlit as st

from src.demo.live import initial_state, set_counters
from src.ui.command_center import _rotate_counter_widget, _sync_controls
from src.ui.command_model import snapshot
from src.ui.operations_tabs import (
    render_evaluation_tab, render_forecast_tab, render_live_tab,
    render_optimizer_tab, render_overview_tab, render_vision_tab,
)

TAB_NAMES = (
    "Overview", "Live Simulation", "Forecast", "Optimizer",
    "ML Lab", "Vision Lab", "Evaluation",
)


def _ensure_state():
    if "qm_live" not in st.session_state:
        st.session_state.qm_live = initial_state("Canteen Lunch Rush")
        _sync_controls()
    counter_key = f'qm_counters_{st.session_state.get("qm_counter_version", 0)}'
    if any(key not in st.session_state for key in (counter_key, "qm_demand", "qm_speed")):
        _sync_controls()
    pending = st.session_state.pop("qm_pending_counters", None)
    if pending is not None:
        state = st.session_state.qm_live
        st.session_state.qm_comparison_from = state["active_counters"]
        set_counters(state, pending)
        _rotate_counter_widget()


def _header(state, view, location):
    if state["simulation_running"]:
        status = "LIVE"
    elif state["complete"]:
        status = "COMPLETE"
    elif state["simulation_elapsed_minutes"]:
        status = "PAUSED"
    else:
        status = "READY"
    name = escape(location["display_name"])
    scenario = escape(state["selected_scenario"])
    st.markdown(
        '<header class="qm-product-header">'
        '<div class="qm-product-brand"><span class="qm-clock">◷</span>'
        '<div><b>QueueMind</b><small>See now. Predict next. Act early.</small></div></div>'
        f'<div class="qm-product-status"><span class="qm-tag">SYNTHETIC DEMO</span>'
        f'<span class="qm-badge {"normal" if status == "LIVE" else "info"}">{status}</span>'
        f'<span>{name}</span><span class="qm-muted">{scenario}</span></div></header>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="qm-context">'
        f'<span><b>SCENARIO</b> {scenario}</span>'
        f'<span><b>FACILITY</b> {name}</span>'
        f'<span><b>SIMULATION</b> {status} · +{state["simulation_elapsed_minutes"]}m</span>'
        f'<span><b>COUNTERS</b> {state["active_counters"]}</span>'
        f'<span><b>RECOMMENDED</b> {view["recommendation"]["recommended_counters"]}</span>'
        '<span class="qm-context-synthetic">SYNTHETIC DATA</span></div>',
        unsafe_allow_html=True,
    )


def render_product(locations):
    """Render only the selected primary tab; simulation state lives above tabs."""
    _ensure_state()
    state = st.session_state.qm_live
    location_by_id = {item["id"]: item for item in locations}
    location = location_by_id[state["selected_location"]]
    view = snapshot(state)
    _header(state, view, location)
    tabs = st.tabs(TAB_NAMES, key="main_tabs", bind="query-params")
    handlers = {
        "Overview": lambda: render_overview_tab(locations),
        "Live Simulation": lambda: render_live_tab(locations),
        "Forecast": lambda: render_forecast_tab(locations),
        "Optimizer": lambda: render_optimizer_tab(locations),
        "Vision Lab": render_vision_tab,
        "Evaluation": lambda: render_evaluation_tab(locations),
    }
    for name, tab in zip(TAB_NAMES, tabs):
        if tab.open:
            with tab:
                if name == "ML Lab":
                    from src.ui.ml_lab import render_ml_lab

                    render_ml_lab(locations)
                else:
                    handlers[name]()
