"""Six clear, lazily rendered pages within the existing Streamlit application."""

import streamlit as st

from src.demo.live import initial_state, set_counters
from src.ui.command_center import _rotate_counter_widget, _sync_controls
from src.ui.operations_tabs import render_forecast_tab, render_live_tab, render_optimizer_tab

TAB_NAMES = ("Home", "Upload Video", "Live Demo", "Predictions", "Best Counters", "Results")


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


def _header():
    video = st.session_state.get("qm_state_source") == "VISION" and st.session_state.get("qm_vision_state") is not None
    source = "From Uploaded Video" if video else "Using Demo Data · SYNTHETIC"
    detail = "Anonymous people counts" if video else "Simulation, not a real CCTV feed"
    st.markdown(
        '<header class="qm-product-header"><div class="qm-product-brand">'
        '<span class="qm-clock">◷</span><div><b>QueueMind</b>'
        '<small>Know your queue. Plan ahead.</small></div></div>'
        f'<div class="qm-product-status"><span class="qm-tag">{source}</span>'
        f'<span class="qm-muted">{detail}</span></div></header>',
        unsafe_allow_html=True,
    )


def render_product(locations):
    """Only the selected tab runs; upload and simulation state survive tab changes."""
    _ensure_state()
    _header()
    tabs = st.tabs(TAB_NAMES, key="main_tabs", bind="query-params")
    for name, tab in zip(TAB_NAMES, tabs):
        if not tab.open:
            continue
        with tab:
            if name == "Home":
                from src.ui.simple_home import render_home
                render_home(locations)
            elif name == "Upload Video":
                from src.ui.vision_lab import render_video_upload
                render_video_upload(locations)
            elif name == "Live Demo":
                render_live_tab(locations)
            elif name == "Predictions":
                render_forecast_tab(locations)
            elif name == "Best Counters":
                render_optimizer_tab(locations)
            else:
                from src.ui.simple_results import render_results
                render_results(locations)
