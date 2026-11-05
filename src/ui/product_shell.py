"""Two-page QueueMind product shell with existing tools in Advanced Details."""

import streamlit as st

from src.demo.live import initial_state, set_counters
from src.ui.command_center import _rotate_counter_widget, _sync_controls

TAB_NAMES = ("Live Camera", "Simulation Studio")


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
    status = st.session_state.get("qm_camera_status", "DISCONNECTED")
    if status == "DISCONNECTED" and st.session_state.get("qm_show_demo"):
        status = "DEMO"
    video = st.session_state.get("qm_state_source") == "VISION" and st.session_state.get("qm_vision_state") is not None
    source = "From Uploaded Video" if video else "SYNTHETIC demo available"
    st.markdown(
        '<header class="qm-product-header"><div class="qm-product-brand">'
        '<span class="qm-clock">◷</span><div><b>QueueMind</b>'
        '<small>See your queue. Predict the wait. Take action.</small></div></div>'
        f'<div class="qm-product-status"><span class="qm-tag">{status}</span>'
        f'<span class="qm-muted">{source}</span></div></header>',
        unsafe_allow_html=True,
    )


def _advanced(locations, page):
    with st.expander("Advanced Details · existing tools and evaluation"):
        choice = st.selectbox("Open tool", (
            "None", "Uploaded Video", "Live Demo", "Predictions", "Best Counters",
            "Results", "Evaluation", "Vision Lab", "ML Lab",
        ), key="qm_advanced_tool")
        if (page == "Live Camera" and st.session_state.get("qm_show_demo")
                and choice in ("Live Demo", "Predictions", "Best Counters", "Results")):
            st.info("Hide the SYNTHETIC demo above to open this advanced tool here.")
            return
        if (page == "Live Camera" and st.session_state.get("qm_show_video")
                and choice in ("Uploaded Video", "Vision Lab")):
            st.info("Hide the video analyzer above to open this advanced tool here.")
            return
        if choice == "Uploaded Video":
            from src.ui.vision_lab import render_video_upload
            render_video_upload(locations)
        elif choice == "Live Demo":
            from src.ui.operations_tabs import render_live_tab
            render_live_tab(locations)
        elif choice == "Predictions":
            from src.ui.operations_tabs import render_forecast_tab
            render_forecast_tab(locations)
        elif choice == "Best Counters":
            from src.ui.operations_tabs import render_optimizer_tab
            render_optimizer_tab(locations)
        elif choice == "Results":
            from src.ui.simple_results import render_results
            render_results(locations)
        elif choice == "Evaluation":
            from src.ui.operations_tabs import render_evaluation_tab
            render_evaluation_tab(locations)
        elif choice == "Vision Lab":
            from src.ui.vision_lab import render_vision_lab
            render_vision_lab()
        elif choice == "ML Lab":
            from src.ui.ml_lab import render_ml_lab
            render_ml_lab(locations)


def render_product(locations):
    """Render only the selected primary page; keep camera and demo state in session."""
    _ensure_state()
    _header()
    tabs = st.tabs(TAB_NAMES, key="main_tabs", bind="query-params")
    for name, tab in zip(TAB_NAMES, tabs):
        if not tab.open:
            continue
        with tab:
            if name == "Live Camera":
                from src.ui.live_camera import render_live_camera
                render_live_camera(locations)
            else:
                from src.ui.simulation_studio import render_simulation_studio
                render_simulation_studio(locations)
            _advanced(locations, name)
