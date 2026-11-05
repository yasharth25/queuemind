"""Low-friction landing view backed by the existing demo and vision state."""

import streamlit as st

from src.ui.command_model import snapshot
from src.ui.feature_help import feature_heading


def _switch(name: str) -> None:
    st.session_state.main_tabs = name


def render_home(locations) -> None:
    st.markdown(
        '<div class="qm-home-hero"><div><span class="qm-eyebrow">QUEUE INTELLIGENCE, MADE SIMPLE</span>'
        '<h1>Know Your Queue Before It Gets Long</h1>'
        '<p>Upload a video or try our live demo to count people, predict waiting times, '
        'and find the best number of counters.</p></div>'
        '<div class="qm-home-steps"><b>1 · Upload</b><b>2 · See the queue</b>'
        '<b>3 · Plan ahead</b></div></div>', unsafe_allow_html=True,
    )
    upload_side, glance_side = st.columns([1.8, 1], gap="large")
    with upload_side:
        feature_heading("Upload Video", "QueueMind counts people, not identities. Only upload videos you may analyze.")
        from src.ui.vision_lab import render_video_upload
        render_video_upload(locations, home=True)
    with glance_side:
        if st.session_state.get("qm_vision_result") is None:
            state = st.session_state.qm_live
            view = snapshot(state)
            st.markdown("#### Or try the Live Demo")
            st.caption("DEMO DATA · SYNTHETIC · Not a real CCTV feed")
            st.metric("People Waiting", f'{view["queue"]:.0f}', help="Current simulated queue count.")
            st.metric("Estimated Wait", f'{view["wait"]:.1f} min', help="Uses the demo's example service time and open counters.")
            st.metric("Queue in 20 Min", f'{view["queue20"]:.0f}', help="Simulation from synthetic demand.")
            st.caption(f'{state["active_counters"]} open counters · Try {view["recommendation"]["recommended_counters"]} in this demo.')
        if st.button("Try Live Demo →", key="qm_home_demo", on_click=_switch, args=("Live Demo",), width="stretch"):
            st.rerun()
