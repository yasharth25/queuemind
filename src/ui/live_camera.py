"""Live Camera landing page; optional local connector work follows in V6."""

import streamlit as st


def render_live_camera(locations):
    st.title("Live Camera")
    st.caption("Connect an authorized camera, analyze a recording, or try a SYNTHETIC demo.")
    st.subheader("Connect CCTV / Camera")
    st.info("DISCONNECTED · No camera has been validated for this session.")
    st.write("A local QueueMind connector is required for private CCTV, RTSP, or host webcams. The public website cannot directly access devices on your network.")
    connect, upload, demo = st.columns(3)
    with connect:
        if st.button("Connect Camera", type="primary", width="stretch"):
            st.session_state.qm_show_camera_setup = True
    with upload:
        if st.button("Upload Video", width="stretch"):
            st.session_state.qm_show_video = True
    with demo:
        if st.button("Try Demo", width="stretch"):
            st.session_state.qm_show_demo = True
    if st.session_state.get("qm_show_camera_setup"):
        st.markdown("#### Use Local Connector")
        st.code("python -m scripts.camera_connector", language="powershell")
        st.caption("Camera discovery and connection will run only on an authorized local machine. No cameras are connected yet.")
    if st.session_state.get("qm_show_video"):
        from src.ui.vision_lab import render_video_upload
        render_video_upload(locations)
    if st.session_state.get("qm_show_demo"):
        from src.ui.operations_tabs import render_live_tab
        st.caption("SIMULATION MODE · SYNTHETIC · No CCTV data")
        render_live_tab(locations)
