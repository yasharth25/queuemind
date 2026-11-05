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
        method = st.radio("How will you connect?", (
            "Find Cameras on My Network", "Add Camera Manually", "Use My Webcam",
        ), horizontal=True, key="qm_camera_method")
        if method == "Find Cameras on My Network":
            st.write("Run discovery on the connector computer using an authorized local network interface.")
            st.code("python -m scripts.camera_connector --discover --interface YOUR_LOCAL_IPV4", language="powershell")
            st.caption("Discovery lists ONVIF candidates only; it does not validate stream access or guess passwords.")
        elif method == "Add Camera Manually":
            st.write("Have the camera IP/hostname, RTSP path or ONVIF support, port, and authorized credentials ready. Brand and model are descriptive only.")
            st.code("python -m scripts.camera_connector --source rtsp --host CAMERA_IP --path STREAM_PATH --test-only", language="powershell")
            st.caption("The connector prompts for credentials locally and keeps them in memory. This cloud page does not receive them.")
        else:
            st.write("A webcam must be attached to the computer running the local connector. The cloud server cannot read your browser's webcam.")
            st.code("python -m scripts.camera_connector --source webcam --webcam-index 0 --test-only", language="powershell")
        st.caption("No camera is connected until the local connector validates decoded frames.")
    if st.session_state.get("qm_show_video"):
        from src.ui.vision_lab import render_video_upload
        render_video_upload(locations)
    if st.session_state.get("qm_show_demo"):
        from src.ui.operations_tabs import render_live_tab
        st.caption("SIMULATION MODE · SYNTHETIC · No CCTV data")
        render_live_tab(locations)
