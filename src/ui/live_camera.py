"""Cloud-safe camera dashboard: authenticated aggregates, local preview guidance."""

from __future__ import annotations

from datetime import datetime, timezone
import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.queueing.simulator import estimate_wait_minutes
from src.vision.live.client import LiveAPIClient, validate_endpoint
from src.vision.live.forecast import project_live_queue
from src.vision.live.service import deserialize_observation


def _service_endpoint():
    value = os.environ.get("QUEUEMIND_LIVE_API_ENDPOINT", "")
    if not value:
        try:
            value = st.secrets.get("QUEUEMIND_LIVE_API_ENDPOINT", "")
        except Exception:
            value = ""
    return validate_endpoint(value) if value else None


def _clear_view():
    for key in ("qm_live_view_token", "qm_connected_cameras", "qm_selected_camera",
                "qm_camera_observation", "qm_camera_status", "qm_studio_baseline",
                "qm_studio_result", "qm_pairing_code", "qm_admin_message",
                "qm_pair_admin_token", "qm_live_admin_token"):
        st.session_state.pop(key, None)


def _admin_action(endpoint, action):
    key = "qm_pair_admin_token" if action == "pair" else "qm_live_admin_token"
    token = st.session_state.get(key, "")
    if not token:
        st.session_state.qm_admin_message = "Enter your workspace admin token first."
        return
    try:
        client = LiveAPIClient(endpoint, token)
        camera_id = st.session_state.get("qm_selected_camera")
        if action == "pair":
            code = client._request("POST", "/api/v1/pairing-codes")["pairing_code"]
            st.session_state.qm_pairing_code = code
            st.session_state.qm_admin_message = "Pairing code created. It expires in five minutes and works once."
        elif action == "revoke" and camera_id:
            client.disconnect(camera_id)
            st.session_state.qm_camera_status = "DISCONNECTED"
            st.session_state.qm_admin_message = "Connector access revoked for this camera."
        elif action == "remove" and camera_id:
            client.remove(camera_id)
            st.session_state.pop("qm_selected_camera", None)
            st.session_state.pop("qm_connected_cameras", None)
            st.session_state.qm_camera_observation = None
            st.session_state.qm_camera_status = "DISCONNECTED"
            st.session_state.qm_admin_message = "Camera and its aggregate history removed from this service process."
    except (ValueError, RuntimeError) as exc:
        st.session_state.qm_admin_message = str(exc)
    finally:
        st.session_state[key] = ""


def _setup_panel(endpoint):
    st.subheader("Connect CCTV / Camera")
    st.caption("Only use cameras you own or are explicitly authorized to monitor.")
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
    if not endpoint:
        st.info("CONNECTOR REQUIRED · This public demo has no live data service configured. No camera is connected.")
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
        st.caption("Connection means the local connector decoded real frames. Manufacturer and model alone cannot connect a camera.")
        st.markdown("#### Mark Your Queue Area")
        st.write("On the connector computer, click the queue polygon, entry line, and service line on a real preview frame. "
                 "The resulting normalized geometry is saved per camera without video or credentials.")
        st.code("python -m scripts.camera_connector --source webcam --calibrate --calibration config/calibrations/my_camera.json", language="powershell")
        if endpoint:
            st.markdown("#### Pair this workspace")
            st.write("1. Start the local connector on the camera network. 2. Create a short-lived code below. "
                     "3. Run the connector with the configured service URL and enter the code at its local prompt.")
            st.code("python -m scripts.camera_connector --source rtsp --host CAMERA_IP --path STREAM_PATH --service-url YOUR_HTTPS_SERVICE", language="powershell")
            st.caption("Aggregate counts are sent by default. Add --share-preview only if authorized viewers may see annotated camera frames; previews then leave the local network and expire in service memory.")
            st.text_input("Workspace admin token", type="password", key="qm_pair_admin_token",
                          help="Used only for pairing/revocation. Never use a camera password here.")
            st.button("Create Pairing Code", on_click=_admin_action, args=(endpoint, "pair"))
            if st.session_state.get("qm_pairing_code"):
                st.code(st.session_state.qm_pairing_code)
            if st.session_state.get("qm_admin_message"):
                st.caption(st.session_state.qm_admin_message)


def _preview_placeholder(observation=None):
    if observation is None:
        st.markdown('<div class="qm-camera-preview"><strong>Camera preview unavailable</strong>'
                    '<span>Connect a local QueueMind connector or analyze an uploaded video.</span></div>',
                    unsafe_allow_html=True)
    else:
        st.markdown('<div class="qm-camera-preview"><strong>Local camera preview stays on the connector</strong>'
                    '<span>Only anonymous aggregate counts reach this page. Run the connector with '
                    '--preview on the camera computer to view detection boxes and the queue area.</span></div>',
                    unsafe_allow_html=True)


@st.fragment(run_every="3s")
def _camera_feed(endpoint, token, camera_id, location, active_counters):
    try:
        client = LiveAPIClient(endpoint, token)
        latest = client.latest(camera_id)
    except (RuntimeError, ValueError) as exc:
        st.session_state.qm_camera_status = "ERROR"
        st.error(str(exc))
        _preview_placeholder()
        return
    status = latest["status"]
    st.session_state.qm_camera_status = status
    data = latest["observation"]
    if data is None:
        st.session_state.qm_camera_observation = None
        st.info(f"{status} · No analyzed frame has been received yet.")
        _preview_placeholder()
        return
    observation = deserialize_observation(data)
    st.session_state.qm_camera_observation = observation
    st.session_state.qm_camera_counters = active_counters
    stale = observation.is_stale() or status != "PROCESSING"
    if stale:
        st.warning(f"Camera is {status.lower()} or data is out of date. Last valid observation: "
                   f"{observation.captured_at.isoformat()}. Predictions are paused.")
    else:
        st.success(f"{status} · Last analyzed frame captured at {observation.captured_at.isoformat()}")
    preview = None
    if not stale:
        try:
            preview = client.preview(camera_id)
        except RuntimeError:
            pass
    if preview:
        st.image(preview[0], caption=f"Annotated preview · CAPTURED {preview[1]} · protected viewer access",
                 width="stretch")
        st.caption("PREVIEW SHARING ENABLED · annotated JPEG frames leave the connector network only with explicit local opt-in.")
    else:
        _preview_placeholder(observation)
    st.caption(f"{observation.source_type.value.replace('_', ' ')} · "
               "preview remains local; no CCTV frame is sent to this service.")
    wait = estimate_wait_minutes(observation.people_waiting, active_counters,
                                 location["average_service_time_minutes"])
    cols = st.columns(4)
    cols[0].metric("People Waiting · OBSERVED", observation.people_waiting,
                   help="Anonymous tracked people inside the marked queue area.")
    cols[1].metric("Estimated Wait · CONFIGURED CAPACITY", f"{wait:.1f} min",
                   help="Queue divided by service capacity from configured counters and service time.")
    cols[2].metric("People Arriving · OBSERVED", observation.entries_in_window,
                   help="Entry-line crossings in the latest analysis window, not identities.")
    cols[3].metric("Service Boundary Crossings · OBSERVED", observation.service_boundary_crossings,
                   help="Crossing the service line does not prove a completed transaction.")
    if stale:
        return
    projection = project_live_queue(observation, location, active_counters=active_counters)
    if not projection["available"]:
        st.info(projection["reason"] + ". No live future values are shown.")
        return
    next20 = projection["points"][3]
    a, b, c = st.columns(3)
    a.metric("Expected Queue in 20 Min · ESTIMATED", f'{next20["queue"]:.1f}')
    b.metric("Congestion Risk · ESTIMATED", next20["congestion"].upper())
    c.metric("Suggested Counters · SIMULATED", projection["recommendation"]["recommended_counters"])
    st.caption("Demand method: observed entry-rate persistence. The existing ML model has not been validated on CCTV streams.")
    for alert in projection["alerts"]:
        st.warning(alert)
    with st.expander("Advanced Camera Details · history and timing"):
        st.write(f"Capture time: {observation.captured_at.isoformat()} · "
                 f"Processed time: {observation.processed_at.isoformat()}")
        st.write(f"Analysis FPS: {observation.analysis_fps or 'unavailable'} · "
                 f"Processing latency: {observation.processing_latency_s:.3f} s · "
                 "Camera FPS: unavailable from aggregate feed")
        st.write(f"Arrival rate: {observation.observed_arrival_rate} people/min · "
                 f"Service-boundary rate: {observation.service_boundary_rate} crossings/min · "
                 f"Configured service capacity: {projection['configured_service_capacity']:.2f} people/min")
        try:
            history = client.history(camera_id)
            if history:
                frame = pd.DataFrame(history)
                figure = go.Figure(go.Scatter(x=frame["captured_at"], y=frame["people_waiting"],
                                              name="Observed queue", line_color="#173B63"))
                figure.update_layout(height=230, margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(figure, width="stretch")
        except RuntimeError as exc:
            st.caption(str(exc))


def _workspace_panel(endpoint, locations):
    token = st.text_input("Workspace viewer token", type="password", key="qm_live_view_token",
                          help="An authorized workspace token; never enter camera credentials here.")
    if st.button("Connect to Workspace", disabled=not bool(token)):
        try:
            cameras = LiveAPIClient(endpoint, token).cameras()
            st.session_state.qm_connected_cameras = cameras
            st.session_state.qm_camera_status = "CONNECTING" if cameras else "DISCONNECTED"
        except (RuntimeError, ValueError) as exc:
            st.error(str(exc))
            st.session_state.qm_connected_cameras = None
    cameras = st.session_state.get("qm_connected_cameras")
    if cameras is None:
        _preview_placeholder()
        return
    if not cameras:
        st.info("No paired cameras in this workspace. Start the connector and pair it first.")
        _preview_placeholder()
        return
    options = {camera["camera_id"]: camera for camera in cameras}
    selected = st.selectbox("Select Camera", tuple(options),
                            format_func=lambda key: f"{options[key]['name']} · {options[key]['location_id']}",
                            key="qm_selected_camera")
    camera = options[selected]
    previous = st.session_state.get("qm_camera_observation")
    if previous is not None and previous.camera_id != selected:
        st.session_state.qm_camera_observation = None
    location = next((item for item in locations if item["id"] == camera["location_id"]), None)
    if location is None:
        st.error("This camera's facility configuration is unavailable.")
        return
    _camera_feed(endpoint, token, selected, location, camera["active_counters"])
    with st.expander("Camera access and privacy"):
        st.write("Disconnecting this view leaves the local connector running. "
                 "Revoking access stops that connector token. Removal also deletes this service process's aggregate history.")
        st.text_input("Workspace admin token for revoke/remove", type="password",
                      key="qm_live_admin_token")
        one, two, three = st.columns(3)
        one.button("Disconnect View", on_click=_clear_view)
        two.button("Revoke Connector Access", on_click=_admin_action, args=(endpoint, "revoke"))
        three.button("Remove Camera", on_click=_admin_action, args=(endpoint, "remove"))
        st.button("Clear Session Data", on_click=_clear_view)
        if st.session_state.get("qm_admin_message"):
            st.caption(st.session_state.qm_admin_message)


def render_live_camera(locations):
    st.title("Live Camera")
    st.caption("See your queue. Predict the wait. Take action. Observations and assumptions stay separately labelled.")
    endpoint = _service_endpoint()
    _setup_panel(endpoint)
    if endpoint:
        _workspace_panel(endpoint, locations)
    else:
        _preview_placeholder()
    if st.session_state.get("qm_show_video"):
        if st.button("Hide Video Analyzer"):
            st.session_state.qm_show_video = False
            st.rerun()
        from src.ui.vision_lab import render_video_upload
        render_video_upload(locations)
    if st.session_state.get("qm_show_demo"):
        if st.button("Hide SYNTHETIC Demo"):
            st.session_state.qm_show_demo = False
            st.rerun()
        from src.ui.operations_tabs import render_live_tab
        st.caption("SIMULATION MODE · SYNTHETIC · No CCTV data")
        render_live_tab(locations)
