"""Session-scoped video analysis and optional advanced vision diagnostics."""

from __future__ import annotations

from contextlib import nullcontext
from html import escape
import importlib.util
import logging
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.config import get_enabled_locations
from src.evaluation.vision import TEMPLATE_CSV, evaluate_vision_labels
from src.vision.analyzer import analyze_video
from src.vision.config import load_vision_config, validate_line, validate_polygon
from src.vision.integration import project_vision_state, state_from_frame
from src.vision.remote import analyze_remote_upload, remote_config
from src.vision.video_io import open_video, read_sample_frame, uploaded_video_file

LOG = logging.getLogger(__name__)
SECTIONS = (
    "Video Analyzer", "ROI Setup", "Live Metrics", "Timeline",
    "Forecast Integration", "Evaluation", "Pipeline / Privacy",
)
MAX_UPLOAD_BYTES = 40 * 1024 * 1024
MAX_DURATION_S = 60
MAX_PIXELS = 1280 * 720
QUEUE_AREAS = {
    "Middle of frame": ((.20, .25), (.80, .25), (.80, .95), (.20, .95)),
    "Left side": ((.02, .20), (.52, .20), (.52, .98), (.02, .98)),
    "Right side": ((.48, .20), (.98, .20), (.98, .98), (.48, .98)),
}


def _defaults():
    config = load_vision_config()
    st.session_state.setdefault("qm_vision_roi", tuple(map(tuple, config["default_roi"])))
    st.session_state.setdefault("qm_vision_entry", tuple(map(tuple, config["entry_line"])))
    st.session_state.setdefault("qm_vision_service", tuple(map(tuple, config["service_line"])))
    st.session_state.setdefault("qm_vision_model_path", config["model_path"])
    st.session_state.setdefault("qm_vision_confidence", float(config["confidence"]))
    st.session_state.setdefault("qm_vision_rate_window", int(config["rate_window_seconds"]))


@st.cache_resource(show_spinner=False)
def _yolo_model(model_path: str):
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Local Vision Mode requires Ultralytics. Install requirements-vision.txt.") from exc
    return YOLO(model_path)


def _person_detector():
    from src.vision.detector import PersonDetector
    return PersonDetector(
        st.session_state.qm_vision_model_path,
        st.session_state.qm_vision_confidence,
        model=_yolo_model(st.session_state.qm_vision_model_path),
    )


def _local_available() -> bool:
    return all(importlib.util.find_spec(name) is not None for name in ("cv2", "ultralytics"))


def _remote_available() -> bool:
    try:
        return remote_config(st.secrets) is not None
    except ValueError:
        return False


def _available() -> bool:
    return _local_available() or _remote_available()


def _source_context():
    uploaded = st.session_state.get("qm_vision_upload_keep")
    if uploaded is not None:
        return uploaded_video_file(uploaded)
    local_path = st.session_state.get("qm_vision_local_path_keep", "").strip()
    if not local_path:
        raise ValueError("Upload a video or select a local video path first.")
    if "://" in local_path:
        raise ValueError("Only local files are supported; RTSP and remote URLs are not enabled.")
    return nullcontext(Path(local_path))


def _runtime_status():
    cv2_ready = importlib.util.find_spec("cv2") is not None
    yolo_ready = importlib.util.find_spec("ultralytics") is not None
    st.markdown("#### Vision runtime status")
    columns = st.columns(5)
    for column, label, value in zip(columns,
        ("OpenCV", "Ultralytics", "YOLO model", "Local camera", "Vision evaluation"),
        ("Available" if cv2_ready else "Not installed",
         "Available" if yolo_ready else "Not installed",
         "Loaded" if st.session_state.get("qm_vision_model_loaded") else "Not loaded",
         "Local CLI only", "Available"),
    ):
        column.metric(label, value)
    st.caption("Cloud inference needs a separately configured private worker. Runtime checks do not load CV libraries or weights.")
    if not _available():
        st.info("VIDEO INFERENCE UNAVAILABLE · Install local vision dependencies or configure an authorized private worker. CSV evaluation remains available.")
        st.code("python -m pip install -r requirements-vision.txt\npython -m streamlit run app.py", language="powershell")


def _video_metadata():
    with _source_context() as path:
        capture, metadata = open_video(path)
        capture.release()
    return metadata


def _video_analyzer(locations, *, home=False):
    st.markdown('<div class="qm-upload-intro"><b>Analyze Your Queue Video</b>'
                '<span>Upload a short queue recording. We detect people only inside the queue area you choose.</span></div>',
                unsafe_allow_html=True)
    current_upload = st.file_uploader(
        "Upload Queue Video · MP4, MOV, AVI, or MKV", type=["mp4", "avi", "mov", "mkv"],
        key=f'qm_vision_video_upload_{st.session_state.get("qm_video_upload_version", 0)}',
        help="Up to 40 MB and 60 seconds. QueueMind counts people, not identities.",
    )
    if current_upload is not None:
        st.session_state.qm_vision_upload_keep = current_upload
    uploaded = current_upload or st.session_state.get("qm_vision_upload_keep")
    with st.expander("Advanced Video Settings · local file and calibration"):
        local_path = st.text_input(
            "Local video path (optional, on this computer)",
            value=st.session_state.get("qm_vision_local_path_keep", ""),
            key=f'qm_vision_local_path_{st.session_state.get("qm_video_upload_version", 0)}',
            help="Used only if no video is uploaded. The path is read on this server; remote URLs and RTSP are unsupported.",
        )
        st.session_state.qm_vision_local_path_keep = local_path.strip()
    if uploaded is not None:
        if uploaded.size > MAX_UPLOAD_BYTES:
            st.error("This video is too large. Choose a file smaller than 40 MB.")
            return
        st.caption(f"{uploaded.name} · {uploaded.size / 1024**2:.1f} MB · held in this session only")
        if Path(uploaded.name).suffix.lower() in {".mp4", ".mov"}:
            st.video(uploaded)
        else:
            st.caption("Your browser may not preview AVI/MKV. Analysis depends on the server decoder.")
    elif st.session_state.get("qm_vision_local_path_keep", "").strip():
        path = Path(st.session_state.qm_vision_local_path_keep)
        if path.is_file() and path.suffix.lower() in {".mp4", ".mov"}:
            st.video(str(path))
    else:
        st.info("Start by uploading a queue video. Upload → Analyze → See Results.")
        if not _available():
            st.caption("This cloud demo currently has no video inference backend. The Live Demo remains available; uploaded-video analysis requires the optional local vision setup.")
        return
    local_ready, remote_ready = _local_available(), _remote_available()
    if not (local_ready or remote_ready):
        st.warning("Video preview is available, but this deployment has no person-detection backend. Analysis is available with the optional local vision setup; this upload has not been analyzed.")
        return
    metadata = None
    if local_ready:
        try:
            metadata = _video_metadata()
        except (OSError, ValueError, RuntimeError) as exc:
            st.error(f"We could not read this video: {exc}")
            return
        if metadata.duration_s is not None and metadata.duration_s > MAX_DURATION_S:
            st.error("This video is longer than 60 seconds. Please trim it before analysis.")
            return
        if metadata.width * metadata.height > MAX_PIXELS:
            st.error("This video is above 1280 × 720. Please use a smaller copy for this demo.")
            return
        st.caption(f"{metadata.duration_s:.1f} seconds · {metadata.width} × {metadata.height} · {metadata.fps:.1f} frames/second")
    else:
        st.caption("The private video worker checks clip duration and resolution before processing. Limit: 30 seconds, 1280 × 720, 40 MB.")
        if uploaded is None:
            st.error("Remote analysis requires an uploaded video, not a server-local path.")
            return
    st.markdown("**Select Queue Area**")
    st.caption("Choose the part of the picture where people wait. People elsewhere are not counted as waiting.")
    area = st.selectbox("Queue area preset", ["Middle of frame", "Left side", "Right side", "Custom area"],
                        key="qm_vision_area_choice")
    if area != "Custom area":
        st.session_state.qm_vision_roi = QUEUE_AREAS[area]
    if local_ready:
        try:
            with _source_context() as path:
                sample, _ = read_sample_frame(path, 0)
            from src.vision.annotation import annotate_frame
            from src.vision.video_io import encode_preview
            outlined = annotate_frame(sample, [], [], st.session_state.qm_vision_roi,
                                      st.session_state.qm_vision_entry, st.session_state.qm_vision_service, 0)
            st.image(encode_preview(outlined), caption="Queue area preview · adjust before analyzing", width=480)
        except (OSError, ValueError, RuntimeError) as exc:
            st.warning(f"Queue-area preview unavailable: {exc}")
    if area == "Custom area":
        _roi_editor()
    names = [item["display_name"] for item in locations]
    selected_location = st.selectbox("Facility type", names, key="qm_vision_analysis_location")
    st.session_state.qm_vision_location = names.index(selected_location)
    mode = st.radio("Analysis speed", ("Quick Analysis", "Full Analysis"), horizontal=True,
                    help="Quick mode skips frames and may miss brief boundary crossings.")
    stride = 1 if mode == "Full Analysis" else 5
    with st.expander("ADVANCED VISION SETTINGS"):
        if mode == "Quick Analysis":
            stride = st.select_slider("Frame skipping", options=[2, 3, 5, 10], value=5)
        st.slider("Person confidence", .10, .90, step=.05, key="qm_vision_confidence")
        st.text_input("YOLO model path/name", key="qm_vision_model_path",
                      help="Default yolo11n.pt may download once on the local machine if absent. No model is loaded in the cloud page.")
        st.select_slider("Rolling event-rate window · seconds", options=[30, 60, 120], key="qm_vision_rate_window")
        iou = st.slider("Anonymous tracker IoU threshold", .10, .80, .30, .05)
        max_previews = st.slider("Maximum stored annotated previews", 10, 150, 80, 10)
        save_video = st.checkbox("GENERATE ANNOTATED VIDEO", value=False,
                                 help="Optional MP4 encoding costs time and memory; OFF by default.")
    st.caption("Advanced queue geometry and entry/service lines can be adjusted below. No face or identity processing.")
    with st.expander("Advanced Queue Area and Crossing Lines"):
        if local_ready:
            _roi_setup()
        else:
            _roi_editor()
            st.caption("Entry and service lines use configured defaults on the private worker.")
    start, reset = st.columns(2)
    if reset.button("Analyze Another Video", width="stretch"):
        for key in ("qm_vision_result", "qm_vision_state", "qm_vision_frame_selection",
                    "qm_vision_optimizer_ran", "qm_vision_upload_keep",
                    "qm_vision_local_path_keep", "qm_vision_demand_choice_keep",
                    "qm_vision_demand_choice"):
            st.session_state.pop(key, None)
        st.session_state.qm_video_upload_version = st.session_state.get("qm_video_upload_version", 0) + 1
        if st.session_state.get("qm_state_source") == "VISION":
            st.session_state.qm_state_source = "SYNTHETIC"
        st.rerun()
    if not start.button("Analyze Video", type="primary", width="stretch"):
        if st.session_state.get("qm_vision_result") is not None:
            _simple_video_result(locations)
        return
    progress_bar = st.progress(0, text="Reading your video…")
    status = st.empty()
    def progress(index, total, elapsed):
        fraction = min(1.0, index / total) if total else 0.0
        progress_bar.progress(fraction, text=f"Finding and counting people · frame {index:,} / {total:,}" if total else f"Finding and counting people · frame {index:,}")
        processed = max(1, (index + stride - 1) // stride)
        status.caption(f"Elapsed {elapsed:.1f}s · analysis {processed / max(elapsed, .001):.1f} FPS · video {metadata.fps:.1f} FPS")
    try:
        if local_ready:
            output_context = TemporaryDirectory(prefix="queuemind_output_") if save_video else nullcontext(None)
            with _source_context() as path, output_context as folder:
                output = Path(folder) / "annotated.mp4" if save_video else None
                result = analyze_video(
                    path, _person_detector(), locations[st.session_state.get("qm_vision_location", 0)],
                    st.session_state.qm_vision_roi, st.session_state.qm_vision_entry,
                    st.session_state.qm_vision_service, stride=stride,
                    rate_window_s=st.session_state.qm_vision_rate_window,
                    max_previews=max_previews, iou_threshold=iou,
                    annotated_output=output, progress=progress,
                )
                if output is not None:
                    size = output.stat().st_size
                    if size > 100 * 1024**2:
                        st.warning("Annotated MP4 exceeded 100 MB and was not retained for download.")
                    elif size:
                        result.annotated_video = output.read_bytes()
        else:
            status.caption("Sending the clip to the authorized private worker; waiting for real analysis…")
            result = analyze_remote_upload(uploaded, locations[st.session_state.qm_vision_location]["id"],
                                           st.session_state.qm_vision_roi, st.session_state.qm_vision_entry,
                                           st.session_state.qm_vision_service, stride, secrets=st.secrets)
        st.session_state.qm_vision_model_loaded = True
        st.session_state.qm_vision_result = result
        st.session_state.pop("qm_vision_state", None)
        st.session_state.pop("qm_vision_optimizer_ran", None)
        st.session_state.pop("qm_vision_demand_choice_keep", None)
        st.session_state.pop("qm_vision_demand_choice", None)
        if st.session_state.get("qm_state_source") == "VISION":
            st.session_state.qm_state_source = "SYNTHETIC"
        progress_bar.progress(1.0, text="Preparing results · analysis complete")
        st.rerun()
    except Exception as exc:
        LOG.exception("Local vision analysis failed")
        st.error(f"This video could not be analyzed: {exc}")


def _roi_editor(key="qm_vision_roi_editor"):
    current = st.session_state.qm_vision_roi
    edited = st.data_editor(
        pd.DataFrame(current, columns=["x", "y"]), num_rows="dynamic", width="stretch",
        column_config={"x": st.column_config.NumberColumn("Horizontal position", min_value=0.0, max_value=1.0, step=.01),
                       "y": st.column_config.NumberColumn("Vertical position", min_value=0.0, max_value=1.0, step=.01)},
        key=key,
    )
    if st.button("Apply Queue Area", key=f"{key}_apply"):
        try:
            st.session_state.qm_vision_roi = validate_polygon(edited[["x", "y"]].values.tolist())
            st.success("Queue area updated for this session. Run analysis again to use it.")
        except (ValueError, TypeError) as exc:
            st.error(f"Invalid queue area: {exc}")


def _roi_setup():
    from src.vision.annotation import annotate_frame
    from src.vision.roi import classify_queue_people
    from src.vision.tracking import AnonymousTracker
    from src.vision.video_io import encode_preview

    st.markdown("#### Queue ROI · normalized camera geometry")
    st.caption("Default polygon is a calibration placeholder. Only tracked bottom-centre points inside it count as queued.")
    _roi_editor()
    for key, title in (("qm_vision_entry", "ENTRY LINE"), ("qm_vision_service", "SERVICE LINE")):
        with st.expander(title):
            line = st.session_state[key]
            columns = st.columns(4)
            values = [columns[i].number_input(label, 0.0, 1.0, float(line[i // 2][i % 2]), .01,
                                             key=f"{key}_{i}")
                      for i, label in enumerate(("Start x", "Start y", "End x", "End y"))]
            if st.button(f"APPLY {title}"):
                try:
                    st.session_state[key] = validate_line(((values[0], values[1]), (values[2], values[3])))
                    st.success(f"{title} updated for this session.")
                except ValueError as exc:
                    st.error(str(exc))
    st.caption("Events require anonymous tracks crossing the finite oriented line from its negative to positive side, with a cooldown. Reverse line endpoints to reverse direction. Camera perspective requires calibration; service-line crossings are not verified completions.")
    if not _available():
        st.info("ROI geometry can be inspected in the cloud; person-detection calibration runs locally.")
        return
    sample_index = st.number_input("Sample frame index", min_value=0, value=0, step=1)
    if st.button("TEST ROI"):
        try:
            with _source_context() as path:
                frame, metadata = read_sample_frame(path, int(sample_index))
            detector = _person_detector()
            detections = detector.detect(frame)
            tracks = AnonymousTracker().update(detections)
            queue = classify_queue_people(tracks, st.session_state.qm_vision_roi,
                                          metadata.width, metadata.height)
            annotated = annotate_frame(frame, tracks, queue["queue_track_ids"],
                                       st.session_state.qm_vision_roi, st.session_state.qm_vision_entry,
                                       st.session_state.qm_vision_service, queue["queue_count"])
            st.image(encode_preview(annotated), caption=f"Frame {sample_index} · ROI calibration preview")
            columns = st.columns(3)
            columns[0].metric("Detected", len(detections))
            columns[1].metric("Inside ROI", queue["queue_count"])
            columns[2].metric("Outside ROI", len(detections) - queue["queue_count"])
            st.caption("Single-frame IDs are temporary. Run the full clip to observe track continuity and events.")
            st.session_state.qm_vision_model_loaded = True
        except Exception as exc:
            LOG.exception("ROI test failed")
            st.error(f"ROI test unavailable: {exc}")


def _result():
    result = st.session_state.get("qm_vision_result")
    if result is None:
        st.info("No local video analysis in this session yet. Run Video Analyzer first.")
    return result


def _simple_video_result(locations):
    result = st.session_state.get("qm_vision_result")
    if result is None:
        return
    frame = result.frames[-1]
    st.markdown("#### What We Found in Your Video")
    st.caption("FROM UPLOADED VIDEO · Counts depend on the selected queue area and camera view")
    columns = st.columns(4)
    columns[0].metric("People Waiting", frame.queue_count, help="People counted inside the selected queue area in the latest sampled frame.")
    columns[1].metric("People Entering", result.summary["entry_events"], help="Tracks that crossed the selected entry line.")
    columns[2].metric("People Served", result.summary["service_events"], help="Tracks that crossed the service line; an actual service completion is not verified.")
    columns[3].metric("People Seen", frame.detected_people, help="All detected people in the latest sampled frame, including those outside the queue area.")
    view = st.radio("Video view", ("Analyzed Video", "Original Video"), horizontal=True,
                    key="qm_vision_view")
    if view == "Original Video":
        uploaded = st.session_state.get("qm_vision_upload_keep")
        if uploaded is not None and Path(uploaded.name).suffix.lower() in {".mp4", ".mov"}:
            st.video(uploaded)
        else:
            st.info("Your browser cannot preview this codec. The analyzed still frame is available below.")
    else:
        preview_index = min(result.previews, key=lambda index: abs(index - frame.frame_index)) if result.previews else None
        if preview_index is not None:
            st.image(result.previews[preview_index], caption=f"Analyzed frame {preview_index}: boxes mark Waiting or Outside Queue", width="stretch")
        if result.annotated_video is not None:
            st.video(result.annotated_video)
        elif preview_index is None:
            st.info("No annotated frame was retained for this analysis.")
    st.caption("The queue-area outline and anonymous person numbers are shown on the analyzed frame. Queue counting accuracy has not been measured on this video.")
    _forecast_integration(locations)
    with st.expander("Advanced Video Details"):
        if st.checkbox("Show frame metrics, timeline, and evaluation tools", key="qm_vision_show_advanced"):
            _live_metrics(locations[st.session_state.get("qm_vision_location", 0)])
            _timeline(locations[st.session_state.get("qm_vision_location", 0)])
            _evaluation()


def _selected_frame(result):
    index = st.slider("Analyzed frame / video time", 0, len(result.frames) - 1,
                      len(result.frames) - 1, key="qm_vision_frame_selection")
    return result.frames[index]


def _live_metrics(location):
    st.markdown("#### Local video metrics")
    result = _result()
    if result is None:
        return
    summary = result.summary
    for group in (
        (("Video frames", "total_video_frames", "count"), ("Processed", "processed_frames", "count"),
         ("Avg detected", "avg_detected_people", "float"), ("Avg queue", "avg_queue_count", "float"),
         ("Max queue", "max_queue_count", "count")),
        (("Entry crossings", "entry_events", "count"), ("Service crossings", "service_events", "count"),
         ("Avg entry rate/min", "avg_arrival_rate", "float"),
         ("Avg service rate/min", "avg_observed_service_rate", "float"),
         ("Analysis FPS", "analysis_fps", "float")),
    ):
        cols = st.columns(len(group))
        for col, (label, key, kind) in zip(cols, group):
            value = summary[key]
            col.metric(label, f"{value:,.1f}" if kind == "float" and value is not None else
                       f"{value:,}" if value is not None else "Not available")
    lifetime = (f"{summary['avg_track_lifetime_s']:.2f}s"
                if summary["avg_track_lifetime_s"] is not None else "N/A")
    confidence = (f"{summary['avg_detection_confidence']:.2f}"
                  if summary["avg_detection_confidence"] is not None else "N/A")
    st.caption(f"Video FPS {result.metadata.fps:.1f} · average anonymous track lifetime {lifetime} · queue visible in {summary['queue_visible_fraction']:.0%} of analyzed frames · mean detection confidence {confidence} (diagnostic, not accuracy).")
    frame = _selected_frame(result)
    preview_key = min(result.previews, key=lambda key: abs(key - frame.frame_index)) if result.previews else None
    if preview_key is not None:
        st.image(result.previews[preview_key], caption=f"Stored annotated preview · source frame {preview_key} (selected metrics: frame {frame.frame_index})", width="stretch")
        st.download_button("DOWNLOAD THIS ANNOTATED FRAME", result.previews[preview_key],
                           f"queuemind_frame_{preview_key}.jpg", "image/jpeg")
    cols = st.columns(6)
    for col, label, value in zip(cols,
        ("Time", "People", "Tracked", "In queue", "Entries", "Services"),
        (f"{frame.timestamp_s:.1f}s", frame.detected_people, frame.tracked_people,
         frame.queue_count, frame.entry_events, frame.service_events),
    ):
        col.metric(label, value)
    capacity = frame.arrival_rate - location["default_counters"] / location["average_service_time_minutes"]
    st.caption(f"Observed entry rate {frame.arrival_rate:.2f}/min · observed service-crossing rate {frame.service_rate:.2f}/min · configured service capacity {location['default_counters'] / location['average_service_time_minutes']:.2f}/min · queue pressure {capacity:+.2f}/min. Crossing rates are video observations; capacity is a configured assumption.")
    table = result.frames_dataframe().tail(100)
    with st.expander("FRAME-LEVEL PEOPLE DETECTION TABLE · latest 100 rows"):
        st.dataframe(table, hide_index=True, width="stretch")


def _timeline(location):
    st.markdown("#### Queue and event timelines · local video time")
    result = _result()
    if result is None:
        return
    frame = result.frames_dataframe()
    queue = go.Figure()
    queue.add_trace(go.Scatter(x=frame.timestamp_s, y=frame.queue_count,
                               mode="lines+markers", name="Queue ROI count", line_color="#173B63"))
    for kind, name, color in (("ENTRY", "Entry", "#2DA36B"),
                              ("SERVICE_EXIT", "Service crossing", "#F36F5D")):
        subset = result.events_dataframe()
        subset = subset[subset.event_type == kind]
        if not subset.empty:
            queue.add_trace(go.Scatter(x=subset.timestamp_s, y=subset.queue_count,
                                       mode="markers", name=name,
                                       marker={"size": 11, "color": color, "symbol": "diamond"}))
    queue.add_hline(y=location["warning_queue_length"], line_dash="dot",
                    line_color="#E5A128", annotation_text="Configured warning threshold")
    queue.update_layout(height=320, xaxis_title="Video time · seconds", yaxis_title="People inside ROI",
                        margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(queue, width="stretch")
    rates = go.Figure()
    rates.add_trace(go.Scatter(x=frame.timestamp_s, y=frame.arrival_rate,
                               name="Observed entry crossings/min", line_color="#F36F5D"))
    rates.add_trace(go.Scatter(x=frame.timestamp_s, y=frame.service_rate,
                               name="Observed service crossings/min", line_color="#159A82"))
    buildup = frame.arrival_rate > frame.service_rate
    rates.add_trace(go.Scatter(x=frame.timestamp_s, y=frame.arrival_rate.where(buildup),
                               fill="tozeroy", fillcolor="rgba(243,111,93,.12)",
                               line={"color": "rgba(0,0,0,0)"}, name="QUEUE BUILD-UP · observed crossings"))
    rates.update_layout(height=300, xaxis_title="Video time · seconds", yaxis_title="Crossings / min",
                        margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(rates, width="stretch")
    st.caption("A crossing-rate gap is diagnostic; it does not prove actual queue growth or service completion without camera calibration.")
    events = result.events_dataframe()
    kind = st.selectbox("Event filter", ("ALL", "ENTRY", "SERVICE_EXIT", "QUEUE_CHANGE"))
    if kind != "ALL":
        events = events[events.event_type == kind]
    if not events.empty:
        events = events.copy()
        events["description"] = events.apply(
            lambda row: (f"Track {int(row.track_id)} entered queue" if row.event_type == "ENTRY"
                         else f"Track {int(row.track_id)} crossed service line" if row.event_type == "SERVICE_EXIT"
                         else f"Queue changed to {int(row.queue_count)}"), axis=1,
        )
    st.dataframe(events.tail(100).sort_values("timestamp_s", ascending=False), hide_index=True, width="stretch")
    if events.empty:
        st.info("No matching crossings or queue changes were observed.")


def _forecast_integration(locations):
    st.markdown("#### Predict Future Queue")
    result = _result()
    if result is None:
        return
    active_state = st.session_state.get("qm_vision_state")
    if active_state is not None:
        remembered = next((item["display_name"] for item in locations
                           if item["id"] == active_state.location_id), None)
        if remembered is not None:
            st.session_state.setdefault("qm_vision_location_label", remembered)
        st.session_state.setdefault("qm_vision_active_counters", active_state.active_counters)
    names = [item["display_name"] for item in locations]
    selected_location = st.selectbox("Which kind of facility is this?", names,
                                     key="qm_vision_location_label",
                                     help="This supplies an example service time and wait target; it does not verify where the video was recorded.")
    location = locations[names.index(selected_location)]
    st.session_state.qm_vision_location = names.index(selected_location)
    st.caption("The video supplies the queue count. Facility settings supply example service speed and the target wait.")
    counters = st.slider("Open Counters", 1, 8, value=int(location["default_counters"]),
                         key="qm_vision_active_counters")
    frame = result.frames[-1]
    from src.demo.scenarios import SCENARIOS
    compatible = [name for name, settings in SCENARIOS.items()
                  if settings["location_id"] == location["id"]]
    if not compatible:
        compatible = list(SCENARIOS)
    entry_count = result.summary["entry_events"]
    measured_available = entry_count >= 2 and frame.timestamp_s >= 20
    options = ["Choose future demand"]
    if measured_available:
        options.append("Use video entry rate")
    options.extend(f"Demo assumption: {name}" for name in compatible)
    if "qm_vision_demand_choice" not in st.session_state:
        previous = st.session_state.get("qm_vision_demand_choice_keep", "Choose future demand")
        st.session_state.qm_vision_demand_choice = previous if previous in options else "Choose future demand"
    demand_choice = st.selectbox("How many people may arrive next?", options,
                                 key="qm_vision_demand_choice")
    st.session_state.qm_vision_demand_choice_keep = demand_choice
    if not measured_available:
        st.info("This short video did not show enough entry crossings to estimate future arrivals. Choose a clearly labelled demo demand assumption to continue.")
    if demand_choice == "Choose future demand" and active_state is None:
        st.caption("No future queue or wait is shown until you choose a demand source.")
    elif demand_choice == "Use video entry rate":
        st.caption(f"Measured from video crossings: {frame.arrival_rate:.2f} people arriving per minute. Future arrivals are held at this rate; no ML forecast is claimed.")
    else:
        chosen = demand_choice.removeprefix("Demo assumption: ")
        st.caption(f"VIDEO COUNT + SYNTHETIC DEMO DEMAND · {chosen} uses {SCENARIOS[chosen]['arrival_rate']:.2f} arrivals/min. This rate was not measured in the video.")
    if st.button("Predict Future Queue", type="primary", width="stretch",
                 disabled=demand_choice == "Choose future demand"):
        if demand_choice == "Use video entry rate":
            state = state_from_frame(result, frame, location, counters)
        else:
            chosen = demand_choice.removeprefix("Demo assumption: ")
            state = state_from_frame(result, frame, location, counters,
                                     forecast_arrival_rate=SCENARIOS[chosen]["arrival_rate"],
                                     demand_source="SYNTHETIC_ASSUMPTION")
        st.session_state.qm_vision_state = state
        st.session_state.qm_state_source = "VISION"
    state = st.session_state.get("qm_vision_state")
    if st.session_state.get("qm_state_source") != "VISION" or state is None:
        return
    forecast = project_vision_state(state, next(item for item in locations if item["id"] == state.location_id))
    st.caption("FROM UPLOADED VIDEO" if state.demand_source == "VIDEO_EVENTS"
               else "VIDEO COUNT + SYNTHETIC DEMO DEMAND ASSUMPTION")
    cols = st.columns(5)
    for col, point in zip(cols, forecast["points"]):
        col.metric("Now" if point["minute"] == 0 else f"+{point['minute']} min",
                   f"{point['queue']:.0f} waiting")
        col.caption(f"Estimated wait {point['wait']:.1f} min")
    st.caption("Future queue is a deterministic simulation. Wait uses the selected facility's example service time, not a measured transaction speed.")
    rec = forecast["recommendation"]
    before, after = rec["current_outcome"], rec["recommended_outcome"]
    st.markdown("#### Best Counters")
    left, right = st.columns(2)
    left.metric(f"Before · {before['active_counters']} counters", f"{before['predicted_wait_minutes']:.1f} min wait")
    right.metric(f"After · {after['active_counters']} counters", f"{after['predicted_wait_minutes']:.1f} min wait")
    st.caption("Simulator-derived +20-minute outcomes · " + rec["reason"])
    if st.button("Back to Demo", key="qm_vision_back_demo"):
        st.session_state.qm_state_source = "SYNTHETIC"
        st.rerun()


def render_shared_vision_state(locations, section):
    """Present the selected observed state across existing operations tabs."""
    state = st.session_state.get("qm_vision_state")
    if st.session_state.get("qm_state_source") != "VISION" or state is None:
        return False
    location = next((item for item in locations if item["id"] == state.location_id), None)
    if location is None:
        st.warning("The selected vision facility is no longer configured.")
        return False
    forecast = project_vision_state(state, location)
    st.info("FROM UPLOADED VIDEO · Queue count measured from frames. " +
            ("Future demand uses observed entry crossings." if state.demand_source == "VIDEO_EVENTS"
             else "Future demand uses a SYNTHETIC demo assumption, not video measurements."))
    if st.button("Back to Demo", key=f"qm_return_synthetic_{section}"):
        st.session_state.qm_state_source = "SYNTHETIC"
        st.rerun()
    st.markdown(f"### {section} · {escape(location['display_name'])}")
    current = forecast["points"][0]
    cols = st.columns(5)
    for col, label, value in zip(cols,
        ("People Waiting", "Estimated Wait", "Video Entry Crossings / Min", "Example Service Capacity", "Open Counters"),
        (f"{state.current_queue}", f"{current['wait']:.1f} min",
         f"{state.observed_arrival_rate:.2f}/min", f"{forecast['configured_service_capacity']:.2f}/min",
         state.active_counters),
    ):
        col.metric(label, value)
    st.caption(f"Observed service-crossing rate: {state.observed_service_rate:.2f}/min. It is not the configured service capacity.")
    st.caption(f"Future demand used in simulation: {forecast['points'][1]['forecast_arrival_rate']:.2f} arrivals/min · "
               + ("measured video crossings held constant" if state.demand_source == "VIDEO_EVENTS"
                  else "SYNTHETIC demo assumption, not video-derived"))
    if section in {"Home", "Predictions"}:
        points = pd.DataFrame(forecast["points"])
        mode = st.radio("Show", ("People Waiting", "Waiting Time"), horizontal=True,
                        key="qm_video_prediction_mode")
        value = "queue" if mode == "People Waiting" else "wait"
        chart = px.line(points, x="minute", y=value, markers=True,
                        title="People Waiting Over Time" if value == "queue" else "Estimated Waiting Time")
        chart.add_vline(x=0, line_dash="dot", annotation_text="NOW")
        st.plotly_chart(chart, width="stretch")
        with st.expander("Advanced Prediction Details"):
            st.dataframe(points.rename(columns={"minute": "Minutes ahead", "queue": "Expected queue",
                                               "wait": "Estimated wait · min"}), hide_index=True, width="stretch")
    if section in {"Home", "Best Counters"}:
        rec = forecast["recommendation"]
        before, after = rec["current_outcome"], rec["recommended_outcome"]
        st.markdown("#### How many counters should be open?")
        left, right = st.columns(2)
        for col, label, outcome in ((left, "Current", before), (right, "Recommended", after)):
            col.metric(f"{label} Counters", outcome["active_counters"])
            col.metric("Queue in 20 Min", f"{outcome['future_queue']:.1f}")
            col.metric("Estimated Wait", f"{outcome['predicted_wait_minutes']:.1f} min")
        st.caption(rec["reason"])
        st.caption("These are simulator results using facility settings and the stated demand source, not measured effects of changing staff.")
    return True


def _evaluation():
    st.markdown("#### Manually labelled queue-count evaluation")
    st.caption("Only user-supplied ground truth is scored. Source provenance is not verified, and no vision accuracy is claimed without labels.")
    st.download_button("DOWNLOAD EMPTY VISION LABEL TEMPLATE", TEMPLATE_CSV,
                       "vision_labels_template.csv", "text/csv")
    result = st.session_state.get("qm_vision_result")
    if result is not None:
        count = st.select_slider("Evenly distributed frames for manual labelling", options=[30, 50, 100], value=30)
        sample = result.labelling_sample(count)
        st.download_button("EXPORT SAMPLE FOR MANUAL LABELLING", sample.to_csv(index=False),
                           "vision_predictions_for_labelling.csv", "text/csv")
        st.caption("Predicted queue counts come from this local analysis. Fill ground_truth_queue_count yourself, then upload the labelled CSV below.")
        st.download_button("DOWNLOAD FRAME METRICS CSV", result.frames_dataframe().to_csv(index=False),
                           "vision_frame_metrics.csv", "text/csv")
        st.download_button("DOWNLOAD EVENT CSV", result.events_dataframe().to_csv(index=False),
                           "vision_events.csv", "text/csv")
        if result.annotated_video is not None:
            st.download_button("DOWNLOAD ANNOTATED VIDEO", result.annotated_video,
                               "queuemind_annotated.mp4", "video/mp4")
    version = st.session_state.get("qm_vision_upload_version", 0)
    uploaded = st.file_uploader("Labelled CSV", type="csv", key=f"qm_vision_upload_{version}")
    if uploaded is not None:
        try:
            metrics, frame = evaluate_vision_labels(pd.read_csv(uploaded))
        except (ValueError, pd.errors.ParserError) as exc:
            st.error(f"Unable to evaluate labels: {exc}")
        else:
            st.session_state.qm_vision_evaluation = metrics
            st.session_state.qm_vision_frame = frame
    metrics, frame = st.session_state.get("qm_vision_evaluation"), st.session_state.get("qm_vision_frame")
    if metrics is None or frame is None:
        st.info("Real-video quantitative evaluation pending. Upload manually labelled frame counts to calculate it.")
        return
    cols = st.columns(4)
    for col, label, value in zip(cols,
        ("Labelled frames", "Queue Count MAE", "RMSE", "Exact count"),
        (metrics["labelled_frames"], f'{metrics["queue_count_mae"]:.3f}',
         f'{metrics["queue_count_rmse"]:.3f}', f'{metrics["exact_count_accuracy"]:.1%}'),
    ):
        col.metric(label, value)
    st.caption(f'Within ±1 person: {metrics["within_one_person_accuracy"]:.1%} · signed error: {metrics["mean_signed_error"]:+.3f} · maximum absolute error: {metrics["max_absolute_error"]}.')
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=frame.frame_id, y=frame.ground_truth_queue_count,
                                name="Ground truth", line_color="#173B63"))
    figure.add_trace(go.Scatter(x=frame.frame_id, y=frame.predicted_queue_count,
                                name="Predicted", line_color="#159A82"))
    figure.update_layout(height=300, title="Ground truth vs predicted queue count")
    st.plotly_chart(figure, width="stretch")
    a, b = st.columns(2)
    a.plotly_chart(px.histogram(frame, x="signed_error", title="Count error distribution"), width="stretch")
    b.plotly_chart(px.bar(frame, x="frame_id", y="absolute_error", title="Per-frame absolute error"), width="stretch")
    st.dataframe(frame, hide_index=True, width="stretch")
    if st.button("CLEAR UPLOADED LABELS"):
        for key in ("qm_vision_evaluation", "qm_vision_frame"):
            st.session_state.pop(key, None)
        st.session_state.qm_vision_upload_version = version + 1
        st.rerun()


def _privacy():
    st.markdown("#### Pipeline / Privacy")
    st.markdown('<div class="qm-panel blue"><div class="qm-label">QUEUE MIND VISION</div>'
                '<h3>CCTV → anonymous queue intelligence</h3>'
                '<div class="qm-flow"><span>VIDEO</span><span>PERSON DETECTION</span>'
                '<span>ANONYMOUS TRACKING</span><span>QUEUE ROI</span>'
                '<span>EVENTS</span><span>QUEUE STATE</span><span>FORECAST</span>'
                '<span>ACTION</span></div></div>', unsafe_allow_html=True)
    st.info("PRIVACY DESIGN · Person class only · temporary anonymous track IDs · no face recognition · no biometric or student identification · upload held only in this browser session for calibration and cleared on Reset; temporary disk copies are deleted after each operation · aggregate queue statistics preferred.")
    st.caption("Anonymous IoU tracking may lose or reassign IDs under heavy occlusion. Virtual-line crossings require camera calibration and are not verified person identities or service completions. Real campus CCTV requires institutional permission.")
    st.code('python -m pip install -r requirements-vision.txt\npython -m streamlit run app.py\npython -m scripts.vision_demo "C:\\path\\to\\queue_video.mp4"\npython -m scripts.vision_demo 0', language="powershell")


def render_video_upload(locations, *, home=False):
    """One upload/analyzer implementation shared by Home and Upload Video."""
    _defaults()
    if not home:
        from src.ui.feature_help import feature_heading
        feature_heading("Upload Video", "Upload → choose queue area → analyze → see results")
    _video_analyzer(locations, home=home)


def render_vision_evaluation():
    _defaults()
    _evaluation()


def render_vision_lab():
    """Render the existing Vision Lab as an optional local-analysis workspace."""
    _defaults()
    st.markdown("### QueueMind Vision · local video analysis")
    st.caption("VIDEO → PERSON DETECTION → ANONYMOUS TRACKING → QUEUE ROI → EVENTS → QUEUE STATE → FORECAST → ACTION")
    _runtime_status()
    locations = get_enabled_locations()
    tabs = st.tabs(SECTIONS, key="vision_sections", bind="query-params")
    for name, tab in zip(SECTIONS, tabs):
        if tab.open:
            with tab:
                if name == "Video Analyzer":
                    _video_analyzer(locations)
                elif name == "ROI Setup":
                    _roi_setup()
                elif name == "Live Metrics":
                    result = st.session_state.get("qm_vision_result")
                    location = locations[st.session_state.get("qm_vision_location", 0)]
                    _live_metrics(location)
                elif name == "Timeline":
                    location = locations[st.session_state.get("qm_vision_location", 0)]
                    _timeline(location)
                elif name == "Forecast Integration":
                    _forecast_integration(locations)
                elif name == "Evaluation":
                    _evaluation()
                else:
                    _privacy()
