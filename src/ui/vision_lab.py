"""Optional local video workspace inside QueueMind's existing Vision Lab tab."""

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
from src.vision.video_io import open_video, read_sample_frame, uploaded_video_file

LOG = logging.getLogger(__name__)
SECTIONS = (
    "Video Analyzer", "ROI Setup", "Live Metrics", "Timeline",
    "Forecast Integration", "Evaluation", "Pipeline / Privacy",
)


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


def _available() -> bool:
    return all(importlib.util.find_spec(name) is not None for name in ("cv2", "ultralytics"))


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
    st.caption("Cloud video inference: DISABLED by optional dependencies. Runtime checks do not load CV libraries or weights.")
    if not _available():
        st.info("LOCAL VISION MODE · CCTV/video inference uses optional local dependencies and is intentionally disabled on the public cloud demo. CSV evaluation remains available.")
        st.code("python -m pip install -r requirements-vision.txt\npython -m streamlit run app.py", language="powershell")


def _video_metadata():
    with _source_context() as path:
        capture, metadata = open_video(path)
        capture.release()
    return metadata


def _video_analyzer(locations):
    st.markdown("#### Video Analyzer · local prerecorded queue video")
    st.caption("PERSON DETECTION → ANONYMOUS TRACKING → QUEUE ROI → EVENTS → QUEUE STATE")
    if not _available():
        st.info("This section is local-only. Install the optional vision requirements; no webcam or YOLO weights are required for the cloud demo.")
        return
    upload_col, path_col = st.columns([1.2, 1])
    with upload_col:
        current_upload = st.file_uploader(
            "UPLOAD CCTV / QUEUE VIDEO", type=["mp4", "avi", "mov", "mkv"],
            key=f'qm_vision_video_upload_{st.session_state.get("qm_video_upload_version", 0)}',
        )
        if current_upload is not None:
            st.session_state.qm_vision_upload_keep = current_upload
        uploaded = current_upload or st.session_state.get("qm_vision_upload_keep")
    with path_col:
        local_path = st.text_input(
            "Local video path (optional)",
            value=st.session_state.get("qm_vision_local_path_keep", ""),
            key=f'qm_vision_local_path_{st.session_state.get("qm_video_upload_version", 0)}',
            help="Used only if no video is uploaded. The path is read on this server; remote URLs and RTSP are unsupported.",
        )
        st.session_state.qm_vision_local_path_keep = local_path.strip()
    if uploaded is not None:
        st.caption(f"Uploaded: {uploaded.name} · {uploaded.size / 1024**2:.2f} MB · held only in this browser session for ROI calibration; no disk copy persists")
        if Path(uploaded.name).suffix.lower() in {".mp4", ".mov"}:
            st.video(uploaded)
        else:
            st.info("Browser preview may be unavailable for AVI/MKV; OpenCV can still analyze supported codecs.")
    elif st.session_state.get("qm_vision_local_path_keep", "").strip():
        path = Path(st.session_state.qm_vision_local_path_keep)
        if path.is_file() and path.suffix.lower() in {".mp4", ".mov"}:
            st.video(str(path))
    else:
        st.info("Upload MP4/AVI/MOV/MKV or enter a path on this local machine. A short prerecorded clip is best for calibration.")
        return
    try:
        metadata = _video_metadata()
    except (OSError, ValueError, RuntimeError) as exc:
        st.error(f"Video preview unavailable: {exc}")
        return
    fields = st.columns(5)
    for column, label, value in zip(fields,
        ("Resolution", "Video FPS", "Frames", "Duration", "File size"),
        (f"{metadata.width} × {metadata.height}", f"{metadata.fps:.1f}",
         f"{metadata.total_frames:,}" if metadata.total_frames else "Unknown",
         f"{metadata.duration_s:.1f} s" if metadata.duration_s is not None else "Unknown",
         f"{metadata.size_bytes / 1024**2:.2f} MB" if metadata.size_bytes else "Unknown"),
    ):
        column.metric(label, value)
    try:
        with _source_context() as path:
            sample, _ = read_sample_frame(path, 0)
        st.image(sample, channels="BGR", caption="Decoded first frame · fallback preview when browser video codecs are unavailable", width=480)
    except (OSError, ValueError, RuntimeError) as exc:
        st.warning(f"Still-frame preview unavailable: {exc}")
    mode = st.radio("Processing mode", ("QUICK ANALYSIS", "FULL ANALYSIS"), horizontal=True,
                    help="Quick mode skips frames and may miss brief boundary crossings.")
    stride = 1 if mode == "FULL ANALYSIS" else st.select_slider(
        "Frame stride", options=[1, 2, 3, 5, 10], value=3,
    )
    st.caption("Higher stride is faster but can reduce anonymous-track continuity and entry/service event precision.")
    with st.expander("ADVANCED VISION SETTINGS"):
        st.slider("Person confidence", .10, .90, step=.05, key="qm_vision_confidence")
        st.text_input("YOLO model path/name", key="qm_vision_model_path",
                      help="Default yolo11n.pt may download once on the local machine if absent. No model is loaded in the cloud page.")
        st.select_slider("Rolling event-rate window · seconds", options=[30, 60, 120], key="qm_vision_rate_window")
        iou = st.slider("Anonymous tracker IoU threshold", .10, .80, .30, .05)
        max_previews = st.slider("Maximum stored annotated previews", 10, 150, 80, 10)
        save_video = st.checkbox("GENERATE ANNOTATED VIDEO", value=False,
                                 help="Optional MP4 encoding costs time and memory; OFF by default.")
    st.caption("Queue ROI and virtual lines can be calibrated in ROI Setup before analysis. No face/identity processing.")
    start, reset = st.columns(2)
    if reset.button("RESET ANALYSIS", width="stretch"):
        for key in ("qm_vision_result", "qm_vision_state", "qm_vision_frame_selection",
                    "qm_vision_optimizer_ran", "qm_vision_upload_keep",
                    "qm_vision_local_path_keep"):
            st.session_state.pop(key, None)
        st.session_state.qm_video_upload_version = st.session_state.get("qm_video_upload_version", 0) + 1
        if st.session_state.get("qm_state_source") == "VISION":
            st.session_state.qm_state_source = "SYNTHETIC"
        st.rerun()
    if not start.button("START ANALYSIS", type="primary", width="stretch"):
        st.info("Use TEST ROI in ROI Setup first, then start analysis. Streamlit's Stop control can interrupt a long run.")
        return
    progress_bar = st.progress(0, text="Preparing local video analysis…")
    status = st.empty()
    def progress(index, total, elapsed):
        fraction = min(1.0, index / total) if total else 0.0
        progress_bar.progress(fraction, text=f"Frame {index:,} / {total:,}" if total else f"Frame {index:,}")
        processed = max(1, (index + stride - 1) // stride)
        status.caption(f"Elapsed {elapsed:.1f}s · analysis {processed / max(elapsed, .001):.1f} FPS · video {metadata.fps:.1f} FPS")
    try:
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
        st.session_state.qm_vision_model_loaded = True
        st.session_state.qm_vision_result = result
        st.session_state.pop("qm_vision_state", None)
        st.session_state.pop("qm_vision_optimizer_ran", None)
        if st.session_state.get("qm_state_source") == "VISION":
            st.session_state.qm_state_source = "SYNTHETIC"
        progress_bar.progress(1.0, text="Analysis complete")
        st.success(f"Analyzed {len(result.frames):,} frames · {len(result.previews)} bounded preview frames retained in this session.")
    except Exception as exc:
        LOG.exception("Local vision analysis failed")
        st.error(f"Local video analysis could not finish: {exc}")


def _roi_setup():
    from src.vision.annotation import annotate_frame
    from src.vision.roi import classify_queue_people
    from src.vision.tracking import AnonymousTracker
    from src.vision.video_io import encode_preview

    st.markdown("#### Queue ROI · normalized camera geometry")
    st.caption("Default polygon is a calibration placeholder. Only tracked bottom-centre points inside it count as queued.")
    current = st.session_state.qm_vision_roi
    st.code(str(list(current)))
    edited = st.data_editor(
        pd.DataFrame(current, columns=["x", "y"]), num_rows="dynamic", width="stretch",
        column_config={"x": st.column_config.NumberColumn("Normalized x", min_value=0.0, max_value=1.0, step=.01),
                       "y": st.column_config.NumberColumn("Normalized y", min_value=0.0, max_value=1.0, step=.01)},
        key="qm_vision_roi_editor",
    )
    if st.button("APPLY QUEUE ROI"):
        try:
            st.session_state.qm_vision_roi = validate_polygon(edited[["x", "y"]].values.tolist())
            st.success("Queue ROI updated for this session.")
        except (ValueError, TypeError) as exc:
            st.error(f"Invalid ROI: {exc}")
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
    st.markdown("#### Use local video state in QueueMind")
    result = _result()
    if result is None:
        return
    names = [item["display_name"] for item in locations]
    selected_location = st.selectbox("Facility profile for configured service capacity", names,
                                     key="qm_vision_location_label")
    location = locations[names.index(selected_location)]
    st.session_state.qm_vision_location = names.index(selected_location)
    st.caption("Facility profile supplies service time and target wait; the video supplies queue count and crossing rates. It does not establish where the video was recorded.")
    counters = st.slider("Active counters", 1, 8, value=int(location["default_counters"]),
                         key="qm_vision_active_counters")
    choice = st.radio("Observation", ("LATEST ANALYZED STATE", "CURRENT ANALYZED FRAME"), horizontal=True)
    frame = result.frames[-1] if choice.startswith("LATEST") else _selected_frame(result)
    state = state_from_frame(result, frame, location, counters)
    forecast = project_vision_state(state, location)
    st.markdown('<div class="qm-panel blue"><div class="qm-label">SOURCE: LOCAL VIDEO ANALYSIS</div>'
                f'<b>Frame {frame.frame_index} · {frame.timestamp_s:.1f}s</b> · '
                f'{frame.queue_count} queued · {frame.arrival_rate:.2f} entry crossings/min · '
                f'{frame.service_rate:.2f} service crossings/min</div>', unsafe_allow_html=True)
    st.caption("DETERMINISTIC QUEUE PROJECTION · observed trailing entry rate held constant. No ML arrival forecast is claimed for this short video session.")
    cols = st.columns(4)
    for col, point in zip(cols, forecast["points"][1:]):
        col.metric(f"+{point['minute']} min queue", f"{point['queue']:.1f}")
        col.caption(f"Wait {point['wait']:.1f} min · {point['congestion'].upper()}")
    projected = pd.DataFrame(forecast["points"])
    figure = go.Figure(go.Scatter(x=projected.minute, y=projected.queue, mode="lines+markers",
                                  line={"color": "#F36F5D", "width": 3}, name="Deterministic projection"))
    figure.update_layout(height=250, xaxis_title="Minutes after selected video frame",
                         yaxis_title="Expected queue · people", margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(figure, width="stretch")
    if st.button("RUN COUNTER OPTIMIZER", type="secondary", width="stretch"):
        st.session_state.qm_vision_optimizer_ran = True
    if st.session_state.get("qm_vision_optimizer_ran"):
        st.markdown("#### Counter optimizer · +20 minutes")
    recommendation = forecast["recommendation"]
    if st.session_state.get("qm_vision_optimizer_ran"):
        left, right = st.columns(2)
        for column, title, outcome in (
            (left, "Current", recommendation["current_outcome"]),
            (right, "Recommended", recommendation["recommended_outcome"]),
        ):
            column.markdown(f"**{title}: {outcome['active_counters']} counters**")
            column.metric("Expected +20 queue", f"{outcome['future_queue']:.1f}")
            column.metric("Expected +20 wait", f"{outcome['predicted_wait_minutes']:.1f} min")
        st.caption(recommendation["reason"])
    st.caption(f"Configured service capacity {forecast['configured_service_capacity']:.2f}/min · observed service crossings {forecast['observed_service_completion_rate']:.2f}/min. These are distinct quantities.")
    if st.button("USE THIS VISION STATE", type="primary", width="stretch"):
        st.session_state.qm_vision_state = state
        st.session_state.qm_state_source = "VISION"
        st.rerun()
    if st.session_state.get("qm_state_source") == "VISION":
        if st.button("RETURN TO SYNTHETIC DEMO"):
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
    st.info("SOURCE: LOCAL VIDEO ANALYSIS · observed video queue and crossings; future queue is a DETERMINISTIC PROJECTION, not an ML forecast.")
    if st.button("RETURN TO SYNTHETIC DEMO", key=f"qm_return_synthetic_{section}"):
        st.session_state.qm_state_source = "SYNTHETIC"
        st.rerun()
    st.markdown(f"### {section} · {escape(location['display_name'])}")
    current = forecast["points"][0]
    cols = st.columns(5)
    for col, label, value in zip(cols,
        ("Current queue", "Estimated wait", "Entry rate", "Configured capacity", "Counters"),
        (f"{state.current_queue}", f"{current['wait']:.1f} min",
         f"{state.observed_arrival_rate:.2f}/min", f"{forecast['configured_service_capacity']:.2f}/min",
         state.active_counters),
    ):
        col.metric(label, value)
    st.caption(f"Observed service-crossing rate: {state.observed_service_rate:.2f}/min. It is not the configured service capacity.")
    if section in {"Overview", "Forecast"}:
        points = pd.DataFrame(forecast["points"])
        st.plotly_chart(px.line(points, x="minute", y="queue", markers=True,
                                title="Observed queue → deterministic future projection"), width="stretch")
        st.dataframe(points.rename(columns={"minute": "Horizon · min", "queue": "Expected queue",
                                           "wait": "Estimated wait · min"}), hide_index=True, width="stretch")
    if section in {"Overview", "Optimizer"}:
        rec = forecast["recommendation"]
        before, after = rec["current_outcome"], rec["recommended_outcome"]
        st.markdown("#### Counter recommendation · existing simulator")
        left, right = st.columns(2)
        for col, label, outcome in ((left, "Current", before), (right, "Recommended", after)):
            col.metric(f"{label} counters", outcome["active_counters"])
            col.metric("+20 queue", f"{outcome['future_queue']:.1f}")
            col.metric("+20 wait", f"{outcome['predicted_wait_minutes']:.1f} min")
        st.caption(rec["reason"])
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
