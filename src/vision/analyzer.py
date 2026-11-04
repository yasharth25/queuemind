"""Streaming, session-scoped local video analysis built from existing vision modules."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from time import monotonic
from typing import Callable, Iterable

import pandas as pd

from src.vision.annotation import annotate_frame
from src.vision.config import validate_line, validate_polygon
from src.vision.pipeline import VisionQueuePipeline
from src.vision.tracking import AnonymousTracker
from src.vision.video_io import VideoMetadata, encode_preview, open_video, require_opencv


@dataclass(frozen=True)
class VisionFrameResult:
    frame_index: int
    timestamp_s: float
    detected_people: int
    tracked_people: int
    queue_count: int
    entry_events: int
    service_events: int
    arrival_rate: float
    service_rate: float
    mean_confidence: float | None


@dataclass(frozen=True)
class VisionEventRecord:
    frame_index: int
    timestamp_s: float
    track_id: int | None
    event_type: str
    queue_count: int


@dataclass
class VisionAnalysisResult:
    metadata: VideoMetadata
    frames: list[VisionFrameResult]
    events: list[VisionEventRecord]
    previews: dict[int, bytes] = field(default_factory=dict)
    elapsed_s: float = 0.0
    track_lifetimes_s: dict[int, float] = field(default_factory=dict)
    annotated_video: bytes | None = None

    @property
    def summary(self) -> dict:
        if not self.frames:
            raise ValueError("No readable video frames were analyzed.")
        count = len(self.frames)
        return {
            "total_video_frames": self.metadata.total_frames,
            "processed_frames": count,
            "avg_detected_people": sum(row.detected_people for row in self.frames) / count,
            "avg_queue_count": sum(row.queue_count for row in self.frames) / count,
            "max_queue_count": max(row.queue_count for row in self.frames),
            "entry_events": sum(row.entry_events for row in self.frames),
            "service_events": sum(row.service_events for row in self.frames),
            "avg_arrival_rate": sum(row.arrival_rate for row in self.frames) / count,
            "avg_observed_service_rate": sum(row.service_rate for row in self.frames) / count,
            "avg_track_lifetime_s": (
                sum(self.track_lifetimes_s.values()) / len(self.track_lifetimes_s)
                if self.track_lifetimes_s else None
            ),
            "queue_visible_fraction": sum(row.queue_count > 0 for row in self.frames) / count,
            "avg_detection_confidence": (
                sum(row.mean_confidence for row in self.frames if row.mean_confidence is not None)
                / sum(row.mean_confidence is not None for row in self.frames)
                if any(row.mean_confidence is not None for row in self.frames) else None
            ),
            "analysis_fps": count / self.elapsed_s if self.elapsed_s > 0 else None,
        }

    def frames_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(row) for row in self.frames])

    def events_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(row) for row in self.events])

    def labelling_sample(self, count: int) -> pd.DataFrame:
        if count < 1:
            raise ValueError("Sample size must be positive.")
        total = len(self.frames)
        indices = sorted({round(index * (total - 1) / (min(count, total) - 1))
                          for index in range(min(count, total))}) if total > 1 and count > 1 else [total - 1]
        return pd.DataFrame([{
            "frame_id": str(self.frames[index].frame_index),
            "timestamp_s": self.frames[index].timestamp_s,
            "predicted_queue_count": self.frames[index].queue_count,
            "ground_truth_queue_count": "",
        } for index in indices])


def rolling_event_rate(event_times: list[float], timestamp_s: float, window_s: int) -> float:
    """Observed events/min over a fixed trailing video-time window."""
    if window_s <= 0 or timestamp_s < 0:
        raise ValueError("Window and video timestamp must be positive/nonnegative.")
    return sum(timestamp_s - window_s < event <= timestamp_s for event in event_times) * 60 / window_s


def analyze_frames(
    frames: Iterable[tuple[int, object]], metadata: VideoMetadata, pipeline: VisionQueuePipeline,
    roi, entry_line, service_line, *, stride: int = 1, rate_window_s: int = 60,
    max_previews: int = 100, preview_encoder: Callable | None = None,
    progress: Callable[[int, int, float], None] | None = None,
    writer=None, cancel: Callable[[], bool] | None = None,
) -> VisionAnalysisResult:
    """Analyze a stream without retaining raw frames; dependencies are injectable for tests."""
    roi = validate_polygon(roi)
    entry_line, service_line = validate_line(entry_line), validate_line(service_line)
    if stride < 1 or rate_window_s < 1 or max_previews < 0:
        raise ValueError("Stride/window must be positive and preview limit nonnegative.")
    result = VisionAnalysisResult(metadata, [], [])
    entry_times: list[float] = []
    service_times: list[float] = []
    track_counts: dict[int, int] = {}
    start = monotonic()
    preview_step = max(1, (metadata.total_frames + stride * max_previews - 1) // (stride * max_previews)) if max_previews and metadata.total_frames else 1
    previous_queue: int | None = None
    for frame_index, frame in frames:
        if cancel and cancel():
            break
        if frame_index % stride:
            continue
        timestamp_s = frame_index / metadata.fps
        # This is a relative video clock, not a claim about original CCTV wall time.
        timestamp = datetime(2000, 1, 1) + timedelta(seconds=timestamp_s)
        output = pipeline.process_frame(frame, timestamp)
        if not output["processed"]:
            continue
        detections, tracks = output["detections"], output["tracks"]
        queue = output["queue"]
        for person in tracks:
            track_counts[person.track_id] = track_counts.get(person.track_id, 0) + 1
        entries = services = 0
        for event in output["events"]:
            if event.event_type == "ENTRY":
                entry_times.append(timestamp_s)
                entries += 1
            elif event.event_type == "SERVICE_EXIT":
                service_times.append(timestamp_s)
                services += 1
            result.events.append(VisionEventRecord(frame_index, timestamp_s, event.track_id,
                                                   event.event_type, queue["queue_count"]))
        if previous_queue is not None and queue["queue_count"] != previous_queue:
            result.events.append(VisionEventRecord(frame_index, timestamp_s, None,
                                                   "QUEUE_CHANGE", queue["queue_count"]))
        previous_queue = queue["queue_count"]
        row = VisionFrameResult(
            frame_index, timestamp_s, len(detections), len(tracks), queue["queue_count"],
            entries, services,
            rolling_event_rate(entry_times, timestamp_s, rate_window_s),
            rolling_event_rate(service_times, timestamp_s, rate_window_s),
            sum(item.confidence for item in detections) / len(detections) if detections else None,
        )
        result.frames.append(row)
        wants_preview = (
            preview_encoder is not None and len(result.previews) < max_previews
            and (len(result.frames) - 1) % preview_step == 0
        )
        if wants_preview or writer is not None:
            annotated = annotate_frame(frame, tracks, queue["queue_track_ids"],
                                       roi, entry_line, service_line, queue["queue_count"],
                                       entries, services)
            if wants_preview:
                result.previews[frame_index] = preview_encoder(annotated)
            if writer is not None:
                writer.write(annotated)
        if progress and (len(result.frames) % 10 == 0 or frame_index == metadata.total_frames - 1):
            progress(frame_index + 1, metadata.total_frames, monotonic() - start)
    if not result.frames:
        raise ValueError("Video contains no readable frames to analyze.")
    result.elapsed_s = monotonic() - start
    result.track_lifetimes_s = {ident: count * stride / metadata.fps
                                for ident, count in track_counts.items()}
    if progress:
        progress(result.frames[-1].frame_index + 1, metadata.total_frames, result.elapsed_s)
    return result


def analyze_video(
    path: str | Path, detector, location: dict, roi, entry_line, service_line,
    *, stride: int = 1, rate_window_s: int = 60, max_previews: int = 100,
    iou_threshold: float = .3, annotated_output: str | Path | None = None,
    progress: Callable[[int, int, float], None] | None = None,
) -> VisionAnalysisResult:
    """Open a local prerecorded video and process it frame-by-frame."""
    capture, metadata = open_video(path)
    cv2 = require_opencv()
    writer = None
    try:
        if stride < 1:
            raise ValueError("Frame stride must be positive.")
        if annotated_output is not None:
            writer = cv2.VideoWriter(str(annotated_output), cv2.VideoWriter_fourcc(*"mp4v"),
                                     metadata.fps / stride, (metadata.width, metadata.height))
            if not writer.isOpened():
                raise RuntimeError("Annotated MP4 encoding failed on this machine.")
        pipeline = VisionQueuePipeline(
            detector, AnonymousTracker(iou_threshold=iou_threshold),
            validate_polygon(roi), location,
            _pixel_line(validate_line(entry_line), metadata),
            _pixel_line(validate_line(service_line), metadata),
        )
        def frame_stream():
            index = 0
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                yield index, frame
                index += 1
        return analyze_frames(
            frame_stream(), metadata, pipeline, roi, entry_line, service_line,
            stride=stride, rate_window_s=rate_window_s, max_previews=max_previews,
            preview_encoder=encode_preview, progress=progress, writer=writer,
        )
    finally:
        capture.release()
        if writer is not None:
            writer.release()


def _pixel_line(line, metadata):
    return tuple((x * metadata.width, y * metadata.height) for x, y in line)
