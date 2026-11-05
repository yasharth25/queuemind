"""Bounded local frame capture and aggregate-only continuous queue monitoring."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from queue import Empty, Full, Queue
from threading import Event, Lock, Thread
from time import monotonic
from uuid import uuid4

from src.queueing.simulator import calculate_service_rate
from src.vision.live.camera_sources import CameraConnection, CameraStatus, SourceType


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class LiveObservation:
    observation_id: str
    camera_id: str
    location_id: str
    captured_at: datetime
    processed_at: datetime
    source_type: SourceType
    people_detected: int
    people_waiting: int
    entries_in_window: int
    service_boundary_crossings: int
    observed_arrival_rate: float | None
    service_boundary_rate: float | None
    observed_completion_rate: float | None
    configured_service_capacity: float
    analysis_fps: float | None
    processing_latency_s: float

    def __post_init__(self):
        if self.captured_at.tzinfo is None or self.processed_at.tzinfo is None:
            raise ValueError("Observation timestamps must include a timezone.")
        if self.processed_at < self.captured_at:
            raise ValueError("Processing cannot precede capture.")
        if any(value < 0 for value in (self.people_detected, self.people_waiting,
                                      self.entries_in_window, self.service_boundary_crossings,
                                      self.configured_service_capacity, self.processing_latency_s)):
            raise ValueError("Observation counts, rates, and latency cannot be negative.")
        if self.people_waiting > self.people_detected:
            raise ValueError("Queue count cannot exceed detected people.")
        for value in (self.observed_arrival_rate, self.service_boundary_rate,
                      self.observed_completion_rate, self.analysis_fps):
            if value is not None and value < 0:
                raise ValueError("Observed rates and analysis FPS cannot be negative.")

    def is_stale(self, *, now: datetime | None = None, threshold_s: float = 15) -> bool:
        if threshold_s <= 0:
            raise ValueError("Freshness threshold must be positive.")
        return ((now or utc_now()) - self.captured_at).total_seconds() > threshold_s


class QueueMonitor:
    """Separate capture and inference; keep only newest frame and bounded history."""

    def __init__(self, camera_id: str, connection: CameraConnection, pipeline,
                 location: dict, *, active_counters: int, service_time_minutes: float,
                 rate_window_s: int = 60, max_history: int = 300,
                 min_inference_interval_s: float = .5, preview_encoder=None,
                 annotator=None, replay_fps: float | None = None):
        if not camera_id or location["id"] != connection.settings.location_id:
            raise ValueError("Camera ID and matching location are required.")
        if rate_window_s < 1 or max_history < 1 or min_inference_interval_s < 0:
            raise ValueError("Monitoring intervals and history limit are invalid.")
        if replay_fps is not None and replay_fps <= 0:
            raise ValueError("Replay FPS must be positive.")
        self.camera_id, self.connection, self.pipeline = camera_id, connection, pipeline
        self.location = location
        self.capacity = calculate_service_rate(active_counters, service_time_minutes)
        self.rate_window_s = rate_window_s
        self.min_inference_interval_s = min_inference_interval_s
        self.replay_fps = replay_fps
        self.preview_encoder, self.annotator = preview_encoder, annotator
        self.history: deque[LiveObservation] = deque(maxlen=max_history)
        self._entries: deque[datetime] = deque()
        self._services: deque[datetime] = deque()
        self._frames: Queue = Queue(maxsize=1)
        self._stop = Event()
        self._capture_done = Event()
        self._lock = Lock()
        self._threads: list[Thread] = []
        self._started_at: datetime | None = None
        self._last_analysis_clock: float | None = None
        self.preview: bytes | None = None
        self.preview_captured_at: datetime | None = None
        self.status = CameraStatus.DISCONNECTED
        self.last_error: str | None = None
        self.dropped_frames = 0

    def process_frame(self, frame, captured_at: datetime | None = None) -> LiveObservation | None:
        """Deterministic processing seam for replay/fake-frame tests."""
        captured_at = captured_at or utc_now()
        if captured_at.tzinfo is None:
            raise ValueError("Capture timestamp must include a timezone.")
        if self._started_at is None:
            self._started_at = captured_at
        clock_start = monotonic()
        output = self.pipeline.process_frame(frame, captured_at)
        if not output["processed"]:
            return None
        entry_count = sum(event.event_type == "ENTRY" for event in output["events"])
        service_count = sum(event.event_type == "SERVICE_EXIT" for event in output["events"])
        self._entries.extend([captured_at] * entry_count)
        self._services.extend([captured_at] * service_count)
        from datetime import timedelta
        start = captured_at - timedelta(seconds=self.rate_window_s)
        while self._entries and self._entries[0] <= start:
            self._entries.popleft()
        while self._services and self._services[0] <= start:
            self._services.popleft()
        window_covered = (captured_at - self._started_at).total_seconds() >= self.rate_window_s
        rate = lambda events: len(events) * 60 / self.rate_window_s if window_covered else None
        processed_at = utc_now()
        elapsed = monotonic() - clock_start
        analysis_fps = (1 / (clock_start - self._last_analysis_clock)
                        if self._last_analysis_clock is not None and clock_start > self._last_analysis_clock
                        else None)
        self._last_analysis_clock = clock_start
        observation = LiveObservation(
            str(uuid4()), self.camera_id, self.location["id"], captured_at,
            max(captured_at, processed_at), self.connection.settings.source_type,
            len(output["detections"]), output["queue"]["queue_count"],
            entry_count, service_count, rate(self._entries), rate(self._services),
            None, self.capacity, analysis_fps, elapsed,
        )
        preview = None
        if self.preview_encoder is not None and self.annotator is not None:
            annotated = self.annotator(
                frame, output["tracks"], output["queue"]["queue_track_ids"],
                self.pipeline.roi, self.pipeline.entry_line,
                self.pipeline.service_line, observation.people_waiting,
                entry_count, service_count,
            )
            preview = self.preview_encoder(annotated)
        with self._lock:
            self.history.append(observation)
            self.preview = preview
            self.preview_captured_at = captured_at if preview is not None else None
            self.status = CameraStatus.PROCESSING
        return observation

    def _offer_latest(self, captured_at, frame):
        try:
            self._frames.put_nowait((captured_at, frame))
        except Full:
            self._frames.get_nowait()
            self.dropped_frames += 1
            self._frames.put_nowait((captured_at, frame))

    def _capture_loop(self):
        backoff = 1.0
        while not self._stop.is_set():
            try:
                self.status = CameraStatus.CONNECTING if backoff == 1 else CameraStatus.RECONNECTING
                first = self.connection.connect()
                self._offer_latest(utc_now(), first)
                self.status = CameraStatus.CONNECTED
                backoff = 1.0
                while not self._stop.is_set():
                    frame = self.connection.read()
                    self._offer_latest(utc_now(), frame)
                    if self.connection.settings.source_type == SourceType.REPLAY_VIDEO and self.replay_fps:
                        self._stop.wait(1 / self.replay_fps)
            except Exception:
                self.connection.stop()
                if self.connection.settings.source_type == SourceType.REPLAY_VIDEO:
                    self._capture_done.set()
                    break
                self.status = CameraStatus.RECONNECTING
                self.last_error = "Camera stream lost. Reconnecting locally."
                self._stop.wait(backoff)
                backoff = min(backoff * 2, 30)
        self.connection.stop()

    def _inference_loop(self):
        last = 0.0
        while not self._stop.is_set() and (not self._capture_done.is_set() or not self._frames.empty()):
            try:
                captured_at, frame = self._frames.get(timeout=.2)
            except Empty:
                continue
            remaining = self.min_inference_interval_s - (monotonic() - last)
            if remaining > 0 and self._stop.wait(remaining):
                break
            try:
                self.process_frame(frame, captured_at)
                last = monotonic()
            except Exception:
                self.status = CameraStatus.ERROR
                self.last_error = "Vision analysis failed. Check local model and calibration."
                self._stop.set()
                break
        if self._capture_done.is_set() and self.status != CameraStatus.ERROR:
            self.status = CameraStatus.OFFLINE

    def start(self):
        if any(thread.is_alive() for thread in self._threads):
            raise RuntimeError("Monitoring is already running.")
        self._stop.clear()
        self._capture_done.clear()
        self._threads = [Thread(target=self._capture_loop, daemon=True),
                         Thread(target=self._inference_loop, daemon=True)]
        for thread in self._threads:
            thread.start()

    def stop(self):
        self._stop.set()
        self.connection.stop()
        for thread in self._threads:
            thread.join(timeout=3)
        self._threads.clear()
        self.status = CameraStatus.DISCONNECTED

    def latest(self) -> LiveObservation | None:
        with self._lock:
            return self.history[-1] if self.history else None

    def recent(self) -> tuple[LiveObservation, ...]:
        with self._lock:
            return tuple(self.history)


def build_pipeline(location: dict, *, roi, entry_line, service_line,
                   confidence: float = .35, model_path: str = "yolo11n.pt", stride: int = 1):
    """Load YOLO only when a local monitoring session actually starts."""
    from src.vision.config import validate_line, validate_polygon
    from src.vision.detector import PersonDetector
    from src.vision.pipeline import VisionQueuePipeline
    from src.vision.tracking import AnonymousTracker

    return VisionQueuePipeline(
        PersonDetector(model_path, confidence, frame_skip=stride),
        AnonymousTracker(), validate_polygon(roi), location,
        validate_line(entry_line), validate_line(service_line),
    )
