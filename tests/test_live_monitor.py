"""Continuous monitor tests with deterministic frames; no YOLO, webcam, or network."""

from datetime import datetime, timedelta, timezone
from time import monotonic, sleep

import numpy as np
import pytest

from src.vision.live.camera_sources import CameraConnection, CameraSettings, SourceType
from src.vision.live.camera_sources import CameraStatus
from src.vision.live.monitor import QueueMonitor
from src.vision.detector import Detection
from src.vision.pipeline import VisionQueuePipeline
from src.vision.tracking import AnonymousTracker


class FakePipeline:
    roi = ((.1, .1), (.9, .1), (.9, .9), (.1, .9))
    events = type("Lines", (), {"entry_line": ((.1, .1), (.9, .1)),
                                 "service_line": ((.1, .9), (.9, .9))})()

    def __init__(self):
        self.calls = 0

    def process_frame(self, frame, timestamp):
        self.calls += 1
        kinds = ("ENTRY", "SERVICE_EXIT") if self.calls == 2 else ()
        events = [type("Event", (), {"event_type": kind})() for kind in kinds]
        return {"processed": True, "detections": [object(), object()],
                "tracks": [], "queue": {"queue_count": self.calls % 3,
                                           "queue_track_ids": []}, "events": events}


def monitor(max_history=2):
    settings = CameraSettings("Test", SourceType.WEBCAM, "bank")
    return QueueMonitor("camera-1", CameraConnection(settings), FakePipeline(),
                        {"id": "bank"}, active_counters=2,
                        service_time_minutes=4, rate_window_s=60,
                        max_history=max_history)


def test_monitor_rolls_events_and_bounds_history_without_frames():
    engine = monitor()
    frame = np.zeros((50, 50, 3), dtype=np.uint8)
    first = datetime.now(timezone.utc) - timedelta(minutes=3)
    a = engine.process_frame(frame, first)
    b = engine.process_frame(frame, first + timedelta(seconds=60))
    c = engine.process_frame(frame, first + timedelta(seconds=120))
    assert a.observed_arrival_rate is None
    assert b.observed_arrival_rate == 1
    assert b.service_boundary_rate == 1
    assert b.observed_completion_rate is None
    assert c.observed_arrival_rate == 0
    assert len(engine.recent()) == 2
    assert engine.latest() == c
    assert c.camera_id == "camera-1"
    assert c.configured_service_capacity == .5
    assert c.source_type == SourceType.WEBCAM
    assert c.is_stale(now=first + timedelta(seconds=140), threshold_s=15)
    assert not c.is_stale(now=first + timedelta(seconds=125), threshold_s=15)


def test_latest_frame_buffer_drops_old_frames():
    engine = monitor()
    first = datetime.now(timezone.utc)
    engine._offer_latest(first, "old")
    engine._offer_latest(first + timedelta(seconds=1), "new")
    assert engine._frames.qsize() == 1
    assert engine._frames.get_nowait()[1] == "new"
    assert engine.dropped_frames == 1


def test_observation_rejects_wrong_location_and_naive_time():
    settings = CameraSettings("Test", SourceType.WEBCAM, "library")
    with pytest.raises(ValueError, match="matching location"):
        QueueMonitor("cam", CameraConnection(settings), FakePipeline(),
                     {"id": "bank"}, active_counters=1, service_time_minutes=2)
    with pytest.raises(ValueError, match="timezone"):
        monitor().process_frame(np.zeros((10, 10, 3)), datetime.now())


def test_pipeline_scales_normalized_lines_before_counting_crossings():
    class Detector:
        frame_skip = 1
        def __init__(self):
            self.calls = 0
        def detect(self, frame):
            self.calls += 1
            return [Detection((30, 10 + 20 * (self.calls - 1),
                               50, 50 + 20 * (self.calls - 1)), .9)]

    pipeline = VisionQueuePipeline(
        Detector(), AnonymousTracker(),
        ((.1, .1), (.9, .1), (.9, .9), (.1, .9)), {"id": "bank"},
        ((.1, .6), (.9, .6)), ((.1, .95), (.9, .95)),
    )
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    first = pipeline.process_frame(frame)
    second = pipeline.process_frame(frame)
    assert first["queue"]["queue_count"] == 1
    assert second["queue"]["queue_count"] == 1
    assert [event.event_type for event in second["events"]] == ["ENTRY"]
    assert first["tracks"][0].track_id == second["tracks"][0].track_id


def test_replay_capture_and_inference_threads_finish_cleanly(tmp_path):
    class Capture:
        def __init__(self):
            self.left = 4
            self.released = False
        def isOpened(self):
            return True
        def read(self):
            self.left -= 1
            return (True, np.zeros((30, 30, 3), dtype=np.uint8)) if self.left >= 0 else (False, None)
        def release(self):
            self.released = True

    path = tmp_path / "queue.mp4"
    path.write_bytes(b"fake frame provider")
    captures = []
    def factory(_):
        capture = Capture()
        captures.append(capture)
        return capture

    connection = CameraConnection(CameraSettings("Replay", SourceType.REPLAY_VIDEO,
                                                  "bank", replay_path=str(path)),
                                  capture_factory=factory)
    engine = QueueMonitor("replay-1", connection, FakePipeline(), {"id": "bank"},
                          active_counters=2, service_time_minutes=4,
                          min_inference_interval_s=.01, replay_fps=20)
    engine.start()
    deadline = monotonic() + 3
    while engine.status != CameraStatus.OFFLINE and monotonic() < deadline:
        sleep(.01)
    assert engine.status == CameraStatus.OFFLINE
    assert engine.latest() is not None
    assert engine.latest().source_type == SourceType.REPLAY_VIDEO
    engine.stop()
    assert all(capture.released for capture in captures)
