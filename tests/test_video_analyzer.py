from dataclasses import replace

import numpy as np
import pytest

from src.vision.analyzer import (
    VisionAnalysisResult, VisionEventRecord, VisionFrameResult,
    analyze_frames, analyze_video, rolling_event_rate,
)
from src.vision.config import validate_line, validate_polygon
from src.vision.integration import project_vision_state, state_from_frame
from src.vision.video_io import VideoMetadata, require_opencv
from src.vision.tracking import TrackedPerson
from src.vision.events import QueueEvent
from src.vision.detector import Detection


META = VideoMetadata("queue.mp4", 10, 100, 100, 10, 1.0, 1024)
ROI = ((.1, .1), (.9, .1), (.9, .9), (.1, .9))
ENTRY = ((.1, .3), (.9, .3))
SERVICE = ((.1, .8), (.9, .8))
LOCATION = {
    "id": "bank", "average_service_time_minutes": 3,
    "target_wait_minutes": 10, "warning_queue_length": 10,
    "critical_queue_length": 20,
}


class FakePipeline:
    def __init__(self):
        self.index = 0

    def process_frame(self, frame, timestamp):
        self.index += 1
        count = min(self.index, 3)
        tracks = [TrackedPerson(i, (10, 10, 20, 30), .8, (15, 30))
                  for i in range(1, count + 1)]
        event_type = "ENTRY" if self.index == 2 else "SERVICE_EXIT" if self.index == 3 else None
        events = [QueueEvent(timestamp.isoformat(), "bank", 1, event_type)] if event_type else []
        return {"processed": True, "detections": tracks, "tracks": tracks,
                "queue": {"queue_count": count, "queue_track_ids": list(range(1, count + 1))},
                "events": events}


def test_analyzer_aggregates_frames_events_rates_and_exports_without_opencv():
    frames = ((i, np.zeros((100, 100, 3), dtype=np.uint8)) for i in range(10))
    result = analyze_frames(frames, META, FakePipeline(), ROI, ENTRY, SERVICE, stride=2)
    assert [row.frame_index for row in result.frames] == [0, 2, 4, 6, 8]
    assert result.summary["processed_frames"] == 5
    assert result.summary["max_queue_count"] == 3
    assert result.summary["entry_events"] == 1
    assert result.summary["service_events"] == 1
    assert result.frames[1].arrival_rate == 1
    assert result.frames[2].service_rate == 1
    assert result.track_lifetimes_s[1] == 1.0
    assert result.frames_dataframe().queue_count.tolist() == [1, 2, 3, 3, 3]
    assert set(result.events_dataframe().event_type) == {"ENTRY", "SERVICE_EXIT", "QUEUE_CHANGE"}
    sample = result.labelling_sample(3)
    assert sample.frame_id.tolist() == ["0", "4", "8"]
    assert sample.ground_truth_queue_count.eq("").all()


def test_rolling_rate_uses_video_time_not_total_duration():
    assert rolling_event_rate([10, 40, 60], 60, 30) == 4
    assert rolling_event_rate([10, 40, 60], 60, 60) == 3
    with pytest.raises(ValueError):
        rolling_event_rate([], 0, 0)


def test_empty_or_invalid_analysis_is_rejected():
    with pytest.raises(ValueError, match="no readable"):
        analyze_frames([], META, FakePipeline(), ROI, ENTRY, SERVICE)
    with pytest.raises(ValueError, match="Stride"):
        analyze_frames([], META, FakePipeline(), ROI, ENTRY, SERVICE, stride=0)
    with pytest.raises(ValueError):
        validate_polygon([(0, 0), (1, 1), (2, 2)])
    with pytest.raises(ValueError):
        validate_line([(0, 0), (0, 0)])


def test_vision_state_projects_with_existing_queue_math_and_preserves_source():
    frame = VisionFrameResult(20, 2.0, 4, 4, 3, 1, 0, 1.5, 0.2, .9)
    result = VisionAnalysisResult(META, [frame], [VisionEventRecord(20, 2, 1, "ENTRY", 3)])
    state = state_from_frame(result, frame, LOCATION, 2)
    forecast = project_vision_state(state, LOCATION)
    assert state.state_source == "VISION"
    assert forecast["points"][0]["queue"] == 3
    assert [point["minute"] for point in forecast["points"]] == [0, 5, 10, 20, 30]
    assert forecast["points"][3]["queue"] > 3
    assert forecast["recommendation"]["recommended_counters"] >= 2
    assert forecast["configured_service_capacity"] == pytest.approx(2 / 3)
    assert forecast["observed_service_completion_rate"] == .2
    with pytest.raises(ValueError):
        state_from_frame(result, replace(frame, frame_index=21), LOCATION, 2)


def test_missing_opencv_error_is_actionable(monkeypatch):
    import builtins
    real_import = builtins.__import__
    def blocked(name, *args, **kwargs):
        if name == "cv2":
            raise ImportError("mock missing")
        return real_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", blocked)
    with pytest.raises(RuntimeError, match="requirements-vision.txt"):
        require_opencv()


def test_optional_local_mp4_streaming_smoke(tmp_path):
    cv2 = pytest.importorskip("cv2")
    source = tmp_path / "tiny_queue.mp4"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 64))
    if not writer.isOpened():
        pytest.skip("Local OpenCV build has no MP4 encoder")
    for _ in range(12):
        writer.write(np.zeros((64, 64, 3), dtype=np.uint8))
    writer.release()

    class MovingDetector:
        frame_skip = 1
        def __init__(self):
            self.index = 0
        def detect(self, frame):
            y = 14 + 4 * self.index
            self.index += 1
            return [Detection((20, y - 10, 35, y), .9)]

    result = analyze_video(source, MovingDetector(), LOCATION, ROI, ENTRY, SERVICE,
                           max_previews=4)
    assert result.summary["processed_frames"] == 12
    assert result.summary["entry_events"] == 1
    assert result.summary["service_events"] == 1
    assert 1 <= len(result.previews) <= 4
    assert result.summary["analysis_fps"] > 0
