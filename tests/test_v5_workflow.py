"""V5 navigation, provenance, and optional worker contract without camera/model access."""

from dataclasses import replace
from io import BytesIO
import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from scripts.vision_service import validate_options, validate_video_metadata
from src.config import get_location_by_id
from src.vision.analyzer import VisionAnalysisResult, VisionFrameResult
from src.vision.integration import project_vision_state, state_from_frame
from src.vision.remote import analyze_remote_upload, remote_config, result_from_payload, result_to_payload
from src.vision.video_io import VideoMetadata


def _result():
    frames = [VisionFrameResult(0, 0, 2, 2, 1, 0, 0, 0, 0, .8),
              VisionFrameResult(10, 1, 3, 3, 2, 1, 0, 1, 0, .9)]
    return VisionAnalysisResult(VideoMetadata("test.mp4", 10, 640, 480, 11, 1.1, 1024),
                                frames, [], previews={10: b"jpeg"})


def test_six_simple_tabs_render_without_camera_or_yolo(monkeypatch):
    monkeypatch.setattr("src.ui.vision_lab._local_available", lambda: False)
    monkeypatch.setattr("src.ui.vision_lab._remote_available", lambda: False)
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py")
    for tab in ("Home", "Upload Video", "Live Demo", "Predictions", "Best Counters", "Results"):
        app.query_params["main_tabs"] = tab
        app.run(timeout=30)
        assert not app.exception, tab
        assert [item.label for item in app.tabs] == [
            "Home", "Upload Video", "Live Demo", "Predictions", "Best Counters", "Results",
        ]


def test_video_count_and_synthetic_demand_remain_separately_labelled():
    result = _result()
    location = get_location_by_id("bank")
    state = state_from_frame(result, result.frames[-1], location, 2,
                             forecast_arrival_rate=1.7,
                             demand_source="SYNTHETIC_ASSUMPTION")
    projection = project_vision_state(state, location)
    assert state.current_queue == result.frames[-1].queue_count
    assert state.observed_arrival_rate == result.frames[-1].arrival_rate
    assert projection["demand_source"] == "SYNTHETIC_ASSUMPTION"
    assert projection["points"][1]["forecast_arrival_rate"] == 1.7
    assert projection["recommendation"]["current_outcome"]["future_queue"] == projection["points"][3]["queue"]
    with pytest.raises(ValueError):
        state_from_frame(result, result.frames[-1], location, 2,
                         forecast_arrival_rate=-1)


def test_worker_payload_round_trip_and_invalid_counts():
    result = _result()
    payload = result_to_payload(result)
    restored = result_from_payload(payload)
    assert restored.frames == result.frames
    assert restored.previews == result.previews
    payload["frames"][1]["queue_count"] = 99
    with pytest.raises(ValueError, match="inconsistent"):
        result_from_payload(payload)


def test_remote_service_requires_https_and_bounded_video(monkeypatch):
    monkeypatch.setenv("QUEUEMIND_VISION_ENDPOINT", "http://example.com/analyze")
    monkeypatch.setenv("QUEUEMIND_VISION_TOKEN", "a" * 32)
    with pytest.raises(ValueError, match="HTTPS"):
        remote_config()
    monkeypatch.setenv("QUEUEMIND_VISION_ENDPOINT", "http://localhost:8765/analyze")
    assert remote_config()[0].endswith("/analyze")
    options = {"location_id": "bank", "roi": ((.1, .1), (.9, .1), (.9, .9), (.1, .9)),
               "entry_line": ((.1, .2), (.9, .2)),
               "service_line": ((.1, .8), (.9, .8)), "stride": 5}
    assert validate_options(options)["location"]["id"] == "bank"
    with pytest.raises(ValueError):
        validate_options({**options, "stride": 1})
    metadata = _result().metadata
    validate_video_metadata(metadata)
    with pytest.raises(ValueError):
        validate_video_metadata(replace(metadata, duration_s=31))


def test_remote_client_returns_actual_worker_result_without_local_cv(monkeypatch):
    monkeypatch.setenv("QUEUEMIND_VISION_ENDPOINT", "http://localhost:8765/analyze")
    monkeypatch.setenv("QUEUEMIND_VISION_TOKEN", "t" * 32)
    payload = json.dumps(result_to_payload(_result())).encode()
    seen = {}

    class Reply(BytesIO):
        pass

    def fake_open(request, timeout):
        seen["authorization"] = request.get_header("Authorization")
        seen["video"] = request.data
        seen["options"] = json.loads(request.get_header("X-queuemind-options"))
        return Reply(payload)

    monkeypatch.setattr("src.vision.remote.urlopen", fake_open)
    upload = type("Upload", (), {"name": "clip.mp4", "getbuffer": lambda self: memoryview(b"video")})()
    result = analyze_remote_upload(upload, "bank", ((.1, .1), (.9, .1), (.9, .9)),
                                   ((.1, .2), (.9, .2)), ((.1, .8), (.9, .8)), 5)
    assert result.frames[-1].queue_count == 2
    assert seen["authorization"] == "Bearer " + "t" * 32
    assert seen["video"] == b"video"
    assert seen["options"]["location_id"] == "bank"
