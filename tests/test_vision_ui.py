"""Vision Lab render checks without YOLO weights, webcam, or a real video."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

from src.vision.analyzer import VisionAnalysisResult, VisionEventRecord, VisionFrameResult
from src.vision.video_io import VideoMetadata
from src.vision.integration import VisionQueueState


APP = "from src.ui.vision_lab import render_vision_lab\nrender_vision_lab()"
SECTIONS = ("Video Analyzer", "ROI Setup", "Live Metrics", "Timeline",
            "Forecast Integration", "Evaluation", "Pipeline / Privacy")


def test_vision_lab_all_sections_render_without_video_or_model(monkeypatch):
    monkeypatch.setattr("src.ui.vision_lab._available", lambda: False)
    app = AppTest.from_string(APP).run(timeout=30)
    assert not app.exception
    assert [tab.label for tab in app.tabs] == list(SECTIONS)
    assert any("LOCAL VISION MODE" in item.value for item in app.info)
    for section in SECTIONS:
        app.query_params["vision_sections"] = section
        app.run(timeout=30)
        assert not app.exception, section


def test_analyzed_result_renders_metrics_timeline_and_forecast():
    frames = [
        VisionFrameResult(0, 0.0, 1, 1, 1, 0, 0, 0, 0, .9),
        VisionFrameResult(10, 1.0, 2, 2, 2, 1, 0, 1, 0, .8),
        VisionFrameResult(20, 2.0, 3, 3, 3, 0, 1, 1, 1, .85),
    ]
    result = VisionAnalysisResult(
        VideoMetadata("mock.mp4", 10, 100, 100, 21, 2.1, 4096), frames,
        [VisionEventRecord(10, 1.0, 1, "ENTRY", 2),
         VisionEventRecord(20, 2.0, 2, "SERVICE_EXIT", 3)],
        track_lifetimes_s={1: 2.0, 2: 1.0}, elapsed_s=1.0,
    )
    app = AppTest.from_string(APP)
    app.session_state["qm_vision_result"] = result
    for section in ("Live Metrics", "Timeline", "Forecast Integration", "Evaluation"):
        app.query_params["vision_sections"] = section
        app.run(timeout=30)
        assert not app.exception, section


def test_selected_vision_state_is_available_in_existing_operations_tabs():
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py")
    app.session_state["qm_state_source"] = "VISION"
    app.session_state["qm_vision_state"] = VisionQueueState("bank", 12, 1.2, 4, 1.5, .6, 2)
    for section in ("Overview", "Forecast", "Optimizer"):
        app.query_params["main_tabs"] = section
        app.run(timeout=30)
        assert not app.exception, section
        assert any("SOURCE: LOCAL VIDEO ANALYSIS" in item.value for item in app.markdown)
        assert [tab.label for tab in app.tabs] == [
            "Overview", "Live Simulation", "Forecast", "Optimizer",
            "ML Lab", "Vision Lab", "Evaluation",
        ]


def test_session_held_upload_remains_available_after_widget_tab_unmount():
    class Upload:
        name = "clip.mp4"
        def getbuffer(self):
            return memoryview(b"small test clip")

    script = (
        "import streamlit as st\n"
        "from src.ui.vision_lab import _source_context\n"
        "with _source_context() as path:\n"
        "    st.write(path.read_bytes().decode())\n"
    )
    app = AppTest.from_string(script)
    app.session_state["qm_vision_upload_keep"] = Upload()
    app.run(timeout=30)
    assert not app.exception
    assert any("small test clip" in item.value for item in app.markdown)
