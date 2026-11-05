"""Cloud frontend renders truthfully with no connector and with authenticated aggregates."""

from datetime import datetime, timezone
from pathlib import Path

from streamlit.testing.v1 import AppTest

from src.vision.live.camera_sources import SourceType
from src.vision.live.monitor import LiveObservation
from src.vision.live.service import serialize_observation


APP = Path(__file__).resolve().parents[1] / "app.py"


def test_public_demo_without_live_endpoint_stays_disconnected(monkeypatch):
    monkeypatch.delenv("QUEUEMIND_LIVE_API_ENDPOINT", raising=False)
    monkeypatch.setattr("src.ui.live_camera._service_endpoint", lambda: None)
    app = AppTest.from_file(APP).run(timeout=30)
    assert not app.exception
    assert [tab.label for tab in app.tabs] == ["Live Camera", "Simulation Studio"]
    assert any("CONNECTOR REQUIRED" in item.value for item in app.info)
    assert app.session_state.get("qm_camera_observation") is None


def test_authenticated_camera_aggregate_reaches_live_dashboard_and_studio(monkeypatch):
    now = datetime.now(timezone.utc)
    observation = LiveObservation("obs-1", "cam-1", "bank", now, now,
                                  SourceType.WEBCAM, 12, 9, 2, 1, 1.5,
                                  1.0, None, 1.0, 2.0, .1)
    camera = {"camera_id": "cam-1", "name": "Bank", "location_id": "bank",
              "active_counters": 3, "source_type": "WEBCAM"}

    class Client:
        def __init__(self, endpoint, token):
            assert endpoint == "https://service.example.test"
            assert token == "viewer-token"
        def latest(self, camera_id):
            assert camera_id == "cam-1"
            return {"camera": camera, "status": "PROCESSING",
                    "observation": serialize_observation(observation)}
        def history(self, camera_id):
            return [serialize_observation(observation)]
        def preview(self, camera_id):
            return None

    monkeypatch.setattr("src.ui.live_camera._service_endpoint",
                        lambda: "https://service.example.test")
    monkeypatch.setattr("src.ui.live_camera.LiveAPIClient", Client)
    app = AppTest.from_file(APP)
    app.session_state["qm_live_view_token"] = "viewer-token"
    app.session_state["qm_connected_cameras"] = [camera]
    app.run(timeout=30)
    assert not app.exception
    assert app.session_state["qm_camera_observation"].people_waiting == 9
    assert any("People Waiting" in metric.label for metric in app.metric)
    app.query_params["main_tabs"] = "Simulation Studio"
    app.run(timeout=30)
    assert not app.exception
    assert app.session_state["qm_studio_baseline"].snapshot_id == "obs-1"
    assert app.session_state["qm_studio_baseline"].source_label == "OBSERVED CAMERA"


def test_demo_and_advanced_tools_do_not_duplicate_streamlit_widget_keys(monkeypatch):
    monkeypatch.setattr("src.ui.live_camera._service_endpoint", lambda: None)
    app = AppTest.from_file(APP).run(timeout=30)
    next(button for button in app.button if button.label == "Try Demo").click().run(timeout=30)
    assert not app.exception
    app.session_state["qm_advanced_tool"] = "Best Counters"
    app.run(timeout=30)
    assert not app.exception
    next(button for button in app.button if button.label == "Hide SYNTHETIC Demo").click().run(timeout=30)
    assert not app.exception


def test_pairing_code_is_requested_only_with_admin_token(monkeypatch):
    class Client:
        def __init__(self, endpoint, token):
            assert endpoint == "https://service.example.test"
            assert token == "admin-secret"
        def _request(self, method, path):
            assert (method, path) == ("POST", "/api/v1/pairing-codes")
            return {"pairing_code": "one-time-code"}

    monkeypatch.setattr("src.ui.live_camera._service_endpoint",
                        lambda: "https://service.example.test")
    monkeypatch.setattr("src.ui.live_camera.LiveAPIClient", Client)
    app = AppTest.from_file(APP).run(timeout=30)
    next(button for button in app.button if button.label == "Connect Camera").click().run(timeout=30)
    app.text_input(key="qm_pair_admin_token").set_value("admin-secret").run(timeout=30)
    next(button for button in app.button if button.label == "Create Pairing Code").click().run(timeout=30)
    assert not app.exception
    assert app.session_state["qm_pairing_code"] == "one-time-code"
    assert app.session_state["qm_pair_admin_token"] == ""
