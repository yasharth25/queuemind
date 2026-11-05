"""Pairing, ownership, bounded aggregate ingestion, and token revocation."""

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from http.server import ThreadingHTTPServer
from threading import Thread

import pytest

from src.vision.live.camera_sources import CameraConnection, CameraSettings, SourceType
from src.vision.live.client import LiveAPIClient, validate_endpoint
from src.vision.live.forecast import project_live_queue
from src.vision.live.monitor import LiveObservation, QueueMonitor
from src.vision.live.simulation import SimulationControls, baseline_from_live, run_what_if
from src.vision.live.service import (
    AccessDenied, InvalidObservation, LiveDataStore, deserialize_observation,
    serialize_observation,
)
from scripts.live_data_service import handler_for


ADMIN_A, VIEW_A = "a" * 40, "b" * 40
ADMIN_B, VIEW_B = "c" * 40, "d" * 40


def paired_store():
    store = LiveDataStore({"alpha": (ADMIN_A, VIEW_A), "beta": (ADMIN_B, VIEW_B)},
                          max_history=2)
    code = store.issue_pairing_code(ADMIN_A)
    registration = store.pair(code, name="Bank", location_id="bank",
                              source_type="WEBCAM", active_counters=2)
    return store, code, registration


def sample_payload(camera_id, seconds_ago=0):
    now = datetime.now(timezone.utc) - timedelta(seconds=seconds_ago)
    observation = LiveObservation(
        "local-id", camera_id, "bank", now, now, SourceType.WEBCAM,
        8, 6, 1, 0, 1.0, 0.0, None, 2 / 3, 2.0, .1,
    )
    payload = serialize_observation(observation)
    payload.pop("observation_id")
    return payload


def test_pairing_is_one_time_and_scoped_to_workspace():
    store, code, registration = paired_store()
    with pytest.raises(AccessDenied):
        store.pair(code, name="Second", location_id="bank",
                   source_type="WEBCAM", active_counters=2)
    assert len(store.cameras(VIEW_A)) == 1
    assert store.cameras(VIEW_B) == []
    with pytest.raises(AccessDenied):
        store.latest(VIEW_B, registration["camera_id"])
    with pytest.raises(AccessDenied):
        store.issue_pairing_code(VIEW_A)


def test_authenticated_ingestion_is_bounded_and_rejects_wrong_camera():
    store, _, registration = paired_store()
    camera_id, token = registration["camera_id"], registration["device_token"]
    payload = sample_payload(camera_id)
    for index in range(3):
        dated = dict(payload)
        timestamp = datetime.now(timezone.utc) + timedelta(milliseconds=index)
        dated["captured_at"] = dated["processed_at"] = timestamp.isoformat()
        store.ingest(token, dated)
    assert len(store.history(VIEW_A, camera_id)) == 2
    assert store.latest(VIEW_A, camera_id)["observation"]["people_waiting"] == 6
    with pytest.raises(AccessDenied):
        store.ingest(token, {**payload, "camera_id": "another-camera"})
    with pytest.raises(InvalidObservation):
        store.ingest(token, {**payload, "raw_frame": "secret"})
    with pytest.raises(InvalidObservation):
        store.ingest(token, sample_payload(camera_id, seconds_ago=600))


def test_revocation_disables_connector_and_viewer_cannot_revoke():
    store, _, registration = paired_store()
    camera_id, token = registration["camera_id"], registration["device_token"]
    with pytest.raises(AccessDenied):
        store.disconnect(VIEW_A, camera_id)
    store.heartbeat(token)
    image = b"\xff\xd8" + b"0" * 100 + b"\xff\xd9"
    store.set_preview(token, camera_id, image, datetime.now(timezone.utc).isoformat())
    assert store.preview(VIEW_A, camera_id)[0] == image
    store._previews[camera_id] = (image, datetime.now(timezone.utc) - timedelta(seconds=30))
    store.expire_previews()
    assert store.preview(VIEW_A, camera_id) is None
    store.set_preview(token, camera_id, image, datetime.now(timezone.utc).isoformat())
    with pytest.raises(AccessDenied):
        store.preview(VIEW_B, camera_id)
    store.disconnect(ADMIN_A, camera_id)
    assert store.latest(VIEW_A, camera_id)["status"] == "DISCONNECTED"
    assert store.preview(VIEW_A, camera_id) is None
    with pytest.raises(AccessDenied):
        store.heartbeat(token)
    store.remove(ADMIN_A, camera_id)
    assert store.cameras(VIEW_A) == []


def test_remote_endpoint_requires_https_and_never_embeds_credentials():
    assert validate_endpoint("http://127.0.0.1:8780") == "http://127.0.0.1:8780"
    with pytest.raises(ValueError, match="HTTPS"):
        validate_endpoint("http://example.com")
    with pytest.raises(ValueError, match="credentials"):
        validate_endpoint("https://user:secret@example.com")


def test_heartbeat_reports_real_reconnecting_state():
    store, _, registration = paired_store()
    camera_id, token = registration["camera_id"], registration["device_token"]
    assert store.latest(VIEW_A, camera_id)["status"] == "CONNECTING"
    store.heartbeat(token, "RECONNECTING")
    assert store.latest(VIEW_A, camera_id)["status"] == "RECONNECTING"
    with pytest.raises(ValueError, match="Invalid camera status"):
        store.heartbeat(token, "FAKE_LIVE")


def test_local_http_pair_ingest_read_and_revoke():
    store = LiveDataStore({"alpha": (ADMIN_A, VIEW_A)})
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(store))
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_port}"
        admin = LiveAPIClient(url, ADMIN_A)
        code = admin._request("POST", "/api/v1/pairing-codes")["pairing_code"]
        agent = LiveAPIClient(url)
        registration = agent.pair(code, name="Test", location_id="bank",
                                  source_type="WEBCAM", active_counters=2)
        agent.token = registration["device_token"]
        payload = sample_payload(registration["camera_id"])
        agent._request("POST", "/api/v1/cameras/observations", payload)
        viewer = LiveAPIClient(url, VIEW_A)
        assert viewer.latest(registration["camera_id"])["observation"]["people_waiting"] == 6
        image = b"\xff\xd8" + b"0" * 100 + b"\xff\xd9"
        agent.send_preview(registration["camera_id"], image,
                           datetime.now(timezone.utc).isoformat())
        assert viewer.preview(registration["camera_id"])[0] == image
        with pytest.raises(RuntimeError, match="403"):
            LiveAPIClient(url, VIEW_B).latest(registration["camera_id"])
        admin.disconnect(registration["camera_id"])
        with pytest.raises(RuntimeError, match="403"):
            agent.heartbeat()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_fake_camera_to_service_to_forecast_and_simulation():
    store = LiveDataStore({"alpha": (ADMIN_A, VIEW_A)})
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(store))
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_port}"
        code = LiveAPIClient(url, ADMIN_A)._request("POST", "/api/v1/pairing-codes")["pairing_code"]
        agent = LiveAPIClient(url)
        paired = agent.pair(code, name="Fake", location_id="bank",
                            source_type="WEBCAM", active_counters=2)
        agent.token = paired["device_token"]

        class Pipeline:
            def __init__(self):
                self.calls = 0
            def process_frame(self, frame, captured_at):
                self.calls += 1
                events = ([type("Event", (), {"event_type": "ENTRY"})()]
                          if self.calls == 2 else [])
                return {"processed": True, "detections": [object()] * 4,
                        "tracks": [], "queue": {"queue_count": 3,
                                                   "queue_track_ids": []}, "events": events}

        connection = CameraConnection(CameraSettings("Fake", SourceType.WEBCAM, "bank"))
        monitor = QueueMonitor(paired["camera_id"], connection, Pipeline(),
                               {"id": "bank"}, active_counters=2,
                               service_time_minutes=3, rate_window_s=60)
        now = datetime.now(timezone.utc)
        monitor.process_frame(object(), now - timedelta(seconds=60))
        observed = monitor.process_frame(object(), now)
        agent.send_observation(observed)
        latest = LiveAPIClient(url, VIEW_A).latest(paired["camera_id"])
        actual = deserialize_observation(latest["observation"])
        assert actual.people_waiting == 3 and actual.observed_arrival_rate == 1
        location = {"id": "bank", "average_service_time_minutes": 3,
                    "target_wait_minutes": 15, "warning_queue_length": 15,
                    "critical_queue_length": 30}
        forecast = project_live_queue(actual, location, active_counters=2)
        assert forecast["available"] and len(forecast["points"]) == 5
        baseline = baseline_from_live(actual, location, 2)
        scenario = run_what_if(baseline, SimulationControls(3, service_time_minutes=3))
        assert scenario["proposed"]["future_queue"] <= scenario["current"]["future_queue"]
        assert store.latest(VIEW_A, paired["camera_id"])["observation"]["people_waiting"] == 3
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
