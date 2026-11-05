"""Camera setup and state checks need no real camera or network access."""

from dataclasses import replace

import numpy as np
import pytest

from src.vision.live.camera_sources import (
    CameraConnection, CameraSettings, CameraStatus, SourceType,
    probe_camera, resolve_onvif_stream, stream_url,
)
from src.vision.live.onvif_discovery import parse_probe_response


class Capture:
    def __init__(self, frames=3, opened=True):
        self.frames = frames
        self.opened = opened
        self.released = False

    def isOpened(self):
        return self.opened

    def read(self):
        self.frames -= 1
        return (True, np.zeros((24, 32, 3), dtype=np.uint8)) if self.frames >= 0 else (False, None)

    def release(self):
        self.released = True


def test_sources_validate_and_keep_credentials_out_of_repr():
    settings = CameraSettings("Bank", SourceType.RTSP, "bank", "192.168.1.9", 554,
                              "stream/main", "operator", "secret")
    assert stream_url(settings) == "rtsp://operator:secret@192.168.1.9:554/stream/main"
    assert "secret" not in repr(settings)
    with pytest.raises(ValueError, match="not a URL"):
        replace(settings, host="https://evil.test/camera")
    with pytest.raises(ValueError, match="path is required"):
        replace(settings, stream_path="")
    with pytest.raises(ValueError, match="does not exist"):
        stream_url(CameraSettings("Replay", SourceType.REPLAY_VIDEO, "bank", replay_path="missing.mp4"))


def test_probe_requires_decoded_frames_and_releases_capture():
    settings = CameraSettings("USB", SourceType.WEBCAM, "bank")
    good = Capture()
    result = probe_camera(settings, capture_factory=lambda _: good)
    assert (result.width, result.height, result.frames_read) == (32, 24, 3)
    assert good.released
    bad = Capture(frames=0)
    with pytest.raises(RuntimeError, match="no frames"):
        probe_camera(settings, capture_factory=lambda _: bad)
    assert bad.released
    closed = Capture(opened=False)
    with pytest.raises(RuntimeError, match="not reachable"):
        probe_camera(settings, capture_factory=lambda _: closed)
    assert closed.released


def test_connection_state_reconnect_and_shutdown():
    captures = []

    def factory(_):
        capture = Capture(frames=2)
        captures.append(capture)
        return capture

    connection = CameraConnection(CameraSettings("USB", SourceType.WEBCAM, "bank"),
                                  capture_factory=factory)
    assert connection.status == CameraStatus.DISCONNECTED
    connection.connect()
    assert connection.status == CameraStatus.CONNECTED
    connection.read()
    assert connection.status == CameraStatus.PROCESSING
    with pytest.raises(RuntimeError, match="lost"):
        connection.read()
    assert connection.status == CameraStatus.DEGRADED
    connection.reconnect()
    assert captures[0].released
    assert connection.status == CameraStatus.CONNECTED
    connection.stop()
    assert captures[1].released


def test_onvif_resolution_and_discovery_parser():
    class Media:
        def GetProfiles(self):
            return [type("Profile", (), {"token": "main"})()]

        def GetStreamUri(self, _):
            return type("Uri", (), {"Uri": "rtsp://10.0.0.5:8554/channel?quality=1"})()

    class Device:
        def create_media_service(self):
            return Media()

    settings = CameraSettings("Camera", SourceType.ONVIF, "bank", host="10.0.0.5",
                              username="user", password="secret")
    resolved = resolve_onvif_stream(settings, camera_factory=lambda *args: Device())
    assert resolved.source_type == SourceType.RTSP
    assert stream_url(resolved).endswith("/channel?quality=1")
    assert resolved.password == "secret"
    payload = b'<Envelope><XAddrs>http://10.0.0.5/onvif/device_service</XAddrs></Envelope>'
    found = parse_probe_response(payload)
    assert len(found) == 1 and found[0].address == "10.0.0.5"
    assert found[0].stream_available is None
