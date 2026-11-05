"""Authorized local camera sources. Credentials remain in process memory only."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from ipaddress import ip_address
from pathlib import Path
from time import monotonic
from urllib.parse import quote, urlsplit
import re


class SourceType(StrEnum):
    WEBCAM = "WEBCAM"
    RTSP = "RTSP"
    ONVIF = "ONVIF"
    REPLAY_VIDEO = "REPLAY_VIDEO"


class CameraStatus(StrEnum):
    DISCONNECTED = "DISCONNECTED"
    CONNECTING = "CONNECTING"
    CONNECTED = "CONNECTED"
    PROCESSING = "PROCESSING"
    DEGRADED = "DEGRADED"
    RECONNECTING = "RECONNECTING"
    OFFLINE = "OFFLINE"
    ERROR = "ERROR"


@dataclass(frozen=True)
class CameraSettings:
    name: str
    source_type: SourceType
    location_id: str
    host: str = ""
    port: int = 554
    stream_path: str = ""
    username: str = field(default="", repr=False)
    password: str = field(default="", repr=False)
    webcam_index: int = 0
    replay_path: str = ""
    manufacturer: str = ""
    model: str = ""

    def __post_init__(self):
        if not self.name.strip() or not self.location_id.strip():
            raise ValueError("Camera name and location are required.")
        if self.source_type in (SourceType.RTSP, SourceType.ONVIF):
            if not 1 <= self.port <= 65535:
                raise ValueError("Camera port must be between 1 and 65535.")
            if not _valid_host(self.host):
                raise ValueError("Enter an IP address or hostname, not a URL or connection string.")
            if self.source_type == SourceType.RTSP and not self.stream_path.strip("/"):
                raise ValueError("RTSP stream path is required; vendor paths are not guessed.")
        if self.source_type == SourceType.WEBCAM and not 0 <= self.webcam_index <= 16:
            raise ValueError("Webcam index must be between 0 and 16 on the local Python host.")
        if self.source_type == SourceType.REPLAY_VIDEO and not self.replay_path:
            raise ValueError("Choose a local replay video file.")


def _valid_host(host: str) -> bool:
    if not host or any(char in host for char in "/:@?#\\ \r\n"):
        return False
    try:
        ip_address(host)
        return True
    except ValueError:
        return bool(re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9.-]{0,251}[A-Za-z0-9])?", host))


def stream_url(settings: CameraSettings) -> str | int:
    """Build a source for local OpenCV only; never display or log the result."""
    if settings.source_type == SourceType.WEBCAM:
        return settings.webcam_index
    if settings.source_type == SourceType.REPLAY_VIDEO:
        path = Path(settings.replay_path).expanduser()
        if not path.is_file():
            raise ValueError("Replay video does not exist on this machine.")
        return str(path)
    if settings.source_type == SourceType.ONVIF:
        raise ValueError("Resolve the ONVIF stream URI locally before opening the source.")
    credentials = ""
    if settings.username:
        credentials = quote(settings.username, safe="")
        if settings.password:
            credentials += ":" + quote(settings.password, safe="")
        credentials += "@"
    return f"rtsp://{credentials}{settings.host}:{settings.port}/{settings.stream_path.lstrip('/')}"


@dataclass(frozen=True)
class CameraProbe:
    width: int
    height: int
    fps: float | None
    frames_read: int
    elapsed_s: float
    preview_frame: object


def _capture_factory(source):
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("Local camera support requires OpenCV. Install requirements-vision.txt.") from exc
    if isinstance(source, str) and source.startswith("rtsp://"):
        params = [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 8000,
                  cv2.CAP_PROP_READ_TIMEOUT_MSEC, 8000]
        return cv2.VideoCapture(source, cv2.CAP_FFMPEG, params)
    return cv2.VideoCapture(source)


def probe_camera(settings: CameraSettings, *, capture_factory=None, frames: int = 3,
                 timeout_s: float = 8) -> CameraProbe:
    """Declare success only after a decoded frame; always release test capture."""
    if frames < 1 or timeout_s <= 0:
        raise ValueError("Probe frames and timeout must be positive.")
    source = stream_url(settings)
    factory = capture_factory or _capture_factory
    start = monotonic()
    try:
        capture = factory(source)
    except Exception:
        raise RuntimeError("Camera could not be opened. Check local decoder and source settings.") from None
    try:
        if not capture.isOpened():
            raise RuntimeError("Camera address is not reachable or authentication failed.")
        valid = 0
        preview = None
        while valid < frames and monotonic() - start < timeout_s:
            ok, frame = capture.read()
            if not ok:
                break
            if frame is not None and len(frame.shape) >= 2 and min(frame.shape[:2]) > 0:
                valid += 1
                preview = frame
        if valid == 0:
            raise RuntimeError("Stream opened, but no frames could be read.")
        width, height = int(preview.shape[1]), int(preview.shape[0])
        fps = None
        try:
            import cv2
            value = float(capture.get(cv2.CAP_PROP_FPS))
            if 0 < value < 1000:
                fps = value
        except (ImportError, AttributeError, TypeError, ValueError):
            pass
        return CameraProbe(width, height, fps, valid, monotonic() - start, preview)
    finally:
        capture.release()


class CameraConnection:
    """Session-scoped capture state; no source credentials in repr/errors."""

    def __init__(self, settings: CameraSettings, *, capture_factory=None):
        self.settings = settings
        self.capture_factory = capture_factory or _capture_factory
        self.status = CameraStatus.DISCONNECTED
        self._capture = None
        self.last_error = None

    def connect(self):
        self.stop()
        self.status = CameraStatus.CONNECTING
        try:
            source = stream_url(self.settings)
            capture = self.capture_factory(source)
            self._capture = capture
            if not capture.isOpened():
                raise RuntimeError("Camera address is not reachable or authentication failed.")
            ok, frame = capture.read()
            if not ok or frame is None:
                raise RuntimeError("Stream opened, but no frames could be read.")
            self.status = CameraStatus.CONNECTED
            self.last_error = None
            return frame
        except Exception as exc:
            self.stop()
            self.status = CameraStatus.ERROR
            self.last_error = (str(exc) if str(exc) in {
                "Camera address is not reachable or authentication failed.",
                "Stream opened, but no frames could be read.",
            } else "Camera connection failed. Check source settings and local decoder.")
            raise RuntimeError(self.last_error) from None

    def read(self):
        if self._capture is None or self.status not in (CameraStatus.CONNECTED, CameraStatus.PROCESSING):
            raise RuntimeError("Camera is disconnected.")
        ok, frame = self._capture.read()
        if not ok or frame is None:
            self.status = CameraStatus.DEGRADED
            raise RuntimeError("Network connection was lost or the stream stopped.")
        self.status = CameraStatus.PROCESSING
        return frame

    def reconnect(self):
        self.status = CameraStatus.RECONNECTING
        return self.connect()

    def stop(self):
        if self._capture is not None:
            self._capture.release()
            self._capture = None
        self.status = CameraStatus.DISCONNECTED


def resolve_onvif_stream(settings: CameraSettings, *, camera_factory=None) -> CameraSettings:
    """Optional ONVIF metadata lookup on the local connector; no vendor URL guesses."""
    if settings.source_type != SourceType.ONVIF:
        raise ValueError("ONVIF settings required.")
    if camera_factory is None:
        try:
            from onvif import ONVIFCamera
        except ImportError as exc:
            raise RuntimeError("ONVIF support requires the optional onvif-zeep package.") from exc
        camera_factory = ONVIFCamera
    camera = camera_factory(settings.host, settings.port, settings.username, settings.password)
    media = camera.create_media_service()
    profiles = media.GetProfiles()
    if not profiles:
        raise RuntimeError("Camera has no ONVIF media profiles.")
    uri = str(media.GetStreamUri({
        "StreamSetup": {"Stream": "RTP-Unicast", "Transport": {"Protocol": "RTSP"}},
        "ProfileToken": profiles[0].token,
    }).Uri)
    parsed = urlsplit(uri)
    if parsed.scheme.lower() != "rtsp" or not parsed.hostname or not parsed.path:
        raise RuntimeError("Camera did not provide an RTSP-compatible ONVIF stream.")
    # The returned URI is used only by the local connector; credentials are supplied in memory.
    return CameraSettings(settings.name, SourceType.RTSP, settings.location_id,
                          parsed.hostname, parsed.port or 554, parsed.path.lstrip("/") +
                          ("?" + parsed.query if parsed.query else ""),
                          settings.username, settings.password, manufacturer=settings.manufacturer,
                          model=settings.model)
