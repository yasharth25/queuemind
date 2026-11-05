"""Small authenticated, bounded live observation store for an academic MVP.

No camera credentials, RTSP URLs, or global mutable per-user UI state. One
short-lived annotated JPEG per camera is optional and requires connector opt-in.
The store is process-local; deploy behind an HTTPS reverse proxy and persistent
storage before treating it as a production multi-user service.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from secrets import token_urlsafe
from threading import RLock
from uuid import uuid4

from src.vision.live.camera_sources import CameraStatus, SourceType
from src.vision.live.monitor import LiveObservation, utc_now
from src.config import get_location_by_id

MAX_PREVIEW_BYTES = 512 * 1024


class AccessDenied(Exception):
    pass


class InvalidObservation(Exception):
    pass


def _hash(secret: str) -> str:
    return sha256(secret.encode()).hexdigest()


def _utc(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (ValueError, AttributeError) as exc:
        raise InvalidObservation("Use ISO 8601 timestamps with timezone.") from exc
    if parsed.tzinfo is None:
        raise InvalidObservation("Observation timestamps require timezone.")
    return parsed.astimezone(timezone.utc)


class LiveDataStore:
    """Workspace-scoped pairing, device ingestion, viewer reads, and revocation."""

    def __init__(self, workspaces: dict[str, tuple[str, str]], *,
                 max_history: int = 300, pairing_ttl_s: int = 300,
                 device_ttl_s: int = 8 * 3600):
        if not workspaces or max_history < 1 or pairing_ttl_s < 1 or device_ttl_s < 1:
            raise ValueError("Workspace credentials and positive bounds are required.")
        self._admins = {}
        self._viewers = {}
        for workspace, (admin, viewer) in workspaces.items():
            if not workspace or min(len(admin), len(viewer)) < 32 or admin == viewer:
                raise ValueError("Distinct admin and viewer tokens of at least 32 characters are required.")
            self._admins[_hash(admin)] = workspace
            self._viewers[_hash(viewer)] = workspace
        self.max_history = max_history
        self.pairing_ttl_s = pairing_ttl_s
        self.device_ttl_s = device_ttl_s
        self._codes = {}
        self._devices = {}
        self._cameras = {}
        self._history = {}
        self._previews = {}
        self._lock = RLock()

    def _workspace(self, token: str, *, admin: bool = False) -> str:
        key = _hash(token or "")
        workspace = self._admins.get(key)
        if workspace is None and not admin:
            workspace = self._viewers.get(key)
        if workspace is None:
            raise AccessDenied("Authorization required.")
        return workspace

    def issue_pairing_code(self, admin_token: str) -> str:
        workspace = self._workspace(admin_token, admin=True)
        code = token_urlsafe(8)
        with self._lock:
            self._codes = {key: record for key, record in self._codes.items()
                           if record[1] > utc_now()}
            if len(self._codes) >= 1000:
                raise ValueError("Too many active pairing codes; try again later.")
            self._codes[_hash(code)] = (workspace, utc_now() + timedelta(seconds=self.pairing_ttl_s))
        return code

    def pair(self, code: str, *, name: str, location_id: str,
             source_type: str, active_counters: int) -> dict:
        if (not isinstance(name, str) or not isinstance(location_id, str)
                or not name.strip() or len(name) > 100 or not location_id.strip()
                or not isinstance(active_counters, int) or isinstance(active_counters, bool)
                or not 1 <= active_counters <= 8):
            raise ValueError("Camera name, location, and positive counter count are required.")
        try:
            get_location_by_id(location_id)
        except ValueError as exc:
            raise ValueError("Choose a configured facility location.") from exc
        try:
            source = SourceType(source_type)
        except ValueError as exc:
            raise ValueError("Unsupported camera source type.") from exc
        with self._lock:
            record = self._codes.pop(_hash(code or ""), None)
            if record is None or record[1] < utc_now():
                raise AccessDenied("Pairing code is invalid or expired.")
            workspace = record[0]
            if sum(camera["workspace_id"] == workspace for camera in self._cameras.values()) >= 50:
                raise ValueError("This workspace has reached the 50-camera development limit.")
            camera_id = str(uuid4())
            token = token_urlsafe(32)
            self._devices[_hash(token)] = (workspace, camera_id,
                                           utc_now() + timedelta(seconds=self.device_ttl_s))
            self._cameras[camera_id] = {
                "camera_id": camera_id, "workspace_id": workspace,
                "name": name.strip(), "location_id": location_id,
                "source_type": source.value, "active_counters": active_counters,
                "status": "CONNECTING", "last_seen": utc_now().isoformat(),
            }
            self._history[camera_id] = deque(maxlen=self.max_history)
        return {"camera_id": camera_id, "device_token": token,
                "expires_in_seconds": self.device_ttl_s}

    def _device(self, token: str) -> tuple[str, str]:
        record = self._devices.get(_hash(token or ""))
        if record is None or record[2] < utc_now():
            raise AccessDenied("Connector token is invalid, expired, or revoked.")
        return record[0], record[1]

    def heartbeat(self, token: str, status: str = "CONNECTED") -> dict:
        try:
            camera_status = CameraStatus(status)
        except ValueError as exc:
            raise ValueError("Invalid camera status.") from exc
        with self._lock:
            workspace, camera_id = self._device(token)
            self._devices[_hash(token)] = (workspace, camera_id,
                                           utc_now() + timedelta(seconds=self.device_ttl_s))
            camera = self._cameras[camera_id]
            camera["last_seen"] = utc_now().isoformat()
            camera["status"] = camera_status.value
            return {"camera_id": camera_id, "status": camera["status"]}

    def ingest(self, token: str, payload: dict) -> LiveObservation:
        allowed = {"camera_id", "location_id", "captured_at", "processed_at", "source_type",
                   "people_detected", "people_waiting", "entries_in_window",
                   "service_boundary_crossings", "observed_arrival_rate",
                   "service_boundary_rate", "observed_completion_rate",
                   "configured_service_capacity", "analysis_fps", "processing_latency_s"}
        if not isinstance(payload, dict) or set(payload) != allowed:
            raise InvalidObservation("Submit only the required aggregate observation fields.")
        with self._lock:
            workspace, camera_id = self._device(token)
            camera = self._cameras[camera_id]
            if payload["camera_id"] != camera_id or payload["location_id"] != camera["location_id"]:
                raise AccessDenied("Connector does not own this camera or location.")
            if payload["source_type"] != camera["source_type"]:
                raise InvalidObservation("Source type changed during monitoring.")
            try:
                source = SourceType(payload["source_type"])
                observation = LiveObservation(
                    str(uuid4()), camera_id, camera["location_id"],
                    _utc(payload["captured_at"]), _utc(payload["processed_at"]), source,
                    payload["people_detected"], payload["people_waiting"],
                    payload["entries_in_window"],
                    payload["service_boundary_crossings"],
                    (float(payload["observed_arrival_rate"])
                     if payload["observed_arrival_rate"] is not None else None),
                    (float(payload["service_boundary_rate"])
                     if payload["service_boundary_rate"] is not None else None),
                    (float(payload["observed_completion_rate"])
                     if payload["observed_completion_rate"] is not None else None),
                    float(payload["configured_service_capacity"]),
                    float(payload["analysis_fps"]) if payload["analysis_fps"] is not None else None,
                    float(payload["processing_latency_s"]),
                )
            except (ValueError, TypeError, KeyError) as exc:
                raise InvalidObservation("Observation fields are invalid.") from exc
            if abs((utc_now() - observation.captured_at).total_seconds()) > 300:
                raise InvalidObservation("Observation is too old or too far in the future.")
            expected_capacity = (camera["active_counters"] /
                                 float(get_location_by_id(camera["location_id"])["average_service_time_minutes"]))
            if abs(observation.configured_service_capacity - expected_capacity) > .01:
                raise InvalidObservation("Configured service capacity does not match camera facility settings.")
            history = self._history[camera_id]
            if history and observation.captured_at < history[-1].captured_at:
                raise InvalidObservation("Observation time cannot move backwards.")
            history.append(observation)
            camera["last_seen"] = utc_now().isoformat()
            camera["status"] = "PROCESSING"
            return observation

    def cameras(self, viewer_token: str) -> list[dict]:
        workspace = self._workspace(viewer_token)
        with self._lock:
            return [self._public_camera(camera) for camera in self._cameras.values()
                    if camera["workspace_id"] == workspace]

    def _public_camera(self, camera):
        return {key: value for key, value in camera.items() if key != "workspace_id"}

    def _authorized_camera(self, token: str, camera_id: str):
        workspace = self._workspace(token)
        camera = self._cameras.get(camera_id)
        if camera is None or camera["workspace_id"] != workspace:
            raise AccessDenied("Camera is unavailable in this workspace.")
        return camera

    def latest(self, viewer_token: str, camera_id: str) -> dict:
        with self._lock:
            camera = self._authorized_camera(viewer_token, camera_id)
            history = self._history[camera_id]
            latest = serialize_observation(history[-1]) if history else None
            last_seen = _utc(camera["last_seen"])
            status = camera["status"]
            if status != "DISCONNECTED" and (utc_now() - last_seen).total_seconds() > 15:
                status = "OFFLINE"
            if status in ("CONNECTED", "PROCESSING") and history and history[-1].is_stale():
                status = "OFFLINE"
            return {"camera": self._public_camera(camera), "status": status,
                    "observation": latest}

    def history(self, viewer_token: str, camera_id: str) -> list[dict]:
        with self._lock:
            self._authorized_camera(viewer_token, camera_id)
            return [serialize_observation(item) for item in self._history[camera_id]]

    def set_preview(self, device_token: str, camera_id: str, image: bytes,
                    captured_at: str) -> None:
        """Keep only the latest authenticated JPEG in process memory."""
        if (not isinstance(image, bytes) or not 100 <= len(image) <= MAX_PREVIEW_BYTES
                or not image.startswith(b"\xff\xd8") or not image.endswith(b"\xff\xd9")):
            raise InvalidObservation("Preview must be a JPEG of at most 512 KiB.")
        timestamp = _utc(captured_at)
        if abs((utc_now() - timestamp).total_seconds()) > 30:
            raise InvalidObservation("Preview capture time is out of date.")
        with self._lock:
            _, owned_camera = self._device(device_token)
            if owned_camera != camera_id:
                raise AccessDenied("Connector does not own this camera.")
            self._previews[camera_id] = (image, timestamp)

    def preview(self, viewer_token: str, camera_id: str) -> tuple[bytes, datetime] | None:
        with self._lock:
            self._authorized_camera(viewer_token, camera_id)
            self.expire_previews()
            stored = self._previews.get(camera_id)
            if stored is None:
                return None
            return stored

    def expire_previews(self):
        """Housekeeping used by the service even when no viewer is polling."""
        with self._lock:
            for camera_id, (_, timestamp) in tuple(self._previews.items()):
                if (utc_now() - timestamp).total_seconds() > 15:
                    self._previews.pop(camera_id, None)

    def disconnect(self, token: str, camera_id: str) -> dict:
        with self._lock:
            try:
                workspace, owned_camera = self._device(token)
                allowed = owned_camera == camera_id
            except AccessDenied:
                workspace = self._workspace(token, admin=True)
                allowed = (camera_id in self._cameras
                           and self._cameras[camera_id]["workspace_id"] == workspace)
            if not allowed:
                raise AccessDenied("Camera cannot be disconnected by this token.")
            self._cameras[camera_id]["status"] = "DISCONNECTED"
            self._previews.pop(camera_id, None)
            for key, record in tuple(self._devices.items()):
                if record[1] == camera_id:
                    del self._devices[key]
            return {"camera_id": camera_id, "status": "DISCONNECTED"}

    def remove(self, admin_token: str, camera_id: str) -> dict:
        with self._lock:
            workspace = self._workspace(admin_token, admin=True)
            camera = self._cameras.get(camera_id)
            if camera is None or camera["workspace_id"] != workspace:
                raise AccessDenied("Camera cannot be removed from this workspace.")
            self.disconnect(admin_token, camera_id)
            del self._cameras[camera_id]
            del self._history[camera_id]
            return {"camera_id": camera_id, "removed": True}


def serialize_observation(observation: LiveObservation) -> dict:
    data = asdict(observation)
    data["captured_at"] = observation.captured_at.isoformat()
    data["processed_at"] = observation.processed_at.isoformat()
    data["source_type"] = observation.source_type.value
    return data


def deserialize_observation(data: dict) -> LiveObservation:
    return LiveObservation(
        data["observation_id"], data["camera_id"], data["location_id"],
        _utc(data["captured_at"]), _utc(data["processed_at"]),
        SourceType(data["source_type"]), int(data["people_detected"]),
        int(data["people_waiting"]), int(data["entries_in_window"]),
        int(data["service_boundary_crossings"]), data["observed_arrival_rate"],
        data["service_boundary_rate"], data["observed_completion_rate"],
        float(data["configured_service_capacity"]), data["analysis_fps"],
        float(data["processing_latency_s"]),
    )
