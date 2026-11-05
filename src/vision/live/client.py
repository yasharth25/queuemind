"""HTTPS polling client for aggregate observations and scoped camera reads."""

from __future__ import annotations

from dataclasses import dataclass
import json
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from src.vision.live.service import MAX_PREVIEW_BYTES, serialize_observation


def validate_endpoint(endpoint: str) -> str:
    parsed = urlsplit(endpoint)
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("Live API endpoint must not contain credentials or query parameters.")
    if parsed.scheme == "https" and parsed.hostname:
        return endpoint.rstrip("/")
    if parsed.scheme == "http" and parsed.hostname in ("localhost", "127.0.0.1", "::1"):
        return endpoint.rstrip("/")
    raise ValueError("Remote live API requires HTTPS; HTTP is allowed only on loopback.")


@dataclass
class LiveAPIClient:
    endpoint: str
    token: str = ""
    timeout_s: float = 5

    def __post_init__(self):
        self.endpoint = validate_endpoint(self.endpoint)
        if not 0 < self.timeout_s <= 30:
            raise ValueError("Live API timeout must be between 0 and 30 seconds.")

    def _request(self, method: str, path: str, payload: dict | None = None):
        body = json.dumps(payload).encode() if payload is not None else None
        headers = {"Accept": "application/json"}
        if body is not None:
            headers["Content-Type"] = "application/json"
        if self.token:
            headers["Authorization"] = "Bearer " + self.token
        request = Request(self.endpoint + path, data=body, headers=headers, method=method)
        try:
            with urlopen(request, timeout=self.timeout_s) as reply:
                return json.loads(reply.read(1_000_000))
        except HTTPError as exc:
            raise RuntimeError(f"Live data service rejected request (HTTP {exc.code}).") from None
        except URLError:
            raise RuntimeError("Live data service is unavailable.") from None

    def pair(self, code: str, *, name: str, location_id: str,
             source_type: str, active_counters: int):
        return self._request("POST", "/api/v1/agents/pair", {
            "pairing_code": code, "name": name, "location_id": location_id,
            "source_type": source_type, "active_counters": active_counters,
        })

    def heartbeat(self, status: str = "CONNECTED"):
        return self._request("POST", "/api/v1/agents/heartbeat", {"status": status})

    def send_observation(self, observation):
        data = serialize_observation(observation)
        data.pop("observation_id")
        return self._request("POST", "/api/v1/cameras/observations", data)

    def cameras(self):
        return self._request("GET", "/api/v1/cameras")["cameras"]

    def latest(self, camera_id: str):
        return self._request("GET", f"/api/v1/cameras/{camera_id}/latest")

    def history(self, camera_id: str):
        return self._request("GET", f"/api/v1/cameras/{camera_id}/history")["observations"]

    def disconnect(self, camera_id: str):
        return self._request("POST", f"/api/v1/cameras/{camera_id}/disconnect")

    def remove(self, camera_id: str):
        return self._request("POST", f"/api/v1/cameras/{camera_id}/remove")

    def send_preview(self, camera_id: str, image: bytes, captured_at: str):
        if not isinstance(image, bytes) or len(image) > MAX_PREVIEW_BYTES:
            raise ValueError("Annotated preview must be a JPEG of at most 512 KiB.")
        request = Request(self.endpoint + f"/api/v1/cameras/{camera_id}/preview",
                          data=image, method="POST", headers={
                              "Authorization": "Bearer " + self.token,
                              "Content-Type": "image/jpeg",
                              "X-Captured-At": captured_at,
                          })
        try:
            with urlopen(request, timeout=self.timeout_s) as reply:
                return json.loads(reply.read(1024))
        except (HTTPError, URLError):
            raise RuntimeError("Live preview could not be delivered.") from None

    def preview(self, camera_id: str):
        request = Request(self.endpoint + f"/api/v1/cameras/{camera_id}/preview",
                          headers={"Authorization": "Bearer " + self.token})
        try:
            with urlopen(request, timeout=self.timeout_s) as reply:
                if reply.status == 204:
                    return None
                image = reply.read(MAX_PREVIEW_BYTES + 1)
                if len(image) > MAX_PREVIEW_BYTES:
                    raise RuntimeError("Live preview exceeds the configured size limit.")
                return image, reply.headers.get("X-Captured-At", "")
        except (HTTPError, URLError):
            raise RuntimeError("Live preview is unavailable.") from None
