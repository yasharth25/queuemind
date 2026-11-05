"""Small, optional HTTP bridge to an authorized QueueMind vision worker."""

from __future__ import annotations

from base64 import b64decode, b64encode
from dataclasses import asdict
import json
import os
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from src.vision.analyzer import VisionAnalysisResult, VisionEventRecord, VisionFrameResult
from src.vision.video_io import VideoMetadata

MAX_REMOTE_UPLOAD_BYTES = 40 * 1024 * 1024
MAX_RESPONSE_BYTES = 6 * 1024 * 1024


def remote_config(secrets=None):
    """Return an explicit endpoint/token pair, or None when no worker exists."""
    endpoint = os.getenv("QUEUEMIND_VISION_ENDPOINT")
    token = os.getenv("QUEUEMIND_VISION_TOKEN")
    if secrets is not None:
        try:
            endpoint = endpoint or secrets.get("QUEUEMIND_VISION_ENDPOINT")
            token = token or secrets.get("QUEUEMIND_VISION_TOKEN")
        except (FileNotFoundError, KeyError):
            pass
    if not endpoint or not token:
        return None
    parsed = urlparse(str(endpoint))
    local = parsed.hostname in {"localhost", "127.0.0.1", "::1"}
    if ((parsed.scheme != "https" and not (local and parsed.scheme == "http"))
            or not parsed.netloc or parsed.username or parsed.password or parsed.query
            or parsed.fragment or parsed.path.rstrip("/") != "/analyze"):
        raise ValueError("Vision endpoint must be an HTTPS /analyze URL (HTTP is allowed only on localhost).")
    return str(endpoint), str(token)


def result_to_payload(result: VisionAnalysisResult) -> dict:
    """Bounded aggregate/preview response; never return a raw video."""
    return {
        "metadata": asdict(result.metadata),
        "frames": [asdict(frame) for frame in result.frames],
        "events": [asdict(event) for event in result.events],
        "previews": {str(index): b64encode(image).decode("ascii") for index, image in result.previews.items()},
        "elapsed_s": result.elapsed_s,
        "track_lifetimes_s": result.track_lifetimes_s,
    }


def result_from_payload(payload: dict) -> VisionAnalysisResult:
    """Validate the worker response before presenting video-derived claims."""
    if not isinstance(payload, dict) or not isinstance(payload.get("frames"), list):
        raise ValueError("Vision service returned an invalid analysis result.")
    if not 1 <= len(payload["frames"]) <= 1000:
        raise ValueError("Vision service returned an invalid frame count.")
    metadata = VideoMetadata(**payload["metadata"])
    frames = [VisionFrameResult(**item) for item in payload["frames"]]
    if any(row.frame_index < 0 or row.queue_count < 0 or row.detected_people < 0
           or row.queue_count > row.tracked_people for row in frames):
        raise ValueError("Vision service returned inconsistent people counts.")
    events = [VisionEventRecord(**item) for item in payload.get("events", [])]
    encoded = payload.get("previews", {})
    if not isinstance(encoded, dict) or len(encoded) > 20:
        raise ValueError("Vision service returned too many preview frames.")
    previews = {int(index): b64decode(image, validate=True) for index, image in encoded.items()}
    if sum(map(len, previews.values())) > MAX_RESPONSE_BYTES:
        raise ValueError("Vision service preview data exceeded the size limit.")
    return VisionAnalysisResult(metadata, frames, events, previews,
                                elapsed_s=float(payload.get("elapsed_s", 0)),
                                track_lifetimes_s={int(key): float(value) for key, value in
                                                   payload.get("track_lifetimes_s", {}).items()})


def analyze_remote_upload(uploaded, location_id: str, roi, entry_line, service_line,
                          stride: int, *, secrets=None) -> VisionAnalysisResult:
    config = remote_config(secrets)
    if config is None:
        raise RuntimeError("No authorized video-analysis service is configured for this deployment.")
    if Path(uploaded.name).suffix.lower() not in {".mp4", ".mov", ".avi", ".mkv"}:
        raise ValueError("Upload MP4, MOV, AVI, or MKV video.")
    body = bytes(uploaded.getbuffer())
    if not body or len(body) > MAX_REMOTE_UPLOAD_BYTES:
        raise ValueError("Remote video analysis accepts non-empty clips up to 40 MB.")
    options = json.dumps({"location_id": location_id, "roi": roi,
                          "entry_line": entry_line, "service_line": service_line,
                          "stride": max(5, int(stride))})
    endpoint, token = config
    request = Request(endpoint, data=body, method="POST", headers={
        "Authorization": f"Bearer {token}", "Content-Type": "application/octet-stream",
        "X-QueueMind-Filename": Path(uploaded.name).name,
        "X-QueueMind-Options": options,
    })
    try:
        with urlopen(request, timeout=180) as response:
            encoded = response.read(MAX_RESPONSE_BYTES + 1)
    except HTTPError as exc:
        raise RuntimeError(f"Video service rejected the request (HTTP {exc.code}).") from exc
    except (URLError, TimeoutError) as exc:
        raise RuntimeError("The video service is unavailable or took too long to respond.") from exc
    if len(encoded) > MAX_RESPONSE_BYTES:
        raise ValueError("Vision service response exceeded the size limit.")
    return result_from_payload(json.loads(encoded))
