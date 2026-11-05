"""Optional local/private vision worker for a separately hosted Streamlit frontend.

Run behind an HTTPS reverse proxy. No public provider is configured by this repo.
"""

from __future__ import annotations

import argparse
from functools import lru_cache
from hmac import compare_digest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
from threading import Lock
from types import SimpleNamespace

from src.config import get_location_by_id
from src.vision.analyzer import analyze_video
from src.vision.config import load_vision_config, validate_line, validate_polygon
from src.vision.detector import PersonDetector
from src.vision.remote import MAX_REMOTE_UPLOAD_BYTES, result_to_payload
from src.vision.video_io import open_video, uploaded_video_file

MAX_SECONDS = 30
MAX_PIXELS = 1280 * 720
MAX_FRAMES = 1800
ANALYSIS_LOCK = Lock()


@lru_cache(maxsize=1)
def _model(path: str):
    from ultralytics import YOLO
    return YOLO(path)


def validate_options(options: dict) -> dict:
    if not isinstance(options, dict) or not isinstance(options.get("location_id"), str):
        raise ValueError("A configured facility is required.")
    location = get_location_by_id(options["location_id"])
    roi = validate_polygon(options["roi"])
    entry = validate_line(options["entry_line"])
    service = validate_line(options["service_line"])
    stride = int(options.get("stride", 5))
    if not 5 <= stride <= 15:
        raise ValueError("Frame skipping must be between 5 and 15 for the video service.")
    return {"location": location, "roi": roi, "entry": entry,
            "service": service, "stride": stride}


def validate_video_metadata(metadata) -> None:
    if (metadata.duration_s is None or metadata.duration_s <= 0
            or metadata.duration_s > MAX_SECONDS or metadata.total_frames > MAX_FRAMES
            or metadata.width * metadata.height > MAX_PIXELS):
        raise ValueError("Use a video up to 30 seconds, 1280 × 720, and 1800 frames.")


def _analyze_bytes(body: bytes, filename: str, options: dict):
    if not body or len(body) > MAX_REMOTE_UPLOAD_BYTES:
        raise ValueError("Use a non-empty video smaller than 40 MB.")
    name = Path(filename).name
    upload = SimpleNamespace(name=name, getbuffer=lambda: memoryview(body))
    settings = validate_options(options)
    with uploaded_video_file(upload) as path:
        capture, metadata = open_video(path)
        capture.release()
        validate_video_metadata(metadata)
        vision_config = load_vision_config()
        detector = PersonDetector(vision_config["model_path"],
                                  float(vision_config["confidence"]),
                                  model=_model(vision_config["model_path"]))
        result = analyze_video(path, detector, settings["location"], settings["roi"],
                               settings["entry"], settings["service"],
                               stride=settings["stride"], max_previews=12)
    return result_to_payload(result)


class VisionHandler(BaseHTTPRequestHandler):
    token = ""

    def _send(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._send(200, {"status": "ready"})
        else:
            self._send(404, {"error": "Not found"})

    def do_POST(self):
        if self.path != "/analyze":
            self._send(404, {"error": "Not found"})
            return
        supplied = self.headers.get("Authorization", "")
        if not compare_digest(supplied, f"Bearer {self.token}"):
            self._send(401, {"error": "Unauthorized"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= MAX_REMOTE_UPLOAD_BYTES:
                raise ValueError("Use a non-empty video smaller than 40 MB.")
            raw_options = self.headers.get("X-QueueMind-Options", "")
            if len(raw_options) > 8192:
                raise ValueError("Video options are too long.")
            options = json.loads(raw_options)
            filename = self.headers.get("X-QueueMind-Filename", "upload.mp4")
            if not ANALYSIS_LOCK.acquire(blocking=False):
                self._send(429, {"error": "Video service is busy. Try again shortly."})
                return
            try:
                result = _analyze_bytes(self.rfile.read(length), filename, options)
            finally:
                ANALYSIS_LOCK.release()
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            self._send(400, {"error": str(exc)})
            return
        except Exception:
            self._send(503, {"error": "Video analysis failed on the worker."})
            return
        self._send(200, result)


def main() -> None:
    parser = argparse.ArgumentParser(description="QueueMind private video inference worker")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    token = os.environ.get("QUEUEMIND_VISION_TOKEN", "")
    if len(token) < 24:
        raise SystemExit("Set a random QUEUEMIND_VISION_TOKEN of at least 24 characters.")
    VisionHandler.token = token
    server = ThreadingHTTPServer((args.host, args.port), VisionHandler)
    print(f"QueueMind vision worker listening on {args.host}:{args.port}; use HTTPS proxy for remote access.")
    server.serve_forever()


if __name__ == "__main__":
    main()
