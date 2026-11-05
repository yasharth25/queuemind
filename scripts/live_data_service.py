"""Aggregate-first local development API; place behind HTTPS for remote use.

No camera URLs or credentials are accepted. Optional annotated JPEG previews
require explicit connector opt-in and stay only in bounded process memory.
"""

from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from secrets import token_urlsafe
from threading import Event, Thread
from urllib.parse import urlsplit

from src.vision.live.service import AccessDenied, InvalidObservation, LiveDataStore, MAX_PREVIEW_BYTES


def handler_for(store: LiveDataStore):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            # Avoid logging bearer tokens, pair codes, or camera metadata.
            pass

        def _token(self):
            header = self.headers.get("Authorization", "")
            return header[7:] if header.startswith("Bearer ") else ""

        def _body(self):
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                raise ValueError("Invalid request size.") from None
            if length < 1 or length > 16384:
                raise ValueError("Aggregate request must be between 1 and 16384 bytes.")
            return json.loads(self.rfile.read(length))

        def _send(self, status, payload):
            body = json.dumps(payload, default=str).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_preview(self, preview):
            if preview is None:
                self.send_response(204)
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                return
            image, timestamp = preview
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Captured-At", timestamp.isoformat())
            self.send_header("Content-Length", str(len(image)))
            self.end_headers()
            self.wfile.write(image)

        def _dispatch(self, method):
            try:
                path = urlsplit(self.path).path
                token = self._token()
                if method == "GET" and path == "/health":
                    return self._send(200, {"status": "ok"})
                if method == "POST" and path == "/api/v1/pairing-codes":
                    return self._send(200, {"pairing_code": store.issue_pairing_code(token),
                                            "expires_in_seconds": store.pairing_ttl_s})
                if method == "POST" and path == "/api/v1/agents/pair":
                    body = self._body()
                    return self._send(200, store.pair(body["pairing_code"], name=body["name"],
                                                       location_id=body["location_id"],
                                                       source_type=body["source_type"],
                                                       active_counters=body["active_counters"]))
                if method == "POST" and path == "/api/v1/agents/heartbeat":
                    body = self._body()
                    return self._send(200, store.heartbeat(token, body["status"]))
                if method == "POST" and path == "/api/v1/cameras/observations":
                    item = store.ingest(token, self._body())
                    return self._send(200, {"observation_id": item.observation_id})
                if method == "GET" and path == "/api/v1/cameras":
                    return self._send(200, {"cameras": store.cameras(token)})
                parts = path.strip("/").split("/")
                if len(parts) == 5 and parts[:3] == ["api", "v1", "cameras"]:
                    camera_id, action = parts[3:]
                    if method == "GET" and action == "latest":
                        return self._send(200, store.latest(token, camera_id))
                    if method == "GET" and action == "history":
                        return self._send(200, {"observations": store.history(token, camera_id)})
                    if method == "GET" and action == "preview":
                        return self._send_preview(store.preview(token, camera_id))
                    if method == "POST" and action == "preview":
                        length = int(self.headers.get("Content-Length", "0"))
                        if not 100 <= length <= MAX_PREVIEW_BYTES:
                            raise ValueError("Preview must be a JPEG of at most 512 KiB.")
                        store.set_preview(token, camera_id, self.rfile.read(length),
                                          self.headers.get("X-Captured-At", ""))
                        return self._send(200, {"accepted": True})
                    if method == "POST" and action == "disconnect":
                        return self._send(200, store.disconnect(token, camera_id))
                    if method == "POST" and action == "remove":
                        return self._send(200, store.remove(token, camera_id))
                return self._send(404, {"error": "Endpoint not found."})
            except AccessDenied as exc:
                return self._send(403, {"error": str(exc)})
            except (InvalidObservation, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
                return self._send(400, {"error": str(exc) if isinstance(exc, (InvalidObservation, ValueError))
                                        else "Invalid aggregate request."})

        def do_GET(self):
            self._dispatch("GET")

        def do_POST(self):
            self._dispatch("POST")

    return Handler


def main(argv=None):
    parser = argparse.ArgumentParser(description="QueueMind aggregate-only live data API")
    parser.add_argument("--port", type=int, default=8780)
    parser.add_argument("--dev", action="store_true", help="Generate ephemeral local development tokens")
    args = parser.parse_args(argv)
    if args.dev:
        admin, viewer = token_urlsafe(32), token_urlsafe(32)
        print("LOCAL DEVELOPMENT ONLY · tokens expire when this service stops")
        print(f"Admin token: {admin}\nViewer token: {viewer}")
    else:
        admin = os.environ.get("QUEUEMIND_LIVE_ADMIN_TOKEN", "")
        viewer = os.environ.get("QUEUEMIND_LIVE_VIEW_TOKEN", "")
    store = LiveDataStore({"default": (admin, viewer)})
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler_for(store))
    stop_housekeeping = Event()
    def housekeeping():
        while not stop_housekeeping.wait(5):
            store.expire_previews()
    cleanup = Thread(target=housekeeping, daemon=True)
    cleanup.start()
    print(f"QueueMind aggregate service listening on http://127.0.0.1:{server.server_port}")
    print("For remote use, terminate HTTPS in a private reverse proxy and configure access controls.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stop_housekeeping.set()
        cleanup.join(timeout=2)
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
