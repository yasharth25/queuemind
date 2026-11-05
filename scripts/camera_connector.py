"""Local authorized camera setup; heavy vision dependencies are optional.

Run ``python -m scripts.camera_connector`` on the computer that can reach the
camera. RTSP credentials are requested interactively and never written to disk.
"""

from __future__ import annotations

import argparse
from getpass import getpass

from src.vision.live.camera_sources import (
    CameraSettings, SourceType, probe_camera, resolve_onvif_stream,
)
from src.vision.live.onvif_discovery import discover_onvif


def build_parser():
    parser = argparse.ArgumentParser(description="QueueMind local authorized camera connector")
    parser.add_argument("--source", choices=[item.value.lower() for item in SourceType])
    parser.add_argument("--name", default="My Camera")
    parser.add_argument("--location", default="bank")
    parser.add_argument("--host", default="")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--path", default="")
    parser.add_argument("--username", default="")
    parser.add_argument("--webcam-index", type=int, default=0)
    parser.add_argument("--replay", default="")
    parser.add_argument("--discover", action="store_true")
    parser.add_argument("--interface", default="", help="Local IPv4 address for ONVIF discovery")
    parser.add_argument("--test-only", action="store_true")
    return parser


def settings_from_args(args, *, ask=input, ask_secret=getpass) -> CameraSettings:
    source = args.source or ask("Source (webcam / rtsp / onvif / replay_video): ").strip().lower()
    try:
        kind = SourceType(source.upper())
    except ValueError as exc:
        raise ValueError("Choose webcam, rtsp, onvif, or replay_video.") from exc
    host = args.host
    path = args.path
    replay = args.replay
    if kind in (SourceType.RTSP, SourceType.ONVIF):
        host = host or ask("Camera IP address or hostname: ").strip()
        if kind == SourceType.RTSP:
            path = path or ask("RTSP stream path from your camera manual: ").strip()
    if kind == SourceType.REPLAY_VIDEO:
        replay = replay or ask("Local replay video path: ").strip()
    username = args.username
    if kind in (SourceType.RTSP, SourceType.ONVIF):
        username = username or ask("Username (blank if none): ").strip()
    password = ask_secret("Camera password (kept in memory only): ") if username else ""
    return CameraSettings(args.name, kind, args.location, host, args.port or
                          (80 if kind == SourceType.ONVIF else 554), path,
                          username, password, args.webcam_index, replay)


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.discover:
        interface = args.interface or input("Authorized local interface IPv4 address: ").strip()
        cameras = discover_onvif(interface)
        if not cameras:
            print("No supported cameras found. Try adding a camera manually.")
        for camera in cameras:
            print(f"ONVIF candidate: {camera.address} · stream unverified")
        return 0
    settings = settings_from_args(args)
    if settings.source_type == SourceType.ONVIF:
        settings = resolve_onvif_stream(settings)
    probe = probe_camera(settings)
    print(f"Validated {settings.name}: {probe.width}×{probe.height}, "
          f"{probe.frames_read} decoded frames, FPS {probe.fps or 'unavailable'}. "
          f"Source: {settings.source_type.value}.")
    if not args.test_only:
        print("Connection test complete. Continuous monitoring is configured separately.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
