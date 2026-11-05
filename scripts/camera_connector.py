"""Local authorized camera setup; heavy vision dependencies are optional.

Run ``python -m scripts.camera_connector`` on the computer that can reach the
camera. RTSP credentials are requested interactively and never written to disk.
"""

from __future__ import annotations

import argparse
from getpass import getpass
from time import sleep
from uuid import uuid4

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
    parser.add_argument("--preview", action="store_true", help="Show local annotated preview window")
    parser.add_argument("--frame-skip", type=int, default=2)
    parser.add_argument("--inference-interval", type=float, default=.5)
    parser.add_argument("--active-counters", type=int, default=None)
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
    if args.test_only:
        return 0
    from src.config import get_location_by_id
    from src.vision.annotation import annotate_frame
    from src.vision.config import load_vision_config
    from src.vision.live.camera_sources import CameraConnection, CameraStatus
    from src.vision.live.forecast import project_live_queue
    from src.vision.live.monitor import QueueMonitor, build_pipeline
    from src.vision.video_io import encode_preview, require_opencv

    config = load_vision_config()
    location = get_location_by_id(settings.location_id)
    pipeline = build_pipeline(location, roi=config["default_roi"],
                              entry_line=config["entry_line"],
                              service_line=config["service_line"],
                              confidence=float(config["confidence"]),
                              model_path=config["model_path"], stride=args.frame_skip)
    engine = QueueMonitor(str(uuid4()), CameraConnection(settings), pipeline, location,
                          active_counters=args.active_counters or location["default_counters"],
                          service_time_minutes=location["average_service_time_minutes"],
                          rate_window_s=int(config["rate_window_seconds"]),
                          min_inference_interval_s=args.inference_interval,
                          preview_encoder=encode_preview if args.preview else None,
                          annotator=annotate_frame if args.preview else None,
                          replay_fps=probe.fps if settings.source_type == SourceType.REPLAY_VIDEO else None)
    cv2 = require_opencv() if args.preview else None
    print(f"Monitoring {settings.name} · {settings.source_type.value}. "
          "ENTRY and SERVICE are boundary crossings, not verified transactions. Ctrl+C to stop.")
    engine.start()
    last_id = None
    try:
        while engine.status not in (CameraStatus.ERROR, CameraStatus.OFFLINE):
            sleep(.25)
            current = engine.latest()
            if current and current.observation_id != last_id:
                last_id = current.observation_id
                print(f"{current.captured_at.isoformat()} | queue {current.people_waiting} | "
                      f"detected {current.people_detected} | entries {current.entries_in_window} | "
                      f"service boundary {current.service_boundary_crossings} | "
                      f"arrival rate {current.observed_arrival_rate if current.observed_arrival_rate is not None else 'insufficient history'}")
                projection = project_live_queue(
                    current, location,
                    active_counters=args.active_counters or location["default_counters"],
                )
                if projection["available"]:
                    after20 = projection["points"][3]
                    print(f"  +20 min estimated queue {after20['queue']:.1f}, "
                          f"wait {after20['wait_minutes']:.1f} min, "
                          f"suggested counters {projection['recommendation']['recommended_counters']} "
                          f"({projection['method']})")
            if cv2 is not None and engine.preview is not None:
                import numpy as np
                image = cv2.imdecode(np.frombuffer(engine.preview, dtype=np.uint8), cv2.IMREAD_COLOR)
                if image is not None:
                    cv2.imshow("QueueMind · Local Camera Preview", image)
                    if cv2.waitKey(1) & 0xFF == 27:
                        break
    except KeyboardInterrupt:
        pass
    finally:
        engine.stop()
        if cv2 is not None:
            cv2.destroyAllWindows()
    if engine.last_error:
        print(engine.last_error)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
