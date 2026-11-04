"""Local video metadata and streaming I/O; importing this module is cloud safe."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from contextlib import contextmanager
from tempfile import TemporaryDirectory
from math import isfinite

SUPPORTED_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv"}


def require_opencv():
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "Local Vision Mode requires OpenCV. Install requirements-vision.txt."
        ) from exc
    return cv2


@dataclass(frozen=True)
class VideoMetadata:
    name: str
    fps: float
    width: int
    height: int
    total_frames: int
    duration_s: float | None
    size_bytes: int | None


def open_video(path: str | Path):
    source = Path(path)
    if source.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise ValueError("Supported local videos: MP4, AVI, MOV, and MKV.")
    if not source.is_file():
        raise ValueError("Video file does not exist on this machine.")
    cv2 = require_opencv()
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        capture.release()
        raise ValueError("Video could not be opened. Check its codec and file integrity.")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if not isfinite(fps) or fps <= 0 or width <= 0 or height <= 0:
        capture.release()
        raise ValueError("Video has invalid FPS or resolution metadata.")
    metadata = VideoMetadata(
        source.name, fps, width, height, max(0, total),
        total / fps if total > 0 else None, source.stat().st_size,
    )
    return capture, metadata


def encode_preview(frame) -> bytes:
    cv2 = require_opencv()
    ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 83])
    if not ok:
        raise RuntimeError("Could not encode an annotated preview frame.")
    return encoded.tobytes()


@contextmanager
def uploaded_video_file(uploaded):
    """Materialize one upload only while it is inspected/analyzed, then remove it."""
    name = Path(uploaded.name).name
    if Path(name).suffix.lower() not in SUPPORTED_SUFFIXES:
        raise ValueError("Supported local videos: MP4, AVI, MOV, and MKV.")
    with TemporaryDirectory(prefix="queuemind_video_") as folder:
        path = Path(folder) / name
        with path.open("wb") as stream:
            stream.write(uploaded.getbuffer())
        yield path


def read_sample_frame(path: str | Path, frame_index: int):
    if frame_index < 0:
        raise ValueError("Sample frame index cannot be negative.")
    capture, metadata = open_video(path)
    cv2 = require_opencv()
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok:
            raise ValueError("Selected frame could not be decoded.")
        return frame, metadata
    finally:
        capture.release()
