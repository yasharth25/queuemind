"""Per-camera normalized queue geometry; optional local click-to-mark editor."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from src.vision.config import load_vision_config, validate_line, validate_polygon
from src.vision.video_io import require_opencv


@dataclass(frozen=True)
class CameraCalibration:
    roi: tuple[tuple[float, float], ...]
    entry_line: tuple[tuple[float, float], tuple[float, float]]
    service_line: tuple[tuple[float, float], tuple[float, float]]

    def __post_init__(self):
        object.__setattr__(self, "roi", validate_polygon(self.roi))
        object.__setattr__(self, "entry_line", validate_line(self.entry_line))
        object.__setattr__(self, "service_line", validate_line(self.service_line))


def default_calibration() -> CameraCalibration:
    config = load_vision_config()
    return CameraCalibration(config["default_roi"], config["entry_line"],
                             config["service_line"])


def load_calibration(path: str | Path | None) -> CameraCalibration:
    if path is None:
        return default_calibration()
    source = Path(path)
    if not source.is_file():
        raise ValueError("Camera calibration file does not exist. Use --calibrate to create it.")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if set(payload) != {"roi", "entry_line", "service_line"}:
        raise ValueError("Calibration must contain only normalized ROI and two virtual lines.")
    return CameraCalibration(payload["roi"], payload["entry_line"], payload["service_line"])


def save_calibration(calibration: CameraCalibration, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {"roi": calibration.roi, "entry_line": calibration.entry_line,
               "service_line": calibration.service_line}
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return destination


def normalized_points(pixel_points, width: int, height: int):
    if width <= 0 or height <= 0:
        raise ValueError("Frame resolution must be positive.")
    return tuple((max(0, min(1, x / width)), max(0, min(1, y / height)))
                 for x, y in pixel_points)


def mark_calibration(frame) -> CameraCalibration:
    """Local OpenCV editor: click polygon, entry line, service line in sequence."""
    cv2 = require_opencv()
    height, width = frame.shape[:2]
    stages = (("Queue Area · click at least 3 points", 3),
              ("Entry Line · click 2 points", 2),
              ("Service/Exit Line · click 2 points", 2))
    selected = []
    window = "QueueMind · Mark Your Queue Area"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    try:
        for title, minimum in stages:
            points = []

            def click(event, x, y, flags, userdata):
                if event == cv2.EVENT_LBUTTONDOWN and (minimum != 2 or len(points) < 2):
                    points.append((x, y))

            cv2.setMouseCallback(window, click)
            while True:
                image = frame.copy()
                cv2.putText(image, title, (12, 27), cv2.FONT_HERSHEY_SIMPLEX,
                            .6, (255, 255, 255), 2)
                cv2.putText(image, "Enter: confirm  Backspace: undo  Esc: cancel", (12, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1)
                for point in points:
                    cv2.circle(image, point, 5, (0, 230, 255), -1)
                for first, second in zip(points, points[1:]):
                    cv2.line(image, first, second, (0, 230, 255), 2)
                cv2.imshow(window, image)
                key = cv2.waitKey(30) & 0xFF
                if key in (27,):
                    raise RuntimeError("Camera calibration cancelled.")
                if key in (8, 127) and points:
                    points.pop()
                if key in (10, 13) and len(points) >= minimum:
                    selected.append(normalized_points(points, width, height))
                    break
        return CameraCalibration(*selected)
    finally:
        cv2.destroyWindow(window)
