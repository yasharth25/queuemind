"""Ultralytics person detection with no biometric or identity processing."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any

PERSON_CLASS_ID = 0

@dataclass(frozen=True)
class Detection:
    bounding_box: tuple[float, float, float, float]
    confidence: float
    class_id: int = PERSON_CLASS_ID

class PersonDetector:
    def __init__(self, model_path: str = "yolo11n.pt", confidence: float = .4, device: str | None = None, frame_skip: int = 1, model: Any | None = None):
        if not 0 < confidence <= 1 or frame_skip < 1:
            raise ValueError("Confidence must be in (0, 1] and frame_skip must be positive.")
        self.confidence, self.device, self.frame_skip = confidence, device, frame_skip
        self._model = model
        self.model_path = model_path

    def _load_model(self):
        if self._model is None:
            from ultralytics import YOLO
            self._model = YOLO(self.model_path)
        return self._model

    def detect(self, frame) -> list[Detection]:
        results = self._load_model()(frame, classes=[PERSON_CLASS_ID], conf=self.confidence, device=self.device, verbose=False)
        boxes = results[0].boxes
        if boxes is None:
            return []
        return [Detection(tuple(map(float, box.xyxy[0].tolist())), float(box.conf[0])) for box in boxes]

    def open_source(self, source: str | int):
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError(
                "OpenCV is required for local video/webcam vision. "
                "Install requirements-vision.txt."
            ) from exc
        capture = cv2.VideoCapture(source)
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"Video source is unavailable: {source}")
        return capture
