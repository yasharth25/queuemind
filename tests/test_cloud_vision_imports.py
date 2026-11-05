"""Cloud requirements must import vision analysis/UI without cv2 or YOLO."""

import subprocess
import sys


def test_cloud_safe_vision_imports_and_missing_yolo_message():
    code = r'''
import builtins
real_import = builtins.__import__
def blocked(name, *args, **kwargs):
    if name in {"cv2", "ultralytics"}:
        raise ImportError("optional local package blocked")
    return real_import(name, *args, **kwargs)
builtins.__import__ = blocked
from src.vision.detector import Detection, PersonDetector
from src.vision.tracking import AnonymousTracker
from src.vision.events import EventDetector
from src.vision.roi import point_in_polygon, classify_queue_people
from src.vision.pipeline import VisionQueuePipeline
from src.vision.video_io import VideoMetadata
from src.vision.analyzer import VisionAnalysisResult
from src.ui.vision_lab import render_vision_lab
from src.vision.live.camera_sources import CameraConnection
from src.vision.live.calibration import CameraCalibration
from src.vision.live.monitor import QueueMonitor
from src.vision.live.forecast import project_live_queue
from src.vision.live.simulation import run_what_if
from src.vision.live.service import LiveDataStore
from src.vision.live.client import LiveAPIClient
from src.ui.live_camera import render_live_camera
from src.ui.simulation_studio import render_simulation_studio
try:
    PersonDetector()._load_model()
except RuntimeError as error:
    assert "requirements-vision.txt" in str(error)
else:
    raise AssertionError("Missing YOLO should provide a local-setup error")
print("cloud-safe vision imports passed")
'''
    completed = subprocess.run([sys.executable, "-c", code], capture_output=True,
                               text=True, timeout=30, check=False)
    assert completed.returncode == 0, completed.stderr
    assert "cloud-safe vision imports passed" in completed.stdout
