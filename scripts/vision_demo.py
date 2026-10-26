"""Run person-only QueueMind vision on a local/uploaded video or webcam index."""
import argparse
import cv2
from src.vision.detector import PersonDetector

parser = argparse.ArgumentParser()
parser.add_argument("source", help="Video path or webcam index, for example 0")
parser.add_argument("--model", default="yolo11n.pt")
parser.add_argument("--confidence", type=float, default=.4)
args = parser.parse_args()
source = int(args.source) if args.source.isdigit() else args.source
detector = PersonDetector(args.model, args.confidence)
capture = detector.open_source(source)
try:
    while True:
        ok, frame = capture.read()
        if not ok: break
        for item in detector.detect(frame):
            x1, y1, x2, y2 = map(int, item.bounding_box)
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.imshow("QueueMind person-only detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"): break
finally:
    capture.release(); cv2.destroyAllWindows()
