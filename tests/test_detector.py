from src.vision.detector import PersonDetector

class Box:
    def __init__(self): self.xyxy=[type("X", (), {"tolist":lambda s:[1,2,3,4]})()]; self.conf=[.8]
class Result: boxes=[Box()]
class Model:
    def __call__(self, *args, **kwargs): return [Result()]

def test_person_detector_returns_structured_detections_without_yolo_download():
    values = PersonDetector(model=Model()).detect(object())
    assert values[0].bounding_box == (1.0, 2.0, 3.0, 4.0)
    assert values[0].confidence == .8
