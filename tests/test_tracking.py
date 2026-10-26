from src.vision.detector import Detection
from src.vision.tracking import AnonymousTracker

def test_tracker_keeps_anonymous_id_across_overlapping_frames():
    tracker=AnonymousTracker()
    first=tracker.update([Detection((10,10,30,50),.9)])[0]
    second=tracker.update([Detection((12,10,32,50),.8)])[0]
    assert first.track_id == second.track_id
    assert second.representative_point == (22,50)
