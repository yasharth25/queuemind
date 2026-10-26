from datetime import datetime
from src.vision.events import EventDetector
from src.vision.tracking import TrackedPerson

def person(y): return TrackedPerson(1,(0,0,1,1),.9,(50,y))
def test_crossing_emits_one_anonymous_entry_event_with_cooldown():
    detector=EventDetector("bank",((0,50),(100,50)),((0,90),(100,90)),cooldown_frames=10)
    assert detector.update([person(40)]) == []
    assert [event.event_type for event in detector.update([person(60)],datetime(2026,1,1))] == ["ENTRY"]
    assert detector.update([person(40)]) == []
