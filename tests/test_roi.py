from src.vision.roi import point_in_polygon, classify_queue_people
from src.vision.tracking import TrackedPerson

def test_polygon_and_queue_classification():
    polygon=[(.1,.1),(.9,.1),(.9,.9),(.1,.9)]
    tracks=[TrackedPerson(1,(0,0,0,0),.9,(50,50)),TrackedPerson(2,(0,0,0,0),.9,(95,50))]
    assert point_in_polygon((50,50),[(10,10),(90,10),(90,90),(10,90)])
    assert not point_in_polygon((95,50),[(10,10),(90,10),(90,90),(10,90)])
    assert classify_queue_people(tracks,polygon,100,100)=={"total_detected_people":2,"queue_count":1,"queue_track_ids":[1]}
