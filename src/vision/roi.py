"""Normalized queue-region geometry and annotation helpers."""
import cv2
import numpy as np
from src.vision.tracking import TrackedPerson

def point_in_polygon(point, polygon):
    x,y=point; inside=False
    for i,(x1,y1) in enumerate(polygon):
        x2,y2=polygon[(i+1)%len(polygon)]
        if (y1>y)!=(y2>y) and x < (x2-x1)*(y-y1)/(y2-y1)+x1: inside=not inside
    return inside

def classify_queue_people(tracks: list[TrackedPerson], normalized_polygon, frame_width, frame_height):
    polygon=[(x*frame_width,y*frame_height) for x,y in normalized_polygon]
    queued=[track for track in tracks if point_in_polygon(track.representative_point, polygon)]
    return {"total_detected_people":len(tracks),"queue_count":len(queued),"queue_track_ids":[item.track_id for item in queued]}

def annotate_roi(frame, tracks, normalized_polygon):
    height,width=frame.shape[:2]; polygon=np.array([(int(x*width),int(y*height)) for x,y in normalized_polygon], np.int32)
    cv2.polylines(frame,[polygon],True,(255,255,0),2)
    ids=classify_queue_people(tracks,normalized_polygon,width,height)["queue_track_ids"]
    for track in tracks:
        x1,y1,x2,y2=map(int,track.bounding_box); color=(0,255,0) if track.track_id in ids else (0,0,255)
        cv2.rectangle(frame,(x1,y1),(x2,y2),color,2); cv2.putText(frame,str(track.track_id),(x1,y1-4),cv2.FONT_HERSHEY_SIMPLEX,.5,color,1)
    return frame
