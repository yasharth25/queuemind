"""Lightweight anonymous multi-object tracking for detector outputs."""
from dataclasses import dataclass
from src.vision.detector import Detection

@dataclass(frozen=True)
class TrackedPerson:
    track_id: int
    bounding_box: tuple[float,float,float,float]
    confidence: float
    representative_point: tuple[float,float]

def _iou(a, b):
    x1,y1,x2,y2 = max(a[0],b[0]),max(a[1],b[1]),min(a[2],b[2]),min(a[3],b[3])
    overlap=max(0,x2-x1)*max(0,y2-y1); union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-overlap
    return overlap/union if union else 0

class AnonymousTracker:
    """Deterministic IoU tracker; IDs are temporary anonymous session IDs."""
    def __init__(self, iou_threshold=.3, max_missed=10): self.iou_threshold,self.max_missed,self._tracks,self._next=iou_threshold,max_missed,{},1
    def update(self, detections: list[Detection]) -> list[TrackedPerson]:
        unmatched=set(self._tracks); assigned=[]
        for detection in detections:
            candidates=[( _iou(detection.bounding_box, state["box"]), ident) for ident,state in self._tracks.items() if ident in unmatched]
            score, ident=max(candidates, default=(0,None))
            if score < self.iou_threshold:
                ident=self._next; self._next+=1; self._tracks[ident]={"box":detection.bounding_box,"missed":0}
            else: unmatched.remove(ident); self._tracks[ident]={"box":detection.bounding_box,"missed":0}
            x1,y1,x2,y2=detection.bounding_box
            assigned.append(TrackedPerson(ident,detection.bounding_box,detection.confidence,((x1+x2)/2,y2)))
        for ident in unmatched: self._tracks[ident]["missed"]+=1
        self._tracks={ident:state for ident,state in self._tracks.items() if state["missed"]<=self.max_missed}
        return assigned
