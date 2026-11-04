"""Anonymous virtual-line queue events with crossing de-duplication."""
from dataclasses import dataclass
from datetime import datetime
from src.vision.tracking import TrackedPerson

@dataclass(frozen=True)
class QueueEvent:
    timestamp: str; location_id: str; track_id: int; event_type: str

def _side(point, line):
    (x1,y1),(x2,y2)=line; return (x2-x1)*(point[1]-y1)-(y2-y1)*(point[0]-x1)


def _directed_segment_crossing(prior, current, line):
    """Count negative-to-positive crossings of the finite oriented line only."""
    first, second = _side(prior, line), _side(current, line)
    if not first < 0 < second:
        return False
    fraction = -first / (second - first)
    crossing = (prior[0] + fraction * (current[0] - prior[0]),
                prior[1] + fraction * (current[1] - prior[1]))
    (x1, y1), (x2, y2) = line
    length_sq = (x2 - x1) ** 2 + (y2 - y1) ** 2
    if length_sq <= 0:
        return False
    along = ((crossing[0] - x1) * (x2 - x1) +
             (crossing[1] - y1) * (y2 - y1)) / length_sq
    return 0 <= along <= 1

class EventDetector:
    def __init__(self, location_id, entry_line, service_line, cooldown_frames=15):
        self.location_id,self.entry_line,self.service_line,self.cooldown_frames=location_id,entry_line,service_line,cooldown_frames; self.previous={}; self.last={}; self.frame=0
    def update(self, tracks: list[TrackedPerson], timestamp: datetime | None=None):
        self.frame+=1; now=(timestamp or datetime.now()).isoformat(); events=[]
        for track in tracks:
            prior=self.previous.get(track.track_id); self.previous[track.track_id]=track.representative_point
            if prior is None: continue
            for kind,line in (("ENTRY",self.entry_line),("SERVICE_EXIT",self.service_line)):
                key=(track.track_id,kind)
                if _directed_segment_crossing(prior,track.representative_point,line) and self.frame-self.last.get(key,-999)>self.cooldown_frames:
                    self.last[key]=self.frame; events.append(QueueEvent(now,self.location_id,track.track_id,kind))
        return events
    @staticmethod
    def persist(events, repository):
        for event in events: repository.save_event(event.timestamp,str(event.track_id),event.location_id,event.event_type,"VISION")
