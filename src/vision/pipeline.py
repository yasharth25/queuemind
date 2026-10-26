"""Composable vision-to-queue intelligence pipeline; no model training occurs here."""
from __future__ import annotations
from datetime import datetime
from src.forecasting.queue_integration import forecast_queue_outcomes
from src.vision.events import EventDetector
from src.vision.roi import classify_queue_people

class VisionQueuePipeline:
    def __init__(self, detector, tracker, normalized_roi, location, entry_line, service_line, repository=None, forecaster=None):
        self.detector,self.tracker,self.roi,self.location=detector,tracker,normalized_roi,location
        self.events=EventDetector(location["id"],entry_line,service_line)
        self.repository,self.forecaster=repository,forecaster
        self.frame_number=0
    def process_frame(self, frame, timestamp: datetime | None=None):
        self.frame_number+=1
        if (self.frame_number-1) % self.detector.frame_skip: return {"processed":False}
        tracks=self.tracker.update(self.detector.detect(frame))
        height,width=frame.shape[:2]
        roi=classify_queue_people(tracks,self.roi,width,height)
        events=self.events.update(tracks,timestamp)
        if self.repository: self.events.persist(events,self.repository)
        return {"processed":True,"tracks":tracks,"queue":roi,"events":events}
    def forecast(self, observation_history):
        if self.forecaster is None: raise RuntimeError("A forecaster is required for demand and queue projections.")
        return forecast_queue_outcomes(self.forecaster,self.location,observation_history)
