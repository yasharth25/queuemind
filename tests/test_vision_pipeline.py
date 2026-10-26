import numpy as np
from src.forecasting.forecaster import ArrivalForecaster
from src.synthetic.generator import generate_synthetic_queue_data
from src.vision.detector import Detection
from src.vision.pipeline import VisionQueuePipeline
from src.vision.tracking import AnonymousTracker

class Detector:
    frame_skip=1
    def detect(self, frame): return [Detection((10,10,30,70),.9)]

LOCATION={"id":"bank","category":"bank","default_counters":2,"average_service_time_minutes":3,"target_wait_minutes":10}
def test_pipeline_links_vision_counts_to_safe_forecast_queue_path():
    pipeline=VisionQueuePipeline(Detector(),AnonymousTracker(),[(0,0),(1,0),(1,1),(0,1)],LOCATION,((0,20),(100,20)),((0,90),(100,90)),forecaster=ArrivalForecaster())
    result=pipeline.process_frame(np.zeros((100,100,3),dtype=np.uint8))
    assert result["queue"]["queue_count"] == 1
    history=generate_synthetic_queue_data([LOCATION],"2026-01-01",days=1)
    assert 20 in pipeline.forecast(history)
