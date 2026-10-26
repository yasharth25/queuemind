import pandas as pd
from src.evaluation.metrics import queue_count_mae
def test_queue_count_mae_uses_labelled_samples():
 assert queue_count_mae(pd.DataFrame({"frame_id":[1,2],"ground_truth_queue_count":[2,5],"predicted_queue_count":[3,4]}))==1.0
