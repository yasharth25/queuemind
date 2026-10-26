import pandas as pd
from sklearn.metrics import mean_absolute_error
def queue_count_mae(labels: pd.DataFrame):
 required={"frame_id","ground_truth_queue_count","predicted_queue_count"}
 if not required.issubset(labels.columns): raise ValueError("Labels require frame_id, ground_truth_queue_count, predicted_queue_count.")
 if labels.empty: return None
 return round(float(mean_absolute_error(labels.ground_truth_queue_count,labels.predicted_queue_count)),4)
