import pandas as pd
import streamlit as st
from src.config import get_enabled_locations
from src.forecasting.forecaster import ArrivalForecaster
from src.synthetic.generator import generate_synthetic_queue_data
from src.ui.components import header
def render_evaluation():
 header("Model evaluation","Synthetic held-out chronological test data")
 st.caption("Synthetic evaluation — not real NIT Rourkela performance.")
 if st.button("Calculate forecasting evaluation"):
  data=generate_synthetic_queue_data(get_enabled_locations(),"2026-01-01",days=8,seed=9); metrics=ArrivalForecaster().train(data)
  rows=[]
  for horizon,value in metrics.items(): rows.append({"horizon":f"+{horizon} min","baseline MAE":value["baseline"]["mae"],"ML MAE":value["ml"]["mae"],"baseline RMSE":value["baseline"]["rmse"],"ML RMSE":value["ml"]["rmse"],"baseline R²":value["baseline"]["r2"],"ML R²":value["ml"]["r2"]})
  table=pd.DataFrame(rows); st.dataframe(table,hide_index=True,use_container_width=True); st.bar_chart(table.set_index("horizon")[["baseline MAE","ML MAE"]]); st.success("Lower MAE/RMSE is better; R² measures explained variation. Results shown are calculated on unseen chronological synthetic samples.")
 st.subheader("Computer Vision Evaluation"); st.info("Computer vision has not yet been quantitatively evaluated on real queue footage."); st.code("frame_id,ground_truth_queue_count,predicted_queue_count")
 st.subheader("Operational Simulation Evaluation"); st.caption("Queue and wait outputs are deterministic simulation results, not measured ground truth. Counter recommendations are derived from those simulations.")
 st.subheader("Privacy / Responsible AI"); st.write("QueueMind uses temporary anonymous tracking IDs only. It does not perform face recognition, student identification, or biometric identification. Raw and processed video retention are disabled by default. Campus CCTV deployment requires institutional authorization and applicable privacy-policy compliance.")
