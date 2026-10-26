import pandas as pd
import streamlit as st
from src.demo.scenarios import SCENARIOS, calculate_demo
from src.ui.components import header, metric, action_card
def render_demo():
 header("Professor demo","Interactive Queue Scenario Simulator")
 st.warning("SIMULATION MODE — SYNTHETIC DATA")
 name=st.selectbox("Scenario",list(SCENARIOS)); settings=SCENARIOS[name]
 counters=st.slider("Active counters",1,8,settings["counters"]); intensity=st.slider("Arrival-demand intensity",.5,2.5,1.0,.1); speed=st.select_slider("Simulation speed",options=["1×","5×","20×"],value="1×")
 result=calculate_demo(name,counters,intensity); columns=st.columns(4)
 for column,(label,value,detail) in zip(columns,[("CURRENT QUEUE",settings["queue"],"Synthetic scenario"),("ESTIMATED WAIT",f"{result['current_wait']:.1f} min","Current state"),("ACTIVE COUNTERS",counters,"Interactive control"),("DEMAND SURGE",f"{intensity:.1f}×","Arrival intensity")]):
  with column: metric(label,value,detail)
 action_card(result["congestion"],f"Service rate {result['service_rate']:.2f}/min",result["recommendation"]["reason"])
 st.subheader("FORECAST / SIMULATED FUTURE")
 future=pd.DataFrame([{"minute":h,"expected_queue":q,"predicted_wait_minutes":round(q/result['service_rate'],2)} for h,q in result["projections"].items()]).set_index("minute"); st.line_chart(future[["expected_queue"]]); st.dataframe(future,use_container_width=True)
 st.subheader("Counter comparison (20-minute simulation)"); st.dataframe(pd.DataFrame(result["counter_comparison"]),hide_index=True,use_container_width=True)
 st.caption(f"Scenario speed setting: {speed}. All scenario inputs and outputs are clearly labelled SYNTHETIC.")
