"""Presentation-oriented operations overview backed by QueueMind services."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from src.queueing.simulator import calculate_service_rate, estimate_wait_minutes, evaluate_counter_counts, get_congestion_status, recommend_counter_count, simulate_queue
from src.synthetic.generator import generate_synthetic_queue_data
from src.ui.components import action_card, header, impact_row, metric
from src.ui.recommendations import recommendation_message

def _chart(history, projection):
 fig=go.Figure(); fig.add_trace(go.Scatter(x=history.index,y=history.queue_count,name="Observed / historical",line=dict(color="#1E4F91",width=2),fill="tozeroy",fillcolor="rgba(30,79,145,.10)"))
 now=history.index[-1]; future=[now+pd.Timedelta(minutes=r["minute"]) for r in projection]; fig.add_trace(go.Scatter(x=future,y=[r["queue"] for r in projection],name="Forecast / simulated",line=dict(color="#F97360",width=2,dash="dash"),fill="tozeroy",fillcolor="rgba(249,115,96,.10)"))
 fig.add_vline(x=now,line_color="#1E3A5F",line_dash="dot"); fig.update_layout(height=330,margin=dict(l=8,r=8,t=20,b=8),paper_bgcolor="#FFF",plot_bgcolor="#FFF",font_color="#334155",legend=dict(orientation="h"),xaxis=dict(gridcolor="#E9EDF2"),yaxis=dict(gridcolor="#E9EDF2",title="People")); return fig

def render_dashboard(locations):
 top_left,top_right=st.columns([3,1]);
 with top_left: header("Campus queue intelligence","Current and future queue conditions from clearly labelled synthetic simulation data.")
 with top_right: location=st.selectbox("Location",locations,format_func=lambda x:x["display_name"]); st.caption("● SIMULATION MODE")
 data=generate_synthetic_queue_data([location],"2026-01-01",days=1,seed=42); latest=data.iloc[-1]
 if "counter_value" not in st.session_state: st.session_state.counter_value=int(location["default_counters"])
 counters=st.session_state.counter_value; horizon=20; projection=simulate_queue(float(latest.queue_count),float(latest.arrival_rate),float(latest.average_service_time),counters,horizon); future=projection[-1]["queue"]; service=calculate_service_rate(counters,float(latest.average_service_time)); current_wait=estimate_wait_minutes(float(latest.queue_count),counters,float(latest.average_service_time)); future_wait=estimate_wait_minutes(future,counters,float(latest.average_service_time)); status=get_congestion_status(future,location["warning_queue_length"],location["critical_queue_length"]); recommendation=recommend_counter_count(float(latest.queue_count),float(latest.arrival_rate),float(latest.average_service_time),counters,float(location["target_wait_minutes"]),horizon,list(range(1,9))); message=recommendation_message(float(latest.queue_count),current_wait,future_wait,float(location["target_wait_minutes"]),counters,recommendation["recommended_counters"],float(latest.arrival_rate),service)
 cols=st.columns(4)
 with cols[0]:metric("CURRENT QUEUE",int(latest.queue_count),f"{future-int(latest.queue_count):+.1f} expected in 20 min")
 with cols[1]:metric("ESTIMATED WAIT",f"{current_wait:.1f} min",f"Target: {location['target_wait_minutes']} min")
 with cols[2]:metric("+20 MIN QUEUE",f"{future:.1f}","Simulation projection")
 with cols[3]:metric("ACTIVE COUNTERS",counters,f"Recommended: {recommendation['recommended_counters']}")
 left,right=st.columns([2,1]);
 with left: st.markdown("#### Queue forecast"); st.plotly_chart(_chart(data.tail(72).set_index("timestamp"),projection),use_container_width=True,config={"displayModeBar":False})
 with right: action_card(status,"Predicted wait is %.1f min at +20 minutes."%future_wait,message); st.slider("Active counters",1,8,counters,key="counter_value"); st.caption(f"Arrival rate {latest.arrival_rate:.2f}/min · Service rate {service:.2f}/min")
 st.markdown(f"#### Counter configuration impact · 20-minute simulation")
 st.caption(f"Arrival rate {latest.arrival_rate:.2f}/min  ·  Service rate {service:.2f}/min  ·  Select a configuration to recalculate")
 comparison=evaluate_counter_counts(float(latest.queue_count),float(latest.arrival_rate),float(latest.average_service_time),list(range(max(1,counters-1),min(8,counters+2)+1))+[recommendation["recommended_counters"]],20)
 max_queue=max(item["future_queue"] for item in comparison); max_wait=max(item["predicted_wait_minutes"] for item in comparison)
 for item in comparison:
  item_status=get_congestion_status(item["future_queue"],location["warning_queue_length"],location["critical_queue_length"]); st.markdown(impact_row(item,item_status,item["active_counters"]==counters,item["active_counters"]==recommendation["recommended_counters"],max_queue,max_wait),unsafe_allow_html=True)
  button_label="Current" if item["active_counters"]==counters else "Select"
  if st.button(button_label,key=f"counter_{item['active_counters']}",disabled=item["active_counters"]==counters): st.session_state.counter_value=item["active_counters"]; st.rerun()
 with st.expander("System status"): st.write("Synthetic simulation backend online. Vision is optional local functionality; no webcam, YOLO model, real data, or persisted forecast model is required.")
