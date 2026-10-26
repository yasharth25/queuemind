"""Presentation-oriented operations overview backed by QueueMind services."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from src.queueing.simulator import calculate_service_rate, estimate_wait_minutes, evaluate_counter_counts, get_congestion_status, recommend_counter_count, simulate_queue
from src.synthetic.generator import generate_synthetic_queue_data
from src.ui.components import action_card, header, metric
from src.ui.recommendations import recommendation_message

def _chart(history, projection):
 fig=go.Figure(); fig.add_trace(go.Scatter(x=history.index,y=history.queue_count,name="Observed / historical",line=dict(color="#4C8DFF",width=2)))
 now=history.index[-1]; future=[now+pd.Timedelta(minutes=r["minute"]) for r in projection]; fig.add_trace(go.Scatter(x=future,y=[r["queue"] for r in projection],name="Forecast / simulated",line=dict(color="#F3B44B",width=2,dash="dash")))
 fig.add_vline(x=now,line_color="#8E99A8",line_dash="dot"); fig.update_layout(height=330,margin=dict(l=8,r=8,t=20,b=8),paper_bgcolor="#11151A",plot_bgcolor="#11151A",font_color="#F4F6F8",legend=dict(orientation="h"),xaxis=dict(gridcolor="#222A33"),yaxis=dict(gridcolor="#222A33",title="People")); return fig

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
 st.markdown("#### Counter impact · 20-minute simulation")
 comparison=pd.DataFrame(evaluate_counter_counts(float(latest.queue_count),float(latest.arrival_rate),float(latest.average_service_time),[max(1,counters-1),counters,counters+1,recommendation["recommended_counters"]],20)); comparison["status"]=comparison.future_queue.apply(lambda x:get_congestion_status(x,location["warning_queue_length"],location["critical_queue_length"])); st.dataframe(comparison,hide_index=True,use_container_width=True)
 with st.expander("System status"): st.write("Synthetic simulation backend online. Vision is optional local functionality; no webcam, YOLO model, real data, or persisted forecast model is required.")
