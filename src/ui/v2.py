"""QueueMind V2 product pages built on existing simulation services."""
from __future__ import annotations
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from src.demo.scenarios import SCENARIOS, calculate_demo
from src.forecasting.queue_integration import forecast_queue_outcomes
from src.forecasting.forecaster import ArrivalForecaster
from src.queueing.simulator import estimate_wait_minutes, evaluate_counter_counts, get_congestion_status, recommend_counter_count, simulate_queue
from src.synthetic.generator import generate_synthetic_queue_data
from src.ui.components import header, metric, impact_row
from src.ui.recommendations import recommendation_message

def _data(location): return generate_synthetic_queue_data([location],"2026-01-01",days=1,seed=42)
def location_state(location,counters=None):
 data=_data(location); row=data.iloc[-1]; counters=counters or location["default_counters"]; projection=simulate_queue(row.queue_count,row.arrival_rate,row.average_service_time,counters,20); queue=projection[-1]["queue"]; wait=estimate_wait_minutes(queue,counters,row.average_service_time); recommendation=recommend_counter_count(row.queue_count,row.arrival_rate,row.average_service_time,counters,location["target_wait_minutes"],20,list(range(1,9))); return {"data":data,"row":row,"projection":projection,"queue20":queue,"wait20":wait,"counters":counters,"risk":get_congestion_status(queue,location["warning_queue_length"],location["critical_queue_length"]),"recommendation":recommendation}
def alerts(locations):
 values=[]
 for location in locations:
  state=location_state(location); severity={"critical":2,"warning":1,"normal":0}[state["risk"]]
  if severity: values.append({"location":location["display_name"],"risk":state["risk"],"queue":state["queue20"],"wait":state["wait20"],"recommendation":state["recommendation"]["recommended_counters"]})
 return sorted(values,key=lambda item:{"critical":2,"warning":1,"normal":0}[item["risk"]],reverse=True)
def command_center(locations):
 header("Command Center","See now. Predict next. Act early.")
 st.caption("CAMPUS QUEUE NETWORK · Schematic · Not to scale · SYNTHETIC / SIMULATION")
 selected=st.selectbox("Facility detail",locations,format_func=lambda x:x["display_name"],key="cc_location")
 nodes=st.columns(len(locations))
 for col,location in zip(nodes,locations):
  state=location_state(location)
  with col: metric(location["display_name"],f"{int(state['row'].queue_count)} people",f"+20m {state['queue20']:.0f} · {state['risk'].upper()}")
 state=location_state(selected); cols=st.columns(4)
 for col,(label,value,detail) in zip(cols,[("CURRENT QUEUE",int(state["row"].queue_count),"Observed synthetic state"),("ESTIMATED WAIT",f"{estimate_wait_minutes(state['row'].queue_count,state['counters'],state['row'].average_service_time):.1f} min",f"Target: {selected['target_wait_minutes']} min"),("+20 MIN QUEUE",f"{state['queue20']:.1f}",state['risk'].upper()),("ACTIVE COUNTERS",state['counters'],f"Recommended: {state['recommendation']['recommended_counters']}")]):
  with col: metric(label,value,detail)
 left,right=st.columns([2,1])
 with left:
  st.subheader("Live queue visualizer")
  dots="● "*min(48,int(state["row"].queue_count)); st.markdown(f'<div class="card"><b>COUNTERS</b> &nbsp; {"● "*state["counters"]}<hr><div style="font-size:22px;color:#1E3A5F;letter-spacing:5px">{dots or "Queue clear"}</div><p class="metric-detail">Anonymous queue representation · derived from current synthetic queue state</p></div>',unsafe_allow_html=True)
 with right:
  st.subheader("Predictive alerts")
  current=alerts(locations)
  if current:
   for alert in current: st.markdown(f'<div class="card"><span class="pill {alert["risk"]}">{alert["risk"].upper()}</span><br><b>{alert["location"]}</b><br><span class="metric-detail">+20m: {alert["queue"]:.0f} people · {alert["wait"]:.1f} min<br>Recommended: {alert["recommendation"]} counters</span></div>',unsafe_allow_html=True)
  else: st.success("No predictive congestion alerts. All configured locations are within simulation thresholds.")
def forecast_page(locations):
 location=st.selectbox("Location",locations,format_func=lambda x:x["display_name"],key="forecast_location"); state=location_state(location); header(f"Queue Forecast · {location['display_name']}","Historical synthetic observations and deterministic future simulation.")
 outcomes=forecast_queue_outcomes(ArrivalForecaster(),location,state["data"])
 timeline=st.columns(5); now_queue=state["row"].queue_count
 entries=[("NOW",now_queue,estimate_wait_minutes(now_queue,state["counters"],state["row"].average_service_time),state["counters"])] + [(f"+{h}",item["expected_queue"],item["predicted_wait_minutes"],item["counter_recommendation"]["recommended_counters"]) for h,item in outcomes.items()]
 for col,(name,queue,wait,count) in zip(timeline,entries):
  with col: metric(name,f"{queue:.0f}",f"{wait:.1f}m · {count} counters")
 fig=go.Figure(); history=state["data"].tail(72); fig.add_trace(go.Scatter(x=history.timestamp,y=history.queue_count,name="Observed",line=dict(color="#1E4F91"))); fig.add_trace(go.Scatter(x=[history.timestamp.iloc[-1]+pd.Timedelta(minutes=r["minute"]) for r in state["projection"]],y=[r["queue"] for r in state["projection"]],name="Forecast",line=dict(color="#F97360",dash="dash"))); fig.update_layout(title="Queue forecast",paper_bgcolor="#FFF",plot_bgcolor="#FFF",height=400); st.plotly_chart(fig,use_container_width=True)
 pressure=state["row"].arrival_rate-(state["counters"]/state["row"].average_service_time); st.info("People are currently arriving faster than they can be served." if pressure>0 else "Current service capacity meets observed synthetic demand.")
def scenario_lab():
 header("Scenario Lab","Interactive Queue Simulation · Professor Demo")
 name=st.radio("Choose scenario",list(SCENARIOS),horizontal=True); settings=SCENARIOS[name]
 if "lab_playing" not in st.session_state: st.session_state.lab_playing=False
 controls=st.columns(5); intensity=controls[0].slider("Demand intensity",.5,2.5,1.0,.1); counters=controls[1].slider("Counters",1,8,settings["counters"]); speed=controls[2].selectbox("Speed",["1×","2×","5×","10×"]); controls[3].button("Play",on_click=lambda:st.session_state.update(lab_playing=True)); controls[4].button("Pause",on_click=lambda:st.session_state.update(lab_playing=False))
 result=calculate_demo(name,counters,intensity); metric_cols=st.columns(4)
 for col,(label,value) in zip(metric_cols,[("QUEUE",settings["queue"]),("WAIT",f"{result['current_wait']:.1f}m"),("+30M QUEUE",result["projections"][30]),("RISK",result["congestion"].upper())]):
  with col: metric(label,value,"SYNTHETIC SIMULATION")
 st.write("**Story**  1. Normal operation → 2. Demand surge → 3. Future congestion → 4. Recommendation → 5. Re-simulate with action")
 st.download_button("Download scenario results",pd.DataFrame([{"time":h,"queue":q,"wait":q/result["service_rate"],"arrival_rate":result["arrival_rate"],"service_rate":result["service_rate"],"counters":counters,"risk":result["congestion"]} for h,q in result["projections"].items()]).to_csv(index=False),"queuemind_scenario.csv","text/csv")
def counter_optimizer(locations):
 location=st.selectbox("Location",locations,format_func=lambda x:x["display_name"],key="optimizer_location"); counters=st.slider("Current counters",1,8,location["default_counters"],key="optimizer_counter"); state=location_state(location,counters); header("Counter Optimizer","Compare simulated capacity options before congestion builds.")
 st.write(f"Target wait: **{location['target_wait_minutes']} min** · Arrival rate: **{state['row'].arrival_rate:.2f}/min**")
 options=evaluate_counter_counts(state["row"].queue_count,state["row"].arrival_rate,state["row"].average_service_time,list(range(1,7)),20); cols=st.columns(3)
 for index,item in enumerate(options):
  status=get_congestion_status(item["future_queue"],location["warning_queue_length"],location["critical_queue_length"])
  with cols[index%3]: impact_row(item,status,item["active_counters"]==counters,item["active_counters"]==state["recommendation"]["recommended_counters"],max(x["future_queue"] for x in options),max(x["predicted_wait_minutes"] for x in options)); st.markdown(impact_row(item,status,item["active_counters"]==counters,item["active_counters"]==state["recommendation"]["recommended_counters"],max(x["future_queue"] for x in options),max(x["predicted_wait_minutes"] for x in options)),unsafe_allow_html=True)
 with st.expander("Why this recommendation?"): st.write(state["recommendation"]["reason"])
def vision_lab():
 header("Vision Lab","Optional local-only anonymous queue analytics")
 st.info("VISION MODULE · Available in local mode. Cloud simulation does not load YOLO, a webcam, or local video.")
 st.graphviz_chart("digraph {Video -> Detection -> Tracking -> ROI -> Events -> Statistics}")
 st.write("No face recognition. No identity recognition. Temporary anonymous tracking IDs only.")
