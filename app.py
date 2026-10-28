import streamlit as st
from src.config import get_enabled_locations
from src.ui.evaluation import render_evaluation
from src.ui.theme import apply_theme
from src.ui.v2 import command_center, counter_optimizer, forecast_page, scenario_lab, vision_lab

st.set_page_config(page_title="QueueMind",page_icon="⏱",layout="wide")
apply_theme()
st.sidebar.markdown("<h2 style='color:#1E3A5F'>◷ QueueMind</h2><p style='color:#6B7280'>See now. Predict next. Act early.</p>",unsafe_allow_html=True)
page=st.sidebar.radio("Navigate",["Command Center","Queue Forecast","Scenario Lab","Counter Optimizer","Vision Lab","Model Evaluation"])
st.sidebar.markdown("---\n**SIMULATION**  \nSynthetic data")
locations=get_enabled_locations()
if page=="Command Center": command_center(locations)
elif page=="Queue Forecast": forecast_page(locations)
elif page=="Scenario Lab": scenario_lab()
elif page=="Counter Optimizer": counter_optimizer(locations)
elif page=="Vision Lab": vision_lab()
else: render_evaluation()
st.caption("QueueMind · Academic Prototype · Simulation data where labelled.")
