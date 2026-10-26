import streamlit as st

from src.config import get_enabled_locations
from src.ui.dashboard import render_dashboard
from src.ui.demo import render_demo
from src.ui.evaluation import render_evaluation
from src.ui.theme import apply_theme

st.set_page_config(
    page_title="QueueMind",
    page_icon="🧠",
    layout="wide"
)

apply_theme()

page=st.sidebar.radio("QueueMind",["Operations Dashboard","Professor Demo Mode","Model Evaluation"])
st.sidebar.caption("Academic demonstration · SYNTHETIC / SIMULATION data · Optional local vision is not enabled in cloud mode.")
if page == "Professor Demo Mode": render_demo()
elif page == "Model Evaluation": render_evaluation()
else: render_dashboard(get_enabled_locations())
