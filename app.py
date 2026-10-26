import streamlit as st

from src.config import get_enabled_locations
from src.ui.dashboard import render_dashboard
from src.ui.demo import render_demo
from src.ui.evaluation import render_evaluation

st.set_page_config(
    page_title="QueueMind",
    page_icon="🧠",
    layout="wide"
)

page=st.sidebar.radio("QueueMind",["Operations Dashboard","Professor Demo Mode","Model Evaluation"])
if page == "Professor Demo Mode": render_demo()
elif page == "Model Evaluation": render_evaluation()
else: render_dashboard(get_enabled_locations())
