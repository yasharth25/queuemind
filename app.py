import streamlit as st

from src.config import get_enabled_locations
from src.ui.dashboard import render_dashboard
from src.ui.demo import render_demo

st.set_page_config(
    page_title="QueueMind",
    page_icon="🧠",
    layout="wide"
)

page=st.sidebar.radio("QueueMind",["Operations Dashboard","Professor Demo Mode"])
if page == "Professor Demo Mode": render_demo()
else: render_dashboard(get_enabled_locations())
