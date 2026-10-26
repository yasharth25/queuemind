import streamlit as st

from src.config import get_enabled_locations
from src.ui.dashboard import render_dashboard

st.set_page_config(
    page_title="QueueMind",
    page_icon="🧠",
    layout="wide"
)

render_dashboard(get_enabled_locations())
