"""QueueMind synthetic academic-demo entrypoint."""

import streamlit as st

from src.config import get_enabled_locations
from src.ui.command_center import render_command_center
from src.ui.command_theme import apply_command_theme


st.set_page_config(page_title="QueueMind · Command Center", page_icon="⏱", layout="wide")
apply_command_theme()
render_command_center(get_enabled_locations())
