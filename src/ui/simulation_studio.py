"""Simulation Studio entry page; deterministic what-if controls follow in V6."""

import streamlit as st


def render_simulation_studio(locations):
    st.title("What Happens If We Change Something?")
    st.caption("Simulation Studio tests hypothetical changes without altering camera observations.")
    st.info("No validated camera snapshot is available. The example below uses SYNTHETIC demo data.")
    from src.ui.operations_tabs import render_live_tab
    render_live_tab(locations)
