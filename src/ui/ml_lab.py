"""ML Lab entrypoint; detailed sections are added in separate modules."""

import streamlit as st


def render_ml_lab(locations):
    st.markdown("### ML Lab · synthetic forecasting lifecycle")
    st.caption("Model evaluation runs only on explicit request and is cached for this demo.")
    if st.button("RUN / REFRESH ML EVALUATION", key="qm_run_ml"):
        st.session_state.qm_ml_ready = True
    if st.session_state.get("qm_ml_ready"):
        from src.ui.ml_service import evaluation_bundle

        with st.spinner("Evaluating held-out synthetic observations…"):
            bundle = evaluation_bundle(tuple(locations))
        st.success("Synthetic chronological evaluation available.")
        st.json(bundle["metrics"])
    else:
        st.info("Evaluation not run in this session. No result is fabricated.")
