"""Plain-language evidence view with the existing diagnostics kept on demand."""

import streamlit as st

from src.ui.command_model import snapshot
from src.ui.feature_help import feature_heading


def render_results(locations) -> None:
    feature_heading("Results", "What has been measured, and what still needs real-world testing.")
    st.markdown("#### How Well Does QueueMind Predict?")
    st.caption("TESTED ON SYNTHETIC DATA · Future arrival counts, not real campus measurements")
    if not st.session_state.get("qm_ml_ready"):
        st.info("Prediction evaluation has not run in this session.")
        if st.button("Run Synthetic Data Evaluation", key="qm_results_evaluate"):
            st.session_state.qm_ml_ready = True
            st.rerun()
    else:
        from src.ui.ml_service import evaluation_bundle
        from src.ui.ml_lab import metric_table

        try:
            metrics = evaluation_bundle(tuple(locations))["metrics"]
        except (ValueError, RuntimeError, ImportError) as exc:
            st.warning(f"Prediction evaluation is unavailable: {exc}")
        else:
            horizon = st.select_slider("Look ahead", [5, 10, 20, 30], value=20,
                                       format_func=lambda value: f"{value} minutes", key="qm_results_horizon")
            record = metrics[horizon]
            ml_error, basic_error = record["ml"]["mae"], record["baseline"]["mae"]
            left, right, difference = st.columns(3)
            left.metric("ML Average Prediction Error", f"{ml_error:.2f} arrivals", help="Mean absolute error on held-out synthetic observations.")
            right.metric("Simple Method Error", f"{basic_error:.2f} arrivals", help="The same test set, using a rolling recent-arrivals baseline.")
            difference.metric("ML Compared with Simple Method", f"{basic_error - ml_error:+.2f} arrivals", help="Positive means ML had lower error; negative means it did worse.")
            st.caption("A lower average error is better. This comparison is for synthetic future arrivals, not video counting accuracy.")
            with st.expander("Advanced Prediction Results"):
                st.dataframe(metric_table(metrics), hide_index=True, width="stretch")

    st.markdown("#### How Much Could Another Counter Help?")
    if st.session_state.get("qm_state_source") == "VISION" and st.session_state.get("qm_vision_state") is not None:
        from src.config import get_location_by_id
        from src.vision.integration import project_vision_state
        state = st.session_state.qm_vision_state
        recommendation = project_vision_state(state, get_location_by_id(state.location_id))["recommendation"]
        st.caption(("VIDEO COUNT + SYNTHETIC DEMO DEMAND" if state.demand_source == "SYNTHETIC_ASSUMPTION"
                    else "VIDEO COUNT + OBSERVED ENTRY CROSSINGS")
                   + " · Configured service settings · Simulated change, not a measured intervention")
    else:
        recommendation = snapshot(st.session_state.qm_live)["recommendation"]
        st.caption("SYNTHETIC DEMO · Simulated change, not a measured intervention")
    before, after = recommendation["current_outcome"], recommendation["recommended_outcome"]
    left, right = st.columns(2)
    left.metric(f'{before["active_counters"]} Current Counters', f'{before["predicted_wait_minutes"]:.1f} min wait')
    right.metric(f'{after["active_counters"]} Suggested Counters', f'{after["predicted_wait_minutes"]:.1f} min wait')
    st.caption("The optimizer chooses the smallest tested counter count that reaches the configured wait target, when one is available.")

    st.markdown("#### How Accurately Does Video Count People?")
    vision = st.session_state.get("qm_vision_evaluation")
    if vision is None:
        st.info("Real-Video Accuracy Has Not Been Measured Yet. Manually labelled frame counts are needed.")
    else:
        st.metric("Average Counting Error", f'{vision["queue_count_mae"]:.2f} people')
        st.caption(f'Computed from {vision["labelled_frames"]} user-labelled frames; recording provenance is not verified.')
    with st.expander("Advanced Details for Professors"):
        if st.checkbox("Open ML Lab and diagnostics", key="qm_results_show_ml"):
            from src.ui.ml_lab import render_ml_lab
            render_ml_lab(locations)
        if st.checkbox("Open vision evaluation and exports", key="qm_results_show_vision"):
            from src.ui.vision_lab import render_vision_evaluation
            render_vision_evaluation()
