"""Inspectable ML lifecycle, with training isolated from ordinary page loads."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import HistGradientBoostingRegressor

from src.forecasting.features import FEATURE_COLUMNS, build_features, chronological_split
from src.forecasting.forecaster import DEFAULT_HORIZONS
from src.ui.ml_service import evaluation_bundle, synthetic_evaluation_data


SECTIONS = ("Overview", "Data", "Features", "Training", "Models", "Predictions",
            "Metrics", "Errors", "Importance", "Uncertainty", "Integrity")


def _bundle(locations):
    if not st.session_state.get("qm_ml_ready"):
        st.info("Evaluation has not run in this session. Select RUN / REFRESH on ML Overview; no metric is fabricated.")
        return None
    with st.spinner("Loading cached synthetic evaluation…"):
        return evaluation_bundle(tuple(locations))


def metric_table(metrics):
    rows = []
    for horizon in DEFAULT_HORIZONS:
        for method in ("baseline", "ml"):
            record = metrics[horizon]
            rows.append({"Horizon": f"+{horizon} min",
                         "Method": "Rolling baseline" if method == "baseline" else "HistGradientBoosting",
                         "MAE": record[method]["mae"], "RMSE": record[method]["rmse"],
                         "R²": record[method]["r2"], "Test rows": record["test_samples"]})
    return pd.DataFrame(rows)


def _overview(locations, data):
    st.markdown("### ML Lab · synthetic forecasting lifecycle")
    st.caption("SYNTHETIC observations → past-only features → chronological split → rolling baseline + HistGradientBoosting → future arrivals → queue simulation → wait estimate → counter optimization")
    left, right = st.columns(2)
    with left:
        st.markdown("#### Model card")
        st.dataframe(pd.DataFrame({"Property": ["Algorithm", "Task", "Horizons", "Dataset", "Evaluation", "Fallback", "Real-campus claim"],
                                   "Value": ["HistGradientBoostingRegressor", "Future interval arrival-count regression", "5 / 10 / 20 / 30 min", "SYNTHETIC queue observations", "Held-out chronological test", "Past-three-arrivals rolling mean", "None"]}), hide_index=True, width="stretch")
    with right:
        st.markdown("#### Evidence snapshot")
        st.metric("Synthetic observations", f"{len(data):,}")
        st.metric("Configured facilities", data.location_id.nunique())
        st.metric("Actual model features", len(FEATURE_COLUMNS))
    if st.button("RUN / REFRESH ML EVALUATION", key="qm_run_ml", type="primary"):
        st.session_state.qm_ml_ready = True
    bundle = _bundle(locations)
    if bundle:
        st.success("Held-out SYNTHETIC evaluation is cached for this session.")
        st.dataframe(metric_table(bundle["metrics"]), hide_index=True, width="stretch")


def _training(locations, data):
    st.markdown("### Training · chronological split")
    featured = build_features(data)
    partitions = chronological_split(featured)
    rows = [{"Partition": name, "Start": frame.timestamp.min(), "End": frame.timestamp.max(),
             "Rows": len(frame), "Share": f"{len(frame)/len(featured):.1%}"}
            for name, frame in zip(("Train", "Validation", "Test"), partitions)]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    fig = px.timeline(pd.DataFrame(rows), x_start="Start", x_end="End", y="Partition", color="Partition",
                      color_discrete_map={"Train": "#173B63", "Validation": "#6B4EAA", "Test": "#F36F5D"})
    fig.update_layout(height=220, showlegend=False, yaxis_autorange="reversed")
    st.plotly_chart(fig, width="stretch")
    st.info("No random shuffling. All facilities at a timestamp share one partition. Train and validation are combined for the final fit. Future targets crossing the test boundary are excluded from fitting.")
    bundle = _bundle(locations)
    if bundle:
        st.markdown("#### Actual fit boundaries")
        st.dataframe(pd.DataFrame([{"Horizon": f"+{h} min", **info} for h, info in bundle["forecaster"].training_summary["horizons"].items()]), hide_index=True, width="stretch")


def _models(locations, data):
    st.markdown("### Models · baseline and multi-horizon regression")
    st.caption("Baseline = mean arrivals in up to three preceding five-minute observations. It uses only past values and is evaluated on the same held-out rows as ML.")
    bundle = _bundle(locations)
    if not bundle:
        return
    model = bundle["forecaster"].models[DEFAULT_HORIZONS[0]]
    params, defaults = model.get_params(), HistGradientBoostingRegressor().get_params()
    keys = ("learning_rate", "max_iter", "max_leaf_nodes", "max_depth", "min_samples_leaf", "l2_regularization", "random_state")
    st.markdown("#### Actual hyperparameters · no tuning claimed")
    st.dataframe(pd.DataFrame([{"Parameter": key, "Value": str(params[key]), "Source": "Default" if params[key] == defaults[key] else "Explicit"} for key in keys]), hide_index=True, width="stretch")
    for horizon in DEFAULT_HORIZONS:
        record = bundle["metrics"][horizon]
        st.markdown(f"**+{horizon} min** · {record['test_samples']:,} held-out rows · ML MAE {record['ml']['mae']:.3f} · RMSE {record['ml']['rmse']:.3f} · R² {record['ml']['r2']:.3f} · baseline MAE {record['baseline']['mae']:.3f}")
    st.caption("Four separate regressors; runtime-trained once on explicit request and cached. No persistent artifact is required on Streamlit Community Cloud.")


def _predictions(locations, data):
    st.markdown("### Predictions · held-out SYNTHETIC test")
    bundle = _bundle(locations)
    if not bundle:
        return
    horizon = st.select_slider("Forecast horizon", options=DEFAULT_HORIZONS, value=20, format_func=lambda h: f"+{h} min", key="qm_ml_horizon")
    frame = bundle["forecaster"].test_predictions[horizon]
    facility = st.selectbox("Facility", sorted(frame.location_id.unique()), key="qm_ml_prediction_facility")
    visible = frame[frame.location_id.eq(facility)].tail(240)
    chart = go.Figure()
    for column, label, color in (("actual", "Actual", "#173B63"), ("predicted", "ML", "#6B4EAA"), ("baseline", "Baseline", "#F36F5D")):
        chart.add_trace(go.Scatter(x=visible.timestamp, y=visible[column], name=label, line_color=color))
    chart.update_layout(height=350, yaxis_title="Arrivals per 5-minute interval")
    st.plotly_chart(chart, width="stretch")
    st.caption("The target is arrival count in the future interval, not cumulative arrivals. Latest 240 held-out rows shown for readability.")
    history = data[data.location_id.eq(facility)]
    inputs = build_features(history).iloc[-1]
    st.markdown("#### Latest SYNTHETIC model-input snapshot")
    st.dataframe(pd.DataFrame({"Feature": FEATURE_COLUMNS, "Value": [inputs[name] for name in FEATURE_COLUMNS]}), hide_index=True, width="stretch")
    predictions = bundle["forecaster"].forecast(facility, history)
    st.dataframe(pd.DataFrame([{"Horizon": f"+{h} min", "Predicted interval arrivals": value["predicted_arrivals"], "Method": value["method"]} for h, value in predictions.items()]), hide_index=True, width="stretch")
    st.caption("Input is the latest generated observation, not the current live simulation or measured campus operations.")


def _metrics(locations, data):
    st.markdown("### Metrics · held-out SYNTHETIC evaluation")
    bundle = _bundle(locations)
    if not bundle:
        return
    values = bundle["metrics"]
    table = metric_table(values)
    st.dataframe(table, hide_index=True, width="stretch")
    metric = st.selectbox("Compare metric", ["MAE", "RMSE", "R²"], key="qm_ml_metric")
    st.plotly_chart(px.bar(table, x="Horizon", y=metric, color="Method", barmode="group",
                           color_discrete_map={"Rolling baseline": "#F36F5D", "HistGradientBoosting": "#6B4EAA"}), width="stretch")
    improvements = []
    for horizon in DEFAULT_HORIZONS:
        base, ml = values[horizon]["baseline"]["rmse"], values[horizon]["ml"]["rmse"]
        improvements.append({"Horizon": f"+{horizon} min", "RMSE improvement vs baseline %": (base - ml) / base * 100 if base else None})
    st.dataframe(pd.DataFrame(improvements), hide_index=True, width="stretch")
    st.caption("Negative improvement means ML performs worse. Regression error is not classification accuracy.")


def render_ml_lab(locations):
    data = synthetic_evaluation_data(tuple(locations))
    st.caption("SYNTHETIC DATA · academic demonstration · no measured NIT Rourkela operations")
    section = st.selectbox("Explore ML lifecycle", SECTIONS, key="qm_ml_section")
    handlers = {"Overview": _overview, "Training": _training, "Models": _models,
                "Predictions": _predictions, "Metrics": _metrics}
    if section == "Data":
        from src.ui.ml_data import render_data
        render_data(locations, data)
    elif section == "Features":
        from src.ui.ml_data import render_features
        render_features(locations, data)
    elif section in handlers:
        handlers[section](locations, data)
    else:
        from src.ui.ml_diagnostics import render_errors, render_importance, render_uncertainty, render_integrity
        diagnostic_handlers = {"Errors": render_errors, "Importance": render_importance,
                               "Uncertainty": render_uncertainty, "Integrity": render_integrity}
        diagnostic_handlers[section](locations, data, _bundle(locations))
