"""Held-out error, importance, empirical uncertainty, and integrity views."""

import platform

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sklearn
import streamlit as st

from src.evaluation.forecast_diagnostics import empirical_band, error_quantile, residual_summary
from src.forecasting.features import FEATURE_COLUMNS, build_features, chronological_split
from src.forecasting.forecaster import DEFAULT_HORIZONS
from src.ui.ml_service import (
    SYNTHETIC_DAYS, SYNTHETIC_FREQUENCY_MINUTES, SYNTHETIC_SEED, SYNTHETIC_START,
    permutation_importance_for_horizon, residuals_for_horizon,
)


def _horizon(key):
    return st.select_slider("Forecast horizon", options=DEFAULT_HORIZONS, value=20,
                            format_func=lambda h: f"+{h} min", key=key)


def render_errors(locations, data, bundle):
    st.markdown("### Error analysis · held-out SYNTHETIC residuals")
    if not bundle:
        return
    horizon = _horizon("qm_error_horizon")
    frame = residuals_for_horizon(tuple(locations), horizon)
    summary = residual_summary(frame)
    cols = st.columns(4)
    for col, label, value in zip(cols, ("Mean residual", "Median absolute error", "Largest error", "90th percentile abs. error"),
                                 (summary["mean_residual"], summary["median_absolute_error"],
                                  summary["max_absolute_error"], summary["p90_absolute_error"])):
        col.metric(label, f"{value:.3f}")
    st.caption(f"{summary['samples']:,} held-out samples · residual = actual − ML prediction · units: arrivals per future five-minute interval")
    left, right = st.columns(2)
    left.plotly_chart(px.histogram(frame, x="residual", nbins=45, title="Residual distribution", color_discrete_sequence=["#6B4EAA"]), width="stretch")
    right.plotly_chart(px.scatter(frame, x="predicted", y="residual", opacity=.45, title="Residual vs prediction", color_discrete_sequence=["#F36F5D"]), width="stretch")
    facility = st.selectbox("Time-series facility", sorted(frame.location_id.unique()), key="qm_error_facility")
    sample = frame[frame.location_id.eq(facility)].tail(260)
    trend = go.Figure()
    trend.add_trace(go.Scatter(x=sample.timestamp, y=sample.residual, name="Signed residual", line_color="#6B4EAA"))
    trend.add_trace(go.Scatter(x=sample.timestamp, y=sample.absolute_error, name="Absolute error", line_color="#F36F5D"))
    trend.update_layout(height=300, title="Error over held-out time")
    st.plotly_chart(trend, width="stretch")
    frame = frame.assign(hour=frame.timestamp.dt.hour, day=frame.timestamp.dt.day_name())
    a, b, c = st.columns(3)
    a.plotly_chart(px.bar(frame.groupby("hour").absolute_error.mean().reset_index(), x="hour", y="absolute_error", title="Mean error by hour"), width="stretch")
    b.plotly_chart(px.bar(frame.groupby("day").absolute_error.mean().reset_index(), x="day", y="absolute_error", title="Mean error by weekday"), width="stretch")
    c.plotly_chart(px.bar(frame.groupby("location_id").absolute_error.mean().reset_index(), x="location_id", y="absolute_error", title="Mean error by facility"), width="stretch")


def render_importance(locations, data, bundle):
    st.markdown("### Feature importance · HELD-OUT DATA")
    if not bundle:
        return
    horizon = _horizon("qm_importance_horizon")
    st.caption("Permutation importance is computed on up to 500 uniformly selected held-out rows, with three seeded repeats. Scoring is negative MAE after nonnegative clipping, matching displayed ML evaluation.")
    try:
        with st.spinner("Computing held-out permutation importance once…"):
            importance = permutation_importance_for_horizon(tuple(locations), horizon)
    except (ValueError, RuntimeError) as exc:
        st.info(f"Permutation importance unavailable: {exc}")
        return
    top = importance.head(10).sort_values("importance")
    st.plotly_chart(px.bar(top, x="importance", y="feature", orientation="h", error_x="std",
                           title="Top 10 · held-out MAE degradation when shuffled",
                           color_discrete_sequence=["#6B4EAA"]), width="stretch")
    st.dataframe(importance, hide_index=True, width="stretch")
    st.caption("Importance measures predictive usefulness on this synthetic test subset, not causality. Negative scores mean shuffling improved this model on the subset.")


def render_uncertainty(locations, data, bundle):
    st.markdown("### Empirical error bands · held-out SYNTHETIC data")
    if not bundle:
        return
    coverage = st.select_slider("Absolute-error quantile", options=[.80, .90, .95], value=.90,
                                format_func=lambda value: f"{value:.0%}")
    facility = st.selectbox("Inference facility", sorted(data.location_id.unique()), key="qm_band_facility")
    history = data[data.location_id.eq(facility)]
    predictions = bundle["forecaster"].forecast(facility, history)
    rows = []
    for horizon in DEFAULT_HORIZONS:
        frame = residuals_for_horizon(tuple(locations), horizon)
        try:
            quantile = error_quantile(frame, coverage)
        except ValueError:
            continue
        prediction = predictions[horizon]["predicted_arrivals"]
        low, high = empirical_band(prediction, quantile)
        stats = residual_summary(frame)
        rows.append({"Horizon": horizon, "Prediction": prediction, "Lower": low, "Upper": high,
                     "Absolute-error quantile": quantile, "MAE": bundle["metrics"][horizon]["ml"]["mae"],
                     "RMSE": bundle["metrics"][horizon]["ml"]["rmse"], "90th-percentile error": stats["p90_absolute_error"]})
    if not rows:
        st.info("Insufficient held-out residuals for an empirical band.")
        return
    frame = pd.DataFrame(rows)
    chart = go.Figure()
    chart.add_trace(go.Scatter(x=frame.Horizon, y=frame.Upper, line_color="rgba(107,78,170,.1)", showlegend=False))
    chart.add_trace(go.Scatter(x=frame.Horizon, y=frame.Lower, fill="tonexty", fillcolor="rgba(107,78,170,.18)", line_color="rgba(107,78,170,.1)", name="Empirical error band"))
    chart.add_trace(go.Scatter(x=frame.Horizon, y=frame.Prediction, line_color="#6B4EAA", mode="lines+markers", name="ML prediction"))
    chart.update_layout(height=330, xaxis_title="Horizon · minutes", yaxis_title="Future five-minute interval arrivals")
    st.plotly_chart(chart, width="stretch")
    st.dataframe(frame, hide_index=True, width="stretch")
    st.caption("Based on held-out synthetic residuals. This is not a formal probabilistic confidence interval. The band is a global absolute-error quantile, not conditional calibration.")
    errors = frame.melt(id_vars="Horizon", value_vars=["MAE", "RMSE", "90th-percentile error"], var_name="Error measure", value_name="Error")
    st.plotly_chart(px.line(errors, x="Horizon", y="Error", color="Error measure", markers=True, title="Error by forecast horizon"), width="stretch")


def render_integrity(locations, data, bundle):
    st.markdown("### ML integrity and reproducibility")
    featured = build_features(data)
    train, validation, test = chronological_split(featured)
    grouped = featured.groupby("location_id")
    lag_verified = (
        np.allclose(featured.arrivals_lag_1, grouped.arrivals.shift(1), equal_nan=True)
        and np.allclose(featured.queue_lag_1, grouped.queue_count.shift(1), equal_nan=True)
    )
    rolling_verified = (
        np.allclose(featured.arrivals_rolling_mean_3,
                    grouped.arrivals.transform(lambda series: series.shift(1).rolling(3, min_periods=1).mean()), equal_nan=True)
        and np.allclose(featured.queue_rolling_mean_3,
                        grouped.queue_count.transform(lambda series: series.shift(1).rolling(3, min_periods=1).mean()), equal_nan=True)
    )
    checks = [
        ("Chronological timestamps do not overlap", train.timestamp.max() < validation.timestamp.min() < test.timestamp.min()),
        ("Every row is labelled SYNTHETIC", bool(data.data_source.eq("SYNTHETIC").all())),
        ("Future target columns are excluded from model features", all(not name.startswith("target_") for name in FEATURE_COLUMNS)),
        ("Lag inputs match previous observations", lag_verified),
        ("Rolling inputs match past-only windows", rolling_verified),
    ]
    if bundle:
        forecaster = bundle["forecaster"]
        checks.extend([
            ("All fitted target timestamps precede held-out test timestamps",
             all(item["last_target_timestamp"] < item["first_test_timestamp"] for item in forecaster.training_summary["horizons"].values())),
            ("Baseline and ML share identical held-out rows",
             all(len(forecaster.test_predictions[h]) == forecaster.metrics[h]["test_samples"] for h in DEFAULT_HORIZONS)),
        ])
    for label, passed in checks:
        st.write(("✓" if passed else "⚠") + " " + label)
    if not bundle:
        st.info("Fit/test and identical-row checks are pending until evaluation runs; they are not marked as verified.")
    st.markdown("#### Reproducibility")
    details = {"Python": platform.python_version(), "scikit-learn": sklearn.__version__,
               "Model": "HistGradientBoostingRegressor", "Synthetic seed": SYNTHETIC_SEED,
               "Synthetic start": SYNTHETIC_START, "Synthetic days": SYNTHETIC_DAYS,
               "Observation interval · min": SYNTHETIC_FREQUENCY_MINUTES,
               "Forecast horizons · min": ", ".join(map(str, DEFAULT_HORIZONS)),
               "Evaluation session": bundle["evaluated_at"].isoformat() if bundle else "Not evaluated in this session"}
    st.dataframe(pd.DataFrame([{"Setting": key, "Value": value} for key, value in details.items()]), hide_index=True, width="stretch")
    st.markdown("#### Limitations")
    st.info("Training and evaluation use synthetic data only; there is no measured NIT Rourkela performance claim. Future campus behavior and sudden events may differ. Representative history is required; vision errors can propagate into forecasts. The displayed empirical bands are not calibrated probabilistic confidence intervals.")
