"""Pure held-out forecast error calculations; never constructs synthetic results."""

import numpy as np
import pandas as pd


def residual_frame(predictions: pd.DataFrame) -> pd.DataFrame:
    """Attach signed and absolute residuals to actual held-out predictions."""
    required = {"actual", "predicted", "baseline", "timestamp", "location_id"}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"Missing prediction fields: {sorted(missing)}")
    if predictions.empty:
        raise ValueError("Held-out predictions cannot be empty.")
    frame = predictions.copy()
    for column in ("actual", "predicted", "baseline"):
        values = pd.to_numeric(frame[column], errors="coerce")
        if not np.isfinite(values.to_numpy()).all():
            raise ValueError("Predictions and actual values must be finite numbers.")
        frame[column] = values
    frame["timestamp"] = pd.to_datetime(frame.timestamp)
    frame["residual"] = frame.actual - frame.predicted
    frame["absolute_error"] = frame.residual.abs()
    frame["baseline_absolute_error"] = (frame.actual - frame.baseline).abs()
    return frame


def residual_summary(frame: pd.DataFrame) -> dict:
    if frame.empty or "residual" not in frame or "absolute_error" not in frame:
        raise ValueError("A non-empty residual frame is required.")
    return {
        "mean_residual": float(frame.residual.mean()),
        "median_absolute_error": float(frame.absolute_error.median()),
        "max_absolute_error": float(frame.absolute_error.max()),
        "p90_absolute_error": float(frame.absolute_error.quantile(.90)),
        "samples": len(frame),
    }


def error_quantile(frame: pd.DataFrame, coverage: float = .90) -> float:
    """Absolute-error quantile from real held-out residuals, not a confidence interval."""
    if not 0 < coverage < 1:
        raise ValueError("Coverage must be between zero and one.")
    if len(frame) < 20 or "absolute_error" not in frame:
        raise ValueError("At least 20 held-out residuals are needed for an empirical band.")
    return float(frame.absolute_error.quantile(coverage))


def empirical_band(prediction: float, absolute_error_quantile: float) -> tuple[float, float]:
    if prediction < 0 or absolute_error_quantile < 0:
        raise ValueError("Prediction and error quantile cannot be negative.")
    return max(0.0, prediction - absolute_error_quantile), prediction + absolute_error_quantile


def improvement_percent(baseline_error: float, model_error: float) -> float | None:
    if baseline_error < 0 or model_error < 0:
        raise ValueError("Errors cannot be negative.")
    return (baseline_error - model_error) / baseline_error * 100 if baseline_error else None
