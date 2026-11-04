"""Strict queue-count evaluation for manually labelled video frames."""

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = ("frame_id", "ground_truth_queue_count", "predicted_queue_count")
TEMPLATE_CSV = ",".join(REQUIRED_COLUMNS) + "\n"


def validate_vision_labels(labels: pd.DataFrame) -> pd.DataFrame:
    missing = set(REQUIRED_COLUMNS).difference(labels.columns)
    if missing:
        raise ValueError(f"Vision labels require columns: {', '.join(REQUIRED_COLUMNS)}. Missing: {sorted(missing)}")
    if labels.empty:
        raise ValueError("At least one labelled frame is required.")
    frame = labels[list(REQUIRED_COLUMNS)].copy()
    ids = frame.frame_id.astype("string").str.strip()
    if ids.isna().any() or ids.eq("").any() or ids.duplicated().any():
        raise ValueError("frame_id must be non-empty and unique for each labelled frame.")
    frame["frame_id"] = ids
    for name in REQUIRED_COLUMNS[1:]:
        raw = frame[name]
        if raw.map(lambda value: isinstance(value, (bool, np.bool_))).any():
            raise ValueError(f"{name} must contain nonnegative integer counts.")
        numeric = pd.to_numeric(raw, errors="coerce")
        if not np.isfinite(numeric.to_numpy(dtype=float)).all() or (numeric < 0).any() or (numeric % 1 != 0).any():
            raise ValueError(f"{name} must contain nonnegative integer counts.")
        frame[name] = numeric.astype(int)
    return frame


def evaluate_vision_labels(labels: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Return measured metrics and per-frame errors; no model inference occurs."""
    frame = validate_vision_labels(labels)
    frame["signed_error"] = frame.predicted_queue_count - frame.ground_truth_queue_count
    frame["absolute_error"] = frame.signed_error.abs()
    metrics = {
        "labelled_frames": len(frame),
        "queue_count_mae": float(frame.absolute_error.mean()),
        "queue_count_rmse": float(np.sqrt(np.mean(frame.signed_error.to_numpy(dtype=float) ** 2))),
        "exact_count_accuracy": float(frame.absolute_error.eq(0).mean()),
        "within_one_person_accuracy": float(frame.absolute_error.le(1).mean()),
        "mean_signed_error": float(frame.signed_error.mean()),
        "max_absolute_error": int(frame.absolute_error.max()),
    }
    return metrics, frame
