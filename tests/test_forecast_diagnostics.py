import pandas as pd
import pytest

from src.evaluation.forecast_diagnostics import (
    empirical_band, error_quantile, improvement_percent, residual_frame,
    residual_summary,
)


def _predictions(count=20):
    return pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=count, freq="5min"),
        "location_id": ["bank"] * count,
        "actual": [2.0] * count,
        "predicted": [1.0] * count,
        "baseline": [0.0] * count,
    })


def test_residuals_and_summary_use_actual_minus_prediction():
    frame = residual_frame(_predictions())
    assert frame.residual.eq(1).all()
    assert frame.absolute_error.eq(1).all()
    assert frame.baseline_absolute_error.eq(2).all()
    assert residual_summary(frame) == {
        "mean_residual": 1, "median_absolute_error": 1,
        "max_absolute_error": 1, "p90_absolute_error": 1, "samples": 20,
    }


def test_empirical_band_and_quantile_are_measured_and_nonnegative():
    frame = residual_frame(_predictions())
    assert error_quantile(frame, .90) == 1
    assert empirical_band(.3, error_quantile(frame)) == (0, 1.3)
    with pytest.raises(ValueError, match="At least 20"):
        error_quantile(frame.head(19))
    with pytest.raises(ValueError, match="Coverage"):
        error_quantile(frame, 1)


def test_improvement_reports_negative_results_and_handles_zero_baseline():
    assert improvement_percent(2, 1) == 50
    assert improvement_percent(1, 2) == -100
    assert improvement_percent(0, 0) is None
    with pytest.raises(ValueError, match="negative"):
        improvement_percent(-1, 1)


def test_residual_validation_rejects_missing_and_nonfinite_values():
    with pytest.raises(ValueError, match="Missing prediction fields"):
        residual_frame(pd.DataFrame({"actual": [1]}))
    bad = _predictions()
    bad.loc[0, "predicted"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        residual_frame(bad)
