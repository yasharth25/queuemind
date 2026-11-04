import pytest
from pathlib import Path
import shutil

from src.forecasting.forecaster import ArrivalForecaster
from src.synthetic.generator import generate_synthetic_queue_data

LOCATIONS = [{"id": "bank", "category": "bank", "default_counters": 2, "average_service_time_minutes": 3}]


def test_training_evaluation_inference_and_persistence():
    data = generate_synthetic_queue_data(LOCATIONS, "2026-01-01", days=8, seed=9)
    forecaster = ArrivalForecaster()
    metrics = forecaster.train(data)
    assert set(metrics) == {5, 10, 20, 30}
    assert set(metrics[5]) >= {"baseline", "ml", "test_samples"}
    result = forecaster.forecast("bank", data)
    assert set(result) == {5, 10, 20, 30}
    artifact_dir = Path("models/forecasting/test_artifacts")
    try:
        forecaster.save(artifact_dir)
        assert ArrivalForecaster.load(artifact_dir).forecast("bank", data)[5]["method"] == "HistGradientBoostingRegressor"
    finally:
        shutil.rmtree(artifact_dir, ignore_errors=True)


def test_insufficient_history_is_rejected():
    data = generate_synthetic_queue_data(LOCATIONS, "2026-01-01", days=1).head(1)
    with pytest.raises(ValueError, match="At least two"):
        ArrivalForecaster().forecast("bank", data)


def test_future_training_targets_cannot_cross_into_held_out_test():
    data = generate_synthetic_queue_data(LOCATIONS, "2026-01-01", days=8, seed=9)
    forecaster = ArrivalForecaster()
    forecaster.train(data)
    for horizon, summary in forecaster.training_summary["horizons"].items():
        assert summary["last_target_timestamp"] < summary["first_test_timestamp"]
        held_out = forecaster.test_predictions[horizon]
        assert len(held_out) == forecaster.metrics[horizon]["test_samples"]
        assert held_out.timestamp.min() == summary["first_test_timestamp"]
        assert {"actual", "predicted", "baseline"}.issubset(held_out)
