from src.forecasting.forecaster import ArrivalForecaster
from src.forecasting.queue_integration import forecast_queue_outcomes
from src.synthetic.generator import generate_synthetic_queue_data

LOCATION = {"id": "bank", "category": "bank", "default_counters": 2, "average_service_time_minutes": 3, "target_wait_minutes": 10}


def test_untrained_forecaster_falls_back_and_drives_queue_simulation():
    history = generate_synthetic_queue_data([LOCATION], "2026-01-01", days=2)
    outcomes = forecast_queue_outcomes(ArrivalForecaster(), LOCATION, history)
    assert set(outcomes) == {5, 10, 20, 30}
    assert outcomes[20]["method"] == "rolling-arrival baseline"
    assert outcomes[20]["expected_queue"] >= 0
    assert "recommended_counters" in outcomes[20]["counter_recommendation"]
