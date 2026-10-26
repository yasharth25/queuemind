from src.forecasting.features import build_features, chronological_split
from src.synthetic.generator import generate_synthetic_queue_data


LOCATION = [{"id": "bank", "category": "bank", "default_counters": 2, "average_service_time_minutes": 3}]


def test_features_have_past_only_lags_and_future_targets():
    data = generate_synthetic_queue_data(LOCATION, "2026-01-01", days=1, seed=3)
    features = build_features(data)
    assert features.loc[0, "arrivals_lag_1"] != features.loc[0, "arrivals_lag_1"]
    assert features.loc[0, "target_arrivals_5m"] == data.loc[1, "arrivals"]


def test_chronological_split_has_no_temporal_overlap():
    features = build_features(generate_synthetic_queue_data(LOCATION, "2026-01-01", days=2))
    train, validation, test = chronological_split(features)
    assert train.timestamp.max() <= validation.timestamp.min() <= test.timestamp.min()
