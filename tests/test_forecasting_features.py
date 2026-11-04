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


def test_same_timestamp_locations_stay_in_one_partition():
    locations = [LOCATION[0], {**LOCATION[0], "id": "bank_two"}]
    features = build_features(generate_synthetic_queue_data(locations, "2026-01-01", days=2))
    train, validation, test = chronological_split(features)
    assert train.timestamp.max() < validation.timestamp.min() < test.timestamp.min()
    assert not set(train.timestamp).intersection(validation.timestamp)
    assert not set(validation.timestamp).intersection(test.timestamp)


def test_lag_and_rolling_features_never_use_current_or_future_arrivals():
    features = build_features(generate_synthetic_queue_data(LOCATION, "2026-01-01", days=1))
    for index in range(1, 8):
        assert features.loc[index, "arrivals_lag_1"] == features.loc[index - 1, "arrivals"]
        expected = features.loc[max(0, index - 3):index - 1, "arrivals"].mean()
        assert features.loc[index, "arrivals_rolling_mean_3"] == expected
    assert features.loc[0, "target_timestamp_20m"] > features.loc[0, "timestamp"]
