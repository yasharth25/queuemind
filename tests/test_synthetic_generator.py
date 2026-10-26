import pandas as pd
import pytest

from src.synthetic.generator import generate_synthetic_queue_data, validate_synthetic_queue_data


LOCATIONS = [
    {"id": "canteen", "category": "canteen", "default_counters": 3, "average_service_time_minutes": 2},
    {"id": "clinic", "category": "healthcare", "default_counters": 2, "average_service_time_minutes": 4},
]


def test_generator_includes_required_synthetic_fields_and_conserves_queue():
    data = generate_synthetic_queue_data(LOCATIONS, "2026-01-01", days=1, seed=7)

    assert len(data) == 2 * 24 * 12
    assert set(data["data_source"]) == {"SYNTHETIC"}
    validate_synthetic_queue_data(data)


def test_generator_is_reproducible_when_seeded():
    first = generate_synthetic_queue_data(LOCATIONS, "2026-01-01", days=1, seed=3)
    second = generate_synthetic_queue_data(LOCATIONS, "2026-01-01", days=1, seed=3)

    pd.testing.assert_frame_equal(first, second)


def test_validation_rejects_unlabelled_data():
    data = generate_synthetic_queue_data(LOCATIONS, "2026-01-01", days=1)
    data.loc[0, "data_source"] = "MEASURED"

    with pytest.raises(ValueError, match="SYNTHETIC"):
        validate_synthetic_queue_data(data)


def test_generator_rejects_invalid_window():
    with pytest.raises(ValueError):
        generate_synthetic_queue_data(LOCATIONS, "2026-01-01", days=0)
