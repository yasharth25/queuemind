import pytest

from src.config import (
    get_enabled_locations,
    get_location_by_id,
    load_locations,
)


REQUIRED_FIELDS = {
    "id",
    "display_name",
    "category",
    "default_counters",
    "average_service_time_minutes",
    "target_wait_minutes",
    "warning_queue_length",
    "critical_queue_length",
    "enabled",
}


def test_locations_can_be_loaded():
    locations = load_locations()

    assert locations is not None
    assert isinstance(locations, list)


def test_at_least_four_locations_exist():
    locations = load_locations()

    assert len(locations) >= 4


def test_location_ids_are_unique():
    locations = load_locations()

    location_ids = [
        location["id"]
        for location in locations
    ]

    assert len(location_ids) == len(set(location_ids))


def test_every_location_has_required_fields():
    locations = load_locations()

    for location in locations:
        assert REQUIRED_FIELDS.issubset(location.keys())


def test_enabled_locations_are_returned():
    locations = get_enabled_locations()

    assert len(locations) > 0

    for location in locations:
        assert location["enabled"] is True


def test_valid_location_lookup():
    location = get_location_by_id("central_canteen")

    assert location["id"] == "central_canteen"
    assert location["display_name"] == "Central Canteen"


def test_invalid_location_lookup():
    with pytest.raises(ValueError):
        get_location_by_id("location_that_does_not_exist")