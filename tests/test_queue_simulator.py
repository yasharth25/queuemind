import pytest

from src.queueing.simulator import (
    calculate_service_rate,
    simulate_queue,
)


def test_service_rate_calculation():
    rate = calculate_service_rate(
        active_counters=3,
        average_service_time_minutes=2.5,
    )

    assert rate == pytest.approx(1.2)


def test_queue_grows_when_arrivals_exceed_service():
    result = simulate_queue(
        current_queue=31,
        arrival_rate_per_minute=1.5,
        average_service_time_minutes=2.5,
        active_counters=3,
        horizon_minutes=20,
    )

    final_queue = result[-1]["queue"]

    assert final_queue == pytest.approx(37.0)


def test_queue_decreases_with_more_counters():
    result = simulate_queue(
        current_queue=31,
        arrival_rate_per_minute=1.5,
        average_service_time_minutes=2.5,
        active_counters=5,
        horizon_minutes=20,
    )

    final_queue = result[-1]["queue"]

    assert final_queue == pytest.approx(21.0)


def test_queue_never_becomes_negative():
    result = simulate_queue(
        current_queue=5,
        arrival_rate_per_minute=0,
        average_service_time_minutes=1,
        active_counters=5,
        horizon_minutes=10,
    )

    for row in result:
        assert row["queue"] >= 0


def test_zero_horizon_returns_current_queue_only():
    result = simulate_queue(
        current_queue=12,
        arrival_rate_per_minute=1,
        average_service_time_minutes=2,
        active_counters=2,
        horizon_minutes=0,
    )

    assert len(result) == 1
    assert result[0]["minute"] == 0
    assert result[0]["queue"] == 12


def test_negative_current_queue_is_rejected():
    with pytest.raises(ValueError):
        simulate_queue(
            current_queue=-1,
            arrival_rate_per_minute=1,
            average_service_time_minutes=2,
            active_counters=2,
            horizon_minutes=10,
        )


def test_negative_arrival_rate_is_rejected():
    with pytest.raises(ValueError):
        simulate_queue(
            current_queue=10,
            arrival_rate_per_minute=-1,
            average_service_time_minutes=2,
            active_counters=2,
            horizon_minutes=10,
        )


def test_zero_counters_is_rejected():
    with pytest.raises(ValueError):
        calculate_service_rate(
            active_counters=0,
            average_service_time_minutes=2,
        )


def test_zero_service_time_is_rejected():
    with pytest.raises(ValueError):
        calculate_service_rate(
            active_counters=2,
            average_service_time_minutes=0,
        )