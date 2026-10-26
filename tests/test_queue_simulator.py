import pytest

from src.queueing.simulator import (
    calculate_service_rate,
    estimate_wait_minutes,
    evaluate_counter_counts,
    get_congestion_status,
    recommend_counter_count,
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


def test_wait_estimate_uses_total_service_capacity():
    assert estimate_wait_minutes(12, 3, 2) == pytest.approx(8)


def test_counter_evaluation_returns_queue_and_wait_for_each_count():
    outcomes = evaluate_counter_counts(
        current_queue=20,
        arrival_rate_per_minute=1,
        average_service_time_minutes=2,
        counter_counts=[2, 3, 4],
        horizon_minutes=10,
    )

    assert [outcome["active_counters"] for outcome in outcomes] == [2, 3, 4]
    assert all("future_queue" in outcome for outcome in outcomes)
    assert outcomes[2]["future_queue"] < outcomes[0]["future_queue"]
    assert outcomes[2]["predicted_wait_minutes"] < outcomes[0]["predicted_wait_minutes"]


def test_optimizer_returns_smallest_feasible_counter_count():
    result = recommend_counter_count(
        current_queue=20,
        arrival_rate_per_minute=1,
        average_service_time_minutes=2,
        current_counters=2,
        target_wait_minutes=3,
        horizon_minutes=10,
        candidate_counts=[2, 3, 4, 5],
    )

    assert result["recommended_counters"] == 5
    assert result["meets_target"] is True
    assert "simulated" in result["reason"]


def test_optimizer_reports_when_no_candidate_meets_target():
    result = recommend_counter_count(
        current_queue=100,
        arrival_rate_per_minute=1,
        average_service_time_minutes=2,
        current_counters=2,
        target_wait_minutes=1,
        horizon_minutes=1,
        candidate_counts=[2, 3],
    )

    assert result["meets_target"] is False
    assert result["recommended_counters"] == 3


@pytest.mark.parametrize(
    ("queue", "expected"),
    [(0, "normal"), (10, "warning"), (20, "critical")],
)
def test_congestion_status_uses_configured_thresholds(queue, expected):
    assert get_congestion_status(queue, 10, 20) == expected


def test_invalid_congestion_threshold_order_is_rejected():
    with pytest.raises(ValueError):
        get_congestion_status(1, 20, 10)
