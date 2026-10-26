def calculate_service_rate(
    active_counters: int,
    average_service_time_minutes: float,
) -> float:
    """
    Calculate total service capacity in people per minute.
    """

    if active_counters <= 0:
        raise ValueError("Active counters must be greater than 0.")

    if average_service_time_minutes <= 0:
        raise ValueError(
            "Average service time must be greater than 0."
        )

    return active_counters / average_service_time_minutes


def estimate_wait_minutes(
    queue_length: float,
    active_counters: int,
    average_service_time_minutes: float,
) -> float:
    """Estimate the time needed to clear the current queue at total capacity."""

    if queue_length < 0:
        raise ValueError("Queue length cannot be negative.")

    service_rate = calculate_service_rate(
        active_counters=active_counters,
        average_service_time_minutes=average_service_time_minutes,
    )
    return queue_length / service_rate


def simulate_queue(
    current_queue: float,
    arrival_rate_per_minute: float,
    average_service_time_minutes: float,
    active_counters: int,
    horizon_minutes: int,
):
    """
    Simulate expected queue length minute by minute.
    """

    if current_queue < 0:
        raise ValueError("Current queue cannot be negative.")

    if arrival_rate_per_minute < 0:
        raise ValueError("Arrival rate cannot be negative.")

    if horizon_minutes < 0:
        raise ValueError("Simulation horizon cannot be negative.")

    service_rate = calculate_service_rate(
        active_counters=active_counters,
        average_service_time_minutes=average_service_time_minutes,
    )

    queue = float(current_queue)

    results = [
        {
            "minute": 0,
            "queue": round(queue, 2),
        }
    ]

    for minute in range(1, horizon_minutes + 1):
        queue = max(
            0.0,
            queue
            + arrival_rate_per_minute
            - service_rate,
        )

        results.append(
            {
                "minute": minute,
                "queue": round(queue, 2),
            }
        )

    return results


def get_congestion_status(
    queue_length: float,
    warning_queue_length: float,
    critical_queue_length: float,
) -> str:
    """Classify a queue using configured, location-specific thresholds."""

    if queue_length < 0:
        raise ValueError("Queue length cannot be negative.")
    if warning_queue_length < 0 or critical_queue_length < 0:
        raise ValueError("Congestion thresholds cannot be negative.")
    if warning_queue_length > critical_queue_length:
        raise ValueError("Warning threshold cannot exceed critical threshold.")

    if queue_length >= critical_queue_length:
        return "critical"
    if queue_length >= warning_queue_length:
        return "warning"
    return "normal"


def evaluate_counter_counts(
    current_queue: float,
    arrival_rate_per_minute: float,
    average_service_time_minutes: float,
    counter_counts: list[int],
    horizon_minutes: int,
) -> list[dict]:
    """Return simulated queue and wait outcomes for each possible counter count."""

    if not counter_counts:
        raise ValueError("At least one counter count must be provided.")

    evaluations = []
    for counters in sorted(set(counter_counts)):
        projected_queue = simulate_queue(
            current_queue=current_queue,
            arrival_rate_per_minute=arrival_rate_per_minute,
            average_service_time_minutes=average_service_time_minutes,
            active_counters=counters,
            horizon_minutes=horizon_minutes,
        )[-1]["queue"]
        evaluations.append(
            {
                "active_counters": counters,
                "future_queue": projected_queue,
                "predicted_wait_minutes": round(
                    estimate_wait_minutes(
                        queue_length=projected_queue,
                        active_counters=counters,
                        average_service_time_minutes=average_service_time_minutes,
                    ),
                    2,
                ),
            }
        )
    return evaluations


def recommend_counter_count(
    current_queue: float,
    arrival_rate_per_minute: float,
    average_service_time_minutes: float,
    current_counters: int,
    target_wait_minutes: float,
    horizon_minutes: int,
    candidate_counts: list[int] | None = None,
) -> dict:
    """Recommend the smallest tested counter count that meets a wait-time target.

    The recommendation is derived solely from deterministic simulations.  If no
    candidate reaches the target, the best tested outcome is returned and marked
    as not meeting the target.
    """

    if target_wait_minutes < 0:
        raise ValueError("Target wait time cannot be negative.")
    if candidate_counts is None:
        candidate_counts = list(range(1, max(current_counters, 1) + 4))
    if current_counters not in candidate_counts:
        candidate_counts = [*candidate_counts, current_counters]

    evaluations = evaluate_counter_counts(
        current_queue=current_queue,
        arrival_rate_per_minute=arrival_rate_per_minute,
        average_service_time_minutes=average_service_time_minutes,
        counter_counts=candidate_counts,
        horizon_minutes=horizon_minutes,
    )
    current = next(
        item for item in evaluations if item["active_counters"] == current_counters
    )
    feasible = [
        item for item in evaluations
        if item["predicted_wait_minutes"] <= target_wait_minutes
    ]
    recommended = feasible[0] if feasible else min(
        evaluations, key=lambda item: item["predicted_wait_minutes"]
    )
    meets_target = bool(feasible)

    if recommended["active_counters"] == current_counters and meets_target:
        reason = (
            f"The simulated {horizon_minutes}-minute wait of "
            f"{current['predicted_wait_minutes']:.2f} minutes meets the "
            f"{target_wait_minutes:.2f}-minute target with current capacity."
        )
    elif meets_target:
        reason = (
            f"The simulated {horizon_minutes}-minute wait exceeds the target "
            f"with {current_counters} counters and first falls to "
            f"{recommended['predicted_wait_minutes']:.2f} minutes at "
            f"{recommended['active_counters']} counters."
        )
    else:
        reason = (
            f"No tested counter count meets the {target_wait_minutes:.2f}-minute "
            f"target; {recommended['active_counters']} counters gives the best "
            f"simulated wait of {recommended['predicted_wait_minutes']:.2f} minutes."
        )

    return {
        "current_counters": current_counters,
        "recommended_counters": recommended["active_counters"],
        "meets_target": meets_target,
        "current_outcome": current,
        "recommended_outcome": recommended,
        "evaluations": evaluations,
        "reason": reason,
    }
