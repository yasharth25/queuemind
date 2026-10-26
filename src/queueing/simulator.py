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