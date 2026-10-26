"""Generate physically consistent, explicitly synthetic queue observations."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd


CATEGORY_PROFILES = {
    "canteen": {"base": 0.25, "peaks": [(8, 0.9), (13, 1.8), (17, 1.1)]},
    "bank": {"base": 0.08, "peaks": [(11, 0.8), (14, 0.6)]},
    "library": {"base": 0.15, "peaks": [(12, 0.35), (18, 0.75)]},
    "healthcare": {"base": 0.16, "peaks": [(10, 0.35), (16, 0.3)]},
    "health_centre": {"base": 0.16, "peaks": [(10, 0.35), (16, 0.3)]},
    "administrative_office": {"base": 0.05, "peaks": [(11, 0.7), (14, 0.45)]},
    "mess": {"base": 0.2, "peaks": [(8, 0.8), (13, 1.5), (20, 1.0)]},
    "campus_store": {"base": 0.12, "peaks": [(13, 0.45), (18, 0.65)]},
}


def _arrival_rate(category: str, timestamp: pd.Timestamp, rng: np.random.Generator) -> float:
    """Return a category-specific per-minute arrival rate with daily peaks."""

    profile = CATEGORY_PROFILES.get(category, CATEGORY_PROFILES["administrative_office"])
    hour = timestamp.hour + timestamp.minute / 60
    rate = profile["base"]
    for peak_hour, amplitude in profile["peaks"]:
        rate += amplitude * np.exp(-((hour - peak_hour) ** 2) / 1.5)
    if timestamp.dayofweek >= 5:
        rate *= 0.45
    if category in {"healthcare", "health_centre"}:
        rate *= rng.uniform(0.7, 1.45)
    return max(0.01, float(rate))


def generate_synthetic_queue_data(
    locations: list[dict],
    start: str | datetime,
    days: int = 60,
    frequency_minutes: int = 5,
    seed: int | None = 42,
) -> pd.DataFrame:
    """Generate a labelled synthetic time series for every supplied location.

    Departures are capped by people available after arrivals, so each row obeys
    ``next_queue = previous_queue + arrivals - departures`` and never becomes
    negative. Values are intended for demonstrations, not measured operations.
    """

    if not locations:
        raise ValueError("At least one location is required.")
    if days <= 0 or frequency_minutes <= 0:
        raise ValueError("Days and frequency must be greater than zero.")

    timestamps = pd.date_range(
        start=pd.Timestamp(start),
        periods=(days * 24 * 60) // frequency_minutes,
        freq=f"{frequency_minutes}min",
    )
    rng = np.random.default_rng(seed)
    rows: list[dict] = []

    for location in locations:
        queue_count = 0
        counters = int(location["default_counters"])
        service_time = float(location["average_service_time_minutes"])
        category = location["category"]
        location_id = location["id"]
        service_rate = counters / service_time

        for timestamp in timestamps:
            arrival_rate = _arrival_rate(category, timestamp, rng)
            arrivals = int(rng.poisson(arrival_rate * frequency_minutes))
            available = queue_count + arrivals
            mean_departures = service_rate * frequency_minutes
            departures = min(available, int(rng.poisson(mean_departures)))
            queue_count = available - departures
            rows.append(
                {
                    "timestamp": timestamp,
                    "location_id": location_id,
                    "category": category,
                    "active_counters": counters,
                    "arrivals": arrivals,
                    "departures": departures,
                    "queue_count": queue_count,
                    "average_service_time": service_time,
                    "arrival_rate": round(arrival_rate, 4),
                    "service_rate": round(service_rate, 4),
                    "data_source": "SYNTHETIC",
                }
            )
    return pd.DataFrame(rows)


def validate_synthetic_queue_data(data: pd.DataFrame) -> None:
    """Raise ValueError when required fields or queue conservation are violated."""

    required = {
        "timestamp", "location_id", "category", "active_counters", "arrivals",
        "departures", "queue_count", "average_service_time", "arrival_rate",
        "service_rate", "data_source",
    }
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Synthetic data is missing fields: {sorted(missing)}")
    if data.empty:
        raise ValueError("Synthetic data cannot be empty.")
    if not data["data_source"].eq("SYNTHETIC").all():
        raise ValueError("Every generated row must be labelled SYNTHETIC.")
    if (data[["arrivals", "departures", "queue_count"]] < 0).any().any():
        raise ValueError("Arrivals, departures, and queue counts cannot be negative.")

    ordered = data.sort_values(["location_id", "timestamp"])
    for _, group in ordered.groupby("location_id"):
        previous = 0
        for row in group.itertuples():
            if row.queue_count != previous + row.arrivals - row.departures:
                raise ValueError("Queue evolution is not physically consistent.")
            previous = row.queue_count
