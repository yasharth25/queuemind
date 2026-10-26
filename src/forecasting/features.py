"""Feature engineering and chronological splits for synthetic or observed queues."""

from __future__ import annotations

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "hour", "minute", "day_of_week", "hour_sin", "hour_cos", "day_sin", "day_cos",
    "queue_count", "active_counters", "arrival_rate", "service_rate", "arrivals_lag_1",
    "queue_lag_1", "arrivals_rolling_mean_3", "queue_rolling_mean_3",
]


def build_features(observations: pd.DataFrame, horizons: list[int] | None = None) -> pd.DataFrame:
    """Build features using only each row's present/past observations.

    Horizon targets are shifts *forward* within a location and are therefore
    unavailable to feature construction, preventing target leakage.
    """
    horizons = horizons or [5, 10, 20, 30]
    required = {"timestamp", "location_id", "arrivals", "queue_count", "active_counters", "arrival_rate", "service_rate"}
    missing = required.difference(observations.columns)
    if missing:
        raise ValueError(f"Missing forecasting fields: {sorted(missing)}")
    data = observations.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    data = data.sort_values(["location_id", "timestamp"]).reset_index(drop=True)
    data["hour"] = data.timestamp.dt.hour
    data["minute"] = data.timestamp.dt.minute
    data["day_of_week"] = data.timestamp.dt.dayofweek
    data["hour_sin"] = np.sin(2 * np.pi * data.hour / 24)
    data["hour_cos"] = np.cos(2 * np.pi * data.hour / 24)
    data["day_sin"] = np.sin(2 * np.pi * data.day_of_week / 7)
    data["day_cos"] = np.cos(2 * np.pi * data.day_of_week / 7)
    grouped = data.groupby("location_id", group_keys=False)
    data["arrivals_lag_1"] = grouped.arrivals.shift(1)
    data["queue_lag_1"] = grouped.queue_count.shift(1)
    data["arrivals_rolling_mean_3"] = grouped.arrivals.transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    data["queue_rolling_mean_3"] = grouped.queue_count.transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    inferred_minutes = int(data.groupby("location_id").timestamp.diff().dropna().dt.total_seconds().median() / 60) or 1
    for horizon in horizons:
        steps = max(1, horizon // inferred_minutes)
        data[f"target_arrivals_{horizon}m"] = grouped.arrivals.shift(-steps)
    return data


def chronological_split(data: pd.DataFrame, train_fraction: float = .7, validation_fraction: float = .15) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split globally time-ordered data into train/validation/test partitions."""
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1 or train_fraction + validation_fraction >= 1:
        raise ValueError("Fractions must be positive and leave a test partition.")
    ordered = data.sort_values("timestamp").reset_index(drop=True)
    train_end = int(len(ordered) * train_fraction)
    valid_end = int(len(ordered) * (train_fraction + validation_fraction))
    return ordered.iloc[:train_end], ordered.iloc[train_end:valid_end], ordered.iloc[valid_end:]
