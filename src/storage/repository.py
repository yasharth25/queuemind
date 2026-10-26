"""Small SQLite repository; UI code should use these functions rather than SQL."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd


SCHEMA = """
CREATE TABLE IF NOT EXISTS locations (
 id TEXT PRIMARY KEY, display_name TEXT NOT NULL, category TEXT NOT NULL,
 default_counters INTEGER NOT NULL, average_service_time_minutes REAL NOT NULL,
 target_wait_minutes REAL NOT NULL, warning_queue_length INTEGER NOT NULL,
 critical_queue_length INTEGER NOT NULL, enabled INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS queue_observations (
 id INTEGER PRIMARY KEY, timestamp TEXT NOT NULL, location_id TEXT NOT NULL,
 category TEXT NOT NULL, active_counters INTEGER NOT NULL, arrivals INTEGER NOT NULL,
 departures INTEGER NOT NULL, queue_count INTEGER NOT NULL, average_service_time REAL NOT NULL,
 arrival_rate REAL NOT NULL, service_rate REAL NOT NULL, data_source TEXT NOT NULL,
 FOREIGN KEY(location_id) REFERENCES locations(id)
);
CREATE TABLE IF NOT EXISTS queue_events (
 id INTEGER PRIMARY KEY, timestamp TEXT NOT NULL, track_id TEXT NOT NULL,
 location_id TEXT NOT NULL, event_type TEXT NOT NULL, data_source TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS predictions (
 id INTEGER PRIMARY KEY, created_at TEXT NOT NULL, location_id TEXT NOT NULL,
 horizon_minutes INTEGER NOT NULL, predicted_arrivals REAL, predicted_queue REAL,
 predicted_wait_minutes REAL, model_name TEXT NOT NULL, data_source TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS recommendations (
 id INTEGER PRIMARY KEY, created_at TEXT NOT NULL, location_id TEXT NOT NULL,
 current_counters INTEGER NOT NULL, recommended_counters INTEGER NOT NULL,
 reason TEXT NOT NULL, data_source TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_observations_location_time ON queue_observations(location_id, timestamp);
"""


class QueueRepository:
    def __init__(self, database_path: str | Path = "data/queuemind.db"):
        self.database_path = Path(database_path)
        self._memory_connection: sqlite3.Connection | None = None

    def _connect(self) -> sqlite3.Connection:
        if str(self.database_path) == ":memory:":
            if self._memory_connection is None:
                self._memory_connection = sqlite3.connect(":memory:")
                self._memory_connection.row_factory = sqlite3.Row
            return self._memory_connection
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        return connection

    def initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(SCHEMA)

    def seed_locations(self, locations: list[dict]) -> None:
        self.initialize()
        fields = ("id", "display_name", "category", "default_counters", "average_service_time_minutes", "target_wait_minutes", "warning_queue_length", "critical_queue_length", "enabled")
        with self._connect() as connection:
            connection.executemany(
                f"INSERT OR REPLACE INTO locations ({','.join(fields)}) VALUES ({','.join('?' for _ in fields)})",
                [tuple(location[field] for field in fields) for location in locations],
            )

    def import_observations(self, observations: pd.DataFrame) -> int:
        self.initialize()
        columns = ["timestamp", "location_id", "category", "active_counters", "arrivals", "departures", "queue_count", "average_service_time", "arrival_rate", "service_rate", "data_source"]
        missing = set(columns).difference(observations.columns)
        if missing:
            raise ValueError(f"Observation import missing fields: {sorted(missing)}")
        values = observations[columns].copy()
        values["timestamp"] = values["timestamp"].astype(str)
        with self._connect() as connection:
            connection.executemany(
                f"INSERT INTO queue_observations ({','.join(columns)}) VALUES ({','.join('?' for _ in columns)})",
                values.itertuples(index=False, name=None),
            )
        return len(values)

    def get_recent_observations(self, location_id: str, limit: int = 500) -> pd.DataFrame:
        with self._connect() as connection:
            return pd.read_sql_query(
                "SELECT * FROM queue_observations WHERE location_id = ? ORDER BY timestamp DESC LIMIT ?",
                connection, params=(location_id, limit),
            ).sort_values("timestamp")

    def save_event(self, timestamp: str, track_id: str, location_id: str, event_type: str, data_source: str = "VISION") -> None:
        with self._connect() as connection:
            connection.execute("INSERT INTO queue_events(timestamp,track_id,location_id,event_type,data_source) VALUES(?,?,?,?,?)", (timestamp, track_id, location_id, event_type, data_source))

    def save_prediction(self, values: dict) -> None:
        fields = ["created_at", "location_id", "horizon_minutes", "predicted_arrivals", "predicted_queue", "predicted_wait_minutes", "model_name", "data_source"]
        with self._connect() as connection:
            connection.execute(f"INSERT INTO predictions ({','.join(fields)}) VALUES ({','.join('?' for _ in fields)})", tuple(values[field] for field in fields))

    def save_recommendation(self, values: dict) -> None:
        fields = ["created_at", "location_id", "current_counters", "recommended_counters", "reason", "data_source"]
        with self._connect() as connection:
            connection.execute(f"INSERT INTO recommendations ({','.join(fields)}) VALUES ({','.join('?' for _ in fields)})", tuple(values[field] for field in fields))
