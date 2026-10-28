"""Regression checks for the single-page Command Center calculations."""

import pandas as pd
import pytest

from src.demo.live import initial_state, set_counters, trigger_surge
from src.ui.command_model import (
    HORIZONS, TIME_SLOTS, WEEKDAYS, alert_items, heatmap_data, snapshot,
)


def test_default_synthetic_snapshot_is_nonzero_and_simulator_derived():
    state = initial_state("Canteen Lunch Rush")
    view = snapshot(state)
    assert state["current_queue"] == 24
    assert view["wait"] == 16
    assert view["queue20"] == 34
    assert view["recommendation"]["recommended_counters"] == 4
    assert view["first_target_exceeded"] == 0
    assert [item["minute"] for item in view["timeline"]] == [0, *HORIZONS]
    assert view["queue20"] == view["recommendation"]["current_outcome"]["future_queue"]


def test_counter_and_demand_changes_recalculate_forecast_and_alerts():
    state = initial_state("Canteen Lunch Rush")
    before = snapshot(state)
    set_counters(state, 4)
    after = snapshot(state)
    assert after["queue20"] < before["queue20"]
    assert after["wait20"] < before["wait20"]
    assert any("counter" in event["text"].lower() for event in state["simulation_events"])
    trigger_surge(state)
    surged = snapshot(state)
    assert surged["queue20"] > after["queue20"]
    assert alert_items(state, surged)


def test_heatmap_aggregates_weekday_synthetic_rows_without_static_values():
    data = pd.DataFrame({
        "timestamp": ["2026-01-05 08:00", "2026-01-05 08:05", "2026-01-06 08:00"],
        "queue_count": [4, 8, 6],
        "data_source": ["SYNTHETIC"] * 3,
    })
    queues = heatmap_data(data, "Queue", 2, 2)
    waits = heatmap_data(data, "Wait", 2, 2)
    assert queues.shape == (len(TIME_SLOTS), len(WEEKDAYS))
    assert queues.loc["08–10", "MON"] == 6
    assert waits.loc["08–10", "MON"] == 6
    assert queues.loc["08–10", "TUE"] == 6
    with pytest.raises(ValueError):
        heatmap_data(data, "Unrecognised", 2, 2)
    with pytest.raises(ValueError):
        heatmap_data(data.assign(data_source="MEASURED"), "Queue", 2, 2)
