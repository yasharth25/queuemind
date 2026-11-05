"""Simulation takes a frozen real/demo snapshot and never mutates observations."""

from datetime import datetime, timezone
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from src.vision.live.camera_sources import SourceType
from src.vision.live.monitor import LiveObservation
from src.vision.live.simulation import (
    SimulationControls, baseline_from_live, run_what_if,
)


LOCATION = {"id": "bank", "average_service_time_minutes": 3,
            "target_wait_minutes": 12, "warning_queue_length": 10,
            "critical_queue_length": 20}


def sample_observation(source=SourceType.WEBCAM, rate=2):
    now = datetime.now(timezone.utc)
    return LiveObservation("obs1", "cam1", "bank", now, now, source,
                           18, 15, 2, 0, rate, 0, None, 1.0, 2.0, .1)


def test_live_snapshot_is_immutable_and_counters_change_outcomes():
    observed = sample_observation()
    baseline = baseline_from_live(observed, LOCATION, 2)
    before = run_what_if(baseline, SimulationControls(2, service_time_minutes=3))
    after = run_what_if(baseline, SimulationControls(5, service_time_minutes=3))
    assert before["proposed"]["future_queue"] > after["proposed"]["future_queue"]
    assert before["proposed"]["predicted_wait_minutes"] > after["proposed"]["predicted_wait_minutes"]
    assert observed.people_waiting == 15 and baseline.queue_count == 15
    assert baseline.snapshot_id == observed.observation_id
    assert after["recommendation"]["recommended_counters"] >= 1


def test_extra_arrivals_crowd_and_service_time_affect_simulator():
    baseline = baseline_from_live(sample_observation(), LOCATION, 2)
    standard = run_what_if(baseline, SimulationControls(2, service_time_minutes=3))
    surge = run_what_if(baseline, SimulationControls(
        2, arrival_multiplier=1.5, extra_arrivals=10,
        service_time_minutes=5, duration_minutes=20, playback_speed=5,
    ))
    assert surge["proposed"]["future_queue"] > standard["proposed"]["future_queue"]
    assert surge["proposed_arrival_rate"] == 3
    assert surge["proposed_path"][0]["queue"] == 25
    assert len(surge["proposed_path"]) == 21


def test_missing_demand_requires_explicit_assumption_and_replay_is_labelled():
    observation = sample_observation(SourceType.REPLAY_VIDEO, None)
    baseline = baseline_from_live(observation, LOCATION, 2)
    assert baseline.source_label == "REPLAY VIDEO"
    with pytest.raises(ValueError, match="demand assumption"):
        run_what_if(baseline, SimulationControls(2))
    assumed = baseline_from_live(observation, LOCATION, 2, synthetic_arrival_rate=1.4)
    assert "SYNTHETIC" in assumed.source_label
    assert run_what_if(assumed, SimulationControls(2))["proposed_arrival_rate"] == 1.4


def test_studio_ui_uses_frozen_synthetic_baseline_and_resets():
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py")
    app.query_params["main_tabs"] = "Simulation Studio"
    app.run(timeout=30)
    assert not app.exception
    assert any("SYNTHETIC DEMO" in item.value for item in app.info)
    original_queue = app.session_state["qm_live"]["current_queue"]
    next(button for button in app.button if button.label == "Run Simulation").click().run(timeout=30)
    assert not app.exception
    result = app.session_state["qm_studio_result"]
    assert result["source_label"] == "SYNTHETIC DEMO"
    assert app.session_state["qm_live"]["current_queue"] == original_queue
    next(button for button in app.button if button.label == "Reset").click().run(timeout=30)
    assert not app.exception
    assert "qm_studio_result" not in app.session_state
