"""Live demand provenance and deterministic counter recommendations."""

from dataclasses import replace
from datetime import datetime, timedelta, timezone

from src.vision.live.camera_sources import SourceType
from src.vision.live.forecast import project_live_queue
from src.vision.live.monitor import LiveObservation


LOCATION = {"id": "bank", "average_service_time_minutes": 2.0,
            "target_wait_minutes": 8, "warning_queue_length": 10,
            "critical_queue_length": 20}


def observation(now, rate=1.5):
    return LiveObservation(
        "obs-1", "camera-1", "bank", now, now, SourceType.WEBCAM,
        15, 12, 1, 0, rate, 0, None, 1.0, 2.0, .1,
    )


def test_measured_rate_projects_all_horizons_and_optimizes():
    now = datetime.now(timezone.utc)
    result = project_live_queue(observation(now), LOCATION, active_counters=2, now=now)
    assert result["available"]
    assert result["demand_source"] == "OBSERVED_ENTRY_RATE"
    assert [point["minute"] for point in result["points"]] == [0, 5, 10, 20, 30]
    assert result["points"][3]["queue"] > result["points"][0]["queue"]
    assert result["recommendation"]["recommended_counters"] >= 2
    assert result["configured_service_capacity"] == 1.0
    assert result["measured_service_boundary_rate"] == 0


def test_stale_and_insufficient_history_withhold_predictions():
    now = datetime.now(timezone.utc)
    stale = project_live_queue(observation(now - timedelta(minutes=2)), LOCATION,
                               active_counters=2, now=now)
    assert not stale["available"] and "out of date" in stale["reason"]
    missing = project_live_queue(observation(now, None), LOCATION,
                                 active_counters=2, now=now)
    assert not missing["available"] and missing["points"] == []
    assumed = project_live_queue(observation(now, None), LOCATION, active_counters=2,
                                 synthetic_arrival_rate=2, now=now)
    assert assumed["available"]
    assert assumed["demand_source"] == "SYNTHETIC_ASSUMPTION"
    assert assumed["points"][3]["queue"] > assumed["points"][0]["queue"]
    assert observation(now).people_waiting == 12
