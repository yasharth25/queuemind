"""Normalized calibration persistence contains geometry only, no camera secrets."""

import json

import pytest

from src.vision.live.calibration import (
    CameraCalibration, load_calibration, normalized_points, save_calibration,
)


def test_calibration_roundtrip_and_normalized_geometry(tmp_path):
    calibration = CameraCalibration(
        ((.1, .1), (.8, .1), (.8, .9), (.1, .9)),
        ((.1, .3), (.8, .3)), ((.1, .8), (.8, .8)),
    )
    path = save_calibration(calibration, tmp_path / "camera.json")
    assert load_calibration(path) == calibration
    assert set(json.loads(path.read_text())) == {"roi", "entry_line", "service_line"}
    assert normalized_points(((20, 30), (200, 100)), 200, 100) == ((.1, .3), (1, 1))


def test_invalid_calibration_rejected(tmp_path):
    with pytest.raises(ValueError, match="nonzero area"):
        CameraCalibration(((0, 0), (.5, .5), (1, 1)), ((0, 0), (1, 0)), ((0, 1), (1, 1)))
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"roi": [], "password": "should not exist"}))
    with pytest.raises(ValueError, match="only normalized"):
        load_calibration(path)
