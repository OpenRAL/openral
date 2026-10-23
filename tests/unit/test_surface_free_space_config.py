"""Explicit free-space bounds validate against a shipped robot's sensor identity."""

import pytest
from openral_core import RobotDescription, SurfaceFreeSpaceCalibration
from pydantic import ValidationError


def _calibration() -> dict[str, object]:
    robot = RobotDescription.from_yaml("robots/so101_follower/robot.yaml")
    return {
        "calibration_ref": "synthetic-rig-only",
        "sensor_ids": [robot.sensors[0].name],
        "depth_error_m": 0.001,
        "pose_error_m": 0.002,
        "pixel_error": 1,
        "max_age_s": 0.2,
    }


def test_explicit_bounds_round_trip() -> None:
    value = SurfaceFreeSpaceCalibration.model_validate(_calibration())
    assert SurfaceFreeSpaceCalibration.model_validate_json(value.model_dump_json()) == value


@pytest.mark.parametrize(
    "updates",
    [
        {"calibration_ref": " "},
        {"sensor_ids": []},
        {"sensor_ids": ["top", "top"]},
        {"sensor_ids": [" top"]},
        {"depth_error_m": float("nan")},
        {"pose_error_m": -0.001},
        {"pixel_error": -1},
        {"max_age_s": 0.0},
    ],
)
def test_incomplete_or_invalid_bounds_refuse(updates: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        SurfaceFreeSpaceCalibration.model_validate(_calibration() | updates)


def test_bounds_have_no_calibration_defaults() -> None:
    with pytest.raises(ValidationError):
        SurfaceFreeSpaceCalibration.model_validate({})
