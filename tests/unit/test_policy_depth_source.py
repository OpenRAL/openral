"""``policy_depth_source`` on real robot manifests: where a policy camera's depth comes from."""

from __future__ import annotations

from pathlib import Path

import pytest
from openral_core import RobotDescription, SensorSpec
from openral_core.depth_extrinsic import policy_depth_source
from openral_core.exceptions import ROSConfigError

_ROBOTS = Path(__file__).resolve().parents[2] / "robots"


def _sensors(robot: str) -> list[SensorSpec]:
    return list(RobotDescription.from_yaml(str(_ROBOTS / robot / "robot.yaml")).sensors)


def test_rgbd_partner_gives_the_camera_depth() -> None:
    # OpenArm's `top` colour stream shares its mount with the `head_zed` depth sensor.
    assert policy_depth_source(_sensors("openarm"), "top") == "head_zed"


def test_rgb_only_camera_without_estimator_is_refused_naming_the_fix() -> None:
    # panda_mobile's `head` is RGB only; `front_depth` is a different camera.
    with pytest.raises(ROSConfigError) as exc:
        policy_depth_source(_sensors("panda_mobile"), "head")
    assert "runtime.estimated_depth_camera: head" in str(exc.value)
    assert "shares_mount_with: head" in str(exc.value)


def test_rgb_only_camera_with_estimated_depth_is_its_own_source() -> None:
    sensors = _sensors("panda_mobile")
    assert policy_depth_source(sensors, "head", estimated_depth_camera="head") == "head"
    # An estimator on another camera does not cover this one.
    with pytest.raises(ROSConfigError):
        policy_depth_source(sensors, "head", estimated_depth_camera="wrist")


def test_unknown_camera_is_refused() -> None:
    with pytest.raises(ROSConfigError, match="not a sensor"):
        policy_depth_source(_sensors("panda_mobile"), "nope")
