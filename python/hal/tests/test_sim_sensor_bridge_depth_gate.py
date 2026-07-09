# SPDX-License-Identifier: Apache-2.0
"""SimSensorBridge depth-PointCloud2 gate — manifest depth-sensor signal is correct.

These tests verify that the gate predicate driving ``_setup_depth()`` reads
the right signal from each robot manifest:

* franka_panda has no depth/point_cloud SensorSpec with intrinsics →
  ``is_depth_sensor`` is False for all sensors → bridge skips depth entirely.
* panda_mobile declares a depth SensorSpec (realsense with intrinsics) →
  ``is_depth_sensor`` is True for ≥1 sensor → bridge would wire publishers
  when live MuJoCo handles are also present.

Both checks use real ``RobotDescription.from_yaml`` against the repo
fixtures — no mocks, no placeholder strings (CLAUDE.md §1.11).
"""

from __future__ import annotations

from openral_core import RobotDescription
from openral_hal.depth_cloud import is_depth_sensor


def test_franka_has_no_depth_sensor() -> None:
    """franka_panda manifest has no depth SensorSpec → bridge gate skips depth."""
    desc = RobotDescription.from_yaml("robots/franka_panda/robot.yaml")
    assert not any(is_depth_sensor(s) for s in desc.sensors)


def test_panda_mobile_has_depth_sensor() -> None:
    """panda_mobile manifest declares a depth SensorSpec → bridge gate proceeds."""
    desc = RobotDescription.from_yaml("robots/panda_mobile/robot.yaml")
    assert any(is_depth_sensor(s) for s in desc.sensors)
