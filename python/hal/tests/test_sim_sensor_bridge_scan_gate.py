"""SimSensorBridge /scan gate — manifest lidar_2d signal is correct.

These tests verify that the gate predicate driving ``_setup_scan()`` reads
the right signal from each robot manifest:

* franka_panda has no lidar_2d sensor → ``lidar_sensor is None`` → bridge skips
  the publisher + timer entirely.
* panda_mobile declares a lidar_2d sensor → ``lidar_sensor is not None`` → bridge
  would wire the publisher when a live MuJoCo handle is also present.

Both checks use real ``RobotDescription.from_yaml`` against the repo
fixtures — no mocks, no placeholder strings (CLAUDE.md §1.11).
"""

from __future__ import annotations

import pytest
from openral_core import RobotDescription


def test_franka_has_no_lidar_sensor_so_bridge_gate_skips() -> None:
    """Franka manifest declares has_lidar: false and no lidar_2d SensorSpec."""
    desc = RobotDescription.from_yaml("robots/franka_panda/robot.yaml")
    assert desc.lidar_sensor is None, (
        "franka_panda must have no lidar_2d sensor so SimSensorBridge._setup_scan() "
        "returns early without creating a publisher."
    )


def test_panda_mobile_has_lidar_sensor_so_bridge_gate_passes() -> None:
    """panda_mobile manifest declares a lidar_2d SensorSpec — gate should proceed."""
    desc = RobotDescription.from_yaml("robots/panda_mobile/robot.yaml")
    lidar = desc.lidar_sensor
    assert lidar is not None, (
        "panda_mobile must expose a lidar_2d SensorSpec so SimSensorBridge._setup_scan() "
        "creates the /scan publisher when live MuJoCo handles are available."
    )
    # Spot-check the key fields the bridge reads.
    # The lidar publishes in its OWN frame, not `base_link` -- it used to
    # share `base_link` and silently lose its 0.40 m mount offset (the fan is
    # cast 0.30 m above the floor while `base_link` is the platform frame at
    # 0.700 m). See `tests/unit/test_lidar_mount_frame.py` for the full
    # reconciliation this manifest field now carries.
    assert lidar.frame_id == "base_scan"
    assert lidar.n_channels == 360
    # The SENSOR minimum, not a self-filter: #194 lowered this from 0.55 m, a
    # radial cutoff sized to hide the chassis that also deleted every real
    # obstacle inside 0.55 m. Self-returns are excluded by identity instead
    # (`synthesize_laser_scan_2d` in sim, `payload_scan_filter_node`'s chassis
    # polygon on hardware). Matches `scan_range_min: 0.05` in both Nav2 configs.
    assert lidar.range_min_m == pytest.approx(0.05)
    assert lidar.range_max_m == pytest.approx(12.0)
    assert lidar.rate_hz == pytest.approx(10.0)
