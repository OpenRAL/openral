"""SimSensorBridge resolves camera obs-key via ``openral_core.sensor_name_to_slot``.

Also covers the dual-keying fallback (issue #88): a frame dict may be keyed by
the VLA slot (``camera1`` — SimAttachedHAL) or by the sensor name (``front`` —
the MujocoArmHAL bare/composed twin), and the bridge must resolve both.
"""

from __future__ import annotations

import numpy as np
from openral_core import RobotDescription, sensor_name_to_slot
from openral_hal.sim_sensor_bridge import _frame_for_camera


def test_franka_sensors_map_to_vla_feature_key_suffix() -> None:
    desc = RobotDescription.from_yaml("robots/franka_panda/robot.yaml")
    slots = sensor_name_to_slot(desc)
    assert slots["top"] == "camera1"
    assert slots["wrist"] == "camera2"


def test_so101_sensors_map_name_to_vla_slot() -> None:
    # so101's sensor names (top / wrist) differ from their VLA slots
    # (camera1 / camera2) — the mismatch that hid issue #88's frame lookup.
    desc = RobotDescription.from_yaml("robots/so101_follower/robot.yaml")
    slots = sensor_name_to_slot(desc)
    assert slots["top"] == "camera1"
    assert slots["wrist"] == "camera2"


def test_frame_lookup_prefers_vla_slot_key() -> None:
    # SimAttachedHAL keying: frames live under the VLA slot.
    cam1 = np.zeros((4, 4, 3), dtype=np.uint8)
    images = {"camera1": cam1, "camera2": np.ones((4, 4, 3), dtype=np.uint8)}
    assert _frame_for_camera(images, "camera1", "front") is cam1


def test_frame_lookup_falls_back_to_sensor_name() -> None:
    # MujocoArmHAL keying: composed/bare twin keys frames by sensor name, while
    # the bridge's obs_key is the VLA slot. The fallback must still resolve.
    front = np.zeros((4, 4, 3), dtype=np.uint8)
    images = {"front": front, "wrist": np.ones((4, 4, 3), dtype=np.uint8)}
    assert _frame_for_camera(images, "camera1", "front") is front


def test_frame_lookup_returns_none_when_absent() -> None:
    images = {"other": np.zeros((2, 2, 3), dtype=np.uint8)}
    assert _frame_for_camera(images, "camera1", "front") is None
