"""SimSensorBridge resolves camera obs-key via vla_feature_key suffix (ADR-0034).

Also covers the dual-keying fallback (issue #88): a frame dict may be keyed by
the VLA slot (``camera1`` — SimAttachedHAL) or by the sensor name (``front`` —
the MujocoArmHAL bare/composed twin), and the bridge must resolve both.
"""

from __future__ import annotations

import numpy as np
from openral_core import RobotDescription
from openral_hal.sim_sensor_bridge import _frame_for_camera, _obs_key_for_sensor


def test_franka_sensors_map_to_vla_feature_key_suffix() -> None:
    desc = RobotDescription.from_yaml("robots/franka_panda/robot.yaml")
    by_name = {s.name: s for s in desc.sensors if s.modality == "rgb"}
    assert _obs_key_for_sensor(by_name["front"]) == "camera1"
    assert _obs_key_for_sensor(by_name["wrist"]) == "camera2"


def test_so101_sensors_map_name_to_vla_slot() -> None:
    # so101's sensor names (front / wrist) differ from their VLA slots
    # (camera1 / camera2) — the mismatch that hid issue #88's frame lookup.
    desc = RobotDescription.from_yaml("robots/so101_follower/robot.yaml")
    by_name = {s.name: s for s in desc.sensors if s.modality == "rgb"}
    assert _obs_key_for_sensor(by_name["front"]) == "camera1"
    assert _obs_key_for_sensor(by_name["wrist"]) == "camera2"


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
