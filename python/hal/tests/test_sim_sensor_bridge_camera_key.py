"""SimSensorBridge resolves camera obs-key via vla_feature_key suffix (ADR-0034)."""

from __future__ import annotations

from openral_core import RobotDescription
from openral_hal.sim_sensor_bridge import _obs_key_for_sensor


def test_franka_sensors_map_to_vla_feature_key_suffix() -> None:
    desc = RobotDescription.from_yaml("robots/franka_panda/robot.yaml")
    by_name = {s.name: s for s in desc.sensors if s.modality == "rgb"}
    assert _obs_key_for_sensor(by_name["agentview"]) == "camera1"
    assert _obs_key_for_sensor(by_name["wrist"]) == "camera2"
