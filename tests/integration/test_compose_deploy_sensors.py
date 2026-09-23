"""``compose_runtime(deploy_sensors=...)``: the scene's cameras reach the policy's slots.

A deploy scene is where a cell says what its cameras are — including the
``vla_feature_key`` naming the view a checkpoint was trained on. The scene used
to be merged into the robot description only for the sensor readers, so the
runner kept the manifest's key: on the OpenArm bench the ZED view arrived as
``base`` while the restock π0.5 asked for ``context``, the slot was dropped, and
lerobot fed the policy a masked blank in place of its head camera on real arms
(qorin1, 2026-09-23).

Real ``compose_runtime``, real robot manifests, real committed deploy scenes;
skips without a sourced ROS 2 workspace.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("openral_msgs") is None or importlib.util.find_spec("rclpy") is None,
    reason="openral_msgs / rclpy not importable — ROS 2 workspace not sourced",
)

_REPO = Path(__file__).resolve().parents[2]
_SCENES = sorted(
    p
    for p in (_REPO / "scenes" / "deploy").glob("*.yaml")
    if "vla_feature_key" in p.read_text(encoding="utf-8")
)


@pytest.mark.parametrize("scene_path", _SCENES, ids=lambda p: p.stem)
def test_scene_feature_keys_reach_the_runner_and_recorder_slots(scene_path: Path) -> None:
    import rclpy  # type: ignore[import-untyped]
    from openral_core import DeployScene, sensor_name_to_slot
    from openral_rskill_ros.compose import compose_runtime

    scene = DeployScene.from_yaml(str(scene_path))
    expected = {
        s.name: s.vla_feature_key.rsplit(".", 1)[-1]
        for s in scene.sensors
        if s.vla_feature_key and s.modality == "rgb"
    }
    assert expected, f"{scene_path.name} declares no RGB feature key; selection is wrong"

    rclpy.init()
    runtime = compose_runtime(
        _REPO / "robots" / scene.robot_id / "robot.yaml", deploy_sensors=scene.sensors
    )
    try:
        # One description for every consumer: the runner builds its camera
        # slots from it, the aggregator and the recorder read the same object.
        assert runtime.skill_runner_node._description is runtime.description
        assert runtime.aggregator.description is runtime.description
        slots = sensor_name_to_slot(runtime.description)
        for sensor, slot in expected.items():
            assert slots.get(sensor) == slot, (sensor, slots)
        names = [s.name for s in runtime.description.sensors]
        assert len(names) == len(set(names)), f"a sensor survived the merge twice: {names}"
    finally:
        runtime.skill_runner_node.destroy_node()
        runtime.world_state_node.destroy_node()
        rclpy.try_shutdown()
