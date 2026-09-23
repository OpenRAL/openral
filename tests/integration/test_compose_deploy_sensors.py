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
from typing import Any

import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("openral_msgs") is None or importlib.util.find_spec("rclpy") is None,
    reason="openral_msgs / rclpy not importable — ROS 2 workspace not sourced",
)

_REPO = Path(__file__).resolve().parents[2]
_SCENES_WITH_SENSORS = sorted(
    p
    for p in (_REPO / "scenes" / "deploy").glob("*.yaml")
    if "\nsensors:" in p.read_text(encoding="utf-8")
)
_SCENES_WITH_FEATURE_KEYS = [
    p for p in _SCENES_WITH_SENSORS if "vla_feature_key" in p.read_text(encoding="utf-8")
]


def _compose(scene_path: Path) -> tuple[Any, Any]:
    from openral_core import DeployScene
    from openral_rskill_ros.compose import compose_runtime

    scene = DeployScene.from_yaml(str(scene_path))
    runtime = compose_runtime(
        _REPO / "robots" / scene.robot_id / "robot.yaml", deploy_sensors=scene.sensors
    )
    return scene, runtime


def _destroy(runtime: Any) -> None:
    import rclpy  # type: ignore[import-untyped]

    runtime.skill_runner_node.destroy_node()
    runtime.world_state_node.destroy_node()
    rclpy.try_shutdown()


@pytest.mark.parametrize("scene_path", _SCENES_WITH_FEATURE_KEYS, ids=lambda p: p.stem)
def test_scene_feature_keys_reach_the_runner_and_recorder_slots(scene_path: Path) -> None:
    import rclpy  # type: ignore[import-untyped]
    from openral_core import sensor_name_to_slot

    rclpy.init()
    scene, runtime = _compose(scene_path)
    expected = {
        s.name: s.vla_feature_key.rsplit(".", 1)[-1]
        for s in scene.sensors
        if s.vla_feature_key and s.modality == "rgb"
    }
    assert expected, f"{scene_path.name} declares no RGB feature key; selection is wrong"
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
        _destroy(runtime)


@pytest.mark.parametrize("scene_path", _SCENES_WITH_SENSORS, ids=lambda p: p.stem)
def test_merging_a_scene_never_drops_or_renames_a_manifest_slot(scene_path: Path) -> None:
    """Robot-agnostic invariant for every committed scene that binds cameras.

    A scene entry naming a manifest sensor overrides only the fields it sets
    (``model_fields_set``), so a manifest ``vla_feature_key`` survives a scene
    that supplies just the device binding: the SO-101 bench's ``top`` /
    ``wrist`` entries carry no key and must keep the manifest's ``camera1`` /
    ``camera2`` slots now that the runner sees the merged list.
    """
    import rclpy  # type: ignore[import-untyped]
    from openral_core import RobotDescription, sensor_name_to_slot

    rclpy.init()
    scene, runtime = _compose(scene_path)
    try:
        manifest_only = RobotDescription.from_yaml(
            str(_REPO / "robots" / scene.robot_id / "robot.yaml")
        )
        before = sensor_name_to_slot(manifest_only)
        after = sensor_name_to_slot(runtime.description)
        # A scene may re-key a camera on purpose (the OpenArm bench binds the
        # ZED view to the restock checkpoint's `context`); only an explicit
        # `vla_feature_key` on the scene entry is allowed to do that.
        rekeyed = {s.name for s in scene.sensors if "vla_feature_key" in s.model_fields_set}
        for sensor, slot in before.items():
            if sensor in rekeyed:
                continue
            assert after.get(sensor) == slot, (scene_path.name, sensor, before, after)
        # Scene-only cameras are appended, never substituted for manifest ones.
        assert set(before) <= set(after)
        names = [s.name for s in runtime.description.sensors]
        assert len(names) == len(set(names)), names
        # Every scene camera the leg would open is one the runner can now name.
        for s in scene.sensors:
            assert s.name in names
    finally:
        _destroy(runtime)
