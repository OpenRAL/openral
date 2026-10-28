"""``compose_runtime(deploy_sensors=...)``: the scene's cameras reach the policy's slots.

A deploy scene is where a cell says what its cameras are — including the
``vla_feature_key`` naming the view a checkpoint was trained on. The scene used
to be merged into the robot description only for the sensor readers, so the
runner kept the manifest's key: on the OpenArm bench the ZED view arrived as
``base`` while the pick-and-place π0.5 asked for ``context``, the slot was dropped, and
lerobot fed the policy a masked blank in place of its head camera on real arms
(lab Orin, 2026-09-23).

Camera bindings now live in the robot unit overlays a deploy scene selects with
``robot_unit``; the tests compose every such committed scene with its unit.

Real ``compose_runtime``, real robot manifests, real committed deploy scenes and
units; skips without a sourced ROS 2 workspace.
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
# Camera bindings live in the robot unit overlays (``robots/<id>/units/<unit>.yaml``),
# selected by a deploy scene's ``robot_unit``; a scene never redefines a robot camera,
# so no committed deploy scene carries a top-level ``sensors:`` block any more.
_SCENES_WITH_UNITS = sorted(
    p
    for p in (_REPO / "scenes" / "deploy").glob("*.yaml")
    if "\nrobot_unit:" in p.read_text(encoding="utf-8")
)


def test_scene_parametrization_is_not_empty() -> None:
    """Guard: an empty list would turn every parametrized test below into a silent no-op."""
    assert _SCENES_WITH_UNITS, "no committed deploy scene selects a robot_unit"


def _compose(scene_path: Path) -> tuple[Any, Any, list[Any]]:
    from openral_core import DeployScene, resolve_sensor_overlays
    from openral_rskill_ros.compose import compose_runtime

    scene = DeployScene.from_yaml(str(scene_path))
    assert scene.robot_id is not None and scene.robot_unit is not None
    robot_yaml = _REPO / "robots" / scene.robot_id / "robot.yaml"
    overlays = resolve_sensor_overlays(robot_yaml, scene.robot_unit, required=True)
    assert overlays, f"{scene_path.name}: unit {scene.robot_unit} overlays no sensor"
    runtime = compose_runtime(robot_yaml, deploy_sensors=scene.sensors, sensor_overlays=overlays)
    return scene, runtime, overlays


def _destroy(runtime: Any) -> None:
    import rclpy

    runtime.skill_runner_node.destroy_node()
    runtime.world_state_node.destroy_node()
    rclpy.try_shutdown()


@pytest.mark.parametrize("scene_path", _SCENES_WITH_UNITS, ids=lambda p: p.stem)
def test_manifest_feature_keys_reach_the_runner_and_recorder_slots(
    scene_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The manifest's RGB ``vla_feature_key`` slots survive the unit overlay on every consumer."""
    import rclpy
    from openral_core import RobotDescription, sensor_name_to_slot

    monkeypatch.delenv("OPENRAL_ROBOT_UNIT", raising=False)
    runtime = None
    try:
        # init and compose inside the try: a compose failure must still reach
        # try_shutdown, or every later scene fails at init with "already
        # initialized" and hides the real error.
        rclpy.init()
        scene, runtime, _ = _compose(scene_path)
        manifest = RobotDescription.from_yaml(str(_REPO / "robots" / scene.robot_id / "robot.yaml"))
        expected = {
            s.name: s.vla_feature_key.rsplit(".", 1)[-1]
            for s in manifest.sensors
            if s.vla_feature_key and s.modality == "rgb"
        }
        assert expected, f"{scene.robot_id} declares no RGB feature key"
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
        if runtime is not None:
            _destroy(runtime)
        else:
            rclpy.try_shutdown()


@pytest.mark.parametrize("scene_path", _SCENES_WITH_UNITS, ids=lambda p: p.stem)
def test_the_unit_overlay_never_drops_or_renames_a_manifest_slot(
    scene_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Robot-agnostic invariant for every committed scene that selects a unit.

    A unit overlay carries only bindings, mounts and intrinsics (``SensorOverlay``
    forbids ``vla_feature_key``), so every manifest slot must come through unchanged,
    and each overlay's binding must land on the one composed description.
    """
    import rclpy
    from openral_core import RobotDescription, sensor_name_to_slot

    monkeypatch.delenv("OPENRAL_ROBOT_UNIT", raising=False)
    runtime = None
    try:
        rclpy.init()
        scene, runtime, overlays = _compose(scene_path)
        manifest_only = RobotDescription.from_yaml(
            str(_REPO / "robots" / scene.robot_id / "robot.yaml")
        )
        before = sensor_name_to_slot(manifest_only)
        after = sensor_name_to_slot(runtime.description)
        assert before == {k: after.get(k) for k in before}, (scene_path.name, before, after)
        names = [s.name for s in runtime.description.sensors]
        assert len(names) == len(set(names)), names
        by_name = {s.name: s for s in runtime.description.sensors}
        for overlay in overlays:
            assert by_name[overlay.name].deploy_binding == overlay.deploy_binding
        for s in scene.sensors:
            assert s.name in names
    finally:
        if runtime is not None:
            _destroy(runtime)
        else:
            rclpy.try_shutdown()


@pytest.mark.parametrize(
    ("robot_id", "expected_frame"),
    [("openarm", "openarm_base"), ("panda_mobile", "map")],
)
def test_the_object_lift_maps_in_the_frame_the_deploy_maps_in(
    robot_id: str, expected_frame: str
) -> None:
    """A fixed-base arm lifts detections in its own base frame; a mobile base keeps SLAM's map.

    The lift's default is ``map``, which the OpenArm cell never has (no odometry, no SLAM),
    so every lifted box was dropped there and a reasoner-named grasp target could never be
    grounded (real pick-and-place design §2.2). Same rule as the launch's ``_octomap_frames``.
    """
    import rclpy
    from openral_core import RobotDescription
    from openral_rskill_ros.compose import compose_runtime

    robot_yaml = _REPO / "robots" / robot_id / "robot.yaml"
    description = RobotDescription.from_yaml(str(robot_yaml))
    if not rclpy.ok():
        rclpy.init()
    runtime = compose_runtime(robot_yaml)
    try:
        frame = runtime.world_state_node.get_parameter("object_lift_map_frame").value
    finally:
        _destroy(runtime)
    assert frame == expected_frame
    if expected_frame != "map":
        assert frame == description.base_frame
