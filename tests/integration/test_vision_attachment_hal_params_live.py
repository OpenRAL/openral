"""Live: every ``vision_attachment_*`` param ``openral deploy`` writes reaches the HAL node.

The CLI maps ``runtime.vision_attachment`` to HAL ROS params in a params file; rclpy keeps
only the names the node declares and drops the rest without a word. So a scene knob the
HAL never declares is silently a no-op — exactly how ``attach_effort`` / ``release_effort``
were lost. This composes the invocation from the real OpenArm cell scene, writes the
params file the way ``deploy`` does, loads it into the real ``ManifestHALLifecycleNode``
through ``--params-file``, and checks every vision key is declared with the scene's value
and that the effort thresholds land on the real ``GripperEffortTrigger``.

Gated on ``OPENRAL_TEST_ROS_LIVE=1``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.skipif(
    not os.getenv("OPENRAL_TEST_ROS_LIVE"),
    reason="live rclpy node — set OPENRAL_TEST_ROS_LIVE=1 and source install/setup.bash.",
)

_REPO = Path(__file__).resolve().parents[2]


def test_scene_vision_params_are_declared_by_the_hal_and_reach_the_trigger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rclpy = pytest.importorskip("rclpy")
    import yaml
    from openral_cli.deploy_sim import resolve_launch_invocation
    from openral_core import RobotDescription
    from openral_hal._grasp_trigger import GripperEffortTrigger
    from openral_hal.lifecycle import ManifestHALLifecycleNode, vision_attachment_trigger_config

    monkeypatch.setenv("OPENRAL_ROBOT_UNIT", "thor")
    data = yaml.safe_load(
        (_REPO / "scenes" / "deploy" / "openarm_real_world_voxels.yaml").read_text()
    )
    data["runtime"]["vision_attachment"].update(
        enabled=True,
        attach_effort=120.0,
        release_effort=40.0,
        grasp_target_enabled=True,
        place_fixture_enabled=True,
    )
    scene = tmp_path / "vision_leg.yaml"
    scene.write_text(yaml.safe_dump(data), encoding="utf-8")
    invocation = resolve_launch_invocation(
        config=scene,
        robot_override="openarm",
        dashboard_port=4318,
        reset_to_pose_service=None,
        hal_param_overrides=None,
        enable_dashboard=False,
    )
    params_file = tmp_path / "hal_params.yaml"
    params_file.write_text(
        yaml.safe_dump({"/**": {"ros__parameters": invocation.hal_params}}), encoding="utf-8"
    )
    vision = {k: v for k, v in invocation.hal_params.items() if k.startswith("vision_attachment_")}
    assert vision["vision_attachment_attach_effort"] == 120.0

    rclpy.init(args=["--ros-args", "--params-file", str(params_file)])
    node: Any = None
    try:
        node = ManifestHALLifecycleNode("openral_hal")
        undeclared = sorted(k for k in vision if not node.has_parameter(k))
        assert undeclared == [], f"the HAL drops these scene params silently: {undeclared}"
        for key, value in vision.items():
            if node.has_parameter(key):
                assert node.get_parameter(key).value == value, key

        openarm = RobotDescription.from_yaml(str(_REPO / "robots" / "openarm" / "robot.yaml"))
        trigger = GripperEffortTrigger(
            openarm,
            joint_name="left_gripper",
            config=vision_attachment_trigger_config(
                openarm,
                attach_effort=node.get_parameter("vision_attachment_attach_effort").value,
                release_effort=node.get_parameter("vision_attachment_release_effort").value,
            ),
        )
        assert trigger.thresholds_n == pytest.approx((120.0, 40.0))
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()
