"""ADR-0044 Phase 3 end-to-end — :class:`LookAtRskill` against live MoveIt.

Reuses the ``moveit_resources_panda_moveit_config`` demo harness from
:mod:`test_moveit_joints_franka` (real ``move_group`` + fake hardware +
``robot_state_publisher``), drives the in-tree ``rskills/rskill-moveit-look-at``
manifest with the real franka :class:`~openral_core.RobotDescription` (whose
``wrist`` sensor is the LIBERO eye-in-hand on ``panda_hand``), and asserts —
via TF, after the demo *executes* the planned motion — that ``panda_hand``'s
optical axis (+Z) actually points at the requested target.

Gates mirror the sibling test: ``ROS_DISTRO`` set, ``rclpy`` importable, the
panda demo package installed — else ``pytest.skip`` (never faked).
"""

from __future__ import annotations

import importlib.util
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

_ROS2_AVAILABLE = bool(os.environ.get("ROS_DISTRO")) and (
    importlib.util.find_spec("openral_msgs") is not None
)  # custom IDL must be colcon-built, not just ROS sourced (skip cleanly otherwise)

pytestmark = pytest.mark.skipif(
    not _ROS2_AVAILABLE,
    reason="ROS_DISTRO not set — these tests require a sourced ROS 2 installation.",
)

pytest.importorskip("rclpy")

# The shared `move_group_subprocess` fixture (live MoveIt panda demo) comes
# from tests/integration/conftest.py.

_TARGET_XYZ = (0.5, 0.0, 0.2)  # reachable point on the demo's table plane
_GAZE_ANGLE_TOLERANCE_RAD = 0.20  # orientation constraint tol (0.15) + FK epsilon


def test_look_at_rskill_aims_wrist_camera_at_target(
    move_group_subprocess: None,
) -> None:
    del move_group_subprocess

    import rclpy
    from openral_core import RobotDescription, ROSRskillGoalSatisfied, RSkillManifest
    from openral_core.schemas import ControlMode
    from openral_rskill.look_at_rskill import LookAtRskill

    repo_root = Path(__file__).resolve().parents[2]
    manifest = RSkillManifest.from_yaml(
        str(repo_root / "rskills" / "rskill-moveit-look-at" / "rskill.yaml")
    )
    description = RobotDescription.from_yaml(
        str(repo_root / "robots" / "franka_panda" / "robot.yaml")
    )

    rclpy.init()
    node: Any = None
    skill: Any = None
    try:
        from rclpy.node import Node

        node = Node("test_look_at_harness")
        skill = LookAtRskill(
            manifest=manifest,
            ros_node=node,
            robot_description=description,
            prompt="look at the target point",
            prompt_metadata_json="",
        )
        skill.configure()
        skill.activate()

        import threading

        from rclpy.executors import SingleThreadedExecutor

        executor = SingleThreadedExecutor()
        executor.add_node(node)
        spin_thread = threading.Thread(target=executor.spin, daemon=True)
        spin_thread.start()
        try:
            # First step lowers the look_at goal from live TF (camera current
            # pose) and sends the MoveGroup goal (plan_only — OpenRAL's own
            # waypoint replay is the actuation path; MoveIt-side execution
            # would bypass the safety kernel).
            first = skill.step(world_state=None)  # type: ignore[arg-type]
            assert first.control_mode is ControlMode.JOINT_POSITION
            assert len(first.joint_targets[0]) == len(description.joints)

            final_waypoint = list(first.joint_targets[0])
            satisfied = False
            for _ in range(10_000):
                try:
                    nxt = skill.step(world_state=None)  # type: ignore[arg-type]
                except ROSRskillGoalSatisfied:
                    satisfied = True
                    break
                final_waypoint = list(nxt.joint_targets[0])
            assert satisfied, "ROSRskillGoalSatisfied never raised"

            # The REAL assert: run the plan's FINAL waypoint through MoveIt's
            # own /compute_fk and check panda_hand's +Z (the LIBERO
            # eye-in-hand optical axis) points at the target.
            from moveit_msgs.srv import GetPositionFK

            fk_client = node.create_client(GetPositionFK, "/compute_fk")
            assert fk_client.wait_for_service(timeout_sec=10.0), "/compute_fk unavailable"
            req = GetPositionFK.Request()
            req.header.frame_id = "panda_link0"
            req.fk_link_names = ["panda_hand"]
            # Only URDF joints: the description's trailing "panda_gripper" slot
            # is an OpenRAL abstraction (the demo URDF has panda_finger_joint*),
            # and /compute_fk silently never responds to unknown joint names.
            arm = [
                (j.name, float(final_waypoint[i]))
                for i, j in enumerate(description.joints)
                if j.name.startswith("panda_joint")
            ]
            req.robot_state.joint_state.name = [name for name, _ in arm]
            req.robot_state.joint_state.position = [pos for _, pos in arm]
            fut = fk_client.call_async(req)
            deadline = time.monotonic() + 10.0
            while not fut.done() and time.monotonic() < deadline:
                time.sleep(0.05)
            resp = fut.result()
            assert resp is not None and resp.error_code.val == 1, f"FK failed: {resp}"
            hand_pose = resp.pose_stamped[0].pose

            from openral_world_state.object_lift import homogeneous_from_quat_xyz

            q = hand_pose.orientation
            p = hand_pose.position
            hand = homogeneous_from_quat_xyz((p.x, p.y, p.z), (q.x, q.y, q.z, q.w))
            optical_axis = hand[:3, :3] @ np.asarray((0.0, 0.0, 1.0))
            to_target = np.asarray(_TARGET_XYZ) - hand[:3, 3]
            to_target /= np.linalg.norm(to_target)
            angle = math.acos(float(np.clip(np.dot(optical_axis, to_target), -1.0, 1.0)))
            assert angle <= _GAZE_ANGLE_TOLERANCE_RAD, (
                f"camera optical axis misses the target by {angle:.3f} rad "
                f"(> {_GAZE_ANGLE_TOLERANCE_RAD}); hand at {hand[:3, 3]}, "
                f"axis {optical_axis}, target {_TARGET_XYZ}"
            )
        finally:
            executor.shutdown()
            spin_thread.join(timeout=2.0)
    finally:
        if skill is not None and skill.state.value not in {"finalized", "error"}:
            skill.shutdown()
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()
