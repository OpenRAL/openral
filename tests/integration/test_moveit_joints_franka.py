"""ADR-0024/0054 end-to-end exercise — joint goal rSkill against live MoveIt.

Brings up the upstream ``moveit_resources_panda_moveit_config`` demo
launch as a subprocess (real ``move_group`` + ``ros2_control`` fake
hardware + ``robot_state_publisher``), then drives the
:class:`~openral_rskill.joint_goal_rskill.JointGoalRskill` adapter selected by
``ros_integration.goal_builder: "joint"`` from the in-tree
``rskills/rskill-moveit-joints/`` manifest. Asserts the planner returns a
non-empty trajectory that the adapter caches, reorders into the host
:class:`~openral_core.RobotDescription` joint order, and emits one
waypoint per :meth:`~openral_rskill.base.rSkillBase.step` call until
:class:`~openral_core.exceptions.ROSRskillGoalSatisfied` fires.

Gates per CLAUDE.md §1.11 / §1.12:

* ``ROS_DISTRO`` env must be set (a sourced ROS 2 workspace).
* ``rclpy`` must import.
* ``moveit_resources_panda_moveit_config`` must be available as an
  ament package (apt: ``ros-${ROS_DISTRO}-moveit-resources-panda-moveit-config``).

When any of those are missing the test ``pytest.skip(reason=...)`` —
never faked.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Any

import pytest

_ROS2_AVAILABLE = bool(os.environ.get("ROS_DISTRO")) and (
    importlib.util.find_spec("openral_msgs") is not None
)  # custom IDL must be colcon-built, not just ROS sourced (skip cleanly otherwise)

pytestmark = pytest.mark.skipif(
    not _ROS2_AVAILABLE,
    reason="ROS_DISTRO not set — these tests require a sourced ROS 2 installation.",
)

pytest.importorskip("rclpy")


# The shared `move_group_subprocess` fixture (live MoveIt panda demo)
# lives in tests/integration/conftest.py (ADR-0044 made it shared with
# test_look_at_franka.py).


# ── Test ────────────────────────────────────────────────────────────────────


def test_moveit_joints_rskill_plans_and_replays_waypoints(
    move_group_subprocess: None,
) -> None:
    """The wrapped MoveIt rSkill returns a real trajectory and replays it.

    End-to-end exercise of the adapter contract:

    1. Load the in-tree ``OpenRAL/rskill-moveit-joints`` manifest from
       ``rskills/rskill-moveit-joints/rskill.yaml`` (no HF Hub fetch).
    2. Construct :class:`ROSActionRskill` against the running
       ``move_group`` on the canonical Franka panda
       :class:`~openral_core.RobotDescription`.
    3. Drive ``configure()`` + ``activate()`` — the adapter opens an
       ``ActionClient`` on ``/move_action`` and parses the manifest's
       ``default_goal_json``.
    4. Call ``step()`` — the first call sends the goal, waits for the
       real planner, caches the returned :class:`JointTrajectory`
       reordered into ``RobotDescription.joints`` order.
    5. Subsequent ``step()`` calls each return one waypoint as a 1-row
       :class:`Action`.
    6. After the last waypoint, ``step()`` raises
       :class:`ROSRskillGoalSatisfied`.

    No mocks (CLAUDE.md §1.11): real rclpy, real ``move_group``, real
    rSkill manifest, real :class:`RobotDescription`.
    """
    del move_group_subprocess  # fixture provides the side effect

    import rclpy
    from openral_core import (
        RobotDescription,
        ROSRskillGoalSatisfied,
        RSkillManifest,
    )
    from openral_core.schemas import ControlMode
    from openral_rskill.joint_goal_rskill import JointGoalRskill

    repo_root = Path(__file__).resolve().parents[2]
    manifest_path = repo_root / "rskills" / "rskill-moveit-joints" / "rskill.yaml"
    franka_yaml = repo_root / "robots" / "franka_panda" / "robot.yaml"
    assert manifest_path.exists(), f"manifest missing: {manifest_path}"
    assert franka_yaml.exists(), f"Franka robot.yaml missing: {franka_yaml}"

    manifest = RSkillManifest.from_yaml(str(manifest_path))
    description = RobotDescription.from_yaml(str(franka_yaml))

    rclpy.init()
    node: Any = None
    skill: Any = None
    try:
        from rclpy.node import Node

        node = Node("test_moveit_rskill_harness")
        skill = JointGoalRskill(
            manifest=manifest,
            ros_node=node,
            robot_description=description,
            prompt="move to home",
            prompt_metadata_json="",
        )
        skill.configure()
        skill.activate()

        # Spin a brief background thread so any action-client / TF / lifecycle
        # callbacks register. The adapter's `_poll_future` already drives
        # progress via `time.sleep`, but the node still needs callback servicing
        # for service / action discovery — handled by a single-threaded executor
        # spinning in the test's main thread.
        import threading

        from rclpy.executors import SingleThreadedExecutor

        executor = SingleThreadedExecutor()
        executor.add_node(node)
        spin_thread = threading.Thread(target=executor.spin, daemon=True)
        spin_thread.start()
        try:
            # First step: full plan + reorder. Bounded above by the
            # adapter's internal deadline (latency_budget * multiplier).
            first = skill.step(world_state=None)  # type: ignore[arg-type]
            assert first.control_mode is ControlMode.JOINT_POSITION, first.control_mode
            assert first.horizon == 1
            assert len(first.joint_targets) == 1
            assert len(first.joint_targets[0]) == len(description.joints), (
                f"first waypoint dimension {len(first.joint_targets[0])} != "
                f"len(description.joints)={len(description.joints)}"
            )

            # Drive every remaining waypoint plus one more so we hit the
            # termination signal. Bound the loop to avoid infinite spin
            # on a regression where the adapter forgets to raise.
            waypoint_count = 1
            satisfied = False
            for _ in range(10_000):
                try:
                    nxt = skill.step(world_state=None)  # type: ignore[arg-type]
                except ROSRskillGoalSatisfied:
                    satisfied = True
                    break
                waypoint_count += 1
                assert nxt.control_mode is ControlMode.JOINT_POSITION
                assert nxt.horizon == 1
                assert len(nxt.joint_targets[0]) == len(description.joints)
            assert satisfied, (
                f"ROSRskillGoalSatisfied never raised after {waypoint_count} waypoints"
            )
            # A one-waypoint plan is degenerate but still a valid
            # MoveIt result (it means the current state is already
            # close enough to the goal that the planner returned a
            # single-point trajectory). The contract the adapter
            # implements is "emit every waypoint then signal
            # completion" — that holds for waypoint_count >= 1.
            assert waypoint_count >= 1, "expected MoveIt to return at least one waypoint"
        finally:
            executor.shutdown()
            spin_thread.join(timeout=2.0)
    finally:
        if skill is not None and skill.state.value not in {"finalized", "error"}:
            skill.shutdown()
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()
