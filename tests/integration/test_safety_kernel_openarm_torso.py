"""The real safety kernel refuses a hand swung into the OpenArm torso, on the real model.

Issue #356: ``robots/openarm/robot.yaml`` lowered the arms and not the body they
bolt to, so ``check_self_collision`` could never report an arm-vs-torso hit; the
2026-10-08 wrist-camera calibration had to bolt on an offline mesh check to drop
a HOME candidate that swung the hands into the body. The torso is in the
manifest now (three box + hull slabs, lowered from the URDF). This proves it
through the real ``safety_kernel_node`` binary with the real manifest's
collision model, the parameters ``deploy_e2e.launch.py`` builds:

===========================================================  =============================
row                                                          verdict
===========================================================  =============================
q = 0 (the rest pose; both pedestals sit inside the torso)   ACCEPTED (the two rigid
                                                             contacts are SRDF-exempt)
left hand folded back into the shoulder block                REFUSED, ``KIND_COLLISION``
(elbow 2.37 rad; link7 box centre 29 mm inside the           self, with ``openarm_body_link0``
torso's watertight collision mesh)                           in the reported pair
===========================================================  =============================

The refused row is the ``_REAL_COLLISIONS`` openarm case of
``tests/unit/test_collision_geometry_zero_pose.py`` (a seeded search over the left
arm's limits; the kernel's own predicates put the gap at -118 mm). World voxels
are off: this is the self-collision path.

Gates: ``OPENRAL_TEST_ROS_LIVE=1`` + ROS_DISTRO + rclpy + openral_msgs + colcon-built kernel.
``scripts/ros_live_tests.sh`` runs it (``tests/unit/test_ros_live_targets.py`` keeps it listed).
"""

from __future__ import annotations

import json
import os
import pathlib
import tempfile
import time
import uuid
from typing import Any

import pytest

_LIVE = os.environ.get("OPENRAL_TEST_ROS_LIVE") == "1" and bool(os.environ.get("ROS_DISTRO"))
pytestmark = pytest.mark.skipif(
    not _LIVE,
    reason="live-ROS test — set OPENRAL_TEST_ROS_LIVE=1 with ROS 2 sourced "
    "(scripts/ros_live_tests.sh does both).",
)
if _LIVE:
    pytest.importorskip("rclpy")
    pytest.importorskip("openral_msgs")

from openral_core import RobotDescription  # noqa: E402
from openral_safety.envelope_loader import (  # noqa: E402
    collision_params_from_description,
    compute_intersection,
    kernel_params_from_envelope,
)

from tests.sim.safety._kernel_subprocess import (  # noqa: E402
    activate_kernel_node,
    isolated_domain_id,
    start_kernel,
    terminate_kernel,
)

_ROBOT_YAML = pathlib.Path(__file__).resolve().parents[2] / "robots" / "openarm" / "robot.yaml"
_RSKILL_ID = "openral/torso-self-collision"
_TORSO = "openarm_body_link0"

# The left hand folded back into the shoulder block (joints 1-7; see the module docstring).
_INTO_TORSO = {
    "left_joint1": 0.02,
    "left_joint2": 0.02,
    "left_joint3": 1.41,
    "left_joint4": 2.37,
    "left_joint5": -0.15,
    "left_joint6": 0.37,
    "left_joint7": -1.4,
}


def _description() -> RobotDescription:
    return RobotDescription.from_yaml(str(_ROBOT_YAML))


def _kernel_params() -> dict[str, object]:
    description = _description()
    params: dict[str, object] = {
        **kernel_params_from_envelope(compute_intersection(description, skill=None, deploy=None)),
        **collision_params_from_description(description),
    }
    params["collision_joint_names"] = [j.name for j in description.joints]
    params["collision_seed_dt_s"] = 0.0
    params["collision_state_deadline_ms"] = 1000.0
    return params


def _row(joint_names: list[str], overrides: dict[str, float], trace: str) -> Any:
    """A whole-robot JOINT_POSITION chunk (every joint named, so nothing is filled from state)."""
    from openral_msgs.msg import ActionChunk

    chunk = ActionChunk()
    chunk.control_mode = 0  # JOINT_POSITION
    chunk.horizon = 1
    chunk.n_dof = len(joint_names)
    chunk.flat = [overrides.get(name, 0.0) for name in joint_names]
    chunk.joint_names = list(joint_names)
    chunk.rskill_id = _RSKILL_ID
    chunk.trace_id = trace
    return chunk


def test_the_kernel_refuses_a_hand_in_the_torso_and_accepts_the_rest_pose() -> None:
    import rclpy
    from openral_msgs.msg import ActionChunk, FailureTrigger
    from rclpy.executors import SingleThreadedExecutor
    from sensor_msgs.msg import JointState
    from std_msgs.msg import Empty

    description = _description()
    joint_names = [j.name for j in description.joints]
    assert any(g.link_name == _TORSO for g in description.collision_geometry), (
        "fixture precondition: the manifest carries the torso"
    )

    node_name = f"safety_kernel_torso_{uuid.uuid4().hex[:8]}"
    with tempfile.TemporaryDirectory() as td:
        log_path = pathlib.Path(td) / "kernel.log"
        proc = start_kernel(
            _kernel_params(),
            node_name,
            isolated_domain_id(),
            log_path=log_path,
            params_file=pathlib.Path(td) / "kernel_params.yaml",
        )
        try:
            time.sleep(1.5)
            rclpy.init()
            try:
                helper = rclpy.create_node("torso_helper")
                assert activate_kernel_node(node_name, helper), "kernel activation failed"
                executor = SingleThreadedExecutor()
                executor.add_node(helper)
                safe: dict[str, Any] = {}
                failures: list[Any] = []
                estops: list[Any] = []
                chunk_pub = helper.create_publisher(ActionChunk, "/openral/candidate_action", 10)
                safe_sub = helper.create_subscription(
                    ActionChunk,
                    "/openral/safe_action",
                    lambda m: safe.__setitem__(m.trace_id, m),
                    10,
                )
                helper.create_subscription(
                    FailureTrigger, "/openral/failure/safety", failures.append, 10
                )
                helper.create_subscription(Empty, "/openral/estop", estops.append, 10)
                joint_pub = helper.create_publisher(JointState, "/joint_states", 10)

                def publish_joint_state() -> None:
                    js = JointState()
                    js.header.stamp = helper.get_clock().now().to_msg()
                    js.name = joint_names
                    js.position = [0.0] * len(joint_names)
                    joint_pub.publish(js)

                helper.create_timer(0.05, publish_joint_state)

                def spin(seconds: float) -> None:
                    end = time.time() + seconds
                    while time.time() < end:
                        executor.spin_once(timeout_sec=0.02)

                deadline = time.time() + 5.0
                while time.time() < deadline and not (
                    chunk_pub.get_subscription_count() >= 1 and safe_sub.get_publisher_count() >= 1
                ):
                    executor.spin_once(timeout_sec=0.05)

                def send(chunk: Any) -> None:
                    chunk_pub.publish(chunk)
                    end = time.time() + 10.0
                    while time.time() < end and chunk.trace_id not in safe and not estops:
                        executor.spin_once(timeout_sec=0.02)
                    spin(0.4)

                # ── Rest: both pedestals sit inside the torso, and that is exempt ──────
                spin(0.6)
                send(_row(joint_names, {}, "rest"))
                assert "rest" in safe, "the rest pose must pass with the torso in the model"
                assert not estops

                # ── The hand folded back into the shoulder block ───────────────────────
                send(_row(joint_names, _INTO_TORSO, "into_torso"))
                assert "into_torso" not in safe, "a hand inside the torso must be refused"
                assert estops, "a self-collision hit must fire /openral/estop"
                trigger = failures[-1]
                assert trigger.kind == FailureTrigger.KIND_COLLISION
                evidence = json.loads(trigger.evidence_json)
                assert evidence["collision_kind"] == "self"
                assert _TORSO in {evidence["link_a"], evidence["link_b_or_object"]}, evidence
            finally:
                rclpy.shutdown()
        except AssertionError as exc:
            raise AssertionError(
                f"{exc}\n\n--- safety_kernel_node log ---\n"
                f"{log_path.read_text(encoding='utf-8', errors='replace')}"
            ) from exc
        finally:
            terminate_kernel(proc)
