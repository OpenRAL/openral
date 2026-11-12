"""The real safety kernel refuses an OpenAMRobot 2.0 hand swung into the lift carriage.

OpenAMRobot reuses OpenArm's arm primitives on a body of its own (chassis, column, lift
carriage, adapter plate, arm profile, tablet, head bracket: boxes lowered from the
URDF), with a 51-pair ACM in which no moving arm link is exempted against any body
part. This runs that model through the real ``safety_kernel_node`` binary with the
parameters ``deploy_e2e.launch.py`` builds:

===========================================================  =============================
row                                                          verdict
===========================================================  =============================
q = 0 (arms hanging, lift at the bottom, base at the origin) ACCEPTED
left hand swung behind the shoulder into the lift carriage   REFUSED, ``KIND_COLLISION``
                                                             self, a left-arm link against a
                                                             body link (the kernel reports the
                                                             first pair it finds: on
                                                             2026-11-05, left link6 against
                                                             the column's L1350 stop sensor)
===========================================================  =============================

The refused row is the ``_REAL_COLLISIONS`` openamrobot_v2 case of
``tests/unit/test_collision_geometry_zero_pose.py``. World voxels are off: this is
the self-collision path.

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

_ROBOT_YAML = (
    pathlib.Path(__file__).resolve().parents[2] / "robots" / "openamrobot_v2" / "robot.yaml"
)
_RSKILL_ID = "openral/openamrobot-body-self-collision"
_BODY = "lift_carriage_link"

# The left hand swung behind the shoulder into the lift carriage (joints 1-7).
_INTO_BODY = {
    "left_joint1": -2.38,
    "left_joint2": -2.88,
    "left_joint3": -1.47,
    "left_joint4": 1.23,
    "left_joint5": -1.18,
    "left_joint6": -0.51,
    "left_joint7": 1.13,
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


def test_the_kernel_refuses_a_hand_in_the_lift_carriage_and_accepts_the_rest_pose() -> None:
    import rclpy
    from openral_msgs.msg import ActionChunk, FailureTrigger
    from rclpy.executors import SingleThreadedExecutor
    from sensor_msgs.msg import JointState
    from std_msgs.msg import Empty

    description = _description()
    joint_names = [j.name for j in description.joints]
    assert any(g.link_name == _BODY for g in description.collision_geometry), (
        "fixture precondition: the manifest carries the lift carriage"
    )

    node_name = f"safety_kernel_openamrobot_{uuid.uuid4().hex[:8]}"
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
                helper = rclpy.create_node("openamrobot_body_helper")
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

                # ── Rest ─────────────────────────────────────────────────────────────────
                spin(0.6)
                send(_row(joint_names, {}, "rest"))
                assert "rest" in safe, "the rest pose must pass with the body in the model"
                assert not estops

                # ── The hand swung into the lift carriage ───────────────────────────────
                send(_row(joint_names, _INTO_BODY, "into_body"))
                assert "into_body" not in safe, "a hand inside the lift carriage must be refused"
                assert estops, "a self-collision hit must fire /openral/estop"
                trigger = failures[-1]
                assert trigger.kind == FailureTrigger.KIND_COLLISION
                evidence = json.loads(trigger.evidence_json)
                assert evidence["collision_kind"] == "self"
                pair = {evidence["link_a"], evidence["link_b_or_object"]}
                arm = {link for link in pair if link.startswith("openarm_left_")}
                assert len(arm) == 1, evidence  # one left-arm link ...
                assert not (pair - arm).pop().startswith("openarm_"), evidence  # ... vs the body
            finally:
                rclpy.shutdown()
        except AssertionError as exc:
            raise AssertionError(
                f"{exc}\n\n--- safety_kernel_node log ---\n"
                f"{log_path.read_text(encoding='utf-8', errors='replace')}"
            ) from exc
        finally:
            terminate_kernel(proc)
