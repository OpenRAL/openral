"""The real safety kernel on the MakerMods Metal manifests: rest passes, folded poses stop.

PR #499 gives the Metal single arm and the two-arm station fitted collision geometry,
lowered from the derived ``makermods:metal`` MJCF (box + hull per link, ``link1`` in two
slabs, the fingers as ``link7`` / ``link8`` posed by the normalised-gripper mimic rows).
The unit tier checks it through the Python port of the kernel predicates; this runs the
real ``safety_kernel_node`` binary with the parameters ``deploy_e2e.launch.py`` builds
from each manifest (hazard-log Entry 077, review item 3):

=====================================================  ===================================
row                                                    verdict
=====================================================  ===================================
single arm, q = 0 (rest, gripper closed)               ACCEPTED (jaw/mount pairs exempt)
single arm, gripper folded down into the base          REFUSED, self, base_link/gripper_base
single arm, gripper folded back onto the upper arm     REFUSED, self, gripper_base/link2
station, q = 0 (both arms at rest, 0.612 m apart)      ACCEPTED
station, both arms yawed inward until the fingertips   REFUSED, self, left_link7/right_link8
meet over the middle of the plate
=====================================================  ===================================

The refused rows are the ``_REAL_COLLISIONS`` Metal cases of
``tests/unit/test_collision_geometry_zero_pose.py`` (mesh evidence there). World voxels
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

_ROBOTS = pathlib.Path(__file__).resolve().parents[2] / "robots"
_RSKILL_ID = "openral/metal-self-collision"

# (manifest dir, trace id, joint overrides, the self-collision pair the kernel must report).
# One kernel process per row: a hit latches the e-stop, so each starts from a fresh kernel
# that first accepts the rest pose.
_FOLDED: list[tuple[str, str, dict[str, float], tuple[str, str]]] = [
    (
        "makermods_metal",
        "into_base",
        {
            "joint1": 0.26,
            "joint2": -1.74,
            "joint3": 0.1,
            "joint4": -0.42,
            "joint5": 0.54,
            "joint6": -0.27,
        },
        ("base_link", "gripper_base"),
    ),
    (
        "makermods_metal",
        "onto_upper_arm",
        {
            "joint1": -0.3,
            "joint2": -0.77,
            "joint3": 0.44,
            "joint4": -1.63,
            "joint5": 0.48,
            "joint6": 0.22,
        },
        ("gripper_base", "link2"),
    ),
    (
        "makermods_metal_station",
        "fingertips_meet",
        {
            "left_joint1": -1.57,
            "left_joint2": -1.2,
            "left_joint3": 1.57,
            "right_joint1": 1.57,
            "right_joint2": -1.2,
            "right_joint3": 1.57,
        },
        ("left_link7", "right_link8"),
    ),
]


def _kernel_params(description: RobotDescription) -> dict[str, object]:
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


@pytest.mark.parametrize(
    ("robot", "trace", "folded", "pair"), _FOLDED, ids=[f"{r[0]}-{r[1]}" for r in _FOLDED]
)
def test_the_kernel_accepts_rest_and_stops_a_folded_metal_pose(
    robot: str, trace: str, folded: dict[str, float], pair: tuple[str, str]
) -> None:
    import rclpy
    from openral_msgs.msg import ActionChunk, FailureTrigger
    from rclpy.executors import SingleThreadedExecutor
    from sensor_msgs.msg import JointState
    from std_msgs.msg import Empty

    description = RobotDescription.from_yaml(str(_ROBOTS / robot / "robot.yaml"))
    joint_names = [j.name for j in description.joints]
    links = {g.link_name for g in description.collision_geometry}
    assert set(pair) <= links, f"fixture precondition: {pair} carry geometry"

    node_name = f"safety_kernel_metal_{uuid.uuid4().hex[:8]}"
    with tempfile.TemporaryDirectory() as td:
        log_path = pathlib.Path(td) / "kernel.log"
        proc = start_kernel(
            _kernel_params(description),
            node_name,
            isolated_domain_id(),
            log_path=log_path,
            params_file=pathlib.Path(td) / "kernel_params.yaml",
        )
        try:
            time.sleep(1.5)
            rclpy.init()
            try:
                helper = rclpy.create_node("metal_helper")
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

                spin(0.6)
                send(_row(joint_names, {}, "rest"))
                assert "rest" in safe, f"{robot}: the rest pose must pass the kernel"
                assert not estops, f"{robot}: no e-stop at rest"

                send(_row(joint_names, folded, trace))
                assert trace not in safe, f"{robot} {trace}: must be refused"
                assert estops, f"{robot} {trace}: a self-collision must fire /openral/estop"
                assert failures, f"{robot} {trace}: no FailureTrigger"
                trigger = failures[-1]
                assert trigger.kind == FailureTrigger.KIND_COLLISION
                evidence = json.loads(trigger.evidence_json)
                assert evidence["collision_kind"] == "self", evidence
                got = {evidence["link_a"], evidence["link_b_or_object"]}
                assert got == set(pair), f"{robot} {trace}: reported {got}, want {pair}"
            finally:
                rclpy.shutdown()
        except AssertionError as exc:
            raise AssertionError(
                f"{exc}\n\n--- safety_kernel_node log ---\n"
                f"{log_path.read_text(encoding='utf-8', errors='replace')}"
            ) from exc
        finally:
            terminate_kernel(proc)
