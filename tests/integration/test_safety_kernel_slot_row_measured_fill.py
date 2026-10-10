"""A slot row's uncommanded joints are checked at their MEASURED pose, on the real OpenArm model.

The real OpenArm runs ADR-0102 slot groups: per tick the runner sends the left arm, left
gripper, right arm and right gripper as separate ``ActionChunk`` s, each ``JOINT_POSITION``
arm chunk zero-padded to the 16 dof at the joints it does not own and naming the ones it does
in ``joint_names``. The kernel used to FK such a row as-is — the other arm at an all-zero
phantom pose — so a left target that hits the right arm where it really is passed, and one
clear of it could stop against a pose nobody commanded. The fix fills every joint the row does
not command from ``/joint_states``. This test proves it on the real ``safety_kernel_node``
binary with the real ``robots/openarm/robot.yaml`` collision model:

=============================================================  ===================================
row                                                            verdict
=============================================================  ===================================
left slot X, right arm MEASURED at M (zero padding hits X)     ACCEPTED (no phantom stop)
left slot T, right arm MEASURED at R (zero padding is clear)   REFUSED, ``KIND_COLLISION`` self,
                                                               evidence carries R, not zeros
=============================================================  ===================================

How the configurations were found: a seeded random search over the envelope's joint limits
with a box-box (SAT) distance mirroring the kernel's stage-1 check over
``collision_params_from_description`` (the FK and SAT of
``tests/unit/test_so101_base_box_collision.py``). Inter-arm box gaps for T / R:
(T, 0) +59.8 mm, (T, R) -41.0 mm; every same-arm pair at or above the
all-zero pose's own -1.9 mm, which the kernel's hull stage clears (the zero pose is accepted
by every other OpenArm kernel test). The live kernel is the oracle; it re-checks these
verdicts every run.

X and M were re-found when the torso entered the model (issue #356): the first X put the left
link5 12 mm from the real torso mesh, which the torso's box + hull slabs refuse, so the phantom
row stopped on the body instead of passing. The same search, with the C++ ``collision.cpp`` as
the oracle over the manifest that now carries the torso, gave the X / M below: (X, 0) is an
inter-arm hit at -39.1 mm; (X, M) clears every non-allowed pair, +59.8 mm between the arms and
+6.0 mm to the torso (kernel predicates, hull-refined).

Real throughout (CLAUDE.md §1.11): the real manifest, the parameters ``deploy_e2e.launch.py``
builds (envelope + manifest collision model + ``collision_joint_names``), real
``ActionChunk`` / ``JointState`` messages. World voxels are off: this is the self/inter-arm
path, and the grid would add nothing but a second reason to stop.

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
_RSKILL_ID = "openral/slot-row-measured-fill"

# Pinned configurations (see the module docstring). Arm joints 1-7; grippers stay at 0.
_X = [0.777, 0.043, 1.264, 1.675, 1.271, -0.673, -1.133]  # left target, hits a zero right arm
_M = [-0.463, 2.049, 0.883, 1.091, 0.393, -0.085, -0.023]  # right measured, clear of X
_T = [-1.497, 0.002, 0.59, 1.49, -0.661, 0.76, -0.809]  # left target, clear of a zero right
_R = [-0.209, 3.049, 1.277, 1.51, 1.277, -0.326, -1.275]  # right measured, hits T


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


def _left_slot(target: list[float], joint_names: list[str], trace: str) -> Any:
    """The left-arm chunk exactly as ``_pad_joint_payload`` + the publishing HAL build it."""
    from openral_msgs.msg import ActionChunk

    left = [n for n in joint_names if n.startswith("left_joint")]
    row = [0.0] * len(joint_names)
    for name, value in zip(left, target, strict=True):
        row[joint_names.index(name)] = value
    chunk = ActionChunk()
    chunk.control_mode = 0  # JOINT_POSITION
    chunk.horizon = 1
    chunk.n_dof = len(joint_names)
    chunk.flat = row
    chunk.joint_names = left
    chunk.rskill_id = _RSKILL_ID
    chunk.trace_id = trace
    return chunk


def test_slot_rows_are_checked_against_the_other_arms_measured_pose() -> None:
    import rclpy
    from openral_msgs.msg import ActionChunk, FailureTrigger
    from rclpy.executors import SingleThreadedExecutor
    from sensor_msgs.msg import JointState
    from std_msgs.msg import Empty

    joint_names = [j.name for j in _description().joints]
    right = [joint_names.index(f"right_joint{i}") for i in range(1, 8)]
    measured = [0.0] * len(joint_names)

    def measure_right(values: list[float]) -> None:
        for i, v in zip(right, values, strict=True):
            measured[i] = v

    node_name = f"safety_kernel_slot_fill_{uuid.uuid4().hex[:8]}"
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
                helper = rclpy.create_node("slot_fill_helper")
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
                    js.position = list(measured)
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

                # ── Phantom: X hits a zero right arm, but the right arm is at M ──────────
                measure_right(_M)
                spin(0.6)
                send(_left_slot(_X, joint_names, "phantom"))
                assert "phantom" in safe, "a slot row must not stop on the padding's phantom pose"
                assert not estops

                # ── Measured: T is clear of a zero right arm, but the right arm is at R ──
                measure_right(_R)
                spin(0.6)
                send(_left_slot(_T, joint_names, "measured"))
                assert "measured" not in safe, "the left target hits the right arm where it is"
                assert estops, "an inter-arm hit must fire /openral/estop"
                trigger = failures[-1]
                assert trigger.kind == FailureTrigger.KIND_COLLISION
                evidence = json.loads(trigger.evidence_json)
                assert evidence["collision_kind"] == "self"
                assert {evidence["link_a"][:13], evidence["link_b_or_object"][:13]} == {
                    "openarm_left_",
                    "openarm_right",
                }, evidence
                q = evidence["joint_positions_rad"]
                assert [q[i] for i in right] == pytest.approx(_R), (
                    "the evidence must carry the right arm's MEASURED pose, not the zero padding"
                )
            finally:
                rclpy.shutdown()
        except AssertionError as exc:
            raise AssertionError(
                f"{exc}\n\n--- safety_kernel_node log ---\n"
                f"{log_path.read_text(encoding='utf-8', errors='replace')}"
            ) from exc
        finally:
            terminate_kernel(proc)
