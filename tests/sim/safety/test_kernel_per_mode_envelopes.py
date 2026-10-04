"""Per-control-mode magnitude bounds, enforced by the real ``safety_kernel_node``.

Before the kernel enforced them, ``max_base_*_speed``, ``max_ee_angular_speed_rad_s``
and ``max_cartesian_step_*`` were declared on manifests but checked only by the
Python ``supervisor_node``, which no launch file starts — so a deployed graph let
any magnitude through. Each case boots the kernel from a real manifest (the same
``compute_intersection`` + ``kernel_params_from_envelope`` path
``deploy_e2e.launch.py`` uses), sends an in-bound chunk that must reach
``/openral/safe_action``, then an out-of-bound chunk that must be dropped with an
E-stop and a ``FailureTrigger`` naming the measured value and the limit.

Every chunk is horizon 2 with the violation on step 1, so a kernel that strides
rows by the robot's joint count instead of the row width cannot pass. Gripper
chunks are bounded per end effector, in that end effector's own encoding.
"""

from __future__ import annotations

import json
import math
import os
import time
import uuid
from dataclasses import dataclass

import pytest

_ROS2_AVAILABLE = bool(os.environ.get("ROS_DISTRO"))
pytestmark = pytest.mark.skipif(
    not _ROS2_AVAILABLE, reason="ROS_DISTRO not set — requires a sourced ROS 2 install."
)
pytest.importorskip("rclpy")
pytest.importorskip("openral_msgs")

from openral_core import RobotDescription  # noqa: E402

from tests.sim.safety._kernel_subprocess import (  # noqa: E402
    activate_kernel_node,
    isolated_domain_id,
    start_kernel,
    terminate_kernel,
)

# openral_safety_kernel::ControlMode wire values (validator.hpp).
_CARTESIAN_DELTA = 5
_CARTESIAN_TWIST = 6
_BODY_TWIST = 7
_GRIPPER_POSITION = 10
_COMPOSITE_MODE = 12

# robosuite OSC_POSE output_max, the cartesian_delta_scale every RoboCasa rSkill ships.
_OSC_SCALE = [0.05, 0.05, 0.05, 0.5, 0.5, 0.5]


@dataclass(frozen=True)
class _Case:
    manifest: str
    mode: int
    width: int
    ok_rows: tuple[tuple[float, ...], tuple[float, ...]]
    bad_rows: tuple[tuple[float, ...], tuple[float, ...]]
    measured: float
    limit: float
    scale: tuple[float, ...] = ()
    ee: str = ""


_CASES = {
    # r1pro declares 0.7 m/s / 0.3 rad/s — the BEHAVIOR-1K demo base never exceeds
    # either (max |v| 0.405, max |wz| 0.300 over 265k steps).
    "r1pro_body_twist_linear": _Case(
        "robots/r1pro/robot.yaml",
        _BODY_TWIST,
        6,
        ((0.4, 0.3, 0.0, 0.0, 0.0, 0.2), (0.0, 0.0, 0.0, 0.0, 0.0, -0.3)),
        ((0.4, 0.3, 0.0, 0.0, 0.0, 0.2), (0.6, 0.6, 0.0, 0.0, 0.0, 0.0)),
        measured=math.hypot(0.6, 0.6),
        limit=0.7,
    ),
    "r1pro_body_twist_angular": _Case(
        "robots/r1pro/robot.yaml",
        _BODY_TWIST,
        6,
        ((0.0, 0.0, 0.0, 0.0, 0.0, 0.1), (0.0, 0.0, 0.0, 0.0, 0.0, 0.3)),
        ((0.0, 0.0, 0.0, 0.0, 0.0, 0.1), (0.0, 0.0, 0.0, 0.0, 0.0, 0.5)),
        measured=0.5,
        limit=0.3,
    ),
    # A saturated OSC command on every axis (the corner of robosuite's output box)
    # is in bound; the same physical delta sent unscaled at 0.2 m is not.
    "panda_mobile_cartesian_delta_scaled_saturation_passes": _Case(
        "robots/panda_mobile/robot.yaml",
        _CARTESIAN_DELTA,
        6,
        ((1.0, 1.0, 1.0, 1.0, 1.0, 1.0), (-1.0, -1.0, -1.0, -1.0, -1.0, -1.0)),
        ((0.0, 0.0, 0.0, 0.0, 0.0, 0.0), (0.2, 0.0, 0.0, 0.0, 0.0, 0.0)),
        measured=0.2,
        limit=0.087,
        scale=(),
    ),
    "panda_mobile_cartesian_delta_rotation": _Case(
        "robots/panda_mobile/robot.yaml",
        _CARTESIAN_DELTA,
        6,
        ((0.0, 0.0, 0.0, 0.5, 0.5, 0.5), (0.01, 0.0, 0.0, 0.0, 0.0, 0.0)),
        ((0.0, 0.0, 0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.9, 0.0, 0.0)),
        measured=0.9,
        limit=0.87,
    ),
    "panda_mobile_cartesian_twist_angular": _Case(
        "robots/panda_mobile/robot.yaml",
        _CARTESIAN_TWIST,
        6,
        ((0.1, 0.0, 0.0, 0.0, 0.0, 0.5), (0.0, 0.0, 0.0, 0.0, 0.0, 1.0)),
        ((0.1, 0.0, 0.0, 0.0, 0.0, 0.5), (0.0, 0.0, 0.0, 1.5, 0.0, 0.0)),
        measured=1.5,
        limit=1.0,
    ),
    # Each OpenArm jaw is bounded by its own radian range. The intersection of
    # the two ([0, 0]) would refuse every command; per jaw, -0.5 closes the
    # right one and is out of range only on the left.
    "openarm_right_jaw_in_its_own_range": _Case(
        "robots/openarm/robot.yaml",
        _GRIPPER_POSITION,
        1,
        ((-0.2,), (-0.5,)),
        ((-0.2,), (0.3,)),
        measured=0.3,
        limit=0.0,
        ee="right_gripper",
    ),
    "openarm_left_jaw_in_its_own_range": _Case(
        "robots/openarm/robot.yaml",
        _GRIPPER_POSITION,
        1,
        ((0.2,), (0.7,)),
        ((0.2,), (-0.5,)),
        measured=-0.5,
        limit=0.0,
        ee="left_gripper",
    ),
    # robosuite's gripper is [-1, 1]; -1 (open) must pass.
    "panda_mobile_gripper_close_symmetric": _Case(
        "robots/panda_mobile/robot.yaml",
        _GRIPPER_POSITION,
        1,
        ((-1.0,), (1.0,)),
        ((-1.0,), (1.5,)),
        measured=1.5,
        limit=1.0,
        ee="panda_gripper",
    ),
    "panda_mobile_composite_mode": _Case(
        "robots/panda_mobile/robot.yaml",
        _COMPOSITE_MODE,
        1,
        ((-1.0,), (1.0,)),
        ((1.0,), (1.5,)),
        measured=1.5,
        limit=1.0,
    ),
}


def _flat(rows: tuple[tuple[float, ...], ...]) -> list[float]:
    return [float(v) for row in rows for v in row]


@pytest.mark.parametrize("name", list(_CASES))
def test_kernel_enforces_per_mode_bound_from_real_manifest(name: str) -> None:
    import rclpy
    from openral_msgs.msg import ActionChunk, FailureTrigger
    from rclpy.executors import SingleThreadedExecutor
    from std_msgs.msg import Empty

    case = _CASES[name]
    desc = RobotDescription.from_yaml(case.manifest)
    node_name = f"safety_kernel_per_mode_{uuid.uuid4().hex[:8]}"
    proc = start_kernel(desc, node_name, isolated_domain_id())
    try:
        time.sleep(1.5)
        rclpy.init()
        try:
            helper = rclpy.create_node(f"per_mode_helper_{uuid.uuid4().hex[:6]}")
            assert activate_kernel_node(node_name, helper), "kernel activation failed"

            safe: dict[str, ActionChunk] = {}
            failures: list[FailureTrigger] = []
            estops: list[Empty] = []
            safe_sub = helper.create_subscription(
                ActionChunk, "/openral/safe_action", lambda m: safe.__setitem__(m.trace_id, m), 10
            )
            helper.create_subscription(
                FailureTrigger, "/openral/failure/safety", failures.append, 50
            )
            helper.create_subscription(Empty, "/openral/estop", estops.append, 10)
            pub = helper.create_publisher(ActionChunk, "/openral/candidate_action", 10)

            executor = SingleThreadedExecutor()
            executor.add_node(helper)
            deadline = time.time() + 5.0
            while time.time() < deadline:
                if pub.get_subscription_count() >= 1 and safe_sub.get_publisher_count() >= 1:
                    break
                executor.spin_once(timeout_sec=0.05)

            def send(trace: str, rows: tuple[tuple[float, ...], ...], scale: list[float]) -> None:
                chunk = ActionChunk()
                chunk.control_mode = case.mode
                chunk.horizon = len(rows)
                chunk.n_dof = case.width
                chunk.flat = _flat(rows)
                chunk.cartesian_delta_scale = scale
                chunk.ee_name = case.ee
                chunk.rskill_id = "openral/per-mode-envelope-test"
                chunk.trace_id = trace
                pub.publish(chunk)
                end = time.time() + 1.2
                while time.time() < end:
                    executor.spin_once(timeout_sec=0.02)

            ok_scale = _OSC_SCALE if case.mode == _CARTESIAN_DELTA else []
            send("in-bound", case.ok_rows, ok_scale)
            assert "in-bound" in safe, f"{name}: in-bound chunk must reach safe_action"
            assert not estops, f"{name}: in-bound chunk must not E-stop"

            send("out-of-bound", case.bad_rows, list(case.scale))
            assert "out-of-bound" not in safe, f"{name}: violating chunk reached safe_action"
            assert estops, f"{name}: violation must fire /openral/estop"
            assert failures, f"{name}: violation must publish a FailureTrigger"
            evidence = json.loads(failures[-1].evidence_json)
            if evidence["kind"] == "force":  # speed / step magnitudes ride ForceEvidence
                measured, limit = evidence["measured_n"], evidence["limit_n"]
            else:  # a 1-D range breach rides WorkspaceEvidence on the x axis
                assert evidence["kind"] == "workspace", evidence
                # A gripper violation names the end effector the reasoner reports.
                if case.mode == _GRIPPER_POSITION:
                    assert evidence["ee_name"] == case.ee, evidence
                # The kernel spans [min(limit, measured), max(limit, measured)].
                measured = evidence["measured_xyz"][0]
                lo, hi = evidence["box_min"][0], evidence["box_max"][0]
                limit = lo if math.isclose(measured, hi) else hi
            assert measured == pytest.approx(case.measured, abs=1e-6), evidence
            assert limit == pytest.approx(case.limit, abs=1e-6), evidence
        finally:
            rclpy.shutdown()
    finally:
        terminate_kernel(proc)
