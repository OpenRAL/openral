"""ADR-0030 phase 2 — end-to-end self-collision rejection through the real kernel.

A synthetic 3-link arm (geometry fully controlled, so no external MJCF assets)
is lowered to kernel collision params via
``openral_safety.envelope_loader.collision_params_from_description`` and driven
through the **real C++ safety_kernel_node** subprocess:

    candidate_action (JOINT_POSITION) → safety_kernel_node → safe_action | estop

* A clear configuration ``q = [0, 0, 0]`` passes through to ``/openral/safe_action``.
* A folded configuration ``q = [0, π, 0]`` brings link1 and link3 (non-adjacent,
  so not in the allowed-collision matrix) collinear and overlapping; the kernel
  must drop it, latch, fire ``/openral/estop``, and publish a
  ``FailureTrigger(KIND_COLLISION)`` carrying ``CollisionEvidence``.

MuJoCo is the independent oracle: the same chain is built as an MJCF and
``mj_geomDistance`` confirms the link1↔link3 sign at each configuration and that
the kernel's reported ``min_distance_m`` agrees with MuJoCo's surface distance.

Gates (CLAUDE.md §1.11 / §1.12): ROS_DISTRO + rclpy + openral_msgs + mujoco; a
sourced + colcon-built workspace. Otherwise pytest.skip — never faked.
"""

from __future__ import annotations

import json
import math
import os
import time
import uuid

import pytest

_ROS2_AVAILABLE = bool(os.environ.get("ROS_DISTRO"))
pytestmark = pytest.mark.skipif(
    not _ROS2_AVAILABLE, reason="ROS_DISTRO not set — requires a sourced ROS 2 install."
)
pytest.importorskip("rclpy")
pytest.importorskip("openral_msgs")
mujoco = pytest.importorskip("mujoco")

from openral_core import (  # noqa: E402
    CapsuleShape,
    ControlMode,
    EmbodimentKind,
    JointSpec,
    JointType,
    LinkCollisionGeometry,
    RobotCapabilities,
    RobotDescription,
    SafetyEnvelope,
)
from openral_safety.envelope_loader import collision_params_from_description  # noqa: E402

from tests.sim.safety._kernel_subprocess import (  # noqa: E402
    activate_kernel_node,
    isolated_domain_id,
    start_kernel,
    terminate_kernel,
)

# Capsule spans z ∈ [0.02, 0.38] in the link frame: origin (0,0,0.2), radius
# 0.04, length 0.36 (half 0.18). Links are 0.4 m long.
_R = 0.04
_CAP = LinkCollisionGeometry  # alias for brevity


def _synthetic_arm() -> RobotDescription:
    """base → link1 → link2 → link3, all revolute about Y, 0.4 m links."""

    def joint(name: str, parent: str, child: str, z: float) -> JointSpec:
        return JointSpec(
            name=name,
            joint_type=JointType.REVOLUTE,
            parent_link=parent,
            child_link=child,
            axis_xyz=(0.0, 1.0, 0.0),
            origin_xyz=(0.0, 0.0, z),
        )

    def capsule(link: str) -> LinkCollisionGeometry:
        return _CAP(
            link_name=link,
            shape=CapsuleShape(radius_m=_R, length_m=0.36),
            origin_xyz_rpy=(0.0, 0.0, 0.2, 0.0, 0.0, 0.0),
        )

    return RobotDescription(
        name="synthetic_3link",
        embodiment_kind=EmbodimentKind.MANIPULATOR,
        joints=[
            joint("j1", "base", "link1", 0.0),
            joint("j2", "link1", "link2", 0.4),
            joint("j3", "link2", "link3", 0.4),
        ],
        collision_geometry=[capsule("link1"), capsule("link2"), capsule("link3")],
        # Adjacent links touch by design; link1↔link3 is deliberately NOT listed.
        allowed_collision_pairs=[("link1", "link2"), ("link2", "link3")],
        capabilities=RobotCapabilities(
            supported_control_modes=[ControlMode.JOINT_POSITION], embodiment_tags=["synthetic"]
        ),
        safety=SafetyEnvelope(),
    )


_MJCF = """
<mujoco>
  <worldbody>
    <body name="link1">
      <joint name="j1" type="hinge" axis="0 1 0"/>
      <geom name="cap1" type="capsule" fromto="0 0 0.02 0 0 0.38" size="0.04"/>
      <body name="link2" pos="0 0 0.4">
        <joint name="j2" type="hinge" axis="0 1 0"/>
        <geom name="cap2" type="capsule" fromto="0 0 0.02 0 0 0.38" size="0.04"/>
        <body name="link3" pos="0 0 0.4">
          <joint name="j3" type="hinge" axis="0 1 0"/>
          <geom name="cap3" type="capsule" fromto="0 0 0.02 0 0 0.38" size="0.04"/>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

_CLEAR = [0.0, 0.0, 0.0]
_COLLIDE = [0.0, math.pi, 0.0]


def _mujoco_link1_link3_distance(qpos: list[float]) -> float:
    """Ground-truth surface distance between the link1 and link3 capsules."""
    model = mujoco.MjModel.from_xml_string(_MJCF)
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    g1 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "cap1")
    g3 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "cap3")
    return float(mujoco.mj_geomDistance(model, data, g1, g3, 2.0, None))


def _kernel_params() -> dict[str, object]:
    robot = _synthetic_arm()
    params: dict[str, object] = {
        "n_dof": 3,
        "robot_name": "synthetic_3link",
        "joint_position_min": [-6.5, -6.5, -6.5],
        "joint_position_max": [6.5, 6.5, 6.5],
        "joint_velocity_max": [100.0, 100.0, 100.0],
        "joint_torque_max": [100.0, 100.0, 100.0],
    }
    params.update(collision_params_from_description(robot))
    return params


def test_mujoco_oracle_confirms_the_configurations() -> None:
    """Sanity-check the geometry against MuJoCo before trusting the kernel."""
    assert _mujoco_link1_link3_distance(_CLEAR) > 0.0, "clear config should not collide"
    assert _mujoco_link1_link3_distance(_COLLIDE) < 0.0, "folded config should self-collide"


def test_kernel_rejects_self_colliding_joint_chunk() -> None:
    """Clear chunk passes; folded chunk is dropped with KIND_COLLISION + estop."""
    import rclpy
    from openral_msgs.msg import ActionChunk, FailureTrigger
    from rclpy.executors import SingleThreadedExecutor
    from std_msgs.msg import Empty

    node_name = f"safety_kernel_selfcoll_{uuid.uuid4().hex[:8]}"
    proc = start_kernel(_kernel_params(), node_name, isolated_domain_id())
    try:
        time.sleep(1.5)
        rclpy.init()
        try:
            helper = rclpy.create_node("selfcoll_helper")
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

            def send(qpos: list[float], trace: str) -> None:
                chunk = ActionChunk()
                chunk.control_mode = 0  # JOINT_POSITION
                chunk.horizon = 1
                chunk.n_dof = 3
                chunk.flat = qpos
                chunk.rskill_id = "openral/selfcoll-test"
                chunk.trace_id = trace
                pub.publish(chunk)
                end = time.time() + 1.0
                while time.time() < end:
                    executor.spin_once(timeout_sec=0.02)

            # 1. Clear configuration passes straight through.
            send(_CLEAR, "clear")
            assert "clear" in safe, "clear chunk should be republished on /openral/safe_action"
            assert not failures, "clear chunk must not raise a FailureTrigger"

            # 2. Folded configuration self-collides → drop + estop + KIND_COLLISION.
            send(_COLLIDE, "collide")
            assert "collide" not in safe, "self-colliding chunk must NOT reach safe_action"
            assert estops, "self-collision must fire /openral/estop"
            assert failures, "self-collision must publish a FailureTrigger"

            trigger = failures[-1]
            assert trigger.kind == FailureTrigger.KIND_COLLISION
            evidence = json.loads(trigger.evidence_json)
            assert evidence["kind"] == "collision"
            assert evidence["collision_kind"] == "self"
            assert {evidence["link_a"], evidence["link_b_or_object"]} == {"link1", "link3"}
            assert evidence["min_distance_m"] < 0.0  # penetration

            # 3. Cross-check the kernel's distance against the MuJoCo oracle.
            mj_distance = _mujoco_link1_link3_distance(_COLLIDE)
            assert mj_distance < 0.0
            assert abs(evidence["min_distance_m"] - mj_distance) < 0.01, (
                f"kernel {evidence['min_distance_m']} vs MuJoCo {mj_distance}"
            )
        finally:
            rclpy.shutdown()
    finally:
        terminate_kernel(proc)
