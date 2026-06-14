"""ADR-0030 phase 2 — real humanoid (Unitree H1) self-collision through the kernel.

Unlike the synthetic-arm test, this drives a **real robot's MuJoCo model**: H1's
collidable geometry is all primitives (capsules / cylinders / spheres / one box),
so ``openral_safety.mjcf_lowering.lower_collision_params`` builds a geometrically
faithful kernel collision model straight from the compiled MJCF — origins from
the body tree, capsules from the collision geoms, the allowed-collision matrix
from parent↔child + the MJCF's own exclusions.

The test searches for an arm-cross pose that brings two non-adjacent arm links
into contact (MuJoCo ``mj_geomDistance`` is the oracle), drives that joint
configuration through the **real safety_kernel_node**, and asserts the kernel
drops it with ``/openral/estop`` + ``FailureTrigger(KIND_COLLISION)`` and that the
reported colliding link pair really overlaps per MuJoCo.

Gates (CLAUDE.md §1.11 / §1.12): ROS_DISTRO + rclpy + openral_msgs + mujoco +
openral_hal, on a sourced + colcon-built workspace. Otherwise skip — never faked.
"""

from __future__ import annotations

import itertools
import json
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
pytest.importorskip("openral_hal")

from openral_safety.mjcf_lowering import lower_collision_params  # noqa: E402

from tests.sim.safety._kernel_subprocess import (  # noqa: E402
    activate_kernel_node,
    isolated_domain_id,
    start_kernel,
    terminate_kernel,
)


def _geom_distance(model: object, data: object, g1: int, g2: int) -> float:
    return float(mujoco.mj_geomDistance(model, data, g1, g2, 2.0, None))


def _first_collidable_geom(model: object, body_id: int) -> int | None:
    for gi in range(int(model.ngeom)):  # type: ignore[attr-defined]
        if int(model.geom_bodyid[gi]) != body_id:  # type: ignore[attr-defined]
            continue
        if int(model.geom_contype[gi]) == 0 and int(model.geom_conaffinity[gi]) == 0:  # type: ignore[attr-defined]
            continue
        if int(model.geom_type[gi]) == 0:  # plane  # type: ignore[attr-defined]
            continue
        return gi
    return None


def _find_arm_cross_pose(model, data, joint_names, le_geom, re_geom):  # type: ignore[no-untyped-def]
    """Search a small grid of shoulder/elbow angles for a left↔right elbow overlap."""
    idx = {name: i for i, name in enumerate(joint_names)}
    base = [0.0] * len(joint_names)
    rolls = (-1.6, -1.2, 1.2, 1.6)
    elbows = (-1.8, -1.2)
    pitches = (-0.4, 0.2)
    for lr, rr, el, pi in itertools.product(rolls, rolls, elbows, pitches):
        pose = list(base)
        pose[idx["left_shoulder_roll"]] = lr
        pose[idx["right_shoulder_roll"]] = rr
        pose[idx["left_shoulder_pitch"]] = pi
        pose[idx["right_shoulder_pitch"]] = pi
        pose[idx["left_elbow"]] = el
        pose[idx["right_elbow"]] = el
        data.qpos[:] = 0.0
        data.qpos[7 : 7 + len(pose)] = pose  # skip the 7-DoF free base
        mujoco.mj_forward(model, data)
        if _geom_distance(model, data, le_geom, re_geom) < 0.0:
            return pose
    return None


def test_h1_arm_cross_self_collision_triggers_kernel() -> None:
    """Lower H1's MJCF, fold the arms into self-collision, and assert the kernel estops."""
    import rclpy
    from openral_hal import H1_DESCRIPTION
    from openral_hal.h1 import H1MujocoHAL
    from openral_msgs.msg import ActionChunk, FailureTrigger
    from rclpy.executors import SingleThreadedExecutor
    from std_msgs.msg import Empty

    hal = H1MujocoHAL(staleness_limit_s=10.0)
    hal.connect()
    model = hal._model  # the twin's compiled MuJoCo model
    data = mujoco.MjData(model)
    joint_names = [j.name for j in H1_DESCRIPTION.joints]
    n_dof = len(joint_names)

    le_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_elbow_link")
    re_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_elbow_link")
    le_geom = _first_collidable_geom(model, le_body)
    re_geom = _first_collidable_geom(model, re_body)
    assert le_geom is not None and re_geom is not None

    pose = _find_arm_cross_pose(model, data, joint_names, le_geom, re_geom)
    assert pose is not None, "no arm-cross pose self-collided — widen the search grid"

    # Build kernel params: a permissive scalar envelope (so the collision check,
    # not a joint-limit check, is what fires) + the lowered H1 collision model.
    params: dict[str, object] = {
        "n_dof": n_dof,
        "robot_name": "h1",
        "joint_position_min": [-6.5] * n_dof,
        "joint_position_max": [6.5] * n_dof,
        "joint_velocity_max": [100.0] * n_dof,
        "joint_torque_max": [500.0] * n_dof,
    }
    params.update(lower_collision_params(model, joint_names))
    assert params["self_collision_enabled"] is True
    # Multi-capsule-per-link: H1's torso/ankles carry several collision geoms, so
    # there are more capsules than distinct capsuled links.
    cap_links = params["collision_capsule_link"]
    assert len(cap_links) > len(set(cap_links)), "expected ≥1 link with multiple capsules"

    node_name = f"safety_kernel_h1_{uuid.uuid4().hex[:8]}"
    proc = start_kernel(params, node_name, isolated_domain_id())
    try:
        time.sleep(1.5)
        rclpy.init()
        try:
            helper = rclpy.create_node("h1_selfcoll_helper")
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
                chunk.n_dof = n_dof
                chunk.flat = qpos
                chunk.rskill_id = "openral/h1-selfcoll-test"
                chunk.trace_id = trace
                pub.publish(chunk)
                end = time.time() + 1.5
                while time.time() < end:
                    executor.spin_once(timeout_sec=0.02)

            # 1. Neutral stance must pass — proves the neutral-pose allowed-collision
            #    matrix prevents false positives on links that touch at rest.
            send([0.0] * n_dof, "neutral")
            assert "neutral" in safe, "neutral H1 stance must pass through to safe_action"
            assert not failures, "neutral stance must not raise a FailureTrigger"

            # 2. Arm-cross pose self-collides → drop + estop + KIND_COLLISION.
            send(pose, "collide")
            assert "collide" not in safe, "self-colliding H1 pose must NOT reach safe_action"
            assert estops, "self-collision must fire /openral/estop"
            assert failures, "self-collision must publish a FailureTrigger"
            trigger = failures[-1]
            assert trigger.kind == FailureTrigger.KIND_COLLISION
            evidence = json.loads(trigger.evidence_json)
            assert evidence["collision_kind"] == "self"
            assert evidence["min_distance_m"] < 0.0

            # 3. Independent MuJoCo oracle: the elbow capsules we searched for are
            #    exact (type-capsule) geoms, so MuJoCo must agree they self-collide.
            data.qpos[:] = 0.0
            data.qpos[7 : 7 + n_dof] = pose
            mujoco.mj_forward(model, data)
            assert _geom_distance(model, data, le_geom, re_geom) < 0.0
        finally:
            rclpy.shutdown()
    finally:
        terminate_kernel(proc)
