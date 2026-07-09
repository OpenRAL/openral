"""Conservatism regression: an MJCF-lowered model with MISMATCHED joint names
enforces self-collision through the real kernel (the dof_index fix, end to end).

``openral deploy sim`` prefers the MJCF-lowered collision model
(``sim_e2e.launch.py`` → ``openral_safety.mjcf_lowering.lower_collision_params``)
over the manifest one. Before the dof_index fix that lowering keyed its
joint→column map by the *manifest* joint names but read the *MJCF* joint names;
real robots name them differently, so every ``collision_dof_index`` collapsed to
``-1`` and the kernel's forward kinematics froze the whole arm at its rest pose.
"self-collision check enabled" was logged but the check was a silent no-op for
franka / so100 / so101 in the prime deploy path.

This test drives the **real safety_kernel_node** with a 3-link MJCF whose joint
names (``Alpha``/``Beta``/``Gamma``) deliberately do NOT appear in the manifest
order passed to the lowering, and asserts the kernel returns DIFFERENT verdicts
for different commanded configurations:

* ``q = [0, 0, 0]`` (straight) → passes.
* ``q = [0, 2.4, 0]`` (bent, still clear) → passes.
* ``q = [0, π, 0]`` (folded: link1 and link3 — non-adjacent, not in the
  allowed-collision matrix — overlap) → dropped, estop, ``KIND_COLLISION``.

A *differential* verdict is only possible when the FK tracks the commanded
joints. Pre-fix (``dof_index`` all ``-1``) every configuration FK'd to the same
rest pose, so the folded config would have passed exactly like the straight one
— the regression this test pins.

Gates (CLAUDE.md §1.11 / §1.12): ROS_DISTRO + rclpy + openral_msgs + mujoco, on a
sourced + colcon-built workspace. Hermetic: inline MJCF, no asset download.
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

from openral_safety.mjcf_lowering import lower_collision_params  # noqa: E402

from tests.sim.safety._kernel_subprocess import (  # noqa: E402
    activate_kernel_node,
    isolated_domain_id,
    start_kernel,
    terminate_kernel,
)

# Three collinear links along +X (0.4 m apart, 0.36 m capsule, radius 0.05),
# each on a hinge about Z. The MJCF joint names are NOT the manifest order.
_TRI_LINK_MJCF = """
<mujoco>
  <worldbody>
    <body name="link1" pos="0 0 0">
      <joint name="Alpha" type="hinge" axis="0 0 1"/>
      <geom type="capsule" fromto="0 0 0 0.36 0 0" size="0.05"/>
      <body name="link2" pos="0.4 0 0">
        <joint name="Beta" type="hinge" axis="0 0 1"/>
        <geom type="capsule" fromto="0 0 0 0.36 0 0" size="0.05"/>
        <body name="link3" pos="0.4 0 0">
          <joint name="Gamma" type="hinge" axis="0 0 1"/>
          <geom type="capsule" fromto="0 0 0 0.36 0 0" size="0.05"/>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

# Manifest joint order — three names, none matching the MJCF's.
_MANIFEST_JOINTS = ["j_a", "j_b", "j_c"]


def _kernel_params() -> dict[str, object]:
    model = mujoco.MjModel.from_xml_string(_TRI_LINK_MJCF)
    collision = lower_collision_params(model, _MANIFEST_JOINTS)
    # The fix: movable joints map to manifest columns 0,1,2 by order, not by name.
    assert collision["collision_dof_index"] == [0, 1, 2], (
        "MJCF dof_index regressed to name-matching — kernel FK would freeze at rest"
    )
    params: dict[str, object] = {
        "n_dof": 3,
        "robot_name": "tri_link_mismatched_names",
        "joint_position_min": [-3.2, -3.2, -3.2],
        "joint_position_max": [3.2, 3.2, 3.2],
        "joint_velocity_max": [100.0, 100.0, 100.0],
        "joint_torque_max": [100.0, 100.0, 100.0],
    }
    params.update(collision)
    return params


def test_mjcf_lowered_model_enforces_self_collision_through_kernel() -> None:
    """Folded config dropped as KIND_COLLISION; straight + bent-clear pass."""
    import rclpy
    from openral_msgs.msg import ActionChunk, FailureTrigger
    from rclpy.executors import SingleThreadedExecutor
    from std_msgs.msg import Empty

    node_name = f"safety_kernel_mjcf_{uuid.uuid4().hex[:8]}"
    proc = start_kernel(_kernel_params(), node_name, isolated_domain_id())
    try:
        time.sleep(1.5)
        rclpy.init()
        try:
            helper = rclpy.create_node("mjcf_self_helper")
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

            def send(q: list[float], trace: str) -> None:
                chunk = ActionChunk()
                chunk.control_mode = 0  # JOINT_POSITION
                chunk.horizon = 1
                chunk.n_dof = 3
                chunk.flat = q
                chunk.rskill_id = "openral/mjcf-self-test"
                chunk.trace_id = trace
                pub.publish(chunk)
                end = time.time() + 1.0
                while time.time() < end:
                    executor.spin_once(timeout_sec=0.02)

            # Two clearly-different non-colliding configs both pass — already proof
            # the FK is live (a frozen-at-rest FK could still pass these, so the
            # folded case below is the decisive one).
            send([0.0, 0.0, 0.0], "straight")
            assert "straight" in safe, "straight arm must pass"
            send([0.0, 2.4, 0.0], "bent-clear")
            assert "bent-clear" in safe, "bent-but-clear arm must pass"
            assert not estops, "no clear config may estop"

            # Folded: link1 and link3 (non-adjacent) overlap. Pre-fix this FK'd to
            # the rest pose and passed; post-fix the kernel sees the real overlap.
            send([0.0, math.pi, 0.0], "folded")
            assert "folded" not in safe, "folded self-colliding config must NOT pass"
            assert estops, "folded self-collision must fire /openral/estop"
            assert failures, "folded self-collision must publish a FailureTrigger"
            trigger = failures[-1]
            assert trigger.kind == FailureTrigger.KIND_COLLISION
            evidence = json.loads(trigger.evidence_json)
            assert evidence["collision_kind"] == "self"
            assert {evidence["link_a"], evidence["link_b_or_object"]} == {"link1", "link3"}
            assert evidence["min_distance_m"] < 0.0, "interpenetration → negative distance"
        finally:
            rclpy.shutdown()
    finally:
        terminate_kernel(proc)
