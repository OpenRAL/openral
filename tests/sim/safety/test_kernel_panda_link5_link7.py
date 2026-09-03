"""Issue #191 — `panda_link5` ↔ `panda_link7` through the real safety kernel.

The pair shipped ACM-exempted "under protest" from PR #169: its real geometry
interpenetrates over part of the (joint6, joint7) grid, but the kernel's OBB
envelopes fire on 86.38 % of that grid while only 6.60 % is real, with no margin
separating the populations. The exemption bought a usable robot at the cost of
an unchecked real self-collision.

This is the end-to-end evidence that both halves are now settled, driven through
the **real C++ safety_kernel_node** subprocess with the **shipped**
``robots/panda_mobile/robot.yaml`` — geometry, hulls and ACM as they land:

* the arm's own robosuite reset pose passes (the pair no longer false-stops);
* a pose whose real links genuinely interpenetrate is dropped, latched, and
  reported as ``KIND_COLLISION`` naming ``panda_link5``/``panda_link7``;
* the distance the kernel reports agrees with an independent convex-geometry
  oracle (``openral_hal.convex_distance``) run on the same MuJoCo meshes.

The unit-level counterpart is ``SelfCollisionHull.*`` in
``cpp/openral_safety_kernel/test/test_collision.cpp``; the hulls' containment
proof is ``tests/unit/test_collision_tight_geometry.py``.

Gates (CLAUDE.md §1.11 / §1.12): ROS_DISTRO + rclpy + openral_msgs + mujoco +
robosuite (the mesh source), and a sourced + colcon-built workspace. Otherwise
pytest.skip — never faked.
"""

from __future__ import annotations

import json
import math
import os
import time
import uuid
from pathlib import Path

import pytest

_ROS2_AVAILABLE = bool(os.environ.get("ROS_DISTRO"))
pytestmark = pytest.mark.skipif(
    not _ROS2_AVAILABLE, reason="ROS_DISTRO not set — requires a sourced ROS 2 install."
)
pytest.importorskip("rclpy")
pytest.importorskip("openral_msgs")
mujoco = pytest.importorskip("mujoco")
pytest.importorskip("robosuite", reason="robosuite ships the Panda collision meshes")
pytest.importorskip("scipy", reason="scipy.spatial.ConvexHull is the hull reference")

import numpy as np  # noqa: E402
from openral_core import BoxShape, RobotDescription  # noqa: E402
from openral_safety.envelope_loader import collision_params_from_description  # noqa: E402

from tests.sim.safety._kernel_subprocess import (  # noqa: E402
    activate_kernel_node,
    isolated_domain_id,
    start_kernel,
    terminate_kernel,
)

_MANIFEST = "robots/panda_mobile/robot.yaml"

# robosuite `PandaOmron.init_qpos` — what `robots: ["PandaMobile"]` resets to,
# and the pose the OBB check stops on when the pair is checked without hulls
# (box gap −11.68 mm, hull gap +21.81 mm).
_ARM_RESET = [
    0.0,
    math.pi / 16 - 0.2,
    0.0,
    -math.pi / 2 - math.pi / 3,
    0.0,
    math.pi - 0.4,
    math.pi / 4,
]
# The SRDF's own `transport` group state, which really does interpenetrate:
# box gap −11.68 mm and hull gap −5.64 mm, so both models agree it is a stop.
_ARM_TRANSPORT = [0.0, -0.5599, 0.0, -2.97, 0.0, 0.0, 0.785]


def _robot() -> RobotDescription:
    return RobotDescription.from_yaml(_MANIFEST)


def _kernel_params() -> tuple[dict[str, object], list[str], list[int]]:
    """(kernel params, link names, the qpos slots of panda_joint1..7)."""
    robot = _robot()
    collision = collision_params_from_description(robot)
    names = list(collision["collision_link_names"])  # type: ignore[arg-type]  # reason: ROS param list
    dof_index = list(collision["collision_dof_index"])  # type: ignore[arg-type]  # reason: ROS param list
    n_dof = max(dof_index) + 1
    arm_slots = [dof_index[names.index(f"panda_link{i}")] for i in range(1, 8)]
    params: dict[str, object] = {
        "n_dof": n_dof,
        "robot_name": robot.name,
        "joint_position_min": [-6.5] * n_dof,
        "joint_position_max": [6.5] * n_dof,
        "joint_velocity_max": [100.0] * n_dof,
        "joint_torque_max": [100.0] * n_dof,
    }
    params.update(collision)
    return params, names, arm_slots


def _qpos(arm: list[float], n_dof: int, arm_slots: list[int]) -> list[float]:
    """A full joint vector with the arm angles in their own slots, base at zero."""
    out = [0.0] * n_dof
    for slot, value in zip(arm_slots, arm, strict=True):
        out[slot] = value
    return out


def _oracle_gap(arm: list[float]) -> float:
    """Signed link5↔link7 distance from the real meshes, independently of the kernel.

    ``conv(mesh)`` rather than the mesh: the Panda's link5 and link7 collision
    meshes are convex to 1e-4 in volume ratio (collision-primitive-study §4.3),
    so the hull IS the mesh for this pair — which is exactly why the hulls can
    retire the exemption. Measured with ``openral_hal.convex_distance``, not
    ``mujoco.mj_geomDistance`` (see that module's docstring for why).
    """
    import sys

    from scipy.spatial import ConvexHull

    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
    from generate_tight_geometry import (
        _PANDA_XML,
        _robosuite_asset_root,
        _rpy_to_matrix,
        link_mesh_in_box_frame,
    )
    from openral_hal.convex_distance import ConvexBody, _signed_distance

    xml = _robosuite_asset_root() / _PANDA_XML
    geoms = {g.link_name: g for g in _robot().collision_geometry}
    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)
    data.qpos[:7] = arm
    mujoco.mj_kinematics(model, data)

    bodies = []
    for link in ("panda_link5", "panda_link7"):
        geom = geoms[link]
        assert isinstance(geom.shape, BoxShape)
        points = link_mesh_in_box_frame(
            xml, f"{link.removeprefix('panda_')}_collision", geom.origin_xyz_rpy
        )
        hull = ConvexHull(points)
        remap = {int(v): i for i, v in enumerate(hull.vertices)}
        faces = np.asarray([[remap[int(v)] for v in s] for s in hull.simplices])
        box_in_link = np.eye(4)
        box_in_link[:3, :3] = _rpy_to_matrix(*geom.origin_xyz_rpy[3:6])
        box_in_link[:3, 3] = geom.origin_xyz_rpy[:3]
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, link.removeprefix("panda_"))
        link_world = np.eye(4)
        link_world[:3, :3] = data.xmat[body_id].reshape(3, 3)
        link_world[:3, 3] = data.xpos[body_id]
        world = link_world @ box_in_link
        bodies.append(
            ConvexBody(
                core=points[hull.vertices] @ world[:3, :3].T + world[:3, 3],
                radius=0.0,
                faces=faces,
            )
        )
    return float(_signed_distance(bodies[0], bodies[1])[0])


def test_the_pair_is_no_longer_exempt() -> None:
    """The manifest and the SRDF both stop hiding the check (issue #191)."""
    robot = _robot()
    assert ("panda_link5", "panda_link7") not in robot.allowed_collision_pairs
    for link in ("panda_link5", "panda_link7"):
        geom = next(g for g in robot.collision_geometry if g.link_name == link)
        assert geom.tight_geometry is not None
        assert geom.tight_geometry.hull_vertices_m, f"{link} needs its stage-2 hull"


def test_oracle_separates_the_two_poses() -> None:
    """Establish the ground truth before asking the kernel to agree with it."""
    assert _oracle_gap(_ARM_RESET) > 0.0, "the reset pose is genuinely clear"
    assert _oracle_gap(_ARM_TRANSPORT) < 0.0, "the transport pose genuinely interpenetrates"


def test_kernel_clears_the_reset_pose_and_stops_the_real_collision() -> None:
    """The whole of #191, end to end: no false stop, and the real one still fires."""
    import rclpy
    from openral_msgs.msg import ActionChunk, FailureTrigger
    from rclpy.executors import SingleThreadedExecutor
    from std_msgs.msg import Empty

    params, _names, arm_slots = _kernel_params()
    n_dof = int(params["n_dof"])  # type: ignore[arg-type]  # reason: set above as int
    node_name = f"safety_kernel_link57_{uuid.uuid4().hex[:8]}"
    proc = start_kernel(params, node_name, isolated_domain_id())
    try:
        time.sleep(1.5)
        rclpy.init()
        try:
            helper = rclpy.create_node("link57_helper")
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

            def send(arm: list[float], trace: str) -> None:
                chunk = ActionChunk()
                chunk.control_mode = 0  # JOINT_POSITION
                chunk.horizon = 1
                chunk.n_dof = n_dof
                chunk.flat = _qpos(arm, n_dof, arm_slots)
                chunk.rskill_id = "openral/link5-link7-test"
                chunk.trace_id = trace
                pub.publish(chunk)
                end = time.time() + 1.0
                while time.time() < end:
                    executor.spin_once(timeout_sec=0.02)

            # 1. The arm's own reset pose. Its boxes overlap by 11.68 mm, so
            #    before the hull refinement this pose was the reason the pair
            #    could not simply be un-exempted (issue #191 Option A).
            send(_ARM_RESET, "reset")
            assert "reset" in safe, (
                "the robot's own reset pose must pass — a box-fidelity check stops here"
            )
            assert not failures, "the reset pose must not raise a FailureTrigger"

            # 2. A pose whose real geometry interpenetrates: still a stop, which
            #    is the half of #191 the exemption was hiding.
            send(_ARM_TRANSPORT, "transport")
            assert "transport" not in safe, "a real self-collision must NOT reach safe_action"
            assert estops, "a real self-collision must fire /openral/estop"
            assert failures, "a real self-collision must publish a FailureTrigger"

            trigger = failures[-1]
            assert trigger.kind == FailureTrigger.KIND_COLLISION
            evidence = json.loads(trigger.evidence_json)
            assert evidence["collision_kind"] == "self"
            assert {evidence["link_a"], evidence["link_b_or_object"]} == {
                "panda_link5",
                "panda_link7",
            }
            assert evidence["min_distance_m"] < 0.0

            # 3. The reported depth is the kernel's `box_box_distance` fallback,
            #    which on overlap is the more conservative of the two numbers.
            #    It must bracket the oracle rather than overstate the clearance.
            oracle = _oracle_gap(_ARM_TRANSPORT)
            assert evidence["min_distance_m"] <= oracle + 1e-9, (
                f"kernel {evidence['min_distance_m']} reported less penetration than the "
                f"oracle's {oracle}"
            )
        finally:
            rclpy.shutdown()
    finally:
        terminate_kernel(proc)
