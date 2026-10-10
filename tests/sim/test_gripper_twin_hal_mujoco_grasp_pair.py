# SPDX-License-Identifier: Apache-2.0
"""Control pair: a declared grasp target is reachable, an undeclared one stops, on the twin.

The grasp-target chain of real pick-and-place design §2.1 / §2.4
(``docs/reference/real-pick-place-design.md``), end to end on one graph:

    dispatch-shaped GraspDeclaration (no region) on /openral/grasp_declaration
      -> SimSensorBridge + SimAttachmentEvidenceTracker measure the target's box
      -> /openral/attachment_state envelope -> World State node -> /openral/world_state_fast
      -> safety_kernel_node (world-voxel check ON, 20 mm margin, grasp_allowance_enabled)

The same approach runs twice. **Undeclared**: the open fingers descending beside the block
come within the 20 mm margin of its cells and the kernel refuses with ``KIND_COLLISION`` on a
finger link. **Declared**: the identical descent is accepted, the fingers close, the sim
tracker attaches the block (attachment revision advances), the kernel hands the exemption
over to the carried payload and keeps it while the payload is inside the measured region,
and a later approach of the same finger to a different obstacle (a post outside the region)
is still refused.

**The region after the grasp, two layers.** The first cut of the sim producer re-measured
the target's box at the body's *live* pose on every envelope, so after the grasp the region
rode along with the carried block and the handover never retired (this test found it). Both
layers now hold it still: the sim producer freezes the region at the attach transition (the
pre-grasp target volume), and the kernel latches the region at the handover edge regardless
and would ignore a later move (``safety.grasp_region_moved_after_handover ... ignored``;
exercised by the kernel's own gtest, not here, because a correct producer never moves it).
So lifting the payload out of the box retires the exemption and the post stop discloses
``grasp_exemption_active=0``.

What is real (CLAUDE.md §1.11): ``safety_kernel_node`` (colcon binary, lifecycle-driven,
attached check on as ``deploy_e2e.launch.py`` sets it for sim), the World State lifecycle
node (in-process), ``SimSensorBridge`` + ``SimAttachedHAL`` on a compiled ``MjModel``, the
sim attachment tracker and its grasp-region producer, ``openral_msgs`` on a real DDS graph.

What is not, and why:

* **The twin.** The OpenArm twin has two gripper parent links and the tracker refuses it
  (``_sim_attachment_evidence.py``, "needs one gripper parent link"); Franka's attach link
  does not resolve. So the twin is the same kind of rig
  ``tests/unit/test_sim_place_phase_witness.py`` uses — a compiled MJCF behind the
  ``SimRollout`` boundary recorder
  ``tests/unit/fakes/fake_sim_env.FakeSimEnv`` — reshaped into a 2-DoF gantry (``slide_x``,
  ``slide_z``) carrying a parallel gripper, so the kernel's collision model can be the
  twin's own geometry exactly (every kernel box below is a box geom in the MJCF).
* **The actuator.** ``FakeSimEnv`` has no integrator, so this test executes what the kernel
  approves: an accepted chunk's joint targets are written into ``qpos`` and ``mj_forward``'d,
  and a carried block is moved rigidly with the hand (the tracker, not the test, decides it
  is carried). The bridge's own idle stepper then runs the tracker at its post-step hook.
* **The occupancy grid.** Not octomap (depth synthesis + octree is heavy for a <10 min
  test): the grid is rasterised directly from the twin's live collision geometry on the
  ``openral_hal._grasp_target.VoxelLattice`` convention (20 mm cells, the real resolution;
  a cell is occupied iff its cube overlaps a collision box) and published on
  ``/openral/world_voxels`` with the kernel's QoS at 10 Hz. Payload masking mirrors the
  bridge: bodies in ``SimAttachedHAL.read_attached_body_ids()`` are dropped from the raster,
  but only once the test "re-integrates" the map — until then the pre-grasp cells stay, which
  is exactly the window the kernel's handover exists for.
* **Dispatch.** A runner-shaped publisher (same message, QoS and fields as
  ``rskill_runner_node``'s arm/retract) rather than the runner itself.

Gated on ``OPENRAL_TEST_ROS_LIVE=1`` (listed in ``scripts/ros_live_tests.sh``); needs the
colcon overlay (``openral_msgs``, ``openral_safety_kernel``) and ``packages/world_state``
on ``PYTHONPATH``.
"""

from __future__ import annotations

import json
import os
import pathlib
import tempfile
import time
import uuid
from typing import Any

import numpy as np
import pytest

_LIVE = os.environ.get("OPENRAL_TEST_ROS_LIVE") == "1" and bool(os.environ.get("ROS_DISTRO"))
pytestmark = pytest.mark.skipif(
    not _LIVE,
    reason="live-ROS test — set OPENRAL_TEST_ROS_LIVE=1 with ROS 2 + the colcon overlay "
    "sourced (scripts/ros_live_tests.sh does both).",
)

# ── Geometry, all in the robot base frame (metres) ───────────────────────────
# The base body sits off the world origin so a region or grid left in the world frame
# cannot pass by accident.
_BASE = np.array([0.1, 0.0, 0.05])
_RES = 0.02  # the real octomap resolution
_GRID_ORIGIN = (0.0, -0.16, 0.0)
_GRID_SIZE = (30, 16, 24)  # 0.60 x 0.32 x 0.48 m

# Table: top face z = 0.12. Block: 40 x 40 x 80 mm, sunk 0.5 mm into the table (a real
# resting contact for the support witness), so its box spans z 0.1195 .. 0.1995.
_TABLE_C, _TABLE_H = (0.36, 0.0, 0.10), (0.16, 0.10, 0.02)
_BLOCK_C, _BLOCK_H = (0.30, 0.0, 0.1595), (0.02, 0.02, 0.04)
# A post standing on the table, outside the block's region: the "different obstacle". Its
# top (z 0.32) stays 40 mm under the palm at carry height, so only a finger can reach it.
_POST_C, _POST_H = (0.48, 0.0, 0.22), (0.02, 0.02, 0.10)

# Fingers hang 100 mm below the hand at x = -/+ 40 mm; pads are 10 x 40 x 60 mm boxes, so an
# open pad's inner face is 35 mm from the hand axis: 15 mm from the block's face when the
# hand is centred over it — inside the 20 mm margin, which is the whole point.
_FINGER_X, _FINGER_Z = 0.040, -0.10
_PAD_H = (0.005, 0.02, 0.03)
_PALM_H = (0.06, 0.03, 0.01)
_CLOSE_Q = 0.0155  # pad centre at 24.5 mm: 0.5 mm into the block, so MuJoCo sees contact

_MARGIN_M = 0.02  # openral_core.depth_extrinsic.REAL_WORLD_VOXEL_MARGIN_M, asserted below

# Joint order = description order = ActionChunk.flat order.
_JOINTS = ("slide_x", "slide_z", "left_finger", "right_finger")
_PREGRASP = (0.30, 0.40, 0.0, 0.0)
# Pads span z 0.17 .. 0.23 — 50 mm over the table. The region's lower face sits one 20 mm
# voxel above the table (the real producer's rule), so the block's bottom cell layer
# (z 0.12 .. 0.14) is NOT exempt and the open pads must clear it by the 20 mm margin.
_GRASP = (0.30, 0.30, 0.0, 0.0)
_CLOSED = (0.30, 0.30, _CLOSE_Q, _CLOSE_Q)
_NUDGE = (0.30, 0.31, _CLOSE_Q, _CLOSE_Q)  # block origin z 0.1695: still in the region
_LIFTED = (0.30, 0.37, _CLOSE_Q, _CLOSE_Q)  # block origin z 0.2295: out of the region
_TOWARD_POST = (0.38, 0.37, _CLOSE_Q, _CLOSE_Q)  # right pad 50.5 mm from the post: clear
_AT_POST = (0.42, 0.37, _CLOSE_Q, _CLOSE_Q)  # right pad 10.5 mm from the post: refused

_TARGET = "sim:block"
_RSKILL = "openral/grasp-pair-twin"


def _mjcf() -> str:
    def at(c: tuple[float, float, float]) -> str:
        return " ".join(f"{v:.4f}" for v in np.asarray(c) + _BASE)

    def size(h: tuple[float, float, float]) -> str:
        return " ".join(f"{v:.4f}" for v in h)

    # contype/conaffinity: robot 1/1, block 1/1, scenery 4/5 — every robot/block geom
    # contacts the scenery and each other; scenery never contacts scenery.
    return f"""
<mujoco model="grasp_pair_twin">
  <worldbody>
    <body name="base" pos="{at((0.0, 0.0, 0.0))}"/>
    <body name="table" pos="{at(_TABLE_C)}">
      <geom name="table_top" type="box" size="{size(_TABLE_H)}" contype="4" conaffinity="5"/>
    </body>
    <body name="post" pos="{at(_POST_C)}">
      <geom name="post" type="box" size="{size(_POST_H)}" contype="4" conaffinity="5"/>
    </body>
    <body name="carriage" pos="{at((0.0, 0.0, 0.0))}">
      <joint name="slide_x" type="slide" axis="1 0 0"/>
      <inertial pos="0 0 0" mass="0.1" diaginertia="1e-4 1e-4 1e-4"/>
      <body name="hand">
        <joint name="slide_z" type="slide" axis="0 0 1"/>
        <geom name="palm" type="box" size="{size(_PALM_H)}" contype="1" conaffinity="1"/>
        <body name="left_finger" pos="{-_FINGER_X} 0 {_FINGER_Z}">
          <joint name="left_finger_joint" type="slide" axis="1 0 0"/>
          <geom name="left_pad" type="box" size="{size(_PAD_H)}" contype="1" conaffinity="1"/>
        </body>
        <body name="right_finger" pos="{_FINGER_X} 0 {_FINGER_Z}">
          <joint name="right_finger_joint" type="slide" axis="-1 0 0"/>
          <geom name="right_pad" type="box" size="{size(_PAD_H)}" contype="1" conaffinity="1"/>
        </body>
      </body>
    </body>
    <body name="block" pos="{at(_BLOCK_C)}">
      <freejoint name="block_joint"/>
      <geom name="block" type="box" size="{size(_BLOCK_H)}" mass="0.1" contype="1"
            conaffinity="1"/>
    </body>
  </worldbody>
</mujoco>
"""


def _description() -> Any:
    from openral_core import (
        BoxShape,
        ControlMode,
        EmbodimentKind,
        EndEffectorSpec,
        JointSpec,
        JointType,
        LinkCollisionGeometry,
        RobotCapabilities,
        RobotDescription,
        SafetyEnvelope,
    )

    def joint(
        name: str, parent: str, child: str, axis: tuple[float, float, float], **kw: Any
    ) -> JointSpec:
        return JointSpec(
            name=name,
            joint_type=JointType.PRISMATIC,
            parent_link=parent,
            child_link=child,
            axis_xyz=axis,
            **kw,
        )

    return RobotDescription(
        name="grasp_pair_twin",
        embodiment_kind=EmbodimentKind.MANIPULATOR,
        joints=[
            joint(
                "slide_x", "base", "carriage", (1.0, 0.0, 0.0), sim_joint_name="slide_x", role="arm"
            ),
            joint(
                "slide_z", "carriage", "hand", (0.0, 0.0, 1.0), sim_joint_name="slide_z", role="arm"
            ),
            joint(
                "left_finger",
                "hand",
                "left_finger",
                (1.0, 0.0, 0.0),
                origin_xyz=(-_FINGER_X, 0.0, _FINGER_Z),
                sim_joint_name="left_finger_joint",
                role="gripper",
            ),
            joint(
                "right_finger",
                "hand",
                "right_finger",
                (-1.0, 0.0, 0.0),
                origin_xyz=(_FINGER_X, 0.0, _FINGER_Z),
                sim_joint_name="right_finger_joint",
                role="gripper",
            ),
        ],
        end_effectors=[
            EndEffectorSpec(name="parallel_gripper", kind="parallel_gripper", parent_link="hand")
        ],
        collision_geometry=[
            LinkCollisionGeometry(link_name="hand", shape=BoxShape(half_extents_m=_PALM_H)),
            LinkCollisionGeometry(link_name="left_finger", shape=BoxShape(half_extents_m=_PAD_H)),
            LinkCollisionGeometry(link_name="right_finger", shape=BoxShape(half_extents_m=_PAD_H)),
        ],
        allowed_collision_pairs=[],
        capabilities=RobotCapabilities(
            supported_control_modes=[ControlMode.JOINT_POSITION], embodiment_tags=["synthetic"]
        ),
        safety=SafetyEnvelope(),
    )


def _kernel_params(description: Any) -> dict[str, object]:
    """The sim deploy launch's kernel parameters, at the REAL world-voxel margin.

    ``deploy_e2e.launch.py`` runs sim at 0 mm; the grasp exemption only means something
    when the margin is non-zero, so this sets the real 20 mm explicitly. The attached check
    is on, as the launch sets it for every sim robot with collision geometry.
    """
    from openral_safety.envelope_loader import collision_params_from_description

    params: dict[str, object] = {
        "n_dof": len(_JOINTS),
        "robot_name": description.name,
        "joint_position_min": [0.0, 0.0, 0.0, 0.0],
        "joint_position_max": [1.0, 1.0, 0.04, 0.04],
        "joint_velocity_max": [10.0] * 4,
        "joint_torque_max": [100.0] * 4,
        "world_voxel_enabled": True,
        "world_voxel_margin_m": _MARGIN_M,
        "world_voxel_deadline_ms": 1000.0,
        "world_voxel_max_cells": int(np.prod(_GRID_SIZE)),
        "attached_collision_enabled": True,
        "attached_collision_margin_m": 0.0,
        "attached_collision_deadline_ms": 2000.0,
        "grasp_allowance_enabled": True,
        "grasp_contact_links": [j.child_link for j in description.joints if j.role == "gripper"],
        "collision_joint_names": list(_JOINTS),
        "collision_state_deadline_ms": 2000.0,
        "collision_seed_dt_s": 0.0,
    }
    params.update(collision_params_from_description(description))
    return params


def _raster(model: Any, data: Any, exclude: frozenset[int]) -> Any:
    """Rasterise the twin's non-robot collision boxes onto the base-frame lattice.

    A cell is occupied iff its cube overlaps a collision box by more than 1 um (so a face
    lying on a cell boundary does not mark the neighbour). Every scenery geom here is an
    axis-aligned box — asserted, not assumed.
    """
    import mujoco
    from openral_hal._grasp_target import VoxelLattice

    sx, sy, sz = _GRID_SIZE
    lo_idx = np.stack(np.meshgrid(np.arange(sx), np.arange(sy), np.arange(sz), indexing="ij"), -1)
    cell_lo = np.asarray(_GRID_ORIGIN) + lo_idx * _RES
    cell_hi = cell_lo + _RES
    occupied = np.zeros((sx, sy, sz), dtype=bool)
    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
    scenery = {"table", "post", "block"}
    for g in range(model.ngeom):
        body = int(model.geom_bodyid[g])
        if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body) not in scenery:
            continue
        if body in exclude:
            continue
        assert model.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX
        assert np.allclose(data.geom_xmat[g].reshape(3, 3), np.eye(3), atol=1e-9)
        centre = data.geom_xpos[g] - data.xpos[base_id]
        lo, hi = centre - model.geom_size[g], centre + model.geom_size[g]
        # 1 um of slack: a face on a cell boundary must not mark the neighbour by rounding.
        occupied |= np.all((cell_lo < hi - 1e-6) & (cell_hi > lo + 1e-6), axis=-1)
    # Flat index is x fastest: transpose to (z, y, x) before ravel.
    return VoxelLattice(
        frame_id="base_link",
        origin=_GRID_ORIGIN,
        orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
        resolution=_RES,
        size=_GRID_SIZE,
        occupancy=occupied.transpose(2, 1, 0).ravel().astype(np.uint8),
    )


def _cells_of(lattice: Any, centre: tuple[float, float, float], half: Any) -> set[int]:
    """Flat indices of occupied cells whose centre lies inside an axis-aligned box."""
    idx = np.flatnonzero(lattice.occupancy)
    centres = lattice.occupied_centers()
    inside = np.all(np.abs(centres - np.asarray(centre)) <= np.asarray(half) + 1e-9, axis=1)
    return {int(i) for i in idx[inside]}


def test_declared_grasp_target_is_reachable_and_undeclared_stops() -> None:
    rclpy = pytest.importorskip("rclpy")
    mujoco = pytest.importorskip("mujoco")
    pytest.importorskip("openral_msgs")
    lifecycle_node = pytest.importorskip(
        "openral_world_state_ros.lifecycle_node",
        reason="packages/world_state must be on PYTHONPATH",
    )

    from geometry_msgs.msg import Point, Quaternion
    from openral_core import GraspDeclaration
    from openral_core.depth_extrinsic import REAL_WORLD_VOXEL_MARGIN_M
    from openral_hal.sim_attached import SimAttachedHAL
    from openral_hal.sim_sensor_bridge import SimSensorBridge
    from openral_msgs.msg import (
        ActionChunk,
        AttachmentState,
        FailureTrigger,
        OccupancyVoxels,
        WorldStateStamped,
    )
    from openral_msgs.msg import GraspDeclaration as GraspDeclarationMsg
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.lifecycle import TransitionCallbackReturn
    from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
    from sensor_msgs.msg import JointState
    from std_msgs.msg import Empty
    from std_srvs.srv import Trigger

    from tests.sim.safety._kernel_subprocess import (
        activate_kernel_node,
        isolated_domain_id,
        start_kernel,
        terminate_kernel,
    )
    from tests.unit.fakes.fake_sim_env import FakeSimEnv

    assert REAL_WORLD_VOXEL_MARGIN_M == _MARGIN_M

    model = mujoco.MjModel.from_xml_string(_mjcf())
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    description = _description()
    env: Any = FakeSimEnv(handles=(model, data))  # the SimRollout subset SimAttachedHAL uses
    hal = SimAttachedHAL(env, description)
    block_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "block")
    block_qpos = int(model.jnt_qposadr[model.body_jntadr[block_id]])
    qpos_of = {
        name: int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, sim)])
        for name, sim in zip(
            _JOINTS,
            ("slide_x", "slide_z", "left_finger_joint", "right_finger_joint"),
            strict=True,
        )
    }

    # The lattice and the cells the assertions name, fixed before anything is spawned.
    pre_grasp_world = _raster(model, data, frozenset())
    block_cells = _cells_of(pre_grasp_world, _BLOCK_C, _BLOCK_H)
    post_cells = _cells_of(pre_grasp_world, _POST_C, _POST_H)
    assert len(block_cells) == 2 * 2 * 4, sorted(block_cells)
    assert len(post_cells) == 2 * 2 * 10, sorted(post_cells)

    node_name = f"safety_kernel_grasp_pair_{uuid.uuid4().hex[:8]}"
    with tempfile.TemporaryDirectory() as td:
        log_path = pathlib.Path(td) / "kernel.log"
        proc = start_kernel(
            _kernel_params(description), node_name, isolated_domain_id(), log_path=log_path
        )

        def kernel_log() -> str:
            return log_path.read_text(encoding="utf-8", errors="replace")

        try:
            time.sleep(1.5)
            rclpy.init()
            try:
                helper = rclpy.create_node("grasp_pair_helper")
                # Before the helper joins our executor: activation spins it on the global one.
                assert activate_kernel_node(node_name, helper), (
                    f"kernel activation failed:\n{kernel_log()}"
                )
                bridge_node = rclpy.create_node("grasp_pair_sim_hal")
                ws_node = lifecycle_node._WorldStateLifecycleNode()
                executor = SingleThreadedExecutor()
                for n in (helper, bridge_node, ws_node):
                    executor.add_node(n)

                def spin(seconds: float) -> None:
                    end = time.time() + seconds
                    while time.time() < end:
                        executor.spin_once(timeout_sec=0.02)

                def spin_until(predicate: Any, seconds: float = 10.0) -> bool:
                    end = time.time() + seconds
                    while time.time() < end:
                        executor.spin_once(timeout_sec=0.02)
                        if predicate():
                            return True
                    return False

                assert ws_node.trigger_configure() == TransitionCallbackReturn.SUCCESS
                assert ws_node.trigger_activate() == TransitionCallbackReturn.SUCCESS

                safe: dict[str, ActionChunk] = {}
                failures: list[FailureTrigger] = []
                estops: list[Empty] = []
                attachments: list[AttachmentState] = []
                world_states: list[WorldStateStamped] = []
                latched = QoSProfile(
                    reliability=QoSReliabilityPolicy.RELIABLE,
                    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
                    depth=1,
                )
                kernel_kl1 = QoSProfile(
                    reliability=QoSReliabilityPolicy.RELIABLE,
                    durability=QoSDurabilityPolicy.VOLATILE,
                    depth=1,
                )
                safe_sub = helper.create_subscription(
                    ActionChunk,
                    "/openral/safe_action",
                    lambda m: safe.__setitem__(m.trace_id, m),
                    10,
                )
                helper.create_subscription(
                    FailureTrigger, "/openral/failure/safety", failures.append, 50
                )
                helper.create_subscription(Empty, "/openral/estop", estops.append, 10)
                helper.create_subscription(
                    AttachmentState, "/openral/attachment_state", attachments.append, latched
                )
                helper.create_subscription(
                    WorldStateStamped,
                    "/openral/world_state_fast",
                    world_states.append,
                    kernel_kl1,
                )
                chunk_pub = helper.create_publisher(ActionChunk, "/openral/candidate_action", 10)
                voxel_pub = helper.create_publisher(
                    OccupancyVoxels, "/openral/world_voxels", kernel_kl1
                )
                joint_pub = helper.create_publisher(JointState, "/joint_states", 10)
                declare_pub = helper.create_publisher(
                    GraspDeclarationMsg, "/openral/grasp_declaration", latched
                )
                reset_client = helper.create_client(Trigger, "/openral/estop_reset")

                # ── The HAL node's job: /joint_states from the twin ─────────────
                hal.connect()

                def publish_joints() -> None:
                    state = hal.read_state()
                    msg = JointState()
                    msg.header.stamp = helper.get_clock().now().to_msg()
                    msg.name = list(state.name)
                    msg.position = list(state.position)
                    joint_pub.publish(msg)

                helper.create_timer(0.02, publish_joints)

                # ── The map: rasterised twin geometry, re-integrated on demand ──
                world = {"lattice": pre_grasp_world}

                def publish_grid() -> None:
                    lattice = world["lattice"]
                    grid = OccupancyVoxels()
                    grid.header.frame_id = lattice.frame_id
                    grid.header.stamp = helper.get_clock().now().to_msg()
                    grid.source_stamp = grid.header.stamp
                    grid.origin = Point(
                        x=lattice.origin[0], y=lattice.origin[1], z=lattice.origin[2]
                    )
                    grid.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
                    grid.resolution = lattice.resolution
                    grid.size_x, grid.size_y, grid.size_z = lattice.size
                    grid.occupancy = lattice.occupancy.tobytes()
                    voxel_pub.publish(grid)

                helper.create_timer(0.1, publish_grid)

                def reintegrate_map() -> None:
                    """What the depth -> octomap path converges to: the masked payload is gone."""
                    world["lattice"] = _raster(model, data, hal.read_attached_body_ids())
                    spin(0.4)

                bridge = SimSensorBridge(bridge_node, hal, description, viewer_enabled=False)
                bridge.setup()
                try:
                    assert bridge._attachment_tracker is not None, "the twin must arm the tracker"

                    # ── The actuator: execute what the kernel approved ──────────
                    def apply(q: tuple[float, ...]) -> None:
                        carried = bool(hal.read_attached_objects())
                        hand_before = np.array(
                            [data.qpos[qpos_of["slide_x"]], 0.0, data.qpos[qpos_of["slide_z"]]]
                        )
                        for name, value in zip(_JOINTS, q, strict=True):
                            data.qpos[qpos_of[name]] = value
                        if carried:
                            delta = np.array([q[0], 0.0, q[1]]) - hand_before
                            data.qpos[block_qpos : block_qpos + 3] += delta
                        mujoco.mj_forward(model, data)
                        spin(0.3)  # idle stepper -> tracker; /joint_states refresh

                    def send(trace: str, q: tuple[float, ...], *, expect_accept: bool) -> None:
                        seen = len(failures)
                        chunk = ActionChunk()
                        chunk.control_mode = 0  # JOINT_POSITION
                        chunk.horizon = 1
                        chunk.n_dof = len(_JOINTS)
                        chunk.flat = [float(v) for v in q]
                        chunk.rskill_id = _RSKILL
                        chunk.trace_id = trace
                        chunk_pub.publish(chunk)
                        spin_until(
                            (lambda: trace in safe)
                            if expect_accept
                            else (lambda: len(failures) > seen and bool(estops))
                        )
                        spin(0.3)  # a late verdict must still be visible
                        if expect_accept:
                            assert trace in safe, f"{trace}: refused, expected accept"
                            assert not estops, f"{trace}: accepted chunk fired the estop"
                            apply(q)
                        else:
                            assert trace not in safe, f"{trace}: accepted, expected refusal"
                            assert estops, f"{trace}: refusal without /openral/estop"

                    def last_collision() -> dict[str, Any]:
                        trigger = failures[-1]
                        assert trigger.kind == FailureTrigger.KIND_COLLISION
                        evidence: dict[str, Any] = json.loads(trigger.evidence_json)
                        assert evidence["collision_kind"] == "world"
                        return evidence

                    def reset_estop() -> None:
                        assert reset_client.wait_for_service(timeout_sec=5.0)
                        spin(0.3)
                        future = reset_client.call_async(Trigger.Request())
                        assert spin_until(future.done, 5.0) and future.result().success
                        estops.clear()
                        spin(0.3)

                    assert spin_until(
                        lambda: (
                            chunk_pub.get_subscription_count() >= 1
                            and safe_sub.get_publisher_count() >= 1
                        )
                    )
                    apply(_PREGRASP)
                    assert spin_until(lambda: len(world_states) > 0), "no WorldStateStamped"
                    spin(1.0)

                    # ════ Pass 1 — undeclared: the approach stops ════════════════
                    send("u-pregrasp", _PREGRASP, expect_accept=True)
                    send("u-descend", _GRASP, expect_accept=False)
                    evidence = last_collision()
                    assert evidence["link_a"] in {"left_finger", "right_finger"}, evidence
                    voxel = int(evidence["link_b_or_object"].removeprefix("voxel_"))
                    assert voxel in block_cells, (voxel, sorted(block_cells))
                    assert 0.0 < evidence["min_distance_m"] <= _MARGIN_M, evidence
                    first_stop = [
                        ln for ln in kernel_log().splitlines() if "safety.collision" in ln
                    ]
                    assert len(first_stop) == 1 and "grasp_exemption_active=0" in first_stop[0]
                    reset_estop()

                    # ════ Pass 2 — declared: the same approach goes through ══════
                    revision_before = attachments[-1].revision if attachments else 0
                    declaration = GraspDeclaration(
                        target_id=_TARGET,
                        object_id=_TARGET,
                        contact_links=("left_finger", "right_finger"),
                        rskill_id=_RSKILL,
                        trace_id=uuid.uuid4().hex,
                        timeout_s=60.0,
                        stamp_ns=int(helper.get_clock().now().nanoseconds),
                    )
                    wire = GraspDeclarationMsg()
                    declaration.fill_idl(wire)
                    assert wire.region_valid is False  # dispatch never supplies a region
                    declare_pub.publish(wire)
                    assert spin_until(
                        lambda: f"safety.grasp_region_armed target={_TARGET}" in kernel_log()
                    ), f"the kernel never armed the measured region\n{kernel_log()}"
                    # The region the kernel armed is the producer's measurement. The
                    # kernel's log can land before this test's own WorldState copy does.
                    assert spin_until(lambda: any(m.grasp_declaration_valid for m in world_states))
                    measured = next(
                        m.grasp_declaration
                        for m in reversed(world_states)
                        if m.grasp_declaration_valid
                    )
                    assert measured.region_valid
                    assert measured.region.evidence_ref == "mujoco_body_subtree:block"
                    assert measured.region.frame_id == "base_link"
                    centre = measured.region.pose.position
                    # The block's box with its lower face lifted one voxel off the table,
                    # the real producer's rule (``_grasp_target.target_region_from_mask``).
                    lifted = (_BLOCK_C[0], _BLOCK_C[1], _BLOCK_C[2] + _RES / 2.0)
                    assert (centre.x, centre.y, centre.z) == pytest.approx(lifted, abs=1e-6)
                    bottom = centre.z - measured.region.half_extents.z
                    assert bottom == pytest.approx(_BLOCK_C[2] - _BLOCK_H[2] + _RES, abs=2e-4)

                    send("d-descend", _GRASP, expect_accept=True)
                    send("d-close", _CLOSED, expect_accept=True)
                    assert spin_until(
                        lambda: any(
                            [o.object_id for o in m.objects] == [_TARGET] for m in attachments
                        )
                    ), "the sim tracker never attached the block"
                    attached = next(
                        m for m in attachments if [o.object_id for o in m.objects] == [_TARGET]
                    )
                    assert attached.revision > revision_before, (
                        "attachment revision did not advance"
                    )
                    assert spin_until(
                        lambda: any(
                            m.attachment_revision == attached.revision
                            and [o.object_id for o in m.attached_objects] == [_TARGET]
                            for m in world_states[-50:]
                        )
                    ), "the envelope revision never reached the kernel's input"
                    assert spin_until(
                        lambda: f"safety.grasp_region_handover target={_TARGET}" in kernel_log()
                    ), kernel_log()

                    # Handover: the map has not re-integrated, so the fingers sit in the
                    # block's stale cells. The payload is still inside the region, so the
                    # exemption must still hold.
                    send("d-nudge", _NUDGE, expect_accept=True)
                    assert "safety.grasp_region_dropped" not in kernel_log()
                    send("d-lift", _LIFTED, expect_accept=True)
                    reintegrate_map()
                    assert not (_cells_of(world["lattice"], _BLOCK_C, _BLOCK_H)), (
                        "the masked payload's cells must leave the re-integrated map"
                    )

                    # The region the kernel holds now, as the producer measures it.
                    lifted_region = next(
                        m.grasp_declaration.region.pose.position
                        for m in reversed(world_states)
                        if m.grasp_declaration_valid and m.grasp_declaration.region_valid
                    )

                    # Lifted out: the next candidate should retire the exemption for good
                    # (design §2.1 "Handover"), and the finger is held at the margin again —
                    # against a different obstacle, outside the region.
                    send("d-toward-post", _TOWARD_POST, expect_accept=True)
                    exited = (
                        f"safety.grasp_region_dropped reason=handover_exit target={_TARGET}"
                        in kernel_log()
                    )
                    send("d-at-post", _AT_POST, expect_accept=False)
                    evidence = last_collision()
                    assert evidence["link_a"] == "right_finger", evidence
                    voxel = int(evidence["link_b_or_object"].removeprefix("voxel_"))
                    assert voxel in post_cells, (voxel, sorted(post_cells))
                    assert 0.0 < evidence["min_distance_m"] <= _MARGIN_M, evidence
                    stop = [
                        line
                        for line in kernel_log().splitlines()
                        if "safety.collision" in line and "a=right_finger" in line
                    ]
                    assert stop, kernel_log()
                    # The producer froze its region at attach, so the lifted envelope still
                    # carries the pre-grasp box; the kernel latched the same box at handover.
                    # Either layer alone retires the exemption on the lift.
                    assert exited, (lifted_region, _BLOCK_C, kernel_log())
                    assert f"safety.grasp_region_latched target={_TARGET}" in kernel_log()
                    # A correct producer never moves the region after handover, so the
                    # kernel's "moved ... ignored" branch must stay silent here.
                    assert kernel_log().count("safety.grasp_region_moved_after_handover") == 0, (
                        kernel_log()
                    )
                    assert "grasp_exemption_active=0" in stop[-1], stop
                finally:
                    bridge.teardown()
                    hal.disconnect()
                    ws_node.destroy_node()
                    bridge_node.destroy_node()
                    helper.destroy_node()
            finally:
                rclpy.try_shutdown()
        except AssertionError as exc:
            raise AssertionError(
                f"{exc}\n\n--- safety_kernel_node log ---\n{kernel_log()}"
            ) from exc
        finally:
            terminate_kernel(proc)
