"""The declaration-scoped place-approach allowance, fired deterministically.

ADR-0097's 2026-08-14 amendment (recalibrated by its Second Amendment, 2026-08-15) reduces a
declared payload's world-collision margin by ``min(1.5 x voxel, 4 cm)`` for occupancy cells
inside the producer-measured place-target region. Prior validation rounds armed the allowance
correctly (``safety.place_region_armed ... allowance_m=0.0375``) but rarely fired it — XR-1's
stochastic trajectories rarely landed the payload in the narrow band between reduced and
unreduced margin. This test removes the policy and drives the band directly.

Real throughout: real ``safety_kernel_node`` lifecycle node, real ``openral_msgs`` IDL, a real
dense occupancy grid on ``/openral/world_voxels``, a real attached payload plus a real
``PlaceDeclaration``/``PlaceRegion`` on ``/openral/world_state_fast``, real ``ActionChunk``
candidates on ``/openral/candidate_action``. No mocks, no simulator, no GPU (CLAUDE.md §1.11).

One-DoF prismatic carriage, so a chunk's single joint value *is* the payload's x position:

    payload sphere centre  = (q, 0, 0),         radius 20 mm
    occupied voxel centre  = (0.1, 0, 0),   half-edge 12.5 mm
    surface distance d(q)  = 0.0875 - q - 0.020 = 0.0675 - q

At the sim grid's 25 mm resolution the allowance is ``min(1.5 x 0.025, 0.04) = 0.0375 m``, so
against the 50 mm attached margin the band is ``0.0125 m < d <= 0.05 m``. Three chunks pin
the three regimes:

===============  ======  ==================  =====================================
q (m)            d (m)   undeclared          declared
===============  ======  ==================  =====================================
0.0              0.0675  accepted            accepted (allowance changes nothing)
0.0475           0.020   REFUSED             ACCEPTED  <- the allowance's verdict
0.0625           0.005   REFUSED             REFUSED, ``place_allowance_active=1``
===============  ======  ==================  =====================================

Last row: the hard stop behind the reduced margin is untouched and disclosed. Middle row is
the previously-unobserved accept, sized to break if the allowance is reduced, not just
deleted: 20 mm is below the pre-Second-Amendment cap (``min(one voxel, 2.5 cm)`` = 25 mm), so
reverting ``kPlaceApproachAllowanceVoxels`` (1.5 -> 1.0) or ``kMaxPlaceApproachAllowanceM``
(0.04 -> 0.025) turns that accept back to refusal.

A second test drives the sibling exemption on the same carriage, stood upright: the
support-contact witness (ADR-0092 D6, hazard log Entry 012) the vision pick now attaches at
ATTACH (``vision_attachment_bridge.region_attachment``, Isaac i42). A grasp-target region
payload pressed 2 mm into the support cell it was measured on (i42: -1.86 mm) is REFUSED
without the witness and ACCEPTED with it; lifted three cells, the kernel retires the witness
(``safety.support_witness_separated``), and the same witness republished cannot bring it back:
set down again, the same contact is REFUSED.

A third drives the vision pick's cell-closed region payload (Isaac i50): the tight held fit
leaves the target's own boundary cell ~4 mm deep at the attach-time snapshot — not embedded
residue — so a payload commanded 19 mm below its measured pose is REFUSED on it; the payload
closed over the cells it touches (what the bridge attaches) is ACCEPTED, a foreign cell outside
the closure still stops it, and a witness patch left at the tight footprint stops on the
support under the closure's corners.

Gates: ``OPENRAL_TEST_ROS_LIVE=1`` + ROS_DISTRO + rclpy + openral_msgs + colcon-built kernel.
``scripts/ros_live_tests.sh`` is the only runner (``just test-ros-live``, docker-build
workflow); ``tests/unit/test_ros_live_targets.py`` keeps this file in TARGETS.
"""

from __future__ import annotations

import json
import os
import pathlib
import re
import tempfile
import time
import uuid
from typing import Any

import numpy as np
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

# The kernel-spawn helpers are shared with the tests/sim/safety kernel suite —
# one definition of "start the real node and drive its lifecycle" (§1.13).
from tests.sim.safety._kernel_subprocess import (  # noqa: E402
    activate_kernel_node,
    isolated_domain_id,
    start_kernel,
    terminate_kernel,
)

# ── The grid ─────────────────────────────────────────────────────────────────
# 25 mm cells: the resolution the sim runs at, and the one the allowance was
# calibrated against (`allowance_m=0.0375` in every declared round's
# `safety.place_region_armed` line).
_RESOLUTION_M = 0.025
_GRID_N = 9  # 9^3 = 729 cells
_GRID_ORIGIN_M = -0.1125  # (min corner of voxel (0,0,0)) on every axis
# Cell (8, 4, 4): centred at (0.1, 0.0, 0.0), i.e. on the carriage's travel axis.
_OCC_IJK = (8, 4, 4)
_OCC_INDEX = _OCC_IJK[0] + _GRID_N * (_OCC_IJK[1] + _GRID_N * _OCC_IJK[2])
_OCC_CENTRE_X = _GRID_ORIGIN_M + (_OCC_IJK[0] + 0.5) * _RESOLUTION_M  # 0.1
_OCC_NEAR_FACE_X = _OCC_CENTRE_X - 0.5 * _RESOLUTION_M  # 0.0875

# ── The payload and the margins ──────────────────────────────────────────────
_PAYLOAD_RADIUS_M = 0.020
_ATTACHED_MARGIN_M = 0.050
# min(1.5 x voxel, 4 cm) — kPlaceApproachAllowanceVoxels / kMaxPlaceApproachAllowanceM
# in cpp/openral_safety_kernel/include/openral_safety_kernel/collision.hpp.
_ALLOWANCE_M = min(1.5 * _RESOLUTION_M, 0.04)
_REDUCED_MARGIN_M = _ATTACHED_MARGIN_M - _ALLOWANCE_M
# What the allowance was before ADR-0097's Second Amendment (2026-08-15):
# min(one voxel, 2.5 cm). Not a live constant — it is here so the band position
# below can be pinned strictly inside the headroom the amendment *added*, which
# is what makes a revert of either constant a red test rather than a silent
# loosening nobody notices.
_PRE_SECOND_AMENDMENT_ALLOWANCE_M = min(1.0 * _RESOLUTION_M, 0.025)


def _distance_at(q: float) -> float:
    """Payload-surface-to-voxel-surface distance for carriage position ``q``."""
    return _OCC_NEAR_FACE_X - q - _PAYLOAD_RADIUS_M


# Carriage positions for the three regimes. Each sits millimetres — not microns —
# from the nearest threshold, so the test pins semantics and not rounding.
_Q_CLEAR = 0.0000  # d = 0.0675 m — 17.5 mm outside the unreduced margin
_Q_BAND = 0.0475  # d = 0.0200 m — 7.5 mm inside the reduced margin, 5 mm inside the old one
_Q_PAST = 0.0625  # d = 0.0050 m — 7.5 mm past the allowance: refused either way

# ── Declaration identities (real shapes, not placeholders — §1.11) ───────────
_TARGET_ID = "sim:cabinet"
_OBJECT_ID = "sim:cup"
_RSKILL_ID = "openral/place-approach-band"


def _carriage_rig(axis_xyz: tuple[float, float, float] = (1.0, 0.0, 0.0)) -> RobotDescription:
    """A 1-DoF prismatic carriage that translates the payload along ``axis_xyz`` (+x).

    The link's capsule sits 300 mm behind the carriage frame so only the attached payload can
    reach the obstacle. Arm-vs-world voxel checking stays enabled and never gets an allowance
    — the kernel only reduces the margin inside ``check_attached_voxel_collision``.
    """
    return RobotDescription(
        name="place_allowance_carriage",
        embodiment_kind=EmbodimentKind.MANIPULATOR,
        joints=[
            JointSpec(
                name="carriage",
                joint_type=JointType.PRISMATIC,
                parent_link="base",
                child_link="carriage",
                axis_xyz=axis_xyz,
                origin_xyz=(0.0, 0.0, 0.0),
            )
        ],
        collision_geometry=[
            LinkCollisionGeometry(
                link_name="carriage",
                shape=CapsuleShape(radius_m=0.01, length_m=0.02),
                origin_xyz_rpy=(-0.30, 0.0, 0.0, 0.0, 0.0, 0.0),
            )
        ],
        allowed_collision_pairs=[],
        capabilities=RobotCapabilities(
            supported_control_modes=[ControlMode.JOINT_POSITION], embodiment_tags=["synthetic"]
        ),
        safety=SafetyEnvelope(),
    )


def _kernel_params(
    rig: RobotDescription | None = None, *, attached_margin_m: float = _ATTACHED_MARGIN_M
) -> dict[str, object]:
    params: dict[str, object] = {
        "n_dof": 1,
        "robot_name": "place_allowance_carriage",
        "joint_position_min": [-1.0],
        "joint_position_max": [1.0],
        "joint_velocity_max": [10.0],
        "joint_torque_max": [100.0],
        "world_voxel_enabled": True,
        "world_voxel_margin_m": 0.0,
        "world_voxel_deadline_ms": 2000.0,
        "world_voxel_max_cells": 4096,
        "attached_collision_enabled": True,
        "attached_collision_margin_m": attached_margin_m,
        "attached_collision_deadline_ms": 30000.0,
        "collision_joint_names": ["carriage"],
        "collision_state_deadline_ms": 30000.0,
        "collision_seed_dt_s": 0.0,
    }
    params.update(collision_params_from_description(rig or _carriage_rig()))
    return params


_COLLISION_LINE = re.compile(r"safety\.collision .*place_allowance_active=(\d)")


def test_place_allowance_band_accepts_only_with_a_live_declaration(
    publish_occupancy_grid, publish_carriage_joint_state, reset_kernel_estop
) -> None:
    """The allowance decides a verdict, and discloses itself when it does not.

    Four phases against one unchanging payload, grid and margin — the only thing
    that varies is the declaration and the commanded carriage position.
    """
    import rclpy
    from geometry_msgs.msg import Point, Pose, Quaternion, Vector3
    from openral_msgs.msg import (
        ActionChunk,
        AttachedCollisionObject,
        AttachedCollisionPrimitive,
        FailureTrigger,
        OccupancyVoxels,
        PlaceDeclaration,
        PlaceRegion,
        WorldStateStamped,
    )
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import JointState
    from std_msgs.msg import Empty
    from std_srvs.srv import Trigger

    # Sanity on the arithmetic the whole test rests on, asserted before a single
    # process is spawned so a mis-sized rig fails loudly rather than passing for
    # the wrong reason.
    assert _ALLOWANCE_M == pytest.approx(0.0375)  # noqa: SIM300 — reads left-to-right
    assert _distance_at(_Q_CLEAR) > _ATTACHED_MARGIN_M
    assert _REDUCED_MARGIN_M < _distance_at(_Q_BAND) <= _ATTACHED_MARGIN_M
    assert 0.0 < _distance_at(_Q_PAST) <= _REDUCED_MARGIN_M
    # The band chunk is inside the headroom the Second Amendment added, so a
    # revert of the cap (either constant) makes phase 3 refuse and this test red.
    assert _distance_at(_Q_BAND) <= _ATTACHED_MARGIN_M - _PRE_SECOND_AMENDMENT_ALLOWANCE_M, (
        "the band chunk would still pass under the pre-Second-Amendment cap"
    )

    node_name = f"safety_kernel_place_band_{uuid.uuid4().hex[:8]}"
    with tempfile.TemporaryDirectory() as td:
        log_path = pathlib.Path(td) / "kernel.log"
        proc = start_kernel(_kernel_params(), node_name, isolated_domain_id(), log_path=log_path)
        try:
            time.sleep(1.5)
            rclpy.init()
            try:
                helper = rclpy.create_node("place_band_helper")
                assert activate_kernel_node(node_name, helper), "kernel activation failed"

                safe: dict[str, ActionChunk] = {}
                failures: list[FailureTrigger] = []
                estops: list[Empty] = []
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
                chunk_pub = helper.create_publisher(ActionChunk, "/openral/candidate_action", 10)
                reliable_kl1 = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE)
                voxel_pub = helper.create_publisher(
                    OccupancyVoxels, "/openral/world_voxels", reliable_kl1
                )
                state_pub = helper.create_publisher(
                    WorldStateStamped, "/openral/world_state_fast", reliable_kl1
                )
                joint_pub = helper.create_publisher(JointState, "/joint_states", 10)
                reset_client = helper.create_client(Trigger, "/openral/estop_reset")

                executor = SingleThreadedExecutor()
                executor.add_node(helper)

                def spin(seconds: float) -> None:
                    end = time.time() + seconds
                    while time.time() < end:
                        executor.spin_once(timeout_sec=0.02)

                deadline = time.time() + 5.0
                while time.time() < deadline:
                    if chunk_pub.get_subscription_count() >= 1 and (
                        safe_sub.get_publisher_count() >= 1
                    ):
                        break
                    executor.spin_once(timeout_sec=0.05)

                # ── The world the kernel checks against ──────────────────────
                # Shared with the sibling test_safety_kernel_place_target_geometry.py
                # via tests/integration/conftest.py (byte-identical closures).
                def publish_grid() -> None:
                    publish_occupancy_grid(
                        voxel_pub,
                        helper,
                        spin,
                        grid_origin_m=_GRID_ORIGIN_M,
                        resolution_m=_RESOLUTION_M,
                        grid_n=_GRID_N,
                        occ_index=_OCC_INDEX,
                    )

                def publish_joint_state() -> None:
                    publish_carriage_joint_state(
                        joint_pub, helper, spin, joint_names=["carriage"], positions=[0.0]
                    )

                def publish_attachment(*, declared: bool) -> None:
                    """One carried payload; the declaration is the only variable.

                    ``attachment_revision`` never changes, so the payload model, its
                    attach-time occupancy baseline, and its (absent) support witness are
                    byte-identical across every phase — accept vs. refusal below differs
                    only in the declaration.
                    """
                    prim = AttachedCollisionPrimitive()
                    prim.shape_type = AttachedCollisionPrimitive.SHAPE_SPHERE
                    prim.shape_dimensions = [_PAYLOAD_RADIUS_M]
                    prim.pose_in_object = Pose(orientation=Quaternion(w=1.0))

                    obj = AttachedCollisionObject()
                    obj.object_id = _OBJECT_ID
                    obj.attach_link = "carriage"
                    obj.touch_links = ["carriage"]
                    obj.pose_in_link = Pose(orientation=Quaternion(w=1.0))
                    obj.primitives = [prim]
                    obj.confidence = 1.0
                    obj.evidence_kind = "sim_geom_distance"
                    obj.evidence_ref = "mujoco_body:sim:cup"
                    obj.stamp_ns = helper.get_clock().now().nanoseconds
                    obj.support_contact_valid = False

                    state = WorldStateStamped()
                    state.header.frame_id = "base"
                    state.header.stamp = helper.get_clock().now().to_msg()
                    state.stamp_ns = helper.get_clock().now().nanoseconds
                    state.attached_objects = [obj]
                    state.attachment_revision = 1
                    state.attachment_stamp_ns = helper.get_clock().now().nanoseconds
                    state.place_declaration_valid = declared
                    if declared:
                        region = PlaceRegion()
                        region.frame_id = "base"  # must match the grid's frame
                        region.pose = Pose(
                            position=Point(x=_OCC_CENTRE_X, y=0.0, z=0.0),
                            orientation=Quaternion(w=1.0),
                        )
                        region.half_extents = Vector3(x=0.06, y=0.06, z=0.06)
                        region.evidence_ref = "mujoco_body_subtree:cabinet"
                        region.stamp_ns = helper.get_clock().now().nanoseconds

                        declaration = PlaceDeclaration()
                        declaration.target_id = _TARGET_ID
                        declaration.object_id = _OBJECT_ID
                        declaration.rskill_id = _RSKILL_ID
                        declaration.trace_id = f"place-band-{uuid.uuid4().hex[:8]}"
                        declaration.timeout_s = 60.0
                        declaration.stamp_ns = helper.get_clock().now().nanoseconds
                        declaration.active = True
                        declaration.region_valid = True
                        declaration.region = region
                        state.place_declaration = declaration
                    state_pub.publish(state)
                    spin(0.4)

                def send(trace: str, q: float, *, expect_accept: bool) -> None:
                    """Publish one candidate chunk and wait for the kernel's verdict.

                    Waits on the outcome, not a fixed duration: a bare sleep makes "no estop
                    arrived" and "the estop has not arrived yet" the same observation, letting
                    a refusal assertion pass for the wrong reason on a loaded host.
                    """
                    seen_failures = len(failures)
                    chunk = ActionChunk()
                    chunk.control_mode = 0  # JOINT_POSITION
                    chunk.horizon = 1
                    chunk.n_dof = 1
                    chunk.flat = [q]
                    chunk.rskill_id = _RSKILL_ID
                    chunk.trace_id = trace
                    chunk_pub.publish(chunk)
                    deadline = time.time() + 10.0
                    while time.time() < deadline:
                        executor.spin_once(timeout_sec=0.02)
                        if expect_accept and trace in safe:
                            break
                        if not expect_accept and len(failures) > seen_failures and estops:
                            break
                    spin(0.4)  # settle: a late accept/estop must still be visible

                def reset_estop() -> None:
                    reset_kernel_estop(reset_client, executor, spin, estops)

                def new_disclosures(seen: int) -> list[int]:
                    """`place_allowance_active` flags of collision lines past ``seen``."""
                    flags = [
                        int(m.group(1))
                        for m in _COLLISION_LINE.finditer(
                            log_path.read_text(encoding="utf-8", errors="replace")
                        )
                    ]
                    return flags[seen:]

                publish_grid()
                publish_joint_state()

                # ── Phase 1: undeclared, outside the unreduced margin ────────
                publish_attachment(declared=False)
                send("clear-undeclared", _Q_CLEAR, expect_accept=True)
                assert "clear-undeclared" in safe, (
                    "a payload 67.5 mm clear of the obstacle must pass with no declaration"
                )
                assert not estops

                # ── Phase 2: undeclared, INSIDE the band → refused ───────────
                # d = 20 mm <= the 50 mm attached margin.
                publish_attachment(declared=False)
                send("band-undeclared", _Q_BAND, expect_accept=False)
                assert "band-undeclared" not in safe, (
                    "without a declaration the band is an ordinary margin violation"
                )
                assert estops, "the undeclared band chunk must fire /openral/estop"
                assert failures
                trigger = failures[-1]
                assert trigger.kind == FailureTrigger.KIND_COLLISION
                evidence = json.loads(trigger.evidence_json)
                assert evidence["collision_kind"] == "world"
                assert evidence["link_a"] == f"attached:{_OBJECT_ID}"
                assert evidence["link_b_or_object"] == f"voxel_{_OCC_INDEX}"
                assert evidence["min_distance_m"] == pytest.approx(_distance_at(_Q_BAND), abs=1e-9)
                undeclared_flags = new_disclosures(0)
                assert undeclared_flags == [0], (
                    f"an undeclared stop must not claim an allowance was active: {undeclared_flags}"
                )

                reset_estop()

                # ── Phase 3: THE FIRING — same chunk, now declared ───────────
                publish_attachment(declared=True)
                send("band-declared", _Q_BAND, expect_accept=True)
                assert "band-declared" in safe, (
                    "the live place declaration must reduce the margin by "
                    f"{_ALLOWANCE_M} m and let the band chunk through"
                )
                assert not estops, "an allowed approach must not fire the estop"
                assert new_disclosures(1) == [], "an accepted chunk reports no collision"

                # ── Phase 4: the hard stop behind the reduced margin ─────────
                # Deeper than the allowance: still refused, and the kernel
                # discloses that the margin it tripped had been reduced.
                publish_attachment(declared=True)
                send("past-allowance-declared", _Q_PAST, expect_accept=False)
                assert "past-allowance-declared" not in safe, (
                    "the allowance is bounded — deepening past it must still stop"
                )
                assert estops, "the past-allowance chunk must fire /openral/estop"
                trigger = failures[-1]
                assert trigger.kind == FailureTrigger.KIND_COLLISION
                evidence = json.loads(trigger.evidence_json)
                assert evidence["link_a"] == f"attached:{_OBJECT_ID}"
                assert evidence["min_distance_m"] == pytest.approx(
                    _distance_at(_Q_PAST), abs=1e-9
                ), "the reported distance stays the pair's TRUE surface distance"
                declared_flags = new_disclosures(1)
                assert declared_flags == [1], (
                    "a stop inside a declared region must disclose "
                    f"place_allowance_active=1, got {declared_flags}"
                )

                # The kernel's own disclosure line, quoted whole, is the
                # evidence this test exists to produce.
                log_text = log_path.read_text(encoding="utf-8", errors="replace")
                disclosure = [
                    line
                    for line in log_text.splitlines()
                    if "safety.collision" in line and "place_allowance_active=1" in line
                ]
                assert len(disclosure) == 1, disclosure
                assert f"place_target={_TARGET_ID}" in disclosure[0]
                assert "sweep_min_distance_m=" in disclosure[0]

                # And the arming line quotes the allowance the geometry used —
                # one definition, so the log can never disagree with the check.
                assert f"safety.place_region_armed target={_TARGET_ID}" in log_text, log_text
                assert f"allowance_m={_ALLOWANCE_M:g}" in log_text
            finally:
                rclpy.shutdown()
        except AssertionError as exc:
            # Every verdict this test asserts on has a matching line in the
            # kernel's own log; surfacing it turns a bare "not in safe" into a
            # diagnosis (which gate dropped the chunk, and why).
            raise AssertionError(
                f"{exc}\n\n--- safety_kernel_node log ---\n"
                f"{log_path.read_text(encoding='utf-8', errors='replace')}"
            ) from exc
        finally:
            terminate_kernel(proc)


# ── The support-contact witness at a vision pick (Isaac i42) ─────────────────
# 15 mm cells (i42's octomap). One support cell, (4, 4, 4), centred on the origin: the
# support's top face — what ``support_top_from_voxels`` measures — is z = 7.5 mm.
_W_RES = 0.015
_W_GRID_N = 9
_W_ORIGIN = -0.0675
_W_CELL = 4 + _W_GRID_N * (4 + _W_GRID_N * 4)
_W_SUPPORT_Z = 0.5 * _W_RES
#: The tight fit, i42's footprint; its lower face one cell above the support
#: (``target_region_from_mask``), measured with the carriage at q = 0.
_W_HALF = (0.06, 0.06, 0.03)
_W_CENTRE_Z = _W_SUPPORT_Z + _W_RES + _W_HALF[2]
#: Pressed 2 mm into the support cell (i42 stopped at -1.86 mm); and three cells above that.
_W_Q_REST = -(_W_RES + 0.002)
_W_Q_LIFT = _W_Q_REST + 3 * _W_RES


def test_the_vision_pick_support_witness_exempts_the_rest_and_dies_on_the_lift(
    publish_occupancy_grid, publish_carriage_joint_state, reset_kernel_estop
) -> None:
    """The region payload's witness, built by the producer's own ``region_attachment``.

    On the real cell's 0 mm attached margin (``deploy_e2e``): resting 2 mm into its support
    it is refused without the witness and accepted with it; the lift retires it in the
    kernel, and the very same witness (same object, support and stamp) does not re-arm.
    """
    import rclpy
    from openral_core import GraspDeclaration, PlaceRegion, Pose6D
    from openral_hal.vision_attachment_bridge import region_attachment
    from openral_msgs.msg import (
        ActionChunk,
        AttachedCollisionObject,
        AttachedCollisionPrimitive,
        FailureTrigger,
        OccupancyVoxels,
        WorldStateStamped,
    )
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import JointState
    from std_msgs.msg import Empty
    from std_srvs.srv import Trigger

    rig = _carriage_rig(axis_xyz=(0.0, 0.0, 1.0))
    node_name = f"safety_kernel_support_witness_{uuid.uuid4().hex[:8]}"
    with tempfile.TemporaryDirectory() as td:
        log_path = pathlib.Path(td) / "kernel.log"
        proc = start_kernel(
            _kernel_params(rig, attached_margin_m=0.0),
            node_name,
            isolated_domain_id(),
            log_path=log_path,
        )
        try:
            time.sleep(1.5)
            rclpy.init()
            try:
                helper = rclpy.create_node("support_witness_helper")
                assert activate_kernel_node(node_name, helper), "kernel activation failed"
                safe: dict[str, ActionChunk] = {}
                failures: list[FailureTrigger] = []
                estops: list[Empty] = []
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
                chunk_pub = helper.create_publisher(ActionChunk, "/openral/candidate_action", 10)
                reliable_kl1 = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE)
                voxel_pub = helper.create_publisher(
                    OccupancyVoxels, "/openral/world_voxels", reliable_kl1
                )
                state_pub = helper.create_publisher(
                    WorldStateStamped, "/openral/world_state_fast", reliable_kl1
                )
                joint_pub = helper.create_publisher(JointState, "/joint_states", 10)
                reset_client = helper.create_client(Trigger, "/openral/estop_reset")
                executor = SingleThreadedExecutor()
                executor.add_node(helper)

                def spin(seconds: float) -> None:
                    end = time.time() + seconds
                    while time.time() < end:
                        executor.spin_once(timeout_sec=0.02)

                deadline = time.time() + 5.0
                while time.time() < deadline and not (
                    chunk_pub.get_subscription_count() >= 1 and safe_sub.get_publisher_count() >= 1
                ):
                    executor.spin_once(timeout_sec=0.05)

                now_ns = helper.get_clock().now().nanoseconds
                region = PlaceRegion(
                    frame_id="base",
                    pose=Pose6D(
                        xyz=(0.0, 0.0, _W_CENTRE_Z), quat_xyzw=(0, 0, 0, 1), frame_id="base"
                    ),
                    half_extents=_W_HALF,
                    evidence_ref="segment_in_view:support-witness-band@0",
                    stamp_ns=now_ns,
                )
                declaration = GraspDeclaration(
                    target_id="approach:carriage:1",
                    contact_links=("carriage",),
                    rskill_id=_RSKILL_ID,
                    trace_id=uuid.uuid4().hex,
                    timeout_s=60.0,
                    stamp_ns=now_ns,
                    region=region,
                )
                payload = {
                    "attach_link": "carriage",
                    "touch_links": ("carriage",),
                    "t_link_from_region": np.eye(4),  # measured with the carriage at q = 0
                    "stamp_ns": now_ns,
                }
                bare = region_attachment(declaration, **payload)
                attested = region_attachment(declaration, support_z=_W_SUPPORT_Z, **payload)
                assert bare.support_contact is None and attested.support_contact is not None

                def publish_attachment(held: Any) -> None:
                    """One payload at one revision: only its witness differs between rows."""
                    obj = AttachedCollisionObject()
                    held.fill_idl(obj, primitive_factory=AttachedCollisionPrimitive)
                    state = WorldStateStamped()
                    state.header.frame_id = "base"
                    state.header.stamp = helper.get_clock().now().to_msg()
                    state.stamp_ns = helper.get_clock().now().nanoseconds
                    state.attached_objects = [obj]
                    state.attachment_revision = 1
                    state.attachment_stamp_ns = state.stamp_ns
                    state_pub.publish(state)
                    spin(0.4)

                def at(q: float) -> None:
                    publish_carriage_joint_state(
                        joint_pub, helper, spin, joint_names=["carriage"], positions=[q]
                    )

                def send(trace: str, q: float, *, expect_accept: bool) -> None:
                    seen = len(failures)
                    chunk = ActionChunk()
                    chunk.control_mode = 0  # JOINT_POSITION
                    chunk.horizon = 1
                    chunk.n_dof = 1
                    chunk.flat = [q]
                    chunk.rskill_id = _RSKILL_ID
                    chunk.trace_id = trace
                    chunk_pub.publish(chunk)
                    end = time.time() + 10.0
                    while time.time() < end:
                        executor.spin_once(timeout_sec=0.02)
                        if expect_accept and trace in safe:
                            break
                        if not expect_accept and len(failures) > seen and estops:
                            break
                    spin(0.4)
                    if expect_accept:
                        assert trace in safe and not estops, f"{trace} was refused"
                    else:
                        assert trace not in safe and estops, f"{trace} was accepted"
                        evidence = json.loads(failures[-1].evidence_json)
                        assert failures[-1].kind == FailureTrigger.KIND_COLLISION
                        assert evidence["link_a"] == f"attached:{declaration.target_id}"
                        assert evidence["link_b_or_object"] == f"voxel_{_W_CELL}"
                        assert evidence["min_distance_m"] == pytest.approx(-0.002, abs=1e-9)

                publish_occupancy_grid(
                    voxel_pub,
                    helper,
                    spin,
                    grid_origin_m=_W_ORIGIN,
                    resolution_m=_W_RES,
                    grid_n=_W_GRID_N,
                    occ_index=_W_CELL,
                )
                at(_W_Q_REST)

                # ── Resting on its measured support, no witness: the i42 stop ──
                publish_attachment(bare)
                send("rest-bare", _W_Q_REST, expect_accept=False)
                reset_kernel_estop(reset_client, executor, spin, estops)

                # ── The producer's witness: the same contact is the support ────
                publish_attachment(attested)
                send("rest-attested", _W_Q_REST, expect_accept=True)

                # ── Lifted three cells: the kernel retires it ─────────────────
                at(_W_Q_LIFT)
                send("lifted", _W_Q_LIFT, expect_accept=True)
                log = log_path.read_text(encoding="utf-8", errors="replace")
                assert "safety.support_witness_separated live=0x0 was=0x1" in log, log

                # ── Set down again under the same witness: dead stays dead ─────
                at(_W_Q_REST)
                publish_attachment(attested)
                send("rest-again", _W_Q_REST, expect_accept=False)
                log = log_path.read_text(encoding="utf-8", errors="replace")
                armed = f"safety.support_witness_armed object={declaration.target_id}"
                assert log.count(armed) == 1, "a republished witness re-armed"
                assert f"support=map_support_under:{declaration.target_id}" in log
            finally:
                rclpy.shutdown()
        except AssertionError as exc:
            raise AssertionError(
                f"{exc}\n\n--- safety_kernel_node log ---\n"
                f"{log_path.read_text(encoding='utf-8', errors='replace')}"
            ) from exc
        finally:
            terminate_kernel(proc)


# ── The cell-closed region payload at a vision pick (Isaac i50) ──────────────
# 15 mm cells on a 16^3 lattice (4096 = `world_voxel_max_cells`), min corner -0.12 m on every
# axis: cell centres on odd multiples of 7.5 mm. Layer k=7 (centre z = -7.5 mm) is the support,
# its top z = 0; k=8 (7.5 mm) the target's bottom layer under the region; the region's lower
# face is one cell above the support (`target_region_from_mask`), z = 15 mm.
_C_RES = 0.015
_C_GRID_N = 16
_C_ORIGIN = -0.12
_C_SUPPORT_Z = 0.0
#: i50's yaw, off-lattice centre, a flat footprint: the tight (held) fit at q = 0.
_C_YAW_DEG = -80.4
_C_TIGHT_HALF = (0.045, 0.03, 0.015)
_C_TIGHT_XYZ = (0.004, -0.003, _C_SUPPORT_Z + _C_RES + 0.015)
#: Measured 2 mm down on the target (the snapshot pose), commanded 19 mm below that (i50).
_C_Q_MEAS = -0.002
_C_Q_PRESS = _C_Q_MEAS - 0.019
#: Raised 14 mm into the shelf cell above the payload.
_C_Q_UP = _C_Q_MEAS + 0.014


def _c_index(i: int, j: int, k: int) -> int:
    return i + _C_GRID_N * (j + _C_GRID_N * k)


_C_SUPPORT = tuple(_c_index(i, j, 7) for i in range(_C_GRID_N) for j in range(_C_GRID_N))
#: Under the payload's centre: the target's own bottom layer, touching the region's lower face.
_C_BOTTOM = _c_index(8, 7, 8)
#: The target's own boundary cell, centre (37.5, 7.5, 22.5) mm: -3.86 mm into the tight fit at
#: the snapshot (not residue), -12.5 mm into its closure (residue) — i50's voxel_125244.
_C_BOUNDARY = _c_index(10, 8, 9)
#: A foreign cell (a shelf) over the payload, centre z = 67.5 mm: 9.5 mm clear of the closed
#: payload at the snapshot, outside the closure.
_C_FOREIGN = _c_index(8, 8, 12)


def test_a_cell_closed_region_payload_embeds_its_targets_boundary_cells(
    publish_occupancy_grid, publish_carriage_joint_state, reset_kernel_estop
) -> None:
    """Isaac i50 on the real kernel: the region payload is the box the kernel latched.

    After a clean region-payload ATTACH the kernel stopped on the target's own boundary
    cell: against the tight held fit it sat ~4 mm deep at the attach-time snapshot — not the
    half-cell embedded residue the kernel exempts — and the support witness's band, riding
    with a payload commanded 19 mm below the measured pose, dropped below it. Built by the
    producer's own ``region_attachment`` on a 1-DoF vertical carriage at the real cell's 0 mm
    attached margin, one attachment revision per row (a fresh snapshot at the measured q):

    * tight payload, pressed 19 mm: REFUSED on the boundary cell (the i50 stop);
    * closed payload (``_kernel_closure``, what ``GraspTargetLeg.kernel_region`` publishes
      and the bridge now attaches), same press: ACCEPTED — the boundary cell is residue;
    * the closed payload raised into a foreign cell outside the closure: REFUSED on it;
    * the closed payload whose witness patch is the *tight* footprint, same press: REFUSED
      on a support cell under the closure's corners — why the witness patch is the
      published footprint.
    """
    import rclpy
    from openral_core import GraspDeclaration, PlaceRegion, Pose6D
    from openral_core.geometry import yaw_to_quat_xyzw
    from openral_hal._grasp_target import VoxelLattice
    from openral_hal._grasp_target_leg import _kernel_closure
    from openral_hal.vision_attachment_bridge import region_attachment
    from openral_msgs.msg import (
        ActionChunk,
        AttachedCollisionObject,
        AttachedCollisionPrimitive,
        FailureTrigger,
        OccupancyVoxels,
        WorldStateStamped,
    )
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import JointState
    from std_msgs.msg import Empty
    from std_srvs.srv import Trigger

    rig = _carriage_rig(axis_xyz=(0.0, 0.0, 1.0))
    node_name = f"safety_kernel_closed_payload_{uuid.uuid4().hex[:8]}"
    with tempfile.TemporaryDirectory() as td:
        log_path = pathlib.Path(td) / "kernel.log"
        proc = start_kernel(
            _kernel_params(rig, attached_margin_m=0.0),
            node_name,
            isolated_domain_id(),
            log_path=log_path,
        )
        try:
            time.sleep(1.5)
            rclpy.init()
            try:
                helper = rclpy.create_node("closed_payload_helper")
                assert activate_kernel_node(node_name, helper), "kernel activation failed"
                safe: dict[str, ActionChunk] = {}
                failures: list[FailureTrigger] = []
                estops: list[Empty] = []
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
                chunk_pub = helper.create_publisher(ActionChunk, "/openral/candidate_action", 10)
                reliable_kl1 = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE)
                voxel_pub = helper.create_publisher(
                    OccupancyVoxels, "/openral/world_voxels", reliable_kl1
                )
                state_pub = helper.create_publisher(
                    WorldStateStamped, "/openral/world_state_fast", reliable_kl1
                )
                joint_pub = helper.create_publisher(JointState, "/joint_states", 10)
                reset_client = helper.create_client(Trigger, "/openral/estop_reset")
                executor = SingleThreadedExecutor()
                executor.add_node(helper)

                def spin(seconds: float) -> None:
                    end = time.time() + seconds
                    while time.time() < end:
                        executor.spin_once(timeout_sec=0.02)

                deadline = time.time() + 5.0
                while time.time() < deadline and not (
                    chunk_pub.get_subscription_count() >= 1 and safe_sub.get_publisher_count() >= 1
                ):
                    executor.spin_once(timeout_sec=0.05)

                lattice = VoxelLattice(
                    "base",
                    (_C_ORIGIN,) * 3,
                    (0.0, 0.0, 0.0, 1.0),
                    _C_RES,
                    (_C_GRID_N,) * 3,
                    np.zeros(_C_GRID_N**3, dtype=np.uint8),
                )
                revision = 0

                def payload(closed: bool) -> Any:
                    """The producer's region payload, freshly stamped (a new witness key)."""
                    now_ns = helper.get_clock().now().nanoseconds
                    tight = PlaceRegion(
                        frame_id="base",
                        pose=Pose6D(
                            xyz=_C_TIGHT_XYZ,
                            quat_xyzw=yaw_to_quat_xyzw(np.deg2rad(_C_YAW_DEG)),
                            frame_id="base",
                        ),
                        half_extents=_C_TIGHT_HALF,
                        evidence_ref="segment_in_view:closed-payload-band@0",
                        stamp_ns=now_ns,
                    )
                    region, note = _kernel_closure(tight, lattice) if closed else (tight, "")
                    assert note == ""
                    declaration = GraspDeclaration(
                        target_id="approach:carriage:1",
                        contact_links=("carriage",),
                        rskill_id=_RSKILL_ID,
                        trace_id=uuid.uuid4().hex,
                        timeout_s=60.0,
                        stamp_ns=now_ns,
                        region=region,
                    )
                    held = region_attachment(
                        declaration,
                        attach_link="carriage",
                        touch_links=("carriage",),
                        t_link_from_region=np.eye(4),  # measured with the carriage at q = 0
                        stamp_ns=now_ns,
                        support_z=_C_SUPPORT_Z,
                        extrinsic_error_m=0.015,
                    )
                    assert held.support_contact is not None
                    return held

                def publish_attachment(held: Any) -> None:
                    """A new attachment revision: the kernel re-snapshots at the measured q."""
                    nonlocal revision
                    revision += 1
                    obj = AttachedCollisionObject()
                    held.fill_idl(obj, primitive_factory=AttachedCollisionPrimitive)
                    state = WorldStateStamped()
                    state.header.frame_id = "base"
                    state.header.stamp = helper.get_clock().now().to_msg()
                    state.stamp_ns = helper.get_clock().now().nanoseconds
                    state.attached_objects = [obj]
                    state.attachment_revision = revision
                    state.attachment_stamp_ns = state.stamp_ns
                    state_pub.publish(state)
                    spin(0.4)

                def send(trace: str, q: float, *, refused_on: tuple[int, ...] = ()) -> None:
                    """One candidate; accepted, or refused on one of the ``refused_on`` cells."""
                    seen = len(failures)
                    chunk = ActionChunk()
                    chunk.control_mode = 0  # JOINT_POSITION
                    chunk.horizon = 1
                    chunk.n_dof = 1
                    chunk.flat = [q]
                    chunk.rskill_id = _RSKILL_ID
                    chunk.trace_id = trace
                    chunk_pub.publish(chunk)
                    end = time.time() + 10.0
                    while time.time() < end:
                        executor.spin_once(timeout_sec=0.02)
                        if not refused_on and trace in safe:
                            break
                        if refused_on and len(failures) > seen and estops:
                            break
                    spin(0.4)
                    if not refused_on:
                        assert trace in safe and not estops, f"{trace} was refused"
                        return
                    assert trace not in safe and estops, f"{trace} was accepted"
                    assert failures[-1].kind == FailureTrigger.KIND_COLLISION
                    evidence = json.loads(failures[-1].evidence_json)
                    assert evidence["link_a"] == "attached:approach:carriage:1"
                    assert evidence["link_b_or_object"] in {f"voxel_{i}" for i in refused_on}, (
                        f"{trace}: stopped on {evidence['link_b_or_object']}"
                    )

                publish_occupancy_grid(
                    voxel_pub,
                    helper,
                    spin,
                    grid_origin_m=_C_ORIGIN,
                    resolution_m=_C_RES,
                    grid_n=_C_GRID_N,
                    occ_index=(*_C_SUPPORT, _C_BOTTOM, _C_BOUNDARY, _C_FOREIGN),
                )
                publish_carriage_joint_state(
                    joint_pub, helper, spin, joint_names=["carriage"], positions=[_C_Q_MEAS]
                )

                # ── The tight held fit as the payload: i50's stop ─────────────
                publish_attachment(payload(closed=False))
                send("tight-pressed", _C_Q_PRESS, refused_on=(_C_BOUNDARY,))
                reset_kernel_estop(reset_client, executor, spin, estops)

                # ── Closed over the cells it touches: the boundary is residue ──
                closed = payload(closed=True)
                publish_attachment(closed)
                send("closed-pressed", _C_Q_PRESS)

                # ── Outside the closure, a foreign cell still stops it ─────────
                send("closed-raised", _C_Q_UP, refused_on=(_C_FOREIGN,))
                reset_kernel_estop(reset_client, executor, spin, estops)

                # ── The witness patch must be the published footprint ──────────
                closed = payload(closed=True)
                witness = closed.support_contact
                tight_patch = float(np.linalg.norm(_C_TIGHT_HALF))
                assert witness.patch_radius_m > tight_patch
                publish_attachment(
                    closed.model_copy(
                        update={
                            "support_contact": witness.model_copy(
                                update={"patch_radius_m": tight_patch}
                            )
                        }
                    )
                )
                send("closed-pressed-tight-patch", _C_Q_PRESS, refused_on=_C_SUPPORT)
                log = log_path.read_text(encoding="utf-8", errors="replace")
                assert log.count("safety.support_witness_armed object=approach:carriage:1") == 3
            finally:
                rclpy.shutdown()
        except AssertionError as exc:
            raise AssertionError(
                f"{exc}\n\n--- safety_kernel_node log ---\n"
                f"{log_path.read_text(encoding='utf-8', errors='replace')}"
            ) from exc
        finally:
            terminate_kernel(proc)
