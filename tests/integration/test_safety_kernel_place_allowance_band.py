"""The declaration-scoped place-approach allowance, fired deterministically.

ADR-0097's 2026-08-14 amendment (as recalibrated by its Second Amendment,
2026-08-15) reduces a declared payload's world-collision margin by
``min(1.5 x voxel, 4 cm)`` for occupancy cells whose centre lies inside the
producer-measured region of the declared place target. Every validation round so
far armed the allowance correctly (``safety.place_region_armed ...
allowance_m=0.0375``) but almost never *fired* it: XR-1's stochastic
trajectories only rarely put the payload in the narrow band between the reduced
and the unreduced margin, so the allowance's effect on an actual verdict rested
on counterfactual arithmetic rather than on an observed accept.

This test removes the policy — and with it the stochasticity — and drives the
band directly. Everything else is real: the real ``safety_kernel_node`` lifecycle
node, the real ``openral_msgs`` IDL, a real dense occupancy grid on
``/openral/world_voxels``, a real attached payload plus a real
``PlaceDeclaration``/``PlaceRegion`` on ``/openral/world_state_fast``, and real
``ActionChunk`` candidates on ``/openral/candidate_action``. No mocks, no
simulator, no GPU (CLAUDE.md §1.11).

The rig is a one-DoF prismatic carriage so a chunk's single joint value *is* the
payload's x position, which makes the payload-to-obstacle distance exact and the
band a matter of arithmetic rather than luck:

    payload sphere centre  = (q, 0, 0),         radius 20 mm
    occupied voxel centre  = (0.1, 0, 0),   half-edge 12.5 mm
    surface distance d(q)  = 0.0875 - q - 0.020 = 0.0675 - q

At the sim grid's 25 mm resolution the allowance is ``min(1.5 x 0.025, 0.04) =
0.0375 m``, so against the 50 mm attached margin the band is
``0.0125 m < d <= 0.05 m``. Three chunks pin the three regimes:

===============  ======  ==================  =====================================
q (m)            d (m)   undeclared          declared
===============  ======  ==================  =====================================
0.0              0.0675  accepted            accepted (allowance changes nothing)
0.0475           0.020   REFUSED             ACCEPTED  <- the allowance's verdict
0.0625           0.005   REFUSED             REFUSED, ``place_allowance_active=1``
===============  ======  ==================  =====================================

The last row is the disclosure the merge package quotes: the hard stop behind the
reduced margin is untouched, and when it fires the kernel says so. The middle row
is the thing that had never been observed.

The middle row is also deliberately sized to break if the allowance is *reduced*,
not only if it is deleted: 20 mm sits below the 25 mm the pre-Second-Amendment
cap (``min(one voxel, 2.5 cm)``) would have left, so reverting either
``kPlaceApproachAllowanceVoxels`` (1.5 -> 1.0) or ``kMaxPlaceApproachAllowanceM``
(0.04 -> 0.025) turns that accept back into a refusal and this test red.

Gates: ``OPENRAL_TEST_ROS_LIVE=1`` + ROS_DISTRO + rclpy + openral_msgs + the
colcon-built kernel, on a sourced workspace. ``scripts/ros_live_tests.sh`` is the
only runner (``just test-ros-live``, and the docker-build workflow), and
``tests/unit/test_ros_live_targets.py`` keeps this file in its TARGETS list.
"""

from __future__ import annotations

import json
import os
import pathlib
import re
import tempfile
import time
import uuid

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


def _carriage_rig() -> RobotDescription:
    """A 1-DoF prismatic carriage that translates the payload along +x.

    The link's own capsule sits 300 mm *behind* the carriage frame so the arm
    never approaches the occupied cell itself: the only geometry that can reach
    the obstacle is the attached payload, which is the geometry the allowance
    applies to. Arm-vs-world voxel checking stays enabled throughout and is
    never granted an allowance — the kernel only reduces the margin inside
    ``check_attached_voxel_collision``.
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
                axis_xyz=(1.0, 0.0, 0.0),
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


def _kernel_params() -> dict[str, object]:
    params: dict[str, object] = {
        "n_dof": 1,
        "robot_name": "place_allowance_carriage",
        "joint_position_min": [-1.0],
        "joint_position_max": [1.0],
        "joint_velocity_max": [10.0],
        "joint_torque_max": [100.0],
        "world_voxel_enabled": True,
        "world_voxel_margin_m": 0.0,
        "world_voxel_deadline_ms": 30000.0,
        "world_voxel_max_cells": 4096,
        "attached_collision_enabled": True,
        "attached_collision_margin_m": _ATTACHED_MARGIN_M,
        "attached_collision_deadline_ms": 30000.0,
        "collision_joint_names": ["carriage"],
        "collision_state_deadline_ms": 30000.0,
        "collision_seed_dt_s": 0.0,
    }
    params.update(collision_params_from_description(_carriage_rig()))
    return params


_COLLISION_LINE = re.compile(r"safety\.collision .*place_allowance_active=(\d)")


def test_place_allowance_band_accepts_only_with_a_live_declaration() -> None:
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
                def publish_grid() -> None:
                    grid = OccupancyVoxels()
                    grid.header.frame_id = "base"
                    grid.header.stamp = helper.get_clock().now().to_msg()
                    grid.origin = Point(x=_GRID_ORIGIN_M, y=_GRID_ORIGIN_M, z=_GRID_ORIGIN_M)
                    grid.resolution = _RESOLUTION_M
                    grid.size_x = _GRID_N
                    grid.size_y = _GRID_N
                    grid.size_z = _GRID_N
                    occupancy = [0] * (_GRID_N**3)
                    occupancy[_OCC_INDEX] = 1
                    grid.occupancy = occupancy
                    voxel_pub.publish(grid)
                    spin(0.4)

                def publish_joint_state() -> None:
                    js = JointState()
                    js.header.stamp = helper.get_clock().now().to_msg()
                    js.name = ["carriage"]
                    js.position = [0.0]
                    joint_pub.publish(js)
                    spin(0.2)

                def publish_attachment(*, declared: bool) -> None:
                    """One carried payload; the declaration is the only variable.

                    ``attachment_revision`` never changes, so the payload model,
                    its attach-time occupancy baseline and its (absent) support
                    witness are byte-identical across every phase — the accept
                    and the refusal below differ in the declaration and nothing
                    else.
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

                    Waits on the *outcome*, not on a fixed duration: a bare sleep
                    makes "no estop arrived" and "the estop has not arrived yet"
                    the same observation, which is how a refusal assertion passes
                    for the wrong reason on a loaded host.
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
                    assert reset_client.wait_for_service(timeout_sec=5.0)
                    spin(0.3)  # clear the reset cooldown
                    future = reset_client.call_async(Trigger.Request())
                    end = time.time() + 5.0
                    while time.time() < end and not future.done():
                        executor.spin_once(timeout_sec=0.02)
                    assert future.done() and future.result().success, "estop reset refused"
                    estops.clear()
                    spin(0.3)

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
