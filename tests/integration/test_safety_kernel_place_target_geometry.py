"""The declared target's own geometry, fired deterministically (ADR-0098).

The sibling ``test_safety_kernel_place_allowance_band.py`` drives ADR-0097's
blanket allowance on this same rig. This file drives the half that replaces the
guess: when the declaration also ships the target's **own** primitives, a cell is
adjudicated against the modelled receptacle instead of against the 25 mm cube it
was quantised into, and the gate for that pair moves from the payload's standoff
margin to the surface itself.

Why a live test rather than gtests alone: #188's graded-velocity band shipped as
dead code that three unit tests failed to catch, because the fixture and the
logic were written from the same wrong picture. The same near-miss happened here
— the first implementation kept the standoff margin against the declared body,
which the collision gtests could not see because they run at margin 0, where
gating at ``margin`` and gating at ``0`` are algebraically identical. This runs
the real ``safety_kernel_node`` binary at the **deployed** margin, where they
are not.

The rig is the sibling's, unchanged, so the two files differ in one field:

    payload sphere centre  = (q, 0, 0),          radius 20 mm
    occupied voxel centre  = (0.1, 0, 0),    half-edge 12.5 mm
    modelled shelf face    = x = 0.100                       <- ADR-0098

so the cube over-states the shelf's surface by exactly 12.5 mm — half a cell,
which is what marking the cell that *contains* a ray endpoint costs on average.
Against the 50 mm attached margin and the 37.5 mm blanket allowance:

===============  ==========  ==========  ==================  =====================
q (m)            d_cell (m)  d_body (m)  declared, box only  declared + geometry
===============  ==========  ==========  ==================  =====================
0.0625           0.0050      0.0175      REFUSED             ACCEPTED
0.0825           −0.0150     −0.0025     refused (advisory)  REFUSED, advisory
0.0950           −0.0275     −0.0150     REFUSED, latched    REFUSED, latched
===============  ==========  ==========  ==================  =====================

Row 1 is the decision: the blanket allowance refuses a payload with 17.5 mm of
**measured** clearance from the receptacle, because it is judging a cube whose
face is not where the shelf is. Row 3 is the bound: past the advisory band the
stop is the latched one it always was, and the evidence names ``place:<target>``
rather than ``voxel_<n>`` — quoting the body's distance under the cell's identity
is the defect class #187 landed to stop.

Gates: ``OPENRAL_TEST_ROS_LIVE=1`` + ROS_DISTRO + rclpy + openral_msgs + the
colcon-built kernel, on a sourced workspace. ``scripts/ros_live_tests.sh`` is the
only runner (``just test-ros-live``, and the docker-build workflow), and
``tests/unit/test_ros_live_targets.py`` keeps this file in its TARGETS list.

CLAUDE.md §1.11 — real kernel binary, real IDL, real DDS, no mocks.
"""

from __future__ import annotations

import json
import os
import pathlib
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

from tests.integration.test_safety_kernel_place_allowance_band import (  # noqa: E402
    _ALLOWANCE_M,
    _ATTACHED_MARGIN_M,
    _GRID_N,
    _GRID_ORIGIN_M,
    _OBJECT_ID,
    _OCC_CENTRE_X,
    _OCC_INDEX,
    _PAYLOAD_RADIUS_M,
    _RESOLUTION_M,
    _TARGET_ID,
    _distance_at,
    _kernel_params,
)

# The kernel-spawn helpers are shared with the tests/sim/safety kernel suite —
# one definition of "start the real node and drive its lifecycle" (§1.13).
from tests.sim.safety._kernel_subprocess import (  # noqa: E402
    activate_kernel_node,
    isolated_domain_id,
    start_kernel,
    terminate_kernel,
)

_RSKILL_ID = "openral/place-target-geometry"

#: The declared shelf's near (−x) face, 12.5 mm behind the cube's. Half a cell
#: is the average cost of marking the cell that *contains* a surface, so this is
#: the ordinary quantisation error rather than a worst case chosen to flatter
#: the mechanism.
_SHELF_FACE_X = 0.100
#: The modelled shelf as the producer would ship it: one box, posed in the
#: region's own frame (the grid's frame), 120 mm on a side behind that face.
_SHELF_HALF_M = 0.06
_SHELF_CENTRE_X = _SHELF_FACE_X + _SHELF_HALF_M

#: min(0.2 × voxel, 5 mm) — kPlaceAdvisoryDepthVoxels / kMaxPlaceAdvisoryDepthM.
_ADVISORY_DEPTH_M = min(0.2 * _RESOLUTION_M, 0.005)


def _body_distance_at(q: float) -> float:
    """Payload-surface-to-**modelled-shelf**-surface distance at carriage ``q``."""
    return _SHELF_FACE_X - q - _PAYLOAD_RADIUS_M


# Three carriage positions, each millimetres from the threshold it pins.
_Q_MEASURED_CLEAR = 0.0625  # d_body = +17.5 mm; d_cell = +5 mm (the box refuses)
_Q_ARRIVED = 0.0825  # d_body = −2.5 mm: inside the advisory band
_Q_DEEP = 0.0950  # d_body = −15 mm: past the band, latched


def test_the_declared_targets_geometry_decides_the_verdict_and_names_the_body() -> None:
    """Same payload, same grid, same margin, same declaration — geometry or not.

    Three phases, and the first is the decision: a chunk the blanket allowance
    refuses is accepted once the kernel is told where the shelf actually is.
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

    # The arithmetic the whole test rests on, asserted before a process is
    # spawned so a mis-sized rig fails loudly rather than passing for the wrong
    # reason. Each line is a claim the docstring's table makes.
    assert _body_distance_at(_Q_MEASURED_CLEAR) == pytest.approx(0.0175)
    assert _distance_at(_Q_MEASURED_CLEAR) <= _ATTACHED_MARGIN_M - _ALLOWANCE_M, (
        "phase 1 is only a decision if the blanket allowance refuses it"
    )
    assert _body_distance_at(_Q_MEASURED_CLEAR) > 0.0, "...and the modelled shelf does not"
    # The substitution cap: the modelled distance is only used while it is
    # within the blanket allowance of the cell's own. 12.5 mm << 37.5 mm.
    for q in (_Q_MEASURED_CLEAR, _Q_ARRIVED, _Q_DEEP):
        assert _body_distance_at(q) <= _distance_at(q) + _ALLOWANCE_M, (
            f"q={q}: the shelf would fall outside the substitution cap"
        )
    assert -_ADVISORY_DEPTH_M < _body_distance_at(_Q_ARRIVED) <= 0.0, (
        "phase 2 must land inside the advisory band"
    )
    assert _body_distance_at(_Q_DEEP) < -_ADVISORY_DEPTH_M, "phase 3 must land past it"

    node_name = f"safety_kernel_place_geom_{uuid.uuid4().hex[:8]}"
    with tempfile.TemporaryDirectory() as td:
        log_path = pathlib.Path(td) / "kernel.log"
        proc = start_kernel(_kernel_params(), node_name, isolated_domain_id(), log_path=log_path)
        try:
            time.sleep(1.5)
            rclpy.init()
            try:
                helper = rclpy.create_node("place_geometry_helper")
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

                def publish_grid() -> None:
                    grid = OccupancyVoxels()
                    grid.header.frame_id = "base"
                    grid.header.stamp = helper.get_clock().now().to_msg()
                    grid.origin = Point(x=_GRID_ORIGIN_M, y=_GRID_ORIGIN_M, z=_GRID_ORIGIN_M)
                    # `OccupancyVoxels` is an oriented grid and its unset
                    # orientation is the all-zero quaternion, which every
                    # consumer refuses rather than reading as identity.
                    grid.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
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

                def publish_attachment(*, with_geometry: bool) -> None:
                    """One carried payload, one live declaration, one variable.

                    ``attachment_revision`` never changes, so the payload model,
                    its attach-time occupancy baseline and its (absent) support
                    witness are byte-identical across every phase. The region box
                    is identical too — the ONLY difference between the two arms
                    of phase 1 is whether the declaration carries the shelf.
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

                    region = PlaceRegion()
                    region.frame_id = "base"  # must match the grid's frame
                    region.pose = Pose(
                        position=Point(x=_OCC_CENTRE_X, y=0.0, z=0.0),
                        orientation=Quaternion(w=1.0),
                    )
                    region.half_extents = Vector3(x=0.06, y=0.06, z=0.06)
                    region.evidence_ref = "mujoco_body_subtree:cabinet"
                    region.stamp_ns = helper.get_clock().now().nanoseconds
                    if with_geometry:
                        shelf = AttachedCollisionPrimitive()
                        shelf.shape_type = AttachedCollisionPrimitive.SHAPE_BOX
                        shelf.shape_dimensions = [_SHELF_HALF_M] * 3
                        # Posed in the region's own frame, not relative to its box.
                        shelf.pose_in_object = Pose(
                            position=Point(x=_SHELF_CENTRE_X, y=0.0, z=0.0),
                            orientation=Quaternion(w=1.0),
                        )
                        region.geometry = [shelf]

                    declaration = PlaceDeclaration()
                    declaration.target_id = _TARGET_ID
                    declaration.object_id = _OBJECT_ID
                    declaration.rskill_id = _RSKILL_ID
                    declaration.trace_id = f"place-geom-{uuid.uuid4().hex[:8]}"
                    declaration.timeout_s = 60.0
                    declaration.stamp_ns = helper.get_clock().now().nanoseconds
                    declaration.active = True
                    declaration.region_valid = True
                    declaration.region = region

                    state = WorldStateStamped()
                    state.header.frame_id = "base"
                    state.header.stamp = helper.get_clock().now().to_msg()
                    state.stamp_ns = helper.get_clock().now().nanoseconds
                    state.attached_objects = [obj]
                    state.attachment_revision = 1
                    state.attachment_stamp_ns = helper.get_clock().now().nanoseconds
                    state.place_declaration_valid = True
                    state.place_declaration = declaration
                    state_pub.publish(state)
                    spin(0.4)

                def send(trace: str, q: float, *, expect_accept: bool) -> None:
                    """Publish one candidate chunk and wait for the kernel's verdict.

                    Waits on the *outcome*, not on a fixed duration: a bare sleep
                    makes "no refusal arrived" and "the refusal has not arrived
                    yet" the same observation.
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
                        # An advisory refusal publishes a FailureTrigger and NO
                        # estop, so waiting on both would hang on phase 2.
                        if not expect_accept and len(failures) > seen_failures:
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

                publish_grid()
                publish_joint_state()

                # ── Phase 1a: declared, box only — the shipped behaviour ─────
                # 5 mm of cube clearance, 7.5 mm inside the reduced margin.
                publish_attachment(with_geometry=False)
                send("box-only", _Q_MEASURED_CLEAR, expect_accept=False)
                assert "box-only" not in safe, (
                    "the blanket allowance judges the 25 mm cube, and the cube says stop"
                )
                assert estops, "and it stops by latching"
                box_trigger = failures[-1]
                box_evidence = json.loads(box_trigger.evidence_json)
                assert box_evidence["link_b_or_object"] == f"voxel_{_OCC_INDEX}"
                assert box_evidence["min_distance_m"] == pytest.approx(
                    _distance_at(_Q_MEASURED_CLEAR), abs=1e-9
                ), "the box-only stop quotes the CELL's distance"

                reset_estop()

                # ── Phase 1b: THE DECISION — same chunk, shelf shipped ───────
                publish_attachment(with_geometry=True)
                send("with-geometry", _Q_MEASURED_CLEAR, expect_accept=True)
                assert "with-geometry" in safe, (
                    "17.5 mm of MEASURED clearance from the declared shelf is not a collision; "
                    "the region box could not tell, and that is what ADR-0098 fixes"
                )
                assert not estops, "an accepted approach must not fire the estop"

                # ── Phase 2: arrival — refused at the surface, not latched ───
                publish_attachment(with_geometry=True)
                send("arrived", _Q_ARRIVED, expect_accept=False)
                assert "arrived" not in safe, "the pair still TRIPS at the surface"
                assert not estops, (
                    "2.5 mm into the declared shelf is inside the advisory band: refused, "
                    "not latched"
                )
                arrived = json.loads(failures[-1].evidence_json)
                assert arrived["link_b_or_object"] == f"place:{_TARGET_ID}#{_OCC_INDEX}", (
                    "a pair adjudicated against the declared body must be reported as that body"
                )
                assert arrived["min_distance_m"] == pytest.approx(
                    _body_distance_at(_Q_ARRIVED), abs=1e-9
                ), "and must quote the distance to the BODY, not to the cell"

                # ── Phase 3: past the band, the latch is unchanged ───────────
                publish_attachment(with_geometry=True)
                send("deep", _Q_DEEP, expect_accept=False)
                assert "deep" not in safe
                assert estops, "15 mm into the declared shelf is the latched stop it always was"
                deep = json.loads(failures[-1].evidence_json)
                assert deep["kind"] == "collision"
                assert deep["link_a"] == f"attached:{_OBJECT_ID}"
                assert deep["link_b_or_object"] == f"place:{_TARGET_ID}#{_OCC_INDEX}"
                assert deep["min_distance_m"] == pytest.approx(_body_distance_at(_Q_DEEP), abs=1e-9)

                log_text = log_path.read_text(encoding="utf-8", errors="replace")
                # The arming line says the geometry armed with the region — one
                # definition, so the log cannot disagree with what was checked.
                assert f"safety.place_region_armed target={_TARGET_ID}" in log_text, log_text
                assert "geometry=1" in log_text, log_text
                # And no adjudicated stop was ever reported under a cell's identity.
                adjudicated = [
                    line
                    for line in log_text.splitlines()
                    if "safety.collision" in line and f"b=place:{_TARGET_ID}" in line
                ]
                assert len(adjudicated) == 2, adjudicated
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
