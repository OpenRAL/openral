"""The grasp-target exemption, end to end, on the real OpenArm collision model.

The real OpenArm cell runs the kernel's world-voxel check at a 20 mm margin on 20 mm cells,
and the finger link (``openarm_<side>_finger_pair``, one hull swept over the stroke) contains
the grasp target during a grasp, so ``check_voxel_collision`` stops on the target's own cells
before any attachment exists. The exemption (``docs/reference/real-pick-place-design.md`` §2.1;
``cpp/openral_safety_kernel/README.md`` "Grasp-target exemption") lets a live, producer-measured
``GraspDeclaration`` exempt exactly the (declared contact link, cell centred in the region)
pairs. This test proves that on the real ``safety_kernel_node`` binary, fed the parameters the
deploy launch would hand it on the real cell, row by row:

=====================================================  ========================================
row                                                    verdict
=====================================================  ========================================
feature OFF, declaration live                          REFUSED (default off, end to end)
undeclared, feature on                                 REFUSED, ``KIND_COLLISION``, left finger
declared, left finger in region                        ACCEPTED, ``/diagnostics`` ``live:``
declared, ``openarm_left_link7`` also in region cells  REFUSED, link7, exemption disclosed
declared for the RIGHT gripper only                    REFUSED, left finger
declared, expired (``timeout_s`` passed)               REFUSED, ``/diagnostics`` ``expired:``
declared, region in another frame                      REFUSED, ``reason=frame_mismatch``
declared, support-plane cells half a voxel below       REFUSED, left finger on a plane cell,
the region's lower face, in the finger's margin        ``grasp_exemption_active=1``
declared, fit tight to a surface inside the target's   REFUSED, left finger on a target cell,
boundary cells (their centres outside the fit)         ``grasp_exemption_active=1``
the same fit cell-closed (``cell_closed_region``,      ACCEPTED; the plane under it still
what the producer leg publishes)                       outside (bottom kept)
=====================================================  ========================================

Real throughout (CLAUDE.md §1.11): the real ``robots/openarm/robot.yaml``, the kernel parameters
built by the same builders ``deploy_e2e.launch.py`` calls (``compute_intersection`` →
``kernel_params_from_envelope`` + ``collision_params_from_description`` — the launch keeps the
manifest model because the bimanual MJCF lowers to no primitive geometry — the real-cell
``REAL_WORLD_VOXEL_MARGIN_M`` and 20 mm resolution, attached checking off as on the real cell
without the vision leg, and ``grasp_contact_links`` = the manifest's ``role: gripper`` child
links), a real dense ``OccupancyVoxels`` grid, a real ``GraspDeclaration`` built through
``openral_core.GraspDeclaration.fill_idl`` on ``/openral/world_state_fast``, and real
``ActionChunk`` candidates. Verdicts are read where the place band tests read them:
``/openral/safe_action`` (accept), ``/openral/failure/safety`` + ``/openral/estop`` (refuse).

How the configuration was found
-------------------------------
The world is built around the hand rather than the hand searched into a world (the simpler
route; every number below is pinned, and the kernel is the oracle that checks them). At the
all-zero configuration both arms hang straight down; the kernel's own FK (``origin · joint
motion``, parent before child, over ``collision_params_from_description``) puts the left finger
pair's tight hull at z ∈ [-0.606, -0.456] m and ``openarm_left_link7``'s at z ∈ [-0.547,
-0.418] m in ``openarm_base``. Exact hull-to-cell gaps (GJK over the manifest's
``hull_vertices_m``, the same hull the kernel's stage-2 narrow phase uses) for the cells this
grid occupies, at that configuration:

====================================  =================  ===============  ===============
cells (layer centre z)                left finger pair   left link7       every other link
====================================  =================  ===============  ===============
target, z = -0.63 / -0.61             0.0 (inside)       ≥ 53.4 mm        > 80 mm
tall-target extra, z = -0.59 / -0.57  0.0 (inside)       13.5 mm (trips)  > 80 mm
support plane, z = -0.65              33.5 mm            > 80 mm          > 80 mm
support plane, world raised 20 mm     13.5 mm (trips)    > 70 mm          > 80 mm
target, world raised 20 mm            0.0 (inside)       33.4 mm          > 80 mm
====================================  =================  ===============  ===============

(Computed offline with a throwaway script: the FK above, then ``openral_hal.convex_distance``'s
GJK between each link's world-placed hull and each occupied cube.) The live kernel agrees: its
stop lines quote 13.505 mm for the tall target's link7 trip and 13.514 mm for the raised
plane's finger trip. So against the 20 mm margin: only the finger trips on the base world (and
only on target cells, all centred in the region); the tall target adds a link7 trip inside the
region; the raised world — equivalently the hand one voxel lower — brings the finger within
margin of the plane while link7 stays clear of the target. The undeclared row asserts that the
finger is the only link the kernel names, which re-checks this table on the kernel every run.

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
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
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

from openral_core import GraspDeclaration, PlaceRegion, Pose6D, RobotDescription  # noqa: E402
from openral_core.depth_extrinsic import REAL_WORLD_VOXEL_MARGIN_M  # noqa: E402
from openral_safety.envelope_loader import (  # noqa: E402
    collision_params_from_description,
    compute_intersection,
    ee_link_index_from_collision_params,
    kernel_params_from_envelope,
)

# The kernel-spawn helpers are shared with the tests/sim/safety kernel suite (§1.13).
from tests.sim.safety._kernel_subprocess import (  # noqa: E402
    activate_kernel_node,
    isolated_domain_id,
    start_kernel,
    terminate_kernel,
)

_REPO = pathlib.Path(__file__).resolve().parents[2]
_ROBOT_YAML = _REPO / "robots" / "openarm" / "robot.yaml"
_FRAME = "openarm_base"  # the manifest's base_frame: the collision FK root and the grid's frame
_LEFT_FINGER = "openarm_left_finger_pair"
_RIGHT_FINGER = "openarm_right_finger_pair"
_LEFT_LINK7 = "openarm_left_link7"

# ── The grid (real-cell resolution: `_octomap_resolution("real")` in deploy_e2e) ─────────────
_RES = 0.02
_SX, _SY, _SZ = 6, 8, 7
_ORIGIN = (-0.06, 0.10, -0.68)  # min corner of cell (0,0,0); cell centres sit on odd cm
#: The world raised one voxel (the support-plane row): same cells, same region, +20 mm in z.
_RAISED_DZ = _RES


def _index(i: int, j: int, k: int) -> int:
    return i + _SX * (j + _SY * k)


#: Layer k=1 (centre z = -0.65): a support plane under the whole footprint.
_PLANE = frozenset(_index(i, j, 1) for i in range(_SX) for j in range(_SY))
#: The target: 2 × 4 cells, two layers (centres z = -0.63, -0.61), standing on the plane.
_TARGET = frozenset(_index(i, j, k) for i in (2, 3) for j in (2, 3, 4, 5) for k in (2, 3))
#: The same target two layers taller (to z = -0.57): it reaches link7's margin.
_TALL_TARGET = frozenset(
    _index(i, j, k) for i in (2, 3) for j in (2, 3, 4, 5) for k in (2, 3, 4, 5)
)

# ── The region the producer would measure (base-frame oriented box) ───────────────────────────
# Lower face at z = -0.64: the plane's top face, so the plane's centres are half a voxel below
# it (outside) while every target centre is inside. Top face one voxel above the target.
_REGION_CENTRE = (0.0, 0.18, -0.61)
_REGION_HALF = (0.03, 0.05, 0.03)
_TALL_REGION_CENTRE = (0.0, 0.18, -0.60)
_TALL_REGION_HALF = (0.03, 0.05, 0.04)

# ── The configuration (pinned; see the module docstring for how it was found) ────────────────
#: Both arms hanging straight down, grippers at 0 rad (the hull covers the whole stroke).
#: Order = the manifest's actuated joints (left 1-7, left_gripper, right 1-7, right_gripper).
_Q = [0.0] * 16

_TARGET_ID = "cell:restock_box"  # tests/unit/fixtures/scenes/openarm_direct_dispatch_grasp.yaml
_RSKILL_ID = "openral/grasp-target-band"
_TIMEOUT_S = 70.0  # that fixture scene's grasp_declaration.timeout_s

_STOP_LINE = re.compile(
    r"safety\.collision kind=world a=(\S+) b=(\S+) .*"
    r"grasp_exemption_active=(\d) grasp_target=(\S*)"
)


def _description() -> RobotDescription:
    return RobotDescription.from_yaml(str(_ROBOT_YAML))


def _kernel_params(*, grasp_allowance_enabled: bool) -> dict[str, object]:
    """The safety kernel's parameters as ``deploy_e2e.launch.py`` builds them for the real cell.

    Same builders, same order: envelope, then the manifest collision model, then the
    world-voxel block at the real margin, then the grasp keys (the allowlist is passed on or
    off, exactly as the launch does). Attached checking stays off: the real cell without the
    vision leg (``_attached_collision_enabled("real", False)``).
    """
    description = _description()
    collision = collision_params_from_description(description)
    params: dict[str, object] = {
        **kernel_params_from_envelope(compute_intersection(description, skill=None, deploy=None)),
        **collision,
    }
    params["collision_joint_names"] = [j.name for j in description.joints]
    params["collision_seed_dt_s"] = 0.0
    params["collision_state_deadline_ms"] = 1000.0
    params["collision_ee_link_index"] = ee_link_index_from_collision_params(collision)
    params["grasp_allowance_enabled"] = grasp_allowance_enabled
    params["grasp_contact_links"] = [
        j.child_link for j in description.joints if j.role == "gripper"
    ]
    params["world_voxel_enabled"] = True
    params["world_voxel_margin_m"] = REAL_WORLD_VOXEL_MARGIN_M
    params["world_voxel_max_cells"] = 4096
    params["world_voxel_deadline_ms"] = 2000.0
    return params


def _region(
    centre: tuple[float, float, float], half: tuple[float, float, float], frame_id: str = _FRAME
) -> PlaceRegion:
    return PlaceRegion(
        frame_id=frame_id,
        pose=Pose6D(xyz=centre, quat_xyzw=(0.0, 0.0, 0.0, 1.0), frame_id=frame_id),
        half_extents=half,
        evidence_ref="grasp_target_leg:seg_mask+voxels",
        stamp_ns=time.time_ns(),
    )


def _declaration(
    region: PlaceRegion,
    *,
    contact_links: tuple[str, ...] = (_LEFT_FINGER,),
    stamp_ns: int | None = None,
    timeout_s: float = _TIMEOUT_S,
) -> GraspDeclaration:
    return GraspDeclaration(
        target_id=_TARGET_ID,
        contact_links=contact_links,
        rskill_id=_RSKILL_ID,
        trace_id=f"grasp-band-{uuid.uuid4().hex[:8]}",
        timeout_s=timeout_s,
        # A NEW declaration per row (fresh stamp): a retired one's heartbeat never re-arms.
        stamp_ns=time.time_ns() if stamp_ns is None else stamp_ns,
        region=region,
    )


@dataclass
class _Cell:
    """One live kernel plus the helper node standing in for HAL, octomap bridge and producer."""

    helper: Any
    executor: Any
    chunk_pub: Any
    reset_client: Any
    log_path: pathlib.Path
    safe: dict[str, Any] = field(default_factory=dict)
    failures: list[Any] = field(default_factory=list)
    estops: list[Any] = field(default_factory=list)
    diagnostics: dict[str, str] = field(default_factory=dict)
    occupied: frozenset[int] = _TARGET | _PLANE
    grid_dz: float = 0.0
    declaration: GraspDeclaration | None = None
    #: What the attachment producer holds (``openral_core.AttachedCollisionObject``) and
    #: its revision; a change of the set is a new revision, as the bridge publishes it.
    attached: list[Any] = field(default_factory=list)
    revision: int = 1

    def spin(self, seconds: float) -> None:
        end = time.time() + seconds
        while time.time() < end:
            self.executor.spin_once(timeout_sec=0.02)

    def world(
        self,
        occupied: frozenset[int],
        declaration: GraspDeclaration | None,
        *,
        grid_dz: float = 0.0,
        attached: list[Any] | None = None,
    ) -> None:
        """Swap what the 5 Hz grid and 10 Hz world-state heartbeats publish, and settle."""
        self.occupied, self.grid_dz, self.declaration = occupied, grid_dz, declaration
        if attached is not None and attached != self.attached:
            self.attached, self.revision = attached, self.revision + 1
        self.spin(0.6)

    def send(self, trace: str, *, expect_accept: bool) -> None:
        """Publish one candidate at the pinned configuration; wait on the outcome, not a sleep."""
        from openral_msgs.msg import ActionChunk

        seen = len(self.failures)
        chunk = ActionChunk()
        chunk.control_mode = 0  # JOINT_POSITION
        chunk.horizon = 1
        chunk.n_dof = len(_Q)
        chunk.flat = list(_Q)
        chunk.rskill_id = _RSKILL_ID
        chunk.trace_id = trace
        self.chunk_pub.publish(chunk)
        deadline = time.time() + 10.0
        while time.time() < deadline:
            self.executor.spin_once(timeout_sec=0.02)
            if expect_accept and trace in self.safe:
                break
            if not expect_accept and len(self.failures) > seen and self.estops:
                break
        self.spin(0.4)  # settle: a late accept/estop must still be visible

    def refused(self, trace: str) -> dict[str, Any]:
        """Assert ``trace`` was refused as a world collision; return the trigger's evidence."""
        from openral_msgs.msg import FailureTrigger

        assert trace not in self.safe, f"{trace} reached /openral/safe_action"
        assert self.estops, f"{trace} must fire /openral/estop"
        trigger = self.failures[-1]
        assert trigger.kind == FailureTrigger.KIND_COLLISION
        evidence: dict[str, Any] = json.loads(trigger.evidence_json)
        assert evidence["collision_kind"] == "world"
        return evidence

    def stop_lines(self) -> list[tuple[str, str, int, str]]:
        text = self.log_path.read_text(encoding="utf-8", errors="replace")
        return [(a, b, int(on), tgt) for a, b, on, tgt in _STOP_LINE.findall(text)]

    def log(self) -> str:
        return self.log_path.read_text(encoding="utf-8", errors="replace")

    def await_diagnostic(self, predicate: Callable[[str], bool]) -> str:
        """The kernel's ``grasp_region`` diagnostic once it satisfies ``predicate`` (1 Hz)."""
        deadline = time.time() + 4.0
        while time.time() < deadline and not predicate(self.diagnostics.get("grasp_region", "")):
            self.executor.spin_once(timeout_sec=0.05)
        return self.diagnostics.get("grasp_region", "<never published>")


@contextmanager
def _live_cell(
    *, grasp_allowance_enabled: bool, reset_kernel_estop: Callable[..., None]
) -> Iterator[tuple[_Cell, Callable[[], None]]]:
    """Spawn the kernel, wire the helper node, start the heartbeats; tear all of it down."""
    import rclpy
    from diagnostic_msgs.msg import DiagnosticArray
    from geometry_msgs.msg import Point, Quaternion
    from openral_msgs.msg import (
        ActionChunk,
        FailureTrigger,
        OccupancyVoxels,
        WorldStateStamped,
    )
    from openral_msgs.msg import AttachedCollisionObject as AttachedCollisionObjectMsg
    from openral_msgs.msg import AttachedCollisionPrimitive as AttachedCollisionPrimitiveMsg
    from openral_msgs.msg import GraspDeclaration as GraspDeclarationMsg
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import JointState
    from std_msgs.msg import Empty
    from std_srvs.srv import Trigger

    node_name = f"safety_kernel_grasp_band_{uuid.uuid4().hex[:8]}"
    joint_names = [j.name for j in _description().joints]
    with tempfile.TemporaryDirectory() as td:
        log_path = pathlib.Path(td) / "kernel.log"
        params = _kernel_params(grasp_allowance_enabled=grasp_allowance_enabled)
        proc = start_kernel(
            params,
            node_name,
            isolated_domain_id(),
            log_path=log_path,
            params_file=pathlib.Path(td) / "kernel_params.yaml",
        )
        try:
            time.sleep(1.5)
            rclpy.init()
            try:
                helper = rclpy.create_node("grasp_band_helper")
                assert activate_kernel_node(node_name, helper), "kernel activation failed"
                executor = SingleThreadedExecutor()
                executor.add_node(helper)
                cell = _Cell(
                    helper=helper,
                    executor=executor,
                    chunk_pub=helper.create_publisher(ActionChunk, "/openral/candidate_action", 10),
                    reset_client=helper.create_client(Trigger, "/openral/estop_reset"),
                    log_path=log_path,
                )
                safe_sub = helper.create_subscription(
                    ActionChunk,
                    "/openral/safe_action",
                    lambda m: cell.safe.__setitem__(m.trace_id, m),
                    10,
                )
                helper.create_subscription(
                    FailureTrigger, "/openral/failure/safety", cell.failures.append, 50
                )
                helper.create_subscription(Empty, "/openral/estop", cell.estops.append, 10)

                def on_diagnostics(msg: Any) -> None:
                    for status in msg.status:
                        for kv in status.values:
                            if kv.key == "grasp_region":
                                cell.diagnostics["grasp_region"] = kv.value

                helper.create_subscription(DiagnosticArray, "/diagnostics", on_diagnostics, 10)
                reliable_kl1 = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE)
                voxel_pub = helper.create_publisher(
                    OccupancyVoxels, "/openral/world_voxels", reliable_kl1
                )
                state_pub = helper.create_publisher(
                    WorldStateStamped, "/openral/world_state_fast", reliable_kl1
                )
                joint_pub = helper.create_publisher(JointState, "/joint_states", 10)

                # Heartbeats, as the octomap bridge (5 Hz), the HAL (joint states) and the
                # attachment producer do: the kernel fails closed on a stale grid, a stale
                # measured state, and — for the exemption — a world-state stream older than
                # attached_collision_deadline_ms (the kernel's 500 ms default here).
                def publish_grid() -> None:
                    grid = OccupancyVoxels()
                    grid.header.frame_id = _FRAME
                    grid.header.stamp = helper.get_clock().now().to_msg()
                    grid.source_stamp = grid.header.stamp
                    grid.origin = Point(x=_ORIGIN[0], y=_ORIGIN[1], z=_ORIGIN[2] + cell.grid_dz)
                    grid.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
                    grid.resolution = _RES
                    grid.size_x, grid.size_y, grid.size_z = _SX, _SY, _SZ
                    occupancy = [0] * (_SX * _SY * _SZ)
                    for idx in cell.occupied:
                        occupancy[idx] = 1
                    grid.occupancy = occupancy
                    voxel_pub.publish(grid)

                def publish_joint_state() -> None:
                    js = JointState()
                    js.header.stamp = helper.get_clock().now().to_msg()
                    js.name = joint_names
                    js.position = list(_Q)
                    joint_pub.publish(js)

                def publish_world_state() -> None:
                    now_ns = helper.get_clock().now().nanoseconds
                    state = WorldStateStamped()
                    state.header.frame_id = _FRAME
                    state.header.stamp = helper.get_clock().now().to_msg()
                    state.stamp_ns = now_ns
                    state.attachment_revision = cell.revision
                    state.attachment_stamp_ns = now_ns
                    for held in cell.attached:
                        item = AttachedCollisionObjectMsg()
                        held.fill_idl(item, primitive_factory=AttachedCollisionPrimitiveMsg)
                        state.attached_objects.append(item)
                    state.grasp_declaration_valid = cell.declaration is not None
                    if cell.declaration is not None:
                        msg = GraspDeclarationMsg()
                        cell.declaration.fill_idl(msg)
                        state.grasp_declaration = msg
                    state_pub.publish(state)

                for period, publish in (
                    (0.2, publish_grid),
                    (0.1, publish_joint_state),
                    (0.1, publish_world_state),
                ):
                    publish()
                    helper.create_timer(period, publish)

                deadline = time.time() + 5.0
                while time.time() < deadline and not (
                    cell.chunk_pub.get_subscription_count() >= 1
                    and safe_sub.get_publisher_count() >= 1
                ):
                    executor.spin_once(timeout_sec=0.05)
                cell.spin(0.6)

                def reset() -> None:
                    reset_kernel_estop(cell.reset_client, executor, cell.spin, cell.estops)

                yield cell, reset
            finally:
                rclpy.shutdown()
        except AssertionError as exc:
            # Every verdict asserted on has a matching kernel log line; surface it.
            raise AssertionError(
                f"{exc}\n\n--- safety_kernel_node log ---\n"
                f"{log_path.read_text(encoding='utf-8', errors='replace')}"
            ) from exc
        finally:
            terminate_kernel(proc)


def test_grasp_exemption_is_off_by_default_end_to_end(
    reset_kernel_estop: Callable[..., None],
) -> None:
    """Feature off (the launch default): a live, valid declaration exempts nothing."""
    with _live_cell(grasp_allowance_enabled=False, reset_kernel_estop=reset_kernel_estop) as (
        cell,
        _,
    ):
        cell.world(_TARGET | _PLANE, _declaration(_region(_REGION_CENTRE, _REGION_HALF)))
        cell.send("off-declared", expect_accept=False)
        evidence = cell.refused("off-declared")
        assert evidence["link_a"] == _LEFT_FINGER
        assert evidence["link_b_or_object"] in {f"voxel_{i}" for i in _TARGET}
        assert cell.await_diagnostic(lambda v: v == "off") == "off"
        assert "safety.grasp_region_armed" not in cell.log()
        assert [line[2] for line in cell.stop_lines()] == [0]


def test_grasp_exemption_band_on_the_real_openarm_model(
    reset_kernel_estop: Callable[..., None],
) -> None:
    """Feature on: every row of the module docstring's table, against one kernel."""
    target_voxels = {f"voxel_{i}" for i in _TARGET}
    with _live_cell(grasp_allowance_enabled=True, reset_kernel_estop=reset_kernel_estop) as (
        cell,
        reset,
    ):
        assert "safety.grasp_allowance enabled" in cell.log()

        # ── Undeclared: the finger stops on the target's own cells ────────────────────────
        cell.world(_TARGET | _PLANE, None)
        cell.send("undeclared", expect_accept=False)
        evidence = cell.refused("undeclared")
        assert evidence["link_a"] == _LEFT_FINGER, "only the finger may reach the target"
        assert evidence["link_b_or_object"] in target_voxels
        assert cell.stop_lines()[-1][2:] == (0, ""), "no exemption was live"
        reset()

        # ── Declared, left finger: THE EXEMPTION — accepted ──────────────────────────────
        cell.world(_TARGET | _PLANE, _declaration(_region(_REGION_CENTRE, _REGION_HALF)))
        cell.send("declared", expect_accept=True)
        assert "declared" in cell.safe, "a live left-finger declaration must exempt the target"
        assert not cell.estops
        assert f"safety.grasp_region_armed target={_TARGET_ID} links=1" in cell.log()
        live = cell.await_diagnostic(lambda v: v.startswith("live:"))
        assert live == f"live:{_TARGET_ID}:links=1", live

        # ── Declared, but link7 also reaches region cells: refused on link7 ──────────────
        cell.world(
            _TALL_TARGET | _PLANE, _declaration(_region(_TALL_REGION_CENTRE, _TALL_REGION_HALF))
        )
        cell.send("link7-in-region", expect_accept=False)
        evidence = cell.refused("link7-in-region")
        assert evidence["link_a"] == _LEFT_LINK7, "the wrist is not a contact link"
        assert evidence["link_b_or_object"] in {f"voxel_{i}" for i in _TALL_TARGET - _TARGET}
        a, _, active, target = cell.stop_lines()[-1]
        assert (a, active, target) == (_LEFT_LINK7, 1, _TARGET_ID), (
            "a stop while the exemption is live must disclose it"
        )
        reset()

        # ── Declared for the RIGHT gripper only: the left finger is not exempt ───────────
        cell.world(
            _TARGET | _PLANE,
            _declaration(_region(_REGION_CENTRE, _REGION_HALF), contact_links=(_RIGHT_FINGER,)),
        )
        cell.send("right-only", expect_accept=False)
        evidence = cell.refused("right-only")
        assert evidence["link_a"] == _LEFT_FINGER
        assert evidence["link_b_or_object"] in target_voxels
        assert cell.stop_lines()[-1][2:] == (1, _TARGET_ID), "armed, but for the other hand"
        reset()

        # ── Declared, expired: armed but not live ─────────────────────────────────────────
        cell.world(
            _TARGET | _PLANE,
            _declaration(
                _region(_REGION_CENTRE, _REGION_HALF),
                stamp_ns=time.time_ns() - 10_000_000_000,
                timeout_s=5.0,
            ),
        )
        cell.send("expired", expect_accept=False)
        evidence = cell.refused("expired")
        assert evidence["link_a"] == _LEFT_FINGER
        assert cell.stop_lines()[-1][2:] == (0, ""), "an expired declaration exempts nothing"
        expired = cell.await_diagnostic(lambda v: v.startswith("expired:"))
        assert expired == f"expired:{_TARGET_ID}:links=1", expired
        reset()

        # ── Declared, region in another frame: rejected, logged ──────────────────────────
        cell.world(
            _TARGET | _PLANE,
            _declaration(_region(_REGION_CENTRE, _REGION_HALF, frame_id="world")),
        )
        cell.send("frame-mismatch", expect_accept=False)
        evidence = cell.refused("frame-mismatch")
        assert evidence["link_a"] == _LEFT_FINGER
        assert cell.stop_lines()[-1][2:] == (0, "")
        assert (
            f"safety.grasp_region_rejected reason=frame_mismatch target={_TARGET_ID}" in cell.log()
        )
        rejected = cell.await_diagnostic(lambda v: v.startswith("frame_mismatch"))
        assert rejected == f"frame_mismatch:{_TARGET_ID}", rejected
        reset()

        # ── Declared, the support plane within the finger's margin: refused on the plane ─
        # World and region raised one voxel together: the plane's centres stay half a voxel
        # below the region's lower face, and are now 13.5 mm from the finger hull.
        cell.world(
            _TARGET | _PLANE,
            _declaration(
                _region(
                    (_REGION_CENTRE[0], _REGION_CENTRE[1], _REGION_CENTRE[2] + _RAISED_DZ),
                    _REGION_HALF,
                )
            ),
            grid_dz=_RAISED_DZ,
        )
        cell.send("support-plane", expect_accept=False)
        evidence = cell.refused("support-plane")
        assert evidence["link_a"] == _LEFT_FINGER
        assert evidence["link_b_or_object"] in {f"voxel_{i}" for i in _PLANE}, (
            "the support surface under the target still stops the finger"
        )
        assert cell.stop_lines()[-1][2:] == (1, _TARGET_ID)


#: A fit tight to the target's measured surface (Isaac i36/i37): x within +-8 mm, inside the
#: boundary cells x -0.02..0.02 whose centres (x = +-0.01) lie outside it; bottom on the plane's
#: top face (z = -0.64), top at -0.605 inside the upper target layer.
_TIGHT_CENTRE = (0.0, 0.18, -0.6225)
_TIGHT_HALF = (0.008, 0.035, 0.0175)


def test_a_tight_fit_exempts_the_targets_boundary_cells_only_once_cell_closed(
    reset_kernel_estop: Callable[..., None],
) -> None:
    """The kernel exempts a cell only when its centre is in the region. A fit tight to the
    measured surface leaves every target cell here (centres at x = +-0.01) outside, so the
    finger hull stops on the target's own cells; closed over the grid's cells — what the
    grasp-target leg publishes — the same fit exempts them, and the plane under it stays
    non-exempt (the closure never grows down)."""
    import numpy as np
    from openral_core import GraspDeclaration as _Declaration
    from openral_hal._grasp_target import VoxelLattice, cell_closed_region

    lattice = VoxelLattice(
        _FRAME,
        _ORIGIN,
        (0.0, 0.0, 0.0, 1.0),
        _RES,
        (_SX, _SY, _SZ),
        np.zeros(_SX * _SY * _SZ, dtype=np.uint8),
    )
    with _live_cell(grasp_allowance_enabled=True, reset_kernel_estop=reset_kernel_estop) as (
        cell,
        reset,
    ):
        tight = _region(_TIGHT_CENTRE, _TIGHT_HALF)
        cell.world(_TARGET | _PLANE, _declaration(tight))
        cell.send("tight-fit", expect_accept=False)
        evidence = cell.refused("tight-fit")
        assert evidence["link_a"] == _LEFT_FINGER
        assert evidence["link_b_or_object"] in {f"voxel_{i}" for i in _TARGET}, (
            "the tight fit's own target cells must be what stops the finger"
        )
        assert cell.stop_lines()[-1][2:] == (1, _TARGET_ID)
        reset()

        closed, clamped = cell_closed_region(
            _region(_TIGHT_CENTRE, _TIGHT_HALF),
            lattice,
            max_half_extent_m=_Declaration.MAX_HALF_EXTENT_M,
        )
        assert not clamped
        assert closed.pose.xyz[2] - closed.half_extents[2] == pytest.approx(-0.64)
        cell.world(_TARGET | _PLANE, _declaration(closed))
        cell.send("cell-closed", expect_accept=True)
        assert "cell-closed" in cell.safe, "the cell-closed fit must exempt the target's cells"
        assert not cell.estops


#: A neighbour standing on the plane one empty column beside the target (x = 0.05, cube
#: 0.04-0.06; the region ends at x = 0.03): outside the measured region, but 13 mm from the
#: left finger pair's hull, whose lower part spans x <= 0.027 at q = 0 (same FK as above).
_NEIGHBOUR = frozenset(_index(5, j, k) for j in (3, 4) for k in (2, 3))
_APPROACH_TARGET = f"approach:{_LEFT_FINGER}:1"  # the tracker's first pick (``first_pick=1``)


def _approach_declaration() -> GraspDeclaration:
    """What the HAL's grasp-target leg publishes with NO named target.

    The runner's goal-scope declaration (both hands, no target, no search box) narrowed by
    the producer's own tracker to the hand whose TCP approached occupied cells — the real
    ``GraspTargetTracker.on_approach`` / ``accept`` / ``envelope`` path, the left finger
    pair's TCP over the target — then the region measured now (a fresh stamp: the tracker
    retracts a region past its freeze TTL, and with it the approach).
    """
    from openral_hal._grasp_target_leg import GraspTargetTracker, approach_box

    now = time.time_ns()
    goal = GraspDeclaration(
        target_id="approach",
        contact_links=(_LEFT_FINGER, _RIGHT_FINGER),
        rskill_id=_RSKILL_ID,
        trace_id=f"grasp-band-{uuid.uuid4().hex[:8]}",
        timeout_s=_TIMEOUT_S,
        stamp_ns=now,
    )
    tracker = GraspTargetTracker(freeze_s=2.0, log=lambda _line: None, first_pick=1)
    tracker.on_declaration(goal)
    tcp = (_REGION_CENTRE[0], _REGION_CENTRE[1], _REGION_CENTRE[2] + 0.05)
    tracker.on_approach(
        [((_LEFT_FINGER,), approach_box([tcp], approach_m=0.10, frame_id=_FRAME))],
        now_ns=now,
        move_m=_RES,
    )
    tracker.accept(_region(_REGION_CENTRE, _REGION_HALF))
    declaration = tracker.envelope(now_ns=now)
    assert declaration is not None and declaration.region is not None
    assert declaration.target_id == _APPROACH_TARGET
    assert declaration.contact_links == (_LEFT_FINGER,)
    return declaration


def test_an_approach_armed_declaration_exempts_one_hand_and_nothing_beside_the_target(
    reset_kernel_estop: Callable[..., None],
) -> None:
    """No dispatch declaration names the target: the producer's approach-armed one-hand
    declaration rides the envelope like any other, and the kernel bounds it the same way.

    ====================================================  =====================================
    row                                                   verdict
    ====================================================  =====================================
    approach-armed (left hand), target only               ACCEPTED, armed ``approach:<left>``
    approach-armed, a neighbour beside the target         REFUSED, left finger on the neighbour,
    (outside the measured region, in the finger margin)   ``grasp_exemption_active=1``
    a region-carrying declaration naming BOTH hands       REFUSED, ``reason=links_span_hands``
    ====================================================  =====================================
    """
    with _live_cell(grasp_allowance_enabled=True, reset_kernel_estop=reset_kernel_estop) as (
        cell,
        reset,
    ):
        # ── Approach-armed, left hand: the finger reaches the target ─────────────────────
        cell.world(_TARGET | _PLANE, _approach_declaration())
        cell.send("approach-armed", expect_accept=True)
        assert "approach-armed" in cell.safe, "the approach-armed region must exempt the target"
        assert f"safety.grasp_region_armed target={_APPROACH_TARGET} links=1" in cell.log()

        # ── A neighbour beside the target, outside the region: still a stop ──────────────
        cell.world(_TARGET | _PLANE | _NEIGHBOUR, _approach_declaration())
        cell.send("approach-neighbour", expect_accept=False)
        evidence = cell.refused("approach-neighbour")
        assert evidence["link_a"] == _LEFT_FINGER
        assert evidence["link_b_or_object"] in {f"voxel_{i}" for i in _NEIGHBOUR}, (
            "the neighbour is not the approached target and stops the exempt finger"
        )
        assert cell.stop_lines()[-1][2:] == (1, _APPROACH_TARGET)
        reset()

        # ── Both hands on one region: refused at ingest, nothing exempt ───────────────────
        cell.world(
            _TARGET | _PLANE,
            _declaration(
                _region(_REGION_CENTRE, _REGION_HALF),
                contact_links=(_LEFT_FINGER, _RIGHT_FINGER),
            ),
        )
        cell.send("two-hands", expect_accept=False)
        evidence = cell.refused("two-hands")
        assert evidence["link_a"] == _LEFT_FINGER
        assert cell.stop_lines()[-1][2:] == (0, ""), "a two-hand declaration exempts nothing"
        assert (
            f"safety.grasp_region_rejected reason=links_span_hands target={_TARGET_ID}"
            in cell.log()
        )


def _t_base_from_link_at_q0(link: str) -> Any:
    """``openarm_base <- link`` at the pinned all-zero configuration: the manifest's chain.

    Joint origins composed parent before child; at ``q = 0`` every joint's motion is the
    identity, so the chain is its origins (the kernel's own FK at ``_Q``).
    """
    import numpy as np
    from openral_hal.vision_attachment_bridge import _xyz_rpy_matrix

    description = _description()
    up = {j.child_link: (j.parent_link, j.origin_xyz, j.origin_rpy) for j in description.joints}
    up |= {
        f.child_link: (f.parent_link, f.origin_xyz, f.origin_rpy)
        for f in description.fixed_attachments
    }
    t = np.eye(4)
    while link != _FRAME:
        parent, xyz, rpy = up[link]
        t = _xyz_rpy_matrix(xyz, rpy) @ t
        link = parent
    return t


def test_two_approach_armed_picks_in_one_goal_on_the_real_kernel(
    reset_kernel_estop: Callable[..., None],
) -> None:
    """Multi-pick per goal: the producer's real tracker and the real bridge records, against
    the real kernel, the right hand holding its own payload throughout.

    ====================================================  =====================================
    row                                                   verdict
    ====================================================  =====================================
    pick 1 (``approach:<left>:1``), measured region       ACCEPTED, finger in the target
    ATTACH: the region payload on the left gripper        handover latched, ACCEPTED
    release: the left payload frozen on the base, the     ``grasp_region_dropped reason=released``
    right hand still holding (attachment set not empty)   — retired at the release; then the
                                                          producer drops the region
    pick 2 (``approach:<left>:2``), re-measured, same     ACCEPTED, finger in the target — a
    goal stamp                                            fresh identity, not the retired one
    pick 1's identity again, fresh region (a restarted    REFUSED — retired, nothing exempt
    producer re-sending it)
    ====================================================  =====================================
    """
    import numpy as np
    from openral_hal._grasp_target_leg import GraspTargetTracker, approach_box
    from openral_hal.vision_attachment_bridge import (
        VisionAttachmentBridge,
        VisionAttachmentConfig,
        freeze_released_attachment,
        region_attachment,
    )

    legs = VisionAttachmentBridge(
        None, _description(), config=VisionAttachmentConfig(camera="head_zed")
    )._legs
    left = next(g for g in legs if g.jaw_link == _LEFT_FINGER)
    right = next(g for g in legs if g.jaw_link == _RIGHT_FINGER)
    t_base_from_left = _t_base_from_link_at_q0(left.producer.attach_link)
    now = time.time_ns()
    goal = GraspDeclaration(
        target_id="approach",
        contact_links=(_LEFT_FINGER, _RIGHT_FINGER),
        rskill_id=_RSKILL_ID,
        trace_id=f"grasp-band-{uuid.uuid4().hex[:8]}",
        timeout_s=_TIMEOUT_S,
        stamp_ns=now,
    )
    tracker = GraspTargetTracker(freeze_s=2.0, log=lambda _line: None, first_pick=1)
    tracker.on_declaration(goal)

    def arm(tcp_dy: float) -> GraspDeclaration:
        """The left TCP over the target arms; its region is measured now."""
        tcp = (_REGION_CENTRE[0], _REGION_CENTRE[1] + tcp_dy, _REGION_CENTRE[2] + 0.05)
        tracker.on_approach(
            [((_LEFT_FINGER,), approach_box([tcp], approach_m=0.10, frame_id=_FRAME))],
            now_ns=time.time_ns(),
            move_m=_RES,
        )
        tracker.accept(_region(_REGION_CENTRE, _REGION_HALF))
        declaration = tracker.envelope(now_ns=time.time_ns())
        assert declaration is not None and declaration.region is not None
        return declaration

    def payload(declaration: GraspDeclaration, gripper: Any) -> Any:
        """The bridge's region payload of ``declaration`` on ``gripper``, posed at ``q = 0``."""
        return region_attachment(
            declaration,
            attach_link=gripper.producer.attach_link,
            touch_links=gripper.producer.touch_links,
            t_link_from_region=np.linalg.inv(_t_base_from_link_at_q0(gripper.producer.attach_link)),
            stamp_ns=time.time_ns(),
        )

    first_id, second_id = f"approach:{_LEFT_FINGER}:1", f"approach:{_LEFT_FINGER}:2"
    with _live_cell(grasp_allowance_enabled=True, reset_kernel_estop=reset_kernel_estop) as (
        cell,
        reset,
    ):
        # The right hand holds its own payload throughout (attached checking is off here).
        right_held = payload(_declaration(_region((0.0, -0.18, -0.61), _REGION_HALF)), right)

        # ── Pick 1: the finger reaches the measured target ──────────────────────────────
        first = arm(0.0)
        assert first.target_id == first_id
        cell.world(_TARGET | _PLANE, first, attached=[right_held])
        cell.send("pick-1", expect_accept=True)
        assert "pick-1" in cell.safe, cell.log()

        # ── ATTACH: the bridge's region payload; the producer hands pick 1 over ─────────
        held_payload = payload(first, left)
        tracker.on_attach(_LEFT_FINGER, confirm=lambda region: region == first.region)
        assert tracker.handed_over == (_LEFT_FINGER,)
        held = tracker.envelope(now_ns=time.time_ns())
        assert held is not None and held.region == first.region
        cell.world(_TARGET | _PLANE, held, attached=[right_held, held_payload])
        cell.send("pick-1-held", expect_accept=True)
        assert "pick-1-held" in cell.safe, cell.log()
        assert f"safety.grasp_region_handover target={first_id}" in cell.log()

        # ── Release: the payload frozen on the base, the right hand still holding ───────
        frozen = freeze_released_attachment(
            held_payload, base_link=_FRAME, t_base_from_link=t_base_from_left
        )
        # The bridge's DETACH publishes the frozen record first, the envelope unchanged...
        cell.world(_TARGET | _PLANE, held, attached=[right_held, frozen])
        assert f"safety.grasp_region_dropped reason=released target={first_id}" in cell.log()
        # ...then hands the record to the leg, which drops the region at once.
        tracker.on_release((_LEFT_FINGER,), frozen)
        released = tracker.envelope(now_ns=time.time_ns())
        assert released is not None and released.region is None, "region dropped at DETACH"
        cell.world(_TARGET | _PLANE, released, attached=[right_held, frozen])
        # The release window closes: the hand holds nothing, the pick is complete.
        cell.world(_TARGET | _PLANE, released, attached=[right_held])
        assert tracker.on_pick_complete((_LEFT_FINGER,), now_ns=time.time_ns())

        # ── Pick 2, same goal: a fresh identity, so the kernel arms it ──────────────────
        second = arm(2.0 * _RES)  # the hand moved on by two voxels: past the backoff
        assert second.target_id == second_id and second.stamp_ns == goal.stamp_ns
        cell.world(_TARGET | _PLANE, second, attached=[right_held])
        cell.send("pick-2", expect_accept=True)
        assert "pick-2" in cell.safe, "the second pick's region was refused"
        assert f"safety.grasp_region_armed target={second_id} links=1" in cell.log()

        # ── Pick 1's identity again (a restarted producer): refused as retired ──────────
        rearmed = first.model_copy(update={"region": _region(_REGION_CENTRE, _REGION_HALF)})
        assert (rearmed.target_id, rearmed.stamp_ns) == (first_id, goal.stamp_ns)
        cell.world(_TARGET | _PLANE, rearmed, attached=[right_held])
        cell.send("pick-1-rearmed", expect_accept=False)
        evidence = cell.refused("pick-1-rearmed")
        assert evidence["link_a"] == _LEFT_FINGER
        assert cell.stop_lines()[-1][2] == 0, "a retired identity re-armed"
        reset()


def test_another_hands_release_retires_a_pre_handover_arming_and_the_producer_re_arms(
    reset_kernel_estop: Callable[..., None],
) -> None:
    """HZ-0115-3, kernel and producer together: the kernel retires an armed declaration at
    any detach edge (the attachment set emptying at a new revision), before its handover too,
    and never re-arms that identity; the producer, which owns the set, keeps the pick live by
    re-arming the other hand under a fresh identity from a measurement taken after the detach.

    ====================================================  =====================================
    row                                                   verdict
    ====================================================  =====================================
    right hand holds; left armed (``approach:<left>:1``)  ACCEPTED, finger in the target
    right hand releases (no release window: the set       ``grasp_region_dropped reason=detached
    empties); the producer refreshes before publishing    target=approach:<left>:1``; the snapshot
                                                          names ``approach:<left>:2``, no region
    left's region measured before the detach, offered     never accepted: REFUSED, nothing exempt
    again to the producer
    re-measured after the detach (``approach:<left>:2``)  ACCEPTED, finger in the target
    ``approach:<left>:1`` again with a fresh region       REFUSED — retired, nothing exempt
    ====================================================  =====================================
    """
    import numpy as np
    from openral_hal._grasp_target_leg import GraspTargetTracker, approach_box
    from openral_hal.vision_attachment_bridge import (
        VisionAttachmentBridge,
        VisionAttachmentConfig,
        region_attachment,
    )

    legs = VisionAttachmentBridge(
        None, _description(), config=VisionAttachmentConfig(camera="head_zed")
    )._legs
    right = next(g for g in legs if g.jaw_link == _RIGHT_FINGER)
    goal = GraspDeclaration(
        target_id="approach",
        contact_links=(_LEFT_FINGER, _RIGHT_FINGER),
        rskill_id=_RSKILL_ID,
        trace_id=f"grasp-band-{uuid.uuid4().hex[:8]}",
        timeout_s=_TIMEOUT_S,
        stamp_ns=time.time_ns(),
    )
    tracker = GraspTargetTracker(freeze_s=2.0, log=lambda _line: None, first_pick=1)
    tracker.on_declaration(goal)
    first_id, second_id = f"approach:{_LEFT_FINGER}:1", f"approach:{_LEFT_FINGER}:2"
    right_held = region_attachment(
        _declaration(_region((0.0, -0.18, -0.61), _REGION_HALF)),
        attach_link=right.producer.attach_link,
        touch_links=right.producer.touch_links,
        t_link_from_region=np.linalg.inv(_t_base_from_link_at_q0(right.producer.attach_link)),
        stamp_ns=time.time_ns(),
    )

    def envelope() -> GraspDeclaration:
        declaration = tracker.envelope(now_ns=time.time_ns())
        assert declaration is not None
        return declaration

    with _live_cell(grasp_allowance_enabled=True, reset_kernel_estop=reset_kernel_estop) as (
        cell,
        reset,
    ):
        # ── The right hand holds; the left hand approaches and is measured ──────────────
        tcp = (_REGION_CENTRE[0], _REGION_CENTRE[1], _REGION_CENTRE[2] + 0.05)
        tracker.on_approach(
            [((_LEFT_FINGER,), approach_box([tcp], approach_m=0.10, frame_id=_FRAME))],
            now_ns=time.time_ns(),
            move_m=_RES,
        )
        before = _region(_REGION_CENTRE, _REGION_HALF)
        tracker.accept(before)
        armed = envelope()
        assert armed.target_id == first_id and armed.region == before
        cell.world(_TARGET | _PLANE, armed, attached=[right_held])
        cell.send("armed", expect_accept=True)
        assert "armed" in cell.safe, cell.log()
        assert tracker.handed_over is None, "pre-handover"

        # ── The right hand releases: the bridge's publish refreshes the arming first ────
        tracker.on_attachment_changed(now_ns=time.time_ns())
        refreshed = envelope()
        assert refreshed.target_id == second_id and refreshed.region is None
        cell.world(_TARGET | _PLANE, refreshed, attached=[])
        assert f"safety.grasp_region_dropped reason=detached target={first_id}" in cell.log()

        # ── The region measured before the detach is never re-used ──────────────────────
        tracker.accept(before)
        stale = envelope()
        assert stale.target_id == second_id and stale.region is None, "pre-detach region reused"
        cell.world(_TARGET | _PLANE, stale, attached=[])
        cell.send("pre-detach-region", expect_accept=False)
        evidence = cell.refused("pre-detach-region")
        assert evidence["link_a"] == _LEFT_FINGER
        assert cell.stop_lines()[-1][2] == 0, "nothing may be exempt before the re-measure"
        reset()

        # ── Re-measured after the detach, under the fresh identity: exempt ──────────────
        after = _region(_REGION_CENTRE, _REGION_HALF)
        tracker.accept(after)
        second = envelope()
        assert second.target_id == second_id and second.region == after
        assert second.stamp_ns == goal.stamp_ns, "the goal's stamp"
        cell.world(_TARGET | _PLANE, second, attached=[])
        cell.send("re-armed", expect_accept=True)
        assert "re-armed" in cell.safe, cell.log()
        assert f"safety.grasp_region_armed target={second_id} links=1" in cell.log()

        # ── The retired identity, with a fresh region: refused ──────────────────────────
        old = armed.model_copy(update={"region": _region(_REGION_CENTRE, _REGION_HALF)})
        cell.world(_TARGET | _PLANE, old, attached=[])
        cell.send("retired", expect_accept=False)
        evidence = cell.refused("retired")
        assert evidence["link_a"] == _LEFT_FINGER
        assert cell.stop_lines()[-1][2] == 0, "a retired identity re-armed"
        assert cell.log().count(f"safety.grasp_region_armed target={first_id} links=1") == 1
        reset()
