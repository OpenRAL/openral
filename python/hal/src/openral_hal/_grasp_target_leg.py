"""Pre-grasp target producer leg — measures the grasp declaration's region on real hardware.

The ROS wiring around ``_grasp_target`` (real pick-and-place design
``docs/reference/real-pick-place-design.md`` §2.2), owned by
``VisionAttachmentBridge`` so ``/openral/attachment_state`` keeps one authority
(§2.4). Default off (``VisionAttachmentConfig.grasp_target_enabled``).

Dispatch names the target on ``/openral/grasp_declaration`` (no region, an
optional ``search_box``); this leg measures where the target is and fills the
region onto every attachment publication. At ``grasp_target_rate_hz``:

1. the support plane is **measured**, never taken from the search box: in a
   column of ``/openral/world_voxels`` under the box (reaching
   ``support_search_below_m`` below its bottom, which is a lifted detection
   bbox and can sit above or below the real table top), the highest layer whose
   top-surface cells ring the footprint of the target standing on it
   (``support_top_from_voxels``, HZ-01xx-6); then the occupied cells of that
   column → one cluster above that plane, whose lowest cell must sit within one
   voxel (+ one of tolerance) of it (else ``not_on_support``) → its top-centre;
2. that seed, which must project into the depth camera through the driver's
   ``CameraInfo`` and tf2 ``optical <- base``, is sent to ``SegmentInView`` as
   the single positive point (no negatives), under the leg's own deadline;
3. the mask + the depth frame it was asked about → ``target_region_from_mask``
   (whose masked cloud must reach within two voxels of the support, else
   ``not_on_support``: a target stacked on another object)
   → ``region_covers_occupied`` against the latest map → ``track_region``
   against the previous accepted region.

**Every failure is an outcome, never a guess** (HZ-01xx-2/-3/-4/-6). Two
classes, each logged once per transition with its typed reason:

* *Contradicting evidence* — no measured support under the target, a target
  not standing on the measured support (``not_on_support``), two
  comparable clusters (``AMBIGUOUS``), a fit over the caps or with no height
  above the support, a re-fit the map does not cover (``map_disagrees``) or
  that moved or resized past one voxel *and* reaches outside the held region
  grown by one voxel (``target_moved``) or has no contact link near it
  (``unoccluded_refit``), a
  frame or calibration mismatch: the region is **retracted at once**.
* *Lost view* — no seed cells, the seed off-image, no mask, a mask captured
  more than ``mask_depth_max_skew_s`` from the depth frame (``mask_depth_skew``),
  a missed deadline,
  too few depth points, stale or missing inputs, and a map-covered re-fit that
  fails the tracking gate but lies inside the held region grown by one voxel
  **while a declared contact link's tf origin is within one voxel +
  ``occluder_margin_m`` of it** (``occluded_refit``: the robot's own hand
  occluding part of the target shrinks and shifts the fit; it never replaces
  the held region). The same shrink with no contact link near is
  ``unoccluded_refit``, a contradiction (a person's hand, the target knocked
  over). The gripper closing in on the target occludes it from the head
  camera exactly then, so the last accepted
  region is **frozen** for at most ``grasp_target_freeze_s`` (unset: twice
  ``grid_max_age_s``, the kernel's voxel deadline) from its own
  ``stamp_ns`` (the depth frame it was measured on), then retracted.

The region dies with the declaration: dispatch retraction (goal end, cancel,
E-stop — the runner retracts on all of them) and ``timeout_s`` expiry clear it.
On an ATTACH of a leg whose jaw link the declaration names, re-measurement stops
and the region is kept for the kernel's handover rule (§2.1).
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray
from openral_core import GraspDeclaration, PlaceRegion
from openral_core.exceptions import ROSConfigError
from openral_core.geometry import homogeneous_from_quat_xyz

from openral_hal._grasp_target import (
    TargetRefusal,
    VoxelLattice,
    _in_region,
    occupied_centers_in_box,
    project_point,
    region_covers_occupied,
    region_within,
    support_top_from_voxels,
    target_region_from_mask,
    target_seed_from_voxels,
    track_region,
)

if TYPE_CHECKING:  # pragma: no cover — typing only
    from openral_core import IntrinsicsPinhole

    from openral_hal.vision_attachment_bridge import VisionAttachmentBridge, VisionAttachmentConfig

__all__ = ["GraspTargetLeg", "GraspTargetTracker", "lattice_from_msg", "search_column"]

#: The design's re-prompt band, Hz (§2.2).
_RATE_BAND_HZ = (2.0, 5.0)

#: Tilt tolerance for the search box: its z axis must be the base frame's.
_GRAVITY_TOL = 1e-6

# Upper bounds that keep the gates meaningful (CLAUDE.md §1.2: chosen, not measured).
#: A contact link's tf origin this far from the held region is not plausibly the hand
#: occluding it; a larger margin lets any nearby arm pose excuse a shrunken re-fit.
_MAX_OCCLUDER_MARGIN_M = 0.10
#: Deeper than this, the column reaches whole furniture levels below the target (a
#: bench under a shelf) and the scan's "first ringed layer" stops meaning "under it".
_MAX_SUPPORT_SEARCH_BELOW_M = 0.5
#: The freeze holds a region with no fresh evidence; past a few kernel voxel deadlines
#: the map the region was vouched against is long superseded.
_MAX_FREEZE_GRID_AGES = 4.0


class GraspTargetTracker:
    """The leg's state machine — declaration, accepted region, freeze TTL. Pure.

    No ROS, no clock: every method takes the caller's ``now_ns`` (the node's ROS
    clock, the same domain as the runner's ``stamp_ns``). ``log`` receives one
    line per state transition, never one per tick.

    Args:
        freeze_s: How long past its ``stamp_ns`` an accepted region survives
            while the view is lost.
        log: Sink for transition lines.

    Example:
        >>> from openral_core import GraspDeclaration
        >>> lines: list[str] = []
        >>> t = GraspTargetTracker(freeze_s=2.0, log=lines.append)
        >>> t.on_declaration(
        ...     GraspDeclaration(
        ...         target_id="cell:box", contact_links=("finger",), timeout_s=10.0, stamp_ns=0
        ...     )
        ... )
        >>> t.envelope(now_ns=1).region is None
        True
        >>> t.on_declaration(t.envelope(now_ns=1).model_copy(update={"active": False}))
        >>> t.envelope(now_ns=2) is None
        True
    """

    def __init__(self, *, freeze_s: float, log: Callable[[str], None]) -> None:
        """Start with no declaration."""
        self._freeze_ns = int(freeze_s * 1e9)
        self._log = log
        self._declaration: GraspDeclaration | None = None
        self._region: PlaceRegion | None = None
        self._handed_over = False
        self._status = ""

    @property
    def declaration(self) -> GraspDeclaration | None:
        """Dispatch's live declaration (region-less), or ``None``."""
        return self._declaration

    @property
    def region(self) -> PlaceRegion | None:
        """The last accepted region still held, or ``None``."""
        return self._region

    def _transition(self, status: str, line: str) -> None:
        if status != self._status:
            self._status = status
            self._log(line)

    def _retract(self, kind: str, detail: str) -> None:
        had = self._region is not None
        self._region = None
        self._transition(
            f"retracted:{kind}",
            f"grasp target region retracted — {kind}: {detail}"
            if had
            else f"grasp target region refused — {kind}: {detail}",
        )

    def on_declaration(self, declaration: GraspDeclaration) -> None:
        """Fold in dispatch's copy; retraction or a new declaration clears the region."""
        current = self._declaration
        if not declaration.active:
            if current is not None:
                self._clear(f"dispatch retracted {current.target_id!r}")
            return
        if current is not None and (current.target_id, current.stamp_ns) == (
            declaration.target_id,
            declaration.stamp_ns,
        ):
            return  # the latched copy again
        self._region = None
        self._handed_over = False
        self._declaration = declaration.model_copy(update={"region": None})
        self._transition(
            f"declared:{declaration.target_id}:{declaration.stamp_ns}",
            f"grasp target declared {declaration.target_id!r} "
            f"search_box={'yes' if declaration.search_box is not None else 'none'} "
            f"rskill={declaration.rskill_id!r} trace={declaration.trace_id!r}",
        )

    def _clear(self, why: str) -> None:
        self._declaration = None
        self._region = None
        self._handed_over = False
        self._transition(f"cleared:{why}", f"grasp target cleared — {why}")

    def on_attach(self, contact_link: str) -> None:
        """A leg attached; if the declaration names its jaw link, stop re-measuring."""
        declaration = self._declaration
        if declaration is None or contact_link not in declaration.contact_links:
            return
        self._handed_over = True
        self._transition(
            "handed_over",
            f"grasp target {declaration.target_id!r}: ATTACH on {contact_link!r} — region "
            f"{'kept' if self._region is not None else 'absent'}, no further re-measurement",
        )

    def wants_measurement(self, *, now_ns: int) -> bool:
        """Whether a measurement tick should run: live, searchable, not handed over."""
        declaration = self._declaration
        return (
            declaration is not None
            and declaration.search_box is not None
            and not self._handed_over
            and declaration.is_live(now_ns=now_ns)
        )

    def accept(self, region: PlaceRegion) -> None:
        """Hold a region that passed every gate; a declaration-cap violation refuses it."""
        declaration = self._declaration
        if declaration is None:
            return
        try:
            GraspDeclaration.model_validate(declaration.model_dump() | {"region": region})
        except ValueError as exc:
            self._retract("declaration_bounds", str(exc).splitlines()[0])
            return
        self._region = region
        self._transition(
            "accepted",
            f"grasp target region accepted for {declaration.target_id!r}: "
            f"centre={tuple(round(v, 3) for v in region.pose.xyz)} "
            f"half_extents={tuple(round(v, 3) for v in region.half_extents)} "
            f"evidence={region.evidence_ref!r}",
        )

    def refuse(self, kind: str, detail: str, *, retract: bool, now_ns: int) -> None:
        """One failed measurement: retract now, or freeze the held region under its TTL."""
        if retract or self._region is None:
            self._retract(kind, detail)
            return
        self._transition(
            f"frozen:{kind}",
            f"grasp target view lost — {kind}: {detail}; holding the last region for at "
            f"most {self._freeze_ns / 1e9:.1f} s from its stamp",
        )
        self._expire_frozen(now_ns)

    def _expire_frozen(self, now_ns: int) -> None:
        region = self._region
        if region is None or self._handed_over:
            return
        if now_ns - region.stamp_ns > self._freeze_ns:
            self._retract(
                "freeze_ttl",
                f"last accepted region is {(now_ns - region.stamp_ns) / 1e9:.2f} s old "
                f"(> {self._freeze_ns / 1e9:.1f} s)",
            )

    def envelope(self, *, now_ns: int) -> GraspDeclaration | None:
        """The declaration to put on the attachment envelope now, or ``None``.

        Dispatch's fields verbatim plus the held region; expiry clears it and an
        over-age region is retracted here too, so a stalled measurement loop
        cannot keep one alive.
        """
        declaration = self._declaration
        if declaration is None:
            return None
        if not declaration.is_live(now_ns=now_ns):
            self._clear(f"{declaration.target_id!r} expired (timeout_s={declaration.timeout_s})")
            return None
        self._expire_frozen(now_ns)
        return declaration.model_copy(update={"region": self._region})


def lattice_from_msg(msg: Any) -> VoxelLattice:
    """Decode one ``openral_msgs/OccupancyVoxels`` into a ``VoxelLattice``.

    Raises:
        ROSConfigError: On a malformed grid (via ``VoxelLattice``).
    """
    o, q = msg.origin, msg.orientation
    return VoxelLattice(
        frame_id=str(msg.header.frame_id),
        origin=(float(o.x), float(o.y), float(o.z)),
        orientation_xyzw=(float(q.x), float(q.y), float(q.z), float(q.w)),
        resolution=float(msg.resolution),
        size=(int(msg.size_x), int(msg.size_y), int(msg.size_z)),
        occupancy=np.array(msg.occupancy, dtype=np.uint8),
    )


def search_column(search_box: PlaceRegion, *, below_m: float) -> PlaceRegion:
    """The column the support is measured in: the box, extended ``below_m`` downward.

    Raises:
        ROSConfigError: If the box is tilted — its footprint then names no
            vertical column to find a horizontal support in (HZ-01xx-6).

    Example:
        >>> from openral_core import PlaceRegion, Pose6D
        >>> box = PlaceRegion(
        ...     frame_id="base",
        ...     half_extents=(0.2, 0.2, 0.1),
        ...     pose=Pose6D(xyz=(0.4, 0.0, 0.1), quat_xyzw=(0, 0, 0.6, 0.8), frame_id="base"),
        ... )
        >>> col = search_column(box, below_m=0.1)
        >>> z, hz = col.pose.xyz[2], col.half_extents[2]
        >>> round(z - hz, 6), round(z + hz, 6)
        (-0.1, 0.2)
    """
    rot = homogeneous_from_quat_xyz((0.0, 0.0, 0.0), search_box.pose.quat_xyzw)[:3, :3]
    if abs(rot[2, 2] - 1.0) > _GRAVITY_TOL:
        raise ROSConfigError("grasp search_box must be gravity-aligned (yaw-only).")
    x, y, z = search_box.pose.xyz
    hx, hy, hz = search_box.half_extents
    return search_box.model_copy(
        update={
            "pose": search_box.pose.model_copy(update={"xyz": (x, y, z - below_m / 2.0)}),
            "half_extents": (hx, hy, hz + below_m / 2.0),
        }
    )


class _Refusal(Exception):  # noqa: N818 — internal control flow, never escapes this module
    """One typed measurement failure; ``retract`` picks the class (see module docstring)."""

    def __init__(self, kind: str, detail: str, *, retract: bool) -> None:
        super().__init__(f"{kind}: {detail}")
        self.kind, self.detail, self.retract = kind, detail, retract


def _lost(kind: str, detail: str) -> _Refusal:
    return _Refusal(kind, detail, retract=False)


def _contradicted(kind: str, detail: str) -> _Refusal:
    return _Refusal(kind, detail, retract=True)


def _hand_near(
    hands: Sequence[tuple[float, float, float]], region: PlaceRegion, *, reach_m: float
) -> bool:
    """Whether any hand point lies inside ``region`` grown by ``reach_m`` on every face."""
    if not hands:
        return False
    grown = region.model_copy(
        update={"half_extents": tuple(h + reach_m for h in region.half_extents)}
    )
    return bool(_in_region(np.asarray(hands, dtype=np.float64), grown).any())


def _gate_refit(
    grid: VoxelLattice,
    region: PlaceRegion,
    previous: PlaceRegion | None,
    *,
    min_cover: float,
    hands: Sequence[tuple[float, float, float]] = (),
    occluder_margin_m: float = 0.05,
) -> PlaceRegion:
    """Map cover + tracking gate on a fresh fit, or the typed refusal (lost vs contradicted).

    A re-fit the map does not cover is always a contradiction (the target is
    gone). A covered re-fit that fails tracking but shrank inside the held
    region (grown by one voxel) is the closing hand occluding part of the
    target — a lost view that keeps the held region under its freeze TTL —
    **only** while a declared contact link (``hands``, base-frame positions) is
    within one voxel + ``occluder_margin_m`` of the held region; otherwise
    nothing of the robot's explains the shrink (a person's hand, the target
    knocked over) and it is retracted.
    """
    count, covered = region_covers_occupied(grid, region, min_fraction=min_cover)
    tracked = previous is None or track_region(
        previous,
        region,
        max_centroid_shift_m=grid.resolution,
        extents_tol_m=grid.resolution,
    )
    if covered and tracked:
        return region
    if not covered:
        raise _contradicted("map_disagrees", f"only {count} occupied cells inside the region")
    assert previous is not None  # a first fit is always "tracked"
    moved = (
        f"re-fit centre {tuple(round(v, 3) for v in region.pose.xyz)} half_extents "
        f"{tuple(round(v, 3) for v in region.half_extents)} vs held centre "
        f"{tuple(round(v, 3) for v in previous.pose.xyz)} half_extents "
        f"{tuple(round(v, 3) for v in previous.half_extents)}"
    )
    if not region_within(region, previous, tol_m=grid.resolution):
        raise _contradicted("target_moved", f"{moved}: reaches outside the held region")
    reach = grid.resolution + occluder_margin_m
    if _hand_near(hands, previous, reach_m=reach):
        # The robot's own hand occluding part of the target: the held region
        # stays under its freeze TTL and is not replaced by the partial fit.
        raise _lost("occluded_refit", f"{moved}, inside the held region, hand within {reach:.3f} m")
    raise _contradicted(
        "unoccluded_refit",
        f"{moved}: inside the held region but no declared contact link within {reach:.3f} m "
        f"of it ({len(hands)} located)",
    )


#: What a request was made against: (depth, depth stamp ns, intrinsics,
#: T_base_from_cam, support_z, the declaration it measures).
_Snapshot = tuple[
    "NDArray[np.float64]", int, "IntrinsicsPinhole", "NDArray[np.float64]", float, GraspDeclaration
]


class GraspTargetLeg:
    """ROS wiring for ``GraspTargetTracker``, owned by ``VisionAttachmentBridge``.

    Reuses the bridge's depth + ``CameraInfo`` caches, tf2 buffer and
    ``SegmentInView`` client; owns its own subscriptions, timer and deadline.

    Args:
        node: The HAL lifecycle node.
        bridge: The owning bridge.
        config: The bridge's config (the ``grasp_target_*`` fields).
        support_search_below_m: How far below the search box's bottom face the
            support may lie, metres — the box is a lifted detection bbox whose
            min-z need not touch the table; at most 0.5 m, beyond which the column
            reaches whole furniture levels below the target. *Calibration point.*
        support_probe_margin_m: Outer reach, from the target's footprint, of the
            ring in which the support layer must hold top-surface cells, metres
            (``support_top_from_voxels``). *Calibration point.*
        occluder_margin_m: How far (beyond one voxel) from the held region a
            declared contact link's tf origin may be for a shrunk re-fit to count
            as the robot's own hand occluding the target (``_gate_refit``).
            *Calibration point* — it covers the link origin's offset from the
            finger surface; at most 0.10 m, beyond which any nearby arm pose
            would excuse a shrunken re-fit.

    Raises:
        ROSConfigError: On a rate outside 2-5 Hz; a non-positive cell count, cover
            fraction or probe margin; a freeze outside ``(0, 4 * grid_max_age_s]``,
            a search depth outside ``(0, 0.5]`` m or an occluder margin outside
            ``[0, 0.10]`` m (``grid_max_age_s`` itself is checked by the bridge).
    """

    def __init__(
        self,
        node: Any,
        bridge: VisionAttachmentBridge,
        config: VisionAttachmentConfig,
        *,
        support_search_below_m: float = 0.15,
        support_probe_margin_m: float = 0.05,
        occluder_margin_m: float = 0.05,
    ) -> None:
        """Validate the config; create no ROS entities yet."""
        if not _RATE_BAND_HZ[0] <= config.grasp_target_rate_hz <= _RATE_BAND_HZ[1]:
            raise ROSConfigError(
                f"grasp_target_rate_hz={config.grasp_target_rate_hz} is outside the design's "
                "2-5 Hz re-prompt band."
            )
        # Unset: twice the grid age the kernel itself accepts (its voxel deadline).
        freeze_s = (
            config.grasp_target_freeze_s
            if config.grasp_target_freeze_s is not None
            else 2.0 * config.grid_max_age_s
        )
        if not (
            0.0 < freeze_s <= _MAX_FREEZE_GRID_AGES * config.grid_max_age_s
            and config.grasp_target_min_cells > 0
            and 0.0 < config.grasp_target_min_cover <= 1.0
        ):
            raise ROSConfigError(
                f"grasp_target_freeze_s must be in (0, {_MAX_FREEZE_GRID_AGES:g} * "
                f"grid_max_age_s], got {freeze_s!r} with grid_max_age_s="
                f"{config.grid_max_age_s!r}; grasp_target_min_cells must be > 0 and "
                "grasp_target_min_cover in (0, 1]."
            )
        if not (
            0.0 < support_search_below_m <= _MAX_SUPPORT_SEARCH_BELOW_M
            and support_probe_margin_m > 0.0
            and 0.0 <= occluder_margin_m <= _MAX_OCCLUDER_MARGIN_M
        ):
            raise ROSConfigError(
                f"support_search_below_m must be in (0, {_MAX_SUPPORT_SEARCH_BELOW_M}], "
                f"support_probe_margin_m > 0 and occluder_margin_m in "
                f"[0, {_MAX_OCCLUDER_MARGIN_M}], got {support_search_below_m!r}, "
                f"{support_probe_margin_m!r} and {occluder_margin_m!r}."
            )
        self._occluder_margin_m = occluder_margin_m
        self._search_below_m = support_search_below_m
        self._probe_margin_m = support_probe_margin_m
        self._node = node
        self._bridge = bridge
        self._config = config
        self._freeze_s = freeze_s
        self.tracker = GraspTargetTracker(
            freeze_s=freeze_s,
            log=lambda line: node.get_logger().info(line),
        )
        # (lattice, monotonic receive time) of the newest grid.
        self._grid: tuple[VoxelLattice, float] | None = None
        self._subs: list[Any] = []
        self._timer: Any = None
        self._inflight: Any = None
        self._deadline_timer: Any = None

    def setup(self) -> None:
        """Subscribe the declaration and the voxel grid; start the measurement timer."""
        from openral_msgs.msg import GraspDeclaration as GraspDeclarationMsg
        from openral_msgs.msg import OccupancyVoxels
        from rclpy.qos import (
            QoSDurabilityPolicy,
            QoSHistoryPolicy,
            QoSProfile,
            QoSReliabilityPolicy,
        )

        declaration_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )
        # The safety kernel's own profile for this topic.
        voxel_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        declaration_sub = self._node.create_subscription(
            GraspDeclarationMsg,
            "/openral/grasp_declaration",
            self._on_declaration,
            declaration_qos,
        )
        voxel_sub = self._node.create_subscription(
            OccupancyVoxels, "/openral/world_voxels", self._on_voxels, voxel_qos
        )
        self._subs = [declaration_sub, voxel_sub]
        self._timer = self._node.create_timer(1.0 / self._config.grasp_target_rate_hz, self._tick)
        self._node.get_logger().info(
            f"grasp target leg: rate={self._config.grasp_target_rate_hz:.1f}Hz "
            f"freeze={self._freeze_s:.2f}s grid_max_age={self._config.grid_max_age_s:.2f}s "
            f"min_cells={self._config.grasp_target_min_cells} "
            f"min_cover={self._config.grasp_target_min_cover:.2f} "
            f"support_search_below={self._search_below_m:.2f}m "
            f"support_probe_margin={self._probe_margin_m:.2f}m "
            f"occluder_margin={self._occluder_margin_m:.2f}m "
            f"deadline={self._config.deadline_s:.3f}s"
        )

    def teardown(self) -> None:
        """Destroy every ROS entity; idempotent."""
        self._cancel_deadline()
        self._inflight = None
        if self._timer is not None:
            self._timer.cancel()
            self._node.destroy_timer(self._timer)
            self._timer = None
        for sub in self._subs:
            self._node.destroy_subscription(sub)
        self._subs = []

    # ── envelope ─────────────────────────────────────────────────────────────

    def fill(self, msg: Any, *, now_ns: int) -> None:
        """Put the declaration in force (region and all) on one ``AttachmentState``."""
        declaration = self.tracker.envelope(now_ns=now_ns)
        msg.grasp_declaration_valid = declaration is not None
        if declaration is not None:
            declaration.fill_idl(msg.grasp_declaration)

    # ── inputs ───────────────────────────────────────────────────────────────

    def _on_declaration(self, msg: Any) -> None:
        try:
            declaration = GraspDeclaration.from_idl(msg)
        except (ValueError, ROSConfigError) as exc:
            # A malformed declaration exempts nothing; whatever was held dies.
            self._node.get_logger().error(f"grasp target: declaration rejected — {exc}")
            held = self.tracker.declaration
            if held is not None:
                self.tracker.on_declaration(held.model_copy(update={"active": False}))
            return
        self.tracker.on_declaration(declaration)

    def _on_voxels(self, msg: Any) -> None:
        try:
            self._grid = (lattice_from_msg(msg), time.monotonic())
        except ROSConfigError as exc:
            self._grid = None
            self._node.get_logger().warning(f"grasp target: voxel grid dropped — {exc}")

    def _fresh_grid(self) -> VoxelLattice:
        if self._grid is None:
            raise _lost("no_grid", "no /openral/world_voxels grid yet")
        lattice, received = self._grid
        age = time.monotonic() - received
        if age > self._config.grid_max_age_s:
            raise _lost("grid_stale", f"newest voxel grid is {age:.2f} s old")
        return lattice

    def _now_ns(self) -> int:
        return int(self._node.get_clock().now().nanoseconds)

    # ── measurement ──────────────────────────────────────────────────────────

    def _tick(self) -> None:
        now_ns = self._now_ns()
        if self._inflight is not None or not self.tracker.wants_measurement(now_ns=now_ns):
            return
        try:
            try:
                self._request(now_ns)
            except ROSConfigError as exc:  # an input shape the geometry refuses
                raise _contradicted("rejected_inputs", str(exc)) from exc
        except _Refusal as refusal:
            self.tracker.refuse(
                refusal.kind, refusal.detail, retract=refusal.retract, now_ns=now_ns
            )

    def _seed(
        self, grid: VoxelLattice, box: PlaceRegion
    ) -> tuple[tuple[float, float, float], float]:
        """Measure the support under the box, then seed the one target above it — or refuse."""
        try:
            column = search_column(box, below_m=self._search_below_m)
        except ROSConfigError as exc:
            raise _contradicted("search_box_tilted", str(exc)) from exc
        if self._probe_margin_m < 2.0 * grid.resolution - 1e-9:
            # A config/grid mismatch, not evidence about the target: a lost view.
            raise _lost(
                "probe_margin_under_two_cells",
                f"support_probe_margin_m={self._probe_margin_m:.3f} m is under two cells of "
                f"the {grid.resolution:.3f} m grid; the support ring would be empty",
            )
        min_cells = self._config.grasp_target_min_cells
        column_centers = occupied_centers_in_box(grid, column)
        near_xy = (box.pose.xyz[0], box.pose.xyz[1])
        # The ring lies outside the target's footprint, mostly outside a search box
        # padded tightly around it: count it on the column grown to hold all of it.
        grow = self._probe_margin_m + grid.resolution
        hx, hy, hz = column.half_extents
        around = column.model_copy(update={"half_extents": (hx + grow, hy + grow, hz)})
        support_z = support_top_from_voxels(
            grid,
            column_centers,
            near_xy=near_xy,
            min_cells=min_cells,
            probe_margin_m=self._probe_margin_m,
            surface_centers=occupied_centers_in_box(grid, around),
        )
        if support_z is None:
            raise _contradicted(
                "no_support",
                f"no layer in the column under the search box (down to "
                f"{self._search_below_m:.2f} m below it) holds >= {min_cells} surface cells "
                f"within {self._probe_margin_m:.2f} m around the target's footprint",
            )
        # Seeded from the whole column, so a lifted box bottom cannot hide the
        # target's lower cells from the contact check below.
        seed = target_seed_from_voxels(
            grid, column_centers, near_xy=near_xy, support_z=support_z, min_cells=min_cells
        )
        if seed.point is None:
            assert seed.refusal is not None
            detail = f"cluster sizes {list(seed.cluster_sizes)}"
            if seed.refusal is TargetRefusal.AMBIGUOUS:
                raise _contradicted(seed.refusal.value, detail)
            raise _lost(seed.refusal.value, detail)
        # The target must stand on the measured support: its lowest kept cell sits
        # one voxel up (the seed drops the layer touching the plane), plus one voxel
        # of tolerance. A lower surface — a bench under the shelf board the target
        # is on — would stand the region on nothing and exempt the gap (HZ-01xx-6).
        assert seed.bottom_z is not None
        if seed.bottom_z - support_z > 2.0 * grid.resolution + 1e-9:
            raise _contradicted(
                "not_on_support",
                f"target's lowest cell at z={seed.bottom_z:.3f} is "
                f"{seed.bottom_z - support_z:.3f} m above the measured support z={support_z:.3f} "
                f"(> {2.0 * grid.resolution:.3f} m)",
            )
        return seed.point, support_z

    def _request(self, now_ns: int) -> None:
        """Seed, project, and send one bounded ``SegmentInView`` request — or refuse."""
        from openral_hal.vision_attachment_bridge import build_segment_request

        declaration = self.tracker.declaration
        assert declaration is not None and declaration.search_box is not None
        box = declaration.search_box
        grid = self._fresh_grid()
        if box.frame_id != grid.frame_id:
            raise _contradicted(
                "frame_mismatch", f"search box in {box.frame_id!r}, grid in {grid.frame_id!r}"
            )
        seed_point, support_z = self._seed(grid, box)

        bridge = self._bridge
        depth_entry = bridge._depth
        if depth_entry is None:
            raise _lost("no_depth", "no depth frame yet")
        depth, depth_stamp_ns, optical = depth_entry
        if not optical:
            raise _lost("no_depth", "depth frame has no header.frame_id")
        # The region is stamped with the frame it was measured on, never later
        # than now (a clock-skewed future stamp would be refused downstream), and
        # a frame already past the freeze TTL could never be held.
        depth_stamp_ns = min(depth_stamp_ns, now_ns)
        if now_ns - depth_stamp_ns > int(self._freeze_s * 1e9):
            raise _lost(
                "depth_stale", f"newest depth frame is {(now_ns - depth_stamp_ns) / 1e9:.2f} s old"
            )
        aligned = bridge._align_to_depth([], depth)
        if isinstance(aligned, str):
            raise _lost("no_intrinsics", aligned)
        intrinsics = aligned[1]
        t_base_from_cam = bridge._lookup(grid.frame_id, optical)
        if t_base_from_cam is None:
            raise _lost("no_tf", f"no tf2 {grid.frame_id} <- {optical}")
        t_cam_from_base = np.asarray(np.linalg.inv(t_base_from_cam), dtype=np.float64)
        if project_point(seed_point, t_cam_from_base, intrinsics) is None:
            raise _lost("seed_off_image", f"seed {seed_point} does not project into {optical!r}")
        client = bridge._client
        if client is None or not client.service_is_ready():
            raise _lost("segmenter_unavailable", f"no server on {self._config.service_name!r}")

        request = build_segment_request(
            stamp_ns=depth_stamp_ns,
            camera=bridge._camera,
            t_link_from_cam=t_base_from_cam,
            tcp_in_link=seed_point,
        )
        snapshot: _Snapshot = (
            depth,
            depth_stamp_ns,
            intrinsics,
            t_base_from_cam,
            support_z,
            declaration,
        )
        future = client.call_async(request)
        self._inflight = future
        future.add_done_callback(lambda fut: self._on_reply(fut, snapshot))
        self._deadline_timer = self._node.create_timer(self._config.deadline_s, self._on_deadline)

    def _on_deadline(self) -> None:
        self._cancel_deadline()
        inflight, self._inflight = self._inflight, None
        if inflight is None:
            return
        inflight.cancel()  # runs _on_reply, which sees it is no longer in flight
        self.tracker.refuse(
            "deadline",
            f"SegmentInView did not answer within {self._config.deadline_s:.3f} s",
            retract=False,
            now_ns=self._now_ns(),
        )

    def _cancel_deadline(self) -> None:
        if self._deadline_timer is not None:
            self._deadline_timer.cancel()
            self._node.destroy_timer(self._deadline_timer)
            self._deadline_timer = None

    def _on_reply(self, future: Any, snapshot: _Snapshot) -> None:
        if future is not self._inflight:
            return  # resolved by the deadline already
        self._cancel_deadline()
        self._inflight = None
        now_ns = self._now_ns()
        try:
            try:
                region = self._measure(future, snapshot)
            except ROSConfigError as exc:  # an input shape the geometry refuses
                raise _contradicted("rejected_inputs", str(exc)) from exc
        except _Refusal as refusal:
            self.tracker.refuse(
                refusal.kind, refusal.detail, retract=refusal.retract, now_ns=now_ns
            )
            return
        if snapshot[5] is not self.tracker.declaration:
            return  # the declaration changed or died while the request was in flight
        self.tracker.accept(region)

    def _measure(self, future: Any, snapshot: _Snapshot) -> PlaceRegion:
        """Reply → region → map cover → tracking gate, or a typed refusal."""
        from openral_hal.vision_attachment_bridge import (
            decode_mono8_mask,
            mask_depth_skew_reason,
            mask_stamps_ns,
            resolve_segment_outcome,
        )

        depth, depth_stamp_ns, _intrinsics, t_base_from_cam, support_z, declaration = snapshot
        try:
            response = future.result()
        except Exception as exc:  # a service exception must degrade, not propagate
            raise _lost("segmenter_failed", f"{type(exc).__name__}: {exc}") from exc
        if response is None:
            raise _lost("segmenter_failed", "future resolved with no response")
        outcome = resolve_segment_outcome(
            timed_out=False,
            ok=bool(response.ok),
            failure_reason=str(response.failure_reason),
            mask_count=len(response.masks),
        )
        if not outcome.use_masks:
            raise _lost("no_mask", outcome.reason)
        skew = mask_depth_skew_reason(
            mask_stamps_ns(response.masks),
            depth_stamp_ns,
            max_skew_s=self._config.mask_depth_max_skew_s,
        )
        if skew:
            raise _lost("mask_depth_skew", skew)
        try:
            masks = [
                decode_mono8_mask(bytes(image.data), height=image.height, width=image.width)
                for image in response.masks
            ]
        except ROSConfigError as exc:
            raise _contradicted("mask_malformed", str(exc)) from exc
        aligned = self._bridge._align_to_depth(masks, depth)
        if isinstance(aligned, str):
            raise _contradicted("mask_misaligned", aligned)
        masks, intrinsics = aligned
        grid = self._fresh_grid()
        model = declaration.rskill_id or self._config.service_name
        fit = None
        for mask in masks:  # candidates in the segmenter's order; first one that fits
            fit = target_region_from_mask(
                mask,
                depth,
                intrinsics,
                t_base_from_cam,
                support_z=support_z,
                resolution=grid.resolution,
                frame_id=grid.frame_id,
                evidence_ref=f"segment_in_view:{model}@{depth_stamp_ns}",
                stamp_ns=depth_stamp_ns,
            )
            if fit.region is not None:
                break
        assert fit is not None
        if fit.region is None:
            assert fit.refusal is not None
            detail = (
                f"points={fit.point_count} depth_valid={fit.depth_valid_fraction:.2f} "
                f"half_extents={tuple(round(v, 3) for v in fit.half_extents)}"
            )
            if fit.refusal is TargetRefusal.TOO_FEW_POINTS:
                raise _lost(fit.refusal.value, detail)
            raise _contradicted(fit.refusal.value, detail)
        bridge = self._bridge
        hands = []
        for link in declaration.contact_links:
            t_base_from_link = bridge._lookup(grid.frame_id, bridge.tf_frame(link))
            if t_base_from_link is not None:
                x, y, z = (float(v) for v in t_base_from_link[:3, 3])
                hands.append((x, y, z))
        return _gate_refit(
            grid,
            fit.region,
            self.tracker.region,
            min_cover=self._config.grasp_target_min_cover,
            hands=hands,
            occluder_margin_m=self._occluder_margin_m,
        )
