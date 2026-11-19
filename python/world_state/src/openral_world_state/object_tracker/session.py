"""``TrackerSession`` — the ROS-free glue the object-tracker lifecycle node drives (issue #418).

The node (``openral_world_state_ros.object_tracker_node``) decodes ROS messages and owns the
wire; everything it decides lives here so it runs without ROS:

* **Goals.** A live ``GraspDeclaration`` with ``target_queries`` (a ``PlaceDeclaration`` with
  a ``destination_query``) opens one standing query per entry, keyed by the declaration
  (``"<target_id>@<stamp_ns>"``, the kernel's ``declaration_key``); a retraction or a new
  declaration closes the goal. Tracks persist across goals: matching live tracks are re-keyed
  both when a goal opens and after fits, because the first fit may have been requested by an
  older standing query. Matching uses the same label and role contract as query grounding;
  competing declaration scopes leave the track unowned.
* **Grounding wait.** ``grounding_ready`` / ``grounding_incomplete`` are logged once per
  goal, in source time, after ``grounding_wait_s``. Refusing the goal under
  ``grounding_required`` is the runner's (it is logged at error level here).
* **Binding** (``binding.GraspBinder`` per gripper): ``ATTACH`` binds what the pinch volume
  meets; ``REGRASP`` grows it; ``DETACH`` — or the jaw-open rule (``JawReleaseMonitor``,
  the kernel's K5 mirrored) — releases it (``release.release_track``). An ambiguous
  (``PINCH_FALLBACK``) union drops every candidate that a post-lift re-fit shows stayed
  behind (re-fitted after the ATTACH, its box no longer meeting the pinch volume). An
  untracked grasp re-opens the live grasp goal's queries (the re-ground after lift).

A RELEASED track is never followed (the released box stayed where the object was let go
while it fell or rolled, and the next approach stopped on it): it lives out the release window, so
the kernel can freeze the payload, and is then retired from the store. The object is plain world
from then on, until the target query finds it afresh under a new id. Safety-WG review owed.

Clocks: every stamp is source time (depth, grid, joint-state, grasp-event stamps); the
segmenter rate runs on the ``wall_s`` the node feeds.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np
import structlog
from numpy.typing import NDArray
from openral_core import (
    AttachedCollisionObject,
    EmbodimentKind,
    GraspDeclaration,
    ObjectQuery,
    ObjectRole,
    PlaceDeclaration,
    PlaceRegion,
    PlaceZone,
    Pose6D,
    RobotDescription,
    TrackedObject,
    TrackedObjectSet,
    TrackStatus,
)
from openral_core.exceptions import ROSConfigError
from openral_core.geometry import (
    freeze_released_attachment,
    homogeneous_from_quat_xyz,
    rebase_box,
    rotation_to_quat_wxyz,
)
from openral_core.kinematics import Box, box_gap_lower_bound_m
from openral_core.perception_geometry import (
    WORKSPACE_MIN_COVER,
    VoxelLattice,
    region_covers_occupied,
)
from openral_core.tracking_thresholds import load_tracking_thresholds

from openral_world_state.object_tracker.binding import (
    Binding,
    BindOutcome,
    GraspBinder,
    _fresh_ns,  # reason: same package; the freshness rule a late bind must share
    _pinch_holds,  # reason: same package; the candidate rule a late bind must share
)
from openral_world_state.object_tracker.cover import Cell, hand_touches
from openral_world_state.object_tracker.fit import ContainingFit
from openral_world_state.object_tracker.grounding import centre_within, in_scope
from openral_world_state.object_tracker.pinch import HandModel
from openral_world_state.object_tracker.release import JawReleaseMonitor, release_track
from openral_world_state.object_tracker.stall import JawStallDetector
from openral_world_state.object_tracker.tracker import (
    CameraFrame,
    ObjectTracker,
    TrackerOutput,
    TrackerParams,
    TrackerResult,
)

__all__ = [
    "RELEASED_CLEAR_TIMEOUT_S",
    "RELEASE_DWELL_S",
    "GraspEventKind",
    "TrackerSession",
    "TrackingMode",
    "declaration_key",
]

#: Instance cells apply only on the tracker's own lattice (origin / resolution to 1 um).
_LATTICE_TOL_M = 1e-6

log = structlog.get_logger(__name__)

_NS = 1_000_000_000
_R = load_tracking_thresholds().release
_L = load_tracking_thresholds().late_bind

#: The jaw-open release dwell (source seconds) the kernel (K5) and the tracker share; the
#: deploy launch passes it to both. *Calibration point.*
RELEASE_DWELL_S = _R.dwell_s
#: Default released-binding lifetime (source seconds); MUST equal the kernel's
#: ``tracked_release_timeout_s`` and the voxel bridge's ``released_clear_timeout_s``.
RELEASED_CLEAR_TIMEOUT_S = _R.clear_timeout_s
#: A late ATTACH (the object may already be moving) is bound at the hand pose of the hold when
#: the hold began at least this long before the event (0: every ATTACH with a hold read takes
#: this path, so a prompt one gets the grasp-pose lookback too; a 0.067 s
#: ATTACH fell to the plain bind after the lift and bound the pinch fallback) ...
_LATE_MIN_NS = round(_L.min_s * _NS)
#: ... and at most this long before it, ...
_LATE_MAX_NS = round(_L.max_s * _NS)
#: ... the hand has travelled no further than this since the hold (m; an ordinary
#: 15 cm lift before a late ATTACH skipped the late bind at 0.12 and bound the
#: pinch fallback; restock scene 08 2026-11-01 lifted 21 cm in 1 s while the jaw still crept
#: closed on the carton, the stall read fired after it and 0.18 bound the pinch fallback; a
#: wrong late bind is a phantom payload, released by `_check_drop`). *Calibration point.* ...
_LATE_MAX_TRAVEL_M = _L.max_travel_m
#: ... (the grasp pose: the hold's, or the newest this long before it with the pinch in a box)
_LATE_LOOKBACK_NS = round(_L.lookback_s * _NS)
#: ... and the track was last fitted within this long of the hold start.
_LATE_CONFIRM_NS = round(_L.confirm_s * _NS)
#: Every hand's pinch box this far (m, box to box) from every target box lets the completion
#: check ask its judge: off the object. The release window's clearance (centre distance less the
#: pinch box's half diagonal, 9.8 cm on the Robotiq) put the hand 18 cm away (the
#: banana lay placed and released 6 s of sim time, the judge never asked).
_JUDGE_CLEAR_M = _R.judge_clear_m
#: Every hand collision box this far (m) from a released box ends its release window early:
#: the kernel's own FK separation (``tracked_release_clear_m``, 0.04), past which it holds the
#: frozen box as a plain obstacle. Retiring later left that box blocking the regrasp of an
#: object never lifted (the window closed by separation at 7.9 s, the release stayed on the
#: wire, the jaw met the frozen box at 9.0 s).
_RELEASE_CLEAR_M = _R.clear_m
#: Consecutive in-hand fits away from every hand box that read as a drop (`_check_drop`).
_DROP_FITS = _R.drop_fits
#: A track a late ATTACH may revive and bind (once it has confirmed).
_LATE_STATUSES = frozenset(
    {TrackStatus.CONFIRMED, TrackStatus.OCCLUDED, TrackStatus.SPLIT, TrackStatus.LOST}
)
#: A joint sample further than this from a stamp is not the pose then.
_JOINT_SAMPLE_MAX_AGE_NS = round(_R.joint_sample_max_age_s * _NS)


#: A source stamp this far behind the newest one is a reset clock, not a late message.
_CLOCK_RESET_NS = 1_000_000_000

#: Kinds whose base cannot carry a base-frame zone away from the furniture (the deploy's
#: ``_check_place_zone_frames`` rule, enforced here too: a launch arg bypasses the CLI).
_FIXED_BASE_KINDS = frozenset({EmbodimentKind.MANIPULATOR, EmbodimentKind.BIMANUAL})


def zone_candidates_target_id(zones: Sequence[PlaceZone]) -> str:
    """The place ``target_id`` naming a candidate set of zones: ``zone:<a>|<b>|...``.

    Station mode: a prompt ("put box x1 into lower slot 1") names the same slot declared
    at several stations (``s1_lower_slot_1`` ... ``s6_lower_slot_1``, each with the alias
    ``lower_slot_1``, all in the map frame). The rSkill runner, which has the prompt but no
    tf2, names every match; the object tracker, which has tf2 (``zone_transform``) and the
    robot's reach, keeps the one in reach at the base's current pose
    (``TrackerSession._zone_region``). Sorted, so the same matches always give the same id.

    Example:
        >>> from openral_core import PlaceRegion, Pose6D
        >>> box = PlaceRegion(
        ...     frame_id="map",
        ...     pose=Pose6D(xyz=(1.0, 0.0, 0.5), quat_xyzw=(0, 0, 0, 1), frame_id="map"),
        ...     half_extents=(0.1, 0.1, 0.1),
        ... )
        >>> zs = [PlaceZone(name=n, region=box) for n in ("s2_lower_slot_1", "s1_lower_slot_1")]
        >>> zone_candidates_target_id(zs)
        'zone:s1_lower_slot_1|s2_lower_slot_1'
        >>> zone_candidate_names(zone_candidates_target_id(zs))
        ['s1_lower_slot_1', 's2_lower_slot_1']
        >>> zone_candidate_names("zone:upper_slot_2") is None
        True
    """
    return PlaceZone.TARGET_PREFIX + ZONE_CANDIDATE_SEP.join(sorted(z.name for z in zones))


def zone_candidate_names(target_id: str) -> list[str] | None:
    """The zone names of a candidate-set ``target_id``; ``None`` for anything else."""
    if not target_id.startswith(PlaceZone.TARGET_PREFIX):
        return None
    names = target_id[len(PlaceZone.TARGET_PREFIX) :].split(ZONE_CANDIDATE_SEP)
    return names if len(names) > 1 else None


#: Separates the zone names of a candidate-set place ``target_id`` (a zone name never holds it).
ZONE_CANDIDATE_SEP = "|"


@dataclass(frozen=True)
class _LateAttachChoice:
    """Late-bind pose and tracks that must not fall through to ordinary ATTACH binding."""

    positions: dict[str, float] | None = None
    excluded_track_ids: frozenset[str] = frozenset()


class TrackingMode(StrEnum):
    """``DeployRuntime.object_tracking`` (``off`` never starts the node).

    ``tracking`` (open): the tracker tracks, grounds and binds a carried payload; every contact
    allowance it publishes comes from a scene declaration (``search_box``, ``workspace_box``, a
    ``PlaceZone``), a track only confirming one is populated, never shaping it. Its one
    track-driven effect is on the map: a payload uniquely bound to a hand is cleared from it by
    identity (2026-11-02), and a released one is frozen and declared (``released_by_link``) so
    the kernel holds it in place of the cleared cells. ``on`` (needs the OpenRAL Pro exemption
    provider): the allowance is track-driven (the provider keys on track ids; a destination
    track is the place region) and the provider decides the release inside the kernel.
    """

    TRACKING = "tracking"
    ON = "on"


class GraspEventKind(StrEnum):
    """``openral_msgs/GraspEvent.kind``."""

    ATTACH = "attach"
    REGRASP = "regrasp"
    DETACH = "detach"


@dataclass
class _Goal:
    goal_id: str
    key: str
    start_ns: int | None  # None until a source stamp exists to start the search at
    queries: tuple[ObjectQuery, ...]
    invocation_id: str = ""
    ready_logged: bool = False
    incomplete_logged: bool = False
    #: The grasp declaration's ``search_box`` (grid frame): its TARGET queries' instances
    #: are the tracks centred in it (``StandingQuery.within``).
    within: PlaceRegion | None = None
    #: The TARGET query's ``seed`` when it names one instance (a memory node's box, often in
    #: ``map``; issue #631): re-expressed in the grid frame on every grid into ``within``, which
    #: it overrides. Unresolvable (no tf2) leaves ``within`` in the seed's frame, which no track
    #: is centred in: the query fails closed instead of binding a same-label neighbour.
    scope: PlaceRegion | None = None
    scope_resolved: bool | None = None  # last logged resolve state of ``scope``
    #: Set when ``scope`` is in another frame than the grid: the ids of the tracks measured
    #: inside it (``TrackerSession._measure_scope``), shared with the goal's TARGET queries
    #: (``StandingQuery.members``). They decide the match; ``within`` only crops the search.
    members: set[str] | None = None
    #: ``object_id -> ((box, fit stamp), inside)``: what ``members`` was last judged from.
    measured: dict[str, tuple[tuple[PlaceRegion, int], bool]] = field(default_factory=dict)

    def owns(self, track: TrackedObject) -> bool:
        """Whether ``track`` is an instance of one of this goal's queries."""
        return any(
            track.label == q.label
            and track.role is q.role
            and (q.role is not ObjectRole.TARGET or in_scope(track, self.within, self.members))
            for q in self.queries
        )


def _box(region: PlaceRegion) -> Box:
    t = homogeneous_from_quat_xyz(region.pose.xyz, region.pose.quat_xyzw)
    return (t[:3, 3], t[:3, :3], np.asarray(region.half_extents, dtype=np.float64))


def _over(region: PlaceRegion, xyz: Sequence[float]) -> bool:
    """Whether ``xyz`` lies over or under ``region``: inside its box's footprint, any height.

    Example:
        >>> from openral_core import Pose6D
        >>> p = Pose6D(xyz=(0.0, 0.0, 0.0), quat_xyzw=(0, 0, 0, 1), frame_id="b")
        >>> bowl = PlaceRegion(frame_id="b", pose=p, half_extents=(0.1, 0.1, 0.05))
        >>> _over(bowl, (0.05, -0.05, 0.4)), _over(bowl, (0.2, 0.0, 0.0))
        (True, False)
    """
    centre, rot, half = _box(region)
    local = rot.T @ (np.asarray(xyz, dtype=np.float64) - centre)
    return bool(np.all(np.abs(local[:2]) <= half[:2]))


def declaration_key(target_id: str, stamp_ns: int) -> str:
    """The kernel's ``declaration_key``: ``"<target_id>@<stamp_ns>"``.

    Example:
        >>> declaration_key("orange", 7)
        'orange@7'
    """
    return f"{target_id}@{stamp_ns}"


class TrackerSession:
    """One deploy's tracker, binders and goals (module docstring).

    Args:
        description: The robot manifest; FK roots at its ``base_frame``, the tracks' frame.
        cell_m: The voxel grid's resolution (the pinch volume is grown by one cell).
        grounding_wait_s: ``DeployRuntime.grounding_wait_s``.
        grounding_required: ``DeployRuntime.grounding_required``.
        release_dwell_s: The jaw-open rule's dwell (source seconds). *Calibration point.*
        released_clear_timeout_s: How long (source seconds) a released ``TRACKED_OBJECT``
            binding stays on ``attachment_state`` after release (frozen at the let-go pose:
            declared ``released_by_link`` in ``tracking``, released by the exemption provider
            in ``on``; the voxel bridge clears its box); the RELEASED track is
            retired with it. MUST equal the kernel's ``tracked_release_timeout_s`` and the
            voxel bridge's ``released_clear_timeout_s`` (all default 3.0 s).
        params: Tracker calibration.
        derive_attach: The tracker reads a stalled jaw itself (``JawStallDetector``: commanded
            jaw vs the median measured jaw, the manifest's ``stall_gap``) and binds as an ATTACH
            does, so a hold the HAL's position-stall trigger misses (a chattering jaw, a command
            past its band) is still bound. ``tracking`` / ``on`` on a sim HAL only; the HAL trigger
            and the legacy / real path are unchanged, and a real ATTACH for the same close is
            never doubled.
        wrist_links: How many links behind each carrier are the hand's wrist
            (``HandModel``), the kernel's ``DeployRuntime.workspace_contact_wrist_links``.
        require_hands: Refuse a gripper the binder cannot model (the node sets it in every
            mode); without it such a hand is logged and skipped.
        mode: ``tracking`` (default, the safer one) publishes only declared grasp / place
            regions; ``on`` keeps the track-driven place region and publishes no grasp region
            (``grasp_region`` / ``place_region``).

    Raises:
        ROSConfigError: ``require_hands`` and a gripper joint cannot be modelled.

    Example:
        >>> d = RobotDescription.from_yaml("robots/so101_follower/robot.yaml")
        >>> s = TrackerSession(d, cell_m=0.015, grounding_wait_s=5.0)
        >>> sorted(s.binders), s.snapshot().frame_id
        (['gripper'], 'base')
    """

    def __init__(
        self,
        description: RobotDescription,
        *,
        cell_m: float,
        grounding_wait_s: float,
        grounding_required: bool = False,
        release_dwell_s: float = RELEASE_DWELL_S,
        released_clear_timeout_s: float = RELEASED_CLEAR_TIMEOUT_S,
        params: TrackerParams | None = None,
        require_hands: bool = False,
        derive_attach: bool = False,
        wrist_links: int = 1,
        mode: TrackingMode = TrackingMode.TRACKING,
    ) -> None:
        """Model every gripper and start with no tracks and no goals."""
        self.description = description
        self.mode = mode
        self.frame_id = description.base_frame
        self.tracker = ObjectTracker(self.frame_id, params)
        self.grounding_wait_ns = int(grounding_wait_s * _NS)
        self.grounding_required = grounding_required
        self.release_dwell_s = release_dwell_s
        self.released_clear_ns = int(released_clear_timeout_s * _NS)
        # Released TRACKED_OBJECT bindings still on the wire: object_id -> (record, stamp).
        self._released: dict[str, tuple[AttachedCollisionObject, int]] = {}
        # EXPERIMENT exp/vision-voxel-fusion: the propagator's latest per-track labelled cells
        # (stamp, frame, lattice origin, resolution, cells), applied before the next feed.
        self._instance_cells: (
            tuple[int, str, tuple[float, float, float], float, dict[str, frozenset[Cell]]] | None
        ) = None
        self._known_at_release: dict[str, frozenset[str]] = {}  # the track ids then
        self._off_hand: dict[str, int] = {}  # consecutive in-hand fits off every hand box
        self.binders: dict[str, GraspBinder] = {}
        self._stall: dict[str, JawStallDetector] = {}
        for joint in description.joints:
            if joint.role != "gripper":
                continue
            try:
                hand = HandModel(description, joint.name, wrist_links)
            except ROSConfigError as exc:
                if require_hands:
                    raise
                log.warning("object_tracker.hand_unbindable", joint=joint.name, error=str(exc))
                continue
            self.binders[joint.name] = GraspBinder(
                hand,
                frame_id=self.frame_id,
                cell_m=cell_m,
                max_track_fit_age_s=self.tracker.params.max_age_s,
            )
            cal = joint.closure_calibration
            if derive_attach and cal is not None:
                self._stall[joint.name] = JawStallDetector(
                    cal.closed_position,
                    cal.closed_rest_offset + cal.stall_gap,
                    cal.settle_tolerance,
                )
        self._joints = frozenset(j.name for j in description.joints)
        # (stamp_ns, complete joint state): the hand pose at the start of a hold, for a late ATTACH.
        self._joint_history: deque[tuple[int, dict[str, float]]] = deque(
            maxlen=load_tracking_thresholds().late_bind.joint_history_len
        )
        self.positions: dict[str, float] = {}
        self.grid: VoxelLattice | None = None
        self.grid_stamp_ns = 0
        self.grasp_declaration: GraspDeclaration | None = None
        self.place_declaration: PlaceDeclaration | None = None
        # The station's declared place zones by ``PlaceZone.target_id`` (``place_region``).
        self._place_zones: dict[str, PlaceZone] = {}
        # Frames a zone may be declared in (``PlaceZone``): the robot's map / odom frame, and
        # its base frame on a fixed-base kind only (a moving base carries a base-frame box
        # away from the furniture). Any other zone is refused (``zone_frame_refusal``).
        self.zone_frames = frozenset(
            {description.map_frame, description.odom_frame}
            | ({self.frame_id} if description.embodiment_kind in _FIXED_BASE_KINDS else set())
        )
        # tf2 ``grid <- frame`` at a stamp, or ``None`` (the node's lookup): re-expresses a zone
        # declared in a fixed frame (``map`` / ``odom``) in the grid frame. Unset: none resolve.
        self._zone_transform: Callable[[str, int], NDArray[np.float64] | None] | None = None
        self._zone_state: dict[str, str] = {}  # zone name -> last resolve state (log on change)
        # ``(target_id, grid_stamp_ns, region)``: ``_zone_region`` once per grid, not per publish.
        self._zone_cache: tuple[str, int, PlaceRegion] | None = None
        # Station-scoped zone choice (``zone_candidates_target_id``): the robot's manipulation
        # reach in the grid frame, ``(centre, radius_m)`` -- the world map's coverage ball the
        # launch measures from the kernel's collision model (``_coverage_ball``: the moving
        # links' reach plus a margin). ``None``: no candidate set ever resolves.
        self.zone_reach: tuple[tuple[float, float, float], float] | None = None
        self._zone_choice_log: tuple[str, ...] = ()
        self.attachment_changes = 0  # bumped whenever the held set changes
        self._monitors: dict[str, JawReleaseMonitor] = {}
        self._attach_ns: dict[str, int] = {}
        # Per gripper: the tracker's own stalled-jaw read, and the joints whose CURRENT hold it
        # bound (a real ATTACH for that close is then a duplicate, not a second bind).
        self._derived: set[str] = set()
        self._goals: dict[str, _Goal | None] = {"grasp": None, "place": None, "deploy": None}
        self._now_ns = 0
        self._clock_reset = False  # the joint stamps went back at least once (a sim reset)

    # -- inputs ------------------------------------------------------------------------

    @property
    def place_zones(self) -> dict[str, PlaceZone]:
        """The station's declared place zones by ``PlaceZone.target_id`` (``place_region``)."""
        return self._place_zones

    @place_zones.setter
    def place_zones(self, zones: dict[str, PlaceZone]) -> None:
        self._place_zones = zones
        self._forget_zones()

    @property
    def zone_transform(self) -> Callable[[str, int], NDArray[np.float64] | None] | None:
        """tf2 ``grid <- frame`` at a stamp (the node's lookup); unset: no fixed-frame zone."""
        return self._zone_transform

    @zone_transform.setter
    def zone_transform(
        self, lookup: Callable[[str, int], NDArray[np.float64] | None] | None
    ) -> None:
        self._zone_transform = lookup
        self._zone_cache = None  # a region resolved through the old lookup

    def _forget_zones(self) -> None:
        """Drop the zone resolve logs and cache: a reload, a new declaration, a reset, a move."""
        self._zone_state.clear()
        self._zone_choice_log = ()
        self._zone_cache = None

    def zone_frame_refusal(self, zone: PlaceZone) -> str | None:
        """Why ``zone`` is refused for its frame (``zone_frames``), or ``None`` if allowed."""
        frame = zone.region.frame_id
        if frame in self.zone_frames:
            return None
        if frame == self.frame_id:
            return (
                f"base frame {frame!r} of a {self.description.embodiment_kind.value!r} robot: "
                "a moving base carries the box away from the furniture; declare it in "
                f"{self.description.map_frame!r}"
            )
        return f"frame {frame!r} is neither the map / odom frame nor a fixed base frame"

    @property
    def now_ns(self) -> int:
        """The newest source stamp seen since the joint stamps last went back (a sim reset)."""
        return self._now_ns

    def _from_old_epoch(self, stamp_ns: int) -> bool:
        """Whether ``stamp_ns`` predates the last sim reset: past it by more than a reset step.

        Inputs lag the joint clock, never lead it, so after a reset a stamp that far ahead of
        it was taken before the reset (a frame or answer still queued in the node).
        """
        return self._clock_reset and stamp_ns > self._now_ns + _CLOCK_RESET_NS

    def _tick(self, stamp_ns: int, *, may_reset: bool = False) -> None:
        if may_reset and stamp_ns < self._now_ns - _CLOCK_RESET_NS:
            # The source clock went back (a sim reset), as ``JawStallDetector.update`` reads it:
            # a running maximum from before it would stamp every jaw command ahead of every
            # jaw sample, and the stall-derived ATTACH would never fire (review of #479). Joint
            # states only: a camera frame or grid can lag a second or more (the node holds a
            # frame up to ``_MASK_TF_WAIT_S`` for its transform) and is late, not a reset.
            self._now_ns = stamp_ns
            self._clock_reset = True
            # Source-time state from before the reset would gate or mislead the new epoch:
            # the newest-grid gate would refuse every grid until sim time passed the old one,
            # and a late ATTACH would look up hand poses from the last episode.
            self.grid, self.grid_stamp_ns = None, 0
            self._forget_zones()
            kept = [h for h in self._joint_history if h[0] <= stamp_ns]
            self._joint_history.clear()
            self._joint_history.extend(kept)
        if self._from_old_epoch(stamp_ns):
            # A frame, grid or answer queued before the reset never lifts the clock back.
            return
        self._now_ns = max(self._now_ns, stamp_ns)
        for goal in self._goals.values():
            if goal is not None and goal.start_ns is None and self._now_ns > 0:
                self._start(goal, self._now_ns)
        self._expire_released()

    def _hands_clear_of(self, box: PlaceRegion) -> bool:
        """Every hand collision box ``_RELEASE_CLEAR_M`` or more from ``box`` (all known)."""
        if not self.binders or not self._joints <= self.positions.keys():
            return False
        target = _box(box)
        return all(
            box_gap_lower_bound_m(target, hb) >= _RELEASE_CLEAR_M
            for binder in self.binders.values()
            for hb in binder.hand.hand_boxes(self.frame_id, self.positions)
        )

    def hands_clear_of_targets(self) -> bool:
        """No hand holds anything and every hand is clear of every TARGET track.

        Clear: every hand's pinch box ``_JUDGE_CLEAR_M`` or more from each CONFIRMED,
        OCCLUDED or RELEASED TARGET box; ``False`` with no such track (nothing to be clear of
        is no evidence the hand let go). The completion check asks its judge only then (asked
        with the open fingers still on the banana in the bowl, the judge said done; the rubric
        requires the robot clear of it).
        """
        if any(b.held is not None for b in self.binders.values()):
            return False
        targets = [
            t.box
            for t in self.tracker.store.all()
            if t.role is ObjectRole.TARGET
            and t.status in (TrackStatus.CONFIRMED, TrackStatus.OCCLUDED, TrackStatus.RELEASED)
        ]
        if not targets or not self.binders or not self._joints <= self.positions.keys():
            return False
        pinches = [b.pinch_in_frame(self.positions) for b in self.binders.values()]
        return all(
            box_gap_lower_bound_m(_box(t), jh) >= _JUDGE_CLEAR_M for t in targets for jh in pinches
        )

    def _expire_released(self) -> None:
        """Retire each released binding, and its RELEASED track, once its window ran out.

        Until then the kernel keeps the payload frozen at the release pose and the voxel
        bridge keeps the object's box clear; after it the object is plain world. The window
        also ends once every hand is clear of the released box (the source-clock window ran
        14 s of wall time, and the frozen box blocked the regrasp of
        the orange it had let go), and once a CONFIRMED track of its label minted after the
        release touches the released box: the object came to rest and the map holds it (the
        orange dropped 3 cm onto the table, was re-tracked there 1 s later, and the box frozen in
        the air stopped the arm carrying the next orange).
        """
        for oid, (_, since) in list(self._released.items()):
            track = self.tracker.store.get(oid)
            clear = track is not None and self._hands_clear_of(track.box)
            known = self._known_at_release.get(oid, frozenset())
            landed = track is not None and any(
                t.object_id not in known
                and t.label == track.label
                and t.status is TrackStatus.CONFIRMED
                and box_gap_lower_bound_m(_box(t.box), _box(track.box)) <= 0.0
                for t in self.tracker.store.all()
            )
            if clear or landed or self._now_ns - since >= self.released_clear_ns:
                del self._released[oid]
                self._known_at_release.pop(oid, None)
                if (track := self.tracker.store.get(oid)) is not None:
                    self.tracker.queries.note_released(track)
                self.tracker.store.drop(oid)
                self.attachment_changes += 1
                why = "hands_clear" if clear else "reacquired" if landed else "window"
                log.info("object_tracker.released_track_retired", object_id=oid, why=why)

    def leave_station(self) -> tuple[list[str], list[str]]:
        """Discard what this station measured; keep what the hands still hold.

        Station mode (the node calls it on every activate and deactivate): tracks, released
        records and the grid are all in the base frame, so they hold only while the base stands
        where they were measured. A released record frozen at the let-go pose would otherwise
        ride the driving robot while the object stays put. Held bindings (``BOUND`` tracks and a
        live hold's candidates) ride the hand, not the base, and are kept: dropping them would
        publish "nothing carried" for a payload the robot still holds. The queries' release
        priors (``QueryBook.note_released``) are base-frame columns too and are cleared.

        Returns:
            The dropped track ids and the released records retired with them.
        """
        held = self._held_ids()
        released = list(self._released)
        for q in self.tracker.queries.queries:
            q.prior = None  # a base-frame release column: wrong once the base drives
        self._released.clear()
        self._known_at_release.clear()
        self._off_hand.clear()
        dropped = [
            t.object_id
            for t in self.tracker.store.all()
            if t.object_id not in held and t.status is not TrackStatus.BOUND
        ]
        for oid in dropped:
            self.tracker.store.drop(oid)
        self.grid, self.grid_stamp_ns = None, 0
        self._instance_cells = None
        self.tracker.drop_requests()  # a late answer is about the last station's frame
        self._forget_zones()
        if released:
            self.attachment_changes += 1
        return dropped, released

    def open_standing_queries(self, queries: Sequence[ObjectQuery]) -> None:
        """The deploy's prompt-time queries (``object_queries_json``), goal ``deploy:deploy@0``.

        They ground at prompt time whatever declarations are in force (a scene with no
        place declaration still grounds its plate); a declaration's own goal re-keys the
        tracks its queries match.
        """
        self._set_goal("deploy", "deploy", 0, True, queries)

    def on_instance_cells(
        self,
        stamp_ns: int,
        frame_id: str,
        origin: tuple[float, float, float],
        resolution: float,
        cells: dict[str, frozenset[Cell]],
    ) -> None:
        """EXPERIMENT: per-track vision-labelled cells, applied on the next frame (our lattice)."""
        self._instance_cells = (stamp_ns, frame_id, origin, resolution, cells)

    def _apply_instance_cells(self, hands: Sequence[PlaceRegion]) -> None:
        pending = self._instance_cells
        if pending is None or self.grid is None:
            return
        self._instance_cells = None
        stamp_ns, frame_id, origin, res, cells = pending
        g = self.grid
        same_lattice = (
            frame_id == g.frame_id
            and math.isclose(res, g.resolution, abs_tol=_LATTICE_TOL_M)
            and all(
                math.isclose(a, b, abs_tol=_LATTICE_TOL_M)
                for a, b in zip(origin, g.origin, strict=True)
            )
        )
        if not same_lattice:
            log.info("object_tracker.instance_cells_lattice_mismatch", frame=frame_id)
            return
        if not np.allclose(g.rotation(), np.eye(3), atol=_LATTICE_TOL_M):
            # ``InstanceCells`` carries no orientation, and ``_vision_fit`` takes the lattice's
            # z for the frame's: on a rotated lattice (OccupancyVoxels may be one) the cells
            # would land on the wrong voxels and layers. Fail closed: no vision box (review of
            # #521). Carrying the orientation in the message lifts this.
            log.info("object_tracker.instance_cells_lattice_rotated", frame=frame_id)
            return
        self.tracker.apply_instance_cells(cells, g, stamp_ns, hands)

    def on_grid(self, grid: VoxelLattice, source_stamp_ns: int) -> None:
        """Keep the newest grid in the tracks' frame; another frame is logged and ignored."""
        if grid.frame_id != self.frame_id:
            log.warning("object_tracker.grid_frame", grid=grid.frame_id, tracks=self.frame_id)
            return
        if self._from_old_epoch(source_stamp_ns):
            log.info("object_tracker.grid_before_reset_ignored", stamp_ns=source_stamp_ns)
            return
        if source_stamp_ns >= self.grid_stamp_ns:
            self.grid, self.grid_stamp_ns = grid, source_stamp_ns
            self._rescope()
            self._measure_scope()
            self._tick(source_stamp_ns)

    def on_jaw_command(self, gripper_joint: str, target: float, stamp_ns: int) -> None:
        """Fold the approved command for a gripper joint into the derived-attach read.

        ``stamp_ns`` is the source time it was received at: jaw samples stamped before it are
        read against the command before it.
        """
        detector = self._stall.get(gripper_joint)
        if detector is not None and math.isfinite(target):
            detector.command(target, stamp_ns)

    def on_joint_state(self, positions: Mapping[str, float], stamp_ns: int) -> None:
        """Fold measured joints in (manifest names); run the jaw-open release rule."""
        self.positions.update(positions)
        if self._joints <= self.positions.keys():
            self._joint_history.append((stamp_ns, dict(self.positions)))
        for joint, binder in self.binders.items():
            if joint in positions:  # the most open jaw before closure bounds a fallback
                binder.observe_jaw(positions[joint])
        self._tick(stamp_ns, may_reset=True)
        for joint, monitor in list(self._monitors.items()):
            jaw = positions.get(joint)
            if jaw is not None and monitor.observe(jaw, stamp_ns):
                log.info("object_tracker.jaw_open_release", hand=joint, jaw=jaw)
                # Decided a dwell after the jaw opened; the object was let go at the opening.
                self._release(joint, stamp_ns, let_go_ns=monitor.opened_ns)
        for joint, detector in self._stall.items():
            jaw = positions.get(joint)
            if jaw is None or not math.isfinite(jaw) or not detector.update(jaw, stamp_ns):
                continue
            holder = self.binders.get(joint)
            if holder is not None and holder.held is not None:
                # The same hold read again after a chattering command unlatched the detector,
                # not a new grasp (14 binds and 13 releases ~2 s apart, each rebind
                # re-centring the box). A real release clears `held` first.
                log.info("object_tracker.attach_duplicate", hand=joint, derived=True)
                continue
            held = detector.held_median
            log.info("object_tracker.derived_attach", hand=joint, jaw=held, stamp_ns=stamp_ns)
            self._derived.add(joint)
            self.on_grasp_event(
                GraspEventKind.ATTACH, joint, jaw=held, hold=held, stamp_ns=stamp_ns, derived=True
            )

    def on_grasp_declaration(self, decl: GraspDeclaration) -> None:
        """A grasp declaration (or its retraction) from dispatch."""
        self.grasp_declaration = decl if decl.active else None
        # Locates crop to the declared workspace (issue #418 restock: ~40 px cartons).
        self.tracker.search_region = decl.workspace_box if decl.active else None
        # `tracking`: the declared search box decides WHICH instance is the goal's target, as
        # it decides the grasp region (`grasp_region`): a same-label object outside it (a
        # shelf carton while the declared one is in the tote, AMR aisle 2026-11-16) neither
        # grounds the query nor is keyed to the declaration. `on` keeps every instance of the
        # class keyed (its provider exempts the one the policy picks, track ids).
        tracking = self.mode is TrackingMode.TRACKING
        within = decl.search_box if decl.active and tracking else None
        if within is not None and within.frame_id != self.frame_id:
            log.warning(
                "object_tracker.search_box_unscoped",
                target_id=decl.target_id,
                box_frame=within.frame_id,
                grid_frame=self.frame_id,
                why="the search box is not in the grid frame; the target query is not scoped",
            )
            within = None
        # A TARGET query's seed names the instance (a recalled memory node, #631): it scopes
        # the query like the search box does, through tf2 (`_rescope`). The runner's seed is
        # the scene's search box itself (handled above), never an instance. One seed only:
        # TARGET queries seeded apart name no single instance (logged; label match).
        seeds: list[PlaceRegion] = []
        for q in decl.target_queries:
            seed = q.seed
            if q.role is not ObjectRole.TARGET or seed is None:
                continue
            if seed not in (decl.search_box, *seeds):
                seeds.append(seed)
        scope = seeds[0] if tracking and decl.active and len(seeds) == 1 else None
        if tracking and decl.active and len(seeds) > 1:
            log.warning(
                "object_tracker.goal_scope_ambiguous",
                target_id=decl.target_id,
                seeds=len(seeds),
                why="TARGET queries seeded apart name no single instance; label match only",
            )
        self._set_goal(
            "grasp",
            decl.target_id,
            decl.stamp_ns,
            decl.active,
            decl.target_queries,
            invocation_id=decl.invocation_id,
            within=within,
            scope=scope,
        )

    def on_place_declaration(self, decl: PlaceDeclaration) -> None:
        """A place declaration (or its retraction) from dispatch."""
        self.place_declaration = decl if decl.active else None
        self._forget_zones()  # a new declaration logs its zone's state afresh
        if decl.active and decl.target_id.startswith(PlaceZone.TARGET_PREFIX):
            # Logged once per declaration: `place_region` itself runs on every publish.
            ids = [
                f"{PlaceZone.TARGET_PREFIX}{n}" for n in zone_candidate_names(decl.target_id) or ()
            ] or [decl.target_id]
            zones = [self.place_zones.get(i) for i in ids]
            log.info(
                "object_tracker.place_zone_declared",
                target_id=decl.target_id,
                known=all(z is not None for z in zones),
                zone_frame=sorted({z.region.frame_id for z in zones if z is not None}),
                grid_frame=self.frame_id,
            )
        queries = () if decl.destination_query is None else (decl.destination_query,)
        self._set_goal(
            "place",
            decl.target_id,
            decl.stamp_ns,
            decl.active,
            queries,
            invocation_id=decl.invocation_id,
        )

    def on_grasp_event(  # noqa: PLR0911  # reason: each event names its own outcome
        self,
        kind: GraspEventKind,
        gripper_joint: str,
        *,
        jaw: float,
        hold: float,
        stamp_ns: int,
        derived: bool = False,
    ) -> Binding | None:
        """Bind, grow or release at the HAL trigger's event; the held binding after it.

        ``derived`` marks the tracker's own stalled-jaw read (``derive_attach``): a real ATTACH
        for a hold it already bound is a duplicate and returns the held binding untouched.
        """
        self._tick(stamp_ns)
        binder = self.binders.get(gripper_joint)
        if binder is None:
            log.warning("object_tracker.grasp_event_unknown_hand", hand=gripper_joint)
            return None
        detector = self._stall.get(gripper_joint)
        if kind is GraspEventKind.ATTACH and not derived and detector is not None:
            if gripper_joint in self._derived and binder.held is not None:
                log.info("object_tracker.attach_duplicate", hand=gripper_joint)
                return binder.held
            detector.suppress()  # the HAL's trigger covers this hold
        elif kind is GraspEventKind.DETACH:
            self._derived.discard(gripper_joint)
        if not math.isfinite(jaw):  # the HAL had no measured jaw: the newest sample
            jaw = self.positions.get(gripper_joint, math.nan)
        if not math.isfinite(jaw):
            log.warning("object_tracker.grasp_event_no_jaw", hand=gripper_joint, kind=kind.value)
            return binder.held
        hold = hold if math.isfinite(hold) else jaw
        positions = {**self.positions, gripper_joint: jaw}
        if not self._joints <= positions.keys():
            missing = sorted(self._joints - positions.keys())
            log.warning("object_tracker.grasp_event_unposed", hand=gripper_joint, missing=missing)
            return None
        if kind is GraspEventKind.DETACH:
            self._release(gripper_joint, stamp_ns)
            return None
        if kind is GraspEventKind.REGRASP:
            if binder.on_regrasp(positions, stamp_ns) is not None:
                self.attachment_changes += 1
            return binder.held
        if binder.held is not None:  # a second ATTACH without a DETACH: the old hold ended
            self._release(gripper_joint, stamp_ns)
        late = self._late_hold(gripper_joint, binder, positions, stamp_ns)
        binding = binder.on_attach(
            self.tracker.store,
            late.positions or positions,
            stamp_ns,
            excluded_track_ids=late.excluded_track_ids,
            oversize_track_ids=frozenset(
                t.object_id
                for t in self.tracker.store.all()
                if self.tracker.oversize_provisional(t)
            ),
        )
        self._monitors[gripper_joint] = JawReleaseMonitor.for_hand(
            binder.hand, hold=hold, release_dwell_s=self.release_dwell_s
        )
        self._attach_ns[gripper_joint] = stamp_ns
        self.attachment_changes += 1
        if binding.reground:
            self._reground()
        return binding

    def _grasp_pose(
        self,
        binder: GraspBinder,
        joint: str,
        positions: Mapping[str, float],
        sample: Mapping[str, float],
        hold_ns: int,
        fresh: Sequence[TrackedObject],
    ) -> tuple[dict[str, float], NDArray[np.float64], list[TrackedObject]] | None:
        """The grasp pose a late bind carries from, with the ``fresh`` tracks holding its pinch.

        The hold's own pose (``sample``) first; else the newest within ``_LATE_LOOKBACK_NS``
        before the hold whose pinch a fresh track holds: the policy lifts while the jaw is still
        settling (by the stall the pinch was 4 cm above the banana's box). ``None``: no pose in
        the window had its pinch in a fresh box.
        """
        older = [
            s for ts, s in reversed(self._joint_history) if 0 <= hold_ns - ts <= _LATE_LOOKBACK_NS
        ]
        # The pinch centre first: the fingertips' level (``_pinch_holds``) also meets poses
        # already lifted a few cm, and the carried box would ride that much off the object.
        for centre_only in (True, False):
            for s in (sample, *older):
                q = {**s, joint: positions[joint]}
                jh = binder.pinch_in_frame(q)
                cands = [
                    o for o in fresh if _pinch_holds(o, jh, binder.cell_m, centre_only=centre_only)
                ]
                if cands:
                    return q, jh[0], cands
        return None

    def _late_hold(  # noqa: PLR0911  # reason: one return per guard
        self,
        joint: str,
        binder: GraspBinder,
        positions: Mapping[str, float],
        stamp_ns: int,
    ) -> _LateAttachChoice:
        """The joint state of the hold's start when this ATTACH is late and safe to bind there.

        A policy lifts within a fraction of a second of the jaw settling, so the ATTACH (the
        HAL's trigger or the stalled-jaw dwell) often lands after the object has begun to
        move and its track has been split or lost (#418 r12: ATTACH +0.3 to +2 s after the lift,
        13 of 13 DROID/restock binds fell back to the 2-4x PINCH_FALLBACK box). Binding the
        track as fitted at the hand pose NOW would put the payload where the object was, not
        where it is. Binding it at the hand pose of the hold's start (``pose_in_link`` from
        there) carries it with the hand, which is right if the grasp is rigid. Guarded so a
        wrong guess cannot under-cover the real object: the hand travelled at most
        ``_LATE_MAX_TRAVEL_M`` since the hold; exactly one track has a geometry fit within
        ``_LATE_CONFIRM_NS`` of the hold and holds the pinch centre at the hold. An object
        left on the table is the camera's to catch after the bind (``_check_drop``), not the
        map's before it. Map confirmation alone does not refresh fit age, except for the goal's
        OCCLUDED target, exactly as the bind itself reads it (``binding._fresh_ns``): under the
        hand the camera cannot refit it (restock bench 2026-10-31, 4 of 4 untracked 6 cm grasps:
        the target occluded, map-confirmed 0.0-0.17 s before the bind, last fit 2.6-11 s old).
        Otherwise ``None``: the normal bind.
        A SPLIT / LOST candidate is revived to OCCLUDED for the bind.
        """
        detector = self._stall.get(joint)
        hold_ns = detector.hold_since_ns if detector is not None else None

        def skip(why: str, **kv: object) -> None:
            log.info("object_tracker.late_attach_skipped", hand=joint, why=why, **kv)

        if hold_ns is None:
            # No stalled-jaw read of this hold: the ATTACH's own time stands in, so the
            # lookback still finds the last second's pose whose pinch a fresh track holds,
            # under the same travel guard (bound where the hand already was, 9.6 cm up,
            # above the banana's 5 cm box, untracked).
            log.info("object_tracker.late_attach_no_hold_read", hand=joint,
                     detector=detector is not None)  # fmt: skip
            hold_ns = stamp_ns
        # A hold read after the ATTACH's stamp (the detector's samples run behind it) means the
        # ATTACH is not late: its own stamp stands in (lag -67 ms, skipped, bound where the hand
        # already was, 12.9 cm up).
        hold_ns = min(hold_ns, stamp_ns)
        if not _LATE_MIN_NS <= stamp_ns - hold_ns <= _LATE_MAX_NS:
            skip("lag_out_of_range", lag_s=(stamp_ns - hold_ns) / _NS)
            return _LateAttachChoice()
        if self.grid is None or not self._joint_history:
            skip("no_grid_or_history")
            return _LateAttachChoice()
        t, sample = min(self._joint_history, key=lambda s: abs(s[0] - hold_ns))
        if abs(t - hold_ns) > _JOINT_SAMPLE_MAX_AGE_NS:
            skip("no_joint_sample_at_hold")
            return _LateAttachChoice()
        store = self.tracker.store
        # A LOST / SPLIT track that never confirmed never held cells of its own: reviving it
        # revives a duplicate (restock bench_fix_08: a generic "carton" query's mint over the
        # target, refused `cover_already_tracked` then LOST, revived OCCLUDED by the late bind
        # and tied with the target, bound ambiguous).
        eligible = [
            o for o in store.all() if o.status in _LATE_STATUSES and o.last_confirmed_stamp_ns > 0
        ]
        fresh = [o for o in eligible if _fresh_ns(o) >= hold_ns - _LATE_CONFIRM_NS]
        if eligible and not fresh:
            skip(
                "stale_fit",
                tracks=[
                    (
                        o.object_id,
                        o.status.value,
                        (hold_ns - o.last_fit_stamp_ns) / _NS,
                        (hold_ns - o.last_confirmed_stamp_ns) / _NS,
                    )
                    for o in eligible
                ],
            )
            return _LateAttachChoice(excluded_track_ids=frozenset(o.object_id for o in eligible))
        pose = self._grasp_pose(binder, joint, positions, sample, hold_ns, fresh)
        if fresh and pose is None:
            skip("pinch_mismatch", tracks=[o.object_id for o in fresh])
            return _LateAttachChoice()
        hold_q = {**sample, joint: positions[joint]}
        hold_q, hold_c, cands = pose or (hold_q, binder.pinch_in_frame(hold_q)[0], [])
        travel = float(np.linalg.norm(binder.pinch_in_frame(positions)[0] - hold_c))
        if travel > _LATE_MAX_TRAVEL_M:
            skip("hand_travelled", travel_m=travel)
            return _LateAttachChoice()
        # No map-occupancy check of the carried box (2026-10-22 decision; the
        # self-filter's padding hides a held object and the map clears its old cells late, so
        # a box carried with the hand showed 0 cells while the banana rode in it). A wrong
        # bind is a phantom payload, more checking not less, which `_check_drop` releases on
        # the camera's first two in-hand fits away from the hand. Safety-WG review owed.
        if not cands:
            skip(
                "no_candidate",
                tracks=[(o.object_id, o.status.value, o.provisional) for o in store.all()],
            )
            return _LateAttachChoice()
        for track in cands:
            if track.status in (TrackStatus.SPLIT, TrackStatus.LOST):
                store.replace(track.model_copy(update={"status": TrackStatus.OCCLUDED}))
        log.info(
            "object_tracker.late_attach",
            hand=joint,
            objects=[o.object_id for o in cands],
            lag_s=(stamp_ns - hold_ns) / _NS,
        )
        return _LateAttachChoice(positions=hold_q)

    def on_frame(
        self, frame: CameraFrame, *, wall_s: float, results: Sequence[TrackerResult] = ()
    ) -> TrackerOutput:
        """Feed one depth frame (and answers to requests issued on it) through the tracker."""
        self._tick(frame.stamp_ns)
        self.tracker.bound_prompts = self._bound_prompts()
        self.tracker.held_ids = self._held_ids()
        hands = self.hand_spaces()
        self._apply_instance_cells(hands)  # EXPERIMENT exp/vision-voxel-fusion
        out = self.tracker.feed(
            frame,
            grid=self.grid,
            grid_stamp_ns=self.grid_stamp_ns,
            wall_s=wall_s,
            results=results,
            hand_spaces=hands,
        )
        self._measure_scope()  # this feed's fits, before ownership reads them
        # A declaration can open before the first matching fit; that fit may be
        # minted by an older standing query. Reconcile ownership after every feed.
        self._rekey_live_tracks()
        for oid, fit in out.bound_fits:
            self._check_drop(oid, fit, hands)
        self._expire_released()
        self._prune_ambiguous()
        self._check_grounding()
        return out

    def _held_ids(self) -> frozenset[str]:
        """Every live hold's candidate tracks, which the tracker never ages (``age_tracks``)."""
        return frozenset(
            oid for b in self.binders.values() if b.held is not None for oid in b.held.candidates
        )

    def _bound_prompts(self) -> dict[str, PlaceRegion]:
        """Each TRACKED payload's box where the hand carries it now (its in-hand prompt)."""
        if not self._joints <= self.positions.keys():
            return {}
        out: dict[str, PlaceRegion] = {}
        for binder in self.binders.values():
            held = binder.held
            if held is None or held.outcome is not BindOutcome.TRACKED:
                continue
            track = self.tracker.store.get(held.attachment.object_id)
            if track is not None:
                out[track.object_id] = track.box.model_copy(
                    update={"pose": binder.object_pose(self.positions)}
                )
        return out

    def _check_drop(self, oid: str, fit: ContainingFit, hands: Sequence[PlaceRegion]) -> None:
        """Release a TRACKED binding the camera twice in a row sees away from the hand.

        A jaw that never opens cannot release a payload that slipped out of it (the banana
        fell beside the bowl, the jaw stayed shut, and the payload rode the hand into the bowl). The
        measured box meeting no hand box (grown by a cell, ``hand_touches``) on ``_DROP_FITS``
        consecutive in-hand fits is a drop: released as a jaw opening is. One stray fit never
        releases. Safety-WG review owed.
        """
        if fit.box is None or self.grid is None:
            return
        if hand_touches(fit.box, hands, self.grid.resolution):
            self._off_hand.pop(oid, None)
            return
        self._off_hand[oid] = n = self._off_hand.get(oid, 0) + 1
        if n < _DROP_FITS:
            return
        for joint, binder in self.binders.items():
            if binder.held is not None and binder.held.attachment.object_id == oid:
                log.info("object_tracker.drop_seen", object_id=oid, hand=joint, fits=n)
                self._release(joint, self._now_ns)
        self._off_hand.pop(oid, None)

    # -- outputs -----------------------------------------------------------------------

    def snapshot(self) -> TrackedObjectSet:
        """The whole track table at the newest source stamp (revision: the store's)."""
        return self.tracker.store.snapshot(self._now_ns)

    def attachments(self) -> list[AttachedCollisionObject]:
        """Every hand's held payload (joint order), then the released ones still on the wire."""
        return self.held_attachments() + [record for record, _ in self._released.values()]

    def held_attachments(self) -> list[AttachedCollisionObject]:
        """Every hand's held payload (joint order), without the released records.

        What the node publishes while the station's perception is off (the base drives): a
        payload in a hand rides it wherever the base goes; a released record is frozen in the
        base frame and is wrong once it moves.
        """
        return [b.held.attachment for b in self.binders.values() if b.held is not None]

    def workspace_region(self) -> PlaceRegion | None:
        """The live grasp declaration's ``workspace_box``, map-confirmed, or ``None``.

        What the HAL producer confirms (``WORKSPACE_MIN_COVER`` of its footprint occupied
        on the newest grid, the same box re-stamped with the grid's ``source_stamp``),
        here for ``tracking`` / ``on`` (both modes: a declared box, confirmed, never shaped by
        a track), where the producer stands down: without it the
        kernel's workspace contact never applies. A box in another frame is not
        confirmed; the kernel ages the stamp out when grids stop.
        """
        decl, grid = self.grasp_declaration, self.grid
        if decl is None or decl.workspace_box is None or grid is None:
            return None
        box = decl.workspace_box
        if box.frame_id != grid.frame_id:
            return None
        _, covered = region_covers_occupied(grid, box, min_fraction=WORKSPACE_MIN_COVER)
        return box.model_copy(update={"stamp_ns": self.grid_stamp_ns}) if covered else None

    def place_region(self) -> PlaceRegion | None:
        """The live place declaration's destination region, or ``None``.

        The HAL's place producer measures a flat support under the payload, which a bowl
        never is (``partial_support`` / ``no_surface``, no region armed in 9 of 9), so the
        tracker stands in. ``None`` without a held payload or a grid: fail closed.

        A declaration whose ``target_id`` names a declared zone (``zone:<name>``, a shelf slot
        no detector grounds) gets that ``PlaceZone``'s box, stamped with the newest grid's
        ``source_stamp``; a zone declared in a fixed frame (``map`` / ``odom``, a docked
        mobile manipulator) is re-expressed in the grid frame through tf2 at that stamp
        (``rebase_box``). An unknown zone, or one with no tf2 at the grid's stamp or tilted past
        the bound, is ``None`` (fail closed, logged once per change).
        A declared, not measured, region: Safety-WG review, a hazard-log row and an ADR-0097
        amendment owed. This is the only place region in ``tracking``.

        ``on`` only: otherwise the CONFIRMED / OCCLUDED DESTINATION track labelled as the
        declaration's ``destination_query``, its containing box stamped with its last map
        confirmation (the kernel ages it out on ``place_region_max_age_s``). With several such
        tracks, the one whose box holds the carried payload's centre in x/y; ``None`` without a
        destination query, without such a track, or when that choice is not exactly one (a
        spurious second DESTINATION left no region at all: Spark pro#51 banana, every run).
        """
        decl, grid = self.place_declaration, self.grid
        if decl is None or grid is None:
            return None
        if all(b.held is None for b in self.binders.values()):
            return None
        if decl.target_id.startswith(PlaceZone.TARGET_PREFIX):
            return self._zone_region(decl.target_id, grid.frame_id)
        if self.mode is not TrackingMode.ON or decl.destination_query is None:
            return None
        label = decl.destination_query.label
        dest = [
            t
            for t in self.tracker.store.all()
            if t.role is ObjectRole.DESTINATION
            and t.label == label
            and t.status in (TrackStatus.CONFIRMED, TrackStatus.OCCLUDED)
        ]
        if len(dest) > 1:
            carried = [b.pose.xyz for b in self._bound_prompts().values()]
            dest = [t for t in dest if any(_over(t.box, xyz) for xyz in carried)]
        if len(dest) != 1 or dest[0].box.frame_id != grid.frame_id:
            return None
        t = dest[0]
        return t.box.model_copy(
            update={
                "stamp_ns": t.last_confirmed_stamp_ns,
                "evidence_ref": f"tracked:{t.object_id}|epoch={t.epoch}",
            }
        )

    def _zone_region(self, target_id: str, grid_frame: str) -> PlaceRegion | None:
        """The declared zone ``target_id`` names, in the grid frame, stamped with the newest grid.

        A zone in another frame is re-expressed through ``zone_transform`` at the grid's
        ``source_stamp`` (where the base was when the map was built). ``None`` for an unknown
        zone, no transform at that stamp, or a tilted one: no region, no allowance.

        A candidate set (``zone_candidates_target_id``: the prompt named the same slot at
        several stations) resolves to the one candidate whose box centre, in the grid frame,
        lies inside ``zone_reach``; none or several, or any candidate unresolved (it may be
        the one in reach), and it is ``None`` (fail closed, retried on the next call). Which
        candidates were considered and why each was dropped is logged on every change.
        """
        cached = self._zone_cache
        if cached is not None and cached[:2] == (target_id, self.grid_stamp_ns):
            return cached[2]  # tf2 is looked up once per grid, not on every publish
        names = zone_candidate_names(target_id)
        if names is not None:
            region = self._zone_choice(names, grid_frame)
        elif (zone := self.place_zones.get(target_id)) is not None:
            region = self._resolve_zone(zone, grid_frame)
        else:
            region = None
        # Only a resolved region is kept: tf2 may still bring the transform at this stamp.
        self._zone_cache = None if region is None else (target_id, self.grid_stamp_ns, region)
        return region

    def _zone_choice(self, names: Sequence[str], grid_frame: str) -> PlaceRegion | None:
        """The one in-reach zone of a candidate set (``_zone_region``), or ``None``."""
        kept: list[PlaceRegion] = []
        why: list[str] = []
        unresolved = False
        for name in names:
            zone = self.place_zones.get(f"{PlaceZone.TARGET_PREFIX}{name}")
            region = None if zone is None else self._resolve_zone(zone, grid_frame)
            if region is None:
                why.append(f"{name}:unresolved")
                unresolved = True
                continue
            if self.zone_reach is None:
                why.append(f"{name}:no_reach")
                continue
            (cx, cy, cz), radius = self.zone_reach
            x, y, z = region.pose.xyz
            dist = math.dist((x, y, z), (cx, cy, cz))
            if dist > radius:
                why.append(f"{name}:out_of_reach({dist:.2f}>{radius:.2f})")
                continue
            why.append(f"{name}:in_reach({dist:.2f}<={radius:.2f})")
            kept.append(region)
        # An unresolved candidate is not proven out of reach: uniqueness is not established.
        chosen = kept[0] if len(kept) == 1 and not unresolved else None
        # Keyed on each candidate's verdict, not its distance: jitter never re-logs.
        record = (*(w.split("(")[0] for w in why), "chosen" if chosen is not None else "none")
        if record != self._zone_choice_log:
            self._zone_choice_log = record
            event = "object_tracker.place_zone_chosen"
            if chosen is None:
                log.warning("object_tracker.place_zone_unchosen", candidates=why)
            else:
                log.info(event, zone=chosen.evidence_ref, candidates=why)
        return chosen

    def _resolve_zone(self, zone: PlaceZone, grid_frame: str) -> PlaceRegion | None:
        """``zone``'s box in ``grid_frame``, stamped with the newest grid (``_zone_region``)."""
        region = zone.region
        refusal = self.zone_frame_refusal(zone)
        if refusal is not None:
            self._zone_log(zone, "refused", grid_frame=grid_frame, why=refusal)
            return None
        if region.frame_id != grid_frame:
            t = None
            if self.zone_transform is not None:
                t = self.zone_transform(region.frame_id, self.grid_stamp_ns)
            if t is None:
                self._zone_log(zone, "no_tf", grid_frame=grid_frame)
                return None
            try:
                region = rebase_box(region, t, grid_frame)
            except ROSConfigError as exc:
                self._zone_log(zone, "rejected", grid_frame=grid_frame, error=str(exc))
                return None
            self._zone_log(zone, "resolved", grid_frame=grid_frame)
        return region.model_copy(
            update={"stamp_ns": self.grid_stamp_ns, "evidence_ref": f"declared:{zone.name}"}
        )

    def _zone_log(self, zone: PlaceZone, state: str, **kv: object) -> None:
        """Log a fixed-frame zone's resolve ``state`` once per change (``place_region`` is hot)."""
        if self._zone_state.get(zone.name) == state:
            return
        self._zone_state[zone.name] = state
        event = f"object_tracker.place_zone_{state}"
        if state == "resolved":
            log.info(event, zone=zone.name, zone_frame=zone.region.frame_id, **kv)
        else:
            log.warning(event, zone=zone.name, zone_frame=zone.region.frame_id, **kv)

    def grasp_region(self) -> PlaceRegion | None:
        """The one target's box while it lies wholly inside the declared ``search_box``.

        ``tracking`` only (``on``'s exemption provider keys on track ids: always ``None``). A
        track only narrows the declaration, never widens it: the region is the target track's
        box, returned only while all eight of its corners lie inside the declaration's
        ``search_box`` (so the allowance is never larger than what the scene declared, nor than
        the object measured; the declared box itself would allow more than ``off``, whose HAL
        grasp-target leg measures a region inside it). Stamped with the track's last map
        confirmation (the kernel ages it out on ``grasp_region_max_age_s``), only while exactly
        one CONFIRMED / OCCLUDED / BOUND TARGET track keyed to the declaration (the scope
        ``_rekey_live_tracks`` assigns, the runner's grounding count) has its box centre inside
        it. ``None`` -- fail closed -- without a declaration, a grid, a ``search_box`` (in the
        grid frame), exactly one such track, a track box reaching outside the ``search_box``,
        or within ``GraspDeclaration``'s caps (the kernel's ``kMaxGraspRegionHalfExtentM`` /
        ``kMaxGraspRegionVolumeM3``).

        The region keeps being published through the hold, as a HAL-measured one is: the
        kernel latches the box the first snapshot carrying the declared attachment brings,
        ignores later moves of it, and retires the declaration for good once the payload's
        measured origin leaves that box (``ingest_grasp_declaration``,
        ``grasp_region_moved_after_handover``, ``handover_exit``). Safety-WG review owed
        (ADR-0122).
        """
        decl, grid = self.grasp_declaration, self.grid
        if decl is None or grid is None or self.mode is not TrackingMode.TRACKING:
            return None
        box = decl.search_box
        if box is None or box.frame_id != grid.frame_id:
            return None
        key = declaration_key(decl.target_id, decl.stamp_ns)
        inv = np.linalg.inv(homogeneous_from_quat_xyz(box.pose.xyz, box.pose.quat_xyzw))
        half = np.asarray(box.half_extents, dtype=np.float64)

        def inside(c: Sequence[float]) -> bool:
            local = inv @ np.array([c[0], c[1], c[2], 1.0])
            return bool(np.all(np.abs(local[:3]) <= half))

        targets = [
            t
            for t in self.tracker.store.all()
            if t.role is ObjectRole.TARGET
            and t.declaration_key == key
            and t.status in (TrackStatus.CONFIRMED, TrackStatus.OCCLUDED, TrackStatus.BOUND)
            and t.box.frame_id == grid.frame_id
            and inside(t.box.pose.xyz)
        ]
        if len(targets) != 1:
            return None
        t = targets[0]
        tb = t.box
        cap = GraspDeclaration.MAX_HALF_EXTENT_M
        if max(tb.half_extents) > cap or tb.volume_m3() > GraspDeclaration.MAX_VOLUME_M3:
            return None
        t_box = homogeneous_from_quat_xyz(tb.pose.xyz, tb.pose.quat_xyzw)
        signs = np.array([(a, b, c) for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)], float)
        corners = (signs * np.asarray(tb.half_extents)) @ t_box[:3, :3].T + t_box[:3, 3]
        if not all(inside(c) for c in corners):
            return None
        return tb.model_copy(
            update={
                "stamp_ns": t.last_confirmed_stamp_ns,
                "evidence_ref": f"tracked:{t.object_id}|epoch={t.epoch}|within:search_box",
            }
        )

    def hand_spaces(self) -> list[PlaceRegion]:
        """Each hand's collision boxes (grid frame), at the measured configuration."""
        if not self._joints <= self.positions.keys():
            return []
        out: list[PlaceRegion] = []
        for binder in self.binders.values():
            for c, r, h in binder.hand.hand_boxes(self.frame_id, self.positions):
                w, x, y, z = rotation_to_quat_wxyz(r)
                pose = Pose6D(
                    xyz=(float(c[0]), float(c[1]), float(c[2])),
                    quat_xyzw=(x, y, z, w),
                    frame_id=self.frame_id,
                )
                cap = PlaceRegion.MAX_HALF_EXTENT_M
                half = (min(float(h[0]), cap), min(float(h[1]), cap), min(float(h[2]), cap))
                out.append(PlaceRegion(frame_id=self.frame_id, pose=pose, half_extents=half))
        return out

    # -- goals -------------------------------------------------------------------------

    def _set_goal(
        self,
        kind: str,
        target_id: str,
        stamp_ns: int,
        active: bool,
        queries: Sequence[ObjectQuery],
        *,
        invocation_id: str = "",
        within: PlaceRegion | None = None,
        scope: PlaceRegion | None = None,
    ) -> None:
        key = declaration_key(target_id, stamp_ns)
        suffix = f"#{invocation_id}" if invocation_id else ""
        goal_id = f"{kind}:{key}{suffix}"
        query_tuple = tuple(queries)
        old = self._goals[kind]
        if old is not None and (
            old.goal_id != goal_id
            or old.queries != query_tuple
            or old.scope != scope
            or (scope is None and old.within != within)
            or not active
        ):
            self.tracker.close_goal(old.goal_id)
            self._goals[kind] = None
            self._clear_declaration_scope(old.key)
            log.info("object_tracker.goal_closed", goal_id=old.goal_id)
        if not active or not query_tuple or self._goals[kind] is not None:
            self._rekey_live_tracks()
            return
        goal = self._goals[kind] = _Goal(
            goal_id,
            key,
            None,
            query_tuple,
            invocation_id=invocation_id,
            within=within,
            scope=scope,
            members=set() if scope is not None and scope.frame_id != self.frame_id else None,
        )
        self._rescope()
        self._measure_scope()
        self._rekey_live_tracks()
        if self._now_ns > 0:
            self._start(goal, self._now_ns)

    def _start(self, goal: _Goal, stamp_ns: int) -> None:
        """Open the goal's standing queries at source time ``stamp_ns``.

        A declaration that arrives before any source stamp starts at the first frame: a
        search timer anchored at 0 would exhaust on the first stamp it is compared with.
        """
        for i, query in enumerate(goal.queries):
            self.tracker.open_query(
                goal.goal_id,
                f"q{i}",
                query,
                goal.key,
                stamp_ns=stamp_ns,
                invocation_id=goal.invocation_id,
                within=goal.within if query.role is ObjectRole.TARGET else None,
                members=goal.members if query.role is ObjectRole.TARGET else None,
            )
        goal.start_ns = stamp_ns

    def _rescope(self) -> None:
        """Re-express each goal's ``scope`` in the grid frame (the base moves under ``map``).

        Through ``zone_transform`` at the newest grid's stamp. A lookup that misses (or a tilted
        transform) after the scope resolved keeps the last resolved box, so one late tf does
        not reset the query; one that never resolved keeps the scope in its own frame, where
        ``centre_within`` matches no track: fail closed. For a scope in another frame than the
        grid this box only crops the search (where the node is now); which tracks are inside
        is ``_measure_scope``'s.
        """
        for goal in self._goals.values():
            if goal is None or goal.scope is None:
                continue
            scope, within, error = goal.scope, goal.scope, ""
            if scope.frame_id != self.frame_id:
                t = None
                if self.zone_transform is not None:
                    t = self.zone_transform(scope.frame_id, self.grid_stamp_ns)
                try:
                    if t is None:
                        raise ROSConfigError(f"no tf2 {self.frame_id} <- {scope.frame_id}")
                    within = rebase_box(scope, t, self.frame_id)
                except ROSConfigError as exc:
                    error = str(exc)
                    if goal.scope_resolved and goal.within is not None:
                        within = goal.within  # the last resolved box
            resolved = within.frame_id == self.frame_id
            if resolved != goal.scope_resolved:
                goal.scope_resolved = resolved
                (log.info if resolved else log.warning)(
                    "object_tracker.goal_scope",
                    goal_id=goal.goal_id,
                    scope_frame=scope.frame_id,
                    grid_frame=self.frame_id,
                    resolved=resolved,
                    error=error,
                )
            goal.within = within
            for q in self.tracker.queries.queries:
                if q.goal_id == goal.goal_id and q.query.role is ObjectRole.TARGET:
                    q.within = within

    def _measure_scope(self) -> None:
        """Judge each track against a fixed-frame goal ``scope`` where the track was measured.

        A track's box is in the grid (base) frame as the base stood at its last fit, so it is
        compared with the scope re-expressed through ``zone_transform`` at that fit's stamp:
        both sides in the scope's own fixed frame (#651). The newest grid's box (``_rescope``)
        would move with the base while an unrefitted track's box does not, and a base that
        moved between the fit and the match would drop the node's instance or bind its
        neighbour. A verdict holds until the track is fitted again (a new box or fit stamp).
        Fail closed: a track with no transform at its fit stamp (tf2 missing, tilted, or older
        than the buffer) is out until a later fit resolves; one already judged keeps its
        verdict over a miss, so one late tf does not reset the query.
        """
        for goal in self._goals.values():
            if goal is None or goal.scope is None or goal.members is None:
                continue
            measured: dict[str, tuple[tuple[PlaceRegion, int], bool]] = {}
            for track in self.tracker.store.all():
                fit = (track.box, track.last_fit_stamp_ns)
                verdict = goal.measured.get(track.object_id)
                if (verdict is None or verdict[0] != fit) and self.zone_transform is not None:
                    t = self.zone_transform(goal.scope.frame_id, track.last_fit_stamp_ns)
                    try:
                        there = None if t is None else rebase_box(goal.scope, t, self.frame_id)
                    except ROSConfigError:
                        there = None  # tilted: unjudged, as a missing transform
                    if there is not None:
                        verdict = (fit, centre_within(track.box, there))
                if verdict is not None:
                    measured[track.object_id] = verdict
            goal.measured = measured
            inside = {oid for oid, (_, ok) in measured.items() if ok}
            if inside != goal.members:
                log.info("object_tracker.goal_scope_members", goal_id=goal.goal_id,
                         inside=sorted(inside))  # fmt: skip
                goal.members.clear()  # in place: the goal's queries hold this set
                goal.members.update(inside)

    def _clear_declaration_scope(self, key: str) -> None:
        """Remove a closed declaration's ownership key from its tracks."""
        store = self.tracker.store
        for t in store.all():
            if t.declaration_key != key:
                continue
            store.replace(t.model_copy(update={"declaration_key": ""}))

    def _rekey_live_tracks(self) -> None:
        """Assign tracks to one unambiguous live declaration query scope.

        Declaration-specific goals take precedence over deploy's standing queries.
        Distinct matching declaration scopes are ambiguous, so that track has no
        declaration owner until the ambiguity ends.
        """
        store = self.tracker.store
        for track in store.all():
            if track.status is TrackStatus.LOST:
                continue
            declaration_goals = {
                goal.goal_id: goal
                for kind, goal in self._goals.items()
                if kind != "deploy" and goal is not None and goal.owns(track)
            }
            if len(declaration_goals) > 1:
                key = ""
            elif declaration_goals:
                key = next(iter(declaration_goals.values())).key
            else:
                deploy = self._goals["deploy"]
                if deploy is None or not any(
                    track.label == query.label and track.role is query.role
                    for query in deploy.queries
                ):
                    key = ""
                else:
                    key = deploy.key
            if track.declaration_key != key:
                store.replace(track.model_copy(update={"declaration_key": key}))

    def _reground(self) -> None:
        """Restart the live grasp goal's search (an untracked grasp, after lift)."""
        goal = self._goals["grasp"]
        if goal is None or goal.start_ns is None:
            log.info("object_tracker.reground_skipped", why="no live grasp goal")
            return
        self._start(goal, self._now_ns)
        log.info("object_tracker.reground_queued", goal_id=goal.goal_id)

    def _check_grounding(self) -> None:
        for goal in self._goals.values():
            if goal is None or goal.start_ns is None or goal.ready_logged:
                continue
            if self.tracker.queries.ready(goal.goal_id):
                goal.ready_logged = True
                log.info("object_tracker.grounding_ready", goal_id=goal.goal_id)
            elif (
                not goal.incomplete_logged and self._now_ns - goal.start_ns > self.grounding_wait_ns
            ):
                goal.incomplete_logged = True
                emit = log.error if self.grounding_required else log.warning
                emit(
                    "object_tracker.grounding_incomplete",
                    goal_id=goal.goal_id,
                    grounding_required=self.grounding_required,
                )

    # -- binding -------------------------------------------------------------------------

    def _positions_at(self, stamp_ns: int) -> Mapping[str, float]:
        """The recorded joint sample nearest ``stamp_ns`` (the newest when none is recorded)."""
        if not self._joint_history:
            return self.positions
        return min(self._joint_history, key=lambda s: abs(s[0] - stamp_ns))[1]

    def _release(self, joint: str, stamp_ns: int, *, let_go_ns: int | None = None) -> None:
        binder = self.binders[joint]
        held = binder.held
        self._monitors.pop(joint, None)
        self._derived.discard(joint)
        detector = self._stall.get(joint)
        if detector is not None:
            detector.reset()  # a regrasp after this release may derive again
        if held is None:
            return
        if held.outcome is BindOutcome.TRACKED and self._joints <= self.positions.keys():
            oid = held.attachment.object_id
            track = self.tracker.store.get(oid)
            # The hand where it let go, not where the (late) decision finds it: a jaw-open
            # release is decided a dwell after the opening, the retreat may have begun.
            let_go_at = stamp_ns if let_go_ns is None else let_go_ns
            q_let_go = self._positions_at(let_go_at)
            pose = binder.object_pose(q_let_go)
            if track is not None:
                support = self._support_under(pose, track.box.half_extents, track.support_z)
                released = release_track(
                    self.tracker.store,
                    oid,
                    object_pose=pose,
                    support_z=support,
                    cell_m=binder.cell_m,
                    stamp_ns=stamp_ns,
                )
                if released is not None:
                    # `tracking`: frozen where it was let go and declared released
                    # (`released_by_link`), so the kernel freezes it and runs the window; left on
                    # the hand it rode the retreating hand 1.0-1.5 s, up to 20 cm (2026-11-02).
                    # `on`: the exemption provider releases it inside the kernel, earlier, keyed
                    # on the hand-attached record, so the record stays as held.
                    record = held.attachment
                    if self.mode is TrackingMode.TRACKING:
                        t_hand = binder.hand.t_frame_from_attach(self.frame_id, q_let_go)
                        record = freeze_released_attachment(
                            record,
                            base_link=self.frame_id,
                            t_base_from_link=t_hand,
                            stamp_ns=let_go_at,  # where the pose was taken
                        )
                    self._released[oid] = (record, stamp_ns)
                    self._known_at_release[oid] = frozenset(
                        t.object_id for t in self.tracker.store.all()
                    )
        binder.on_release()
        self.attachment_changes += 1

    def _support_under(
        self, pose: Pose6D, half: tuple[float, float, float], fallback_z: float
    ) -> float:
        """The highest occupied cell top in the column under the released body.

        ponytail: an axis-aligned column of the box's largest horizontal half extent; the
        track's previous support when the grid shows nothing under it.
        """
        if self.grid is None:
            return fallback_z
        centres = self.grid.occupied_centers()
        reach = max(half[:2]) + self.grid.resolution
        bottom = pose.xyz[2] - half[2]
        col = centres[
            np.all(np.abs(centres[:, :2] - np.asarray(pose.xyz[:2])) <= reach, axis=1)
            & (centres[:, 2] < bottom)
        ]
        if not len(col):
            return fallback_z
        return float(col[:, 2].max()) + self.grid.resolution / 2.0

    def _prune_ambiguous(self) -> None:
        """Drop each candidate a post-lift re-fit shows stayed behind from an ambiguous union."""
        if not self._joints <= self.positions.keys():
            return
        store = self.tracker.store
        for joint, binder in self.binders.items():
            held = binder.held
            if held is None or held.outcome is not BindOutcome.AMBIGUOUS:
                continue
            att, cands = held.attachment, held.candidates
            if len(att.primitives) != len(cands) + 1:
                continue  # merged into one box past the primitive cap: nothing to drop
            jh = binder.pinch_in_frame(self.positions)
            since = self._attach_ns.get(joint, 0)
            behind = set()
            for oid in cands:
                t = store.get(oid)
                if (
                    t is not None
                    and t.status is TrackStatus.CONFIRMED
                    and t.last_fit_stamp_ns > since
                    and box_gap_lower_bound_m(_box(t.box), jh) > 0.0
                ):
                    behind.add(oid)
            if not behind:
                continue
            keep = [i for i, oid in enumerate(cands) if oid not in behind]
            remaining = tuple(cands[i] for i in keep)
            pruned = att.model_copy(
                update={
                    "primitives": [att.primitives[0], *(att.primitives[i + 1] for i in keep)],
                    "geometry_revision": att.geometry_revision + 1,
                    "evidence_ref": "ambiguous:" + ",".join(remaining),
                    "stamp_ns": self._now_ns,
                }
            )
            binder.held = Binding(
                held.outcome,
                AttachedCollisionObject.model_validate(pruned.model_dump()),
                remaining,
                held.reground,
            )
            self.attachment_changes += 1
            log.info("object_tracker.ambiguous_pruned", hand=joint, dropped=sorted(behind))
