"""``TrackerSession`` — the ROS-free glue the object-tracker lifecycle node drives (issue #418).

The node (``openral_world_state_ros.object_tracker_node``) decodes ROS messages and owns the
wire; everything it decides lives here so it runs without ROS:

* **Goals.** A live ``GraspDeclaration`` with ``target_queries`` (a ``PlaceDeclaration`` with
  a ``destination_query``) opens one standing query per entry, keyed by the declaration
  (``"<target_id>@<stamp_ns>"``, the kernel's ``declaration_key``); a retraction or a new
  declaration closes the goal. Tracks persist across goals: a new goal re-keys the live
  tracks its queries match (label and role first, design step 1.6), so the kernel's
  "live ``declaration_key``" eligibility follows them.
* **Grounding wait.** ``grounding_ready`` / ``grounding_incomplete`` are logged once per
  goal, in source time, after ``grounding_wait_s``. Refusing the goal under
  ``grounding_required`` is the runner's (it is logged at error level here).
* **Binding** (``binding.GraspBinder`` per gripper): ``ATTACH`` binds what the pinch volume
  meets; ``REGRASP`` grows it; ``DETACH`` — or the jaw-open rule (``JawReleaseMonitor``,
  the kernel's K5 mirrored) — releases it (``release.release_track``). A fit the tracker
  hands back for a bound track triggers the in-hand protrusion growth. An ambiguous
  (``PINCH_FALLBACK``) union drops every candidate that a post-lift re-fit shows stayed
  behind (re-fitted after the ATTACH, its box no longer meeting the pinch volume). An
  untracked grasp re-opens the live grasp goal's queries (the re-ground after lift).

A RELEASED track's return to CONFIRMED (same id, ``epoch + 1``) is the tracker core's own
(association in the fall column, then map confirmation): ``release.reassociate_released``
is not called a second time here.

Clocks: every stamp is source time (depth, grid, joint-state, grasp-event stamps); the
segmenter rate runs on the ``wall_s`` the node feeds.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import structlog
from openral_core import (
    AttachedCollisionObject,
    GraspDeclaration,
    ObjectQuery,
    PlaceDeclaration,
    PlaceRegion,
    Pose6D,
    RobotDescription,
    TrackedObjectSet,
    TrackStatus,
)
from openral_core.exceptions import ROSConfigError
from openral_core.geometry import homogeneous_from_quat_xyz, rotation_to_quat_wxyz
from openral_core.kinematics import Box, box_gap_lower_bound_m
from openral_core.perception_geometry import (
    WORKSPACE_MIN_COVER,
    VoxelLattice,
    region_covers_occupied,
)

from openral_world_state.object_tracker.binding import (
    Binding,
    BindOutcome,
    GraspBinder,
    _holds,  # reason: same package; the candidate rule a late bind must share
    _t,
)
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

__all__ = ["GraspEventKind", "TrackerSession", "TrackingMode", "declaration_key"]

log = structlog.get_logger(__name__)

_NS = 1_000_000_000
#: A late ATTACH (the object may already be moving) is bound at the hand pose of the hold when
#: the hold began at least this long before the event ...
_LATE_MIN_NS = 100_000_000
#: ... and at most this long before it, ...
_LATE_MAX_NS = 3 * _NS
#: ... the hand has travelled no further than this since the hold (m), ...
_LATE_MAX_TRAVEL_M = 0.12
#: ... the track was last confirmed within this long of the hold start, and ...
_LATE_CONFIRM_NS = 1_500_000_000
#: ... the box carried to the hand's pose now is at least this occupied on the newest grid.
_LATE_MIN_COVER = 0.5
#: A track a late ATTACH may revive and bind.
_LATE_STATUSES = frozenset(
    {TrackStatus.CONFIRMED, TrackStatus.OCCLUDED, TrackStatus.SPLIT, TrackStatus.LOST}
)
#: A joint sample further than this from a stamp is not the pose then.
_JOINT_SAMPLE_MAX_AGE_NS = 250_000_000


class TrackingMode(StrEnum):
    """``DeployRuntime.object_tracking`` (``off`` never starts the node)."""

    SHADOW = "shadow"
    BIND = "bind"
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
    ready_logged: bool = False
    incomplete_logged: bool = False


def _box(region: PlaceRegion) -> Box:
    t = homogeneous_from_quat_xyz(region.pose.xyz, region.pose.quat_xyzw)
    return (t[:3, 3], t[:3, :3], np.asarray(region.half_extents, dtype=np.float64))


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
            binding stays on ``attachment_state`` after release (the kernel freezes it at
            the release pose, the voxel bridge clears its box) unless the track leaves
            RELEASED first. MUST equal the kernel's ``tracked_release_timeout_s`` and the
            voxel bridge's ``released_clear_timeout_s`` (all default 3.0 s).
        params: Tracker calibration.
        derive_attach: The tracker reads a stalled jaw itself (``JawStallDetector``: commanded
            jaw vs the median measured jaw, the manifest's ``stall_gap``) and binds as an ATTACH
            does, so a hold the HAL's position-stall trigger misses (a chattering jaw, a command
            past its band) is still bound. ``bind`` / ``on`` on a sim HAL only; the HAL trigger
            and the legacy / real path are unchanged, and a real ATTACH for the same close is
            never doubled.
        require_hands: Refuse a gripper the binder cannot model (``bind`` / ``on``); in
            ``shadow`` such a hand is logged and skipped.

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
        release_dwell_s: float = 0.3,
        released_clear_timeout_s: float = 3.0,
        params: TrackerParams | None = None,
        require_hands: bool = False,
        derive_attach: bool = False,
    ) -> None:
        """Model every gripper and start with no tracks and no goals."""
        self.description = description
        self.frame_id = description.base_frame
        self.tracker = ObjectTracker(self.frame_id, params)
        self.grounding_wait_ns = int(grounding_wait_s * _NS)
        self.grounding_required = grounding_required
        self.release_dwell_s = release_dwell_s
        self.released_clear_ns = int(released_clear_timeout_s * _NS)
        # Released TRACKED_OBJECT bindings still on the wire: object_id -> (record, stamp).
        self._released: dict[str, tuple[AttachedCollisionObject, int]] = {}
        self.binders: dict[str, GraspBinder] = {}
        self._stall: dict[str, JawStallDetector] = {}
        for joint in description.joints:
            if joint.role != "gripper":
                continue
            try:
                hand = HandModel(description, joint.name)
            except ROSConfigError as exc:
                if require_hands:
                    raise
                log.warning("object_tracker.hand_unbindable", joint=joint.name, error=str(exc))
                continue
            self.binders[joint.name] = GraspBinder(hand, frame_id=self.frame_id, cell_m=cell_m)
            cal = joint.closure_calibration
            if derive_attach and cal is not None:
                self._stall[joint.name] = JawStallDetector(
                    cal.closed_position,
                    cal.closed_rest_offset + cal.stall_gap,
                    cal.settle_tolerance,
                )
        self._joints = frozenset(j.name for j in description.joints)
        # (stamp_ns, complete joint state): the hand pose at the start of a hold, for a late ATTACH.
        self._joint_history: deque[tuple[int, dict[str, float]]] = deque(maxlen=480)
        self.positions: dict[str, float] = {}
        self.grid: VoxelLattice | None = None
        self.grid_stamp_ns = 0
        self.grasp_declaration: GraspDeclaration | None = None
        self.place_declaration: PlaceDeclaration | None = None
        self.attachment_changes = 0  # bumped whenever the held set changes
        self._monitors: dict[str, JawReleaseMonitor] = {}
        self._attach_ns: dict[str, int] = {}
        # Per gripper: the tracker's own stalled-jaw read, and the joints whose CURRENT hold it
        # bound (a real ATTACH for that close is then a duplicate, not a second bind).
        self._derived: set[str] = set()
        self._goals: dict[str, _Goal | None] = {"grasp": None, "place": None, "deploy": None}
        self._now_ns = 0

    # -- inputs ------------------------------------------------------------------------

    @property
    def now_ns(self) -> int:
        """The newest source stamp seen."""
        return self._now_ns

    def _tick(self, stamp_ns: int) -> None:
        self._now_ns = max(self._now_ns, stamp_ns)
        for goal in self._goals.values():
            if goal is not None and goal.start_ns is None and self._now_ns > 0:
                self._start(goal, self._now_ns)
        self._expire_released()

    def _expire_released(self) -> None:
        """Drop each released binding whose track left RELEASED or whose window ran out.

        The same update that re-confirms the track (``epoch + 1``) or loses it takes the
        old ``(id, epoch)`` off the wire; until then the kernel keeps it frozen at the
        release pose and the voxel bridge keeps the object's box clear.
        """
        for oid, (record, since) in list(self._released.items()):
            t = self.tracker.store.get(oid)
            gone = t is None or t.status is not TrackStatus.RELEASED or t.epoch != record.epoch
            if gone or self._now_ns - since >= self.released_clear_ns:
                del self._released[oid]
                self.attachment_changes += 1
                log.info("object_tracker.released_binding_dropped", object_id=oid, left=gone)

    def open_standing_queries(self, queries: Sequence[ObjectQuery]) -> None:
        """The deploy's prompt-time queries (``object_queries_json``), goal ``deploy:deploy@0``.

        They ground at prompt time whatever declarations are in force (a scene with no
        place declaration still grounds its plate); a declaration's own goal re-keys the
        tracks its queries match.
        """
        self._set_goal("deploy", "deploy", 0, True, queries)

    def on_grid(self, grid: VoxelLattice, source_stamp_ns: int) -> None:
        """Keep the newest grid in the tracks' frame; another frame is logged and ignored."""
        if grid.frame_id != self.frame_id:
            log.warning("object_tracker.grid_frame", grid=grid.frame_id, tracks=self.frame_id)
            return
        if source_stamp_ns >= self.grid_stamp_ns:
            self.grid, self.grid_stamp_ns = grid, source_stamp_ns
            self._tick(source_stamp_ns)
            if self._joints <= self.positions.keys():
                for binder in self.binders.values():  # untracked payloads grow on the map
                    if binder.grow_untracked(
                        self.tracker.store, grid, self.positions, source_stamp_ns
                    ):
                        self.attachment_changes += 1

    def on_jaw_command(self, gripper_joint: str, target: float, stamp_ns: int) -> None:
        """Fold the approved command for a gripper joint into the derived-attach read."""
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
        self._tick(stamp_ns)
        for joint, monitor in list(self._monitors.items()):
            jaw = positions.get(joint)
            if jaw is not None and monitor.observe(jaw, stamp_ns):
                log.info("object_tracker.jaw_open_release", hand=joint, jaw=jaw)
                self._release(joint, stamp_ns)
        for joint, detector in self._stall.items():
            jaw = positions.get(joint)
            if jaw is None or not math.isfinite(jaw) or not detector.update(jaw, stamp_ns):
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
        self._set_goal("grasp", decl.target_id, decl.stamp_ns, decl.active, decl.target_queries)

    def on_place_declaration(self, decl: PlaceDeclaration) -> None:
        """A place declaration (or its retraction) from dispatch."""
        self.place_declaration = decl if decl.active else None
        queries = () if decl.destination_query is None else (decl.destination_query,)
        self._set_goal("place", decl.target_id, decl.stamp_ns, decl.active, queries)

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
        binding = binder.on_attach(self.tracker.store, late or positions, stamp_ns)
        self._released.pop(binding.attachment.object_id, None)  # re-grasped before re-grounding
        self._monitors[gripper_joint] = JawReleaseMonitor.for_hand(
            binder.hand, hold=hold, release_dwell_s=self.release_dwell_s
        )
        self._attach_ns[gripper_joint] = stamp_ns
        self.attachment_changes += 1
        if binding.reground:
            self._reground()
        return binding

    def _late_hold(  # noqa: PLR0911  # reason: one return per guard
        self,
        joint: str,
        binder: GraspBinder,
        positions: Mapping[str, float],
        stamp_ns: int,
    ) -> dict[str, float] | None:
        """The joint state of the hold's start when this ATTACH is late and safe to bind there.

        A policy lifts within a fraction of a second of the jaw settling, so the ATTACH (the
        HAL's trigger or the stalled-jaw dwell) often lands after the object has begun to
        move and its track has been split or lost (#418 r12: ATTACH +0.3 to +2 s after the lift,
        13 of 13 DROID/restock binds fell back to the 2-4x PINCH_FALLBACK box). Binding the
        track as fitted at the hand pose NOW would put the payload where the object was, not
        where it is. Binding it at the hand pose of the hold's start (``pose_in_link`` from
        there) carries it with the hand, which is right if the grasp is rigid. Guarded so a
        wrong guess cannot under-cover the real object: the hand travelled at most
        ``_LATE_MAX_TRAVEL_M`` since the hold; exactly one track, last confirmed within
        ``_LATE_CONFIRM_NS`` of the hold, holds the pinch centre at the hold; and the box
        carried to the hand's pose now is at least ``_LATE_MIN_COVER`` occupied on the newest
        grid (an object left on the table fails this). Otherwise ``None``: the normal bind.
        A SPLIT / LOST candidate is revived to OCCLUDED for the bind.
        """
        detector = self._stall.get(joint)
        hold_ns = detector.hold_since_ns if detector is not None else None
        if hold_ns is None or not _LATE_MIN_NS <= stamp_ns - hold_ns <= _LATE_MAX_NS:
            return None
        if self.grid is None or not self._joint_history:
            return None
        t, sample = min(self._joint_history, key=lambda s: abs(s[0] - hold_ns))
        if abs(t - hold_ns) > _JOINT_SAMPLE_MAX_AGE_NS:
            return None
        hold_q = {**sample, joint: positions[joint]}
        now_c = binder.pinch_in_frame(positions)[0]
        hold_c = binder.pinch_in_frame(hold_q)[0]
        if float(np.linalg.norm(now_c - hold_c)) > _LATE_MAX_TRAVEL_M:
            return None
        store = self.tracker.store
        cands = [
            o
            for o in store.all()
            if o.status in _LATE_STATUSES
            and not o.provisional
            and o.last_confirmed_stamp_ns >= hold_ns - _LATE_CONFIRM_NS
            and _holds(o, hold_c, binder.cell_m)
        ]
        if len(cands) != 1:
            return None
        track = cands[0]
        t_now = binder.hand.t_frame_from_attach(self.frame_id, positions)
        t_hold = binder.hand.t_frame_from_attach(self.frame_id, hold_q)
        carried = t_now @ np.linalg.inv(t_hold) @ _t(track.box.pose)
        w, x, y, z = rotation_to_quat_wxyz(carried[:3, :3])
        region = PlaceRegion(
            frame_id=self.frame_id,
            pose=Pose6D(
                xyz=(float(carried[0, 3]), float(carried[1, 3]), float(carried[2, 3])),
                quat_xyzw=(x, y, z, w),
                frame_id=self.frame_id,
            ),
            half_extents=track.box.half_extents,
        )
        _, covered = region_covers_occupied(self.grid, region, min_fraction=_LATE_MIN_COVER)
        if not covered:
            log.info("object_tracker.late_attach_refused", hand=joint, object_id=track.object_id,
                     why="carried_box_not_occupied")  # fmt: skip
            return None
        if track.status in (TrackStatus.SPLIT, TrackStatus.LOST):
            store.replace(track.model_copy(update={"status": TrackStatus.OCCLUDED}))
        log.info("object_tracker.late_attach", hand=joint, object_id=track.object_id,
                 lag_s=(stamp_ns - hold_ns) / _NS)  # fmt: skip
        return hold_q

    def on_frame(
        self, frame: CameraFrame, *, wall_s: float, results: Sequence[TrackerResult] = ()
    ) -> TrackerOutput:
        """Feed one depth frame (and answers to requests issued on it) through the tracker."""
        self._tick(frame.stamp_ns)
        out = self.tracker.feed(
            frame,
            grid=self.grid,
            grid_stamp_ns=self.grid_stamp_ns,
            wall_s=wall_s,
            results=results,
            hand_spaces=self.hand_spaces(),
        )
        for oid, _fit in out.bound_fits:
            self._grow(oid)
        self._expire_released()
        self._prune_ambiguous()
        self._check_grounding()
        return out

    # -- outputs -----------------------------------------------------------------------

    def snapshot(self) -> TrackedObjectSet:
        """The whole track table at the newest source stamp (revision: the store's)."""
        return self.tracker.store.snapshot(self._now_ns)

    def attachments(self) -> list[AttachedCollisionObject]:
        """Every hand's held payload (joint order), then the released ones still on the wire."""
        held = [b.held.attachment for b in self.binders.values() if b.held is not None]
        return held + [record for record, _ in self._released.values()]

    def workspace_region(self) -> PlaceRegion | None:
        """The live grasp declaration's ``workspace_box``, map-confirmed, or ``None``.

        What the HAL producer confirms (``WORKSPACE_MIN_COVER`` of its footprint occupied
        on the newest grid, the same box re-stamped with the grid's ``source_stamp``),
        here for ``bind`` / ``on``, where the producer stands down: without it the
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
    ) -> None:
        key = declaration_key(target_id, stamp_ns)
        goal_id = f"{kind}:{key}"
        old = self._goals[kind]
        if old is not None and (old.goal_id != goal_id or not active):
            self.tracker.close_goal(old.goal_id)
            self._goals[kind] = None
            log.info("object_tracker.goal_closed", goal_id=old.goal_id)
        if not active or not queries or self._goals[kind] is not None:
            return
        goal = self._goals[kind] = _Goal(goal_id, key, None, tuple(queries))
        self._rekey(queries, key)
        if self._now_ns > 0:
            self._start(goal, self._now_ns)

    def _start(self, goal: _Goal, stamp_ns: int) -> None:
        """Open the goal's standing queries at source time ``stamp_ns``.

        A declaration that arrives before any source stamp starts at the first frame: a
        search timer anchored at 0 would exhaust on the first stamp it is compared with.
        """
        for i, query in enumerate(goal.queries):
            self.tracker.open_query(goal.goal_id, f"q{i}", query, goal.key, stamp_ns=stamp_ns)
        goal.start_ns = stamp_ns

    def _rekey(self, queries: Sequence[ObjectQuery], key: str) -> None:
        """Hand the live tracks a new goal's queries match to that goal's declaration."""
        store = self.tracker.store
        for t in store.all():
            if t.status is TrackStatus.LOST or t.declaration_key == key:
                continue
            if any(t.label == q.label and t.role is q.role for q in queries):
                store.replace(t.model_copy(update={"declaration_key": key}))

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

    def _release(self, joint: str, stamp_ns: int) -> None:
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
            pose = binder.object_pose(self.positions)
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
                    self._released[oid] = (held.attachment, stamp_ns)
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

    def _grow(self, object_id: str) -> None:
        """In-hand growth on a fit of the bound ``object_id`` (map protrusion)."""
        if self.grid is None or not self._joints <= self.positions.keys():
            return
        for binder in self.binders.values():
            held = binder.held
            if held is not None and held.attachment.object_id == object_id:
                grown = binder.grow_by_protrusion(
                    self.tracker.store, self.grid, self.positions, self._now_ns
                )
                if grown is not None:
                    self.attachment_changes += 1

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
