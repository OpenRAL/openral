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
   (``support_top_from_voxels``, HZ-0115-6); then the occupied cells of that
   column → one cluster above that plane, whose lowest cell must sit within one
   voxel (+ one of tolerance) of it (else ``not_on_support``) → its top-centre;
2. that seed, which must project into the depth camera through the driver's
   ``CameraInfo`` and tf2 ``optical <- base``, is sent to ``SegmentInView`` as
   the single positive point (no negatives), under the leg's own deadline;
3. the mask + the depth frame it was asked about → ``target_region_from_mask``
   (whose masked cloud must reach within two voxels of the support, else
   ``not_on_support``: a target stacked on another object)
   → ``region_covers_occupied`` against the latest map → ``track_region``
   against the previous accepted region → the whole-target gate: the region the
   kernel would get must hold the target's whole map component
   (``occupied_touching_outside``, else ``partial_fit``).

**Every failure is an outcome, never a guess** (HZ-0115-2/-3/-4/-6). Two
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
  **while a declared contact link's hand point (the bridge's TCP for the leg
  whose jaw link it is, ``VisionAttachmentBridge.jaw_point``) is within one voxel
  + ``occluder_margin_m`` of it** (``occluded_refit``: the robot's own hand
  occluding part of the target shrinks and shifts the fit; it never replaces
  the held region). The same shrink with no contact link near is
  ``unoccluded_refit``, a contradiction (a person's hand, the target knocked
  over). "Near" is measured over the target: the jaw's hand point (its TCP, a
  finger length above the fingertips) over the held footprint, up to the approach
  distance above its top. A fit refused outright with the hand there (what is left
  of an occluded target no longer reaches the support) is the lost view
  ``hand_at_target``. A fit that passed every gate above but whose cell-closed region
  leaves an occupied cell of the target's map component outside — a cell more than a
  voxel above the support, touching the region's cells (the far side the hovering hand
  hid from the head camera, Isaac i40/i43) — is the lost view ``partial_fit``; it is
  checked last, so it never changes another refusal's class. The gripper closing in on
  the target occludes it from the head camera exactly then, so the last accepted
  region is **frozen** for at most ``grasp_target_freeze_s`` (unset: twice
  ``grid_max_age_s``, the kernel's voxel deadline; never more — the kernel's
  ``grasp_region_max_age_s``) from its own ``stamp_ns``, then retracted. That
  stamp is the older of the depth frame it was measured on and the grid's
  ``source_stamp`` (the world data the map cover was checked against), so a
  stalled octomap — whose bridge keeps republishing a fresh ``header.stamp`` —
  ages the region out instead of vouching for it.

**What the kernel gets is cell-closed** (``GraspTargetLeg.fill``). The kernel exempts
a voxel only when the cell's centre lies in the region; a fit tight to the measured
surface leaves the target's own boundary cells (surface inside, centre outside)
unexempt, and the finger hull swept over the target's top edge stops on them. The
published region is ``cell_closed_region`` of the held fit against the newest grid:
grown by ``r/2·(|cos θ|+|sin θ|)`` (≤ ``r/√2``) horizontally and ``r/2`` up, never
down, so the support layer stays non-exempt. Every gate here (``_gate_refit``,
``region_within``, ``track_region``, the cover check, the hand tests) and the region
payload keep the tight fit.

The region dies with the declaration: dispatch retraction (goal end, cancel,
E-stop — the runner retracts on all of them) and ``timeout_s`` expiry clear it.
On an ATTACH of a leg whose jaw link the *armed* declaration names (the
approach-armed hand, or a named ``search_box`` declaration), re-measurement stops
and the region is kept for the kernel's handover rule (§2.1) — but only for an
ATTACH on what was measured (``GraspTargetLeg.on_attach``): the payload built from
the region itself, or a segmented payload lying in the region (+1 voxel) with the
jaw at it (``VisionAttachmentBridge.jaw_at``: its TCP within ``occluder_margin_m`` of
the region, or its jaw link's collision geometry, posed at the jaw's angle, overlapping
it — the TCP may be the finger hinge, a finger length above what the jaws close on).
Any other ATTACH (jaws closed on a neighbour) ends the arming as
``attach_off_target``: handed over with no region. Handover is per hand: an ATTACH
on a hand with nothing armed hands nothing over.

**Approach-armed target** (``approach_m`` set, default off; design §2.2
"Approach-armed target"). The policy, not the reasoner, decides what to pick, so
a live declaration need not name a target: one with no ``search_box`` (the
runner's goal-scope declaration, or a named one without a box) is measured
where the robot's own hand goes. Each tick, for every hand of the robot
(``openral_core.gripper_hands``) whose links the declaration names, the hand's
TCP points (``VisionAttachmentBridge.jaw_point``) span a gravity-aligned box
grown by ``approach_m`` (``approach_box``); a hand holding nothing is
*approaching* when that box holds at least ``grasp_target_min_cells`` occupied
cells above its lowest occupied layer (``approach_cell_count``: the table under
the hand is not a target). Exactly one
approaching hand arms a one-hand declaration (``target_id="approach:<first
link>:<n>"``, ``n`` the tracker's pick counter — seeded from the wall clock at
construction, advanced at every completed pick — ``contact_links`` = that hand, the
box as its ``search_box``, every other field — ``stamp_ns`` included — the goal's);
two at once arm none. The box then
follows the TCP and runs the measurement above unchanged; the hand leaving the
approach distance retracts the region at once (``approach_ended``) — but not while
one of its legs holds, releases or is segmenting a payload: its grasp is being
resolved, and the ATTACH that resolves it (after ``SegmentInView``, or at once from
the region) hands the arming over. A measurement in flight when the target changes
or is handed over is discarded (``GraspTargetTracker.generation``), its refusal or
region dropped, and a handed-over target takes no ``accept`` / ``refuse``. After a
retraction before any handover a hand re-arms only once it moved more than one
voxel or a freeze window elapsed, under the same identity — so one the kernel
retired for a fault stays refused — except a hand that left the approach distance
and was then seen away from it (located, its box off the column over its last target)
on every sample for a continuous freeze window: it approaches afresh, under a fresh
identity (liveness after a kernel fault retirement the producer cannot see — a grid
re-frame, a rejected attachment model). The kernel also retires an armed declaration at
every detach edge (the attachment set emptying at a new revision), before its handover
too and whichever hand let go; the bridge, which owns the set, calls
``GraspTargetLeg.on_attachment_changed`` before every publish whose set lost or
replaced an object or is empty, so the other hand's pre-handover arming drops its region,
takes a fresh identity in that very snapshot, and accepts only a region measured after
the change — except that a hand mid-grasp (an ATTACH being resolved: a segmentation in
flight, or jaws loaded with nothing latched) keeps its identity when the set stays
non-empty (no kernel retirement there; refreshing would fail the grasp), though it still
drops its region: that ATTACH is handed over with no region (fail closed).
**Multi-pick per goal**: once a hand is handed over nothing re-arms until that pick
completes. The bridge's DETACH on the hand
(``on_detach``, with the release window's frozen record) drops its region at once
while keeping the handover, so nothing re-latches it; once the hand's legs hold
nothing (no attachment, release window, pending segmentation or loaded trigger —
re-checked on every DETACH, the window's close and every barrier release, never polled) the
pick is complete: the counter
advances and the hand may re-arm under a fresh identity behind the same backoff —
but not on the payload it just released: until the column its next measurement would
search clears that payload's frozen pose by a voxel it does not arm, and a pick whose
release left no record blocks it for the rest of the goal (HZ-0115-11). The kernel retires every
pick's identity at its release and never re-arms one it retired (HZ-0115-3). A
named ``search_box`` wins: no approach detection runs for it, and it stays handed
over after its pick.

Threading: every entry point here (tick, reply, deadline, declaration, teardown) and every
bridge hook runs under the vision bridge's one re-entrant lock (``VisionAttachmentBridge``
"Threading") — the bridge's ATTACH runs on the HAL's proprio thread, the leg's timers
and replies on the executor. The tracker's own lock nests strictly inside it. Every read
or write of leg, tracker or bridge state (the request in flight and its deadline timer,
tf2, the depth and grid caches) happens under it; only pure work on a snapshot runs
outside — the tick's support/seed scan of the grid column (``_seed``) and the reply's mask
decode, back-projection and gates (``_measure``) — and is committed under the lock only
while the tracker generation is the one snapshotted (and, for a request, while the leg is
not torn down: ``teardown`` sets a flag under the lock that ``_tick`` and ``_on_deadline``
check, so no timer outlives it). The voxel grid is the bridge's (``_on_voxels``: one
subscription and decode shared with the place leg); its whole-grid occupied-cell scan runs
on first use and is cached on the lattice.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager
from functools import update_wrapper
from typing import TYPE_CHECKING, Any, Concatenate, ParamSpec, Protocol, TypeVar

import numpy as np
from numpy.typing import NDArray
from openral_core import GraspDeclaration, PlaceRegion, Pose6D, gripper_hands
from openral_core.exceptions import ROSConfigError, ROSSafetyViolation
from openral_core.geometry import homogeneous_from_quat_xyz

from openral_hal._grasp_target import (
    TargetRefusal,
    VoxelLattice,
    _in_region,
    cell_closed_region,
    mask_without_removed_points,
    occupied_centers_in_box,
    occupied_touching_outside,
    project_point,
    region_covers_occupied,
    region_within,
    support_top_from_voxels,
    target_region_from_mask,
    target_seed_from_voxels,
    track_region,
)

if TYPE_CHECKING:  # pragma: no cover — typing only
    from openral_core import AttachedCollisionObject, IntrinsicsPinhole

    from openral_hal.vision_attachment_bridge import VisionAttachmentBridge, VisionAttachmentConfig

__all__ = [
    "APPROACH_TARGET_PREFIX",
    "GraspTargetLeg",
    "GraspTargetTracker",
    "approach_box",
    "approach_cell_count",
    "lattice_from_msg",
    "search_column",
]

#: ``target_id`` prefix of an approach-armed declaration; the hand's first link follows.
APPROACH_TARGET_PREFIX = "approach:"

#: The design's re-prompt band, Hz (§2.2).
_RATE_BAND_HZ = (2.0, 5.0)

#: Tilt tolerance for the search box: its z axis must be the base frame's.
_GRAVITY_TOL = 1e-6

# Upper bounds that keep the gates meaningful (CLAUDE.md §1.2: chosen, not measured).
#: A contact link's hand point this far from the held region is not plausibly the hand
#: occluding it; a larger margin lets any nearby arm pose excuse a shrunken re-fit.
_MAX_OCCLUDER_MARGIN_M = 0.10
#: Deeper than this, the column reaches whole furniture levels below the target (a
#: bench under a shelf) and the scan's "first ringed layer" stops meaning "under it".
_MAX_SUPPORT_SEARCH_BELOW_M = 0.5
#: The freeze holds a region with no fresh evidence; the kernel ages a grasp region out
#: at ``grasp_region_max_age_s`` = 2 x its voxel deadline (deploy passes 2 x
#: ``world_voxel_deadline_s``; ``grid_max_age_s`` is that deadline), so a longer freeze
#: would publish a region the kernel already refuses (the place leg's ceiling too).
_MAX_FREEZE_GRID_AGES = 2.0


_P = ParamSpec("_P")
_R = TypeVar("_R")


class _Serialized(Protocol):
    """Anything with a ``_lock`` — the tracker's own, or the vision bridge's (legs alias it)."""

    @property
    def _lock(self) -> AbstractContextManager[Any]: ...


_T = TypeVar("_T", bound=_Serialized)


def _locked(method: Callable[Concatenate[_T, _P], _R]) -> Callable[Concatenate[_T, _P], _R]:
    """Run a method under its object's ``_lock`` (re-entrant in every use here)."""

    def locked(self: _T, /, *args: _P.args, **kwargs: _P.kwargs) -> _R:
        with self._lock:
            return method(self, *args, **kwargs)

    update_wrapper(locked, method)  # keeps the docstring (doctests, docs)
    return locked


class GraspTargetTracker:
    """The leg's state machine — declaration, accepted region, freeze TTL. Pure.

    No ROS, no clock: every method takes the caller's ``now_ns`` (the node's ROS
    clock, the same domain as the runner's ``stamp_ns``). ``log`` receives one
    line per state transition, never one per tick.

    Thread-safe on its own: every method runs under the tracker's lock. In the HAL
    every caller also holds the vision bridge's lock (``VisionAttachmentBridge``
    "Threading"), always taken first: bridge lock -> tracker lock, never the reverse —
    nothing called under the tracker lock (``log``, ``on_attach``'s ``confirm``) takes
    the bridge lock.

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

    def __init__(
        self, *, freeze_s: float, log: Callable[[str], None], first_pick: int | None = None
    ) -> None:
        """Start with no declaration.

        ``first_pick`` seeds the pick counter in ``approach:<link>:<n>``; ``None`` (the
        default) seeds it from the wall clock in nanoseconds, like the bridge's
        attachment revision, so a tracker built by a re-activated HAL mid-goal never
        re-mints an identity the kernel already retired.
        """
        self._freeze_ns = int(freeze_s * 1e9)
        # ponytail: wall-clock seed; a clock stepped back across a HAL re-activation by
        # more than the picks since could reuse an identity — the kernel's retired set
        # still refuses it (fail closed).
        self._pick = time.time_ns() if first_pick is None else first_pick
        # Whether an approach identity was armed under the current ``_pick`` (and so may
        # have reached the kernel, which retires it at a detach edge): the next attachment
        # change advances the counter even when that arming has since retracted.
        self._minted = False
        self._log = log
        # ponytail: one re-entrant lock for the whole tracker; every call is O(1)
        # bookkeeping, so contention is negligible next to the 2-5 Hz measurement.
        self._lock = threading.RLock()
        self._declaration: GraspDeclaration | None = None
        # The one-hand declaration armed from the robot's own approach, or ``None``.
        self._approach: GraspDeclaration | None = None
        self._region: PlaceRegion | None = None
        # The links of the hand whose ATTACH ended re-measurement, or ``None``. Held
        # until that hand holds nothing again (``on_pick_complete``) or the
        # declaration changes.
        self._handed_over: tuple[str, ...] | None = None
        # Per hand whose approach-armed pick completed this goal: the released
        # payload's frozen record (base frame), or ``None`` when no record was taken.
        self._spent: dict[tuple[str, ...], AttachedCollisionObject | None] = {}
        # Per hand: (approach box centre, now_ns, the hand left the approach distance)
        # of its last refused/ended arming.
        self._backoff: dict[tuple[str, ...], tuple[tuple[float, float, float], int, bool]] = {}
        # Per hand that left the approach distance: since when every sample saw it
        # located and away from its last target's column (``_track_away``).
        self._away_since: dict[tuple[str, ...], int] = {}
        # Hands away continuously for a freeze window: their next arming takes a fresh
        # identity (``on_approach``).
        self._away: set[tuple[str, ...]] = set()
        # When the attachment set last lost or replaced an object (or published empty):
        # no region measured at or before it is accepted (``on_attachment_changed``).
        self._changed_ns = 0
        self._status = ""
        # Bumped whenever what a measurement in flight was asked for stops being what is
        # measured (new/cleared declaration, arming, retraction, handover).
        self._generation = 0

    @property
    def declaration(self) -> GraspDeclaration | None:
        """Dispatch's live declaration (region-less), or ``None``."""
        return self._declaration

    @property
    def target(self) -> GraspDeclaration | None:
        """What is measured and published: the approach-armed declaration, else dispatch's."""
        with self._lock:
            return self._approach if self._approach is not None else self._declaration

    def measured(self) -> tuple[GraspDeclaration | None, int]:
        """``(target, generation)`` read together, for tagging one measurement request."""
        with self._lock:
            return self.target, self._generation

    @property
    def region(self) -> PlaceRegion | None:
        """The last accepted region still held, or ``None``."""
        return self._region

    @property
    def handed_over(self) -> tuple[str, ...] | None:
        """The links of the hand whose ATTACH ended re-measurement, or ``None``."""
        return self._handed_over

    @property
    def generation(self) -> int:
        """Tag a measurement request with this; a reply under another one is stale."""
        return self._generation

    def _transition(self, status: str, line: str) -> None:
        if status != self._status:
            self._status = status
            self._log(line)

    def _retract(self, kind: str, detail: str, *, now_ns: int) -> None:
        self._generation += 1
        had = self._region is not None
        self._region = None
        # An approach-armed target exists only for its region; it re-arms from scratch,
        # but not before the hand moves or a backoff elapses (``on_approach``).
        held = self._approach
        if held is not None and held.search_box is not None:
            self._away_since.pop(held.contact_links, None)  # each leaving starts its own window
            self._backoff[held.contact_links] = (
                held.search_box.pose.xyz,
                now_ns,
                kind == "approach_ended",
            )
        self._approach = None
        self._transition(
            f"retracted:{kind}",
            f"grasp target region retracted — {kind}: {detail}"
            if had
            else f"grasp target region refused — {kind}: {detail}",
        )

    @_locked
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
        self._generation += 1
        self._region = None
        self._approach = None
        self._handed_over = None
        self._backoff.clear()
        self._away.clear()
        self._away_since.clear()
        self._spent.clear()
        self._declaration = declaration.model_copy(update={"region": None})
        self._transition(
            f"declared:{declaration.target_id}:{declaration.stamp_ns}",
            f"grasp target declared {declaration.target_id!r} "
            f"search_box={'yes' if declaration.search_box is not None else 'none'} "
            f"rskill={declaration.rskill_id!r} trace={declaration.trace_id!r}",
        )

    def _clear(self, why: str) -> None:
        self._generation += 1
        self._declaration = None
        self._approach = None
        self._region = None
        self._handed_over = None
        self._backoff.clear()
        self._away.clear()
        self._away_since.clear()
        self._spent.clear()
        self._transition(f"cleared:{why}", f"grasp target cleared — {why}")

    @_locked
    def wants_approach(self, *, now_ns: int) -> bool:
        """Whether approach detection should run: a live declaration with no search box."""
        declaration = self._declaration
        return (
            declaration is not None
            and declaration.search_box is None
            and self._handed_over is None
            and declaration.is_live(now_ns=now_ns)
        )

    @_locked
    def on_approach(
        self,
        near: Sequence[tuple[tuple[str, ...], PlaceRegion]],
        *,
        now_ns: int,
        move_m: float,
        holding: Sequence[tuple[str, ...]] = (),
        generation: int | None = None,
        located: Mapping[tuple[str, ...], PlaceRegion] | None = None,
    ) -> None:
        """Arm, follow or end the approach-armed target from this tick's approaching hands.

        Args:
            near: ``(hand links, approach box)`` for every hand whose TCP is within
                the approach distance of occupied cells this tick.
            now_ns: The caller's clock (backoff).
            move_m: How far a hand's approach box centre must move from where its
                last arming was refused or ended before it re-arms inside the
                backoff (the leg passes one voxel).
            holding: Hands whose legs hold, release or are segmenting a payload. The
                held hand among them keeps its arming and box unchanged (no
                retraction, no backoff): its grasp is being resolved, and the
                ATTACH that resolves it hands the region over (``on_attach``). Only
                the hand holding nothing again — no attachment — ends it, on a
                later tick, by leaving the approach distance.
            generation: The tracker generation ``near`` was computed under; ``None``
                skips the check. A pick completing on another thread since (its
                released payload is now guarded, ``spent``) makes ``near`` stale.
            located: The approach box of every hand located this tick and holding
                nothing (``near`` or not); ``None`` = none. Only these samples can
                show a hand away (``_track_away``).

        Only hands whose links the declaration names count (a named declaration
        narrows the choice). The held hand absent from ``near`` retracts at once
        (``approach_ended``); otherwise its search box follows the TCP. With none
        held, exactly one approaching hand arms; two at once arm none. A hand whose
        arming was retracted re-arms only once it moved more than ``move_m`` or a
        freeze window (``freeze_s``) elapsed: the same map at the same pose gives
        the same verdict, so re-arming every tick only floods the log and the
        segmenter. It keeps the pick's identity — unless its arming ended by
        leaving the approach distance (``approach_ended``) and every sample for a
        continuous freeze window since showed it located with its approach box off
        its last target's column (``_track_away``): then it arms under a fresh one.
        An unlocated sample (a tf2 gap), a holding one, or one back over the column
        restarts that window. The kernel's fault retirements (``grid_frame_changed``,
        ``attachment_rejected``) are invisible here; this is the one way out of them,
        and wiggling, a tf2 gap or coming straight back is not it.
        """
        goal = self._declaration
        if (
            goal is None
            or goal.search_box is not None
            or self._handed_over is not None
            or (generation is not None and generation != self._generation)
        ):
            self._away_since.clear()  # no sample: unknown is not away
            return
        named = set(goal.contact_links)
        near = [(links, box) for links, box in near if set(links) <= named]
        self._track_away(located or {}, now_ns=now_ns)
        held = self._approach
        if held is not None:
            # Mid-grasp (``holding``) the arming stays as it is: what is near the TCP
            # now is what the hand closes on, and its ATTACH hands the region over.
            box = next((b for links, b in near if links == held.contact_links), None)
            if held.contact_links in holding:
                pass
            elif box is None:
                self._retract(
                    "approach_ended",
                    f"{held.target_id!r}: the hand's TCP left the approach distance "
                    "(or is no longer located)",
                    now_ns=now_ns,
                )
            else:
                self._approach = held.model_copy(update={"search_box": box})
            return
        if len(near) > 1:
            self._transition(
                "approach_ambiguous",
                f"grasp target: {len(near)} hands approaching at once "
                f"{[list(links) for links, _ in near]}; arming none (one hand at a time)",
            )
            return
        if not near:
            return
        links, box = near[0]
        backoff = self._backoff.get(links)
        if backoff is not None:
            where, since_ns, _ = backoff
            moved = float(np.linalg.norm(np.subtract(box.pose.xyz, where)))
            if moved <= move_m and now_ns - since_ns <= self._freeze_ns:
                return  # same place, same map: the refusal would repeat
            del self._backoff[links]
        fresh = links in self._away
        self._away.discard(links)
        self._away_since.pop(links, None)
        if fresh:
            # Liveness after a fault the kernel retired (HZ-0115-3): the producer
            # cannot see a retirement, and a re-arm keeps the pick's identity, so a
            # grid re-frame or a rejected attachment model would refuse this pick for
            # the rest of the goal. A hand that left the approach distance and was
            # still away from it once a freeze window had passed approaches afresh: a
            # new identity, measured from scratch. Wiggling in place or coming
            # straight back keeps the retired one.
            self._advance()
        self._generation += 1
        self._minted = True
        self._approach = goal.model_copy(
            update={
                # One identity per pick, ``stamp_ns`` the goal's (the kernel's
                # timeout_s backstop runs from it): the kernel retires each pick's
                # identity at its release, and the next pick arms under a fresh one
                # (``on_pick_complete``). A re-arm after a pre-handover retraction
                # keeps the pick's identity, so one the kernel retired for a fault
                # stays refused as retired — until the hand left the approach
                # distance and backed off (above).
                "target_id": f"{APPROACH_TARGET_PREFIX}{links[0]}:{self._pick}",
                "contact_links": links,
                "search_box": box,
                "region": None,
            }
        )
        self._transition(
            f"approach:{links[0]}",
            f"grasp target armed from the approach of {list(links)}: search box centre "
            f"{tuple(round(v, 3) for v in box.pose.xyz)} half_extents "
            f"{tuple(round(v, 3) for v in box.half_extents)} "
            f"rskill={goal.rskill_id!r} trace={goal.trace_id!r}",
        )

    def _track_away(self, located: Mapping[tuple[str, ...], PlaceRegion], *, now_ns: int) -> None:
        """Latch every hand continuously away from its last target for a freeze window.

        Away is a positive observation: the hand located, holding nothing, its approach
        box no longer containing the column over its last arming's search-box centre.
        Anything else — unlocated, holding, or over the column again — restarts the
        window; absence from ``near`` alone (a ``min_cells`` dip) is not away.
        """
        for links, (where, _, left) in self._backoff.items():
            if not left:
                continue
            box = located.get(links)
            column = None if box is None else (where[0], where[1], box.pose.xyz[2])
            if box is None or bool(_in_region(np.asarray([column], dtype=np.float64), box)[0]):
                self._away_since.pop(links, None)
                continue
            since_ns = self._away_since.setdefault(links, now_ns)
            if now_ns - since_ns >= self._freeze_ns:
                self._away.add(links)

    @_locked
    def presence_unknown(self) -> None:
        """A tick that could not sample the hands: every away window restarts.

        No fresh grid, a request in flight, a stale verdict: unknown is not away.
        """
        self._away_since.clear()

    def _advance(self) -> None:
        self._pick += 1
        self._minted = False

    def _armed(self) -> GraspDeclaration | None:
        """What is being measured for a hand: the approach arming, else a named search box."""
        if self._approach is not None:
            return self._approach
        declaration = self._declaration
        return declaration if declaration is not None and declaration.search_box else None

    @_locked
    def on_attach(
        self, contact_link: str, *, confirm: Callable[[PlaceRegion | None], bool]
    ) -> None:
        """A leg attached; if the armed declaration names its jaw link, stop re-measuring.

        Scoped to the hand: only an ATTACH on a link of the *armed* declaration (the
        approach-armed hand, or a named ``search_box`` declaration) hands over. The
        goal-scope declaration names every hand before any approach arms, so an
        ATTACH there — a false positive, or a grasp of something never measured —
        neither hands over nor stops the other hand's approach detection.

        Args:
            contact_link: The attaching leg's jaw link.
            confirm: Called, under the lock, with the region held now (``None`` when
                none is); ``True`` only when the attached payload is that region's
                target (the leg's ``GraspTargetLeg.on_attach``: the payload *is* the
                region — so ``None``, the region it was built from gone, is not — or a
                segmented payload lies in it with the jaw at it, or no region is held
                for it to contradict). ``False`` ends the arming as
                ``attach_off_target``: the hand is handed over with **no** region — the
                region is dropped, never handed to the kernel for a payload it was not
                measured for, and re-measurement stops (the hand holds something else).
                An exception from ``confirm`` is the same ``attach_off_target``
                (evaluated before any state changes). Either way the hand stays handed
                over until it holds nothing again (``on_pick_complete``).
        """
        if self._handed_over is not None:
            return
        armed = self._armed()
        if armed is None or contact_link not in armed.contact_links:
            if self._declaration is not None:
                self._transition(
                    f"attach_unarmed:{contact_link}",
                    f"grasp target: ATTACH on {contact_link!r} with no armed target for its "
                    "hand — nothing handed over, approach detection continues",
                )
            return
        region = self._region
        try:
            on_target = confirm(region)
            why = "is not on the measured region"
        except ROSSafetyViolation:
            raise
        except Exception as exc:  # tf2/geometry failing in ``confirm`` is never "on target"
            on_target = False
            why = f"could not be confirmed ({type(exc).__name__}: {exc})"
        self._handed_over = armed.contact_links
        self._generation += 1  # a measurement in flight was asked before the grasp
        if not on_target:
            self._region = None
            where = (
                "no region held"
                if region is None
                else f"centre {tuple(round(v, 3) for v in region.pose.xyz)}"
            )
            self._transition(
                "attach_off_target",
                f"grasp target {armed.target_id!r}: ATTACH on {contact_link!r} {why} "
                f"({where}) — attach_off_target: region dropped, nothing handed over",
            )
            return
        self._transition(
            "handed_over",
            f"grasp target {armed.target_id!r}: ATTACH on {contact_link!r} — region "
            f"{'kept' if region is not None else 'absent'}, no further re-measurement or "
            "approach arming until the hand holds nothing",
        )

    @_locked
    def on_release(self, hand: tuple[str, ...], record: AttachedCollisionObject | None) -> None:
        """A DETACH on the handed-over hand: drop its region now, keep the handover.

        The kernel retires the pick's declaration at the release (HZ-0115-3), and
        the region is never offered again: it is dropped at once, while the handover
        record stays so nothing re-measures or re-latches it — the hand may still be
        in its release window. ``record`` (the released payload frozen in the base
        frame, ``None`` when none was taken) is what ``spent`` guards the hand's
        next arming against (HZ-0115-11). A named ``search_box`` declaration and a
        hand not handed over are untouched.
        """
        goal = self._declaration
        if self._handed_over != hand or goal is None or goal.search_box is not None:
            return
        self._generation += 1
        self._region = None
        self._spent[hand] = record
        self._transition(
            "released",
            f"grasp target {(self._approach or goal).target_id!r}: {list(hand)} released "
            "its payload — region dropped",
        )

    @_locked
    def on_attachment_changed(
        self, *, now_ns: int, emptied: bool = True, holding: Sequence[str] = ()
    ) -> None:
        """The attachment set lost or replaced an object, or was published empty.

        The kernel retires an armed grasp declaration whenever the set empties at a new
        revision — before or after its handover, whichever hand let go (HZ-0115-3) —
        and never re-arms a retired identity. The producer sees every such change (it
        owns the set), so it keeps the pick live here: an arming not handed over drops
        its region at once and, approach-armed, advances to a fresh identity; and no
        region measured at or before ``now_ns`` (its ``stamp_ns``: the older of its
        depth frame and its grid's ``source_stamp``) is accepted afterwards, for any
        arming. The counter advances too when the arming has retracted since its
        identity was armed (the kernel may still hold it, and retires it at this edge),
        so the next re-arm is never a retired identity. A handed-over arming follows its
        own pick-complete path. A named ``search_box`` declaration keeps dispatch's
        identity: the kernel refuses it once retired (fail closed), and dispatch must
        redeclare.

        Args:
            now_ns: The change instant (the node clock).
            emptied: Whether the set is now empty — the kernel's retirement edge.
            holding: Links of every hand with an ATTACH being resolved — a
                segmentation in flight, or jaws read loaded with nothing latched yet
                (``GraspTargetLeg._resolving``) — not a carried payload or a release
                window. An arming whose hand is among them is mid-grasp: across a
                change that leaves the set non-empty it keeps its identity (the kernel
                does not retire there; refreshing would fail the grasp) but still
                drops its region, measured before the change — the ATTACH resolving it
                is handed over with no region (fail closed). On an emptying change it
                is refreshed regardless: the kernel retires it, so that grasp hands
                nothing over (fail closed).
        """
        self._changed_ns = max(self._changed_ns, now_ns)
        if self._handed_over is not None:
            return
        armed = self._armed()
        self._generation += 1  # a measurement in flight saw the old scene
        self._region = None
        if not emptied and armed is not None and set(armed.contact_links) & set(holding):
            self._transition(
                f"kept_mid_grasp:{armed.target_id}",
                f"grasp target {armed.target_id!r}: the attachment set changed (not emptied) "
                "while its hand is mid-grasp — identity kept, region dropped",
            )
            return
        held = self._approach
        if held is None:
            if self._minted:
                self._advance()
            return
        self._advance()
        self._minted = True
        self._approach = held.model_copy(
            update={"target_id": f"{APPROACH_TARGET_PREFIX}{held.contact_links[0]}:{self._pick}"}
        )
        self._transition(
            f"refreshed:{self._approach.target_id}",
            f"grasp target {held.target_id!r}: the attachment set changed before its handover "
            f"— region dropped, re-armed as {self._approach.target_id!r} for a measurement "
            "taken after the change",
        )

    @_locked
    def on_pick_complete(self, hand: tuple[str, ...], *, now_ns: int) -> bool:
        """The handed-over hand holds nothing any more: end its approach-armed pick.

        Called by the leg when the hand's legs hold no attachment, release window or
        pending segmentation. The arming is dropped, the pick counter advances (the
        next arming has a fresh ``target_id``) and the hand may re-arm through
        ``on_approach`` behind the same backoff as a refusal — never on the payload
        it released (``spent``; a pick with no ``on_release`` record keeps the hand
        from arming for the rest of the goal). A named ``search_box`` declaration
        stays handed over: a new target needs a new declaration from dispatch.

        Returns:
            Whether a pick completed.
        """
        goal = self._declaration
        if self._handed_over != hand or goal is None or goal.search_box is not None:
            return False
        self._spent.setdefault(hand, None)
        self._advance()
        held = self._approach
        self._retract(
            "picked",
            f"{(held or goal).target_id!r}: {list(hand)} holds nothing; may re-arm for "
            "the next pick",
            now_ns=now_ns,
        )
        self._handed_over = None
        return True

    @_locked
    def spent(self, hand: tuple[str, ...]) -> tuple[bool, AttachedCollisionObject | None]:
        """``(guarded, released record)`` for ``hand``'s last completed pick this goal."""
        return hand in self._spent, self._spent.get(hand)

    @_locked
    def clear_spent(self, hand: tuple[str, ...], record: AttachedCollisionObject | None) -> None:
        """End ``hand``'s guard — only if ``record`` is still the one guarded. Logged once."""
        if hand in self._spent and self._spent[hand] is record:
            del self._spent[hand]
            self._log(f"grasp target: {list(hand)} cleared the payload it released; may arm again")

    @_locked
    def wants_measurement(self, *, now_ns: int) -> bool:
        """Whether a measurement tick should run: live, searchable, not handed over."""
        declaration = self.target
        return (
            declaration is not None
            and declaration.search_box is not None
            and self._handed_over is None
            and declaration.is_live(now_ns=now_ns)
        )

    @_locked
    def accept(self, region: PlaceRegion, *, generation: int | None = None) -> None:
        """Hold a region that passed every gate; a declaration-cap violation refuses it.

        Ignored once handed over (the region the ATTACH took is frozen) and, given
        ``generation`` (the request's), when the target changed since — checked
        atomically with the hold.
        """
        declaration = self.target
        if declaration is None or self._handed_over is not None:
            return
        if generation is not None and generation != self._generation:
            return
        if region.stamp_ns <= self._changed_ns:
            # Measured on a depth frame or grid from before the attachment set changed:
            # not the scene the kernel now holds (HZ-0115-3). Dropped; the next tick
            # re-measures on newer inputs.
            self._transition(
                "measured_before_change",
                f"grasp target region for {declaration.target_id!r} dropped — measured at "
                f"{region.stamp_ns}, not after the attachment set changed at {self._changed_ns}",
            )
            return
        try:
            GraspDeclaration.model_validate(declaration.model_dump() | {"region": region})
        except ValueError as exc:
            self._retract("declaration_bounds", str(exc).splitlines()[0], now_ns=region.stamp_ns)
            return
        self._region = region
        qx, qy, qz, qw = region.pose.quat_xyzw
        yaw_deg = math.degrees(
            math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
        )
        self._transition(
            "accepted",
            f"grasp target region accepted for {declaration.target_id!r}: "
            f"centre={tuple(round(v, 3) for v in region.pose.xyz)} "
            f"half_extents={tuple(round(v, 3) for v in region.half_extents)} "
            # The box is yaw-only; without its yaw a stopped cell near its edge
            # cannot be placed inside or outside it after the fact.
            f"yaw_deg={yaw_deg:.1f} "
            f"evidence={region.evidence_ref!r}",
        )

    @_locked
    def refuse(
        self, kind: str, detail: str, *, retract: bool, now_ns: int, generation: int | None = None
    ) -> None:
        """One failed measurement: retract now, or freeze the held region under its TTL.

        Ignored once handed over: no measurement runs for a hand that grasped, so a
        refusal is a stale one and must not retract the handed-over arming. Ignored
        too, given ``generation``, when the target changed since it was asked.
        """
        if self._handed_over is not None:
            return
        if generation is not None and generation != self._generation:
            return
        if retract or self._region is None:
            self._retract(kind, detail, now_ns=now_ns)
            return
        self._transition(
            f"frozen:{kind}",
            f"grasp target view lost — {kind}: {detail}; holding the last region for at "
            f"most {self._freeze_ns / 1e9:.1f} s from its stamp",
        )
        self._expire_frozen(now_ns)

    def _expire_frozen(self, now_ns: int) -> None:
        region = self._region
        if region is None or self._handed_over is not None:
            return
        if now_ns - region.stamp_ns > self._freeze_ns:
            self._retract(
                "freeze_ttl",
                f"last accepted region is {(now_ns - region.stamp_ns) / 1e9:.2f} s old "
                f"(> {self._freeze_ns / 1e9:.1f} s)",
                now_ns=now_ns,
            )

    @_locked
    def envelope(self, *, now_ns: int) -> GraspDeclaration | None:
        """The declaration to put on the attachment envelope now, or ``None``.

        Dispatch's fields verbatim plus the held region — or, while an
        approach-armed target is held, that one-hand declaration (the goal's
        attribution, its own ``target_id`` / ``contact_links`` / search box);
        expiry clears it and an over-age region is retracted here too, so a
        stalled measurement loop cannot keep one alive.
        """
        declaration = self._declaration
        if declaration is None:
            return None
        if not declaration.is_live(now_ns=now_ns):
            self._clear(f"{declaration.target_id!r} expired (timeout_s={declaration.timeout_s})")
            return None
        self._expire_frozen(now_ns)
        target = self.target
        assert target is not None
        return target.model_copy(update={"region": self._region})


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
            vertical column to find a horizontal support in (HZ-0115-6).

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


def approach_box(
    points: Sequence[tuple[float, float, float]], *, approach_m: float, frame_id: str
) -> PlaceRegion:
    """The gravity-aligned box a hand approaches in: its TCP points grown by ``approach_m``.

    Each half-extent is capped at ``GraspDeclaration.MAX_HALF_EXTENT_M``: the box is
    both the approach test (occupied cells inside it) and the approach-armed
    target's search box, which seeds no more than one graspable object.

    Raises:
        ROSConfigError: On no points.

    Example:
        >>> box = approach_box([(0.4, 0.1, 0.2), (0.4, 0.14, 0.2)], approach_m=0.1, frame_id="b")
        >>> box.pose.xyz, box.half_extents
        ((0.4, 0.12, 0.2), (0.1, 0.12, 0.1))
    """
    if not points:
        raise ROSConfigError("approach_box needs at least one TCP point.")
    lo = np.min(np.asarray(points, dtype=np.float64), axis=0)
    hi = np.max(np.asarray(points, dtype=np.float64), axis=0)
    cap = GraspDeclaration.MAX_HALF_EXTENT_M
    centre = tuple(round(float(v), 9) for v in (lo + hi) / 2.0)
    half = tuple(round(min(float(v) + approach_m, cap), 9) for v in (hi - lo) / 2.0)
    return PlaceRegion(
        frame_id=frame_id,
        pose=Pose6D(xyz=centre, quat_xyzw=(0.0, 0.0, 0.0, 1.0), frame_id=frame_id),
        half_extents=half,
        evidence_ref="approach:tcp",
    )


def approach_cell_count(grid: VoxelLattice, box: PlaceRegion) -> int:
    """Occupied cells in an approach box that are not the surface under the hand.

    The box's lowest occupied layer — a table or shelf board the hand works over —
    is not counted: a hand low over an empty table holds only that, and would
    otherwise arm, refuse and re-arm every tick. Only that one layer: the box's
    bottom face can cut through a small object the hand is over, whose lowest
    layer in the box is then the object's own.

    Example:
        >>> import numpy as np
        >>> def grid(occ: np.ndarray) -> VoxelLattice:
        ...     return VoxelLattice("b", (0, 0, 0), (0, 0, 0, 1), 0.05, (5, 5, 4), occ)
        >>> table = np.zeros(5 * 5 * 4, dtype=np.uint8)
        >>> table[:25] = 1  # the whole k=0 layer: a table top
        >>> box = approach_box([(0.125, 0.125, 0.1)], approach_m=0.15, frame_id="b")
        >>> approach_cell_count(grid(table), box)
        0
        >>> post = table.copy()
        >>> post[[25 + 12, 50 + 12, 75 + 12]] = 1  # a 3-cell post on it at i=j=2
        >>> approach_cell_count(grid(post), box)
        3
    """
    centers = occupied_centers_in_box(grid, box)
    if len(centers) == 0:
        return 0
    # ponytail: the lowest layer is a heuristic, not a measured support; a surface two
    # or more layers thick (or tilted across layers) inside the box still counts, and
    # the refusal backoff (``GraspTargetTracker.on_approach``) bounds what that costs.
    floor = float(centers[:, 2].min()) + 0.5 * grid.resolution
    return int((centers[:, 2] > floor).sum())


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


def _hand_over_target(
    hands: Sequence[tuple[float, float, float]],
    held: PlaceRegion,
    *,
    reach_m: float,
    hand_rise_m: float,
) -> bool:
    """A declared contact link's hand point over the held target, within ``hand_rise_m``.

    The hand point is the jaw's TCP (the finger hinge), a finger length above the
    fingertips, so "near the target" is measured over its footprint: the held region
    raised by ``hand_rise_m`` (the approach distance that armed it), grown by ``reach_m``.
    """
    x, y, z = held.pose.xyz
    hx, hy, hz = held.half_extents
    column = held.model_copy(
        update={
            "pose": held.pose.model_copy(update={"xyz": (x, y, z + hand_rise_m / 2.0)}),
            "half_extents": (hx, hy, hz + hand_rise_m / 2.0),
        }
    )
    return _hand_near(hands, column, reach_m=reach_m)


def _refused_fit(
    kind: str,
    detail: str,
    previous: PlaceRegion | None,
    hands: Sequence[tuple[float, float, float]],
    *,
    reach_m: float,
    hand_rise_m: float,
) -> _Refusal:
    """A re-fit ``target_region_from_mask`` refused: a lost view or a contradiction.

    Too few points is a lost view. So is any refusal with a region held and a declared
    contact link's hand over it (``hand_at_target``): the closing hand occludes the
    target's lower part, so what is left no longer reaches the support. The held region
    stays, never replaced, under its freeze TTL until the stall's ATTACH hands over.
    Before anything is held, or with the hand elsewhere, it is the contradiction it names.
    """
    if kind == TargetRefusal.TOO_FEW_POINTS.value:
        return _lost(kind, detail)
    if previous is not None and _hand_over_target(
        hands, previous, reach_m=reach_m, hand_rise_m=hand_rise_m
    ):
        return _lost("hand_at_target", f"{kind}: {detail}, hand over the held target")
    return _contradicted(kind, detail)


def _kernel_closure(region: PlaceRegion, grid: VoxelLattice) -> tuple[PlaceRegion, str]:
    """``region`` as the kernel gets it over ``grid``, and a note when not cell-closed.

    ``cell_closed_region`` clamped at ``GraspDeclaration.MAX_HALF_EXTENT_M`` (noted); a
    closure over ``GraspDeclaration.MAX_VOLUME_M3`` is the tight fit (noted). Raises
    ``ROSConfigError`` as ``cell_closed_region`` does (another frame, a tilted region).
    """
    closed, clamped = cell_closed_region(
        region, grid, max_half_extent_m=GraspDeclaration.MAX_HALF_EXTENT_M
    )
    if closed.volume_m3() > GraspDeclaration.MAX_VOLUME_M3:
        return region, (
            f"published tight: the closure exceeds the {GraspDeclaration.MAX_VOLUME_M3} "
            "m^3 volume cap"
        )
    if clamped:
        return closed, f"clamped at the {GraspDeclaration.MAX_HALF_EXTENT_M} m half-extent cap"
    return closed, ""


def _gate_refit(
    grid: VoxelLattice,
    region: PlaceRegion,
    previous: PlaceRegion | None,
    *,
    support_z: float,
    min_cover: float,
    hands: Sequence[tuple[float, float, float]] = (),
    occluder_margin_m: float = 0.05,
    hand_rise_m: float = 0.0,
) -> PlaceRegion:
    """Map cover + tracking gate + whole-target gate on a fresh fit, or the typed refusal.

    A re-fit the map does not cover is always a contradiction (the target is
    gone). A covered re-fit that fails tracking but shrank inside the held
    region (grown by one voxel) is the closing hand occluding part of the
    target — a lost view that keeps the held region under its freeze TTL —
    **only** while a declared contact link (``hands``, base-frame positions) is
    within one voxel + ``occluder_margin_m`` of the held region; otherwise
    nothing of the robot's explains the shrink (a person's hand, the target
    knocked over) and it is retracted.

    A fit that passes both is accepted only when the region the kernel would get
    (``_kernel_closure``) holds the target's whole map component: no occupied cell
    more than a voxel above ``support_z`` touches it from outside
    (``occupied_touching_outside``). Otherwise it is the lost view ``partial_fit`` —
    the head camera saw part of the target (its far side occluded by the hovering
    hand, Isaac i40/i43) and the kernel would stop the fingers on the unexempt rest.
    Nothing new is accepted; a held region stays under its freeze TTL, never retracted.
    Checked last, so every refusal above keeps its class (a lost view here can never
    turn a contradiction into a hold).
    """
    count, covered = region_covers_occupied(grid, region, min_fraction=min_cover)
    tracked = previous is None or track_region(
        previous,
        region,
        max_centroid_shift_m=grid.resolution,
        extents_tol_m=grid.resolution,
    )
    if covered and tracked:
        kernel, _ = _kernel_closure(region, grid)
        # ponytail: anything 26-touching the target above the support (a neighbour within
        # one cell, a wall it leans on) is "the target" here too — such a target is never
        # armed; a per-object map segmentation is the upgrade.
        left = occupied_touching_outside(grid, kernel, support_z=support_z)
        if len(left):
            raise _lost(
                "partial_fit",
                f"fit centre {tuple(round(v, 3) for v in region.pose.xyz)} half_extents "
                f"{tuple(round(v, 3) for v in region.half_extents)}: {len(left)} occupied "
                f"cell(s) of the target's map component touch the kernel's region from "
                f"outside, e.g. {tuple(round(float(v), 4) for v in left[0])}",
            )
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
    if _hand_over_target(hands, previous, reach_m=reach, hand_rise_m=hand_rise_m):
        # The robot's own hand occluding part of the target: the held region
        # stays under its freeze TTL and is not replaced by the partial fit.
        raise _lost("occluded_refit", f"{moved}, inside the held region, hand within {reach:.3f} m")
    raise _contradicted(
        "unoccluded_refit",
        f"{moved}: inside the held region but no declared contact link within {reach:.3f} m "
        f"of it ({len(hands)} located)",
    )


#: What a request was made against: (depth, depth stamp ns, intrinsics,
#: T_base_from_cam, support_z, the declaration it measures, the tracker generation).
_Snapshot = tuple[
    "NDArray[np.float64]",
    int,
    "IntrinsicsPinhole",
    "NDArray[np.float64]",
    float,
    GraspDeclaration,
    int,
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
        approach_m: Arm an approach-armed target (module docstring) when a hand's
            TCP comes within this distance of occupied cells, metres; ``None`` (the
            default) = off, only a declared ``search_box`` is ever measured. In
            ``(0, GraspDeclaration.MAX_HALF_EXTENT_M]``. *Calibration point.*
        occluder_margin_m: How far (beyond one voxel) from the held region a
            declared contact link's hand point (its leg's TCP,
            ``VisionAttachmentBridge.jaw_point``) may be for a shrunk re-fit to
            count as the robot's own hand occluding the target (``_gate_refit``).
            *Calibration point* — it covers the TCP's offset from the finger
            surface; at most 0.10 m, beyond which any nearby arm pose
            would excuse a shrunken re-fit.

    Raises:
        ROSConfigError: On a rate outside 2-5 Hz; a non-positive cell count, cover
            fraction or probe margin; a freeze outside ``(0, 2 * grid_max_age_s]`` (the
            kernel's ``grasp_region_max_age_s``),
            a search depth outside ``(0, 0.5]`` m or an occluder margin outside
            ``[0, 0.10]`` m, or an approach distance outside
            ``(0, GraspDeclaration.MAX_HALF_EXTENT_M]`` (``grid_max_age_s`` itself is
            checked by the bridge).
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
        approach_m: float | None = None,
    ) -> None:
        """Validate the config; create no ROS entities yet."""
        if approach_m is not None and not 0.0 < approach_m <= GraspDeclaration.MAX_HALF_EXTENT_M:
            raise ROSConfigError(
                f"grasp target approach_m must be in (0, {GraspDeclaration.MAX_HALF_EXTENT_M}] "
                f"m, got {approach_m!r}."
            )
        self._approach_m = approach_m
        # The robot's hands, for the approach-armed target (one hand arms at a time).
        self._hands_of_robot = gripper_hands(bridge._description)
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
        self._subs: list[Any] = []
        self._timer: Any = None
        # (future, tracker generation it was asked under) of the request in flight.
        self._inflight: tuple[Any, int] | None = None
        self._deadline_timer: Any = None
        # Set by ``teardown`` (under the bridge lock): no request or deadline after it.
        self._torn_down = False
        # Consecutive fits that found no self-filtered cloud although a topic is
        # configured: logged on entering and on leaving that state (CLAUDE.md §1.4).
        self.unfiltered_fits = 0
        # Why the last published region was not exactly the cell-closed fit ("" when it
        # was): logged on every change, never per publish (``_kernel_region``).
        self._closure_note = ""
        # The last accepted region and the support top it was fitted on
        # (``measured_support``).
        self._support: tuple[PlaceRegion, float] | None = None

    @property
    def _lock(self) -> AbstractContextManager[Any]:
        """The owning bridge's lock: one serialization domain for the bridge and its legs."""
        return self._bridge._lock

    @property
    def _grid(self) -> tuple[VoxelLattice, int, float] | None:
        """The bridge's newest grid (``VisionAttachmentBridge._on_voxels``).

        Shared with the place leg: (lattice, ``source_stamp`` ns, monotonic receive time).
        """
        return self._bridge._grid

    def setup(self) -> None:
        """Subscribe the declaration; start the measurement timer (the bridge owns the grid)."""
        from openral_msgs.msg import GraspDeclaration as GraspDeclarationMsg
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
        declaration_sub = self._node.create_subscription(
            GraspDeclarationMsg,
            "/openral/grasp_declaration",
            self._on_declaration,
            declaration_qos,
        )
        self._subs = [declaration_sub]
        self._torn_down = False
        self._timer = self._node.create_timer(1.0 / self._config.grasp_target_rate_hz, self._tick)
        self._node.get_logger().info(
            f"grasp target leg: rate={self._config.grasp_target_rate_hz:.1f}Hz "
            f"freeze={self._freeze_s:.2f}s grid_max_age={self._config.grid_max_age_s:.2f}s "
            f"min_cells={self._config.grasp_target_min_cells} "
            f"min_cover={self._config.grasp_target_min_cover:.2f} "
            f"support_search_below={self._search_below_m:.2f}m "
            f"support_probe_margin={self._probe_margin_m:.2f}m "
            f"occluder_margin={self._occluder_margin_m:.2f}m "
            f"approach={'off' if self._approach_m is None else f'{self._approach_m:.2f}m'} "
            f"deadline={self._config.deadline_s:.3f}s"
        )

    @_locked
    def teardown(self) -> None:
        """Destroy every ROS entity; idempotent. No request or deadline timer outlives it."""
        self._torn_down = True
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

    @_locked
    def fill(self, msg: Any, *, now_ns: int) -> None:
        """Put the declaration in force (region and all) on one ``AttachmentState``.

        The region goes out cell-closed (``_kernel_region``); the tracker keeps the
        tight fit every producer gate compares against.
        """
        declaration = self.tracker.envelope(now_ns=now_ns)
        msg.grasp_declaration_valid = declaration is not None
        if declaration is None:
            return
        if declaration.region is not None:
            declaration = declaration.model_copy(
                update={"region": self._kernel_region(declaration.target_id, declaration.region)}
            )
        declaration.fill_idl(msg.grasp_declaration)

    def _kernel_region(self, target_id: str, region: PlaceRegion) -> PlaceRegion:
        """The held fit as the kernel gets it: closed over the cells of the newest grid.

        The kernel exempts a voxel only when its centre lies in the region, so the
        tight fit leaves the target's own boundary cells unexempt; ``cell_closed_region``
        grows it by at most half a cell per side (``r/√2`` horizontally, ``r/2`` up,
        never down). The newest grid is the one the kernel checks the same publication
        against. Each half-extent is clamped at ``GraspDeclaration.MAX_HALF_EXTENT_M``;
        a closure over the volume cap, no grid or a grid in another frame (the kernel
        refuses that region itself) publishes the tight fit. Every such case is logged
        once per change.
        """
        grid = self._grid
        try:
            if grid is None:
                raise ROSConfigError("no voxel grid to close the region over")
            closed, note = _kernel_closure(region, grid[0])
        except ROSConfigError as exc:
            closed = region
            note = f"published tight: {exc}"
        if note != self._closure_note:
            self._closure_note = note
            if note:
                self._node.get_logger().warning(
                    f"grasp target region for {target_id!r}: cell closure {note}"
                )
            else:
                self._node.get_logger().info(
                    f"grasp target region for {target_id!r}: cell-closed again"
                )
        return closed

    # ── inputs ───────────────────────────────────────────────────────────────

    @_locked
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

    def _fresh_grid(self, now_ns: int) -> tuple[VoxelLattice, int]:
        """The bridge's newest grid, checked by ``_checked_grid``."""
        return self._checked_grid(now_ns, self._grid)

    def _checked_grid(
        self, now_ns: int, entry: tuple[VoxelLattice, int, float] | None
    ) -> tuple[VoxelLattice, int]:
        """A grid snapshot ``entry`` and its ``source_stamp`` (ns), or a lost-view refusal.

        Two ages: receipt (the kernel's voxel deadline, ``grid_max_age_s``) and the
        world data behind it (``source_stamp``). A grid whose data is older than the
        freeze could only stand a region that is dead at birth (its region is stamped
        no later than that data); an unset ``source_stamp`` is unknown age — stale,
        as the kernel treats it.
        """
        if entry is None:
            raise _lost("no_grid", "no /openral/world_voxels grid yet")
        lattice, source_ns, received = entry
        age = time.monotonic() - received
        if age > self._config.grid_max_age_s:
            raise _lost("grid_stale", f"newest voxel grid is {age:.2f} s old")
        if source_ns <= 0:
            raise _lost("grid_source_unknown", "voxel grid has no source_stamp (unknown data age)")
        if now_ns - source_ns > int(self._freeze_s * 1e9):
            raise _lost(
                "grid_source_stale",
                f"voxel grid's world data is {(now_ns - source_ns) / 1e9:.2f} s old "
                f"(> freeze {self._freeze_s:.2f} s)",
            )
        return lattice, source_ns

    def _now_ns(self) -> int:
        return int(self._node.get_clock().now().nanoseconds)

    # ── measurement ──────────────────────────────────────────────────────────

    def _tick(self) -> None:
        """One measurement tick: snapshot under the bridge lock, scan outside, commit under it.

        Approach detection, the arming's state and the grid snapshot run under the lock
        (``_begin_tick``); the support/seed scan of the snapshot's column (pure) runs
        outside it; the request is committed under the lock (``_commit_request``).
        """
        now_ns = self._now_ns()
        begun = self._begin_tick(now_ns)
        if begun is None:
            return
        target, generation, grid = begun
        assert target.search_box is not None  # checked by ``_begin_tick``
        try:
            try:
                seed_point, support_z = self._seed(grid, target.search_box)
            except ROSConfigError as exc:
                raise _contradicted("rejected_inputs", str(exc)) from exc
        except _Refusal as refusal:
            self._refuse(refusal, now_ns=now_ns, generation=generation)
            return
        self._commit_request(now_ns, target, generation, grid, seed_point, support_z)

    @_locked
    def _begin_tick(self, now_ns: int) -> tuple[GraspDeclaration, int, VoxelLattice] | None:
        """The tick's bookkeeping; ``(target, generation, grid)`` to measure, or ``None``."""
        if self._torn_down:
            return None
        # A refusal applies to what was measured when it was raised: an arming or
        # handover meanwhile makes it stale (``refuse(generation=)``).
        generation = self.tracker.generation
        if self._inflight is not None:
            self.tracker.presence_unknown()  # no sample this tick
            return None
        try:
            try:
                if self._approach_m is not None and self.tracker.wants_approach(now_ns=now_ns):
                    self._detect_approach(self._approach_m, now_ns, generation)
                else:
                    self.tracker.presence_unknown()
                if not self.tracker.wants_measurement(now_ns=now_ns):
                    return None
                target, generation = self.tracker.measured()
                if target is None or target.search_box is None:
                    return None
                box = target.search_box
                grid, _ = self._fresh_grid(now_ns)
                if box.frame_id != grid.frame_id:
                    raise _contradicted(
                        "frame_mismatch",
                        f"search box in {box.frame_id!r}, grid in {grid.frame_id!r}",
                    )
            except ROSConfigError as exc:  # an input shape the geometry refuses
                raise _contradicted("rejected_inputs", str(exc)) from exc
        except _Refusal as refusal:
            self.tracker.presence_unknown()  # e.g. no fresh grid: no sample
            self._refuse(refusal, now_ns=now_ns, generation=generation)
            return None
        return target, generation, grid

    @_locked
    def _commit_request(
        self,
        now_ns: int,
        target: GraspDeclaration,
        generation: int,
        grid: VoxelLattice,
        seed_point: tuple[float, float, float],
        support_z: float,
    ) -> None:
        """Send the tick's request, unless torn down or the target changed during the scan."""
        if self._torn_down or self._inflight is not None or generation != self.tracker.generation:
            return
        try:
            try:
                self._request(now_ns, target, generation, grid, seed_point, support_z)
            except ROSConfigError as exc:
                raise _contradicted("rejected_inputs", str(exc)) from exc
        except _Refusal as refusal:
            self._refuse(refusal, now_ns=now_ns, generation=generation)

    @_locked
    def _refuse(self, refusal: _Refusal, *, now_ns: int, generation: int) -> None:
        self.tracker.refuse(
            refusal.kind,
            refusal.detail,
            retract=refusal.retract,
            now_ns=now_ns,
            generation=generation,
        )

    def _holding(self, links: Sequence[str]) -> bool:
        """Whether a gripper leg of these links holds, releases or is segmenting a payload.

        A trigger that reads the jaws loaded counts too (``trigger.attached``): jaws
        closed on something are holding it, whatever the bridge has latched yet.
        Called under the bridge lock (it reads the bridge's legs).
        """
        return any(
            leg.attachment is not None
            or leg.release is not None
            or leg.pending
            or leg.trigger.attached
            for leg in self._bridge._legs
            if leg.jaw_link in links
        )

    def _hand_of(self, jaw_link: str) -> tuple[str, ...] | None:
        return next((hand for hand in self._hands_of_robot if jaw_link in hand), None)

    def on_detach(self, jaw_link: str, record: AttachedCollisionObject | None) -> None:
        """The bridge's DETACH of ``jaw_link``, after it dropped the leg's attachment.

        ``record`` is the released payload frozen in the base frame (the release
        window's record), or ``None`` when no window opened (no tf2). On the
        handed-over hand the region is dropped at once (``GraspTargetTracker.on_release``)
        and, when the hand's legs hold nothing more, the pick completes
        (``on_pick_complete``): the hand may re-arm, under a fresh identity, but not on
        ``record`` (``_near_spent``). Event-driven: never polled.
        """
        hand = self._hand_of(jaw_link)
        if hand is None:
            return
        self.tracker.on_release(hand, record)
        self._complete_if_empty(hand)

    def on_attachment_changed(self, *, emptied: bool) -> None:
        """The bridge is about to publish a set that lost or replaced an object, or is empty.

        Called before the envelope is filled, so that snapshot already carries the
        refreshed identity and no pre-change region (``GraspTargetTracker.on_attachment_changed``,
        which keeps a mid-grasp hand's identity — never its region — across a change that
        leaves the set non-empty).
        """
        holding = [link for hand in self._hands_of_robot if self._resolving(hand) for link in hand]
        self.tracker.on_attachment_changed(now_ns=self._now_ns(), emptied=emptied, holding=holding)

    def _resolving(self, links: Sequence[str]) -> bool:
        """Whether a leg of these links has an ATTACH being resolved, nothing latched yet.

        A segmentation in flight (``pending``) or jaws read loaded (``trigger.attached``)
        with no attachment: the mid-grasp sense of ``on_attachment_changed``. A carried
        payload or a release window is not mid-grasp. Called under the bridge lock.
        """
        return any(
            leg.attachment is None and (leg.pending or leg.trigger.attached)
            for leg in self._bridge._legs
            if leg.jaw_link in links
        )

    def on_hand_settled(self, jaw_link: str) -> None:
        """``jaw_link``'s leg settled: complete the pick if its hand holds nothing.

        Its release window closed (clear or timed out), its barrier was released, or it
        detached with nothing latched. Never polled, so every such edge must call it.
        """
        hand = self._hand_of(jaw_link)
        if hand is not None:
            self._complete_if_empty(hand)

    def _complete_if_empty(self, hand: tuple[str, ...]) -> None:
        if not self._holding(hand):
            self.tracker.on_pick_complete(hand, now_ns=self._now_ns())

    def on_attach(
        self, jaw_link: str, payload: AttachedCollisionObject, *, region: PlaceRegion | None
    ) -> None:
        """Hand the armed region over to the ATTACH of ``jaw_link`` only if it is its target.

        The bridge's one entry for an ATTACH that resolved (design §2.2):

        * ``region`` set — the payload *is* that measured region (``_region_payload``
          confirmed the jaw at it); handed over only while the tracker still holds
          exactly that region (a re-fit accepted on another thread since is not it).
        * ``region`` ``None`` — the grasp was segmented (``_finish``), possibly because
          the jaw was *not* at the region: a neighbouring object. Handed over only when
          the jaw is at the held region (``VisionAttachmentBridge.jaw_at``, the region
          payload's own test, so a region first accepted mid-close passes it too) **and** every
          primitive centre of the segmented payload lies in the held region grown by one
          voxel. Otherwise the arming ends as ``attach_off_target`` (no handover).

        Args:
            jaw_link: The attaching leg's jaw link.
            payload: The attachment the bridge latched.
            region: The region the payload was built from, or ``None`` when segmented.
        """
        if region is not None:
            # Built from ``region``: on target only while the tracker still holds exactly
            # it — gone (``None``) or replaced since, the payload is a stale measurement.
            self.tracker.on_attach(jaw_link, confirm=lambda held: held == region)
            return
        self.tracker.on_attach(
            jaw_link,
            confirm=lambda held: held is None or self._payload_on(held, jaw_link, payload),
        )

    def _payload_on(
        self, region: PlaceRegion, jaw_link: str, payload: AttachedCollisionObject
    ) -> bool:
        """Whether a segmented payload is the region's target (``on_attach``)."""
        from openral_hal.vision_attachment_bridge import primitive_poses

        bridge = self._bridge
        if not bridge.jaw_at(jaw_link, region, reach_m=self._occluder_margin_m)[0]:
            return False
        t_region_from_link = bridge._lookup(region.frame_id, bridge.tf_frame(payload.attach_link))
        if t_region_from_link is None or not payload.primitives:
            return False
        # One voxel of tolerance (none before any grid: the stricter side).
        voxel = self._grid[0].resolution if self._grid is not None else 0.0
        grown = region.model_copy(
            update={"half_extents": tuple(h + voxel for h in region.half_extents)}
        )
        points = np.asarray(
            [t[:3, 3] for t in primitive_poses(payload, t_region_from_link)], dtype=np.float64
        )
        return bool(_in_region(points, grown).all())

    def _near_spent(self, hand: tuple[str, ...], box: PlaceRegion, grid: VoxelLattice) -> bool:
        """Whether ``hand`` must not arm yet: it is still by the payload it just released.

        After a pick completes, the hand's next arming must not take in the cells of
        the payload it put down (HZ-0115-11: a hand lingering by its just-placed
        object would arm on it). The guard holds until the column the hand's next
        measurement would search (``search_column`` of its approach box, reaching
        ``support_search_below_m`` below it — where ``_seed`` looks for the target and
        its support) clears that payload's last pose (the DETACH's frozen release
        record, each primitive bounded by its oriented box, ``bounding_half_extents``)
        by more than one voxel — the separating-axis lower bound on the gap
        (``box_gap_lower_bound_m``), so an elongated payload is not inflated to its
        bounding sphere — then ends for good.
        A pick with
        no record (no tf2 at the DETACH) keeps the hand from arming for the rest of
        the goal; a tf2 gap to the record's frame keeps the guard.
        """
        from openral_hal.vision_attachment_bridge import (
            bounding_half_extents,
            box_gap_lower_bound_m,
            primitive_poses,
        )

        guarded, record = self.tracker.spent(hand)
        if not guarded:
            return False
        if record is None:
            return True
        t_grid_from_link = self._bridge._lookup(
            grid.frame_id, self._bridge.tf_frame(record.attach_link)
        )
        if t_grid_from_link is None:
            return True
        column = search_column(box, below_m=self._search_below_m)
        t_box = homogeneous_from_quat_xyz(column.pose.xyz, column.pose.quat_xyzw)
        searched = (t_box[:3, 3], t_box[:3, :3], np.asarray(column.half_extents, dtype=np.float64))
        for t, primitive in zip(
            primitive_poses(record, t_grid_from_link), record.primitives, strict=True
        ):
            payload = (t[:3, 3], t[:3, :3], bounding_half_extents(primitive.shape))
            if box_gap_lower_bound_m(searched, payload) <= grid.resolution:
                return True
        self.tracker.clear_spent(hand, record)
        return False

    @_locked
    def _detect_approach(
        self, approach_m: float, now_ns: int, generation: int | None = None
    ) -> None:
        """Which hands are within ``approach_m`` of occupied cells now → the tracker.

        A hand holding a payload (or segmenting one) is never approaching: what is
        near its TCP is what it carries. A hand still by the payload it just released
        (``_near_spent``) is not a candidate either: it neither arms nor counts toward
        "two hands at once" — its box holds the cells of what it put down, and a hand
        guarded for the rest of the goal (no release record) would otherwise block the
        other hand's every pick. One hand per declaration is the kernel's own rule
        (HZ-0115-12), whatever the other hand does. Given
        ``generation`` (the tick's), the verdict applies only under it. Runs under the
        bridge lock (it reads the legs and moves the tracker); the grid's whole scan was
        done at receipt.
        """
        grid, _ = self._fresh_grid(now_ns)
        near: list[tuple[tuple[str, ...], PlaceRegion]] = []
        located: dict[tuple[str, ...], PlaceRegion] = {}
        holding = [hand for hand in self._hands_of_robot if self._holding(hand)]
        for hand in self._hands_of_robot:
            if hand in holding:
                continue
            tcps = [self._bridge.jaw_point(link, grid.frame_id) for link in hand]
            points = [point for point in tcps if point is not None]
            if len(points) != len(hand):
                continue  # an unlocated hand is not approaching (and ends a held approach)
            box = approach_box(points, approach_m=approach_m, frame_id=grid.frame_id)
            located[hand] = box
            # The guard is checked before the cell count: a hand lifted clear of the
            # payload it released ends it, wherever it goes next.
            if self._near_spent(hand, box, grid):
                continue
            if approach_cell_count(grid, box) >= self._config.grasp_target_min_cells:
                near.append((hand, box))
        self.tracker.on_approach(
            near,
            now_ns=now_ns,
            move_m=grid.resolution,
            holding=holding,
            generation=generation,
            located=located,
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
        # is on — would stand the region on nothing and exempt the gap (HZ-0115-6).
        assert seed.bottom_z is not None
        if seed.bottom_z - support_z > 2.0 * grid.resolution + 1e-9:
            raise _contradicted(
                "not_on_support",
                f"target's lowest cell at z={seed.bottom_z:.3f} is "
                f"{seed.bottom_z - support_z:.3f} m above the measured support z={support_z:.3f} "
                f"(> {2.0 * grid.resolution:.3f} m)",
            )
        return seed.point, support_z

    def _request(
        self,
        now_ns: int,
        declaration: GraspDeclaration,
        generation: int,
        grid: VoxelLattice,
        seed_point: tuple[float, float, float],
        support_z: float,
    ) -> None:
        """Project the seed and send one bounded ``SegmentInView`` request — or refuse.

        ``declaration`` and ``generation`` are ``GraspTargetTracker.measured()``, read
        together: the reply is accepted only under that generation. Runs under the bridge
        lock (it reads the bridge's depth and tf2, and records the request in flight).
        """
        from openral_hal.vision_attachment_bridge import build_segment_request

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
            generation,
        )
        future = client.call_async(request)
        self._inflight = (future, snapshot[6])
        future.add_done_callback(lambda fut: self._on_reply(fut, snapshot))
        self._deadline_timer = self._node.create_timer(self._config.deadline_s, self._on_deadline)

    def _stale(self, generation: int) -> bool:
        """Whether a request asked under ``generation`` no longer measures the target."""
        if generation == self.tracker.generation:
            return False
        self._node.get_logger().info(
            "grasp target: measurement in flight discarded — the target changed or was "
            "handed over while it ran"
        )
        return True

    @_locked
    def _on_deadline(self) -> None:
        if self._torn_down:
            return
        self._cancel_deadline()
        inflight, self._inflight = self._inflight, None
        if inflight is None:
            return
        future, generation = inflight
        future.cancel()  # runs _on_reply, which sees it is no longer in flight
        if self._stale(generation):
            return
        self.tracker.refuse(
            "deadline",
            f"SegmentInView did not answer within {self._config.deadline_s:.3f} s",
            retract=False,
            now_ns=self._now_ns(),
            generation=generation,
        )

    def _cancel_deadline(self) -> None:
        if self._deadline_timer is not None:
            self._deadline_timer.cancel()
            self._node.destroy_timer(self._deadline_timer)
            self._deadline_timer = None

    def _on_reply(self, future: Any, snapshot: _Snapshot) -> None:
        now_ns = self._now_ns()
        with self._lock:
            if self._inflight is None or future is not self._inflight[0]:
                return  # resolved by the deadline already (or torn down)
            self._cancel_deadline()
            self._inflight = None
            if self._stale(snapshot[6]):
                return  # neither its refusal nor its region applies any more
            # The bridge state ``_measure`` needs, read here: the grid and the declared
            # contact links' hand points (tf2) in its frame, and the held region.
            grid = self._grid
            hands = self._hands(snapshot[5], grid[0].frame_id) if grid is not None else []
            previous = self.tracker.region
        try:
            try:
                # Outside the lock: mask decode, back-projection and gates on the snapshot.
                region = self._measure(future, snapshot, now_ns, grid, hands, previous)
            except ROSConfigError as exc:  # an input shape the geometry refuses
                raise _contradicted("rejected_inputs", str(exc)) from exc
        except _Refusal as refusal:
            self._refuse(refusal, now_ns=now_ns, generation=snapshot[6])
            return
        with self._lock:
            # An ATTACH (proprio thread) may have handed over or re-armed while
            # ``_measure`` ran: ``accept`` drops the region when the generation moved.
            self.tracker.accept(region, generation=snapshot[6])
            if self.tracker.region is region:
                self._support = (region, snapshot[4])

    def measured_support(self, region: PlaceRegion) -> float | None:
        """The support top ``region`` was fitted on, in ``region.frame_id``, or ``None``.

        ``support_top_from_voxels``' measurement for the request whose reply the tracker
        accepted as exactly ``region`` (``target_region_from_mask`` pins its lower face one
        voxel above it). ``None`` for any other region — nothing was measured for it. The
        region payload's support witness stands on this (``region_attachment``).
        """
        support = self._support
        return support[1] if support is not None and support[0] == region else None

    def _measure(
        self,
        future: Any,
        snapshot: _Snapshot,
        now_ns: int,
        grid_entry: tuple[VoxelLattice, int, float] | None,
        hands: Sequence[tuple[float, float, float]],
        previous: PlaceRegion | None,
    ) -> PlaceRegion:
        """Reply → region → map cover → tracking gate, or a typed refusal.

        Pure on its arguments (snapshots taken under the bridge lock by ``_on_reply``),
        but for ``_align_to_depth``'s read of the cached ``CameraInfo``, the read of the
        self-filtered cloud cache (``_kept_points``) and the ``unfiltered_fits`` count.
        """
        from openral_hal.vision_attachment_bridge import (
            decode_mono8_mask,
            mask_depth_skew_reason,
            mask_stamps_ns,
            resolve_segment_outcome,
        )

        depth, depth_stamp_ns, _intrinsics, t_base_from_cam, support_z, declaration, _ = snapshot
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
        grid, source_ns = self._checked_grid(now_ns, grid_entry)
        # Stamped no later than the map it is vouched against: a stalled octomap ages
        # the region out (freeze TTL here, grasp_region_max_age_s in the kernel).
        stamp_ns = min(depth_stamp_ns, source_ns)
        # The robot's own points (the fingers closing in on the target) leave the fit
        # exactly as the self-filter took them out of the map, at this capture.
        kept = self._kept_points(depth_stamp_ns, grid.frame_id)
        if kept is None and self._config.self_filtered_cloud_topic:
            # With the filter configured, a capture it did not cover is a lost view: the
            # robot's own points would enter the fit (Isaac i38: a hand-grown 32 cm column
            # reached the support and was armed). Nothing new is accepted from it; a held
            # region stays under its freeze TTL.
            raise _lost(
                "unfiltered",
                f"no self-filtered cloud within the mask/depth skew of stamp {depth_stamp_ns}",
            )
        if kept is not None:
            masks = [
                mask_without_removed_points(mask, depth, intrinsics, t_base_from_cam, kept)
                for mask in masks
            ]
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
                stamp_ns=stamp_ns,
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
            raise _refused_fit(
                fit.refusal.value,
                detail,
                previous,
                hands,
                reach_m=grid.resolution + self._occluder_margin_m,
                hand_rise_m=self._approach_m or 0.0,
            )
        return _gate_refit(
            grid,
            fit.region,
            previous,
            support_z=support_z,
            min_cover=self._config.grasp_target_min_cover,
            hands=hands,
            occluder_margin_m=self._occluder_margin_m,
            hand_rise_m=self._approach_m or 0.0,
        )

    def _kept_points(self, stamp_ns: int, frame: str) -> NDArray[np.float64] | None:
        """``VisionAttachmentBridge.kept_points``, saying when a configured filter missed.

        With ``self_filtered_cloud_topic`` set, a capture with no self-filtered cloud is not
        fitted (``_measure`` refuses it as the lost view ``unfiltered``): the first such
        capture warns (topic, stamp, cache span), the next filtered one reports how many
        were skipped (CLAUDE.md §1.4).
        """
        kept = self._bridge.kept_points(stamp_ns, frame)
        topic = self._config.self_filtered_cloud_topic
        if kept is None and topic:
            if not self.unfiltered_fits:
                stamps = [cloud[0] for cloud in self._bridge._kept_clouds]
                span = f"{min(stamps)}..{max(stamps)} ns" if stamps else "empty"
                self._node.get_logger().warning(
                    f"grasp target fit unfiltered: no self-filtered cloud on {topic!r} at "
                    f"stamp {stamp_ns} (or no tf2 to {frame!r}; cache {len(stamps)} clouds, "
                    f"span {span}) — not fitted (the robot's own points would enter it)"
                )
            self.unfiltered_fits += 1
        elif kept is not None and self.unfiltered_fits:
            self._node.get_logger().info(
                f"grasp target fit self-filtered again after {self.unfiltered_fits} "
                "unfiltered fit(s)"
            )
            self.unfiltered_fits = 0
        return kept

    def _hands(self, declaration: GraspDeclaration, frame: str) -> list[tuple[float, float, float]]:
        """The located hand points of the declared contact links, in ``frame``."""
        located = (self._bridge.jaw_point(link, frame) for link in declaration.contact_links)
        return [point for point in located if point is not None]
