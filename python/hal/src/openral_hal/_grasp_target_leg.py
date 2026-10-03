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
  **while a declared contact link's hand point (the bridge's TCP for the leg
  whose jaw link it is, ``VisionAttachmentBridge.jaw_point``) is within one voxel
  + ``occluder_margin_m`` of it** (``occluded_refit``: the robot's own hand
  occluding part of the target shrinks and shifts the fit; it never replaces
  the held region). The same shrink with no contact link near is
  ``unoccluded_refit``, a contradiction (a person's hand, the target knocked
  over). The gripper closing in on the target occludes it from the head
  camera exactly then, so the last accepted
  region is **frozen** for at most ``grasp_target_freeze_s`` (unset: twice
  ``grid_max_age_s``, the kernel's voxel deadline; never more — the kernel's
  ``grasp_region_max_age_s``) from its own ``stamp_ns``, then retracted. That
  stamp is the older of the depth frame it was measured on and the grid's
  ``source_stamp`` (the world data the map cover was checked against), so a
  stalled octomap — whose bridge keeps republishing a fresh ``header.stamp`` —
  ages the region out instead of vouching for it.

The region dies with the declaration: dispatch retraction (goal end, cancel,
E-stop — the runner retracts on all of them) and ``timeout_s`` expiry clear it.
On an ATTACH of a leg whose jaw link the *armed* declaration names (the
approach-armed hand, or a named ``search_box`` declaration), re-measurement stops
and the region is kept for the kernel's handover rule (§2.1) — but only for an
ATTACH on what was measured (``GraspTargetLeg.on_attach``): the payload built from
the region itself, or a segmented payload lying in the region (+1 voxel) with the
jaw at it. Any other ATTACH (jaws closed on a neighbour) ends the arming as
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
link>:<n>"``, ``n`` counting the goal's armings — the kernel retires each pick's
``(target_id, stamp_ns)`` at its detach, so every arming needs its own identity;
``contact_links`` = that hand, the box as its ``search_box``, every other field —
``stamp_ns`` included — the goal's); two at once arm none. The box then follows the TCP and
runs the measurement above unchanged; the hand leaving the approach distance
retracts the region at once (``approach_ended``) — but not while one of its legs
holds, releases or is segmenting a payload: its grasp is being resolved, and the
ATTACH that resolves it (after ``SegmentInView``, or at once from the region) hands
the arming over. A measurement in flight when the target changes or is handed over
is discarded (``GraspTargetTracker.generation``), its refusal or region dropped,
and a handed-over target takes no ``accept`` / ``refuse``. After any retraction a hand
re-arms only once it moved more than one voxel or a freeze window elapsed. Once
the handed-over hand's legs hold nothing (polled each tick), the pick is
complete: the region is dropped and the hand may re-arm for a second pick — not
on the payload it just released: until its approach box clears that payload's last
pose (the release window's frozen record) by a voxel, the hand does not arm, and a
release no tick saw blocks it for the rest of the goal (HZ-01xx-11). A
named ``search_box`` wins: no approach detection runs for it, and it stays
handed over after its pick. The tracker is serialized by one lock: the bridge's
ATTACH runs on the HAL's proprio thread, the leg's ticks and replies on the executor.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Sequence
from functools import update_wrapper
from typing import TYPE_CHECKING, Any, Concatenate, ParamSpec, TypeVar

import numpy as np
from numpy.typing import NDArray
from openral_core import GraspDeclaration, PlaceRegion, Pose6D, gripper_hands
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


def _locked(
    method: Callable[Concatenate[GraspTargetTracker, _P], _R],
) -> Callable[Concatenate[GraspTargetTracker, _P], _R]:
    """Run a tracker method under the tracker's lock."""

    def locked(self: GraspTargetTracker, /, *args: _P.args, **kwargs: _P.kwargs) -> _R:
        with self._lock:
            return method(self, *args, **kwargs)

    update_wrapper(locked, method)  # keeps the docstring (doctests, docs)
    return locked


class GraspTargetTracker:
    """The leg's state machine — declaration, accepted region, freeze TTL. Pure.

    No ROS, no clock: every method takes the caller's ``now_ns`` (the node's ROS
    clock, the same domain as the runner's ``stamp_ns``). ``log`` receives one
    line per state transition, never one per tick.

    Thread-safe: every method runs under one lock. The leg's executor callbacks
    (tick, reply, deadline) and the bridge's grasp trigger — which runs on the HAL's
    proprio publisher thread (``observe_joint_state`` → ``on_attach``) — mutate it
    concurrently, and each check-then-act (``accept`` after the handover check,
    ``on_attach`` reading the arming and handing it over) must be atomic.

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
        # ponytail: one re-entrant lock for the whole tracker; every call is O(1)
        # bookkeeping, so contention is negligible next to the 2-5 Hz measurement.
        self._lock = threading.RLock()
        # Approach armings minted under the live declaration: each gets its own
        # ``target_id`` so the kernel's per-(target_id, stamp_ns) retirement of one
        # pick never refuses the next.
        self._armings = 0
        self._declaration: GraspDeclaration | None = None
        # The one-hand declaration armed from the robot's own approach, or ``None``.
        self._approach: GraspDeclaration | None = None
        self._region: PlaceRegion | None = None
        # The links of the hand whose ATTACH ended re-measurement, or ``None``.
        self._handed_over: tuple[str, ...] | None = None
        # Per hand: (approach box centre, now_ns) of its last refused/ended arming.
        self._backoff: dict[tuple[str, ...], tuple[tuple[float, float, float], int]] = {}
        self._status = ""
        # Bumped whenever what a measurement in flight was asked for stops being what is
        # measured (new/cleared declaration, arming, retraction, handover, pick done).
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
            self._backoff[held.contact_links] = (held.search_box.pose.xyz, now_ns)
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
        self._armings = 0
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
        self._armings = 0
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

        Only hands whose links the declaration names count (a named declaration
        narrows the choice). The held hand absent from ``near`` retracts at once
        (``approach_ended``); otherwise its search box follows the TCP. With none
        held, exactly one approaching hand arms; two at once arm none. A hand whose
        arming was retracted re-arms only once it moved more than ``move_m`` or a
        freeze window (``freeze_s``) elapsed: the same map at the same pose gives
        the same verdict, so re-arming every tick only floods the log and the
        segmenter.
        """
        goal = self._declaration
        if goal is None or goal.search_box is not None or self._handed_over is not None:
            return
        named = set(goal.contact_links)
        near = [(links, box) for links, box in near if set(links) <= named]
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
            where, since_ns = backoff
            moved = float(np.linalg.norm(np.subtract(box.pose.xyz, where)))
            if moved <= move_m and now_ns - since_ns <= self._freeze_ns:
                return  # same place, same map: the refusal would repeat
            del self._backoff[links]
        self._generation += 1
        self._armings += 1
        self._approach = goal.model_copy(
            update={
                # Its own identity per arming: the kernel retires a declaration's
                # (target_id, stamp_ns) at the pick's detach, and the goal's stamp_ns is
                # kept (the kernel's timeout_s backstop runs from it), so a reused id
                # would refuse every later pick of the goal as ``retired``.
                "target_id": f"{APPROACH_TARGET_PREFIX}{links[0]}:{self._armings}",
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

    def _armed(self) -> GraspDeclaration | None:
        """What is being measured for a hand: the approach arming, else a named search box."""
        if self._approach is not None:
            return self._approach
        declaration = self._declaration
        return declaration if declaration is not None and declaration.search_box else None

    @_locked
    def on_attach(self, contact_link: str, *, confirm: Callable[[PlaceRegion], bool]) -> None:
        """A leg attached; if the armed declaration names its jaw link, stop re-measuring.

        Scoped to the hand: only an ATTACH on a link of the *armed* declaration (the
        approach-armed hand, or a named ``search_box`` declaration) hands over. The
        goal-scope declaration names every hand before any approach arms, so an
        ATTACH there — a false positive, or a grasp of something never measured —
        neither hands over nor stops the other hand's approach detection.

        Args:
            contact_link: The attaching leg's jaw link.
            confirm: Called, under the lock, with the region held now; ``True`` only
                when the attached payload is that region's target (the leg's
                ``GraspTargetLeg.on_attach``: the payload *is* the region, or the
                segmented payload lies in it with the jaw at it). ``False`` ends the
                arming as ``attach_off_target``: the hand is handed over with **no**
                region — the region is dropped, never handed to the kernel for a
                payload it was not measured for, and re-measurement stops (the hand
                holds something else). No region held: handed over as ``absent``.
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
        self._handed_over = armed.contact_links
        self._generation += 1  # a measurement in flight was asked before the grasp
        region = self._region
        if region is not None and not confirm(region):
            self._region = None
            self._transition(
                "attach_off_target",
                f"grasp target {armed.target_id!r}: ATTACH on {contact_link!r} is not on the "
                f"measured region (centre {tuple(round(v, 3) for v in region.pose.xyz)}) — "
                "attach_off_target: region dropped, nothing handed over",
            )
            return
        self._transition(
            "handed_over",
            f"grasp target {armed.target_id!r}: ATTACH on {contact_link!r} — region "
            f"{'kept' if self._region is not None else 'absent'}, no further re-measurement",
        )

    @_locked
    def on_detach(self, contact_link: str, *, now_ns: int) -> None:
        """The handed-over hand holds nothing any more: end an approach-armed pick.

        Only an approach-armed handover is released: its region and arming are
        dropped (less exemption, never more) and the hand may re-arm for a second
        pick within the goal through ``on_approach`` — from a fresh approach, behind
        the same backoff as a refusal. A named ``search_box`` declaration stays
        handed over: its target was picked, and a new target needs a new
        declaration from dispatch.
        """
        handed = self._handed_over
        goal = self._declaration
        if handed is None or goal is None or contact_link not in handed:
            return
        held = self._approach
        if held is None and goal.search_box is not None:
            return  # a named target stays handed over
        # ``held`` is ``None`` only if the arming was dropped after the handover; the
        # hand is released either way, never left handed over for the rest of the goal.
        self._retract(
            "picked",
            f"{(held or goal).target_id!r}: {contact_link!r} released its payload; may re-arm",
            now_ns=now_ns,
        )
        self._handed_over = None

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
        try:
            GraspDeclaration.model_validate(declaration.model_dump() | {"region": region})
        except ValueError as exc:
            self._retract("declaration_bounds", str(exc).splitlines()[0], now_ns=region.stamp_ns)
            return
        self._region = region
        self._transition(
            "accepted",
            f"grasp target region accepted for {declaration.target_id!r}: "
            f"centre={tuple(round(v, 3) for v in region.pose.xyz)} "
            f"half_extents={tuple(round(v, 3) for v in region.half_extents)} "
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
        # (lattice, source_stamp ns, monotonic receive time) of the newest grid.
        self._grid: tuple[VoxelLattice, int, float] | None = None
        self._subs: list[Any] = []
        self._timer: Any = None
        # (future, tracker generation it was asked under) of the request in flight.
        self._inflight: tuple[Any, int] | None = None
        self._deadline_timer: Any = None
        # The handed-over hand's frozen release record (base frame), seen while open.
        self._released: AttachedCollisionObject | None = None
        # Per hand whose approach-armed pick completed: (goal (target_id, stamp_ns),
        # the released payload's frozen record, or ``None`` when no tick saw it).
        self._spent: dict[
            tuple[str, ...], tuple[tuple[str, int], AttachedCollisionObject | None]
        ] = {}

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
            f"approach={'off' if self._approach_m is None else f'{self._approach_m:.2f}m'} "
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
        # ``source_stamp``, not ``header.stamp``: the octomap bridge restamps the header
        # on every republish of a stalled octree; only this says how old the world is.
        source = msg.source_stamp
        source_ns = int(source.sec) * 1_000_000_000 + int(source.nanosec)
        try:
            self._grid = (lattice_from_msg(msg), source_ns, time.monotonic())
        except ROSConfigError as exc:
            self._grid = None
            self._node.get_logger().warning(f"grasp target: voxel grid dropped — {exc}")

    def _fresh_grid(self, now_ns: int) -> tuple[VoxelLattice, int]:
        """The newest grid and its ``source_stamp`` (ns), or a lost-view refusal.

        Two ages: receipt (the kernel's voxel deadline, ``grid_max_age_s``) and the
        world data behind it (``source_stamp``). A grid whose data is older than the
        freeze could only stand a region that is dead at birth (its region is stamped
        no later than that data); an unset ``source_stamp`` is unknown age — stale,
        as the kernel treats it.
        """
        if self._grid is None:
            raise _lost("no_grid", "no /openral/world_voxels grid yet")
        lattice, source_ns, received = self._grid
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
        now_ns = self._now_ns()
        if self._inflight is not None:
            return
        self._release_detached(now_ns)
        # A refusal applies to what was measured when it was raised: an arming or
        # handover on another thread meanwhile makes it stale (``refuse(generation=)``).
        generation = self.tracker.generation
        try:
            try:
                if self._approach_m is not None and self.tracker.wants_approach(now_ns=now_ns):
                    self._detect_approach(self._approach_m, now_ns)
                if not self.tracker.wants_measurement(now_ns=now_ns):
                    return
                target, generation = self.tracker.measured()
                self._request(now_ns, target, generation)
            except ROSConfigError as exc:  # an input shape the geometry refuses
                raise _contradicted("rejected_inputs", str(exc)) from exc
        except _Refusal as refusal:
            self.tracker.refuse(
                refusal.kind,
                refusal.detail,
                retract=refusal.retract,
                now_ns=now_ns,
                generation=generation,
            )

    def _holding(self, links: Sequence[str]) -> bool:
        """Whether a gripper leg of these links holds, releases or is segmenting a payload."""
        return any(
            leg.attachment is not None or leg.release is not None or leg.pending
            for leg in self._bridge._legs
            if leg.jaw_link in links
        )

    def _release_detached(self, now_ns: int) -> None:
        """A handed-over hand whose legs hold nothing any more has completed its pick.

        Its release window's frozen record — the payload's last pose — is kept as the
        hand's spent payload (``_spent``, checked by ``_detect_approach``).
        """
        handed = self.tracker.handed_over
        if handed is None:
            self._released = None
            return
        for leg in self._bridge._legs:
            if leg.jaw_link in handed and leg.release is not None:
                self._released = leg.release.record
        if self._holding(handed):
            return
        goal = self.tracker.declaration
        self.tracker.on_detach(handed[0], now_ns=now_ns)
        if self.tracker.handed_over is None and goal is not None:
            self._spent[handed] = ((goal.target_id, goal.stamp_ns), self._released)
        self._released = None

    def _near_spent(self, hand: tuple[str, ...], box: PlaceRegion, grid: VoxelLattice) -> bool:
        """Whether ``hand`` must not arm yet: it is still by the payload it just released.

        After a pick completes (``_release_detached``) the hand's next arming must not
        take in the cells of the payload it put down (HZ-01xx-11: the hand lingering
        by its just-placed object would arm on it). The guard holds until the hand's
        approach box clears that payload's last pose (its frozen release record, each
        primitive bounded by a sphere) by more than one voxel — the hand moved more
        than the approach distance away — and then ends for good. The release window
        is always closed by then (the pick completes only when it is), so it is no
        exit. A release no tick saw (its window opened and closed between two ticks,
        or never opened: no tf2) is unknown: the hand does not re-arm for the rest of
        the goal; a tf2 gap to the record's frame keeps the guard.
        """
        spent = self._spent.get(hand)
        if spent is None:
            return False
        goal = self.tracker.declaration
        (target_id, stamp_ns), record = spent
        if goal is None or (goal.target_id, goal.stamp_ns) != (target_id, stamp_ns):
            del self._spent[hand]  # a new goal: dispatch declared afresh
            return False
        if record is None:
            return True
        from openral_core.geometry import homogeneous_from_quat_xyz

        from openral_hal.vision_attachment_bridge import _bounding_half_extents

        t_grid_from_record = self._bridge._lookup(
            grid.frame_id, self._bridge.tf_frame(record.attach_link)
        )
        if t_grid_from_record is None:
            return True
        t_object = t_grid_from_record @ homogeneous_from_quat_xyz(
            record.pose_in_link.xyz, record.pose_in_link.quat_xyzw
        )
        reach = np.asarray(box.half_extents) + grid.resolution
        for primitive in record.primitives:
            t = t_object @ homogeneous_from_quat_xyz(
                primitive.pose_in_object.xyz, primitive.pose_in_object.quat_xyzw
            )
            radius = float(np.linalg.norm(_bounding_half_extents(primitive.shape)))
            if bool(np.all(np.abs(t[:3, 3] - np.asarray(box.pose.xyz)) <= reach + radius)):
                return True
        del self._spent[hand]
        return False

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
          the jaw is within ``occluder_margin_m`` of the held region (the region payload's
          own test, so a region first accepted mid-close passes it too) **and** every
          primitive centre of the segmented payload lies in the held region grown by one
          voxel. Otherwise the arming ends as ``attach_off_target`` (no handover).

        Args:
            jaw_link: The attaching leg's jaw link.
            payload: The attachment the bridge latched.
            region: The region the payload was built from, or ``None`` when segmented.
        """
        if region is not None:
            self.tracker.on_attach(jaw_link, confirm=lambda held: held == region)
            return
        self.tracker.on_attach(
            jaw_link, confirm=lambda held: self._payload_on(held, jaw_link, payload)
        )

    def _payload_on(
        self, region: PlaceRegion, jaw_link: str, payload: AttachedCollisionObject
    ) -> bool:
        """Whether a segmented payload is the region's target (``on_attach``)."""
        from openral_core.geometry import homogeneous_from_quat_xyz

        bridge = self._bridge
        hand = bridge.jaw_point(jaw_link, region.frame_id)
        if hand is None or not _hand_near([hand], region, reach_m=self._occluder_margin_m):
            return False
        t_region_from_link = bridge._lookup(region.frame_id, bridge.tf_frame(payload.attach_link))
        if t_region_from_link is None or not payload.primitives:
            return False
        t_object = t_region_from_link @ homogeneous_from_quat_xyz(
            payload.pose_in_link.xyz, payload.pose_in_link.quat_xyzw
        )
        centres = [
            (t_object @ homogeneous_from_quat_xyz(q.pose_in_object.xyz, q.pose_in_object.quat_xyzw))
            for q in payload.primitives
        ]
        # One voxel of tolerance (none before any grid: the stricter side).
        voxel = self._grid[0].resolution if self._grid is not None else 0.0
        grown = region.model_copy(
            update={"half_extents": tuple(h + voxel for h in region.half_extents)}
        )
        points = np.asarray([t[:3, 3] for t in centres], dtype=np.float64)
        return bool(_in_region(points, grown).all())

    def _detect_approach(self, approach_m: float, now_ns: int) -> None:
        """Which hands are within ``approach_m`` of occupied cells now → the tracker.

        A hand holding a payload (or segmenting one) is never approaching: what is
        near its TCP is what it carries.
        """
        grid, _ = self._fresh_grid(now_ns)
        near: list[tuple[tuple[str, ...], PlaceRegion]] = []
        spent: list[tuple[tuple[str, ...], PlaceRegion]] = []
        holding = [hand for hand in self._hands_of_robot if self._holding(hand)]
        for hand in self._hands_of_robot:
            if hand in holding:
                continue
            located = [self._bridge.jaw_point(link, grid.frame_id) for link in hand]
            points = [point for point in located if point is not None]
            if len(points) != len(hand):
                continue  # an unlocated hand is not approaching (and ends a held approach)
            box = approach_box(points, approach_m=approach_m, frame_id=grid.frame_id)
            if self._near_spent(hand, box, grid):
                if approach_cell_count(grid, box) >= self._config.grasp_target_min_cells:
                    spent.append((hand, box))
                continue
            if approach_cell_count(grid, box) >= self._config.grasp_target_min_cells:
                near.append((hand, box))
        if spent and near:
            # A hand by its just-released payload never arms, but still counts toward
            # "two hands at once" (it would have, before the guard): none arms.
            near += spent
        self.tracker.on_approach(near, now_ns=now_ns, move_m=grid.resolution, holding=holding)

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

    def _request(self, now_ns: int, declaration: GraspDeclaration | None, generation: int) -> None:
        """Seed, project, and send one bounded ``SegmentInView`` request — or refuse.

        ``declaration`` and ``generation`` are ``GraspTargetTracker.measured()``, read
        together: the reply is accepted only under that generation.
        """
        from openral_hal.vision_attachment_bridge import build_segment_request

        if declaration is None or declaration.search_box is None:
            return  # disarmed on another thread since ``wants_measurement``
        box = declaration.search_box
        grid, _ = self._fresh_grid(now_ns)
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

    def _on_deadline(self) -> None:
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
        if self._inflight is None or future is not self._inflight[0]:
            return  # resolved by the deadline already
        self._cancel_deadline()
        self._inflight = None
        if self._stale(snapshot[6]):
            return  # neither its refusal nor its region applies any more
        now_ns = self._now_ns()
        try:
            try:
                region = self._measure(future, snapshot, now_ns)
            except ROSConfigError as exc:  # an input shape the geometry refuses
                raise _contradicted("rejected_inputs", str(exc)) from exc
        except _Refusal as refusal:
            self.tracker.refuse(
                refusal.kind,
                refusal.detail,
                retract=refusal.retract,
                now_ns=now_ns,
                generation=snapshot[6],
            )
            return
        # An ATTACH on the HAL's proprio thread (``observe_joint_state`` → ``on_attach``)
        # can hand over or re-arm while ``_measure`` runs; ``accept`` drops the region
        # atomically when the generation moved (the check above was only for the log).
        self.tracker.accept(region, generation=snapshot[6])

    def _measure(self, future: Any, snapshot: _Snapshot, now_ns: int) -> PlaceRegion:
        """Reply → region → map cover → tracking gate, or a typed refusal."""
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
        grid, source_ns = self._fresh_grid(now_ns)
        # Stamped no later than the map it is vouched against: a stalled octomap ages
        # the region out (freeze TTL here, grasp_region_max_age_s in the kernel).
        stamp_ns = min(depth_stamp_ns, source_ns)
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
            if fit.refusal is TargetRefusal.TOO_FEW_POINTS:
                raise _lost(fit.refusal.value, detail)
            raise _contradicted(fit.refusal.value, detail)
        return _gate_refit(
            grid,
            fit.region,
            self.tracker.region,
            min_cover=self._config.grasp_target_min_cover,
            hands=self._hands(declaration, grid.frame_id),
            occluder_margin_m=self._occluder_margin_m,
        )

    def _hands(self, declaration: GraspDeclaration, frame: str) -> list[tuple[float, float, float]]:
        """The located hand points of the declared contact links, in ``frame``."""
        located = (self._bridge.jaw_point(link, frame) for link in declaration.contact_links)
        return [point for point in located if point is not None]
