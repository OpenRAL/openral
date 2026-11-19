# SPDX-License-Identifier: Apache-2.0
"""Judge a ``SuccessCondition`` over the object tracker's sets (issue #635).

``SuccessJudge`` is the one place the predicate and its settle clock live: the rSkill
runner feeds it the tracked-object set of every world-state snapshot and reports its
``TaskOutcome``; the sim tests run the same class. It reads tracker output only, commands
nothing, and is never a safety-kernel input.
"""

from __future__ import annotations

import numpy as np

from openral_core.perception_geometry import _in_region
from openral_core.schemas import (
    SuccessCondition,
    TaskOutcome,
    TaskVerdict,
    TrackedObject,
    TrackedObjectSet,
    TrackStatus,
)

__all__ = ["SuccessJudge"]

# Statuses whose box is where the tracker last SAW the object. A RELEASED track's box is
# frozen at the let-go pose (no fit until it is re-grounded), a TENTATIVE / SPLIT one is
# not confirmed: they may yet come to rest in the region, so they keep a verdict open.
# A LOST track says nothing.
_SEEN = frozenset({TrackStatus.CONFIRMED, TrackStatus.OCCLUDED})
_PENDING = frozenset({TrackStatus.RELEASED, TrackStatus.TENTATIVE, TrackStatus.SPLIT})


class SuccessJudge:
    """The verdict of one ``SuccessCondition`` over a stream of ``TrackedObjectSet`` s.

    ``observe`` every set the tracker publishes, then read ``outcome``:

    * ``succeeded`` — at least ``min_count`` tracks of the label are seen (confirmed or
      occluded, not in a hand) with their box centre in the region (a track whose centre
      lies in another counted track's box is the same object, counted once), on every set over
      ``settle_s`` of tracker time (the sets' own ``stamp_ns``), with no gap between two
      sets longer than ``max_evidence_gap_s``.
    * ``failed`` — too few rest there even counting the tracks that may still settle
      (released, tentative, split), and the tracker sees an object of the label held or
      elsewhere.
    * ``unknown`` — anything else: no set, a set in another frame, too few tracks of the
      label to tell, or the predicate holding for less than ``settle_s``.

    Example:
        >>> from openral_core import PlaceRegion, Pose6D
        >>> region = PlaceRegion(
        ...     frame_id="base",
        ...     pose=Pose6D(xyz=(0.3, 0.1, 0.02), quat_xyzw=(0, 0, 0, 1), frame_id="base"),
        ...     half_extents=(0.03, 0.03, 0.03),
        ... )
        >>> judge = SuccessJudge(SuccessCondition(object_label="cube", region=region))
        >>> judge.outcome().verdict.value, judge.outcome().reason
        ('unknown', 'no tracked-object set to judge on')
    """

    def __init__(self, condition: SuccessCondition) -> None:
        """Start judging ``condition`` with no evidence."""
        self.condition = condition
        self._last: TrackedObjectSet | None = None
        self._held_since_ns: int | None = None
        self._outcome = self._judged(None)

    def observe(self, tracked: TrackedObjectSet | None) -> None:
        """Take the newest tracked set (``None``: the tracker has none, or it went stale).

        A set that is not newer than the last one (same revision, stamp not advanced) is
        ignored; the tracker's heartbeat, the same revision at a later stamp, is evidence.
        A missing set, a stamp that went backwards (a clock reset) or a gap longer than
        ``max_evidence_gap_s`` restarts the settle.
        """
        last = self._last
        if tracked is not None and last is not None:
            ahead_ns = tracked.stamp_ns - last.stamp_ns
            if ahead_ns == 0 and tracked.revision == last.revision:
                return
            if ahead_ns < 0 or ahead_ns > int(self.condition.max_evidence_gap_s * 1e9):
                self._held_since_ns = None
        else:
            self._held_since_ns = None
        self._last = tracked
        self._outcome = self._judged(tracked)

    def outcome(self) -> TaskOutcome:
        """The verdict on the evidence observed so far."""
        return self._outcome

    def _judged(self, tracked: TrackedObjectSet | None) -> TaskOutcome:
        cond = self.condition

        def outcome(
            verdict: TaskVerdict, reason: str, tracks: list[TrackedObject], resting: int = 0
        ) -> TaskOutcome:
            return TaskOutcome(
                verdict=verdict,
                source="success_condition",
                reason=reason[: TaskOutcome.MAX_REASON_CHARS],
                object_ids=tuple(t.object_id for t in tracks),
                resting=resting,
                required=cond.min_count,
                held_s=held_s,
            )

        held_s = 0.0
        if tracked is None:
            return outcome(TaskVerdict.UNKNOWN, "no tracked-object set to judge on", [])
        if tracked.frame_id != cond.region.frame_id:
            self._held_since_ns = None
            return outcome(
                TaskVerdict.UNKNOWN,
                f"the region is in {cond.region.frame_id!r}, the tracked objects in "
                f"{tracked.frame_id!r}",
                [],
            )
        label = cond.object_label.strip().casefold()
        named = [t for t in tracked.objects if t.label.strip().casefold() == label]
        centres = np.asarray([t.box.pose.xyz for t in named], dtype=np.float64).reshape(-1, 3)
        inside = _in_region(centres, cond.region)
        resting: list[TrackedObject] = []
        for t, ok in zip(named, inside, strict=True):
            # Two tracks, one inside the other's box, are one object seen twice (a
            # re-grounded object beside its stale track): it counts once.
            centre = np.asarray([t.box.pose.xyz], dtype=np.float64)
            if ok and t.status in _SEEN and not any(_in_region(centre, r.box)[0] for r in resting):
                resting.append(t)
        pending = [t for t in named if t.status in _PENDING]
        elsewhere = [
            t
            for t, ok in zip(named, inside, strict=True)
            if t.status is TrackStatus.BOUND or (t.status in _SEEN and not ok)
        ]
        count = f"{len(resting)} of {cond.min_count} {cond.object_label!r}"
        if len(resting) < cond.min_count:
            self._held_since_ns = None
            if elsewhere and len(resting) + len(pending) < cond.min_count:
                seen = elsewhere[0]
                where = "is held" if seen.status is TrackStatus.BOUND else "is outside it"
                return outcome(
                    TaskVerdict.FAILED,
                    f"{count} rest in the region; {seen.object_id} {where}",
                    resting or [seen],
                    len(resting),
                )
            why = f"{len(pending)} not yet re-confirmed" if pending else "too few are tracked"
            return outcome(
                TaskVerdict.UNKNOWN, f"{count} rest in the region; {why}", resting, len(resting)
            )
        if self._held_since_ns is None:
            self._held_since_ns = tracked.stamp_ns
        held_s = max(0.0, (tracked.stamp_ns - self._held_since_ns) * 1e-9)
        if held_s < cond.settle_s:
            return outcome(
                TaskVerdict.UNKNOWN,
                f"{count} rest in the region, for {held_s:.1f} s of the {cond.settle_s:.1f} s "
                "they must",
                resting,
                len(resting),
            )
        return outcome(
            TaskVerdict.SUCCEEDED,
            f"{count} rest released in the region ({held_s:.1f} s)",
            resting,
            len(resting),
        )
