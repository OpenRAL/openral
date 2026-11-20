"""IoU-gated, evidence-based object memory.

Pure, ROS-free, stateful. Holds a table of remembered ``DetectedObject``s with
stable track ids and, per object, an existence belief kept as log-odds — the
same bookkeeping the occupancy voxels use (issue #636). Associates new lifted
candidates by label + 3D AABB IoU and freezes matched objects (no re-write of a
known object). A matched detection adds ``hit_log_odds``; an unmatched object
adds whatever its ``MissEvidence`` is worth: nothing when it could not be
observed (out of view, out of range, occluded), ``miss_occupied_log_odds`` when
its place is still occupied, ``miss_free_log_odds`` when the sensor saw through
it. The belief is clamped to ``[belief_absent, belief_max]`` and the object is
evicted when it reaches ``belief_absent`` — never on a single miss.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

from openral_core.schemas import (
    DetectedObject,
    ObjectBeliefState,
    SceneGraph,
    SpatialNode,
    SpatialNodeKind,
    belief_state,
)

from openral_world_state.object_lift import MissEvidence, aabb_iou_3d

__all__ = ["ObjectMemory"]


def _probability(log_odds: float) -> float:
    return 0.5 * (1.0 + math.tanh(0.5 * log_odds))  # the logistic, without exp's overflow


@dataclass
class _Tracked:
    obj: DetectedObject
    last_seen_ns: int
    belief: float  # existence log-odds


class ObjectMemory:
    """Stateful, IoU-gated spatial memory of detected objects.

    Args:
        iou_threshold: Minimum 3D AABB IoU (same label) to associate a new
            detection with an existing track.
        hit_log_odds: Added to a track's belief on a matched detection (``> 0``).
        miss_free_log_odds: Added when the track went undetected and the sensor
            saw through its box (``< 0``) — the strong miss.
        miss_occupied_log_odds: Added when the track went undetected by a detector
            that ran, but its box is still occupied (``<= 0``). Likely occlusion or
            a missed detection, so it is weak — yet not zero, or a false detection
            lifted onto a wall would be remembered forever.
        belief_max: Upper clamp (``> 0``); bounds how many clear observations a
            long-seen object takes to forget.
        belief_absent: Lower clamp and eviction threshold (``< 0``).

    Example:
        >>> from openral_core.schemas import DetectedObject, Pose6D
        >>> from openral_world_state.object_lift import MissEvidence
        >>> mem = ObjectMemory()
        >>> cand = DetectedObject(
        ...     label="cup",
        ...     confidence=0.9,
        ...     pose=Pose6D(xyz=(0.5, 0.5, 0.5), quat_xyzw=(0, 0, 0, 1), frame_id="map"),
        ...     bbox_3d=(0.0, 0.0, 0.0, 1.0, 1.0, 1.0),
        ... )
        >>> _ = mem.tick([cand], stamp_ns=1, miss_evidence=lambda o: MissEvidence.NONE)
        >>> free = lambda o: MissEvidence.FREE
        >>> tuple(len(mem.tick([], stamp_ns=t, miss_evidence=free)) for t in (2, 3, 4, 5))
        (1, 1, 1, 0)
    """

    def __init__(
        self,
        *,
        iou_threshold: float = 0.3,
        hit_log_odds: float = 0.85,
        miss_free_log_odds: float = -0.85,
        miss_occupied_log_odds: float = -0.1,
        belief_max: float = 3.5,
        belief_absent: float = -2.0,
    ) -> None:
        """Validate the IoU threshold and the evidence weights' signs."""
        if not 0.0 <= iou_threshold <= 1.0:
            raise ValueError(f"iou_threshold must be in [0,1]; got {iou_threshold}")
        if hit_log_odds <= 0.0 or miss_free_log_odds >= 0.0 or miss_occupied_log_odds > 0.0:
            raise ValueError(
                "need hit_log_odds > 0, miss_free_log_odds < 0, miss_occupied_log_odds <= 0; "
                f"got {hit_log_odds}, {miss_free_log_odds}, {miss_occupied_log_odds}"
            )
        if not belief_absent < 0.0 < belief_max:
            raise ValueError(
                f"need belief_absent < 0 < belief_max; got {belief_absent}, {belief_max}"
            )
        self._iou = iou_threshold
        self._hit = hit_log_odds
        self._miss = {
            MissEvidence.NONE: 0.0,
            MissEvidence.OCCUPIED: miss_occupied_log_odds,
            MissEvidence.FREE: miss_free_log_odds,
        }
        self._belief_max = belief_max
        self._belief_absent = belief_absent
        self._tracks: list[_Tracked] = []
        self._next_id = 0

    def tick(
        self,
        candidates: list[DetectedObject],
        *,
        stamp_ns: int,
        miss_evidence: Callable[[DetectedObject], MissEvidence],
    ) -> list[DetectedObject]:
        """Associate candidates, freeze matches, create new, weigh the misses.

        Args:
            candidates: Newly lifted ``DetectedObject``s for this run (map frame).
            stamp_ns: Run timestamp in nanoseconds (stored as last-seen).
            miss_evidence: What this run's sensing says about a remembered object
                nothing matched (see ``build_miss_classifier``).

        Returns:
            The current remembered object set (stable ``track_id``s), each carrying
            its ``existence`` probability. Evicted objects are absent from it.
        """
        matched: set[int] = set()
        for cand in sorted(candidates, key=lambda d: d.confidence, reverse=True):
            best_i, best_iou = -1, self._iou
            for i, tr in enumerate(self._tracks):
                if tr.obj.label != cand.label:
                    continue
                if cand.bbox_3d is None or tr.obj.bbox_3d is None:
                    continue
                iou = aabb_iou_3d(cand.bbox_3d, tr.obj.bbox_3d)
                if iou >= best_iou:
                    best_iou, best_i = iou, i
            if best_i >= 0:
                tr = self._tracks[best_i]
                tr.last_seen_ns = stamp_ns
                if best_i not in matched:  # one hit per run, however many boxes matched
                    tr.belief = min(self._belief_max, tr.belief + self._hit)
                if cand.confidence > tr.obj.confidence:
                    tr.obj = tr.obj.model_copy(update={"confidence": cand.confidence})
                matched.add(best_i)
            else:
                # A new track adopts the detection-time id propagated
                # from the 2D detector (DetectedObject.track_id, from
                # ObjectDetection2D.det_id) when present, so a physical object
                # carries one id across the 2D in_view line and the 3D
                # scene_objects line. Fall back to minting for legacy detectors
                # (track_id None / < 0). Matched tracks keep their stored id (3D
                # association is more stable than 2D re-id). An id a live track
                # already holds (a restored one, a recycled 2D id) is not adopted.
                adopted = cand.track_id
                taken = {t.obj.track_id for t in self._tracks}
                new_id = (
                    adopted
                    if adopted is not None and adopted >= 0 and adopted not in taken
                    else self._next_id
                )
                self._tracks.append(
                    _Tracked(
                        obj=cand.model_copy(update={"track_id": new_id}),
                        last_seen_ns=stamp_ns,
                        belief=min(self._belief_max, self._hit),
                    ),
                )
                matched.add(len(self._tracks) - 1)
                self._next_id = max(self._next_id, new_id) + 1

        survivors: list[_Tracked] = []
        for i, tr in enumerate(self._tracks):
            if i not in matched:
                tr.belief += self._miss[miss_evidence(tr.obj)]
                if tr.belief <= self._belief_absent:
                    continue
            survivors.append(tr)
        self._tracks = survivors
        return self.objects()

    def to_scene_graph(self) -> SceneGraph:
        """The remembered objects as a ``SceneGraph`` (the persistence contract).

        One ``OBJECT`` node per track, keyed ``obj_track_<id>`` like
        ``SpatialMemory``'s, carrying the track's belief state.
        """
        return SceneGraph(
            nodes=[
                SpatialNode(
                    node_id=f"obj_track_{tr.obj.track_id}",
                    kind=SpatialNodeKind.OBJECT,
                    pose=tr.obj.pose,
                    label=tr.obj.label,
                    confidence=min(1.0, max(0.0, tr.obj.confidence)),
                    bbox_3d=tr.obj.bbox_3d,
                    first_seen_ns=tr.last_seen_ns,
                    last_seen_ns=tr.last_seen_ns,
                    state=belief_state(_probability(tr.belief)),
                )
                for tr in self._tracks
            ]
        )

    def restore(self, graph: SceneGraph, *, map_frame: str) -> int:
        """Seed the memory from a persisted graph; every object starts uncertain.

        Takes the ``OBJECT`` nodes that are not absent, are anchored in
        ``map_frame`` and carry a ``bbox_3d`` (association and the miss test both
        need the box). Each becomes a track at belief 0 — neither confirmed nor
        refuted — so the first observations decide. A node keeps the track id its
        ``obj_track_<id>`` key names (a consumer that outlived the restart still
        holds it); any other key, or an id already taken, gets a fresh one.

        Returns:
            The number of objects restored.
        """
        before = len(self._tracks)
        for node in graph.nodes:
            if (
                node.kind is not SpatialNodeKind.OBJECT
                or node.state is ObjectBeliefState.ABSENT
                or node.bbox_3d is None
                or node.pose.frame_id != map_frame
            ):
                continue
            taken = {t.obj.track_id for t in self._tracks}
            suffix = node.node_id.removeprefix("obj_track_")
            kept = int(suffix) if suffix != node.node_id and suffix.isdecimal() else -1
            track_id = kept if kept >= 0 and kept not in taken else self._next_id
            self._tracks.append(
                _Tracked(
                    obj=DetectedObject(
                        label=node.label,
                        confidence=node.confidence,
                        pose=node.pose,
                        bbox_3d=node.bbox_3d,
                        track_id=track_id,
                    ),
                    last_seen_ns=node.last_seen_ns,
                    belief=0.0,
                )
            )
            self._next_id = max(self._next_id, track_id + 1)
        return len(self._tracks) - before

    def objects(self) -> list[DetectedObject]:
        """The remembered set as ``tick`` returns it, without running a tick."""
        return [
            tr.obj.model_copy(update={"existence": _probability(tr.belief)}) for tr in self._tracks
        ]
