"""ADR-0035 — IoU-gated object memory.

Pure, ROS-free, stateful. Holds a table of remembered ``DetectedObject``s with
stable track ids. Associates new lifted candidates by label + 3D AABB IoU,
freezes matched objects (no re-write of a known object), creates tracks for new
ones, and evicts objects that were in view but not re-detected (they may have
moved). Out-of-view objects are retained — the camera looking away is not
evidence of absence.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from openral_core.schemas import DetectedObject

from openral_world_state.object_lift import aabb_iou_3d

__all__ = ["ObjectMemory"]


@dataclass
class _Tracked:
    obj: DetectedObject
    last_seen_ns: int
    miss_count: int


class ObjectMemory:
    """Stateful, IoU-gated spatial memory of detected objects.

    Args:
        iou_threshold: Minimum 3D AABB IoU (same label) to associate a new
            detection with an existing track.
        max_misses: Evict an in-view, unmatched track after this many
            consecutive misses. ``1`` ⇒ remove if not re-seen the next run.

    Example:
        >>> from openral_core.schemas import DetectedObject, Pose6D
        >>> mem = ObjectMemory(iou_threshold=0.3, max_misses=1)
        >>> cand = DetectedObject(
        ...     label="cup",
        ...     confidence=0.9,
        ...     pose=Pose6D(xyz=(0.5, 0.5, 0.5), quat_xyzw=(0, 0, 0, 1), frame_id="map"),
        ...     bbox_3d=(0.0, 0.0, 0.0, 1.0, 1.0, 1.0),
        ... )
        >>> out = mem.tick([cand], stamp_ns=1, in_fov=lambda o: True)
        >>> out[0].track_id
        0
    """

    def __init__(self, *, iou_threshold: float = 0.3, max_misses: int = 1) -> None:
        """Validate the IoU threshold and miss budget."""
        if not 0.0 <= iou_threshold <= 1.0:
            raise ValueError(f"iou_threshold must be in [0,1]; got {iou_threshold}")
        if max_misses < 1:
            raise ValueError(f"max_misses must be >= 1; got {max_misses}")
        self._iou = iou_threshold
        self._max_misses = max_misses
        self._tracks: list[_Tracked] = []
        self._next_id = 0

    def tick(
        self,
        candidates: list[DetectedObject],
        *,
        stamp_ns: int,
        in_fov: Callable[[DetectedObject], bool],
    ) -> list[DetectedObject]:
        """Associate candidates, freeze matches, create new, evict in-view misses.

        Args:
            candidates: Newly lifted ``DetectedObject``s for this run (map frame).
            stamp_ns: Run timestamp in nanoseconds (stored as last-seen).
            in_fov: Predicate — was this remembered object in the camera's view
                this run? Unmatched in-view objects accrue a miss; out-of-view
                objects are retained.

        Returns:
            The current remembered object set (stable ``track_id``s).
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
                tr.miss_count = 0
                if cand.confidence > tr.obj.confidence:
                    tr.obj = tr.obj.model_copy(update={"confidence": cand.confidence})
                matched.add(best_i)
            else:
                self._tracks.append(
                    _Tracked(
                        obj=cand.model_copy(update={"track_id": self._next_id}),
                        last_seen_ns=stamp_ns,
                        miss_count=0,
                    ),
                )
                matched.add(len(self._tracks) - 1)
                self._next_id += 1

        survivors: list[_Tracked] = []
        for i, tr in enumerate(self._tracks):
            if i in matched:
                survivors.append(tr)
                continue
            if in_fov(tr.obj):
                tr.miss_count += 1
                if tr.miss_count >= self._max_misses:
                    continue
            survivors.append(tr)
        self._tracks = survivors
        return [tr.obj for tr in self._tracks]
