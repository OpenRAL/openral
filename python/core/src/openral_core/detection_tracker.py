"""ADR-0076 — camera-space 2D detection tracker.

Pure, ROS-free, stateful. The 2D analog of ADR-0035's :class:`ObjectMemory`:
assigns a **stable per-camera `det_id`** to each :class:`ObjectDetection2D` by
greedy same-label 2D-AABB IoU association across frames, so an object can be
referred to and de-duplicated **even when the 3D lift cannot run** (RGB-only / no
depth). One tracker instance per camera; identity is camera-space, not a world
entity. The id is propagated into the 3D path by the lift (ADR-0076 §3), so a
physical object keeps one id with or without depth.
"""

from __future__ import annotations

from dataclasses import dataclass

from openral_core.schemas import ObjectDetection2D

__all__ = ["DetectionTracker2D", "aabb_iou_2d"]


def aabb_iou_2d(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
) -> float:
    """2D axis-aligned bbox IoU (pixel space).

    Args:
        a: First box as ``(x_min, y_min, x_max, y_max)``.
        b: Second box as ``(x_min, y_min, x_max, y_max)``.

    Returns:
        Intersection-over-union in ``[0, 1]``; ``0.0`` for disjoint or
        degenerate (zero-area) boxes.

    Example:
        >>> aabb_iou_2d((0, 0, 10, 10), (0, 0, 10, 10))
        1.0
        >>> aabb_iou_2d((0, 0, 10, 10), (100, 100, 110, 110))
        0.0
    """
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    if inter <= 0:
        return 0.0
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


@dataclass
class _Track2D:
    bbox_xyxy: tuple[int, int, int, int]
    label: str
    det_id: int
    miss_count: int


class DetectionTracker2D:
    """Stateful 2D-IoU tracker assigning stable ``det_id``s (ADR-0076).

    Args:
        iou_threshold: Minimum 2D AABB IoU (same label) to associate a detection
            with an existing track.
        max_misses: Retire a track after this many consecutive frames without a
            match (detectors flicker, so a few frames of grace).

    Example:
        >>> trk = DetectionTracker2D()
        >>> a = [ObjectDetection2D(label="milk", confidence=0.9, bbox_xyxy=(10, 10, 50, 50))]
        >>> trk.assign(a)[0].det_id
        0
        >>> b = [ObjectDetection2D(label="milk", confidence=0.9, bbox_xyxy=(12, 11, 52, 51))]
        >>> trk.assign(b)[0].det_id  # same object next frame keeps its id
        0
    """

    def __init__(self, *, iou_threshold: float = 0.3, max_misses: int = 3) -> None:
        """Validate the IoU threshold and miss budget."""
        if not 0.0 <= iou_threshold <= 1.0:
            raise ValueError(f"iou_threshold must be in [0,1]; got {iou_threshold}")
        if max_misses < 1:
            raise ValueError(f"max_misses must be >= 1; got {max_misses}")
        self._iou = iou_threshold
        self._max_misses = max_misses
        self._tracks: list[_Track2D] = []
        self._next_id = 0

    def assign(self, detections: list[ObjectDetection2D]) -> list[ObjectDetection2D]:
        """Stamp each detection with a stable ``det_id`` and age unmatched tracks.

        Greedy highest-confidence-first same-label IoU association: a matched
        detection keeps its track's id (and refreshes the track's box); an
        unmatched detection opens a new track with a fresh monotonic id. Tracks
        not matched this frame accrue a miss and are retired past ``max_misses``.

        Args:
            detections: This frame's 2D detections (``det_id`` ignored on input).

        Returns:
            The same detections, in input order, each with ``det_id`` assigned.
        """
        matched_tracks: set[int] = set()
        # Greedy by confidence so the strongest detection claims a track first.
        order = sorted(range(len(detections)), key=lambda i: detections[i].confidence, reverse=True)
        assigned: dict[int, int] = {}  # detection index -> det_id
        for di in order:
            det = detections[di]
            best_t, best_iou = -1, self._iou
            for ti, tr in enumerate(self._tracks):
                if ti in matched_tracks or tr.label != det.label:
                    continue
                iou = aabb_iou_2d(det.bbox_xyxy, tr.bbox_xyxy)
                if iou >= best_iou:
                    best_iou, best_t = iou, ti
            if best_t >= 0:
                tr = self._tracks[best_t]
                tr.bbox_xyxy = det.bbox_xyxy
                tr.miss_count = 0
                matched_tracks.add(best_t)
                assigned[di] = tr.det_id
            else:
                self._tracks.append(
                    _Track2D(
                        bbox_xyxy=det.bbox_xyxy,
                        label=det.label,
                        det_id=self._next_id,
                        miss_count=0,
                    ),
                )
                matched_tracks.add(len(self._tracks) - 1)
                assigned[di] = self._next_id
                self._next_id += 1

        survivors: list[_Track2D] = []
        for ti, tr in enumerate(self._tracks):
            if ti in matched_tracks:
                survivors.append(tr)
                continue
            tr.miss_count += 1
            if tr.miss_count < self._max_misses:
                survivors.append(tr)
        self._tracks = survivors

        return [det.model_copy(update={"det_id": assigned[i]}) for i, det in enumerate(detections)]
