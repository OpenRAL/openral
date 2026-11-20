"""Geometry for lifting 2D detections to 3D object centers.

Pure, ROS-free. The world-state lifecycle node feeds this with occupied voxel
centers, the camera intrinsics, and homogeneous transforms; it returns
``DetectedObject`` candidates anchored in the map frame. No ROS, no I/O, no
global state — fully unit-testable.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Sequence
from enum import Enum

import numpy as np
from numpy.typing import NDArray
from openral_core.exceptions import ROSConfigError
from openral_core.geometry import homogeneous_from_quat_xyz as _core_homogeneous_from_quat_xyz
from openral_core.schemas import (
    DetectedObject,
    IntrinsicsPinhole,
    ObjectDetection2D,
    Pose6D,
)

__all__ = [
    "MissEvidence",
    "ObjectsLiftError",
    "VoxelFrustumLifter",
    "aabb_iou_3d",
    "build_miss_classifier",
    "decode_occupied_centers",
    "depth_cloud_to_centers_base",
    "homogeneous_from_quat_xyz",
]

# Numerical guards.
_DEPTH_EPS = 1e-6  # minimum +z (metres) for a voxel/point to be "in front" of the camera
# Tolerance on |q|^2 for a grid orientation. Wide enough for a quaternion that
# has been through a wire round-trip, far too tight for the all-zero one an
# unset field carries.
_QUAT_NORM_TOL = 1e-6


class ObjectsLiftError(ValueError):
    """Raised on malformed lift inputs (bad quaternion, grid size mismatch)."""


def homogeneous_from_quat_xyz(
    translation: tuple[float, float, float],
    quat_xyzw: tuple[float, float, float, float],
) -> NDArray[np.float64]:
    """Build a 4x4 homogeneous transform from a translation + xyzw quaternion.

    Args:
        translation: ``(x, y, z)`` translation in metres.
        quat_xyzw: Rotation quaternion as ``(x, y, z, w)``; need not be unit
            length (it is normalised internally).

    Returns:
        A ``(4, 4)`` float64 homogeneous transform matrix.

    Raises:
        ObjectsLiftError: If the quaternion norm is effectively zero.

    Note:
        The math now lives in ``openral_core.geometry.homogeneous_from_quat_xyz``
        — the layer-0 HAL needs the same TF→matrix step for its vision
        attachment bridge and must not reach up to layer 2 for it. This wrapper
        keeps the world-state-local ``ObjectsLiftError`` contract that every
        caller here already handles.
    """
    try:
        return _core_homogeneous_from_quat_xyz(translation, quat_xyzw)
    except ROSConfigError as exc:
        raise ObjectsLiftError(str(exc)) from exc


def decode_occupied_centers(
    *,
    origin: tuple[float, float, float],
    resolution: float,
    size_xyz: tuple[int, int, int],
    occupancy: bytes,
    orientation_xyzw: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
) -> NDArray[np.float64]:
    """Occupied-voxel centers ``(N, 3)`` in the grid's reference frame.

    Cell ``(ix, iy, iz)``'s center is ``origin + R * ((index + 0.5) *
    resolution)``, where ``R`` is ``orientation_xyzw``. A cell is occupied when
    its byte is non-zero.

    ``openral_msgs/OccupancyVoxels`` is an ORIENTED lattice: its cell axes are
    the source map's, not ``header.frame_id``'s, so the rotation is part of
    placing a cell and not decoration. Dropping it puts every returned center in
    the wrong place whenever the robot is not aligned with the map.

    Args:
        origin: ``(x, y, z)`` position of cell ``(0, 0, 0)``'s minimum corner,
            in ``header.frame_id``, in metres.
        resolution: Edge length of a cubic voxel in metres.
        size_xyz: Grid dimensions ``(size_x, size_y, size_z)`` in voxels.
        occupancy: Flat occupancy buffer, one byte per voxel, row-major with x
            fastest: ``idx = ix + size_x * (iy + size_y * iz)``. Non-zero byte
            means occupied.
        orientation_xyzw: Grid orientation as ``(x, y, z, w)``. Defaults to
            identity for callers building an axis-aligned grid themselves;
            a caller decoding a wire message must pass the message's own, and
            an all-zero quaternion (the unset default) is rejected rather than
            read as identity.

    Returns:
        A ``(N, 3)`` float64 array of occupied-voxel centers in the grid's
        reference frame, or shape ``(0, 3)`` when none are occupied.

    Raises:
        ObjectsLiftError: If ``len(occupancy)`` does not equal
            ``size_x * size_y * size_z``, or ``orientation_xyzw`` is not a unit
            quaternion.

    Example:
        >>> centers = decode_occupied_centers(
        ...     origin=(0.0, 0.0, 0.0),
        ...     resolution=0.1,
        ...     size_xyz=(2, 1, 1),
        ...     occupancy=bytes([0, 1]),
        ... )
        >>> tuple(round(float(v), 3) for v in centers[0])
        (0.15, 0.05, 0.05)
    """
    sx, sy, sz = size_xyz
    expected = sx * sy * sz
    arr = np.frombuffer(occupancy, dtype=np.uint8)
    if arr.size != expected:
        raise ObjectsLiftError(
            f"occupancy length {arr.size} != size_x*size_y*size_z {expected}",
        )
    quat = np.asarray(orientation_xyzw, dtype=np.float64)
    norm2 = float(quat @ quat)
    if not np.isfinite(norm2) or abs(norm2 - 1.0) > _QUAT_NORM_TOL:
        raise ObjectsLiftError(
            f"orientation {orientation_xyzw!r} is not a unit quaternion "
            "(an unset OccupancyVoxels.orientation is all zeros, not identity)",
        )
    occ = np.nonzero(arr)[0]
    if occ.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    ix = occ % sx
    iy = (occ // sx) % sy
    iz = occ // (sx * sy)
    idx = np.stack([ix, iy, iz], axis=1).astype(np.float64)
    local = (idx + 0.5) * float(resolution)
    x, y, z, w = quat
    rot = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    return np.asarray(origin, dtype=np.float64) + local @ rot.T


def depth_cloud_to_centers_base(
    points_cloud: NDArray[np.float64],
    t_base_from_cloud: NDArray[np.float64],
    *,
    max_points: int = 0,
) -> NDArray[np.float64]:
    """Depth-cloud points → occupied centers ``(M, 3)`` in the base frame (#11).

    The octomap-free fallback depth source for ``VoxelFrustumLifter``. Drops
    non-finite returns (depth holes), uniformly subsamples to ``max_points`` so
    a dense cloud can't stall the per-detection projection, and maps the cloud
    from its optical frame into the robot base frame. The result is
    interchangeable with ``decode_occupied_centers`` output as the lifter's
    ``occupied_centers_base`` argument.

    Args:
        points_cloud: ``(N, 3)`` raw cloud points in the camera optical frame.
        t_base_from_cloud: ``(4, 4)`` transform mapping cloud-frame points into
            the robot base frame.
        max_points: Cap on returned points (uniform stride). ``0`` = no cap.

    Returns:
        An ``(M, 3)`` float64 array of finite points in the base frame, or shape
        ``(0, 3)`` when the cloud has no finite points.
    """
    pts: NDArray[np.float64] = np.asarray(points_cloud, dtype=np.float64).reshape(-1, 3)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if pts.shape[0] == 0:
        return np.empty((0, 3), dtype=np.float64)
    if max_points and pts.shape[0] > max_points:
        stride = pts.shape[0] // max_points + 1
        pts = pts[::stride]
    homog = np.hstack([pts, np.ones((pts.shape[0], 1), dtype=np.float64)])
    return np.ascontiguousarray((homog @ np.asarray(t_base_from_cloud, dtype=np.float64).T)[:, :3])


def aabb_iou_3d(
    a: tuple[float, float, float, float, float, float],
    b: tuple[float, float, float, float, float, float],
) -> float:
    """3D axis-aligned bbox IoU.

    Args:
        a: First box as ``(xmin, ymin, zmin, xmax, ymax, zmax)``.
        b: Second box as ``(xmin, ymin, zmin, xmax, ymax, zmax)``.

    Returns:
        Intersection-over-union in ``[0, 1]``; ``0.0`` for disjoint or
        degenerate (zero-volume) boxes.
    """
    ix = max(0.0, min(a[3], b[3]) - max(a[0], b[0]))
    iy = max(0.0, min(a[4], b[4]) - max(a[1], b[1]))
    iz = max(0.0, min(a[5], b[5]) - max(a[2], b[2]))
    inter = ix * iy * iz
    if inter <= 0.0:
        return 0.0
    vol_a = max(0.0, a[3] - a[0]) * max(0.0, a[4] - a[1]) * max(0.0, a[5] - a[2])
    vol_b = max(0.0, b[3] - b[0]) * max(0.0, b[4] - b[1]) * max(0.0, b[5] - b[2])
    union = vol_a + vol_b - inter
    return inter / union if union > 0.0 else 0.0


class VoxelFrustumLifter:
    """Lift 2D detections to 3D object centres via voxel-frustum K-nearest.

    For each detection: project the occupied voxels into the camera image, keep
    those inside the (resolution-scaled) box, take the ``k_nearest`` whose
    projection is closest to the box centre, and return their mean position +
    axis-aligned bbox in the map frame. Detections with fewer than
    ``min_voxels`` in-box voxels are skipped (insufficient 3D evidence).
    """

    def __init__(self, *, k_nearest: int = 25, min_voxels: int = 3) -> None:
        """Validate K and the min-voxel floor."""
        if k_nearest < 1:
            raise ValueError(f"k_nearest must be >= 1; got {k_nearest}")
        if min_voxels < 1:
            raise ValueError(f"min_voxels must be >= 1; got {min_voxels}")
        self._k = k_nearest
        self._min_voxels = min_voxels

    def lift(
        self,
        *,
        detections: Sequence[ObjectDetection2D],
        occupied_centers_base: NDArray[np.float64],
        intrinsics: IntrinsicsPinhole,
        frame_size: tuple[int, int],
        t_cam_from_base: NDArray[np.float64],
        t_map_from_base: NDArray[np.float64],
        map_frame: str = "map",
        pad_m: float = 0.0,
    ) -> list[DetectedObject]:
        """Lift each 2D detection to a ``DetectedObject`` in the map frame.

        Args:
            detections: Per-camera 2D detections (pixel ``bbox_xyxy``).
            occupied_centers_base: ``(N, 3)`` occupied voxel centres in the
                robot base frame.
            intrinsics: Pinhole intrinsics of the detection camera.
            frame_size: ``(width, height)`` in pixels of the frame the detector
                ran on; the box is scaled to the intrinsics resolution.
            t_cam_from_base: ``(4, 4)`` transform mapping base-frame points into
                the camera optical frame.
            t_map_from_base: ``(4, 4)`` transform mapping base-frame points into
                the map frame.
            map_frame: tf2 frame id stamped on the returned poses.
            pad_m: Half the extent one occupied point stands for (half a voxel
                edge), added on every face of the box. The points are cell
                *centres*: without it one voxel layer lifts to a zero-volume box,
                whose IoU is 0 even with itself.

        Returns:
            One ``DetectedObject`` per detection with enough in-box voxels;
            detections below ``min_voxels`` are omitted.
        """
        out: list[DetectedObject] = []
        n = int(occupied_centers_base.shape[0])
        if n == 0 or not detections:
            return out

        homo = np.concatenate(
            [occupied_centers_base, np.ones((n, 1), dtype=np.float64)],
            axis=1,
        )  # (N,4)
        cam = (t_cam_from_base @ homo.T).T[:, :3]  # (N,3) camera-optical
        z = cam[:, 2]
        in_front = z > _DEPTH_EPS
        u = np.full(n, -1.0, dtype=np.float64)
        v = np.full(n, -1.0, dtype=np.float64)
        u[in_front] = intrinsics.fx * cam[in_front, 0] / z[in_front] + intrinsics.cx
        v[in_front] = intrinsics.fy * cam[in_front, 1] / z[in_front] + intrinsics.cy

        map_pts = (t_map_from_base @ homo.T).T[:, :3]  # (N,3) map frame

        fw, fh = frame_size
        scale_x = intrinsics.width / fw if fw else 1.0
        scale_y = intrinsics.height / fh if fh else 1.0

        for det in detections:
            x0, y0, x1, y1 = det.bbox_xyxy
            bx0, bx1 = x0 * scale_x, x1 * scale_x
            by0, by1 = y0 * scale_y, y1 * scale_y
            ucen, vcen = (bx0 + bx1) / 2.0, (by0 + by1) / 2.0
            inside = in_front & (u >= bx0) & (u <= bx1) & (v >= by0) & (v <= by1)
            idx = np.nonzero(inside)[0]
            if idx.size < self._min_voxels:
                continue
            d2 = (u[idx] - ucen) ** 2 + (v[idx] - vcen) ** 2
            nearest = idx[np.argpartition(d2, self._k)[: self._k]] if idx.size > self._k else idx
            pts = map_pts[nearest]
            center = pts.mean(axis=0)
            mn = pts.min(axis=0)
            mx = pts.max(axis=0)
            out.append(
                DetectedObject(
                    label=det.label,
                    confidence=det.confidence,
                    pose=Pose6D(
                        xyz=(float(center[0]), float(center[1]), float(center[2])),
                        quat_xyzw=(0.0, 0.0, 0.0, 1.0),
                        frame_id=map_frame,
                    ),
                    bbox_3d=(
                        float(mn[0]) - pad_m,
                        float(mn[1]) - pad_m,
                        float(mn[2]) - pad_m,
                        float(mx[0]) + pad_m,
                        float(mx[1]) + pad_m,
                        float(mx[2]) + pad_m,
                    ),
                    # Carry the detection-time id through the lift so a
                    # physical object keeps one id across 2D (in_view) and 3D
                    # (scene_objects). >= 0 → propagate; -1 (untracked) → None,
                    # leaving ObjectMemory to mint as before.
                    track_id=det.det_id if det.det_id >= 0 else None,
                ),
            )
        return out


class MissEvidence(str, Enum):
    """What one sensing run says about a remembered object nothing matched (issue #636).

    ``NONE``: it could not be observed — no box, not wholly in view, out of range,
    occluded, or no depth behind it to see through to. ``OCCUPIED``: its box still
    holds occupied points, so it is probably there and merely undetected. ``FREE``:
    the sensor saw through the box to something behind it — a strong miss.
    """

    NONE = "none"
    OCCUPIED = "occupied"
    FREE = "free"


def build_miss_classifier(
    *,
    occupied_centers_map: NDArray[np.float64],
    intrinsics: IntrinsicsPinhole,
    t_cam_from_map: NDArray[np.float64],
    max_range_m: float,
    min_points: int = 3,
    margin_m: float = 0.03,
) -> Callable[[DetectedObject], MissEvidence]:
    """Return ``classify(obj)``: what this camera's depth evidence says about an undetected object.

    The object is observable only when all eight corners of its ``bbox_3d`` project
    inside the image, in front of the camera and within ``max_range_m``. Then, over
    the occupied points (voxel centres or depth returns, map frame):

    * ``min_points`` or more inside the box, grown by ``margin_m`` on every face
      (the box is a frozen lift of surface points; depth noise and map drift must
      not empty it), at least one of them more than ``margin_m`` above the box's
      floor → ``OCCUPIED``. Points in that floor band alone are the shelf a removed
      object stood on, not the object; a box no taller than twice the margin has
      no band and is counted whole.
    * otherwise, among the points projecting into the box's image rectangle:
      ``min_points`` or more beyond the box by over ``margin_m``, and fewer than
      ``min_points`` nearer than it → ``FREE``. Anything else (an occluder in front,
      however partial; a depth hole; nothing mapped behind) → ``NONE``.

    The cloud is projected once, on the first object asked about, and shared by all.

    Args:
        occupied_centers_map: ``(N, 3)`` occupied points in the map frame.
        intrinsics: Pinhole intrinsics of the camera.
        t_cam_from_map: ``(4, 4)`` transform, map frame → camera optical frame.
        max_range_m: The detector's working range along the optical axis.
        min_points: Points needed to call the box occupied or seen through.
        margin_m: Tolerance on the box's faces and the depth band of its support.

    Returns:
        ``classify(obj) -> MissEvidence``; ``NONE`` when this camera gives none.

    Example:
        >>> from openral_core.schemas import DetectedObject, IntrinsicsPinhole, Pose6D
        >>> cup = DetectedObject(
        ...     label="cup",
        ...     confidence=0.9,
        ...     pose=Pose6D(xyz=(0.0, 0.0, 2.0), quat_xyzw=(0, 0, 0, 1), frame_id="map"),
        ...     bbox_3d=(-0.1, -0.1, 1.9, 0.1, 0.1, 2.1),
        ... )
        >>> k = IntrinsicsPinhole(width=100, height=100, fx=100, fy=100, cx=50, cy=50)
        >>> wall = np.array([[x, 0.0, 3.0] for x in (-0.05, 0.0, 0.05)])
        >>> classify = build_miss_classifier(
        ...     occupied_centers_map=wall, intrinsics=k, t_cam_from_map=np.eye(4), max_range_m=5.0
        ... )
        >>> classify(cup).value
        'free'
    """
    pts = occupied_centers_map

    def project(points: NDArray[np.float64]) -> tuple[NDArray[np.float64], ...]:
        cam = points @ t_cam_from_map[:3, :3].T + t_cam_from_map[:3, 3]
        z = np.where(cam[:, 2] > _DEPTH_EPS, cam[:, 2], np.nan)  # behind the camera → NaN
        return (
            intrinsics.fx * cam[:, 0] / z + intrinsics.cx,
            intrinsics.fy * cam[:, 1] / z + intrinsics.cy,
            z,
        )

    projected: list[tuple[NDArray[np.float64], ...]] = []

    def classify(obj: DetectedObject) -> MissEvidence:
        box = obj.bbox_3d
        if box is None or pts.shape[0] == 0:
            return MissEvidence.NONE
        lo = np.asarray(box[:3], dtype=np.float64)
        hi = np.asarray(box[3:], dtype=np.float64)
        corners = np.array(list(itertools.product(*zip(lo, hi, strict=True))), dtype=np.float64)
        cu, cv, cz = project(corners)
        in_view = (cu >= 0.0) & (cu <= intrinsics.width) & (cv >= 0.0) & (cv <= intrinsics.height)
        if not (in_view.all() and cz.max() <= max_range_m):  # NaN compares False
            return MissEvidence.NONE

        inside = pts[((pts >= lo - margin_m) & (pts <= hi + margin_m)).all(axis=1)]
        band_top = lo[2] + margin_m if hi[2] - lo[2] > 2.0 * margin_m else -np.inf
        if inside.shape[0] >= min_points and bool((inside[:, 2] > band_top).any()):
            return MissEvidence.OCCUPIED

        if not projected:
            projected.append(project(pts))
        u, v, z = projected[0]
        in_rect = (u >= cu.min()) & (u <= cu.max()) & (v >= cv.min()) & (v <= cv.max())
        behind = int((in_rect & (z > cz.max() + margin_m)).sum())
        in_front = int((in_rect & (z < cz.min() - margin_m)).sum())
        # ponytail: counts points, not rays — space the depth sensor never observed
        # (the view is this detection camera's, the points are another sensor's or a
        # map's) reads the same as free when something is mapped behind it. A
        # z-buffered depth image from this camera (object_tracker.fit.visible_fraction)
        # is the upgrade if that bites.
        if behind >= min_points and in_front < min_points:
            return MissEvidence.FREE
        return MissEvidence.NONE

    return classify
