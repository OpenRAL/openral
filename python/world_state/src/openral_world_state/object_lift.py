"""ADR-0035 — geometry for lifting 2D detections to 3D object centers.

Pure, ROS-free. The world-state lifecycle node feeds this with occupied voxel
centers, the camera intrinsics, and homogeneous transforms; it returns
``DetectedObject`` candidates anchored in the map frame. No ROS, no I/O, no
global state — fully unit-testable.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from numpy.typing import NDArray
from openral_core.schemas import (
    DetectedObject,
    IntrinsicsPinhole,
    ObjectDetection2D,
    Pose6D,
)

__all__ = [
    "ObjectsLiftError",
    "VoxelFrustumLifter",
    "aabb_iou_3d",
    "build_in_fov_predicate",
    "decode_occupied_centers",
    "depth_cloud_to_centers_base",
    "homogeneous_from_quat_xyz",
]

# Numerical guards.
_QUAT_NORM_EPS = 1e-12  # below this squared-norm a quaternion is degenerate
_DEPTH_EPS = 1e-6  # minimum +z (metres) for a voxel/point to be "in front" of the camera


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
    """
    x, y, z, w = quat_xyzw
    n = x * x + y * y + z * z + w * w
    if n < _QUAT_NORM_EPS:
        raise ObjectsLiftError(f"degenerate quaternion {quat_xyzw!r}")
    s = 2.0 / n
    xx, yy, zz = x * x * s, y * y * s, z * z * s
    xy, xz, yz = x * y * s, x * z * s, y * z * s
    wx, wy, wz = w * x * s, w * y * s, w * z * s
    m = np.eye(4, dtype=np.float64)
    m[0, 0] = 1.0 - (yy + zz)
    m[0, 1] = xy - wz
    m[0, 2] = xz + wy
    m[1, 0] = xy + wz
    m[1, 1] = 1.0 - (xx + zz)
    m[1, 2] = yz - wx
    m[2, 0] = xz - wy
    m[2, 1] = yz + wx
    m[2, 2] = 1.0 - (xx + yy)
    m[0, 3], m[1, 3], m[2, 3] = translation
    return m


def decode_occupied_centers(
    *,
    origin: tuple[float, float, float],
    resolution: float,
    size_xyz: tuple[int, int, int],
    occupancy: bytes,
) -> NDArray[np.float64]:
    """Occupied-voxel centers ``(N, 3)`` in the grid frame.

    Cell ``(ix, iy, iz)`` center is ``origin + (index + 0.5) * resolution``. A
    cell is occupied when its byte is non-zero.

    Args:
        origin: ``(x, y, z)`` grid origin (corner of cell ``(0, 0, 0)``) in
            metres.
        resolution: Edge length of a cubic voxel in metres.
        size_xyz: Grid dimensions ``(size_x, size_y, size_z)`` in voxels.
        occupancy: Flat occupancy buffer, one byte per voxel, row-major with x
            fastest: ``idx = ix + size_x * (iy + size_y * iz)``. Non-zero byte
            means occupied.

    Returns:
        A ``(N, 3)`` float64 array of occupied-voxel centers in the grid frame,
        or shape ``(0, 3)`` when none are occupied.

    Raises:
        ObjectsLiftError: If ``len(occupancy)`` does not equal
            ``size_x * size_y * size_z``.
    """
    sx, sy, sz = size_xyz
    expected = sx * sy * sz
    arr = np.frombuffer(occupancy, dtype=np.uint8)
    if arr.size != expected:
        raise ObjectsLiftError(
            f"occupancy length {arr.size} != size_x*size_y*size_z {expected}",
        )
    occ = np.nonzero(arr)[0]
    if occ.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    ix = occ % sx
    iy = (occ // sx) % sy
    iz = occ // (sx * sy)
    idx = np.stack([ix, iy, iz], axis=1).astype(np.float64)
    return np.asarray(origin, dtype=np.float64) + (idx + 0.5) * float(resolution)


def depth_cloud_to_centers_base(
    points_cloud: NDArray[np.float64],
    t_base_from_cloud: NDArray[np.float64],
    *,
    max_points: int = 0,
) -> NDArray[np.float64]:
    """Depth-cloud points → occupied centers ``(M, 3)`` in the base frame (#11).

    The octomap-free fallback depth source for :class:`VoxelFrustumLifter`. Drops
    non-finite returns (depth holes), uniformly subsamples to ``max_points`` so
    a dense cloud can't stall the per-detection projection, and maps the cloud
    from its optical frame into the robot base frame. The result is
    interchangeable with :func:`decode_occupied_centers` output as the lifter's
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
                        float(mn[0]),
                        float(mn[1]),
                        float(mn[2]),
                        float(mx[0]),
                        float(mx[1]),
                        float(mx[2]),
                    ),
                    # ADR-0076: carry the detection-time id through the lift so a
                    # physical object keeps one id across 2D (in_view) and 3D
                    # (scene_objects). >= 0 → propagate; -1 (untracked) → None,
                    # leaving ObjectMemory to mint as before.
                    track_id=det.det_id if det.det_id >= 0 else None,
                ),
            )
        return out


def build_in_fov_predicate(
    *,
    intrinsics: IntrinsicsPinhole,
    t_cam_from_map: NDArray[np.float64],
) -> Callable[[DetectedObject], bool]:
    """Return a predicate: does a map-frame object project into the camera image?

    Used by ``ObjectMemory`` to decide whether an unmatched object *should* have
    been re-seen (in view => a real miss) or is merely out of view (retain).
    Objects whose centre is behind the camera or outside the image bounds return
    ``False``. Image bounds are inclusive of the far edge (``0 <= u <= width``),
    using continuous pixel coordinates.

    Args:
        intrinsics: Pinhole intrinsics of the camera; ``width``/``height`` bound
            the image rectangle.
        t_cam_from_map: ``(4, 4)`` transform mapping map-frame points into the
            camera optical frame.

    Returns:
        A callable ``in_fov(obj) -> bool``.
    """
    w, h = float(intrinsics.width), float(intrinsics.height)

    def in_fov(obj: DetectedObject) -> bool:
        x, y, z = obj.pose.xyz
        p = t_cam_from_map @ np.array([x, y, z, 1.0], dtype=np.float64)
        if p[2] <= _DEPTH_EPS:
            return False
        u = intrinsics.fx * p[0] / p[2] + intrinsics.cx
        v = intrinsics.fy * p[1] / p[2] + intrinsics.cy
        return bool(0.0 <= u <= w and 0.0 <= v <= h)

    return in_fov
