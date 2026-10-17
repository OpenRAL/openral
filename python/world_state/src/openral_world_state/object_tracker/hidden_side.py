"""Grow a PROVISIONAL track's box into the space the robot hides (issue #418, OT-6/OT-14).

A provisional fit sees only what the robot leaves visible, so its box can stop short of the
object on the side the robot covers (#418 PickOrange r6po_dbg4: orange_3 under the open jaw,
a 5.2 cm box on a 5.8 cm orange; the gripper touched the hidden half and stopped). Once the
tracker knows the object's size, from its own earlier clear fit or from same-label
neighbours it measured clear, the box grows along each horizontal axis to that size, on the
side the robot hides, and only into space the camera cannot see because the robot is in
front. The grown box is K2 exemption geometry only: a provisional track is never a payload
(K3 refuses it).

Pure numpy, no ROS, no clock.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
from openral_core import IntrinsicsPinhole, PlaceRegion, Pose6D
from openral_core.geometry import homogeneous_from_quat_xyz
from openral_core.perception_geometry import VoxelLattice

__all__ = ["grow_hidden_side"]

#: Robot / band pixels of a robot mask (``fit.UNCERTAIN_PIXEL`` and up).
_ROBOT_MIN = 128
#: Fraction of the new face's central samples that must lie behind the robot.
_HIDDEN_FRACTION = 0.5


def _local(points: NDArray[np.float64], box: PlaceRegion) -> NDArray[np.float64]:
    t = homogeneous_from_quat_xyz(box.pose.xyz, box.pose.quat_xyzw)
    return (points - t[:3, 3]) @ t[:3, :3]


def _inside(points: NDArray[np.float64], box: PlaceRegion) -> NDArray[np.bool_]:
    return np.asarray(np.all(np.abs(_local(points, box)) <= np.asarray(box.half_extents), axis=1))


def _project(
    points: NDArray[np.float64], k: IntrinsicsPinhole, t_base_from_cam: NDArray[np.float64]
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.float64], NDArray[np.bool_]]:
    t_cam = np.linalg.inv(t_base_from_cam)
    cam = points @ t_cam[:3, :3].T + t_cam[:3, 3]
    z = cam[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = np.floor(k.fx * cam[:, 0] / z + k.cx).astype(np.int64)
        v = np.floor(k.fy * cam[:, 1] / z + k.cy).astype(np.int64)
    ok = (z > 0) & (u >= 0) & (u < k.width) & (v >= 0) & (v < k.height)
    return np.where(ok, u, 0), np.where(ok, v, 0), z, ok


def _grown(box: PlaceRegion, axis: int, sign: int, g: float) -> PlaceRegion:
    t = homogeneous_from_quat_xyz(box.pose.xyz, box.pose.quat_xyzw)
    centre = t[:3, 3] + t[:3, axis] * sign * g / 2.0
    half = list(box.half_extents)
    half[axis] += g / 2.0
    return box.model_copy(
        update={
            "half_extents": (half[0], half[1], half[2]),
            "pose": Pose6D(
                xyz=(float(centre[0]), float(centre[1]), float(centre[2])),
                quat_xyzw=box.pose.quat_xyzw,
                frame_id=box.pose.frame_id,
            ),
        }
    )


def _slab(box: PlaceRegion, axis: int, sign: int, g: float) -> PlaceRegion:
    """The part ``_grown(box, axis, sign, g)`` adds."""
    t = homogeneous_from_quat_xyz(box.pose.xyz, box.pose.quat_xyzw)
    h = box.half_extents[axis]
    centre = t[:3, 3] + t[:3, axis] * sign * (h + g / 2.0)
    half = list(box.half_extents)
    half[axis] = g / 2.0
    return box.model_copy(
        update={
            "half_extents": (half[0], half[1], half[2]),
            "pose": Pose6D(
                xyz=(float(centre[0]), float(centre[1]), float(centre[2])),
                quat_xyzw=box.pose.quat_xyzw,
                frame_id=box.pose.frame_id,
            ),
        }
    )


def _samples(box: PlaceRegion, n: int, spread: float) -> NDArray[np.float64]:
    """An ``n``^3 lattice of points over the central ``spread`` of ``box``."""
    s = np.linspace(-spread, spread, n)
    local = np.array([(a, b, c) for a in s for b in s for c in s]) * np.asarray(box.half_extents)
    t = homogeneous_from_quat_xyz(box.pose.xyz, box.pose.quat_xyzw)
    return np.asarray(local @ t[:3, :3].T + t[:3, 3], dtype=np.float64)


def _occupied_centres(grid: VoxelLattice, near: PlaceRegion) -> NDArray[np.float64]:
    sx, sy, _ = grid.size
    idx = np.flatnonzero(grid.occupancy)
    ijk = np.stack([idx % sx, (idx // sx) % sy, idx // (sx * sy)], axis=1).astype(np.float64)
    centres = (ijk + 0.5) * grid.resolution @ grid.rotation().T + np.asarray(grid.origin)
    return np.asarray(centres[_inside(centres, near)], dtype=np.float64)


def _hidden(  # reason: one frame's evidence, passed through
    box: PlaceRegion,
    slab: PlaceRegion,
    depth_m: NDArray[np.float64],
    robot_mask: NDArray[np.uint8],
    k: IntrinsicsPinhole,
    t_base_from_cam: NDArray[np.float64],
    grid: VoxelLattice,
    others: Sequence[PlaceRegion],
    support_z: float,
    tol_m: float,
) -> bool:
    """Is ``slab`` (grown onto ``box``) space the robot hides, holding no other body?

    - Its central samples are mostly behind a robot pixel (the measured surface there is the
      robot, nearer than the sample).
    - No sample lies in ``others`` (another track's box).
    - No occupied cell centred in it, more than one cell past ``box``, shows the camera a
      non-robot surface at the cell: a body the camera sees there is not hidden, and is not
      this object (whose visible surface the fit's box holds to within half a cell). The
      support's own cells (centred within one cell of ``support_z``) are skipped.

    An occupied cell the camera sees *past* (observed free at its centre) is no reason to
    stop: a partial-volume edge of the object itself reads so (r6po_dbg4: one of orange_3's
    cells, 2.5 cm), and space observed free holds nothing an exemption could act on.
    """
    pts = _samples(slab, 4, 0.9)
    if any(_inside(pts, o).any() for o in others):
        return False
    u, v, z, ok = _project(pts, k, t_base_from_cam)
    if not ok.all():
        return False  # out of view: unknown, not hidden
    d = depth_m[v, u]
    behind_robot = (robot_mask[v, u] >= _ROBOT_MIN) & np.isfinite(d) & (d < z - tol_m)
    if behind_robot.mean() < _HIDDEN_FRACTION:
        return False
    cells = _occupied_centres(grid, slab)
    cells = cells[cells[:, 2] > support_z + grid.resolution]
    if len(cells):
        cu, cv, cz, cok = _project(cells, k, t_base_from_cam)
        if not cok.all():
            return False
        cd = depth_m[cv, cu]
        surface = (robot_mask[cv, cu] < _ROBOT_MIN) & np.isfinite(cd)
        surface &= np.abs(cd - cz) <= grid.resolution
        out = np.abs(_local(cells, box)) - np.asarray(box.half_extents)
        beyond = np.max(out, axis=1) > grid.resolution
        if (surface & beyond).any():
            return False
    return True


def grow_hidden_side(  # reason: one frame's evidence, passed through
    box: PlaceRegion,
    size_m: float,
    *,
    depth_m: NDArray[np.float64],
    robot_mask: NDArray[np.uint8] | None,
    intrinsics: IntrinsicsPinhole,
    t_base_from_cam: NDArray[np.float64],
    grid: VoxelLattice,
    support_z: float,
    others: Sequence[PlaceRegion] = (),
    tol_m: float = 0.01,
) -> tuple[PlaceRegion, list[tuple[int, int, float]]]:
    """``box`` grown along each horizontal axis toward ``size_m`` where the robot hides it.

    Per horizontal axis whose extent is under ``size_m``, the side whose added slab
    ``_hidden`` accepts, by the largest step in half-cell decrements from the full deficit
    (so never past ``size_m``, within the size plus one cell). Returns the box and the
    ``(axis, sign, grown_m)`` applied. Unchanged without a robot mask, or for a box not
    gravity-aligned.

    Example:
        >>> import numpy as np
        >>> from openral_core import IntrinsicsPinhole, PlaceRegion, Pose6D
        >>> from openral_core.perception_geometry import VoxelLattice
        >>> k = IntrinsicsPinhole(width=64, height=48, fx=60.0, fy=60.0, cx=32.0, cy=24.0)
        >>> t = np.eye(4)
        >>> t[:3, :3] = [[1, 0, 0], [0, -1, 0], [0, 0, -1]]
        >>> t[2, 3] = 0.5
        >>> g = VoxelLattice(
        ...     "b", (-0.2, -0.2, 0.0), (0, 0, 0, 1.0), 0.02, (20, 20, 5), np.zeros(2000, np.uint8)
        ... )
        >>> b = PlaceRegion(
        ...     frame_id="b",
        ...     half_extents=(0.02, 0.02, 0.02),
        ...     pose=Pose6D(xyz=(0.0, 0.0, 0.02), quat_xyzw=(0, 0, 0, 1), frame_id="b"),
        ... )
        >>> depth = np.full((48, 64), 0.5)
        >>> robot = np.zeros((48, 64), np.uint8)
        >>> robot[:, 34:] = 255  # the arm over +x, 0.3 m from the camera
        >>> depth[:, 34:] = 0.3
        >>> out, grown = grow_hidden_side(
        ...     b,
        ...     0.06,
        ...     depth_m=depth,
        ...     robot_mask=robot,
        ...     intrinsics=k,
        ...     t_base_from_cam=t,
        ...     grid=g,
        ...     support_z=0.0,
        ... )
        >>> grown, round(out.half_extents[0], 3), round(out.pose.xyz[0], 3)
        ([(0, 1, 0.02)], 0.03, 0.01)
    """
    t = homogeneous_from_quat_xyz(box.pose.xyz, box.pose.quat_xyzw)
    if robot_mask is None or abs(t[2, 2]) < 0.99:  # noqa: PLR2004  # reason: gravity-aligned
        return box, []
    depth = np.asarray(depth_m, dtype=np.float64)
    step = grid.resolution / 2.0
    grown: list[tuple[int, int, float]] = []
    fitted = box  # each axis is judged against the fitted box, never a half-grown one
    for axis in (0, 1):
        deficit = size_m - 2.0 * fitted.half_extents[axis]
        if deficit <= 0.0:
            continue
        best: tuple[int, float] | None = None
        for sign in (1, -1):
            for g in np.arange(deficit, 0.0, -step):
                slab = _slab(fitted, axis, sign, float(g))
                ok = _hidden(fitted, slab, depth, robot_mask, intrinsics, t_base_from_cam, grid,
                             others, support_z, tol_m)  # fmt: skip
                if ok:
                    if best is None or g > best[1]:
                        best = (sign, float(g))
                    break
        if best is not None:
            box = _grown(box, axis, best[0], best[1])
            grown.append((axis, best[0], round(best[1], 4)))
    return box, grown
