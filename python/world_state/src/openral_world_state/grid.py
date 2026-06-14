"""Occupancy-grid queries + approach-pose refinement (ADR-0044 Phase 2).

Pure planning-layer geometry over a ``nav_msgs/OccupancyGrid`` snapshot —
deliberately **not** a safety surface. The ADR-0030 Phase 6 kernel gate
(``/openral/check_nav_goal``, on its own branch) answers "is this goal safe?"
as enforcement; this module answers "which nearby pose is free *and* sees the
object?" as a proposal. A refined pose still crosses every downstream check.

ROS-free by design (like the rest of ``openral_world_state``):
:meth:`OccupancyGridIndex.from_msg` duck-types the message, so units tests run
on plain objects and the live caller passes the real subscription payload.

Occupancy semantics follow ``nav_msgs/OccupancyGrid``: ``-1`` unknown,
``0..100`` occupancy probability. We are conservative — a cell is *free* only
when ``0 <= value <= FREE_MAX``; unknown and mid-probability cells block both
placement and line-of-sight.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from numpy.typing import NDArray
from openral_core import ApproachViewpoint, Pose6D

from openral_world_state.spatial_memory import compute_approach_viewpoint

__all__ = ["FREE_MAX", "OccupancyGridIndex", "refine_approach_pose"]

_GRID_NDIM = 2
FREE_MAX = 25
"""Highest ``nav_msgs/OccupancyGrid`` value still treated as free (map_server's
``free_thresh`` is ~20/100; everything above, plus ``-1`` unknown, blocks)."""


def _yaw_from_quat_xyzw(quat: tuple[float, float, float, float]) -> float:
    x, y, z, w = quat
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


class OccupancyGridIndex:
    """Queryable view over one ``nav_msgs/OccupancyGrid`` snapshot.

    Args:
        data: ``(height, width)`` int8 occupancy array (row-major, as on the
            wire: row 0 is the cell at the grid origin).
        resolution_m: Cell edge length in metres.
        origin_xy: World x, y of the corner of cell ``(0, 0)``.
        origin_yaw: Grid rotation about +Z (slam_toolbox maps are normally
            axis-aligned, but the origin pose may carry yaw).

    Example:
        >>> import numpy as np
        >>> idx = OccupancyGridIndex(
        ...     np.zeros((4, 4), dtype=np.int8), resolution_m=0.5, origin_xy=(0.0, 0.0)
        ... )
        >>> idx.is_free(1.0, 1.0)
        True
    """

    def __init__(
        self,
        data: NDArray[np.int8],
        *,
        resolution_m: float,
        origin_xy: tuple[float, float],
        origin_yaw: float = 0.0,
    ) -> None:
        """See the class docstring for argument semantics."""
        if data.ndim != _GRID_NDIM:
            raise ValueError(f"occupancy data must be 2-D (height, width); got {data.shape}")
        if resolution_m <= 0.0:
            raise ValueError(f"resolution_m must be > 0; got {resolution_m}")
        self._data = data
        self._res = float(resolution_m)
        self._ox, self._oy = float(origin_xy[0]), float(origin_xy[1])
        self._cos = math.cos(origin_yaw)
        self._sin = math.sin(origin_yaw)

    @classmethod
    def from_msg(cls, msg: Any) -> OccupancyGridIndex:  # noqa: ANN401  # reason: duck-typed nav_msgs/OccupancyGrid keeps world_state ROS-free
        """Build from a (duck-typed) ``nav_msgs/OccupancyGrid`` message."""
        info = msg.info
        data = np.asarray(msg.data, dtype=np.int8).reshape(info.height, info.width)
        origin = info.origin
        yaw = _yaw_from_quat_xyzw(
            (
                origin.orientation.x,
                origin.orientation.y,
                origin.orientation.z,
                origin.orientation.w,
            )
        )
        return cls(
            data,
            resolution_m=float(info.resolution),
            origin_xy=(float(origin.position.x), float(origin.position.y)),
            origin_yaw=yaw,
        )

    @property
    def resolution_m(self) -> float:
        """Cell edge length in metres."""
        return self._res

    def world_to_cell(self, x: float, y: float) -> tuple[int, int] | None:
        """``(row, col)`` containing the world point, or ``None`` off-grid."""
        dx, dy = x - self._ox, y - self._oy
        gx = self._cos * dx + self._sin * dy
        gy = -self._sin * dx + self._cos * dy
        col = math.floor(gx / self._res)
        row = math.floor(gy / self._res)
        h, w = self._data.shape
        if 0 <= row < h and 0 <= col < w:
            return (row, col)
        return None

    def is_free(self, x: float, y: float, *, inflation_m: float = 0.0) -> bool:
        """True iff the point — and every cell within ``inflation_m`` of it — is free.

        Conservative: the disc is measured in **world space** from the query
        point to each cell's nearest face (not centre-cell to centre-cell,
        which under-inflates near cell edges). Off-grid points, and any part
        of the disc extending off-grid, count as blocked — the robot footprint
        must sit inside known-free space.
        """
        centre = self.world_to_cell(x, y)
        if centre is None:
            return False
        row0, col0 = centre
        h, w = self._data.shape
        # Work in (rotated) grid coordinates so cell rectangles are axis-aligned.
        dx, dy = x - self._ox, y - self._oy
        gx = self._cos * dx + self._sin * dy
        gy = -self._sin * dx + self._cos * dy
        steps = max(0, math.ceil(inflation_m / self._res) + 1)
        for dr in range(-steps, steps + 1):
            for dc in range(-steps, steps + 1):
                r, c = row0 + dr, col0 + dc
                # Nearest point of cell (r, c) to the query point, grid coords.
                cx = min(max(gx, c * self._res), (c + 1) * self._res)
                cy = min(max(gy, r * self._res), (r + 1) * self._res)
                if math.hypot(cx - gx, cy - gy) > inflation_m + 1e-12:
                    continue
                if not (0 <= r < h and 0 <= c < w):
                    return False
                value = int(self._data[r, c])
                if value < 0 or value > FREE_MAX:
                    return False
        return True

    def line_of_sight(self, a_xy: tuple[float, float], b_xy: tuple[float, float]) -> bool:
        """True iff every cell strictly before ``b`` on the segment is free.

        Bresenham over the grid; unknown cells block sight and either endpoint
        off-grid is no sight. The **final cell is exempt**: the target is
        typically an object whose own footprint is occupied in the 2-D grid
        (a mug shares its cell with the counter it sits on) — sight means an
        unobstructed ray *up to* it, not through it.
        """
        a = self.world_to_cell(*a_xy)
        b = self.world_to_cell(*b_xy)
        if a is None or b is None:
            return False
        r0, c0 = a
        r1, c1 = b
        dr, dc = abs(r1 - r0), abs(c1 - c0)
        sr = 1 if r1 >= r0 else -1
        sc = 1 if c1 >= c0 else -1
        err = dc - dr
        r, c = r0, c0
        while True:
            if (r, c) == (r1, c1):
                return True
            value = int(self._data[r, c])
            if value < 0 or value > FREE_MAX:
                return False
            e2 = 2 * err
            if e2 > -dr:
                err -= dr
                c += sc
            if e2 < dc:
                err += dc
                r += sr


def refine_approach_pose(
    grid: OccupancyGridIndex,
    viewpoint: ApproachViewpoint,
    target_xyz: tuple[float, float, float],
    *,
    inflation_m: float = 0.25,
    max_radius_m: float = 2.0,
    min_standoff_m: float | None = None,
    max_standoff_m: float | None = None,
) -> ApproachViewpoint | None:
    """Snap an :class:`~openral_core.ApproachViewpoint` to the occupancy grid.

    Returns the viewpoint unchanged when its pose already sits on a free cell
    (under ``inflation_m``) with line-of-sight to the target. Otherwise ring-
    searches outward (at grid resolution) for the nearest point that is free,
    keeps the standoff within ``[min_standoff_m, max_standoff_m]`` (default
    0.5x / 2.0x the ideal standoff), and still sees the target — and re-aims
    the yaw from there via :func:`compute_approach_viewpoint`. Returns ``None``
    when nothing qualifies inside ``max_radius_m``: the caller reports "no
    reachable viewpoint" rather than fabricating one (ADR-0038 posture).

    Args:
        grid: Decoded occupancy snapshot.
        viewpoint: The geometric (grid-blind) viewpoint to validate/refine.
        target_xyz: The object the camera must face, same frame as the grid.
        inflation_m: Robot-footprint radius every candidate must clear.
        max_radius_m: Search radius around the ideal viewpoint.
        min_standoff_m: Closest admissible standoff (default ``0.5 *
            viewpoint.standoff_m``).
        max_standoff_m: Farthest admissible standoff (default ``2.0 *
            viewpoint.standoff_m``).
    """
    lo = 0.5 * viewpoint.standoff_m if min_standoff_m is None else min_standoff_m
    hi = 2.0 * viewpoint.standoff_m if max_standoff_m is None else max_standoff_m
    ideal_x, ideal_y, _ideal_z = viewpoint.pose.xyz
    tx, ty = target_xyz[0], target_xyz[1]

    def _admissible(x: float, y: float) -> bool:
        standoff = math.hypot(x - tx, y - ty)
        return (
            lo <= standoff <= hi
            and grid.is_free(x, y, inflation_m=inflation_m)
            and grid.line_of_sight((x, y), (tx, ty))
        )

    if _admissible(ideal_x, ideal_y):
        return viewpoint

    target_pose = Pose6D(
        xyz=target_xyz, quat_xyzw=(0.0, 0.0, 0.0, 1.0), frame_id=viewpoint.pose.frame_id
    )
    step = grid.resolution_m
    radius = step
    while radius <= max_radius_m + 1e-9:
        # Enough samples that consecutive points on the ring are ~one cell apart.
        n_samples = max(8, math.ceil(2.0 * math.pi * radius / step))
        best: tuple[float, float, float] | None = None  # (standoff_error, x, y)
        for k in range(n_samples):
            theta = 2.0 * math.pi * k / n_samples
            x = ideal_x + radius * math.cos(theta)
            y = ideal_y + radius * math.sin(theta)
            if not _admissible(x, y):
                continue
            err = abs(math.hypot(x - tx, y - ty) - viewpoint.standoff_m)
            if best is None or err < best[0]:
                best = (err, x, y)
        if best is not None:
            _err, x, y = best
            candidate = Pose6D(
                xyz=(x, y, viewpoint.pose.xyz[2]),
                quat_xyzw=(0.0, 0.0, 0.0, 1.0),
                frame_id=viewpoint.pose.frame_id,
            )
            return compute_approach_viewpoint(
                target_pose,
                standoff_m=math.hypot(x - tx, y - ty),
                camera_frame_id=viewpoint.camera_frame_id,
                approach_from=candidate,
            )
        radius += step
    return None
