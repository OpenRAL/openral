"""Bounded PCL pose tracking and object-local surface accumulation for shadow evaluation.

Only observed surface voxels are represented. An absent voxel is unknown. These
models never replace containing fits, collision geometry or world-map occupancy.
"""

from __future__ import annotations

import importlib
import itertools
from dataclasses import dataclass, field
from typing import Protocol, cast

import numpy as np
from numpy.typing import NDArray
from openral_core import IntrinsicsPinhole, ObjectSurfaceModelConfig, TrackedObjectSet
from openral_core.exceptions import ROSConfigError

__all__ = ["ObjectSurfaceModels"]

_MIN_POINTS = 4
_MAX_COORDINATE_M = 10_000.0  # Matches the native registration input cap.


class _Registration(Protocol):
    def align(
        self,
        source: NDArray[np.float64],
        target: NDArray[np.float64],
        initial: NDArray[np.float64],
        correspondence_m: float,
        iterations: int,
    ) -> tuple[list[list[float]], bool, float, float]: ...


@dataclass
class _Surface:
    epoch: int
    pose: NDArray[np.float64]
    stamp_ns: int
    cells: dict[tuple[int, int, int], tuple[float, float, float]] = field(default_factory=dict)
    reason: str = "seeded"
    rmse_m: float = 0.0
    overlap: float = 1.0
    cleared_cells: int = 0
    free_stamp_ns: int = 0
    free_cursor: int = 0


class ObjectSurfaceModels:
    """Observation-only models keyed by track identity and lifecycle generation.

    Requires the ROS package's native PCL module. Configuration must explicitly
    opt in; missing native support then raises ``ROSConfigError``. Registration
    residual/overlap gates are diagnostic acceptance criteria, not calibrated
    containment bounds. Symmetry and unseen geometry remain unresolved.
    """

    def __init__(self, config: ObjectSurfaceModelConfig) -> None:
        """Load native PCL support and start with no observed objects."""
        self.config = config
        try:
            self._registration = cast(
                _Registration, importlib.import_module("_openral_pcl_registration")
            )
        except ImportError as exc:
            raise ROSConfigError(
                "object_surface_model requires a built openral_world_state PCL module; "
                "build and source the ROS workspace"
            ) from exc
        self._models: dict[str, _Surface] = {}

    def retain(self, identities: set[str]) -> None:
        """Drop models whose track no longer exists; cap matches the tracker table."""
        self._models = {key: value for key, value in self._models.items() if key in identities}

    def observe(
        self,
        object_id: str,
        epoch: int,
        points_world: NDArray[np.float64],
        initial_world_pose: NDArray[np.float64],
        stamp_ns: int,
    ) -> str:
        """Register a newer surface and fuse accepted points in the object's frame.

        The initial frame is a gauge taken from the first sensor fit, not a CAD
        alignment. Subsequent poses come from PCL registration seeded by the
        previous observed pose. Rejected/stale samples change neither pose nor
        geometry. If a same-epoch target has fewer than four voxels, its earlier
        diagnostic surface is discarded and a new model is seeded from this fit's
        pose; this does not preserve every partial observation. Full models stop
        adding cells and report capacity explicitly. These models never authorize
        collision geometry or world-map clearing.
        """
        p = np.asarray(points_world, dtype=np.float64)
        pose = np.asarray(initial_world_pose, dtype=np.float64)
        if (
            p.ndim != 2  # noqa: PLR2004  # reason: XYZ points form a matrix
            or p.shape[1:] != (3,)
            or len(p) < _MIN_POINTS
            or not np.isfinite(p).all()
            or np.any(np.abs(p) > _MAX_COORDINATE_M)
            or pose.shape != (4, 4)
            or not np.isfinite(pose).all()
            or epoch < 0
            or stamp_ns < 0
        ):
            raise ROSConfigError("surface model requires finite XYZ points, pose and timestamps")
        if (
            not np.allclose(pose[3], (0, 0, 0, 1))
            or not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(pose[:3, :3]), 1.0)
        ):
            raise ROSConfigError("surface model initial pose must be rigid")
        p = self._sample(p)
        rmse, overlap = 0.0, 1.0
        old = self._models.get(object_id)
        if old is not None and (epoch < old.epoch or stamp_ns <= old.stamp_ns):
            return "stale"
        reseeded = False
        if old is None or epoch != old.epoch:
            if old is None and len(self._models) >= TrackedObjectSet.MAX_OBJECTS:
                return "object_capacity"
            state = _Surface(epoch, pose.copy(), stamp_ns)
        else:
            state = old
            target = np.asarray(list(state.cells.values()), dtype=np.float64)
            if len(target) < _MIN_POINTS:
                # A fit pose is a real tracker observation. Start a fresh gauge rather
                # than merging unregistered points or keeping an unusable ICP target.
                state = _Surface(epoch, pose.copy(), stamp_ns)
                reseeded = True
            else:
                rows, converged, rmse, overlap = self._registration.align(
                    p,
                    self._sample(target),
                    np.asarray(np.linalg.inv(state.pose), dtype=np.float64),
                    self.config.max_correspondence_m,
                    self.config.max_iterations,
                )
                transform = np.asarray(rows, dtype=np.float64)
                if (
                    not converged
                    or not np.isfinite(transform).all()
                    or rmse > self.config.max_rmse_m
                    or not np.isfinite(rmse)
                    or not np.isfinite(overlap)
                    or overlap < self.config.min_overlap
                ):
                    state.reason = "registration_rejected"
                    return state.reason
                pose = np.asarray(np.linalg.inv(transform), dtype=np.float64)
        local = (p - pose[:3, 3]) @ pose[:3, :3]
        with np.errstate(over="ignore", invalid="ignore"):
            scaled = np.floor(local / self.config.voxel_m)
        if not np.isfinite(scaled).all() or np.any(np.abs(scaled) >= 2**62):
            raise ROSConfigError("surface model voxel indices exceed the supported range")
        keys = scaled.astype(np.int64)
        state.pose, state.stamp_ns = pose, stamp_ns
        state.rmse_m, state.overlap = float(rmse), float(overlap)
        capped = False
        for key, point in zip(keys, local, strict=True):
            cell = (int(key[0]), int(key[1]), int(key[2]))
            if cell not in state.cells and len(state.cells) >= self.config.max_voxels:
                capped = True
                continue
            state.cells[cell] = (float(point[0]), float(point[1]), float(point[2]))
        state.reason = "model_capacity" if capped else "model_reseeded" if reseeded else "accepted"
        self._models[object_id] = state
        return state.reason

    def world_points(self, object_id: str) -> NDArray[np.float64]:
        """Project observed local surfaces at the latest accepted pose, with no old trail."""
        state = self._models[object_id]
        local = np.asarray(list(state.cells.values()), dtype=np.float64).reshape(-1, 3)
        return np.asarray(local @ state.pose[:3, :3].T + state.pose[:3, 3])

    def world_contributions(self) -> dict[str, NDArray[np.float64]]:
        """Snapshot independently owned surfaces; overlap never transfers ownership.

        Replacing or retiring one identity changes only its contribution. These
        points are diagnostic occupied surfaces, never instructions to erase
        cells from OctoMap, static geometry or another identity.
        """
        return {key: self.world_points(key).copy() for key in self._models}

    def clear_observed_free(
        self,
        object_id: str,
        epoch: int,
        stamp_ns: int,
        depth_m: NDArray[np.float64],
        robot_mask: NDArray[np.uint8] | None,
        intrinsics: IntrinsicsPinhole,
        t_world_from_cam: NDArray[np.float64],
        *,
        now_ns: int,
        max_age_ns: int,
        depth_error_m: float,
        pose_error_m: float,
        pixel_error: int,
        min_depth_m: float,
        max_depth_m: float,
    ) -> int:
        """Remove only owned local cells independently seen free at the accepted pose.

        Every pixel covering the entire voxel, enlarged by the supplied pose
        and pixel error bounds, must have a valid unmasked return beyond its
        farthest depth plus depth error. Missing, occluded, stale, future and
        off-image evidence retains the cell. Bounds are explicit evaluation
        inputs; this diagnostic method does not establish their calibration.
        Projected pixel work is capped by ``config.max_points`` per invocation.
        """
        if (
            max_age_ns <= 0
            or now_ns < 0
            or not isinstance(pixel_error, int)
            or pixel_error < 0
            or not np.isfinite((depth_error_m, pose_error_m)).all()
            or min(depth_error_m, pose_error_m) < 0
            or not np.isfinite((min_depth_m, max_depth_m)).all()
            or min_depth_m < 0
            or max_depth_m <= min_depth_m
        ):
            raise ROSConfigError("surface clearing requires finite nonnegative error bounds")
        state = self._models.get(object_id)
        if (
            state is None
            or epoch != state.epoch
            or stamp_ns != state.stamp_ns
            or stamp_ns <= state.free_stamp_ns
            or stamp_ns <= 0
            or stamp_ns > now_ns
            or now_ns - stamp_ns > max_age_ns
            or robot_mask is None
        ):
            return 0
        shape = (intrinsics.height, intrinsics.width)
        depth = np.asarray(depth_m, dtype=np.float64)
        mask = np.asarray(robot_mask)
        tf = np.asarray(t_world_from_cam, dtype=np.float64)
        if (
            depth.shape != shape
            or mask.shape != shape
            or tf.shape != (4, 4)
            or not np.isfinite(tf).all()
            or np.any(np.abs(tf[:3, 3]) > _MAX_COORDINATE_M)
            or not np.allclose(tf[3], (0, 0, 0, 1))
            or not np.allclose(tf[:3, :3].T @ tf[:3, :3], np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(tf[:3, :3]), 1.0)
        ):
            raise ROSConfigError(
                "surface clearing requires matching depth/mask and rigid camera TF"
            )
        if not state.cells:
            return 0
        state.free_stamp_ns = stamp_ns
        cam_from_local = np.linalg.inv(tf) @ state.pose
        # Enclose the rotated local voxel in a camera-axis box, then grow by pose error.
        half = np.abs(cam_from_local[:3, :3]) @ np.full(3, self.config.voxel_m / 2)
        half += pose_error_m
        signs = np.asarray(list(itertools.product((-1, 1), repeat=3)), dtype=np.float64)
        budget = self.config.max_points
        removed = []
        keys = list(state.cells)
        # Rotate the budget across frames so early cells cannot starve later contributions.
        start = state.free_cursor % len(keys)
        for offset in range(len(keys)):
            index = (start + offset) % len(keys)
            key = keys[index]
            centre = (np.asarray(key, dtype=np.float64) + 0.5) * self.config.voxel_m
            cam_centre = cam_from_local[:3, :3] @ centre + cam_from_local[:3, 3]
            corners = cam_centre + signs * half
            if np.min(corners[:, 2]) <= 0:
                continue
            u = intrinsics.fx * corners[:, 0] / corners[:, 2] + intrinsics.cx
            v = intrinsics.fy * corners[:, 1] / corners[:, 2] + intrinsics.cy
            x0, x1 = int(np.floor(u.min())) - pixel_error, int(np.ceil(u.max())) + pixel_error
            y0, y1 = int(np.floor(v.min())) - pixel_error, int(np.ceil(v.max())) + pixel_error
            if x0 < 0 or y0 < 0 or x1 >= intrinsics.width or y1 >= intrinsics.height:
                continue
            cost = (x1 - x0 + 1) * (y1 - y0 + 1)
            if cost > budget:
                continue  # Unexamined space remains unknown; capacity never licenses clearing.
            budget -= cost
            state.free_cursor = (index + 1) % len(keys)
            measured = depth[y0 : y1 + 1, x0 : x1 + 1]
            if (
                np.all(mask[y0 : y1 + 1, x0 : x1 + 1] == 0)
                and np.isfinite(measured).all()
                and np.all((measured > min_depth_m) & (measured < max_depth_m))
                and np.all(measured > corners[:, 2].max() + depth_error_m)
            ):
                removed.append(key)
        for key in removed:
            del state.cells[key]
        state.cleared_cells += len(removed)
        return len(removed)

    def diagnostics(self) -> dict[str, str]:
        """Bounded model size, source age inputs and registration outcomes per identity."""
        return {
            key: (
                f"{s.reason};voxels={len(s.cells)};stamp_ns={s.stamp_ns};"
                f"rmse_m={s.rmse_m:.6g};overlap={s.overlap:.6g};"
                f"epoch={s.epoch};cleared_cells={s.cleared_cells}"
            )
            for key, s in self._models.items()
        }

    def _sample(self, points: NDArray[np.float64]) -> NDArray[np.float64]:
        # Deterministic bounded input; dropping samples is allowed only for this shadow model.
        count = min(len(points), self.config.max_points)
        indices = np.linspace(0, len(points) - 1, count, dtype=np.intp)
        return np.ascontiguousarray(points[indices], dtype=np.float64)
