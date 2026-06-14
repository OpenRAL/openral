"""ADR-0044 Phase 2 — occupancy-grid queries + approach-pose refinement.

Scenario grids encode the spatial situations the refinement exists for — an
ideal standoff inside a kitchen counter, a wall breaking line-of-sight, a
viewpoint walled into a dead end — at a realistic slam resolution (0.1 m).
(Precedent: the ADR-0030 Phase 6 kernel nav-goal sim test also drives a
synthetic grid; a slam-captured fixture upgrade rides with ADR-0044 Phase 4.)
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest
from openral_core import Pose6D
from openral_world_state.grid import FREE_MAX, OccupancyGridIndex, refine_approach_pose
from openral_world_state.spatial_memory import compute_approach_viewpoint

RES = 0.1  # metres per cell — typical slam_toolbox resolution


def _room_with_counter() -> OccupancyGridIndex:
    """4 m x 4 m room: bordering walls + a 1.2 m x 0.6 m counter mid-room.

    The counter spans x in [1.4, 2.6), y in [1.6, 2.2) world metres.
    """
    grid = np.zeros((40, 40), dtype=np.int8)
    grid[0, :] = 100
    grid[-1, :] = 100
    grid[:, 0] = 100
    grid[:, -1] = 100
    grid[16:22, 14:26] = 100  # counter: rows = y cells, cols = x cells
    return OccupancyGridIndex(grid, resolution_m=RES, origin_xy=(0.0, 0.0))


def _yaw_of(pose: Pose6D) -> float:
    x, y, z, w = pose.quat_xyzw
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


# --------------------------------------------------------------------------
# OccupancyGridIndex primitives.
# --------------------------------------------------------------------------


def test_world_to_cell_and_off_grid() -> None:
    idx = _room_with_counter()
    assert idx.world_to_cell(0.05, 0.05) == (0, 0)
    assert idx.world_to_cell(3.95, 0.05) == (0, 39)
    assert idx.world_to_cell(-0.1, 1.0) is None
    assert idx.world_to_cell(1.0, 4.1) is None
    assert not idx.is_free(-0.1, 1.0), "off-grid is blocked"


def test_is_free_semantics_and_inflation() -> None:
    idx = _room_with_counter()
    assert idx.is_free(1.0, 1.0)
    assert not idx.is_free(2.0, 1.9), "counter cell is occupied"
    # 10 cm from the counter face: free bare, blocked under a 25 cm footprint.
    assert idx.is_free(2.0, 1.45)
    assert not idx.is_free(2.0, 1.45, inflation_m=0.25)
    # Unknown blocks like occupied.
    data = np.zeros((10, 10), dtype=np.int8)
    data[5, 5] = -1
    unknown = OccupancyGridIndex(data, resolution_m=RES, origin_xy=(0.0, 0.0))
    assert not unknown.is_free(0.55, 0.55)
    # Boundary value: exactly FREE_MAX is free, one above is not.
    data2 = np.full((4, 4), FREE_MAX, dtype=np.int8)
    assert OccupancyGridIndex(data2, resolution_m=RES, origin_xy=(0.0, 0.0)).is_free(0.2, 0.2)
    data2[:] = FREE_MAX + 1
    assert not OccupancyGridIndex(data2, resolution_m=RES, origin_xy=(0.0, 0.0)).is_free(0.2, 0.2)


def test_line_of_sight_blocked_by_counter() -> None:
    idx = _room_with_counter()
    # Across open floor: clear.
    assert idx.line_of_sight((0.5, 0.5), (3.5, 0.5))
    # Through the counter: blocked.
    assert not idx.line_of_sight((2.0, 1.0), (2.0, 3.0))
    # Around the counter (same endpoints' sides, path not crossing it): clear.
    assert idx.line_of_sight((0.5, 1.0), (0.5, 3.0))
    # Off-grid endpoint: no sight.
    assert not idx.line_of_sight((2.0, 1.0), (2.0, 4.5))


def test_from_msg_duck_typed_with_origin_offset() -> None:
    msg = SimpleNamespace(
        info=SimpleNamespace(
            resolution=0.5,
            width=4,
            height=2,
            origin=SimpleNamespace(
                position=SimpleNamespace(x=-1.0, y=-0.5, z=0.0),
                orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
            ),
        ),
        data=[0, 0, 100, 0, 0, 0, 0, 0],
    )
    idx = OccupancyGridIndex.from_msg(msg)
    # Cell (0, 2) is occupied: world x in [0.0, 0.5), y in [-0.5, 0.0).
    assert idx.world_to_cell(0.2, -0.2) == (0, 2)
    assert not idx.is_free(0.2, -0.2)
    assert idx.is_free(-0.8, -0.2)


def test_validation_rejects_bad_shapes() -> None:
    with pytest.raises(ValueError, match="2-D"):
        OccupancyGridIndex(np.zeros(8, dtype=np.int8), resolution_m=0.1, origin_xy=(0, 0))
    with pytest.raises(ValueError, match="resolution"):
        OccupancyGridIndex(np.zeros((2, 2), dtype=np.int8), resolution_m=0.0, origin_xy=(0, 0))


# --------------------------------------------------------------------------
# refine_approach_pose.
# --------------------------------------------------------------------------


def _target_on_counter() -> Pose6D:
    # A mug on the counter's south (front) edge cell: its own cell is occupied
    # (it shares it with the counter), so only the endpoint-exempt sight rule
    # makes it observable at all — exactly the live situation.
    return Pose6D(xyz=(2.0, 1.65, 0.9), quat_xyzw=(0.0, 0.0, 0.0, 1.0), frame_id="map")


def test_valid_ideal_viewpoint_returned_unchanged() -> None:
    idx = _room_with_counter()
    target = _target_on_counter()
    # Approach from open floor south of the counter: ideal already free + LoS.
    vp = compute_approach_viewpoint(
        target,
        standoff_m=0.6,
        camera_frame_id="wrist",
        approach_from=Pose6D(xyz=(2.0, 0.5, 0.0), quat_xyzw=(0, 0, 0, 1), frame_id="map"),
    )
    assert refine_approach_pose(idx, vp, target.xyz) is vp


def test_blocked_ideal_snaps_to_free_cell_with_sight() -> None:
    idx = _room_with_counter()
    target = _target_on_counter()
    # A tight 0.25 m standoff puts the ideal viewpoint right at the counter
    # face: the bare cell is free but the robot footprint (inflation) overlaps
    # the counter — the realistic "ideal goal is too close to furniture" case.
    vp = compute_approach_viewpoint(
        target,
        standoff_m=0.25,
        camera_frame_id="wrist",
        approach_from=Pose6D(xyz=(2.0, 0.5, 0.0), quat_xyzw=(0, 0, 0, 1), frame_id="map"),
    )
    assert idx.is_free(*vp.pose.xyz[:2]), "bare ideal cell is free"
    assert not idx.is_free(*vp.pose.xyz[:2], inflation_m=0.25), "but the footprint overlaps"
    refined = refine_approach_pose(idx, vp, target.xyz, inflation_m=0.25)
    assert refined is not None
    assert refined is not vp
    rx, ry, _rz = refined.pose.xyz
    assert idx.is_free(rx, ry, inflation_m=0.25)
    assert idx.line_of_sight((rx, ry), (target.xyz[0], target.xyz[1]))
    standoff = math.hypot(rx - target.xyz[0], ry - target.xyz[1])
    assert 0.125 <= standoff <= 0.5, "standoff stays within [0.5x, 2.0x] of ideal"
    assert refined.standoff_m == pytest.approx(standoff)
    # Re-aimed: the +X (approach convention) yaw points from the snap at the target.
    expected_yaw = math.atan2(target.xyz[1] - ry, target.xyz[0] - rx)
    assert _yaw_of(refined.pose) == pytest.approx(expected_yaw, abs=1e-6)
    assert refined.camera_frame_id == "wrist"


def test_candidate_without_line_of_sight_is_rejected() -> None:
    # A wall splits the room; the target sits across it. Cells behind the wall
    # are free but blind — refinement must walk around to the doorway side.
    grid = np.zeros((40, 40), dtype=np.int8)
    grid[0, :] = 100
    grid[-1, :] = 100
    grid[:, 0] = 100
    grid[:, -1] = 100
    grid[20, 1:30] = 100  # wall at y = 2.0..2.1 with a doorway at x >= 3.0
    idx = OccupancyGridIndex(grid, resolution_m=RES, origin_xy=(0.0, 0.0))
    target = Pose6D(xyz=(2.0, 2.8, 0.8), quat_xyzw=(0, 0, 0, 1), frame_id="map")
    # Ideal viewpoint south of the wall: free floor, but the wall blocks sight.
    vp = compute_approach_viewpoint(
        target,
        standoff_m=1.0,
        camera_frame_id="wrist",
        approach_from=Pose6D(xyz=(2.0, 1.0, 0.0), quat_xyzw=(0, 0, 0, 1), frame_id="map"),
    )
    assert idx.is_free(*vp.pose.xyz[:2]), "ideal cell itself is free"
    refined = refine_approach_pose(
        idx, vp, target.xyz, inflation_m=0.1, max_radius_m=2.5, max_standoff_m=2.5
    )
    assert refined is not None
    rx, ry, _rz = refined.pose.xyz
    assert idx.line_of_sight((rx, ry), (2.0, 2.8)), "snapped pose must actually see the target"


def test_no_reachable_viewpoint_returns_none() -> None:
    # The target sits deep inside a solid block: every free cell near it is
    # farther than the admissible standoff. Honest None, never a fabricated pose.
    grid = np.zeros((40, 40), dtype=np.int8)
    grid[14:27, 14:27] = 100  # solid 1.3 m block; target at its centre
    idx = OccupancyGridIndex(grid, resolution_m=RES, origin_xy=(0.0, 0.0))
    target = Pose6D(xyz=(2.0, 2.0, 0.5), quat_xyzw=(0, 0, 0, 1), frame_id="map")
    vp = compute_approach_viewpoint(target, standoff_m=0.3, camera_frame_id="wrist")
    # Free space starts >= 0.6 m from the target (block face) + 0.1 m inflation,
    # beyond the 2.0x standoff ceiling of 0.6 m.
    assert refine_approach_pose(idx, vp, target.xyz, inflation_m=0.1, max_radius_m=1.0) is None
