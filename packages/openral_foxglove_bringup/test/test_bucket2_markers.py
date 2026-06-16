"""Hermetic unit tests for the Bucket-2 pure conversion functions.

Loads ``bucket2_markers.py`` by filesystem path (no ament package
install required) mirroring the approach in ``test_foxglove_launch.py``.

Only the two pure, ROS-free functions are exercised here:
  - ``capsule_markers``   — WorldCollision arrays → list[MarkerSpec]
  - ``occupied_voxel_centers`` — OccupancyVoxels → list[(cx, cy, cz)]

No ROS context, no mocks, no stubs (CLAUDE.md §1.11).  All inputs are
real numeric values; all assertions are on real numeric outputs.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Load the module by path so this test runs under plain pytest without the
# ament package installed (same pattern as test_foxglove_launch.py).
# ---------------------------------------------------------------------------

_PKG_DIR = Path(__file__).resolve().parent.parent
_MODULE_PATH = _PKG_DIR / "openral_foxglove_bringup" / "bucket2_markers.py"


def _load_bucket2() -> object:
    import sys

    spec = importlib.util.spec_from_file_location("_bucket2_markers", _MODULE_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    # Register before exec_module so @dataclass can resolve cls.__module__.
    sys.modules["_bucket2_markers"] = mod
    spec.loader.exec_module(mod)
    return mod


_b2 = _load_bucket2()
capsule_markers = _b2.capsule_markers  # type: ignore[attr-defined]
occupied_voxel_centers = _b2.occupied_voxel_centers  # type: ignore[attr-defined]
MarkerSpec = _b2.MarkerSpec  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _approx(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(a - b) <= tol


# ---------------------------------------------------------------------------
# capsule_markers — WorldCollision → list[MarkerSpec]
# ---------------------------------------------------------------------------


class TestCapsuleMarkers:
    def test_two_capsules_basic_geometry(self) -> None:
        """Two real capsules: verify radius→scale, half_length→length, pose."""
        radius = [0.05, 0.10]
        half_length = [0.20, 0.00]  # second is a sphere (half_length == 0)
        # Obstacle 0: at (1, 2, 3), yaw=π/4; obstacle 1: at (0, 0, 0.5), no rotation
        origin_xyzrpy = [
            1.0, 2.0, 3.0, 0.0, 0.0, math.pi / 4,
            0.0, 0.0, 0.5, 0.0, 0.0, 0.0,
        ]
        object_id = ["box_left", "sphere_right"]

        specs = capsule_markers(radius, half_length, origin_xyzrpy, object_id)

        assert len(specs) == 2

        s0 = specs[0]
        assert s0.ns == "box_left"
        assert s0.marker_id == 0
        # scale_x/y = 2 * radius = 0.10
        assert _approx(s0.scale_x, 0.10)
        assert _approx(s0.scale_y, 0.10)
        # scale_z = 2 * half_length = 0.40
        assert _approx(s0.scale_z, 0.40)
        # Position
        assert _approx(s0.pos_x, 1.0)
        assert _approx(s0.pos_y, 2.0)
        assert _approx(s0.pos_z, 3.0)
        # Quaternion for yaw = π/4 (ZYX, roll=pitch=0):
        # qw = cos(π/8), qz = sin(π/8), qx = qy = 0
        assert _approx(s0.q_x, 0.0, tol=1e-9)
        assert _approx(s0.q_y, 0.0, tol=1e-9)
        assert _approx(s0.q_z, math.sin(math.pi / 8), tol=1e-9)
        assert _approx(s0.q_w, math.cos(math.pi / 8), tol=1e-9)
        # Quaternion must be unit
        norm = math.sqrt(s0.q_x**2 + s0.q_y**2 + s0.q_z**2 + s0.q_w**2)
        assert _approx(norm, 1.0, tol=1e-9)

        s1 = specs[1]
        assert s1.ns == "sphere_right"
        assert s1.marker_id == 1
        # Sphere: radius 0.10, half_length 0 → scale_z = 0
        assert _approx(s1.scale_x, 0.20)
        assert _approx(s1.scale_y, 0.20)
        assert _approx(s1.scale_z, 0.0)
        # Identity quaternion
        assert _approx(s1.q_x, 0.0, tol=1e-9)
        assert _approx(s1.q_y, 0.0, tol=1e-9)
        assert _approx(s1.q_z, 0.0, tol=1e-9)
        assert _approx(s1.q_w, 1.0, tol=1e-9)

    def test_ns_falls_back_to_index_when_empty_object_id(self) -> None:
        """Empty object_id string → ns becomes 'obstacle_<i>'."""
        specs = capsule_markers(
            radius=[0.1],
            half_length=[0.0],
            origin_xyzrpy=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            object_id=[""],
        )
        assert specs[0].ns == "obstacle_0"

    def test_empty_inputs_return_empty_list(self) -> None:
        """No obstacles → empty list, no crash."""
        specs = capsule_markers(
            radius=[], half_length=[], origin_xyzrpy=[], object_id=[]
        )
        assert specs == []

    def test_mismatched_lengths_raise_value_error(self) -> None:
        with pytest.raises(ValueError, match="radius and half_length"):
            capsule_markers(
                radius=[0.1, 0.2],
                half_length=[0.1],
                origin_xyzrpy=[0.0] * 12,
                object_id=["a", "b"],
            )

    def test_bad_origin_length_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="origin_xyzrpy"):
            capsule_markers(
                radius=[0.1],
                half_length=[0.1],
                origin_xyzrpy=[0.0] * 5,  # needs 6
                object_id=["a"],
            )

    def test_marker_type_is_cylinder(self) -> None:
        """CYLINDER = 3 (visualization_msgs/Marker constant)."""
        specs = capsule_markers(
            radius=[0.05],
            half_length=[0.1],
            origin_xyzrpy=[0.0] * 6,
            object_id=["obj"],
        )
        assert specs[0].marker_type == 3  # CYLINDER


# ---------------------------------------------------------------------------
# occupied_voxel_centers — OccupancyVoxels → list[(cx, cy, cz)]
# ---------------------------------------------------------------------------


class TestOccupiedVoxelCenters:
    def test_2x2x2_grid_two_occupied(self) -> None:
        """2×2×2 grid with voxels (0,0,0) and (1,1,1) occupied.

        Index formula: idx = x + size_x*(y + size_y*z)
          (0,0,0) → 0 + 2*(0 + 2*0) = 0
          (1,1,1) → 1 + 2*(1 + 2*1) = 7
        Centers at resolution=0.1, origin=(0,0,0):
          (0,0,0) → (0.05, 0.05, 0.05)
          (1,1,1) → (0.15, 0.15, 0.15)
        """
        size_x, size_y, size_z = 2, 2, 2
        occupancy = [0] * 8
        occupancy[0] = 1  # voxel (0,0,0)
        occupancy[7] = 1  # voxel (1,1,1)

        centers = occupied_voxel_centers(
            origin=(0.0, 0.0, 0.0),
            resolution=0.1,
            size=(size_x, size_y, size_z),
            occupancy=occupancy,
        )

        assert len(centers) == 2
        c_set = {(round(cx, 9), round(cy, 9), round(cz, 9)) for cx, cy, cz in centers}
        assert (0.05, 0.05, 0.05) in c_set
        assert (0.15, 0.15, 0.15) in c_set

    def test_center_offset_is_half_resolution(self) -> None:
        """Single voxel (0,0,0) with non-unit resolution and non-zero origin."""
        occupancy = [0] * (3 * 3 * 3)
        # Voxel (1, 2, 0): idx = 1 + 3*(2 + 3*0) = 1 + 6 = 7
        occupancy[7] = 255

        centers = occupied_voxel_centers(
            origin=(10.0, 20.0, 30.0),
            resolution=0.5,
            size=(3, 3, 3),
            occupancy=occupancy,
        )

        assert len(centers) == 1
        cx, cy, cz = centers[0]
        assert _approx(cx, 10.0 + 1.5 * 0.5)   # 10.75
        assert _approx(cy, 20.0 + 2.5 * 0.5)   # 21.25
        assert _approx(cz, 30.0 + 0.5 * 0.5)   # 30.25

    def test_all_free_returns_empty_list(self) -> None:
        occupancy = [0] * (2 * 2 * 2)
        centers = occupied_voxel_centers(
            origin=(0.0, 0.0, 0.0), resolution=0.05, size=(2, 2, 2), occupancy=occupancy
        )
        assert centers == []

    def test_empty_grid_returns_empty_list(self) -> None:
        """Zero-size grid (no voxels) → empty list, no crash."""
        centers = occupied_voxel_centers(
            origin=(0.0, 0.0, 0.0), resolution=0.1, size=(0, 0, 0), occupancy=[]
        )
        assert centers == []

    def test_wrong_occupancy_length_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="occupancy length"):
            occupied_voxel_centers(
                origin=(0.0, 0.0, 0.0),
                resolution=0.1,
                size=(2, 2, 2),
                occupancy=[0] * 5,  # needs 8
            )

    def test_row_major_x_fastest_ordering(self) -> None:
        """Verify x-fastest indexing: (1,0,0)=idx1, (0,1,0)=idx2."""
        # 3×2×1 grid: size_x=3, size_y=2, size_z=1
        # (1,0,0) → 1 + 3*(0 + 2*0) = 1
        # (0,1,0) → 0 + 3*(1 + 2*0) = 3
        occupancy = [0] * 6
        occupancy[1] = 1  # voxel (1,0,0)
        occupancy[3] = 1  # voxel (0,1,0)

        centers = occupied_voxel_centers(
            origin=(0.0, 0.0, 0.0), resolution=1.0, size=(3, 2, 1), occupancy=occupancy
        )

        assert len(centers) == 2
        c_set = {(round(cx, 9), round(cy, 9), round(cz, 9)) for cx, cy, cz in centers}
        # (1,0,0) → center (1.5, 0.5, 0.5)
        assert (1.5, 0.5, 0.5) in c_set
        # (0,1,0) → center (0.5, 1.5, 0.5)
        assert (0.5, 1.5, 0.5) in c_set
