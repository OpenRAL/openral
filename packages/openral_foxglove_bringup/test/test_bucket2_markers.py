"""Hermetic unit tests for the Bucket-2 pure conversion functions.

Loads ``bucket2_markers.py`` by filesystem path (no ament package
install required) mirroring the approach in ``test_foxglove_launch.py``.

Only the pure, ROS-free function is exercised here:
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
    # Register before exec_module, the standard importlib pattern.
    sys.modules["_bucket2_markers"] = mod
    spec.loader.exec_module(mod)
    return mod


_b2 = _load_bucket2()
occupied_voxel_centers = _b2.occupied_voxel_centers  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _approx(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(a - b) <= tol


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
        assert _approx(cx, 10.0 + 1.5 * 0.5)  # 10.75
        assert _approx(cy, 20.0 + 2.5 * 0.5)  # 21.25
        assert _approx(cz, 30.0 + 0.5 * 0.5)  # 30.25

    def test_the_grids_orientation_places_the_voxel(self) -> None:
        """The lattice is the OctoMap's, so its axes are not `base_frame`'s.

        Drawing a voxel without the grid's rotation puts the obstacle somewhere
        it is not on every dashboard the operator judges a stop by.
        """
        occupancy = [0, 1]  # voxel (1, 0, 0)
        quat = (0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4))  # +90° about z
        centers = occupied_voxel_centers(
            origin=(0.0, 0.0, 0.0),
            resolution=0.1,
            size=(2, 1, 1),
            occupancy=occupancy,
            orientation_xyzw=quat,
        )
        assert len(centers) == 1
        cx, cy, cz = centers[0]
        assert _approx(cx, -0.05)
        assert _approx(cy, 0.15)
        assert _approx(cz, 0.05)

    def test_an_unset_orientation_is_refused_not_read_as_identity(self) -> None:
        """All-zeros is what an unset field carries, and it is not a rotation."""
        with pytest.raises(ValueError, match="unit quaternion"):
            occupied_voxel_centers(
                origin=(0.0, 0.0, 0.0),
                resolution=0.1,
                size=(1, 1, 1),
                occupancy=[1],
                orientation_xyzw=(0.0, 0.0, 0.0, 0.0),
            )

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
