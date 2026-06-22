"""Unit checks for the nvblox depth height filter."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from openral_core import RobotDescription
from openral_slam_bringup.depth_height_filter_node import (
    derive_robot_relative_height_band,
    filter_depth_by_global_height,
    quaternion_to_matrix_z_row,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]


def test_filter_depth_by_global_height_keeps_only_band() -> None:
    depth = np.array([[0.5, 1.0, 1.5]], dtype=np.float32)

    filtered = filter_depth_by_global_height(
        depth,
        fx=1.0,
        fy=1.0,
        cx=0.0,
        cy=0.0,
        rotation_z_row=(0.0, 0.0, 1.0),
        translation_z_m=0.0,
        min_height_m=0.8,
        max_height_m=1.3,
    )

    np.testing.assert_array_equal(filtered, np.array([[0.0, 1.0, 0.0]], dtype=np.float32))


def test_filter_depth_by_global_height_uses_camera_pose() -> None:
    depth = np.array([[0.2, 0.4]], dtype=np.float32)

    filtered = filter_depth_by_global_height(
        depth,
        fx=1.0,
        fy=1.0,
        cx=0.0,
        cy=0.0,
        rotation_z_row=(0.0, 0.0, 1.0),
        translation_z_m=0.7,
        min_height_m=0.8,
        max_height_m=1.3,
    )

    np.testing.assert_array_equal(filtered, np.array([[0.2, 0.4]], dtype=np.float32))


def test_derive_height_band_from_real_robot_manifest_geometry() -> None:
    description = RobotDescription.from_yaml(str(_REPO_ROOT / "robots/panda_mobile/robot.yaml"))

    band = derive_robot_relative_height_band(description)

    assert band.min_z_m == pytest.approx(0.10)
    assert band.max_z_m > 1.0
    assert "footprint" in band.source
    assert "collision_geometry" in band.source


def test_quaternion_to_matrix_z_row_identity() -> None:
    assert quaternion_to_matrix_z_row(0.0, 0.0, 0.0, 1.0) == pytest.approx((0.0, 0.0, 1.0))


def test_filter_depth_by_global_height_rejects_invalid_intrinsics() -> None:
    with pytest.raises(ValueError, match="invalid pinhole intrinsics"):
        filter_depth_by_global_height(
            np.ones((1, 1), dtype=np.float32),
            fx=0.0,
            fy=1.0,
            cx=0.0,
            cy=0.0,
            rotation_z_row=(0.0, 0.0, 1.0),
            translation_z_m=0.0,
            min_height_m=0.8,
            max_height_m=1.3,
        )
