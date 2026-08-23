"""Unit checks for the nvblox depth height filter."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from openral_core import BoxShape, RobotDescription, ROSConfigError
from openral_slam_bringup.depth_height_filter_node import (
    derive_robot_relative_height_band,
    filter_depth_by_global_height,
    quaternion_to_matrix_z_row,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class _Geom:
    """One collision volume in the duck-typed shape the band derivation reads.

    ``derive_robot_relative_height_band`` takes ``description: Any`` on purpose
    (the node must not import ``openral_core`` to compute a band), so this is
    the real calling contract, not a stand-in for ``LinkCollisionGeometry``. It
    exists to place a *real* :class:`openral_core.BoxShape` at an orientation no
    shipped manifest happens to use, and to carry the one shape kind the schema
    cannot express yet.
    """

    link_name: str
    shape: Any
    origin_xyz_rpy: tuple[float, float, float, float, float, float]


@dataclass(frozen=True)
class _Description:
    """The five attributes the band derivation reads off a ``RobotDescription``."""

    base_frame: str = "base_link"
    joints: tuple[Any, ...] = ()
    collision_geometry: tuple[_Geom, ...] = field(default_factory=tuple)
    footprint_polygon: tuple[tuple[float, float], ...] = ()
    footprint_radius: float | None = None


@dataclass(frozen=True)
class _CylinderShape:
    """A fourth primitive kind — what a future ``CollisionShape`` member looks like.

    ``openral_core.CollisionShape`` is closed over sphere/capsule/box today, so
    no fixture in ``robots/`` can produce this. It is here to prove the band
    refuses an extent it cannot measure instead of inventing one.
    """

    shape: str = "cylinder"
    radius_m: float = 0.1
    length_m: float = 0.4


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
    """``panda_mobile`` bounds every arm link with a rotated ``BoxShape``.

    The upper edge is the exact OBB projection onto world z of the tallest
    link at ``q = 0``. It is pinned tightly enough to exclude both radius
    surrogates a box invites: the inscribed radius ``min(half_extents)``
    yields 1.0879 m (35 mm of real obstacle height silently dropped from the
    occupancy grid) and the circumscribed radius ``|half_extents|`` yields
    1.1663 m (43 mm of ceiling integrated as obstacle).
    """
    description = RobotDescription.from_yaml(str(_REPO_ROOT / "robots/panda_mobile/robot.yaml"))

    band = derive_robot_relative_height_band(description)

    assert band.min_z_m == pytest.approx(0.10)
    assert band.max_z_m == pytest.approx(1.1233, abs=1e-3)
    assert "footprint" in band.source
    assert "collision_geometry" in band.source


def test_derive_height_band_from_capsule_manifest() -> None:
    """``franka_panda`` bounds every link with a ``CapsuleShape``.

    Its lowest capsule reaches below ``base_frame`` z=0, so the floor edge is
    driven by the geometry rather than clamped at the origin — the case where
    an over-approximated lower extent would drag the band down through the
    real floor and re-mark the floor this node exists to remove.
    """
    description = RobotDescription.from_yaml(str(_REPO_ROOT / "robots/franka_panda/robot.yaml"))

    band = derive_robot_relative_height_band(description)

    assert band.min_z_m == pytest.approx(-0.0154, abs=1e-3)
    assert band.max_z_m == pytest.approx(1.1749, abs=1e-3)
    assert band.source == "minimum+collision_geometry"


def test_derive_height_band_from_sphere_and_capsule_manifest() -> None:
    """``h1`` mixes ``SphereShape`` and ``CapsuleShape`` in one manifest."""
    description = RobotDescription.from_yaml(str(_REPO_ROOT / "robots/h1/robot.yaml"))
    kinds = {geom.shape.shape for geom in description.collision_geometry}
    assert kinds == {"sphere", "capsule"}

    band = derive_robot_relative_height_band(description)

    assert band.min_z_m == pytest.approx(-0.9434, abs=1e-3)
    assert band.max_z_m == pytest.approx(0.4721, abs=1e-3)
    assert "collision_geometry" in band.source


def test_derive_height_band_projects_a_rotated_box_onto_world_z() -> None:
    """A box's z span is its OBB support projection, not any of its extents.

    Rolled 90 degrees about x, the box's local +y axis points along world z, so
    the span is set by ``half_extents_m[1]`` (0.20 m) — neither the local z
    half-extent (0.05 m) nor the inscribed/circumscribed radius.
    """
    description = _Description(
        collision_geometry=(
            _Geom(
                link_name="base_link",
                shape=BoxShape(half_extents_m=(0.10, 0.20, 0.05)),
                origin_xyz_rpy=(0.0, 0.0, 0.0, math.pi / 2.0, 0.0, 0.0),
            ),
        )
    )

    band = derive_robot_relative_height_band(description)

    # floor_z = min(0.0, -0.20) = -0.20; min_z = floor_z + clearance.
    assert band.min_z_m == pytest.approx(-0.10)
    assert band.max_z_m == pytest.approx(0.20)


def test_derive_height_band_rejects_a_shape_it_cannot_measure() -> None:
    """An unhandled primitive raises rather than contributing a guessed extent."""
    description = _Description(
        collision_geometry=(
            _Geom(
                link_name="base_link",
                shape=_CylinderShape(),
                origin_xyz_rpy=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            ),
        )
    )

    with pytest.raises(ROSConfigError, match="cylinder"):
        derive_robot_relative_height_band(description)


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
