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
    _link_transforms_at_zero,
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
    """``rizon4`` bounds every link with a ``CapsuleShape``.

    Its lowest capsule reaches below ``base_frame`` z=0, so the floor edge is
    driven by the geometry rather than clamped at the origin — the case where
    an over-approximated lower extent would drag the band down through the
    real floor and re-mark the floor this node exists to remove.
    """
    description = RobotDescription.from_yaml(str(_REPO_ROOT / "robots/rizon4/robot.yaml"))
    assert {geom.shape.shape for geom in description.collision_geometry} == {"capsule"}

    band = derive_robot_relative_height_band(description)

    assert band.min_z_m == pytest.approx(0.0090, abs=1e-3)
    assert band.max_z_m == pytest.approx(1.3930, abs=1e-3)
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


@pytest.mark.parametrize(
    ("robot_id", "expected_max_z_m"),
    [("ur5e", 0.3354), ("ur10e", 0.4204)],
)
def test_derive_height_band_bridges_the_declared_urdf_root(
    robot_id: str, expected_max_z_m: float
) -> None:
    """UR manifests place their arm through ``assets.urdf.root_frame``.

    ``joints`` enumerates only movable joints, so UR's upstream URDF root
    (``base_link``) is not a child of any of them — every collision volume
    hangs off a link the ``base_frame`` (``ur5e_base_link``) chain alone cannot
    reach. The manifest is not missing the transform: it declares it as
    ``assets.urdf.root_frame`` + ``base_to_root_xyz_rpy``, the same pair
    ``sim_e2e.launch.py`` publishes as a static TF. Reading it is what makes
    the band cover the arm instead of collapsing onto the
    ``min_body_height_m`` floor.
    """
    description = RobotDescription.from_yaml(str(_REPO_ROOT / f"robots/{robot_id}/robot.yaml"))
    assert description.assets is not None
    assert description.assets.urdf is not None
    assert description.assets.urdf.root_frame == "base_link"
    assert description.base_frame != "base_link"

    band = derive_robot_relative_height_band(description)

    assert "collision_geometry" in band.source
    assert band.max_z_m == pytest.approx(expected_max_z_m, abs=1e-3)
    # The defect this pins: with the bridge unread every geom was skipped and
    # the band was exactly the 0.30 m floor, hiding the arm from the map.
    assert band.max_z_m > 0.30


def test_derive_height_band_refuses_an_unplaceable_collision_volume() -> None:
    """Declared geometry that cannot be placed refuses; it is never skipped.

    Partial and total placement failure get the same answer on purpose: a band
    covering an arbitrary reachable subset of the robot is not a measurement of
    the robot, and the partial case is the more dangerous one — it keeps
    reporting ``collision_geometry`` as its source, so nothing downstream can
    tell a measured band from a truncated one.

    ``franka_panda``, ``g1`` and ``openarm`` used to reach this refusal, because
    their manifests could not place a rigid mount. That was a defect in the
    manifests, not in the robots, and it is fixed — they now measure real bands
    (see the test below). So the case is reproduced the only honest way left:
    a real manifest with its declared mount taken away.
    """
    description = RobotDescription.from_yaml(str(_REPO_ROOT / "robots/franka_panda/robot.yaml"))
    assert description.collision_geometry
    assert description.fixed_attachments, "fixture precondition: the manifest declares a mount"
    orphaned = description.model_copy(update={"fixed_attachments": []})

    with pytest.raises(ROSConfigError) as excinfo:
        derive_robot_relative_height_band(orphaned)

    message = str(excinfo.value)
    assert "panda_hand" in message
    assert orphaned.base_frame in message


@pytest.mark.parametrize(
    ("robot_id", "recovered_link", "expected_max_z_m"),
    [
        # 1 of 9 volumes was unplaceable: the gripper at the top of the chain.
        ("franka_panda", "panda_hand", 1.175),
        # 15 of 27 — every arm link plus the torso. The band was [-0.72, -0.02] m,
        # a humanoid whose entire upper body sat outside its own "measured"
        # navigation height; it now reaches above the pelvis. The top is set by
        # the torso volume, so it is stated in the URDF's waist convention
        # (torso_link at z=0.054) — the model this manifest is lowered from.
        ("g1", "torso_link", 0.518),
        # 16 of 16 — total. Both arms hang off ``openarm_*_link0``.
        ("openarm", "openarm_left_link1", 0.150),
    ],
)
def test_derive_height_band_places_rigidly_mounted_links(
    robot_id: str, recovered_link: str, expected_max_z_m: float
) -> None:
    """Rigid mounts come from ``fixed_attachments``, so these robots measure.

    ``joints`` enumerates only movable joints, so a bolted-on hand or a
    bimanual rig's arm pedestals are not in it — which is why these three
    refused. The manifests are not missing the transforms: they declare them as
    ``fixed_attachments``, the same list the safety kernel builds its collision
    tree from. Reading the same source here is what keeps the height band
    measuring the robot the kernel actually checks.
    """
    description = RobotDescription.from_yaml(str(_REPO_ROOT / f"robots/{robot_id}/robot.yaml"))
    assert description.collision_geometry
    assert any(g.link_name == recovered_link for g in description.collision_geometry)

    band = derive_robot_relative_height_band(description)

    assert "collision_geometry" in band.source
    assert band.max_z_m == pytest.approx(expected_max_z_m, abs=1e-3)
    # Every declared volume was placed — a partial walk would have refused.
    placed = _link_transforms_at_zero(description)
    for geom in description.collision_geometry:
        assert str(geom.link_name) in placed


def test_derive_height_band_keeps_the_floor_for_a_manifest_without_collision_geometry() -> None:
    """No declared geometry is not a placement failure — it is the floor's job.

    ``min_body_height_m`` exists for exactly this manifest shape, so refusing
    here would break a robot that is not broken. The refusal above must stay
    narrower than "the extent is unknown": it fires only when the manifest
    declares volumes the derivation then cannot place.
    """
    description = RobotDescription.from_yaml(str(_REPO_ROOT / "robots/aloha_agilex/robot.yaml"))
    assert description.collision_geometry == []

    band = derive_robot_relative_height_band(description)

    assert band.source == "minimum"
    assert band.min_z_m == pytest.approx(0.10)
    assert band.max_z_m == pytest.approx(0.30)


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
