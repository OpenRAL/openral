"""The payload-inclusive Nav2 footprint, against the real panda_mobile manifest.

Everything here is the geometry the ROS node publishes, exercised without a ROS
graph: the node's pure half takes plain transforms, so these are the same code
paths that run on the robot. The live half — that ``nav2_costmap_2d`` actually
adopts what we publish — is
``tests/integration/test_nav2_payload_footprint_live.py``.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import yaml
from openral_core import RobotDescription
from openral_nav2_bringup.payload_footprint_node import (
    SHAPE_BOX,
    SHAPE_CAPSULE,
    SHAPE_SPHERE,
    base_footprint_polygon,
    convex_hull_2d,
    footprint_with_payload,
    primitive_ground_points,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_PANDA_MOBILE = _REPO_ROOT / "robots" / "panda_mobile" / "robot.yaml"
_NAV2_CONFIG = (
    _REPO_ROOT / "packages" / "openral_nav2_bringup" / "config" / "nav2_panda_mobile.yaml"
)


@pytest.fixture(scope="module")
def panda_mobile() -> RobotDescription:
    return RobotDescription.from_yaml(_PANDA_MOBILE)


def _translation(x: float, y: float, z: float) -> np.ndarray:
    m = np.eye(4)
    m[:3, 3] = (x, y, z)
    return m


def _yaw(angle: float) -> np.ndarray:
    m = np.eye(4)
    c, s = math.cos(angle), math.sin(angle)
    m[0, 0], m[0, 1] = c, -s
    m[1, 0], m[1, 1] = s, c
    return m


def test_base_polygon_is_the_manifests_outline(panda_mobile: RobotDescription) -> None:
    """No payload, no invention: the nominal footprint is the manifest's."""
    polygon = base_footprint_polygon(panda_mobile)

    assert set(polygon) == {(0.35, 0.25), (-0.35, 0.25), (-0.35, -0.25), (0.35, -0.25)}
    assert polygon == convex_hull_2d(polygon), "must already be a CCW hull"


def test_nav2_config_footprint_still_matches_the_manifest(
    panda_mobile: RobotDescription,
) -> None:
    """The shipped Nav2 base config and robot.yaml must not drift apart.

    ``nav2.launch.py`` rewrites ``footprint`` from the manifest when a
    ``robot_yaml`` is passed, so a mismatch here is invisible in deployment and
    loud only when someone launches the base config standalone — with a
    silently wrong robot outline on Nav2's collision surface.
    """
    cfg = yaml.safe_load(_NAV2_CONFIG.read_text())
    expected = panda_mobile.nav2_footprint_param()

    for scope in ("local_costmap", "global_costmap"):
        assert cfg[scope][scope]["ros__parameters"]["footprint"] == expected


def test_radius_only_robot_gets_nav2s_use_the_radius_sentinel() -> None:
    """A manifest with no polygon must not inherit some other robot's shape."""
    description = RobotDescription.from_yaml(_PANDA_MOBILE).model_copy(
        update={"footprint_polygon": None}
    )

    assert description.nav2_footprint_param() == "[]"
    assert description.nav2_param_overrides()["footprint"] == "[]"
    # …and the fallback polygon the node publishes still contains the circle.
    polygon = base_footprint_polygon(description)
    assert min(math.hypot(x, y) for x, y in polygon) >= 0.35


def test_a_carried_object_reaches_past_the_chassis(panda_mobile: RobotDescription) -> None:
    """The whole point: a payload held in front grows the footprint forward."""
    base = base_footprint_polygon(panda_mobile)
    # A 30 cm baguette held 0.55 m ahead of base_link, its local +Z axis (the
    # capsule's segment) rotated onto the robot's +X.
    lying_along_x = np.array(
        [[0.0, 0.0, 1.0, 0.55], [0.0, 1.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.9], [0.0, 0.0, 0.0, 1.0]]
    )

    grown = footprint_with_payload(base, [(SHAPE_CAPSULE, (0.04, 0.30), lying_along_x)])

    assert max(x for x, _ in grown) > max(x for x, _ in base)
    # Capsule half-length 0.15 + radius 0.04 = 0.19 forward of 0.55.
    assert max(x for x, _ in grown) >= 0.55 + 0.15 + 0.04
    # The chassis is still inside the result.
    for vertex in base:
        assert _point_in_polygon(vertex, grown)


def test_detaching_restores_the_nominal_outline(panda_mobile: RobotDescription) -> None:
    """No attachments on the wire means the bare chassis, exactly."""
    base = base_footprint_polygon(panda_mobile)

    assert footprint_with_payload(base, []) == base


def test_a_payload_inside_the_chassis_changes_nothing(panda_mobile: RobotDescription) -> None:
    """A small object tucked over the base must not inflate the footprint."""
    base = base_footprint_polygon(panda_mobile)
    tucked = (SHAPE_SPHERE, (0.03,), _translation(0.0, 0.0, 1.1))

    assert footprint_with_payload(base, [tucked]) == base


def test_round_shapes_circumscribe_rather_than_inscribe() -> None:
    """A sampled circle would lose payload; the projection must contain it."""
    sphere_at_origin = primitive_ground_points(SHAPE_SPHERE, (0.10,), np.eye(4), circle_samples=8)
    hull = convex_hull_2d(sphere_at_origin)

    # Every edge of the polygon is at least the true radius from the centre.
    for (x0, y0), (x1, y1) in zip(hull, hull[1:] + hull[:1], strict=True):
        edge = math.hypot(x1 - x0, y1 - y0)
        area2 = abs(x0 * y1 - x1 * y0)
        assert area2 / edge >= 0.10 - 1e-12


def test_a_rotated_box_projects_exactly() -> None:
    """A box's projection is the hull of its projected corners — no slop."""
    half = 0.10
    rotated = _yaw(math.pi / 4.0)

    hull = convex_hull_2d(primitive_ground_points(SHAPE_BOX, (half, half, half), rotated))

    assert max(x for x, _ in hull) == pytest.approx(half * math.sqrt(2.0))
    assert max(y for _, y in hull) == pytest.approx(half * math.sqrt(2.0))


def test_the_capsule_axis_is_local_z_like_the_kernels() -> None:
    """Same convention as the kernel and the octomap bridge, or the shapes diverge."""
    upright = primitive_ground_points(SHAPE_CAPSULE, (0.02, 1.00), np.eye(4), circle_samples=8)
    # An upright capsule projects to a disc of the capsule RADIUS: the length
    # runs up the z axis and contributes nothing on the ground.
    assert max(math.hypot(x, y) for x, y in upright) < 0.05

    lying = np.array(
        [[0.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    )
    flat = primitive_ground_points(SHAPE_CAPSULE, (0.02, 1.00), lying, circle_samples=8)
    assert max(x for x, _ in flat) >= 0.50


def test_the_margin_only_ever_grows_the_footprint(panda_mobile: RobotDescription) -> None:
    base = base_footprint_polygon(panda_mobile)
    payload = (SHAPE_BOX, (0.10, 0.05, 0.05), _translation(0.55, 0.0, 0.9))

    tight = footprint_with_payload(base, [payload])
    padded = footprint_with_payload(base, [payload], margin_m=0.05)

    assert max(x for x, _ in padded) == pytest.approx(max(x for x, _ in tight) + 0.05)
    for vertex in tight:
        assert _point_in_polygon(vertex, padded)


@pytest.mark.parametrize(
    ("shape_type", "dims"),
    [
        (SHAPE_SPHERE, ()),
        (SHAPE_SPHERE, (-0.1,)),
        (SHAPE_CAPSULE, (0.02,)),
        (SHAPE_BOX, (0.1, 0.1)),
        (99, (0.1,)),
    ],
)
def test_a_primitive_the_kernel_would_reject_is_refused_here_too(
    shape_type: int, dims: tuple[float, ...]
) -> None:
    """Malformed geometry must raise, so the node holds the last footprint."""
    with pytest.raises(ValueError, match=r"shape_type|radius|length|half-extent"):
        primitive_ground_points(shape_type, dims, np.eye(4))


def test_a_negative_margin_is_refused() -> None:
    """Negative margin would shrink the payload — the one forbidden direction."""
    with pytest.raises(ValueError, match="non-negative"):
        primitive_ground_points(SHAPE_SPHERE, (0.05,), np.eye(4), margin_m=-0.01)


def test_collinear_points_are_not_a_polygon() -> None:
    with pytest.raises(ValueError, match=r"collinear|>= 3"):
        convex_hull_2d([(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)])


def _point_in_polygon(point: tuple[float, float], polygon: list[tuple[float, float]]) -> bool:
    """Inside-or-on test for a CCW convex polygon."""
    px, py = point
    for (x0, y0), (x1, y1) in zip(polygon, polygon[1:] + polygon[:1], strict=True):
        if (x1 - x0) * (py - y0) - (y1 - y0) * (px - x0) < -1e-9:
            return False
    return True
