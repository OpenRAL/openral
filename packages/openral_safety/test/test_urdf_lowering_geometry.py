"""Capsule/sphere fitting from a vertex cloud — the conservative-containment law.

The geometry half of the offline lowering tool (ADR-0030): a fitted capsule MUST
contain every source vertex so the safety check is a conservative
over-approximation (never under-covers). Tests use real geometry (the panda URDF
and tiny inline URDFs), no mocks (CLAUDE.md §1.11).
"""

from __future__ import annotations

import numpy as np
import pytest
from openral_core import CapsuleShape, SphereShape
from openral_safety.urdf_lowering import fit_capsule_to_vertices, lower_link_geometry


def _point_in_capsule(p, shape: CapsuleShape, origin_xyz_rpy) -> bool:
    """True if world point ``p`` is inside the capsule (segment along +Z, length_m)."""
    from openral_safety.mjcf_lowering import _rpy_to_mat

    x, y, z, roll, pitch, yaw = origin_xyz_rpy
    r = np.array(_rpy_to_mat(roll, pitch, yaw)).reshape(3, 3)
    t = np.array([x, y, z])
    local = r.T @ (np.asarray(p) - t)
    h = shape.length_m / 2.0
    axial = float(np.clip(local[2], -h, h))
    radial = float(np.linalg.norm(local - np.array([0.0, 0.0, axial])))
    return radial <= shape.radius_m + 1e-9


# ── fit_capsule_to_vertices: containment is the contract ──────────────────────


def test_fit_capsule_contains_every_vertex_of_a_rod() -> None:
    rng = np.random.default_rng(0)
    axis = np.linspace(-0.2, 0.2, 200)
    pts = np.stack([axis, rng.normal(0, 0.005, 200), rng.normal(0, 0.005, 200)], axis=1)
    shape, origin = fit_capsule_to_vertices(pts)
    assert isinstance(shape, CapsuleShape)
    assert all(_point_in_capsule(p, shape, origin) for p in pts), "capsule must contain all pts"
    assert shape.length_m > 0.30
    assert 0.0 < shape.radius_m < 0.05


def test_fit_capsule_contains_an_offaxis_box_cloud() -> None:
    corners = np.array(
        [[sx * 0.1, sy * 0.05, sz * 0.15] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)],
        dtype=float,
    )
    shape, origin = fit_capsule_to_vertices(corners)
    assert all(_point_in_capsule(c, shape, origin) for c in corners)


def test_fit_capsule_diagonal_rod_containment() -> None:
    # A rod along the (1,1,1) diagonal — exercises the +Z→axis rotation thoroughly.
    t = np.linspace(-0.3, 0.3, 150)
    axis = np.array([1.0, 1.0, 1.0]) / np.sqrt(3)
    rng = np.random.default_rng(1)
    pts = np.outer(t, axis) + rng.normal(0, 0.004, (150, 3))
    shape, origin = fit_capsule_to_vertices(pts)
    assert all(_point_in_capsule(p, shape, origin) for p in pts)


# ── lower_link_geometry: real + inline URDFs ──────────────────────────────────

_BOX_URDF = """<robot name="t">
  <link name="base"><collision><origin xyz="0.01 0.02 0.03" rpy="0 0 0"/>
    <geometry><box size="0.1 0.2 0.3"/></geometry></collision></link>
</robot>"""

_SPHERE_URDF = """<robot name="t">
  <link name="base"><collision><origin xyz="0.05 0 0" rpy="0 0 0"/>
    <geometry><sphere radius="0.07"/></geometry></collision></link>
</robot>"""

_CYLINDER_URDF = """<robot name="t">
  <link name="base"><collision><origin xyz="0 0 0" rpy="0 0 0"/>
    <geometry><cylinder radius="0.04" length="0.5"/></geometry></collision></link>
</robot>"""


def _write_and_lower(tmp_path, urdf_text):
    # lower_link_geometry loads the URDF via yourdfpy (optional dep, absent in the
    # base CI tier — its sibling test_urdf_lowering_*.py modules skip the same way).
    pytest.importorskip("yourdfpy")
    path = tmp_path / "robot.urdf"
    path.write_text(urdf_text, encoding="utf-8")
    return lower_link_geometry(str(path))


def test_box_collision_fits_containing_capsule(tmp_path) -> None:
    geoms = _write_and_lower(tmp_path, _BOX_URDF)
    assert len(geoms) == 1
    g = geoms[0]
    assert g.link_name == "base"
    assert isinstance(g.shape, CapsuleShape)
    # box half-diagonal = sqrt(0.05^2+0.1^2+0.15^2) ≈ 0.187 → capsule must reach every corner.
    corners = np.array(
        [
            [0.01 + sx * 0.05, 0.02 + sy * 0.1, 0.03 + sz * 0.15]
            for sx in (-1, 1)
            for sy in (-1, 1)
            for sz in (-1, 1)
        ],
        dtype=float,
    )
    assert all(_point_in_capsule(c, g.shape, g.origin_xyz_rpy) for c in corners)


def test_sphere_collision_maps_to_sphere(tmp_path) -> None:
    geoms = _write_and_lower(tmp_path, _SPHERE_URDF)
    assert len(geoms) == 1
    g = geoms[0]
    assert isinstance(g.shape, SphereShape)
    assert g.shape.radius_m == pytest.approx(0.07, abs=1e-6)
    assert g.origin_xyz_rpy[0] == pytest.approx(0.05, abs=1e-6)


def test_cylinder_collision_fits_capsule(tmp_path) -> None:
    geoms = _write_and_lower(tmp_path, _CYLINDER_URDF)
    assert len(geoms) == 1
    g = geoms[0]
    assert isinstance(g.shape, CapsuleShape)
    # Capsule must span the 0.5 m cylinder and have radius >= 0.04.
    assert g.shape.length_m >= 0.49
    assert g.shape.radius_m >= 0.04 - 1e-6


def test_lower_link_geometry_panda_arm_links() -> None:
    pytest.importorskip("yourdfpy")
    pytest.importorskip("robot_descriptions")
    from openral_core.assets import resolve_asset

    urdf = resolve_asset("rd:panda_description", "urdf")
    assert urdf is not None
    geoms = lower_link_geometry(str(urdf))
    by_link = {g.link_name: g for g in geoms}
    assert any(name.startswith("panda_link") for name in by_link)
    for g in geoms:
        assert g.shape.radius_m > 0.0
