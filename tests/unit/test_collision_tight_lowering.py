"""A link the lowering refines is a box plus the exact hull, and both hold the whole link.

``openral collision lower --tight-link <link>`` (and every link a manifest
already refines) lowers to one box and its ``tight_geometry`` (26-DOP + exact
convex hull), which the kernel re-asks a box pair of when both boxes carry one
(``refine_self_pair``) and runs on every voxel check. The hull has no headroom:
it IS the link's hull, so the safety argument is ``mesh ⊆ hull ⊆ DOP ⊆ box``
on the box as the manifest writes it (4-dp origin), with the refinement written
at 1 nm. Pinned here on a real URDF link (the SO-100 wrist, collision + visual
meshes) and on the MJCF ACM sweep, which must never exempt a pair the kernel
would re-ask of two hulls.

Real fixtures only (CLAUDE.md §1.11).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("yourdfpy")
pytest.importorskip("trimesh")
pytest.importorskip("scipy")
pytest.importorskip("robot_descriptions")

import yaml
from openral_core import RobotDescription
from openral_core.assets import resolve_asset
from openral_core.schemas import TIGHT_CONTAINMENT_EPSILON_M, BoxShape, LinkCollisionGeometry
from openral_safety.urdf_lowering import (
    CAPSULE_HEADROOM_M,
    LoweredCollisionModel,
    _apply,
    _collision_local_vertices,
    _load_urdf,
    _origin_matrix,
    _xyzrpy_matrix,
    lower_link_geometry,
)

from tests.unit._collision_kernel_verdict import tight_escape_m

_REPO = Path(__file__).resolve().parents[2]


def _so100() -> tuple[RobotDescription, str]:
    manifest = _REPO / "robots" / "so100_follower" / "robot.yaml"
    robot = RobotDescription.from_yaml(str(manifest))
    urdf = resolve_asset(robot.assets.urdf.ref, "urdf", manifest_dir=manifest.parent)  # type: ignore[union-attr]  # reason: so100_follower declares assets.urdf
    return robot, str(urdf)


def _link_cloud(urdf: str, link_name: str) -> np.ndarray:
    model = _load_urdf(urdf)
    handler = model._filename_handler  # type: ignore[attr-defined]  # reason: yourdfpy URDF
    link = model.link_map[link_name]  # type: ignore[attr-defined]  # reason: yourdfpy URDF
    return np.vstack(
        [
            _apply(_origin_matrix(el.origin), _collision_local_vertices(el, handler))
            for el in list(link.collisions or []) + list(link.visuals or [])
        ]
    )


@pytest.fixture(scope="module")
def wrist() -> LinkCollisionGeometry:
    _, urdf = _so100()
    lowered = lower_link_geometry(urdf, tight_links={"wrist"})
    entries = [g for g in lowered if g.link_name == "wrist"]
    assert len(entries) == 1, entries
    return entries[0]


def test_a_refined_link_is_one_box_whose_dop_and_hull_hold_the_link(
    wrist: LinkCollisionGeometry,
) -> None:
    _, urdf = _so100()
    assert isinstance(wrist.shape, BoxShape)
    tight = wrist.tight_geometry
    assert tight is not None and tight.hull_vertices_m, "a refined link must ship a stage-2 hull"
    assert tight.hull_overhang_m is not None and tight.hull_overhang_m >= 0.0
    # The origin is already at the manifest's 4-dp precision, so the frame the
    # refinement was derived in is the frame the kernel will place.
    assert all(round(v, 4) == v for v in wrist.origin_xyz_rpy)

    tf = _xyzrpy_matrix(wrist.origin_xyz_rpy)
    local = (_link_cloud(urdf, "wrist") - tf[:3, 3]) @ tf[:3, :3]
    # Box: every vertex inside with (nearly) the declared headroom.
    over = np.abs(local) - np.asarray(wrist.shape.half_extents_m)
    assert over.max() <= -0.9 * CAPSULE_HEADROOM_M, f"box headroom {-over.max() * 1e3:.3f} mm"
    # DOP and hull: every vertex inside (the hull is the link's own hull).
    assert tight_escape_m(wrist, local) <= TIGHT_CONTAINMENT_EPSILON_M


def test_only_the_named_links_are_refined() -> None:
    _, urdf = _so100()
    geoms = lower_link_geometry(urdf, tight_links={"wrist"})
    assert {g.link_name for g in geoms if g.tight_geometry is not None} == {"wrist"}


def test_a_refined_link_survives_the_manifest_round_trip(wrist: LinkCollisionGeometry) -> None:
    """``render_blocks`` writes the refinement exactly; the manifest loads it back equal."""
    from openral_cli.collision import render_blocks

    geo_block, _ = render_blocks(
        LoweredCollisionModel(collision_geometry=[wrist], allowed_collision_pairs=[])
    )
    entry = yaml.safe_load(geo_block)["collision_geometry"][0]
    loaded = LinkCollisionGeometry.model_validate(entry)
    assert loaded == wrist


def test_the_mjcf_sweep_never_exempts_a_pair_of_hulled_links() -> None:
    """A pair whose boxes overlap at every sampled pose is exempted — unless both carry hulls.

    The kernel re-asks such a pair of the two hulls, so a box overlap says
    nothing about what it will check; the sweep must leave the pair checked, as
    ``_certified_always_colliding`` does on the URDF path. The control half
    shows the same pair IS exempted when neither box carries a hull, so the
    withholding (not the geometry) is what keeps it checked.
    """
    pytest.importorskip("mujoco")
    from openral_safety.tight_geometry import tight_geometry_for_box
    from openral_safety.urdf_lowering import lower_robot_from_mjcf

    manifest = _REPO / "robots" / "openarm" / "robot.yaml"
    robot = RobotDescription.from_yaml(str(manifest))
    # The shipped OpenArm refines every link; strip the refinements so the
    # control half sees plain boxes, then refine exactly one always-colliding pair.
    robot = robot.model_copy(
        update={
            "assets": robot.assets.model_copy(update={"srdf": None}),
            "collision_geometry": [
                g.model_copy(update={"tight_geometry": None}) for g in robot.collision_geometry
            ],
        }
    )
    boxes = [g for g in robot.collision_geometry if isinstance(g.shape, BoxShape)]
    by_link = {g.link_name: g for g in boxes}
    plain = lower_robot_from_mjcf(robot, manifest_dir=manifest.parent, acm_only=True, n_samples=24)
    exempt_boxes = [
        (a, b) for a, b in plain.allowed_collision_pairs if a in by_link and b in by_link
    ]
    assert exempt_boxes, "no box pair is always-colliding on the OpenArm; pick another fixture"
    a, b = exempt_boxes[0]

    import trimesh

    def hulled(g: LinkCollisionGeometry) -> LinkCollisionGeometry:
        assert isinstance(g.shape, BoxShape)
        half = g.shape.half_extents_m
        box = trimesh.creation.box(extents=[2.0 * h - 2e-4 for h in half])
        return g.model_copy(update={"tight_geometry": tight_geometry_for_box(box, half)})

    refined = [hulled(g) if g.link_name in (a, b) else g for g in robot.collision_geometry]
    lowered = lower_robot_from_mjcf(
        robot.model_copy(update={"collision_geometry": refined}),
        manifest_dir=manifest.parent,
        acm_only=True,
        n_samples=24,
    )
    assert (a, b) not in lowered.allowed_collision_pairs
    # Nothing else moved: withholding touches only the hulled pair.
    assert set(lowered.allowed_collision_pairs) == set(plain.allowed_collision_pairs) - {(a, b)}
