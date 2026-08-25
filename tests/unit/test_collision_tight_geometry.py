"""The containment obligation the safety kernel cannot discharge for itself.

The kernel checks, at ``on_configure``, that every declared tight
representation sits inside the shipped OBB whose broad-phase window it will be
checked in (``validate_tight_geometry``), and that each stage-2 hull vertex
satisfies its own 26-DOP slabs. That is two of the three links in the chain::

    link mesh  ⊆  exact convex hull  ⊆  26-DOP  ⊆  shipped OBB

The first link is the one the kernel structurally cannot verify: it never sees
a mesh. It is discharged here, against the **real** robosuite collision meshes
the manifest's numbers were derived from — not against a fixture, and not
against the generator's own intermediate output, because a generator checking
itself proves nothing about the numbers that shipped.

This is the file a safety-WG reviewer should read to satisfy the "containment,
per link, as a proof and not a sample" obligation in
``docs/reference/collision-tight-geometry.md`` §10.5 item 1.

Requires robosuite + mujoco (the mesh source). Without them the mesh-side
proofs cannot run and are skipped; the pure-schema invariants below still do.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
from openral_core.schemas import (
    DOP_AXES,
    MAX_TIGHT_HULL_VERTICES,
    BoxShape,
    LinkCollisionGeometry,
    RobotDescription,
    TightCollisionGeometry,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
PANDA_MANIFEST = REPO_ROOT / "robots" / "panda_mobile" / "robot.yaml"
# The VSLAM variant is the same arm on the same base, and
# `tests/unit/test_collision_geometry_contracts.py` requires the two manifests'
# `panda_link*` entries to be identical -- so its geometry is verified against
# the same meshes rather than trusted to have been copied correctly.
VSLAM_MANIFEST = REPO_ROOT / "robots" / "panda_mobile_vslam" / "robot.yaml"

pytest.importorskip("numpy", reason="numpy is required to check mesh containment")


def _mesh_tools():
    """Import the generator + its mesh dependencies, or skip."""
    pytest.importorskip("mujoco", reason="mujoco is the mesh source for the containment proof")
    pytest.importorskip("robosuite", reason="robosuite ships the Panda collision meshes")
    pytest.importorskip("scipy", reason="scipy.spatial.ConvexHull is the hull reference")
    import sys

    sys.path.insert(0, str(REPO_ROOT / "tools"))
    import generate_tight_geometry as gen

    return gen


@pytest.fixture(scope="module")
def panda() -> RobotDescription:
    return RobotDescription.from_yaml(PANDA_MANIFEST)


def _declared(robot: RobotDescription) -> list[LinkCollisionGeometry]:
    return [g for g in robot.collision_geometry if g.tight_geometry is not None]


def test_the_manifest_actually_declares_tight_geometry(panda: RobotDescription) -> None:
    """Guard against this whole file silently passing on an empty set."""
    declared = _declared(panda)
    assert {g.link_name for g in declared} == {"panda_link1", "panda_link2"}, (
        "the scope is link1 + link2 (the two links holding 60 of the census's 72 stops); "
        "widening it needs its own measurement and its own hazard-entry line"
    )


def test_link1_ships_stage_one_only_because_its_hull_is_over_budget(
    panda: RobotDescription,
) -> None:
    """`panda_link1`'s exact hull is 1588 vertices and was measured slower than the box.

    Recorded as a test rather than a comment because it is the one place the
    scoping decision is falsifiable: if a future change makes the exact hull
    affordable at that vertex count, this fails and the decision gets revisited
    deliberately instead of by drift.
    """
    link1 = next(g for g in panda.collision_geometry if g.link_name == "panda_link1")
    assert link1.tight_geometry is not None
    assert link1.tight_geometry.hull_vertices_m == (), (
        "link1 runs the 26-DOP only; its exact hull is 1588 vertices, over "
        f"MAX_TIGHT_HULL_VERTICES={MAX_TIGHT_HULL_VERTICES}, and measured 0.77x the shipped "
        "routine's speed at 400 occupied cells"
    )


def test_link2_ships_its_exact_hull(panda: RobotDescription) -> None:
    link2 = next(g for g in panda.collision_geometry if g.link_name == "panda_link2")
    assert link2.tight_geometry is not None
    assert len(link2.tight_geometry.hull_vertices_m) == 152
    assert len(link2.tight_geometry.hull_vertices_m) <= MAX_TIGHT_HULL_VERTICES


def test_every_declared_dop_sits_inside_its_shipped_box(panda: RobotDescription) -> None:
    """`26-DOP ⊆ shipped OBB` — the link that keeps the broad-phase window correct.

    The kernel sizes its voxel window from ``half_extents_m`` alone. A tight
    representation reaching outside that box would make the kernel skip cells it
    must visit, which is a **missed collision**, not a lost conservatism — the
    one way this whole change could make the kernel unsafe rather than merely
    tighter. Checked here with no tolerance at all, because the shipped boxes
    carry a real measured margin rather than a numerical one.
    """
    for geom in _declared(panda):
        assert isinstance(geom.shape, BoxShape)
        tight = geom.tight_geometry
        assert tight is not None
        margins = []
        for k, half in enumerate(geom.shape.half_extents_m):
            assert tight.dop_lo_m[k] >= -half, f"{geom.link_name} axis {k} escapes below"
            assert tight.dop_hi_m[k] <= half, f"{geom.link_name} axis {k} escapes above"
            margins.append(min(half - tight.dop_hi_m[k], half + tight.dop_lo_m[k]))
        # The margin is 0.055-0.132 mm across the fleet -- accidental headroom in
        # the shipped fit, which is exactly why nothing is allowed to consume it.
        assert 0.0 < min(margins) < 2e-4, f"{geom.link_name} margin {min(margins)}"


def test_the_real_link_mesh_is_inside_every_declared_dop(panda: RobotDescription) -> None:
    """`mesh ⊆ 26-DOP`, against robosuite's own collision meshes.

    Definitional, not fitted: each slab bound is ``max`` of ``u·x`` over the
    mesh, so the correct answer is exactly zero escape, and any drift between
    the manifest and the asset shows up immediately rather than as a tolerance
    being slowly eaten.
    """
    gen = _mesh_tools()
    import numpy as np

    xml, geoms = gen._sources_for(panda.name)
    axes = np.asarray(DOP_AXES, dtype=float)
    for geom in _declared(panda):
        points = gen.link_mesh_in_box_frame(xml, geoms[geom.link_name], geom.origin_xyz_rpy)
        tight = geom.tight_geometry
        assert tight is not None
        proj = points @ axes.T
        over = float((proj - np.asarray(tight.dop_hi_m)).max())
        under = float((np.asarray(tight.dop_lo_m) - proj).max())
        assert over <= 1e-9, f"{geom.link_name}: mesh escapes its DOP by {over * 1e3:.6f} mm"
        assert under <= 1e-9, f"{geom.link_name}: mesh escapes its DOP by {under * 1e3:.6f} mm"


def test_the_real_link_mesh_is_inside_every_declared_hull(panda: RobotDescription) -> None:
    """`mesh ⊆ exact convex hull`, checked facet by facet rather than by sampling.

    Every mesh vertex must satisfy every facet halfspace of the declared hull.
    A truncated, stale, or reordered vertex list fails this; a hull that merely
    *looks* right does not pass it.
    """
    gen = _mesh_tools()
    import numpy as np
    from scipy.spatial import ConvexHull

    xml, geoms = gen._sources_for(panda.name)
    checked = 0
    for geom in _declared(panda):
        tight = geom.tight_geometry
        assert tight is not None
        if not tight.hull_vertices_m:
            continue
        points = gen.link_mesh_in_box_frame(xml, geoms[geom.link_name], geom.origin_xyz_rpy)
        hull = ConvexHull(np.asarray(tight.hull_vertices_m, dtype=float))
        normals = hull.equations[:, :3]
        offsets = hull.equations[:, 3]
        slack = (points @ normals.T + offsets) / np.linalg.norm(normals, axis=1)
        worst = float(slack.max())
        assert worst <= 1e-9, f"{geom.link_name}: mesh escapes its hull by {worst * 1e3:.6f} mm"
        checked += 1
    assert checked == 1, "panda_link2 is the only link shipping a stage-2 hull today"


def test_the_declared_geometry_reproduces_from_the_mesh(panda: RobotDescription) -> None:
    """The manifest is what the generator produces today, not a hand-edited drift.

    ``tools/generate_tight_geometry.py check`` is the same gate; running it from
    here keeps it inside `just test` rather than depending on someone
    remembering to invoke it.
    """
    gen = _mesh_tools()
    for manifest in (PANDA_MANIFEST, VSLAM_MANIFEST):
        assert gen.main(["check", "--robot", str(manifest)]) == 0, manifest


def test_the_tightening_is_real_and_measured(panda: RobotDescription) -> None:
    """The DOP must actually be smaller than the box, by the amount claimed.

    A containment proof alone would be satisfied by a DOP identical to the box.
    This pins the *benefit* side of the hazard entry: the support excess drops
    from 53.3 mm to 25.7 mm on link1 and from 46.8 mm to 23.0 mm on link2, and a
    change that quietly lost that would still pass every containment test above.
    """
    gen = _mesh_tools()
    import numpy as np

    xml, geoms = gen._sources_for(panda.name)
    axes = np.asarray(DOP_AXES, dtype=float)
    # A 20 000-direction Fibonacci set, matching the study's own metric
    # (docs/reference/collision-tight-geometry.md §11).
    n = 20000
    i = np.arange(n) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / n)
    theta = math.pi * (1.0 + 5.0**0.5) * i
    dirs = np.stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], axis=1)
    expected = {"panda_link1": (53.27, 25.69), "panda_link2": (46.83, 23.01)}
    for geom in _declared(panda):
        points = gen.link_mesh_in_box_frame(xml, geoms[geom.link_name], geom.origin_xyz_rpy)
        tight = geom.tight_geometry
        assert tight is not None
        assert isinstance(geom.shape, BoxShape)
        h_mesh = (points @ dirs.T).max(axis=0)
        h_box = np.abs(dirs) @ np.asarray(geom.shape.half_extents_m)
        box_excess = float((h_box - h_mesh).max()) * 1e3
        # The DOP's support along its OWN axes is its slab bound, exactly, so the
        # per-axis excess is available without enumerating the polytope.
        dop_axis_excess = float((np.asarray(tight.dop_hi_m) - (points @ axes.T).max(axis=0)).max())
        assert abs(dop_axis_excess) <= 1e-12, "the slabs must be tangent, not padded"
        want_box, want_dop = expected[geom.link_name]
        assert box_excess == pytest.approx(want_box, abs=0.1), (
            f"{geom.link_name}: the shipped box's support excess moved; the hazard entry "
            "quotes this number"
        )
        # And the DOP is strictly tighter than the box on every direction it bounds.
        assert box_excess > want_dop, f"{geom.link_name} would not be a tightening"


def test_a_hull_that_escapes_its_dop_is_refused_at_the_manifest_boundary() -> None:
    """Fail-closed at the schema, not only in the kernel."""
    with pytest.raises(ValueError, match="escapes DOP slab"):
        TightCollisionGeometry(
            dop_lo_m=(-0.05,) * len(DOP_AXES),
            dop_hi_m=(0.05,) * len(DOP_AXES),
            hull_vertices_m=((0.06, 0.0, 0.0),),
        )


def test_tight_geometry_reaching_outside_its_box_is_refused() -> None:
    """The broad-phase guard, at the manifest boundary.

    This is the failure mode that would make the kernel *unsafe* rather than
    merely less tight, so it is refused in three independent places: here, in
    the kernel's ``validate_tight_geometry``, and by the generator, which never
    emits geometry it cannot place inside the box.
    """
    slabs = TightCollisionGeometry(
        dop_lo_m=(-0.05,) * len(DOP_AXES), dop_hi_m=(0.05,) * len(DOP_AXES)
    )
    with pytest.raises(ValueError, match="escapes its box"):
        LinkCollisionGeometry(
            link_name="panda_link1",
            shape=BoxShape(half_extents_m=(0.04, 0.06, 0.06)),
            tight_geometry=slabs,
        )


def test_tight_geometry_on_a_capsule_is_refused() -> None:
    """The containment proof is stated against a box; there is nothing to refine on a capsule."""
    from openral_core.schemas import CapsuleShape

    slabs = TightCollisionGeometry(
        dop_lo_m=(-0.05,) * len(DOP_AXES), dop_hi_m=(0.05,) * len(DOP_AXES)
    )
    with pytest.raises(ValueError, match="refines a BoxShape only"):
        LinkCollisionGeometry(
            link_name="h1_link",
            shape=CapsuleShape(radius_m=0.04, length_m=0.3),
            tight_geometry=slabs,
        )


def test_a_hull_over_the_vertex_budget_is_refused() -> None:
    """The cost ceiling is enforced, not advised.

    ``MAX_TIGHT_HULL_VERTICES`` is a real-time bound: past it the kernel's
    exhaustive support scan costs more than the ``box_box_distance`` it
    replaces, and the staged path stops paying for itself.
    """
    lo = (-1.0,) * len(DOP_AXES)
    hi = (1.0,) * len(DOP_AXES)
    too_many = tuple((0.0, 0.0, 0.001 * k) for k in range(MAX_TIGHT_HULL_VERTICES + 1))
    with pytest.raises(ValueError, match="over the"):
        TightCollisionGeometry(dop_lo_m=lo, dop_hi_m=hi, hull_vertices_m=too_many)
