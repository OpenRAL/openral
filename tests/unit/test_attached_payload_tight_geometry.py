# SPDX-License-Identifier: Apache-2.0
"""Stage-2 geometry for a CARRIED payload — the producer half of issue #266.

Robot links got the staged 26-DOP → exact-hull narrow phase in #166. Attached
payloads never did: ``extract_body_primitives`` lowers a carried **mesh** geom
to its local AABB, and that is not a fallback after a failed refinement — it was
the only lowering a carried mesh had.

Measured across all 80 rounds of the 2026-09-10 resolution A/B (424 samples, 4
scenes) the payload box's corners stood a **median 50.78 mm** (max 88.22 mm)
proud of the mesh. For scale the world-voxel half-diagonal #253 proposed to
shrink is 21.65 mm at 25 mm cells and 12.99 mm at 15 mm — the payload box was
2.3-3.9x the entire quantisation term, which is why that A/B returned a null
while 97 % of the 15 mm arm's stops were payload-vs-world.

What is pinned here is the producer's side of the containment chain the kernel's
ingest re-proves: ``mesh ⊆ hull ⊆ DOP ⊆ box``, definitional at every link.

Real compiled MuJoCo models throughout, no mocks (CLAUDE.md §1.11).
"""

from __future__ import annotations

import math
from typing import Any, ClassVar

import numpy as np
import pytest
from openral_core.schemas import (
    DOP_AXES,
    MAX_TIGHT_HULL_VERTICES,
    AttachedCollisionPrimitive,
    BoxShape,
)
from openral_hal._sim_attachment_evidence import extract_body_primitives

mujoco = pytest.importorskip("mujoco")

# An octahedron with vertices at ±_R on each axis. Its AABB is the cube of
# half-side _R, whose corner (R,R,R) is `R*sqrt(2)` from the nearest surface
# point — the whole defect, at a size a test can name exactly.
_R = 0.03
_OCTAHEDRON_VERTS = f"{_R} 0 0  {-_R} 0 0  0 {_R} 0  0 {-_R} 0  0 0 {_R}  0 0 {-_R}"
_MJCF = f"""
<mujoco model="payload_tight_geometry">
  <option gravity="0 0 0"/>
  <asset>
    <mesh name="octahedron" vertex="{_OCTAHEDRON_VERTS}"/>
  </asset>
  <worldbody>
    <body name="mesh_payload" pos="0.5 0 0.5">
      <freejoint name="mesh_payload_free"/>
      <geom name="mesh_payload_g0" type="mesh" mesh="octahedron"/>
    </body>
    <body name="box_payload" pos="0.5 0.4 0.5">
      <freejoint name="box_payload_free"/>
      <geom name="box_payload_g0" type="box" size="0.02 0.03 0.04"/>
    </body>
    <body name="sphere_payload" pos="0.5 0.8 0.5">
      <freejoint name="sphere_payload_free"/>
      <geom name="sphere_payload_g0" type="sphere" size="0.025"/>
    </body>
    <!-- More geoms than the default primitive cap, so the CLUSTERED lowering
         runs: that path merges several geoms into one box and is the looser of
         the two, and it is the one a real RoboCasa baguette (16 primitives)
         takes. -->
    <body name="many_mesh_payload" pos="0.5 1.2 0.5">
      <freejoint name="many_mesh_payload_free"/>
      <geom name="m0" type="mesh" mesh="octahedron" pos="0 0 0"/>
      <geom name="m1" type="mesh" mesh="octahedron" pos="0.06 0 0"/>
      <geom name="m2" type="mesh" mesh="octahedron" pos="0.12 0 0"/>
      <geom name="m3" type="mesh" mesh="octahedron" pos="0.18 0 0"/>
      <geom name="m4" type="mesh" mesh="octahedron" pos="0.24 0 0"/>
    </body>
  </worldbody>
</mujoco>
"""


def _compiled() -> tuple[Any, Any]:
    model = mujoco.MjModel.from_xml_string(_MJCF)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def _body(model: Any, name: str) -> int:
    return int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name))


def _primitives(name: str, *, max_primitives: int = 16) -> list[AttachedCollisionPrimitive]:
    model, data = _compiled()
    return extract_body_primitives(
        model,
        data,
        root_body_id=_body(model, name),
        object_id=f"sim:{name}",
        max_primitives=max_primitives,
    )


def _payload_points(name: str) -> Any:
    """Every surface point of one payload, in its root body frame."""
    from openral_hal._sim_attachment_evidence import (
        _body_subtree,
        _geom_surface_points_in_root,
    )

    model, data = _compiled()
    root = _body(model, name)
    subtree = _body_subtree(model, root)
    return np.vstack(
        [
            _geom_surface_points_in_root(model, data, geom_id=g, root_body_id=root)
            for g in range(int(model.ngeom))
            if int(model.geom_bodyid[g]) in subtree
        ]
    )


def test_a_carried_mesh_now_ships_its_own_hull() -> None:
    """The fix, stated at its narrowest: a mesh payload is no longer just a box."""
    (prim,) = _primitives("mesh_payload")
    assert isinstance(prim.shape, BoxShape)
    tight = prim.tight_geometry
    assert tight is not None, "a mesh geom's AABB is exactly what #266 exists to refine"
    # The octahedron's exact hull IS its six vertices.
    assert len(tight.hull_vertices_m) == 6
    assert (
        sorted(round(abs(c), 6) for v in tight.hull_vertices_m for c in v).count(round(_R, 6)) == 6
    )


def test_the_corner_slop_the_ab_measured_is_what_the_refinement_removes() -> None:
    """50.78 mm median, reproduced in the small and then recovered.

    The box corner is ``R*sqrt(2)`` from the octahedron's surface. The DOP's
    corner slab is tangent to the (1,1,1)/sqrt(3) face, so it gives that
    distance back entirely — before stage 2 is even consulted, which matters
    because stage 2 is budget-capped in the kernel and stage 1 is not.
    """
    (prim,) = _primitives("mesh_payload")
    tight = prim.tight_geometry
    assert tight is not None
    assert isinstance(prim.shape, BoxShape)

    box_corner = np.array(prim.shape.half_extents_m)
    points = _payload_points("mesh_payload") - np.array(prim.pose_in_object.xyz)
    box_slop = float(np.min(np.linalg.norm(points - box_corner, axis=1)))
    # `R*sqrt(2)` from the corner to the nearest vertex, plus the producer's
    # own 1e-4 m inflation of the AABB.
    assert box_slop == pytest.approx(_R * math.sqrt(2.0), abs=3e-4)

    # The DOP's own corner support along (1,1,1)/sqrt(3): tangent to the
    # octahedron's face plane x+y+z = R, i.e. R/sqrt(3), not R*sqrt(3).
    corner_axis = 3  # DOP_AXES[3] is (1,1,1)/sqrt(3), the first corner axis
    assert DOP_AXES[corner_axis] == pytest.approx((3**-0.5,) * 3)
    assert tight.dop_hi_m[corner_axis] == pytest.approx(_R / math.sqrt(3.0), abs=1e-6)
    box_support = float(np.dot(box_corner, DOP_AXES[corner_axis]))
    assert box_support == pytest.approx(_R * math.sqrt(3.0), abs=3e-4)
    # Two thirds of the box's reach along the worst axis, gone at stage 1.
    assert tight.dop_hi_m[corner_axis] < box_support / 2.0


def test_the_containment_chain_holds_for_every_lowered_payload() -> None:
    """``mesh ⊆ hull ⊆ DOP ⊆ box`` — the entire safety argument, on real models.

    Each link is definitional rather than fitted, so this is a proof re-run, not
    a tolerance check: slabs are tangent halfspaces over the same points the
    hull is built from, and the box is those points' own AABB grown by 1e-4 m.
    The kernel re-proves the last two links at ingest and refuses what it cannot
    prove; it never sees the mesh, which is why the first link is checked here.
    """
    axes = np.asarray(DOP_AXES, dtype=np.float64)
    for name, cap in (("mesh_payload", 16), ("many_mesh_payload", 2)):
        points = _payload_points(name)
        covered = np.zeros(len(points), dtype=bool)
        for prim in _primitives(name, max_primitives=cap):
            tight = prim.tight_geometry
            assert tight is not None, name
            assert isinstance(prim.shape, BoxShape)
            centre = np.array(prim.pose_in_object.xyz)
            half = np.array(prim.shape.half_extents_m)
            hull = np.asarray(tight.hull_vertices_m, dtype=np.float64)
            lo = np.asarray(tight.dop_lo_m)
            hi = np.asarray(tight.dop_hi_m)

            # DOP ⊆ box, on the three axes that prove it for the whole polytope.
            assert np.all(lo[:3] >= -half - 1e-12), name
            assert np.all(hi[:3] <= half + 1e-12), name

            # hull ⊆ DOP.
            support = hull @ axes.T
            assert np.all(support <= hi + 1e-9), name
            assert np.all(support >= lo - 1e-9), name
            assert len(hull) <= MAX_TIGHT_HULL_VERTICES

            # mesh ⊆ DOP, for the geometry THIS primitive covers. A cluster box
            # bounds only its own geoms, so the payload-wide statement is that
            # the primitives' DOPs cover every surface point BETWEEN them —
            # accumulated below. (mesh ⊆ hull needs the hull's faces; the DOP is
            # the stage the kernel always runs and the solid the budget-capped
            # stage 2 falls back to, so it is the one that must contain.)
            mesh_support = (points - centre) @ axes.T
            covered |= np.all((mesh_support <= hi + 1e-9) & (mesh_support >= lo - 1e-9), axis=1)
        assert covered.all(), f"{name}: {int((~covered).sum())} surface points escape every DOP"


def test_an_exact_lowering_is_refined_by_nothing() -> None:
    """A sphere/box geom has no representation gap, so it ships no refinement.

    ``None`` here is a fact about the geometry, not a failure to build one —
    the kernel checks these as the sphere and box they are, with no corner to
    overhang.
    """
    for name in ("box_payload", "sphere_payload"):
        (prim,) = _primitives(name)
        assert prim.tight_geometry is None, name


def test_the_clustered_lowering_is_refined_too() -> None:
    """The looser of the two lowerings is the one a real payload takes.

    A cluster box bounds several geoms at once, so it stands proud of the merged
    AABB as well as of each mesh — and a RoboCasa ``baguette`` lowers to 16
    primitives. Leaving this path on the plain box would have refined the case
    that barely occurs and missed the one the A/B measured.
    """
    prims = _primitives("many_mesh_payload", max_primitives=2)
    assert len(prims) == 2, "the cluster path must actually have run"
    for prim in prims:
        assert prim.tight_geometry is not None
        assert len(prim.tight_geometry.hull_vertices_m) > 0


def test_a_refinement_that_escapes_its_box_is_refused_at_the_schema() -> None:
    """The one failure that could make the kernel's broad phase skip a cell.

    The window is sized from ``half_extents_m`` alone, so a representation
    reaching outside the box is a missed collision rather than a lost
    conservatism. Refused at the producer boundary, and again by the kernel at
    ingest (``validate_tight_hull``) — a defence per side of the wire, since
    neither can prove the other ran.
    """
    from openral_core.schemas import Pose6D, TightCollisionGeometry

    pose = Pose6D(xyz=(0.0, 0.0, 0.0), quat_xyzw=(0.0, 0.0, 0.0, 1.0), frame_id="sim:obj")
    escaping = TightCollisionGeometry(
        dop_lo_m=(-0.05,) * 13,
        dop_hi_m=(0.05,) * 13,
    )
    with pytest.raises(ValueError, match="escapes its box"):
        AttachedCollisionPrimitive(
            shape=BoxShape(half_extents_m=(0.01, 0.1, 0.1)),
            pose_in_object=pose,
            tight_geometry=escaping,
        )
    # ... and a refinement on a shape that is not the broad-phase box at all.
    from openral_core.schemas import SphereShape

    with pytest.raises(ValueError, match="refines a BoxShape only"):
        AttachedCollisionPrimitive(
            shape=SphereShape(radius_m=0.1),
            pose_in_object=pose,
            tight_geometry=escaping,
        )


def test_the_refinement_survives_the_wire_unchanged() -> None:
    """Round-trip through the duck-typed IDL the kernel actually decodes.

    A refinement that does not reach the kernel is a producer talking to itself,
    so the encode/decode pair is pinned rather than assumed.
    """

    class _Pose:
        class _V:
            x = y = z = w = 0.0

        position = _V()
        orientation = _V()

    class _Msg:
        SHAPE_SPHERE = 1
        SHAPE_CAPSULE = 2
        SHAPE_BOX = 3
        shape_type = 0
        shape_dimensions: ClassVar[list[float]] = []
        tight_dop_lo: ClassVar[list[float]] = []
        tight_dop_hi: ClassVar[list[float]] = []
        tight_hull_vertices: ClassVar[list[float]] = []
        pose_in_object = _Pose()

    (original,) = _primitives("mesh_payload")
    assert original.tight_geometry is not None
    msg = _Msg()
    original.fill_idl(msg)
    assert len(msg.tight_dop_lo) == len(DOP_AXES)
    assert len(msg.tight_hull_vertices) == 3 * len(original.tight_geometry.hull_vertices_m)

    decoded = AttachedCollisionPrimitive.from_idl(msg, object_id="sim:mesh_payload")
    assert decoded.tight_geometry == original.tight_geometry

    # A publisher that ships nothing decodes to nothing, which is the pre-#266
    # wire and every exactly-lowered primitive.
    plain = _Msg()
    plain.shape_type = _Msg.SHAPE_BOX
    plain.shape_dimensions = [0.1, 0.1, 0.1]
    assert AttachedCollisionPrimitive.from_idl(plain, object_id="sim:x").tight_geometry is None


def test_the_budget_discloses_which_primitives_the_kernel_refined() -> None:
    """#264's ``has_stage2_hull`` lesson, on the payload side.

    ``corner_slop_m`` stays the right budget for an attached-payload SELF stop —
    that path is box-vs-box either way. Charging it to a *world-voxel* payload
    stop over-budgets a refined primitive by exactly what the refinement
    recovered, which is how a real defect hides behind a generous gap. So the
    count is published rather than left to be inferred from the round's date.
    """
    from openral_hal.sim_sensor_bridge import attached_payload_mesh_slop

    model, data = _compiled()
    slop = attached_payload_mesh_slop(
        model,
        data,
        attached_body_ids=frozenset({_body(model, "mesh_payload"), _body(model, "box_payload")}),
    )
    objects = slop["objects"]
    assert isinstance(objects, dict)
    assert objects["mesh_payload"]["n_stage2_primitives"] == 1
    assert objects["box_payload"]["n_stage2_primitives"] == 0
    # The box term itself is untouched: the self-collision budget it feeds is
    # measured against the same box the kernel still checks there.
    assert objects["mesh_payload"]["corner_slop_m"] == pytest.approx(_R * math.sqrt(2.0), abs=3e-4)
