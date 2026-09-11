# SPDX-License-Identifier: Apache-2.0
"""The payload side of the adjudication budget — the other OBB in a self stop.

The 2026-08-22 ``baguette`` round logged an attached-payload SELF stop
(``attached:sim:obj_main`` vs ``panda_link2``) at -4.63 mm, while the
ground-truth probe put the nearest payload mesh 75.86 mm from that link —
looking like a misattributed defect (the class ``fold_pair`` guards against).
It wasn't: reproducing the kernel's own arithmetic puts the pair at
+21.71 mm, ``panda_link2`` genuinely nearest (``panda_link1`` runner-up,
25.5 mm further; pinned in the kernel's own ``BaguettePayloadSelfStop``
gtests). The gap is representation, not error.

The budget only covered the world-voxel case (``corner_slop(link) +
voxel_half_diagonal``). An attached-payload self stop has no voxel and an OBB
on both sides, so the payload's own corner slop applies instead; charging
only the link's share under-counts the admissible gap.

``attached_payload_mesh_slop`` is that missing term, pinned here.

Real compiled MuJoCo models throughout, no mocks (CLAUDE.md §1.11).
"""

from __future__ import annotations

import math
from typing import Any

import pytest
from openral_hal.sim_sensor_bridge import (
    attached_payload_mesh_slop,
    estop_ground_truth_snapshot,
)

mujoco = pytest.importorskip("mujoco")

# `mesh_payload`: octahedron with vertices at ±_R on each axis.
# `extract_body_primitives` lowers a MESH geom to its enclosing AABB (cube),
# loose by an exact amount: cube corner (h,h,h) is `h*sqrt(2)` from vertex
# (h,0,0).
# `box_payload`: a real box geom lowers EXACTLY — proves slop is charged only
# where the producer actually creates it, not a uniform inflation.
_R = 0.03
_OCTAHEDRON_VERTS = f"{_R} 0 0  {-_R} 0 0  0 {_R} 0  0 {-_R} 0  0 0 {_R}  0 0 {-_R}"
_TETRA_VERTS = f"{_R} 0 0  0 {_R} 0  0 0 {_R}  {-_R * 0.2} {-_R * 0.2} {-_R * 0.2}"
_MJCF = f"""
<mujoco model="estop_payload_slop">
  <option gravity="0 0 0"/>
  <asset>
    <mesh name="octahedron" vertex="{_OCTAHEDRON_VERTS}"/>
    <!-- A SKEW tetrahedron, deliberately: a regular one inscribed in the cube
         has its four faces on the DOP's own corner axes, so its refinement is
         exact and it would test the same thing the octahedron already does.
         This one's faces lie on no DOP axis, so the tangent slabs bound it
         strictly loosely — a real positive overhang, strictly inside the
         box's corner slop, which is the case the budget exists for. -->
    <mesh name="tetra" vertex="{_TETRA_VERTS}"/>
  </asset>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1"/>
    <body name="robot0_base" pos="0 0 0.5">
      <joint name="robot0_joint1" type="hinge" axis="0 0 1"/>
      <geom name="robot0_link1_collision" type="capsule" fromto="0 0 0 0.3 0 0" size="0.04"/>
    </body>
    <body name="mesh_payload" pos="0.5 0 0.5">
      <freejoint name="mesh_payload_free"/>
      <geom name="mesh_payload_g0" type="mesh" mesh="octahedron"/>
    </body>
    <body name="tetra_payload" pos="0.5 -0.4 0.5">
      <freejoint name="tetra_payload_free"/>
      <geom name="tetra_payload_g0" type="mesh" mesh="tetra"/>
    </body>
    <body name="box_payload" pos="0.5 0.4 0.5">
      <freejoint name="box_payload_free"/>
      <geom name="box_payload_g0" type="box" size="0.02 0.03 0.04"/>
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


def _robot_bodies(model: Any) -> frozenset[int]:
    return frozenset({_body(model, "robot0_base")})


def test_a_mesh_payload_is_charged_its_aabb_corner_slop() -> None:
    """A mesh geom lowers to its AABB, and the corner term is that box's own.

    This is the payload half of the ``baguette`` gap. The producer publishes
    the enclosing box, the kernel checks that box, and the ground-truth probe
    measures the mesh inside it — so the two differ by exactly this, before
    the robot link's corner slop is even counted.
    """
    model, data = _compiled()
    slop = attached_payload_mesh_slop(
        model, data, attached_body_ids=frozenset({_body(model, "mesh_payload")})
    )

    entry = slop["objects"]["mesh_payload"]  # type: ignore[index]
    assert entry["n_primitives"] == 1  # type: ignore[index]
    assert entry["n_box_primitives"] == 1, "a mesh geom lowers to a box"  # type: ignore[index]
    assert entry["collision_points_sampled"] == 6  # type: ignore[index]
    # The AABB half-extent is _R + 1e-4 (the producer's padding); the corner is
    # that times sqrt(2) from the nearest octahedron vertex.
    expected = (_R + 1e-4) * math.sqrt(2.0)
    assert entry["corner_slop_m"] == pytest.approx(expected, abs=5e-5)  # type: ignore[index]
    assert float(slop["max_corner_slop_m"]) == pytest.approx(expected, abs=5e-5)  # type: ignore[arg-type]
    assert slop["unresolved_objects"] == []


def test_an_exactly_lowered_payload_is_charged_nothing() -> None:
    """A box geom lowers exactly, so it contributes no budget.

    The measurement has to track what the producer actually publishes. A
    uniform payload inflation would hand every self stop an unearned excuse,
    which is the opposite of what a budget is for.
    """
    model, data = _compiled()
    slop = attached_payload_mesh_slop(
        model, data, attached_body_ids=frozenset({_body(model, "box_payload")})
    )

    entry = slop["objects"]["box_payload"]  # type: ignore[index]
    assert entry["corner_slop_m"] == pytest.approx(0.0, abs=1e-9)  # type: ignore[index]
    assert float(slop["max_corner_slop_m"]) == pytest.approx(0.0, abs=1e-9)  # type: ignore[arg-type]


def test_nothing_carried_reports_no_budget_rather_than_zero() -> None:
    """An empty dict, not ``0.0`` — "no payload" is not "a payload with no slop"."""
    model, data = _compiled()
    assert attached_payload_mesh_slop(model, data, attached_body_ids=frozenset()) == {}


def test_the_self_collision_budget_rides_with_a_payload_stop() -> None:
    """``adjudication_budget.self_collision`` is what a payload stop needs.

    Distinct from the world-voxel block beside it: no voxel term (there is no
    voxel), and a payload term (there is a second OBB). Without it the reader
    of a payload stop has only the link's share and reaches the wrong verdict.
    """
    model, data = _compiled()
    snapshot = estop_ground_truth_snapshot(
        model,
        data,
        robot_body_ids=_robot_bodies(model),
        attached_body_ids=frozenset({_body(model, "mesh_payload")}),
        base_frame_body="robot0_base",
    )

    assert snapshot["stop_class"] == "attached_payload"
    budget = snapshot["adjudication_budget"]
    assert isinstance(budget, dict)
    self_block = budget["self_collision"]
    assert isinstance(self_block, dict)
    # The payload's share is real and is carried per object.
    payload_term = float(self_block["max_payload_corner_slop_m"])  # type: ignore[arg-type]
    assert payload_term == pytest.approx((_R + 1e-4) * math.sqrt(2.0), abs=5e-5)
    assert self_block["payload_slop"]["objects"]["mesh_payload"]  # type: ignore[index]
    # No voxel term: the world-voxel half is a separate block, and mixing them
    # is precisely the misreading this exists to stop.
    assert "voxel_half_diagonal_m" not in self_block
    # No description passed, so the link's share is unknown rather than zero —
    # the budget says "no link term" instead of quietly claiming there is none.
    assert self_block["max_link_corner_slop_m"] is None
    assert self_block["admissible_gap_m"] == pytest.approx(payload_term, abs=5e-5)


def test_a_payloadless_stop_carries_no_self_collision_block() -> None:
    """Nothing carried, nothing to adjudicate as a self stop."""
    model, data = _compiled()
    snapshot = estop_ground_truth_snapshot(
        model,
        data,
        robot_body_ids=_robot_bodies(model),
        base_frame_body="robot0_base",
    )

    budget = snapshot["adjudication_budget"]
    assert isinstance(budget, dict)
    assert budget["self_collision"] is None


def test_the_probe_window_widens_to_the_self_collision_budget() -> None:
    """A payload term wider than the default window must widen the probe.

    Same rule as the world-voxel side: widen only, never narrow. A payload
    whose primitives reach further than the probe looks would otherwise put
    the geometry backing its own stop outside the record.
    """
    model, data = _compiled()
    snapshot = estop_ground_truth_snapshot(
        model,
        data,
        robot_body_ids=_robot_bodies(model),
        attached_body_ids=frozenset({_body(model, "mesh_payload")}),
        base_frame_body="robot0_base",
        distmax_m=0.001,  # far below the payload's own corner term
    )

    budget = snapshot["adjudication_budget"]
    assert isinstance(budget, dict)
    used = float(budget["probe_distmax_used_m"])  # type: ignore[arg-type]
    assert used == pytest.approx((_R + 1e-4) * math.sqrt(2.0), abs=5e-5)
    assert used > 0.001, "the window widened to the budget, it did not narrow"


def test_the_world_voxel_budget_is_the_models_overhang_not_the_boxs() -> None:
    """#266's missing budget. A payload-vs-world stop has no robot link in it.

    Until this, ``hal_admissible_gap_m`` routed that class to the top-level
    block — worst LINK corner slop plus the cell half-diagonal — a budget for a
    pair the stop does not involve, and that class is **97 %** of the
    2026-09-10 A/B's 15 mm stops and 79 % of its 25 mm ones.

    The right term is how far the solid the kernel *actually checks* reaches
    past the real mesh: the refinement's DOP where one ships, the box
    otherwise. Both are convex, so the maximum of a convex distance over them
    is at a vertex — the box's eight corners, or the DOP's.

    The octahedron pins it exactly: it **is** a 26-DOP (its eight faces are the
    DOP's four corner axes, both signs), so the refinement bounds it with zero
    slack while its AABB stands ``R*sqrt(2)`` proud at the corners.
    """
    model, data = _compiled()
    slop = attached_payload_mesh_slop(
        model,
        data,
        attached_body_ids=frozenset(
            {_body(model, n) for n in ("mesh_payload", "tetra_payload", "box_payload")}
        ),
    )
    objects = slop["objects"]
    assert isinstance(objects, dict)

    octa = objects["mesh_payload"]
    assert octa["corner_slop_m"] == pytest.approx(_R * math.sqrt(2.0), abs=3e-4)
    assert octa["model_overhang_m"] == pytest.approx(0.0, abs=1e-9), (
        "an octahedron is exactly a 26-DOP; the refinement overhangs it by nothing"
    )

    # The skew tetrahedron is not a DOP, so its overhang is real — 18.25 mm
    # against the box's 32.55 mm. Charging the box would hand this stop 14 mm
    # of budget the kernel's own model does not use, which is how a real defect
    # hides behind a generous gap.
    tetra = objects["tetra_payload"]
    assert tetra["model_overhang_m"] == pytest.approx(0.018249, abs=1e-5)
    assert tetra["corner_slop_m"] == pytest.approx(0.032545, abs=1e-5)
    assert 0.0 < tetra["model_overhang_m"] < tetra["corner_slop_m"]

    # A box geom lowers exactly and carries no refinement, so the two terms
    # agree: the model IS the box, and the budget does not change for it.
    box = objects["box_payload"]
    assert box["n_stage2_primitives"] == 0
    assert box["model_overhang_m"] == pytest.approx(box["corner_slop_m"], abs=1e-9)

    assert slop["max_model_overhang_m"] == pytest.approx(
        max(float(o["model_overhang_m"]) for o in objects.values()),  # type: ignore[index]
        abs=1e-9,
    )


def test_the_payload_world_budget_rides_with_a_payload_stop() -> None:
    """The block itself, composed and carried on the snapshot.

    One model and one voxel, so the composition is the payload's own overhang
    plus the cell half-diagonal — never the link term beside it, and never
    ``max_payload_corner_slop_m``, which is the box's and belongs to the
    self-collision block.
    """
    model, data = _compiled()
    snapshot = estop_ground_truth_snapshot(
        model,
        data,
        robot_body_ids=_robot_bodies(model),
        attached_body_ids=frozenset({_body(model, "tetra_payload")}),
        base_frame_body="robot0_base",
        description=None,
    )
    budget = snapshot["adjudication_budget"]
    assert isinstance(budget, dict)
    block = budget["payload_world_voxel"]
    assert isinstance(block, dict)
    overhang = float(block["max_payload_model_overhang_m"])  # type: ignore[arg-type]
    half = float(block["voxel_half_diagonal_m"])  # type: ignore[arg-type]
    assert block["admissible_gap_m"] == pytest.approx(overhang + half, abs=1e-6)
    # Strictly tighter than charging the box would have been.
    payload_slop = block["payload_slop"]
    assert isinstance(payload_slop, dict)
    assert overhang < float(payload_slop["max_corner_slop_m"])  # type: ignore[arg-type]


def test_a_payloadless_stop_carries_no_payload_world_block() -> None:
    """Nothing carried, nothing to charge a payload budget to."""
    model, data = _compiled()
    snapshot = estop_ground_truth_snapshot(
        model,
        data,
        robot_body_ids=_robot_bodies(model),
        base_frame_body="robot0_base",
    )
    budget = snapshot["adjudication_budget"]
    assert isinstance(budget, dict)
    assert budget["payload_world_voxel"] is None
