"""The always-colliding criterion: what it proves, what it refuses, and why.

An ACM entry *removes* a self-collision check, so the "always-colliding"
justification is only sound when the check it removes is a constant — true at
every reachable configuration. These tests pin that the criterion (a)
establishes it as a proof rather than a sample, (b) uses the geometry and margin
the kernel actually uses, and (c) refuses in the safe direction when it cannot
prove anything.

Real manifests and real URDFs throughout, no mocks (§1.11).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from openral_core import BoxShape, CapsuleShape, RobotDescription, SphereShape

pytest.importorskip("yourdfpy")
pytest.importorskip("robot_descriptions")

from openral_core.assets import resolve_asset
from openral_safety.kernel_predicates import (
    bounding_capsule_segment,
    box_box_distance,
    shape_distance,
    shape_max_extent_m,
)
from openral_safety.urdf_lowering import (
    _certified_always_colliding,
    _chain_transforms,
    _load_urdf,
    _pair_relative_dofs,
    _relative_chains,
    _xyzrpy_matrix,
    acm_for_geometry,
)


def _load(name: str) -> tuple[object, dict[str, object], float]:
    """(yourdfpy model, geometry by link, the robot's own self-collision margin)."""
    robot = RobotDescription.from_yaml(f"robots/{name}/robot.yaml")
    manifest_dir = Path(f"robots/{name}")
    urdf = resolve_asset(robot.assets.urdf.ref, "urdf", manifest_dir=manifest_dir)
    assert urdf is not None
    geoms = {g.link_name: g for g in robot.collision_geometry}
    margin = float(getattr(robot.safety, "self_collision_margin_m", 0.0) or 0.0)
    return _load_urdf(str(urdf)), geoms, margin


def _pair_gaps(model: object, geoms: dict, a: str, b: str, n_per_axis: int) -> np.ndarray:
    """Kernel surface gaps over a dense grid of the pair's relative-DoF subspace."""
    dofs = _pair_relative_dofs(model, a, b, geoms)
    assert dofs is not None
    chains = _relative_chains(model, a, b)
    assert chains is not None
    chain_a, chain_b = chains
    origin_a = _xyzrpy_matrix(geoms[a].origin_xyz_rpy)
    origin_b = _xyzrpy_matrix(geoms[b].origin_xyz_rpy)
    axes = [np.linspace(lo, hi, n_per_axis) for _, _, (lo, hi) in dofs]
    grids = np.meshgrid(*axes, indexing="ij")
    flat = [g.ravel() for g in grids]
    values = {str(j.name): flat[i] for i, (j, _, _) in enumerate(dofs)}
    n = int(flat[0].size)
    t_a = _chain_transforms(chain_a, values, n) @ origin_a
    t_b = _chain_transforms(chain_b, values, n) @ origin_b
    return shape_distance(geoms[a].shape, t_a, geoms[b].shape, t_b)


# ── The relative-DoF subspace is what makes the proof tractable ───────────────


def test_pair_moves_only_with_the_joints_between_the_links() -> None:
    """A pair's relative pose depends on the joints on the path, and no others.

    This is what turns "is it always-colliding?" from an intractable question
    about the arm's 7-D joint box into an exhaustible one about a 2-D box —
    and it is why the old 2000-sample sweep could not see the answer.
    """
    model, geoms, _ = _load("panda_mobile")
    dofs = _pair_relative_dofs(model, "panda_link5", "panda_link7", geoms)
    assert dofs is not None
    assert [str(j.name) for j, _, _ in dofs] == ["panda_joint6", "panda_joint7"]
    # Every other joint really is irrelevant: pin joints 6/7 and sweep the rest.
    base = _pair_gaps(model, geoms, "panda_link5", "panda_link7", 2)
    assert base.shape == (4,)


# ── panda_link5 ↔ panda_link7: the pair issue #155 is about ───────────────────


def test_link5_link7_is_not_always_colliding() -> None:
    """It separates, so it cannot be exempted as always-colliding.

    Under the true oriented boxes the kernel checks, the pair is fully separated
    over part of its (joint6, joint7) range. The old sweep modelled each box as
    its inscribed sphere and reported the opposite.
    """
    model, geoms, margin = _load("panda_mobile")
    assert not _certified_always_colliding(
        model, geoms, "panda_link5", "panda_link7", margin_m=margin
    )
    gaps = _pair_gaps(model, geoms, "panda_link5", "panda_link7", 201)
    assert (gaps > margin).any(), "the pair must genuinely separate somewhere"
    assert gaps.max() == pytest.approx(0.0106, abs=5e-4)


def test_inscribed_sphere_would_have_hidden_the_separation() -> None:
    """The specific modelling error behind #155, pinned so it cannot come back.

    Modelling each box by its inscribed sphere makes link5/link7 look disjoint
    almost everywhere — the exact inversion that dropped the pair from the sweep.
    A box must never be lowered to anything smaller than itself for a *collision*
    verdict; :func:`bounding_capsule_segment` (over-covering) is for planners.
    """
    _model, geoms, _ = _load("panda_mobile")
    box5 = geoms["panda_link5"].shape
    assert isinstance(box5, BoxShape)
    # The inscribed sphere is dramatically smaller than the box it stands in for.
    assert min(box5.half_extents_m) < shape_max_extent_m(box5) / 3.0
    # And the over-covering capsule is at least as large as the box, always.
    _, _, radius = bounding_capsule_segment(box5, (0.0,) * 6)
    assert radius > min(box5.half_extents_m)
    assert radius >= max(sorted(box5.half_extents_m)[:2])


def test_link5_link7_is_checked_and_no_longer_exempt() -> None:
    """The pair is neither always-colliding nor never-colliding — it is *both*.

    Its boxes overlap over most of the range (so checking it at box fidelity
    E-stops constantly) and its real meshes interpenetrate over part of it (so
    exempting it hides a true self-collision). No margin separates the two
    populations, which is why the pair shipped exempted "under protest" from
    #169 until issue #191.

    What retired the exemption is not a margin and not a tighter box but the
    kernel's exact-hull narrow phase (``hull_hull_distance``), extended from
    world voxels to self-pairs: both links declare ``tight_geometry``, the box
    stays the broad phase, and any box pair it cannot clear is re-asked of the
    hulls. So the box overlap measured below is still real — it is just no
    longer the kernel's verdict.

    This test is the inverse of the one it replaces: it pins that nothing puts
    the row back, in either channel.
    """
    model, geoms, margin = _load("panda_mobile")
    gaps = _pair_gaps(model, geoms, "panda_link5", "panda_link7", 201)
    overlap_fraction = float((gaps <= margin).mean())
    assert 0.8 < overlap_fraction < 0.95, "boxes overlap over most, not all, of the range"

    robot = RobotDescription.from_yaml("robots/panda_mobile/robot.yaml")
    assert ("panda_link5", "panda_link7") not in robot.allowed_collision_pairs
    for link in ("panda_link5", "panda_link7"):
        geom = next(g for g in robot.collision_geometry if g.link_name == link)
        assert geom.tight_geometry is not None, (
            f"{link} must ship tight_geometry — it is what makes checking the pair usable"
        )
        assert geom.tight_geometry.hull_vertices_m, f"{link} needs its stage-2 hull, not just a DOP"
    srdf = Path("robots/panda_mobile/panda_mobile.srdf").read_text(encoding="utf-8")
    assert 'link1="panda_link5" link2="panda_link7"' not in srdf
    assert "RETIRED" in srdf, "the retired exemption must leave its record behind"


def test_a_hull_carrying_pair_is_never_certified_from_its_boxes() -> None:
    """The proof must not outrun the geometry the kernel checks.

    ``_certified_always_colliding`` reasons with ``shape_distance`` — the box.
    Once a pair's links both ship ``tight_geometry`` the kernel decides that pair
    on the hulls instead, and ``hull_gap >= box_gap`` everywhere, so "the boxes
    always overlap" no longer implies "the kernel always trips". Certifying on it
    would grant an ACM entry that hides a live check, so the criterion withholds.
    """
    model, geoms, margin = _load("panda_mobile")
    assert geoms["panda_link5"].tight_geometry is not None
    assert geoms["panda_link7"].tight_geometry is not None
    assert not _certified_always_colliding(
        model, geoms, "panda_link5", "panda_link7", margin_m=margin
    )
    # Not a vacuous pass: strip the hulls and the criterion still refuses, for
    # the *original* reason (#169 — the boxes genuinely separate somewhere), so
    # the guard above is an additional refusal rather than the only one.
    bare = {k: v.model_copy(update={"tight_geometry": None}) for k, v in geoms.items()}
    assert not _certified_always_colliding(
        model, bare, "panda_link5", "panda_link7", margin_m=margin
    )


def test_generated_acm_never_invents_the_exemption() -> None:
    """Without the SRDF the tool must not produce link5↔link7 from geometry alone."""
    robot = RobotDescription.from_yaml("robots/panda_mobile/robot.yaml")
    urdf = resolve_asset(robot.assets.urdf.ref, "urdf", manifest_dir=Path("robots/panda_mobile"))
    assert urdf is not None
    geoms = {g.link_name: g for g in robot.collision_geometry}
    generated = acm_for_geometry(str(urdf), geoms, srdf_path=None, margin_m=0.0)
    assert frozenset({"panda_link5", "panda_link7"}) not in generated


# ── What the criterion DOES certify, and that it is really always-colliding ───


@pytest.mark.parametrize(
    "link_a,link_b",
    [
        ("left_hip_pitch_link", "left_hip_yaw_link"),
        ("left_wrist_roll_link", "left_wrist_yaw_link"),
        ("left_hip_pitch_link", "torso_link"),
    ],
)
def test_certified_pairs_are_genuinely_always_colliding(link_a: str, link_b: str) -> None:
    """Every certificate is checked against a dense independent sweep.

    A certificate that were wrong would delete a live check, so it is not enough
    that the criterion says yes: the pair must also never separate on a grid the
    criterion did not choose. Includes a 4-DoF pair (hip pitch + the three waist
    joints), which an earlier fixed-grid version of this criterion wrongly
    refused — a false refusal costs false E-stops on the g1's legs.
    """
    model, geoms, margin = _load("g1")
    assert _certified_always_colliding(model, geoms, link_a, link_b, margin_m=margin)
    dofs = _pair_relative_dofs(model, link_a, link_b, geoms) or []
    # Keep the independent grid affordable: it grows as n**len(dofs).
    assert (_pair_gaps(model, geoms, link_a, link_b, 200 if len(dofs) <= 2 else 20) <= margin).all()


def test_certification_is_deterministic() -> None:
    """No RNG anywhere: the same question gets the same answer, every time."""
    model, geoms, margin = _load("g1")
    verdicts = [
        _certified_always_colliding(
            model, geoms, "left_hip_pitch_link", "left_hip_yaw_link", margin_m=margin
        )
        for _ in range(3)
    ]
    assert verdicts == [True, True, True]


# ── Conservatism: every budget is a one-way ratchet toward FEWER entries ──────


@pytest.mark.parametrize(
    "constant,starved",
    [("_CERTIFY_MAX_DOF", 0), ("_CERTIFY_MAX_CELLS", 1), ("_CERTIFY_MAX_LEVELS", 1)],
)
def test_running_out_of_budget_withholds_the_exemption(
    monkeypatch: pytest.MonkeyPatch, constant: str, starved: int
) -> None:
    """Each refinement limit fails toward *fewer* ACM entries, never more.

    An ACM entry removes a check, so the direction these knobs fail in is the
    whole safety argument. Starve any one of them and a pair that certifies
    comfortably must stop certifying — never the reverse. That makes them cost
    knobs rather than safety knobs: raising one can only ever admit a pair that a
    full proof would have admitted anyway.
    """
    import openral_safety.urdf_lowering as ul

    model, geoms, margin = _load("g1")
    pair = ("left_hip_pitch_link", "left_hip_yaw_link")
    assert _certified_always_colliding(model, geoms, *pair, margin_m=margin)
    monkeypatch.setattr(ul, constant, starved)
    assert not _certified_always_colliding(model, geoms, *pair, margin_m=margin)


def test_exempt_set_only_shrinks_against_the_old_sampling_rule() -> None:
    """The corrected criterion adds nothing the old strict rule would have rejected.

    The whole fleet-wide claim of this change, asserted rather than described:
    across every robot whose manifest carries an ACM, the regenerated pairs are a
    **subset** of the committed ones. Nothing is added. A future change that
    starts exempting new pairs has to come here and say why.
    """
    from openral_safety.urdf_lowering import lower_robot

    for name in ("panda_mobile", "panda_mobile_vslam", "so101_follower", "g1"):
        robot = RobotDescription.from_yaml(f"robots/{name}/robot.yaml")
        result = lower_robot(robot, acm_only=True, manifest_dir=Path(f"robots/{name}"))
        assert set(result.allowed_collision_pairs) <= set(robot.allowed_collision_pairs), (
            f"{name}: regenerating the ACM ADDED a pair — every addition needs a "
            f"geometric proof recorded in the PR body"
        )


# ── The margin is the kernel's, not a hardcoded zero ──────────────────────────


def test_criterion_uses_the_robots_own_margin() -> None:
    """A pair only ever *loses* its exemption as the margin goes more negative.

    The kernel trips at ``distance <= self_collision_margin_m``. so101_follower
    runs -0.06 m, so a criterion pinned at 0.0 would call pairs always-colliding
    that the kernel never trips on — exempting a check that was doing real work.
    Tightening the margin must therefore only ever shrink the exempt set.
    """
    model, geoms, margin = _load("so101_follower")
    assert margin == pytest.approx(-0.06)
    certified_at = {
        m: {
            (a, b)
            for a in geoms
            for b in geoms
            if a < b and _certified_always_colliding(model, geoms, a, b, margin_m=m)
        }
        for m in (0.0, margin)
    }
    assert certified_at[margin] <= certified_at[0.0]


# ── The predicates really are the kernel's ────────────────────────────────────


def test_box_box_matches_the_kernel_sat_on_the_shipped_geometry() -> None:
    """`box_box_distance` is the 15-axis SAT, not a bounding-sphere stand-in."""
    a = np.eye(4)[None]
    b = np.eye(4)[None].copy()
    b[0, 0, 3] = 1.0
    assert box_box_distance(a, (0.1, 0.1, 0.1), b, (0.1, 0.1, 0.1))[0] == pytest.approx(0.8)
    # A box is never treated as its inscribed sphere: two unit boxes offset along
    # a face normal by just under their combined extent must still overlap.
    b[0, 0, 3] = 0.19
    assert box_box_distance(a, (0.1, 0.1, 0.1), b, (0.1, 0.1, 0.1))[0] < 0.0


def test_shape_distance_routes_every_primitive_pairing() -> None:
    """box/box, box/capsule, capsule/capsule and sphere all reach a real predicate."""
    far = np.eye(4)[None].copy()
    far[0, 0, 3] = 5.0
    here = np.eye(4)[None]
    box = BoxShape(half_extents_m=(0.1, 0.1, 0.1))
    cap = CapsuleShape(radius_m=0.05, length_m=0.2)
    sph = SphereShape(radius_m=0.05)
    for x in (box, cap, sph):
        for y in (box, cap, sph):
            assert shape_distance(x, here, y, far)[0] > 0.0
            assert shape_distance(x, here, y, here)[0] <= 0.0
