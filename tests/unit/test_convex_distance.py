"""The replacement distance instrument is exact, certified, and refuses when it cannot be.

``openral_hal.convex_distance`` exists because ``mujoco.mj_geomDistance`` is
unreliable for RoboCasa-fixture-vs-panda-mesh pairs under mujoco 3.8.0 (that
module's docstring carries the measurements). An instrument adopted for that
reason has to be held to a higher standard than the one it replaces, so this
file pins three separate things:

* **Accuracy** against distances that are known analytically — boxes at a
  stated offset, spheres, capsules, cylinders, and the penetrating branch —
  rather than against another implementation.
* **The certificate.** Every separated answer must close its own
  separating-axis duality gap; a bracketed round type must report a bracket
  that contains the truth; and an answer that cannot be certified must say so
  instead of being emitted (CLAUDE.md §1.4, no hidden fallbacks).
* **The witness validator**, which is the contradiction detector that exposed
  the original defect: a nearest-point segment whose endpoints lie outside
  both geoms cannot be a nearest pair.

No mocks (CLAUDE.md §1.11): real compiled ``MjModel``s throughout, and the
mesh case is the *real* panda link-7 collision mesh, read out of the robot's
own MJCF description rather than invented here.
"""

from __future__ import annotations

import math

import pytest

mujoco = pytest.importorskip("mujoco")
np = pytest.importorskip("numpy")

from openral_hal.convex_distance import (  # noqa: E402
    convex_geom_distance,
    mesh_hull,
    separating_axis_bound,
    witness_clearance_m,
)

_PRIMITIVES = """<mujoco>
 <worldbody>
  <body pos="0 0 0"><geom name="box_a" type="box" size="0.1 0.1 0.1"/></body>
  <body pos="0.35 0 0"><geom name="box_far" type="box" size="0.1 0.1 0.1"/></body>
  <body pos="0.5 0.5 0.5"><geom name="box_corner" type="box" size="0.1 0.1 0.1"/></body>
  <body pos="0.15 0 0"><geom name="box_overlap" type="box" size="0.1 0.1 0.1"/></body>
  <body pos="0 2 0"><geom name="sphere_a" type="sphere" size="0.05"/></body>
  <body pos="0 2 0.3"><geom name="sphere_b" type="sphere" size="0.07"/></body>
  <body pos="0 4 0"><geom name="capsule_a" type="capsule" size="0.03 0.1"/></body>
  <body pos="0 4.25 0"><geom name="capsule_b" type="capsule" size="0.04 0.1"/></body>
  <body pos="0 6 0"><geom name="cylinder_a" type="cylinder" size="0.05 0.1"/></body>
  <body pos="0 6.3 0"><geom name="cylinder_b" type="cylinder" size="0.06 0.1"/></body>
  <body pos="0 8 0"><geom name="plane_a" type="plane" size="1 1 0.1"/></body>
  <body pos="0 8 0.4"><geom name="box_over_plane" type="box" size="0.1 0.1 0.1"/></body>
 </worldbody>
</mujoco>"""


@pytest.fixture(scope="module")
def primitives() -> tuple[object, object]:
    model = mujoco.MjModel.from_xml_string(_PRIMITIVES)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def _geom(model: object, name: str) -> int:
    return int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name))


@pytest.mark.parametrize(
    ("name_a", "name_b", "truth_m"),
    [
        ("box_a", "box_far", 0.15),
        ("box_a", "box_corner", math.sqrt(3.0) * 0.3),
        ("box_a", "box_overlap", -0.05),
        ("sphere_a", "sphere_b", 0.3 - 0.05 - 0.07),
        ("capsule_a", "capsule_b", 0.25 - 0.03 - 0.04),
        ("cylinder_a", "cylinder_b", 0.3 - 0.05 - 0.06),
    ],
)
def test_distance_matches_the_analytic_truth(
    primitives: tuple[object, object], name_a: str, name_b: str, truth_m: float
) -> None:
    """Each answer is right to a tolerance the instrument itself declares.

    The tolerance is the reported bracket, not a number chosen here: a box or
    a sphere brackets exactly (width 0.0) and must match to floating point,
    while a cylinder is polygonised and must match to its own stated width.
    """
    model, data = primitives
    result = convex_geom_distance(model, data, _geom(model, name_a), _geom(model, name_b))

    assert result.certified, result.uncertified_reason
    assert result.lower_m <= result.upper_m
    tolerance = max(result.upper_m - result.lower_m, 1e-12)
    assert result.distance_m == pytest.approx(truth_m, abs=tolerance)
    assert result.lower_m - tolerance <= truth_m <= result.upper_m + tolerance


def test_the_separating_axis_certificate_closes(primitives: tuple[object, object]) -> None:
    """A separated answer is *proved*, not merely produced.

    ``min_B u·b - max_A u·a`` is a lower bound on the distance along any unit
    ``u``, and ``||p_b - p_a||`` is an upper bound. When the two meet, no other
    direction can do better — which is the whole reason this instrument can be
    trusted where ``mj_geomDistance`` could not.
    """
    model, data = primitives
    result = convex_geom_distance(model, data, _geom(model, "box_a"), _geom(model, "box_corner"))

    assert result.duality_gap_m < 1e-9
    assert result.witness_a is not None and result.witness_b is not None
    segment = float(np.linalg.norm(np.array(result.witness_b) - np.array(result.witness_a)))
    assert segment == pytest.approx(result.distance_m, abs=1e-12)


def test_a_witness_on_its_own_geom_has_no_clearance(primitives: tuple[object, object]) -> None:
    """Both endpoints of the reported segment lie on their own geoms.

    This is the property ``mj_geomDistance``'s native-CCD failure violated by
    half a metre while reporting ``+0.000000``.
    """
    model, data = primitives
    a, b = _geom(model, "box_a"), _geom(model, "box_far")
    result = convex_geom_distance(model, data, a, b)

    assert witness_clearance_m(model, data, a, result.witness_a) < 1e-9
    assert witness_clearance_m(model, data, b, result.witness_b) < 1e-9


def test_the_witness_validator_refutes_a_point_outside_the_geom(
    primitives: tuple[object, object],
) -> None:
    """A positive clearance is a *proof* the point is not on the geom.

    ``box_a`` is a 0.2 m cube at the origin; a point 0.5 m out along +x is
    0.4 m clear of it, and the validator must not under-report that — a
    contradiction detector that hedges detects nothing.
    """
    model, data = primitives
    clearance = witness_clearance_m(model, data, _geom(model, "box_a"), np.array([0.5, 0.0, 0.0]))

    assert clearance == pytest.approx(0.4, abs=1e-9)


def test_a_plane_is_refused_rather_than_guessed(primitives: tuple[object, object]) -> None:
    """A geom with no bounded convex hull yields no number at all.

    Fail closed (CLAUDE.md §1.4): the caller gets an uncertified result naming
    the geom type, not a plausible distance it cannot defend.
    """
    model, data = primitives
    result = convex_geom_distance(
        model, data, _geom(model, "box_over_plane"), _geom(model, "plane_a")
    )

    assert not result.certified
    assert result.method == "unsupported-geom-type"
    assert "mjGEOM_PLANE" in result.uncertified_reason
    assert "distance_bracket_m" in result.as_record()


def test_the_window_rejection_is_a_proof_not_a_heuristic(
    primitives: tuple[object, object],
) -> None:
    """A pair dropped for being outside ``distmax_m`` is provably outside it.

    The early-out is what makes an exact instrument affordable on a few
    hundred candidate pairs; it is only admissible because the bound it uses
    is valid for every direction, so it can never discard a pair that was
    actually inside the window.
    """
    model, data = primitives
    a, b = _geom(model, "box_a"), _geom(model, "box_corner")
    rejected = convex_geom_distance(model, data, a, b, distmax_m=0.1)
    solved = convex_geom_distance(model, data, a, b)

    assert rejected.method == "beyond-window"
    assert rejected.certified
    assert rejected.lower_m > 0.1
    assert rejected.lower_m <= solved.distance_m
    # And a pair genuinely inside the window is still solved.
    inside = convex_geom_distance(model, data, a, _geom(model, "box_far"), distmax_m=0.3)
    assert inside.method == "gjk"
    assert inside.distance_m == pytest.approx(0.15, abs=1e-12)


def test_the_record_is_strict_json_even_with_an_unbounded_side(
    primitives: tuple[object, object],
) -> None:
    """No bound is ever serialised as ``Infinity``.

    A stop record is read back as strict JSON; a bound that cannot be written
    is reported as ``None``, which is the same claim honestly made.
    """
    import json

    model, data = primitives
    record = convex_geom_distance(
        model, data, _geom(model, "box_over_plane"), _geom(model, "plane_a")
    ).as_record()

    assert json.loads(json.dumps(record))["distance_bracket_m"] == [None, None]


def test_the_bound_holds_for_every_direction_not_just_the_witness(
    primitives: tuple[object, object],
) -> None:
    """Weak duality, checked on directions the instrument did not choose.

    If some other direction beat the reported distance, the certificate would
    be worthless — so this samples the sphere and asserts none of them do.
    """
    model, data = primitives
    a, b = _geom(model, "box_a"), _geom(model, "box_corner")
    from openral_hal.convex_distance import geom_convex_bracket

    hull_a = geom_convex_bracket(model, data, a)
    hull_b = geom_convex_bracket(model, data, b)
    assert hull_a is not None and hull_b is not None
    result = convex_geom_distance(model, data, a, b)

    rng = np.random.default_rng(20260825)
    directions = rng.normal(size=(512, 3))
    for direction in directions:
        bound = separating_axis_bound(hull_a[1].core, hull_b[1].core, direction)
        assert bound <= result.distance_m + 1e-12


def test_the_mesh_hull_is_the_hull_mujoco_itself_collides() -> None:
    """The panda meshes, read through MuJoCo's own compiled hull graph.

    MuJoCo collides a mesh's convex hull, never its triangles, so measuring
    anything else would answer a different question than the one the safety
    kernel's own contacts answer. The graph is decoded rather than a hull
    recomputed — no computational-geometry dependency is added to the HAL —
    and this pins that the decode is right: the hull's vertices come from the
    mesh, and the hull *encloses* every raw vertex of it.
    """
    descriptions = pytest.importorskip("robot_descriptions.loaders.mujoco")
    try:
        model = descriptions.load_robot_description("panda_mj_description")
    except Exception as exc:  # reason: description fetch is a network dependency
        pytest.skip(f"panda_mj_description unavailable: {exc}")

    meshes = [
        m
        for m in range(model.nmesh)
        if int(model.mesh_graphadr[m]) >= 0 and int(model.mesh_vertnum[m]) > 4
    ]
    assert meshes, "the panda description carries no meshes with a compiled hull graph"

    for mesh_id in meshes:
        hull, faces = mesh_hull(model, mesh_id)
        assert faces is not None
        adr = int(model.mesh_vertadr[mesh_id])
        count = int(model.mesh_vertnum[mesh_id])
        raw = np.asarray(model.mesh_vert[adr : adr + count], dtype=np.float64).reshape(count, 3)

        assert len(hull) <= count
        assert {tuple(np.round(v, 12)) for v in hull} <= {tuple(np.round(v, 12)) for v in raw}

        # Orient each hull face outward against the hull's own centroid, then
        # assert containment: no raw vertex escapes any supporting halfspace.
        centroid = hull.mean(axis=0)
        tri = hull[faces]
        normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        lengths = np.linalg.norm(normals, axis=1)
        keep = lengths > 1e-12
        normals = normals[keep] / lengths[keep][:, None]
        offsets = np.einsum("fk,fk->f", normals, tri[keep][:, 0])
        outward = np.where(normals @ centroid - offsets > 0.0, -1.0, 1.0)
        normals, offsets = normals * outward[:, None], offsets * outward
        assert float((raw @ normals.T - offsets).max()) < 1e-6
