"""A static link — a ``fixed_attachments`` child the MJCF has no body for — lowers from the URDF.

The OpenArm's safety model is lowered from its v2.0 bimanual MJCF, whose
worldbody hangs the two arm base links directly: there is no torso body, so the
kernel never saw the pedestal the hands can swing into (issue #356). The URDF
assembly carries it (``openarm_body_link0``: a 250 × 190 mm foot plate, a 60 mm
column, a shoulder block). This pins the three pieces of that path, on the real
OpenArm manifest, MJCF and URDF (CLAUDE.md §1.11):

* ``_fixed_link_world`` places a link the MJCF does not carry from the mounts
  it does carry, and refuses two mounts that disagree;
* ``_static_link_meshes`` reads a static link's collision **and** visual
  geometry through ``resolve_package_uri`` and fails closed on anything less;
* ``_split_tight_pieces`` cuts the torso into ``k`` box + hull slabs whose
  union holds the whole mesh, at cut planes a re-lower reproduces — because
  one hull of the torso (5.6× its volume, 0.12 m past the column) would stop
  the arms across the bimanual workspace.

The pinned ``openarm_description`` clone is fetched on first use; a host that
cannot fetch it skips the reader and split tests rather than faking a mesh.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("yourdfpy")
pytest.importorskip("trimesh")
pytest.importorskip("scipy")

import trimesh
from openral_core import RobotDescription
from openral_core.assets import AssetFetchError, AssetRefError, resolve_package_uri
from openral_core.exceptions import ROSConfigError
from openral_core.schemas import (
    MAX_TIGHT_HULL_VERTICES,
    TIGHT_CONTAINMENT_EPSILON_M,
    BoxShape,
    FixedAttachment,
    LinkCollisionGeometry,
)
from openral_safety.urdf_lowering import (
    _best_cuts,
    _fixed_link_world,
    _split_tight_pieces,
    _static_link_meshes,
    _tight_links_of,
    _xyzrpy_matrix,
    fit_link_with_tight_geometry,
)

from tests.unit._collision_kernel_verdict import surface_pieces, tight_escape_m

_REPO = Path(__file__).resolve().parents[2]
_OPENARM_DIR = _REPO / "robots" / "openarm"
_TORSO = "openarm_body_link0"
# URDF: world -> body_link0 identity, body_link0 -> *_base_link z = 0.698; the
# manifest's root `openarm_base` is the arm-mount frame (robot.yaml assets comment).
_TORSO_IN_BASE = (0.0, 0.0, -0.698)


def _openarm() -> RobotDescription:
    return RobotDescription.from_yaml(str(_OPENARM_DIR / "robot.yaml"))


def _with_torso_mount(robot: RobotDescription) -> RobotDescription:
    """The manifest plus the torso row, whether or not the committed one carries it yet."""
    if any(a.child_link == _TORSO for a in robot.fixed_attachments):
        return robot
    mount = FixedAttachment(
        name="openarm_body_mount",
        parent_link="openarm_base",
        child_link=_TORSO,
        origin_xyz=_TORSO_IN_BASE,
        origin_rpy=(0.0, 0.0, 0.0),
    )
    return robot.model_copy(update={"fixed_attachments": [*robot.fixed_attachments, mount]})


@pytest.fixture(scope="module")
def torso_mesh() -> trimesh.Trimesh:
    """The torso's collision + visual geometry in the link frame, via the static-link reader."""
    robot = _with_torso_mount(_openarm())
    # Every arm link maps to an MJCF body; only the torso is static.
    link_body = {
        a.child_link: i for i, a in enumerate(robot.fixed_attachments) if a.child_link != _TORSO
    }
    try:
        meshes = _static_link_meshes(robot, link_body, _OPENARM_DIR)
    except ROSConfigError as exc:
        if isinstance(exc.__cause__, AssetFetchError):
            pytest.skip(f"openarm_description clone unavailable on this host: {exc}")
        raise
    assert set(meshes) == {_TORSO}
    return meshes[_TORSO]


# ── resolve_package_uri ────────────────────────────────────────────────────────


def test_package_uri_grammar_is_enforced() -> None:
    with pytest.raises(AssetRefError, match="not a package"):
        resolve_package_uri("file:///tmp/x.stl")
    with pytest.raises(AssetRefError, match="malformed"):
        resolve_package_uri("package://openarm_description")
    with pytest.raises(AssetRefError, match="not one OpenRAL fetches"):
        resolve_package_uri("package://no_such_description/meshes/x.stl")


def test_the_torso_collision_mesh_resolves_inside_the_pinned_clone() -> None:
    uri = "package://openarm_description/assets/robot/openarm_v2.0/meshes/body/collision/body_link0_symp.stl"
    try:
        path = resolve_package_uri(uri)
    except AssetFetchError as exc:
        pytest.skip(f"openarm_description clone unavailable on this host: {exc}")
    assert path.is_file() and path.name == "body_link0_symp.stl"
    with pytest.raises(AssetRefError, match="is not a file"):
        resolve_package_uri(uri.replace("body_link0_symp", "no_such_mesh"))


# ── _fixed_link_world ──────────────────────────────────────────────────────────


def test_the_root_and_the_torso_are_placed_from_the_arm_mounts() -> None:
    mujoco = pytest.importorskip("mujoco")
    from openral_core.assets import resolve_asset
    from openral_safety.urdf_lowering import _mjcf_link_bodies

    robot = _with_torso_mount(_openarm())
    try:
        mjcf = resolve_asset(robot.assets.mjcf, "mjcf", manifest_dir=_OPENARM_DIR)  # type: ignore[arg-type]  # reason: openarm declares assets.mjcf
    except AssetRefError as exc:
        pytest.skip(f"OpenArm MJCF unavailable on this host: {exc}")
    model = mujoco.MjModel.from_xml_path(str(mjcf))
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    link_body = _mjcf_link_bodies(robot, model)
    assert _TORSO not in link_body and "openarm_base" not in link_body

    def body_tf(b: int) -> np.ndarray:
        tf = np.eye(4)
        tf[:3, :3] = data.xmat[b].reshape(3, 3)
        tf[:3, 3] = data.xpos[b]
        return tf

    placed = _fixed_link_world(robot, link_body, body_tf)
    assert set(placed) == {"openarm_base", _TORSO}
    # Both mounts agree the root is MuJoCo's world; the torso hangs 0.698 m under it.
    assert placed["openarm_base"] == pytest.approx(np.eye(4), abs=1e-9)
    assert placed[_TORSO] == pytest.approx(
        _xyzrpy_matrix((*_TORSO_IN_BASE, 0.0, 0.0, 0.0)), abs=1e-9
    )

    # A mount that contradicts the other is a manifest error, not a tolerance.
    skewed = robot.model_copy(
        update={
            "fixed_attachments": [
                a.model_copy(update={"origin_xyz": (0.0, 0.031, 0.005)})
                if a.child_link == "openarm_left_link0"
                else a
                for a in robot.fixed_attachments
            ]
        }
    )
    with pytest.raises(ROSConfigError, match="two ways"):
        _fixed_link_world(skewed, link_body, body_tf)


# ── _static_link_meshes ────────────────────────────────────────────────────────


def test_no_static_link_means_the_urdf_is_never_opened() -> None:
    """Every other MJCF robot lowers exactly as before: the reader returns early."""
    robot = _openarm()
    if any(a.child_link == _TORSO for a in robot.fixed_attachments):
        robot = robot.model_copy(
            update={
                "fixed_attachments": [a for a in robot.fixed_attachments if a.child_link != _TORSO]
            }
        )
    link_body = {a.child_link: i for i, a in enumerate(robot.fixed_attachments)}
    no_urdf = robot.model_copy(update={"assets": robot.assets.model_copy(update={"urdf": None})})
    assert _static_link_meshes(no_urdf, link_body, _OPENARM_DIR) == {}


def test_the_torso_reads_as_collision_plus_visual_geometry(torso_mesh: trimesh.Trimesh) -> None:
    lo, hi = torso_mesh.bounds
    # The plate and the shoulder block span the footprint; the column is in between.
    assert lo == pytest.approx((-0.155, -0.095, 0.0), abs=2e-3)
    assert hi == pytest.approx((0.095, 0.095, 0.773), abs=2e-3)
    # The visual CAD carries a fitting 22 mm behind the simplified collision
    # column (x to -0.052 at z 0.18-0.20, where the collision mesh stops at
    # -0.030); the union keeps it, so the kernel holds the real part.
    v = np.asarray(torso_mesh.vertices)
    fitting = v[(v[:, 2] > 0.15) & (v[:, 2] < 0.25) & (v[:, 0] < -0.045)]
    assert len(fitting) > 0, "the visual geometry was not folded into the static link"


def test_a_static_link_absent_from_the_urdf_or_half_loaded_fails_closed(
    tmp_path: Path,
) -> None:
    robot = _with_torso_mount(_openarm())
    link_body = {
        a.child_link: i for i, a in enumerate(robot.fixed_attachments) if a.child_link != _TORSO
    }
    # The torso counted as MJCF-mapped here, so the ghost is the only static link
    # and the refusal needs no mesh fetch.
    ghost_body = {**link_body, _TORSO: -1}
    ghost = robot.model_copy(
        update={
            "fixed_attachments": [
                *robot.fixed_attachments,
                FixedAttachment(
                    name="ghost_mount",
                    parent_link="openarm_base",
                    child_link="openarm_no_such_link",
                    origin_xyz=(0.0, 0.0, 0.0),
                    origin_rpy=(0.0, 0.0, 0.0),
                ),
            ]
        }
    )
    with pytest.raises(ROSConfigError, match="openarm_no_such_link"):
        _static_link_meshes(ghost, ghost_body, _OPENARM_DIR)
    # A URDF whose torso mesh ref cannot resolve raises instead of warning.
    broken = (_OPENARM_DIR / "openarm.urdf").read_text(encoding="utf-8")
    broken = broken.replace("body_link0_symp.stl", "no_such_mesh.stl")
    assert "no_such_mesh.stl" in broken
    (tmp_path / "broken.urdf").write_text(broken, encoding="utf-8")
    urdf_asset = robot.assets.urdf.model_copy(update={"ref": "file:broken.urdf"})  # type: ignore[union-attr]  # reason: openarm declares assets.urdf
    half = robot.model_copy(update={"assets": robot.assets.model_copy(update={"urdf": urdf_asset})})
    with pytest.raises(ROSConfigError, match="no_such_mesh") as excinfo:
        try:
            _static_link_meshes(half, link_body, tmp_path)
        except ROSConfigError as exc:
            if isinstance(exc.__cause__, AssetFetchError):
                pytest.skip(f"openarm_description clone unavailable on this host: {exc}")
            raise
    assert not isinstance(excinfo.value.__cause__, AssetFetchError)


# ── _split_tight_pieces ────────────────────────────────────────────────────────


def test_the_cut_search_finds_the_cut_trying_every_cut_finds() -> None:
    """``_best_cuts`` (a dynamic program) returns the minimum and the planes of brute force.

    Pure: the slab volumes are seeded random numbers, so this is the search
    alone, against ``itertools.combinations`` over every cut.
    """
    import itertools

    rng = np.random.default_rng(356)
    for _ in range(300):
        n = int(rng.integers(1, 10))
        k = int(rng.integers(2, min(4, n + 1) + 1))
        vol = {(a, b): float(rng.random()) for a in range(-1, n) for b in range(a + 1, n + 1)}
        brute = min(
            (sum(vol[ab] for ab in itertools.pairwise((-1, *cut, n))), list(cut))
            for cut in itertools.combinations(range(n), k - 1)
        )
        total, cut = _best_cuts(lambda a, b, vol=vol: vol[(a, b)], n, k)
        assert total == pytest.approx(brute[0], abs=1e-12)
        assert cut == brute[1]


@pytest.fixture(scope="module")
def torso_pieces(torso_mesh: trimesh.Trimesh) -> list[LinkCollisionGeometry]:
    return _split_tight_pieces(_TORSO, torso_mesh, 3)


def _box_frame(g: LinkCollisionGeometry, pts: np.ndarray) -> np.ndarray:
    tf = _xyzrpy_matrix(g.origin_xyz_rpy)
    return (pts - tf[:3, 3]) @ tf[:3, :3]


def test_three_slabs_hold_the_torso_and_cut_where_the_shape_changes(
    torso_mesh: trimesh.Trimesh, torso_pieces: list[LinkCollisionGeometry]
) -> None:
    pieces = torso_pieces
    assert len(pieces) == 3
    pts = np.asarray(torso_mesh.vertices, dtype=float)
    for g in pieces:
        assert g.link_name == _TORSO and isinstance(g.shape, BoxShape)
        tight = g.tight_geometry
        assert tight is not None and tight.hull_vertices_m, "every slab ships a stage-2 hull"
        assert len(tight.hull_vertices_m) <= MAX_TIGHT_HULL_VERTICES
        assert tight.hull_overhang_m is not None
    # The union of the three boxes holds the whole surface (a triangle across a
    # cut could leave the union with all corners inside, so the surface, not the vertices).
    _, uncovered = surface_pieces(pts, np.asarray(torso_mesh.faces), pieces)
    assert uncovered == 0
    # Each slab's own vertices sit inside its DOP and hull: the piece the kernel
    # refines a box with is the piece's exact hull.
    centres = np.array([g.origin_xyz_rpy[2] for g in pieces])
    order = np.argsort(centres)
    low, mid, high = (pieces[i] for i in order)
    z = pts[:, 2]
    for g, mask in ((low, z <= 0.07), (mid, (z >= 0.07) & (z <= 0.61)), (high, z >= 0.61)):
        assert tight_escape_m(g, _box_frame(g, pts[mask])) <= TIGHT_CONTAINMENT_EPSILON_M
    # The cuts land where the cross-section changes (plate | column | shoulder
    # block), not at equal thirds: the middle slab is the long, 60 mm thin
    # column (plus the flange at its foot), the other two are the squat ends.
    assert isinstance(mid.shape, BoxShape)
    assert min(mid.shape.half_extents_m) < 0.035, mid.shape
    assert max(mid.shape.half_extents_m) > 0.25, mid.shape
    for end in (low, high):
        assert isinstance(end.shape, BoxShape)
        assert max(end.shape.half_extents_m) < 0.15, end.shape
    # And the split is the point: three hulls are far smaller than the one hull.
    whole = fit_link_with_tight_geometry(_TORSO, torso_mesh)
    whole_volume = trimesh.convex.convex_hull(
        np.asarray(whole.tight_geometry.hull_vertices_m)
    ).volume  # type: ignore[union-attr]  # reason: a refined link ships a hull
    split_volume = sum(
        trimesh.convex.convex_hull(np.asarray(g.tight_geometry.hull_vertices_m)).volume  # type: ignore[union-attr]  # reason: asserted above
        for g in pieces
    )
    assert split_volume < 0.5 * whole_volume


def test_the_cut_planes_are_reproducible_and_one_slab_is_the_plain_refinement(
    torso_mesh: trimesh.Trimesh, torso_pieces: list[LinkCollisionGeometry]
) -> None:
    assert _split_tight_pieces(_TORSO, torso_mesh, 3) == torso_pieces
    assert _split_tight_pieces(_TORSO, torso_mesh, 1) == [
        fit_link_with_tight_geometry(_TORSO, torso_mesh)
    ]
    with pytest.raises(ROSConfigError, match="cannot cut"):
        _split_tight_pieces(_TORSO, torso_mesh, 200)


def test_a_split_link_stays_split_on_a_relower_and_an_extra_count_never_shrinks_it(
    torso_pieces: list[LinkCollisionGeometry],
) -> None:
    robot = _openarm().model_copy(update={"collision_geometry": torso_pieces})
    assert _tight_links_of(robot, None) == {_TORSO: 3}
    assert _tight_links_of(robot, (_TORSO,)) == {_TORSO: 3}
    assert _tight_links_of(robot, {_TORSO: 2}) == {_TORSO: 3}
    assert _tight_links_of(robot, {_TORSO: 4, "openarm_left_link0": 1}) == {
        _TORSO: 4,
        "openarm_left_link0": 1,
    }
