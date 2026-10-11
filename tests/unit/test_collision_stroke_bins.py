"""The OpenArm's stroke-binned fingers hold the real fingers at every opening each bin covers.

``finger_pair`` is the leader finger's link with the equality-coupled follower
finger folded in. The lowering splits it (``_fit_stroke_binned``): the leader
as one rigid box + hull, the follower as a box + whole-stroke hull carrying
``stroke_bins``, one hull per sub-range of the gripper joint. The kernel's
world-voxel check uses only the bins covering the joint interval of the step
it checks (``docs/reference/finger-stroke-geometry.md``), so the safety
argument is per bin: ``follower mesh at q ⊆ bin hull`` for EVERY q in the bin,
not only at the positions it was fitted to. Pinned here densely (20 positions
per bin, the fit used 5) against the real MJCF meshes, with the schema and
parameter contracts on the real manifest.

Real fixtures only (CLAUDE.md §1.11).
"""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco", reason="needs mujoco (sim group) for the MJCF meshes")
pytest.importorskip("trimesh")
pytest.importorskip("scipy")

from openral_core import RobotDescription  # noqa: E402
from openral_core.assets import AssetRefError, resolve_asset  # noqa: E402
from openral_core.schemas import LinkCollisionGeometry  # noqa: E402
from openral_safety.envelope_loader import collision_params_from_description  # noqa: E402
from openral_safety.urdf_lowering import (  # noqa: E402
    _body_mesh,
    _follower_meshes_at,
    _mjcf_link_bodies,
    _xyzrpy_matrix,
    mjcf_coupled_joints,
)
from pydantic import ValidationError  # noqa: E402

from tests.unit._collision_kernel_verdict import tight_escape_per_point_m  # noqa: E402

_MANIFEST = Path(__file__).resolve().parents[2] / "robots" / "openarm" / "robot.yaml"
_FINGERS = ("openarm_left_finger_pair", "openarm_right_finger_pair")
# Positions per bin the dense check samples; the fit used 5.
_DENSE = 20
# A swept link may poke out between fitted samples by this much (the enclosure
# test's tolerance for swept links); measured here, it is ~1e-5 m.
_SWEPT_TOLERANCE_M = 1e-3
# A rigid hull is exact: only the 1 nm write rounding separates it from its mesh.
_RIGID_TOLERANCE_M = 1e-6


@pytest.fixture(scope="module")
def openarm() -> tuple[RobotDescription, object, object, dict[str, int], dict]:
    robot = RobotDescription.from_yaml(str(_MANIFEST))
    try:
        mjcf = resolve_asset(robot.assets.mjcf, "mjcf", manifest_dir=_MANIFEST.parent)
    except AssetRefError as exc:
        pytest.skip(f"openarm MJCF unavailable on this host: {exc}")
    model = mujoco.MjModel.from_xml_path(str(mjcf))
    data = mujoco.MjData(model)
    return robot, model, data, _mjcf_link_bodies(robot, model), mjcf_coupled_joints(model)


def _pieces(
    robot: RobotDescription, link: str
) -> tuple[LinkCollisionGeometry, LinkCollisionGeometry]:
    pieces = [g for g in robot.collision_geometry if g.link_name == link]
    rigid = [g for g in pieces if not g.stroke_bins]
    binned = [g for g in pieces if g.stroke_bins]
    assert len(rigid) == 1 and len(binned) == 1, f"{link}: expected leader + binned follower"
    return rigid[0], binned[0]


def _in_box(geom: LinkCollisionGeometry, points: np.ndarray) -> np.ndarray:
    tf = _xyzrpy_matrix(geom.origin_xyz_rpy)
    return (points - tf[:3, 3]) @ tf[:3, :3]


@pytest.mark.parametrize("link", _FINGERS)
def test_every_bin_holds_the_follower_at_every_opening_it_covers(openarm, link: str) -> None:
    robot, model, data, link_body, coupled = openarm
    _, follower = _pieces(robot, link)
    joint = next(j for j in robot.joints if j.child_link == link)
    assert follower.stroke_joint == joint.name and joint.role == "gripper"
    leader = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint.sim_joint_name))
    worst = -np.inf
    for b in follower.stroke_bins:
        as_bin = follower.model_copy(update={"tight_geometry": b.tight_geometry})
        for q in np.linspace(b.q_lo, b.q_hi, _DENSE):
            meshes = _follower_meshes_at(
                model, data, link_body[link], leader, coupled[leader], float(q)
            )
            pts = np.vstack([np.asarray(m.vertices) for m in meshes])
            local = _in_box(follower, pts)
            worst = max(worst, float(tight_escape_per_point_m(as_bin, local).max()))
            # The whole-stroke hull (unknown gripper, self-collision) holds it too.
            assert float(tight_escape_per_point_m(follower, local).max()) <= _SWEPT_TOLERANCE_M
    assert worst <= _SWEPT_TOLERANCE_M, f"{link}: a bin misses its follower by {worst * 1e3:.4f} mm"


@pytest.mark.parametrize("link", _FINGERS)
def test_the_leader_finger_is_one_rigid_exact_hull(openarm, link: str) -> None:
    robot, model, data, link_body, _ = openarm
    rigid, _ = _pieces(robot, link)
    mujoco.mj_resetData(model, data)
    mujoco.mj_kinematics(model, data)
    mesh = _body_mesh(model, data, link_body[link])
    local = _in_box(rigid, np.asarray(mesh.vertices))
    assert float(tight_escape_per_point_m(rigid, local).max()) <= _RIGID_TOLERANCE_M


@pytest.mark.parametrize("link", _FINGERS)
def test_the_open_jaw_leaves_the_space_between_the_fingers_free(openarm, link: str) -> None:
    """The point of the change: at full open, the gap is no longer solid.

    The midpoint of the two fingers' centroids, at the open end of the stroke,
    sits outside both the leader's hull and the open bin's — the kernel no
    longer stops on a cell there — while the follower's whole-stroke hull (the
    pre-bin model's behaviour for this volume) still holds it.
    """
    robot, model, data, link_body, coupled = openarm
    rigid, follower = _pieces(robot, link)
    joint = next(j for j in robot.joints if j.child_link == link)
    leader = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint.sim_joint_name))
    lo, hi = joint.position_limits  # type: ignore[misc]  # reason: the OpenArm gripper declares limits
    q_open = hi if abs(hi) > abs(lo) else lo
    mujoco.mj_resetData(model, data)
    mujoco.mj_kinematics(model, data)
    lead = np.asarray(_body_mesh(model, data, link_body[link]).vertices).mean(axis=0)
    meshes = _follower_meshes_at(model, data, link_body[link], leader, coupled[leader], q_open)
    foll = np.vstack([np.asarray(m.vertices) for m in meshes]).mean(axis=0)
    mid = ((lead + foll) / 2.0)[None, :]
    open_bin = next(b for b in follower.stroke_bins if b.q_lo <= q_open <= b.q_hi)
    as_bin = follower.model_copy(update={"tight_geometry": open_bin.tight_geometry})
    clear = min(
        float(tight_escape_per_point_m(rigid, _in_box(rigid, mid))[0]),
        float(tight_escape_per_point_m(as_bin, _in_box(follower, mid))[0]),
    )
    assert clear > 0.005, f"{link}: the open-jaw gap is {clear * 1e3:.1f} mm from solid"
    assert float(tight_escape_per_point_m(follower, _in_box(follower, mid))[0]) <= 0.0


def test_the_manifest_bins_are_contiguous_and_cover_the_gripper_range() -> None:
    robot = RobotDescription.from_yaml(str(_MANIFEST))
    for link in _FINGERS:
        _, follower = _pieces(robot, link)
        joint = next(j for j in robot.joints if j.name == follower.stroke_joint)
        bins = follower.stroke_bins
        assert joint.position_limits == (bins[0].q_lo, bins[-1].q_hi)
        assert all(a.q_hi == b.q_lo for a, b in itertools.pairwise(bins))


def test_a_gapped_or_unrefined_bin_set_is_refused() -> None:
    robot = RobotDescription.from_yaml(str(_MANIFEST))
    _, follower = _pieces(robot, _FINGERS[0])
    raw = follower.model_dump()
    gapped = dict(raw, stroke_bins=[raw["stroke_bins"][0], *raw["stroke_bins"][2:]])
    with pytest.raises(ValidationError, match="not contiguous"):
        LinkCollisionGeometry.model_validate(gapped)
    with pytest.raises(ValidationError, match="refine tight_geometry"):
        LinkCollisionGeometry.model_validate(dict(raw, tight_geometry=None))
    with pytest.raises(ValidationError, match="declared together"):
        LinkCollisionGeometry.model_validate(dict(raw, stroke_joint=None))


def test_a_bin_set_on_a_non_gripper_joint_or_a_short_cover_is_refused() -> None:
    raw = RobotDescription.from_yaml(str(_MANIFEST)).model_dump()
    i = next(
        k
        for k, g in enumerate(raw["collision_geometry"])
        if g["link_name"] == _FINGERS[0] and g["stroke_bins"]
    )
    wrong_joint = {**raw, "collision_geometry": list(raw["collision_geometry"])}
    wrong_joint["collision_geometry"][i] = {
        **raw["collision_geometry"][i],
        "stroke_joint": "left_joint7",
    }
    with pytest.raises(ValidationError, match="gripper's opening only"):
        RobotDescription.model_validate(wrong_joint)
    short = {**raw, "collision_geometry": list(raw["collision_geometry"])}
    short["collision_geometry"][i] = {
        **raw["collision_geometry"][i],
        "stroke_bins": raw["collision_geometry"][i]["stroke_bins"][:-1],
    }
    with pytest.raises(ValidationError, match="whole range"):
        RobotDescription.model_validate(short)


def test_the_kernel_params_carry_every_bin_grouped_by_box() -> None:
    robot = RobotDescription.from_yaml(str(_MANIFEST))
    params = collision_params_from_description(robot)
    boxes = params["collision_stroke_bin_box"]
    assert boxes == sorted(boxes), "bins must be grouped by box in ascending order"
    assert len(boxes) == sum(len(g.stroke_bins) for g in robot.collision_geometry)
    columns = {j.name: i for i, j in enumerate(robot.joints)}
    hulls = params["collision_stroke_bin_hull"]
    for g in (g for g in robot.collision_geometry if g.stroke_bins):
        rows = [
            k
            for k, d in enumerate(params["collision_stroke_bin_dof"])
            if d == columns[str(g.stroke_joint)]
        ]
        assert [params["collision_stroke_bin_q_lo"][k] for k in rows] == [
            b.q_lo for b in g.stroke_bins
        ]
        for k, b in zip(rows, g.stroke_bins, strict=True):
            n = params["collision_hull_vertex_count"][hulls[k]]
            assert n == len(b.tight_geometry.hull_vertices_m)
            # The box's own hull is the whole-stroke one, never a bin's.
            assert params["collision_box_hull"][boxes[k]] not in hulls
