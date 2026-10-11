"""The OpenArm's follower finger as a rigid mimic-joint link (issue #349).

``finger_pair`` is the leader finger (the ``role: gripper`` joint's child);
``finger_follower`` is the URDF ``<mimic>`` / MJCF-equality second jaw, a
``mimic_joints`` row the kernel poses at ``multiplier * q[gripper] + offset``
every check (``docs/reference/finger-mimic-geometry.md``). Each finger is one
rigid box + exact hull, so the safety argument is the same as for every other
link: the hull holds its mesh (rigid, so once), and the kernel's placement of
the follower agrees with the source model at every opening. Pinned here against
the real MJCF meshes with the schema, lowering and parameter contracts on the
real manifest. Real fixtures only (CLAUDE.md §1.11).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco", reason="needs mujoco (sim group) for the MJCF meshes")
pytest.importorskip("trimesh")
pytest.importorskip("scipy")

from openral_core import MimicJoint, RobotDescription  # noqa: E402
from openral_core.assets import AssetRefError, resolve_asset  # noqa: E402
from openral_core.exceptions import ROSConfigError  # noqa: E402
from openral_core.schemas import JointType  # noqa: E402
from openral_safety.envelope_loader import collision_params_from_description  # noqa: E402
from openral_safety.urdf_lowering import (  # noqa: E402
    _body_mesh,
    _declared_mimic_followers,
    _mjcf_link_bodies,
    _xyzrpy_matrix,
    mjcf_coupled_joints,
)
from pydantic import ValidationError  # noqa: E402

from tests.unit._collision_kernel_verdict import tight_escape_per_point_m  # noqa: E402

_MANIFEST = Path(__file__).resolve().parents[2] / "robots" / "openarm" / "robot.yaml"
_SIDES = ("left", "right")
# A rigid hull is exact: only the 1 nm write rounding separates it from its mesh.
_RIGID_TOLERANCE_M = 1e-6
# The kernel's follower placement against MuJoCo's, on every follower mesh vertex.
_FK_TOLERANCE_M = 1e-3
_OPENINGS = 25


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


def _gripper(robot: RobotDescription, side: str) -> tuple[object, MimicJoint]:
    joint = next(j for j in robot.joints if j.name == f"{side}_gripper")
    mimic = next(m for m in robot.mimic_joints if m.mimics == joint.name)
    return joint, mimic


def _pose(model: object, data: object, coupled: dict, leader: int, q: float) -> None:
    mujoco.mj_resetData(model, data)
    data.qpos[int(model.jnt_qposadr[leader])] = q  # type: ignore[attr-defined]  # reason: MjModel
    for follower, c0, c1 in coupled.get(leader, []):
        data.qpos[int(model.jnt_qposadr[follower])] = c0 + c1 * q  # type: ignore[attr-defined]  # reason: MjModel
    mujoco.mj_kinematics(model, data)


def _body_tf(data: object, body: int) -> np.ndarray:
    tf = np.eye(4)
    tf[:3, :3] = data.xmat[body].reshape(3, 3)  # type: ignore[attr-defined]  # reason: MjData
    tf[:3, 3] = data.xpos[body]  # type: ignore[attr-defined]  # reason: MjData
    return tf


def _rotation(axis: tuple[float, float, float], angle: float) -> np.ndarray:
    a = np.asarray(axis, dtype=float)
    a = a / np.linalg.norm(a)
    k = np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])
    return np.eye(3) + np.sin(angle) * k + (1.0 - np.cos(angle)) * (k @ k)


@pytest.mark.parametrize("side", _SIDES)
def test_each_finger_is_one_rigid_exact_hull_holding_its_own_mesh(openarm, side: str) -> None:
    robot, model, data, link_body, _ = openarm
    mujoco.mj_resetData(model, data)
    mujoco.mj_kinematics(model, data)
    for link in (f"openarm_{side}_finger_pair", f"openarm_{side}_finger_follower"):
        pieces = [g for g in robot.collision_geometry if g.link_name == link]
        assert len(pieces) == 1 and pieces[0].tight_geometry is not None, f"{link}: one box + hull"
        mesh = _body_mesh(model, data, link_body[link])
        tf = _xyzrpy_matrix(pieces[0].origin_xyz_rpy)
        local = (np.asarray(mesh.vertices) - tf[:3, 3]) @ tf[:3, :3]
        escape = float(tight_escape_per_point_m(pieces[0], local).max())
        assert escape <= _RIGID_TOLERANCE_M, f"{link}: mesh escapes its hull by {escape:.2e} m"


@pytest.mark.parametrize("side", _SIDES)
def test_the_kernel_places_the_follower_where_the_mjcf_does_at_every_opening(
    openarm, side: str
) -> None:
    """``origin · R(axis, multiplier · q + offset)`` from link7 is the MJCF's follower body."""
    robot, model, data, link_body, coupled = openarm
    joint, mimic = _gripper(robot, side)
    leader = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint.sim_joint_name))
    assert mimic.parent_link == joint.parent_link and mimic.joint_type == JointType.REVOLUTE
    lo, hi = joint.position_limits  # type: ignore[misc]  # reason: the OpenArm gripper declares limits
    worst = 0.0
    for q in np.linspace(lo, hi, _OPENINGS):
        _pose(model, data, coupled, leader, float(q))
        follower_body = link_body[mimic.child_link]
        verts = np.asarray(_body_mesh(model, data, follower_body).vertices)
        kernel = _xyzrpy_matrix((*mimic.origin_xyz, *mimic.origin_rpy))
        kernel[:3, :3] = kernel[:3, :3] @ _rotation(mimic.axis_xyz, mimic.value_at(float(q)))
        mjcf = np.linalg.inv(_body_tf(data, link_body[mimic.parent_link])) @ _body_tf(
            data, follower_body
        )
        placed_k = (kernel[:3, :3] @ verts.T).T + kernel[:3, 3]
        placed_m = (mjcf[:3, :3] @ verts.T).T + mjcf[:3, 3]
        worst = max(worst, float(np.linalg.norm(placed_k - placed_m, axis=1).max()))
    assert worst <= _FK_TOLERANCE_M, f"{side}: follower placement differs by {worst * 1e3:.3f} mm"


@pytest.mark.parametrize("side", _SIDES)
def test_the_open_jaw_leaves_the_space_between_the_fingers_free(openarm, side: str) -> None:
    """The point of the change: at full open, neither finger's hull covers the gap."""
    robot, model, data, link_body, coupled = openarm
    joint, mimic = _gripper(robot, side)
    leader = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint.sim_joint_name))
    lo, hi = joint.position_limits  # type: ignore[misc]  # reason: the OpenArm gripper declares limits
    _pose(model, data, coupled, leader, hi if abs(hi) > abs(lo) else lo)
    links = (joint.child_link, mimic.child_link)
    centroids = [
        _body_tf(data, link_body[ln])[:3, :3]
        @ np.asarray(_body_mesh(model, data, link_body[ln]).vertices).mean(axis=0)
        + _body_tf(data, link_body[ln])[:3, 3]
        for ln in links
    ]
    mid = ((centroids[0] + centroids[1]) / 2.0)[None, :]
    clear = np.inf
    for ln in links:
        g = next(g for g in robot.collision_geometry if g.link_name == ln)
        tf = _body_tf(data, link_body[ln]) @ _xyzrpy_matrix(g.origin_xyz_rpy)
        clear = min(clear, float(tight_escape_per_point_m(g, (mid - tf[:3, 3]) @ tf[:3, :3])[0]))
    assert clear > 0.02, f"{side}: the open-jaw gap midpoint is {clear * 1e3:.1f} mm from solid"


def test_the_manifest_mimic_relation_is_the_mjcf_equality_and_a_mismatch_is_refused(
    openarm,
) -> None:
    robot, model, _, _, coupled = openarm
    declared = _declared_mimic_followers(robot, model, coupled)
    assert len(declared) == len(robot.mimic_joints) == 2
    wrong = robot.model_copy(
        update={
            "mimic_joints": [
                robot.mimic_joints[0].model_copy(update={"multiplier": -1.0}),
                *robot.mimic_joints[1:],
            ]
        }
    )
    with pytest.raises(ROSConfigError, match="disagree with the MJCF equality"):
        _declared_mimic_followers(wrong, model, coupled)


def test_the_kernel_params_carry_the_mimic_kind_on_the_gripper_column() -> None:
    robot = RobotDescription.from_yaml(str(_MANIFEST))
    params = collision_params_from_description(robot)
    names = list(params["collision_link_names"])
    kinds = list(params["collision_joint_kind"])
    dofs = list(params["collision_dof_index"])
    mult = list(params["collision_mimic_multiplier"])
    off = list(params["collision_mimic_offset"])
    assert len(mult) == len(off) == len(names)
    for mimic in robot.mimic_joints:
        follower = names.index(mimic.child_link)
        leader = names.index(next(j.child_link for j in robot.joints if j.name == mimic.mimics))
        assert (
            kinds[follower] == 3 and kinds[leader] == 1
        )  # mimic-revolute reads the leader's column
        assert dofs[follower] == dofs[leader] >= 0
        assert (mult[follower], off[follower]) == (mimic.multiplier, mimic.offset)
    assert all(
        m == 1.0 and o == 0.0
        for i, (m, o) in enumerate(zip(mult, off, strict=True))
        if kinds[i] < 3
    )


def test_grasp_contact_links_name_both_jaws_of_each_gripper() -> None:
    robot = RobotDescription.from_yaml(str(_MANIFEST))
    assert robot.grasp_contact_links() == (
        "openarm_left_finger_pair",
        "openarm_right_finger_pair",
        "openarm_left_finger_follower",
        "openarm_right_finger_follower",
    )


def test_the_generated_acm_exempts_the_follower_against_its_leader_and_its_parent_only() -> None:
    """The two jaws of one gripper and the follower's own mount are exempt (the lowering's
    adjacency rule); against link5 / link6 the follower stays CHECKED — unlike the leader's
    SRDF rows, its hull never trips them over 500 seeded in-limit poses (design note §3)."""
    robot = RobotDescription.from_yaml(str(_MANIFEST))
    pairs = {frozenset(p) for p in robot.allowed_collision_pairs}
    for side in _SIDES:
        follower = f"openarm_{side}_finger_follower"
        exempt = {other for p in pairs if follower in p for other in p if other != follower}
        assert exempt == {f"openarm_{side}_finger_pair", f"openarm_{side}_link7"}, exempt


def test_mimic_joint_schema_refusals() -> None:
    good = MimicJoint(name="f", parent_link="hand", child_link="jaw", mimics="g")
    assert good.value_at(0.5) == 0.5
    with pytest.raises(ValidationError, match="multiplier 0"):
        MimicJoint(name="f", parent_link="hand", child_link="jaw", mimics="g", multiplier=0.0)
    with pytest.raises(ValidationError, match="to itself"):
        MimicJoint(name="f", parent_link="jaw", child_link="jaw", mimics="g")
    with pytest.raises(ValidationError, match="a mimic joint moves"):
        MimicJoint(
            name="f", parent_link="hand", child_link="jaw", mimics="g", joint_type=JointType.FIXED
        )
    raw = RobotDescription.from_yaml(str(_MANIFEST)).model_dump()
    unknown = {**raw, "mimic_joints": [{**raw["mimic_joints"][0], "mimics": "no_such_joint"}]}
    with pytest.raises(ValidationError, match="not in joints"):
        RobotDescription.model_validate(unknown)
    clash = {
        **raw,
        "mimic_joints": [{**raw["mimic_joints"][0], "child_link": "openarm_left_finger_pair"}],
    }
    with pytest.raises(ValidationError, match="already defined by joint"):
        RobotDescription.model_validate(clash)
