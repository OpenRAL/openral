"""No non-allowed collision pair may interpenetrate at a robot's rest pose.

The C++ safety kernel evaluates every candidate chunk against the manifest's
``collision_geometry``, skipping only ``allowed_collision_pairs``. A pair that
already overlaps at the pose the robot rests in is not a hazard the kernel can
act on: it is a modelling error that makes the kernel refuse *every* chunk near
rest. That is exactly what stopped the first OpenArm policy dispatch on qorin1
(2026-09-22): ``openarm_*_link3``'s hand-authored capsule was 0.22 m long from
a joint 0.154 m above the elbow, so it reached 6.6 cm past joint 4 into link 5's
capsule — a constant ``-0.04575 m`` at every elbow angle, on both arms, on the
twin and on the real cell.

The rest pose is ``q = 0`` when every joint's ``position_limits`` admit it.
When they do not (the Panda's joint 4 tops out at -0.0698 rad, and its meshes
interpenetrate at q = 0 anyway), zero is not a pose the robot can be in, and
the rest pose is the ``ready`` / ``home`` ``group_state`` of the manifest's own
SRDF. A robot with neither fails loudly rather than being checked at a pose it
cannot reach.

Real fixtures only: every ``robots/*/robot.yaml`` that declares collision
geometry is checked with a small forward-kinematics pass over its own
``joints`` and ``fixed_attachments`` — the same inputs
``collision_params_from_description`` lowers — and the kernel's own distance
predicates (``openral_safety.kernel_predicates.shape_distance``, a line-by-line
port of ``collision.cpp``), tripping at ``d <= safety.self_collision_margin_m``
as the kernel does. Every robot whose rest pose is fixed here also gets a
negative case: a configuration where the real meshes collide must still trip.
"""

from __future__ import annotations

import itertools
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
from openral_core.schemas import JointType, RobotDescription
from openral_safety.kernel_predicates import shape_distance

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MANIFESTS = sorted(_REPO_ROOT.glob("robots/*/robot.yaml"))
# SRDF group_state names read as the robot's rest pose, in preference order.
_REST_STATE_NAMES = ("ready", "home")


def _rot(rpy: tuple[float, float, float]) -> np.ndarray:
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    return rz @ ry @ rx


def _tf(xyz: tuple[float, ...], rpy: tuple[float, ...]) -> np.ndarray:
    m = np.eye(4)
    m[:3, :3] = _rot((rpy[0], rpy[1], rpy[2]))
    m[:3, 3] = xyz
    return m


def _joint_motion(joint_type: JointType, axis: tuple[float, float, float], q: float) -> np.ndarray:
    """The joint's own motion (URDF: applied after its origin transform)."""
    m = np.eye(4)
    a = np.asarray(axis, dtype=float)
    if joint_type in (JointType.REVOLUTE, JointType.CONTINUOUS):
        a = a / np.linalg.norm(a)
        k = np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])
        m[:3, :3] = np.eye(3) + np.sin(q) * k + (1.0 - np.cos(q)) * (k @ k)
    elif joint_type == JointType.PRISMATIC:
        m[:3, 3] = a * q
    return m


def _link_poses(robot: RobotDescription, q: dict[str, float]) -> dict[str, np.ndarray]:
    """World pose of every link at joint positions ``q`` (missing joints at 0)."""
    poses: dict[str, np.ndarray] = {robot.base_frame: np.eye(4)}
    pending = [
        (a.parent_link, a.child_link, _tf(a.origin_xyz, a.origin_rpy))
        for a in robot.fixed_attachments
    ] + [
        (
            j.parent_link,
            j.child_link,
            _tf(j.origin_xyz, j.origin_rpy) @ _joint_motion(j.joint_type, j.axis_xyz, q[j.name]),
        )
        for j in robot.joints
    ]
    # Fixed-point pass: parents may be declared after children.
    progressed = True
    while pending and progressed:
        progressed = False
        for entry in list(pending):
            parent, child, local = entry
            if parent in poses:
                poses[child] = poses[parent] @ local
                pending.remove(entry)
                progressed = True
    if pending:
        # Links that do not chain back to ``base_frame`` would have to be
        # placed by guessing a root, and a guessed root superimposes subtrees
        # at the origin — false overlaps. Refuse rather than guess.
        raise LookupError(sorted({child for _p, child, _l in pending}))
    return poses


def _srdf_states(robot: RobotDescription, manifest_path: Path) -> dict[str, dict[str, float]]:
    ref = robot.assets.srdf
    if not ref or not ref.startswith("file:"):
        return {}
    root = ET.parse(manifest_path.parent / ref.removeprefix("file:")).getroot()
    return {
        state.attrib["name"]: {j.attrib["name"]: float(j.attrib["value"]) for j in state}
        for state in root.iter("group_state")
    }


def _rest_pose(robot: RobotDescription, manifest_path: Path) -> tuple[str, dict[str, float]]:
    """``q = 0`` when the joint limits admit it, else the SRDF ready/home state."""
    q = {j.name: 0.0 for j in robot.joints}
    outside = [
        j.name
        for j in robot.joints
        if j.position_limits is not None
        and not (j.position_limits[0] <= 0.0 <= j.position_limits[1])
    ]
    if not outside:
        return "q = 0", q
    states = _srdf_states(robot, manifest_path)
    for name in _REST_STATE_NAMES:
        if name in states and set(outside) <= states[name].keys():
            return f"SRDF group_state {name!r}", q | states[name]
    pytest.fail(
        f"{robot.name}: q = 0 is outside the limits of {outside} and the manifest's SRDF "
        f"declares no {' / '.join(_REST_STATE_NAMES)} group_state covering them, so "
        "there is no declared rest pose to check."
    )


def _tripping_pairs(robot: RobotDescription, q: dict[str, float]) -> list[tuple[str, str, float]]:
    """Every non-allowed pair the kernel would trip on at ``q``, with its gap."""
    poses = _link_poses(robot, q)
    margin = float(robot.safety.self_collision_margin_m or 0.0)
    allowed = {frozenset(p) for p in robot.allowed_collision_pairs}
    placed = [
        (
            g.link_name,
            g.shape,
            (poses[g.link_name] @ _tf(g.origin_xyz_rpy[:3], g.origin_xyz_rpy[3:]))[None],
        )
        for g in robot.collision_geometry
        if g.link_name in poses
    ]
    out = []
    for (a, sa, ta), (b, sb, tb) in itertools.combinations(sorted(placed, key=lambda c: c[0]), 2):
        if a == b or frozenset((a, b)) in allowed:
            continue
        gap = float(shape_distance(sa, ta, sb, tb)[0])
        if gap <= margin:
            out.append((a, b, round(gap, 4)))
    return out


def _load(name: str) -> RobotDescription:
    return RobotDescription.from_yaml(str(_REPO_ROOT / "robots" / name / "robot.yaml"))


# Manifests whose geometry still trips at rest, pending a safety-WG decision.
# Strict xfails so a fix flips them green and a regression is loud.
_UNADJUDICATED: dict[str, str] = {
    # Arms hanging at q = 0 sit 1-4 cm from the torso mesh (MuJoCo hull gaps
    # elbow +0.037, shoulder_yaw +0.009, shoulder_roll +0.002 m), but the torso
    # is a tapered block whose widest point (|y| = 0.1077 m, at the shoulder
    # mounts) is wider than the gap to the upper arms below it: no single
    # primitive per link that bounds the torso mesh clears both upper arms —
    # not the capsule (r = 0.169 m), not its bounding box (-0.041 m against
    # the upper-arm capsules). Needs the kernel's box tight_geometry (hull
    # narrow phase) on torso/upper arm, or a safety-WG exemption.
    "g1": "arms-vs-torso at q = 0; one bounding primitive per link cannot clear it",
    # q = 0 of this URDF is the folded rest (upper arm lying on the base): the
    # meshes are in contact there (3 base vertices inside upper_arm's mesh and
    # 4 the other way; MuJoCo hull gap -0.0003 m), so no primitive that bounds
    # both links can clear it at margin 0, and there is no declared rest pose
    # (no SRDF) to check instead. Needs a safety-WG call: a measured rest pose,
    # an allowed pair, or the SO-101 box model + negative margin of issue #84.
    "so100_follower": "base/upper_arm meshes in contact at the folded q = 0 pose",
}


@pytest.mark.parametrize(
    "manifest_path",
    [
        pytest.param(
            p,
            id=p.parent.name,
            marks=(
                pytest.mark.xfail(strict=True, reason=_UNADJUDICATED[p.parent.name])
                if p.parent.name in _UNADJUDICATED
                else ()
            ),
        )
        for p in _MANIFESTS
    ],
)
def test_no_non_allowed_pair_interpenetrates_at_rest(manifest_path: Path) -> None:
    robot = RobotDescription.from_yaml(str(manifest_path))
    if not robot.collision_geometry:
        pytest.skip("manifest declares no collision_geometry")
    label, q = _rest_pose(robot, manifest_path)
    try:
        overlaps = _tripping_pairs(robot, q)
    except LookupError as unrooted:
        pytest.skip(f"links not rooted at base_frame, FK unvalidated here: {unrooted.args[0]}")
    assert not overlaps, (
        f"{robot.name}: collision pairs the kernel trips on at its rest pose ({label}), "
        f"so it would refuse every chunk near rest: {overlaps}"
    )


# A configuration per fixed robot, inside every joint limit, where the real
# meshes interpenetrate — the fix must not have bought the rest pose by
# blinding the kernel there. Evidence (vertices of the first link inside the
# second link's watertight mesh at this q):
#   franka_panda: 40 panda_hand vertices inside panda_link1 (panda_description
#     URDF collision meshes) — elbow fully folded, hand driven into the shoulder.
#   h1: 627 left_elbow_link vertices inside torso_link (mujoco_menagerie
#     unitree_h1 meshes) — left forearm swung into the chest.
_REAL_COLLISIONS: list[tuple[str, dict[str, float], tuple[str, str]]] = [
    (
        "franka_panda",
        {"panda_joint2": -0.3, "panda_joint4": -3.0, "panda_joint6": 0.2},
        ("panda_hand", "panda_link1"),
    ),
    (
        "h1",
        {
            "left_shoulder_pitch": -0.84,
            "left_shoulder_roll": -0.3,
            "left_shoulder_yaw": 4.05,
            "left_elbow": -0.33,
        },
        ("left_elbow_link", "torso_link"),
    ),
]


@pytest.mark.parametrize(
    ("name", "q_overrides", "pair"), _REAL_COLLISIONS, ids=[r[0] for r in _REAL_COLLISIONS]
)
def test_real_self_collision_still_trips(
    name: str, q_overrides: dict[str, float], pair: tuple[str, str]
) -> None:
    robot = _load(name)
    q = {j.name: 0.0 for j in robot.joints} | q_overrides
    tripped = {frozenset((a, b)) for a, b, _gap in _tripping_pairs(robot, q)}
    assert frozenset(pair) in tripped, f"{name}: {pair} no longer trips at {q_overrides}"
