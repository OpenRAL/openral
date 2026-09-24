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
    edges = [*robot.fixed_attachments, *robot.joints]
    roots = {e.parent_link for e in edges} - {e.child_link for e in edges}
    # The kernel roots its model at the chain's single root; a `base_frame`
    # outside the chain (UR's `ur5e_base_link`) names that same frame.
    root = robot.base_frame if robot.base_frame in roots or len(roots) != 1 else roots.pop()
    poses: dict[str, np.ndarray] = {root: np.eye(4)}
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
    # A box pair whose links both declare `tight_geometry` is decided by the
    # kernel's exact-hull narrow phase (`hull_hull_distance`), which this port
    # does not model; the box distance would report the box's corner slop as a
    # trip. That verdict is pinned end to end by
    # tests/sim/safety/test_kernel_panda_link5_link7.py instead.
    tight = {g.link_name for g in robot.collision_geometry if g.tight_geometry is not None}
    out = []
    for (a, sa, ta), (b, sb, tb) in itertools.combinations(sorted(placed, key=lambda c: c[0]), 2):
        if a == b or frozenset((a, b)) in allowed or {a, b} <= tight:
            continue
        gap = float(shape_distance(sa, ta, sb, tb)[0])
        if gap <= margin:
            out.append((a, b, round(gap, 4)))
    return out


def _load(name: str) -> RobotDescription:
    return RobotDescription.from_yaml(str(_REPO_ROOT / "robots" / name / "robot.yaml"))


# Manifests whose geometry still trips at rest, pending a safety-WG decision.
# Strict xfails so a fix flips them green and a regression is loud. All three
# are the same finding (docs/reference/collision-geometry-review.md §5): the
# capsules now bound the whole link (collision + visual geometry, 1 mm headroom),
# and at rest these links sit closer than any capsule set can express — the
# exact convex hulls are clear, but single capsules AND chains of 2-4 capsules
# per link still overlap. The honest fix is a representation change (the
# kernel's existing box + tight_geometry hull narrow phase), not an exemption.
_UNADJUDICATED: dict[str, str] = {
    # q = 0, arms hanging: shoulder_yaw <-> torso hull gap +7.8 / +9.2 mm
    # (left/right); the capsules overlap by -12.6 / -17.1 mm.
    "g1": "shoulder_yaw vs torso hull gap +7.8 mm at q = 0; no capsule set clears it",
    # q = 0: shoulder_roll <-> torso hull gap +1.2 mm, shoulder_yaw +27.4 mm,
    # elbow +55.0 mm, hip_pitch <-> pelvis +38.3 mm; capsules -0.4 ... -12.1 mm.
    "h1": "shoulder_roll vs torso hull gap +1.2 mm at q = 0; no capsule set clears it",
    # q = 0 is this URDF's folded rest: base <-> upper_arm hulls overlap (nearest
    # vertices 1.0 mm apart, the arm resting on the base), lower_arm <-> shoulder
    # +0.9 mm, shoulder/upper_arm <-> wrist +14-16 mm. There is no declared rest
    # pose (no SRDF) to check instead. Needs a measured rest pose or the hull
    # narrow phase, and a WG call on the resting contact.
    "so100_follower": "folded q = 0: base/upper_arm resting contact, lower_arm/shoulder +0.9 mm",
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


# A configuration per robot, inside every joint limit, where the real meshes
# interpenetrate — tightening a primitive must never have bought anything by
# blinding the kernel there. Evidence (vertices of the first link's visual mesh
# inside the second link's watertight mesh at this q, robot_descriptions meshes):
#   franka_panda: 40 panda_hand vertices inside panda_link1 — elbow fully folded.
#   h1 (1): 627 left_elbow_link vertices inside torso_link — forearm into the chest.
#   h1 (2): the PR #324 review's "forearm miss": 13 604 left_elbow_link vertices
#     inside torso_link, which the placeholder-derived capsules did not trip on.
#   ur5e / ur10e: 32 601 / 33 530 wrist_2_link vertices inside upper_arm_link.
#   rizon4: 93 base_link vertices inside link6.
#   so100_follower: 207 wrist vertices inside base.
#   g1: 163 torso_link vertices inside right_elbow_link.
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
    (
        "h1",
        {
            "left_shoulder_pitch": 0.99,
            "left_shoulder_roll": -0.15,
            "left_shoulder_yaw": -1.03,
            "left_elbow": -0.62,
        },
        ("left_elbow_link", "torso_link"),
    ),
    (
        "ur5e",
        {
            "shoulder_pan_joint": -0.04,
            "shoulder_lift_joint": -3.17,
            "elbow_joint": -3.07,
            "wrist_1_joint": -3.87,
            "wrist_2_joint": 2.41,
            "wrist_3_joint": -3.76,
        },
        ("upper_arm_link", "wrist_2_link"),
    ),
    (
        "ur10e",
        {
            "shoulder_pan_joint": -0.04,
            "shoulder_lift_joint": -3.17,
            "elbow_joint": -3.07,
            "wrist_1_joint": -3.87,
            "wrist_2_joint": 2.41,
            "wrist_3_joint": -3.76,
        },
        ("upper_arm_link", "wrist_2_link"),
    ),
    (
        "rizon4",
        {
            "joint1": 1.82,
            "joint2": -2.29,
            "joint3": 0.78,
            "joint4": 1.8,
            "joint5": 0.08,
            "joint6": 2.95,
            "joint7": -1.67,
        },
        ("base_link", "link6"),
    ),
    (
        "so100_follower",
        {
            "shoulder_pan": 0.52,
            "shoulder_lift": 1.39,
            "elbow_flex": 0.96,
            "wrist_flex": -0.96,
            "wrist_roll": -1.26,
            "gripper": 0.87,
        },
        ("base", "wrist"),
    ),
    (
        "g1",
        {
            "left_hip_pitch_joint": 2.76,
            "left_hip_roll_joint": 1.54,
            "left_hip_yaw_joint": 0.58,
            "left_knee_joint": 1.81,
            "left_ankle_pitch_joint": 0.07,
            "left_ankle_roll_joint": -0.18,
            "right_hip_pitch_joint": -0.15,
            "right_hip_roll_joint": -2.13,
            "right_hip_yaw_joint": -0.54,
            "right_knee_joint": 0.2,
            "right_ankle_pitch_joint": 0.48,
            "right_ankle_roll_joint": -0.15,
            "waist_yaw_joint": 0.9,
            "waist_roll_joint": -0.21,
            "waist_pitch_joint": 0.39,
            "left_shoulder_pitch_joint": 0.72,
            "left_shoulder_roll_joint": -1.08,
            "left_shoulder_yaw_joint": 1.81,
            "left_elbow_joint": 1.92,
            "left_wrist_roll_joint": 1.59,
            "left_wrist_pitch_joint": 0.23,
            "left_wrist_yaw_joint": -1.14,
            "right_shoulder_pitch_joint": -1.98,
            "right_shoulder_roll_joint": 1.31,
            "right_shoulder_yaw_joint": 0.27,
            "right_elbow_joint": -0.48,
            "right_wrist_roll_joint": 1.51,
            "right_wrist_pitch_joint": 0.46,
            "right_wrist_yaw_joint": 0.23,
        },
        ("right_elbow_link", "torso_link"),
    ),
]


@pytest.mark.parametrize(
    ("name", "q_overrides", "pair"),
    _REAL_COLLISIONS,
    ids=[f"{r[0]}-{i}" for i, r in enumerate(_REAL_COLLISIONS)],
)
def test_real_self_collision_still_trips(
    name: str, q_overrides: dict[str, float], pair: tuple[str, str]
) -> None:
    robot = _load(name)
    q = {j.name: 0.0 for j in robot.joints} | q_overrides
    tripped = {frozenset((a, b)) for a, b, _gap in _tripping_pairs(robot, q)}
    assert frozenset(pair) in tripped, f"{name}: {pair} no longer trips at {q_overrides}"
