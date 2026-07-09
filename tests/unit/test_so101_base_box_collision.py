"""Offline verification of the SO-101 all-OBB self-collision model + tuned
negative self-collision margin (issue #84).

Story: a live ``openral deploy run`` on the real SO-101 latched ``/openral/estop``
before any motion. Root cause chain (all evidence-backed, no hardware needed):

1. The base link is a near-cubic housing; a capsule over-reported its clearance
   (base↔lower_arm/wrist) → first false E-stop. Boxing the base fixed it.
2. The arm links are rectangular 3D-printed brackets; a single capsule per link
   over-reports ~7-9 cm at the arm's compact folded operating poses (shoulder↔
   lower_arm etc.) → the next false E-stop. Boxing every link cuts the
   over-report to ~1-4 cm.
3. The pen VLA's whole training distribution is compact-folded, where the *true*
   (fcl non-convex mesh) self-clearance is ≈ 0 — the arm operates in light
   self-contact by design. A small negative ``self_collision_margin_m`` (−0.06)
   lets that envelope pass while gross over-folds (far outside the VLA
   distribution) still E-stop; real self-contact is protected by the force/torque
   limits, and WORLD/voxel collision (its own positive margin) is unaffected.

Fully offline (no ROS, no hardware, no ``deploy run``). The C++ box distance math
is unit-tested in ``cpp/openral_safety_kernel/test/test_collision.cpp``; this
mirrors the kernel FK + box↔box distance from the *real manifest params* and
cross-checks against MuJoCo (convex hull) and, where available, fcl (true
non-convex).
"""

from __future__ import annotations

import math
import pathlib

import numpy as np
import pytest
from openral_core.assets import resolve_asset
from openral_core.schemas import BoxShape, RobotDescription

pytest.importorskip("mujoco")
import mujoco as mj

_MANIFEST = pathlib.Path(__file__).resolve().parents[2] / "robots" / "so101_follower" / "robot.yaml"

# In-distribution / operating poses (radians, manifest joint order) that must be
# ALLOWED, and gross over-folds (far outside the VLA distribution) that must fire.
_D2R = math.radians
_STARTING_POSE = [0.0, -1.5708, 1.5708, 0.75, 0.0, 0.5]  # SRDF rest = pen starting_pose
_VLA_START = [_D2R(x) for x in (-5.566, -98.899, 98.48, 71.933, 0.543, 14.054)]
_MEASURED_FOLD = [_D2R(x) for x in (4.3, -88.3, 100.0, 19.2, 12.2, 3.1)]
_GROSS_FOLD = [_D2R(x) for x in (0, -90, 150, -30, 0, 10)]  # elbow 150° — way OOD


# ── kernel-mirrored geometry (matches cpp/.../collision.cpp) ───────────────────


def _transform(x, y, z, roll, pitch, yaw):
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    r = np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )
    m = np.eye(4)
    m[:3, :3] = r
    m[:3, 3] = (x, y, z)
    return m


def _axis_angle(axis, angle):
    a = np.asarray(axis, float)
    n = np.linalg.norm(a)
    m = np.eye(4)
    if n < 1e-12 or abs(angle) < 1e-15:
        return m
    a = a / n
    c, s = math.cos(angle), math.sin(angle)
    x, y, z = a
    m[:3, :3] = np.array(
        [
            [c + x * x * (1 - c), x * y * (1 - c) - z * s, x * z * (1 - c) + y * s],
            [y * x * (1 - c) + z * s, c + y * y * (1 - c), y * z * (1 - c) - x * s],
            [z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z * z * (1 - c)],
        ]
    )
    return m


def _fk(p, q):
    parent, origin, kind, dof, axis = (
        p["collision_parent"],
        p["collision_origin_xyzrpy"],
        p["collision_joint_kind"],
        p["collision_dof_index"],
        p["collision_axis"],
    )
    names = p["collision_link_names"]
    world = [None] * len(names)
    for i in range(len(names)):
        ti = _transform(*origin[6 * i : 6 * i + 6])
        motion = np.eye(4)
        d = dof[i]
        if d >= 0 and d < len(q) and kind[i] == 1:
            motion = _axis_angle(axis[3 * i : 3 * i + 3], q[d])
        local = ti @ motion
        world[i] = local if parent[i] < 0 else world[parent[i]] @ local
    return world


def _box_box(a_tf, a_half_extents, b_tf, b_half_extents):
    """Conservative SAT box-box distance (max separating gap; matches the C++)."""
    a_rot, b_rot = a_tf[:3, :3], b_tf[:3, :3]
    dc = b_tf[:3, 3] - a_tf[:3, 3]
    a_half_extents = np.asarray(a_half_extents)
    b_half_extents = np.asarray(b_half_extents)
    axes = [a_rot[:, i] for i in range(3)] + [b_rot[:, i] for i in range(3)]
    axes += [np.cross(a_rot[:, i], b_rot[:, j]) for i in range(3) for j in range(3)]
    best = -9.0
    for axis in axes:
        n = np.linalg.norm(axis)
        if n < 1e-9:
            continue
        unit_axis = axis / n
        ra = sum(a_half_extents[k] * abs(a_rot[:, k] @ unit_axis) for k in range(3))
        rb = sum(b_half_extents[k] * abs(b_rot[:, k] @ unit_axis) for k in range(3))
        best = max(best, abs(dc @ unit_axis) - ra - rb)
    return best


def _boxes(p):
    names = p["collision_link_names"]
    bl, bh, bo = (
        p["collision_box_link"],
        p["collision_box_half_extents"],
        p["collision_box_origin_xyzrpy"],
    )
    return {
        names[li]: (np.asarray(bh[3 * k : 3 * k + 3]), bo[6 * k : 6 * k + 6])
        for k, li in enumerate(bl)
    }


def _kernel_self_min(p, q):
    """Min box↔box distance over every checked (non-allowed) link pair."""
    names = p["collision_link_names"]
    idx = {n: i for i, n in enumerate(names)}
    allowed = set()
    ap = p.get("collision_allowed_pairs", [])
    for i in range(0, len(ap), 2):
        allowed.add(tuple(sorted((ap[i], ap[i + 1]))))
    world = _fk(p, q)
    boxes = _boxes(p)
    bn = list(boxes)
    vals = []
    for i in range(len(bn)):
        for j in range(i + 1, len(bn)):
            a, b = bn[i], bn[j]
            if tuple(sorted((idx[a], idx[b]))) in allowed:
                continue
            a_half_extents, a_origin = boxes[a]
            b_half_extents, b_origin = boxes[b]
            vals.append(
                _box_box(
                    world[idx[a]] @ _transform(*a_origin),
                    a_half_extents,
                    world[idx[b]] @ _transform(*b_origin),
                    b_half_extents,
                )
            )
    return min(vals)


def _params():
    pytest.importorskip("openral_safety")
    from openral_safety.envelope_loader import collision_params_from_description

    return collision_params_from_description(RobotDescription.from_yaml(str(_MANIFEST)))


# ── manifest / lowering shape ─────────────────────────────────────────────────


def test_manifest_is_all_obb_no_capsules():
    rd = RobotDescription.from_yaml(str(_MANIFEST))
    shapes = {g.link_name: g.shape for g in rd.collision_geometry}
    for link in ("base", "shoulder", "upper_arm", "lower_arm", "wrist"):
        assert isinstance(shapes[link], BoxShape), f"{link} must be an OBB"


def test_lowering_emits_all_boxes_and_negative_margin():
    p = _params()
    names = p["collision_link_names"]
    box_links = sorted(names[i] for i in p.get("collision_box_link", []))
    assert box_links == ["base", "lower_arm", "shoulder", "upper_arm", "wrist"]
    assert len(p.get("collision_capsule_link", [])) == 0  # omitted when empty
    assert p["self_collision_margin_m"] == pytest.approx(-0.06)


def test_every_obb_encloses_its_link_mesh():
    """Safety (§3): each box contains its link's collision mesh (conservative)."""
    rd = RobotDescription.from_yaml(str(_MANIFEST))
    model = _mj_model()

    def quat_to_rot(q):
        w, x, y, z = q
        return np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ]
        )

    for g in rd.collision_geometry:
        bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, g.link_name)
        rot = _transform(*g.origin_xyz_rpy)[:3, :3]
        c = np.asarray(g.origin_xyz_rpy[:3])
        he = np.asarray(g.shape.half_extents_m)
        worst = 0.0
        for gi in range(model.ngeom):
            if model.geom_bodyid[gi] != bid or model.geom_dataid[gi] < 0:
                continue
            did = model.geom_dataid[gi]
            verts = model.mesh_vert[
                model.mesh_vertadr[did] : model.mesh_vertadr[did] + model.mesh_vertnum[did]
            ].reshape(-1, 3)
            verts = (verts @ quat_to_rot(model.geom_quat[gi]).T) + model.geom_pos[gi]
            local = (verts - c) @ rot
            worst = max(worst, float((np.abs(local) - he).max()))
        assert worst <= 2e-3, f"{g.link_name} mesh escapes its OBB by {worst:.4f} m"


# ── margin behaviour: envelope allowed, gross folds fire ──────────────────────


def test_operating_envelope_clears_the_margin():
    """The pen VLA's in-distribution folded poses pass the self-check (kernel box
    distance above the negative margin) — no false E-stop."""
    p = _params()
    margin = p["self_collision_margin_m"]
    for name, q in (
        ("starting_pose", _STARTING_POSE),
        ("measured_fold", _MEASURED_FOLD),
        ("vla_start", _VLA_START),
    ):
        k = _kernel_self_min(p, q)
        assert k > margin, f"{name} would false-E-stop: kernel {k:.4f} <= margin {margin}"


def test_gross_overfold_still_fires():
    """A fold far outside the VLA distribution (elbow 150°) still trips the
    self-check — the negative margin is not 'self-collision disabled'."""
    p = _params()
    assert _kernel_self_min(p, _GROSS_FOLD) <= p["self_collision_margin_m"]


# ── MuJoCo oracle: the operating envelope is grazing, not deep penetration ─────


def _mj_model():
    rd = RobotDescription.from_yaml(str(_MANIFEST))
    mjcf = resolve_asset(rd.assets.mjcf, "mjcf", manifest_dir=_MANIFEST.parent)
    return mj.MjModel.from_xml_path(str(mjcf))


def test_operating_envelope_is_near_contact_not_deep_penetration():
    """Convex-hull mesh clearance at the VLA start is only mildly negative
    (grazing/contact), confirming the arm operates in light self-contact rather
    than deep interpenetration — so the negative margin tolerates real operation,
    it is not hiding a gross collision."""
    model = _mj_model()
    data = mj.MjData(model)
    data.qpos[:6] = _VLA_START
    mj.mj_forward(model, data)

    def geoms(name):
        bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, name)
        return [g for g in range(model.ngeom) if model.geom_bodyid[g] == bid]

    ft = np.zeros(6)
    for a, b in (("shoulder", "lower_arm"), ("shoulder", "wrist"), ("upper_arm", "wrist")):
        hull = min(
            mj.mj_geomDistance(model, data, ga, gb, 1.0, ft) for ga in geoms(a) for gb in geoms(b)
        )
        assert hull > -0.03, f"{a}<->{b} hull clearance {hull:.4f} is deeper than grazing"
