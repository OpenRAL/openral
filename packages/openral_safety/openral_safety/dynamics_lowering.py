"""MJCF → ``ForceObserverSpec`` lowering for the kernel's external-torque observer (#373).

The safety kernel's momentum observer (``cpp/openral_safety_kernel/src/force_observer.cpp``)
runs rigid-body dynamics over the manifest's own kinematic tree — the one
``envelope_loader.collision_params_from_description`` lowers — so every inertial
must be expressed in the **manifest** link frame. This module reads them from
the robot's MJCF and proves the frames agree before emitting anything:

* manifest links map to MJCF bodies by the joint correspondence
  (``urdf_lowering._mjcf_link_bodies``, the mapping the collision lowering uses);
* at the zero configuration, every mapped body's MJCF pose must equal the
  manifest's forward kinematics up to one rigid root transform, or the lowering
  refuses (a frame mismatch would put every mass in the wrong place);
* an MJCF body no manifest link names is folded into the nearest mapped
  ancestor if it is welded to it, and refused if a joint moves it;
* joint terms come from the MJCF's ``armature`` / ``damping`` /
  ``frictionloss`` (``frictionloss`` → the Coulomb term).

The MJCF is the simulator's model. On a real arm it is the starting point, not a
calibration: the OpenArm's real residual carries stiction and offsets the MJCF
does not (``docs/reference/manipulation-workspace-contact.md`` §8).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from openral_core import (
    ForceObserverSpec,
    JointFriction,
    LinkInertial,
    RobotDescription,
)
from openral_core.exceptions import ROSConfigError

from openral_safety.envelope_loader import _ordered_collision_links
from openral_safety.urdf_lowering import _mjcf_link_bodies, _xyzrpy_matrix

__all__ = ["lower_force_observer_from_mjcf"]

# Frame agreement tolerance between the manifest FK and the MJCF at q = 0. The
# manifest's origins are the MJCF's own transforms (``openral collision lower``
# re-injects them), so the only expected difference is float formatting.
_FRAME_TOL = 1e-5


def lower_force_observer_from_mjcf(
    robot: RobotDescription,
    *,
    manifest_dir: Path | None = None,
    gain_per_s: float = 50.0,
    coulomb_velocity_rad_s: float = 0.05,
    limit_margin_rad: float = 0.02,
) -> ForceObserverSpec:
    """Lower the observer's inertials and joint friction from ``robot``'s MJCF.

    Args:
        robot: The manifest; its ``assets.mjcf`` is the source.
        manifest_dir: Resolves a ``file:`` MJCF ref against the manifest directory.
        gain_per_s: Observer bandwidth to record (1/s).
        coulomb_velocity_rad_s: Coulomb ``tanh`` velocity scale to record.
        limit_margin_rad: Joint-limit margin to record.

    Returns:
        A ``ForceObserverSpec`` whose links are the manifest's.

    Raises:
        ROSConfigError: No MJCF; a moving manifest link with no MJCF body; an MJCF
            joint not at its body origin; a frame that disagrees with the
            manifest; or an unmapped MJCF body moved by its own joint.
    """
    import mujoco
    import numpy as np

    m, d = _zero_pose_model(robot, manifest_dir)
    ordered, link_body, world_root = _checked_link_bodies(robot, m, d)
    links = [
        _link_inertial(link, parts, _body_tf(d, link_body[link]))
        for link, parts in _fold_bodies(robot, m, d, ordered, link_body)
    ]
    joints: list[JointFriction] = []
    for spec in robot.joints:
        ji = int(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, spec.sim_joint_name or spec.name))
        if ji < 0:
            continue
        dof = int(m.jnt_dofadr[ji])
        joints.append(
            JointFriction(
                joint=spec.name,
                armature_kgm2=_r(m.dof_armature[dof]),
                viscous_nm_s_per_rad=_r(m.dof_damping[dof]),
                coulomb_nm=_r(m.dof_frictionloss[dof]),
            )
        )
    gravity = world_root[:3, :3].T @ np.asarray(m.opt.gravity, dtype=np.float64)
    return ForceObserverSpec(
        source=f"openral_safety.dynamics_lowering from mjcf {robot.assets.mjcf}",
        gravity_xyz_m_s2=(_r(gravity[0]), _r(gravity[1]), _r(gravity[2])),
        gain_per_s=gain_per_s,
        coulomb_velocity_rad_s=coulomb_velocity_rad_s,
        limit_margin_rad=limit_margin_rad,
        links=links,
        joints=joints,
    )


def _zero_pose_model(robot: RobotDescription, manifest_dir: Path | None) -> tuple[Any, Any]:
    """The robot's compiled MJCF and its data at q = 0 (every hinge / slide joint)."""
    import mujoco
    from openral_core.assets import AssetRefError, resolve_asset

    if not robot.assets.mjcf:
        raise ROSConfigError(f"{robot.name}: no assets.mjcf to lower the force observer from")
    try:
        mjcf_path = resolve_asset(robot.assets.mjcf, "mjcf", manifest_dir=manifest_dir)
    except AssetRefError as exc:
        raise ROSConfigError(f"{robot.name}: {exc}") from exc
    if mjcf_path is None:
        raise ROSConfigError(f"{robot.name}: assets.mjcf={robot.assets.mjcf!r} did not resolve")
    m: Any = mujoco.MjModel.from_xml_path(str(mjcf_path))
    d: Any = mujoco.MjData(m)
    hinge_slide = (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE))
    for j in range(m.njnt):
        if int(m.jnt_type[j]) in hinge_slide:
            d.qpos[m.jnt_qposadr[j]] = 0.0
    mujoco.mj_kinematics(m, d)
    return m, d


def _body_tf(d: Any, b: int) -> Any:
    import numpy as np

    tf = np.eye(4)
    tf[:3, :3] = d.xmat[b].reshape(3, 3)
    tf[:3, 3] = d.xpos[b]
    return tf


def _checked_link_bodies(
    robot: RobotDescription, m: Any, d: Any
) -> tuple[list[str], dict[str, int], Any]:
    """(tree order, manifest link → MJCF body, MJCF world ← kinematic root), frames proven equal."""
    import mujoco
    import numpy as np

    # Manifest FK at q = 0: compose the edge origins down the tree.
    ordered, _, edge_of_child = _ordered_collision_links(robot)
    manifest_tf: dict[str, Any] = {}
    moves: dict[str, bool] = {}
    for name in ordered:
        edge = edge_of_child.get(name)
        if edge is None:
            manifest_tf[name], moves[name] = np.eye(4), False
            continue
        local = _xyzrpy_matrix((*edge.origin_xyz, *edge.origin_rpy))
        manifest_tf[name] = manifest_tf[edge.parent_link] @ local
        moves[name] = moves[edge.parent_link] or edge.dof_index >= 0

    link_body = {ln: b for ln, b in _mjcf_link_bodies(robot, m).items() if ln in manifest_tf}
    if missing := sorted(ln for ln in ordered if moves[ln] and ln not in link_body):
        raise ROSConfigError(
            f"{robot.name}: moving links {missing} have no MJCF body; their mass would be lost"
        )
    for ln, b in link_body.items():
        for j in range(int(m.body_jntadr[b]), int(m.body_jntadr[b]) + int(m.body_jntnum[b])):
            if float(np.linalg.norm(m.jnt_pos[j])) > _FRAME_TOL:
                name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
                raise ROSConfigError(
                    f"{robot.name}: MJCF joint {name!r} of link {ln!r} sits off its body "
                    "origin; the kernel rotates links about their frame origin"
                )
    # One rigid root transform must explain every mapped body (frame check).
    anchor = next(iter(link_body))
    world_root = _body_tf(d, link_body[anchor]) @ np.linalg.inv(manifest_tf[anchor])
    for ln, b in link_body.items():
        err = np.abs(world_root @ manifest_tf[ln] - _body_tf(d, b)).max()
        if err > _FRAME_TOL:
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b)
            raise ROSConfigError(
                f"{robot.name}: link {ln!r} frame differs from MJCF body {name!r} by "
                f"{err:.3g} at q = 0"
            )
    return ordered, link_body, world_root


def _fold_bodies(
    robot: RobotDescription, m: Any, d: Any, ordered: list[str], link_body: dict[str, int]
) -> list[tuple[str, list[tuple[float, Any, Any]]]]:
    """Every massive MJCF body as (mass, world COM, world inertia), grouped by manifest link."""
    import mujoco

    body_link = {b: ln for ln, b in link_body.items()}
    parts: dict[str, list[tuple[float, Any, Any]]] = {}
    for b in range(1, m.nbody):
        owner, walk, welded = None, b, True
        while walk > 0:
            if walk in body_link:
                owner = body_link[walk]
                break
            welded = welded and int(m.body_jntnum[walk]) == 0
            walk = int(m.body_parentid[walk])
        if owner is None:
            continue  # world-attached, not part of the manifest's tree
        if not welded:
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b)
            raise ROSConfigError(
                f"{robot.name}: MJCF body {name!r} moves on its own joint under link "
                f"{owner!r}, which the manifest does not model"
            )
        mass = float(m.body_mass[b])
        if mass > 0.0:
            rot = d.ximat[b].reshape(3, 3)  # principal inertia axes, world frame
            inertia = rot @ (m.body_inertia[b][:, None] * rot.T)
            parts.setdefault(owner, []).append((mass, d.xipos[b].copy(), inertia))
    return [(ln, parts[ln]) for ln in ordered if ln in parts]


def _link_inertial(link: str, parts: list[tuple[float, Any, Any]], frame: Any) -> LinkInertial:
    """Combine ``parts`` (parallel axis) and express them in the link ``frame``."""
    import numpy as np

    total = sum(p[0] for p in parts)
    com_world = sum(p[0] * p[1] for p in parts) / total
    inertia_world = np.zeros((3, 3))
    for mass, com, inertia in parts:
        r = com - com_world
        inertia_world += inertia + mass * (float(r @ r) * np.eye(3) - np.outer(r, r))
    rot = frame[:3, :3]
    com = rot.T @ (com_world - frame[:3, 3])
    i = rot.T @ inertia_world @ rot
    return LinkInertial(
        link=link,
        mass_kg=_r(total),
        com_xyz_m=(_r(com[0]), _r(com[1]), _r(com[2])),
        inertia_kgm2=(_r(i[0, 0]), _r(i[1, 1]), _r(i[2, 2]), _r(i[0, 1]), _r(i[0, 2]), _r(i[1, 2])),
    )


def _r(x: Any) -> float:
    """Round to 1e-9 (nanometre / nano-kg·m²) so the manifest diff stays readable."""
    return round(float(x), 9) + 0.0  # + 0.0 normalises -0.0
