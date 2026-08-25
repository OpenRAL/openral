"""Envelope loader — Python helper that bridges Pydantic to the C++ kernel.

The C++ safety kernel (``cpp/openral_safety_kernel/``) needs a
robot ceiling + (optional) skill envelope intersection at ``configure()``
time. Re-implementing Pydantic validation in C++ would duplicate the
source-of-truth schema (CLAUDE.md §1.3) and create drift; instead, this
Python helper reads the Pydantic manifests once, validates the
intersection, and converts the result to a ROS-parameter dict that the
kernel reads via :func:`load_envelope_from_ros_parameters`
(added 2026-05-24).

The legacy flat-YAML envelope-file path the kernel used before this
transport is gone — there is exactly one transport: ROS parameters.

The safety envelope contract enforced here:

* The robot manifest declares the **ceiling**.
* Each rSkill manifest may declare a **tighter envelope**.
* **Loosening beyond the robot ceiling is rejected at goal-acceptance**
  (never silently honored) — :func:`compute_intersection` raises
  :class:`~openral_core.exceptions.ROSConfigError`.

CLAUDE.md §1.4 ("Explicit beats implicit"): the loader rejects, never
clamps. A skill that asks for a max force higher than the robot's
ceiling fails to load.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping
from typing import cast

from openral_core import (
    BoxShape,
    CapsuleShape,
    JointType,
    LinkCollisionGeometry,
    RobotDescription,
    RSkillManifest,
    SafetyEnvelope,
    SphereShape,
)
from openral_core.exceptions import ROSConfigError

__all__ = [
    "EnvelopeIntersection",
    "collision_params_from_description",
    "compute_intersection",
    "ee_link_index_from_collision_params",
    "kernel_params_from_envelope",
    "merge_deploy_envelope",
    "merge_extra_allowed_pairs",
]


@dataclasses.dataclass(frozen=True)
class EnvelopeIntersection:
    """The numerical product of ``robot.safety ∩ skill.envelope``.

    Every field is a *single* numeric or array value — no nested
    Pydantic models, no Python-only types. This is the bridge surface
    the C++ kernel reads; keep it boring.

    Attributes:
        robot_name: ``RobotDescription.name`` — for diagnostic logs.
        rskill_id: ``RSkillManifest.name`` — for diagnostic logs (or
            ``""`` when no skill is loaded yet).
        rskill_revision: ``RSkillManifest.version`` (or ``""``).
        n_dof: Number of revolute / prismatic joints in the robot.
        joint_position_min: Per-joint lower bound (rad or m).
        joint_position_max: Per-joint upper bound (rad or m).
        joint_velocity_max: Per-joint max |velocity|, already pre-multiplied
            by :attr:`SafetyEnvelope.max_joint_speed_factor`.
        joint_torque_max: Per-joint max |effort| (Nm or N).
        workspace_box_min_xyz: Cartesian workspace AABB lower corner; ``None``
            when both the robot and the skill leave it unset.
        workspace_box_max_xyz: AABB upper corner; symmetric.
        max_ee_speed_m_s: Cartesian end-effector speed cap.
        max_ee_accel_m_s2: Cartesian end-effector acceleration cap.
        max_force_n: External force cap (Newtons).
        max_torque_nm: External torque cap (Nm).
        contact_force_threshold_n: Below this, no contact; above, contact.
        deadman_required: Logical OR of the two manifests.
    """

    robot_name: str
    rskill_id: str
    rskill_revision: str
    n_dof: int
    joint_position_min: tuple[float, ...]
    joint_position_max: tuple[float, ...]
    joint_velocity_max: tuple[float, ...]
    joint_torque_max: tuple[float, ...]
    workspace_box_min_xyz: tuple[float, float, float] | None
    workspace_box_max_xyz: tuple[float, float, float] | None
    max_ee_speed_m_s: float
    max_ee_accel_m_s2: float
    max_force_n: float
    max_torque_nm: float
    contact_force_threshold_n: float
    deadman_required: bool


# Joint kinds that have a meaningful position / velocity / torque limit
# the kernel can enforce. Fixed joints have no DoF; we skip them.
_ACTUATED_JOINT_TYPES: frozenset[str] = frozenset({"revolute", "prismatic", "continuous"})


def _extract_joint_limits(
    robot: RobotDescription,
    skill_max_joint_speed_factor: float,
) -> tuple[
    tuple[float, ...],
    tuple[float, ...],
    tuple[float, ...],
    tuple[float, ...],
]:
    """Pull per-joint min/max position, max |velocity|, max |torque|.

    Velocity is pre-multiplied by ``skill_max_joint_speed_factor`` so the
    kernel does a single ``|v| > limit`` check per joint without needing
    to know about the factor.

    Joints without explicit limits get sentinel values — ``-inf`` for
    lower bounds, ``+inf`` for upper bounds, and ``+inf`` for velocity /
    torque caps — which the C++ kernel treats as "no enforcement on this
    joint." This matches how URDF/MJCF treats missing ``<limit>`` tags.
    """
    pos_min: list[float] = []
    pos_max: list[float] = []
    vel_max: list[float] = []
    tau_max: list[float] = []
    for j in robot.joints:
        # ``j.joint_type`` is the typed Enum; compare on .value to stay loose.
        if j.joint_type.value not in _ACTUATED_JOINT_TYPES:
            continue
        if j.position_limits is not None:
            pos_min.append(float(j.position_limits[0]))
            pos_max.append(float(j.position_limits[1]))
        else:
            pos_min.append(-math.inf)
            pos_max.append(math.inf)
        v = j.velocity_limit if j.velocity_limit is not None else math.inf
        vel_max.append(float(v) * float(skill_max_joint_speed_factor))
        tau = j.effort_limit if j.effort_limit is not None else math.inf
        tau_max.append(float(tau))
    return (tuple(pos_min), tuple(pos_max), tuple(vel_max), tuple(tau_max))


def _check_box_subset(
    skill_min: tuple[float, float, float] | None,
    skill_max: tuple[float, float, float] | None,
    robot_min: tuple[float, float, float] | None,
    robot_max: tuple[float, float, float] | None,
    *,
    label: str = "rSkill",
) -> None:
    """Raise ROSConfigError if the skill's workspace box loosens the robot's.

    ``robot_*`` is the ceiling; ``skill_*`` is the requested floor. The
    skill must declare a workspace **at most as large** as the robot's
    on every axis (i.e. ``skill_min >= robot_min`` and ``skill_max <=
    robot_max`` componentwise). When the robot leaves the box unset, the
    skill may declare anything (the ceiling is unbounded).
    """
    if skill_min is None and skill_max is None:
        return
    if robot_min is None or robot_max is None:
        # The robot leaves the box unset: anything goes; nothing to compare.
        return
    if skill_min is None or skill_max is None:
        raise ROSConfigError(
            f"{label} envelope declared one of workspace_box_{{min,max}}_xyz "
            "but not the other; both must be set together."
        )
    axes = ("x", "y", "z")
    for i, axis in enumerate(axes):
        if skill_min[i] < robot_min[i] - 1e-9:
            raise ROSConfigError(
                f"{label} workspace_box_min_xyz[{axis}]={skill_min[i]!r} "
                f"loosens the robot ceiling "
                f"workspace_box_min_xyz[{axis}]={robot_min[i]!r}; "
                f"{label} envelope must be contained in the robot box."
            )
        if skill_max[i] > robot_max[i] + 1e-9:
            raise ROSConfigError(
                f"{label} workspace_box_max_xyz[{axis}]={skill_max[i]!r} "
                f"loosens the robot ceiling "
                f"workspace_box_max_xyz[{axis}]={robot_max[i]!r}."
            )


def _check_scalar_not_loosened(
    field: str,
    skill_value: float,
    robot_value: float,
    *,
    label: str = "rSkill",
) -> None:
    """Raise when ``skill_value > robot_value`` on a ``max_*`` field."""
    if skill_value > robot_value + 1e-9:
        raise ROSConfigError(
            f"{label} envelope {field}={skill_value!r} loosens robot ceiling "
            f"{field}={robot_value!r}; {label} envelope must be tighter "
            "or equal to the robot ceiling."
        )


def _validate_envelope_tightens(
    ceiling: SafetyEnvelope,
    candidate: SafetyEnvelope,
    explicit_fields: frozenset[str],
    *,
    label: str,
) -> None:
    if "workspace_box_min_xyz" in explicit_fields or "workspace_box_max_xyz" in explicit_fields:
        _check_box_subset(
            candidate.workspace_box_min_xyz,
            candidate.workspace_box_max_xyz,
            ceiling.workspace_box_min_xyz,
            ceiling.workspace_box_max_xyz,
            label=label,
        )
    for field in (
        "max_ee_speed_m_s",
        "max_ee_accel_m_s2",
        "max_joint_speed_factor",
        "max_force_n",
        "max_torque_nm",
        "contact_force_threshold_n",
    ):
        if field in explicit_fields:
            _check_scalar_not_loosened(
                field, getattr(candidate, field), getattr(ceiling, field), label=label
            )
    if (
        "deadman_required" in explicit_fields
        and ceiling.deadman_required
        and not candidate.deadman_required
    ):
        raise ROSConfigError(
            f"{label} envelope clears deadman_required while the robot ceiling "
            "requires it; loosening rejected."
        )


def merge_deploy_envelope(
    robot_env: SafetyEnvelope, deploy: SafetyEnvelope | None
) -> SafetyEnvelope:
    """Apply explicit deploy/workcell safety fields to the robot ceiling.

    Only fields explicitly present in the deploy YAML are considered; omitted
    fields keep the robot manifest's tighter values instead of resetting to
    ``SafetyEnvelope`` schema defaults.
    """
    if deploy is None:
        return robot_env
    deploy_set = frozenset(deploy.model_fields_set)
    _validate_envelope_tightens(robot_env, deploy, deploy_set, label="deploy")
    updates = {field: getattr(deploy, field) for field in deploy_set}
    return robot_env.model_copy(update=updates)


def _intersect_workspace_boxes(
    base_min: tuple[float, float, float] | None,
    base_max: tuple[float, float, float] | None,
    skill_min: tuple[float, float, float] | None,
    skill_max: tuple[float, float, float] | None,
) -> tuple[tuple[float, float, float] | None, tuple[float, float, float] | None]:
    if skill_min is None and skill_max is None:
        return base_min, base_max
    if base_min is None or base_max is None:
        return skill_min, skill_max
    if skill_min is None or skill_max is None:
        raise ROSConfigError(
            "rSkill envelope declared one of workspace_box_{min,max}_xyz "
            "but not the other; both must be set together."
        )
    out_min = tuple(max(base_min[i], skill_min[i]) for i in range(3))
    out_max = tuple(min(base_max[i], skill_max[i]) for i in range(3))
    for i, axis in enumerate(("x", "y", "z")):
        if out_min[i] > out_max[i] + 1e-9:
            raise ROSConfigError(
                f"workspace intersection is empty on {axis}: min={out_min[i]!r}, "
                f"max={out_max[i]!r}."
            )
    return out_min, out_max


def compute_intersection(
    robot: RobotDescription,
    skill: RSkillManifest | None,
    *,
    deploy: SafetyEnvelope | None = None,
) -> EnvelopeIntersection:
    """Return the validated intersection of a robot ceiling and a skill envelope.

    When ``skill`` is ``None`` or its ``envelope`` is unset, the
    intersection is simply the robot ceiling (no per-skill tightening).
    Otherwise every scalar ``max_*`` field is intersected with ``min(robot,
    skill)``; the workspace AABB is intersected via the
    ``[max(mins), min(maxes)]`` rule; ``deadman_required`` is the logical
    OR of the two.

    Args:
        robot: The robot manifest (the ceiling).
        skill: Optional rSkill manifest (the optional tighter envelope).
        deploy: Optional deploy-scene workcell envelope; explicit fields must
            tighten the robot ceiling before skill intersection.

    Returns:
        An :class:`EnvelopeIntersection` ready to be serialized for the
        C++ kernel.

    Raises:
        ROSConfigError: When ``skill.envelope`` loosens the robot ceiling on
            any field. The loader refuses to honor a looser envelope
            (CLAUDE.md §1.1, §1.4).
    """
    robot_env: SafetyEnvelope = robot.safety
    merged_env = merge_deploy_envelope(robot_env, deploy)
    skill_env: SafetyEnvelope | None = skill.envelope if skill is not None else None

    # ``model_fields_set`` tells us which fields the user *explicitly set*
    # on the skill manifest vs which fields took the SafetyEnvelope schema
    # default. We only validate / apply fields that were explicitly set —
    # a skill that only declares ``max_force_n`` does NOT silently
    # override the robot's tighter ``max_torque_nm`` with the schema
    # default of 10 Nm.
    skill_set: frozenset[str] = (
        frozenset(skill_env.model_fields_set) if skill_env is not None else frozenset()
    )

    # Validate the skill envelope first — if it loosens the robot, fail loudly.
    if skill_env is not None:
        _validate_envelope_tightens(robot_env, skill_env, skill_set, label="rSkill")

    # Intersection: pick the tighter of each scalar, but only consider the
    # skill value when it was explicitly set.
    def _pick_min(field: str) -> float:
        r = getattr(merged_env, field)
        if skill_env is None or field not in skill_set:
            return float(r)
        return float(min(r, getattr(skill_env, field)))

    # Workspace AABB: ``robot/deploy ∩ skill`` axis-by-axis when both corners are
    # explicitly set on the skill; otherwise use the deploy-tightened robot box.
    skill_set_box = (
        "workspace_box_min_xyz" in skill_set
        and "workspace_box_max_xyz" in skill_set
        and skill_env is not None
        and skill_env.workspace_box_min_xyz is not None
        and skill_env.workspace_box_max_xyz is not None
    )
    if skill_set_box:
        ws_min, ws_max = _intersect_workspace_boxes(
            merged_env.workspace_box_min_xyz,
            merged_env.workspace_box_max_xyz,
            skill_env.workspace_box_min_xyz,  # type: ignore[union-attr]  # reason: skill_set_box implies non-None
            skill_env.workspace_box_max_xyz,  # type: ignore[union-attr]  # reason: skill_set_box implies non-None
        )
    else:
        ws_min = merged_env.workspace_box_min_xyz
        ws_max = merged_env.workspace_box_max_xyz

    # Joint-level limits — pull from JointSpec and pre-multiply velocity.
    factor = _pick_min("max_joint_speed_factor")
    pos_min, pos_max, vel_max, tau_max = _extract_joint_limits(robot, factor)

    # OR with the skill's deadman_required only when explicitly set.
    deadman_required = merged_env.deadman_required or (
        skill_env.deadman_required
        if skill_env is not None and "deadman_required" in skill_set
        else False
    )

    return EnvelopeIntersection(
        robot_name=robot.name,
        rskill_id=skill.name if skill is not None else "",
        rskill_revision=skill.version if skill is not None else "",
        n_dof=len(pos_min),
        joint_position_min=pos_min,
        joint_position_max=pos_max,
        joint_velocity_max=vel_max,
        joint_torque_max=tau_max,
        workspace_box_min_xyz=ws_min,
        workspace_box_max_xyz=ws_max,
        max_ee_speed_m_s=_pick_min("max_ee_speed_m_s"),
        max_ee_accel_m_s2=_pick_min("max_ee_accel_m_s2"),
        max_force_n=_pick_min("max_force_n"),
        max_torque_nm=_pick_min("max_torque_nm"),
        contact_force_threshold_n=_pick_min("contact_force_threshold_n"),
        deadman_required=deadman_required,
    )


def kernel_params_from_envelope(envelope: EnvelopeIntersection) -> dict[str, object]:
    """Translate :class:`EnvelopeIntersection` → safety_kernel ROS parameters.

    The C++ safety kernel (``cpp/openral_safety_kernel/``) reads its
    envelope exclusively from per-field ROS parameters (added 2026-05-24).
    This function is the canonical Python → ROS-params
    converter — used by ``openral deploy sim``'s ``sim_e2e.launch.py`` to
    feed the kernel from ``robots/<id>/robot.yaml``, by ``kernel_only``
    launches, and by every C++ / Python kernel test fixture.

    Workspace-box corners are *omitted* (not passed as empty arrays)
    when unset — launch_ros's parameter validator rejects empty
    ``double_array`` parameters, and the kernel already declares them
    defaulted-empty so omission has the same "unbounded Cartesian
    envelope" semantics.

    Args:
        envelope: Validated envelope intersection (typically from
            :func:`compute_intersection`).

    Returns:
        A dict mapping each ROS parameter name to a value of the right
        type, ready to plug into a ``LifecycleNode(parameters=[…])``
        list or a ``rclcpp::NodeOptions.parameter_overrides({…})``
        block.

    Raises:
        ValueError: If any scalar field is NaN — a bug in
            ``compute_intersection`` we want to surface, not propagate.
    """
    if any(
        math.isnan(value)
        for value in (
            envelope.max_ee_speed_m_s,
            envelope.max_ee_accel_m_s2,
            envelope.max_force_n,
            envelope.max_torque_nm,
            envelope.contact_force_threshold_n,
        )
    ):
        raise ValueError(f"NaN scalar in envelope: {envelope!r}")

    params: dict[str, object] = {
        "n_dof": int(envelope.n_dof),
        "robot_name": envelope.robot_name,
        "rskill_id": envelope.rskill_id,
        "skill_revision": envelope.rskill_revision,
        "joint_position_min": [float(v) for v in envelope.joint_position_min],
        "joint_position_max": [float(v) for v in envelope.joint_position_max],
        "joint_velocity_max": [float(v) for v in envelope.joint_velocity_max],
        "joint_torque_max": [float(v) for v in envelope.joint_torque_max],
        "max_ee_speed_m_s": float(envelope.max_ee_speed_m_s),
        "max_ee_accel_m_s2": float(envelope.max_ee_accel_m_s2),
        "max_force_n": float(envelope.max_force_n),
        "max_torque_nm": float(envelope.max_torque_nm),
        "contact_force_threshold_n": float(envelope.contact_force_threshold_n),
        "deadman_required": bool(envelope.deadman_required),
    }
    if envelope.workspace_box_min_xyz is not None:
        params["workspace_box_min_xyz"] = [float(v) for v in envelope.workspace_box_min_xyz]
    if envelope.workspace_box_max_xyz is not None:
        params["workspace_box_max_xyz"] = [float(v) for v in envelope.workspace_box_max_xyz]
    return params


_JOINT_KIND_CODE = {
    JointType.REVOLUTE: 1,
    JointType.CONTINUOUS: 1,
    JointType.PRISMATIC: 2,
}


@dataclasses.dataclass(frozen=True)
class _Edge:
    """One parent→child edge of the collision tree.

    ``dof_index`` is the edge's column in ``RobotDescription.joints`` for a
    movable joint, or ``-1`` for a rigid :class:`~openral_core.FixedAttachment`
    (which has no commanded column). ``joint_type`` is
    :attr:`~openral_core.JointType.FIXED` for an attachment, so the existing
    ``_JOINT_KIND_CODE`` lookup already lowers it to the kernel's static kind.
    """

    parent_link: str
    joint_type: JointType
    dof_index: int
    origin_xyz: tuple[float, float, float]
    origin_rpy: tuple[float, float, float]
    axis_xyz: tuple[float, float, float]


def _ordered_collision_links(
    robot: RobotDescription,
) -> tuple[list[str], dict[str, int], dict[str, _Edge]]:
    """Topologically order the kinematic links (every parent before its children).

    The tree is the union of the robot's **movable** ``joints`` and its rigid
    ``fixed_attachments``. Both are needed: ``joints`` alone is only the
    actuated skeleton, so a robot with any rigid mount (a Franka hand on the
    flange, a bimanual rig's arm pedestals) would otherwise present as a
    *forest* of several disconnected components.

    A forest has no safe interpretation. Treating a component's top link as a
    second base — the behaviour this function replaced — silently places that
    entire subtree at the robot's origin, which both fabricates contacts that
    cannot happen and, far worse, leaves the subtree's real swept volume
    completely unmodelled. So this refuses, naming the disconnected roots
    (CLAUDE.md §1.1, §1.4: the loader rejects, never guesses).

    Returns the ordered link names, a name→index map, and a child-link → edge
    map.

    Raises:
        ROSConfigError: When the collision links do not form exactly one
            connected tree — several roots (missing ``fixed_attachments``), or
            a cycle.
    """
    edge_of_child: dict[str, _Edge] = {
        j.child_link: _Edge(
            parent_link=j.parent_link,
            joint_type=j.joint_type,
            dof_index=idx,
            origin_xyz=j.origin_xyz,
            origin_rpy=j.origin_rpy,
            axis_xyz=j.axis_xyz,
        )
        for idx, j in enumerate(robot.joints)
    }
    # ``RobotDescription._validate_fixed_attachments`` already rejects an
    # attachment that re-defines a link a joint defines, so this cannot clobber.
    for att in robot.fixed_attachments:
        edge_of_child[att.child_link] = _Edge(
            parent_link=att.parent_link,
            joint_type=JointType.FIXED,
            dof_index=-1,
            origin_xyz=att.origin_xyz,
            origin_rpy=att.origin_rpy,
            axis_xyz=(0.0, 0.0, 1.0),
        )

    children_of: dict[str, list[str]] = {}
    all_links: set[str] = set()
    for child, edge in edge_of_child.items():
        all_links.add(edge.parent_link)
        all_links.add(child)
        children_of.setdefault(edge.parent_link, []).append(child)
    for kids in children_of.values():
        kids.sort()

    roots = sorted(link for link in all_links if link not in edge_of_child)
    if not roots:
        raise ROSConfigError(
            f"robot {robot.name!r}: every collision link has a parent, so the "
            "kinematic chain contains a cycle and no link can be placed relative "
            "to a base."
        )
    if len(roots) != 1:
        detail = ", ".join(f"{r!r} ({len(_component(r, children_of))} links)" for r in roots)
        raise ROSConfigError(
            f"robot {robot.name!r}: collision links form {len(roots)} disconnected "
            f"trees, not one — roots: {detail}. Every link must be placeable "
            "relative to the robot's base, or the safety kernel cannot know where "
            "it sweeps. 'joints' enumerates only movable joints, so declare each "
            "rigid mount in 'fixed_attachments' (name, parent_link, child_link, "
            "origin_xyz, origin_rpy), taking the transform from the robot's real "
            "URDF/MJCF at the zero configuration — never estimate one."
        )

    ordered: list[str] = []
    queue = list(roots)
    while queue:
        link = queue.pop(0)
        ordered.append(link)
        queue.extend(children_of.get(link, []))
    if len(ordered) != len(all_links):
        orphaned = sorted(all_links - set(ordered))
        raise ROSConfigError(
            f"robot {robot.name!r}: collision links {orphaned} are unreachable from "
            f"root {roots[0]!r} — the kinematic chain contains a cycle."
        )
    index = {name: i for i, name in enumerate(ordered)}
    return ordered, index, edge_of_child


def _component(root: str, children_of: dict[str, list[str]]) -> list[str]:
    """Links reachable from ``root`` — used only to size a diagnostic message."""
    seen: list[str] = []
    queue = [root]
    while queue:
        link = queue.pop(0)
        seen.append(link)
        queue.extend(children_of.get(link, []))
    return seen


def _capsules_by_link(
    robot: RobotDescription, index: dict[str, int]
) -> dict[str, LinkCollisionGeometry]:
    """Map each link to its single collision primitive, validating references."""
    capsule_of: dict[str, LinkCollisionGeometry] = {}
    for geom in robot.collision_geometry:
        if geom.link_name not in index:
            msg = f"collision_geometry references unknown link {geom.link_name!r}"
            raise ROSConfigError(msg)
        if geom.link_name in capsule_of:
            msg = (
                f"link {geom.link_name!r} has >1 collision primitive; "
                "split it into separate links (unsupported in this lowering phase)"
            )
            raise ROSConfigError(msg)
        capsule_of[geom.link_name] = geom
    return capsule_of


def collision_params_from_description(  # noqa: PLR0912, PLR0915
    robot: RobotDescription, *, margin_m: float | None = None
) -> dict[str, object]:
    """Flatten a robot's collision geometry into safety_kernel ROS parameters.

    Lowers :attr:`RobotDescription.collision_geometry` +
    :attr:`~RobotDescription.allowed_collision_pairs` + the kinematic chain
    (``joints`` **and** :attr:`~RobotDescription.fixed_attachments`, with their
    ``origin_xyz`` / ``origin_rpy`` / ``axis_xyz``) into the flat parallel
    arrays the C++ kernel's ``load_collision_model`` reads. The manifest stays
    the normative kinematic source; this never parses URDF/MJCF — the offline
    lowering tool populates the joint origins + capsules in the manifest first.

    The two lists together must describe exactly **one** connected tree. Any
    link a robot's movable ``joints`` do not reach — a rigidly mounted hand, a
    bimanual rig's arm pedestals — belongs in ``fixed_attachments``; a manifest
    that leaves it out is rejected rather than lowered with that subtree
    dumped at the origin.

    Links are emitted in a topological order (every parent precedes its
    children) so the kernel's forward kinematics can resolve each link from its
    already-computed parent frame. The chunk's per-row joint index for a link is
    the link-defining joint's position in ``robot.joints`` (the same ordering
    the envelope joint arrays and ``ActionChunk.flat`` use).

    Args:
        robot: The robot manifest. No collision geometry → returns
            ``{"self_collision_enabled": False}`` (the kernel runs the scalar
            envelope check only, exactly as before this lowering was added).
        margin_m: Clearance margin in metres; a pair closer than this fires
            (default ``0.0`` = collide on touch).

    Returns:
        A ROS-parameter dict to merge into :func:`kernel_params_from_envelope`'s
        output.

    Raises:
        ROSConfigError: If the collision links do not form exactly one
            connected tree (missing ``fixed_attachments``, or a cycle); if a
            link carries more than one collision primitive (unsupported in this
            version — split it into separate links); or if a capsule references
            an unknown link.
    """
    if not robot.collision_geometry:
        return {"self_collision_enabled": False}

    # An explicit margin_m arg overrides; otherwise use the manifest's
    # safety.self_collision_margin_m (default 0.0 = collide on touch).
    if margin_m is None:
        margin_m = float(getattr(robot.safety, "self_collision_margin_m", 0.0) or 0.0)

    ordered, index, edge_of_child = _ordered_collision_links(robot)
    capsule_of = _capsules_by_link(robot, index)

    parent: list[int] = []
    joint_kind: list[int] = []
    dof_index: list[int] = []
    origin_xyzrpy: list[float] = []
    axis: list[float] = []

    for name in ordered:
        edge = edge_of_child.get(name)
        if edge is None:
            # The tree's single root (``_ordered_collision_links`` guarantees
            # there is exactly one): the base, at the identity frame.
            parent.append(-1)
            joint_kind.append(0)
            dof_index.append(-1)
            origin_xyzrpy.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            axis.extend([0.0, 0.0, 1.0])
        else:
            parent.append(index[edge.parent_link])
            joint_kind.append(_JOINT_KIND_CODE.get(edge.joint_type, 0))
            dof_index.append(edge.dof_index if edge.joint_type in _JOINT_KIND_CODE else -1)
            origin_xyzrpy.extend([float(v) for v in (*edge.origin_xyz, *edge.origin_rpy)])
            axis.extend([float(v) for v in edge.axis_xyz])

    # Each link's primitive is routed by shape: capsules/spheres to the capsule
    # arrays (sphere = zero-length capsule), boxes to the OBB arrays
    # (issue #84). Both are flat per-primitive lists tagged with the link index.
    capsule_link: list[int] = []
    capsule_radius: list[float] = []
    capsule_half_length: list[float] = []
    capsule_origin_xyzrpy: list[float] = []
    box_link: list[int] = []
    box_half_extents: list[float] = []
    box_origin_xyzrpy: list[float] = []
    for name in ordered:
        geom = capsule_of.get(name)
        if geom is None:
            continue
        shape = geom.shape
        if isinstance(shape, BoxShape):
            box_link.append(index[name])
            box_half_extents.extend([float(h) for h in shape.half_extents_m])
            box_origin_xyzrpy.extend([float(v) for v in geom.origin_xyz_rpy])
        elif isinstance(shape, CapsuleShape | SphereShape):
            # A sphere is a zero-length capsule; both share `radius_m`.
            half_length = shape.length_m / 2.0 if isinstance(shape, CapsuleShape) else 0.0
            capsule_link.append(index[name])
            capsule_radius.append(float(shape.radius_m))
            capsule_half_length.append(float(half_length))
            capsule_origin_xyzrpy.extend([float(v) for v in geom.origin_xyz_rpy])
        else:
            # Fail closed on a primitive this lowering does not understand.
            # The branch above used to be a bare `else`, so ANY future member of
            # `CollisionShape` was lowered as a zero-length capsule of its
            # `radius_m` — silently, and in the UNSAFE direction: a capsule of
            # radius r is contained in every non-spherical primitive that
            # carries r, so the kernel would have received a strictly SMALLER
            # volume than the manifest declared and missed real contacts inside
            # the difference. Refusing to build the params instead is
            # at-least-as-conservative by construction: no envelope is emitted,
            # so no motion is authorised against a wrong one.
            raise ROSConfigError(
                f"link {name!r} declares collision primitive {shape.shape!r}, which "
                f"{__name__} cannot lower. Add it to the capsule/box routing here "
                "(and to the kernel's distance routines) before using it in a "
                "manifest — it must never be lowered as an assumed capsule."
            )

    allowed_pairs: list[int] = []
    for a, b in robot.allowed_collision_pairs:
        if a in index and b in index:
            allowed_pairs.extend([index[a], index[b]])

    params: dict[str, object] = {
        "self_collision_enabled": True,
        "self_collision_margin_m": float(margin_m),
        "collision_n_links": len(ordered),
        "collision_parent": parent,
        "collision_joint_kind": joint_kind,
        "collision_dof_index": dof_index,
        "collision_origin_xyzrpy": origin_xyzrpy,
        "collision_axis": axis,
        "collision_link_names": ordered,
    }
    # Per-primitive arrays (capsules, boxes) and the allowed-pair list are omitted
    # when empty: launch_ros collapses an empty Python list to ``()`` and
    # ensure_argument_type rejects it. An all-box robot (SO-101) has zero
    # capsules; a capsule-only robot has zero boxes; both are valid. The kernel
    # declares its own ``[]`` default for each (same guard as
    # ``collision_base_dofs`` in sim_e2e.launch.py).
    if capsule_link:
        params["collision_capsule_link"] = capsule_link
        params["collision_capsule_radius"] = capsule_radius
        params["collision_capsule_half_length"] = capsule_half_length
        params["collision_capsule_origin_xyzrpy"] = capsule_origin_xyzrpy
    if box_link:
        params["collision_box_link"] = box_link
        params["collision_box_half_extents"] = box_half_extents
        params["collision_box_origin_xyzrpy"] = box_origin_xyzrpy
    if allowed_pairs:
        params["collision_allowed_pairs"] = allowed_pairs
    return params


def merge_extra_allowed_pairs(
    params: Mapping[str, object], pairs: list[tuple[str, str]]
) -> dict[str, object]:
    """Append deploy-scene allowed collision pairs to kernel collision params."""
    merged = dict(params)
    if not pairs or not bool(merged.get("self_collision_enabled", False)):
        return merged

    raw_names = merged.get("collision_link_names")
    if not isinstance(raw_names, list) or not all(isinstance(name, str) for name in raw_names):
        raise ROSConfigError("collision_link_names missing from self-collision params.")
    names = cast(list[str], raw_names)
    index = {name: i for i, name in enumerate(names)}

    existing_raw = merged.get("collision_allowed_pairs", [])
    if not isinstance(existing_raw, list) or not all(isinstance(i, int) for i in existing_raw):
        raise ROSConfigError("collision_allowed_pairs must be a flat list of integer indices.")
    if len(existing_raw) % 2:
        raise ROSConfigError("collision_allowed_pairs must contain index pairs.")

    allowed_pairs = list(cast(list[int], existing_raw))
    seen: set[tuple[int, int]] = set()
    for i in range(0, len(allowed_pairs), 2):
        seen.add(tuple(sorted((allowed_pairs[i], allowed_pairs[i + 1]))))

    valid = ", ".join(names)
    for a, b in pairs:
        if a == b:
            raise ROSConfigError(f"extra_allowed_collision_pairs cannot pair {a!r} with itself.")
        if a not in index or b not in index:
            raise ROSConfigError(
                f"unknown extra_allowed_collision_pairs link {a!r}<->{b!r}; "
                f"valid collision links: {valid}"
            )
        pair = tuple(sorted((index[a], index[b])))
        if pair in seen:
            continue
        seen.add(pair)
        allowed_pairs.extend([pair[0], pair[1]])

    merged["collision_allowed_pairs"] = allowed_pairs
    return merged


def ee_link_index_from_collision_params(params: Mapping[str, object]) -> int:
    """Pick the predictive-Cartesian end-effector link.

    The C++ kernel reconstructs where a ``CARTESIAN_DELTA`` chunk's EE deltas
    drive the arm using the geometric Jacobian of one *control* link. For a
    serial manipulator that link is the kinematically **deepest** collision link
    — the wrist/tip the Cartesian command moves — so we return the index with the
    longest parent chain to the root. The choice only sets the Jacobian's control
    point; the predicted configuration is still checked against the *whole* arm's
    capsules, and the kernel's reactive measured-config check is the guaranteed
    floor, so a mis-identified EE link can only weaken the *early-warning* margin,
    never make the kernel unsafe.

    Args:
        params: The dict from :func:`collision_params_from_description` (or
            :func:`~openral_safety.mjcf_lowering.lower_collision_params`).

    Returns:
        The deepest collision-link index, or ``-1`` when there is no collision
        model (predictive Cartesian stays disabled — reactive check only).
    """
    parent = cast("list[int]", params.get("collision_parent") or [])
    n = len(parent)
    if n == 0:
        return -1
    best_index = -1
    best_depth = -1
    for i in range(n):
        depth = 0
        p = parent[i]
        guard = 0
        while p is not None and p >= 0 and guard <= n:
            depth += 1
            p = parent[p]
            guard += 1
        if depth > best_depth:
            best_depth = depth
            best_index = i
    return best_index
