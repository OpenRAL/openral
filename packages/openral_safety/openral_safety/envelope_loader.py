"""Envelope loader — Python helper that bridges Pydantic to the C++ kernel.

The C++ safety kernel (``cpp/openral_safety_kernel/``, ADR-0020) needs a
robot ceiling + (optional) skill envelope intersection at ``configure()``
time. Re-implementing Pydantic validation in C++ would duplicate the
source-of-truth schema (CLAUDE.md §1.3) and create drift; instead, this
Python helper reads the Pydantic manifests once, validates the
intersection, and converts the result to a ROS-parameter dict that the
kernel reads via :func:`load_envelope_from_ros_parameters` (ADR-0020
PR-K, 2026-05-24).

The legacy flat-YAML envelope-file path the kernel used pre-PR-K is
gone — there is exactly one transport: ROS parameters.

ADR-0018 §5 contract enforced here:

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
    CapsuleShape,
    JointSpec,
    JointType,
    LinkCollisionGeometry,
    RobotDescription,
    RSkillManifest,
    SafetyEnvelope,
)
from openral_core.exceptions import ROSConfigError

__all__ = [
    "EnvelopeIntersection",
    "collision_params_from_description",
    "compute_intersection",
    "ee_link_index_from_collision_params",
    "kernel_params_from_envelope",
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
            "rSkill envelope declared one of workspace_box_{min,max}_xyz "
            "but not the other; both must be set together."
        )
    axes = ("x", "y", "z")
    for i, axis in enumerate(axes):
        if skill_min[i] < robot_min[i] - 1e-9:
            raise ROSConfigError(
                f"rSkill workspace_box_min_xyz[{axis}]={skill_min[i]!r} "
                f"loosens the robot ceiling "
                f"workspace_box_min_xyz[{axis}]={robot_min[i]!r}; "
                "skill envelope must be contained in the robot box."
            )
        if skill_max[i] > robot_max[i] + 1e-9:
            raise ROSConfigError(
                f"rSkill workspace_box_max_xyz[{axis}]={skill_max[i]!r} "
                f"loosens the robot ceiling "
                f"workspace_box_max_xyz[{axis}]={robot_max[i]!r}."
            )


def _check_scalar_not_loosened(
    field: str,
    skill_value: float,
    robot_value: float,
) -> None:
    """Raise when ``skill_value > robot_value`` on a ``max_*`` field."""
    if skill_value > robot_value + 1e-9:
        raise ROSConfigError(
            f"rSkill envelope {field}={skill_value!r} loosens robot ceiling "
            f"{field}={robot_value!r}; skill envelope must be tighter "
            "or equal to the robot ceiling (ADR-0018 §5)."
        )


def compute_intersection(
    robot: RobotDescription,
    skill: RSkillManifest | None,
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

    Returns:
        An :class:`EnvelopeIntersection` ready to be serialized for the
        C++ kernel.

    Raises:
        ROSConfigError: When ``skill.envelope`` loosens the robot ceiling on
            any field. The loader refuses to honor a looser envelope
            (CLAUDE.md §1.1, §1.4; ADR-0018 §5).
    """
    robot_env: SafetyEnvelope = robot.safety
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
        # Workspace box: the two corners are treated as a unit; a partial
        # declaration is itself an error inside _check_box_subset.
        if "workspace_box_min_xyz" in skill_set or "workspace_box_max_xyz" in skill_set:
            _check_box_subset(
                skill_env.workspace_box_min_xyz,
                skill_env.workspace_box_max_xyz,
                robot_env.workspace_box_min_xyz,
                robot_env.workspace_box_max_xyz,
            )
        for field in (
            "max_ee_speed_m_s",
            "max_ee_accel_m_s2",
            "max_joint_speed_factor",
            "max_force_n",
            "max_torque_nm",
            "contact_force_threshold_n",
        ):
            if field in skill_set:
                _check_scalar_not_loosened(
                    field, getattr(skill_env, field), getattr(robot_env, field)
                )
        # deadman_required: robot=True, skill=False is a loosening only
        # when the skill *explicitly* set deadman_required=False.
        if (
            "deadman_required" in skill_set
            and robot_env.deadman_required
            and not skill_env.deadman_required
        ):
            raise ROSConfigError(
                "rSkill envelope clears deadman_required while the robot "
                "ceiling requires it; loosening rejected (ADR-0018 §5)."
            )

    # Intersection: pick the tighter of each scalar, but only consider the
    # skill value when it was explicitly set.
    def _pick_min(field: str) -> float:
        r = getattr(robot_env, field)
        if skill_env is None or field not in skill_set:
            return float(r)
        return float(min(r, getattr(skill_env, field)))

    # Workspace AABB: ``robot ∩ skill`` axis-by-axis when both corners are
    # explicitly set on the skill; otherwise use the robot's box.
    skill_set_box = (
        "workspace_box_min_xyz" in skill_set
        and "workspace_box_max_xyz" in skill_set
        and skill_env is not None
        and skill_env.workspace_box_min_xyz is not None
        and skill_env.workspace_box_max_xyz is not None
    )
    if skill_set_box:
        # _check_box_subset already proved skill is inside robot (or robot
        # is unset); use the skill's box verbatim.
        ws_min = skill_env.workspace_box_min_xyz  # type: ignore[union-attr]  # reason: skill_set_box implies non-None
        ws_max = skill_env.workspace_box_max_xyz  # type: ignore[union-attr]  # reason: skill_set_box implies non-None
    else:
        ws_min = robot_env.workspace_box_min_xyz
        ws_max = robot_env.workspace_box_max_xyz

    # Joint-level limits — pull from JointSpec and pre-multiply velocity.
    factor = _pick_min("max_joint_speed_factor")
    pos_min, pos_max, vel_max, tau_max = _extract_joint_limits(robot, factor)

    # OR with the skill's deadman_required only when explicitly set.
    deadman_required = robot_env.deadman_required or (
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
    envelope exclusively from per-field ROS parameters (ADR-0020 PR-K,
    2026-05-24). This function is the canonical Python → ROS-params
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


def _ordered_collision_links(
    joints: list[JointSpec],
) -> tuple[list[str], dict[str, int], dict[str, tuple[int, JointSpec]]]:
    """Topologically order the kinematic links (every parent before its children).

    Returns the ordered link names, a name→index map, and a child-link →
    (joint index in ``joints``, JointSpec) map.
    """
    joint_of_child = {j.child_link: (idx, j) for idx, j in enumerate(joints)}
    children_of: dict[str, list[str]] = {}
    all_links: set[str] = set()
    for j in joints:
        all_links.add(j.parent_link)
        all_links.add(j.child_link)
        children_of.setdefault(j.parent_link, []).append(j.child_link)

    roots = sorted(link for link in all_links if link not in joint_of_child)
    ordered: list[str] = []
    queue = list(roots)
    while queue:
        link = queue.pop(0)
        ordered.append(link)
        queue.extend(children_of.get(link, []))
    index = {name: i for i, name in enumerate(ordered)}
    return ordered, index, joint_of_child


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
                "split it into separate links (unsupported in ADR-0030 phase 2)"
            )
            raise ROSConfigError(msg)
        capsule_of[geom.link_name] = geom
    return capsule_of


def collision_params_from_description(
    robot: RobotDescription, *, margin_m: float = 0.0
) -> dict[str, object]:
    """Flatten a robot's collision geometry into safety_kernel ROS parameters.

    Lowers :attr:`RobotDescription.collision_geometry` +
    :attr:`~RobotDescription.allowed_collision_pairs` + the kinematic chain
    (``joints`` with their ADR-0030 ``origin_xyz`` / ``origin_rpy`` / ``axis_xyz``)
    into the flat parallel arrays the C++ kernel's ``load_collision_model``
    reads. ``joints`` stays the normative kinematic source; this never parses
    URDF/MJCF — the offline lowering tool populates the joint origins + capsules
    in the manifest first.

    Links are emitted in a topological order (every parent precedes its
    children) so the kernel's forward kinematics can resolve each link from its
    already-computed parent frame. The chunk's per-row joint index for a link is
    the link-defining joint's position in ``robot.joints`` (the same ordering
    the envelope joint arrays and ``ActionChunk.flat`` use).

    Args:
        robot: The robot manifest. No collision geometry → returns
            ``{"self_collision_enabled": False}`` (the kernel runs the scalar
            envelope check only, exactly as before ADR-0030).
        margin_m: Clearance margin in metres; a pair closer than this fires
            (default ``0.0`` = collide on touch).

    Returns:
        A ROS-parameter dict to merge into :func:`kernel_params_from_envelope`'s
        output.

    Raises:
        ROSConfigError: If a link carries more than one collision primitive
            (unsupported in this version — split it into separate links), or a
            capsule references an unknown link.
    """
    if not robot.collision_geometry:
        return {"self_collision_enabled": False}

    ordered, index, joint_of_child = _ordered_collision_links(list(robot.joints))
    capsule_of = _capsules_by_link(robot, index)

    parent: list[int] = []
    joint_kind: list[int] = []
    dof_index: list[int] = []
    origin_xyzrpy: list[float] = []
    axis: list[float] = []

    for name in ordered:
        child = joint_of_child.get(name)
        if child is None:
            # Root link: no joint, identity frame.
            parent.append(-1)
            joint_kind.append(0)
            dof_index.append(-1)
            origin_xyzrpy.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            axis.extend([0.0, 0.0, 1.0])
        else:
            jidx, j = child
            parent.append(index[j.parent_link])
            joint_kind.append(_JOINT_KIND_CODE.get(j.joint_type, 0))
            dof_index.append(jidx if j.joint_type in _JOINT_KIND_CODE else -1)
            origin_xyzrpy.extend([float(v) for v in (*j.origin_xyz, *j.origin_rpy)])
            axis.extend([float(v) for v in j.axis_xyz])

    # Capsules are a flat per-capsule list tagged with their link index (a link
    # may carry zero or — once the manifest supports it — several).
    capsule_link: list[int] = []
    capsule_radius: list[float] = []
    capsule_half_length: list[float] = []
    capsule_origin_xyzrpy: list[float] = []
    for name in ordered:
        cap = capsule_of.get(name)
        if cap is None:
            continue
        shape = cap.shape
        half_length = shape.length_m / 2.0 if isinstance(shape, CapsuleShape) else 0.0
        capsule_link.append(index[name])
        capsule_radius.append(float(shape.radius_m))
        capsule_half_length.append(float(half_length))
        capsule_origin_xyzrpy.extend([float(v) for v in cap.origin_xyz_rpy])

    allowed_pairs: list[int] = []
    for a, b in robot.allowed_collision_pairs:
        if a in index and b in index:
            allowed_pairs.extend([index[a], index[b]])

    return {
        "self_collision_enabled": True,
        "self_collision_margin_m": float(margin_m),
        "collision_n_links": len(ordered),
        "collision_parent": parent,
        "collision_joint_kind": joint_kind,
        "collision_dof_index": dof_index,
        "collision_origin_xyzrpy": origin_xyzrpy,
        "collision_axis": axis,
        "collision_capsule_link": capsule_link,
        "collision_capsule_radius": capsule_radius,
        "collision_capsule_half_length": capsule_half_length,
        "collision_capsule_origin_xyzrpy": capsule_origin_xyzrpy,
        "collision_allowed_pairs": allowed_pairs,
        "collision_link_names": ordered,
    }


def ee_link_index_from_collision_params(params: Mapping[str, object]) -> int:
    """Pick the predictive-Cartesian end-effector link (ADR-0040 Phase 3).

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
