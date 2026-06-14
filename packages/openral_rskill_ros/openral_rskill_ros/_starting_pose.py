"""Pure starting-pose dispatch decision + MoveIt goal shaping (ADR-0053).

Before the first inference tick of an ``ExecuteSkill`` goal, the runner can move
the HAL to the rSkill's in-distribution ``starting_pose`` two ways:

* **approach** — dispatch the MoveIt ``rskill-moveit-joints`` rSkill
  (``kind: ros_action``) retargeted at ``starting_pose``: MoveIt plans a
  collision-free joint-space motion (self + planning-scene/world collision) and
  ``ROSActionRskill`` replays it per-waypoint through ``/openral/candidate_action``
  (the kernel checks every step). A failure is **fatal** — the runner aborts the
  goal rather than start the policy from an unreachable / colliding state.
* **reset** — the legacy ``ResetToPose`` snap (instantaneous ``qpos`` teleport),
  **best-effort** — a failure only warns (pre-ADR-0053 behaviour, kept for HALs
  without a MoveIt config).

This module holds only the *decision* and *pure goal-shaping* (no ROS) so it is
unit-testable. The MoveGroup dispatch itself lives in the node.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal


@dataclass(frozen=True, slots=True)
class StartingPoseAction:
    """What the runner should do to reach an rSkill's ``starting_pose``.

    Attributes:
        mode: ``"approach"`` (MoveIt rSkill), ``"reset"`` (legacy snap), or
            ``"none"`` (nothing to do).
        pose: The target joint positions (empty when ``mode == "none"``).
        fatal_on_failure: If the dispatched action fails, abort the ExecuteSkill
            goal (``True``, approach) vs. warn and continue (``False``, reset).
    """

    mode: Literal["approach", "reset", "none"]
    pose: list[float]
    fatal_on_failure: bool


def resolve_starting_pose_action(
    *,
    approach_skill_id: str,
    reset_to_pose_service: str,
    starting_pose: Sequence[float] | None,
) -> StartingPoseAction:
    """Decide how to reach ``starting_pose`` given the wired knobs.

    Prefers the MoveIt approach skill over the legacy snap; with neither wired
    (or no ``starting_pose``) there is nothing to do.

    Args:
        approach_skill_id: ``approach_skill_id`` param — the MoveIt approach
            rSkill URI (empty = not wired).
        reset_to_pose_service: ``reset_to_pose_service`` param (empty = not wired).
        starting_pose: The manifest's ``starting_pose`` (or ``None``).

    Returns:
        The :class:`StartingPoseAction` to execute.

    Example:
        >>> resolve_starting_pose_action(
        ...     approach_skill_id="rskills/rskill-moveit-joints",
        ...     reset_to_pose_service="/openral/ur5e/reset_to_pose",
        ...     starting_pose=[0.0, -1.2, 1.2, -1.0, -1.4, 0.0],
        ... ).mode
        'approach'
    """
    pose = [float(v) for v in starting_pose] if starting_pose else []
    if not pose:
        return StartingPoseAction("none", [], False)
    if approach_skill_id:
        return StartingPoseAction("approach", pose, True)
    if reset_to_pose_service:
        return StartingPoseAction("reset", pose, False)
    return StartingPoseAction("none", pose, False)


def joint_names_from_goal_json(default_goal_json: str) -> list[str]:
    """Extract the planning-group joint names from a MoveGroup ``default_goal_json``.

    Reads the ADR-0054 ``joint`` block's ``joint_names`` — the joint order the
    ``rskill-moveit-joints`` (``goal_builder: "joint"``) approach manifest
    declares for its MoveIt planning group. Used to length-check a robot's flat
    ``starting_pose`` before building the retarget override
    (:func:`moveit_joint_goal_override`).

    Args:
        default_goal_json: The approach rSkill's
            ``ros_integration.default_goal_json`` string.

    Returns:
        Joint names in the manifest's declared order.

    Raises:
        ValueError: If the JSON is malformed or lacks a ``joint.joint_names`` list.

    Example:
        >>> joint_names_from_goal_json(
        ...     '{"joint": {"joint_names": ["panda_joint1"], "positions": [0.0]}}'
        ... )
        ['panda_joint1']
    """
    try:
        goal: Any = json.loads(default_goal_json)
        names = [str(n) for n in goal["joint"]["joint_names"]]
    except (json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
        raise ValueError(
            f"approach manifest default_goal_json lacks joint.joint_names: {exc}"
        ) from exc
    if not names:
        raise ValueError("approach manifest declares no joint.joint_names to retarget.")
    return names


def moveit_joint_goal_override(joint_names: Sequence[str], positions: Sequence[float]) -> str:
    """Build the ``goal_params_json`` that retargets the joint goal at ``positions``.

    Produces the ADR-0026 deep-merge override that replaces the approach
    manifest's ``joint.positions`` with ``starting_pose`` — i.e. plan to the next
    skill's pose instead of the manifest's home default. ``joint_names`` (from
    :func:`joint_names_from_goal_json`) is used only to length-check ``positions``;
    the manifest's ``joint.joint_names`` order is authoritative and is preserved
    by the deep-merge.

    Args:
        joint_names: Planning-group joint names (for the length check).
        positions: Target joint positions aligned 1:1 with ``joint_names``.

    Returns:
        A JSON string suitable for ``ROSActionRskill``'s ``goal_params_json``.

    Raises:
        ValueError: If ``joint_names`` and ``positions`` differ in length.
    """
    if len(joint_names) != len(positions):
        raise ValueError(
            f"starting_pose length {len(positions)} != approach planning-group "
            f"joint count {len(joint_names)} ({list(joint_names)!r})"
        )
    override = {"joint": {"positions": [float(p) for p in positions]}}
    return json.dumps(override)
