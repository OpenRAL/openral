"""Pure MoveIt goal shaping for the runner's starting-pose approach."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any


def joint_names_from_goal_json(default_goal_json: str) -> list[str]:
    """Extract the planning-group joint names from a MoveGroup ``default_goal_json``.

    Reads the ``joint`` block's ``joint_names`` — the joint order the
    ``rskill-moveit-multi-joints-none`` (``goal_builder: "joint"``) approach manifest
    declares for its MoveIt planning group. Used to length-check a robot's flat
    ``starting_pose`` before building the retarget override
    (``moveit_joint_goal_override``).

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

    Produces the deep-merge override that replaces the approach
    manifest's ``joint.positions`` with ``starting_pose`` — i.e. plan to the next
    skill's pose instead of the manifest's home default. ``joint_names`` (from
    ``joint_names_from_goal_json``) is used only to length-check ``positions``;
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
