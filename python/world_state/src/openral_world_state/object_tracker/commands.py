"""Which gripper command a safety-approved chunk carries (issue #418, tracking/on, sim only).

The tracker's own stalled-jaw read (``stall.JawStallDetector``) needs the commanded jaw. The
approved chunk carries it three ways, and the node read only the first:

* a full-dof row, the gripper at its manifest index (DROID, 8 dof; SO-101, 6 dof): positional;
* a compact row with ``joint_names``;
* an ADR-0102 slot group (``tick_group_size > 1``, the OpenArm's four slots per tick): the arm
  slots are joint rows, a gripper slot is one value (``n_dof == 1``, no names) that names its
  gripper by ``ee_name`` (``left_gripper`` is the gripper joint's own name).

Ignoring every slot group meant the restock tracker never held a command, so the stalled-jaw
read never fired in any restock run (r12rs02_on1/on2/camlong_b: ATTACH came from the HAL's
trigger 0.3 to 0.7 s after the carton had started to lift).

Pure: no ROS.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence

__all__ = ["jaw_commands"]


def jaw_commands(
    joints: Collection[str],
    joint_order: Sequence[str],
    *,
    n_dof: int,
    flat: Sequence[float],
    joint_names: Sequence[str],
    tick_group_size: int,
    ee_name: str,
) -> dict[str, float]:
    """The gripper targets of one approved chunk, by gripper joint name.

    Args:
        joints: The gripper joints the tracker models.
        joint_order: The manifest's joint order (a full-dof row's layout).
        n_dof: The chunk's row width.
        flat: The chunk's rows, row-major; the last row is the applied command.
        joint_names: The chunk's joint names (empty for a full-dof or gripper-slot row).
        tick_group_size: ``> 1`` for one slot of an ADR-0102 group.
        ee_name: The end-effector a gripper slot names.

    Returns:
        ``{joint: target}`` for each modelled gripper the chunk commands, else ``{}``.

    Example:
        >>> jaw_commands(
        ...     {"left_gripper"},
        ...     ["a", "left_gripper"],
        ...     n_dof=1,
        ...     flat=[0.4],
        ...     joint_names=[],
        ...     tick_group_size=4,
        ...     ee_name="left_gripper",
        ... )
        {'left_gripper': 0.4}
        >>> jaw_commands(
        ...     {"left_gripper"},
        ...     ["a", "left_gripper"],
        ...     n_dof=2,
        ...     flat=[0.1, 0.2],
        ...     joint_names=[],
        ...     tick_group_size=1,
        ...     ee_name="",
        ... )
        {'left_gripper': 0.2}
    """
    if n_dof <= 0 or len(flat) < n_dof:
        return {}
    row, names = list(flat[-n_dof:]), list(joint_names)
    if tick_group_size > 1:  # a slot of a group: only a one-value gripper slot is a jaw command
        return {ee_name: float(row[0])} if n_dof == 1 and ee_name in joints else {}
    out: dict[str, float] = {}
    for joint in joints:
        if names:
            index = names.index(joint) if joint in names and len(names) == n_dof else None
        elif n_dof == len(joint_order) and joint in joint_order:
            index = list(joint_order).index(joint)
        else:
            index = None
        if index is not None:
            out[joint] = float(row[index])
    return out
