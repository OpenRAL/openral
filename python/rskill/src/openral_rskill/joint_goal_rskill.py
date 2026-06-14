"""``JointGoalRskill`` — move the arm to a joint configuration via MoveGroup (ADR-0054).

A :class:`~openral_rskill.ros_action_rskill.ROSActionRskill` whose goal is a
``joint`` block (``joint_names`` + ``positions``), lowered at dispatch time into
a MoveGroup ``joint_constraints`` goal. The clean, LLM-facing replacement for
hand-writing the ``request.goal_constraints[0].joint_constraints`` array in
``default_goal_json``: the reasoner provides positions, not raw constraint dicts.
"""

from __future__ import annotations

from typing import Any

import structlog
from openral_core.exceptions import ROSConfigError

from openral_rskill.ros_action_rskill import ROSActionRskill

__all__ = ["JointGoalRskill", "joint_constraints_from_block"]

log = structlog.get_logger(__name__)

_DEFAULT_JOINT_TOLERANCE_RAD = 0.001
_JOINT_WEIGHT = 1.0


def joint_constraints_from_block(block: dict[str, Any]) -> dict[str, Any]:
    """Lower a ``joint`` goal block into one MoveGroup ``goal_constraints`` entry.

    Args:
        block: ``{joint_names: [...], positions: [...], position_tolerance_rad?}``.
            ``joint_names`` and ``positions`` must be equal-length and parallel.

    Returns:
        A ``goal_constraints`` entry: ``{"joint_constraints": [{joint_name,
        position, tolerance_above, tolerance_below, weight}, …]}``.

    Raises:
        ROSConfigError: On missing/ill-typed fields or a name/position length
            mismatch.
    """
    names = block.get("joint_names")
    positions = block.get("positions")
    if not isinstance(names, (list, tuple)) or not all(isinstance(n, str) for n in names):
        raise ROSConfigError(f"joint.joint_names must be a list of strings; got {names!r}.")
    if not isinstance(positions, (list, tuple)) or not all(
        isinstance(p, (int, float)) for p in positions
    ):
        raise ROSConfigError(f"joint.positions must be a list of numbers; got {positions!r}.")
    if len(names) != len(positions):
        raise ROSConfigError(
            f"joint.joint_names ({len(names)}) and joint.positions ({len(positions)}) "
            "differ in length."
        )
    tol = float(block.get("position_tolerance_rad", _DEFAULT_JOINT_TOLERANCE_RAD))
    joint_constraints = [
        {
            "joint_name": name,
            "position": float(pos),
            "tolerance_above": tol,
            "tolerance_below": tol,
            "weight": _JOINT_WEIGHT,
        }
        for name, pos in zip(names, positions, strict=True)
    ]
    return {"joint_constraints": joint_constraints}


class JointGoalRskill(ROSActionRskill):
    """Joint-space MoveGroup skill (``ros_integration.goal_builder: "joint"``).

    Consumes the merged goal's ``joint`` block and lowers it into a
    ``joint_constraints`` MoveGroup goal on configure, then dispatches + replays
    exactly like the parent.
    """

    def _configure_impl(self) -> None:
        super()._configure_impl()
        block = self._goal_dict.pop("joint", None)
        if not isinstance(block, dict):
            raise ROSConfigError(
                f"JointGoalRskill({self.name!r}): the merged goal JSON has no 'joint' object "
                "— the manifest's default_goal_json must carry one (the LLM's "
                "goal_params_json may override its fields)."
            )
        entry = joint_constraints_from_block(block)
        request = self._goal_dict.setdefault("request", {})
        request["goal_constraints"] = [entry]
        log.info(
            "joint_goal_rskill.configured",
            name=self.name,
            n_joints=len(entry["joint_constraints"]),
        )
