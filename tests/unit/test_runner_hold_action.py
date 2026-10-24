"""``hold_action``: the completion check's hold keeps commanding and does not move (2026-10-22).

Going quiet while a VLM answered let the deadman watchdog e-stop after 8.4 s (Spark R225);
re-sending the last command's final step let the arm finish its chunk into a collision (R232).
The hold is the arm at its measured joints, the jaw at its last command.
"""

from __future__ import annotations

import pytest
from openral_core.schemas import Action, ControlMode

runner = pytest.importorskip("openral_rskill_ros.rskill_runner_node")

_NAMES = ["j1", "j2", "finger_joint"]


def _chunk() -> Action:
    return Action(
        control_mode=ControlMode.JOINT_POSITION,
        horizon=3,
        joint_targets=[[0.0, 0.5, 0.1], [0.1, 0.6, 0.3], [0.2, 0.7, 0.39]],
        joint_names=_NAMES,
        tick_index=7,
    )


def test_the_arm_holds_where_it_is_and_the_jaw_keeps_its_command() -> None:
    measured = {"j1": 0.04, "j2": 0.52, "finger_joint": 0.31}
    held = runner.hold_action(_chunk(), measured, {"finger_joint"}, _NAMES)
    assert held is not None and held.horizon == 1
    assert held.joint_targets == [[0.04, 0.52, 0.39]], "arm measured, jaw as commanded"
    final = _chunk().joint_targets
    assert final is not None and held.joint_targets != [final[-1]], "never the final step"


def test_an_unnamed_row_takes_the_manifest_order() -> None:
    a = _chunk().model_copy(update={"joint_names": None})
    held = runner.hold_action(a, {"j1": 0.0, "j2": 0.0}, {"finger_joint"}, _NAMES)
    assert held is not None and held.joint_names == _NAMES
    assert held.joint_targets == [[0.0, 0.0, 0.39]]


@pytest.mark.parametrize(
    ("action", "measured"),
    [
        (Action(control_mode=ControlMode.JOINT_VELOCITY, joint_velocities=[[0.0, 0.0]]), {}),
        (_chunk(), {"j1": 0.0}),  # j2 unmeasured: nothing to hold it at
        (_chunk().model_copy(update={"joint_names": None}), {"j1": 0.0, "j2": 0.0, "x": 0.0}),
    ],
)
def test_nothing_to_hold_is_refused(action: Action, measured: dict[str, float]) -> None:
    names = _NAMES if action.joint_names else ["j1", "j2"]
    assert runner.hold_action(action, measured, {"finger_joint"}, names) is None
