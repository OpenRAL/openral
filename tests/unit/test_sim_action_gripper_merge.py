"""Regression: the non-composite packer merges the split arm+gripper action.

A single VLA policy step on a non-composite sim env (LIBERO OSC_POSE franka,
SimplerEnv widowx — every ``delta_ee_6d_plus_gripper`` rSkill) is dispatched as
TWO typed Actions, CARTESIAN_DELTA (arm) then GRIPPER_POSITION (finger), and the
HAL ``env.step``\\s each one. Before the fix the stateless ``pack_action_for_env``
built a fresh zero vector per Action, so the arm stepped with gripper=0 and the
gripper stepped with arm=0 — the arm advanced only every other env step with a
flickering gripper and never coordinated a grasp (env.step received all-zero arm
commands; the reward sat flat). The composite path already merged via
``_last_env_action``; this brings the same merge to the legacy packer.

These pin the merge at the packer level (the live two-``env.step`` flow is
exercised by the deploy-sim run); they drive the REAL ``pack_action_for_env``
with the REAL franka manifest (CLAUDE.md §1.11 — no mocks).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("numpy")

from openral_core import Action, ControlMode, RobotDescription
from openral_hal.sim_attached import pack_action_for_env

_FRANKA = Path("robots/franka_panda/robot.yaml")
_ENV_DIM = 7  # LIBERO OSC_POSE: 6-D arm delta + 1 gripper, fixed base (base_dim=0).


def _franka() -> RobotDescription:
    if not _FRANKA.exists():
        pytest.skip(f"robot fixture missing: {_FRANKA}")
    return RobotDescription.from_yaml(str(_FRANKA))


def _arm(delta: tuple[float, ...]) -> Action:
    return Action(control_mode=ControlMode.CARTESIAN_DELTA, cartesian_delta=[delta])


def _grip(v: float) -> Action:
    return Action(control_mode=ControlMode.GRIPPER_POSITION, gripper=[v])


def test_arm_step_preserves_the_last_commanded_gripper() -> None:
    desc = _franka()
    # Previous command latched a closed gripper at the last slot.
    prev = np.zeros(_ENV_DIM, dtype=np.float32)
    prev[-1] = -1.0
    out = pack_action_for_env(_arm((0.3, -0.7, -0.04, 0.0, 0.0, 0.0)), desc, _ENV_DIM, prev)
    # Arm slots get the delta...
    np.testing.assert_allclose(out[0:6], [0.3, -0.7, -0.04, 0.0, 0.0, 0.0], atol=1e-6)
    # ...and the gripper is CARRIED from prev (the bug zeroed it).
    assert out[-1] == pytest.approx(-1.0)


def test_gripper_step_holds_the_arm() -> None:
    desc = _franka()
    # Previous command had the arm mid-delta; the gripper step must HOLD it
    # (zero) so the per-step OSC delta is not applied a second time.
    prev = np.array([0.3, -0.7, -0.04, 0.0, 0.0, 0.0, -1.0], dtype=np.float32)
    out = pack_action_for_env(_grip(1.0), desc, _ENV_DIM, prev)
    np.testing.assert_allclose(out[0:6], [0.0] * 6, atol=1e-6)
    assert out[-1] == pytest.approx(1.0)


def test_one_policy_step_lands_full_arm_plus_gripper() -> None:
    # Simulate the two-Action sequence one policy step splits into, threading
    # the previous env vector like SimAttachedHAL.send_action does.
    desc = _franka()
    prev = np.zeros(_ENV_DIM, dtype=np.float32)
    prev[-1] = -1.0  # gripper open from the prior step
    arm_out = pack_action_for_env(_arm((0.1, 0.2, 0.3, 0.0, 0.0, 0.0)), desc, _ENV_DIM, prev)
    grip_out = pack_action_for_env(_grip(1.0), desc, _ENV_DIM, arm_out)
    # Arm step carries the moving arm AND a real (non-zero) gripper command.
    assert np.any(arm_out[0:6] != 0.0)
    assert arm_out[-1] != 0.0
    # Gripper step holds the arm and updates the gripper.
    np.testing.assert_allclose(grip_out[0:6], [0.0] * 6, atol=1e-6)
    assert grip_out[-1] == pytest.approx(1.0)


def test_no_prev_is_safe_first_step() -> None:
    # First step of an episode: no prev → fresh zeros + the delta (gripper 0
    # until the first gripper Action lands, which is correct).
    desc = _franka()
    out = pack_action_for_env(_arm((0.1, 0.0, 0.0, 0.0, 0.0, 0.0)), desc, _ENV_DIM, None)
    assert out[0] == pytest.approx(0.1)
    assert out[-1] == pytest.approx(0.0)
