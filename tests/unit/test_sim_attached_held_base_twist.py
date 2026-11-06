"""A held ``/cmd_vel`` setpoint on a base the env integrates (the Isaac kinematic base).

``SimAttachedHAL`` holds an ungrouped ``BODY_TWIST`` as a velocity setpoint for
``_BASE_TWIST_HOLD_NS`` and every env step applies it. These pin the edges found in
review on PR #497: an arm step during navigation keeps the base driving (arm actions
suppress the idle stepper), a reset drops the setpoint, and a grouped lateral twist on
a differential base is refused before any backend sees it.

The env is a recording fake at the ``SimRollout`` boundary (the legitimate seam) that
packs by name like the Isaac sidecar; the description is the real openamrobot_v2 manifest.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray
from openral_core import Action, ControlMode, RobotDescription, ROSConfigError, SceneSpec, TaskSpec
from openral_hal.sim_attached import SimAttachedHAL
from openral_sim.rollout import StepResult

_ROBOTS = Path(__file__).resolve().parents[2] / "robots"
_DIM = 24


class _PackingSim:
    """Non-MuJoCo env that packs by name (twist in the last three slots) and records steps."""

    scene: SceneSpec = None  # type: ignore[assignment]  # reason: the HAL never reads scene
    task: TaskSpec = None  # type: ignore[assignment]  # reason: the HAL never reads task
    action_dim = _DIM

    def __init__(self, n_joints: int) -> None:
        self._n = n_joints
        self.stepped: list[NDArray[np.float32]] = []

    def _obs(self) -> dict[str, Any]:
        return {"images": {}, "joint_positions": np.zeros(self._n, dtype=np.float32)}

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        del seed
        return self._obs()

    def pack_action(
        self, action: Action, prev: NDArray[np.float32] | None, carry_twist: bool = False
    ) -> NDArray[np.float32]:
        out = np.zeros(_DIM, dtype=np.float32) if prev is None else prev.copy()
        if not carry_twist:
            out[-3:] = 0.0
        if action.control_mode is ControlMode.BODY_TWIST and action.body_twist:
            vx, vy, *_, wz = action.body_twist[0]
            out[-3:] = (vx, vy, wz)
        elif action.control_mode is ControlMode.GRIPPER_POSITION and action.gripper:
            out[0] = float(action.gripper[0])
        return out

    def step(self, action: NDArray[np.float32]) -> StepResult:
        self.stepped.append(np.asarray(action, dtype=np.float32).copy())
        return StepResult(self._obs(), 0.0, False, False, {})

    def render(self) -> NDArray[np.uint8] | None:
        return None

    def close(self) -> None:
        return None


@pytest.fixture
def rig() -> tuple[SimAttachedHAL, _PackingSim]:
    description = RobotDescription.from_yaml(str(_ROBOTS / "openamrobot_v2" / "robot.yaml"))
    env = _PackingSim(len(description.joints))
    hal = SimAttachedHAL(env, description)
    hal.connect()
    return hal, env


def _twist(vx: float, vy: float, wz: float, **kw: Any) -> Action:
    return Action(
        control_mode=ControlMode.BODY_TWIST, body_twist=[(vx, vy, 0.0, 0.0, 0.0, wz)], **kw
    )


def _grip(value: float, **kw: Any) -> Action:
    return Action(
        control_mode=ControlMode.GRIPPER_POSITION, gripper=[value], ee_name="left_gripper", **kw
    )


def test_an_arm_step_keeps_a_live_setpoint_driving_the_base(
    rig: tuple[SimAttachedHAL, _PackingSim],
) -> None:
    hal, env = rig
    hal.send_action(_twist(0.3, 0.0, 0.2))
    assert env.stepped == []  # a setpoint steps nothing itself
    hal.send_action(_grip(0.5))
    assert env.stepped[-1][-3:].tolist() == pytest.approx([0.3, 0.0, 0.2])
    assert hal.base_twist == pytest.approx((0.3, 0.0, 0.0, 0.0, 0.0, 0.2))
    hal.idle_step()
    assert env.stepped[-1][-3:].tolist() == pytest.approx([0.3, 0.0, 0.2])


def test_a_reset_drops_the_held_setpoint(rig: tuple[SimAttachedHAL, _PackingSim]) -> None:
    hal, env = rig
    hal.send_action(_twist(0.3, 0.0, 0.0))
    hal.reset_episode()
    hal.idle_step()
    assert env.stepped[-1][-3:].tolist() == [0.0, 0.0, 0.0]
    assert hal.base_twist == (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def test_a_grouped_lateral_twist_is_refused_before_the_env_packs_it(
    rig: tuple[SimAttachedHAL, _PackingSim],
) -> None:
    hal, env = rig
    common = {"tick_index": 3, "tick_group_size": 2, "runner_session_id": 1}
    hal.send_action(_twist(0.2, 0.1, 0.0, **common))
    with pytest.raises(ROSConfigError, match="differential base"):
        hal.send_action(_grip(0.5, **common))
    assert env.stepped == []


def test_estop_reports_the_base_stopped(rig: tuple[SimAttachedHAL, _PackingSim]) -> None:
    hal, _env = rig
    hal.send_action(_twist(0.3, 0.0, 0.2))
    hal.idle_step()
    assert hal.base_twist[0] == pytest.approx(0.3)
    hal.estop()
    assert hal.base_twist == (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


class _PersistentHoldSim(_PackingSim):
    """A backend whose idle_action hands back its own persistent hold vector."""

    def __init__(self, n_joints: int) -> None:
        super().__init__(n_joints)
        self.hold = np.zeros(_DIM, dtype=np.float32)

    def idle_action(self) -> NDArray[np.float32]:
        return self.hold


def test_a_held_twist_never_writes_into_the_backends_hold_vector() -> None:
    description = RobotDescription.from_yaml(str(_ROBOTS / "openamrobot_v2" / "robot.yaml"))
    env = _PersistentHoldSim(len(description.joints))
    hal = SimAttachedHAL(env, description)
    hal.connect()
    hal.send_action(_twist(0.3, 0.0, 0.2))
    hal.idle_step()
    assert env.stepped[-1][-3:].tolist() == pytest.approx([0.3, 0.0, 0.2])
    assert env.hold.tolist() == [0.0] * _DIM  # the backend's own vector is untouched


def test_a_groups_own_twist_supersedes_the_held_setpoint(
    rig: tuple[SimAttachedHAL, _PackingSim],
) -> None:
    """A skill's stop (a grouped zero twist) must not be overridden by an older /cmd_vel."""
    hal, env = rig
    hal.send_action(_twist(0.3, 0.0, 0.2))
    common = {"tick_index": 3, "tick_group_size": 2, "runner_session_id": 1}
    hal.send_action(_twist(0.0, 0.0, 0.0, **common))
    hal.send_action(_grip(0.5, **common))
    assert env.stepped[-1][-3:].tolist() == [0.0, 0.0, 0.0]
    hal.idle_step()
    assert env.stepped[-1][-3:].tolist() == [0.0, 0.0, 0.0]
    hal.send_action(_grip(0.4))
    assert env.stepped[-1][-3:].tolist() == [0.0, 0.0, 0.0]


class _HeadTwistSim(_PackingSim):
    """An env whose own layout puts the base twist first, not in the last three slots."""

    def pack_action(
        self, action: Action, prev: NDArray[np.float32] | None, carry_twist: bool = False
    ) -> NDArray[np.float32]:
        out = np.zeros(_DIM, dtype=np.float32) if prev is None else prev.copy()
        if not carry_twist:
            out[:3] = 0.0
        if action.control_mode is ControlMode.BODY_TWIST and action.body_twist:
            vx, vy, *_, wz = action.body_twist[0]
            out[:3] = (vx, vy, wz)
        elif action.control_mode is ControlMode.GRIPPER_POSITION and action.gripper:
            out[-1] = float(action.gripper[0])
        return out


def test_a_held_twist_goes_where_the_env_packs_it() -> None:
    description = RobotDescription.from_yaml(str(_ROBOTS / "openamrobot_v2" / "robot.yaml"))
    env = _HeadTwistSim(len(description.joints))
    hal = SimAttachedHAL(env, description)
    hal.connect()
    hal.send_action(_twist(0.3, 0.0, 0.2))
    hal.send_action(_grip(0.5))
    assert env.stepped[-1][:3].tolist() == pytest.approx([0.3, 0.0, 0.2])
    assert env.stepped[-1][-1] == pytest.approx(0.5)  # the gripper slot is not overwritten
    hal.idle_step()
    assert env.stepped[-1][:3].tolist() == pytest.approx([0.3, 0.0, 0.2])
