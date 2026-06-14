"""Unit tests: SimAttachedHAL is backend-agnostic for non-MuJoCo SimRollouts.

ADR-0034 amendment (deploy sim with a non-MuJoCo backend, e.g. the Isaac Sim
sidecar). A `SimRollout` with no `mujoco_handles` must still drive
`openral deploy sim`:

* `read_state()` sources real joint angles from `obs["joint_positions"]` (in
  description-joint order), not all-zeros, when the backend provides them;
* `idle_step()` steps the env (no MuJoCo-handle gate) so cameras stay live.

Exercised against a tiny fake `SimRollout` at the env boundary (the legitimate
seam; no MuJoCo, no GPU, no ROS) + a real `RobotDescription`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from openral_core import RobotDescription
from openral_hal.sim_attached import SimAttachedHAL
from openral_sim.rollout import StepResult


def _franka_description() -> RobotDescription:
    here = Path(__file__).resolve()
    for ancestor in (here, *here.parents):
        if (ancestor / "robots").is_dir() and (ancestor / "pyproject.toml").is_file():
            return RobotDescription.from_yaml(
                str(ancestor / "robots" / "franka_panda" / "robot.yaml")
            )
    raise RuntimeError("repo root not found")


class _FakeSim:
    """Minimal non-MuJoCo SimRollout: no `mujoco_handles`, optional joint_positions."""

    scene = None
    task = None
    action_dim = 8

    def __init__(
        self,
        joint_positions: list[float] | None,
        joint_velocities: list[float] | None = None,
    ) -> None:
        self._jp = joint_positions
        self._jv = joint_velocities
        self.steps = 0

    def _obs(self) -> dict:
        obs: dict = {"images": {}, "state": np.zeros(8, dtype=np.float32), "task": ""}
        if self._jp is not None:
            obs["joint_positions"] = np.asarray(self._jp, dtype=np.float32)
        if self._jv is not None:
            obs["joint_velocities"] = np.asarray(self._jv, dtype=np.float32)
        return obs

    def reset(self, seed: int | None = None) -> dict:
        del seed
        return self._obs()

    def step(self, action: np.ndarray) -> StepResult:
        del action
        self.steps += 1
        return StepResult(self._obs(), 0.0, False, False, {})

    def render(self) -> np.ndarray | None:
        return None

    def close(self) -> None:
        return None


def test_read_state_uses_obs_joint_positions() -> None:
    description = _franka_description()
    jp = [0.1 * i for i in range(len(description.joints))]
    hal = SimAttachedHAL(_FakeSim(jp), description)
    hal.connect()
    try:
        state = hal.read_state()
        assert state.name == [j.name for j in description.joints]
        np.testing.assert_allclose(state.position, jp, atol=1e-5)
    finally:
        hal.disconnect()


def test_read_state_uses_obs_joint_velocities() -> None:
    description = _franka_description()
    n = len(description.joints)
    jp = [0.1 * i for i in range(n)]
    jv = [0.01 * i for i in range(n)]
    hal = SimAttachedHAL(_FakeSim(jp, jv), description)
    hal.connect()
    try:
        state = hal.read_state()
        np.testing.assert_allclose(state.position, jp, atol=1e-5)
        np.testing.assert_allclose(state.velocity, jv, atol=1e-5)
    finally:
        hal.disconnect()


def test_read_state_velocity_zeros_without_joint_velocities() -> None:
    description = _franka_description()
    n = len(description.joints)
    hal = SimAttachedHAL(_FakeSim([0.1] * n), description)  # positions only
    hal.connect()
    try:
        state = hal.read_state()
        assert all(v == 0.0 for v in state.velocity)
    finally:
        hal.disconnect()


def test_read_state_falls_back_to_zeros_without_joint_positions() -> None:
    description = _franka_description()
    hal = SimAttachedHAL(_FakeSim(None), description)
    hal.connect()
    try:
        state = hal.read_state()
        assert state.name == [j.name for j in description.joints]
        assert all(p == 0.0 for p in state.position)
    finally:
        hal.disconnect()


def test_idle_step_steps_without_mujoco_handles() -> None:
    description = _franka_description()
    env = _FakeSim([0.0] * len(description.joints))
    hal = SimAttachedHAL(env, description)
    hal.connect()
    try:
        before = env.steps
        assert hal.idle_step() is True
        assert env.steps == before + 1
    finally:
        hal.disconnect()


@pytest.mark.parametrize("jp_len", [6, 10])
def test_read_state_tolerates_length_mismatch(jp_len: int) -> None:
    # A backend whose joint_positions vector is shorter/longer than the manifest
    # must still yield a description-shaped JointState (pad / truncate).
    description = _franka_description()
    hal = SimAttachedHAL(_FakeSim([0.5] * jp_len), description)
    hal.connect()
    try:
        state = hal.read_state()
        assert len(state.position) == len(description.joints)
    finally:
        hal.disconnect()
