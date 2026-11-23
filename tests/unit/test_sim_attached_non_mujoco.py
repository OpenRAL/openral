"""Unit tests: SimAttachedHAL is backend-agnostic for non-MuJoCo SimRollouts.

Amendment for deploy sim with a non-MuJoCo backend (e.g. the Isaac Sim
sidecar). A `SimRollout` with no `mujoco_handles` must still drive
`openral deploy sim`:

* `read_state()` sources real joint angles from `obs["joint_positions"]` (in
  description-joint order, or keyed by `obs["joint_names"]`) and refuses with a
  typed `ROSRuntimeError` when the backend provides none for some joint — never a
  fabricated all-zero pose the safety kernel would read as measured;
* `idle_step()` steps the env (no MuJoCo-handle gate) so cameras stay live.

Exercised against a tiny fake `SimRollout` at the env boundary (the legitimate
seam; no MuJoCo, no GPU, no ROS) + a real `RobotDescription`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray
from openral_core import Action, ControlMode, RobotDescription, SceneSpec, TaskSpec
from openral_core.exceptions import ROSRuntimeError
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

    scene: SceneSpec = None  # type: ignore[assignment]  # reason: the HAL never reads scene
    task: TaskSpec = None  # type: ignore[assignment]  # reason: the HAL never reads task
    action_dim = 8

    def __init__(
        self,
        joint_positions: list[float] | None,
        joint_velocities: list[float] | None = None,
        joint_efforts: list[float] | None = None,
        joint_names: list[str] | None = None,
    ) -> None:
        self._jn = joint_names
        self._jp = joint_positions
        self._jv = joint_velocities
        self._je = joint_efforts
        self.steps = 0

    def _obs(self) -> dict[str, Any]:
        obs: dict[str, Any] = {"images": {}, "state": np.zeros(8, dtype=np.float32), "task": ""}
        if self._jp is not None:
            obs["joint_positions"] = np.asarray(self._jp, dtype=np.float32)
        if self._jv is not None:
            obs["joint_velocities"] = np.asarray(self._jv, dtype=np.float32)
        if self._je is not None:
            obs["joint_efforts"] = np.asarray(self._je, dtype=np.float32)
        if self._jn is not None:
            obs["joint_names"] = list(self._jn)
        return obs

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        del seed
        return self._obs()

    def step(self, action: NDArray[np.float32]) -> StepResult:
        del action
        self.steps += 1
        return StepResult(self._obs(), 0.0, False, False, {})

    def render(self) -> NDArray[np.uint8] | None:
        return None

    def close(self) -> None:
        return None


class _GroupedSim(_FakeSim):
    action_group_size = 2

    def __init__(self, joint_positions: list[float]) -> None:
        super().__init__(joint_positions)
        self.groups: list[list[Action]] = []

    def step(self, action: NDArray[np.float32]) -> StepResult:
        raise AssertionError(f"flat step must not run for grouped backend: {action}")

    def step_action_group(self, actions: list[Action]) -> StepResult:
        self.groups.append(actions)
        self.steps += 1
        obs = self._obs()
        obs["policy_state"] = np.arange(8, dtype=np.float32)
        return StepResult(obs, 0.0, False, False, {})


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


def test_read_state_velocity_empty_without_joint_velocities() -> None:
    """Unmeasured velocity → none published, never a zero that reads as at rest."""
    description = _franka_description()
    n = len(description.joints)
    hal = SimAttachedHAL(_FakeSim([0.1] * n), description)  # positions only
    hal.connect()
    try:
        assert hal.read_state().velocity == []
    finally:
        hal.disconnect()


def test_read_state_uses_obs_joint_efforts() -> None:
    description = _franka_description()
    n = len(description.joints)
    je = [1.5 * i - 3.0 for i in range(n)]
    hal = SimAttachedHAL(_FakeSim([0.0] * n, joint_efforts=je), description)
    hal.connect()
    try:
        np.testing.assert_allclose(hal.read_state().effort, je, atol=1e-5)
    finally:
        hal.disconnect()


def test_read_state_effort_empty_for_a_short_joint_efforts_vector() -> None:
    """Efforts for only some joints → none published, not zero-padded rows."""
    description = _franka_description()
    n = len(description.joints)
    hal = SimAttachedHAL(_FakeSim([0.0] * n, joint_efforts=[1.0] * (n - 1)), description)
    hal.connect()
    try:
        assert hal.read_state().effort == []
    finally:
        hal.disconnect()


def test_read_state_effort_empty_without_joint_efforts() -> None:
    """No measured effort → none published; zeros would read as a free, unloaded arm."""
    description = _franka_description()
    hal = SimAttachedHAL(_FakeSim([0.0] * len(description.joints)), description)
    hal.connect()
    try:
        assert hal.read_state().effort == []
    finally:
        hal.disconnect()


def test_read_state_refuses_without_joint_positions() -> None:
    """No measured positions → a typed refusal, never a fabricated all-zero pose."""
    description = _franka_description()
    hal = SimAttachedHAL(_FakeSim(None), description)
    hal.connect()
    try:
        with pytest.raises(ROSRuntimeError, match="no measured joint positions"):
            hal.read_state()
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
def test_read_state_refuses_length_mismatch(jp_len: int) -> None:
    """An unnamed vector that is not one value per joint cannot be placed: refuse.

    Padding a short vector fabricates rows; truncating a long one guesses which rows
    belong to which joint. Both used to publish as measured.
    """
    description = _franka_description()
    hal = SimAttachedHAL(_FakeSim([0.5] * jp_len), description)
    hal.connect()
    try:
        with pytest.raises(ROSRuntimeError, match="no measured joint positions"):
            hal.read_state()
    finally:
        hal.disconnect()


def test_read_state_maps_named_joints_onto_the_description() -> None:
    """``joint_names`` keys the vectors: reordered, with extra backend joints ignored."""
    description = _franka_description()
    names = [j.name for j in description.joints]
    reported = ["extra_backend_joint", *reversed(names)]
    jp = [9.0, *(0.1 * i for i in reversed(range(len(names))))]
    jv = [9.0, *(0.01 * i for i in reversed(range(len(names))))]
    hal = SimAttachedHAL(_FakeSim(jp, jv, joint_names=reported), description)
    hal.connect()
    try:
        state = hal.read_state()
        assert state.name == names
        np.testing.assert_allclose(state.position, [0.1 * i for i in range(len(names))], atol=1e-5)
        np.testing.assert_allclose(state.velocity, [0.01 * i for i in range(len(names))], atol=1e-5)
    finally:
        hal.disconnect()


def test_read_state_refuses_a_description_joint_the_backend_does_not_name() -> None:
    description = _franka_description()
    names = [j.name for j in description.joints][1:]
    hal = SimAttachedHAL(_FakeSim([0.0] * len(names), joint_names=names), description)
    hal.connect()
    try:
        with pytest.raises(ROSRuntimeError, match=description.joints[0].name):
            hal.read_state()
    finally:
        hal.disconnect()


def test_read_policy_state_requires_explicit_key() -> None:
    # A backend with only the generic obs["state"] exposes NO policy state:
    # WorldState.policy_state is never inferred, so the HAL must not fall
    # back to obs["state"] and silently publish /openral/policy_state.
    description = _franka_description()
    hal = SimAttachedHAL(_FakeSim([0.0] * len(description.joints)), description)
    hal.connect()
    try:
        assert hal.idle_step() is True
        assert hal.read_policy_state() is None
    finally:
        hal.disconnect()


def test_atomic_action_group_steps_once_after_every_safe_slot() -> None:
    description = _franka_description()
    env = _GroupedSim([0.0] * len(description.joints))
    hal = SimAttachedHAL(env, description)
    hal.connect()
    first = Action(
        control_mode=ControlMode.BODY_TWIST,
        body_twist=[(0.1, 0.0, 0.0, 0.0, 0.0, 0.0)],
        tick_index=7,
    )
    second = Action(
        control_mode=ControlMode.GRIPPER_POSITION,
        gripper=[1.0],
        ee_name="panda_hand",
        tick_index=7,
    )

    hal.send_action(first)
    assert env.steps == 0
    assert hal.idle_step() is False
    hal.send_action(second)

    assert env.steps == 1
    assert len(env.groups) == 1
    assert [action.tick_index for action in env.groups[0]] == [7, 7]
    assert hal.read_policy_state() == np.arange(8, dtype=np.float32).tolist()


def test_incomplete_action_group_fails_on_first_tick_transition() -> None:
    """An incomplete group (fewer typed slots per tick than ``action_group_size``)
    raises instead of silently never stepping — it previously froze the sim with
    only a stdout print.
    """
    from openral_core.exceptions import ROSRuntimeError

    description = _franka_description()
    env = _GroupedSim([0.0] * len(description.joints))
    hal = SimAttachedHAL(env, description)
    hal.connect()

    def one_slot(tick: int) -> Action:
        return Action(
            control_mode=ControlMode.BODY_TWIST,
            body_twist=[(0.1, 0.0, 0.0, 0.0, 0.0, 0.0)],
            tick_index=tick,
        )

    hal.send_action(one_slot(1))  # stages tick 1 (1/2 slots)
    with pytest.raises(ROSRuntimeError, match="incomplete slot group"):
        hal.send_action(one_slot(2))
    assert env.steps == 0  # atomicity preserved: nothing partial ever stepped


def test_a_stop_boundary_discards_a_half_staged_group() -> None:
    """A kernel stop cuts a tick short: the HAL lifecycle node calls
    ``discard_staged_slots`` at the estop latch/clear (as for the MuJoCo and real HALs),
    so the next goal's first slot starts clean instead of raising "incomplete action
    group". The committed watermark is kept: the pre-stop tick is still stale."""
    description = _franka_description()
    env = _GroupedSim([0.0] * len(description.joints))
    hal = SimAttachedHAL(env, description)
    hal.connect()

    def slot(tick: int, mode: ControlMode = ControlMode.BODY_TWIST) -> Action:
        if mode is ControlMode.BODY_TWIST:
            return Action(
                control_mode=mode, body_twist=[(0.1, 0.0, 0.0, 0.0, 0.0, 0.0)], tick_index=tick
            )
        return Action(control_mode=mode, gripper=[1.0], ee_name="panda_hand", tick_index=tick)

    hal.send_action(slot(3))
    hal.send_action(slot(3, ControlMode.GRIPPER_POSITION))  # tick 3 committed
    hal.send_action(slot(4))  # tick 4 half-staged when the stop lands
    assert hal.discard_staged_slots() == 1
    assert hal.discard_staged_slots() == 0
    hal.send_action(slot(5))  # no "incomplete slot group"
    hal.send_action(slot(5, ControlMode.GRIPPER_POSITION))
    assert env.steps == 2
    from openral_core.exceptions import ROSRuntimeError

    with pytest.raises(ROSRuntimeError):
        hal.send_action(slot(3))  # the committed watermark survived the discard


def test_group_commit_latches_commanded_base_twist() -> None:
    """The group path maintains base_twist like the per-mode send_action paths."""
    description = _franka_description()
    env = _GroupedSim([0.0] * len(description.joints))
    hal = SimAttachedHAL(env, description)
    hal.connect()
    hal.send_action(
        Action(
            control_mode=ControlMode.BODY_TWIST,
            body_twist=[(0.1, 0.2, 0.0, 0.0, 0.0, 0.3)],
            tick_index=5,
        )
    )
    hal.send_action(
        Action(
            control_mode=ControlMode.GRIPPER_POSITION,
            gripper=[1.0],
            ee_name="panda_hand",
            tick_index=5,
        )
    )
    assert env.steps == 1
    assert hal.base_twist == pytest.approx((0.1, 0.2, 0.0, 0.0, 0.0, 0.3))


def test_stale_pending_group_releases_idle_step(monkeypatch: pytest.MonkeyPatch) -> None:
    """A group whose slots stopped arriving must not block idle stepping forever."""
    import time as time_module

    class _IdleSteppableGroupedSim(_GroupedSim):
        # The real grouped backend (BEHAVIOR) accepts flat env.step too —
        # that is exactly what idle_step drives (its hold vector).
        def step(self, action: NDArray[np.float32]) -> StepResult:
            del action
            self.steps += 1
            return StepResult(self._obs(), 0.0, False, False, {})

    description = _franka_description()
    env = _IdleSteppableGroupedSim([0.0] * len(description.joints))
    hal = SimAttachedHAL(env, description)
    hal.connect()

    now = [time_module.monotonic_ns()]
    monkeypatch.setattr("openral_hal.sim_attached.time.monotonic_ns", lambda: now[0])

    hal.send_action(
        Action(
            control_mode=ControlMode.BODY_TWIST,
            body_twist=[(0.1, 0.0, 0.0, 0.0, 0.0, 0.0)],
            tick_index=9,
        )
    )
    assert hal.idle_step() is False  # slots mid-flight: never interleave a HOLD

    now[0] += 6_000_000_000  # +6 s: the skill died mid-tick
    assert hal.idle_step() is True  # stale group discarded, scene stays live
    assert env.steps == 1


class _SharedWorldSim(_FakeSim):
    """One robot of a shared world: offers a stop for its last observation (ADR-0137)."""

    def __init__(self, joint_positions: list[float], *, shared: bool) -> None:
        super().__init__(joint_positions)
        self.shared = shared
        self.actions: list[NDArray[np.float32]] = []

    def estop_hold_action(self, obs: dict[str, Any] | None) -> NDArray[np.float32] | None:
        if not self.shared or obs is None:
            return None
        return np.full(self.action_dim, 0.25, dtype=np.float32)

    def step(self, action: NDArray[np.float32]) -> StepResult:
        self.actions.append(np.asarray(action))
        return super().step(action)


def test_a_latched_robot_of_a_shared_world_keeps_stepping_its_stop() -> None:
    """Its world moves under it, so its joint states must keep coming (peer_voxels)."""
    description = _franka_description()
    env = _SharedWorldSim([0.0] * len(description.joints), shared=True)
    hal = SimAttachedHAL(env, description)
    hal.connect()
    try:
        hal.estop()
        before = env.steps
        assert hal.idle_step() is True, "the HAL's own latch"
        assert env.steps == before + 1
        before = env.steps
        assert hal.step_estop_hold() is True, "the node's latch (SimSensorBridge)"
        assert env.steps == before + 1
        np.testing.assert_array_equal(env.actions[-1], np.full(env.action_dim, 0.25))
    finally:
        hal.disconnect()


def test_a_latched_robot_alone_in_its_world_still_freezes() -> None:
    description = _franka_description()
    env = _SharedWorldSim([0.0] * len(description.joints), shared=False)
    hal = SimAttachedHAL(env, description)
    hal.connect()
    try:
        hal.estop()
        before = env.steps
        assert hal.step_estop_hold() is False
        assert hal.idle_step() is False
        assert env.steps == before
    finally:
        hal.disconnect()


class _PackingSim(_FakeSim):
    """An env that packs by name (RoboTwin's hook): records the ``prev`` it is handed."""

    def __init__(self, joint_positions: list[float]) -> None:
        super().__init__(joint_positions)
        self.prevs: list[NDArray[np.float32] | None] = []

    def pack_action(
        self, action: Action, prev: NDArray[np.float32] | None, *, carry_twist: bool = False
    ) -> NDArray[np.float32]:
        del carry_twist
        self.prevs.append(None if prev is None else prev.copy())
        out = np.full(self.action_dim, np.nan, dtype=np.float32) if prev is None else prev.copy()
        if action.joint_targets:
            out[: len(action.joint_targets[0])] = action.joint_targets[0]
        if action.gripper:
            out[-1] = action.gripper[0]
        return out


def test_a_stop_drops_the_cut_short_target_from_the_next_pack() -> None:
    """After ``stop_stale_motion`` the next pack seeds from the measured pose, not the
    target the stop cut short: a gripper command must not resume the arm's old move."""
    description = _franka_description()
    env = _PackingSim([0.0] * len(description.joints))
    hal = SimAttachedHAL(env, description)
    hal.connect()
    far = [0.5] * len(description.joints)
    hal.send_action(Action(control_mode=ControlMode.JOINT_POSITION, joint_targets=[far]))
    hal.stop_stale_motion()
    hal.send_action(
        Action(control_mode=ControlMode.GRIPPER_POSITION, gripper=[0.02], ee_name="panda_hand")
    )
    assert env.prevs[0] is None
    assert env.prevs[1] is None, f"the stopped arm target rode along: {env.prevs[1]}"
