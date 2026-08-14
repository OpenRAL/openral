"""Unit tests for RoboCasa adapter success extraction helpers."""

from __future__ import annotations

import numpy as np
from openral_core import SceneSpec, TaskSpec
from openral_sim.backends.robocasa import _RoboCasaSim


class _InnerEnv:
    def _check_success(self) -> bool:
        return True


class _WrappedEnv:
    unwrapped = type("_Unwrapped", (), {"env": _InnerEnv()})()


class _RawEnv:
    action_dim = 1

    def __init__(self) -> None:
        self.success_checks = 0

    def step(self, _action: object) -> tuple[dict[str, object], float, bool, dict[str, object]]:
        return {}, 0.0, True, {}

    def _check_success(self) -> bool:
        self.success_checks += 1
        return True


class _MinimalRoboCasaSim(_RoboCasaSim):
    def _wrap_obs(self, _raw: dict[str, object]) -> dict[str, object]:
        return {"images": {}, "state": np.zeros(0, dtype=np.float32)}

    def _log_eef_distance(
        self,
        _raw: dict[str, object],
        *,
        step: int,
        action: np.ndarray,
    ) -> None:
        del step, action


def test_gr1_gym_wrapper_success_is_read_from_inner_env() -> None:
    rollout = _RoboCasaSim(
        scene=SceneSpec(id="robocasa/gr1/PnPCupToDrawerClose", backend="mujoco"),
        task=TaskSpec(
            id="robocasa/gr1/PnPCupToDrawerClose/0",
            scene_id="robocasa/gr1/PnPCupToDrawerClose",
            success_key="is_success",
        ),
        _env=_WrappedEnv(),
        _camera_keys=("camera1",),
        _state_layout="gr1",
        _last_image=np.zeros((4, 4, 3), dtype=np.uint8),
        _is_gymnasium_wrapped=True,
        _robots=("GR1ArmsAndWaistFourierHands",),
    )
    assert rollout._check_success_fallback(False) is True


def test_deploy_continuous_mode_skips_success_and_terminal_synthesis() -> None:
    raw_env = _RawEnv()
    rollout = _MinimalRoboCasaSim(
        scene=SceneSpec(id="robocasa/PickPlaceCounterToCabinet", backend="mujoco"),
        task=TaskSpec(
            id="robocasa/PickPlaceCounterToCabinet/0",
            scene_id="robocasa/PickPlaceCounterToCabinet",
            success_key="is_success",
        ),
        _env=raw_env,
        _camera_keys=("camera1",),
    )
    rollout.enable_continuous()
    result = rollout.step(np.zeros(1, dtype=np.float32))
    assert raw_env.success_checks == 0
    assert result.terminated is False
    assert result.truncated is False
    assert "is_success" not in result.info


class _NoPredicateEnv:
    """A robosuite-shaped env that ships no ``_check_success`` at all.

    Not every backend the adapter can wrap carries a task-success
    predicate (the procedural scenario surface, a scene built purely to
    host a deploy twin). The adapter must report "unknown", never "failed".
    """

    action_dim = 1

    def step(self, _action: object) -> tuple[dict[str, object], float, bool, dict[str, object]]:
        return {}, 0.0, False, {}


def _pnp_sim(env: object) -> _RoboCasaSim:
    """Wrap ``env`` in the adapter against a real shipped RoboCasa scene id."""
    return _RoboCasaSim(
        scene=SceneSpec(id="robocasa/PickPlaceCounterToSink", backend="mujoco"),
        task=TaskSpec(
            id="robocasa/PickPlaceCounterToSink/0",
            scene_id="robocasa/PickPlaceCounterToSink",
            success_key="is_success",
        ),
        _env=env,
        _camera_keys=("camera1",),
    )


def test_task_success_reads_the_raw_robosuite_predicate() -> None:
    """`task_success()` returns the env's own `_check_success()` verdict."""
    raw_env = _RawEnv()
    rollout = _pnp_sim(raw_env)
    assert rollout.task_success() is True
    assert raw_env.success_checks == 1


def test_task_success_reads_through_the_gymnasium_wrapper() -> None:
    """The GR1 gym path hides `_check_success` at `env.unwrapped.env`."""
    rollout = _pnp_sim(_WrappedEnv())
    assert rollout.task_success() is True


def test_task_success_is_none_when_the_backend_has_no_predicate() -> None:
    """No `_check_success` → `None` ("unknown"), never `False`."""
    rollout = _pnp_sim(_NoPredicateEnv())
    assert rollout.task_success() is None


def test_task_success_does_not_step_the_env() -> None:
    """Reading the predicate is side-effect free — it must not advance physics."""
    raw_env = _RawEnv()
    rollout = _pnp_sim(raw_env)
    rollout.task_success()
    rollout.task_success()
    assert not hasattr(raw_env, "step_calls")  # _RawEnv counts only success checks
    assert raw_env.success_checks == 2


def test_check_success_fallback_uses_terminated_only_without_a_predicate() -> None:
    """The eval path still falls back to `terminated` when no predicate exists."""
    rollout = _pnp_sim(_NoPredicateEnv())
    assert rollout._check_success_fallback(True) is True
    assert rollout._check_success_fallback(False) is False


def test_deploy_continuous_mode_still_never_polls_success_during_step() -> None:
    """`step()` in continuous mode leaves the predicate untouched.

    The new `task_success()` accessor is polled by the HAL AFTER the step,
    for observability. `step()` itself must stay exactly as evaluation-free
    as before, so nothing about termination changes (CLAUDE.md §1.4).
    """
    raw_env = _RawEnv()
    rollout = _MinimalRoboCasaSim(
        scene=SceneSpec(id="robocasa/PickPlaceCounterToSink", backend="mujoco"),
        task=TaskSpec(
            id="robocasa/PickPlaceCounterToSink/0",
            scene_id="robocasa/PickPlaceCounterToSink",
            success_key="is_success",
        ),
        _env=raw_env,
        _camera_keys=("camera1",),
    )
    rollout.enable_continuous()
    result = rollout.step(np.zeros(1, dtype=np.float32))
    assert raw_env.success_checks == 0
    assert result.terminated is False
    # …and the accessor still works on the same env, out of band.
    assert rollout.task_success() is True
    assert raw_env.success_checks == 1
