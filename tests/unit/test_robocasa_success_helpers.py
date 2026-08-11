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
