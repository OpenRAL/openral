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
