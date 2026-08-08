"""The first CUDA inference must not land in the first control tick.

Measured on an RTX 4070 with the ACT so101-pen checkpoint (resnet18 +
transformer, two 480x640 cameras):

    call 1   330.4 ms      <- 10x the 33.3 ms budget at 30 Hz
    call 2+   14.9 ms

Charged to tick 1 that is a guaranteed deadline miss, and under
``DeadlineOverrunPolicy.DROP`` the robot's first commanded action is
discarded. After wiring the warm-up into ``activate()``, the same
checkpoint's first real tick measured 14.9 ms — inside budget.

Uses a real ``torch.nn.Module`` with a real lerobot-shaped config rather
than a mock (CLAUDE.md §1.11): the thing under test is precisely whether
shapes are read correctly off a config and fed to a real forward pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
from openral_rskill._vla_core import warm_up_lerobot_policy

torch = pytest.importorskip("torch", reason="warm_up_lerobot_policy is a torch seam")


@dataclass
class _Feature:
    """Stands in for a lerobot ``PolicyFeature`` (it only exposes ``shape``)."""

    shape: tuple[int, ...]


class _Config:
    def __init__(self, state_dim: int, cameras: dict[str, tuple[int, int, int]]) -> None:
        self.image_features = {k: _Feature(v) for k, v in cameras.items()}
        self.input_features = {
            "observation.state": _Feature((state_dim,)),
            **self.image_features,
        }


class _Policy(torch.nn.Module):
    """A real torch policy that records the batch it was called with."""

    def __init__(self, state_dim: int, cameras: dict[str, tuple[int, int, int]]) -> None:
        super().__init__()
        self.config = _Config(state_dim, cameras)
        self.linear = torch.nn.Linear(state_dim, 4)
        self.seen: dict[str, Any] | None = None
        self.reset_calls = 0

    def reset(self) -> None:
        self.reset_calls += 1

    def select_action(self, batch: dict[str, Any]) -> torch.Tensor:
        self.seen = batch
        return self.linear(batch["observation.state"])


class _Adapter:
    def __init__(self, policy: object | None, device: str = "cpu") -> None:
        self._policy = policy
        self.device = device


def test_warms_up_and_reports_true() -> None:
    policy = _Policy(6, {"observation.images.wrist": (3, 480, 640)})

    assert warm_up_lerobot_policy(_Adapter(policy), prompt="pick up the pen") is True
    assert policy.seen is not None
    assert policy.reset_calls == 1


def test_shapes_come_from_the_policy_config() -> None:
    """A guessed resolution would autotune kernels the real ticks never use."""
    policy = _Policy(
        7,
        {
            "observation.images.wrist": (3, 480, 640),
            "observation.images.front": (3, 256, 256),
        },
    )

    warm_up_lerobot_policy(_Adapter(policy), prompt="x")

    seen = policy.seen
    assert seen is not None
    assert tuple(seen["observation.state"].shape) == (1, 7)
    assert tuple(seen["observation.images.wrist"].shape) == (1, 3, 480, 640)
    assert tuple(seen["observation.images.front"].shape) == (1, 3, 256, 256)


def test_prompt_is_passed_for_language_conditioned_families() -> None:
    policy = _Policy(6, {"observation.images.wrist": (3, 224, 224)})

    warm_up_lerobot_policy(_Adapter(policy), prompt="place the eraser on the blue square")

    assert policy.seen is not None
    assert policy.seen["task"] == ["place the eraser on the blue square"]


def test_image_dtype_is_honoured_when_the_adapter_declares_one() -> None:
    """SmolVLA casts images; warming in float32 would autotune the wrong kernels."""
    policy = _Policy(6, {"observation.images.wrist": (3, 224, 224)})
    adapter = _Adapter(policy)
    adapter._image_dtype = torch.bfloat16  # type: ignore[attr-defined]

    warm_up_lerobot_policy(adapter, prompt="x")

    assert policy.seen is not None
    assert policy.seen["observation.images.wrist"].dtype == torch.bfloat16


@pytest.mark.parametrize(
    "adapter",
    [
        _Adapter(None),  # HF-based adapters (molmoact2 / openvla) expose no _policy
        _Adapter(object()),  # no .config
    ],
)
def test_unintrospectable_adapters_are_skipped(adapter: _Adapter) -> None:
    """Skipping is the contract, not an error — those families just stay cold."""
    assert warm_up_lerobot_policy(adapter, prompt="x") is False


def test_policy_without_a_state_feature_is_skipped() -> None:
    policy = _Policy(6, {"observation.images.wrist": (3, 224, 224)})
    del policy.config.input_features["observation.state"]

    assert warm_up_lerobot_policy(_Adapter(policy), prompt="x") is False


class _PolicyWithoutReset(torch.nn.Module):
    """Some families expose no per-episode ``reset()``."""

    def __init__(self) -> None:
        super().__init__()
        self.config = _Config(6, {"observation.images.wrist": (3, 224, 224)})
        self.linear = torch.nn.Linear(6, 4)
        self.seen: dict[str, Any] | None = None

    def select_action(self, batch: dict[str, Any]) -> torch.Tensor:
        self.seen = batch
        return self.linear(batch["observation.state"])


def test_a_policy_without_reset_still_warms() -> None:
    """``reset()`` is optional; its absence must not abort the warm-up."""
    policy = _PolicyWithoutReset()

    assert warm_up_lerobot_policy(_Adapter(policy), prompt="x") is True
    assert policy.seen is not None
