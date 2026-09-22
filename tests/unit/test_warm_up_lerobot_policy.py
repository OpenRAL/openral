"""The first CUDA inference must not land in the first control tick.

Measured on an RTX 4070 with the ACT so101-pen checkpoint (resnet18 +
transformer, two 480x640 cameras):

    call 1   330.4 ms      <- 10x the 33.3 ms budget at 30 Hz
    call 2+   14.9 ms

Charged to tick 1, that's a deadline miss and, under
``DeadlineOverrunPolicy.DROP``, a discarded first action. After wiring the
warm-up into ``activate()``, the first real tick measured 14.9 ms — inside
budget.

Uses a real ``torch.nn.Module`` with a real lerobot-shaped config (CLAUDE.md
§1.11): tests whether shapes are read off the config and fed to a real
forward pass.
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


def _tokenising_preprocessor(batch: dict[str, Any]) -> dict[str, Any]:
    """What lerobot's π0.5 pipeline does: tokenise ``task`` onto the CPU, leave state float32."""
    batch["observation.language.tokens"] = torch.zeros(1, 8, dtype=torch.long)
    return batch


def test_preprocessed_batch_is_cast_to_the_adapters_input_dtype() -> None:
    """The real path casts floating tensors after preprocessing; so must the warm-up."""
    policy = _Policy(6, {"observation.images.wrist": (3, 224, 224)})
    adapter = _Adapter(policy)
    adapter._preprocessor = _tokenising_preprocessor  # type: ignore[attr-defined]
    adapter._input_dtype = torch.bfloat16  # type: ignore[attr-defined]
    policy.linear.to(torch.bfloat16)
    assert warm_up_lerobot_policy(adapter, prompt="x") is True
    assert policy.seen is not None
    assert policy.seen["observation.state"].dtype == torch.bfloat16
    # Integer tokens are not floating point and must not be cast.
    assert policy.seen["observation.language.tokens"].dtype == torch.long


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_preprocessed_cpu_tensors_are_moved_to_the_policy_device() -> None:
    """A CPU token tensor left in a CUDA batch made the warm-up raise and warm nothing.

    Live on qorin1 (2026-09-22): ``rskill_runner.warmup_failed error='Expected all
    tensors to be on the same device'`` and the first real π0.5 tick then paid
    15.3 s. Every tensor the preprocessor returns has to end up on the device.
    """
    policy = _Policy(6, {"observation.images.wrist": (3, 224, 224)}).to("cuda:0")
    adapter = _Adapter(policy, device="cuda:0")
    adapter._preprocessor = _tokenising_preprocessor  # type: ignore[attr-defined]
    assert warm_up_lerobot_policy(adapter, prompt="x") is True
    assert policy.seen is not None
    assert str(policy.seen["observation.language.tokens"].device).startswith("cuda")
    assert str(policy.seen["observation.state"].device).startswith("cuda")
