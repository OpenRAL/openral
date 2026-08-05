"""Unit tests for the deploy-runtime chunk-prefetch wiring in `_SmolVLAAdapter`.

The deploy path (`rskill_runner_node._build_runtime_skill_from_manifest`) sets
``VLASpec.extra["chunk_prefetch"]`` so the adapter overlaps chunk N+1 inference
with the tail of chunk N via :class:`openral_rskill.ChunkedExecutor`
(SmolVLA's async inference mode). The eval path never sets the flag and must
keep the synchronous, paper-faithful chunk replay.

Follows the established `_NullPolicy` pattern from
``tests/unit/test_smolvla_adapter.py``: a minimal chunk-producing policy whose
``predict_action_chunk`` stamps each chunk with a monotonically increasing id,
so tests can assert WHICH inference produced each action. The adapter, the
executor, the batch build, the preprocessors, and the ``VLASpec`` schema are
all real.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from openral_core import VLASpec
from openral_sim.policies.smolvla import _SmolVLAAdapter, maybe_enable_chunk_prefetch

# ── Fixtures / helpers ────────────────────────────────────────────────────────


class _FakeConfig:
    def __init__(self, chunk_size: int, n_dof: int) -> None:
        self.chunk_size = chunk_size
        self.n_action_steps = chunk_size
        self.n_dof = n_dof
        self.adapt_to_pi_aloha = False


class _NullPolicy:
    """Chunk-producing policy stub; chunk axis stamped with the inference id."""

    def __init__(self, chunk_size: int = 6, n_dof: int = 6) -> None:
        self.config = _FakeConfig(chunk_size, n_dof)
        self._call_count = 0
        self._select_action_calls = 0
        self._reset_calls = 0

    def reset(self) -> None:
        self._reset_calls += 1

    def predict_action_chunk(self, batch: dict[str, Any]) -> torch.Tensor:
        self._call_count += 1
        return torch.full(
            (1, self.config.n_action_steps, self.config.n_dof), float(self._call_count)
        )

    def select_action(self, batch: dict[str, Any]) -> torch.Tensor:
        self._select_action_calls += 1
        return torch.zeros(1, self.config.n_dof)


def _identity(x: Any) -> Any:
    return x


def _make_adapter(policy: _NullPolicy, extra: dict[str, Any]) -> _SmolVLAAdapter:
    spec = VLASpec(id="smolvla", weights_uri="rskills/template", extra=extra)
    adapter = _SmolVLAAdapter(
        spec=spec,
        device="cpu",
        _policy=policy,
        _preprocessor=_identity,
        _postprocessor=_identity,
        _torch=torch,
        _camera_keys=("camera1",),
    )
    maybe_enable_chunk_prefetch(adapter, spec.extra)
    return adapter


def _obs(n_dof: int = 6) -> dict[str, Any]:
    return {
        "images": {"camera1": np.zeros((48, 64, 3), dtype=np.uint8)},
        "state": np.zeros(n_dof, dtype=np.float32),
    }


# ── Tests ─────────────────────────────────────────────────────────────────────


class TestMaybeEnableChunkPrefetch:
    def test_flag_unset_keeps_synchronous_eval_path(self) -> None:
        adapter = _make_adapter(_NullPolicy(), extra={})
        assert adapter._chunk_executor is None

    def test_flag_set_attaches_started_executor(self) -> None:
        adapter = _make_adapter(_NullPolicy(chunk_size=6), extra={"chunk_prefetch": True})
        assert adapter._chunk_executor is not None
        assert adapter._chunk_executor._running is True

    def test_per_step_policy_is_skipped(self) -> None:
        """n_action_steps == 1 has no chunk to overlap — stay synchronous."""
        adapter = _make_adapter(_NullPolicy(chunk_size=1), extra={"chunk_prefetch": True})
        assert adapter._chunk_executor is None


class TestPrefetchStepPath:
    def test_one_inference_per_chunk_and_actions_from_that_chunk(self) -> None:
        chunk_size = 6
        policy = _NullPolicy(chunk_size=chunk_size)
        adapter = _make_adapter(policy, extra={"chunk_prefetch": True})

        # prefetch_at (default 15) > chunk_size → no bg thread in this window:
        # deterministic one-cold-start-per-chunk behaviour.
        actions = [adapter.step(_obs(), "place the erase") for _ in range(chunk_size)]
        assert policy._call_count == 1
        assert policy._select_action_calls == 0  # never falls back to the queue path
        assert all(a.shape == (6,) for a in actions)  # (n_dof,) after squeeze
        assert all(float(a[0]) == 1.0 for a in actions)  # all from inference #1

        nxt = adapter.step(_obs(), "place the erase")
        assert policy._call_count == 2
        assert float(nxt[0]) == 2.0

    def test_reset_delegates_to_executor(self) -> None:
        policy = _NullPolicy(chunk_size=6)
        adapter = _make_adapter(policy, extra={"chunk_prefetch": True})
        adapter.step(_obs(), "task")
        assert len(adapter._chunk_executor._buffer) == 5
        adapter.reset()
        assert len(adapter._chunk_executor._buffer) == 0
        assert policy._reset_calls == 1

    def test_close_stops_and_clears_executor(self) -> None:
        policy = _NullPolicy(chunk_size=6)
        adapter = _make_adapter(policy, extra={"chunk_prefetch": True})
        adapter.step(_obs(), "task")
        adapter.close()
        assert adapter._chunk_executor is None
