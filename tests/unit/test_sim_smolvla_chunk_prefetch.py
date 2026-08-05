"""Unit tests for the shared chunk-executor wiring in the sim policy adapters.

`_vla_core.build_chunk_executor` is the ONE seam every chunked adapter uses:
the executor-owned buffer replaces both lerobot's internal queue pops and the
adapter-private replay lists, and ``VLASpec.extra["chunk_prefetch"]`` (set by
the deploy runtime) only controls whether the chunk-boundary inference
overlaps execution — flag unset keeps a synchronous buffer with action
semantics identical to the old queue replay, so eval stays paper-faithful.

Follows the established `_NullPolicy` pattern from
``tests/unit/test_smolvla_adapter.py``: a minimal chunk-producing policy whose
``predict_action_chunk`` stamps each chunk with a monotonically increasing id,
so tests can assert WHICH inference produced each action. The adapter, the
executor, the batch build, the processors, and the ``VLASpec`` schema are all
real.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from openral_core import VLASpec
from openral_rskill._vla_core import attach_chunk_executor, build_chunk_executor
from openral_sim.policies.smolvla import _SmolVLAAdapter

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
        """Legacy single-step interface — the executor must not use it."""
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
    return attach_chunk_executor(adapter, spec.extra, policy=policy, adapter_name="smolvla")


def _obs(n_dof: int = 6) -> dict[str, Any]:
    return {
        "images": {"camera1": np.zeros((48, 64, 3), dtype=np.uint8)},
        "state": np.zeros(n_dof, dtype=np.float32),
    }


# ── build_chunk_executor semantics ────────────────────────────────────────────


class TestBuildChunkExecutor:
    def test_flag_unset_builds_synchronous_buffer(self) -> None:
        """Eval path: executor attached, but with prefetch disabled."""
        adapter = _make_adapter(_NullPolicy(chunk_size=6), extra={})
        assert adapter._chunk_executor is not None
        assert adapter._chunk_executor._prefetch_at == 0

    def test_flag_set_enables_prefetch(self) -> None:
        adapter = _make_adapter(_NullPolicy(chunk_size=6), extra={"chunk_prefetch": True})
        assert adapter._chunk_executor is not None
        assert adapter._chunk_executor._prefetch_at > 0
        assert adapter._chunk_executor._running is True

    def test_per_step_policy_gets_no_executor(self) -> None:
        """n_action_steps == 1 has no chunk to buffer — plain per-step path."""
        adapter = _make_adapter(_NullPolicy(chunk_size=1), extra={"chunk_prefetch": True})
        assert adapter._chunk_executor is None

    def test_chunk_fn_mode_requires_no_policy(self) -> None:
        """Custom producers (gr00t / molmoact2 / openvla shape) work policy-free."""
        chunks = []

        def chunk_fn(payload: Any) -> list[np.ndarray]:
            chunks.append(payload)
            return [np.full(6, float(len(chunks))) for _ in range(4)]

        ex = build_chunk_executor({}, chunk_fn=chunk_fn, chunk_size=4, adapter_name="test")
        assert ex is not None
        got = [ex.select_action(("obs", "instr")) for _ in range(5)]
        assert len(chunks) == 2  # one inference per chunk of 4, then a fresh one
        assert all(float(a[0]) == 1.0 for a in got[:4])
        assert float(got[4][0]) == 2.0


# ── Adapter step semantics through the executor ──────────────────────────────


class TestExecutorStepPath:
    def test_one_inference_per_chunk_and_actions_from_that_chunk(self) -> None:
        chunk_size = 6
        policy = _NullPolicy(chunk_size=chunk_size)
        adapter = _make_adapter(policy, extra={})  # eval mode: synchronous buffer

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
        adapter = _make_adapter(policy, extra={})
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
