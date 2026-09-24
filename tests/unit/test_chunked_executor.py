"""Unit tests for ``ChunkedExecutor`` and ``build_chunk_executor``.

Moved out of the deleted ``test_smolvla_adapter.py`` (the unused
``openral_rskill.smolvla`` adapter was removed); ``ChunkedExecutor`` lives in
``openral_rskill.executor`` and is shared by every chunked VLA family.

All tests run without a GPU, without lerobot, and without network access.
A ``_NullPolicy`` stands in for a lerobot policy. ``TestChunkedExecutor``
exercises executor-owned chunk ordering, reset, prefetch timing, and
background error propagation; the policy returns numbered chunks so any
duplicated or reordered action fails visibly.
"""

from __future__ import annotations

import threading
import time
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch
from openral_core.exceptions import ROSConfigError, ROSRuntimeError
from openral_rskill._vla_core import build_chunk_executor
from openral_rskill.executor import ChunkedExecutor

# ── Fixtures / helpers ────────────────────────────────────────────────────────


class _FakeConfig:
    """Minimal stand-in for SmolVLAPolicy.config."""

    def __init__(self, chunk_size: int = 10, n_dof: int = 6) -> None:
        self.chunk_size = chunk_size
        self.n_action_steps = chunk_size
        self.n_dof = n_dof
        self.input_features = {
            "observation.state": MagicMock(shape=(n_dof,)),
            "observation.images.camera1": MagicMock(shape=(3, 256, 256)),
        }
        self.output_features = {
            "action": MagicMock(shape=(n_dof,)),
        }


class _NullPolicy:
    """Minimal lerobot policy stub for unit tests.

    ``predict_action_chunk`` returns numbered chunks: inference 1 contains
    actions 100, 101, 102…, inference 2 contains 200, 201… Numbering EVERY
    step (rather than stamping the whole chunk with one id) is what makes
    intra-chunk reordering and duplication visible — a chunk filled with a
    single value hides both.

    ``select_action`` (the legacy single-step interface) is counted separately
    so tests can assert the executor never falls back to it.
    """

    def __init__(self, chunk_size: int = 10, n_dof: int = 6) -> None:
        self.config = _FakeConfig(chunk_size=chunk_size, n_dof=n_dof)
        self._inference_count = 0
        self._select_action_calls = 0
        self._reset_count = 0

    def reset(self) -> None:
        """Record resets; the executor must never call this from prefetch."""
        self._reset_count += 1

    def predict_action_chunk(self, batch: dict[str, Any]) -> torch.Tensor:
        """Return one numbered ``(1, steps, n_dof)`` chunk."""
        del batch
        self._inference_count += 1
        values = (
            torch.arange(
                self.config.n_action_steps,
                dtype=torch.float32,
            )
            + self._inference_count * 100
        )
        return values.view(1, -1, 1).expand(1, -1, self.config.n_dof).clone()

    def select_action(self, batch: dict[str, Any]) -> torch.Tensor:
        """Legacy single-step surface — kept warmup-compatible, but counted:
        the chunk executor must never route through it."""
        self._select_action_calls += 1
        return self.predict_action_chunk(batch)[:, 0]


# ── ChunkedExecutor tests ─────────────────────────────────────────────────────


class TestChunkedExecutor:
    def test_first_call_hits_policy(self) -> None:
        """The very first select_action call must run a chunk inference."""
        p = _NullPolicy(chunk_size=5)
        ex = ChunkedExecutor(p, prefetch_at=2)
        ex.start()
        batch: dict[str, Any] = {}
        action = ex.select_action(batch)
        assert p._inference_count == 1
        assert action[0, 0].item() == 100

    def test_subsequent_calls_drain_executor_owned_chunk(self) -> None:
        """Calls 2..chunk_size drain the executor buffer IN ORDER — no new inference.

        prefetch_at=0 keeps the background pre-fetch out of the tested window,
        so the exact action sequence proves the buffer served every step.
        """
        chunk_size = 8
        p = _NullPolicy(chunk_size=chunk_size)
        ex = ChunkedExecutor(p, prefetch_at=0)
        ex.start()
        batch: dict[str, Any] = {}
        got = [int(ex.select_action(batch)[0, 0].item()) for _ in range(chunk_size)]
        assert got == list(range(100, 100 + chunk_size))
        assert p._inference_count == 1  # one chunk inference for the whole chunk
        assert p._select_action_calls == 0  # legacy per-step interface unused

    def test_lazy_batch_only_materialised_on_inference(self) -> None:
        """A callable batch must be invoked only when an inference launches."""
        chunk_size = 6
        p = _NullPolicy(chunk_size=chunk_size)
        ex = ChunkedExecutor(p, prefetch_at=0)
        ex.start()
        calls = 0

        def batch_fn() -> dict[str, Any]:
            nonlocal calls
            calls += 1
            return {}

        for _ in range(chunk_size):
            ex.select_action(batch_fn)
        assert calls == 1  # cold start only — pops never build a batch

    def test_prefetch_never_resets_policy_or_reinfers_foreground(self) -> None:
        """Regression: the pre-fetch must not clear live policy state.

        The old implementation called ``policy.reset()`` from the background
        thread, emptying the live action queue — the foreground then ran a
        SECOND full inference racing the pre-fetch. With the executor-owned
        buffer: exactly one inference per chunk, zero policy resets, and the
        actions handed out during chunk N all carry chunk N's stamp.
        """
        chunk_size = 6
        prefetch_at = 2
        p = _NullPolicy(chunk_size=chunk_size)
        ex = ChunkedExecutor(p, prefetch_at=prefetch_at)
        ex.start()
        batch: dict[str, Any] = {}

        first_chunk = [int(ex.select_action(batch)[0, 0].item()) for _ in range(chunk_size)]
        # Every action of chunk 1 must come from inference #1, IN ORDER — the
        # old bug served the tail of the chunk from a racing second inference.
        assert first_chunk == list(range(100, 100 + chunk_size))

        # Crossing into chunk 2 consumes the pre-fetched result: total
        # inferences == 2 (one per chunk), and the policy was never reset.
        assert int(ex.select_action(batch)[0, 0].item()) == 200
        assert p._inference_count == 2
        assert p._reset_count == 0
        ex.stop()

    def test_reset_clears_state(self) -> None:
        """reset() clears the owned buffer and resets the policy once."""
        p = _NullPolicy(chunk_size=5)
        ex = ChunkedExecutor(p, prefetch_at=0)
        ex.start()
        ex.select_action({})
        ex.select_action({})
        assert len(ex._buffer) == 3
        ex.reset()
        assert len(ex._buffer) == 0
        assert p._reset_count == 1

    def test_stop_is_idempotent(self) -> None:
        """stop() must be callable multiple times without error."""
        p = _NullPolicy()
        ex = ChunkedExecutor(p)
        ex.start()
        ex.stop()
        ex.stop()  # second stop must not raise

    def test_prefetch_is_clamped_for_small_chunks(self) -> None:
        """A default larger than the checkpoint chunk cannot silently disable prefetch."""
        ex = ChunkedExecutor(_NullPolicy(chunk_size=4), prefetch_at=20)
        assert ex._prefetch_at == 3

    def test_zero_prefetch_waits_for_fresh_boundary_batch(self) -> None:
        """Zero disables background inference instead of capturing a one-step-stale batch."""
        policy = _NullPolicy(chunk_size=2)
        ex = ChunkedExecutor(policy, prefetch_at=0)
        ex.start()

        assert [int(ex.select_action({})[0, 0]) for _ in range(2)] == [100, 101]
        assert policy._inference_count == 1
        assert int(ex.select_action({})[0, 0]) == 200
        assert policy._inference_count == 2

    def test_negative_prefetch_is_rejected(self) -> None:
        """Negative queue depths are configuration errors."""
        with pytest.raises(ROSConfigError, match="prefetch_at"):
            ChunkedExecutor(_NullPolicy(), prefetch_at=-1)

    def test_background_error_propagates_as_ros_runtime_error(self) -> None:
        """If the pre-fetch thread raises, the next foreground call must re-raise.

        The failure is raised by the policy on the real background path rather
        than poked into the executor's private ``_bg_error``, so this exercises
        the actual propagation seam (CLAUDE.md §1.11).
        """

        class _FailingPolicy(_NullPolicy):
            def predict_action_chunk(self, batch: dict[str, Any]) -> torch.Tensor:
                # Inference 1 is the foreground chunk; fail the prefetch (2nd).
                if self._inference_count == 1:
                    raise ValueError("simulated GPU OOM")
                return super().predict_action_chunk(batch)

        chunk_size = 4
        p = _FailingPolicy(chunk_size=chunk_size)
        ex = ChunkedExecutor(p, prefetch_at=2)
        ex.start()
        batch: dict[str, Any] = {}

        # Drain the whole chunk so the next call must consume the BG result.
        for _ in range(chunk_size):
            ex.select_action(batch)

        with pytest.raises(ROSRuntimeError, match="simulated GPU OOM"):
            ex.select_action(batch)

    def test_prefetch_triggers_background_thread(self) -> None:
        """A background thread must launch when the buffer falls to prefetch_at."""
        chunk_size = 6
        prefetch_at = 2
        p = _NullPolicy(chunk_size=chunk_size)
        ex = ChunkedExecutor(p, prefetch_at=prefetch_at)
        ex.start()
        batch: dict[str, Any] = {}

        # Pop until the buffer depth reaches prefetch_at.
        for _ in range(chunk_size - prefetch_at):
            ex.select_action(batch)

        # Allow the daemon thread to start.
        time.sleep(0.05)
        assert ex._bg_thread is not None
        ex.stop()

    def test_prefetch_preserves_chunk_order_without_resetting_policy(self) -> None:
        """Background inference cannot reset, duplicate, or reorder foreground actions.

        The companion to ``test_prefetch_triggers_background_thread``: that one
        proves the thread launches, this one proves what it hands back is still
        the right sequence. With prefetch ACTIVE, the whole of chunk 1 must be
        served in order and the boundary must hand over chunk 2's first action —
        not a duplicate, not a reordered pop, not a re-inference.
        """
        chunk_size = 6
        p = _NullPolicy(chunk_size=chunk_size)
        ex = ChunkedExecutor(p, prefetch_at=2)
        ex.start()
        batch: dict[str, Any] = {}

        got = [int(ex.select_action(batch)[0, 0].item()) for _ in range(chunk_size + 1)]
        assert got == [100, 101, 102, 103, 104, 105, 200]
        assert p._inference_count == 2
        assert p._reset_count == 0
        assert p._select_action_calls == 0
        ex.stop()


class TestChunkedExecutorChunkFn:
    """Generalised chunk_fn mode — custom producers, sequence chunks."""

    def test_requires_policy_or_chunk_fn_with_size(self) -> None:
        with pytest.raises(ValueError, match="chunk_fn together with chunk_size"):
            ChunkedExecutor()
        with pytest.raises(ValueError, match="chunk_fn together with chunk_size"):
            ChunkedExecutor(chunk_fn=lambda payload: [])

    def test_sequence_chunk_size_mismatch_is_rejected(self) -> None:
        ex = ChunkedExecutor(
            chunk_fn=lambda payload: [torch.full((1, 6), 7.0)] * 5, chunk_size=3, prefetch_at=0
        )
        ex.start()
        with pytest.raises(ROSRuntimeError, match="expected 3 actions, producer returned 5"):
            ex.select_action({})

    def test_short_chunk_prefetches_right_after_first_pop(self) -> None:
        """chunks shorter than prefetch_at trigger at chunk_size - 1."""
        calls = 0

        def chunk_fn(payload: Any) -> list[torch.Tensor]:
            nonlocal calls
            calls += 1
            return [torch.zeros(1, 6)] * 4

        ex = ChunkedExecutor(chunk_fn=chunk_fn, chunk_size=4, prefetch_at=15)
        ex.start()
        ex.select_action({})  # cold start; buffer 3 == min(15, 4-1) → prefetch fires
        time.sleep(0.05)
        assert ex._bg_thread is not None
        assert calls == 2  # cold start + prefetch
        ex.stop()

    def test_chunk_fn_with_policy_delegates_reset(self) -> None:
        """policy passed alongside chunk_fn is used for reset() only."""
        p = _NullPolicy(chunk_size=4)
        ex = ChunkedExecutor(
            p, chunk_fn=lambda payload: [torch.zeros(1, 6)] * 4, chunk_size=4, prefetch_at=0
        )
        ex.start()
        ex.select_action({})
        ex.reset()
        assert p._reset_count == 1
        assert p._inference_count == 0  # producer used, not policy.predict_action_chunk

    def test_stop_waits_for_running_prefetch(self) -> None:
        started = threading.Event()
        release = threading.Event()
        calls = 0

        def chunk_fn(payload: Any) -> list[torch.Tensor]:
            nonlocal calls
            calls += 1
            if calls == 2:
                started.set()
                release.wait()
            return [torch.zeros(1, 6)] * 4

        ex = ChunkedExecutor(chunk_fn=chunk_fn, chunk_size=4, prefetch_at=1)
        ex.start()
        for _ in range(3):
            ex.select_action({})
        assert started.wait(timeout=1.0)

        stopper = threading.Thread(target=ex.stop)
        stopper.start()
        stopper.join(timeout=0.05)
        assert stopper.is_alive()
        release.set()
        stopper.join(timeout=1.0)
        assert not stopper.is_alive()


class TestBuildChunkExecutor:
    def test_custom_single_action_producer_keeps_synchronous_buffer(self) -> None:
        ex = build_chunk_executor(
            {"chunk_prefetch": True},
            chunk_fn=lambda payload: [torch.zeros(1, 6)],
            chunk_size=1,
            adapter_name="test",
        )
        assert ex is not None
        assert ex._prefetch_at == 0

    def test_prefetch_lead_is_configurable_and_clamped(self) -> None:
        ex = build_chunk_executor(
            {"chunk_prefetch": True, "chunk_prefetch_at": 99},
            chunk_fn=lambda payload: [torch.zeros(1, 6)] * 4,
            chunk_size=4,
            adapter_name="test",
        )
        assert ex is not None
        assert ex._prefetch_at == 3

    @pytest.mark.parametrize("value", [0, -1, True, "2"])
    def test_invalid_prefetch_lead_is_rejected(self, value: Any) -> None:
        with pytest.raises(ROSConfigError, match="chunk_prefetch_at"):
            build_chunk_executor(
                {"chunk_prefetch": True, "chunk_prefetch_at": value},
                chunk_fn=lambda payload: [torch.zeros(1, 6)] * 4,
                chunk_size=4,
                adapter_name="test",
            )
