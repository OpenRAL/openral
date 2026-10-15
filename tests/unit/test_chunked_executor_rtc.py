"""RTC-mode ChunkedExecutor: replace-on-merge, index-delta delay, leftover tails.

Drives the executor's documented ``chunk_fn`` API with real tensors and the
real lerobot ActionQueue/RTCConfig — the policy-level guided denoise is
covered by the GPU test (test_smolvla_rtc.py) and the Pro A/B harness.
"""

from __future__ import annotations

import threading
from typing import Any

import pytest
import torch
from openral_core.exceptions import ROSConfigError, ROSRuntimeError
from openral_rskill.executor import ChunkedExecutor
from structlog.testing import capture_logs

CHUNK, DOF = 10, 3


def _rtc_config(**overrides: Any) -> Any:
    from lerobot.policies.rtc import RTCConfig

    kwargs: dict[str, Any] = {"enabled": True, "execution_horizon": 4}
    kwargs.update(overrides)
    return RTCConfig(**kwargs)


def _chunk(base: float) -> torch.Tensor:
    """(1, CHUNK, DOF) ramp chunk: row t is filled with base + t."""
    t = torch.arange(CHUNK, dtype=torch.float32) + base
    return t.view(1, CHUNK, 1).expand(1, CHUNK, DOF).clone()


class Producer:
    """chunk_fn returning ramp chunks; records every kwargs dict it receives."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.release = threading.Event()
        self.blocking = False

    def __call__(self, batch: Any, **kwargs: Any) -> torch.Tensor:
        if self.blocking and self.calls:  # never block the cold-start call
            assert self.release.wait(timeout=5.0), "test deadlock"
        self.calls.append(dict(kwargs))
        return _chunk(base=100.0 * len(self.calls))


def _executor(producer: Producer, *, prefetch_at: int = 4, **rtc_overrides: Any) -> ChunkedExecutor:
    ex = ChunkedExecutor(
        chunk_fn=producer,
        chunk_size=CHUNK,
        prefetch_at=prefetch_at,
        rtc_config=_rtc_config(**rtc_overrides),
    )
    ex.start()
    return ex


def _join_bg(ex: ChunkedExecutor) -> None:
    thread = ex._bg_thread
    if thread is not None:
        thread.join(timeout=5.0)
        assert not thread.is_alive()


def test_rtc_requires_prefetch() -> None:
    with pytest.raises(ROSConfigError):
        ChunkedExecutor(
            chunk_fn=Producer(), chunk_size=CHUNK, prefetch_at=0, rtc_config=_rtc_config()
        )


def test_cold_start_has_no_guidance_kwargs() -> None:
    producer = Producer()
    ex = _executor(producer)
    action = ex.select_action({})
    assert action.shape == (1, DOF)
    assert torch.allclose(action.squeeze(0), torch.full((DOF,), 100.0))
    assert producer.calls[0] == {"inference_delay": 0, "prev_chunk_left_over": None}
    ex.stop()


def test_prefetch_passes_leftover_tail_and_merge_replaces() -> None:
    producer = Producer()
    # horizon == CHUNK so the tail is handed over whole (truncation has its own test).
    ex = _executor(producer, prefetch_at=CHUNK - 1, execution_horizon=CHUNK)
    first = ex.select_action({})  # cold start (chunk1 = 100..109), pops 100
    _join_bg(ex)  # prefetch (chunk2 = 200..209) merged now
    second = ex.select_action({})
    assert torch.allclose(first.squeeze(0), torch.full((DOF,), 100.0))
    # chunk2 merged with real_delay=0 actions consumed during the (instant)
    # prefetch, so the very next pop comes from chunk2's head — replaced, not appended.
    assert torch.allclose(second.squeeze(0), torch.full((DOF,), 200.0))
    prev = producer.calls[1]["prev_chunk_left_over"]
    # Leftover captured at launch: chunk1 rows 1..9 (row 0 already consumed).
    assert prev.shape == (CHUNK - 1, DOF)
    assert torch.allclose(prev[0], torch.full((DOF,), 101.0))
    ex.stop()


def test_leftover_tail_is_truncated_to_execution_horizon() -> None:
    """A long tail is clipped to the horizon — guidance weights past it are zero."""
    producer = Producer()
    ex = _executor(producer, prefetch_at=CHUNK - 1, execution_horizon=4)
    ex.select_action({})  # cold start; 9 rows unconsumed, prefetch launched
    _join_bg(ex)
    prev = producer.calls[1]["prev_chunk_left_over"]
    assert prev.shape == (4, DOF)  # not (9, DOF)
    assert torch.allclose(prev[0], torch.full((DOF,), 101.0))
    assert torch.allclose(prev[-1], torch.full((DOF,), 104.0))
    ex.stop()


def test_prefetch_below_horizon_warns() -> None:
    """A lead shorter than the horizon cannot fill the tail — degradation, not error."""
    with capture_logs() as logs:
        ChunkedExecutor(
            chunk_fn=Producer(),
            chunk_size=CHUNK,
            prefetch_at=2,
            rtc_config=_rtc_config(execution_horizon=8),
        )
    warned = [e for e in logs if e["event"] == "chunked_executor.rtc_prefetch_below_horizon"]
    assert len(warned) == 1
    assert warned[0]["log_level"] == "warning"
    assert warned[0]["prefetch_at"] == 2
    assert warned[0]["execution_horizon"] == 8


def test_prefetch_at_horizon_does_not_warn() -> None:
    with capture_logs() as logs:
        ChunkedExecutor(
            chunk_fn=Producer(),
            chunk_size=CHUNK,
            prefetch_at=8,
            rtc_config=_rtc_config(execution_horizon=8),
        )
    assert not [e for e in logs if e["event"] == "chunked_executor.rtc_prefetch_below_horizon"]


def test_real_delay_is_actions_consumed_during_inference() -> None:
    producer = Producer()
    producer.blocking = True
    ex = _executor(producer, prefetch_at=CHUNK - 1, execution_horizon=CHUNK)
    ex.select_action({})  # cold start; prefetch launched, blocked
    for _ in range(3):  # consume 3 more of chunk1 while "inference" runs
        ex.select_action({})
    producer.release.set()
    _join_bg(ex)
    nxt = ex.select_action({})
    # 4 consumed at merge, 1 consumed at launch -> real_delay == 3; merge drops
    # chunk2's first 3 rows: next action is chunk2 row 3 == 203.
    assert torch.allclose(nxt.squeeze(0), torch.full((DOF,), 203.0))
    ex.stop()


def test_next_prefetch_reuses_last_real_delay_as_estimate() -> None:
    producer = Producer()
    producer.blocking = True
    ex = _executor(producer, prefetch_at=CHUNK - 1, execution_horizon=CHUNK)
    ex.select_action({})
    for _ in range(3):
        ex.select_action({})
    producer.release.set()
    _join_bg(ex)
    producer.blocking = False
    ex.select_action({})  # pops chunk2 head; relaunches prefetch (call 3)
    _join_bg(ex)
    assert producer.calls[2]["inference_delay"] == 3
    ex.stop()


def test_buffer_drain_blocks_until_merge() -> None:
    producer = Producer()
    producer.blocking = True
    ex = _executor(producer, prefetch_at=2, execution_horizon=2)
    drained = [ex.select_action({}) for _ in range(CHUNK)]  # consume ALL of chunk1
    assert len(drained) == CHUNK

    def _unblock() -> None:
        producer.release.set()

    threading.Timer(0.2, _unblock).start()
    nxt = ex.select_action({})  # empty queue + pending prefetch -> blocks
    assert nxt.squeeze(0)[0].item() >= 200.0
    # Served by the merge, not by a synchronous cold start behind its back.
    assert len(producer.calls) == 2
    ex.stop()


def test_merge_landing_during_empty_read_is_not_discarded() -> None:
    """A merge can land between the queue read and the ``_bg_pending`` check.

    Forces exactly that interleaving with a thin delegator over the *real*
    ActionQueue: the first empty ``get()`` releases the blocked producer and
    waits for the merge, so the executor's next look must find the fresh chunk.
    Without the second look it falls into the cold-start branch, which
    *replaces* the queue and throws the just-merged chunk away.
    """
    producer = Producer()
    producer.blocking = True
    ex = _executor(producer, prefetch_at=2, execution_horizon=2)
    for _ in range(CHUNK):
        ex.select_action({})  # drain chunk1; prefetch launched at qsize==2, blocked
    real_queue = ex._rtc_queue

    class _MergeRacer:
        """Delegates to the real ActionQueue; lands the merge on the first miss."""

        def __init__(self) -> None:
            self.raced = False

        def get(self) -> Any:
            action = real_queue.get()
            if action is None and not self.raced:
                self.raced = True
                producer.release.set()
                assert ex._bg_thread is not None
                ex._bg_thread.join(timeout=5.0)  # merge lands, _bg_pending -> False
            return action

        def __getattr__(self, name: str) -> Any:
            return getattr(real_queue, name)

    racer = _MergeRacer()
    ex._rtc_queue = racer
    action = ex.select_action({})
    assert racer.raced  # the interleaving really happened
    assert len(producer.calls) == 2  # no third, synchronous, cold-start inference
    # real_delay == 2 (rows 8,9 popped while inference ran) -> chunk2 row 2.
    assert torch.allclose(action.squeeze(0), torch.full((DOF,), 202.0))
    ex.stop()


def test_stop_while_waiting_raises() -> None:
    producer = Producer()
    producer.blocking = True
    ex = _executor(producer, prefetch_at=2, execution_horizon=2)
    for _ in range(CHUNK):
        ex.select_action({})  # drain chunk1; prefetch launched, blocked
    stopper = threading.Timer(0.2, ex.stop)
    stopper.start()
    with pytest.raises(ROSRuntimeError, match="stopped while waiting"):
        ex.select_action({})
    producer.release.set()  # let the bg thread — and stop()'s join — finish
    stopper.join(timeout=5.0)


def test_bg_error_propagates() -> None:
    class Boom(Producer):
        def __call__(self, batch: Any, **kwargs: Any) -> torch.Tensor:
            if self.calls:
                raise RuntimeError("kaput")
            return super().__call__(batch, **kwargs)

    producer = Boom()
    ex = _executor(producer, prefetch_at=CHUNK - 1, execution_horizon=CHUNK)
    ex.select_action({})
    _join_bg(ex)
    with pytest.raises(ROSRuntimeError, match="kaput"):
        for _ in range(CHUNK + 1):
            ex.select_action({})
    ex.stop()


def test_non_tensor_producer_rejected() -> None:
    """RTC needs a tensor; the sequence producers the deque path accepts won't do."""

    def _sequence(batch: Any, **kwargs: Any) -> list[float]:
        return [0.0] * CHUNK

    ex = ChunkedExecutor(
        chunk_fn=_sequence,
        chunk_size=CHUNK,
        prefetch_at=CHUNK - 1,
        rtc_config=_rtc_config(execution_horizon=CHUNK),
    )
    ex.start()
    with pytest.raises(ROSRuntimeError, match="tensor producer"):
        ex.select_action({})
    ex.stop()


def test_batched_producer_rejected() -> None:
    """squeeze(0) is a no-op on batch>1, which would consume batch rows as time."""

    def _batched(batch: Any, **kwargs: Any) -> torch.Tensor:
        return torch.zeros(2, CHUNK, DOF)

    ex = ChunkedExecutor(
        chunk_fn=_batched,
        chunk_size=CHUNK,
        prefetch_at=CHUNK - 1,
        rtc_config=_rtc_config(execution_horizon=CHUNK),
    )
    ex.start()
    with pytest.raises(ROSRuntimeError, match="batch dimension must be 1"):
        ex.select_action({})
    ex.stop()


def test_reset_clears_rtc_state() -> None:
    producer = Producer()
    ex = _executor(producer)
    ex.select_action({})  # cold start (call 1, chunk1 = 100..109), pops 100
    ex.reset()
    action = ex.select_action({})  # fresh cold start (call 2, chunk2 = 200..209)
    assert producer.calls[-1]["prev_chunk_left_over"] is None
    assert action.shape == (1, DOF)
    # A second inference really ran, and it served chunk2's head — a queue that
    # survived reset() would have served the stale chunk1 row 1 (== 101) instead.
    assert len(producer.calls) == 2
    assert torch.allclose(action.squeeze(0), torch.full((DOF,), 200.0))
    ex.stop()


def test_rtc_off_is_unchanged_append_mode() -> None:
    producer = Producer()
    ex = ChunkedExecutor(chunk_fn=producer, chunk_size=CHUNK, prefetch_at=0)
    ex.start()
    actions = [ex.select_action({}) for _ in range(CHUNK + 1)]
    # Append semantics: chunk1 fully replayed, then chunk2 begins.
    assert torch.allclose(actions[CHUNK - 1].squeeze(0), torch.full((DOF,), 109.0))
    assert torch.allclose(actions[CHUNK].squeeze(0), torch.full((DOF,), 200.0))
    assert producer.calls == [{}, {}]  # no RTC kwargs leak into non-RTC producers
    ex.stop()


# ── Relative-action policies: the RTC prefix is re-anchored (openral#378) ───
#
# Real lerobot processor steps, configured like a `use_relative_actions` π0.5
# checkpoint (relative_actions → normalizer / unnormalizer → absolute_actions,
# gripper excluded), with stats chosen so every number below is hand-checkable.


def _relative_processors() -> tuple[Any, Any]:
    from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature
    from lerobot.processor import (
        AbsoluteActionsProcessorStep,
        NormalizerProcessorStep,
        PolicyProcessorPipeline,
        RelativeActionsProcessorStep,
        UnnormalizerProcessorStep,
        policy_action_to_transition,
        transition_to_policy_action,
    )

    features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(DOF,))}
    norm_map = {FeatureType.ACTION: NormalizationMode.MEAN_STD}
    # joints: x / 2; gripper: x - 0.5
    stats = {
        "action": {"mean": torch.tensor([0.0, 0.0, 0.5]), "std": torch.tensor([2.0, 2.0, 1.0])}
    }
    relative = RelativeActionsProcessorStep(
        enabled=True, exclude_joints=["gripper"], action_names=["j1", "j2", "gripper"]
    )
    pre = PolicyProcessorPipeline(
        steps=[relative, NormalizerProcessorStep(features=features, norm_map=norm_map, stats=stats)]
    )
    post = PolicyProcessorPipeline(
        steps=[
            UnnormalizerProcessorStep(features=features, norm_map=norm_map, stats=stats),
            AbsoluteActionsProcessorStep(enabled=True, relative_step=relative),
        ],
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )
    return pre, post


class _RelativeProducer(Producer):
    """Model-space chunk: row t = [0.1 t, -0.1 t, 0.2] (normalized deltas + gripper)."""

    def __call__(self, batch: Any, **kwargs: Any) -> torch.Tensor:
        self.calls.append(dict(kwargs))
        t = torch.arange(CHUNK, dtype=torch.float32)
        return torch.stack([0.1 * t, -0.1 * t, torch.full((CHUNK,), 0.2)], dim=-1).unsqueeze(0)


def _state_batches(pre: Any, *states: list[float]) -> Any:
    """Payload factory: each inference preprocesses the next observation state."""
    pending = [torch.tensor([s]) for s in states]
    return lambda: pre({"observation.state": pending.pop(0)})


def test_relative_prefix_is_reanchored_to_the_new_state() -> None:
    from openral_rskill._vla_core import relative_rtc_prefix_fn

    pre, post = _relative_processors()
    producer = _RelativeProducer()
    ex = ChunkedExecutor(
        chunk_fn=producer,
        chunk_size=CHUNK,
        prefetch_at=CHUNK - 1,
        rtc_config=_rtc_config(execution_horizon=4),
        postprocess_action=post,
        rtc_prefix_fn=relative_rtc_prefix_fn(pre, device="cpu"),
    )
    ex.start()
    # Cold start at state [1, -1, 0.3]; the prefetch reads the arm at [1.5, -0.5, 0.3].
    batches = _state_batches(pre, [1.0, -1.0, 0.3], [1.5, -0.5, 0.3])
    served = ex.select_action(batches)
    _join_bg(ex)
    # Served absolute row 0: joints 2 * 0 + state, gripper 0.2 + 0.5.
    assert served == pytest.approx([1.0, -1.0, 0.7])
    # Leftover rows 1..4 absolute: [1 + 0.2 t, -1 - 0.2 t, 0.7]. Against the NEW state
    # [1.5, -0.5]: deltas [0.2 t - 0.5, -0.5 - 0.2 t], normalized (/2), and the gripper
    # (0.7 - 0.5): [0.1 t - 0.25, -0.1 t - 0.25, 0.2]. The raw rows would be
    # [0.1 t, -0.1 t, 0.2] — off by the 0.5 rad the arm moved, over the std.
    expected = torch.tensor(
        [[0.1 * t - 0.25, -0.1 * t - 0.25, 0.2] for t in range(1, 5)], dtype=torch.float32
    )
    prefix = producer.calls[1]["prev_chunk_left_over"]
    assert torch.allclose(prefix, expected, atol=1e-6)
    ex.stop()


def test_absolute_policy_prefix_is_the_raw_rows() -> None:
    """Regression: without rtc_prefix_fn the prefix is the raw leftover, postprocess or not."""
    producer = Producer()
    ex = ChunkedExecutor(
        chunk_fn=producer,
        chunk_size=CHUNK,
        prefetch_at=CHUNK - 1,
        rtc_config=_rtc_config(execution_horizon=4),
        postprocess_action=lambda row: row * 2.0 + 1.0,  # robot-facing copy differs
    )
    ex.start()
    served = ex.select_action({})
    _join_bg(ex)
    assert served == pytest.approx([201.0] * DOF)  # processed row 0 of chunk1
    prefix = producer.calls[1]["prev_chunk_left_over"]
    assert torch.equal(prefix, _chunk(100.0)[0, 1:5])  # raw rows 101..104, untouched
    ex.stop()


def test_relative_policy_without_prefix_fn_is_refused() -> None:
    from lerobot.policies.pi05 import PI05Config

    class _RelativePolicy:  # the real config is what the guard reads
        config = PI05Config(use_relative_actions=True)

    with pytest.raises(ROSConfigError, match="use_relative_actions"):
        ChunkedExecutor(
            _RelativePolicy(),
            chunk_fn=Producer(),
            chunk_size=CHUNK,
            prefetch_at=CHUNK - 1,
            rtc_config=_rtc_config(),
        )


def test_prefix_fn_without_postprocess_is_refused() -> None:
    """The hook reads the postprocessed (absolute) rows — without one there are none."""
    from openral_rskill._vla_core import relative_rtc_prefix_fn

    pre, _ = _relative_processors()
    with pytest.raises(ROSConfigError, match="postprocess_action"):
        ChunkedExecutor(
            chunk_fn=Producer(),
            chunk_size=CHUNK,
            prefetch_at=CHUNK - 1,
            rtc_config=_rtc_config(),
            rtc_prefix_fn=relative_rtc_prefix_fn(pre, device="cpu"),
        )


def test_merge_logs_tick_and_delay(cap: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """DEBUG, so the `cap` fixture (no level filter) rather than capture_logs."""
    import structlog
    from openral_rskill import executor

    # An earlier test may have pinned the module logger to a cached pipeline.
    monkeypatch.setattr(executor, "log", structlog.get_logger(executor.__name__))
    producer = Producer()
    producer.blocking = True
    ex = _executor(producer, prefetch_at=CHUNK - 1, execution_horizon=CHUNK)
    ex.select_action({})  # cold start merge (tick 0); prefetch launched, blocked
    for _ in range(3):
        ex.select_action({})
    producer.release.set()
    _join_bg(ex)
    merges = cap.named("chunked_executor.rtc_merge")
    # Cold start before any pop; then 4 served (1 at launch + 3 during inference).
    assert [(e["merge_tick"], e["rtc_delay"]) for e in merges] == [(0, 0), (4, 3)]
    ex.stop()
