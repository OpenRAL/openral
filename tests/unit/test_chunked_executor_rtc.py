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
    ex = _executor(producer, prefetch_at=CHUNK - 1)  # trigger on the very first pop
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


def test_real_delay_is_actions_consumed_during_inference() -> None:
    producer = Producer()
    producer.blocking = True
    ex = _executor(producer, prefetch_at=CHUNK - 1)
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
    ex = _executor(producer, prefetch_at=CHUNK - 1)
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
    ex = _executor(producer, prefetch_at=2)
    drained = [ex.select_action({}) for _ in range(CHUNK)]  # consume ALL of chunk1
    assert len(drained) == CHUNK

    def _unblock() -> None:
        producer.release.set()

    threading.Timer(0.2, _unblock).start()
    nxt = ex.select_action({})  # empty queue + pending prefetch -> blocks
    assert nxt.squeeze(0)[0].item() >= 200.0
    ex.stop()


def test_bg_error_propagates() -> None:
    class Boom(Producer):
        def __call__(self, batch: Any, **kwargs: Any) -> torch.Tensor:
            if self.calls:
                raise RuntimeError("kaput")
            return super().__call__(batch, **kwargs)

    producer = Boom()
    ex = _executor(producer, prefetch_at=CHUNK - 1)
    ex.select_action({})
    _join_bg(ex)
    with pytest.raises(ROSRuntimeError, match="kaput"):
        for _ in range(CHUNK + 1):
            ex.select_action({})
    ex.stop()


def test_reset_clears_rtc_state() -> None:
    producer = Producer()
    ex = _executor(producer)
    ex.select_action({})
    ex.reset()
    action = ex.select_action({})  # fresh cold start
    assert producer.calls[-1]["prev_chunk_left_over"] is None
    assert action.shape == (1, DOF)
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
