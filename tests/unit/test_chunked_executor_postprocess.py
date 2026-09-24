"""ChunkedExecutor ``postprocess_action``: chunks are finished where they are produced.

The regression this guards was measured on the OpenArm/Thor cell with the pi0.5
restock policy, whose lerobot pipeline pairs ``RelativeActionsProcessorStep`` (pre)
with ``AbsoluteActionsProcessorStep`` (post). Postprocessing at pop time made the
post step add the state cached by the *latest* preprocessor call, and the prefetch
launch replaces that state mid-chunk, so the rest of the chunk was shifted by how far
the arm moved since the chunk's own inference. It also put a device-to-CPU copy on
every control tick, which waits behind the prefetch's CUDA kernels.

Uses lerobot's real processor steps and the executor's documented ``chunk_fn`` API.
"""

from __future__ import annotations

import threading
from typing import Any

import pytest
import torch
from openral_rskill.executor import ChunkedExecutor

CHUNK, DOF = 6, 3


def _processor_pair() -> tuple[Any, Any]:
    from lerobot.processor.relative_action_processor import (
        AbsoluteActionsProcessorStep,
        RelativeActionsProcessorStep,
    )

    pre = RelativeActionsProcessorStep(enabled=True)
    return pre, AbsoluteActionsProcessorStep(enabled=True, relative_step=pre)


def _preprocess(pre: Any, state: torch.Tensor) -> dict[str, Any]:
    from lerobot.processor.converters import create_transition
    from lerobot.utils.constants import OBS_STATE

    pre(create_transition(observation={OBS_STATE: state}))  # caches the state, as lerobot does
    return {OBS_STATE: state}


def _postprocess(post: Any) -> Any:
    from lerobot.processor.converters import create_transition
    from lerobot.types import TransitionKey

    return lambda a: post(create_transition(action=a))[TransitionKey.ACTION]


def _hold_position(batch: dict[str, Any], **_: Any) -> torch.Tensor:
    """A policy that predicts 'stay where you are': all-zero relative actions."""
    return torch.zeros(1, CHUNK, DOF)


def test_every_action_of_a_chunk_uses_its_own_inference_state() -> None:
    pre, post = _processor_pair()
    ex = ChunkedExecutor(
        chunk_fn=_hold_position,
        chunk_size=CHUNK,
        prefetch_at=3,
        postprocess_action=_postprocess(post),
    )
    ex.start()
    states: list[torch.Tensor] = []

    def batch() -> dict[str, Any]:
        # The arm has moved by the time each inference starts: 1.0, 2.0, 3.0, ...
        states.append(torch.full((1, DOF), float(len(states) + 1)))
        return _preprocess(pre, states[-1])

    popped = [ex.select_action(batch) for _ in range(2 * CHUNK)]
    ex.stop()
    assert len(states) >= 2
    # Chunk 0 was inferred at state 1.0: holding position means every action is 1.0,
    # including the 3 popped after the prefetch launch re-ran the preprocessor at 2.0.
    assert all(torch.equal(a, states[0]) for a in popped[:CHUNK])
    assert all(torch.equal(a, states[1]) for a in popped[CHUNK:])


def test_prefetched_chunk_is_postprocessed_off_the_control_thread() -> None:
    tick_thread = threading.get_ident()
    seen: list[int] = []

    def post(a: torch.Tensor) -> torch.Tensor:
        seen.append(threading.get_ident())
        return a

    ex = ChunkedExecutor(
        chunk_fn=_hold_position, chunk_size=CHUNK, prefetch_at=3, postprocess_action=post
    )
    ex.start()
    for _ in range(2 * CHUNK):
        ex.select_action({})
    ex.stop()
    assert seen[:CHUNK] == [tick_thread] * CHUNK  # cold start: nothing to overlap with
    assert tick_thread not in seen[CHUNK : 2 * CHUNK]  # prefetch: finished in the bg thread
    assert len(seen) >= 2 * CHUNK  # every action once; the next prefetch may have landed too


def test_rtc_postprocesses_at_merge_and_keeps_raw_rows_for_guidance() -> None:
    from lerobot.policies.rtc import RTCConfig

    pre, post = _processor_pair()
    post_fn = _postprocess(post)
    release, produced = threading.Event(), []

    def slow_prefetch(batch: dict[str, Any], **kwargs: Any) -> torch.Tensor:
        if produced:  # hold the prefetch until chunk 0 is popped past its launch
            assert release.wait(timeout=5.0), "test deadlock"
        produced.append(1)
        return _hold_position(batch)

    ex = ChunkedExecutor(
        chunk_fn=slow_prefetch,
        chunk_size=CHUNK,
        prefetch_at=3,
        rtc_config=RTCConfig(enabled=True, execution_horizon=2),
        postprocess_action=lambda a: post_fn(a).reshape(-1).numpy(),
    )
    ex.start()
    states: list[torch.Tensor] = []

    def batch() -> dict[str, Any]:
        states.append(torch.full((1, DOF), float(len(states) + 1)))
        return _preprocess(pre, states[-1])

    first = [ex.select_action(batch) for _ in range(CHUNK)]  # launch fires on the 3rd pop
    left_over = ex._rtc_queue.get_left_over()
    release.set()
    ex.stop()
    assert len(states) == 2  # the prefetch's preprocess ran and cached state 2.0
    # Served rows are finished NumPy vectors at the state of their own inference, even
    # after the prefetch launch re-ran the preprocessor; the queue's raw rows stay raw.
    assert all(a.shape == (DOF,) and (a == 1.0).all() for a in first)
    assert left_over is not None and torch.count_nonzero(left_over) == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_device_chunk_is_copied_to_host_once() -> None:
    def on_gpu(batch: Any, **_: Any) -> torch.Tensor:
        return torch.arange(CHUNK * DOF, dtype=torch.float32, device="cuda").view(1, CHUNK, DOF)

    ex = ChunkedExecutor(chunk_fn=on_gpu, chunk_size=CHUNK, prefetch_at=3)
    ex.start()
    popped = [ex.select_action({}) for _ in range(2 * CHUNK)]
    ex.stop()
    assert all(a.device.type == "cpu" for a in popped)
    assert torch.equal(popped[1], torch.tensor([[3.0, 4.0, 5.0]]))
