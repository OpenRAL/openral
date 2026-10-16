"""``CudaGraphChunkForward``: replay equals eager, never aliases, and falls back explicitly.

A small real CUDA network shaped like a flow-matching chunk forward (fresh noise
from the default generator every call, then a few denoise-style matmuls) stands
in for π0.5 here; the real checkpoint is exercised in
``tests/sim/test_pi05_cuda_graph.py``. Needs a CUDA device; skips without one
(CLAUDE.md §1.12).
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from typing import Any

import pytest

pytest.importorskip("torch")
import torch
from openral_rskill._vla_core import CudaGraphChunkForward

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")

_CHUNK, _DOF = 10, 8


class _Flow(torch.nn.Module):
    """Noise in, three denoise steps conditioned on the observation, chunk out."""

    def __init__(self) -> None:
        super().__init__()
        self.cond = torch.nn.Linear(16, _DOF)
        self.step = torch.nn.Linear(_DOF, _DOF)

    def forward(self, batch: dict[str, Any], **_: Any) -> Any:
        obs = batch["observation.state"]
        x = torch.randn(obs.shape[0], _CHUNK, _DOF, device=obs.device)
        c = self.cond(obs)[:, None, :]
        for _step in range(3):
            x = x - 0.3 * torch.tanh(self.step(x) + c)
        return x


@pytest.fixture
def flow() -> Any:
    torch.manual_seed(0)
    return _Flow().cuda().eval()


def _batch(seed: int, width: int = 16) -> dict[str, Any]:
    g = torch.Generator().manual_seed(seed)
    return {"observation.state": torch.randn(1, width, generator=g).cuda(), "task": "pick"}


def _wrapped(flow: Any, **kwargs: Any) -> CudaGraphChunkForward:
    def forward(batch: dict[str, Any], **kw: Any) -> Any:
        with torch.no_grad():
            return flow(batch, **kw)

    return CudaGraphChunkForward(forward, torch=torch, name="test", **kwargs)


def test_replay_equals_eager_under_the_same_seed(flow: Any) -> None:
    graphed = _wrapped(flow)
    graphed(_batch(1))  # capture
    assert graphed.captured
    for seed in (2, 3):
        batch = _batch(seed)
        torch.manual_seed(seed)
        with torch.no_grad():
            eager = flow(batch)
        torch.manual_seed(seed)
        assert torch.equal(graphed(batch), eager)


def test_each_replay_draws_fresh_noise(flow: Any) -> None:
    graphed = _wrapped(flow)
    batch = _batch(1)
    first, second = graphed(batch), graphed(batch)
    assert not torch.equal(first, second)


def test_a_held_chunk_is_not_overwritten_by_the_next_replay(flow: Any) -> None:
    graphed = _wrapped(flow)
    held = graphed(_batch(1))
    snapshot = held.clone()
    graphed(_batch(2))
    assert torch.equal(held, snapshot)


def test_inputs_reach_the_graph(flow: Any) -> None:
    """A replay must read the new batch, not the one it was captured with."""
    graphed = _wrapped(flow)
    graphed(_batch(1))
    torch.manual_seed(7)
    a = graphed(_batch(2))
    torch.manual_seed(7)
    b = graphed(_batch(3))
    assert not torch.equal(a, b)


def test_a_keyword_call_runs_eager_and_keeps_the_graph(flow: Any) -> None:
    graphed = _wrapped(flow)
    graphed(_batch(1))
    out = graphed(_batch(2), inference_delay=3)
    assert out.shape == (1, _CHUNK, _DOF)
    assert graphed.captured and graphed.disabled_reason is None


def test_a_changed_input_layout_latches_eager(flow: Any) -> None:
    graphed = _wrapped(flow)
    graphed(_batch(1))
    extra = {**_batch(2), "observation.extra": torch.zeros(1, 2, device="cuda")}
    out = graphed(extra)
    assert out.shape == (1, _CHUNK, _DOF)
    assert graphed.disabled_reason == "input_layout_changed"
    assert not graphed.captured
    assert graphed(_batch(3)).shape == (1, _CHUNK, _DOF)  # stays eager, still works


_FAILED_CAPTURE = """
import torch
from openral_core.exceptions import ROSRuntimeError
from openral_rskill._vla_core import CudaGraphChunkForward

lin = torch.nn.Linear(4, 4).cuda()

def syncing(batch):
    with torch.no_grad():
        out = lin(batch["x"] + torch.randn(1, 4, device="cuda"))
        float(out.sum().item())  # a host sync: illegal under capture
        return out

graphed = CudaGraphChunkForward(syncing, torch=torch, name="test")
try:
    graphed({"x": torch.zeros(1, 4, device="cuda")})
except ROSRuntimeError as exc:
    assert "Remove policy_extras.cuda_graph" in str(exc), exc
    assert graphed.disabled_reason == "capture_failed"
    print("RAISED")
"""


def test_a_failed_capture_raises_rather_than_fall_back() -> None:
    """Run in a subprocess: an aborted capture poisons the CUDA generator process-wide."""
    import subprocess
    import sys

    done = subprocess.run(
        [sys.executable, "-c", _FAILED_CAPTURE],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert done.returncode == 0, done.stderr[-2000:]
    assert done.stdout.strip().endswith("RAISED"), done.stdout[-500:]


def test_capture_ctx_wraps_warmup_and_capture_only(flow: Any) -> None:
    entered: list[int] = []

    @contextlib.contextmanager
    def ctx() -> Iterator[None]:
        entered.append(1)
        yield

    graphed = _wrapped(flow, capture_ctx=ctx)
    for seed in range(4):
        graphed(_batch(seed))
    assert entered == [1]
