"""π0.5 with ``policy_extras.cuda_graph``: the real forward captures, and a replay equals eager.

The graph path exists for the deploy runtime on a Jetson AGX Orin, where an
eager π0.5 forward takes 7.2-7.5 s inside the live graph against 1.95 s idle
(GIL contention over ~67k kernel launches) and trips the 5 s deadman between
chunks. This drives the real adapter (``_build_pi05``) on the real LIBERO
checkpoint behind ``rskills/pi05-libero-int8``, so lerobot's actual
``predict_action_chunk`` — including the three per-forward host uploads the
pinned-upload shim makes capturable — goes through capture and replay.

bf16 on a >= 10 GiB GPU (Orin / Thor / a desktop card), else the rSkill's own
int8 (an 8 GiB laptop). Needs the cached checkpoint (~7.5 GB) and a GPU; absent
either is the legitimate skip (CLAUDE.md §1.12). Run under the shared GPU lock:
    flock <gpu.lock> ./.venv/bin/pytest tests/sim/test_pi05_cuda_graph.py -v
"""

from __future__ import annotations

import gc
import statistics
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from openral_rskill.quantization_gate import pro_quantization_installed

torch = pytest.importorskip("torch")
pytest.importorskip("lerobot.policies.pi05.modeling_pi05")

_REPO = Path(__file__).resolve().parents[2]
_RSKILL = _REPO / "rskills" / "pi05-libero-int8"
_CONFIG = _REPO / "scenes" / "sim" / "libero_spatial.yaml"
_HF_REPO = "lerobot/pi05_libero_finetuned_v044"
_BF16_MIN_VRAM_GIB = 10
_PROMPT = "pick up the black bowl between the plate and the ramekin and place it on the plate"

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a local GPU")


def _dtype() -> str:
    vram = float(torch.cuda.get_device_properties(0).total_memory) / 2**30
    if vram >= _BF16_MIN_VRAM_GIB:
        return "bf16"
    if not pro_quantization_installed():
        pytest.skip(f"{vram:.1f} GiB needs the int8 path, which needs OpenRAL Pro (ADR-0123)")
    return "int8"


@pytest.fixture(scope="module")
def adapter() -> Iterator[Any]:
    from huggingface_hub import try_to_load_from_cache
    from openral_sim.policies.pi05 import _build_pi05

    from tests.sim.conftest import compose_sim_env

    if not isinstance(try_to_load_from_cache(_HF_REPO, "model.safetensors"), str):
        pytest.skip(f"{_HF_REPO} is not in the local HF cache (~7.5 GB download)")
    env = compose_sim_env(_CONFIG, rskill_uri=str(_RSKILL))
    extra = {**env.vla.extra, "dtype": _dtype(), "cuda_graph": True}
    vla = env.vla.model_copy(update={"device": "cuda:0", "extra": extra})
    built = _build_pi05(env.model_copy(update={"vla": vla}))
    yield built
    built.close()
    gc.collect()
    torch.cuda.empty_cache()


def _observation(adapter: Any, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    images = {k: rng.integers(0, 256, (224, 224, 3), dtype=np.uint8) for k in adapter._camera_keys}
    state = rng.normal(size=adapter._state_dim or 8).astype(np.float32)
    return {"images": images, "state": state, "task": _PROMPT}


def _timed(fn: Any, n: int) -> float:
    ts = []
    for _ in range(n):
        torch.cuda.synchronize()
        t = time.monotonic()
        fn()
        torch.cuda.synchronize()
        ts.append(time.monotonic() - t)
    return statistics.median(ts)


def test_the_real_forward_captures_and_replays_eager_exactly(adapter: Any) -> None:
    graphed = adapter._cuda_graph
    assert graphed is not None
    first = adapter._prepared_batch(_observation(adapter, 0), _PROMPT)
    adapter._chunk_forward(first)  # captures through the pinned-upload shim
    assert graphed.captured, graphed.disabled_reason
    assert adapter._cuda_graph_keepalive  # the recorded uploads' host buffers are held

    batch = adapter._prepared_batch(_observation(adapter, 1), _PROMPT)
    torch.manual_seed(11)
    eager = adapter._graph_forward(batch)
    torch.manual_seed(11)
    replay = adapter._chunk_forward(batch)
    assert replay.shape == eager.shape
    assert torch.isfinite(replay).all()
    assert torch.equal(replay, eager), (replay - eager).abs().max().item()


def test_a_replay_is_faster_than_eager(adapter: Any) -> None:
    batch = adapter._prepared_batch(_observation(adapter, 2), _PROMPT)
    adapter._chunk_forward(batch)
    assert adapter._cuda_graph.captured
    eager = _timed(lambda: adapter._graph_forward(batch), 3)
    replay = _timed(lambda: adapter._chunk_forward(batch), 5)
    print(f"pi05 {_dtype()} chunk: eager {eager * 1e3:.0f} ms, replay {replay * 1e3:.0f} ms")
    assert replay < eager
