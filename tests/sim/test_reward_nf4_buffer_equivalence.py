"""The NF4 reward loaders must not leave a buffer uninitialised (issue #304).

PR #289 found π0.5's fast meta-init loader left the non-persistent
``embed_scale`` buffer as garbage: every state_dict tensor matched, yet forward
cosine was 0.03. A state_dict comparison cannot see that bug class, so each
reward loader is checked here against a reference whose buffers come from the
modules' own ``__init__``:

* Robometer (``tools/_robometer_scorer.py``) builds on meta and fills buffers
  from the checkpoint. The reference build keeps parameters on meta but builds
  buffers for real, then installs the same checkpoint. ``named_buffers()`` and
  the per-frame scores on a fixed real LIBERO clip must be bit-identical.
* TOPReward loads through stock ``transformers`` ``from_pretrained``. Its buffers
  must equal a real-init build of the same config.

Gated on a local GPU + the native deps; GPU-less CI is the legitimate skip
(CLAUDE.md §12). Run with (single GPU — take the shared lock):
    flock /tmp/openral-gpu.lock -c \
      "./.venv/bin/pytest tests/sim/test_reward_nf4_buffer_equivalence.py -v"
"""

from __future__ import annotations

import gc
import os
import pathlib
import sys

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("lerobot")
pytest.importorskip("bitsandbytes")
pytest.importorskip("accelerate")
pytest.importorskip("qwen_vl_utils")
pytest.importorskip("datasets")

_REPO = pathlib.Path(__file__).resolve().parents[2]
_ROBOMETER = "OpenRAL/rskill-robometer_4b-any-general-nf4"
_TOPREWARD = "OpenRAL/rskill-topreward_qwen3vl_4b-any-general-nf4"

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a local GPU")


def _buffers(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    # remove_duplicate=False: an aliased buffer (original_inv_freq) must still show.
    return {n: b.detach().cpu() for n, b in model.named_buffers(remove_duplicate=False)}


def _assert_same_buffers(got: dict[str, torch.Tensor], ref: dict[str, torch.Tensor]) -> None:
    assert got.keys() == ref.keys(), got.keys() ^ ref.keys()
    for name, r in ref.items():
        assert got[name].dtype == r.dtype, (name, got[name].dtype, r.dtype)
        assert torch.equal(got[name], r), name


def _free_cuda() -> None:
    gc.collect()
    torch.cuda.empty_cache()


def _clip(n_frames: int = 6) -> tuple[object, str]:
    """Real LIBERO success-demo frames (RGB uint8, T x H x W x 3) + their task."""
    import numpy as np
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset("lerobot/libero_object_image", episodes=[0], download_videos=True)
    ep = ds.meta.episodes[0]
    s, e = int(ep["dataset_from_index"]), int(ep["dataset_to_index"])
    key = "observation.images.image"
    frames = [
        ds[int(i)][key].permute(1, 2, 0).clamp(0, 1).mul(255).to(torch.uint8).numpy()
        for i in np.linspace(s, e - 1, n_frames).round().astype(int)
    ]
    return np.stack(frames), str(ds[s]["task"])


def test_robometer_meta_load_matches_real_buffer_reference() -> None:
    from openral_sim._sidecar_common import installed_alloc_conf_var

    os.environ.setdefault(installed_alloc_conf_var(), "expandable_segments:True")
    sys.path.insert(0, str(_REPO / "tools"))
    import _robometer_scorer as scorer_mod

    clip, task = _clip()

    # Sequential, not side by side: two 3.3 GB models do not fit an 8 GB GPU.
    meta = scorer_mod.Scorer(_ROBOMETER, device="cuda")
    meta_bufs = _buffers(meta.model)
    meta_score = meta.score(clip, task, num_bins=10)
    del meta
    _free_cuda()

    ref = scorer_mod.Scorer(_ROBOMETER, device="cuda", meta_buffers=False)
    ref_bufs = _buffers(ref.model)
    ref_score = ref.score(clip, task, num_bins=10)
    del ref
    _free_cuda()

    assert any(n.endswith("original_inv_freq") for n in ref_bufs), ref_bufs.keys()
    _assert_same_buffers(meta_bufs, ref_bufs)
    # Determinism is pinned (math SDP), so equal buffers + weights => equal scores.
    assert meta_score == ref_score, (meta_score, ref_score)


def test_topreward_load_buffers_match_real_init() -> None:
    from accelerate import init_empty_weights
    from openral_runner.backends.reward.topreward_reward import TOPRewardMonitor
    from transformers import AutoConfig, Qwen3VLForConditionalGeneration

    with init_empty_weights(include_buffers=False):
        ref = Qwen3VLForConditionalGeneration._from_config(AutoConfig.from_pretrained(_TOPREWARD))
    monitor = TOPRewardMonitor(model_id="topreward", weights_source=_TOPREWARD)
    try:
        monitor._ensure_ready()
        got = _buffers(monitor._model.model)
    finally:
        monitor.close()
        _free_cuda()

    assert any(n.endswith("original_inv_freq") for n in got), got.keys()
    _assert_same_buffers(got, _buffers(ref))
