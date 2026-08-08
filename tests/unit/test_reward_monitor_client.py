"""Unit tests for the Robometer reward backend factory + input guards.

No GPU needed for most tests — these cover manifest wiring and pre-flight
validation. The live scoring path is gated on local GPU/deps.

Run with:
    uv run pytest tests/unit/test_reward_monitor_client.py -v
"""

from __future__ import annotations

import importlib.util
import pathlib
import shutil

import pytest
import yaml
from openral_core.exceptions import ROSConfigError
from openral_core.schemas import RSkillManifest
from openral_runner.backends.reward.frame_source import Frame
from openral_runner.backends.reward.robometer_reward import (
    RobometerInProcessReward,
    build_reward_monitor,
)

_REPO_ROOT = pathlib.Path(__file__).parent.parent.parent
_FIXTURE = _REPO_ROOT / "rskills" / "robometer-4b" / "rskill.yaml"


def _load_manifest() -> RSkillManifest:
    with open(_FIXTURE, encoding="utf-8") as fh:
        return RSkillManifest.model_validate(yaml.safe_load(fh))


def test_build_reward_monitor_propagates_contract() -> None:
    """The factory carries num_bins + success_threshold + weights from the manifest."""
    manifest = _load_manifest()
    mon = build_reward_monitor(manifest)
    assert isinstance(mon, RobometerInProcessReward)
    assert mon._num_bins == manifest.reward.num_bins
    assert mon._success_threshold == manifest.reward.success_threshold
    # hf:// scheme stripped from the weights source; the manifest now points at the
    # published pre-quantized NF4 repo (the sidecar meta-loads it directly as 4-bit).
    assert mon._weights_source == "OpenRAL/rskill-robometer_4b-any-general-nf4"


def test_build_reward_monitor_local_scheme() -> None:
    """``local://`` weights resolve to a bare path (pre-quantized checkpoint dir)."""
    manifest = _load_manifest()
    local = manifest.model_copy(update={"weights_uri": "local:///tmp/robometer-nf4-ckpt"})
    mon = build_reward_monitor(local)
    # local:// stripped to the absolute dir; the in-process loader meta-loads it directly.
    assert mon._weights_source == "/tmp/robometer-nf4-ckpt"


def test_evenly_spaced_indices_bounds_frames() -> None:
    """Subsampling a frame window keeps a bounded, end-inclusive, unique set."""
    from openral_runner.backends.reward.robometer_reward import _evenly_spaced_indices

    # n <= k: identity (no subsampling).
    assert _evenly_spaced_indices(5, 8) == [0, 1, 2, 3, 4]
    # n > k: exactly k unique, sorted, spanning first..last (newest always kept).
    idx = _evenly_spaced_indices(19, 8)  # the openarm-window case observed live
    assert len(idx) == 8
    assert idx == sorted(set(idx))
    assert idx[0] == 0 and idx[-1] == 18
    assert all(0 <= i < 19 for i in idx)
    # k == 1: just the newest frame.
    assert _evenly_spaced_indices(20, 1) == [19]


def test_build_reward_monitor_rejects_wrong_kind() -> None:
    """A non-reward manifest is rejected by the factory."""
    manifest = _load_manifest()
    bad = manifest.model_copy(update={"kind": "vlm", "reward": None})
    with pytest.raises(ROSConfigError, match="requires kind='reward'"):
        build_reward_monitor(bad)


def test_score_rejects_empty_clip() -> None:
    """Scoring with no frames is a config error (never loads the model)."""
    mon = RobometerInProcessReward(model_id="t")
    with pytest.raises(ROSConfigError, match="at least one frame"):
        mon.score([], "do the task")


def test_score_rejects_empty_task() -> None:
    """Scoring with a blank task is a config error."""
    mon = RobometerInProcessReward(model_id="t")
    f = Frame(stamp_ns=0, bgr=b"\x00\x00\x00", width=1, height=1)
    with pytest.raises(ROSConfigError, match="non-empty task"):
        mon.score([f], "   ")


def test_score_rejects_mismatched_frame_sizes() -> None:
    """All frames in a clip must share width/height."""
    mon = RobometerInProcessReward(model_id="t")
    frames = [
        Frame(stamp_ns=0, bgr=b"\x00\x00\x00", width=1, height=1),
        Frame(stamp_ns=1, bgr=b"\x00" * 12, width=2, height=2),
    ]
    with pytest.raises(ROSConfigError, match="share width/height"):
        mon.score(frames, "do the task")


def _gpu_present() -> bool:
    """Whether a CUDA device is actually reachable *from torch*.

    ``nvidia-smi`` on PATH proves the driver is installed; it does not prove
    the installed torch wheel was built with CUDA. On a host that has an
    NVIDIA GPU but a CPU-only torch, the nvidia-smi probe passes and the test
    then dies inside torch with "Torch not compiled with CUDA enabled"
    instead of skipping (CLAUDE.md §1.11 — an unavailable dependency skips).
    Mirrors the ``torch.cuda.is_available()`` guard already used by
    ``tests/unit/test_release_torch_modules.py``.
    """
    if shutil.which("nvidia-smi") is None:
        return False
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def _missing_live_deps() -> list[str]:
    return [
        mod
        for mod in ("bitsandbytes", "huggingface_hub", "lerobot", "safetensors", "torch")
        if importlib.util.find_spec(mod) is None
    ]


@pytest.mark.skipif(
    not _gpu_present() or _missing_live_deps(),
    reason="needs a local GPU plus Robometer deps in the current env",
)
def test_e2e_score_clip_in_process() -> None:
    """The real NF4 Robometer backend scores a clip end-to-end via build_reward_monitor.

    Exercises the production path (manifest → in-process lerobot scorer) and
    asserts per-frame progress/success arrays of the right shape and range.
    """
    import numpy as np

    manifest = _load_manifest()
    mon = build_reward_monitor(manifest)
    n, h, w = 6, 224, 224
    frames = [
        Frame(
            stamp_ns=i * 333_000_000,
            bgr=np.random.randint(0, 255, (h, w, 3), dtype=np.uint8).tobytes(),
            width=w,
            height=h,
        )
        for i in range(n)
    ]
    try:
        progress, success = mon.score(frames, "pick up the cube and place it in the bowl")
        assert len(progress) == n
        assert len(success) == n
        assert all(0.0 <= s <= 1.0 for s in success)
        assert all(0.0 <= p <= 1.0 for p in progress)  # discrete mode → [0,1]
    finally:
        mon.close()
