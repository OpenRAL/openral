"""Live native Robometer reward on a real clip (lerobot 0.6.0 port).

Runs the in-process scorer — lerobot's in-tree
``RobometerRewardModel`` (Qwen3-VL-4B + NF4 pre-quantized weights, loaded with
plain ``transformers`` and no ``robometer`` git package) — on a real LIBERO
rollout clip and asserts a per-frame progress/success series in ``[0, 1]``.

Gated on a local GPU + the native deps (lerobot Robometer + qwen-vl-utils); the
GPU-less CI runner is the legitimate skip path (CLAUDE.md §12). No mocks: real
checkpoint, real frames from ``example_videos/libero_spatial.mp4``.

Run with (single GPU — take the shared lock):
    flock /tmp/openral-gpu.lock -c \
      "PYTORCH_ALLOC_CONF=expandable_segments:True \
       ./.venv/bin/pytest tests/sim/test_robometer_native_reward.py -v"
"""

from __future__ import annotations

import os
import pathlib
import shutil
import sys

import pytest

_REPO = pathlib.Path(__file__).resolve().parents[2]
_MP4 = _REPO / "example_videos" / "libero_spatial.mp4"
_WEIGHTS = "OpenRAL/rskill-robometer-4b-nf4"
_N_FRAMES = 6
_RES = 256


def _gpu_present() -> bool:
    return shutil.which("nvidia-smi") is not None


def _native_available() -> bool:
    try:
        import lerobot.rewards.robometer.modeling_robometer  # noqa: F401
        import qwen_vl_utils  # noqa: F401

        return True
    except Exception:
        return False


@pytest.mark.skipif(
    not _gpu_present() or not _native_available() or not _MP4.exists(),
    reason="needs a local GPU + lerobot Robometer + qwen-vl-utils + a real clip",
)
def test_native_robometer_scores_real_clip() -> None:
    # expandable_segments must be set before the first CUDA alloc (8 GB fit).
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    import cv2
    import numpy as np

    sys.path.insert(0, str(_REPO / "tools"))
    import _robometer_scorer as scorer_mod

    cap = cv2.VideoCapture(str(_MP4))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idx = np.linspace(0, total - 1, _N_FRAMES).round().astype(int)
    frames = []
    for i in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, bgr = cap.read()
        assert ok, f"could not read frame {i}"
        bgr = cv2.resize(bgr, (_RES, _RES), interpolation=cv2.INTER_AREA)
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    clip = np.stack(frames).astype(np.uint8)
    assert clip.shape == (_N_FRAMES, _RES, _RES, 3)

    scorer = scorer_mod._Scorer(_WEIGHTS, device="cuda")
    progress, success = scorer.score(clip, "pick up the object and place it", num_bins=100)

    # Per-frame series, one value per input frame.
    assert len(progress) == _N_FRAMES, progress
    assert len(success) == _N_FRAMES, success
    # Discrete-mode progress + sigmoid success are both normalized to [0, 1].
    assert all(0.0 <= p <= 1.0 for p in progress), progress
    assert all(0.0 <= s <= 1.0 for s in success), success
