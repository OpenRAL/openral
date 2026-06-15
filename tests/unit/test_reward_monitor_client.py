"""Unit tests for the Robometer reward client factory + input guards (ADR-0057).

No GPU / sidecar needed — these cover manifest→client wiring and the client's
pre-flight validation (the live ZMQ scoring path is validated separately on a
GPU host, see rskills/robometer-4b/PHASE0.md Phase 3).

Run with:
    uv run pytest tests/unit/test_reward_monitor_client.py -v
"""

from __future__ import annotations

import os
import pathlib
import shutil

import pytest
import yaml
from openral_core.exceptions import ROSConfigError
from openral_core.schemas import RSkillManifest
from openral_runner.backends.reward.frame_source import Frame
from openral_runner.backends.reward.robometer_reward import (
    RobometerReward,
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
    mon = build_reward_monitor(manifest, port=5769)
    assert isinstance(mon, RobometerReward)
    assert mon._num_bins == manifest.reward.num_bins  # noqa: SLF001 — test asserts wiring
    assert mon._success_threshold == manifest.reward.success_threshold  # noqa: SLF001
    # hf:// scheme + @revision stripped from the weights source (loadable upstream)
    assert mon._weights_source == "robometer/Robometer-4B"  # noqa: SLF001 — @sha stripped


def test_build_reward_monitor_rejects_wrong_kind() -> None:
    """A non-reward manifest is rejected by the factory."""
    manifest = _load_manifest()
    bad = manifest.model_copy(update={"kind": "vlm", "reward": None})
    with pytest.raises(ROSConfigError, match="requires kind='reward'"):
        build_reward_monitor(bad)


def test_score_rejects_empty_clip() -> None:
    """Scoring with no frames is a config error (never reaches the sidecar)."""
    mon = RobometerReward(model_id="t", auto_spawn=False)
    with pytest.raises(ROSConfigError, match="at least one frame"):
        mon.score([], "do the task")


def test_score_rejects_empty_task() -> None:
    """Scoring with a blank task is a config error."""
    mon = RobometerReward(model_id="t", auto_spawn=False)
    f = Frame(stamp_ns=0, bgr=b"\x00\x00\x00", width=1, height=1)
    with pytest.raises(ROSConfigError, match="non-empty task"):
        mon.score([f], "   ")


def test_score_rejects_mismatched_frame_sizes() -> None:
    """All frames in a clip must share width/height."""
    mon = RobometerReward(model_id="t", auto_spawn=False)
    frames = [
        Frame(stamp_ns=0, bgr=b"\x00\x00\x00", width=1, height=1),
        Frame(stamp_ns=1, bgr=b"\x00" * 12, width=2, height=2),
    ]
    with pytest.raises(ROSConfigError, match="share width/height"):
        mon.score(frames, "do the task")


def _gpu_present() -> bool:
    return shutil.which("nvidia-smi") is not None


@pytest.mark.skipif(
    not os.environ.get("OPENRAL_ROBOMETER_SIDECAR_VENV") or not _gpu_present(),
    reason="needs a provisioned Robometer sidecar venv + a local GPU "
    "(set OPENRAL_ROBOMETER_SIDECAR_VENV).",
)
def test_e2e_score_clip_over_zmq() -> None:
    """The real NF4 Robometer sidecar scores a clip end-to-end via build_reward_monitor.

    Exercises the production wire (manifest → client → auto-spawned sidecar → ZMQ
    score) and asserts per-frame progress/success arrays of the right shape and
    range. Semantic correctness (progress ramping on a real rollout) is validated
    separately — see rskills/robometer-4b/PHASE0.md Phase 3.
    """
    import numpy as np

    manifest = _load_manifest()
    mon = build_reward_monitor(manifest, port=5770)
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
