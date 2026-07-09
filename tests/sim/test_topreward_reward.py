"""Live TOPReward reward-monitor backend on a real clip (lerobot 0.6.0).

Exercises the deploy path end-to-end: `build_reward_monitor` dispatches the
`reward.backend: "topreward"` manifest to the in-process `TOPRewardMonitor`,
which loads the HOSTED pre-quantized NF4 checkpoint (weights_uri) 4-bit directly
and scores real LIBERO frames (converted to the node's BGR888 `Frame`s). Asserts
lerobot-style min-max normalized progress + the reasoner `assess` dict.

Gated on a local GPU + the topreward deps; CI runners without a GPU skip.
"""

from __future__ import annotations

import os

import pytest

pytest.importorskip("torch")
pytest.importorskip("lerobot")
pytest.importorskip("bitsandbytes")

import numpy as np
import torch

_HAS_GPU = torch.cuda.is_available()


@pytest.mark.skipif(not _HAS_GPU, reason="needs a local GPU + lerobot TOPReward + NF4 checkpoint")
def test_topreward_monitor_scores_real_clip() -> None:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from openral_core.schemas import RSkillManifest
    from openral_runner.backends.reward import TOPRewardMonitor, build_reward_monitor
    from openral_runner.backends.reward.frame_source import Frame

    manifest = RSkillManifest.from_yaml("rskills/topreward-qwen3vl-4b-nf4/rskill.yaml")
    assert manifest.reward is not None and manifest.reward.backend == "topreward"
    monitor = build_reward_monitor(manifest)
    assert isinstance(monitor, TOPRewardMonitor)

    # Real LIBERO success demo → sample 16 frames spanning the whole attempt.
    ds = LeRobotDataset("lerobot/libero_object_image", episodes=[0], download_videos=True)
    ep = ds.meta.episodes[0]
    s, e = int(ep["dataset_from_index"]), int(ep["dataset_to_index"])
    idx = np.linspace(s, e - 1, 16).round().astype(int)
    key = "observation.images.image"
    task = ds[s].get("task")

    frames: list[Frame] = []
    for i in idx:
        rgb = (
            ds[int(i)][key].permute(1, 2, 0).clamp(0, 1).mul(255).to(torch.uint8).numpy()
        )  # HWC RGB
        bgr = np.ascontiguousarray(rgb[:, :, ::-1])
        h, w = bgr.shape[:2]
        frames.append(Frame(stamp_ns=int(i) * 1_000_000_000, bgr=bgr.tobytes(), width=w, height=h))

    try:
        progress, success = monitor.score(frames, task)
        assess = monitor.assess(frames, task)
    finally:
        monitor.close()

    peak = torch.cuda.max_memory_allocated() / 1e9
    head = float(np.mean(progress[:4]))
    tail = float(np.mean(progress[-4:]))
    print(f"\ntask={task!r} frames={len(frames)} peak={peak:.2f}GB head={head:.3f} tail={tail:.3f}")
    print("assess:", {k: round(v, 3) if isinstance(v, float) else v for k, v in assess.items()})

    # per-frame series is a min-max normalized [0,1] curve, one per frame
    assert len(progress) == len(frames) == len(success)
    assert all(0.0 <= p <= 1.0 for p in progress)
    assert max(progress) == pytest.approx(1.0)
    assert min(progress) == pytest.approx(0.0)
    assert tail > head + 0.1, f"progress should rise (head={head:.3f} tail={tail:.3f})"
    # assess exposes the exact reasoner contract (same keys as Robometer)
    assert set(assess) == {
        "progress_now",
        "success_now",
        "progress_trend",
        "success_trend",
        "stalled",
        "succeeded",
        "frames_seen",
    }
    assert assess["progress_now"] == progress[-1]
    assert assess["succeeded"] is True
    assert assess["frames_seen"] == len(frames)
    assert peak < 8.0, f"must fit 8 GB, peaked {peak:.2f} GB"


if __name__ == "__main__":
    test_topreward_monitor_scores_real_clip()
    print("PASS")
