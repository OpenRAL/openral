#!/usr/bin/env python
"""Prove lerobot 0.6.0 TOPReward yields a per-frame progress curve on 8 GB.

TOPReward (``lerobot.rewards.topreward``) is a *zero-shot, clip-level* reward:
``compute_reward`` returns a single ``log P("True" | video, instruction)`` scalar.
Per-frame progress is not a separate model — it is the **prefix sweep** lerobot
ships in :mod:`lerobot.rewards.topreward.compute_rabc_weights`: score growing
trajectory prefixes, min-max normalise per episode, interpolate to one value per
frame (``[0, 1]``). This script reuses that exact machinery; the only thing we
add is NF4 quantization of the Qwen3-VL backbone, because every lerobot 0.6.0
reward model loads bf16 and an 8 B (or even 4 B) bf16 VLM does not fit the 8 GB
dev GPU.

Run (downloads Qwen3-VL-4B + one LIBERO episode the first time)::

    .venv/bin/python tools/topreward_per_frame_demo.py

Exits non-zero (asserts) if the curve is not a valid rising [0,1] progress
signal on a known-successful demo episode. ponytail: no test framework — one
real episode, real weights, real asserts.
"""

from __future__ import annotations

# expandable_segments must be set before the first CUDA allocation (fits NF4 +
# an 8-frame Qwen3-VL forward on 8 GB — same trick as the MolmoAct2/Robometer rSkills).
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
from typing import Any, cast

import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.rewards.pretrained import PreTrainedRewardModel
from lerobot.rewards.topreward.compute_rabc_weights import (
    compute_instruction_rewards_for_prefixes,
)
from lerobot.rewards.topreward.configuration_topreward import TOPRewardConfig
from lerobot.rewards.topreward.modeling_topreward import TOPRewardModel
from lerobot.rewards.topreward.processor_topreward import TOPRewardEncoderProcessorStep
from numpy.typing import NDArray
from transformers import BitsAndBytesConfig, Qwen3VLForConditionalGeneration


class NF4TOPRewardModel(TOPRewardModel):  # type: ignore[misc]  # reason: lerobot model is untyped.
    """TOPReward whose Qwen3-VL backbone is loaded in NF4 (4-bit) to fit 8 GB.

    lerobot's :class:`TOPRewardModel.__init__` hard-codes ``model_kwargs`` with
    no quantization knob, so we override it to inject a bitsandbytes NF4 config.
    Everything downstream (``compute_reward``, the processor) is unchanged — the
    model is still a frozen zero-shot scorer.
    """

    def __init__(self, config: TOPRewardConfig) -> None:
        # Skip TOPRewardModel.__init__ (bf16 loader); run the abstract base only.
        PreTrainedRewardModel.__init__(self, config)
        self.config = config
        bits_and_bytes_config = cast(Any, BitsAndBytesConfig)
        qcfg = bits_and_bytes_config(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            config.vlm_name,
            quantization_config=qcfg,
            dtype=torch.bfloat16,
            device_map={"": 0},
            trust_remote_code=True,
        )


def per_frame_progress(
    *,
    dataset_repo_id: str,
    episode: int,
    vlm_name: str,
    image_key: str,
    num_samples: int,
    max_frames: int,
    device: str = "cuda",
) -> tuple[NDArray[np.float32], str, int, NDArray[np.uint8]]:
    """Return the TOPReward per-frame progress curve for one dataset episode.

    Returns ``(progress[num_frames] in [0,1], task_string, num_frames,
    frames[num_frames, H, W, 3] uint8)``.
    """
    config = TOPRewardConfig(
        vlm_name=vlm_name,
        device=device,
        image_key=image_key,
        # max_frames bounds every prefix forward to `max_frames` (tail-crop),
        # so the full-episode prefix stays inside 8 GB.
        max_frames=max_frames,
    )
    model = NF4TOPRewardModel(config).eval()
    encoder = TOPRewardEncoderProcessorStep(
        vlm_name=config.vlm_name,
        image_key=config.image_key,
        task_key=config.task_key,
        default_task=config.default_task,
        max_frames=max_frames,
        fps=config.fps,
        prompt_prefix=config.prompt_prefix,
        prompt_suffix_template=config.prompt_suffix_template,
        add_chat_template=config.add_chat_template,
        max_length=config.max_input_length,
    )

    ds = LeRobotDataset(dataset_repo_id, episodes=[episode], download_videos=True)
    ep = ds.meta.episodes[episode]
    ep_start = int(ep["dataset_from_index"])
    num_frames = int(ep["dataset_to_index"]) - ep_start
    sample: dict[str, Any] = ds[ep_start]
    task = sample.get("task") or config.default_task or "perform the task"

    progress = compute_instruction_rewards_for_prefixes(
        model=model,
        encoder=encoder,
        dataset=ds,
        ep_start=ep_start,
        num_frames=num_frames,
        task=task,
        image_key=image_key,
        num_samples=num_samples,
        device=device,
    )
    # Grab the RGB frames (same key/order) as uint8 HWC for overlay rendering.
    frames = np.stack(
        [
            (
                ds[ep_start + i][image_key]
                .permute(1, 2, 0)
                .clamp(0, 1)
                .mul(255)
                .to(torch.uint8)
                .numpy()
            )
            for i in range(num_frames)
        ]
    )
    return np.asarray(progress, dtype=np.float32), task, num_frames, frames


def render_overlay(frame: NDArray[np.uint8], value: float, task: str) -> NDArray[np.uint8]:
    """Draw a progress bar + value + task caption under an RGB frame (uint8 HWC)."""
    import cv2

    h, w = frame.shape[:2]
    scale = max(1, 384 // w)
    img = cv2.resize(frame, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    W = img.shape[1]
    panel = np.full((70, W, 3), 24, np.uint8)
    canvas = np.vstack([img, panel])
    y0 = img.shape[0]
    # progress bar
    x1, x2, by = 12, W - 12, y0 + 44
    cv2.rectangle(canvas, (x1, by), (x2, by + 14), (70, 70, 70), -1)
    fillx = x1 + int((x2 - x1) * float(np.clip(value, 0, 1)))
    cv2.rectangle(canvas, (x1, by), (fillx, by + 14), (80, 200, 90), -1)
    cv2.putText(
        canvas,
        f"progress {value:0.2f}",
        (x1, y0 + 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (235, 235, 235),
        1,
        cv2.LINE_AA,
    )
    cap = task if len(task) < 46 else task[:43] + "..."
    cv2.putText(
        canvas, cap, (x1, y0 + 68), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (170, 200, 235), 1, cv2.LINE_AA
    )
    return cast(NDArray[np.uint8], cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))


def write_media(
    frames: NDArray[np.uint8], progress: NDArray[np.float32], task: str, media_dir: str
) -> None:
    """Write progress.mp4 + start/mid/end stills with the progress overlay."""
    import imageio.v2 as imageio

    os.makedirs(media_dir, exist_ok=True)
    overlaid = [render_overlay(frames[i], float(progress[i]), task) for i in range(len(frames))]
    imageio.mimwrite(
        os.path.join(media_dir, "progress.mp4"),
        cast(list[Any], overlaid),
        fps=15,
        codec="libx264",
        quality=8,
        macro_block_size=1,
    )
    n = len(frames)
    for label, idx in (("start", 2), ("mid", n // 2), ("end", n - 3)):
        imageio.imwrite(os.path.join(media_dir, f"frame_{label}.png"), overlaid[idx])


def _sparkline(x: NDArray[np.float32]) -> str:
    blocks = "▁▂▃▄▅▆▇█"
    return "".join(blocks[min(len(blocks) - 1, int(round(v * (len(blocks) - 1))))] for v in x)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="lerobot/libero_object_image")
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--vlm", default="Qwen/Qwen3-VL-4B-Instruct")
    p.add_argument("--image-key", default="observation.images.image")
    p.add_argument("--num-samples", type=int, default=10)
    p.add_argument("--max-frames", type=int, default=8)
    p.add_argument("--out", default="tools/_topreward_progress.png")
    p.add_argument("--media-dir", default=None, help="if set, write progress.mp4 + 3 stills here")
    args = p.parse_args()

    if not torch.cuda.is_available():
        print("SKIP: no CUDA GPU")
        return 0

    progress, task, num_frames, frames = per_frame_progress(
        dataset_repo_id=args.dataset,
        episode=args.episode,
        vlm_name=args.vlm,
        image_key=args.image_key,
        num_samples=args.num_samples,
        max_frames=args.max_frames,
    )
    peak_gb = torch.cuda.max_memory_allocated() / 1e9

    head = float(progress[: max(1, num_frames // 5)].mean())
    tail = float(progress[-max(1, num_frames // 5) :].mean())
    print(f"\ntask: {task!r}")
    print(f"frames: {num_frames}   peak VRAM: {peak_gb:.2f} GB   backbone: {args.vlm} (NF4)")
    print(
        f"per-frame progress [{progress.min():.2f}..{progress.max():.2f}]  {_sparkline(progress)}"
    )
    print(f"first-20% mean={head:.3f}   last-20% mean={tail:.3f}")

    # --- asserts: a valid, rising per-frame progress signal on a success demo ---
    assert progress.shape == (num_frames,), f"expected one value per frame, got {progress.shape}"
    assert progress.min() >= 0.0 and progress.max() <= 1.0, "progress must be in [0, 1]"
    assert np.isclose(progress.max(), 1.0) and np.isclose(progress.min(), 0.0), (
        "min-max curve should span [0,1]"
    )
    assert tail > head + 0.15, (
        f"progress should rise on a successful demo (head={head:.3f} tail={tail:.3f})"
    )
    assert peak_gb < 8.0, f"must fit 8 GB, peaked {peak_gb:.2f} GB"

    if args.media_dir:
        write_media(frames, progress, task, args.media_dir)
        print(f"wrote progress.mp4 + start/mid/end stills to {args.media_dir}")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure(figsize=(7, 3))
        plt.plot(progress, lw=2)
        plt.ylim(-0.05, 1.05)
        plt.xlabel("frame")
        plt.ylabel("progress")
        plt.title(f"TOPReward per-frame progress — {task}", fontsize=8)
        plt.tight_layout()
        plt.savefig(args.out, dpi=110)
        print(f"saved {args.out}")
    except Exception as e:
        print(f"(plot skipped: {e})")

    print("\nPASS: TOPReward produces a valid rising per-frame progress curve within 8 GB.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
