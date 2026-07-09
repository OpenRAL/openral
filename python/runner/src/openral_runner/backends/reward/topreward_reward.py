"""In-process TOPReward reward monitor (``reward.backend: "topreward"``).

TOPReward (arXiv 2602.19313) is a **zero-shot** reward: it asks an off-the-shelf
Qwen3-VL VLM ``P("True" | video, instruction)`` and reads the token log-prob as
the signal. Unlike Robometer this needs **no fine-tuned checkpoint and no ZMQ
sidecar** — lerobot's stock :class:`~lerobot.rewards.topreward.TOPRewardModel`
loads the pre-quantized NF4 weights (``weights_uri``) 4-bit directly and runs in
the node's own process (transformers 5.x, bitsandbytes).

:class:`TOPRewardMonitor` mirrors
:class:`~openral_runner.backends.reward.robometer_reward.RobometerInProcessReward`'s
``score`` / ``assess`` / ``close`` surface, so ``reward_monitor_node`` and the
Reasoner's ``query_task_progress`` path consume it unchanged. The clip-level
scalar becomes a **per-frame** series via the same prefix sweep + per-window
min-max normalization used by lerobot's offline TOPReward labeler.

Nothing here imports torch / transformers at module load; the model is built
lazily on the first :meth:`TOPRewardMonitor.score`.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from openral_core import RSkillManifest
from openral_core.exceptions import ROSConfigError

if TYPE_CHECKING:
    from openral_runner.backends.reward.frame_source import Frame

_STALL_TREND_EPS = 0.002


class TOPRewardMonitor:
    """In-process TOPReward scorer with the reward-monitor ``score``/``assess`` API."""

    def __init__(
        self,
        *,
        model_id: str,
        weights_source: str,
        success_threshold: float = 0.8,
        max_frames: int = 8,
        num_samples: int = 6,
        fps: float = 2.0,
        device: str = "cuda",
    ) -> None:
        """Store config; the VLM is loaded lazily on first :meth:`score`.

        Args:
            model_id: Manifest name (for logs).
            weights_source: HF repo id / local dir of the pre-quantized NF4
                checkpoint (``weights_uri`` with the ``hf://`` prefix stripped).
            success_threshold: Manifest ``reward.success_threshold`` — the
                advisory bar stamped on ``succeeded`` in :meth:`assess`.
            max_frames: Frames per forward (each prefix is tail-cropped to this),
                to bound the Qwen3-VL activation on an 8 GB GPU.
            num_samples: Prefix-sweep anchor count (forwards per :meth:`score`);
                traded off against the S2 latency budget.
            fps: Frames-per-second metadata for the Qwen video processor.
            device: Torch device for the model.
        """
        self._model_id = model_id
        self._weights_source = weights_source
        self._success_threshold = float(success_threshold)
        self._max_frames = int(max_frames)
        self._num_samples = int(num_samples)
        self._fps = float(fps)
        self._device = device
        self._model: Any = None
        self._encoder: Any = None

    def _ensure_ready(self) -> None:
        """Load lerobot's stock TOPRewardModel (4-bit) + encoder once."""
        if self._model is not None:
            return
        # expandable_segments must precede the first CUDA alloc (8 GB NF4 fit).
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        try:
            from lerobot.rewards.topreward.configuration_topreward import (  # noqa: PLC0415
                TOPRewardConfig,
            )
            from lerobot.rewards.topreward.modeling_topreward import (  # noqa: PLC0415
                TOPRewardModel,
            )
            from lerobot.rewards.topreward.processor_topreward import (  # noqa: PLC0415
                TOPRewardEncoderProcessorStep,
            )
        except ImportError as exc:  # pragma: no cover — env-dependent
            raise ROSConfigError(
                "TOPReward monitor needs lerobot 0.6.0 + transformers + bitsandbytes; "
                "install with `uv sync --group topreward`."
            ) from exc

        # success_threshold stays -inf (TOPRewardConfig default) so compute_reward
        # returns the raw log-prob; score() normalizes the prefix sweep.
        cfg = TOPRewardConfig(vlm_name=self._weights_source, device=self._device, fps=self._fps)
        self._model = TOPRewardModel(cfg).eval()
        self._encoder = TOPRewardEncoderProcessorStep(
            vlm_name=cfg.vlm_name,
            image_key=cfg.image_key,
            task_key=cfg.task_key,
            default_task=cfg.default_task,
            max_frames=self._max_frames,
            fps=cfg.fps,
            prompt_prefix=cfg.prompt_prefix,
            prompt_suffix_template=cfg.prompt_suffix_template,
            add_chat_template=cfg.add_chat_template,
            max_length=cfg.max_input_length,
        )

    def _score_clip(self, video: object, task: str) -> float:
        """One raw ``log P("True")`` for a clip tensor ``(1, T, H, W, 3)`` uint8."""
        import torch  # noqa: PLC0415
        from lerobot.types import TransitionKey  # noqa: PLC0415

        transition = {
            TransitionKey.OBSERVATION: {self._encoder.image_key: video},
            TransitionKey.COMPLEMENTARY_DATA: {"task": task},
        }
        obs = self._encoder(transition)[TransitionKey.OBSERVATION]
        batch = {
            k: (v.to(self._device) if isinstance(v, torch.Tensor) else v) for k, v in obs.items()
        }
        with torch.no_grad():
            return float(self._model.compute_reward(batch).item())

    def score(self, frames: list[Frame], task: str) -> tuple[list[float], list[float]]:
        """Score a clip → ``(progress, success)`` per-frame lists in ``[0, 1]``.

        Prefix-sweep: score ``frames[:L]`` at ``num_samples`` anchor lengths,
        min-max normalize raw log-probs within the queried window, then
        interpolate to one value per frame. ``success`` mirrors ``progress`` —
        TOPReward has a single head, so normalized progress IS the signal.
        """
        import numpy as np  # noqa: PLC0415
        import torch  # noqa: PLC0415

        if not frames:
            raise ROSConfigError("reward score requires at least one frame")
        if not task.strip():
            raise ROSConfigError("reward score requires a non-empty task instruction")
        w, h = frames[0].width, frames[0].height
        if any(f.width != w or f.height != h for f in frames):
            raise ROSConfigError("all frames in a clip must share width/height")

        self._ensure_ready()
        # BGR888 bytes -> RGB uint8 (T, H, W, 3)
        video = torch.stack(
            [
                torch.from_numpy(
                    np.ascontiguousarray(
                        np.frombuffer(f.bgr, dtype=np.uint8).reshape(h, w, 3)[:, :, ::-1]
                    )
                )
                for f in frames
            ]
        )
        n = len(frames)
        anchors = np.unique(np.linspace(1, n, min(self._num_samples, n)).round().astype(np.int64))
        vals = np.asarray(
            [self._score_clip(video[: int(L)].unsqueeze(0), task.strip()) for L in anchors],
            dtype=np.float64,
        )
        if vals.size == 1 or float(vals.max()) == float(vals.min()):
            vals = np.ones_like(vals, dtype=np.float64)
        else:
            vals = (vals - vals.min()) / (vals.max() - vals.min())
        if len(anchors) == n:
            progress = vals
        else:
            progress = np.interp(
                np.arange(1, n + 1, dtype=np.float64), anchors.astype(np.float64), vals
            )
        series = [float(x) for x in progress]
        return series, list(series)

    def assess(self, frames: list[Frame], task: str) -> dict[str, Any]:
        """Score ``frames`` and summarize the window for the Reasoner.

        Same keys as
        :meth:`~openral_runner.backends.reward.robometer_reward.RobometerInProcessReward.assess`.
        """
        from openral_runner.backends.reward.frame_source import trend  # noqa: PLC0415

        progress, success = self.score(frames, task)
        p_trend = trend(progress)
        return {
            "progress_now": progress[-1],
            "success_now": success[-1],
            "progress_trend": p_trend,
            "success_trend": trend(success),
            "stalled": abs(p_trend) < _STALL_TREND_EPS,
            "succeeded": success[-1] >= self._success_threshold,
            "frames_seen": len(frames),
        }

    def close(self) -> None:
        """Release the in-process model + free CUDA memory."""
        self._encoder = None
        if self._model is not None:
            self._model = None
            try:
                import torch  # noqa: PLC0415

                torch.cuda.empty_cache()
            except Exception:  # pragma: no cover — torch optional / no CUDA
                pass


def build_topreward_monitor(manifest: RSkillManifest, *, device: str = "cuda") -> TOPRewardMonitor:
    """Build a :class:`TOPRewardMonitor` from a ``reward.backend == "topreward"`` manifest."""
    if manifest.kind != "reward" or manifest.reward is None:
        raise ROSConfigError(
            f"build_topreward_monitor requires a reward manifest, got {manifest.name!r}"
        )
    raw = manifest.weights_uri or manifest.source_repo or "OpenRAL/rskill-topreward-qwen3vl-4b-nf4"
    weights_source = (
        raw.removeprefix("local://")
        if raw.startswith("local://")
        else raw.removeprefix("hf://").split("@", 1)[0]
    )
    return TOPRewardMonitor(
        model_id=manifest.name,
        weights_source=weights_source,
        success_threshold=manifest.reward.success_threshold,
        fps=manifest.reward.target_fps,
        device=device,
    )
