"""Robometer reward-monitor backends.

The backend loads lerobot 0.6.0's native
Robometer model inside ``reward_monitor_node`` and scores clips on demand. That
keeps the heavy VLM out of the VLA runner / reasoner / HAL processes without a
second process boundary. Nothing here imports torch / transformers / numpy at
module load.
"""

from __future__ import annotations

import contextlib
import importlib.util
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast

from openral_core import RSkillManifest
from openral_core.exceptions import ROSConfigError

if TYPE_CHECKING:
    from openral_runner.backends.reward.frame_source import Frame
    from openral_runner.backends.reward.topreward_reward import TOPRewardMonitor


class _RobometerScorer(Protocol):
    def score(
        self, frames_rgb: object, task: str, num_bins: int
    ) -> tuple[list[float], list[float]]: ...


class _RobometerScorerClass(Protocol):
    def __call__(self, weights: str, device: str = "cuda") -> _RobometerScorer: ...


# |progress trend per sample| below this reads as "stalled" (no meaningful change).
_STALL_TREND_EPS = 0.002


def critic_score_from_assessment(
    assessment: Mapping[str, object], *, threshold: float
) -> tuple[float, float]:
    """Map a reward-monitor assessment to a generic critic ``(score, threshold)``.

    The Tier-C critic bus consumes a higher-is-better scalar per
    sample; a reward model's per-window ``progress_now`` (in [0, 1]) is exactly
    that. The producer's ``CriticWatchdogGroup`` decides when the score has
    *stalled* — this helper only normalises one reward assessment
    result into the ``openral_msgs/CriticScore`` ``(score, threshold)`` pair,
    clamping the score to ``[0, 1]`` and defaulting a missing/non-numeric
    ``progress_now`` to ``0.0`` (a conservative "no progress").

    Args:
        assessment: A Robometer assessment result. Reads
            ``progress_now``.
        threshold: The pass bar to stamp on the CriticScore — the watchdog fires
            when the score stays below it without improving.

    Returns:
        ``(score, threshold)`` ready for an ``openral_msgs/CriticScore``.

    Example:
        >>> critic_score_from_assessment({"progress_now": 0.42}, threshold=0.8)
        (0.42, 0.8)
        >>> critic_score_from_assessment({"progress_now": 1.5}, threshold=0.8)
        (1.0, 0.8)
    """
    raw = assessment.get("progress_now", 0.0)
    score = 0.0 if isinstance(raw, bool) or not isinstance(raw, (int, float)) else float(raw)
    score = max(0.0, min(1.0, score))
    return score, float(threshold)


def _evenly_spaced_indices(n: int, k: int) -> list[int]:
    """``k`` evenly-spaced indices into ``range(n)``, always including the last.

    Used to subsample a frame window to a fixed budget so the reward model's
    vision-transformer activation stays bounded on an 8 GB GPU. The
    newest frame (index ``n-1``) is always kept — the reasoner reads
    ``progress_now`` from it. Returns ``list(range(n))`` when ``n <= k``.
    """
    if n <= k:
        return list(range(n))
    step = (n - 1) / (k - 1) if k > 1 else 0.0
    idx = sorted({min(n - 1, round(i * step)) for i in range(k)})
    if idx[-1] != n - 1:
        idx[-1] = n - 1
    return idx


def _find_robometer_scorer_script() -> Path:
    """Locate ``tools/_robometer_scorer.py``."""
    for parent in Path(__file__).resolve().parents:
        cand = parent / "tools" / "_robometer_scorer.py"
        if cand.exists():
            return cand
    raise ROSConfigError("could not locate tools/_robometer_scorer.py")


def _load_inprocess_scorer_class() -> _RobometerScorerClass:
    """Load the Robometer NF4 scorer lazily."""
    scorer_path = _find_robometer_scorer_script()
    tools_dir = str(scorer_path.parent)
    if tools_dir not in sys.path:
        sys.path.insert(0, tools_dir)
    spec = importlib.util.spec_from_file_location("_openral_robometer_scorer", scorer_path)
    if spec is None or spec.loader is None:
        raise ROSConfigError(f"could not import Robometer scorer from {scorer_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    scorer = getattr(module, "_Scorer", None)
    if scorer is None:
        raise ROSConfigError(f"Robometer scorer {scorer_path} does not expose _Scorer")
    return cast(_RobometerScorerClass, scorer)


def _bgr_frames_to_rgb_array(frames: list[Frame]) -> object:
    """Convert validated BGR888 frames to the RGB ndarray Robometer expects."""
    import numpy as np  # type: ignore[import-not-found,import-untyped,unused-ignore]  # noqa: PLC0415

    w, h = frames[0].width, frames[0].height
    bgr = np.frombuffer(b"".join(f.bgr for f in frames), dtype=np.uint8).reshape(
        len(frames), h, w, 3
    )
    return np.ascontiguousarray(bgr[:, :, :, ::-1])


def _validate_and_bound_frames(
    frames: list[Frame], task: str, *, max_frames: int, label: str
) -> list[Frame]:
    """Shared Robometer pre-flight: input guards + frame budget."""
    if not frames:
        raise ROSConfigError("reward score requires at least one frame")
    if not task.strip():
        raise ROSConfigError("reward score requires a non-empty task instruction")
    if len(frames) > max_frames:
        idx = _evenly_spaced_indices(len(frames), max_frames)
        print(
            f"[{label}] subsampling {len(frames)} -> {len(idx)} frames "
            f"(max_frames={max_frames}) to bound activation memory",
            flush=True,
        )
        frames = [frames[i] for i in idx]
    w, h = frames[0].width, frames[0].height
    if any(f.width != w or f.height != h for f in frames):
        raise ROSConfigError("all frames in a clip must share width/height")
    return frames


class RobometerInProcessReward:
    """In-process Robometer scorer for ``reward_monitor_node``.

    Reuses ``tools/_robometer_scorer.py::_Scorer`` so deploy-sim/run get the
    native lerobot 0.6.0 + NF4 loader without an extra process boundary.
    """

    def __init__(
        self,
        *,
        model_id: str,
        weights_source: str = "OpenRAL/rskill-robometer_4b-any-general-nf4",
        num_bins: int = 100,
        success_threshold: float = 0.5,
        max_frames: int = 8,
        device: str = "cuda",
    ) -> None:
        """Store config; the VLM is loaded lazily on first score."""
        self._model_id = model_id
        self._weights_source = weights_source
        self._num_bins = num_bins
        self._success_threshold = success_threshold
        self._max_frames = max(1, max_frames)
        self._device = device
        self._scorer: _RobometerScorer | None = None

    def _ensure_ready(self) -> None:
        if self._scorer is not None:
            return
        scorer_cls = _load_inprocess_scorer_class()
        self._scorer = scorer_cls(self._weights_source, device=self._device)

    def score(self, frames: list[Frame], task: str) -> tuple[list[float], list[float]]:
        """Score a clip in-process."""
        frames = _validate_and_bound_frames(
            frames, task, max_frames=self._max_frames, label="robometer"
        )
        self._ensure_ready()
        assert self._scorer is not None
        progress, success = self._scorer.score(
            _bgr_frames_to_rgb_array(frames), task.strip(), self._num_bins
        )
        return [float(x) for x in progress], [float(x) for x in success]

    def assess(self, frames: list[Frame], task: str) -> dict[str, Any]:
        """Score ``frames`` and summarize the window for the Reasoner."""
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
        """Release the model."""
        self._scorer = None
        with contextlib.suppress(ImportError):
            import torch  # noqa: PLC0415

            torch.cuda.empty_cache()


def build_reward_monitor(
    manifest: RSkillManifest,
) -> RobometerInProcessReward | TOPRewardMonitor:
    """Build a reward monitor from a ``kind: "reward"`` rSkill manifest.

    Args:
        manifest: A validated rSkill manifest with ``kind == "reward"``.

    Returns:
        TOPReward or Robometer in-process reward monitor.

    Raises:
        ROSConfigError: If the manifest is not ``kind == "reward"`` or lacks a
            ``reward`` block.
    """
    if manifest.kind != "reward":
        raise ROSConfigError(
            f"build_reward_monitor requires kind='reward', got {manifest.kind!r} "
            f"for {manifest.name!r}"
        )
    if manifest.reward is None:  # pragma: no cover — validator guarantees this
        raise ROSConfigError(f"reward manifest {manifest.name!r} has no `reward` block")
    # Backend dispatch: TOPReward and Robometer run in the reward
    # monitor process.
    if manifest.reward.backend == "topreward":
        from openral_runner.backends.reward.topreward_reward import (  # noqa: PLC0415
            build_topreward_monitor,
        )

        return build_topreward_monitor(manifest)
    raw = (
        manifest.weights_uri
        or manifest.source_repo
        or "OpenRAL/rskill-robometer_4b-any-general-nf4"
    )
    # hf://org/repo[@rev] -> "org/repo"; local:///path
    # -> "/path" (a pre-quantized checkpoint dir loaded directly as 4-bit).
    if raw.startswith("local://"):
        weights_source = raw.removeprefix("local://")
    else:
        weights_source = raw.removeprefix("hf://").split("@", 1)[0]
    return RobometerInProcessReward(
        model_id=manifest.name,
        weights_source=weights_source,
        num_bins=manifest.reward.num_bins,
        success_threshold=manifest.reward.success_threshold,
    )
