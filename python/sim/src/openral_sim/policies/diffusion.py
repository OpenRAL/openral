"""Diffusion Policy adapter (Chi et al., 2023).

Wraps :class:`lerobot.policies.diffusion.modeling_diffusion.DiffusionPolicy`.
The PushT checkpoint uses a different obs key naming than the multi-camera
SmolVLA / ACT family:

- Single image stream under ``observation.image`` (NOT
  ``observation.images.<name>`` — PushT predates the multi-cam convention).
- 2-D ``observation.state`` (the agent tip xy).
- 2-D action chunk of length ``n_action_steps=8`` from a 16-step horizon.

The published ``lerobot/diffusion_pusht`` checkpoint predates lerobot's
``PolicyProcessorPipeline`` migration; ``select_action`` accepts raw
float32 [0, 1] inputs directly. Set
``vla.extra.use_lerobot_processors=True`` for any future re-published
checkpoint that ships normaliser stats.

This module imports torch / lerobot lazily.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray
from openral_core.exceptions import ROSConfigError
from openral_rskill._vla_core import (
    call_make_processors_cached_first,
    resolve_device,
    resolve_rskill_repo_revision,
    run_inference,
    to_numpy_action,
)

from openral_sim.policies._processors import resolve_processor_dir
from openral_sim.registry import POLICIES

if TYPE_CHECKING:
    from openral_core import VLASpec

    from openral_sim.rollout import Observation


@dataclass
class _DiffusionAdapter:
    """Diffusion Policy adapter — single ``observation.image`` + state."""

    spec: VLASpec
    device: str
    _policy: Any
    _preprocessor: Any | None
    _postprocessor: Any | None
    _torch: Any
    _image_key: str = "camera1"  # eval-layer Observation.images key to consume
    _last_input_frame: NDArray[np.uint8] | None = None
    # Normalization stats extracted from the checkpoint when the
    # underlying lerobot DiffusionPolicy class no longer carries
    # normalize_inputs / unnormalize_outputs modules. Without these the
    # action returned by select_action stays in the [-1, 1] training
    # space and the env interprets ~0 as "push toward origin", producing
    # wrong-direction motion. Set on construction; applied per step.
    _state_min: Any = None  # tensor (D,) on device
    _state_max: Any = None
    _image_mean: Any = None  # tensor (3, 1, 1) on device
    _image_std: Any = None
    _action_min: Any = None  # tensor (A,) on device
    _action_max: Any = None

    def last_input_frame(self) -> NDArray[np.uint8] | None:
        return self._last_input_frame

    def reset(self) -> None:
        if hasattr(self._policy, "reset"):
            self._policy.reset()

    def step(self, observation: Observation, instruction: str) -> NDArray[np.float32]:
        batch = self._build_batch(observation, instruction)
        if self._preprocessor is not None:
            batch = self._preprocessor(batch)
        elif self._state_min is not None:
            self._normalize_inplace(batch)
        action_tensor = run_inference(self._policy, batch)
        if self._postprocessor is not None:
            action_tensor = self._postprocessor(action_tensor)
        elif self._action_min is not None:
            # Unnormalize from [-1, 1] training space → action units (PushT
            # pixel coords).
            action_tensor = (action_tensor + 1.0) * 0.5 * (
                self._action_max - self._action_min
            ) + self._action_min
        return to_numpy_action(action_tensor)

    def _normalize_inplace(self, batch: dict[str, Any]) -> None:
        torch = self._torch
        img = batch.get("observation.image")
        if img is not None and self._image_mean is not None:
            batch["observation.image"] = (img - self._image_mean) / self._image_std
        state = batch.get("observation.state")
        if state is not None and self._state_min is not None:
            denom = self._state_max - self._state_min
            denom = torch.where(denom == 0, torch.ones_like(denom), denom)
            normed = (state - self._state_min) / denom
            batch["observation.state"] = normed * 2.0 - 1.0

    def close(self) -> None:
        if self.device.startswith("cuda"):
            import contextlib

            with contextlib.suppress(Exception):
                self._torch.cuda.empty_cache()

    def _build_batch(self, observation: Observation, instruction: str) -> dict[str, Any]:
        torch = self._torch
        batch: dict[str, Any] = {"task": [instruction or str(observation.get("task", ""))]}

        images = observation.get("images", {})
        img = images.get(self._image_key)
        if img is not None:
            from openral_sim.policies._video_capture import to_input_frame

            self._last_input_frame = to_input_frame(img)
            t = torch.from_numpy(np.asarray(img)).float().div(255.0).permute(2, 0, 1)
            batch["observation.image"] = t.unsqueeze(0).to(self.device)

        state = observation.get("state")
        if state is not None:
            state_np = np.asarray(state, dtype=np.float32)
            batch["observation.state"] = torch.from_numpy(state_np).unsqueeze(0).to(self.device)

        return batch


@POLICIES.register("diffusion")
def _build_diffusion(env_cfg: Any) -> _DiffusionAdapter:
    """Load a Diffusion Policy checkpoint."""
    spec = env_cfg.vla
    device = resolve_device(spec)

    try:
        import torch
        from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
        from openral_rskill import _lerobot_compat  # noqa: F401
    except ImportError as exc:  # pragma: no cover - opt-in
        raise ROSConfigError(
            "Diffusion Policy adapter requires torch + lerobot; install with: "
            "just sync --all-packages --group sim"
        ) from exc

    repo_id, revision = resolve_rskill_repo_revision(
        spec.weights_uri, adapter_name="Diffusion Policy"
    )
    policy = DiffusionPolicy.from_pretrained(repo_id, revision=revision).to(device)
    policy.eval()

    preprocessor: Any | None = None
    postprocessor: Any | None = None
    if bool(spec.extra.get("use_lerobot_processors", False)):
        from lerobot.policies.factory import make_pre_post_processors

        pretrained_path = resolve_processor_dir(spec, repo_id)
        preprocessor, postprocessor = call_make_processors_cached_first(
            make_pre_post_processors,
            policy.config,
            pretrained_path=pretrained_path,
        )

    image_key = str(spec.extra.get("image_key", "camera1"))

    # Recover normalization stats from the checkpoint's safetensors when
    # the loaded DiffusionPolicy class doesn't expose normalize_inputs /
    # unnormalize_outputs modules. Without this the action stays in the
    # [-1, 1] training space and PushT's eef pushes toward origin.
    stats: dict[str, Any] = {}
    if not preprocessor and not postprocessor:
        stats = _try_load_norm_stats(repo_id, device, torch)

    return _DiffusionAdapter(
        spec=spec,
        device=device,
        _policy=policy,
        _preprocessor=preprocessor,
        _postprocessor=postprocessor,
        _torch=torch,
        _image_key=image_key,
        _state_min=stats.get("state_min"),
        _state_max=stats.get("state_max"),
        _image_mean=stats.get("image_mean"),
        _image_std=stats.get("image_std"),
        _action_min=stats.get("action_min"),
        _action_max=stats.get("action_max"),
    )


def _try_load_norm_stats(repo_id: str, device: str, torch: Any) -> dict[str, Any]:
    """Pull normalize_* / unnormalize_* tensors from the checkpoint's safetensors.

    Returns an empty dict if no usable stats are found — adapters then
    fall back to passing raw inputs and trust the policy's own
    normalisation.
    """
    try:
        from huggingface_hub import snapshot_download
        from safetensors import safe_open
    except ImportError:
        return {}

    try:
        local = snapshot_download(repo_id=repo_id, ignore_patterns=["*.md"])
    except Exception:
        return {}

    import os

    weights = os.path.join(local, "model.safetensors")
    if not os.path.exists(weights):
        return {}

    out: dict[str, Any] = {}
    key_map = {
        "state_min": "normalize_inputs.buffer_observation_state.min",
        "state_max": "normalize_inputs.buffer_observation_state.max",
        "image_mean": "normalize_inputs.buffer_observation_image.mean",
        "image_std": "normalize_inputs.buffer_observation_image.std",
        "action_min": "unnormalize_outputs.buffer_action.min",
        "action_max": "unnormalize_outputs.buffer_action.max",
    }
    try:
        with safe_open(weights, framework="pt") as f:  # type: ignore[no-untyped-call]  # reason: safetensors lacks py.typed
            available = set(f.keys())
            for short, long in key_map.items():
                if long in available:
                    t = f.get_tensor(long).to(device=device, dtype=torch.float32)
                    # State / action stats are 1-D; broadcast over batch dim.
                    if t.ndim == 1:
                        t = t.unsqueeze(0)
                    out[short] = t
    except Exception:
        return {}
    return out
