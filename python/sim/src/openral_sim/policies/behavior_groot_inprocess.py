"""BEHAVIOR-1K GR00T N1.7 checkpoint run in-process on lerobot's ``groot`` policy.

The opt-in alternative to the Python 3.10 Isaac-GR00T sidecar
(``openral_sim.policies.behavior_groot``), selected with
``vla.extra.runtime: inprocess`` or ``OPENRAL_BEHAVIOR_GROOT_RUNTIME=inprocess``.

lerobot 0.7 reads the checkpoint's own ``processor_config.json`` /
``statistics.json`` / ``embodiment_id.json``, so normalization, the per-group
relative/absolute action decode (torso and arms relative to their state via
``state_key``) and the embodiment id need no OpenRAL code. What the organizers'
``gr00t.eval.eval_b1k_wrapper.B1KPolicyWrapper`` does around the model is ported
here, to the line (fork ``wensi-ai/Isaac-GR00T@ace36d9``):

- the 61-D R1Pro proprio is sliced into the six state groups, concatenated in
  the checkpoint's modality order (25-D);
- each camera drops its alpha channel and is letterboxed to 224x224
  (``cv2.INTER_LINEAR``, centred zero padding);
- the 16-step chunks are temporally ensembled (``temporal_ensemble`` mode).

The output is the same 23-D wire action the sidecar returns, so the scene side
is unchanged.

Precision follows the manifest like every family (``int4`` → NF4, OpenRAL Pro);
``OPENRAL_QUANTIZATION_DTYPE=fp32`` (or ``none``) loads the open fp32 path.
bf16 is refused, as for the ``gr00t`` family: an unquantized ``GrootPolicy``
load casts every param to fp32.
"""

from __future__ import annotations

import os
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import structlog
from numpy.typing import NDArray
from openral_core.exceptions import ROSConfigError, ROSRuntimeError
from openral_observability import inference_span

from openral_sim import _behavior_wire

if TYPE_CHECKING:
    from openral_core import RSkillManifest, VLASpec

    from openral_sim.policies.behavior_groot import _EnvCfg
    from openral_sim.rollout import Observation

_log = structlog.get_logger(__name__)

RUNTIME_ENV = "OPENRAL_BEHAVIOR_GROOT_RUNTIME"

_EMBODIMENT_TAG = "new_embodiment"
_OBS_SIZE = (224, 224)
# B1KPolicyWrapper's temporal-ensemble constants.
_ENSEMBLE_MAX = 5
_ENSEMBLE_K = 0.005
_CHUNK_NDIM = 2  # (T, D)

# Pro's ``quantize_groot_nf4`` packs Linears of at least this many params.
_PRO_NF4_MIN_PARAMS = 4_000_000

STATE_DIM = sum(end - start for _, start, end in _behavior_wire.R1PRO_STATE_GROUPS)


def r1pro_state(proprio: NDArray[np.floating[Any]]) -> NDArray[np.float32]:
    """Slice the 61-D R1Pro proprio into the checkpoint's 25-D state layout."""
    flat = np.asarray(proprio, dtype=np.float32).reshape(-1)
    if flat.shape != (_behavior_wire.STATE_DIM,):
        raise ROSRuntimeError(
            f"BEHAVIOR GR00T proprio must have {_behavior_wire.STATE_DIM} values, "
            f"got {flat.shape[0]}."
        )
    groups = _behavior_wire.R1PRO_STATE_GROUPS
    return np.concatenate([flat[start:end] for _, start, end in groups])


def resize_with_pad(image: NDArray[np.uint8], height: int, width: int) -> NDArray[np.uint8]:
    """Aspect-preserving resize + centred zero pad (``eval_b1k_wrapper.resize_with_pad``)."""
    import cv2

    h, w = image.shape[:2]
    scale = min(height / h, width / w)
    new_h, new_w = int(h * scale), int(w * scale)
    out = np.zeros((height, width, image.shape[2]), dtype=image.dtype)
    top, left = (height - new_h) // 2, (width - new_w) // 2
    out[top : top + new_h, left : left + new_w] = cv2.resize(
        image, (new_w, new_h), interpolation=cv2.INTER_LINEAR
    )
    return out


@dataclass
class TemporalEnsemble:
    """``B1KPolicyWrapper.act_temporal_ensemble`` for one environment.

    Every step adds the newest chunk; the last ``_ENSEMBLE_MAX`` chunks are each
    read at their age and averaged with weights ``exp(k * slot)`` (oldest slot
    0). Gripper dims come from the newest chunk's first step.
    """

    _chunks: deque[NDArray[np.float32]] = field(default_factory=lambda: deque(maxlen=_ENSEMBLE_MAX))

    def reset(self) -> None:
        """Drop every buffered chunk (episode boundary)."""
        self._chunks.clear()

    def step(self, chunk: NDArray[np.float32]) -> NDArray[np.float32]:
        """Add the newest ``(T, 23)`` chunk and return this step's 23-D action."""
        chunk = np.asarray(chunk, dtype=np.float32)
        if chunk.ndim != _CHUNK_NDIM or chunk.shape[0] < _ENSEMBLE_MAX:
            raise ROSRuntimeError(
                f"BEHAVIOR GR00T temporal ensemble needs (T>={_ENSEMBLE_MAX}, D) chunks, "
                f"got {chunk.shape}."
            )
        self._chunks.append(chunk)
        n = len(self._chunks)
        # Slot i (oldest first) was added n-1-i steps ago, so it is read there.
        current = np.stack([c[n - 1 - i] for i, c in enumerate(self._chunks)])
        weights = np.exp(_ENSEMBLE_K * np.arange(n))
        action = (current * (weights / weights.sum())[:, None]).sum(axis=0)
        newest = self._chunks[-1][0]
        for idx in _behavior_wire.GRIPPER_ACTION_INDICES:
            action[idx] = newest[idx]
        out: NDArray[np.float32] = action.astype(np.float32)
        return out


def wants_inprocess(extra: dict[str, object]) -> bool:
    """Whether the in-process runtime was selected (env var wins over ``extra``)."""
    runtime = os.environ.get(RUNTIME_ENV) or str(extra.get("runtime", "sidecar"))
    if runtime not in {"sidecar", "inprocess"}:
        raise ROSConfigError(
            f"BEHAVIOR GR00T runtime must be 'sidecar' or 'inprocess', got {runtime!r}."
        )
    return runtime == "inprocess"


@dataclass
class _BehaviorGrootInprocessAdapter:
    """PolicyAdapter running the B1K checkpoint on lerobot ``GrootPolicy``."""

    spec: VLASpec
    device: str
    # reason: lerobot / torch objects are imported lazily inside the builder.
    _policy: Any
    _preprocessor: Any
    _postprocessor: Any
    _torch: Any
    _ensemble: TemporalEnsemble = field(default_factory=TemporalEnsemble)
    _last_input: NDArray[np.uint8] | None = None

    def reset(self) -> None:
        self._ensemble.reset()
        if self._policy is not None:  # None after close()
            self._policy.reset()

    def step(self, observation: Observation, instruction: str) -> NDArray[np.float32]:
        from openral_rskill._vla_core import image_to_chw_tensor

        from openral_sim.policies.behavior_groot import _behavior_wire_observation

        wire = _behavior_wire_observation(observation, instruction=instruction)
        missing = [
            key
            for key in (*_behavior_wire.CAMERA_RGB_KEYS.values(), _behavior_wire.STATE_KEY)
            if key not in wire
        ]
        if missing:
            raise ROSRuntimeError(f"BEHAVIOR GR00T observation is missing {missing}.")
        torch = self._torch
        batch: dict[str, Any] = {"task": instruction}
        for role, key in _behavior_wire.CAMERA_RGB_KEYS.items():
            rgb = np.asarray(wire[key])[..., :3].astype(np.uint8)
            img = resize_with_pad(rgb, *_OBS_SIZE)
            if role == "head":
                self._last_input = img
            batch[f"observation.images.{role}"] = image_to_chw_tensor(
                img, torch=torch, device=self.device
            )
        state = r1pro_state(np.asarray(wire[_behavior_wire.STATE_KEY]))
        batch["observation.state"] = torch.from_numpy(state).to(self.device)

        with inference_span(kind="single", engine="pytorch"), torch.inference_mode():
            raw = self._policy.predict_action_chunk(self._preprocessor(batch))
            chunk = self._postprocessor(raw)
        chunk_np = chunk.detach().to("cpu").float().numpy()
        if chunk_np.ndim == _CHUNK_NDIM + 1 and chunk_np.shape[0] == 1:  # batch of one
            chunk_np = chunk_np[0]
        if chunk_np.ndim != _CHUNK_NDIM or chunk_np.shape[1] != _behavior_wire.ACTION_DIM:
            raise ROSRuntimeError(
                f"BEHAVIOR GR00T in-process policy returned a {chunk_np.shape} chunk; "
                f"expected (T, {_behavior_wire.ACTION_DIM})."
            )
        action = self._ensemble.step(chunk_np)
        if not np.isfinite(action).all():
            raise ROSRuntimeError("BEHAVIOR GR00T in-process policy produced a non-finite action.")
        return action

    def last_input_frame(self) -> NDArray[np.uint8] | None:
        return None if self._last_input is None else self._last_input.copy()

    def close(self) -> None:
        from openral_rskill._vla_core import release_torch_modules

        release_torch_modules(
            self,
            "_policy",
            "_preprocessor",
            "_postprocessor",
            device=self.device,
            torch=self._torch,
        )


def build_behavior_groot_inprocess_policy(
    env_cfg: _EnvCfg,
    manifest: RSkillManifest,
    extra: dict[str, object],
    checkpoint: Path,
) -> _BehaviorGrootInprocessAdapter:
    """Load the B1K checkpoint on lerobot's ``GrootPolicy`` (N1.7)."""
    from openral_rskill._vla_core import resolve_device
    from openral_rskill.loader import rSkill
    from openral_rskill.quantization_gate import pro_quantization

    from openral_sim._quantization import require_supported_dtype, resolve_quant_plan
    from openral_sim.policies._policy_loading import lazy_import_lerobot
    from openral_sim.policies.gr00t import (
        _build_groot_config,
        _patch_groot_dtype_property,
        _redirect_groot_backbone_processor,
    )

    control_mode = str(extra.get("control_mode", "temporal_ensemble"))
    if control_mode != "temporal_ensemble":
        raise ROSConfigError(
            f"The in-process BEHAVIOR GR00T runtime implements only temporal_ensemble, "
            f"not {control_mode!r}; use runtime: sidecar for it."
        )
    if not checkpoint.is_dir():
        raise ROSConfigError(f"BEHAVIOR GR00T checkpoint not found at {checkpoint}.")
    rSkill._check_provenance(manifest, source=str(checkpoint))

    spec = env_cfg.vla
    device = resolve_device(spec)
    plan = resolve_quant_plan(spec, manifest, default="nf4")
    require_supported_dtype(plan, frozenset({"nf4", "fp32", "none"}), "gr00t_b1k (inprocess)")
    quantize_scope = str(plan.extra.get("quantize_scope", "model")).lower()
    requested_min = plan.extra.get("nf4_min_params")
    if plan.quantize and requested_min is not None and requested_min != _PRO_NF4_MIN_PARAMS:
        # ponytail: Pro's quantize_groot_nf4 has no min-params knob; add one if
        # the in-process NF4 run needs to match the sidecar's packing.
        _log.warning(
            "behavior_groot.inprocess.nf4_min_params_ignored",
            requested=requested_min,
            used=_PRO_NF4_MIN_PARAMS,
        )

    install_hint = "just sync --all-packages --group gr00t"
    torch, make_processors = lazy_import_lerobot("GR00T", install_hint=install_hint)
    try:
        from lerobot.policies.groot.modeling_groot import GrootPolicy
    except ImportError as exc:
        raise ROSConfigError(
            f"The in-process BEHAVIOR GR00T runtime needs lerobot's groot policy: {install_hint}"
        ) from exc

    _patch_groot_dtype_property(torch)
    _redirect_groot_backbone_processor()
    config = _build_groot_config(
        local_path=str(checkpoint),
        embodiment_tag=_EMBODIMENT_TAG,
        quantize=plan.quantize,
        image_keys=tuple(f"observation.images.{role}" for role in _behavior_wire.CAMERA_RGB_KEYS),
        state_dim=STATE_DIM,
        action_dim=_behavior_wire.ACTION_DIM,
    )
    prev_default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    try:
        policy = GrootPolicy.from_pretrained(str(checkpoint), config=config)
        if plan.quantize:
            pro_quantization(what="gr00t_b1k").quantize_groot_nf4(
                policy, torch, scope=quantize_scope
            )
        policy = policy.to(device)
    finally:
        torch.set_default_dtype(prev_default_dtype)
    policy.eval()
    preprocessor, postprocessor = make_processors(policy.config, pretrained_path=str(checkpoint))
    _log.info(
        "behavior_groot.inprocess.loaded",
        checkpoint=str(checkpoint),
        quantization=plan.dtype or "none",
        device=device,
    )
    return _BehaviorGrootInprocessAdapter(
        spec=spec,
        device=device,
        _policy=policy,
        _preprocessor=preprocessor,
        _postprocessor=postprocessor,
        _torch=torch,
    )


__all__ = [
    "RUNTIME_ENV",
    "STATE_DIM",
    "TemporalEnsemble",
    "build_behavior_groot_inprocess_policy",
    "r1pro_state",
    "resize_with_pad",
    "wants_inprocess",
]
