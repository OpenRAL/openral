r"""OpenVLA / OpenVLA-OFT policy adapter.

Wraps the `OpenVLA <https://openvla.github.io>`_ family — a Prismatic VLM
(DINOv2 + SigLIP fused vision backbone + Llama-2 7B) with a discrete
action head — and its OFT fine-tuning recipe (arXiv:2502.19645). The first
in-tree checkpoint is ``RLinf/RLinf-OpenVLAOFT-PPO-ManiSkill3-25ood``: an
OpenVLA-OFT bridge policy RL-tuned (PPO) on ManiSkill3 ``PutOnPlateInScene25``,
run here on the SimplerEnv WidowX put-on-plate tasks it actually solves
(``unnorm_key=bridge_orig``) — see ADR-0061 for the WidowX-vs-Panda rationale.

Like MolmoAct2 (and unlike the lerobot adapters), OpenVLA is **not** a lerobot
policy. It ships as a transformers *custom-code* model (``trust_remote_code``,
``auto_map`` → ``OpenVLAForActionPrediction``) and is driven through its own
``predict_action`` API rather than lerobot's ``select_action`` queue:

- The eval-layer :class:`~openral_sim.rollout.Observation` (a flat ``state`` +
  ``images`` dict) is turned into a single 224×224 RGB + the prompt
  ``In: What action should the robot take to {instruction.lower()}?\nOut: `` and
  passed to ``predict_action(**inputs, unnorm_key=...)``. The RLinf checkpoint
  uses no proprio (``use_proprio=False``). RLinf-family checkpoints may instead
  request ``generate_action_verl`` via ``policy_extras`` to mirror their PPO eval
  path (right-padded prompt, temperature sampling).
- ``predict_action`` decodes the 256-bin discrete action tokens and
  de-normalizes them with the checkpoint's embedded ``unnorm_key`` stats
  (``bridge_orig``: 6 EE deltas rescaled, gripper passed through). For
  checkpoints whose custom code returns *normalized* tokens instead, set
  ``vla.extra['openvla_actions_prenormalized']=True`` and the adapter applies
  :func:`_unnormalize_action` itself.
- The returned chunk (OFT: 8 × 7-D; base OpenVLA: a single 7-D action) is
  replayed one step at a time and re-inferred when the queue empties — the same
  closed-loop replay MolmoAct2 / the lerobot adapters get.

NF4 quantization reuses the adapter-agnostic helpers in
:mod:`openral_sim._quantization`; the 7.5 B bf16 backbone is ~16 GB and OOMs an
8 GB consumer GPU, so int4 (~7 GB, matching bf16 accuracy per the OpenVLA paper
Table 2) brings it into reach. The CUDA expandable-segments allocator
(:func:`_enable_expandable_segments`) keeps the inference peak placeable on a
tight 8 GB card, mirroring the MolmoAct2 recipe.

This module imports torch / transformers lazily so installing ``openral-sim``
never pulls them transitively.
"""

from __future__ import annotations

import contextlib
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import structlog
from numpy.typing import NDArray
from openral_core.exceptions import ROSConfigError
from openral_observability import inference_span
from openral_rskill._diagnostics import phase_timer
from openral_rskill._vla_core import (
    resolve_camera_keys,
    resolve_device,
    resolve_rskill_repo_id,
)

from openral_sim._quantization import (
    default_dtype_for_device,
    manifest_dtype,
)
from openral_sim.policies._policy_loading import load_manifest_for_spec
from openral_sim.registry import POLICIES

if TYPE_CHECKING:
    from openral_core import VLASpec

    from openral_sim.rollout import Observation

_log = structlog.get_logger(__name__)

# OpenVLA prompt template (openvla README / modeling code). The instruction is
# lowercased; the model was trained on this exact wrapper.
_PROMPT_TEMPLATE = "In: What action should the robot take to {instr}?\nOut: "

# RLinf OpenVLA-OFT default de-normalization key (config.json ``norm_stats``).
_DEFAULT_UNNORM_KEY = "bridge_orig"

# predict_action emits either a single (action_dim,) action (base OpenVLA) or a
# (chunk, action_dim) chunk (OFT). 7-D bridge action: 3 EE pos Δ + 3 rot Δ + 1
# gripper.
_DEFAULT_ACTION_DIM = 7
_DEFAULT_GENERATION_METHOD = "predict_action"
_GENERATE_ACTION_VERL = "generate_action_verl"
_ACTION_CHUNK_NDIM = 2

_CUDA_ALLOC_ENV = "PYTORCH_CUDA_ALLOC_CONF"
_EXPANDABLE_SEGMENTS = "expandable_segments:True"
_ALLOW_REMOTE_CODE_ENV = "OPENRAL_ALLOW_REMOTE_CODE"


# ── Pure helpers (unit-tested in tests/unit/sim/test_openvla_adapter.py) ─────


def _decode_prompt(instruction: str) -> str:
    """Wrap a task instruction in the OpenVLA prompt template (lowercased)."""
    return _PROMPT_TEMPLATE.format(instr=instruction.strip().lower())


def _unnormalize_action(norm: NDArray[np.float32], stats: dict[str, Any]) -> NDArray[np.float32]:
    """Un-normalize an OpenVLA action in ``[-1, 1]`` with embedded q01/q99 stats.

    Implements OpenVLA's ``BOUNDS_Q99`` de-normalization, per masked dim::

        action = 0.5 * (norm + 1) * (q99 - q01) + q01

    Dims whose ``mask`` is ``False`` (the gripper) are passed through unchanged.
    Only needed when a checkpoint's custom ``predict_action`` returns *normalized*
    tokens; the stock OpenVLA code already un-normalizes internally.

    Args:
        norm: Normalized action in ``[-1, 1]`` of length ``action_dim``.
        stats: The checkpoint's ``norm_stats[unnorm_key]`` dict with an
            ``"action"`` entry carrying ``q01`` / ``q99`` / ``mask`` lists.

    Returns:
        1-D ``float32`` un-normalized action of length ``action_dim``.
    """
    action_stats = stats["action"]
    q01 = np.asarray(action_stats["q01"], dtype=np.float32)
    q99 = np.asarray(action_stats["q99"], dtype=np.float32)
    mask = np.asarray(action_stats["mask"], dtype=bool)
    norm = np.asarray(norm, dtype=np.float32)
    scaled = 0.5 * (norm + 1.0) * (q99 - q01) + q01
    return np.where(mask, scaled, norm).astype(np.float32)


def _as_action_chunk(arr: Any, action_dim: int) -> NDArray[np.float32]:
    """Normalize an OpenVLA action return value to ``(chunk, action_dim)``."""
    if action_dim <= 0:
        raise ROSConfigError(f"openvla_action_dim must be > 0, got {action_dim!r}.")
    out = np.asarray(arr, dtype=np.float32)
    if out.ndim == 0:
        raise ROSConfigError("OpenVLA returned a scalar action; expected a vector or chunk.")
    if out.ndim == 1 and out.shape[0] > action_dim and out.shape[0] % action_dim == 0:
        out = out.reshape(-1, action_dim)
    elif out.ndim == 1:
        out = out[None, :]
    elif out.ndim > _ACTION_CHUNK_NDIM:
        out = out.reshape(-1, out.shape[-1])
    if out.shape[-1] > action_dim:
        out = out[:, :action_dim]
    if out.shape[-1] != action_dim:
        raise ROSConfigError(
            f"OpenVLA returned action chunk with last dimension {out.shape[-1]}, "
            f"expected openvla_action_dim={action_dim}."
        )
    return np.ascontiguousarray(out, dtype=np.float32)


def _postprocess_action_chunk(
    arr: NDArray[np.float32],
    *,
    action_scale: float,
    binarize_gripper: bool,
    gripper_threshold: float,
) -> NDArray[np.float32]:
    """Apply explicit env-side action transforms from ``policy_extras``."""
    out = np.asarray(arr, dtype=np.float32).copy()
    if action_scale != 1.0:
        out[:, : min(6, out.shape[1])] *= np.float32(action_scale)
    if binarize_gripper and out.shape[1] >= _DEFAULT_ACTION_DIM:
        out[:, 6] = np.where(out[:, 6] > gripper_threshold, 1.0, -1.0).astype(np.float32)
    return np.ascontiguousarray(out, dtype=np.float32)


def _extra_bool(extra: dict[str, object], key: str, default: bool) -> bool:
    value = extra.get(key, default)
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    if isinstance(value, int) and value in {0, 1}:
        return bool(value)
    raise ROSConfigError(f"OpenVLA policy_extras.{key} must be a bool, got {value!r}.")


def _extra_float(extra: dict[str, object], key: str, default: float) -> float:
    value = extra.get(key, default)
    try:
        out = float(str(value))
    except (TypeError, ValueError) as exc:
        raise ROSConfigError(
            f"OpenVLA policy_extras.{key} must be a finite float, got {value!r}."
        ) from exc
    if not np.isfinite(out):
        raise ROSConfigError(f"OpenVLA policy_extras.{key} must be a finite float, got {value!r}.")
    return out


def _extra_int_or_none(extra: dict[str, object], key: str) -> int | None:
    value = extra.get(key)
    if value is None:
        return None
    try:
        out = int(str(value))
    except (TypeError, ValueError) as exc:
        raise ROSConfigError(
            f"OpenVLA policy_extras.{key} must be an integer, got {value!r}."
        ) from exc
    if out < 0:
        raise ROSConfigError(f"OpenVLA policy_extras.{key} must be >= 0, got {out!r}.")
    return out


def _extra_positive_int_or_none(extra: dict[str, object], key: str) -> int | None:
    out = _extra_int_or_none(extra, key)
    if out == 0:
        raise ROSConfigError(f"OpenVLA policy_extras.{key} must be > 0, got 0.")
    return out


# ── Load-phase helpers (per-adapter, mirroring pi05 / molmoact2 convention) ──


def _enable_expandable_segments() -> None:
    """Enable the CUDA expandable-segments allocator before the OpenVLA load.

    Sets ``PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`` via
    :meth:`os.environ.setdefault` semantics (an operator export wins) **before
    the first CUDA allocation** in this process. The caching allocator reads the
    variable lazily on its first allocation, so setting it at the top of the
    build — ahead of the model's device placement — takes effect even though
    torch is already imported. No-op when the variable is already set. Mirrors
    the MolmoAct2 recipe that lets a 7B NF4 backbone fit an 8 GB card.
    """
    current = os.environ.get(_CUDA_ALLOC_ENV)
    if current is not None:
        if "expandable_segments" not in current:
            _log.info(
                "openvla_alloc_conf_preset",
                value=current,
                note="PYTORCH_CUDA_ALLOC_CONF already set; not adding expandable_segments.",
            )
        return
    os.environ[_CUDA_ALLOC_ENV] = _EXPANDABLE_SEGMENTS
    _log.info("openvla_expandable_segments_enabled", value=_EXPANDABLE_SEGMENTS)


def _openvla_phase(name: str, **fields: Any) -> Any:
    """``phase_timer`` shortcut for OpenVLA load phases (GPU footprint on)."""
    return phase_timer(name, prefix="openvla", gpu_mb=True, log=_log, **fields)


def _import_transformers() -> tuple[Any, Any, Any]:
    """Import the OpenVLA auto-model + ``AutoProcessor`` + ``BitsAndBytesConfig``.

    OpenVLA is a transformers custom-code model (``auto_map`` →
    ``OpenVLAForActionPrediction``), loaded with ``trust_remote_code=True``. It is
    NOT a lerobot policy. ``BitsAndBytesConfig`` is returned so the 4-bit NF4
    config is built at ``from_pretrained`` time (a 4-bit model cannot be
    ``.to()``-moved afterwards).

    OpenVLA's ``auto_map`` keys the model on ``AutoModelForVision2Seq``, which
    transformers <5 exposes directly; transformers 5.x renamed it to
    ``AutoModelForImageTextToText``. Prefer the legacy class when present (so the
    auto_map dispatch matches) and fall back to the new name otherwise.

    Returns ``(auto_model_cls, AutoProcessor, BitsAndBytesConfig)``, untyped
    (transformers has no strict stubs in this workspace).

    Raises:
        ROSConfigError: If transformers / torch are not installed.
    """
    try:
        import transformers
        from transformers import AutoProcessor, BitsAndBytesConfig
    except ImportError as exc:  # pragma: no cover - opt-in dependency
        raise ROSConfigError(
            "OpenVLA adapter requires transformers (custom-code model loaded via "
            "AutoModelForVision2Seq + trust_remote_code). Install with: "
            f"just sync --all-packages --group simpler-env (underlying: {exc!r})"
        ) from exc
    auto_model_cls: Any = getattr(transformers, "AutoModelForVision2Seq", None)
    if auto_model_cls is None:
        auto_model_cls = transformers.AutoModelForImageTextToText
    return auto_model_cls, AutoProcessor, BitsAndBytesConfig


def _strip_hf_uri(uri: str | None, *, field_name: str) -> tuple[str, str | None]:
    """Strip ``hf://`` and split a trailing ``@revision``.

    Args:
        uri: A manifest URI like ``hf://owner/repo`` or ``hf://owner/repo@sha``.
        field_name: The manifest field name, for the error message.

    Returns:
        ``(repo_id, revision_or_None)``.

    Raises:
        ROSConfigError: If the URI is missing / not an ``hf://`` repo.
    """
    value = (uri or "").strip()
    if not value.startswith("hf://"):
        raise ROSConfigError(
            f"OpenVLA adapter needs the rSkill manifest's {field_name} to be an "
            f"hf:// repo (e.g. 'hf://RLinf/RLinf-OpenVLAOFT-PPO-ManiSkill3-25ood'), "
            f"got {value!r}."
        )
    repo_id, _, revision = value[len("hf://") :].partition("@")
    return repo_id, (revision or None)


def _require_remote_code_ack(source_repo: str, revision: str | None) -> None:
    """Refuse to load a ``trust_remote_code`` model unless the operator opts in.

    OpenVLA's ``from_pretrained`` executes ``modeling_prismatic.py`` shipped in
    the repo. The repo id is manifest/operator-supplied and rSkill signature
    verification is not yet implemented (ADR-0006), so this is a remote-code
    execution sink. Require ``OPENRAL_ALLOW_REMOTE_CODE=1``, mirroring the
    MolmoAct2 gate and ``OPENRAL_ALLOW_UNSAFE_PICKLE``.

    Raises:
        ROSConfigError: If the acknowledgement env var is not set to ``"1"``.
    """
    if os.environ.get(_ALLOW_REMOTE_CODE_ENV, "0") != "1":
        raise ROSConfigError(
            f"OpenVLA loads custom code from '{source_repo}' via "
            "trust_remote_code=True, which executes arbitrary Python from the repo "
            "(remote-code-execution risk for untrusted or unverified weights). rSkill "
            "signature verification is not yet implemented (ADR-0006), so this is "
            f"blocked by default. To load a TRUSTED repo, set: export {_ALLOW_REMOTE_CODE_ENV}=1 "
            "(pin a revision SHA in the manifest's weights_uri for reproducibility)."
        )
    if revision is None:
        _log.warning(
            "openvla.remote_code_unpinned",
            repo=source_repo,
            env=_ALLOW_REMOTE_CODE_ENV,
            note="Executing custom code from an UNPINNED repo; pin @<sha> in weights_uri.",
        )


def _resolve_unnorm_key(spec: VLASpec, model: Any) -> str:
    """Resolve the de-normalization key: vla.extra → model.config → default.

    OpenVLA checkpoints embed multiple Open-X norm-stat tables in ``config.json``;
    ``unnorm_key`` selects the right one. The RLinf checkpoint ships
    ``norm_stats`` keyed ``bridge_orig``.
    """
    extra = getattr(spec, "extra", {}) or {}
    if extra.get("openvla_unnorm_key"):
        return str(extra["openvla_unnorm_key"])
    cfg_key = getattr(getattr(model, "config", None), "unnorm_key", None)
    if isinstance(cfg_key, str) and cfg_key:
        return cfg_key
    norm_stats = getattr(getattr(model, "config", None), "norm_stats", None)
    if isinstance(norm_stats, dict) and len(norm_stats) == 1:
        return str(next(iter(norm_stats)))
    return _DEFAULT_UNNORM_KEY


# ── Adapter ──────────────────────────────────────────────────────────────────


@dataclass
class _OpenVLAAdapter:
    """OpenVLA / OpenVLA-OFT adapter — drives ``predict_action`` with chunk replay."""

    spec: VLASpec
    device: str
    _model: Any
    _processor: Any
    _torch: Any
    _unnorm_key: str = _DEFAULT_UNNORM_KEY
    _action_dim: int = _DEFAULT_ACTION_DIM
    _camera_keys: tuple[str, ...] = field(default_factory=lambda: ("camera1",))
    # Stock OpenVLA un-normalizes inside predict_action; set True for forks that
    # return normalized tokens (then the adapter applies _unnormalize_action).
    _actions_prenormalized: bool = False
    _autocast_dtype: Any = None
    _generation_method: str = _DEFAULT_GENERATION_METHOD
    _do_sample: bool = False
    _temperature: float = 0.6
    _padding_max_length: int | None = None
    _action_scale: float = 1.0
    _binarize_gripper: bool = False
    _gripper_threshold: float = 0.5
    _torch_seed: int | None = None
    _last_input_frame: NDArray[np.uint8] | None = None
    _action_queue: list[NDArray[np.float32]] = field(default_factory=list)

    def last_input_frame(self) -> NDArray[np.uint8] | None:
        return self._last_input_frame

    def reset(self) -> None:
        self._action_queue = []
        _seed_torch_for_sampling(self._torch, self._torch_seed)

    def step(self, observation: Observation, instruction: str) -> NDArray[np.float32]:
        if not self._action_queue:
            self._action_queue = self._predict_chunk(observation, instruction)
        return self._action_queue.pop(0)

    def close(self) -> None:
        if self.device.startswith("cuda"):
            with contextlib.suppress(Exception):
                self._torch.cuda.empty_cache()

    def _collect_image(self, observation: Observation) -> Any:
        """Return the single RGB frame OpenVLA consumes, as a PIL Image."""
        from PIL import Image

        raw = observation.get("images", {})
        for cam_key in self._camera_keys:
            img = raw.get(cam_key)
            if img is not None:
                frame = np.ascontiguousarray(np.asarray(img), dtype=np.uint8)
                self._last_input_frame = frame
                return Image.fromarray(frame)
        raise ROSConfigError(
            "OpenVLA adapter got no camera frames; expected an observation image "
            f"for one of {self._camera_keys!r}, saw {list(raw)!r}."
        )

    def _predict_chunk(
        self, observation: Observation, instruction: str
    ) -> list[NDArray[np.float32]]:
        """Run ``predict_action`` once and return the replayable action chunk."""
        torch = self._torch
        image = self._collect_image(observation)
        task = instruction or observation.get("task", "")
        prompt = _decode_prompt(task)

        if self._generation_method == _GENERATE_ACTION_VERL:
            proc_kwargs: dict[str, Any] = {"text": [prompt], "images": [image]}
            if self._padding_max_length is not None:
                proc_kwargs.update(
                    {"padding": "max_length", "max_length": self._padding_max_length}
                )
            inputs = self._processor(**proc_kwargs)
        else:
            inputs = self._processor(prompt, image)
        inputs = inputs.to(self.device, dtype=self._autocast_dtype or torch.bfloat16)

        device_type = self.device.split(":", 1)[0]
        if self._autocast_dtype is not None and device_type in {"cuda", "cpu"}:
            autocast_ctx: Any = torch.amp.autocast(
                device_type=device_type, dtype=self._autocast_dtype
            )
        else:
            autocast_ctx = contextlib.nullcontext()

        with inference_span(kind="chunk"), torch.no_grad(), autocast_ctx:
            if self._generation_method == _GENERATE_ACTION_VERL:
                if not hasattr(self._model, _GENERATE_ACTION_VERL):
                    raise ROSConfigError(
                        "OpenVLA policy_extras.openvla_generation_method="
                        f"{_GENERATE_ACTION_VERL!r} was requested, but the loaded model "
                        "does not expose generate_action_verl()."
                    )
                padding_idx = getattr(
                    getattr(self._processor, "tokenizer", None), "pad_token_id", None
                )
                if padding_idx is None:
                    padding_idx = getattr(
                        getattr(self._processor, "tokenizer", None), "eos_token_id", 2
                    )
                out = self._model.generate_action_verl(
                    **inputs,
                    unnorm_key=self._unnorm_key,
                    do_sample=self._do_sample,
                    temperature=self._temperature,
                    padding_idx=padding_idx,
                )
            else:
                out = self._model.predict_action(
                    **inputs, unnorm_key=self._unnorm_key, do_sample=self._do_sample
                )

        # OpenVLA-OFT ``predict_action`` returns a ``(actions, hidden_states)``
        # tuple; the action half is the (already un-normalized) chunk, flattened
        # to ``(num_chunk * action_dim,)`` (verified live 2026-06-19: 8*7=56).
        act = out[0] if isinstance(out, (tuple, list)) else out
        if hasattr(act, "detach"):
            act = act.detach().to(torch.float32).cpu().numpy()
        arr = _as_action_chunk(act, self._action_dim)
        # Fallback de-norm only for checkpoints whose custom code returns
        # *normalized* tokens (the stock OpenVLA path already un-normalizes).
        if self._actions_prenormalized:
            stats = self._model.config.norm_stats[self._unnorm_key]
            arr = np.stack([_unnormalize_action(row, stats) for row in arr])
        arr = _postprocess_action_chunk(
            arr,
            action_scale=self._action_scale,
            binarize_gripper=self._binarize_gripper,
            gripper_threshold=self._gripper_threshold,
        )
        return [np.ascontiguousarray(row, dtype=np.float32) for row in arr]


def _load_openvla_model(
    *,
    torch: Any,
    auto_model_cls: Any,
    auto_processor_cls: Any,
    bnb_config_cls: Any,
    repo_id: str,
    revision: str | None,
    device: str,
    dtype_str: str | None,
) -> tuple[Any, Any]:
    """Load the OpenVLA model + processor, NF4-quantized when requested."""
    use_nf4 = (dtype_str or "").lower() in {"nf4", "int4"} and device.startswith("cuda")
    common = {"trust_remote_code": True, "revision": revision}

    with _openvla_phase("processor"):
        processor = auto_processor_cls.from_pretrained(repo_id, **common)

    load_kwargs: dict[str, Any] = {
        **common,
        "torch_dtype": torch.bfloat16,
        "low_cpu_mem_usage": True,
    }
    if use_nf4:
        load_kwargs["quantization_config"] = bnb_config_cls(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        # A 4-bit model is placed at load time; it cannot be .to()-moved after.
        # Use the integer CUDA index ({"": 0}) — accelerate's single-device
        # dispatch path is sensitive to the device_map form for quantized
        # custom-code models (verified live 2026-06-19 with accelerate 0.33).
        cuda_index = int(device.split(":", 1)[1]) if ":" in device else 0
        load_kwargs["device_map"] = {"": cuda_index}

    with _openvla_phase("from_pretrained", nf4=use_nf4):
        model = auto_model_cls.from_pretrained(repo_id, **load_kwargs)

    if not use_nf4:
        with _openvla_phase("to_device", device=device):
            model = model.to(device)
    model.eval()
    _patch_unnormalize_for_accelerate(model)
    return model, processor


def _patch_unnormalize_for_accelerate(model: Any) -> None:
    """Make OpenVLA's ``_unnormalize_actions`` tolerate a CUDA predicted tensor.

    OpenVLA's custom ``_unnormalize_actions`` runs ``np.where(...)`` directly on
    the predicted-action tensor, which assumes it is on CPU. Under accelerate's
    device-map hooks (used for the 4-bit load) the tensor stays on CUDA, so the
    implicit ``np.asarray`` raises "can't convert cuda tensor to numpy". Wrap the
    bound method to move the tensor to CPU first (verified live 2026-06-19). A
    no-op for checkpoints without this method.
    """
    orig = getattr(model, "_unnormalize_actions", None)
    if orig is None:
        return

    def _cpu_first(normalized_actions: Any, unnorm_key: Any = None) -> Any:
        if hasattr(normalized_actions, "detach"):
            normalized_actions = normalized_actions.detach().cpu().numpy()
        return orig(normalized_actions, unnorm_key)

    model._unnormalize_actions = _cpu_first


def _seed_torch_for_sampling(torch: Any, seed: int | None) -> None:
    """Pin torch sampling for stochastic OpenVLA generation when requested."""
    if seed is None:
        return
    torch.manual_seed(seed)
    if hasattr(torch, "cuda") and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@POLICIES.register("openvla")
def _build_openvla(env_cfg: Any) -> _OpenVLAAdapter:
    """Load an OpenVLA / OpenVLA-OFT checkpoint as a transformers custom-code model."""
    spec = env_cfg.vla
    device = resolve_device(spec)

    if device.startswith("cuda"):
        _enable_expandable_segments()

    with _openvla_phase("imports"):
        import torch

        auto_model_cls, auto_processor_cls, bnb_config_cls = _import_transformers()

    # The rSkill manifest is the source of truth for the repo id + dtype.
    resolve_rskill_repo_id(spec.weights_uri, adapter_name="OpenVLA")  # validates the reference
    manifest = load_manifest_for_spec(spec)
    if manifest is None:
        raise ROSConfigError(
            "OpenVLA adapter requires a bare rSkill reference as weights_uri "
            f"resolving to a manifest; got {spec.weights_uri!r}."
        )
    repo_id, revision = _strip_hf_uri(manifest.weights_uri, field_name="weights_uri")
    _require_remote_code_ack(repo_id, revision)

    dtype_str = manifest_dtype(spec, manifest=manifest) or default_dtype_for_device(device)
    model, processor = _load_openvla_model(
        torch=torch,
        auto_model_cls=auto_model_cls,
        auto_processor_cls=auto_processor_cls,
        bnb_config_cls=bnb_config_cls,
        repo_id=repo_id,
        revision=revision,
        device=device,
        dtype_str=dtype_str,
    )
    extra = getattr(spec, "extra", {}) or {}
    torch_seed = _extra_int_or_none(extra, "openvla_torch_seed")
    _seed_torch_for_sampling(torch, torch_seed)

    scene_cameras = getattr(env_cfg.scene, "cameras", None)
    cam_keys = resolve_camera_keys(manifest, spec.extra, scene_cameras=scene_cameras)

    action_dim = int(extra.get("openvla_action_dim", _DEFAULT_ACTION_DIM))
    generation_method = str(extra.get("openvla_generation_method", _DEFAULT_GENERATION_METHOD))
    if generation_method not in {_DEFAULT_GENERATION_METHOD, _GENERATE_ACTION_VERL}:
        raise ROSConfigError(
            "OpenVLA policy_extras.openvla_generation_method must be "
            f"{_DEFAULT_GENERATION_METHOD!r} or {_GENERATE_ACTION_VERL!r}, got "
            f"{generation_method!r}."
        )

    return _OpenVLAAdapter(
        spec=spec,
        device=device,
        _model=model,
        _processor=processor,
        _torch=torch,
        _unnorm_key=_resolve_unnorm_key(spec, model),
        _action_dim=action_dim,
        _camera_keys=tuple(cam_keys) if cam_keys else ("camera1",),
        _actions_prenormalized=bool(extra.get("openvla_actions_prenormalized", False)),
        _autocast_dtype=torch.bfloat16 if device.startswith("cuda") else None,
        _generation_method=generation_method,
        _do_sample=_extra_bool(extra, "openvla_do_sample", False),
        _temperature=_extra_float(extra, "openvla_temperature", 0.6),
        _padding_max_length=_extra_positive_int_or_none(extra, "openvla_padding_max_length"),
        _action_scale=_extra_float(extra, "openvla_action_scale", 1.0),
        _binarize_gripper=_extra_bool(extra, "openvla_binarize_gripper", False),
        _gripper_threshold=_extra_float(extra, "openvla_gripper_threshold", 0.5),
        _torch_seed=torch_seed,
    )
