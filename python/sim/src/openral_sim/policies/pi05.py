"""π0.5 (Physical Intelligence) policy adapter.

Wraps :class:`lerobot.policies.pi05.modeling_pi05.PI05Policy`. π0.5 shares
the same observation contract as SmolVLA on LIBERO (8-D state + 2 RGB
cameras) but uses a different lerobot policy class and a 3.4 B-parameter
PaliGemma backbone, so it gets its own adapter.

Mirrors :mod:`openral_sim.policies.smolvla` in shape:
- Bare rSkill reference required as weights URI.
- Reuses the lerobot ``make_pre_post_processors`` factory.
- Builds the policy input batch from the eval-layer ``Observation`` (flat
  ``state`` + ``images`` dict).

Bf16 is the default for π0.5 — fp32 weights are ~13.6 GiB and OOM on an
8 GiB GPU. The rSkill manifest's ``QuantizationConfig.dtype`` is honoured
when set; otherwise the lerobot default applies.

This module imports torch / lerobot lazily so installing
``openral-sim`` never pulls them transitively.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import structlog
from numpy.typing import NDArray
from openral_core.exceptions import ROSConfigError
from openral_observability import inference_span
from openral_rskill._diagnostics import phase_timer
from openral_rskill._vla_core import (
    apply_chunk_replay,
    call_make_processors_cached_first,
    resolve_camera_keys,
    resolve_device,
    resolve_image_preprocessing,
    resolve_rskill_repo_revision,
    resolve_state_dim,
    to_numpy_action,
)

# π0.5 nf4 quantization + prequantized state load both live in the
# adapter-agnostic helper module so smolvla / xvla / future-pi06 can
# reuse them. The dtype-resolution helpers (manifest_dtype,
# torch_dtype_for, default_dtype_for_device) also live there so every
# adapter speaks the same QuantizationDtype enum vocabulary.
from openral_sim._quantization import (
    default_dtype_for_device,
    detect_prequantized_nf4,
    load_prequantized_state_for_rskill,
    manifest_dtype,
    peek_safetensors_keys,
    quantize_int8_in_place,
    quantize_nf4_in_place,
    torch_dtype_for,
)
from openral_sim._quantization import (
    targeted_reset_parameters as _targeted_reset_parameters,
)
from openral_sim._quantization import (
    tie_transformers_weights as _tie_transformers_weights,
)
from openral_sim.policies._policy_loading import (
    lazy_import_lerobot,
    load_manifest_for_spec,
)
from openral_sim.policies._processors import resolve_processor_dir
from openral_sim.registry import POLICIES

if TYPE_CHECKING:
    from openral_core import VLASpec

    from openral_sim.rollout import Observation


@dataclass
class _PI05Adapter:
    """π0.5 policy adapter — applies preprocessor/postprocessor per step."""

    spec: VLASpec
    device: str
    _policy: Any
    _preprocessor: Any
    _postprocessor: Any
    _torch: Any
    _flip_images_180: bool = True
    # Independent from `_flip_images_180`. When True, applies a *vertical*
    # flip (`img[::-1, :, :]`) BEFORE any other transform -- this matches
    # `robocasa.wrappers.gym_wrapper.RoboCasaGymEnv.get_basic_observation`,
    # which the canonical openpi-robocasa eval uses to feed images to the
    # policy. Both `RoMALab/pi05_robocasa-MG_*` and
    # `robocasa/robocasa365_checkpoints/pi05_pretrain_human300` were
    # benchmarked against vertically-flipped frames, so leaving this off
    # presents the vision encoder with an upside-down scene relative to
    # training -- empirically the policy then produces small drift
    # actions and never commits to a grasp.
    _flip_vertical: bool = False
    _state_dim: int | None = None
    _camera_keys: tuple[str, ...] = field(default_factory=lambda: ("camera1", "camera2"))
    # Format string applied to each image-batch key. The default
    # ``"observation.images.{cam}"`` matches the SmolVLA + RoMALab
    # pi0.5 RoboCasa-MG_300 input feature naming; the lerobot pi05
    # 10tasks-200k checkpoint by ruiname uses the singular
    # ``"observation.image.{cam}"`` instead, so the YAML can override.
    _image_input_template: str = "observation.images.{cam}"
    # Per-camera-key -> alias remap merged on top of the built-in
    # ``{"camera1": "image", "camera2": "image2"}`` map. Used by the
    # ruiname pi05 RoboCasa checkpoint to rewrite the robosuite key
    # ``robot0_agentview_left_image`` to the model's input feature name
    # ``agentview`` (and ``robot0_eye_in_hand_image`` to ``wrist``).
    _camera_aliases: dict[str, str] = field(default_factory=dict)
    _last_input_frame: NDArray[np.uint8] | None = None
    # Dtype to cast float inputs to before forward. None → leave whatever
    # the preprocessor produced. Used by the nf4 path so PaliGemma's
    # bf16 Linear weights match the activation dtype.
    _input_dtype: Any = None
    # Wrap forward in ``torch.amp.autocast(device_type, dtype)`` when set
    # — needed in nf4 mode because some intermediate tensors (RMSNorm
    # outputs, RoPE rotations, etc.) are computed in fp32 inside
    # PaliGemma even when params are bf16, and the subsequent Linear
    # then trips a dtype mismatch. autocast silently up/down-casts so
    # mixed-precision compute Just Works.
    _autocast_dtype: Any = None

    def last_input_frame(self) -> NDArray[np.uint8] | None:
        return self._last_input_frame

    def reset(self) -> None:
        if hasattr(self._policy, "reset"):
            self._policy.reset()

    def step(self, observation: Observation, instruction: str) -> NDArray[np.float32]:
        batch = self._build_batch(observation, instruction)
        batch = self._preprocessor(batch)
        # Belt-and-suspenders device move (preprocessor sometimes returns CPU tensors).
        device_kind = self.device.split(":", 1)[0]
        for k, v in list(batch.items()):
            if hasattr(v, "device") and getattr(v, "device", None) is not None:
                v_dev = str(v.device)
                if v_dev != self.device and not v_dev.startswith(device_kind):
                    batch[k] = v.to(self.device)

        if self._input_dtype is not None:
            for k, v in list(batch.items()):
                if (
                    hasattr(v, "dtype")
                    and v.dtype.is_floating_point
                    and v.dtype != self._input_dtype
                ):
                    batch[k] = v.to(self._input_dtype)
        device_type = self.device.split(":", 1)[0]
        autocast_ctx: Any
        if self._autocast_dtype is not None and device_type in {"cuda", "cpu"}:
            autocast_ctx = self._torch.amp.autocast(
                device_type=device_type, dtype=self._autocast_dtype
            )
        else:
            import contextlib

            autocast_ctx = contextlib.nullcontext()
        with inference_span(kind="single"), self._torch.no_grad(), autocast_ctx:
            action_tensor = self._policy.select_action(batch)
        action_tensor = self._postprocessor(action_tensor)
        return to_numpy_action(action_tensor)

    def close(self) -> None:
        if self.device.startswith("cuda"):
            import contextlib

            with contextlib.suppress(Exception):
                self._torch.cuda.empty_cache()

    def _build_batch(self, observation: Observation, instruction: str) -> dict[str, Any]:
        torch = self._torch
        batch: dict[str, Any] = {"task": instruction or observation.get("task", "")}

        images = observation.get("images", {})
        cam_alias = {"camera1": "image", "camera2": "image2", **self._camera_aliases}
        from openral_sim.policies._video_capture import to_input_frame

        # See smolvla.py for rationale: capture the wrist / mounted
        # camera (last entry in ``camera_keys``), not the static agent
        # view, so the top-left "VLA input" panel matches what the
        # policy is actually attending to during manipulation.
        wrist_idx = len(self._camera_keys) - 1
        for i, cam_key in enumerate(self._camera_keys):
            img = images.get(cam_key)
            if img is None:
                continue
            # `flip_vertical` is the canonical openpi-robocasa flip (H only,
            # matching `RoboCasaGymEnv.process_img`). Apply it before any
            # other transform so the input-frame debug panel and the
            # tensor handed to the policy share the same orientation.
            if self._flip_vertical:
                img = np.ascontiguousarray(np.asarray(img)[::-1, :, :])
            if i == wrist_idx:
                self._last_input_frame = to_input_frame(img, flip_180=self._flip_images_180)
            t = torch.from_numpy(np.asarray(img)).float().div(255.0).permute(2, 0, 1)
            if self._flip_images_180:
                t = torch.flip(t, dims=[1, 2])
            t = t.unsqueeze(0).to(self.device)
            batch_key = self._image_input_template.format(cam=cam_alias.get(cam_key, cam_key))
            batch[batch_key] = t

        state = observation.get("state")
        if state is not None:
            state_np = np.asarray(state, dtype=np.float32)
            if self._state_dim is not None and state_np.shape[0] != self._state_dim:
                if state_np.shape[0] < self._state_dim:
                    pad = np.zeros(self._state_dim - state_np.shape[0], dtype=np.float32)
                    state_np = np.concatenate([state_np, pad])
                else:
                    state_np = state_np[: self._state_dim]
            batch["observation.state"] = torch.from_numpy(state_np).unsqueeze(0).to(self.device)

        return batch


_log = structlog.get_logger(__name__)


def _expand_covered_keys_via_tied_storage(policy: Any, covered_keys: set[str]) -> set[str]:
    """Extend ``covered_keys`` to include every param tied to a covered one.

    PaliGemma's ``language_model.embed_tokens.weight`` is tied to the
    LM head (Gemma's ``model.embed_tokens`` ↔ ``lm_head.weight``). The
    source safetensors only stores the head; transformers' real
    ``from_pretrained`` plumbs the tie automatically, but our manual
    fast path materialises both slots separately via ``to_empty`` and
    only learns about the tie when ``policy.tie_weights()`` runs.

    If the caller invokes that ``tie_weights`` between ``to_empty`` and
    the upcoming reset / load, the tied params end up sharing storage.
    This helper detects shared-storage groups via the state_dict's
    ``Tensor.untyped_storage().data_ptr()`` and adds every member of a
    group to ``covered_keys`` whenever any member is already covered.

    Without it, the targeted reset walk fires a ~10 s ``normal_``
    init across the 256k×2048 ``embed_tokens.weight`` slot — the value
    that the immediately-following ``load_state_dict`` would overwrite
    anyway via the tie.

    Uses ``named_parameters()`` rather than ``state_dict()`` because the
    latter triggers ``Linear8bitLt._save_to_state_dict``, which on a
    pre-``.to(<cuda>)`` policy crashes looking for the ``SCB`` attr
    that bnb only populates during the int8 pack (i.e. after the
    upcoming ``.to(<cuda>)`` we're trying to optimise).
    ``remove_duplicate=False`` is critical here — by default
    ``named_parameters`` yields each Parameter exactly once even when
    multiple module paths point to it (which is *the whole point* of
    weight tying), so the tied group's second key would be invisible.
    """
    groups: dict[int, set[str]] = {}
    for key, param in policy.named_parameters(remove_duplicate=False):
        try:
            sid = param.untyped_storage().data_ptr()
        except (AttributeError, RuntimeError):
            continue
        groups.setdefault(sid, set()).add(key)
    expanded = set(covered_keys)
    for keys_group in groups.values():
        if len(keys_group) > 1 and (expanded & keys_group):
            expanded.update(keys_group)
    return expanded


def _rebuild_int8_params_for_linear8bitlt(policy: Any) -> int:
    """Re-wrap each ``Linear8bitLt.weight`` as a fresh ``bnb.nn.Int8Params``.

    ``torch.nn.Module.to_empty(device=...)`` allocates fresh storage for
    every parameter but **strips Parameter subclasses** — after
    ``to_empty`` runs, what used to be an ``Int8Params`` is now a plain
    ``torch.nn.Parameter`` wrapping a bf16 tensor. The downstream
    ``policy.to(<cuda>)`` then walks params with the standard
    ``Tensor.to``, not ``Int8Params.to`` / ``Int8Params.cuda``, so
    bnb's int8 pack never fires and the whole bf16 model (~7 GiB on
    a 3.4 B-param backbone) lands on the GPU as bf16 — OOMs an 8 GiB
    consumer GPU long before the fast path can finish.

    The nf4 fast path side-steps the same class-stripping by calling
    :func:`install_prequantized_linears` which explicitly rebuilds
    each ``Linear4bit.weight`` as a fresh ``Params4bit.from_prequantized``.
    The int8 path has no prequant pack to install from, so this helper
    just re-wraps the (now plain-Parameter) bf16 storage in a fresh
    ``Int8Params`` with ``has_fp16_weights=False``. The next
    ``policy.to(<cuda>)`` then dispatches through ``Int8Params.cuda``
    which packs to int8 and frees the bf16 source.

    Returns the number of modules rewrapped (useful for the structlog
    sanity counter; should match the count
    ``quantize_int8_in_place`` reported on the same policy).
    """
    try:
        import bitsandbytes as bnb
    except ImportError as exc:  # pragma: no cover
        raise ROSConfigError(
            "int8 fast meta-init requires bitsandbytes; install with: "
            "uv pip install 'bitsandbytes>=0.45'"
        ) from exc

    rebuilt = 0
    for module in policy.modules():
        if not isinstance(module, bnb.nn.Linear8bitLt):
            continue
        # ``weight.data`` is bf16 CPU storage after ``to_empty`` +
        # ``load_state_dict``. Wrapping it in a fresh Int8Params with
        # ``has_fp16_weights=False`` re-arms the bnb.cuda() pack path.
        # We re-share the existing storage (no clone) — the source
        # bf16 weight is about to be replaced anyway when .to(<cuda>)
        # calls Int8Params.cuda().
        module.weight = bnb.nn.Int8Params(
            module.weight.data,
            requires_grad=False,
            has_fp16_weights=False,
        )
        rebuilt += 1
    return rebuilt


def _load_bf16_state_for_int8(policy: Any, repo_id: str, *, torch: Any) -> None:
    """Download ``<repo>/model.safetensors`` and apply via ``load_state_dict``.

    The int8 fast meta-init path's substitute for lerobot's slow
    ``PI05Policy.from_pretrained`` graph allocation. The policy has
    already been built on the meta device (via ``init_empty_weights``)
    and materialised to real CPU storage (via ``to_empty``); this
    helper fills that storage with the source bf16 weights so the
    upcoming ``policy.to(<cuda>)`` has real data for bnb's int8 pack.

    Routes through :func:`_hf_download_cached_first` so the
    ``local_files_only=True`` fast path skips the HF Hub HEAD
    validation on a warm cache. Logs ``missing`` / ``unexpected``
    key counts via structlog so the operator can spot a mismatched
    checkpoint without enabling debug logging.

    The Linear8bitLt modules' ``Int8Params`` slots already have real
    bf16 CPU storage (from ``to_empty``), so ``load_state_dict``'s
    ``param.data.copy_(...)`` is a plain bf16→bf16 copy. bnb's
    ``Int8Params.cuda()`` later computes the int8 SCB pack from the
    loaded bf16 values; nothing here is bnb-aware.

    The ``state`` dict is explicitly dropped + GC'd before the function
    returns so the source bf16 tensors don't stay resident alongside
    the policy's bf16 copy through the subsequent ``.to(<cuda>)``.
    Without that, the peak CPU footprint is 2× the model (source
    state + policy params), and the ``.to(<cuda>)`` allocator dance
    can overshoot 8 GiB GPUs that handled the slow path fine
    (observed 6.65 GiB peak vs the slow path's 4.72 GiB final on a
    7.62 GiB RTX 4070 Laptop with ``pi05-libero-nf4`` + int8).
    """
    del torch  # consumed by the caller's `.to(device)`; kept for API parity
    try:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import LocalEntryNotFoundError
        from openral_rskill._vla_core import _hf_download_cached_first
        from safetensors.torch import load_file
    except ImportError as exc:  # pragma: no cover
        raise ROSConfigError(
            "int8 fast meta-init requires huggingface_hub + safetensors; "
            "install with: just sync --all-packages --group sim"
        ) from exc

    weights_path = _hf_download_cached_first(
        hf_hub_download,
        LocalEntryNotFoundError,
        repo_id=repo_id,
        filename="model.safetensors",
    )
    state = load_file(weights_path, device="cpu")
    missing, unexpected = policy.load_state_dict(state, strict=False)
    _log.info(
        "pi05_int8_bf16_state_loaded",
        repo=repo_id,
        keys=len(state),
        missing=len(missing),
        unexpected=len(unexpected),
    )
    # Drop the source tensors before the caller's `.to(<cuda>)` so the
    # peak CPU + GPU footprint matches the slow-path baseline.
    import gc

    del state
    gc.collect()


def _pi05_phase(name: str, **fields: Any) -> Any:
    """Shortcut for ``phase_timer(name, prefix="pi05", gpu_mb=True, log=_log)``.

    Keeps the call-site short while every pi05 phase consistently emits
    ``pi05_<name>_{start,heartbeat,done}`` events on the per-module
    logger. ``gpu_mb=True`` because every measurable pi05 phase either
    moves tensors to / from the GPU or sits adjacent to one that does
    (``init_empty_weights`` → ``quantize_nf4`` → ``prequant_state_load``
    → ``to_device``); the heartbeat's GPU footprint reading is what
    distinguishes "stuck on CPU allocation" from "stuck on a CUDA call".
    """
    return phase_timer(name, prefix="pi05", gpu_mb=True, log=_log, **fields)


def _resolve_pretrained_path(spec: Any, repo_id: str) -> str:
    """Return a local directory containing the lerobot processor sidecars.

    Three URI shapes are honoured:

    * Absolute local filesystem path (``/.../checkpoint_dir``) -- returned
      verbatim. This is the shape produced for manifests whose
      ``weights_uri`` points at a pre-converted lerobot-format checkpoint
      directory.
    * Bare rSkill reference whose manifest declares a ``processors`` block
      -- per-file ``hf_hub_download`` of exactly the two processor URIs
      via :func:`openral_sim.policies._processors.resolve_processor_dir`
      (which delegates to :func:`materialize_processor_dir`). Mirrors
      the SmolVLA / modern-ACT path; ADR-0013.
    * Bare HF Hub repo id (``namespace/name``) -- snapshot-downloaded as
      before. The prequantized fast path
      (``load_prequantized_state_for_rskill``) pulls only ``config.json``
      + ``model.safetensors`` + ``quantization_metadata.json``, leaving
      the snapshot dir missing the ``policy_preprocessor.json`` /
      ``policy_postprocessor.json`` sidecars that
      ``make_pre_post_processors`` needs. A ``local_files_only=True``
      shortcut would silently return the incomplete dir; the snapshot
      call inside :func:`resolve_processor_dir` runs without it so any
      missing sidecars are fetched (~5 HEADs, ~0.5 s when fully cached).
    """
    import os

    if os.path.isabs(repo_id) or os.path.exists(repo_id):
        return repo_id

    return resolve_processor_dir(spec, repo_id)


@POLICIES.register("pi05")
def _build_pi05(env_cfg: Any) -> _PI05Adapter:  # noqa: PLR0915  # reason: load-phase orchestration (from_pretrained/meta-init/quantize/prequant/to_device) is naturally long
    """Load a π0.5 LIBERO/SO-100 finetune via the lerobot ``PI05Policy``."""
    spec = env_cfg.vla
    device = resolve_device(spec)

    # Heavy first-import cost (torch + transformers + lerobot pulling in
    # safetensors / huggingface_hub / accelerate) is paid once per
    # process but is invisible from the operator's perspective — wrap it
    # so the 10–30 s first-call cost shows up in the load timeline
    # alongside the GPU-side phases. Shared torch + ``make_pre_post_processors``
    # import lives in ``_policy_loading``; ``PI05Policy`` is pulled here
    # so the lerobot dispatch choice doesn't leak into adapters that
    # don't need it.
    with _pi05_phase("imports"):
        torch, make_pre_post_processors = lazy_import_lerobot("π0.5")
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy

    repo_id, revision = resolve_rskill_repo_revision(spec.weights_uri, adapter_name="π0.5")

    # Load the rSkill manifest up-front so its ``quantization.dtype``
    # pin can feed dtype resolution alongside any ``spec.extra["dtype"]``
    # override. The manifest is also reused below for chunk-replay and
    # image-preprocessing resolution; loading it once is cheaper than
    # twice. ``load_manifest_for_spec`` returns ``None`` for bare
    # ``hf://`` / local URIs — pi05 tolerates either shape.
    manifest = load_manifest_for_spec(spec)

    # Pick a load dtype:
    #   * dtype="nf4" / "int4"  → 4-bit NF4 quantization via bitsandbytes;
    #                             the 3.4 B PaliGemma backbone fits in
    #                             ≤4 GiB this way, leaving room for
    #                             activations on an 8 GiB GPU.
    #   * dtype="int8"          → 8-bit LLM.int8 (``bnb.nn.Linear8bitLt``).
    #                             ~7 GiB peak post-pack — between bf16 and
    #                             nf4 on memory, but more accurate than
    #                             nf4 on outlier-heavy attention blocks.
    #   * dtype="bf16"          → bf16 weights (≈ 7 GiB; tight on 8 GiB).
    #   * dtype="fp32"          → fp32 weights (≈ 13.6 GiB; needs a bigger GPU).
    #   * default               → nf4 on CUDA (fits 8 GiB), fp32 on CPU.
    #
    # ``PI05Policy.from_pretrained`` doesn't expose a dtype kwarg, so we
    # set torch's default dtype while loading; both nf4 and int8 take a
    # different code path that quantizes Linear weights post-load.
    dtype_str = manifest_dtype(spec, manifest=manifest) or default_dtype_for_device(device)
    use_nf4 = dtype_str.lower() in {"nf4", "4bit", "int4"}
    use_int8 = dtype_str.lower() in {"int8", "8bit", "llm_int8"}
    torch_dtype = torch_dtype_for(torch, None if (use_nf4 or use_int8) else dtype_str, device)

    # PI05Policy.__init__ ends with ``self.model.to(config.device)`` —
    # if config.device == "cuda" we have already OOM'd by the time
    # ``from_pretrained`` returns. Force the construction onto CPU so we
    # can quantize / cast at our leisure, then move once.
    import lerobot.policies.pi05.modeling_pi05 as _pi05_mod  # noqa: F401  registers PI05Config in the choice registry
    from lerobot.configs.policies import PreTrainedConfig

    # ``PreTrainedConfig.from_pretrained`` hits the Hub for a
    # ``config.json`` HEAD validation on every load (even with a
    # cached file), so it can stall 1–5 s on a cold connection.
    with _pi05_phase("config_load", repo=repo_id):
        pi05_cfg = PreTrainedConfig.from_pretrained(repo_id, revision=revision)
    pi05_cfg.device = "cpu"
    # torch.compile bakes the graph at construction time and assumes the
    # model's dtypes match across all sub-modules. Once we quantize a
    # subset of Linear weights to nf4 (compute_dtype=bf16) the cached
    # graph trips "mat1 and mat2 to have the same dtype" errors at
    # forward time. Disable compile when we plan to mutate dtypes.
    if hasattr(pi05_cfg, "compile_model"):
        pi05_cfg.compile_model = False

    # When the rSkill ships a prequantized nf4 pack, both
    # ``from_pretrained`` AND the plain ``PI05Policy(cfg)`` constructor
    # take ~143 s on CPU just to allocate + zero-initialise the
    # 3.4 B-param graph -- a per-tensor ``torch.empty(shape)`` allocation
    # on real RAM is the actual bottleneck, not the safetensors load.
    # Skip the allocation entirely with ``accelerate.init_empty_weights``
    # (assigns ``meta``-device tensors with shape but no storage) and
    # let the downstream ``Linear4bit`` rewrite + prequant state load
    # materialise real storage exactly once. Measured: ~14 s total load
    # vs ~157 s on the slow path (11x speedup).
    # Pick the safetensors repo that will land via ``load_state_dict``
    # after the meta-init build. Two cases qualify for the fast path:
    #
    # * nf4 with a prequant pack — ``detect_prequantized_nf4`` finds
    #   the matching ``quantization_metadata.json`` sentinel + a
    #   sibling nf4 safetensors that the upcoming
    #   ``load_prequantized_state_for_rskill`` overlays via
    #   ``install_prequantized_linears``.
    # * int8 — there is no int8 prequant pack today, but we can still
    #   skip lerobot's ~152 s ``PI05Policy.from_pretrained`` allocation
    #   by building the graph on meta and loading the bf16 source
    #   ``model.safetensors`` ourselves. bnb's ``Int8Params.cuda()``
    #   computes its int8 SCB pack from the loaded bf16 weights on the
    #   final ``.to(<cuda>)``.
    #
    # ``detect_prequantized_nf4`` makes 1 HF HEAD request (~100 ms warm,
    # 1–3 s cold); the int8 path doesn't probe — it commits to the
    # bf16 ``repo_id`` resolved from the bare rSkill reference above.
    if use_nf4:
        with _pi05_phase("detect_prequant"):
            prequant_repo = detect_prequantized_nf4(spec)
    else:
        prequant_repo = None
    nf4_fast_meta_init = prequant_repo is not None
    int8_fast_meta_init = use_int8 and device.startswith("cuda")
    use_fast_meta_init = nf4_fast_meta_init or int8_fast_meta_init
    # The safetensors source that will provide the state_dict after
    # meta init. None means "no fast path; load via from_pretrained".
    fast_state_repo: str | None = (
        prequant_repo if nf4_fast_meta_init else (repo_id if int8_fast_meta_init else None)
    )

    # Peek the safetensors header so the upcoming `reset_parameters`
    # walk can skip every module whose params will be overwritten by
    # the state-dict load. On a 3.4 B-param graph that walk runs
    # kaiming / normal init across every Linear / Embedding / RMSNorm
    # and was the dominant CPU cost (~60 s of a 95 s warm-cache load)
    # before this gate was added. `None` means the peek failed and we
    # fall back to the full reset.
    fast_state_keys: set[str] | None = (
        peek_safetensors_keys(fast_state_repo) if fast_state_repo is not None else None
    )

    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch_dtype)
    try:
        if use_fast_meta_init:
            from accelerate import init_empty_weights  # type: ignore[import-untyped]

            # Setting cfg.device='meta' suppresses the internal
            # ``self.model.to(config.device)`` call inside
            # ``PI05Policy.__init__`` (which would otherwise raise
            # ``Cannot copy out of meta tensor`` because the meta
            # parameters have no storage to copy from).
            pi05_cfg.device = "meta"
            with (
                _pi05_phase(
                    "init_empty_weights",
                    repo=repo_id,
                    prequant_repo=prequant_repo,
                    dtype=dtype_str,
                ),
                init_empty_weights(),
            ):
                policy = PI05Policy(pi05_cfg)
            # Reset the config device immediately so any later
            # ``cfg.device``-aware code path (position_id allocation,
            # input-batch device placement) sees the real device. The
            # parameters themselves are still meta until ``to_empty``
            # materialises them below.
            pi05_cfg.device = "cpu"
            if hasattr(policy, "config"):
                with contextlib.suppress(Exception):
                    policy.config.device = "cpu"
        else:
            # PaliGemma graph allocation + state_dict load is fully silent
            # inside transformers/lerobot and takes ~2 min on a warm cache.
            # Heartbeat so the user can tell loading from hung.
            with _pi05_phase("from_pretrained", repo=repo_id, dtype=dtype_str):
                policy = PI05Policy.from_pretrained(repo_id, config=pi05_cfg, revision=revision)
    finally:
        torch.set_default_dtype(prev_dtype)

    if use_nf4:
        if not device.startswith("cuda"):
            raise ROSConfigError(
                "nf4 quantization for π0.5 requires a CUDA device; got "
                f"device={device!r}. Set vla.extra.dtype='bf16' to load on CPU/MPS."
            )
        # Pre-cast every fp32 leaf parameter / buffer to bf16 BEFORE
        # quantization, so the surviving non-Linear bits (embeddings,
        # norms, biases, small projection heads kept in compute dtype)
        # are uniformly bf16 — otherwise PaliGemma forward later trips
        # "expected mat1 and mat2 to have the same dtype, but got:
        # float != BFloat16".
        if not use_fast_meta_init:
            # When loaded via init_empty_weights the parameters are
            # ``meta``-device tensors with no .data to cast. The
            # prequant state load materialises bf16 storage directly
            # in the right dtype, so this cast is a no-op there.
            with _pi05_phase("precast_bf16"):
                for p in policy.parameters():
                    if p.dtype == torch.float32:
                        p.data = p.data.to(torch.bfloat16)
                for b in policy.buffers():
                    if b.dtype == torch.float32:
                        b.data = b.data.to(torch.bfloat16)
        with _pi05_phase("quantize_nf4"):
            # On the fast meta-init path the policy tree itself is on
            # meta device; constructing fresh bnb.nn.Linear4bit shells
            # on real CPU storage would allocate ~7 GiB of bf16
            # placeholder weights only to immediately overwrite them
            # with meta clones plus the prequant state load. Build
            # the new modules on meta too — the upcoming `to_empty`
            # phase materialises real storage exactly once.
            quantize_nf4_in_place(
                policy,
                torch=torch,
                compute_dtype=torch.bfloat16,
                new_modules_on_meta=use_fast_meta_init,
            )
        if use_fast_meta_init:
            # Linear4bit modules created in the previous step already
            # carry real bnb storage; the residual params (norms /
            # biases / embeddings / conv2d / rope caches) are still
            # meta. ``to_empty`` allocates real CPU storage but with
            # uninitialized memory; the upcoming prequant state load
            # reports ~254 ``missing`` keys (bnb sub-state Params4bit
            # owns internally + ``inv_freq`` RoPE buffers + similar)
            # that the load *cannot* overwrite. Leaving those at
            # zero (or uninit garbage) is fatal: an RMSNorm.weight=0
            # zeros its block output → downstream softmax saturates
            # → NaNs propagate → an F.embedding gather later reads
            # an out-of-bounds index and CUDA asserts.
            #
            # Restore PyTorch's default ``__init__``-time values by
            # calling each module's ``reset_parameters()`` after the
            # materialisation. This re-applies kaiming for Linear,
            # ones for {LayerNorm,RMSNorm}.weight, zeros for biases,
            # normal for embeddings, etc. The prequant load below
            # then overwrites everything it has data for; the few
            # remaining keys stay at the same canonical values the
            # slow path's ``PI05Policy.from_pretrained`` would have
            # produced.
            # Materialise the meta-device parameters on CPU first, then
            # move to the target device at the end. Going straight to
            # `to_empty(device=cuda)` looks tempting (cuts ~19 s of CPU
            # allocation) but OOMs on 8 GiB GPUs: `to_empty` allocates a
            # tensor with the *meta* shape, and bitsandbytes' Params4bit
            # `__torch_function__` doesn't intercept `empty_like` -- so
            # every Linear weight tries to allocate its full pre-quant
            # bf16 footprint on GPU (~6.5 GiB) before the prequant state
            # load replaces it with the much smaller nf4 pack (~4 GiB).
            # Staging on CPU + a single final `.to(device)` keeps GPU
            # peak memory at the nf4 footprint.
            # Split into measurable sub-phases — `to_empty` allocates
            # ~7 GiB of CPU storage for the 3.4 B-param graph (mmap-lazy
            # under Linux, so the cost is paged in lazily by downstream
            # accessors), `reset_parameters` walks every module and
            # runs kaiming / normal / ones init on its tensors, and
            # `init_buffers` re-derives the position_ids + RoPE inv_freq
            # buffers the meta-init dropped on the floor. Profiling on
            # an RTX 4070 host showed the reset_parameters walk as the
            # dominant cost (~30–60 s on a warm cache) and almost
            # entirely wasted because the prequant state load below
            # overwrites every value it produces. `fast_state_keys`
            # carries the set of safetensors keys we'll load next; we
            # use it to skip reset on modules whose params are all
            # going to be replaced.
            with _pi05_phase("to_empty"):
                policy.to_empty(device="cpu")
            with _pi05_phase("reset_parameters"):
                _targeted_reset_parameters(policy, covered_keys=fast_state_keys)
            # 2. Reconstruct buffers that the original __init__
            # would have computed via ``register_buffer`` with a
            # data tensor -- ``init_empty_weights`` redirects
            # those allocations to the meta device, leaving the
            # buffers uninitialised after ``to_empty``. The two
            # buffer families we care about:
            #
            #   * ``*.position_ids`` (int64): the vision embedding
            #     index lookup table; the constructor sets it to
            #     ``arange(num_patches).unsqueeze(0)``. With
            #     garbage int64 values, the embedding gather
            #     trips ``CUDA index out of bounds`` and the
            #     kernel asserts (see git blame on this comment
            #     for the original traceback).
            #
            #   * ``*.inv_freq`` / ``*.original_inv_freq``
            #     (bfloat16/float32): RoPE rotary frequency
            #     coefficients -- ``1.0 / (theta ** (arange(0, d, 2)/d))``.
            #     The PaliGemma + gemma_expert language models
            #     both carry these, with ``d=128`` and ``theta``
            #     read from the model config. With garbage
            #     values the RoPE rotation produces NaNs that
            #     contaminate the entire attention path.
            with _pi05_phase("init_buffers"), torch.no_grad():
                for name, buf in policy.named_buffers():
                    if name.endswith(".position_ids") and buf.dtype == torch.int64:
                        n = buf.shape[-1]
                        arange = torch.arange(n, dtype=torch.int64, device=buf.device)
                        buf.copy_(arange.expand_as(buf))
                    elif name.endswith((".inv_freq", ".original_inv_freq")):
                        d = buf.shape[-1] * 2
                        # Walk up to the owning module to find
                        # the rope ``theta`` (a.k.a. ``base``).
                        mod = policy
                        for part in name.split(".")[:-1]:
                            mod = getattr(mod, part)
                        theta = (
                            getattr(mod, "rope_theta", None)
                            or getattr(mod, "base", None)
                            or 10000.0
                        )
                        freqs = 1.0 / (
                            float(theta)
                            ** (
                                torch.arange(
                                    0, d, 2, dtype=torch.float32, device=buf.device
                                ).float()
                                / d
                            )
                        )
                        buf.copy_(freqs.to(buf.dtype))
        # If the rSkill ships a pre-quantized state dict (a
        # `quantization_metadata.json` at the HF repo root), load it
        # OVER the freshly-rewritten Linear4bit modules so the ~30 s
        # bf16->nf4 quantization on .to(cuda) is replaced by a ~1-2 s
        # state-dict load. The bf16 weights we just loaded from
        # `from_pretrained` are discarded; ``tools/quantize_rskill.py``
        # is the one-shot script that produces the matching upload.
        with _pi05_phase("prequant_state_load"):
            load_prequantized_state_for_rskill(policy, spec, torch=torch, log_event_prefix="pi05")
        # Move the materialised model (CPU staging from `to_empty_cpu`
        # above, then prequant nf4 state load) to the target device. This
        # is where the peak GPU memory hits -- the nf4-packed Linear
        # weights are smaller than the bf16 placeholders they replaced,
        # so the final GPU footprint is the post-quantization one.
        with _pi05_phase("to_device", device=device):
            policy = policy.to(device=device)
    elif use_int8:
        if not device.startswith("cuda"):
            raise ROSConfigError(
                "int8 quantization for π0.5 requires a CUDA device; got "
                f"device={device!r}. Set vla.extra.dtype='bf16' to load on CPU/MPS."
            )
        if int8_fast_meta_init:
            # Fast meta-init path. There's no int8 prequant safetensors
            # (the LLM.int8 SCB sub-state inside ``Int8Params`` doesn't
            # round-trip cleanly through safetensors, so an artefact
            # producer like ``tools/quantize_rskill.py`` is impractical),
            # but we can still skip lerobot's ~152 s
            # ``PI05Policy.from_pretrained`` graph allocation: build
            # the policy on meta via ``init_empty_weights`` (already
            # done above), swap Linears → Linear8bitLt also on meta,
            # ``to_empty`` the whole tree to real CPU storage, then
            # load the source bf16 ``model.safetensors`` directly via
            # ``policy.load_state_dict``. The bnb int8 pack happens
            # on the final ``policy.to(<cuda>)`` exactly as in the
            # slow path; this branch just skips the wasted lerobot
            # allocation.
            with _pi05_phase("quantize_int8"):
                quantize_int8_in_place(
                    policy,
                    torch=torch,
                    compute_dtype=torch.bfloat16,
                    new_modules_on_meta=True,
                )
            with _pi05_phase("to_empty"):
                policy.to_empty(device="cpu")
            # Re-establish PaliGemma's weight tying BEFORE the
            # targeted reset walk, then expand ``fast_state_keys``
            # so any param sharing storage with a covered key is
            # also treated as covered. Without this, the
            # ``embed_tokens.weight`` slot (~0.5 B params) — which is
            # tied to the loaded ``lm_head.weight`` — eats a ~10 s
            # ``normal_`` init that the next ``load_state_dict`` would
            # immediately discard via the tie.
            _tie_transformers_weights(policy)
            int8_reset_keys = (
                _expand_covered_keys_via_tied_storage(policy, fast_state_keys)
                if fast_state_keys is not None
                else None
            )
            with _pi05_phase("reset_parameters"):
                _targeted_reset_parameters(policy, covered_keys=int8_reset_keys)
            # Same buffer reconstruction as the nf4 fast meta-init
            # branch (see the long comment above the nf4 init_buffers
            # block) — position_ids + RoPE inv_freq buffers need
            # canonical values because ``load_state_dict`` won't fill
            # them (they're not parameters in the source safetensors).
            with _pi05_phase("init_buffers"), torch.no_grad():
                for name, buf in policy.named_buffers():
                    if name.endswith(".position_ids") and buf.dtype == torch.int64:
                        n = buf.shape[-1]
                        arange = torch.arange(n, dtype=torch.int64, device=buf.device)
                        buf.copy_(arange.expand_as(buf))
                    elif name.endswith((".inv_freq", ".original_inv_freq")):
                        d = buf.shape[-1] * 2
                        mod = policy
                        for part in name.split(".")[:-1]:
                            mod = getattr(mod, part)
                        theta = (
                            getattr(mod, "rope_theta", None)
                            or getattr(mod, "base", None)
                            or 10000.0
                        )
                        freqs = 1.0 / (
                            float(theta)
                            ** (
                                torch.arange(
                                    0, d, 2, dtype=torch.float32, device=buf.device
                                ).float()
                                / d
                            )
                        )
                        buf.copy_(freqs.to(buf.dtype))
            with _pi05_phase("bf16_state_load", repo=repo_id):
                _load_bf16_state_for_int8(policy, repo_id, torch=torch)
            # ``to_empty`` above stripped the ``Int8Params`` subclass
            # from every ``Linear8bitLt.weight`` — re-wrap each one so
            # the upcoming ``policy.to(<cuda>)`` dispatches through
            # ``Int8Params.cuda`` (which packs bf16 → int8) instead
            # of the default ``Tensor.to`` (which would copy 7 GiB of
            # bf16 weights to the GPU and OOM the 8 GiB card).
            with _pi05_phase("rebuild_int8_params"):
                rebuilt = _rebuild_int8_params_for_linear8bitlt(policy)
                _log.info("pi05_int8_params_rewrapped", modules=rebuilt)
            with _pi05_phase("to_device", device=device):
                policy = policy.to(device=device)
        else:
            # Slow fallback: bf16 from_pretrained (already paid above,
            # ~152 s) → bf16 precast → bnb rewrite → device move.
            # Kept so a CPU-only host (``int8_fast_meta_init`` is
            # gated on CUDA) still has a working code path even
            # though the int8 raise above currently rejects CPU.
            with _pi05_phase("precast_bf16"):
                for p in policy.parameters():
                    if p.dtype == torch.float32:
                        p.data = p.data.to(torch.bfloat16)
                for b in policy.buffers():
                    if b.dtype == torch.float32:
                        b.data = b.data.to(torch.bfloat16)
            with _pi05_phase("quantize_int8"):
                quantize_int8_in_place(
                    policy,
                    torch=torch,
                    compute_dtype=torch.bfloat16,
                    new_modules_on_meta=True,
                )
            with _pi05_phase("to_device", device=device):
                policy = policy.to(device=device)
    else:
        # Cast on CPU first, then move to GPU. Doing both in one .to() loads
        # each parameter onto CUDA in fp32 before the dtype conversion, which
        # peaks at full fp32 (~13.6 GiB for π0.5) and OOMs on 8 GiB GPUs.
        with _pi05_phase("cast_and_to_device", device=device, dtype=dtype_str):
            policy = policy.to(dtype=torch_dtype).to(device=device)
    policy.eval()

    # Chunk replay: same lerobot ``select_action`` queue as SmolVLA. The
    # shipped pi05 checkpoint defaults to ``n_action_steps=1``, so a
    # single env step pays a full PaliGemma forward (~3.4 B params).
    # ``torch.compile`` is intentionally NOT plumbed here because
    # ``pi05_cfg.compile_model`` is forced off above to keep the
    # quantization path stable; opt back in by editing the adapter.
    # ``manifest`` was loaded at the top of this function so its
    # ``quantization.dtype`` pin could feed dtype resolution; the
    # resolvers below reuse the same instance.
    apply_chunk_replay(policy, spec.extra, manifest=manifest)

    # `_resolve_pretrained_path` → `resolve_processor_dir` →
    # `materialize_processor_dir`, which fans out into 2 `hf_hub_download`
    # calls for `policy_preprocessor.json` + `policy_postprocessor.json`
    # plus any sibling `state_file` safetensors. Each call HEAD-checks
    # the cache (~100 ms warm, several seconds cold) — wrap so the cost
    # shows up in the timeline.
    with _pi05_phase("processor_dir", repo=repo_id):
        pretrained_path = _resolve_pretrained_path(spec, repo_id)
    with _pi05_phase("make_processors"):
        # ``call_make_processors_cached_first`` suppresses the 5 HF HEAD /
        # metadata round-trips the lerobot ``TokenizerProcessorStep`` would
        # otherwise fire at ``google/paligemma-3b-pt-224`` on every load
        # against an already-warm cache.
        preprocessor, postprocessor = call_make_processors_cached_first(
            make_pre_post_processors,
            policy.config,
            pretrained_path=pretrained_path,
        )

    # Manifest-first resolution. flip_vertical (the openpi-robocasa
    # `RoboCasaGymEnv.process_img` H-only flip) is part of the typed
    # ImagePreprocessing contract; the human300 manifests carry it on,
    # the RoMALab MG_300 manifests carry it off.
    ip = resolve_image_preprocessing(manifest, spec.extra)
    flip_vertical = ip.flip_vertical
    state_dim = resolve_state_dim(manifest, spec.extra)
    scene_cameras = getattr(env_cfg.scene, "cameras", None)
    cam_keys = resolve_camera_keys(manifest, spec.extra, scene_cameras=scene_cameras)

    return _PI05Adapter(
        spec=spec,
        device=device,
        _policy=policy,
        _preprocessor=preprocessor,
        _postprocessor=postprocessor,
        _torch=torch,
        _flip_images_180=ip.flip_180,
        _flip_vertical=flip_vertical,
        _state_dim=state_dim,
        _camera_keys=cam_keys,
        _image_input_template=ip.input_template,
        _camera_aliases=dict(ip.aliases),
        # The mixed-precision plumbing is needed whenever the policy
        # parameters are in a reduced precision (nf4 with bf16 compute
        # OR pure bf16 / fp16), not just under nf4. PaliGemma's
        # forward materialises some intermediate tensors in fp32
        # (RMSNorm outputs, RoPE rotations, image-embedding projections)
        # which then collide with bf16 Linear weights -- ``mat1 and mat2
        # must have the same dtype: Float vs BFloat16``. Enabling
        # ``autocast`` lets torch up/down-cast at op boundaries; the
        # input-cast ensures the preprocessor's fp32 image/state
        # tensors enter the model at the matching dtype.
        _input_dtype=(
            torch.bfloat16
            if (use_nf4 or use_int8 or torch_dtype == torch.bfloat16)
            else (torch.float16 if torch_dtype == torch.float16 else None)
        ),
        _autocast_dtype=(
            torch.bfloat16
            if (use_nf4 or use_int8 or torch_dtype == torch.bfloat16)
            else (torch.float16 if torch_dtype == torch.float16 else None)
        ),
    )
