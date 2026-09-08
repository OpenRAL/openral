"""π0.5 (Physical Intelligence) policy adapter.

Wraps :class:`lerobot.policies.pi05.modeling_pi05.PI05Policy`. Same
observation contract as SmolVLA on LIBERO (8-D state + 2 RGB cameras),
different lerobot policy class and a 3.4 B-parameter PaliGemma backbone.
Mirrors :mod:`openral_sim.policies.smolvla`: bare rSkill reference as
weights URI, lerobot ``make_pre_post_processors`` factory, batch built
from the eval-layer ``Observation`` (flat ``state`` + ``images`` dict).

Bf16 is the default — fp32 weights are ~13.6 GiB and OOM on an 8 GiB GPU.
``QuantizationConfig.dtype`` from the rSkill manifest is honoured when set.

Imports torch / lerobot lazily so installing ``openral-sim`` never pulls
them transitively.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import structlog
from numpy.typing import NDArray
from openral_core.exceptions import ROSConfigError
from openral_observability import inference_span
from openral_rskill._diagnostics import phase_timer
from openral_rskill._vla_core import (
    apply_chunk_replay,
    build_chunk_executor,
    call_make_processors_cached_first,
    release_torch_modules,
    resolve_camera_keys,
    resolve_device,
    resolve_image_preprocessing,
    resolve_rskill_repo_revision,
    resolve_state_dim,
    to_numpy_action,
)
from openral_rskill.backend_registry import maybe_attach_pro_hooks

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
    # Independent of `_flip_images_180`: applies a vertical flip
    # (`img[::-1, :, :]`) before any other transform, matching
    # `RoboCasaGymEnv.get_basic_observation`'s H-only flip used by the
    # canonical openpi-robocasa eval. `RoMALab/pi05_robocasa-MG_*` and
    # `robocasa365_checkpoints/pi05_pretrain_human300` were both
    # benchmarked against vertically-flipped frames.
    _flip_vertical: bool = False
    _state_dim: int | None = None
    _camera_keys: tuple[str, ...] = field(default_factory=lambda: ("camera1", "camera2"))
    # Default matches SmolVLA + RoMALab pi0.5 RoboCasa-MG_300 naming; the
    # lerobot pi05 10tasks-200k (ruiname) checkpoint uses the singular
    # "observation.image.{cam}" instead — override via YAML.
    _image_input_template: str = "observation.images.{cam}"
    # Merged on top of the built-in {"camera1": "image", "camera2": "image2"}
    # map. Used by the ruiname RoboCasa checkpoint to rewrite
    # robosuite's `robot0_agentview_left_image` -> `agentview` (and
    # `robot0_eye_in_hand_image` -> `wrist`).
    _camera_aliases: dict[str, str] = field(default_factory=dict)
    _last_input_frame: NDArray[np.uint8] | None = None
    # Dtype to cast float inputs to before forward; None leaves the
    # preprocessor's dtype. Used by nf4 so PaliGemma's bf16 Linear
    # weights match the activation dtype.
    _input_dtype: Any = None
    # `torch.amp.autocast(device_type, dtype)` wrapper for forward, needed
    # in nf4/bf16 mode: some PaliGemma intermediates (RMSNorm, RoPE) run
    # in fp32 even when params are bf16, tripping a dtype mismatch.
    _autocast_dtype: Any = None
    _chunk_executor: Any = None

    def last_input_frame(self) -> NDArray[np.uint8] | None:
        return self._last_input_frame

    def reset(self) -> None:
        if self._chunk_executor is not None:
            self._chunk_executor.reset()
        elif hasattr(self._policy, "reset"):
            self._policy.reset()

    def step(self, observation: Observation, instruction: str) -> NDArray[np.float32]:
        if self._chunk_executor is not None:
            action_tensor = self._chunk_executor.select_action(
                lambda: self._prepared_batch(observation, instruction)
            )
            return to_numpy_action(self._postprocessor(action_tensor))

        batch = self._prepared_batch(observation, instruction)
        with inference_span(kind="single"), self._torch.no_grad(), self._autocast_ctx():
            action_tensor = self._policy.select_action(batch)
        action_tensor = self._postprocessor(action_tensor)
        return to_numpy_action(action_tensor)

    def _prepared_batch(self, observation: Observation, instruction: str) -> dict[str, Any]:
        """Build the full on-device, dtype-cast batch for one inference."""
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
        return cast("dict[str, Any]", batch)

    def _autocast_ctx(self) -> Any:
        """The adapter's mixed-precision context (see `_autocast_dtype` note)."""
        device_type = self.device.split(":", 1)[0]
        if self._autocast_dtype is not None and device_type in {"cuda", "cpu"}:
            return self._torch.amp.autocast(device_type=device_type, dtype=self._autocast_dtype)
        import contextlib

        return contextlib.nullcontext()

    def _chunk_forward(self, batch: dict[str, Any], **kwargs: Any) -> Any:
        """Chunk producer for the executor — predict under this adapter's autocast.

        ``kwargs`` carries the executor's RTC arguments (``inference_delay`` /
        ``prev_chunk_left_over``) straight through to lerobot; empty otherwise.
        """
        with self._torch.no_grad(), self._autocast_ctx():
            return self._policy.predict_action_chunk(batch, **kwargs)

    def close(self) -> None:
        """Drop the loaded modules, then reclaim their VRAM.

        Order matters: ``empty_cache()`` only returns already-free blocks,
        so flushing while this adapter still holds the policy frees nothing.
        See :func:`openral_rskill._vla_core.release_torch_modules`.
        """
        if self._chunk_executor is not None:
            self._chunk_executor.stop()
            self._chunk_executor = None
        release_torch_modules(
            self,
            "_policy",
            "_preprocessor",
            "_postprocessor",
            device=self.device,
            torch=self._torch,
        )

    def _build_batch(self, observation: Observation, instruction: str) -> dict[str, Any]:
        torch = self._torch
        batch: dict[str, Any] = {"task": instruction or observation.get("task", "")}

        images = observation.get("images", {})
        cam_alias = {"camera1": "image", "camera2": "image2", **self._camera_aliases}
        from openral_sim.policies._video_capture import tile_input_frames, to_input_frame

        # Record a stitched preview of every camera handed to the policy so
        # the debug video shows the real multi-camera input, not only the last
        # stream in the adapter loop.
        preview_frames: list[NDArray[np.uint8]] = []
        for cam_key in self._camera_keys:
            img = images.get(cam_key)
            if img is None:
                continue
            # `flip_vertical` is the canonical openpi-robocasa flip (H only,
            # matching `RoboCasaGymEnv.process_img`). Apply it before any
            # other transform so the input-frame debug panel and the
            # tensor handed to the policy share the same orientation.
            if self._flip_vertical:
                img = np.ascontiguousarray(np.asarray(img)[::-1, :, :])
            preview = to_input_frame(img, flip_180=self._flip_images_180)
            if preview is not None:
                preview_frames.append(preview)
            t = torch.tensor(np.asarray(img), dtype=torch.float32).div(255.0).permute(2, 0, 1)
            if self._flip_images_180:
                t = torch.flip(t, dims=[1, 2])
            t = t.unsqueeze(0).to(self.device)
            batch_key = self._image_input_template.format(cam=cam_alias.get(cam_key, cam_key))
            batch[batch_key] = t

        self._last_input_frame = tile_input_frames(preview_frames)

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

    PaliGemma's ``language_model.embed_tokens.weight`` is tied to the LM
    head; source safetensors store only the head, and our manual fast
    path (``to_empty`` + a later ``policy.tie_weights()``) materialises
    both slots separately, so the tie must be detected explicitly via
    shared-storage groups (``Tensor.untyped_storage().data_ptr()``).

    Without this, the targeted reset walk fires a ~10 s ``normal_`` init
    across the 256k×2048 ``embed_tokens.weight`` slot that the following
    ``load_state_dict`` would overwrite anyway via the tie.

    Uses ``named_parameters(remove_duplicate=False)`` rather than
    ``state_dict()``: the latter triggers ``Linear8bitLt._save_to_state_dict``,
    which crashes pre-``.to(<cuda>)`` looking for the bnb-only ``SCB`` attr;
    ``remove_duplicate=False`` is required so both tied keys are visible
    (default ``named_parameters`` yields each Parameter once).
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

    ``torch.nn.Module.to_empty(device=...)`` strips Parameter subclasses:
    an ``Int8Params`` becomes a plain ``torch.nn.Parameter`` wrapping a
    bf16 tensor, so the downstream ``policy.to(<cuda>)`` walks it via
    plain ``Tensor.to`` instead of ``Int8Params.cuda`` — bnb's int8 pack
    never fires and the ~7 GiB bf16 model lands on the GPU as bf16, OOMing
    an 8 GiB card. The nf4 fast path avoids this via
    :func:`install_prequantized_linears` / ``Params4bit.from_prequantized``;
    int8 has no prequant pack, so this re-wraps the bf16 storage in a
    fresh ``Int8Params(has_fp16_weights=False)`` so the next
    ``policy.to(<cuda>)`` dispatches through ``Int8Params.cuda`` (packs
    to int8, frees the bf16 source).

    Returns the rewrapped-module count (should match
    ``quantize_int8_in_place``'s count on the same policy).
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
    ``PI05Policy.from_pretrained`` graph allocation: the policy is already
    built on meta (``init_empty_weights``) and materialised to real CPU
    storage (``to_empty``); this fills that storage with the source bf16
    weights so the upcoming ``policy.to(<cuda>)`` has data for bnb's int8
    pack. Routes through :func:`_hf_download_cached_first` so
    ``local_files_only=True`` skips the HF Hub HEAD on a warm cache. Logs
    ``missing``/``unexpected`` key counts via structlog.

    The loaded ``state`` dict is dropped + GC'd before returning so source
    bf16 tensors don't stay resident alongside the policy's bf16 copy
    through the subsequent ``.to(<cuda>)`` — otherwise peak CPU footprint
    is 2x the model and can overshoot 8 GiB GPUs (observed 6.65 GiB peak
    vs the slow path's 4.72 GiB final, on a 7.62 GiB RTX 4070 Laptop with
    ``pi05-libero-int8`` + int8).
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

    Three URI shapes: an absolute local path is returned verbatim (a
    pre-converted lerobot checkpoint dir); a bare rSkill reference with a
    manifest ``processors`` block downloads exactly the two processor
    files via :func:`openral_sim.policies._processors.resolve_processor_dir`
    (mirrors SmolVLA / modern-ACT); a bare HF Hub repo id is
    snapshot-downloaded. The prequantized fast path
    (``load_prequantized_state_for_rskill``) pulls only ``config.json`` +
    ``model.safetensors`` + ``quantization_metadata.json``, so the
    snapshot here fetches without ``local_files_only`` to backfill any
    missing ``policy_preprocessor.json`` / ``policy_postprocessor.json``
    (~5 HEADs, ~0.5 s warm cache).
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
    # safetensors/huggingface_hub/accelerate) is ~10-30s per process;
    # wrap it so it shows in the load timeline. Shared torch +
    # `make_pre_post_processors` import lives in `_policy_loading`;
    # `PI05Policy` is pulled here to keep the lerobot dispatch local.
    with _pi05_phase("imports"):
        torch, make_pre_post_processors = lazy_import_lerobot("π0.5")
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy

    repo_id, revision = resolve_rskill_repo_revision(spec.weights_uri, adapter_name="π0.5")

    # Loaded once, reused below for dtype/chunk-replay/image-preprocessing
    # resolution. Returns None for bare hf:// / local URIs; pi05 tolerates
    # either shape.
    manifest = load_manifest_for_spec(spec)

    # Load dtype: nf4/int4 -> bitsandbytes NF4, 3.4B backbone fits <=4 GiB;
    # int8 -> bnb.nn.Linear8bitLt, ~7 GiB peak, more accurate than nf4 on
    # outlier-heavy attention; bf16 -> ~7 GiB (tight on 8 GiB); fp32 ->
    # ~13.6 GiB; default -> nf4 on CUDA, fp32 on CPU.
    # `PI05Policy.from_pretrained` has no dtype kwarg, so torch's default
    # dtype is set while loading; nf4/int8 quantize Linear weights post-load.
    dtype_str = manifest_dtype(spec, manifest=manifest) or default_dtype_for_device(device)
    use_nf4 = dtype_str.lower() in {"nf4", "4bit", "int4"}
    use_int8 = dtype_str.lower() in {"int8", "8bit", "llm_int8"}
    torch_dtype = torch_dtype_for(torch, None if (use_nf4 or use_int8) else dtype_str, device)

    # PI05Policy.__init__ ends with `self.model.to(config.device)` — if
    # device="cuda" we'd OOM before `from_pretrained` returns. Construct
    # on CPU, quantize/cast there, then move once.
    import lerobot.policies.pi05.modeling_pi05 as _pi05_mod  # noqa: F401  registers PI05Config in the choice registry
    from lerobot.configs.policies import PreTrainedConfig

    # `config.json` HEAD validation on every load; ~1-5s on a cold connection.
    with _pi05_phase("config_load", repo=repo_id):
        pi05_cfg = PreTrainedConfig.from_pretrained(repo_id, revision=revision)
    pi05_cfg.device = "cpu"
    # torch.compile bakes the graph at construction time; a later nf4
    # requantization (compute_dtype=bf16) trips a cached-graph dtype
    # mismatch at forward time, so disable compile whenever dtypes may mutate.
    if hasattr(pi05_cfg, "compile_model"):
        pi05_cfg.compile_model = False

    # Fast meta-init: `from_pretrained`/`PI05Policy(cfg)` take ~143s on CPU
    # to allocate + zero-init the 3.4B-param graph (per-tensor
    # `torch.empty` is the bottleneck, not the safetensors load).
    # `accelerate.init_empty_weights` assigns meta-device tensors (shape,
    # no storage); the Linear4bit rewrite + prequant state load below
    # materialise real storage once — measured ~14s vs ~157s slow path
    # (11x). Two cases qualify: nf4 with a prequant pack
    # (`detect_prequantized_nf4` finds `quantization_metadata.json` +
    # sibling nf4 safetensors, 1 HF HEAD, ~100ms warm/1-3s cold), or int8
    # (no prequant pack; loads bf16 `model.safetensors` directly, bnb's
    # `Int8Params.cuda()` packs to int8 on the final `.to(<cuda>)`).
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

    # Peek the safetensors header so `reset_parameters` can skip modules
    # the state-dict load will overwrite anyway — was the dominant CPU
    # cost (~60s of a 95s warm-cache load) before this gate. None means
    # the peek failed; fall back to the full reset.
    fast_state_keys: set[str] | None = (
        peek_safetensors_keys(fast_state_repo) if fast_state_repo is not None else None
    )

    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch_dtype)
    try:
        if use_fast_meta_init:
            # reason: accelerate ships no py.typed; the exact code differs by install state
            from accelerate import init_empty_weights  # type: ignore[import-untyped,unused-ignore]

            # cfg.device='meta' suppresses `PI05Policy.__init__`'s internal
            # `self.model.to(config.device)`, which would otherwise raise
            # "Cannot copy out of meta tensor" (no storage yet).
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
            # Reset immediately so later cfg.device-aware code (position_id
            # allocation, batch placement) sees the real device; params
            # stay meta until `to_empty` below.
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
        # Pre-cast every fp32 leaf param/buffer to bf16 before quantization
        # so the surviving non-Linear bits (embeddings, norms, biases) are
        # uniformly bf16 — else PaliGemma forward trips a Float/BFloat16
        # mat1/mat2 dtype mismatch.
        if not use_fast_meta_init:
            # init_empty_weights params are meta-device (no .data to cast);
            # the prequant state load below materialises bf16 directly, so
            # this cast is a no-op on that path.
            with _pi05_phase("precast_bf16"):
                for p in policy.parameters():
                    if p.dtype == torch.float32:
                        p.data = p.data.to(torch.bfloat16)
                for b in policy.buffers():
                    if b.dtype == torch.float32:
                        b.data = b.data.to(torch.bfloat16)
        with _pi05_phase("quantize_nf4"):
            # Fast meta-init path: build new bnb.nn.Linear4bit shells on
            # meta too, not real CPU storage (would allocate ~7 GiB of bf16
            # placeholders only to be overwritten by `to_empty` + prequant load).
            quantize_nf4_in_place(
                policy,
                torch=torch,
                compute_dtype=torch.bfloat16,
                new_modules_on_meta=use_fast_meta_init,
            )
        if use_fast_meta_init:
            # `to_empty` materialises real (uninitialised) CPU storage for
            # every still-meta param (Linear4bit modules from the previous
            # step already have real bnb storage). The prequant state load
            # below reports ~254 "missing" keys (bnb Params4bit sub-state +
            # RoPE inv_freq buffers, etc.) it cannot fill — leaving those at
            # uninit garbage is fatal: RMSNorm.weight=0 zeros its block
            # output, softmax saturates, NaNs propagate, and an
            # F.embedding gather later reads an out-of-bounds index and
            # CUDA asserts. `reset_parameters()` restores PyTorch's normal
            # __init__ values (kaiming for Linear, ones for norms, zeros
            # for biases, normal for embeddings) as a safe baseline before
            # the prequant load overwrites what it has data for.
            # Staging on CPU (vs `to_empty(device=cuda)` directly, ~19s
            # faster) avoids an 8 GiB OOM: bitsandbytes' Params4bit
            # `__torch_function__` doesn't intercept `empty_like`, so
            # `to_empty` on cuda would allocate the full pre-quant bf16
            # footprint (~6.5 GiB) before the nf4 pack (~4 GiB) replaces it.
            # Split into measurable sub-phases: `to_empty` (~7 GiB CPU,
            # mmap-lazy under Linux), `reset_parameters` (measured dominant
            # cost, ~30-60s warm-cache on an RTX 4070 host, mostly wasted
            # since the prequant load overwrites it — `fast_state_keys`
            # lets it skip already-covered modules), `init_buffers`
            # (re-derives position_ids + RoPE inv_freq, dropped by meta-init).
            with _pi05_phase("to_empty"):
                policy.to_empty(device="cpu")
            with _pi05_phase("reset_parameters"):
                _targeted_reset_parameters(policy, covered_keys=fast_state_keys)
            # `*.position_ids` (int64): vision embedding index table, set
            # to `arange(num_patches)`; garbage → CUDA index-out-of-bounds
            # assert on the embedding gather. `*.inv_freq` /
            # `*.original_inv_freq` (bf16/fp32): RoPE rotary coefficients
            # `1/(theta**(arange(0,d,2)/d))`, carried by PaliGemma +
            # gemma_expert with d=128 and theta from the model config;
            # garbage → NaN RoPE rotation contaminating the attention path.
            with _pi05_phase("init_buffers"), torch.no_grad():
                for name, buf in policy.named_buffers():
                    if name.endswith(".position_ids") and buf.dtype == torch.int64:
                        n = buf.shape[-1]
                        arange = torch.arange(n, dtype=torch.int64, device=buf.device)
                        buf.copy_(arange.expand_as(buf))
                    elif name.endswith((".inv_freq", ".original_inv_freq")):
                        d = buf.shape[-1] * 2
                        # Walk up to the owning module for its rope `theta` (a.k.a. `base`).
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
        # If the rSkill ships a prequantized state dict
        # (`quantization_metadata.json` at the HF repo root), load it over
        # the rewritten Linear4bit modules, replacing the ~30s bf16->nf4
        # quantize-on-`.to(cuda)` with a ~1-2s state-dict load. Produced by
        # `tools/quantize_rskill.py`.
        with _pi05_phase("prequant_state_load"):
            load_prequantized_state_for_rskill(policy, spec, torch=torch, log_event_prefix="pi05")
        # Peak GPU memory hits here: the nf4-packed weights are smaller
        # than the bf16 placeholders they replace.
        with _pi05_phase("to_device", device=device):
            policy = policy.to(device=device)
    elif use_int8:
        if not device.startswith("cuda"):
            raise ROSConfigError(
                "int8 quantization for π0.5 requires a CUDA device; got "
                f"device={device!r}. Set vla.extra.dtype='bf16' to load on CPU/MPS."
            )
        if int8_fast_meta_init:
            # Fast meta-init path: no int8 prequant safetensors exist (the
            # LLM.int8 SCB sub-state inside `Int8Params` doesn't round-trip
            # through safetensors), but we still skip lerobot's ~152s
            # `PI05Policy.from_pretrained` allocation: build on meta
            # (already done above), swap Linears -> Linear8bitLt also on
            # meta, `to_empty` to real CPU storage, then load the source
            # bf16 `model.safetensors` via `policy.load_state_dict`. bnb's
            # int8 pack happens on the final `policy.to(<cuda>)`, same as
            # the slow path.
            with _pi05_phase("quantize_int8"):
                quantize_int8_in_place(
                    policy,
                    torch=torch,
                    compute_dtype=torch.bfloat16,
                    new_modules_on_meta=True,
                )
            with _pi05_phase("to_empty"):
                policy.to_empty(device="cpu")
            # Re-establish weight tying before the reset walk, then expand
            # `fast_state_keys` to cover tied params. Without this, the
            # ~0.5B-param `embed_tokens.weight` (tied to `lm_head.weight`)
            # eats a ~10s `normal_` init that `load_state_dict` would
            # immediately discard via the tie.
            _tie_transformers_weights(policy)
            int8_reset_keys = (
                _expand_covered_keys_via_tied_storage(policy, fast_state_keys)
                if fast_state_keys is not None
                else None
            )
            with _pi05_phase("reset_parameters"):
                _targeted_reset_parameters(policy, covered_keys=int8_reset_keys)
            # Same buffer reconstruction as the nf4 fast meta-init branch
            # above: position_ids + RoPE inv_freq need canonical values
            # since `load_state_dict` won't fill them (not parameters in
            # the source safetensors).
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
            # `to_empty` above stripped the `Int8Params` subclass from
            # every `Linear8bitLt.weight`; re-wrap so `policy.to(<cuda>)`
            # dispatches through `Int8Params.cuda` (bf16->int8 pack)
            # instead of `Tensor.to` (would copy 7 GiB bf16 to GPU, OOM).
            with _pi05_phase("rebuild_int8_params"):
                rebuilt = _rebuild_int8_params_for_linear8bitlt(policy)
                _log.info("pi05_int8_params_rewrapped", modules=rebuilt)
            with _pi05_phase("to_device", device=device):
                policy = policy.to(device=device)
        else:
            # Slow fallback: bf16 from_pretrained (already paid above,
            # ~152s) -> bf16 precast -> bnb rewrite -> device move. Kept
            # for a CPU-only host even though the int8 raise above
            # currently rejects CPU (int8_fast_meta_init is CUDA-gated).
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

    # Chunk replay: same lerobot `select_action` queue as SmolVLA. The
    # shipped pi05 checkpoint defaults to `n_action_steps=1`, so a single
    # env step pays a full PaliGemma forward (~3.4B params). torch.compile
    # is not plumbed here since `pi05_cfg.compile_model` is forced off
    # above to keep the quantization path stable.
    apply_chunk_replay(policy, spec.extra, manifest=manifest)

    # `_resolve_pretrained_path` -> `resolve_processor_dir` ->
    # `materialize_processor_dir` fans out into 2 `hf_hub_download` calls
    # for the two processor jsons plus any sibling `state_file`
    # safetensors; each HEAD-checks the cache (~100ms warm, seconds cold).
    with _pi05_phase("processor_dir", repo=repo_id):
        pretrained_path = _resolve_pretrained_path(spec, repo_id)
    with _pi05_phase("make_processors"):
        # Suppresses the 5 HF HEAD/metadata round-trips lerobot's
        # `TokenizerProcessorStep` would otherwise fire at
        # `google/paligemma-3b-pt-224` on every load, warm cache or not.
        preprocessor, postprocessor = call_make_processors_cached_first(
            make_pre_post_processors,
            policy.config,
            pretrained_path=pretrained_path,
        )

    # flip_vertical (the openpi-robocasa `RoboCasaGymEnv.process_img`
    # H-only flip) is part of the typed ImagePreprocessing contract; the
    # human300 manifests carry it on, RoMALab MG_300 manifests carry it off.
    ip = resolve_image_preprocessing(manifest, spec.extra)
    flip_vertical = ip.flip_vertical
    state_dim = resolve_state_dim(manifest, spec.extra)
    scene_cameras = getattr(env_cfg.scene, "cameras", None)
    cam_keys = resolve_camera_keys(manifest, spec.extra, scene_cameras=scene_cameras)

    # Opt-in TensorRT runtime: swaps sample_actions for split-ONNX TRT
    # engines, as SmolVLA/ACT do. The hook ships in the private
    # openral-pro-trt package, looked up by name; a host without it falls
    # through to the torch path unwired above.
    if maybe_attach_pro_hooks(
        "pi05", policy, repo_id=repo_id, device=device, n_cameras=len(cam_keys)
    ):
        _log.info("pi05.runtime_tensorrt", repo_id=repo_id, n_cameras=len(cam_keys))

    adapter = _PI05Adapter(
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
        # Needed whenever policy params are reduced precision (nf4 with
        # bf16 compute, or pure bf16/fp16): PaliGemma's forward computes
        # some intermediates (RMSNorm, RoPE, image-embedding projections)
        # in fp32, colliding with bf16 Linear weights ("mat1 and mat2 must
        # have the same dtype"). autocast up/down-casts at op boundaries;
        # the input-cast matches the preprocessor's fp32 tensors to it.
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
    adapter._chunk_executor = build_chunk_executor(
        spec.extra,
        policy=policy,
        chunk_fn=adapter._chunk_forward,
        adapter_name="pi05",
    )
    return adapter
