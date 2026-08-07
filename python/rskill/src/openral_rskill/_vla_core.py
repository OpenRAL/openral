"""Shared helpers for VLA adapters (Layer 3 — Skill / S1).

Internal module. Not part of the public ``openral_rskill`` surface.

Every VLA family — SmolVLA, π0.5, xVLA, ACT, Diffusion Policy — needs the
same three things at the boundary:

1. Resolve ``VLASpec.device`` (``"auto"`` → ``"cuda:0"`` / ``"mps"`` / ``"cpu"``).
2. Resolve ``VLASpec.weights_uri`` (a bare rSkill reference — name, path, or HF repo id)
   to a bare HuggingFace Hub repo id.
3. Call ``policy.select_action(batch)`` inside an ``inference_span`` and a
   ``torch.no_grad()`` context, then squeeze the result to a 1-D float32
   NumPy action.

Before this module these three steps were copy-pasted across each eval
adapter under ``openral_sim.{policies,backends}`` and (with thread-aware extras)
``openral_rskill.smolvla.ChunkedExecutor``. The duplication had a real
cost: the ``inference_span`` instrumentation only existed on the skill-side
copy, so ``openral sim run`` runs produced no inference spans at all.

This module owns those three seams; family-specific batch construction,
camera handling, and post-processor pipelines stay where they are.
"""

from __future__ import annotations

import contextlib
import gc
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from time import perf_counter_ns
from typing import TYPE_CHECKING, Any

import numpy as np
import structlog
from numpy.typing import NDArray
from openral_core.exceptions import ROSConfigError
from openral_observability import InferenceKind as _InferenceKind
from openral_observability import inference_span, semconv

if TYPE_CHECKING:
    from openral_core import ImagePreprocessing, RSkillManifest, VLASpec

    from openral_rskill.executor import ChunkedExecutor

# Re-exported from the span helper that owns the label (design §9 closed set);
# kept in `__all__` here for the adapters that import it from _vla_core.
InferenceKind = _InferenceKind


def resolve_device(spec: VLASpec) -> str:
    """Resolve ``VLASpec.device`` to a concrete torch device string.

    ``"auto"`` resolves to ``"cuda:0"`` if CUDA is available, then ``"mps"``
    on Apple Silicon, then ``"cpu"``. Any other value is returned as-is.

    Args:
        spec: The :class:`openral_core.VLASpec` from a SimEnvironment config.

    Returns:
        A torch device string (``"cpu"``, ``"cuda:0"``, ``"mps"``).
    """
    if spec.device != "auto":
        return spec.device
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda:0"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_rskill_repo_id(weights_uri: str, *, adapter_name: str) -> str:
    """Resolve a bare rSkill reference to a bare HF Hub repo id.

    VLA adapters only accept rSkill-backed weights so the manifest (and
    its embodiment / capability / sensor contract) is always the source
    of truth. Explicit URI schemes (``hf://``, ``local://``, ``file://``)
    are rejected — pass a bare name or path instead.

    Args:
        weights_uri: The full ``VLASpec.weights_uri`` value.
        adapter_name: Human-readable adapter name used in the error message
            (e.g. ``"SmolVLA"``, ``"xVLA"``, ``"Diffusion Policy"``).

    Returns:
        Bare HF Hub repo id, e.g. ``"lerobot/smolvla_libero"``.

    Raises:
        ROSConfigError: If ``weights_uri`` carries an explicit URI scheme.
    """
    for bad in ("hf://", "local://", "file://", "http://", "https://"):
        if weights_uri.startswith(bad):
            raise ROSConfigError(
                f"{adapter_name} adapter requires a bare rSkill reference, got {weights_uri!r}; "
                "package the policy as an rSkill (rskills/<name>/rskill.yaml) and "
                "reference it by name or path."
            )
    from openral_rskill.loader import resolve_rskill_to_hf

    return str(resolve_rskill_to_hf(weights_uri))


def resolve_rskill_repo_revision(weights_uri: str, *, adapter_name: str) -> tuple[str, str | None]:
    """Resolve a bare rSkill reference to ``(repo_id_or_path, revision)``.

    Like :func:`resolve_rskill_repo_id`, but also returns the optional
    ``@<branch-or-sha>`` revision pin so callers can thread it into
    ``from_pretrained`` / ``snapshot_download`` (HF ignores an ``@<sha>`` glued
    onto the repo id — security audit 2026-06, finding H4). Emits a structured
    ``rskill.unpinned_weights`` warning when an ``hf://`` skill is unpinned,
    surfacing the reproducibility/supply-chain risk (CLAUDE.md §1.8) without
    breaking unpinned manifests.

    Args:
        weights_uri: The full ``VLASpec.weights_uri`` value (bare rSkill ref).
        adapter_name: Human-readable adapter name used in the error message.

    Returns:
        ``(repo_id, revision)`` for ``hf://`` weights (``revision`` is ``None``
        when unpinned), or ``(absolute_path, None)`` for ``local://`` weights.

    Raises:
        ROSConfigError: If ``weights_uri`` carries an explicit URI scheme.
    """
    for bad in ("hf://", "local://", "file://", "http://", "https://"):
        if weights_uri.startswith(bad):
            raise ROSConfigError(
                f"{adapter_name} adapter requires a bare rSkill reference, got {weights_uri!r}; "
                "package the policy as an rSkill (rskills/<name>/rskill.yaml) and "
                "reference it by name or path."
            )
    from openral_rskill.loader import resolve_rskill_to_hf_with_revision

    repo_id, revision = resolve_rskill_to_hf_with_revision(weights_uri)
    if revision is None and not repo_id.startswith("/"):
        log = structlog.get_logger("openral_rskill._vla_core")
        log.warning(
            "rskill.unpinned_weights",
            adapter=adapter_name,
            repo=repo_id,
            note="weights_uri is unpinned; pin '@<sha>' for reproducible loads (CLAUDE.md §1.8).",
        )
    return repo_id, revision


def resolve_image_preprocessing(
    manifest: RSkillManifest | None, spec_extra: dict[str, Any]
) -> ImagePreprocessing:
    """Build the ``ImagePreprocessing`` block the adapter should apply.

    Precedence (strict, no auto-derivation):

    1. ``spec_extra`` keys (``flip_180``, ``image_input_template``,
       ``camera_aliases``, ``image_max_crops``) — per-rollout YAML override.
    2. ``manifest.image_preprocessing`` — per-checkpoint contract from
       ``rskill.yaml``.
    3. ``ImagePreprocessing()`` schema defaults (``flip_180=False``,
       ``input_template="observation.images.{cam}"``, empty aliases,
       ``image_max_crops=None``).

    No fallback to ``policy.config.input_features`` or any other
    heuristic; missing per-checkpoint hints surface as the schema default
    so adapters fail loud on first run instead of silently changing
    behaviour when the manifest's free-text ``metadata.notes`` block
    documented a flip the resolver didn't know about.

    Args:
        manifest: The loaded rSkill manifest, or ``None`` when the
            adapter is invoked with a non-rSkill weights_uri (legacy
            tests; this path falls through to defaults + ``spec_extra``).
        spec_extra: ``VLASpec.extra`` dict from the SimEnvironment YAML.

    Returns:
        A fresh :class:`openral_core.ImagePreprocessing` instance
        combining the inputs by precedence.
    """
    from openral_core import ImagePreprocessing as _ImagePreprocessing

    manifest_ip = manifest.image_preprocessing if manifest is not None else None

    flip_180 = bool(
        spec_extra.get(
            "flip_180",
            spec_extra.get(
                "flip_images_180",
                manifest_ip.flip_180 if manifest_ip is not None else False,
            ),
        )
    )
    flip_vertical = bool(
        spec_extra.get(
            "flip_vertical",
            manifest_ip.flip_vertical if manifest_ip is not None else False,
        )
    )
    input_template = str(
        spec_extra.get(
            "image_input_template",
            manifest_ip.input_template if manifest_ip is not None else "observation.images.{cam}",
        )
    )
    aliases_obj = spec_extra.get("camera_aliases")
    if isinstance(aliases_obj, dict):
        aliases: dict[str, str] = {str(k): str(v) for k, v in aliases_obj.items()}
    elif manifest_ip is not None:
        aliases = dict(manifest_ip.aliases)
    else:
        aliases = {}

    raw_max_crops = spec_extra.get(
        "image_max_crops",
        manifest_ip.image_max_crops if manifest_ip is not None else None,
    )
    image_max_crops = int(raw_max_crops) if raw_max_crops is not None else None

    return _ImagePreprocessing(
        flip_180=flip_180,
        flip_vertical=flip_vertical,
        input_template=input_template,
        aliases=aliases,
        norm_tag=manifest_ip.norm_tag if manifest_ip is not None else None,
        image_max_crops=image_max_crops,
    )


def resolve_state_dim(manifest: RSkillManifest | None, spec_extra: dict[str, Any]) -> int | None:
    """Return the per-checkpoint proprio state dimension, or ``None``.

    Precedence:

    1. ``spec_extra["state_dim"]`` — YAML override.
    2. ``manifest.state_contract.dim`` — per-checkpoint contract.
    3. ``None`` — adapter falls back to the policy's own preprocessor
       width (no clipping / padding applied).

    No auto-derivation from ``policy.config.input_features`` — keeping
    the resolver heuristic-free is the whole point.
    """
    dim_obj = spec_extra.get("state_dim")
    if isinstance(dim_obj, int) and dim_obj > 0:
        return dim_obj
    if manifest is not None and manifest.state_contract is not None:
        return manifest.state_contract.dim
    return None


def resolve_camera_keys(
    manifest: RSkillManifest | None,
    spec_extra: dict[str, Any],
    *,
    scene_cameras: list[str] | tuple[str, ...] | None = None,
    default: tuple[str, ...] = ("camera1", "camera2"),
) -> tuple[str, ...]:
    """Resolve which scene camera keys the adapter pulls from the observation.

    Precedence:

    1. ``spec_extra["camera_keys"]`` — YAML override (list of strings).
    2. ``scene_cameras`` — the ``scene.cameras`` field from the
       SimEnvironment YAML when present. Auto-uses the scene's actual
       camera names.
    3. ``default`` — adapter-supplied fallback (typically the LIBERO
       ``("camera1", "camera2")`` pair).

    The manifest itself does **not** carry a camera-key list — that's
    a scene-side property, not a checkpoint property. The manifest's
    ``ImagePreprocessing.aliases`` *renames* these keys to the
    checkpoint's input-feature names; resolving the source key happens
    here.
    """
    extra_keys = spec_extra.get("camera_keys")
    if isinstance(extra_keys, (list, tuple)) and extra_keys:
        return tuple(str(k) for k in extra_keys)
    if scene_cameras:
        return tuple(str(k) for k in scene_cameras)
    return default


def apply_chunk_replay(
    policy: Any,
    spec_extra: dict[str, Any],
    *,
    manifest: RSkillManifest | None = None,
    default_n_action_steps: int | None = None,
) -> int:
    """Override ``policy.config.n_action_steps`` for chunk replay.

    Lerobot policies emit ``chunk_size`` actions per heavy forward but the
    shipped checkpoints typically set ``n_action_steps=1``, throwing the
    rest of the chunk away and paying a full forward every env step.
    Adapters call this helper with the **paper-faithful default** for
    their VLA family so ``openral benchmark run`` reproduces published numbers
    without per-suite extras. ``vla.extra.n_action_steps`` always wins.

    Per-family paper defaults (passed in by each adapter):

    - SmolVLA / π0.5: ``chunk_size`` (full chunk, synchronous mode the
      SmolVLA paper documents as ``inference_mode: synchronous``).
    - ACT: ``1`` (per-step re-inference; paper uses temporal ensembling,
      see ``temporal_ensemble_coeff`` plumbing in the ACT adapter).
    - Diffusion Policy: not via this helper (the adapter pins
      ``n_action_steps=8`` directly from the checkpoint config).

    Args:
        policy: A lerobot-style policy with a ``config`` attribute that
            exposes ``chunk_size`` and ``n_action_steps``.
        spec_extra: ``VLASpec.extra`` dict; ``n_action_steps`` overrides
            the default.
        manifest: Loaded rSkill manifest, or ``None`` when no manifest is
            available. When set and its ``n_action_steps`` field is
            populated, that value is preferred over
            ``default_n_action_steps`` -- paper-faithful per-checkpoint
            default from ``rskill.yaml``.
        default_n_action_steps: Adapter-supplied fallback when neither
            ``spec_extra`` nor the manifest carries ``n_action_steps``.
            ``None`` (the historical default) means "use ``chunk_size``"
            -- paper-faithful for SmolVLA / π0.5; pass ``1`` from the
            ACT adapter.

    Returns:
        The applied ``n_action_steps`` value (clamped to ``[1, chunk_size]``).
    """
    chunk_size = int(getattr(policy.config, "chunk_size", 1) or 1)
    # Precedence: spec_extra > manifest.n_action_steps > caller default > chunk_size.
    if "n_action_steps" in spec_extra:
        n_steps: int = int(spec_extra["n_action_steps"])
    elif manifest is not None and manifest.n_action_steps is not None:
        n_steps = manifest.n_action_steps
    elif default_n_action_steps is not None:
        n_steps = default_n_action_steps
    else:
        n_steps = chunk_size
    n_action_steps = max(1, min(n_steps, chunk_size))
    policy.config.n_action_steps = n_action_steps
    return n_action_steps


_CUDAGRAPH_COMPILE_MODES = frozenset({"reduce-overhead", "max-autotune"})
"""``torch.compile`` modes that may capture CUDA graphs.

CUDA-graph replay reuses static output buffers, so any tensor a caller
holds across two invocations of the compiled callable (lerobot's internal
action queue holds *views* of the chunk tensor; ``ChunkedExecutor``
pre-fetches chunk N+1 while up to ``prefetch_at`` actions of chunk N are
still queued) is silently overwritten by the next replay. Outputs under
these modes must be cloned before they escape the compiled boundary.
"""


def _has_bnb_quantized_modules(policy: Any) -> bool:
    """Return True when any submodule of *policy* comes from ``bitsandbytes``.

    Detects nf4 / LLM.int8 quantized policies (``bnb.nn.Linear4bit`` /
    ``bnb.nn.Linear8bitLt`` rewrites from ``openral_sim._quantization``) via
    the class' module path, so this never imports bitsandbytes itself.
    Non-``nn.Module`` policies (no ``modules()``) report False.
    """
    modules = getattr(policy, "modules", None)
    if not callable(modules):
        return False
    return any(type(m).__module__.startswith("bitsandbytes") for m in policy.modules())


def _clone_chunk_output(out: Any, torch: Any) -> Any:
    """Clone every tensor in a chunk forward's output (tensor / tuple / list / dict).

    Detaches the result from CUDA-graph static buffers so downstream
    holders (lerobot's action queue, ``ChunkedExecutor._bg_result``) own
    their storage. Non-tensor leaves pass through unchanged.
    """
    if isinstance(out, torch.Tensor):
        return out.clone()
    if isinstance(out, tuple):
        return tuple(_clone_chunk_output(o, torch) for o in out)
    if isinstance(out, list):
        return [_clone_chunk_output(o, torch) for o in out]
    if isinstance(out, dict):
        return {k: _clone_chunk_output(v, torch) for k, v in out.items()}
    return out


def maybe_compile_chunk_forward(
    policy: Any,
    spec_extra: dict[str, Any],
    device: str,
    torch: Any,
    *,
    method_name: str = "_get_action_chunk",
) -> bool:
    """Best-effort ``torch.compile`` of the policy's heavy chunk forward.

    Wraps the compiled callable so a backend failure (Triton missing CC,
    OOM at first forward, CUDA-graph recapture errors with
    ``reduce-overhead``) latches into eager mode for the rest of the
    rollout instead of crashing the episode. Skipped on CPU because the
    Inductor backend gives ~nothing without a GPU. Opt-in via
    ``spec_extra['compile'] = True``; mode via ``spec_extra['compile_mode']``
    (``default``, ``reduce-overhead``, ``max-autotune``).

    Two safety gates:

    * **bitsandbytes-quantized policies are never compiled.** Mixed
      nf4/bf16 graphs trip ``"mat1 and mat2 must have the same dtype"``
      at forward time (the documented reason the pi05 adapter forces
      ``compile_model = False``), and bnb custom ops graph-break away
      most of the benefit. Logs ``vla_compile_skipped_bnb_quantized``
      and returns False.
    * **CUDA-graph modes clone their output.** Under ``reduce-overhead``
      / ``max-autotune`` the compiled callable may return views of a
      static replay buffer; the wrapper routes every output through
      :func:`_clone_chunk_output` so queued action views are never
      overwritten by the next chunk's replay (the pre-fetch pattern in
      ``ChunkedExecutor`` holds chunk-N views while chunk N+1 runs).

    Args:
        policy: Policy whose ``method_name`` attribute is the heavy
            chunk forward to compile.
        spec_extra: ``VLASpec.extra`` dict; reads ``compile`` /
            ``compile_mode``.
        device: Resolved device string (``cpu`` / ``cuda:0`` / ``mps``).
        torch: The imported ``torch`` module (passed in to keep this file
            import-light).
        method_name: Name of the policy attribute to wrap; defaults to
            lerobot's ``_get_action_chunk``.

    Returns:
        True if a compiled wrapper was installed (or queued lazily),
        False if compile was skipped or setup failed.
    """
    if not bool(spec_extra.get("compile", False)):
        return False
    if not device.startswith("cuda"):
        return False
    target = getattr(policy, method_name, None)
    if not callable(target):
        return False

    compile_mode = str(spec_extra.get("compile_mode", "default"))
    log = structlog.get_logger("openral_rskill._vla_core")
    if _has_bnb_quantized_modules(policy):
        log.warning(
            "vla_compile_skipped_bnb_quantized",
            mode=compile_mode,
            method=method_name,
        )
        return False
    try:
        compiled = torch.compile(target, mode=compile_mode)
    except Exception as exc:
        log.warning(
            "vla_compile_setup_failed",
            error=str(exc),
            mode=compile_mode,
            method=method_name,
        )
        return False

    fell_back = [False]
    clone_output = compile_mode in _CUDAGRAPH_COMPILE_MODES

    def _safe_compiled(*args: Any, **kwargs: Any) -> Any:
        if fell_back[0]:
            out = target(*args, **kwargs)
        else:
            try:
                out = compiled(*args, **kwargs)
            except Exception as exc:
                fell_back[0] = True
                log.warning(
                    "vla_compile_runtime_fallback",
                    error=str(exc),
                    mode=compile_mode,
                    method=method_name,
                )
                out = target(*args, **kwargs)
        # Clone on the eager-fallback branch too: the cudagraph-mode
        # guarantee — outputs never alias policy-internal storage —
        # must hold regardless of which branch produced them.
        return _clone_chunk_output(out, torch) if clone_output else out

    setattr(policy, method_name, _safe_compiled)
    return True


def run_inference(
    policy: Any,
    batch: dict[str, Any],
    *,
    chunk_index: int | None = None,
    kind: InferenceKind = "single",
    chunk_size: int | None = None,
    engine: str | None = None,
    call: Callable[..., Any] | None = None,
    call_kwargs: dict[str, Any] | None = None,
) -> Any:
    """Call ``policy.select_action(batch)`` inside an OTel span and ``no_grad``.

    This is the single seam where every VLA inference call is instrumented;
    every adapter on both the eval and skill paths must go through here so
    ``inference.kind`` / ``inference.chunk_index`` / ``inference.chunk_size``
    / ``inference.engine`` / ``inference.device`` spans show up uniformly
    in traces.

    Args:
        policy: A lerobot-style policy with a ``select_action(batch)``
            method that returns a torch tensor.
        batch: Pre-processed observation dict already on the inference device.
        chunk_index: Sequence number of the chunk being computed (skill
            path with chunked execution); ``None`` for single-step eval.
        kind: ``"foreground"`` / ``"prefetch"`` for chunked execution,
            ``"single"`` for per-step eval adapters.
        chunk_size: Chunk length recorded as a span attribute when known.
        engine: Inference engine label (``"torch"`` / ``"trt"`` /
            ``"onnx"`` / ``"jit"`` / …). Defaults to ``"torch"`` since
            every shipped adapter dispatches through PyTorch today; TRT
            and ONNX adapters pass their own value.
        call: Custom inference callable invoked as ``call(batch)`` INSTEAD of
            ``policy.select_action(batch)``. The seam stays the single
            instrumented entry point for chunk producers with autocast,
            decoding, or non-lerobot APIs. ``policy`` may be ``None`` in this
            mode.
        call_kwargs: Extra keyword arguments splatted into ``call`` — the RTC
            executor threads ``inference_delay`` / ``prev_chunk_left_over``
            through here. ``None`` (the default) calls ``call(batch)`` exactly
            as before, so non-RTC producers keep their 1-arg signature. When
            ``inference_delay`` is present it is recorded on the span as
            ``inference.rtc_delay``.

    Returns:
        The raw tensor returned by the invoked callable / policy method.
    """
    import torch

    # ``policy.device`` is the lerobot convention; fall back to None so the
    # span helper omits the attribute on adapters that don't track it.
    device = getattr(policy, "device", None)
    extras: dict[str, Any] = {"engine": resolve_inference_engine(policy, engine)}
    if chunk_size is not None:
        extras["chunk_size"] = chunk_size
    if device is not None:
        extras["device"] = str(device)
    if call_kwargs and "inference_delay" in call_kwargs:
        extras["rtc_delay"] = int(call_kwargs["inference_delay"] or 0)
    with (
        inference_span(chunk_index=chunk_index, kind=kind, **extras) as span,
        # NOT ``torch.inference_mode()``: lerobot's RTC guidance calls
        # ``autograd.grad`` inside ``RTCProcessor.denoise_step``, which raises on
        # inference-mode tensors. ``no_grad`` still suppresses graph building for
        # every non-RTC adapter. Pinned by
        # tests/sim/test_smolvla_rtc.py::test_inference_mode_would_break_the_guidance.
        torch.no_grad(),
    ):
        started_ns = perf_counter_ns()
        try:
            if call is not None:
                return call(batch, **call_kwargs) if call_kwargs else call(batch)
            return policy.select_action(batch)
        finally:
            elapsed_ms = (perf_counter_ns() - started_ns) / 1_000_000.0
            span.set_attribute(semconv.INFERENCE_DURATION_MS, elapsed_ms)


def resolve_inference_engine(owner: Any, declared: str | None = None) -> str:
    """Return the active inference backend, preferring runtime attachments.

    Optional policy plugins replace callables after the manifest is loaded, so
    the manifest runtime may no longer describe the code actually executing.
    Plugins can expose ``_openral_inference_engine`` explicitly; the released
    OpenRAL Pro TRT plugin predates that marker, so its entry-point module is
    also recognized.
    """
    candidates = [owner]
    adapter = getattr(owner, "_adapter", None)
    if adapter is not None:
        candidates.append(adapter)
    for candidate in tuple(candidates):
        policy = getattr(candidate, "_policy", None)
        if policy is not None:
            candidates.append(policy)
    for candidate in tuple(candidates):
        model = getattr(candidate, "model", None)
        if model is not None:
            candidates.extend((model, getattr(model, "sample_actions", None)))

    for candidate in candidates:
        if candidate is None:
            continue
        marker = getattr(candidate, "_openral_inference_engine", None)
        if isinstance(marker, str) and marker:
            return _normalize_inference_engine(marker)
        module = str(getattr(candidate, "__module__", type(candidate).__module__))
        if module == "openral_pro_trt" or module.startswith("openral_pro_trt."):
            return "trt"

    return _normalize_inference_engine(declared or "torch")


def _normalize_inference_engine(engine: str) -> str:
    """Normalize manifest/runtime backend names to telemetry labels."""
    return {"pytorch": "torch", "tensorrt": "trt"}.get(engine.lower(), engine.lower())


_RTC_ADAPTERS = frozenset({"smolvla", "pi05"})
"""Adapters whose lerobot policies are flow-matching and carry ``rtc_config``.

molmoact2/pi0_fast also support RTC upstream but are out of Phase A scope —
extend this set (and the adapter's chunk_fn kwargs pass-through) to add one.
"""

_RTC_KEYS = frozenset(
    {"enabled", "execution_horizon", "max_guidance_weight", "prefix_attention_schedule", "debug"}
)


def _parse_rtc_config(spec_extra: dict[str, Any], *, adapter_name: str) -> Any:
    """Parse ``policy_extras.rtc`` into a lerobot :class:`RTCConfig` (or ``None``).

    Args:
        spec_extra: The ``VLASpec.extra`` dict (manifest ``policy_extras``).
        adapter_name: Adapter label; RTC is refused outside ``_RTC_ADAPTERS``.

    Returns:
        A ``lerobot.policies.rtc.RTCConfig``, or ``None`` when no ``rtc`` block.

    Raises:
        ROSConfigError: Non-mapping block, unknown key, unknown schedule,
            non-boolean ``enabled``/``debug``, non-numeric or non-positive
            ``max_guidance_weight``, non-positive ``execution_horizon``, or a
            non-flow-matching adapter.
    """
    raw = spec_extra.get("rtc")
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ROSConfigError(f"{adapter_name}: policy_extras.rtc must be a mapping")
    if adapter_name not in _RTC_ADAPTERS:
        raise ROSConfigError(
            f"{adapter_name}: RTC needs a flow-matching policy; supported adapters: "
            f"{sorted(_RTC_ADAPTERS)}"
        )
    unknown = set(raw) - _RTC_KEYS
    if unknown:
        raise ROSConfigError(f"{adapter_name}: unknown policy_extras.rtc keys {sorted(unknown)}")

    from lerobot.configs import RTCAttentionSchedule  # deferred: lerobot is a heavy optional dep
    from lerobot.policies.rtc import RTCConfig  # deferred: lerobot is a heavy optional dep

    schedule_raw = raw.get("prefix_attention_schedule", "exp")
    try:
        schedule = RTCAttentionSchedule[str(schedule_raw).upper()]
    except KeyError as exc:
        raise ROSConfigError(
            f"{adapter_name}: unknown rtc.prefix_attention_schedule {schedule_raw!r} "
            f"(expected one of {[s.name.lower() for s in RTCAttentionSchedule]})"
        ) from exc
    horizon_raw = raw.get("execution_horizon", 10)
    if isinstance(horizon_raw, bool) or not isinstance(horizon_raw, int) or horizon_raw < 1:
        raise ROSConfigError(f"{adapter_name}: rtc.execution_horizon must be a positive integer")
    # Flags are checked, never coerced: `bool("false")` is True, so a quoted
    # YAML boolean would silently arm RTC (or its debug tensors) on a manifest
    # whose author meant the opposite.
    for flag in ("enabled", "debug"):
        if flag in raw and not isinstance(raw[flag], bool):
            raise ROSConfigError(f"{adapter_name}: rtc.{flag} must be a boolean")
    weight_raw = raw.get("max_guidance_weight", 10.0)
    if isinstance(weight_raw, bool) or not isinstance(weight_raw, (int, float)):
        raise ROSConfigError(f"{adapter_name}: rtc.max_guidance_weight must be a number")
    try:
        return RTCConfig(
            enabled=bool(raw.get("enabled", True)),
            execution_horizon=horizon_raw,
            max_guidance_weight=float(weight_raw),
            prefix_attention_schedule=schedule,
            debug=bool(raw.get("debug", False)),
        )
    except ValueError as exc:  # RTCConfig.__post_init__ validation
        raise ROSConfigError(f"{adapter_name}: invalid policy_extras.rtc: {exc}") from exc


def _rtc_enabled_in_extra(spec_extra: dict[str, Any], *, adapter_name: str) -> bool:
    """Whether ``policy_extras`` carries an *enabled* ``rtc`` block.

    For adapter factories that must decide something before the executor exists —
    smolvla skips ``maybe_compile_chunk_forward`` on this, since RTC and
    ``torch.compile`` rewrite the same flow-matching forward. Keyed on the parsed
    ``enabled`` flag rather than the block's presence, so ``rtc: {enabled: false}``
    still gets compiled.

    Args:
        spec_extra: The ``VLASpec.extra`` dict (manifest ``policy_extras``).
        adapter_name: Adapter label, for the error messages.

    Returns:
        True only for a present, well-formed, enabled ``rtc`` block.

    Raises:
        ROSConfigError: The ``rtc`` block is malformed — a bad manifest fails here
            rather than surviving to the later parse in
            :func:`build_chunk_executor`; the error is identical either way.
    """
    cfg = _parse_rtc_config(spec_extra, adapter_name=adapter_name)
    return cfg is not None and bool(cfg.enabled)


def build_chunk_executor(
    spec_extra: dict[str, Any],
    *,
    policy: Any = None,
    chunk_fn: Callable[[Any], Any] | None = None,
    chunk_size: int | None = None,
    adapter_name: str = "policy",
) -> ChunkedExecutor | None:
    """Build + start a :class:`ChunkedExecutor` for a chunked adapter.

    Pop ticks come from the executor-owned buffer, so no observation batch is
    built and no policy call runs. ``chunk_prefetch`` enables background
    inference; ``chunk_prefetch_at`` tunes its lead in actions (default 15).

    NOT for policies whose ``select_action`` consumes the observation on
    every call (Diffusion Policy keeps ``n_obs_steps`` of observation
    history) — those cannot use a chunk buffer at all and must keep the
    plain per-tick path.

    A ``policy_extras.rtc`` block (see :func:`_parse_rtc_config`) additionally
    installs the policy's lerobot ``RTCProcessor`` — ``config.rtc_config`` is
    set and ``init_rtc_processor()`` called *before* the executor is built —
    and hands the same ``RTCConfig`` to the executor so it serves actions from
    an ``ActionQueue``. RTC needs a real overlap between chunks, so it refuses
    single-step policies and ``chunk_prefetch: false``.

    Args:
        spec_extra: The ``VLASpec.extra`` dict.
        policy: lerobot-style policy (default chunk producer + reset target).
        chunk_fn: Custom chunk producer for adapters whose forward is not a
            bare ``predict_action_chunk`` — see :class:`ChunkedExecutor`.
        chunk_size: Actions consumed per inference; defaults to
            ``policy.config.n_action_steps``.
        adapter_name: Label for the enable log line.

    Returns:
        A started executor. Returns ``None`` for single-step lerobot policies;
        custom producers still get a synchronous one-action buffer so their
        output contract is checked.

    Raises:
        ROSConfigError: Non-positive chunk size, malformed ``chunk_prefetch`` /
            ``chunk_prefetch_at``, an invalid ``rtc`` block, or an ``rtc`` block
            enabled on a policy that cannot run it (chunk size 1, no
            pre-fetch, no ``init_rtc_processor``, bitsandbytes-quantized).
    """
    from openral_rskill.executor import ChunkedExecutor  # deferred: avoids import cycle

    n = int(
        chunk_size
        if chunk_size is not None
        else getattr(getattr(policy, "config", None), "n_action_steps", 1) or 1
    )
    if n < 1:
        raise ROSConfigError(f"{adapter_name}: chunk size must be positive, got {n}")
    rtc_cfg = _parse_rtc_config(spec_extra, adapter_name=adapter_name)
    rtc_on = rtc_cfg is not None and bool(rtc_cfg.enabled)
    if n == 1:
        # Never let an enabled rtc block fall through the single-step return below:
        # RTC blends a *previous chunk's* tail, which one-action inference never has.
        if rtc_on:
            raise ROSConfigError(
                f"{adapter_name}: policy_extras.rtc requires chunked execution, but this "
                "policy emits a single action per inference (chunk size 1)"
            )
        if chunk_fn is None:
            return None
    prefetch = spec_extra.get("chunk_prefetch", False)
    if not isinstance(prefetch, bool):
        raise ROSConfigError(f"{adapter_name}: policy_extras.chunk_prefetch must be a boolean")
    prefetch_at = 0
    if prefetch and n > 1:
        raw_prefetch_at = spec_extra.get("chunk_prefetch_at", 15)
        if isinstance(raw_prefetch_at, bool) or not isinstance(raw_prefetch_at, int):
            raise ROSConfigError(
                f"{adapter_name}: policy_extras.chunk_prefetch_at must be an integer"
            )
        if raw_prefetch_at < 1:
            raise ROSConfigError(
                f"{adapter_name}: policy_extras.chunk_prefetch_at must be positive"
            )
        prefetch_at = min(raw_prefetch_at, n - 1)
    if rtc_on:
        if prefetch_at < 1:
            raise ROSConfigError(
                f"{adapter_name}: policy_extras.rtc requires chunk_prefetch: true — RTC "
                "blends the prefetched chunk with the executing one"
            )
        if policy is None or not hasattr(policy, "init_rtc_processor"):
            raise ROSConfigError(
                f"{adapter_name}: this policy does not expose init_rtc_processor; "
                "RTC needs a lerobot flow-matching policy"
            )
        if _has_bnb_quantized_modules(policy):
            raise ROSConfigError(
                f"{adapter_name}: RTC guidance backpropagates through the denoiser each "
                "step; bitsandbytes-quantized weights (nf4/int8) are not supported"
            )
        policy.config.rtc_config = rtc_cfg
        policy.init_rtc_processor()
    executor = ChunkedExecutor(
        policy,
        chunk_fn=chunk_fn,
        chunk_size=n,
        prefetch_at=prefetch_at,
        rtc_config=rtc_cfg,
    )
    executor.start()
    log = structlog.get_logger("openral_rskill._vla_core")
    log.info(
        "vla.chunk_executor_enabled",
        adapter=adapter_name,
        n_action_steps=n,
        prefetch=prefetch,
        prefetch_at=prefetch_at,
        rtc=rtc_on,
    )
    return executor


def to_numpy_action(action_tensor: Any) -> NDArray[np.float32]:
    """Squeeze a single-batch action tensor to a 1-D float32 NumPy array.

    The eval ``PolicyAdapter.step`` contract requires a flat per-step action
    of length ``action_dim``; lerobot policies emit ``(1, action_dim)``.

    Args:
        action_tensor: Torch tensor of shape ``(1, action_dim)``.

    Returns:
        1-D ``float32`` NumPy array of length ``action_dim``.
    """
    out: NDArray[np.float32] = action_tensor.squeeze(0).detach().cpu().numpy().astype(np.float32)
    return out


def _hf_download_cached_first(
    hf_hub_download: Any,
    local_not_found_exc: type[BaseException],
    *,
    repo_id: str,
    filename: str,
    revision: str | None = None,
    **extra: Any,
) -> str:
    """Resolve an HF Hub file via the cache first, fall back to the Hub.

    Every ``hf_hub_download(...)`` call without ``local_files_only=True``
    HEAD-validates the cached file against the Hub, even when the file
    is already on disk. On a cold TLS connection that HEAD is 0.5 - 3 s
    per call; with N processor + state_file URIs the cost stacks into
    the visible portion of a 90 s policy load.

    This helper tries ``local_files_only=True`` first. On cache hit
    (the common case for a robot bring-up against a known checkpoint)
    no network call happens at all. On miss — manifest pins a new
    revision, cache was cleared, first download — falls back to the
    normal call so behaviour is unchanged when no cached file exists.

    Set ``HF_HUB_OFFLINE=1`` to force offline mode for every HF call in
    the process, including those inside ``Policy.from_pretrained``
    (which this helper does not wrap). That env-var is the broader knob
    when even the inner lerobot / transformers cache validation is the
    bottleneck.

    Args:
        hf_hub_download: The imported ``huggingface_hub.hf_hub_download``
            function. Injected to avoid an import in every caller.
        local_not_found_exc: ``huggingface_hub.errors.LocalEntryNotFoundError``.
            Same injection rationale.
        repo_id: HF Hub repo id.
        filename: File path within the repo.
        revision: Optional git revision to pin.
        **extra: Forwarded verbatim to both ``hf_hub_download`` calls.

    Returns:
        Absolute local path to the (now cached) file.
    """
    try:
        return str(
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                revision=revision,
                local_files_only=True,
                **extra,
            )
        )
    except local_not_found_exc:
        return str(
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                revision=revision,
                **extra,
            )
        )


def parse_hf_file_uri(uri: str) -> tuple[str, str | None, str]:
    """Split an ``hf://owner/repo[@rev]/path/to/file`` URI into its parts.

    Used by :func:`materialize_processor_dir` to drive per-file
    ``huggingface_hub.hf_hub_download`` calls from a
    :class:`openral_core.RSkillProcessors` URI. Closes Gap 1 + Gap 3 of
    the rSkill self-containment audit: the adapter no longer needs to
    ``snapshot_download(repo_id)`` and trust that the artefact happens to
    live at a particular filename.

    Args:
        uri: URI of the form ``hf://owner/repo[@rev]/path/to/file.ext``.
            The trailing ``path/to/file.ext`` is required (a bare
            ``hf://owner/repo`` is the implicit-snapshot shape that the
            schema rejects).

    Returns:
        ``(repo_id, revision, filename)`` tuple. ``revision`` is ``None``
        when the URI did not include an ``@<rev>`` segment.

    Raises:
        ROSConfigError: The URI does not start with ``hf://`` or is missing
            a file tail.

    Example:
        >>> parse_hf_file_uri("hf://lerobot/smolvla_base/policy_preprocessor.json")
        ('lerobot/smolvla_base', None, 'policy_preprocessor.json')
        >>> parse_hf_file_uri("hf://lerobot/smolvla_base@abc123/a/b/c.json")
        ('lerobot/smolvla_base', 'abc123', 'a/b/c.json')
    """
    if not uri.startswith("hf://"):
        raise ROSConfigError(f"parse_hf_file_uri only accepts hf:// URIs, got {uri!r}.")
    body = uri[len("hf://") :]
    # owner / repo[@rev] / path/to/file — three '/'-separated segments minimum.
    parts = body.split("/", 2)
    expected_segments = 3
    if len(parts) < expected_segments:
        raise ROSConfigError(
            f"hf:// URI {uri!r} is missing a file tail "
            "(expected hf://owner/repo[@rev]/path/to/file.ext)."
        )
    owner, repo_with_rev, filename = parts[0], parts[1], parts[2]
    if "@" in repo_with_rev:
        repo, revision = repo_with_rev.split("@", 1)
    else:
        repo, revision = repo_with_rev, None
    repo_id = f"{owner}/{repo}"
    return repo_id, revision, filename


def materialize_processor_dir(manifest: RSkillManifest) -> str:
    """Download the manifest's per-file processor artefacts into a single directory.

    Closes Gap 1 + Gap 3 of the rSkill self-containment audit. Replaces
    the implicit ``snapshot_download(repo_id)`` path with two explicit
    :func:`huggingface_hub.hf_hub_download` calls driven by
    ``manifest.processors``. The downloads are then symlinked under the
    fixed names ``policy_preprocessor.json`` /
    ``policy_postprocessor.json`` that
    :func:`lerobot.policies.factory.make_pre_post_processors` reads when
    given a ``pretrained_path``.

    Single seam — both the SmolVLA and the modern-ACT adapter call this
    helper, so the URI-driven path is exercised uniformly.

    Args:
        manifest: An rSkill manifest. ``manifest.processors`` MUST be set.

    Returns:
        Absolute path to a directory containing
        ``policy_preprocessor.json`` and ``policy_postprocessor.json``
        symlinks pointing at the downloaded files.

    Raises:
        ROSConfigError: The manifest has no ``processors`` block, or
            ``huggingface_hub`` is not installed.
    """
    if manifest.processors is None:
        raise ROSConfigError(
            f"materialize_processor_dir({manifest.name!r}) called but the "
            "manifest has no `processors` block. Only the legacy ACT path "
            "(model_family=act with norm stats inside model.safetensors) "
            "may omit it; that path does not call this helper."
        )
    try:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import LocalEntryNotFoundError
    except ImportError as exc:
        raise ROSConfigError(
            "materialize_processor_dir requires 'huggingface_hub'. "
            "Install it: uv add huggingface_hub --package openral-rskill"
        ) from exc

    import json
    import os
    import tempfile

    pre_repo, pre_rev, pre_file = parse_hf_file_uri(manifest.processors.preprocessor_uri)
    post_repo, post_rev, post_file = parse_hf_file_uri(manifest.processors.postprocessor_uri)

    pre_path = _hf_download_cached_first(
        hf_hub_download,
        LocalEntryNotFoundError,
        repo_id=pre_repo,
        filename=pre_file,
        revision=pre_rev,
    )
    post_path = _hf_download_cached_first(
        hf_hub_download,
        LocalEntryNotFoundError,
        repo_id=post_repo,
        filename=post_file,
        revision=post_rev,
    )

    # lerobot's PolicyProcessorPipeline.from_pretrained(<dir>) reads two top-level
    # JSON configs and then walks `steps[*]` for any entry that carries a
    # `state_file` key (a sibling .safetensors blob holding normalizer stats,
    # tokenizer state, etc.). When `<dir>` does not contain a step's
    # state_file, lerobot falls back to `hf_hub_download(repo_id=<dir>, ...)`
    # which fails because <dir> is a local path, not a repo id. So we have to
    # materialize the referenced state files into the same staging dir.
    staging = tempfile.mkdtemp(prefix="openral-processors-")

    def _materialize(json_local_path: str, canonical_name: str, repo: str, rev: str | None) -> None:
        link = os.path.join(staging, canonical_name)
        os.symlink(json_local_path, link)
        with open(json_local_path) as f:
            data = json.load(f)
        for step in data.get("steps", []):
            state_file = step.get("state_file")
            if not state_file:
                continue
            state_local = _hf_download_cached_first(
                hf_hub_download,
                LocalEntryNotFoundError,
                repo_id=repo,
                filename=state_file,
                revision=rev,
            )
            os.symlink(state_local, os.path.join(staging, state_file))

    _materialize(pre_path, "policy_preprocessor.json", pre_repo, pre_rev)
    _materialize(post_path, "policy_postprocessor.json", post_repo, post_rev)
    return staging


def _read_tokenizer_repo_from_preprocessor(pretrained_path: str | None) -> str | None:
    """Return the ``tokenizer_name`` baked into a saved preprocessor JSON.

    Walks ``<pretrained_path>/policy_preprocessor.json`` for the
    ``tokenizer_processor`` step (lerobot ``ProcessorStepRegistry``
    name) and returns its ``config.tokenizer_name``. Returns ``None``
    when the file is absent, malformed, or carries no tokenizer step
    (ACT / Diffusion Policy preprocessors).
    """
    if pretrained_path is None:
        return None
    import json
    from pathlib import Path

    pre_json = Path(pretrained_path) / "policy_preprocessor.json"
    if not pre_json.exists():
        return None
    try:
        data = json.loads(pre_json.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    for step in data.get("steps", []):
        if not isinstance(step, dict):
            continue
        if step.get("registry_name") != "tokenizer_processor":
            continue
        name = step.get("config", {}).get("tokenizer_name")
        if isinstance(name, str) and name:
            return name
    return None


def _hf_tokenizer_is_cached(repo_id: str) -> bool:
    """Probe the local HF cache for ``<repo_id>/tokenizer_config.json``.

    ``tokenizer_config.json`` is the first file
    ``AutoTokenizer.from_pretrained`` resolves; if it is on disk the rest
    of the tokenizer family (vocab, special tokens, processor config)
    was downloaded alongside it on the initial pull.
    ``try_to_load_from_cache`` returns the cached path (``str``),
    ``None`` for "unknown", or the ``_CACHED_NO_EXIST`` sentinel for
    "known not to exist upstream" — only the ``str`` return means we
    have a real file. Returns ``False`` on any import error so the
    caller falls back to a normal (online) load.
    """
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return False
    cached = try_to_load_from_cache(repo_id=repo_id, filename="tokenizer_config.json")
    return isinstance(cached, str)


def call_make_processors_cached_first(
    make_pre_post_processors: Any,
    policy_config: Any,
    *,
    pretrained_path: str | None,
    **kwargs: Any,
) -> tuple[Any, Any]:
    """Call ``make_pre_post_processors`` with HF revalidation suppressed when warm.

    Lerobot's ``TokenizerProcessorStep.__post_init__`` unconditionally
    calls ``AutoTokenizer.from_pretrained(tokenizer_name)`` whenever a
    saved preprocessor is reloaded. Transformers then issues 5 HEAD /
    metadata round-trips to the Hub against ``tokenizer_name`` (typically
    ``google/paligemma-3b-pt-224`` for π0.5) on *every* load, even
    against a fully-cached tokenizer — a noticeable stall on cold TLS.

    This wrapper:

    1. Reads ``tokenizer_name`` out of the preprocessor JSON.
    2. Probes the local HF cache for its ``tokenizer_config.json``.
    3. If the file is present, flips
       ``huggingface_hub.constants.HF_HUB_OFFLINE`` to ``True`` for the
       duration of the inner call (``transformers.utils.hub.is_offline_mode``
       reads the same constant). The probe-then-flip turns 5 HEADs per
       reload into 0.

    Adapters whose preprocessor has no tokenizer step (ACT, Diffusion
    Policy) hit the ``return None`` early-out in
    :func:`_read_tokenizer_repo_from_preprocessor` and fall through to a
    plain passthrough call. Cold caches do the same — the inner load is
    free to talk to the Hub and warm the cache exactly once.

    Args:
        make_pre_post_processors: The lerobot factory function imported
            in the caller (``lerobot.policies.factory.make_pre_post_processors``).
            Injected to avoid an import at the wrapper level so opt-in
            install groups (``just sync --all-packages --group sim``) stay opt-in.
        policy_config: ``policy.config`` — the
            :class:`lerobot.configs.policies.PreTrainedConfig` instance.
        pretrained_path: Absolute path to the directory containing
            ``policy_preprocessor.json`` / ``policy_postprocessor.json``.
            Forwarded verbatim to the inner call; ``None`` is treated as
            "no preprocessor on disk" and skips the offline-mode probe.
        **kwargs: Forwarded verbatim to ``make_pre_post_processors``
            (e.g. ``preprocessor_overrides``, ``dataset_stats``).

    Returns:
        ``(preprocessor, postprocessor)`` — whatever lerobot returns.
    """
    tokenizer_repo = _read_tokenizer_repo_from_preprocessor(pretrained_path)
    if tokenizer_repo is None or not _hf_tokenizer_is_cached(tokenizer_repo):
        result: tuple[Any, Any] = make_pre_post_processors(
            policy_config, pretrained_path=pretrained_path, **kwargs
        )
        return result

    import huggingface_hub.constants as _hc

    saved = _hc.HF_HUB_OFFLINE
    _hc.HF_HUB_OFFLINE = True
    try:
        result = make_pre_post_processors(policy_config, pretrained_path=pretrained_path, **kwargs)
        return result
    finally:
        _hc.HF_HUB_OFFLINE = saved


__all__ = [
    "InferenceKind",
    "apply_chunk_replay",
    "call_make_processors_cached_first",
    "materialize_processor_dir",
    "maybe_compile_chunk_forward",
    "parse_hf_file_uri",
    "release_torch_modules",
    "resolve_camera_keys",
    "resolve_device",
    "resolve_image_preprocessing",
    "resolve_rskill_repo_id",
    "resolve_rskill_repo_revision",
    "resolve_state_dim",
    "run_inference",
    "to_numpy_action",
    "warm_up_lerobot_policy",
]


def warm_up_lerobot_policy(adapter: object, *, prompt: str = "", torch: Any = None) -> bool:
    """Run one dummy forward so the first *real* tick doesn't blow its deadline.

    The first inference on a CUDA policy pays cuDNN autotune, kernel JIT and
    lazy-module materialisation. Measured on an RTX 4070 with the ACT
    so101-pen checkpoint (resnet18 + transformer, two 480x640 cameras)::

        call 1   330.4 ms      <- 10x the 33.3 ms budget at 30 Hz
        call 2+   14.9 ms

    Charged to tick 1 that is a guaranteed deadline miss, and under
    ``DeadlineOverrunPolicy.DROP`` the robot's first commanded action is
    discarded. Paying it during ``activate()`` instead costs the same
    wall-clock but lands where an operator is already waiting.

    Shapes come from the policy's own ``config``
    (``image_features[k].shape``, ``input_features["observation.state"]``),
    so the warm-up exercises the exact kernels the real ticks will — a
    guessed resolution would autotune the wrong ones and waste the pass.

    Best-effort and non-fatal by contract: a policy this cannot introspect
    is skipped, and any failure is swallowed. A warm-up is an optimisation;
    it must never be the reason a skill fails to activate.

    Args:
        adapter: The policy adapter. Must expose ``_policy`` holding a
            lerobot policy (all six in-tree lerobot families do); anything
            else returns ``False`` unchanged.
        prompt: Task string for language-conditioned families.
        torch: The caller's torch module; imported on demand when omitted.

    Returns:
        ``True`` when a dummy forward actually ran, ``False`` when skipped.
    """
    policy = getattr(adapter, "_policy", None)
    config = getattr(policy, "config", None)
    if policy is None or config is None:
        return False
    image_features = getattr(config, "image_features", None)
    input_features = getattr(config, "input_features", None)
    if not image_features or not input_features:
        return False

    if torch is None:
        import torch as torch_mod

        torch = torch_mod

    device = str(getattr(adapter, "device", "") or "cpu")
    state_feature = input_features.get("observation.state")
    if state_feature is None:
        return False
    # SmolVLA casts images to a non-default dtype; warming in the wrong one
    # autotunes kernels the real path will not use.
    image_dtype = getattr(adapter, "_image_dtype", None) or torch.float32

    batch: dict[str, Any] = {
        "observation.state": torch.zeros(
            1, int(state_feature.shape[0]), dtype=torch.float32, device=device
        ),
        "task": [prompt],
    }
    for key, feature in image_features.items():
        channels, height, width = (int(v) for v in feature.shape)
        batch[key] = torch.zeros(1, channels, height, width, dtype=image_dtype, device=device)

    # Run the batch through the adapter's own preprocessor first, exactly as
    # `step()` does. Skipping it warms the wrong thing — or nothing at all:
    # SmolVLA's `select_action` reads `observation.language.tokens`, which the
    # preprocessor produces by tokenising `task`, so a raw batch raises
    # KeyError and the warm-up is silently lost. Found on a live deploy sim;
    # the guard downgraded it to a warning, which is exactly why it went
    # unnoticed until the log was read.
    preprocessor = getattr(adapter, "_preprocessor", None)
    if callable(preprocessor):
        batch = preprocessor(batch)

    with contextlib.suppress(AttributeError, TypeError):
        policy.reset()
    with torch.no_grad():
        policy.select_action(batch)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    return True


def release_torch_modules(owner: object, *attrs: str, device: str = "", torch: Any = None) -> None:
    """Drop references to loaded torch modules, then reclaim their VRAM.

    **Order is the whole point.** ``torch.cuda.empty_cache()`` returns
    *already-free* cached blocks to the driver; it cannot free memory the
    allocator still considers live. So calling it while the adapter still
    holds the policy frees exactly nothing. Measured on an RTX 4070 with a
    768 MiB module resident::

        after load                       768.2 MiB allocated
        empty_cache() alone              768.2 MiB allocated   <- unchanged
        drop the reference, then flush     0.0 MiB allocated   <- reclaimed

    Every VLA adapter's ``close()`` used to do the second thing, which is
    why an rSkill swap did not actually give the card back and a second
    skill OOM'd on an 8 GB machine even though each fits alone.

    ``gc.collect()`` is not optional here either: a policy is typically part
    of a reference cycle (module ↔ parameters ↔ hooks), so dropping the last
    named reference does not necessarily run its finaliser on the spot.

    Best-effort by contract — teardown must always reach the code behind it,
    so a missing attribute or a torch that will not import is swallowed.

    Args:
        owner: The adapter holding the modules.
        *attrs: Attribute names to clear (e.g. ``"_policy"``, ``"_processor"``).
        device: The adapter's device string. The cache flush is skipped
            unless it names CUDA; dropping the references still happens.
        torch: The caller's already-imported torch module. Adapters pass
            their own ``self._torch`` handle — the same one they use
            everywhere else — rather than have this helper re-import it.

    Example:
        >>> class _Adapter:
        ...     def __init__(self) -> None:
        ...         self._policy = object()
        >>> a = _Adapter()
        >>> release_torch_modules(a, "_policy", device="cpu")
        >>> a._policy is None
        True
    """
    for attr in attrs:
        with contextlib.suppress(AttributeError):
            setattr(owner, attr, None)
    with contextlib.suppress(Exception):
        gc.collect()
        if device.startswith("cuda"):
            if torch is None:
                import torch as torch_mod

                torch = torch_mod
            torch.cuda.empty_cache()


@contextmanager
def suppress_hf_weight_init() -> Iterator[None]:
    """Skip HF's random weight init while a checkpoint is being loaded.

    ``transformers`` fills every parameter with its per-module init
    distribution at construction time, then ``from_pretrained`` immediately
    overwrites all of it with the stored tensors — the init is pure waste. HF
    knows this and skips it internally, but only on its own
    ``from_pretrained`` path. lerobot's SmolVLA builds the backbone by calling
    the model class *directly*
    (``SmolVLMForConditionalGeneration(config=...)`` in
    ``smolvlm_with_expert.py``, taken whenever the checkpoint sets
    ``load_vlm_weights=False`` — which every SmolVLA finetune does), so it pays
    the full init. Worse, it then truncates the text stack to
    ``num_vlm_layers`` and throws half those freshly-initialised layers away.

    Measured on the SO-101 eraser-place checkpoint (SmolVLM2-500M backbone,
    507 M params built, 16 of 32 layers kept):

    ==========================================  ========
    construction                                  wall
    ==========================================  ========
    baseline (allocate + init)                   8.45 s
    with this context (allocate only)            2.38 s
    meta device (allocate nothing)               0.03 s
    ==========================================  ========

    i.e. ~6 s of the ~15 s cold load is init math for values nothing reads.

    The remaining time is allocation. Reclaiming it needs a meta-device build
    plus an assign-mode state-dict load — a deeper change into lerobot's
    construction path, deliberately not attempted here. **Measured ceiling on
    an RTX 4070 host: ~1.6 s** (508 M params in transformer-shaped blocks —
    1.79 s allocated on CPU vs 0.21 s under ``accelerate.init_empty_weights``),
    so the win is smaller than the 2.4 s this note used to imply. Against a
    SmolVLA load that is ~10 s in-graph that is 15-20%, bought by taking
    ownership of construction code lerobot owns and re-validating it on every
    lerobot bump. π0.5 does take that path (``pi05.py``) because there the same
    change is worth 157 s → 14 s on a 3.4 B model; at 500 M it is not.

    Safety: only sound when the checkpoint supplies **every** parameter —
    otherwise a param that would have been randomly initialised is left as
    whatever ``malloc`` returned. Callers must therefore validate the loaded
    model; :func:`assert_all_parameters_finite` is the guard used by the
    SmolVLA adapter.

    Process-global for the duration (it patches the ``PreTrainedModel``
    class), so it must not wrap a block that loads models on several threads
    at once. The skill runner serialises loads behind its resident-skill lock,
    which is the only in-process caller.

    Yields:
        Nothing; the caller's construction runs with init suppressed.

    Example:
        >>> from openral_rskill._vla_core import suppress_hf_weight_init
        >>> with suppress_hf_weight_init():
        ...     pass  # SmolVLAPolicy.from_pretrained(...) goes here
    """
    try:
        from transformers.modeling_utils import PreTrainedModel
    except ImportError:  # transformers is an opt-in extra
        yield
        return

    original = PreTrainedModel._init_weights  # reason: documented monkeypatch

    def _skip(self: Any, module: Any) -> None:  # reason: matches HF signature
        return

    PreTrainedModel._init_weights = _skip  # type: ignore[method-assign]  # reason: restored in finally
    try:
        yield
    finally:
        PreTrainedModel._init_weights = original  # type: ignore[method-assign]  # reason: restore


def assert_all_parameters_finite(
    policy: Any, *, repo_id: str
) -> None:  # reason: torch.nn.Module without importing torch here
    """Raise if any parameter is NaN/Inf — the guard for suppressed init.

    Uninitialised memory read as float is overwhelmingly NaN or a wild
    magnitude, so this catches a checkpoint that failed to cover the graph
    while :func:`suppress_hf_weight_init` was active. Without the check a
    partially-loaded policy would silently emit garbage actions; the safety
    kernel would clamp them, but the robot would still move wrongly.

    Freshly-mapped pages are the other uninitialised state: all ZEROS, which
    are finite and would sail through the isfinite check. A trained weight
    *matrix* (ndim >= 2) is never exactly all-zero, so those are rejected
    too; 1-D parameters (biases, norms) are exempt — zero-init biases are
    legitimate and common.

    Args:
        policy: The loaded ``torch.nn.Module``.
        repo_id: Checkpoint id, for the error message.

    Raises:
        ROSConfigError: If any floating-point parameter is non-finite, or a
            floating-point weight matrix is entirely zero.
    """
    import torch  # reason: torch is an opt-in extra

    # 1-D parameters (biases, norms) are legitimately zero-init; only
    # matrices/convs (>= this many dims) are held to the all-zero check.
    matrix_min_dims = 2
    bad: list[str] = []
    zeroed: list[str] = []
    for name, tensor in policy.named_parameters():
        if not tensor.is_floating_point():
            continue
        if not torch.isfinite(tensor).all():
            bad.append(name)
        elif tensor.dim() >= matrix_min_dims and tensor.numel() > 0 and not tensor.count_nonzero():
            zeroed.append(name)
    if bad:
        raise ROSConfigError(
            f"{repo_id!r}: {len(bad)} parameter tensor(s) are NaN/Inf after load "
            f"(first: {bad[0]!r}). The checkpoint does not cover the whole graph, "
            "so weight-init suppression left them uninitialised."
        )
    if zeroed:
        raise ROSConfigError(
            f"{repo_id!r}: {len(zeroed)} weight matrix(es) are entirely zero after "
            f"load (first: {zeroed[0]!r}). A trained checkpoint never ships an "
            "all-zero weight matrix — a non-strict load most likely skipped the "
            "key (renamed after a dependency bump?) and weight-init suppression "
            "left the tensor on fresh zero pages."
        )
