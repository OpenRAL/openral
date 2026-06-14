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

from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import structlog
from numpy.typing import NDArray
from openral_core.exceptions import ROSConfigError
from openral_observability import inference_span

if TYPE_CHECKING:
    from openral_core import ImagePreprocessing, RSkillManifest, VLASpec

InferenceKind = Literal["foreground", "prefetch", "single"]


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

    Returns:
        The raw action tensor returned by ``policy.select_action``.
    """
    import torch

    # ``policy.device`` is the lerobot convention; fall back to None so the
    # span helper omits the attribute on adapters that don't track it.
    device = getattr(policy, "device", None)
    extras: dict[str, Any] = {"engine": engine if engine is not None else "torch"}
    if chunk_size is not None:
        extras["chunk_size"] = chunk_size
    if device is not None:
        extras["device"] = str(device)
    with (
        inference_span(chunk_index=chunk_index, kind=kind, **extras),
        torch.no_grad(),
    ):
        return policy.select_action(batch)


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
    "resolve_camera_keys",
    "resolve_device",
    "resolve_image_preprocessing",
    "resolve_rskill_repo_id",
    "resolve_rskill_repo_revision",
    "resolve_state_dim",
    "run_inference",
    "to_numpy_action",
]
