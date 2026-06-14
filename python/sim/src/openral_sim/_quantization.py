"""Shared policy quantization helpers for openral_sim policy adapters.

Pure-PyTorch / bitsandbytes utilities, extracted from the pi0.5
adapter so any future policy adapter (smolvla, xvla, pi0.6, ...) can
reuse the same quantization rules + fast-path loader without
copy-pasting:

* :func:`quantize_nf4_in_place` -- walk a policy's module tree and
  rewrite every large ``torch.nn.Linear`` into a ``bnb.nn.Linear4bit``.
  Defers the actual nf4 pack to the next ``.to(device)`` call, just
  like bitsandbytes does when used via ``BitsAndBytesConfig``.
* :func:`quantize_int8_in_place` -- sibling rewrite that swaps the
  same large Linears for ``bnb.nn.Linear8bitLt`` (LLM.int8 mixed
  decomposition, ~50% the bf16 footprint, lossless on most attention
  workloads). Used when the rSkill manifest declares ``dtype: int8``.
  bitsandbytes only offers 4-bit and 8-bit Linear classes — there is
  no ``nf8``; ``int8`` here means LLM.int8, not torchao dynamic int8.
* :func:`install_prequantized_linears` -- overlay a state dict produced
  by ``tools/quantize_rskill.py`` directly onto already-rewritten
  Linear4bit modules. Uses ``Params4bit.from_prequantized`` to avoid
  the ~30 s on-line bf16->nf4 conversion the standard ``.to(cuda)``
  path triggers.
* :func:`load_prequantized_state_for_rskill` -- combined entry point:
  read the rSkill manifest, probe the Hub for a
  ``quantization_metadata.json`` sentinel, download the prequantized
  weights, and call :func:`install_prequantized_linears`. The function
  is a silent no-op when the rSkill ships bf16 weights, so adapters
  can call it unconditionally after their own ``quantize_nf4_in_place``.
  Only nf4 prequant packs are recognised today; int8 always runs the
  on-line ``bnb.nn.Linear8bitLt`` rewrite.

These helpers are deliberately framework-agnostic. They don't know
what a pi0.5 / smolvla / xvla *is* -- they only know that the policy
is an ``nn.Module`` whose interesting Linears live ``>=4M elements``
deep in the tree. The matching upload-side script lives at
``tools/quantize_rskill.py``.
"""

from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog
from openral_core.exceptions import ROSConfigError

if TYPE_CHECKING:
    from openral_core import VLASpec


log = structlog.get_logger(__name__)


# ── nf4 quantization rule ─────────────────────────────────────────────────────


DEFAULT_MIN_PARAMS_TO_QUANTIZE = 4_000_000
"""Per-Linear threshold for the nf4 rewrite.

Linears with fewer than this many weight elements are *not* swapped to
``Linear4bit`` -- they stay in the policy's compute dtype. 4M is the
PaliGemma / SmolVLA paper threshold: keeps the small projection
heads / LM-output heads in higher precision (where quantization noise
matters more), quantizes the bulk transformer blocks (where the
parameter count justifies the precision drop).
"""


def quantize_nf4_in_place(
    policy: Any,
    *,
    torch: Any,
    compute_dtype: Any,
    min_params: int = DEFAULT_MIN_PARAMS_TO_QUANTIZE,
    new_modules_on_meta: bool = False,
) -> None:
    """In-place rewrite of ``torch.nn.Linear`` modules to NF4 ``Linear4bit``.

    Only modules whose ``weight.numel() >= min_params`` are rewritten;
    smaller projection / norm / output heads stay in ``compute_dtype``
    for numerical safety. The bf16 weights are cloned into a
    ``Params4bit`` placeholder; bitsandbytes packs them into nf4 on the
    next ``policy.to(device=<cuda>)`` call.

    Args:
        policy: Root ``nn.Module`` of the policy.
        torch: Imported ``torch`` module (deferred-import pattern so
            ``openral_sim`` install doesn't pull torch).
        compute_dtype: Target dtype for bnb's de-quantized matmul output
            and for the small Linear bias terms.
        min_params: Per-Linear weight-element threshold. Defaults to
            :data:`DEFAULT_MIN_PARAMS_TO_QUANTIZE`.
        new_modules_on_meta: When True, wrap the module-replacement walk
            in :func:`accelerate.init_empty_weights` so each new
            ``bnb.nn.Linear4bit`` is constructed on the ``meta`` device.
            Use this when the caller is going to ``to_empty(device=...)``
            the resulting tree (and overwrite every weight via a
            prequant state load) — without it, the bnb constructor
            allocates a full bf16 placeholder per module that gets
            thrown away seconds later. Measured saving on a 3.4 B-param
            π0.5 backbone: ~5–10 s of CPU allocation per load.

    Raises:
        ROSConfigError: If ``bitsandbytes`` is not installed.
    """
    try:
        import bitsandbytes as bnb
    except ImportError as exc:  # pragma: no cover
        raise ROSConfigError(
            "nf4 quantization requires bitsandbytes; install with: "
            "uv pip install 'bitsandbytes>=0.45'"
        ) from exc

    # Build the constructor context once. ``init_empty_weights`` is a
    # context manager that overrides ``nn.Module._register_parameter``
    # to redirect new params/buffers to the meta device — we MUST exit
    # it before reassigning ``new.weight`` to the source-data clone,
    # otherwise the override intercepts the assignment and discards
    # the real bf16 data (a silent corruption that the e2e prequant
    # load happens to mask on the nf4 path but would surface on any
    # caller whose source weights are already real). Narrowing the
    # context to just the bnb constructor preserves the placeholder-
    # allocation saving without touching the data path.
    if new_modules_on_meta:
        try:
            from accelerate import init_empty_weights  # type: ignore[import-untyped]
        except ImportError as exc:  # pragma: no cover
            raise ROSConfigError(
                "nf4 quantization on meta requires accelerate; install with: uv add accelerate"
            ) from exc
        ctor_ctx = init_empty_weights
    else:
        ctor_ctx = contextlib.nullcontext

    def _replace(module: Any, prefix: str = "") -> None:
        for name, child in list(module.named_children()):
            full = f"{prefix}.{name}" if prefix else name
            if isinstance(child, torch.nn.Linear) and child.weight.numel() >= min_params:
                with ctor_ctx():
                    new = bnb.nn.Linear4bit(
                        child.in_features,
                        child.out_features,
                        bias=child.bias is not None,
                        compute_dtype=compute_dtype,
                        quant_type="nf4",
                    )
                # Copy bf16/fp32 weights into the new module; bnb does
                # the actual nf4 pack the first time the module is
                # moved to CUDA.
                new.weight = bnb.nn.Params4bit(
                    child.weight.data.clone(),
                    requires_grad=False,
                    quant_type="nf4",
                )
                if child.bias is not None:
                    new.bias = torch.nn.Parameter(
                        child.bias.data.clone().to(compute_dtype),
                        requires_grad=False,
                    )
                setattr(module, name, new)
            else:
                _replace(child, full)

    _replace(policy)


# ── LLM.int8 (bnb.nn.Linear8bitLt) rewrite ────────────────────────────────────


def quantize_int8_in_place(
    policy: Any,
    *,
    torch: Any,
    compute_dtype: Any,
    min_params: int = DEFAULT_MIN_PARAMS_TO_QUANTIZE,
    threshold: float = 6.0,
    new_modules_on_meta: bool = False,
) -> None:
    """In-place rewrite of ``torch.nn.Linear`` modules to ``bnb.nn.Linear8bitLt``.

    Same selection rule as :func:`quantize_nf4_in_place`: modules whose
    ``weight.numel() >= min_params`` are rewritten, smaller projection
    heads stay in ``compute_dtype``. The bf16 weight is cloned into a
    ``bnb.nn.Int8Params`` placeholder; bitsandbytes packs it on the
    next ``policy.to(device=<cuda>)`` call.

    Unlike the nf4 path, there is no pre-quantized fast-path -- LLM.int8
    weights are not commonly shipped as a separate Hub artefact today
    (the SCB sub-state ownership inside ``Int8Params`` makes round-
    tripping through ``state_dict`` brittle). Callers that want a
    prequant cache should stick to nf4.

    Args:
        policy: Root ``nn.Module`` of the policy.
        torch: Imported ``torch`` module (deferred-import pattern so
            ``openral_sim`` install doesn't pull torch).
        compute_dtype: Target dtype for bnb's mixed-precision compute
            path. bf16 on CUDA matches what PaliGemma forward expects.
        min_params: Per-Linear weight-element threshold. Defaults to
            :data:`DEFAULT_MIN_PARAMS_TO_QUANTIZE`.
        threshold: ``Linear8bitLt``'s outlier-detection threshold. The
            LLM.int8 paper uses 6.0; lower values keep more of the
            activation matrix in fp16 (slower, slightly more accurate).
        new_modules_on_meta: When True, wrap the module-replacement walk
            in :func:`accelerate.init_empty_weights` so each new
            ``bnb.nn.Linear8bitLt`` is constructed on the ``meta``
            device. Mirrors the nf4 path's identically-named kwarg:
            without it, the bnb constructor allocates a full bf16
            placeholder per Linear (roughly doubling the peak CPU
            footprint during the rewrite since the source Linears
            already hold their real bf16 weights). The source
            ``child.weight.data.clone()`` carries the real data into
            the new module's ``Int8Params``, so the meta-shaped
            placeholder we skipped contributed nothing to the final
            state — and bitsandbytes still does its on-line int8 pack
            on the upcoming ``policy.to(<cuda>)`` because the cloned
            data lands in real CPU storage at the assignment.

    Raises:
        ROSConfigError: If ``bitsandbytes`` is not installed.
    """
    try:
        import bitsandbytes as bnb
    except ImportError as exc:  # pragma: no cover
        raise ROSConfigError(
            "int8 quantization requires bitsandbytes; install with: "
            "uv pip install 'bitsandbytes>=0.45'"
        ) from exc

    # See the matching commentary in :func:`quantize_nf4_in_place` for
    # why the meta context wraps only the bnb constructor and not the
    # subsequent ``new.weight = ...`` assignment — under
    # ``init_empty_weights`` that assignment would be redirected to
    # meta too, silently dropping the real bf16 source data on the
    # floor (the LLM.int8 path has no prequant safety net to recover
    # from that, unlike nf4).
    if new_modules_on_meta:
        try:
            from accelerate import init_empty_weights
        except ImportError as exc:  # pragma: no cover
            raise ROSConfigError(
                "int8 quantization on meta requires accelerate; install with: uv add accelerate"
            ) from exc
        ctor_ctx = init_empty_weights
    else:
        ctor_ctx = contextlib.nullcontext

    def _replace(module: Any, prefix: str = "") -> None:
        for name, child in list(module.named_children()):
            full = f"{prefix}.{name}" if prefix else name
            if isinstance(child, torch.nn.Linear) and child.weight.numel() >= min_params:
                with ctor_ctx():
                    new = bnb.nn.Linear8bitLt(
                        child.in_features,
                        child.out_features,
                        bias=child.bias is not None,
                        has_fp16_weights=False,
                        threshold=threshold,
                    )
                new.weight = bnb.nn.Int8Params(
                    child.weight.data.clone(),
                    requires_grad=False,
                    has_fp16_weights=False,
                )
                if child.bias is not None:
                    new.bias = torch.nn.Parameter(
                        child.bias.data.clone().to(compute_dtype),
                        requires_grad=False,
                    )
                setattr(module, name, new)
            else:
                _replace(child, full)

    _replace(policy)


# ── Prequantized state-dict overlay ───────────────────────────────────────────


_BNB_META_SUFFIXES = (
    ".absmax",
    ".quant_map",
    ".nested_absmax",
    ".nested_quant_map",
    ".quant_state.bitsandbytes__nf4",
    ".quant_state.bitsandbytes__fp4",
)


def install_prequantized_linears(
    policy: Any,
    state: dict[str, Any],
    *,
    device: str,
    torch: Any,
) -> tuple[int, set[str]]:
    """Replace each ``Linear4bit.weight`` with ``Params4bit.from_prequantized`` data.

    Walks the policy, finds every ``bnb.nn.Linear4bit`` module, looks
    up the matching prequantized weight + bnb metadata in ``state``,
    and rebuilds the module's ``weight`` via
    ``Params4bit.from_prequantized`` so the packed uint8 tensor goes
    straight to ``device`` without an intermediate bf16
    materialisation. This is the "fast-path" that replaces the ~30 s
    on-line bf16->nf4 conversion done by the standard ``.to(cuda)``
    path on Linear4bit modules carrying bf16 placeholder weights.

    Args:
        policy: The policy ``nn.Module`` tree, after
            :func:`quantize_nf4_in_place` has already swapped Linear
            modules for Linear4bit shells (with bf16 placeholder
            weights).
        state: The state dict loaded from the prequantized safetensors
            on disk. Keys are expected to follow the bnb serialisation
            shape: ``<prefix>.weight`` is the packed uint8 tensor and
            ``<prefix>.weight.<absmax|quant_map|...>`` are the bnb
            metadata siblings.
        device: Target device for the resulting Params4bit. Typically
            ``"cuda"`` on machines with a GPU.
        torch: Imported torch module (passed in for symmetry with the
            other helpers, currently unused inside the function).

    Returns:
        ``(n_quantized_modules, consumed_state_keys)`` -- the number
        of Linear4bit modules we rebuilt and the set of state-dict
        keys consumed. Callers drop the consumed keys from any
        residual ``load_state_dict`` overlay so PyTorch doesn't
        complain about missing slots on already-rebuilt modules.
    """
    del torch  # kept in the signature for parity with sibling helpers
    import bitsandbytes as bnb

    consumed: set[str] = set()
    quantized_count = 0

    for prefix, module in policy.named_modules():
        if not isinstance(module, bnb.nn.Linear4bit):
            continue
        weight_key = f"{prefix}.weight"
        if weight_key not in state:
            continue
        packed = state[weight_key]
        # Collect every sibling metadata tensor that bnb expects to see
        # inside QuantState.from_dict. The dict keys it expects don't
        # carry the leading dot (it's ``absmax``, not ``.absmax``).
        quantized_stats: dict[str, Any] = {}
        for suffix in _BNB_META_SUFFIXES:
            full = f"{weight_key}{suffix}"
            if full in state:
                quantized_stats[suffix.lstrip(".")] = state[full]
                consumed.add(full)
        consumed.add(weight_key)
        new_weight = bnb.nn.Params4bit.from_prequantized(
            data=packed,
            quantized_stats=quantized_stats,
            requires_grad=False,
            device=device,
        )
        module.weight = new_weight
        bias_key = f"{prefix}.bias"
        if module.bias is not None and bias_key in state:
            module.bias.data = state[bias_key].to(device).to(module.bias.dtype)
            consumed.add(bias_key)
        quantized_count += 1

    return quantized_count, consumed


# ── rSkill-level entry point ──────────────────────────────────────────────────


def detect_prequantized_nf4(spec: VLASpec) -> str | None:  # noqa: PLR0911  # reason: linear early-return chain is clearer than nesting
    """Probe the rSkill's HF repo for a ``quantization_metadata.json`` sentinel.

    Returns the resolved HF repo id (e.g.
    ``"OpenRAL/rskill-pi05-robocasa365-human300-nf4"``) when the
    repo carries a prequantized nf4 pack we can fast-load, ``None``
    otherwise. The check is a single ``HEAD`` request (~100 ms warm,
    cached after the first call).

    Adapters use this to decide whether to bypass
    ``PolicyClass.from_pretrained`` -- when a prequant pack is present
    the entire ``from_pretrained`` flow (fp32 graph allocation +
    safetensors load + 100s of size-mismatch warnings) is wasted work,
    because ``load_prequantized_state_for_rskill`` will overwrite
    every Linear weight + every residual key from the pack a few
    seconds later. See ADR notes / git blame on the pi05 adapter for
    why we keep `from_pretrained` as the slow fallback.
    """
    weights_uri = (spec.weights_uri or "").strip()
    if weights_uri.startswith(("hf://", "local://", "file://", "http://", "https://")):
        return None

    try:
        from openral_rskill.loader import load_rskill_manifest
    except ImportError:  # pragma: no cover
        return None
    try:
        manifest = load_rskill_manifest(weights_uri)
    except Exception:
        return None

    hub_uri = (manifest.weights_uri or "").strip()
    if not hub_uri.startswith("hf://"):
        return None
    target_repo = hub_uri[len("hf://") :]

    try:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import (
            EntryNotFoundError,
            LocalEntryNotFoundError,
            RepositoryNotFoundError,
        )
        from openral_rskill._vla_core import _hf_download_cached_first
    except ImportError:  # pragma: no cover
        return None

    try:
        meta_path = _hf_download_cached_first(
            hf_hub_download,
            LocalEntryNotFoundError,
            repo_id=target_repo,
            filename="quantization_metadata.json",
        )
    except (EntryNotFoundError, RepositoryNotFoundError):
        return None

    try:
        meta = json.loads(Path(meta_path).read_text())
    except (json.JSONDecodeError, OSError):
        return None
    if meta.get("quantization", {}).get("scheme") != "nf4":
        return None
    return target_repo


def load_prequantized_state_for_rskill(  # noqa: PLR0911  # reason: linear early-return chain is clearer than nesting
    policy: Any,
    spec: VLASpec,
    *,
    torch: Any,
    log_event_prefix: str = "rskill",
) -> None:
    """Optionally overlay a pre-quantized state dict onto a freshly nf4-rewritten policy.

    Pre-quantized rSkills (produced by ``tools/quantize_rskill.py``)
    ship a ``quantization_metadata.json`` alongside
    ``model.safetensors``. When the rSkill's HF repo carries that
    metadata file with ``scheme == "nf4"``, we download the matching
    safetensors and call :func:`install_prequantized_linears`. The
    Linear4bit modules built by :func:`quantize_nf4_in_place` get
    their weights replaced by the prequantized data; the implicit
    bf16->nf4 packing that ``.to(device)`` would otherwise run is
    skipped.

    Silently no-ops when the rSkill manifest does not point at an
    ``hf://`` repo, or when that repo does not carry a
    ``quantization_metadata.json``. Back-compat: rSkills that ship
    bf16 weights (e.g. ``rskills/pi05-libero-nf4``) keep their existing
    on-line nf4 pack path.

    Args:
        policy: Policy ``nn.Module`` already rewritten via
            :func:`quantize_nf4_in_place`.
        spec: The :class:`openral_core.VLASpec` carrying the
            bare rSkill reference. The rSkill manifest's
            ``weights_uri`` field (after resolution) must point at an
            ``hf://`` repo for this function to do anything.
        torch: Imported torch module.
        log_event_prefix: Per-adapter event-name prefix for structlog
            ("pi05" -> ``pi05_prequantized_fastpath`` etc.). Lets
            adapter-level traces distinguish which policy hit the
            fast-path.
    """
    weights_uri = spec.weights_uri or ""
    if weights_uri.startswith(("hf://", "local://", "file://", "http://", "https://")):
        return
    try:
        from openral_rskill.loader import load_rskill_manifest
    except ImportError:  # pragma: no cover -- openral_rskill always available
        return
    try:
        manifest = load_rskill_manifest(weights_uri)
    except Exception:
        # If the manifest doesn't resolve, the upstream loader has
        # already validated it earlier -- fall through to the standard
        # bf16->nf4 path the caller will run on the next .to(device).
        return

    hub_uri = (manifest.weights_uri or "").strip()
    if not hub_uri.startswith("hf://"):
        return
    target_repo = hub_uri[len("hf://") :]

    try:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import (
            EntryNotFoundError,
            LocalEntryNotFoundError,
            RepositoryNotFoundError,
        )
        from openral_rskill._vla_core import _hf_download_cached_first
    except ImportError:  # pragma: no cover
        return

    try:
        meta_path = _hf_download_cached_first(
            hf_hub_download,
            LocalEntryNotFoundError,
            repo_id=target_repo,
            filename="quantization_metadata.json",
        )
    except (EntryNotFoundError, RepositoryNotFoundError):
        return

    meta = json.loads(Path(meta_path).read_text())
    scheme = meta.get("quantization", {}).get("scheme")
    if scheme != "nf4":
        return

    log.info(
        f"{log_event_prefix}_prequantized_fastpath",
        repo=target_repo,
        source_repo=meta.get("source_repo"),
        source_revision=meta.get("source_revision"),
    )

    weights_path = _hf_download_cached_first(
        hf_hub_download,
        LocalEntryNotFoundError,
        repo_id=target_repo,
        filename="model.safetensors",
    )
    from safetensors.torch import load_file

    state = load_file(weights_path, device="cpu")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        loaded, skipped = install_prequantized_linears(policy, state, device=device, torch=torch)
    except Exception as exc:
        log.warning(
            f"{log_event_prefix}_prequantized_load_skipped",
            reason=str(exc).splitlines()[0][:200],
            note=(
                "Falling back to the standard bf16->nf4 path "
                "(`.to(cuda)` will re-pack the bf16 weights). The "
                "pre-quantized rSkill still loaded cleanly; only the "
                "fast state-dict overlay is disabled."
            ),
        )
        return

    leftover = {k: v for k, v in state.items() if k not in skipped}
    missing, unexpected = policy.load_state_dict(leftover, strict=False)
    log.info(
        f"{log_event_prefix}_prequantized_loaded",
        keys=len(state),
        quantized_modules=loaded,
        residual_keys=len(leftover),
        missing=len(missing),
        unexpected=len(unexpected),
    )


def peek_safetensors_keys(repo_id: str, *, filename: str = "model.safetensors") -> set[str] | None:
    """Return the key set of a safetensors file without loading tensors.

    Used by adapters that want to skip ``reset_parameters()`` on
    modules whose params are about to be overwritten by an upcoming
    state-dict load — on a 3.4 B-param model that walk dominates the
    ``to_empty_cpu`` phase even though every kaiming / normal / ones
    init it produces is thrown away seconds later.

    Reads only the safetensors header (~10 ms warm), so calling this
    eagerly during the build phase is cheap. Routes the file fetch
    through :func:`_hf_download_cached_first` so the
    ``local_files_only=True`` fast path applies. Works for both
    prequantized packs (caller passes the nf4 prequant repo id) and
    bare source checkpoints (caller passes the bf16 source repo id —
    used by the int8 fast meta-init path that loads bf16 weights via
    ``load_state_dict`` instead of going through lerobot's
    ``PI05Policy.from_pretrained``).

    Args:
        repo_id: HF Hub repo id carrying the safetensors file.
        filename: Path within the repo. Defaults to the standard
            single-file checkpoint layout (``model.safetensors``).

    Returns:
        The set of tensor keys present in ``filename``, or ``None``
        if the file cannot be downloaded / parsed (caller falls back
        to a full ``reset_parameters`` walk).
    """
    try:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import (
            EntryNotFoundError,
            LocalEntryNotFoundError,
            RepositoryNotFoundError,
        )
        from openral_rskill._vla_core import _hf_download_cached_first
        from safetensors import safe_open
    except ImportError:  # pragma: no cover
        return None

    try:
        weights_path = _hf_download_cached_first(
            hf_hub_download,
            LocalEntryNotFoundError,
            repo_id=repo_id,
            filename=filename,
        )
    except (EntryNotFoundError, RepositoryNotFoundError, OSError):
        return None

    try:
        # safetensors ships a stub for `safe_open` only as a class wrapper;
        # mypy --strict flags the call as untyped. The library is widely
        # used and stable; the ignore is the maintenance-cheap path.
        with safe_open(weights_path, framework="pt") as f:  # type: ignore[no-untyped-call]
            return set(f.keys())
    except (OSError, ValueError):
        return None


# ── Manifest-driven dtype resolution ──────────────────────────────────────────


def normalise_manifest_dtype(manifest: Any) -> str | None:
    """Pull ``manifest.quantization.dtype`` as a string, if present.

    The rSkill schema's ``QuantizationDtype`` enum values
    (``"int4"``, ``"int8"``, ``"bf16"``, ``"fp32"`` ...) already match
    the keys the adapter's dispatch checks; this helper just guards
    against a manifest without a quantization block (``None`` →
    :func:`default_dtype_for_device` kicks in at the call site).
    """
    quant = getattr(manifest, "quantization", None)
    if quant is None:
        return None
    dtype = getattr(quant, "dtype", None)
    if dtype is None:
        return None
    return str(getattr(dtype, "value", dtype))


def manifest_dtype(spec: Any, manifest: Any | None = None) -> str | None:
    """Return the dtype the adapter should load with, if any.

    Resolution order (first hit wins):

    1. ``spec.extra["dtype"]`` — a per-run override (``openral sim run
       --vla-extra dtype=int8`` or a programmatic VLASpec). Lets the
       operator pick a dtype without editing the rSkill manifest.
    2. ``manifest.quantization.dtype`` — the rSkill's pinned dtype,
       when an rSkill manifest is in hand. Mapped through
       :func:`normalise_manifest_dtype` so the enum value (``"int4"``,
       ``"int8"``, ``"bf16"`` ...) lands as a string the adapter's own
       dispatch already understands.

    Returns ``None`` when neither source supplies a dtype, leaving
    :func:`default_dtype_for_device` to pick a CUDA-aware default.
    """
    raw = spec.extra.get("dtype") if hasattr(spec, "extra") else None
    if raw:
        return str(raw)
    if manifest is not None:
        return normalise_manifest_dtype(manifest)
    return None


def torch_dtype_for(torch: Any, dtype_str: str | None, device: str) -> Any:
    """Map a manifest dtype string to a torch dtype, with a CUDA-aware default.

    Recognised dtype strings (case-insensitive): ``bf16`` / ``bfloat16``,
    ``fp16`` / ``float16`` / ``half``, ``fp32`` / ``float32``. Anything
    else falls through to the device-aware default (bf16 on CUDA, fp32
    elsewhere) so adapters can pass through ``"nf4"`` / ``"int8"`` and
    let this helper pick a sensible *compute* dtype for the leaves that
    won't be quantized.
    """
    if dtype_str:
        s = dtype_str.lower()
        if s in {"bf16", "bfloat16"}:
            return torch.bfloat16
        if s in {"fp16", "float16", "half"}:
            return torch.float16
        if s in {"fp32", "float32"}:
            return torch.float32
    # Default: bf16 on CUDA (large VLAs are too big for fp32 on consumer
    # GPUs), fp32 elsewhere.
    return torch.bfloat16 if device.startswith("cuda") else torch.float32


def default_dtype_for_device(device: str) -> str:
    """Pick a default load dtype when the manifest doesn't specify one.

    Large VLAs (π0.5 / RLDX-1) are >3 B params, so we default to 4-bit
    NF4 on CUDA (fits in ~4 GiB) and fp32 on CPU (where bf16 inference
    is far slower than fp32 and bnb is unavailable). Adapters that
    serve smaller models (SmolVLA at ~400 M) typically override the
    default by passing ``"bf16"`` directly.
    """
    return "nf4" if device.startswith("cuda") else "fp32"


def targeted_reset_parameters(policy: Any, *, covered_keys: set[str] | None) -> None:
    """Restore canonical init for modules whose params are NOT in ``covered_keys``.

    ``init_empty_weights`` builds the policy on the meta device, then
    ``to_empty(device="cpu")`` materialises real CPU storage with
    *uninitialised* memory. PyTorch's standard remedy is to walk every
    module and call its ``reset_parameters()`` so weights start from the
    same canonical values ``__init__`` would have produced (kaiming for
    Linear, ones for {LayerNorm,RMSNorm}, normal for Embedding, etc.). On a
    multi-billion-param model that walk is the single slowest CPU phase of
    the load — the init RNG over every parameter dominates wall-time.

    When the prequant state load is about to overwrite a module's params
    anyway, the reset is pure waste — the init values are discarded a few
    seconds later. ``covered_keys`` is the set of safetensors keys we know
    will land via :func:`install_prequantized_linears` + ``load_state_dict``.
    For any module whose immediate (non-recursive) parameter keys are a
    subset of ``covered_keys``, we skip reset. Modules carrying parameters
    that won't be filled fall through to the standard reset so their values
    stay canonical.

    Adapter-agnostic: it walks ``named_modules`` generically and makes no
    assumption about the policy architecture, so π0.5 / MolmoAct2 / future
    meta-init families share it.

    Args:
        policy: The meta-then-materialised policy ``nn.Module`` tree.
        covered_keys: Set of fully-qualified state-dict keys that will be
            overwritten by the upcoming prequant state load. Pass ``None``
            to fall back to the unconditional reset; the helper still works,
            it just doesn't save any time.
    """
    for prefix, module in policy.named_modules():
        reset = getattr(module, "reset_parameters", None)
        if not callable(reset):
            continue
        if covered_keys is not None:
            # Only inspect direct (non-recursive) params so we don't treat a
            # child module's coverage as this module's coverage. Linear4bit /
            # Linear8bitLt expose only a ``weight`` (and optional ``bias``)
            # parameter key locally; bnb metadata keys (``weight.absmax`` etc.)
            # live alongside in the state dict and are consumed by
            # ``Params4bit.from_prequantized`` / ``Int8Params.cuda`` without
            # needing a separate parameter slot here.
            own_param_keys = {
                f"{prefix}.{name}" if prefix else name
                for name, _ in module.named_parameters(recurse=False)
            }
            if not own_param_keys:
                # Container with no direct params (root policy, ModuleList,
                # etc.). Its inherited ``reset_parameters`` would recursively
                # re-init every child param — precisely the wasted walk this
                # helper skips. Children are visited in their own iterations.
                continue
            if own_param_keys.issubset(covered_keys):
                continue
        with contextlib.suppress(Exception):
            reset()


def tie_transformers_weights(policy: Any) -> None:
    """Best-effort call ``tie_weights`` on each outermost transformers backbone.

    Many VLA backbones tie their input ``embed_tokens`` to the output
    ``lm_head`` via the standard transformers ``tie_weights()`` convention,
    but the policy wrapper itself does not always expose a top-level
    ``tie_weights``. Calling it in pre-order (and skipping descendants of an
    already-tied module) keeps the tie state consistent so the missing tied
    slot does not eat a wasted ``normal_`` init in
    :func:`targeted_reset_parameters`. Failures are non-fatal (a meta-init
    expert backbone can raise ``embed_tokens is not an nn.Module``); the
    worst case is paying the reset cost for that one slot.
    """
    tied_prefixes: list[str] = []
    for name, module in policy.named_modules():
        if not (hasattr(module, "tie_weights") and callable(module.tie_weights)):
            continue
        # Skip descendants of an already-tied module — calling ``tie_weights``
        # on a child of an already-tied parent has been observed to undo the
        # parent's tie on some transformers versions.
        if any(name.startswith(p + ".") for p in tied_prefixes):
            continue
        try:
            module.tie_weights()
        except Exception:
            continue
        tied_prefixes.append(name)


__all__ = [
    "DEFAULT_MIN_PARAMS_TO_QUANTIZE",
    "default_dtype_for_device",
    "detect_prequantized_nf4",
    "install_prequantized_linears",
    "load_prequantized_state_for_rskill",
    "manifest_dtype",
    "normalise_manifest_dtype",
    "peek_safetensors_keys",
    "quantize_int8_in_place",
    "quantize_nf4_in_place",
    "targeted_reset_parameters",
    "tie_transformers_weights",
    "torch_dtype_for",
]


# Suppress mypy unused-import warnings on optional bnb imports.
_ = os
