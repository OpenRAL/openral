"""Shared precision helpers for openral_sim policy adapters.

Open OpenRAL runs fp32, fp16 and bf16 (ADR-0123). This module keeps that
plumbing: the manifest-driven dtype resolver (``plan_quantization`` /
``resolve_quant_plan``, which refuse sub-16-bit without OpenRAL Pro), dtype
casting (``torch_dtype_for``, ``default_dtype_for_device``), and the
safetensors streaming and meta-init helpers full-precision loads use.

The bitsandbytes NF4 / LLM.int8 rewrites and the pre-quantized pack loaders
live in OpenRAL Pro's quantization module; adapters reach them through
``openral_rskill.quantization_gate.pro_quantization``.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import struct
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, NamedTuple

import structlog
from openral_core.exceptions import ROSConfigError
from openral_rskill.quantization_gate import require_pro_for_checkpoint, require_pro_for_dtype

log = structlog.get_logger(__name__)


WEIGHTS_FILE = "model.safetensors"
"""The single-file transformers / lerobot checkpoint layout."""

WEIGHTS_INDEX_FILE = "model.safetensors.index.json"
"""The sharded layout's index (``save_pretrained`` above its shard size): ``weight_map`` names
each tensor's shard file."""


def _shards_from_index(index_path: str) -> list[str]:
    """Shard file names listed in a ``model.safetensors.index.json``, in first-seen order."""
    weight_map = json.loads(Path(index_path).read_text()).get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ROSConfigError(f"{index_path}: no 'weight_map' to name the checkpoint shards")
    return list(dict.fromkeys(str(v) for v in weight_map.values()))


def resolve_weights_files(repo_id: str, *, revision: str | None = None) -> list[str]:
    """Local paths of the weights for ``repo_id``: one file, or every shard of a sharded one.

    ``repo_id`` is a *directory* for an rSkill installed with ``openral rskill install``
    (the snapshot ``resolve_rskill_to_hf_with_revision`` returns) and a bare Hub repo id
    otherwise; every weights read that went straight to ``hf_hub_download`` broke on the
    directory case, and this is the one place that tells the two apart. Both shapes
    accept ``model.safetensors`` or a sharded ``model.safetensors.index.json`` + the
    shards it names. A directory holding neither falls through to the cached-first Hub
    download. ``revision`` is the pinned Hub revision the adapter resolved for
    ``config.json``, forwarded so weights never come from a different revision.

    Example:
        >>> import tempfile, pathlib
        >>> d = pathlib.Path(tempfile.mkdtemp())
        >>> _ = (d / "model.safetensors").write_bytes(b"")
        >>> resolve_weights_files(str(d)) == [str(d / "model.safetensors")]
        True
    """
    candidate = Path(repo_id)
    if candidate.is_dir():
        if (candidate / WEIGHTS_FILE).is_file():
            return [str(candidate / WEIGHTS_FILE)]
        if (candidate / WEIGHTS_INDEX_FILE).is_file():
            return [
                str(candidate / shard)
                for shard in _shards_from_index(str(candidate / WEIGHTS_INDEX_FILE))
            ]
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError, LocalEntryNotFoundError
    from openral_rskill._vla_core import hf_download_cached_first

    def _fetch(filename: str) -> str:
        return str(
            hf_download_cached_first(
                hf_hub_download,
                LocalEntryNotFoundError,
                repo_id=repo_id,
                filename=filename,
                revision=revision,
            )
        )

    try:
        return [_fetch(WEIGHTS_FILE)]
    except EntryNotFoundError as single_file_missing:
        try:
            index_path = _fetch(WEIGHTS_INDEX_FILE)
        except EntryNotFoundError:
            raise single_file_missing from None
    return [_fetch(shard) for shard in _shards_from_index(index_path)]


def peek_safetensors_keys(repo_id: str, *, revision: str | None = None) -> set[str] | None:
    """Return the key set of a (possibly sharded) safetensors checkpoint without loading tensors.

    Used by adapters that want to skip ``reset_parameters()`` on
    modules whose params are about to be overwritten by an upcoming
    state-dict load — on a 3.4 B-param model that walk dominates the
    ``to_empty_cpu`` phase even though every kaiming / normal / ones
    init it produces is thrown away seconds later.

    Reads only the safetensors headers (~10 ms warm), so calling this
    eagerly during the build phase is cheap. Resolves through
    ``resolve_weights_files`` (directory-aware, cached-first, sharded). Works for
    prequantized packs and bare source checkpoints alike. The keys are the
    file's raw spelling: an adapter whose loader renames keys maps them itself.

    Args:
        repo_id: HF Hub repo id or local snapshot directory.
        revision: Pinned Hub revision, forwarded to the download.

    Returns:
        The set of tensor keys, or ``None`` if the checkpoint cannot be
        downloaded / parsed (caller falls back to a full ``reset_parameters`` walk).
    """
    try:
        from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError
        from safetensors import safe_open
    except ImportError:  # pragma: no cover
        return None

    try:
        paths = resolve_weights_files(repo_id, revision=revision)
    except (EntryNotFoundError, RepositoryNotFoundError, OSError, ROSConfigError):
        return None

    keys: set[str] = set()
    try:
        for path in paths:
            # reason: safetensors ships no type stubs for safe_open
            with safe_open(path, framework="pt") as f:  # type: ignore[no-untyped-call]
                keys.update(f.keys())
    except (OSError, ValueError):
        return None
    return keys


# ── Meta-init loading (family-neutral) ───────────────────────────────────────
#
# A multi-billion-parameter VLA built by ``from_pretrained`` is allocated and
# initialised on the CPU before its weights are read — minutes on a Jetson.
# The fast path builds it on meta (``accelerate.init_empty_weights``),
# allocates with ``to_empty``, and fills it from the checkpoint. Two things
# ``from_pretrained`` did implicitly then become the loader's job, and both
# fail *silently* (a plausible-looking, wrong policy) when missed: the
# non-persistent buffers the checkpoint does not carry, and the checkpoint's
# key spelling. These helpers make both explicit and fail closed.

BufferRebuilder = Callable[[str, Any, Any], bool]
"""A family hook ``(buffer_name, owning_module, buffer) -> handled``.

It rebuilds, in place, a non-persistent buffer only its family knows how to
compute (π0.5: Gemma RoPE ``inv_freq`` and ``embed_scale``) and returns
``True``; ``False`` leaves the buffer to the generic rules / the refusal."""

KeyMap = Callable[[list[str]], dict[str, str]]
"""A family hook mapping a checkpoint's raw keys to ``{parameter_name: source_key}``.

It is the family's own ``from_pretrained`` key normalisation (renames,
prefixes, deliberately dropped keys), so the fast path loads exactly what the
slow path would. A source key it leaves out is a deliberate drop."""


def _persistent_state_names(policy: Any) -> tuple[set[str], set[str]]:
    """``(persistent_buffer_names, non_persistent_buffer_names)`` of ``policy``.

    Read from each module's ``_non_persistent_buffers_set`` rather than
    ``state_dict()``, which a pre-``.to(cuda)`` bitsandbytes ``Linear8bitLt``
    cannot serialise.
    """
    persistent: set[str] = set()
    transient: set[str] = set()
    for prefix, module in policy.named_modules(remove_duplicate=False):
        hidden: set[str] = getattr(module, "_non_persistent_buffers_set", set())
        for name, buf in module._buffers.items():
            if buf is None:
                continue
            full = f"{prefix}.{name}" if prefix else name
            (transient if name in hidden else persistent).add(full)
    return persistent, transient


def rebuild_non_persistent_buffers(
    policy: Any, *, torch: Any, family: str, rebuild: BufferRebuilder | None = None
) -> None:
    """Recompute every buffer a meta-initialised policy cannot load from its checkpoint.

    After ``init_empty_weights`` + ``to_empty`` every buffer holds garbage, and
    the *non-persistent* ones are not in the checkpoint, so each must be rebuilt
    the way its module builds it. Generic rule: ``*.position_ids`` (int64
    ``arange``; garbage indices assert on the embedding gather). Everything
    else goes to the family's ``rebuild`` hook.

    A non-persistent buffer nobody rebuilt is refused (``ROSRuntimeError``)
    rather than run on garbage — the only symptom of a missed one is a policy
    that is silently wrong (π0.5's ``embed_scale``: cosine 0.03 against
    ``from_pretrained`` with every loaded tensor matching). Call under
    ``torch.no_grad()``.

    Args:
        policy: The meta-then-materialised policy ``nn.Module``.
        torch: The imported ``torch`` module.
        family: Adapter family, named in the refusal.
        rebuild: The family's hook for its own buffers (see ``BufferRebuilder``).

    Raises:
        ROSRuntimeError: A non-persistent buffer no rule rebuilt.
    """
    from openral_core.exceptions import ROSRuntimeError

    _, transient = _persistent_state_names(policy)
    unhandled: list[str] = []
    # Persistent buffers the rules know are rebuilt too (a strict=False loader,
    # e.g. the nf4 prequant overlay, may not fill them); only an unhandled
    # *non-persistent* one is refused, the checkpoint covers the rest.
    for name, buf in policy.named_buffers():
        mod = policy.get_submodule(name.rpartition(".")[0])
        if name.endswith(".position_ids") and buf.dtype == torch.int64:
            arange = torch.arange(buf.shape[-1], dtype=torch.int64, device=buf.device)
            buf.copy_(arange.expand_as(buf))
        elif rebuild is not None and rebuild(name, mod, buf):
            continue
        elif name in transient:
            unhandled.append(name)
    if unhandled:
        raise ROSRuntimeError(
            f"{family} fast meta-init: non-persistent buffers this loader cannot rebuild "
            f"(they are not in the checkpoint and would run as garbage): {unhandled[:8]}"
        )


def expand_covered_keys_via_tied_storage(policy: Any, covered_keys: set[str]) -> set[str]:
    """Extend ``covered_keys`` to include every parameter tied to a covered one.

    A checkpoint stores a tied pair (PaliGemma's ``embed_tokens`` / ``lm_head``)
    under one name; after ``to_empty`` + ``tie_transformers_weights`` both names
    share storage, detected here by ``untyped_storage().data_ptr()``. Without
    this, ``targeted_reset_parameters`` pays a ~10 s ``normal_`` init on a slot
    the load fills through the tie. ``named_parameters(remove_duplicate=False)``
    keeps both alias names visible (and avoids ``state_dict()``, which a
    pre-``.to(cuda)`` ``Linear8bitLt`` cannot serialise).
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


#: Bytes of the little-endian u64 header length a safetensors file starts with.
_SAFETENSORS_PREFIX_BYTES = 8
#: A header entry's ``data_offsets`` is a ``[start, end)`` pair.
_OFFSETS_PAIR = 2
#: safetensors header dtype tags -> torch dtype attribute names.
_SAFETENSORS_DTYPES: dict[str, str] = {
    "BF16": "bfloat16",
    "F16": "float16",
    "F32": "float32",
    "F64": "float64",
    "I64": "int64",
    "I32": "int32",
    "I16": "int16",
    "I8": "int8",
    "U8": "uint8",
    "BOOL": "bool",
}


def read_safetensors_header(path: str) -> tuple[dict[str, dict[str, Any]], int]:
    """A safetensors file's header entries and the byte its data section starts at.

    Returns ``(tensors, data_start)``: per key ``dtype``, ``shape`` and ``data_offsets``
    (relative to ``data_start``).

    Validates what the reader relies on -- a JSON object of entries with integer
    ``data_offsets`` pairs and ``shape`` lists, a known dtype, and a byte span inside the
    file (``_stream_safetensors_file`` also checks it against ``shape`` × element size).
    Every malformed header raises ``ROSConfigError``, so a truncated or hand-edited file
    cannot read past its end or mis-view a tensor.

    Example:
        >>> import tempfile, torch
        >>> from safetensors.torch import save_file
        >>> with tempfile.TemporaryDirectory() as d:
        ...     save_file({"w": torch.zeros(2, 3)}, f"{d}/m.safetensors")
        ...     tensors, start = read_safetensors_header(f"{d}/m.safetensors")
        >>> tensors["w"]["dtype"], tensors["w"]["shape"], tensors["w"]["data_offsets"]
        ('F32', [2, 3], [0, 24])
    """
    size = os.path.getsize(path)
    with open(path, "rb") as fh:
        prefix = fh.read(_SAFETENSORS_PREFIX_BYTES)
        if len(prefix) != _SAFETENSORS_PREFIX_BYTES:
            raise ROSConfigError(f"{path}: too short for a safetensors header")
        (n,) = struct.unpack("<Q", prefix)
        if n > size - _SAFETENSORS_PREFIX_BYTES:
            raise ROSConfigError(f"{path}: safetensors header length {n} exceeds the file")
        try:
            header = json.loads(fh.read(n))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ROSConfigError(f"{path}: safetensors header is not valid JSON") from exc
    if not isinstance(header, dict):
        raise ROSConfigError(f"{path}: safetensors header is not a JSON object")
    header.pop("__metadata__", None)
    data_start = _SAFETENSORS_PREFIX_BYTES + n
    for key, meta in header.items():
        offsets = meta.get("data_offsets") if isinstance(meta, dict) else None
        shape = meta.get("shape") if isinstance(meta, dict) else None
        if not (
            isinstance(offsets, list)
            and len(offsets) == _OFFSETS_PAIR
            and all(isinstance(o, int) for o in offsets)
            and isinstance(shape, list)
            and all(isinstance(d, int) and d >= 0 for d in shape)
        ):
            raise ROSConfigError(f"{path}: tensor {key!r} has a malformed header entry")
        start, end = offsets
        if not 0 <= start <= end <= size - data_start:
            raise ROSConfigError(
                f"{path}: tensor {key!r} spans {meta['data_offsets']} outside the file"
            )
        if meta["dtype"] not in _SAFETENSORS_DTYPES:
            raise ROSConfigError(f"{path}: tensor {key!r} has unsupported dtype {meta['dtype']!r}")
    return header, data_start


def _stream_safetensors_file(
    path: str,
    torch: Any,
    targets: Mapping[str, Any],
    key_map: KeyMap | None,
    loaded: set[str],
    unexpected: list[str],
    *,
    pin_memory: bool,
) -> int:
    """Copy one shard's tensors into ``targets`` (name -> parameter / buffer).

    Returns the number of source keys ``key_map`` dropped on purpose.

    Reads each tensor's bytes straight from the file into one reusable staging buffer
    (pinned when the targets live on CUDA) and ``copy_``-es the typed view into its
    target. ``safe_open``'s memory-mapped ``get_tensor`` + CUDA copy ran at ~0.2 GB/s on
    a GB10 (a 3.4 B-param π0.5 took 34-45 s); plain reads into pinned memory stream at
    >4 GB/s (1.7 s) -- the same bytes into the same slots.
    """
    header, data_start = read_safetensors_header(path)
    raw = list(header)
    mapping = key_map(raw) if key_map is not None else {k: k for k in raw}
    by_source: dict[str, list[str]] = {}
    for name, source in mapping.items():
        if targets.get(name) is None:
            unexpected.append(name)
            continue
        by_source.setdefault(source, []).append(name)
    order = sorted(by_source, key=lambda k: header[k]["data_offsets"][0])
    largest = max(
        (header[k]["data_offsets"][1] - header[k]["data_offsets"][0] for k in order), default=0
    )
    staging = torch.empty(max(1, largest), dtype=torch.uint8, pin_memory=pin_memory)
    view = memoryview(staging.numpy())
    with open(path, "rb", buffering=0) as fh:
        for source in order:
            meta = header[source]
            start, end = meta["data_offsets"]
            nbytes = end - start
            dtype = getattr(torch, _SAFETENSORS_DTYPES[meta["dtype"]])
            if nbytes != math.prod(meta["shape"]) * torch.empty((), dtype=dtype).element_size():
                raise ROSConfigError(
                    f"{path}: tensor {source!r} byte span {nbytes} does not match its shape"
                    f" {meta['shape']} and dtype {meta['dtype']}"
                )
            fh.seek(data_start + start)
            got = 0
            while got < nbytes:
                read = fh.readinto(view[got:nbytes])
                if not read:
                    raise ROSConfigError(f"{path}: file ended inside tensor {source!r}")
                got += read
            tensor = staging[:nbytes].view(dtype).view(meta["shape"])
            for name in by_source[source]:
                targets[name].copy_(tensor)
                loaded.add(name)
    return len(set(raw) - set(mapping.values()))


def stream_state_into_policy(
    policy: Any,
    repo_id: str,
    *,
    torch: Any,
    family: str,
    revision: str | None = None,
    key_map: KeyMap | None = None,
) -> None:
    """Copy a (possibly sharded) safetensors checkpoint tensor-by-tensor into ``policy``.

    The meta-init substitute for a family's ``from_pretrained`` weight load. The
    policy is already materialised where it will run (``to_empty(device)``);
    each tensor is read from the file into one staging buffer (pinned when the
    policy is on CUDA; ``_stream_safetensors_file``) and ``copy_``-ed into its
    target, which moves and casts it — so the device peak is one model plus one
    tensor, never a second full state dict, and never a source-dtype temporary on
    the GPU.

    ``key_map`` applies the family's own key normalisation first (π0.5:
    lerobot's ``_fix_pytorch_state_dict_keys`` + the ``model.`` prefix), so the
    fast path loads the same tensors into the same slots as ``from_pretrained``.
    Then it fails closed:

    * a mapped key that names no parameter / persistent buffer is *unexpected*;
    * a parameter or persistent buffer nothing was copied into is *missing* —
      it would keep its reset/uninitialised value. The one allowed exception
      is a **tied alias**: a slot sharing storage with one that was loaded
      (a checkpoint stores a tied ``embed_tokens`` / ``lm_head`` pair once).

    Either raises ``ROSConfigError``. Source keys the family's ``key_map`` drops
    on purpose are counted as ``dropped`` in the ``meta_init_state_streamed`` log.
    Tie weights (``tie_transformers_weights``) before calling.

    Raises:
        ROSConfigError: safetensors missing, or any missing / unexpected key.
    """
    paths = resolve_weights_files(repo_id, revision=revision)
    persistent, _ = _persistent_state_names(policy)
    targets: dict[str, Any] = dict(policy.named_parameters(remove_duplicate=False))
    targets.update(
        (name, buf)
        for name, buf in policy.named_buffers(remove_duplicate=False)
        if name in persistent
    )
    loaded: set[str] = set()
    unexpected: list[str] = []
    dropped = 0
    pin = any(getattr(t, "is_cuda", False) for t in targets.values())
    with torch.no_grad():
        for path in paths:
            dropped += _stream_safetensors_file(
                path, torch, targets, key_map, loaded, unexpected, pin_memory=pin
            )
    loaded_storage = {targets[n].untyped_storage().data_ptr() for n in loaded}
    missing = sorted(
        n
        for n in set(targets) - loaded
        if targets[n].untyped_storage().data_ptr() not in loaded_storage
    )
    log.info(
        "meta_init_state_streamed",
        family=family,
        repo=repo_id,
        shards=len(paths),
        loaded=len(loaded),
        tied_aliases=len(set(targets) - loaded) - len(missing),
        dropped=dropped,
    )
    if missing or unexpected:
        raise ROSConfigError(
            f"{family} fast meta-init: checkpoint {repo_id!r} does not match the policy "
            f"after key normalisation — {len(missing)} missing (would run on reset/garbage "
            f"values): {missing[:8]}; {len(unexpected)} unexpected: {sorted(unexpected)[:8]}"
        )


# ── Manifest-driven dtype resolution ──────────────────────────────────────────


def normalise_manifest_dtype(manifest: Any) -> str | None:
    """Pull ``manifest.quantization.dtype`` as a string, if present.

    The rSkill schema's ``QuantizationDtype`` enum values
    (``"int4"``, ``"int8"``, ``"bf16"``, ``"fp32"`` ...) already match
    the keys the adapter's dispatch checks; this helper just guards
    against a manifest without a quantization block (``None`` →
    ``default_dtype_for_device`` kicks in at the call site).
    """
    quant = getattr(manifest, "quantization", None)
    if quant is None:
        return None
    dtype = getattr(quant, "dtype", None)
    if dtype is None:
        return None
    return str(getattr(dtype, "value", dtype))


#: Per-run dtype override, honoured by every policy family.
#:
#: Replaces the per-family ``OPENRAL_GR00T_QUANTIZATION`` /
#: ``OPENRAL_RLDX_QUANTIZATION`` vars, which covered only the two sidecar
#: adapters and left the in-process ones (pi05 / molmoact2 / openvla) with no
#: per-run lever at all after the ``--vla-extra`` flag was removed.
QUANTIZATION_DTYPE_ENV = "OPENRAL_QUANTIZATION_DTYPE"

#: Tokens that name a *packing* format rather than a plain compute precision.
#: ``resolve_quant_plan`` sets ``quantize`` when the resolved dtype is one of
#: these, which is the single definition of "does this load rewrite Linears".
_PACKING_TOKENS: frozenset[str] = frozenset({"nf4", "int8"})

#: Canonical spelling per accepted alias. The rSkill schema's enum value for
#: 4-bit is ``int4``, but every consumer downstream — the pi05 / molmoact2 /
#: openvla dispatch, ``tools/rldx_sidecar.py``, and the argparse ``choices`` on
#: ``tools/behavior_groot_sidecar.py`` — spells it ``nf4``. Normalising here is
#: what lets one resolved value feed all of them.
_QUANT_TOKEN_ALIASES: dict[str, str] = {
    "int4": "nf4",
    "4bit": "nf4",
    "nf4": "nf4",
    "int8": "int8",
    "bf16": "bf16",
    "bfloat16": "bf16",
    "fp16": "fp16",
    "float16": "fp16",
    "half": "fp16",
    "fp32": "fp32",
    "float32": "fp32",
    "none": "none",
    "": "none",
}


def canonical_quant_token(raw: object) -> str | None:
    """Normalise a dtype spelling to the token every consumer understands.

    Maps the schema's ``int4`` (and the legacy ``4bit``) onto ``nf4``, folds
    the float aliases, and passes an unknown token through lowercased so a new
    format surfaces at the adapter rather than being silently dropped here.

    Returns ``None`` for ``None`` so callers can distinguish "unset" from the
    explicit ``"none"`` (which means *do not quantize*).

    Example:
        >>> canonical_quant_token("int4")
        'nf4'
        >>> canonical_quant_token("BF16")
        'bf16'
        >>> canonical_quant_token(None) is None
        True
    """
    if raw is None:
        return None
    token = str(raw).strip().lower()
    return _QUANT_TOKEN_ALIASES.get(token, token)


class QuantPlan(NamedTuple):
    """How one policy load should be quantized, resolved from all sources.

    Attributes:
        dtype: Canonical token (``nf4`` / ``int8`` / ``bf16`` / ``fp16`` /
            ``fp32`` / ``none``), or ``None`` when nothing declared one and
            the caller passed no default.
        quantize: ``True`` when ``dtype`` names a packing format, so the load
            rewrites Linears. ``False`` for a plain precision and for
            ``none``.
        source: Which input won — ``env`` / ``spec_extra`` / ``manifest`` /
            ``default`` / ``unset``. Carried so the adapter can log it.
        extra: The manifest's ``quantization.extra`` knobs (``quantize_scope``,
            ``nf4_min_params``, …), empty when the manifest declares none.
    """

    dtype: str | None
    quantize: bool
    source: str
    extra: Mapping[str, Any]


def quantization_extra(manifest: Any | None) -> Mapping[str, Any]:
    """Return the manifest's ``quantization.extra`` knob map, or ``{}``.

    This is the one home for the per-family packing knobs (``quantize_scope``
    on GR00T, ``nf4_min_params`` on the BEHAVIOR sidecar). They used to live in
    ``policy_extras`` next to unrelated adapter settings, which is why each
    family grew its own reader.
    """
    quant = getattr(manifest, "quantization", None)
    extra = getattr(quant, "extra", None) if quant is not None else None
    return dict(extra) if isinstance(extra, dict) else {}


def plan_quantization(
    spec: Any,
    manifest: Any | None = None,
    *,
    default: str | None = None,
) -> QuantPlan:
    """Resolve one load's quantization from every source, without the Pro gate.

    The precedence logic of :func:`resolve_quant_plan`, which every loader calls;
    that function adds the ADR-0123 check on top. Call this one only to inspect a
    plan, never to load weights.

    Precedence, first hit wins:

    1. ``$OPENRAL_QUANTIZATION_DTYPE`` — the per-run override, honoured
       identically by every family.
    2. ``spec.quantization.dtype`` — the typed ``VLASpec`` override.
    3. ``spec.extra["dtype"]`` — the untyped override (a programmatic
       ``VLASpec``, or a scene that pins the dtype for one pairing).
    4. ``manifest.quantization.dtype`` — the rSkill's declared *runtime*
       dtype, the normal case. A checkpoint stored at a different precision
       than it runs records that in ``quantization.extra.stored_dtype``; the
       resolver never reads it.
    5. ``default`` — the adapter's own fallback, for families that quantize
       unless told otherwise.

    The resolver does not know which tokens a family can load; every adapter
    passes the plan through :func:`require_supported_dtype`, so an
    unsupported request raises instead of loading something else.

    Every resolution is logged; a resolved dtype that *differs* from the
    manifest's declared one is logged at WARNING naming both, because the
    package's recorded ``benchmarks:`` numbers then no longer describe what is
    about to run (CLAUDE.md §1.4 — quantization is never silent).

    Args:
        spec: The ``VLASpec`` for this run; its ``quantization`` and
            ``extra`` are read.
        manifest: The resolved ``RSkillManifest``, when the caller has one.
        default: Fallback token when no source declares a dtype.

    Returns:
        A :class:`QuantPlan`.
    """
    declared = normalise_manifest_dtype(manifest) if manifest is not None else None
    manifest_token = canonical_quant_token(declared)

    env_raw = os.environ.get(QUANTIZATION_DTYPE_ENV)
    typed_raw = normalise_manifest_dtype(spec)
    spec_raw = spec.extra.get("dtype") if hasattr(spec, "extra") and spec.extra else None

    if env_raw:
        token, source = canonical_quant_token(env_raw), "env"
    elif typed_raw:
        token, source = canonical_quant_token(typed_raw), "spec_quantization"
    elif spec_raw:
        token, source = canonical_quant_token(spec_raw), "spec_extra"
    elif manifest_token:
        token, source = manifest_token, "manifest"
    elif default:
        token, source = canonical_quant_token(default), "default"
    else:
        token, source = None, "unset"

    quantize = token in _PACKING_TOKENS
    if token is not None and manifest_token is not None and token != manifest_token:
        log.warning(
            "quantization.override",
            resolved=token,
            declared=manifest_token,
            source=source,
            skill=getattr(manifest, "name", None),
            note=(
                "running at a dtype the manifest does not declare; its "
                "benchmarks / eval results describe the declared one"
            ),
        )
    else:
        log.info("quantization.resolved", dtype=token, source=source, quantize=quantize)

    return QuantPlan(
        dtype=token, quantize=quantize, source=source, extra=quantization_extra(manifest)
    )


def resolve_quant_plan(
    spec: Any,
    manifest: Any | None = None,
    *,
    default: str | None = None,
) -> QuantPlan:
    """Resolve one load's quantization, refusing sub-16-bit weights without OpenRAL Pro.

    :func:`plan_quantization` (precedence: ``$OPENRAL_QUANTIZATION_DTYPE``,
    ``spec.quantization``, ``spec.extra["dtype"]``, the manifest, ``default``),
    then the ADR-0123 checks: a resolved dtype of 8 bits or fewer, or a
    ``spec.weights_uri`` checkpoint carrying a ``quantization_config``, needs the
    OpenRAL Pro quantization package.

    Args:
        spec: The ``VLASpec`` for this run.
        manifest: The resolved ``RSkillManifest``, when the caller has one.
        default: Fallback token when no source declares a dtype.

    Returns:
        A :class:`QuantPlan`.

    Raises:
        ROSConfigError: Sub-16-bit weights requested, or a quantized checkpoint,
            and OpenRAL Pro's quantization module is not installed.
    """
    plan = plan_quantization(spec, manifest, default=default)
    what = (
        f"rSkill {getattr(manifest, 'name', '?')!r}"
        if manifest is not None
        else f"policy {getattr(spec, 'id', '?')!r}"
    )
    require_pro_for_dtype(plan.dtype, what=what)
    weights_uri = getattr(spec, "weights_uri", None)
    if isinstance(weights_uri, str) and weights_uri:
        require_pro_for_checkpoint(weights_uri, what=what)
    return plan


def require_supported_dtype(plan: QuantPlan, supported: frozenset[str], family: str) -> None:
    """Refuse a resolved dtype the ``family`` cannot actually load.

    Every adapter calls this with the set of canonical tokens (see
    :func:`canonical_quant_token`) its load path honours, so a request it
    would otherwise silently reinterpret — ``int8`` loading as bf16, ``bf16``
    loading as fp32 — fails at build time instead (CLAUDE.md §1.4). An unset
    plan (``dtype is None``) or an explicit ``"none"`` (do not quantize) means
    the adapter's own native precision applies and is always accepted.

    Raises:
        ROSConfigError: Naming the family, the requested token, where it came
            from, and the supported set.

    Example:
        >>> plan = QuantPlan(dtype="int8", quantize=True, source="env", extra={})
        >>> try:
        ...     require_supported_dtype(plan, frozenset({"nf4", "bf16"}), "molmoact2")
        ... except ROSConfigError as exc:
        ...     print(str(exc).split(";")[0])
        molmoact2 cannot load dtype 'int8' (from env)
        >>> require_supported_dtype(
        ...     QuantPlan(dtype="none", quantize=False, source="env", extra={}),
        ...     frozenset({"fp32"}),
        ...     "xvla",
        ... )
    """
    if plan.dtype in (None, "none") or plan.dtype in supported:
        return
    raise ROSConfigError(
        f"{family} cannot load dtype {plan.dtype!r} (from {plan.source}); "
        f"supported: {sorted(supported)}. Set {QUANTIZATION_DTYPE_ENV} or the "
        "rSkill's quantization.dtype to one of those."
    )


def torch_dtype_for(torch: Any, dtype_str: str | None, device: str) -> Any:
    """Map a manifest dtype string to a torch dtype, with a CUDA-aware default.

    Recognised dtype strings (case-insensitive, via
    :func:`canonical_quant_token`): ``bf16`` / ``bfloat16``, ``fp16`` /
    ``float16`` / ``half``, ``fp32`` / ``float32``. ``None`` / ``"none"``
    pick the device-aware default (bf16 on CUDA, fp32 elsewhere). A packing
    token (``nf4`` / ``int8``) or an unknown one raises: it is not a torch
    dtype, and silently mapping it to the default is how an ``int8`` request
    used to load as bf16. Packing adapters pass ``None`` for the compute
    dtype of the leaves they do not quantize.

    Raises:
        ROSConfigError: On a token that names no torch dtype.
    """
    token = canonical_quant_token(dtype_str)
    if token in (None, "none"):
        # Default: bf16 on CUDA (large VLAs are too big for fp32 on consumer
        # GPUs), fp32 elsewhere.
        return torch.bfloat16 if device.startswith("cuda") else torch.float32
    mapped = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}.get(str(token))
    if mapped is None:
        raise ROSConfigError(
            f"dtype {dtype_str!r} is not a torch compute dtype (expected bf16 / fp16 / "
            "fp32, or none for the device default)."
        )
    return mapped


def sidecar_quant_token(plan: QuantPlan, accepted: frozenset[str], family: str) -> str:
    """Map a resolved plan onto a sidecar's ``--quantization`` token, or raise.

    Every OpenRAL sidecar loads its ``none`` mode at bf16, so a ``bf16``
    request maps onto ``none`` when the sidecar accepts it; any other token
    outside ``accepted`` raises via :func:`require_supported_dtype` rather than
    collapsing to something the sidecar can parse.

    Example:
        >>> sidecar_quant_token(
        ...     QuantPlan("bf16", False, "env", {}), frozenset({"none", "nf4"}), "lingbot_vla2"
        ... )
        'none'
        >>> try:
        ...     sidecar_quant_token(QuantPlan(None, False, "default", {}), frozenset({"nf4"}), "x")
        ... except ROSConfigError as exc:
        ...     print(str(exc).split(";")[0])
        x sidecar cannot load dtype None (from default)
    """
    supported = accepted | {"bf16"} if "none" in accepted else accepted
    require_supported_dtype(plan, supported, family)
    token = "none" if plan.dtype in (None, "none", "bf16") else str(plan.dtype)
    if token not in accepted:  # an unset / "none" plan on a sidecar with no "none" mode
        raise ROSConfigError(
            f"{family} sidecar cannot load dtype {plan.dtype!r} (from {plan.source}); "
            f"accepted: {sorted(accepted)}."
        )
    return token


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
    will land via Pro's ``install_prequantized_linears`` + ``load_state_dict``.
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
    ``targeted_reset_parameters``. Failures are non-fatal (a meta-init
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
    "WEIGHTS_FILE",
    "WEIGHTS_INDEX_FILE",
    "BufferRebuilder",
    "KeyMap",
    "default_dtype_for_device",
    "expand_covered_keys_via_tied_storage",
    "normalise_manifest_dtype",
    "peek_safetensors_keys",
    "rebuild_non_persistent_buffers",
    "require_supported_dtype",
    "resolve_quant_plan",
    "resolve_weights_files",
    "sidecar_quant_token",
    "stream_state_into_policy",
    "targeted_reset_parameters",
    "tie_transformers_weights",
    "torch_dtype_for",
]


# Suppress mypy unused-import warnings on optional bnb imports.
_ = os
