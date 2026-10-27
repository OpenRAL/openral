"""Sub-16-bit weights need OpenRAL Pro (ADR-0123).

Open OpenRAL loads and runs weights in ``fp32``, ``fp16`` and ``bf16``. Every
format of 8 bits or fewer (``int8``, ``fp8``, ``int4`` / NF4, ``fp4_nvfp4``) needs
the OpenRAL Pro quantization package, which registers the ``quantization``
module in the ``openral.pro_modules`` entry-point group. This covers quantizing at
load time, loading an OpenRAL-made pre-quantized pack, and loading a checkpoint
quantized upstream (any ``quantization_config`` in its ``config.json``).

Three checks, one rule. Each raises ``ROSConfigError`` naming the dtype, the
caller and OpenRAL Pro, and never falls back to a wider dtype: a skill's recorded
benchmarks describe its declared dtype (CLAUDE.md §1.4).

- :func:`require_pro_for_dtype`: a resolved dtype token (the policy adapters,
  through ``openral_sim._quantization.resolve_quant_plan``).
- :func:`require_pro_for_manifest`: an rSkill manifest's declared dtype (the
  reward monitors, the scene VLM and the manifest detector, which load their
  checkpoint without the policy resolver).
- :func:`require_pro_for_checkpoint`: a checkpoint's own ``config.json``.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any

import structlog
from openral_core.exceptions import ROSConfigError

from openral_rskill.backend_registry import resolve_pro_module

if TYPE_CHECKING:
    from openral_core import RSkillManifest

log = structlog.get_logger(__name__)

#: Logical name of the Pro module, in the ``openral.pro_modules`` group.
PRO_QUANTIZATION_MODULE = "quantization"

#: Every spelling of a format of 8 bits or fewer that a manifest, a scene, the
#: ``OPENRAL_QUANTIZATION_DTYPE`` override or a sidecar flag may carry.
SUB_16_BIT_DTYPES: frozenset[str] = frozenset(
    {"int8", "fp8", "int4", "nf4", "4bit", "8bit", "fp4", "nvfp4", "fp4_nvfp4"}
)


def is_sub_16_bit(dtype: object) -> bool:
    """Whether ``dtype`` (a token or a ``QuantizationDtype``) is 8 bits or fewer.

    Example:
        >>> is_sub_16_bit("int4"), is_sub_16_bit("bf16"), is_sub_16_bit(None)
        (True, False, False)
    """
    if dtype is None:
        return False
    return str(getattr(dtype, "value", dtype)).strip().lower() in SUB_16_BIT_DTYPES


def pro_quantization_installed() -> bool:
    """Whether the OpenRAL Pro quantization module is installed.

    Example:
        >>> isinstance(pro_quantization_installed(), bool)
        True
    """
    return resolve_pro_module(PRO_QUANTIZATION_MODULE) is not None


def _pro_refusal(token: str, *, what: str) -> ROSConfigError:
    return ROSConfigError(
        f"{what} requests {token!r} weights. Formats of 8 bits or fewer (int8, fp8, "
        "int4/NF4, nvfp4) need the OpenRAL Pro quantization package (ADR-0123); open "
        "OpenRAL runs fp32, fp16 and bf16. Install OpenRAL Pro, or run a 16/32-bit "
        "variant of this model."
    )


def require_pro_for_dtype(dtype: object, *, what: str) -> None:
    """Refuse a sub-16-bit ``dtype`` unless OpenRAL Pro's quantization is installed.

    Args:
        dtype: The requested dtype token (``"nf4"``, ``"int8"``, ``"bf16"`` …) or a
            ``QuantizationDtype``. ``None`` and 16/32-bit dtypes always pass.
        what: Names the load for the error (the rSkill or the policy family).

    Raises:
        ROSConfigError: ``dtype`` is 8 bits or fewer and the Pro module is absent.

    Example:
        >>> require_pro_for_dtype("bf16", what="smolvla")  # open dtype: passes
    """
    if not is_sub_16_bit(dtype):
        return
    if pro_quantization_installed():
        log.info("quantization.pro_provider", dtype=str(getattr(dtype, "value", dtype)), what=what)
        return
    raise _pro_refusal(str(getattr(dtype, "value", dtype)), what=what)


def pro_quantization(*, what: str) -> ModuleType:
    """The OpenRAL Pro quantization module, which carries every sub-16-bit loader.

    Open adapters call its NF4 / LLM.int8 rewrites and pre-quantized pack loaders
    through this (ADR-0123 step 3); open ships no bitsandbytes code of its own.

    Args:
        what: Names the load for the error (the rSkill or the policy family).

    Raises:
        ROSConfigError: the Pro module is absent.
    """
    mod = resolve_pro_module(PRO_QUANTIZATION_MODULE)
    if mod is None:
        raise _pro_refusal("sub-16-bit", what=what)
    return mod


#: Sidecar ``--quantization`` tokens of 8 bits or fewer.
_SIDECAR_SUB_16_BIT = frozenset({"nf4", "int8", "prequantized_nf4"})


def sidecar_quantization_args(dtype: str | None, *, what: str, rewrites: bool = False) -> list[str]:
    """The ``--quantization-config`` argv a sidecar takes for a sub-16-bit load.

    A sidecar runs in its own venv, where OpenRAL Pro is not installed, so the
    adapter asks Pro for the ``BitsAndBytesConfig`` keyword arguments here and
    passes them as JSON; the sidecar only calls ``BitsAndBytesConfig(**kwargs)``
    and refuses any sub-16-bit load without them (ADR-0123 step 3), so a
    sidecar runs only fp16 / bf16 / fp32 unless OpenRAL Pro is installed.

    Args:
        dtype: The sidecar's ``--quantization`` token (``"nf4"``, ``"int8"``,
            ``"prequantized_nf4"``, ``"none"`` …). 16/32-bit tokens get no config.
        what: Names the load for the error.
        rewrites: Also pass ``--pro-quant-module``, the path of Pro's sidecar
            Linear-rewrite file, for a sidecar that rewrites its own modules
            (LingBot, BEHAVIOR GR00T) rather than handing the config to
            ``from_pretrained``.

    Returns:
        ``["--quantization-config", "<json>"]`` (plus ``["--pro-quant-module",
        "<path>"]`` with ``rewrites``), or ``[]``.

    Raises:
        ROSConfigError: ``dtype`` is sub-16-bit and Pro is absent.

    Example:
        >>> sidecar_quantization_args("none", what="xr1")
        []
    """
    if dtype not in _SIDECAR_SUB_16_BIT:
        return []
    pro = pro_quantization(what=what)
    kwargs = pro.hf_quantization_kwargs("int8" if dtype == "int8" else "nf4")
    args = ["--quantization-config", json.dumps(kwargs, sort_keys=True)]
    if rewrites:
        args += ["--pro-quant-module", pro.sidecar_module_path()]
    return args


def require_pro_for_manifest(manifest: RSkillManifest) -> None:
    """Refuse an rSkill whose declared dtype is sub-16-bit, unless Pro is installed.

    Args:
        manifest: The rSkill manifest. One without a quantization block passes.

    Raises:
        ROSConfigError: As :func:`require_pro_for_dtype`.
    """
    quant = getattr(manifest, "quantization", None)
    require_pro_for_dtype(
        getattr(quant, "dtype", None), what=f"rSkill {getattr(manifest, 'name', '?')!r}"
    )


def _checkpoint_config(source: str) -> dict[str, Any] | None:
    """Read a checkpoint's ``config.json`` from a local dir or the HF Hub, if any."""
    raw = source.removeprefix("local://")
    local = Path(raw)
    if local.is_dir():
        config = local / "config.json"
        return dict(json.loads(config.read_text())) if config.is_file() else None
    repo, _, revision = raw.removeprefix("hf://").partition("@")
    try:
        from huggingface_hub import hf_hub_download  # noqa: PLC0415  # reason: optional at import
        from huggingface_hub.errors import (  # noqa: PLC0415  # reason: optional at import
            EntryNotFoundError,
            HfHubHTTPError,
            LocalEntryNotFoundError,
        )
    except ImportError:
        log.warning("quantization.checkpoint_unchecked", source=source, reason="no huggingface_hub")
        return None
    try:
        downloaded = hf_hub_download(repo, "config.json", revision=revision or None)
    except EntryNotFoundError:
        return None
    except (LocalEntryNotFoundError, HfHubHTTPError, OSError) as exc:
        # Offline, private or missing: the load that follows fails the same way.
        log.warning("quantization.checkpoint_unchecked", source=source, reason=repr(exc))
        return None
    return dict(json.loads(Path(downloaded).read_text()))


def require_pro_for_checkpoint(source: str, *, what: str) -> None:
    """Refuse a checkpoint quantized upstream (a ``quantization_config``) without Pro.

    Args:
        source: ``hf://org/repo[@rev]``, ``org/repo``, ``local:///dir`` or a dir path.
        what: Names the load for the error.

    Raises:
        ROSConfigError: The checkpoint's ``config.json`` carries a
            ``quantization_config`` and the Pro module is absent.
    """
    if pro_quantization_installed():
        return
    quant = (_checkpoint_config(source) or {}).get("quantization_config")
    if not quant:
        return
    method = quant.get("quant_method", "unknown") if isinstance(quant, dict) else "unknown"
    raise ROSConfigError(
        f"{what}: checkpoint {source} is quantized ({method}). Quantized checkpoints need "
        "the OpenRAL Pro quantization package (ADR-0123); open OpenRAL runs fp32, fp16 and "
        "bf16. Install OpenRAL Pro, or point at a 16/32-bit checkpoint."
    )
