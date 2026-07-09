"""Shared loader helpers for openral_sim policy adapters.

Three things every full-graph policy adapter does on `__build_*` and
which used to be open-coded in :mod:`openral_sim.policies.pi05`,
:mod:`openral_sim.policies.rldx`, and :mod:`openral_sim.policies.smolvla`:

1. **Manifest resolution.** Pull the :class:`RSkillManifest` from
   ``spec.weights_uri`` when it is a bare rSkill reference (no explicit
   scheme); return ``None`` for ``hf://`` / local URIs and other
   explicit-scheme URIs so callers can decide whether the lack of a
   manifest is fatal. See :func:`load_manifest_for_spec`.

2. **Lazy torch + lerobot import.** Both the SmolVLA and π0.5 adapters
   defer ``torch`` / ``lerobot`` imports so installing ``openral-sim``
   never pulls them transitively (CLAUDE.md §3 — Python toolchain).
   The matching ``ROSConfigError`` install-hint is identical across the
   adapters; :func:`lazy_import_lerobot` returns the imported torch +
   ``make_pre_post_processors`` callable, with adapter-specific
   ``policy_class`` resolved by the caller.

3. **Processor pipeline materialisation.** ``make_pre_post_processors``
   needs a ``pretrained_path`` that already carries
   ``policy_preprocessor.json`` + ``policy_postprocessor.json`` sidecars.
   Both SmolVLA and π0.5 build that dir per-call via the rSkill
   manifest's ``processors`` block; the choice between
   :func:`openral_rskill._vla_core.materialize_processor_dir` (manifest
   only) and :func:`openral_sim.policies._processors.resolve_processor_dir`
   (manifest with snapshot fallback) is adapter-specific so callers
   pass a callable.

The RLDX adapter consumes the manifest helper here too — it doesn't run
``make_pre_post_processors`` (its sidecar holds the processor), but the
manifest is still its single source of truth for ``state_contract`` /
``image_preprocessing`` / ``n_action_steps``.

These helpers are deliberately small. We do **not** try to wrap the
quantization branch from pi05 (that flow is genuinely adapter-specific —
SmolVLA never quantizes, RLDX delegates to the sidecar). See
:mod:`openral_sim._quantization` for the dtype-resolution + bnb rewrite
helpers that *are* generic.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from openral_core.exceptions import ROSConfigError

if TYPE_CHECKING:
    from openral_core import RSkillManifest


def load_manifest_for_spec(spec: Any) -> RSkillManifest | None:
    """Load the rSkill manifest pinned in ``spec.weights_uri``.

    Returns the parsed :class:`openral_core.RSkillManifest` when
    ``spec.weights_uri`` is a bare rSkill reference (name, local path
    like ``rskills/smolvla-libero``, or HF repo id with no explicit
    scheme); returns ``None`` for ``hf://``, ``local://``, ``file://``,
    ``http://``, or ``https://`` URIs so the caller can decide whether
    the missing manifest is fatal (SmolVLA raises; pi05 / RLDX fall
    back to the URI directly).

    The function is intentionally tolerant of ``spec=None`` /
    ``spec.weights_uri=None`` -- both shapes resolve to ``None``.

    Args:
        spec: A ``VLASpec``-shaped object (anything with a
            ``weights_uri`` attribute).

    Returns:
        Parsed manifest or ``None``.
    """
    weights_uri = str(getattr(spec, "weights_uri", "") or "")
    if not weights_uri or weights_uri.startswith(
        ("hf://", "local://", "file://", "http://", "https://")
    ):
        return None
    from openral_rskill.loader import load_rskill_manifest

    return load_rskill_manifest(weights_uri)


def lazy_import_lerobot(
    adapter_name: str,
    *,
    install_hint: str = "just sync --all-packages --group libero",
) -> tuple[Any, Any]:
    """Import torch + lerobot's ``make_pre_post_processors`` factory.

    Both the SmolVLA and π0.5 adapters defer torch + lerobot imports to
    keep a bare ``openral-sim`` install free of the ~2 GiB of CUDA /
    transformers dependencies. The install-hint string is identical
    between the two; centralising it here keeps the message in sync
    when the recommended sync group changes.

    Args:
        adapter_name: Human-readable adapter name (``"SmolVLA"`` /
            ``"π0.5"`` / ...) inlined into the ``ROSConfigError`` so
            the operator sees which adapter triggered the failure.
        install_hint: Override the default
            ``just sync --all-packages --group libero`` hint when an
            adapter pulls a different optional group (e.g.
            ``"just sync --all-packages --group metaworld"``).

    Returns:
        ``(torch, make_pre_post_processors)``. The caller is expected
        to import the adapter-specific ``Policy`` class separately,
        since the registry shape varies (``PI05Policy`` vs
        ``SmolVLAPolicy`` vs custom ACT modules) and pulling the
        choice into this helper would force every adapter to share
        the same lerobot dispatch convention.

    Raises:
        ROSConfigError: If any of the lazy imports fail.
    """
    try:
        import torch
        from lerobot.policies.factory import make_pre_post_processors
        from openral_rskill import _lerobot_compat  # noqa: F401
    except ImportError as exc:  # pragma: no cover - opt-in
        raise ROSConfigError(
            f"{adapter_name} adapter requires torch + lerobot; "
            f"install with: {install_hint} (underlying: {exc!r})"
        ) from exc
    return torch, make_pre_post_processors


__all__ = [
    "lazy_import_lerobot",
    "load_manifest_for_spec",
]
