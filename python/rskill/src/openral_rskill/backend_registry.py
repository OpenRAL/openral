"""Runtime-backend + policy-attach-hook registry (OpenRAL Pro extraction seam).

OpenRAL Pro ships proprietary inference backends — the TensorRT
engine runtime and the NVMM zero-copy consumers — as a separate, privately
distributed package (``openral-pro-trt``) rather than in-tree. This module is
the seam that lets those packages plug into the open ``openral-rskill`` /
``openral-runner`` stack without either side importing the other directly:

- :func:`resolve_runtime_backend` — name → :class:`~openral_rskill.runtime.Runtime`
  implementation class. Built-in backends (``pytorch``, ``onnx``, ``null``)
  resolve from an in-tree dict; anything else is looked up via the
  ``openral.runtime_backends`` entry-point group, so a privately installed
  package (e.g. ``openral-pro-trt`` registering ``tensorrt``) is discovered
  automatically. A backend that is neither built-in nor installed raises a
  typed :exc:`~openral_core.exceptions.ROSConfigError` naming the package to
  install (CLAUDE.md §1.4 — explicit, no silent fallback).
- :func:`maybe_attach_pro_hooks` — generic "does OpenRAL Pro want to swap this
  policy's inference path" lookup via the ``openral.policy_attach_hooks``
  entry-point group. Absent hook (open-source-only install) is a no-op with a
  debug log; an attached hook logs at info level. Replaces the ad hoc
  per-policy ``try: from openral_rskill.smolvla_trt import ...`` /
  ``act_trt`` imports that used to live directly in
  :mod:`openral_rskill.smolvla` and :mod:`openral_sim.policies.act`.
"""

from __future__ import annotations

from importlib import import_module
from importlib.metadata import entry_points
from typing import TYPE_CHECKING, Any

import structlog
from openral_core.exceptions import ROSConfigError

if TYPE_CHECKING:
    from openral_rskill.runtime import Runtime

__all__ = ["maybe_attach_pro_hooks", "resolve_runtime_backend"]

log = structlog.get_logger(__name__)

_RUNTIME_BACKENDS_GROUP = "openral.runtime_backends"
_POLICY_ATTACH_HOOKS_GROUP = "openral.policy_attach_hooks"

# Built-in backends, resolved lazily by module path: `openral_rskill.__init__`
# imports this module eagerly, and `runtime_pytorch` imports torch at module
# top — an eager class reference here would drag torch into every
# `import openral_cli.main` (test_bh_cli_import_is_light guards this).
_BUILTIN_RUNTIME_BACKENDS: dict[str, str] = {
    "pytorch": "openral_rskill.runtime_pytorch:PyTorchRuntime",
    "onnx": "openral_rskill.runtime_onnx:ONNXRuntime",
    "null": "openral_rskill.runtime:NullRuntime",
}


def resolve_runtime_backend(kind: str) -> type[Runtime]:
    """Resolve an inference-runtime backend class by name.

    Checks the built-in backends first (``pytorch``, ``onnx``, ``null``),
    then the ``openral.runtime_backends`` entry-point group for privately
    installed backends (e.g. ``openral-pro-trt`` registering ``tensorrt``).

    Args:
        kind: Backend name, e.g. ``"pytorch"``, ``"onnx"``, ``"null"``, or
            ``"tensorrt"`` (only resolvable with ``openral-pro-trt``
            installed).

    Returns:
        The :class:`~openral_rskill.runtime.Runtime`-conforming class
        (not an instance — callers construct it themselves, e.g.
        ``resolve_runtime_backend("pytorch")(device="cuda:0")``).

    Raises:
        ROSConfigError: *kind* is neither a built-in backend nor a
            registered entry point. Names the private package to install.

    Example:
        >>> from openral_rskill.runtime import NullRuntime
        >>> resolve_runtime_backend("null") is NullRuntime
        True
        >>> resolve_runtime_backend("does-not-exist")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ...
        openral_core.exceptions.ROSConfigError: ...
    """
    builtin = _BUILTIN_RUNTIME_BACKENDS.get(kind)
    if builtin is not None:
        module_path, _, attr = builtin.partition(":")
        loaded_builtin: type[Runtime] = getattr(import_module(module_path), attr)
        return loaded_builtin

    for ep in entry_points(group=_RUNTIME_BACKENDS_GROUP):
        if ep.name == kind:
            loaded: type[Runtime] = ep.load()
            log.debug("runtime_backend.resolved", kind=kind, entry_point=ep.value)
            return loaded

    raise ROSConfigError(
        f"resolve_runtime_backend: unknown runtime backend {kind!r}. Built-in "
        f"backends are {sorted(_BUILTIN_RUNTIME_BACKENDS)!r}; 'tensorrt' and "
        "other proprietary backends require the private openral-pro-trt "
        "package, which registers itself via the "
        f"{_RUNTIME_BACKENDS_GROUP!r} entry-point group."
    )


def maybe_attach_pro_hooks(policy_name: str, skill: Any, **kwargs: Any) -> bool:  # noqa: ANN401  # reason: forwarded verbatim to whatever hook is registered
    """Look up and invoke an OpenRAL Pro policy-attach hook, if installed.

    Replaces the per-policy ``try: from openral_rskill.<x>_trt import
    maybe_attach_<x>_trt_from_env`` calls that used to be hardcoded in
    :mod:`openral_rskill.smolvla` and :mod:`openral_sim.policies.act`. A hook
    registered under the ``openral.policy_attach_hooks`` entry-point group
    (name = *policy_name*, e.g. ``"smolvla"`` or ``"act"``) is loaded and
    called as ``hook(skill, **kwargs)``; its truthy/falsy return says whether
    it swapped the policy's inference path.

    No hook installed (open-source-only install, i.e. no ``openral-pro-trt``)
    is expected and explicit: a debug log, and ``False`` — the caller falls
    through to its own eager/host path. This is not a silent skip (CLAUDE.md
    §1.4): the debug log records that the lookup happened and found nothing.

    Args:
        policy_name: Entry-point name to look up, e.g. ``"smolvla"``,
            ``"act"``.
        skill: The policy/skill object the hook may mutate in place (e.g. a
            ``SmolVLAPolicy`` or ``ACTPolicy`` instance).
        **kwargs: Forwarded verbatim to the hook (e.g. ``repo_id``,
            ``device``, ``n_cameras``, ``onnx_uri``).

    Returns:
        ``True`` if a hook was found and reported attaching; ``False``
        otherwise (no hook installed, or the hook declined — e.g. env var
        not set).

    Example:
        >>> from openral_rskill.runtime import NullRuntime
        >>> maybe_attach_pro_hooks("smolvla", NullRuntime())
        False
    """
    for ep in entry_points(group=_POLICY_ATTACH_HOOKS_GROUP):
        if ep.name == policy_name:
            hook = ep.load()
            attached = bool(hook(skill, **kwargs))
            if attached:
                log.info("pro_hooks.attached", policy=policy_name, entry_point=ep.value)
            else:
                log.debug("pro_hooks.declined", policy=policy_name, entry_point=ep.value)
            return attached

    log.debug("pro_hooks.absent", policy=policy_name, reason="openral-pro-trt not installed")
    return False
