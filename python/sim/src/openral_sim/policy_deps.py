"""Policy-family dependency probing — shared by reasoner + skill_runner.

The policy factories in ``openral_sim.policies`` live behind opt-in extras
groups (``sim`` / ``libero`` / ``metaworld`` / ``robocasa``): ``transformers``,
``bitsandbytes``, ``lerobot[…]``, etc. When the right group isn't installed
the factory raises ``ImportError`` deep inside lerobot — confusing to surface,
and it leaves partially-loaded modules in ``sys.modules`` so subsequent calls
fail with a *different* ``cannot import name 'X'`` cascade error.

Two contracts live here:

* ``model_family_install_hint`` — the actionable uv-sync command for each
  known family, used by ``openral_rskill_ros.rskill_runner_node`` when
  translating a factory ``ImportError`` into ``ROSRuntimeError``.
* ``can_import_policy_family`` / ``filter_importable_manifests`` — pre-flight
  probes the reasoner runs at ``on_configure`` to drop rSkills whose deps
  aren't installed before the palette is built, so the operator sees one
  warning at boot ("dropped X: missing transformers; run ``just sync
  --all-packages --group sim``") instead of a per-tick dispatch failure.

Install commands always use ``just sync --all-packages --group <X>``, never
bare ``uv sync --group <X>``: ``--all-packages`` keeps the workspace members
(openral-core, openral-cli, …) installed — plain ``uv sync`` would uninstall
them and the next ROS launch would fail with ``No module named
'openral_core'``. ``just sync`` also repairs the ``hf-libero==0.1.3``
distutils-uninstall trap before+after the sync.

The probe never instantiates a factory or loads weights: by default it only
resolves the *top-level* package of each required import via
``importlib.util.find_spec`` (~0 ms, the same idiom
``openral_cli.deploy_sim._omdet_runtime_available`` uses). It deliberately
skips the deep module — ``lerobot/policies/__init__.py`` eagerly imports
every policy family's config class, so touching ``lerobot.policies.<anything>``
costs the whole tree (measured 6.6 s, and identically so via ``find_spec``,
which must import the parent to find the child). That cost was being paid in
three processes per deploy (CLI preflight, reasoner palette seed,
``runtime_node``) when only ``runtime_node`` needs the modules resolved.

The fast probe catches a dependency group that was never installed; it
cannot catch one that's installed but *broken* (a half-written editable
``.pth``, say) — that still surfaces at dispatch via
``rskill_runner_node``'s ``ROSRuntimeError``. Set
``OPENRAL_STRICT_POLICY_PROBE=1`` to restore the deep import probe when that
distinction matters.

Adding a new policy family: pass its ``install_groups`` and
``required_imports`` to ``@POLICIES.register`` — the facts live on the
registration, so there is no second table to forget.
``test_every_registered_family_declares_its_deps`` walks the registry.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from collections.abc import Callable, Iterable
from typing import Any

from openral_sim.registry import POLICIES

__all__ = [
    "can_import_policy_family",
    "can_import_policy_manifest",
    "filter_importable_manifests",
    "manifest_install_groups",
    "manifest_install_hint",
    "model_family_install_groups",
    "model_family_install_hint",
    "model_family_required_imports",
    "purge_partial_imports",
]


# The per-family facts (``install_groups``, ``required_imports`` — the
# leaf module(s) whose presence proves the factory clears its import gates,
# mirroring its FIRST heavy import — and an optional ``install_note``) are
# declared on each ``@POLICIES.register(...)`` call and read back here.
# Importing ``openral_sim`` registers every built-in family without importing
# torch (heavy imports live inside the factory bodies).
_STRICT_PROBE_ENV = "OPENRAL_STRICT_POLICY_PROBE"


def _meta_strings(family: str, key: str) -> tuple[str, ...]:
    raw = POLICIES.meta(family).get(key, ())
    return tuple(str(item) for item in raw) if isinstance(raw, (tuple, list)) else ()


def model_family_install_hint(family: str) -> str:
    """Return an actionable install command for a given model_family.

    Derived from the family's registered ``install_groups`` (plus its
    ``install_note``). Falls back to a generic hint when the family is
    unknown — better than silence, but the operator still has to map to a
    uv extras group.
    """
    if family not in POLICIES:
        return (
            f"Unknown model_family {family!r}; check the rSkill manifest's "
            "runtime declarations and install the matching uv extras group "
            "(`just sync --all-packages --group <name>`)."
        )
    groups = model_family_install_groups(family)
    hint = (
        "Install the extras: `just sync --all-packages "
        + " ".join(f"--group {g}" for g in groups)
        + "`."
        if groups
        else "No extras required."
    )
    note = POLICIES.meta(family).get("install_note")
    return f"{hint} {note}" if note else hint


def model_family_install_groups(family: str) -> tuple[str, ...]:
    """Return the ``uv sync --group …`` group names that install ``family``.

    Empty tuple for unknown families (caller should fall back to
    ``model_family_install_hint`` for display) and for families that need
    no extras (``zero`` / ``random``).
    """
    return _meta_strings(family, "install_groups")


def model_family_required_imports(family: str) -> tuple[str, ...]:
    """Return the leaf modules whose presence proves the factory will load.

    Returns an empty tuple for unknown families — the pre-flight probe
    then assumes the family is importable (no false negatives on
    fresh / out-of-tree policies).
    """
    return _meta_strings(family, "required_imports")


def can_import_policy_family(family: str) -> tuple[bool, str | None]:
    """Probe whether ``family``'s policy factory can resolve its imports.

    Resolves each of ``model_family_required_imports(family)`` via
    ``_can_import_modules`` — top-level ``find_spec`` by default,
    or a full import under ``OPENRAL_STRICT_POLICY_PROBE=1``. Returns
    ``(True, None)`` on full success, else ``(False, reason)`` where
    ``reason`` carries the leaf import error. The strict tier also
    purges the partially-loaded module tree from ``sys.modules`` so a
    subsequent call sees the same primary error, not a cascade.
    """
    return _can_import_modules(model_family_required_imports(family))


def _can_import_modules(required: tuple[str, ...]) -> tuple[bool, str | None]:
    """Probe an explicit import set, cheaply by default.

    Resolves only the top-level package of each entry via
    ``importlib.util.find_spec`` unless ``OPENRAL_STRICT_POLICY_PROBE=1``
    is set, in which case the historical deep-import probe runs instead.
    See the module docstring for why the deep probe is not the default.
    """
    if os.environ.get(_STRICT_PROBE_ENV) == "1":
        return _deep_import_probe(required)
    for mod in required:
        top = mod.split(".", 1)[0]
        try:
            found = importlib.util.find_spec(top) is not None
        except (ImportError, ValueError):
            # A top-level name can still raise when the package exists but
            # its loader is unusable (a broken editable install). Treat it
            # exactly as absent — the message is what the operator acts on.
            found = False
        if not found:
            return False, f"ModuleNotFoundError: No module named {top!r}"
    return True, None


def _deep_import_probe(required: tuple[str, ...]) -> tuple[bool, str | None]:
    """Import every entry for real, with partial-import cleanup.

    The historical probe. Opt-in via ``OPENRAL_STRICT_POLICY_PROBE=1``:
    it is the only tier that catches an installed-but-broken dependency,
    and it costs ~6.6 s the first time it touches ``lerobot.policies``.
    """
    for mod in required:
        try:
            importlib.import_module(mod)
        except ImportError as exc:
            # Drop the half-baked tree so other code paths that retry
            # the same import don't get the cascade variant. NOTE:
            # ``torch`` is intentionally NOT purged — its C++ side
            # holds process-global state that breaks (``INTERNAL ASSERT
            # FAILED at DynamicTypes.cpp``) when the Python module is
            # removed from ``sys.modules``. ``lerobot`` and
            # ``transformers`` are pure-Python at the import edge and
            # safe to purge.
            purge_partial_imports(("lerobot", "transformers", mod.split(".", 1)[0]))
            return False, f"{type(exc).__name__}: {exc}"
    return True, None


def can_import_policy_manifest(manifest: Any) -> tuple[bool, str | None]:
    """Probe the runtime of ``manifest.model_family``."""
    return can_import_policy_family(getattr(manifest, "model_family", None) or "")


def manifest_install_groups(manifest: Any) -> tuple[str, ...]:
    """Return dependency groups for the manifest's ``model_family``."""
    return model_family_install_groups(getattr(manifest, "model_family", None) or "")


def manifest_install_hint(manifest: Any) -> str:
    """Return the install hint for the manifest's ``model_family``."""
    return model_family_install_hint(getattr(manifest, "model_family", None) or "")


def filter_importable_manifests(
    manifests: Iterable[Any],
    *,
    log_fn: Callable[[str], None] | None = None,
) -> list[Any]:
    """Return the subset of ``manifests`` whose policy family can be imported.

    Each manifest is expected to expose ``.model_family`` and ``.name``
    (matches ``openral_core.RSkillManifest``). Dropped manifests
    are reported via ``log_fn`` (e.g. ``self.get_logger().warning``)
    with an actionable install hint.

    Manifests whose ``model_family`` is not registered in
    ``openral_sim.POLICIES`` are kept unchanged (unknown
    families are assumed importable — better to surface a clearer
    runtime error from the factory than to drop a manifest the
    operator may want).
    """
    kept: list[Any] = []
    for manifest in manifests:
        family = getattr(manifest, "model_family", None) or ""
        ok, reason = can_import_policy_manifest(manifest)
        if ok:
            kept.append(manifest)
            continue
        if log_fn is not None:
            name = getattr(manifest, "name", "<unknown>")
            hint = manifest_install_hint(manifest)
            log_fn(f"palette: dropping rSkill {name!r} (model_family={family!r}): {reason}. {hint}")
    return kept


def purge_partial_imports(prefixes: tuple[str, ...]) -> None:
    """Drop modules under ``prefixes`` from ``sys.modules`` after a failed import.

    Python caches a partially-imported module in ``sys.modules`` even
    when the ``import`` raised. Subsequent imports then see the stale
    module and fail with ``cannot import name 'X'`` instead of the
    original ``ModuleNotFoundError``. This helper purges those entries
    so the next attempt hits the original error message (which the
    operator can actually act on).
    """
    for name in list(sys.modules):
        if any(name == p or name.startswith(p + ".") for p in prefixes):
            sys.modules.pop(name, None)
