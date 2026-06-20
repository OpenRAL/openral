"""Policy-family dependency probing — shared by reasoner + skill_runner.

The policy factories in :mod:`openral_sim.policies` live behind opt-in
extras groups (``sim`` / ``libero`` / ``metaworld`` / ``robocasa``):
``transformers``, ``bitsandbytes``, ``lerobot[…]``, etc. When the right
group isn't installed the factory raises ``ImportError`` deep inside
lerobot, which (a) is confusing to surface and (b) leaves
partially-loaded modules in ``sys.modules`` so subsequent calls fail
with a *different* ``cannot import name 'X'`` cascade error.

This module is the single source of truth for two related contracts:

* :func:`model_family_install_hint` — the actionable uv-sync command for
  each known family. Used by :mod:`openral_rskill_ros.rskill_runner_node`
  when translating a factory ``ImportError`` into ``ROSRuntimeError``.
* :func:`can_import_policy_family` /
  :func:`filter_importable_manifests` — pre-flight probes used by the
  reasoner at ``on_configure`` to drop rSkills whose deps aren't
  installed *before* the palette is built. This way the operator sees
  one warning at boot ("dropped X: missing transformers; run
  ``just sync --all-packages --group sim``"), not a confusing per-tick
  failure at goal dispatch time.

All install commands recommended by this module go through
``just sync --all-packages --group <X>`` rather than bare
``uv sync --group <X>``. ``--all-packages`` is required so the
workspace members (openral-core, openral-cli, …) survive the install
— ``uv sync`` without it would uninstall every workspace member and
the next ROS launch would fail with ``No module named 'openral_core'``.
``just sync`` additionally repairs the ``hf-libero==0.1.3``
distutils-uninstall trap before+after the sync.

The probe never instantiates a factory or loads weights — it only
imports the deepest lerobot/transformers module each family touches.
That's fast (≈100 ms for the lerobot tree on a warm filesystem) and
deterministic.

Adding a new policy family: register a new entry in
:data:`_FAMILY_REQUIRED_IMPORTS` AND :data:`_FAMILY_INSTALL_HINTS`.
The reasoner's ``test_reasoner_palette_filters_unimportable_families``
test (and ``test_known_model_families_get_concrete_install_hints``)
walks both dicts so a half-registered family fails at unit-test time.
"""

from __future__ import annotations

import importlib
import sys
from collections.abc import Callable, Iterable
from typing import Any

__all__ = [
    "can_import_policy_family",
    "filter_importable_manifests",
    "model_family_install_groups",
    "model_family_install_hint",
    "model_family_required_imports",
    "purge_partial_imports",
]


# Model-family → ``uv sync`` install hint. Surfaced both at
# pre-flight (reasoner drops the skill) and at runtime (skill_runner
# translates the factory ImportError).
_FAMILY_INSTALL_HINTS: dict[str, str] = {
    "smolvla": (
        "Install the sim extras: `just sync --all-packages --group sim` "
        "(provides transformers + lerobot smolvla deps)."
    ),
    "pi05": (
        "Install the sim + libero extras: `just sync --all-packages "
        "--group sim --group libero` (provides transformers + bitsandbytes "
        "+ lerobot)."
    ),
    "act": "Install the sim extras: `just sync --all-packages --group sim`.",
    "diffusion": "Install the sim extras: `just sync --all-packages --group sim`.",
    "xvla": "Install the sim extras: `just sync --all-packages --group sim`.",
    "rldx": (
        "Install the rldx extras: `just sync --all-packages --group rldx` "
        "(adds pyzmq + msgpack for the RLDX adapter sidecar)."
    ),
    "gr00t": (
        "Install the gr00t extras: `just sync --all-packages --group gr00t` "
        "(adds pyzmq + msgpack for the GR00T sidecar client). The GR00T 3B "
        "policy itself runs in tools/gr00t_sidecar.py's own Python 3.10 venv "
        "(ADR-0046)."
    ),
    "diffuser_actor": (
        "Install the rlbench extras: `just sync --all-packages --group rlbench` "
        "(adds pyzmq + msgpack for the 3D Diffuser Actor sidecar client). The "
        "policy + the CoppeliaSim/PyRep RLBench env run in tools/rlbench_*"
        "_sidecar.py's own externally-provisioned Python 3.10 venv (ADR-0061)."
    ),
    # `mock` has no external deps — included so a smoke that mentions a
    # mock-family rSkill never gets filtered out.
    "mock": "No extras required.",
}


# Model-family → ``uv sync --group …`` group names. Parallel to
# :data:`_FAMILY_INSTALL_HINTS` but machine-readable so callers can
# compose a single ``uv sync`` for the union of missing families
# (deploy_sim's pre-flight prompt joins these across all blocked
# rSkills into one command).
_FAMILY_INSTALL_GROUPS: dict[str, tuple[str, ...]] = {
    "smolvla": ("sim",),
    "pi05": ("sim", "libero"),
    "act": ("sim",),
    "diffusion": ("sim",),
    "xvla": ("sim",),
    "rldx": ("rldx",),
    "gr00t": ("gr00t",),
    "diffuser_actor": ("rlbench",),
    "mock": (),
}


# Model-family → leaf module(s) whose presence proves the factory will
# clear its import gates. Picked to mirror the FIRST import inside each
# policy's factory (e.g.
# ``from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy``
# is what ``_build_smolvla`` does on line 300 of smolvla.py).
_FAMILY_REQUIRED_IMPORTS: dict[str, tuple[str, ...]] = {
    "smolvla": ("transformers", "lerobot.policies.smolvla.modeling_smolvla"),
    "pi05": ("transformers", "bitsandbytes", "lerobot.policies.pi05.modeling_pi05"),
    "act": ("lerobot.policies.act.modeling_act",),
    "diffusion": ("lerobot.policies.diffusion.modeling_diffusion",),
    "xvla": ("lerobot.policies.xvla.modeling_xvla",),
    "rldx": ("zmq", "msgpack"),
    # GR00T shares the rldx out-of-process contract; the openral-side client
    # only needs the ZMQ + msgpack wire (the 3B policy lives in the sidecar's
    # own Py3.10 venv). See openral_sim.policies.gr00t / ADR-0046.
    "gr00t": ("zmq", "msgpack"),
    # 3D Diffuser Actor shares the out-of-process sidecar contract; the
    # openral-side client only needs the ZMQ + msgpack wire (the policy + the
    # CoppeliaSim/PyRep RLBench env live in the sidecar's own py3.10 venv).
    # See openral_sim.policies.rlbench_3dda / ADR-0061.
    "diffuser_actor": ("zmq", "msgpack"),
    "mock": (),
}


def model_family_install_hint(family: str) -> str:
    """Return an actionable install command for a given model_family.

    Falls back to a generic hint when the family is unknown — better
    than silence, but the operator still has to map to a uv extras
    group.
    """
    return _FAMILY_INSTALL_HINTS.get(
        family,
        f"Unknown model_family {family!r}; check the rSkill manifest's "
        "runtime declarations and install the matching uv extras group "
        "(`just sync --all-packages --group <name>`).",
    )


def model_family_install_groups(family: str) -> tuple[str, ...]:
    """Return the ``uv sync --group …`` group names that install ``family``.

    Empty tuple for unknown families (caller should fall back to
    :func:`model_family_install_hint` for display). Empty tuple for
    ``"mock"`` (no extras needed).
    """
    return _FAMILY_INSTALL_GROUPS.get(family, ())


def model_family_required_imports(family: str) -> tuple[str, ...]:
    """Return the leaf modules whose presence proves the factory will load.

    Returns an empty tuple for unknown families — the pre-flight probe
    then assumes the family is importable (no false negatives on
    fresh / out-of-tree policies).
    """
    return _FAMILY_REQUIRED_IMPORTS.get(family, ())


def can_import_policy_family(family: str) -> tuple[bool, str | None]:
    """Probe whether ``family``'s policy factory can resolve its imports.

    Imports each module in :data:`_FAMILY_REQUIRED_IMPORTS[family]` via
    :func:`importlib.import_module`. Returns ``(True, None)`` on full
    success. On the first failure: purges the partially-loaded module
    tree from ``sys.modules`` (so a subsequent call sees the same
    primary error, not a cascade) and returns ``(False, reason)``
    where ``reason`` carries the leaf import error.
    """
    required = model_family_required_imports(family)
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


def filter_importable_manifests(
    manifests: Iterable[Any],
    *,
    log_fn: Callable[[str], None] | None = None,
) -> list[Any]:
    """Return the subset of ``manifests`` whose policy family can be imported.

    Each manifest is expected to expose ``.model_family`` and ``.name``
    (matches :class:`openral_core.RSkillManifest`). Dropped manifests
    are reported via ``log_fn`` (e.g. ``self.get_logger().warning``)
    with an actionable install hint.

    Manifests whose ``model_family`` is not in
    :data:`_FAMILY_REQUIRED_IMPORTS` are kept unchanged (unknown
    families are assumed importable — better to surface a clearer
    runtime error from the factory than to drop a manifest the
    operator may want).
    """
    kept: list[Any] = []
    for manifest in manifests:
        family = getattr(manifest, "model_family", None) or ""
        ok, reason = can_import_policy_family(family)
        if ok:
            kept.append(manifest)
            continue
        if log_fn is not None:
            name = getattr(manifest, "name", "<unknown>")
            hint = model_family_install_hint(family)
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
