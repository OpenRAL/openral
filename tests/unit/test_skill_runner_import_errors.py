"""Tests for the rSkill-factory ImportError translation in rskill_runner_node.

The runtime path in ``openral deploy sim`` calls
``_build_runtime_skill_from_manifest`` which delegates to
``openral_sim.factory.make_policy``. Most policy factories live behind
opt-in extras groups (``sim`` / ``libero`` / ``metaworld`` /
``robocasa``) — ``transformers``, ``bitsandbytes``, ``lerobot[…]``, etc.
When that group isn't installed the factory raises ``ImportError``
deep inside lerobot, which (a) is confusing to surface and (b) leaves
partially-loaded modules in ``sys.modules`` so the next call fails
with a *different* ``cannot import name 'X'`` error.

These tests cover the two helpers that translate that error into an
actionable ``ROSRuntimeError`` + purge stale module state.
"""

from __future__ import annotations

import sys

import pytest

# The module under test depends on rclpy / openral_msgs being importable
# (because it bundles the full lifecycle node). Skip cleanly when those
# aren't sourced — the helpers we exercise are still defined at module
# top-level, just unreachable.
pytest.importorskip("rclpy")
pytest.importorskip("openral_msgs")

from openral_core.exceptions import ROSRuntimeError
from openral_rskill_ros.rskill_runner_node import (
    _build_runtime_skill_from_manifest,
)
from openral_sim.policy_deps import (
    _FAMILY_INSTALL_HINTS,
    model_family_install_hint,
    purge_partial_imports,
)


def test_known_model_families_get_concrete_install_hints() -> None:
    """Every family in the canonical set lists a `just sync` command."""
    for family in ("smolvla", "pi05", "act", "diffusion", "xvla"):
        hint = model_family_install_hint(family)
        assert "just sync" in hint, hint
        assert family in _FAMILY_INSTALL_HINTS


def test_unknown_model_family_falls_back_to_generic_hint() -> None:
    """Unfamiliar families still get a hint pointing at the manifest."""
    hint = model_family_install_hint("imaginary_vla")
    assert "imaginary_vla" in hint
    assert "just sync" in hint
    assert "rSkill manifest" in hint


def test_purge_partial_imports_drops_matching_modules() -> None:
    """The cascade-fix helper removes the named prefix from ``sys.modules``."""
    sys.modules["__test_purge_target__"] = object()  # type: ignore[assignment]
    sys.modules["__test_purge_target__.child"] = object()  # type: ignore[assignment]
    sys.modules["__test_purge_keep__"] = object()  # type: ignore[assignment]
    try:
        purge_partial_imports(("__test_purge_target__",))
        assert "__test_purge_target__" not in sys.modules
        assert "__test_purge_target__.child" not in sys.modules
        # Unrelated modules survive.
        assert "__test_purge_keep__" in sys.modules
    finally:
        sys.modules.pop("__test_purge_target__", None)
        sys.modules.pop("__test_purge_target__.child", None)
        sys.modules.pop("__test_purge_keep__", None)


def test_build_runtime_skill_translates_import_error(
    tmp_path: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``ImportError`` from the policy factory → ``ROSRuntimeError`` with hint.

    Drives the real ``_build_runtime_skill_from_manifest`` against a
    minimal in-tree rSkill manifest, with ``make_policy`` monkey-patched
    to raise ``ImportError`` (the deep-lerobot symptom the operator
    would actually see). Asserts the translation hits — no stack trace
    from a missing transformers dep, just the install hint.
    """
    from pathlib import Path

    import openral_sim.factory as _sim_factory

    # Real in-tree manifest — smolvla family. The CLI never gets far
    # enough to need the weights; we only exercise the factory wrapper.
    repo_root = Path(__file__).resolve().parents[2]
    yaml_path = repo_root / "rskills" / "smolvla-libero" / "rskill.yaml"
    if not yaml_path.is_file():
        pytest.skip(f"missing in-tree fixture: {yaml_path}")

    def _raise_import_error(_env_cfg: object) -> object:
        raise ImportError("No module named 'transformers'")

    monkeypatch.setattr(_sim_factory, "make_policy", _raise_import_error)

    with pytest.raises(ROSRuntimeError) as ei:
        _build_runtime_skill_from_manifest(
            yaml_path=yaml_path,
            prompt="pick up the cup",
            scene_cameras=(),
        )
    msg = str(ei.value)
    assert "smolvla" in msg
    assert "transformers" in msg
    assert "just sync --all-packages --group sim" in msg
