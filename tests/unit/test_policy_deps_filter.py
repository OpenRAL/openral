"""Tests for ``openral_sim.policy_deps`` — pre-flight palette filter.

The reasoner calls :func:`filter_importable_manifests` at
``on_configure`` to drop rSkills whose ``model_family`` lives behind
an extras group that isn't installed in this venv. The skill_runner
calls :func:`model_family_install_hint` to translate runtime
``ImportError`` into actionable error messages.

Both contracts share :data:`_FAMILY_INSTALL_HINTS` /
:data:`_FAMILY_REQUIRED_IMPORTS` — a half-registered family (one dict
updated but not the other) fails these tests rather than at the
operator's first ``openral deploy sim``.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

import pytest
from openral_sim.policy_deps import (
    _FAMILY_INSTALL_GROUPS,
    _FAMILY_INSTALL_HINTS,
    _FAMILY_REQUIRED_IMPORTS,
    can_import_policy_family,
    filter_importable_manifests,
    model_family_install_groups,
    model_family_required_imports,
    purge_partial_imports,
)


@dataclass
class _StubManifest:
    """Minimum-shape stand-in for :class:`openral_core.RSkillManifest`."""

    name: str
    model_family: str


def test_install_hints_and_required_imports_cover_the_same_families() -> None:
    """Catch half-registered families at unit-test time.

    A new family must land in BOTH dicts, otherwise the reasoner's
    pre-flight filter will keep it but the skill_runner won't have a
    useful install hint to surface (or vice versa).
    """
    hint_keys = set(_FAMILY_INSTALL_HINTS)
    import_keys = set(_FAMILY_REQUIRED_IMPORTS)
    groups_keys = set(_FAMILY_INSTALL_GROUPS)
    assert hint_keys == import_keys == groups_keys, (
        f"families in _FAMILY_INSTALL_HINTS but not _FAMILY_REQUIRED_IMPORTS: "
        f"{hint_keys - import_keys}; "
        f"families in _FAMILY_REQUIRED_IMPORTS but not _FAMILY_INSTALL_HINTS: "
        f"{import_keys - hint_keys}; "
        f"families missing from _FAMILY_INSTALL_GROUPS: "
        f"{(hint_keys | import_keys) - groups_keys}"
    )


def test_model_family_install_groups_returns_empty_for_unknown() -> None:
    """Unknown families produce an empty tuple — caller falls back to the hint string."""
    assert model_family_install_groups("not_a_real_family") == ()
    # Mock family is deliberately empty (no extras needed).
    assert model_family_install_groups("mock") == ()


def test_model_family_install_groups_returns_uv_groups_for_known_families() -> None:
    """pi05 needs both sim + libero; rldx needs only rldx; act / smolvla / xvla need sim."""
    assert set(model_family_install_groups("pi05")) == {"sim", "libero"}
    assert set(model_family_install_groups("rldx")) == {"rldx"}
    for fam in ("smolvla", "act", "diffusion", "xvla"):
        assert set(model_family_install_groups(fam)) == {"sim"}


def test_model_family_required_imports_returns_empty_for_unknown() -> None:
    """Unknown families return ``()`` so the probe assumes importable."""
    assert model_family_required_imports("imaginary_vla") == ()


@pytest.mark.parametrize("family", ["smolvla", "pi05", "act", "diffusion", "xvla"])
def test_lerobot_family_probes_the_real_adapter_module(family: str) -> None:
    """Each in-tree lerobot policy family probes the module its adapter imports.

    ``openral_sim.policies.<family>`` loads
    ``lerobot.policies.<family>.modeling_<family>``, so the import-deps
    probe must check that exact module — not a bare top-level name that
    doesn't exist. Regression for ``xvla``, which mapped to ``("xvla",)``
    and so got dropped from the deploy-sim reasoner palette ("No module
    named 'xvla'") even though it runs fine via ``openral sim run``.
    """
    imports = model_family_required_imports(family)
    expected = f"lerobot.policies.{family}.modeling_{family}"
    assert expected in imports, (
        f"{family}: required-imports {imports} must include {expected!r} "
        f"(the module openral_sim.policies.{family} imports at load time)."
    )


def test_can_import_policy_family_succeeds_for_stdlib_smoke(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A family whose required-imports are all stdlib modules probes OK."""
    monkeypatch.setitem(_FAMILY_REQUIRED_IMPORTS, "_test_stdlib", ("json", "math"))
    ok, reason = can_import_policy_family("_test_stdlib")
    assert ok is True
    assert reason is None


def test_can_import_policy_family_fails_for_missing_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing required import → ``(False, ImportError reason)`` + purges sys.modules."""
    monkeypatch.setitem(
        _FAMILY_REQUIRED_IMPORTS,
        "_test_missing",
        ("__definitely_not_a_real_module__",),
    )
    ok, reason = can_import_policy_family("_test_missing")
    assert ok is False
    assert reason is not None
    assert "ModuleNotFoundError" in reason or "ImportError" in reason
    assert "__definitely_not_a_real_module__" in reason


def test_filter_importable_manifests_keeps_known_good() -> None:
    """A manifest whose family probes OK survives the filter."""
    kept = filter_importable_manifests([_StubManifest(name="x", model_family="mock")])
    assert len(kept) == 1
    assert kept[0].name == "x"


def test_filter_importable_manifests_drops_unimportable_and_calls_logger(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unimportable manifests are dropped and the logger is told why + how."""
    monkeypatch.setitem(
        _FAMILY_INSTALL_HINTS,
        "_test_broken",
        "Install the broken extras: `uv sync --group broken`.",
    )
    monkeypatch.setitem(
        _FAMILY_REQUIRED_IMPORTS,
        "_test_broken",
        ("__definitely_not_a_real_module__",),
    )
    logged: list[str] = []
    kept = filter_importable_manifests(
        [
            _StubManifest(name="ok-skill", model_family="mock"),
            _StubManifest(name="broken-skill", model_family="_test_broken"),
        ],
        log_fn=logged.append,
    )
    assert [m.name for m in kept] == ["ok-skill"]
    assert len(logged) == 1
    assert "broken-skill" in logged[0]
    assert "uv sync --group broken" in logged[0]
    assert "_test_broken" in logged[0]


def test_filter_importable_manifests_keeps_unknown_families() -> None:
    """Unknown families pass the filter — better to surface a runtime error than drop silently."""
    kept = filter_importable_manifests(
        [_StubManifest(name="exotic", model_family="some_new_family")]
    )
    assert len(kept) == 1


def test_purge_partial_imports_drops_only_matching_prefixes() -> None:
    """Belt-and-braces: ``can_import_policy_family`` purges on failure."""
    sys.modules["__pd_test_target__"] = object()  # type: ignore[assignment]
    sys.modules["__pd_test_target__.sub"] = object()  # type: ignore[assignment]
    sys.modules["__pd_test_keep__"] = object()  # type: ignore[assignment]
    try:
        purge_partial_imports(("__pd_test_target__",))
        assert "__pd_test_target__" not in sys.modules
        assert "__pd_test_target__.sub" not in sys.modules
        assert "__pd_test_keep__" in sys.modules
    finally:
        for k in ("__pd_test_target__", "__pd_test_target__.sub", "__pd_test_keep__"):
            sys.modules.pop(k, None)
