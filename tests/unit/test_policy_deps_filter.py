"""Tests for ``openral_sim.policy_deps`` — pre-flight palette filter.

The reasoner calls ``filter_importable_manifests`` at
``on_configure`` to drop rSkills whose ``model_family`` lives behind
an extras group that isn't installed in this venv. The skill_runner
calls ``model_family_install_hint`` to translate runtime
``ImportError`` into actionable error messages.

Both contracts read the facts each family declares on its
``@POLICIES.register(...)`` call — a family registered without them
fails these tests rather than at the operator's first ``openral deploy
sim``.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

import pytest
from openral_sim.policy_deps import (
    can_import_policy_family,
    can_import_policy_manifest,
    filter_importable_manifests,
    manifest_install_groups,
    manifest_install_hint,
    model_family_install_groups,
    model_family_required_imports,
    purge_partial_imports,
)
from openral_sim.registry import POLICIES


def _fake_family(monkeypatch: pytest.MonkeyPatch, name: str, **meta: object) -> None:
    """Register a throwaway family for one test (undone by monkeypatch)."""
    monkeypatch.setitem(POLICIES._items, name, lambda _env: None)
    monkeypatch.setitem(POLICIES._meta, name, meta)


@dataclass
class _StubManifest:
    """Minimum-shape stand-in for ``openral_core.RSkillManifest``."""

    name: str
    model_family: str
    policy_extras: dict[str, object] | None = None


def test_every_registered_family_declares_its_deps() -> None:
    """Each registered policy carries both facts, so no family is silently treated as
    always-importable. Regression: molmoact2 / openvla / lingbot_vla had no entry in
    the old parallel dicts, and ``mock`` was keyed though the ids are zero/random."""
    missing = [
        name
        for name in POLICIES.names()
        if not {"install_groups", "required_imports"} <= set(POLICIES.meta(name))
    ]
    assert missing == [], f"families registered without dependency facts: {missing}"
    for name in POLICIES.names():
        if model_family_install_groups(name):
            assert model_family_required_imports(name), name
            assert "just sync --all-packages" in manifest_install_hint(
                _StubManifest(name="x", model_family=name)
            )


def test_model_family_install_groups_returns_empty_for_unknown() -> None:
    """Unknown families produce an empty tuple — caller falls back to the hint string."""
    assert model_family_install_groups("not_a_real_family") == ()
    # The no-op policies are deliberately empty (no extras needed).
    assert model_family_install_groups("zero") == ()
    assert model_family_install_groups("random") == ()


def test_model_family_install_groups_returns_uv_groups_for_known_families() -> None:
    """pi05 needs both sim + libero; rldx needs only rldx; act / smolvla / xvla need sim."""
    assert set(model_family_install_groups("pi05")) == {"sim", "libero"}
    assert set(model_family_install_groups("rldx")) == {"rldx"}
    assert set(model_family_install_groups("xr1")) == {"sidecar-wire"}
    for fam in ("smolvla", "act", "diffusion", "xvla", "openvla"):
        assert set(model_family_install_groups(fam)) == {"sim"}
    assert model_family_install_groups("molmoact2") == ("libero",)
    assert model_family_install_groups("lingbot_vla") == ("lingbot",)


def test_model_family_required_imports_returns_empty_for_unknown() -> None:
    """Unknown families return ``()`` so the probe assumes importable."""
    assert model_family_required_imports("imaginary_vla") == ()


@pytest.mark.parametrize("family", ["smolvla", "pi05", "act", "diffusion", "xvla"])
def test_lerobot_family_probes_the_real_adapter_module(family: str) -> None:
    """Each family probes the exact module ``openral_sim.policies.<family>`` loads
    (``lerobot.policies.<family>.modeling_<family>``), not a bare top-level name.
    Regression for ``xvla``, which mapped to ``("xvla",)`` and got dropped from the
    deploy-sim palette ("No module named 'xvla'") despite running fine via ``openral sim run``.
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
    _fake_family(monkeypatch, "_test_stdlib", required_imports=("json", "math"))
    ok, reason = can_import_policy_family("_test_stdlib")
    assert ok is True
    assert reason is None


def test_can_import_policy_family_fails_for_missing_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing required import → ``(False, ImportError reason)`` + purges sys.modules."""
    _fake_family(
        monkeypatch, "_test_missing", required_imports=("__definitely_not_a_real_module__",)
    )
    ok, reason = can_import_policy_family("_test_missing")
    assert ok is False
    assert reason is not None
    assert "ModuleNotFoundError" in reason or "ImportError" in reason
    assert "__definitely_not_a_real_module__" in reason


def test_filter_importable_manifests_keeps_known_good() -> None:
    """A manifest whose family probes OK survives the filter."""
    kept = filter_importable_manifests([_StubManifest(name="x", model_family="zero")])
    assert len(kept) == 1
    assert kept[0].name == "x"


def test_filter_importable_manifests_drops_unimportable_and_calls_logger(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unimportable manifests are dropped and the logger is told why + how."""
    _fake_family(
        monkeypatch,
        "_test_broken",
        install_groups=("broken",),
        required_imports=("__definitely_not_a_real_module__",),
    )
    logged: list[str] = []
    kept = filter_importable_manifests(
        [
            _StubManifest(name="ok-skill", model_family="zero"),
            _StubManifest(name="broken-skill", model_family="_test_broken"),
        ],
        log_fn=logged.append,
    )
    assert [m.name for m in kept] == ["ok-skill"]
    assert len(logged) == 1
    assert "broken-skill" in logged[0]
    assert "just sync --all-packages --group broken" in logged[0]
    assert "_test_broken" in logged[0]


def test_filter_importable_manifests_keeps_unknown_families() -> None:
    """Unknown families pass the filter — better to surface a runtime error than drop silently."""
    kept = filter_importable_manifests(
        [_StubManifest(name="exotic", model_family="some_new_family")]
    )
    assert len(kept) == 1


def test_behavior_groot_is_its_own_family() -> None:
    """The B1K checkpoint is the registered ``gr00t_b1k`` family (it used to hide
    inside ``gr00t`` behind ``policy_extras.implementation``), so its dependency
    facts come from the same registry as every other family. Only the
    environment-independent naming contract is asserted; the real import outcome
    needs the opt-in ``behavior-groot`` group (see ``pytest.importorskip``).
    """
    from openral_rskill.loader import load_rskill_manifest

    manifest = load_rskill_manifest("rskills/gr00t-n17-b1k-turning-on-radio")
    assert manifest.model_family == "gr00t_b1k"
    assert manifest_install_groups(manifest) == ("behavior-groot",)
    assert "behavior-groot" in manifest_install_hint(manifest)

    pytest.importorskip("zmq", reason="behavior-groot's real probe needs the sidecar-wire group")
    pytest.importorskip(
        "msgpack", reason="behavior-groot's real probe needs the sidecar-wire group"
    )
    ok, reason = can_import_policy_manifest(manifest)
    assert ok is True
    assert reason is None


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


def test_fast_probe_does_not_import_the_deep_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The default probe resolves only the top-level package.

    ``lerobot/policies/__init__`` eagerly imports every family's config class, so
    resolving ``lerobot.policies.<x>.modeling_<x>`` drags in the entire tree (6.6 s,
    measured) — CLI preflight and the reasoner's palette seed paid that on every
    deploy though only ``runtime_node`` needs the deep import. Uses a stdlib module
    as stand-in so the assertion holds without the lerobot extras installed.
    """
    deep = "email.mime.audio"
    _fake_family(monkeypatch, "_test_deep", required_imports=(deep,))
    purge_partial_imports((deep,))

    ok, reason = can_import_policy_family("_test_deep")

    assert ok is True
    assert reason is None
    assert deep not in sys.modules, (
        f"{deep} was imported by the default probe; the fast tier must resolve "
        "only the top-level package."
    )


def test_strict_probe_env_restores_the_deep_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``OPENRAL_STRICT_POLICY_PROBE=1`` opts back into the full import.

    The strict tier is the only one that catches an installed-but-broken
    dependency, so it has to stay reachable.
    """
    deep = "email.mime.audio"
    _fake_family(monkeypatch, "_test_deep", required_imports=(deep,))
    monkeypatch.setenv("OPENRAL_STRICT_POLICY_PROBE", "1")
    purge_partial_imports((deep,))

    ok, reason = can_import_policy_family("_test_deep")

    assert ok is True
    assert reason is None
    assert deep in sys.modules


def test_fast_probe_rejects_a_missing_top_level_of_a_dotted_requirement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A dotted requirement whose root is absent still fails, with the root named."""
    _fake_family(
        monkeypatch,
        "_test_missing_top",
        required_imports=("__definitely_not_a_real_module__.policies.modeling",),
    )

    ok, reason = can_import_policy_family("_test_missing_top")

    assert ok is False
    assert reason is not None
    assert "__definitely_not_a_real_module__" in reason


def test_gr00t_probe_covers_the_lazy_diffusers_import() -> None:
    """``diffusers`` loads lazily inside GrootPolicy's build, not via ``modeling_groot`` —
    probing only the modeling module let gr00t rSkills into the palette on hosts missing
    the gr00t extras, and dispatch aborted at runtime ("'diffusers' is required but not
    installed"; observed deploy-sim 2026-07-20)."""
    assert "diffusers" in model_family_required_imports("gr00t")


def test_xr1_probe_uses_only_the_sidecar_wire() -> None:
    """XR-1's incompatible torch stack stays outside the workspace."""
    assert model_family_required_imports("xr1") == ("zmq", "msgpack")
