"""Typo guard for the open rSkill registry ids (CI, not the core schema).

``EmbodimentTag`` / ``ModelFamily`` / ``BenchmarkName`` / ``StateLayout`` are
open, pattern-validated strings in ``openral_core.schemas`` so adding a robot,
policy family, benchmark suite or state layout needs no core-schema edit. The
closed-``Literal`` typo protection moves here: every in-tree
``rskills/*/rskill.yaml`` must name ids that some registry actually defines.

CLAUDE.md §1.11 — real manifests, real ``robots/*/robot.yaml``, real
``benchmarks/*.yaml``, the real ``openral_sim.POLICIES`` and
``openral_state_adapter`` registries. No placeholders.
"""

from __future__ import annotations

from typing import Any

import pytest
from openral_core import CANONICAL_ROBOT_NAME_TOKENS, WRAPPED_TASK_SPACE_LAYOUTS, RSkillManifest
from openral_core.schemas import _EMBODIMENT_TO_ROBOT_TOKEN
from openral_rskill.loader import (
    discover_intree_rskills,
    intree_embodiment_tags,
    known_benchmark_ids,
)
from openral_sim import POLICIES
from openral_sim.policies.rldx import _RLDX_LAYOUT_TO_EMBODIMENT_TAG
from openral_state_adapter import registered_layouts

# ``rskills/template/`` is the scaffolder stub; its ids are placeholders.
_MANIFESTS: list[tuple[str, RSkillManifest]] = [
    (d, m) for d, m in discover_intree_rskills() if d != "template"
]
_IDS = [d for d, _ in _MANIFESTS]
_ROBOT_TAGS = intree_embodiment_tags()  # parses every robots/*/robot.yaml once
_BENCHMARKS = known_benchmark_ids()


def test_manifests_discovered() -> None:
    assert _MANIFESTS, "no in-tree rskills/*/rskill.yaml discovered"


@pytest.mark.parametrize(("skill", "manifest"), _MANIFESTS, ids=_IDS)
def test_embodiment_tags_are_declared_by_a_robot(skill: str, manifest: RSkillManifest) -> None:
    unknown = set(manifest.embodiment_tags) - _ROBOT_TAGS
    assert not unknown, (
        f"{skill}: embodiment_tags {sorted(unknown)} are declared by no robots/*/robot.yaml "
        "(and are not any/custom/multi) — the loader's compat check would never match"
    )


@pytest.mark.parametrize(("skill", "manifest"), _MANIFESTS, ids=_IDS)
def test_robot_name_token_is_a_known_robot(skill: str, manifest: RSkillManifest) -> None:
    """``repo_name_is_canonical`` shape-checks ``<robot>``; the vocabulary check lives here."""
    parts = manifest.name.split("/", 1)[-1].split("-")
    if manifest.kind == "playbook" or len(parts) < 3:
        return
    robot_tokens = {
        _EMBODIMENT_TO_ROBOT_TOKEN.get(t, t) for t in _ROBOT_TAGS
    } | CANONICAL_ROBOT_NAME_TOKENS
    assert parts[2] in robot_tokens, f"{skill}: <robot> token {parts[2]!r} names no known robot"


@pytest.mark.parametrize(("skill", "manifest"), _MANIFESTS, ids=_IDS)
def test_model_family_is_a_registered_policy(skill: str, manifest: RSkillManifest) -> None:
    if manifest.model_family is None:
        return
    assert manifest.model_family in POLICIES, (
        f"{skill}: model_family {manifest.model_family!r} is not registered in "
        f"openral_sim.POLICIES ({sorted(POLICIES)})"
    )


@pytest.mark.parametrize(("skill", "manifest"), _MANIFESTS, ids=_IDS)
def test_benchmark_keys_exist_on_disk(skill: str, manifest: RSkillManifest) -> None:
    assert _BENCHMARKS is not None, "no checkout found — this guard must run in-tree"
    unknown = set(manifest.benchmarks) - _BENCHMARKS
    assert not unknown, (
        f"{skill}: benchmarks keys {sorted(unknown)} are neither a benchmarks/*.yaml suite "
        "nor a scenes/benchmark/*.yaml id"
    )


@pytest.mark.parametrize(("skill", "manifest"), _MANIFESTS, ids=_IDS)
def test_state_layout_has_a_consumer(skill: str, manifest: RSkillManifest) -> None:
    """A layout is real if some code assembles or consumes it.

    Task-space layouts (``WRAPPED_TASK_SPACE_LAYOUTS``) need a deploy-time
    assembler in ``openral_state_adapter``. Joint-space layouts have no
    assembler (the runner serves ``JointState`` verbatim); their only named
    consumer is the sim-only RLDX sidecar's layout map, which is the honest
    registry for them (the openvla SimplerEnv manifest reuses ``simpler_widowx``
    as a joint-space shape tag).
    """
    sc = manifest.state_contract
    if sc is None or sc.layout is None:
        return
    if sc.layout in WRAPPED_TASK_SPACE_LAYOUTS:
        assert sc.layout in registered_layouts(), (
            f"{skill}: task-space layout {sc.layout!r} has no openral_state_adapter assembler"
        )
        return
    known = registered_layouts() | set(_RLDX_LAYOUT_TO_EMBODIMENT_TAG)
    assert sc.layout in known, (
        f"{skill}: state_contract.layout {sc.layout!r} has no assembler and no sim adapter "
        f"consumes it (known: {sorted(known)})"
    )


def test_hub_manifest_with_unknown_benchmark_warns_not_fails(cap: Any) -> None:
    """``rSkill.from_pretrained`` WARNS on a suite this checkout lacks (it may be newer)."""
    from openral_rskill.loader import _warn_unknown_benchmarks

    _, real = next((d, m) for d, m in _MANIFESTS if m.benchmarks)
    newer = real.model_copy(update={"benchmarks": {**real.benchmarks, "suite_from_future": 0.5}})
    _warn_unknown_benchmarks(newer, source="hf://OpenRAL/example")  # must not raise
    [event] = cap.named("rskill.unknown_benchmarks")
    assert event["unknown"] == ["suite_from_future"]
