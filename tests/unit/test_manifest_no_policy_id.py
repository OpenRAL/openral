"""Regression test: the historical ``policy_id`` field is gone.

The rSkill self-containment audit F1 follow-up dropped the
redundant ``policy_id`` field on :class:`RSkillManifest` — every reader
now dispatches the policy adapter via ``manifest.model_family``.
Carrying a stale ``policy_id:`` line on a user-side manifest would
silently parse before the change (str-typed unused field); after the
change it must be rejected so the rename is noisy, not silent.
"""

from __future__ import annotations

import pytest
import yaml
from openral_core import RSkillManifest
from openral_rskill.loader import discover_intree_rskills
from pydantic import ValidationError


def test_policy_id_not_a_schema_field() -> None:
    """The ``policy_id`` attribute must be removed from RSkillManifest."""
    assert "policy_id" not in RSkillManifest.model_fields, (
        "RSkillManifest.policy_id should be removed; dispatch is via "
        "model_family. If you need to re-introduce it, update the loader / "
        "CLI readers in the same PR."
    )


def test_manifest_with_policy_id_is_rejected() -> None:
    """A user-side manifest still carrying ``policy_id:`` must fail loudly."""
    base = """
schema_version: "0.1"
name: openral/rskill-test-no-policy-id
version: "0.1.0"
license: apache-2.0
role: s1
model_family: smolvla
embodiment_tags: [so100_follower]
runtime: pytorch
weights_uri: hf://test/skill
chunk_size: 16
latency_budget: {per_chunk_ms: 100.0}
actuators_required:
  - kind: joint_position
    control_mode_semantics: {mode: absolute}
processors:
  preprocessor_uri: hf://test/skill/policy_preprocessor.json
  postprocessor_uri: hf://test/skill/policy_postprocessor.json
description: regression test for dropped policy_id field
actions:
  - generalist
policy_id: smolvla
"""
    with pytest.raises(ValidationError, match="policy_id"):
        RSkillManifest.model_validate(yaml.safe_load(base))


def test_every_intree_manifest_lacks_policy_id() -> None:
    """No in-tree manifest may carry the dropped field."""
    manifests = list(discover_intree_rskills())
    assert manifests, "no in-tree manifests discovered"
    stale: list[str] = []
    for name, _ in manifests:
        # Re-parse the raw YAML to catch the literal presence (pydantic strips
        # unknown keys silently in some configurations, but extra="forbid"
        # should raise — this is belt-and-suspenders).
        from pathlib import Path

        repo_root = Path(__file__).resolve().parents[2]
        text = (repo_root / "rskills" / name / "rskill.yaml").read_text()
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("policy_id:") and not stripped.startswith("#"):
                stale.append(name)
                break
    assert not stale, f"manifests still carry policy_id: {stale}"
