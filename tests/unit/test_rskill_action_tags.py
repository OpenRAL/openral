"""ADR-0022 — every in-tree rSkill manifest declares an action vocabulary.

Walks the real ``rskills/*/rskill.yaml`` files (no mocks, no synthetic
manifests) and asserts:

1. Every manifest loads under the new :class:`RSkillManifest` schema
   (the new ``description`` / ``actions`` fields are required; the file
   on disk must satisfy them).
2. Every manifest's ``actions`` is non-empty and resolves into
   :class:`RSkillAction` enum members (closed vocabulary).
3. ``description`` is non-empty and within the schema limit.
4. ``objects`` / ``scenes`` (free-form lists) round-trip as lists of
   strings.

Also exercises :class:`RSkillManifest.actions` with hypothesis to
confirm the closed vocabulary is enforced — anything not in
:class:`RSkillAction` is rejected.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st
from openral_core import RSkillAction, RSkillManifest
from openral_rskill.loader import discover_intree_rskills
from pydantic import ValidationError

_REPO_ROOT = Path(__file__).resolve().parents[2]
_RSKILLS_DIR = _REPO_ROOT / "rskills"


def _manifest_paths() -> list[Path]:
    paths = sorted(_RSKILLS_DIR.glob("*/rskill.yaml"))
    assert paths, f"no rskill manifests found under {_RSKILLS_DIR!s}"
    return paths


@pytest.mark.parametrize("manifest_path", _manifest_paths(), ids=lambda p: p.parent.name)
def test_intree_manifest_loads_with_action_vocabulary(manifest_path: Path) -> None:
    """Every in-tree manifest parses and has non-empty actions / description."""
    manifest = RSkillManifest.from_yaml(str(manifest_path))
    assert manifest.description, (
        f"{manifest_path.parent.name}: description must be a non-empty string"
    )
    assert 1 <= len(manifest.description) <= 500, (
        f"{manifest_path.parent.name}: description length {len(manifest.description)} "
        "out of [1, 500]"
    )
    assert len(manifest.actions) >= 1, (
        f"{manifest_path.parent.name}: at least one action verb required"
    )
    for action in manifest.actions:
        assert isinstance(action, RSkillAction), (
            f"{manifest_path.parent.name}: action {action!r} is not a closed-vocabulary "
            "RSkillAction enum member"
        )
    for obj in manifest.objects:
        assert isinstance(obj, str) and obj, f"{manifest_path.parent.name}: bad object {obj!r}"
    for scene in manifest.scenes:
        assert isinstance(scene, str) and scene, f"{manifest_path.parent.name}: bad scene {scene!r}"


def test_every_intree_rskill_discovered_with_actions() -> None:
    """The shared :func:`discover_intree_rskills` produces actions for every manifest."""
    manifests = list(discover_intree_rskills())
    assert manifests, "no in-tree rskills discovered"
    for name, manifest in manifests:
        assert manifest.actions, f"{name}: actions is empty"
        assert manifest.description, f"{name}: description is empty"


def test_specialist_skill_actions_match_curation() -> None:
    """A few representative manifests carry the expected verbs.

    Anchors against the curated values committed to the in-tree manifests so a
    drive-by edit that strips action tags is caught. Uses the real on-disk
    manifest (no mocks).
    """
    pusht = RSkillManifest.from_yaml(str(_RSKILLS_DIR / "diffusion-pusht" / "rskill.yaml"))
    assert pusht.actions == [RSkillAction.PUSH]
    assert "t_shape" in pusht.objects

    aloha_insert = RSkillManifest.from_yaml(
        str(_RSKILLS_DIR / "act-aloha-insertion" / "rskill.yaml"),
    )
    assert RSkillAction.INSERT in aloha_insert.actions
    assert "peg" in aloha_insert.objects

    generalist = RSkillManifest.from_yaml(str(_RSKILLS_DIR / "rldx1-pt-nf4" / "rskill.yaml"))
    assert generalist.actions == [RSkillAction.GENERALIST]


@given(st.text(min_size=1, max_size=32).filter(lambda s: s not in {a.value for a in RSkillAction}))
def test_unknown_action_string_rejected(unknown: str) -> None:
    """A string not in the closed vocabulary must fail schema validation."""
    # Load a real in-tree manifest's dict, then poison the actions field.
    template = RSkillManifest.from_yaml(str(_RSKILLS_DIR / "template" / "rskill.yaml"))
    raw = template.model_dump()
    raw["actions"] = [unknown]
    with pytest.raises(ValidationError):
        RSkillManifest.model_validate(raw)


def test_empty_actions_rejected() -> None:
    """``actions`` is required with min_length=1."""
    template = RSkillManifest.from_yaml(str(_RSKILLS_DIR / "template" / "rskill.yaml"))
    raw = template.model_dump()
    raw["actions"] = []
    with pytest.raises(ValidationError, match="actions"):
        RSkillManifest.model_validate(raw)


def test_empty_description_rejected() -> None:
    """``description`` is required with min_length=1."""
    template = RSkillManifest.from_yaml(str(_RSKILLS_DIR / "template" / "rskill.yaml"))
    raw = template.model_dump()
    raw["description"] = ""
    with pytest.raises(ValidationError, match="description"):
        RSkillManifest.model_validate(raw)
