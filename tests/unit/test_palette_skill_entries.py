"""ADR-0022 — :func:`build_tool_palette` carries per-skill metadata.

Drives the real :class:`RSkillManifest` loader against the in-tree
``rskills/*/rskill.yaml`` files (no synthetic manifests), then asserts
that the palette returned by :func:`build_tool_palette` carries
:class:`RSkillToolEntry` records with the manifest's description /
actions / objects / scenes — not just opaque ids.

Per CLAUDE.md §1.11 / §5.4: no mocks, no smoke tests. Every assertion
exercises the real Pydantic schemas and the real palette filter.
"""

from __future__ import annotations

from pathlib import Path

from openral_core import RobotCapabilities, RSkillAction, RSkillManifest
from openral_reasoner.palette import RSkillToolEntry, ToolPalette, build_tool_palette

_REPO_ROOT = Path(__file__).resolve().parents[2]
_RSKILLS_DIR = _REPO_ROOT / "rskills"


def _load_intree() -> list[RSkillManifest]:
    return [RSkillManifest.from_yaml(str(p)) for p in sorted(_RSKILLS_DIR.glob("*/rskill.yaml"))]


def test_palette_carries_per_skill_metadata_for_aloha() -> None:
    """An aloha robot pulls in the two ALOHA skills with full metadata."""
    caps = RobotCapabilities(embodiment_tags=["aloha"])
    palette = build_tool_palette(installed_skills=_load_intree(), robot_capabilities=caps)
    assert palette.skills, "expected ALOHA palette to be non-empty"
    skill_ids = {entry.rskill_id for entry in palette.skills}
    assert "OpenRAL/rskill-act-aloha" in skill_ids
    assert "OpenRAL/rskill-act-aloha-insertion" in skill_ids
    for entry in palette.skills:
        assert isinstance(entry, RSkillToolEntry)
        assert entry.description, f"{entry.rskill_id} missing description"
        assert entry.actions, f"{entry.rskill_id} missing actions"
        # ALOHA skills carry the bimanual transfer / insertion verb set:
        assert {RSkillAction.PICK, RSkillAction.PLACE}.intersection(entry.actions), (
            f"{entry.rskill_id}: expected at least one of PICK / PLACE in actions"
        )


def test_palette_excludes_detector_kind_skills() -> None:
    """``kind: detector`` rSkills are perception producers, never in the ExecuteSkill palette.

    The in-tree RT-DETR detectors (``rtdetr-coco-r18`` / ``rtdetr-v2-r50vd``)
    carry ``role: s1`` and broad embodiment tags, so they pass the role +
    embodiment filters — but they are activated as the perception ROS node /
    GStreamer tee (ADR-0035/0037), not dispatched via ExecuteSkill. They must
    not appear in any robot's palette regardless of embodiment match.
    """
    intree = _load_intree()
    detector_ids = {m.name for m in intree if m.kind == "detector"}
    assert detector_ids, "expected in-tree detector rSkills as a real fixture"
    # A franka robot matches the detectors' embodiment tags + manipulation VLAs.
    caps = RobotCapabilities(embodiment_tags=["franka_panda"])
    palette = build_tool_palette(installed_skills=intree, robot_capabilities=caps)
    palette_ids = {s.rskill_id for s in palette.skills}
    assert palette_ids, "expected a non-empty franka palette (manipulation VLAs)"
    assert not (palette_ids & detector_ids), (
        f"detector rSkills leaked into the ExecuteSkill palette: {palette_ids & detector_ids}"
    )


def test_palette_execute_rskill_ids_match_skills_when_both_present() -> None:
    """``execute_rskill_ids`` is auto-derived from ``skills`` when only the latter is set."""
    caps = RobotCapabilities(embodiment_tags=["pusht"])
    palette = build_tool_palette(installed_skills=_load_intree(), robot_capabilities=caps)
    assert {s.rskill_id for s in palette.skills} == set(palette.execute_rskill_ids)


def test_palette_filters_skills_not_matching_embodiment() -> None:
    """Skills whose embodiment_tags don't intersect get dropped (existing behaviour)."""
    caps = RobotCapabilities(embodiment_tags=["pusht"])
    palette = build_tool_palette(installed_skills=_load_intree(), robot_capabilities=caps)
    ids = {s.rskill_id for s in palette.skills}
    assert ids == {"OpenRAL/rskill-diffusion-pusht"}
    (entry,) = palette.skills
    assert entry.actions == (RSkillAction.PUSH,)
    assert entry.objects == ("t_shape",)
    assert entry.scenes == ("tabletop_2d",)


def test_empty_palette_when_no_skill_matches_embodiment() -> None:
    """A robot whose embodiment matches no in-tree skill gets an empty palette."""
    # ``unitree_h1`` is a valid tag in the EmbodimentTag closed set but no
    # in-tree skill currently targets it; ensures the filter still runs cleanly.
    caps = RobotCapabilities(embodiment_tags=["h1"])
    palette = build_tool_palette(installed_skills=_load_intree(), robot_capabilities=caps)
    assert palette.skills == ()
    assert palette.execute_rskill_ids == frozenset()


def test_palette_is_frozen() -> None:
    """The palette and its entries must be immutable per ADR-0018."""
    caps = RobotCapabilities(embodiment_tags=["aloha"])
    palette = build_tool_palette(installed_skills=_load_intree(), robot_capabilities=caps)
    import pytest

    with pytest.raises((TypeError, ValueError)):
        palette.skills = ()  # type: ignore[misc]  # reason: pydantic frozen


def test_palette_id_only_construction_still_validates() -> None:
    """Palettes built with only ``execute_rskill_ids`` (no per-skill metadata) validate.

    Exercised by synthetic test palettes in tests/integration and by the
    default empty palette in ``reasoner_node``
    (``ToolPalette(execute_rskill_ids=frozenset())``).
    """
    palette = ToolPalette(execute_rskill_ids=frozenset({"openral/id-only-skill"}))
    assert palette.skills == ()
    assert palette.execute_rskill_ids == frozenset({"openral/id-only-skill"})


def test_palette_order_is_stable_across_calls() -> None:
    """``build_tool_palette`` produces a deterministic order — required for
    reproducible LLM tool schemas (CLAUDE.md operating principle 8)."""
    caps = RobotCapabilities(
        embodiment_tags=["franka_panda", "aloha", "so100_follower", "pusht_2d"],
    )
    manifests = _load_intree()
    p1 = build_tool_palette(installed_skills=manifests, robot_capabilities=caps)
    p2 = build_tool_palette(installed_skills=list(reversed(manifests)), robot_capabilities=caps)
    assert [s.rskill_id for s in p1.skills] == [s.rskill_id for s in p2.skills]
