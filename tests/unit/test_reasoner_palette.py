"""Unit tests for :func:`openral_reasoner.build_tool_palette` (ADR-0018 F4).

Loads **real** ``rskill.yaml`` manifests from ``rskills/`` (CLAUDE.md
§1.11 — real components, no mocks) and asserts the palette filter
matches the documented contract.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from openral_core import RobotCapabilities, RSkillManifest
from openral_reasoner import ToolPalette, build_tool_palette

REPO_ROOT = Path(__file__).resolve().parents[2]
RSKILLS_DIR = REPO_ROOT / "rskills"


def _load_manifest(skill_dirname: str) -> RSkillManifest:
    """Load a real rskill.yaml from rskills/<skill_dirname>."""
    path = RSKILLS_DIR / skill_dirname / "rskill.yaml"
    if not path.exists():
        pytest.skip(f"rskill fixture missing: {path}")
    with path.open() as fh:
        return RSkillManifest.model_validate(yaml.safe_load(fh))


def test_palette_is_empty_when_no_skills_installed() -> None:
    """A robot with no installed skills gets an empty execute_rskill_ids set."""
    palette = build_tool_palette(
        installed_skills=[],
        robot_capabilities=RobotCapabilities(embodiment_tags=["so100_follower"]),
    )
    assert isinstance(palette, ToolPalette)
    assert palette.execute_rskill_ids == frozenset()


def test_palette_includes_capability_matched_skill() -> None:
    """A skill whose embodiment_tags intersect the robot is included."""
    manifest = _load_manifest("pi05-libero-nf4")
    capabilities = RobotCapabilities(
        embodiment_tags=list(manifest.embodiment_tags),
    )
    palette = build_tool_palette(
        installed_skills=[manifest],
        robot_capabilities=capabilities,
    )
    assert manifest.name in palette.execute_rskill_ids


def test_palette_excludes_skill_with_non_intersecting_embodiment() -> None:
    """A skill targeting a different embodiment is excluded."""
    manifest = _load_manifest("pi05-libero-nf4")
    capabilities = RobotCapabilities(embodiment_tags=["aloha"])  # mismatched
    palette = build_tool_palette(
        installed_skills=[manifest],
        robot_capabilities=capabilities,
    )
    assert manifest.name not in palette.execute_rskill_ids


def test_palette_sensor_ids_are_forwarded_verbatim() -> None:
    """sensor_ids is a passthrough (no filtering, no normalisation)."""
    palette = build_tool_palette(
        installed_skills=[],
        robot_capabilities=RobotCapabilities(),
        sensor_ids=["wrist_rgb", "overhead"],
    )
    assert palette.sensor_ids == frozenset({"wrist_rgb", "overhead"})


def test_palette_node_ids_are_forwarded_verbatim() -> None:
    """node_ids is a passthrough."""
    palette = build_tool_palette(
        installed_skills=[],
        robot_capabilities=RobotCapabilities(),
        node_ids=["/openral/hal/so100", "/openral/safety"],
    )
    assert palette.node_ids == frozenset({"/openral/hal/so100", "/openral/safety"})


def test_palette_is_frozen() -> None:
    """Mutation attempts fail — the palette is immutable once built."""
    palette = build_tool_palette(
        installed_skills=[],
        robot_capabilities=RobotCapabilities(),
    )
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        palette.execute_rskill_ids = frozenset({"x"})  # type: ignore[misc]  # reason: pydantic frozen


def test_commercial_deployment_excludes_rldx_noncommercial_skill() -> None:
    """Explicit assertion: commercial_deployment=True drops RLDX-1 non-commercial weights.

    ``rskills/rldx1-ft-libero-nf4`` carries RLDX's non-commercial
    license posture (``RSkillLicensePosture.RLWRLD_NON_COMMERCIAL``).
    CLAUDE.md §1.9 requires the loader / palette / action server to
    refuse these in a commercial deployment.
    """
    manifest = _load_manifest("rldx1-ft-libero-nf4")
    if manifest.is_commercial_use_allowed:
        pytest.skip(
            "Fixture license posture changed — rldx1-ft-libero-nf4 must remain "
            "non-commercial for this test to exercise the filter.",
        )
    capabilities = RobotCapabilities(embodiment_tags=list(manifest.embodiment_tags))
    research_palette = build_tool_palette(
        installed_skills=[manifest],
        robot_capabilities=capabilities,
        commercial_deployment=False,
    )
    commercial_palette = build_tool_palette(
        installed_skills=[manifest],
        robot_capabilities=capabilities,
        commercial_deployment=True,
    )
    # Research deployment: the skill is in the palette.
    assert manifest.name in research_palette.execute_rskill_ids
    # Commercial deployment: the skill is filtered out.
    assert manifest.name not in commercial_palette.execute_rskill_ids


def test_palette_excludes_noncommercial_skill_under_commercial_deployment() -> None:
    """commercial_deployment=True drops skills whose license blocks commercial use."""
    manifest = _load_manifest("rldx1-ft-libero-nf4")
    capabilities = RobotCapabilities(embodiment_tags=list(manifest.embodiment_tags))
    research = build_tool_palette(
        installed_skills=[manifest],
        robot_capabilities=capabilities,
        commercial_deployment=False,
    )
    commercial = build_tool_palette(
        installed_skills=[manifest],
        robot_capabilities=capabilities,
        commercial_deployment=True,
    )
    # Research deployment allows the skill; commercial drops it iff the
    # license posture blocks commercial use. If this particular fixture
    # is Apache-licensed the filter is a no-op; if it's a non-commercial
    # weights repo (e.g. RLDX), the commercial palette is strictly smaller.
    assert manifest.name in research.execute_rskill_ids
    if not manifest.is_commercial_use_allowed:
        assert manifest.name not in commercial.execute_rskill_ids
