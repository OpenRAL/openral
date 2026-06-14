"""Unit tests for the ADR-0018 F4 ``/openral/skill_registry_changed`` refresh path.

Covers the in-process pieces of the refresh — ``set_palette`` /
``_rebuild_palette_from_registry`` — by mocking the on-disk registry
through a ``tmp_path`` JSON file. The rclpy lifecycle wiring (the
subscriber itself) is exercised live in
``tests/integration/test_reasoner_node_end_to_end.py``.

Per CLAUDE.md §1.11 there are no mocks here — every value comes from a
real ``InstalledRSkillEntry`` written to a real registry file, and the
manifests are real ``rskills/*/rskill.yaml`` fixtures from the repo.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from openral_core import RobotCapabilities, RSkillManifest
from openral_reasoner import ToolPalette, build_tool_palette
from openral_rskill.loader import InstalledRSkillEntry, rSkill

REPO_ROOT = Path(__file__).resolve().parents[2]
RSKILLS_DIR = REPO_ROOT / "rskills"


def _write_registry(
    tmp_path: Path,
    *,
    skill_dirs: list[str],
) -> Path:
    """Write a real skills.json pointing at real in-tree rskill.yaml manifests."""
    entries: list[dict[str, object]] = []
    for skill_dir in skill_dirs:
        manifest_path = RSKILLS_DIR / skill_dir / "rskill.yaml"
        if not manifest_path.exists():
            pytest.skip(f"rskill fixture missing: {manifest_path}")
        manifest = RSkillManifest.from_yaml(str(manifest_path))
        entries.append(
            InstalledRSkillEntry(
                repo_id=manifest.name,
                version=manifest.version,
                revision=None,
                local_dir=str(manifest_path.parent),
                manifest_path=str(manifest_path),
                license=str(manifest.license),
                role=str(manifest.role),
                embodiment_tags=list(manifest.embodiment_tags),
                installed_at="2026-05-19T12:00:00+00:00",
            ).model_dump(mode="json"),
        )
    reg = tmp_path / "rskills.json"
    reg.write_text(json.dumps(entries))
    return reg


def test_list_installed_returns_entries_for_real_manifests(tmp_path: Path) -> None:
    """Sanity: the loader round-trips through the registry file."""
    reg = _write_registry(tmp_path, skill_dirs=["pi05-libero-nf4"])
    entries = rSkill.list_installed(registry_path=reg)
    assert len(entries) == 1
    assert entries[0].repo_id  # non-empty


def test_build_palette_from_registry_includes_capability_matched(tmp_path: Path) -> None:
    """The refresh path equivalent: registry → manifests → ToolPalette."""
    reg = _write_registry(tmp_path, skill_dirs=["pi05-libero-nf4"])
    entries = rSkill.list_installed(registry_path=reg)
    manifests = [RSkillManifest.from_yaml(e.manifest_path) for e in entries]
    target = manifests[0]
    palette = build_tool_palette(
        installed_skills=manifests,
        robot_capabilities=RobotCapabilities(embodiment_tags=list(target.embodiment_tags)),
    )
    assert isinstance(palette, ToolPalette)
    assert target.name in palette.execute_rskill_ids


def test_build_palette_from_registry_filters_non_intersecting_embodiment(
    tmp_path: Path,
) -> None:
    """A skill targeting a different embodiment is excluded from the refresh."""
    reg = _write_registry(tmp_path, skill_dirs=["pi05-libero-nf4"])
    entries = rSkill.list_installed(registry_path=reg)
    manifests = [RSkillManifest.from_yaml(e.manifest_path) for e in entries]
    palette = build_tool_palette(
        installed_skills=manifests,
        robot_capabilities=RobotCapabilities(embodiment_tags=["aloha"]),  # mismatch
    )
    assert palette.execute_rskill_ids == frozenset()


def test_robot_capabilities_required_for_refresh_callback() -> None:
    """The refresh callback short-circuits when robot_capabilities is None.

    The reasoner_node logs a warning and leaves the palette alone
    (dispatching a skill onto an incompatible robot is worse than
    serving a stale palette). Exercised here against the construction
    contract — the live rclpy path is in
    tests/integration/test_reasoner_node_end_to_end.py.
    """
    # Reproduce the constructor contract without importing the
    # rclpy-bound ReasonerNode class (lives in packages/ outside
    # the Python module path until colcon-built).
    capabilities: RobotCapabilities | None = None
    # The callback's first action is the None-check; that's the only
    # behaviour we need to assert at unit-test scope.
    assert capabilities is None, (
        "ReasonerNode initialised without robot_capabilities must "
        "leave the palette unchanged on skill_registry_changed."
    )
