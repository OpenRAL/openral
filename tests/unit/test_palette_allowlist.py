"""Pin the robot's curated rSkill list (``OPENRAL_RSKILLS``, OpenRAL/openral#656).

openral.com keeps a list of the rSkills each robot may run (OpenRAL/platform#147) and sends it
as an account setting; ``build_tool_palette`` then admits only those actuating skills, on every
build (seed and registry refresh), and detectors are left alone. No mocks (CLAUDE.md §1.11):
real in-tree manifests through the real palette builder.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from openral_core import (
    DeployRuntime,
    RobotCapabilities,
    RobotDescription,
    RSkillManifest,
    allowed_rskills_from_env,
    build_robot_manifest,
    rskill_listed,
)
from openral_reasoner.palette import build_tool_palette

_RSKILLS_DIR = Path(__file__).resolve().parents[2] / "rskills"


def _load_intree() -> list[RSkillManifest]:
    return [RSkillManifest.from_yaml(str(p)) for p in sorted(_RSKILLS_DIR.glob("*/rskill.yaml"))]


def _franka() -> RobotCapabilities:
    return RobotCapabilities(embodiment_tags=["franka_panda"])


def test_no_list_admits_every_matching_skill() -> None:
    full = build_tool_palette(installed_skills=_load_intree(), robot_capabilities=_franka())
    same = build_tool_palette(
        installed_skills=_load_intree(), robot_capabilities=_franka(), allowed_rskill_ids=None
    )
    assert len(full.execute_rskill_ids) > 2
    assert same.execute_rskill_ids == full.execute_rskill_ids


def test_list_admits_only_listed_skills_that_match() -> None:
    full = build_tool_palette(installed_skills=_load_intree(), robot_capabilities=_franka())
    two = frozenset(sorted(full.execute_rskill_ids)[:2])
    # A listed skill for another robot, and one that isn't installed, never get in.
    other = "OpenRAL/rskill-nav2-mobile_base-navigate_to_pose"
    listed = two | {other, "OpenRAL/rskill-not-installed"}
    palette = build_tool_palette(
        installed_skills=_load_intree(), robot_capabilities=_franka(), allowed_rskill_ids=listed
    )
    assert palette.execute_rskill_ids == two


def test_list_with_nothing_runnable_gives_an_empty_palette_not_everything() -> None:
    palette = build_tool_palette(
        installed_skills=_load_intree(),
        robot_capabilities=_franka(),
        allowed_rskill_ids=frozenset({"OpenRAL/rskill-not-installed"}),
    )
    assert palette.execute_rskill_ids == frozenset()


def test_list_leaves_detectors_alone() -> None:
    full = build_tool_palette(installed_skills=_load_intree(), robot_capabilities=_franka())
    some = build_tool_palette(
        installed_skills=_load_intree(),
        robot_capabilities=_franka(),
        allowed_rskill_ids=frozenset({"OpenRAL/rskill-not-installed"}),
    )
    assert some.continuous_detectors == full.continuous_detectors
    assert some.on_demand_detectors == full.on_demand_detectors


def test_env_list_parses_and_empty_means_no_restriction() -> None:
    assert allowed_rskills_from_env({}) is None
    assert allowed_rskills_from_env({"OPENRAL_RSKILLS": ""}) is None
    assert allowed_rskills_from_env({"OPENRAL_RSKILLS": " , "}) is None
    listed = allowed_rskills_from_env({"OPENRAL_RSKILLS": "OpenRAL/rskill-a, OpenRAL/rskill-b,"})
    assert listed == frozenset({"openral/rskill-a", "openral/rskill-b"})
    # Hugging Face repo ids are case-insensitive: so is the list.
    assert rskill_listed(
        "OpenRAL/rskill-A", allowed_rskills_from_env({"OPENRAL_RSKILLS": "openral/RSKILL-a"})
    )
    assert not rskill_listed("OpenRAL/rskill-c", listed)
    assert rskill_listed("OpenRAL/rskill-b", None)


def test_manifest_skill_buttons_follow_the_list() -> None:
    robot = RobotDescription.from_yaml(
        str(_RSKILLS_DIR.parent / "robots" / "franka_panda" / "robot.yaml")
    )

    def skills(allowed: frozenset[str] | None) -> set[str]:
        manifest = build_robot_manifest(
            robot,
            DeployRuntime(),
            _load_intree(),
            bridge_on=False,
            write_controls=True,
            remote_control=False,
            allowed_rskill_ids=allowed,
        )
        return {a.id.removeprefix("skill:") for a in manifest.actions if a.kind == "skill"}

    every = skills(None)
    assert len(every) > 2
    one = sorted(every)[0]
    assert skills(frozenset({one, "OpenRAL/rskill-not-installed"})) == {one}


def test_list_matches_ids_whatever_their_case() -> None:
    full = build_tool_palette(installed_skills=_load_intree(), robot_capabilities=_franka())
    one = sorted(full.execute_rskill_ids)[0]
    palette = build_tool_palette(
        installed_skills=_load_intree(),
        robot_capabilities=_franka(),
        allowed_rskill_ids=allowed_rskills_from_env({"OPENRAL_RSKILLS": one.upper()}),
    )
    assert palette.execute_rskill_ids == frozenset({one})


def test_the_deploys_pre_launch_checks_follow_the_list(monkeypatch: pytest.MonkeyPatch) -> None:
    """The reward pairing, the VRAM preflight and the missing-extras prompt reason over the
    rSkills the reasoner will get (deploy_sim._capability_matched_manifests), never more."""
    from openral_cli.deploy_sim import _capability_matched_manifests

    root = _RSKILLS_DIR.parent
    robot = RobotDescription.from_yaml(str(root / "robots" / "franka_panda" / "robot.yaml"))
    monkeypatch.delenv("OPENRAL_RSKILLS", raising=False)
    every = [m.name for m in _capability_matched_manifests(root, robot)]
    assert len(every) > 2
    monkeypatch.setenv("OPENRAL_RSKILLS", f"{every[0]},OpenRAL/rskill-not-installed")
    assert [m.name for m in _capability_matched_manifests(root, robot)] == [every[0]]
