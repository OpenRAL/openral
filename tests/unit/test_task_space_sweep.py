"""Repo-wide TaskSpace compliance sweep (ADR-0071).

Verifies that **every** shipped robot and rSkill complies with the schema, and
pins the cross-layer task-space compatibility of every actuating skill against
every robot it claims (by ``embodiment_tags``). Real fixtures only — no mocks
(CLAUDE.md §1.11).

Two kinds of compliance are distinguished:

* **Structural** — the manifest loads and validates against the Pydantic schema
  (`RobotDescription` / `RSkillManifest`). Enforced at definition time today.
  Every robot and rSkill must pass (`test_all_*_load`).
* **Cross-layer** — the skill's `TaskSpace` is executable on the robot. Checked
  in `hal_mode="sim"` (the path everything we ship actually runs on). 20/24 pairs
  pass; the remaining 4 are *genuine latent manifest inconsistencies* this sweep
  surfaced (see `KNOWN_SIM_GAPS`). The test asserts the gap set exactly, so a
  fix that isn't recorded here (or a new regression) trips it.
"""

from __future__ import annotations

import glob
import os
from pathlib import Path

import pytest
from openral_core import (
    RobotDescription,
    RSkillManifest,
    TaskSpace,
    task_space_compatible,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
ROBOT_YAMLS = sorted(glob.glob(str(REPO_ROOT / "robots" / "*" / "robot.yaml")))
RSKILL_YAMLS = sorted(glob.glob(str(REPO_ROOT / "rskills" / "*" / "rskill.yaml")))


def _name(path: str) -> str:
    return os.path.basename(os.path.dirname(path))


# Cross-layer gaps the sweep surfaces today — each is a real inconsistency
# between a shipped rSkill and the robot it targets, NOT a schema bug. Fixing a
# manifest must remove its entry here (the test enforces that the set is exact).
# These are Phase-3 cleanups in ADR-0071; tracked, not silently tolerated.
KNOWN_SIM_GAPS: frozenset[tuple[str, str]] = frozenset(
    {
        # cartesian_pose is not synthesizable by the default-sim OSC packers
        # (SIM_EXECUTABLE_CONTROL_MODES); this is an RLBench-sidecar benchmark
        # skill. Its action slots also name ee='panda_gripper', but franka_panda
        # declares ee='panda_hand'.
        ("3d-diffuser-actor-rlbench", "franka_panda"),
        # rc365 skills' cartesian/gripper slots name ee='panda_hand', but
        # panda_mobile declares its gripper EE as 'panda_gripper'.
        ("pi05-robocasa365-human300-nf4", "panda_mobile"),
        ("rldx1-ft-rc365-nf4", "panda_mobile"),
        # The gr1 robot manifest models 17 joints, but the RLDX-1 GR1 checkpoint
        # (and the robocasa GR1 sim) drive a 29-DOF waist+arms+hands body — the
        # manifest under-models the dexterous hands.
        ("rldx1-ft-gr1-nf4", "gr1"),
    }
)


@pytest.mark.parametrize("path", ROBOT_YAMLS, ids=_name)
def test_all_robots_load(path: str) -> None:
    """Every robot manifest validates against RobotDescription (structural)."""
    desc = RobotDescription.from_yaml(path)
    assert desc.name
    # supported_control_modes deserialize as ControlMode enum members.
    for mode in desc.capabilities.supported_control_modes:
        assert mode.value


@pytest.mark.parametrize("path", RSKILL_YAMLS, ids=_name)
def test_all_rskills_load(path: str) -> None:
    """Every rSkill manifest validates against RSkillManifest (structural)."""
    manifest = RSkillManifest.from_yaml(path)
    assert manifest.name
    # When an action contract is present its dim is positive and any declared
    # slots already passed ActionContract's coverage validator at load.
    if manifest.action_contract is not None:
        assert manifest.action_contract.dim > 0


def _robots() -> dict[str, RobotDescription]:
    return {_name(p): RobotDescription.from_yaml(p) for p in ROBOT_YAMLS}


def _matching_robot_names(skill: RSkillManifest, robots: dict[str, RobotDescription]) -> list[str]:
    tags = set(skill.embodiment_tags or [])
    return [
        name
        for name, rd in robots.items()
        if tags & set(rd.capabilities.embodiment_tags or [])
    ]


def test_sim_executability_matches_known_state() -> None:
    """Pin the sim-mode task-space compatibility of every actuating pair.

    For each actuating rSkill × each robot it claims by embodiment tag, the
    skill's `TaskSpace` must be sim-executable — UNLESS the pair is a recorded
    `KNOWN_SIM_GAPS` entry. The assertion is exact in both directions: a newly
    broken pair fails, and a fixed pair still listed as a gap also fails (forcing
    the list to stay honest as Phase-3 cleanups land).
    """
    robots = _robots()
    observed_gaps: set[tuple[str, str]] = set()
    pair_count = 0

    for path in RSKILL_YAMLS:
        skill = RSkillManifest.from_yaml(path)
        if skill.action_contract is None:
            continue  # detector / vlm / reward / ros_action — no task space.
        sname = _name(path)
        for rname in _matching_robot_names(skill, robots):
            pair_count += 1
            space = TaskSpace.from_action_contract(skill.action_contract, robots[rname])
            match = task_space_compatible(space, robots[rname], hal_mode="sim")
            if not match.ok:
                observed_gaps.add((sname, rname))

    assert pair_count > 0, "no actuating skill×robot pairs discovered"
    new_gaps = observed_gaps - KNOWN_SIM_GAPS
    fixed_gaps = KNOWN_SIM_GAPS - observed_gaps
    assert not new_gaps, f"new sim-incompatible pair(s) introduced: {sorted(new_gaps)}"
    assert not fixed_gaps, (
        f"pair(s) now sim-compatible — remove from KNOWN_SIM_GAPS: {sorted(fixed_gaps)}"
    )


def test_every_actuating_skill_has_a_matching_robot() -> None:
    """No actuating rSkill ships pointing at an embodiment no robot provides."""
    robots = _robots()
    orphans = []
    for path in RSKILL_YAMLS:
        skill = RSkillManifest.from_yaml(path)
        if skill.action_contract is None:
            continue
        if not _matching_robot_names(skill, robots):
            orphans.append((_name(path), skill.embodiment_tags))
    assert not orphans, f"actuating skills with no matching robot: {orphans}"
