"""Repo-wide TaskSpace compliance sweep (ADR-0071).

Verifies that **every** shipped robot and rSkill complies with the schema, and
pins the cross-layer task-space compatibility of every actuating skill against
every robot it claims (by ``embodiment_tags``). Real fixtures only — no mocks
(CLAUDE.md §1.11).

Three kinds of compliance are distinguished:

* **Structural** — the manifest loads and validates against the Pydantic schema
  (`RobotDescription` / `RSkillManifest`). Enforced at definition time today.
  Every robot and rSkill must pass (`test_all_*_load`).
* **Cross-layer (rSkill × robot)** — the skill's `TaskSpace` is executable on the
  robot. Checked in `hal_mode="sim"` (the path everything we ship actually runs
  on). All pairs pass except the recorded `KNOWN_SIM_GAPS` (rlbench cartesian-pose
  + gr1 29-DoF — both dedicated-controller paths, not the default packers). The
  test asserts the gap set exactly, so a fix that isn't recorded here (or a new
  regression) trips it.
* **Scene leg (rSkill × scene)** — ADR-0071 Phase 4. For each scene family an
  rSkill is `evaluated_tasks`-on, the skill's `TaskSpace` modes (and fixed dim)
  must be executable by that family's declared `SceneTaskSpace`. `KNOWN_SCENE_GAPS`
  is empty; every declared family must have an entry in `SCENE_FAMILY_TASK_SPACE`.
"""

from __future__ import annotations

import glob
import os
from pathlib import Path

import pytest
from openral_core import (
    SCENE_FAMILY_TASK_SPACE,
    RobotDescription,
    RSkillManifest,
    TaskSpace,
    scene_family,
    scene_task_space_compatible,
    task_space_compatible,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
ROBOT_YAMLS = sorted(glob.glob(str(REPO_ROOT / "robots" / "*" / "robot.yaml")))
RSKILL_YAMLS = sorted(glob.glob(str(REPO_ROOT / "rskills" / "*" / "rskill.yaml")))


def _name(path: str) -> str:
    return os.path.basename(os.path.dirname(path))


# Pairs that are NOT executable by the DEFAULT-sim SimAttachedHAL OSC packers
# (SIM_EXECUTABLE_CONTROL_MODES). After the ADR-0071 manifest fixes these are no
# longer cross-layer *bugs* — both remaining entries run sim through a dedicated
# controller path, not the default packers, so their control mode is correctly
# outside the default-packer set (same category as any sidecar skill):
#
#   * 3d-diffuser-actor-rlbench emits absolute CARTESIAN_POSE (pos+quat) consumed
#     by the RLBench / CoppeliaSim (PyRep) sidecar — never the SimAttachedHAL
#     packers. (Its gripper-slot EE name was fixed: panda_gripper -> panda_hand.)
#   * rldx1-ft-gr1-nf4 emits a 29-D GR1 whole-body action (3 waist + 7+7 arms +
#     6+6 Fourier-hand finger DoF). The robot enumerates 17 joints; the 12 hand
#     DoF are EE-owned (the two dexterous-hand EEs, n_dof=6 each) by deliberate
#     design, NOT joints. So the bare-dim contract's single 29-wide joint segment
#     exceeds the 17 enumerated joints. It runs via the robosuite BASIC composite
#     controller (gr1_unified wrapper), not the default packers. Full DOF-aware
#     accounting (counting EE-owned hand DoF toward the joint budget) is an
#     ADR-0071 follow-up; the empty-modes bug on the gr1 robot IS fixed here.
#
# The test asserts this set EXACTLY: a regression adds an entry, a fix that
# isn't recorded here removes one — either trips it.
KNOWN_SIM_GAPS: frozenset[tuple[str, str]] = frozenset(
    {
        ("3d-diffuser-actor-rlbench", "franka_panda"),  # CARTESIAN_POSE -> RLBench sidecar
        ("rldx1-ft-gr1-nf4", "gr1"),  # 29-D body+hands > 17 enumerated joints (hands EE-owned)
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
        name for name, rd in robots.items() if tags & set(rd.capabilities.embodiment_tags or [])
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


def test_gr1_empty_modes_bug_fixed() -> None:
    """ADR-0071 fix: the gr1 robot now advertises joint_position (was empty) and
    documents the two Fourier hands as 6-DoF dexterous EEs.

    The 29-vs-17 DOF-accounting gap remains a recorded KNOWN_SIM_GAP (the 12
    hand DoF are EE-owned, not enumerated joints), but the genuine empty
    supported_control_modes bug — which dropped gr1 from the reasoner palette /
    task-space gate entirely — is fixed.
    """
    robot = RobotDescription.from_yaml(str(REPO_ROOT / "robots" / "gr1" / "robot.yaml"))
    from openral_core import ControlMode

    assert ControlMode.JOINT_POSITION in robot.capabilities.supported_control_modes
    hands = {ee.name: ee.n_dof for ee in robot.end_effectors if ee.kind == "dexterous_hand"}
    assert hands == {"right_hand": 6, "left_hand": 6}  # 12 hand DoF of the 29-D action


def test_rc365_sim_executable_after_fix() -> None:
    """ADR-0071 fix: rc365 cartesian slot EE corrected panda_hand -> panda_gripper.

    Both robocasa-365 checkpoints are now sim-executable on panda_mobile (the
    cartesian/gripper/base/composite modes are all in SIM_EXECUTABLE and the EE
    names match the robot).
    """
    robot = RobotDescription.from_yaml(str(REPO_ROOT / "robots" / "panda_mobile" / "robot.yaml"))
    for sname in ("pi05-robocasa365-human300-nf4", "rldx1-ft-rc365-nf4"):
        skill = RSkillManifest.from_yaml(str(REPO_ROOT / "rskills" / sname / "rskill.yaml"))
        assert skill.action_contract is not None
        space = TaskSpace.from_action_contract(skill.action_contract, robot)
        match = task_space_compatible(space, robot, hal_mode="sim")
        assert match.ok is True, (sname, match.reasons)


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


# ─── Scene leg (ADR-0071 Phase 4) ──────────────────────────────────────────────
#
# The third side of the triangle: an actuating rSkill must be executable not only
# on the robot it claims (above) but by the SCENE it is evaluated on. The scene's
# executed control interface is declared per adapter family in
# SCENE_FAMILY_TASK_SPACE, keyed by the same vocabulary the rSkill already uses in
# `evaluated_tasks` (the leading token before any "/"). These two pins close the
# rSkill x robot x scene triangle the audit found unconnected.

# Scene-leg pairs that are NOT executable by their declared scene family. After
# the Phase-4 manifest fixes (metaworld 3-D EE-delta+gripper; pusht joint) this
# is EMPTY. The assertion is exact: a regression adds an entry, a fix that isn't
# recorded removes one.
KNOWN_SCENE_GAPS: frozenset[tuple[str, str]] = frozenset()


def test_scene_families_are_declared() -> None:
    """Every scene family an rSkill is evaluated on has a declared task space.

    Forces a new backend / benchmark family to record what control interface it
    executes in SCENE_FAMILY_TASK_SPACE, instead of silently going unchecked.
    """
    undeclared: set[str] = set()
    for path in RSKILL_YAMLS:
        skill = RSkillManifest.from_yaml(path)
        if skill.action_contract is None:
            continue  # non-actuating — no scene task space to satisfy.
        for task in skill.evaluated_tasks or []:
            fam = scene_family(task)
            if fam not in SCENE_FAMILY_TASK_SPACE:
                undeclared.add(fam)
    assert not undeclared, (
        f"actuating rSkills evaluated on scene families with no declared task "
        f"space in SCENE_FAMILY_TASK_SPACE: {sorted(undeclared)}"
    )


def test_scene_executability_matches_known_state() -> None:
    """Pin the scene-leg compatibility of every actuating rSkill x its scenes.

    For each actuating rSkill that declares `evaluated_tasks`, the skill's
    `TaskSpace` (built against a robot it claims) must be executable by every
    scene family it is evaluated on — UNLESS the pair is a recorded
    `KNOWN_SCENE_GAPS` entry. Exact in both directions.
    """
    robots = _robots()
    observed_gaps: set[tuple[str, str]] = set()
    pair_count = 0

    for path in RSKILL_YAMLS:
        skill = RSkillManifest.from_yaml(path)
        if skill.action_contract is None or not skill.evaluated_tasks:
            continue
        sname = _name(path)
        rnames = _matching_robot_names(skill, robots)
        assert rnames, f"{sname}: evaluated skill with no matching robot"
        space = TaskSpace.from_action_contract(skill.action_contract, robots[rnames[0]])
        for task in skill.evaluated_tasks:
            fam = scene_family(task)
            pair_count += 1
            if not scene_task_space_compatible(fam, space).ok:
                observed_gaps.add((sname, fam))

    assert pair_count > 0, "no actuating skill x scene pairs discovered"
    new_gaps = observed_gaps - KNOWN_SCENE_GAPS
    fixed_gaps = KNOWN_SCENE_GAPS - observed_gaps
    assert not new_gaps, f"new scene-incompatible pair(s) introduced: {sorted(new_gaps)}"
    assert not fixed_gaps, (
        f"pair(s) now scene-compatible — remove from KNOWN_SCENE_GAPS: {sorted(fixed_gaps)}"
    )


def test_metaworld_skill_is_ee_delta_not_joint() -> None:
    """ADR-0071 Phase 4: smolvla-metaworld is 3-D EE-delta + gripper, not joint.

    The MetaWorld mocap controller drives EE translation + gripper (4-D); the
    checkpoint previously fell through the undeclared-layout path and was modelled
    as 4 Sawyer joints, which only passed the gate because Sawyer has >=4 joints.
    """
    skill = RSkillManifest.from_yaml(
        str(REPO_ROOT / "rskills" / "smolvla-metaworld" / "rskill.yaml")
    )
    sawyer = RobotDescription.from_yaml(str(REPO_ROOT / "robots" / "sawyer" / "robot.yaml"))
    assert skill.action_contract is not None
    space = TaskSpace.from_action_contract(skill.action_contract, sawyer)
    modes = {m.value for m in space.control_modes}
    assert modes == {"cartesian_delta", "gripper_position"}
    assert space.total_dim == 4
    assert scene_task_space_compatible("metaworld", space).ok


def test_pusht_skill_and_robot_agree_on_joint_space() -> None:
    """ADR-0071 Phase 4: pusht is joint-space end to end (robot mode + contract).

    The pusht_2d robot advertises joint_position over its two prismatic tip
    joints, and diffusion-pusht declares joint_positions(2) — so the skill is now
    real-executable (was real-incompatible while the robot claimed cartesian_pose).
    """
    robot = RobotDescription.from_yaml(str(REPO_ROOT / "robots" / "pusht_2d" / "robot.yaml"))
    from openral_core import ControlMode

    assert ControlMode.JOINT_POSITION in robot.capabilities.supported_control_modes
    skill = RSkillManifest.from_yaml(str(REPO_ROOT / "rskills" / "diffusion-pusht" / "rskill.yaml"))
    assert skill.action_contract is not None
    space = TaskSpace.from_action_contract(skill.action_contract, robot)
    assert task_space_compatible(space, robot, hal_mode="real").ok
    assert scene_task_space_compatible("pusht", space).ok
