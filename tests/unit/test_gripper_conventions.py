"""Gripper encodings: every in-tree skill declares one, and it matches where it runs.

A ``GRIPPER_*`` value reaches the HAL unconverted, so ``rSkill.check_gripper_conventions``
refuses a skill whose ``gripper_convention`` differs from what the target end
effector (or the simulated scene's environment) consumes. These tests load the
real manifests under ``rskills/`` and ``robots/`` and the real scene registry.
"""

from __future__ import annotations

import pathlib

import pytest
from openral_core import ControlMode, RobotDescription, RSkillManifest
from openral_core.exceptions import ROSCapabilityMismatch
from openral_core.schemas import canonical_slots_for_representation
from openral_rskill.loader import rSkill

_REPO = pathlib.Path(__file__).resolve().parents[2]
_GRIPPER_MODES = (ControlMode.GRIPPER_POSITION, ControlMode.GRIPPER_BINARY)

# Every in-tree skill that sends gripper commands, and the scene it is built
# for (None: the robot's own HAL, no simulated scene). A new gripper-emitting
# skill fails `test_every_gripper_emitting_skill_is_listed` until it is added.
_SKILL_SCENES: dict[str, str | None] = {
    "act-libero": "libero_spatial",
    "molmoact2-libero-nf4": "libero_spatial",
    "pi05-libero-int8": "libero_spatial",
    "gr00t-n17-libero": "libero_spatial",
    "rldx1-ft-libero-nf4": "libero_spatial",
    "smolvla-libero": "libero_spatial",
    "xvla-libero": "libero_spatial",
    "smolvla-vlabench": "vlabench",
    "xr1-vlabench": "vlabench",
    "openvla-oft-simpler-widowx-nf4": "simpler_env",
    "rldx1-ft-simpler-widowx-nf4": "simpler_env",
    "smolvla-metaworld": "metaworld",
    "xr1-robocasa": "robocasa",
    "xr1-robocasa365": "robocasa",
    "rldx1-ft-rc365-nf4": "robocasa",
    "3d-diffuser-actor-rlbench": "rlbench",
    "gr00t-n17-b1k-turning-on-radio": "behavior",
    "lingbot-va-galaxea-a1-fruit-placement": None,
}


def _skill(name: str) -> RSkillManifest:
    return RSkillManifest.from_yaml(str(_REPO / "rskills" / name / "rskill.yaml"))


def _robot(name: str) -> RobotDescription:
    return RobotDescription.from_yaml(str(_REPO / "robots" / name / "robot.yaml"))


def _emits_gripper(manifest: RSkillManifest, robot: RobotDescription) -> bool:
    if any(a.kind in _GRIPPER_MODES for a in manifest.actuators_required):
        return True
    contract = manifest.action_contract
    if contract is None:
        return False
    slots = contract.slots
    if not slots and contract.representation is not None:
        slots = canonical_slots_for_representation(
            contract.representation, dim=contract.dim, description=robot
        )
    return any(s.control_mode in _GRIPPER_MODES for s in slots or [])


def _scene_convention(scene: str | None) -> str | None:
    if scene is None:
        return None
    from openral_sim import SCENES

    value = SCENES.meta(scene).get("gripper_convention")
    assert value, f"scene {scene!r} declares no gripper_convention"
    return str(value)


def test_every_gripper_emitting_skill_is_listed() -> None:
    emitting = set()
    for path in sorted((_REPO / "rskills").glob("*/rskill.yaml")):
        manifest = RSkillManifest.from_yaml(str(path))
        robots = [t for t in manifest.embodiment_tags if (_REPO / "robots" / t).is_dir()]
        if robots and _emits_gripper(manifest, _robot(robots[0])):
            emitting.add(path.parent.name)
    assert emitting == set(_SKILL_SCENES)


@pytest.mark.parametrize(("skill", "scene"), sorted(_SKILL_SCENES.items()))
def test_skill_encoding_matches_where_it_runs(skill: str, scene: str | None) -> None:
    manifest = _skill(skill)
    robot = _robot(next(t for t in manifest.embodiment_tags if (_REPO / "robots" / t).is_dir()))
    rSkill.check_gripper_conventions(
        manifest, robot, scene_gripper_convention=_scene_convention(scene)
    )


def test_libero_skill_on_the_bare_franka_twin_is_refused() -> None:
    """LIBERO's env closes on +1; the MuJoCo Franka twin opens on 1."""
    with pytest.raises(ROSCapabilityMismatch, match="normalized_open_unit"):
        rSkill.check_gripper_conventions(_skill("smolvla-libero"), _robot("franka_panda"))


def test_robocasa_skill_in_an_open_unit_scene_is_refused() -> None:
    with pytest.raises(ROSCapabilityMismatch, match="this scene's environment"):
        rSkill.check_gripper_conventions(
            _skill("xr1-robocasa365"),
            _robot("panda_mobile"),
            scene_gripper_convention="normalized_open_unit",
        )


def test_gripper_slot_without_a_declared_convention_is_refused() -> None:
    manifest = _skill("xr1-robocasa365")
    stripped = manifest.model_copy(
        update={
            "actuators_required": [
                a for a in manifest.actuators_required if a.kind not in _GRIPPER_MODES
            ]
        }
    )
    with pytest.raises(ROSCapabilityMismatch, match="declares no gripper_convention"):
        rSkill.check_gripper_conventions(stripped, _robot("panda_mobile"))


def test_end_effector_without_a_declared_convention_is_refused() -> None:
    robot = _robot("galaxea_a1")
    undeclared = robot.model_copy(
        update={
            "end_effectors": [
                ee.model_copy(update={"command_convention": None}) for ee in robot.end_effectors
            ]
        }
    )
    with pytest.raises(ROSCapabilityMismatch, match="<undeclared>"):
        rSkill.check_gripper_conventions(
            _skill("lingbot-va-galaxea-a1-fruit-placement"), undeclared
        )


def test_slot_naming_an_unknown_end_effector_is_refused() -> None:
    with pytest.raises(ROSCapabilityMismatch, match="addresses gripper end effector"):
        rSkill.check_gripper_conventions(_skill("xr1-robocasa365"), _robot("galaxea_a1"))


def test_skill_without_gripper_actuation_is_not_checked() -> None:
    rSkill.check_gripper_conventions(_skill("act-aloha"), _robot("aloha_bimanual"))


def test_check_compatibility_runs_the_gripper_check() -> None:
    with pytest.raises(ROSCapabilityMismatch, match="gripper"):
        rSkill.check_compatibility(_skill("smolvla-libero"), _robot("franka_panda"))
