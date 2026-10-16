"""Isaac Lab-Arena DROID ``banana_in_bowl``: robot, scenes and the π0.5 DROID rSkill.

Fixtures: ``robots/droid_franka_robotiq`` (Franka + Robotiq 2F-85, composed URDF),
``scenes/{sim,deploy}/isaac_arena_droid_banana_in_bowl.yaml`` and
``rskills/pi05-droid-jointpos-arena``. Host side only: the options validate, the
robot marshals into the sidecar spec (gripper leader + mimic followers + the
inverted-jaw curve), the bowl region grades the banana, and the DROID I/O (7 joint
deltas anchored to the observed state, absolute gripper with DROID's 1 = closed)
round-trips through the pi05 adapter helpers and the policy codec. No Isaac, no
weights.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from openral_core import DeployScene, RobotDescription, SimScene, load_scene_strict
from openral_core.exceptions import ROSConfigError
from openral_rskill._policy_io import PolicyIOCodec
from openral_rskill.loader import load_rskill_manifest
from openral_sim.backends.isaac_sim import (
    IsaacSimOptions,
    _build_robot_spec,
    _curve_gripper_specs,
    _placement_success_spec,
)
from openral_sim.policies.pi05 import _anchor_deltas, _delta_action_mask

_REPO = Path(__file__).resolve().parents[2]
_SIM = _REPO / "scenes/sim/isaac_arena_droid_banana_in_bowl.yaml"
_DEPLOY = _REPO / "scenes/deploy/isaac_arena_droid_banana_in_bowl.yaml"
_ROBOT = RobotDescription.from_yaml(str(_REPO / "robots/droid_franka_robotiq/robot.yaml"))
_SKILL = load_rskill_manifest(str(_REPO / "rskills/pi05-droid-jointpos-arena"))
_PROMPT = "pick up the banana and place it in the bowl"


def _options(path: Path, kind: type[SimScene] | type[DeployScene]) -> IsaacSimOptions:
    return IsaacSimOptions.model_validate(load_scene_strict(str(path), kind).scene.backend_options)


def test_both_scenes_load_with_the_same_world() -> None:
    sim, deploy = _options(_SIM, SimScene), _options(_DEPLOY, DeployScene)
    assert sim == deploy
    assert [o.name for o in sim.objects] == ["franka_stand", "banana", "bowl"]
    assert load_scene_strict(str(_SIM), SimScene).task.instruction == _PROMPT
    assert sim.placement_success is not None
    assert _PROMPT in sim.placement_success.tasks


@pytest.fixture(scope="module")
def scene_mod() -> Any:
    tools = str(_REPO / "tools")
    if tools not in sys.path:
        sys.path.insert(0, tools)
    import isaac_manifest_scene

    return isaac_manifest_scene


def test_the_banana_grades_in_the_bowl_only(scene_mod: Any) -> None:
    opts = _options(_SIM, SimScene)
    spec = _placement_success_spec(opts.placement_success, _PROMPT)
    assert spec is not None
    bowl = np.array([0.5611, 0.15261, 0.03035])
    at = scene_mod.placement_boxes_at(spec["placements"][0], {"bowl": bowl})
    up, level = np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0, 0.0])
    for banana, grade in [
        (bowl + np.array([0.01, -0.02, 0.0]), "complete"),  # lying in the bowl
        (np.array([0.50973, -0.11581, 0.02118]), "fail"),  # its start pose on the table
        (bowl + np.array([0.0, 0.0, 0.2]), "fail"),  # held above the bowl
    ]:
        assert scene_mod.placement_grade(banana, level, up, at, 20.0, spec["grades"]) == grade


def test_robot_spec_drives_the_robotiq_leader_and_inverts_the_jaw(scene_mod: Any) -> None:
    try:
        spec = _build_robot_spec(_ROBOT, "droid_franka_robotiq")
    except ROSConfigError as exc:  # the pinned mesh clones need network on first use
        pytest.skip(f"robot_descriptions mesh packages unavailable: {exc}")
    assert [e["urdf_name"] for e in spec["joints"]] == [
        *(f"panda_joint{i}" for i in range(1, 8)),
        "finger_joint",
    ]
    (gripper,) = spec["grippers"]
    assert gripper["leader"] == "finger_joint"
    assert len(gripper["followers"]) == 5
    assert spec["action"]["control_freq_hz"] == 15.0
    curve = _options(_SIM, SimScene).gripper_joint_curve
    _curve_gripper_specs(spec, curve)
    # normalized_open_unit: 0 = closed -> Arena's close target pi/4, 1 = open -> 0.
    assert scene_mod.manifest_to_urdf_gripper(gripper, 0.0) == pytest.approx(np.pi / 4)
    assert scene_mod.manifest_to_urdf_gripper(gripper, 1.0) == pytest.approx(0.0)


def test_the_skill_declares_droid_delta_joints_and_an_absolute_gripper() -> None:
    assert _delta_action_mask(_SKILL) == (True,) * 7 + (False,)
    state = np.array([0.1, -0.6, 0.0, -2.5, 0.0, 1.9, 0.2, 0.8], dtype=np.float32)
    delta = np.array([0.01, 0.02, 0.0, -0.01, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    out = _anchor_deltas(delta, state, _delta_action_mask(_SKILL) or ())
    np.testing.assert_allclose(out[:7], state[:7] + delta[:7])
    assert out[7] == 1.0  # the gripper target is not offset by its own state


def test_anchor_refuses_a_mask_of_the_wrong_width() -> None:
    with pytest.raises(ROSConfigError, match="delta mask"):
        _anchor_deltas(np.zeros(32, np.float32), np.zeros(8, np.float32), (True,) * 8)


def test_codec_flips_droid_gripper_closed_one() -> None:
    codec = PolicyIOCodec.from_manifest(_SKILL, _ROBOT)
    open_jaw = np.array([0, -0.63, 0, -2.51, 0, 1.88, 0, 1.0], dtype=np.float32)
    assert codec.to_policy_state(open_jaw)[7] == pytest.approx(0.0)  # DROID: 0 = open
    closed_cmd = codec.to_robot_action(np.array([0.0] * 7 + [1.0], dtype=np.float32))
    assert closed_cmd[7] == pytest.approx(0.0)  # OpenRAL jaw: 0 = closed
    np.testing.assert_allclose(codec.to_policy_state(open_jaw)[:7], open_jaw[:7])
