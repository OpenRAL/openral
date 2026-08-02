"""BEHAVIOR scene adapter, R1Pro manifest, and atomic action mapping."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from openral_core import Action, ControlMode, JointState, RobotDescription, VLASpec, WorldState
from openral_core.loaders import load_scene_strict
from openral_core.schemas import DeployScene, SimScene
from openral_rskill.loader import load_rskill_manifest
from openral_sim.backends.behavior import (
    _ACTION_DIM,
    _CAMERA_KEYS,
    _BehaviorSidecar,
    _compose_action_group,
    _joint_state_from_policy_state,
)

_ROOT = Path(__file__).resolve().parents[2]


def _joint_action(values: list[float], tick: int) -> Action:
    return Action(
        control_mode=ControlMode.JOINT_POSITION,
        joint_targets=[values],
        tick_index=tick,
    )


def test_r1pro_manifest_and_behavior_scenes_load() -> None:
    robot = RobotDescription.from_yaml(str(_ROOT / "robots" / "r1pro" / "robot.yaml"))
    deploy = load_scene_strict(
        str(_ROOT / "scenes" / "deploy" / "behavior_r1pro.yaml"),
        DeployScene,
    )
    sim = load_scene_strict(
        str(_ROOT / "scenes" / "sim" / "behavior_turning_on_radio.yaml"),
        SimScene,
    )
    assert robot.name == "r1pro"
    assert len(robot.joints) == 22
    assert deploy.scene.id == "behavior"
    assert sim.task.id == "behavior/turning_on_radio"
    assert list(_CAMERA_KEYS) == ["head", "left_wrist", "right_wrist"]


def test_policy_state_extracts_manifest_order_joint_state() -> None:
    state = np.arange(61, dtype=np.float32)
    positions, velocities = _joint_state_from_policy_state(state)
    expected_positions = np.concatenate(
        [state[53:57], state[3:10], state[24:26], state[28:35], state[49:51]]
    )
    expected_velocities = np.concatenate(
        [state[57:61], state[10:17], state[26:28], state[35:42], state[51:53]]
    )
    np.testing.assert_array_equal(positions, expected_positions)
    np.testing.assert_array_equal(velocities, expected_velocities)


def test_behavior_idle_action_holds_current_pose() -> None:
    state = np.zeros(61, dtype=np.float32)
    state[53:57] = [0.5, -0.5, 0.25, 0.0]
    state[3:10] = np.arange(7, dtype=np.float32)
    state[28:35] = -np.arange(7, dtype=np.float32)
    state[24:26] = 0.05
    state[49:51] = 0.0
    rollout = object.__new__(_BehaviorSidecar)
    rollout._last_state = state
    action = rollout.idle_action()
    assert action.shape == (23,)
    np.testing.assert_array_equal(action[:3], np.zeros(3))
    np.testing.assert_array_equal(action[3:7], state[53:57])
    assert action[14] == 1.0
    assert action[22] == -1.0


def test_six_safe_slots_compose_one_official_action_vector() -> None:
    tick = 9
    torso = np.zeros(22, dtype=np.float32)
    torso[0:4] = [3, 4, 5, 6]
    left = np.zeros(22, dtype=np.float32)
    left[4:11] = np.arange(7, 14)
    right = np.zeros(22, dtype=np.float32)
    right[13:20] = np.arange(15, 22)
    actions = [
        Action(
            control_mode=ControlMode.BODY_TWIST,
            body_twist=[(0.1, 0.2, 0.0, 0.0, 0.0, 0.3)],
            frame_id="base_link",
            tick_index=tick,
        ),
        _joint_action(torso.tolist(), tick),
        _joint_action(left.tolist(), tick),
        Action(
            control_mode=ControlMode.GRIPPER_POSITION,
            gripper=[0.5],
            ee_name="left_gripper",
            tick_index=tick,
        ),
        _joint_action(right.tolist(), tick),
        Action(
            control_mode=ControlMode.GRIPPER_POSITION,
            gripper=[-0.5],
            ee_name="right_gripper",
            tick_index=tick,
        ),
    ]
    vector = _compose_action_group(actions)
    assert vector.shape == (_ACTION_DIM,)
    np.testing.assert_array_equal(
        vector,
        np.asarray(
            [0.1, 0.2, 0.3, *range(3, 14), 0.5, *range(15, 22), -0.5],
            dtype=np.float32,
        ),
    )


def test_runtime_skill_consumes_world_state_policy_state() -> None:
    from openral_rskill_ros.rskill_runner_node import _make_policy_adapter_skill

    robot = RobotDescription.from_yaml(str(_ROOT / "robots" / "r1pro" / "robot.yaml"))
    manifest = load_rskill_manifest(str(_ROOT / "rskills" / "gr00t-n17-b1k-turning-on-radio"))

    class Adapter:
        spec = VLASpec(
            id="gr00t", weights_uri=str(_ROOT / "rskills" / "gr00t-n17-b1k-turning-on-radio")
        )
        device = "cpu"

        def __init__(self) -> None:
            self.state: np.ndarray | None = None

        def reset(self) -> None:
            return None

        def step(self, observation: dict[str, object], instruction: str) -> np.ndarray:
            del instruction
            self.state = np.asarray(observation["state"], dtype=np.float32)
            return np.zeros(23, dtype=np.float32)

        def close(self) -> None:
            return None

    adapter = Adapter()
    skill = _make_policy_adapter_skill(
        manifest=manifest,
        adapter=adapter,
        prompt="turn on the radio",
        description=robot,
    )
    state = np.arange(61, dtype=np.float32)
    actions = skill.step(
        WorldState(
            stamp_ns=1,
            joint_state=JointState(
                name=[joint.name for joint in robot.joints],
                position=[0.0] * len(robot.joints),
                velocity=[0.0] * len(robot.joints),
                stamp_ns=1,
            ),
            policy_state=state.tolist(),
            diagnostics={"policy_state": "ok"},
        )
    )
    assert isinstance(actions, list)
    assert len(actions) == 6
    np.testing.assert_array_equal(adapter.state, state)
