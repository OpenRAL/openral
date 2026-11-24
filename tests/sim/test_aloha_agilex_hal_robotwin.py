"""Sim test: ``SimAttachedHAL.read_state`` publishes RoboTwin's *measured* joints.

The RoboTwin rollout used to return only ``state`` (LeRobot's ``agent_pos``, which is
RoboTwin's ``joint_action.vector``: the joints' drive targets and the commanded gripper,
zero-filled by LeRobot when absent), so a scene-attached ``openral deploy sim`` on
``aloha_agilex`` published an all-zero ``/joint_states``. The sidecar now sends the
articulation's qpos/qvel as ``joint_positions`` / ``joint_velocities``.

What is asserted, on the real sidecar and the real ``aloha_agilex`` manifest:

* ``read_state`` names every manifest joint and carries finite positions, grippers in
  the manifest's ``[0, 1]`` range, and a velocity per joint.
* The state is measured, not commanded: after a 0.4 rad ``left_joint1`` target the
  reported joint has followed it (within 0.05 rad) yet differ from it by the drive's tracking error,
  where a commanded value would equal the target exactly.

Skips like the other RoboTwin sim tests when the sidecar venv is not provisioned.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

_REPO = Path(__file__).resolve().parents[2]


def _sidecar_python_available() -> bool:
    from openral_sim.backends.robotwin import _find_sidecar_python

    return _find_sidecar_python() is not None


_WIRE_MISSING = [m for m in ("zmq", "msgpack") if importlib.util.find_spec(m) is None]

pytestmark = [
    pytest.mark.sim,
    pytest.mark.skipif(
        bool(_WIRE_MISSING),
        reason="robotwin wire needs " + ", ".join(_WIRE_MISSING) + " (just sync --group robotwin)",
    ),
    pytest.mark.skipif(
        not _sidecar_python_available(),
        reason=(
            "RoboTwin sidecar venv not provisioned from the current lock "
            "(set OPENRAL_ROBOTWIN_SIDECAR_PYTHON or OPENRAL_ROBOTWIN_AUTO_PROVISION=1)"
        ),
    ),
]


@pytest.fixture(scope="module")
def hal() -> Iterator[Any]:
    import openral_sim.backends  # noqa: F401 — registers the scene factory
    from openral_core import RobotDescription, SceneSpec, SimEnvironment, TaskSpec, VLASpec
    from openral_core.schemas import PhysicsBackend
    from openral_hal.sim_attached import SimAttachedHAL
    from openral_sim.registry import SCENES

    scene = SceneSpec(
        id="robotwin",
        backend=PhysicsBackend.SAPIEN,
        observation_height=128,
        observation_width=128,
        cameras=["camera1", "camera2", "camera3"],
        backend_options={"boot_timeout_s": 900},
    )
    task = TaskSpec(
        id="robotwin/lift_pot",
        scene_id="robotwin",
        instruction="lift the pot with both arms",
        success_key="is_success",
        max_steps=300,
    )
    env_cfg = SimEnvironment(
        robot_id="aloha_agilex",
        scene=scene,
        task=task,
        vla=VLASpec(id="random", weights_uri="none"),
    )
    rollout = SCENES.get("robotwin")(env_cfg)
    desc = RobotDescription.from_yaml(str(_REPO / "robots" / "aloha_agilex" / "robot.yaml"))
    sim_hal = SimAttachedHAL(rollout, desc)
    sim_hal.connect()
    yield sim_hal
    sim_hal.disconnect()
    rollout.close()


def test_read_state_is_the_measured_manifest_pose(hal: Any) -> None:
    state = hal.read_state()
    names = [j.name for j in hal.description.joints]
    assert state.name == names
    pos = np.asarray(state.position)
    assert pos.shape == (14,) and np.all(np.isfinite(pos))
    assert len(state.velocity) == 14
    for i, joint in enumerate(hal.description.joints):
        if joint.role == "gripper":
            assert 0.0 <= pos[i] <= 1.0, (joint.name, pos[i])
    print(f"/joint_states at reset: {dict(zip(names, np.round(pos, 4).tolist(), strict=True))}")


def test_read_state_follows_the_arm_not_the_command(hal: Any) -> None:
    from openral_core import Action, ControlMode

    start = np.asarray(hal.read_state().position)
    target = start.copy()
    target[0] += 0.4  # left_joint1 (waist), well inside its +-2.618 rad range
    # An arm-only row (12 joints, manifest order); the grippers ride GRIPPER_POSITION.
    arm = [i for i, j in enumerate(hal.description.joints) if j.role != "gripper"]
    action = Action(control_mode=ControlMode.JOINT_POSITION, joint_targets=[target[arm].tolist()])

    hal.send_action(action)
    moved = np.asarray(hal.read_state().position)
    print(f"left_joint1 start={start[0]:.5f} target={target[0]:.5f} measured={moved[0]:.5f}")
    err = abs(moved[0] - target[0])
    # The arm followed the command (RoboTwin's step runs many physics substeps) ...
    assert err < 0.05, err
    # ... but read_state is the articulation, not the drive target it was given: a
    # commanded value would equal the target to float precision.
    assert err > 1e-4, err


def _arm_indices(hal: Any) -> list[int]:
    return [i for i, j in enumerate(hal.description.joints) if j.role != "gripper"]


def _step(hal: Any, action: Any, n: int) -> NDArray[np.float64]:
    for _ in range(n):
        hal.send_action(action)
    return np.asarray(hal.read_state().position)


def test_every_arm_joint_drives_its_own_manifest_joint(hal: Any) -> None:
    """Each arm target reaches the joint it names, on both arms.

    RoboTwin's action vector is ``[left 6, left gripper, right 6, right gripper]``; the
    HAL's default packer used to lay an arm row into slots 0..11, so the right arm's
    targets landed one slot early (``right_joint1`` on the left gripper). Distinct
    offsets per joint, reversed on the right arm, make any mis-slot visible.
    """
    from openral_core import Action, ControlMode

    arm = _arm_indices(hal)
    start = np.asarray(hal.read_state().position)
    offsets = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    target = start.copy()
    for k, i in enumerate(arm):
        target[i] += offsets[k] if k < 6 else offsets[11 - k]
    action = Action(control_mode=ControlMode.JOINT_POSITION, joint_targets=[target[arm].tolist()])
    reached = _step(hal, action, 3)
    names = [j.name for j in hal.description.joints]
    print({names[i]: (round(target[i], 3), round(reached[i], 3)) for i in arm})
    np.testing.assert_allclose(reached[arm], target[arm], atol=0.03)


def test_each_gripper_command_reaches_its_own_gripper(hal: Any) -> None:
    from openral_core import Action, ControlMode

    names = [j.name for j in hal.description.joints]
    left, right = names.index("left_gripper"), names.index("right_gripper")
    arm = _arm_indices(hal)
    before = np.asarray(hal.read_state().position)
    close_left = Action(
        control_mode=ControlMode.GRIPPER_POSITION, gripper=[0.0], ee_name="left_gripper"
    )
    after_left = _step(hal, close_left, 20)
    print(f"close left: left {before[left]:.3f}->{after_left[left]:.3f}", end=", ")
    print(f"right {before[right]:.3f}->{after_left[right]:.3f}")
    # Closed: the measured fingers stop at ~0.18 of RoboTwin's gripper_scale span.
    assert after_left[left] < 0.25 and before[left] - after_left[left] > 0.5
    assert abs(after_left[right] - before[right]) < 0.05
    # A gripper command holds the arms where they are, not at 0 rad.
    np.testing.assert_allclose(after_left[arm], before[arm], atol=0.03)
    close_right = Action(
        control_mode=ControlMode.GRIPPER_POSITION, gripper=[0.0], ee_name="right_gripper"
    )
    after_right = _step(hal, close_right, 20)
    assert after_right[right] < 0.25
    assert after_right[left] < 0.25


def test_idle_step_holds_the_measured_pose(hal: Any) -> None:
    """The HAL's idle tick holds RoboTwin's absolute targets; zeros would drive to 0 rad."""
    from openral_core import Action, ControlMode

    arm = _arm_indices(hal)
    target = np.asarray(hal.read_state().position)
    target[arm[0]] += 0.3  # off zero, so a zero "hold" would show
    target[arm[6]] += 0.3
    hal.send_action(
        Action(control_mode=ControlMode.JOINT_POSITION, joint_targets=[target[arm].tolist()])
    )
    before = np.asarray(hal.read_state().position)
    assert abs(before[arm[0]]) > 0.2 and abs(before[arm[6]]) > 0.2
    for _ in range(5):
        assert hal.idle_step() is True
    after = np.asarray(hal.read_state().position)
    print(f"idle: max |delta| {np.max(np.abs(after - before)):.4f}")
    np.testing.assert_allclose(after, before, atol=0.03)


def test_a_stop_does_not_let_a_gripper_command_resume_the_cut_short_target(hal: Any) -> None:
    """After ``stop_stale_motion``, a gripper command holds the arm where the stop left it.

    RoboTwin's targets are absolute, and the HAL packs each command onto the previous
    one: before the fix a stop kept that cached vector, so the next gripper command
    re-sent the interrupted arm target and the arm finished the move the stop cut short.
    """
    from openral_core import Action, ControlMode

    arm = _arm_indices(hal)
    start = np.asarray(hal.read_state().position)
    target = start.copy()
    target[arm[1]] += 2.6  # left_joint2 (range 0..3.14): too long for one step to finish
    hal.send_action(
        Action(control_mode=ControlMode.JOINT_POSITION, joint_targets=[target[arm].tolist()])
    )
    hal.stop_stale_motion()
    stopped = np.asarray(hal.read_state().position)
    close_right = Action(
        control_mode=ControlMode.GRIPPER_POSITION, gripper=[0.0], ee_name="right_gripper"
    )
    after = _step(hal, close_right, 10)
    print(
        f"left_joint2 start={start[arm[1]]:.3f} target={target[arm[1]]:.3f} "
        f"stopped={stopped[arm[1]]:.3f} after={after[arm[1]]:.3f}"
    )
    # The stop really cut the move short ...
    assert abs(stopped[arm[1]] - target[arm[1]]) > 0.15
    # ... and the gripper command holds the arm there instead of finishing the move.
    assert abs(after[arm[1]] - stopped[arm[1]]) < 0.03
