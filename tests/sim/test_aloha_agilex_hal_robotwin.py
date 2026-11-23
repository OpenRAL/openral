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
    # Only the left arm: the default packer lays the 12 arm targets into env slots
    # 0..11, but RoboTwin's vector is [left 6, left gripper, right 6, right gripper], so
    # the right arm's targets land one slot early (a separate actuation-path defect).
    # The default packer takes the 12 arm joints; the grippers ride GRIPPER_POSITION.
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
