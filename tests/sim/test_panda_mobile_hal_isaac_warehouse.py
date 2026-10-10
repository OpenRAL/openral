"""Sim test: panda_mobile in NVIDIA's Isaac warehouse, with pickable YCB props.

Boots the shipped ``scenes/deploy/isaac_panda_mobile_warehouse.yaml`` through the
real deploy loader on a real Isaac Sim (environment USD + spawn pose + scene
objects) and checks what the three fields promise:

* the robot is physically at its spawn (aisle, facing +y) — PhysX ground truth,
  not the odometry it reports;
* the three YCB props settle on the pallet (graspable rigid bodies, not
  floating or falling through);
* the arm holds under zero action, the gripper opens/closes in manifest units;
* a base twist moves the robot physically down the aisle.

Skip policy: needs pyzmq/msgpack on the openral venv AND an Isaac Sim install
(binary install or sidecar venv; RTX GPU) plus network access to NVIDIA's asset
server. CI without those skips (§1.12).
"""

from __future__ import annotations

import importlib.util
from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest

from tests.sim.conftest import _repo_root, _sidecar_python_available

_WIRE_MISSING = [m for m in ("zmq", "msgpack") if importlib.util.find_spec(m) is None]

pytestmark = [
    pytest.mark.sim,
    pytest.mark.skipif(
        bool(_WIRE_MISSING),
        reason="isaac_sim wire needs " + ", ".join(_WIRE_MISSING) + " (just sync --group isaacsim)",
    ),
    pytest.mark.skipif(
        not _sidecar_python_available(),
        reason="Isaac Sim sidecar venv not provisioned (set OPENRAL_ISAAC_SIDECAR_PYTHON)",
    ),
]

# panda_mobile action (IsaacActionLayout): 7 absolute arm targets, the gripper in
# manifest units ([0, 1] width), then the base twist (vx, vy, wz). NaN = hold.
_GRIPPER_SLOT = 7
_VX_SLOT = 8
_SPAWN_XY = (-4.8, 0.0)
_PALLET_TOP_Z = 0.21


@pytest.fixture(scope="module")
def env() -> Iterator[Any]:
    import openral_sim.backends  # noqa: F401 — registers the isaac_sim scene factory
    from openral_hal.sim_bringup import build_sim_env_from_yaml

    yaml = _repo_root() / "scenes" / "deploy" / "isaac_panda_mobile_warehouse.yaml"
    sim_env, seed = build_sim_env_from_yaml(str(yaml))
    sim_env.reset(seed=seed)
    yield sim_env
    sim_env.close()


def _steps(env: Any, action: np.ndarray, n: int) -> Any:
    result = None
    for _ in range(n):
        result = env.step(action)
    return result


def test_robot_is_at_its_spawn_and_props_rest_on_the_pallet(env: Any) -> None:
    result = _steps(env, np.zeros(env.action_dim, dtype=np.float32), 60)
    x, y, _z = result.info["robot_position"]
    assert (x, y) == pytest.approx(_SPAWN_XY, abs=1e-3)
    objects = result.info["object_positions"]
    assert set(objects) == {"cracker_box", "sugar_box", "mustard_bottle"}
    for name, (ox, oy, oz) in objects.items():
        assert -0.75 < ox < 0.26 and 4.48 < oy < 5.70, f"{name} left the pallet: {ox, oy}"
        assert _PALLET_TOP_Z < oz < _PALLET_TOP_Z + 0.15, f"{name} not resting on it: z={oz}"
    # A zero action is a zero-pose target (joint4 sits at its -0.07 limit).
    joints = result.observation["joint_positions"]
    assert np.max(np.abs(joints[3:10] - np.array([0, 0, 0, -0.07, 0, 0, 0]))) < 0.02


def test_gripper_opens_and_closes_in_manifest_units(env: Any) -> None:
    # Commands arrive in the end effector's command_convention: panda_mobile's is
    # normalized_close_symmetric (+1 closed, -1 open, #342); /joint_states reads
    # the manifest's normalised [0, 1] width (URDF: 0..0.04 m).
    action = np.zeros(env.action_dim, dtype=np.float32)
    action[_GRIPPER_SLOT] = -1.0
    opened = _steps(env, action, 40).observation["joint_positions"][10]
    action[_GRIPPER_SLOT] = 1.0
    closed = _steps(env, action, 40).observation["joint_positions"][10]
    assert opened == pytest.approx(1.0, abs=0.05)
    assert closed == pytest.approx(0.0, abs=0.05)


def test_base_twist_drives_the_robot_down_the_aisle(env: Any) -> None:
    start = _steps(env, np.zeros(env.action_dim, dtype=np.float32), 1)
    action = np.zeros(env.action_dim, dtype=np.float32)
    action[_VX_SLOT] = 0.5  # forward in the base frame = +y (spawn faces down the aisle)
    end = _steps(env, action, 40)
    dx, dy = (end.info["robot_position"][i] - start.info["robot_position"][i] for i in range(2))
    # 40 commands x 0.5 m/s x 0.05 s body_twist interval = 1.0 m, physically.
    assert dy == pytest.approx(1.0, abs=0.02)
    assert dx == pytest.approx(0.0, abs=0.02)
    assert end.observation["base_pose"][0] - start.observation["base_pose"][0] == pytest.approx(
        1.0, abs=0.02
    )


def test_hal_arm_command_moves_the_arm_and_not_the_base(env: Any) -> None:
    """Regression: the HAL's robosuite-shaped packer put an arm JOINT_POSITION
    into this scene's base-twist slots (the base drove 1 m) and a gripper
    command into wz (the base spun 1 rad). Through the env's own name-addressed
    packer the arm reaches its targets and the base stays put."""
    from openral_core import Action, RobotDescription
    from openral_core.schemas import ControlMode
    from openral_hal.sim_attached import SimAttachedHAL

    desc = RobotDescription.from_yaml(str(_repo_root() / "robots" / "panda_mobile" / "robot.yaml"))
    hal = SimAttachedHAL(env, desc)
    hal.connect()
    hold = np.full(env.action_dim, np.nan, dtype=np.float32)
    hold[-3:] = 0.0
    start = env.step(hold).info["robot_position"]
    arm = [0.0, 0.5, 0.0, -1.0, 0.0, 1.0, 0.0]
    for _ in range(60):
        hal.send_action(Action(control_mode=ControlMode.JOINT_POSITION, joint_targets=[arm]))
    for _ in range(40):
        # normalized_close_symmetric: -1 = open (the width reads 1.0).
        hal.send_action(Action(control_mode=ControlMode.GRIPPER_POSITION, gripper=[-1.0]))
    end = env.step(hold)
    joints = dict(zip([j.name for j in desc.joints], hal.read_state().position, strict=True))
    for i, target in enumerate(arm, start=1):
        assert joints[f"panda_joint{i}"] == pytest.approx(target, abs=0.05)
    assert joints["panda_gripper"] == pytest.approx(1.0, abs=0.05)
    assert end.info["robot_position"] == pytest.approx(start, abs=1e-3)
    assert hal.base_pose == pytest.approx((0.0, 0.0, 0.0), abs=1e-4)
