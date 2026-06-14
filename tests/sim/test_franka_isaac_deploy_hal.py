"""Sim test: `openral deploy sim` HAL bring-up against the Isaac Sim sidecar (ADR-0045).

Exercises the deploy-sim seam — `build_sim_env_from_yaml(<DeployScene>)` →
`SimAttachedHAL(SimRollout)` — for a real Isaac Sim scene, in-process (no ROS
launch). This is the path `openral deploy sim --config
scenes/deploy/isaac_franka.yaml` drives through the manifest HAL lifecycle node.

What is asserted
----------------
* `build_sim_env_from_yaml` resolves the `isaac_sim` DeployScene and auto-spawns
  the sidecar.
* `SimAttachedHAL.connect()` succeeds — i.e. `_probe_env_action_dim` reads
  `_IsaacSimSidecar.action_dim` (the hard blocker this PR fixes; it raised
  `ROSConfigError` before).
* `read_images()` returns the Isaac RTX camera frame(s).
* `read_state()` returns a JointState shaped to the franka manifest (8 joints,
  matching names) with REAL joint angles sourced from the sidecar's
  `obs["joint_positions"]` (ADR-0034 amendment — non-MuJoCo backends are no
  longer stuck at all-zeros).
* `send_action()` (JOINT_POSITION, 8 targets → env_action_dim=8) steps the env
  without raising.

Skip policy: same as `test_franka_random_isaac` — needs pyzmq/msgpack on the
openral venv AND a provisioned Isaac sidecar venv (RTX GPU). CI without those
skips (§1.12).
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest


def _sidecar_python_available() -> bool:
    override = os.environ.get("OPENRAL_ISAAC_SIDECAR_PYTHON")
    if override:
        return Path(override).is_file()
    default = Path.home() / ".cache" / "openral" / "isaac-sidecar" / ".venv" / "bin" / "python"
    return default.is_file()


_WIRE_MISSING = [m for m in ("zmq", "msgpack") if importlib.util.find_spec(m) is None]

pytestmark = [
    pytest.mark.sim,
    pytest.mark.skipif(
        bool(_WIRE_MISSING),
        reason="isaac_sim wire needs " + ", ".join(_WIRE_MISSING) + " (uv sync --group isaacsim)",
    ),
    pytest.mark.skipif(
        not _sidecar_python_available(),
        reason="Isaac Sim sidecar venv not provisioned (set OPENRAL_ISAAC_SIDECAR_PYTHON)",
    ),
]


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for ancestor in (here, *here.parents):
        if (ancestor / "robots").is_dir() and (ancestor / "pyproject.toml").is_file():
            return ancestor
    raise RuntimeError("could not locate repo root from test file")


@pytest.fixture(scope="module")
def hal():
    import openral_sim.backends  # noqa: F401 — registers the isaac_sim scene factory
    from openral_core import RobotDescription
    from openral_hal.sim_attached import SimAttachedHAL
    from openral_hal.sim_bringup import build_sim_env_from_yaml

    root = _repo_root()
    description = RobotDescription.from_yaml(str(root / "robots" / "franka_panda" / "robot.yaml"))
    env, seed = build_sim_env_from_yaml(
        str(root / "scenes" / "deploy" / "isaac_franka.yaml"),
        robot_id_fallback="franka_panda",
    )
    hal = SimAttachedHAL(env, description, env_reset_seed=seed)
    hal.connect()  # resets the env + probes env.action_dim (the deploy-sim blocker)
    yield hal, description
    hal.disconnect()


def test_connect_resolves_action_dim(hal) -> None:
    _hal, _desc = hal
    # If connect() returned, _probe_env_action_dim resolved env.action_dim;
    # lift_cube is 8-D (7 arm joints + gripper).
    assert _hal._env_action_dim == 8


def test_read_state_has_real_joint_values(hal) -> None:
    _hal, description = hal
    state = _hal.read_state()
    assert len(state.position) == len(description.joints)
    assert state.name == [j.name for j in description.joints]
    # ADR-0034 amendment: SimAttachedHAL now sources real joint angles from the
    # sidecar's obs["joint_positions"] for a non-MuJoCo backend (was all-zeros).
    # The Franka's default reset pose has non-zero arm angles.
    assert any(abs(p) > 1e-6 for p in state.position)


def test_read_images_returns_isaac_frame(hal) -> None:
    import numpy as np

    _hal, _desc = hal
    images = _hal.read_images()
    assert "camera1" in images
    frame = np.asarray(images["camera1"])
    assert frame.shape == (256, 256, 3)
    assert frame.dtype == np.uint8


def test_send_action_steps_without_raising(hal) -> None:
    from openral_core import Action
    from openral_core.schemas import ControlMode

    _hal, description = hal
    state = _hal.read_state()
    # pack_action_for_env(JOINT_POSITION) wants the ARM joints only (the gripper
    # is packed separately); franka_panda is 7 arm joints + 1 gripper.
    n_arm = sum(1 for j in description.joints if "gripper" not in j.name)
    action = Action(
        control_mode=ControlMode.JOINT_POSITION,
        joint_targets=[list(state.position[:n_arm])],
        horizon=1,
        stamp_ns=state.stamp_ns,
    )
    _hal.send_action(action)  # → env.step over ZMQ; must not raise
    state2 = _hal.read_state()
    assert len(state2.position) == len(description.joints)
