"""Sim test: SimAttachedHAL drives the Isaac panda_mobile base via BODY_TWIST.

ADR-0045 amendment — the deploy-sim HAL generalization. ``SimAttachedHAL`` was
MuJoCo-coupled for the mobile base (``base_pose`` read qpos; a BODY_TWIST raised
without MuJoCo handles). This verifies the obs-fallback path on a real
non-MuJoCo backend (Isaac kinematic base), through the SAME ``SimAttachedHAL``
class (no Mujoco/Isaac subclass split):

* ``base_pose`` reads ``obs["base_pose"]`` (x, y, yaw) — what the ``/odom``
  publisher consumes — instead of returning the neutral (0,0,0);
* a BODY_TWIST ``Action`` (what the ``/cmd_vel`` bridge emits from Nav2) is
  packed into the env action's base slots and stepped, moving the base;
* ``read_state`` stays the 11-joint vector with the base joints tracking the
  kinematic pose.

Skip policy: needs pyzmq/msgpack on the openral venv AND a provisioned Isaac
sidecar venv (RTX GPU). CI without those skips (§1.12).
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
    description = RobotDescription.from_yaml(str(root / "robots" / "panda_mobile" / "robot.yaml"))
    env, seed = build_sim_env_from_yaml(
        str(root / "scenes" / "deploy" / "isaac_panda_mobile_urdf.yaml"),
        robot_id_fallback="panda_mobile",
    )
    hal = SimAttachedHAL(env, description, env_reset_seed=seed)
    hal.connect()
    yield hal, description
    hal.disconnect()


def test_action_dim_is_11(hal) -> None:
    _hal, _desc = hal
    assert _hal._env_action_dim == 11  # 7 arm + 1 gripper + 3 base twist


def test_read_state_is_11_joints(hal) -> None:
    _hal, description = hal
    state = _hal.read_state()
    assert len(state.position) == len(description.joints) == 11
    assert state.name == [j.name for j in description.joints]


def test_base_pose_reads_obs_not_zero_after_twist(hal) -> None:
    from openral_core import Action
    from openral_core.schemas import ControlMode

    _hal, _desc = hal
    # base_pose starts at the origin (obs base_pose = 0,0,0 after reset).
    assert _hal.base_pose == pytest.approx((0.0, 0.0, 0.0), abs=1e-5)

    # Drive the base forward via BODY_TWIST — exactly what the /cmd_vel bridge
    # emits from a Nav2 Twist. vx = +0.5 m/s (base frame), several commands.
    for _ in range(10):
        state = _hal.read_state()
        action = Action(
            control_mode=ControlMode.BODY_TWIST,
            body_twist=[(0.5, 0.0, 0.0, 0.0, 0.0, 0.0)],
            horizon=1,
            stamp_ns=state.stamp_ns,
        )
        _hal.send_action(action)

    # base_pose now reflects the moved kinematic base (obs-fallback path, no
    # MuJoCo) — and base_twist latched the command for /odom.
    x = _hal.base_pose[0]
    assert x > 0.05
    assert _hal.base_twist[0] == pytest.approx(0.5)

    # The base joints in /joint_states track the kinematic pose.
    state = _hal.read_state()
    assert state.position[0] == pytest.approx(x, abs=1e-4)  # base_x
