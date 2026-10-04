"""Sim test: bimanual OpenArm imported from its URDF into the Isaac warehouse.

Boots the shipped ``scenes/deploy/isaac_openarm_warehouse.yaml`` through the
real deploy loader on a real Isaac Sim and checks the URDF import holds up:

* all 16 manifest joints map onto URDF DOFs (manifest ``left_joint1`` vs URDF
  ``openarm_left_joint1``) and hold under zero action — the regression for
  Isaac 6's mimic constraint, which spun the wrists until the importer was fed a
  mimic-free URDF;
* each jaw opens toward its own manifest range (left ``[0, 0.785]``, right
  ``[-0.785, 0]``) and closes, its second finger following via the URDF mimic
  relation the scene drives itself;
* the robot is physically at its spawn and the props rest on the pallet.

Skip policy: as the panda_mobile warehouse test. The URDF's ``package://``
meshes come from a sourced workspace or, failing that, a pinned clone of
Enactic's public ``openarm_description`` (network on first use).
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

# OpenArm action: 14 arm deltas (manifest order), then left + right gripper.
_N_ARM = 14
# Manifest joint order: left_joint1..7, left_gripper, right_joint1..7, right_gripper.
_LEFT_GRIPPER, _RIGHT_GRIPPER = 7, 15


@pytest.fixture(scope="module")
def env() -> Iterator[Any]:
    import openral_sim.backends  # noqa: F401 — registers the isaac_sim scene factory
    from openral_hal.sim_bringup import build_sim_env_from_yaml

    yaml = _repo_root() / "scenes" / "deploy" / "isaac_openarm_warehouse.yaml"
    sim_env, seed = build_sim_env_from_yaml(str(yaml))
    sim_env.reset(seed=seed)
    yield sim_env
    sim_env.close()


def _steps(env: Any, action: np.ndarray, n: int) -> Any:
    result = None
    for _ in range(n):
        result = env.step(action)
    return result


def test_all_joints_hold_and_the_robot_is_at_its_spawn(env: Any) -> None:
    assert env.action_dim == _N_ARM + 2
    result = _steps(env, np.zeros(env.action_dim, dtype=np.float32), 60)
    joints = result.observation["joint_positions"]
    assert joints.shape == (16,)
    assert np.max(np.abs(joints)) < 0.02, f"joints drifted under zero action: {joints}"
    x, y, _z = result.info["robot_position"]
    assert (x, y) == pytest.approx((-1.05, 5.10), abs=1e-3)
    for name, (_ox, _oy, oz) in result.info["object_positions"].items():
        assert 0.21 < oz < 0.36, f"{name} not resting on the pallet: z={oz}"


def test_each_jaw_opens_into_its_own_manifest_range(env: Any) -> None:
    action = np.zeros(env.action_dim, dtype=np.float32)
    action[_N_ARM:] = 1.0
    opened = _steps(env, action, 40).observation["joint_positions"]
    assert opened[_LEFT_GRIPPER] == pytest.approx(0.7854, abs=0.03)
    assert opened[_RIGHT_GRIPPER] == pytest.approx(-0.7854, abs=0.03)
    action[_N_ARM:] = -1.0
    closed = _steps(env, action, 40).observation["joint_positions"]
    assert closed[_LEFT_GRIPPER] == pytest.approx(0.0, abs=0.03)
    assert closed[_RIGHT_GRIPPER] == pytest.approx(0.0, abs=0.03)
    # The arms never moved while the jaws did.
    arm = np.r_[closed[0:7], closed[8:15]]
    assert np.max(np.abs(arm)) < 0.02
