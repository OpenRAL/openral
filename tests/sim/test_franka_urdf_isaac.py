"""Sim test: URDF-driven (robot-agnostic) Isaac scene via the deploy-sim seam.

ADR-0045 amendment (M1). Unlike ``test_franka_isaac_deploy_hal`` — which uses the
``lift_cube`` layout's hardcoded Isaac built-in Franka example asset — this test
drives ``scenes/deploy/isaac_franka_urdf.yaml`` (``--layout manifest``): the
sidecar **imports the franka_panda manifest's URDF** via Isaac's URDF importer,
maps DOFs to the manifest joint order by name, and drives the arm with a
JOINT_POSITION controller. This is the path that makes any manifest robot
pluggable into Isaac.

What is asserted
----------------
* the manifest scene's robot spec is marshalled and the sidecar imports the URDF
  (``SimAttachedHAL.connect()`` succeeds and probes ``action_dim == 8``);
* ``read_state()`` is shaped to the franka manifest (8 joints, matching names)
  with values sourced from the *imported articulation* — not the Isaac example
  asset;
* ``read_images()`` returns the RTX camera frame;
* ``send_action()`` actually **drives the imported articulation** — commanding a
  non-trivial JOINT_POSITION target over several steps moves the arm off its
  start pose (proving generic joint mapping + controller work end-to-end).

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
    description = RobotDescription.from_yaml(str(root / "robots" / "franka_panda" / "robot.yaml"))
    env, seed = build_sim_env_from_yaml(
        str(root / "scenes" / "deploy" / "isaac_franka_urdf.yaml"),
        robot_id_fallback="franka_panda",
    )
    hal = SimAttachedHAL(env, description, env_reset_seed=seed)
    hal.connect()  # marshals the robot spec, spawns the sidecar, imports the URDF
    yield hal, description
    hal.disconnect()


def test_connect_resolves_action_dim(hal) -> None:
    _hal, _desc = hal
    # 7 arm joints + gripper, derived from the manifest (not a hardcoded 8).
    assert _hal._env_action_dim == 8


def test_read_state_shaped_to_manifest(hal) -> None:
    _hal, description = hal
    state = _hal.read_state()
    assert len(state.position) == len(description.joints)
    assert state.name == [j.name for j in description.joints]


def test_read_images_returns_isaac_frame(hal) -> None:
    import numpy as np

    _hal, _desc = hal
    images = _hal.read_images()
    assert "camera1" in images
    frame = np.asarray(images["camera1"])
    assert frame.shape == (256, 256, 3)
    assert frame.dtype == np.uint8


def test_send_action_drives_imported_articulation(hal) -> None:
    import numpy as np
    from openral_core import Action
    from openral_core.schemas import ControlMode

    _hal, description = hal
    n_arm = sum(1 for j in description.joints if "gripper" not in j.name)
    start = np.asarray(_hal.read_state().position[:n_arm], dtype=float)

    # Command a clearly non-trivial joint target for several steps; the manifest
    # scene applies it as a per-step delta, so the arm should accumulate motion.
    for _ in range(8):
        state = _hal.read_state()
        action = Action(
            control_mode=ControlMode.JOINT_POSITION,
            joint_targets=[[0.5] * n_arm],
            horizon=1,
            stamp_ns=state.stamp_ns,
        )
        _hal.send_action(action)

    end = np.asarray(_hal.read_state().position[:n_arm], dtype=float)
    # The imported URDF articulation actually moved — generic DOF mapping +
    # JOINT_POSITION controller drive the arm, not a no-op.
    assert float(np.max(np.abs(end - start))) > 1e-3
