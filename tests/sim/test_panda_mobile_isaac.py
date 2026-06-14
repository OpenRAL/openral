"""Sim test: panda_mobile (arm + kinematic holonomic base) in Isaac Sim.

ADR-0045 amendment (M3). The robot-agnostic manifest scene that brings up
franka_panda (``test_franka_urdf_isaac``) brings up **panda_mobile** by swapping
``robot_id`` — same Panda arm URDF, plus a kinematically-synthesized 3-DOF
holonomic base (the base exists nowhere as an Isaac-importable asset). Driven at
the ``SimRollout`` level (the deploy-sim ROS-graph odom/twist plumbing is a
separate step), so this asserts the scene itself is correct:

* the action surface is 11-D (7 arm + 1 gripper + 3 base twist);
* ``reset`` returns an 11-joint proprio vector (3 base + 7 arm + 1 gripper) and a
  ``base_pose``;
* a forward ``base_twist`` actually moves the base (``base_pose`` x grows and the
  base joints in ``joint_positions`` track it) — the kinematic base works;
* an arm joint-delta still drives the imported articulation.

Skip policy: needs pyzmq/msgpack on the openral venv AND a provisioned Isaac
sidecar venv (RTX GPU). CI without those skips (§1.12).
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import numpy as np
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
def env():
    import openral_sim.backends  # noqa: F401 — registers the isaac_sim scene factory
    from openral_hal.sim_bringup import build_sim_env_from_yaml

    rollout, seed = build_sim_env_from_yaml(
        str(_repo_root() / "scenes" / "deploy" / "isaac_panda_mobile_urdf.yaml"),
        robot_id_fallback="panda_mobile",
    )
    rollout.reset(seed=seed)
    yield rollout
    rollout.close()


def test_action_surface_is_11d(env) -> None:
    # 7 arm joint-deltas + 1 gripper + 3 base twist (vx, vy, wyaw).
    assert env.action_dim == 11


def test_reset_obs_has_11_joints_and_base_pose(env) -> None:
    obs = env.reset()
    assert "base_pose" in obs
    assert np.asarray(obs["base_pose"]).shape == (3,)
    joints = np.asarray(obs["joint_positions"])
    assert joints.shape == (11,)  # 3 base + 7 arm + 1 gripper


def test_reset_obs_has_all_manifest_cameras(env) -> None:
    # panda_mobile declares 3 RGB sensors → camera1/2/3, each a real RTX frame.
    obs = env.reset()
    images = obs["images"]
    assert set(images) == {"camera1", "camera2", "camera3"}
    for key, frame in images.items():
        arr = np.asarray(frame)
        assert arr.shape == (256, 256, 3), key
        assert arr.dtype == np.uint8


def test_reset_obs_has_depth_cloud_in_front(env) -> None:
    # panda_mobile declares a depth sensor (front_depth) → an (N, 3) base_link
    # cloud the bridge publishes as PointCloud2. Never fabricated for a robot
    # without a depth SensorSpec. The forward-facing depth camera sees the ground
    # ahead of the base, so the cloud sits in +x (in front) near z≈0 (ground).
    obs = env.reset()
    assert "depth_points" in obs
    assert "front_depth" in obs["depth_points"]
    cloud = np.asarray(obs["depth_points"]["front_depth"], dtype=float)
    assert cloud.ndim == 2 and cloud.shape[1] == 3
    assert cloud.shape[0] > 100  # a real depth frame, not a stub
    # Geometric sanity (Isaac's get_pointcloud, world→base_link): the bulk of the
    # cloud is in front of the base and at/below the camera height.
    assert float(np.median(cloud[:, 0])) > 0.0  # +x, ahead of the base
    assert float(np.min(cloud[:, 2])) < 0.5  # reaches down toward the ground


def test_reset_obs_has_lidar_scan_seeing_obstacles(env) -> None:
    # panda_mobile declares base_scan (lidar_2d, 360 beams) → a real PhysX
    # raycast fan. The scene seeds static obstacles, so some beams return a hit
    # (range < range_max) while the rest read max. Never a fabricated scan.
    obs = env.reset()
    assert "scan" in obs
    scan = np.asarray(obs["scan"], dtype=float)
    assert scan.shape == (360,)
    assert np.all(scan >= 0.0)
    assert np.all(scan <= 12.0 + 1e-3)  # range_max
    # Obstacles are within range → a meaningful fraction of beams hit something.
    hits = int(np.sum(scan < 12.0 - 1e-3))
    assert hits > 10  # not an all-max (empty-world) fan


def test_base_twist_moves_the_kinematic_base(env) -> None:
    env.reset()
    # Hold the arm/gripper still; drive the base forward (vx = +0.6 m/s).
    forward = np.zeros(11, dtype=np.float32)
    forward[8] = 0.6  # base vx
    last = env.step(forward)
    for _ in range(9):
        last = env.step(forward)

    base_pose = np.asarray(last.observation["base_pose"], dtype=float)
    # Base advanced in +x (world) from the integrated forward twist.
    assert base_pose[0] > 0.05
    # And the base joints in /joint_states track the kinematic pose.
    joints = np.asarray(last.observation["joint_positions"], dtype=float)
    assert joints[0] == pytest.approx(base_pose[0], abs=1e-5)  # base_x


def test_arm_joint_delta_drives_articulation(env) -> None:
    env.reset()
    start = np.asarray(env.reset()["joint_positions"], dtype=float)[3:10]  # arm slots
    delta = np.zeros(11, dtype=np.float32)
    delta[:7] = 0.5  # arm joint deltas
    last = None
    for _ in range(8):
        last = env.step(delta)
    end = np.asarray(last.observation["joint_positions"], dtype=float)[3:10]
    assert float(np.max(np.abs(end - start))) > 1e-3
