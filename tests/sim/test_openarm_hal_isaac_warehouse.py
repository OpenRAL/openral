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

# OpenArm action (IsaacActionLayout): one absolute target per manifest joint, in
# manifest order — left_joint1..7, left_gripper, right_joint1..7, right_gripper.
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
    assert env.action_dim == 16
    result = _steps(env, np.zeros(env.action_dim, dtype=np.float32), 60)
    joints = result.observation["joint_positions"]
    assert joints.shape == (16,)
    assert np.max(np.abs(joints)) < 0.02, f"joints drifted under zero action: {joints}"
    x, y, _z = result.info["robot_position"]
    assert (x, y) == pytest.approx((-1.05, 5.10), abs=1e-3)
    for name, (_ox, _oy, oz) in result.info["object_positions"].items():
        assert 0.21 < oz < 0.36, f"{name} not resting on the pallet: z={oz}"


def test_head_depth_is_a_real_cloud_in_openarm_base(env: Any) -> None:
    """The head depth camera, mounted on openarm_base at the manifest's nominal ZED
    pose, yields finite points expressed in openarm_base: the floor 0.698 m below it.

    Regressions: the near clip plane sat at Isaac's default 1 stage unit, so every
    surface within a metre was inf and the cloud was all-NaN; the cloud was
    stamped openarm_base while expressed in the URDF root frame, 0.698 m too high;
    and the camera rode the robot root Xform, which Isaac >= 6 leaves at the origin.
    """
    result = _steps(env, np.zeros(env.action_dim, dtype=np.float32), 10)
    cloud = result.observation["depth_points"]["head_zed"]
    assert cloud.shape[0] > 1000
    assert np.isfinite(cloud).all()
    assert float(np.min(cloud[:, 2])) == pytest.approx(-0.698, abs=0.02)
    assert float(np.median(cloud[:, 0])) > 0.0  # it looks forward, at the pallet
    # ...from the robot, not from the stage origin: on Isaac >= 6 the pinned robot root
    # Xform stays at the import origin, so a camera parented to it looked at the floor
    # 5 m away (median y -5.1 here) while every robot link rendered at the spawn.
    assert abs(float(np.median(cloud[:, 1]))) < 0.3


def test_head_camera_ships_registered_rgbd_frames(env: Any) -> None:
    """The head camera's colour and metric depth come from ONE render, with Isaac's K
    and the camera's optical pose in openarm_base — what the vision attachment leg
    needs (a real RGB-D driver publishes the same four streams).

    Registration is checked against Isaac's own deprojection: depth pixels lifted
    through ``k`` and ``optical_in_base`` land on the ``depth_points`` cloud Isaac
    computed for the same step (``Camera.get_pointcloud``).
    """
    from openral_core import RobotDescription
    from openral_hal.sim_attached import SimAttachedHAL

    result = None
    for _ in range(6):
        result = env.step(np.full(env.action_dim, np.nan, dtype=np.float32))
        if "depth_frames" in result.observation:
            break
    assert result is not None and "depth_frames" in result.observation
    frame = result.observation["depth_frames"]["head_zed"]
    depth, rgb, k, pose = frame["depth"], frame["rgb"], frame["k"], frame["optical_in_base"]
    h, w = depth.shape
    assert rgb.shape == (h, w, 3) and rgb.dtype == np.uint8
    fx, fy, cx, cy = k
    assert fx == pytest.approx(fy, rel=1e-3)
    assert (cx, cy) == pytest.approx((w / 2.0, h / 2.0), abs=1.0)
    # Optical frame at the head mount (openarm_base + 0.2 m z), looking forward-down.
    assert pose[:3, 3] == pytest.approx((0.0, 0.0, 0.2), abs=1e-3)
    assert pose[0, 2] > 0.2 and pose[2, 2] < -0.8  # optical z: forward and down
    v, u = np.nonzero(depth > 0.0)
    pick = np.linspace(0, len(u) - 1, 200).astype(int)
    z = depth[v[pick], u[pick]]
    optical = np.stack([(u[pick] - cx) * z / fx, (v[pick] - cy) * z / fy, z], axis=1)
    lifted = optical @ pose[:3, :3].T + pose[:3, 3]
    cloud = result.observation["depth_points"]["head_zed"]
    nearest = np.min(np.linalg.norm(lifted[:, None, :] - cloud[None, ::7, :], axis=2), axis=1)
    assert float(np.median(nearest)) < 0.01
    # The HAL keeps the newest frame for the sensor bridge between the steps that carry one.
    desc = RobotDescription.from_yaml(str(_repo_root() / "robots" / "openarm" / "robot.yaml"))
    hal = SimAttachedHAL(env, desc)
    hal.connect()
    for _ in range(3):
        hal.idle_step()
    assert set(hal.read_depth_frames()) == {"head_zed"}


def test_each_jaw_opens_into_its_own_manifest_range(env: Any) -> None:
    action = np.zeros(env.action_dim, dtype=np.float32)
    action[_LEFT_GRIPPER], action[_RIGHT_GRIPPER] = 0.7854, -0.7854  # each jaw's open end
    opened = _steps(env, action, 40).observation["joint_positions"]
    assert opened[_LEFT_GRIPPER] == pytest.approx(0.7854, abs=0.03)
    assert opened[_RIGHT_GRIPPER] == pytest.approx(-0.7854, abs=0.03)
    action[_LEFT_GRIPPER] = action[_RIGHT_GRIPPER] = 0.0
    closed = _steps(env, action, 40).observation["joint_positions"]
    assert closed[_LEFT_GRIPPER] == pytest.approx(0.0, abs=0.03)
    assert closed[_RIGHT_GRIPPER] == pytest.approx(0.0, abs=0.03)
    # The arms never moved while the jaws did.
    arm = np.r_[closed[0:7], closed[8:15]]
    assert np.max(np.abs(arm)) < 0.02


def test_hal_commits_a_bimanual_slot_tick(env: Any) -> None:
    """Regression: the HAL refused every OpenArm slot tick (a zero-padded 16-wide
    JOINT_POSITION row vs its 14-wide arm packer). Placed by joint_names /
    ee_name, one tick drives both arms and both jaws; joints stop at their URDF
    limits (left/right_joint2 at +-0.1745, right_joint4 at 0)."""
    from openral_core import Action, RobotDescription
    from openral_core.schemas import ControlMode
    from openral_hal.sim_attached import SimAttachedHAL

    desc = RobotDescription.from_yaml(str(_repo_root() / "robots" / "openarm" / "robot.yaml"))
    names = [j.name for j in desc.joints]
    left = [f"left_joint{i}" for i in range(1, 8)]
    right = [f"right_joint{i}" for i in range(1, 8)]

    def padded(value: float, sel: list[str]) -> list[list[float]]:
        row = [0.0] * len(names)
        for name in sel:
            row[names.index(name)] = value
        return [row]

    hal = SimAttachedHAL(env, desc)
    hal.connect()
    for tick in range(1, 61):
        for action in (
            Action(
                control_mode=ControlMode.JOINT_POSITION,
                joint_targets=padded(0.3, left),
                joint_names=left,
                tick_index=tick,
                tick_group_size=4,
            ),
            Action(
                control_mode=ControlMode.GRIPPER_POSITION,
                gripper=[0.7],
                ee_name="left_gripper",
                tick_index=tick,
                tick_group_size=4,
            ),
            Action(
                control_mode=ControlMode.JOINT_POSITION,
                joint_targets=padded(-0.3, right),
                joint_names=right,
                tick_index=tick,
                tick_group_size=4,
            ),
            Action(
                control_mode=ControlMode.GRIPPER_POSITION,
                gripper=[-0.7],
                ee_name="right_gripper",
                tick_index=tick,
                tick_group_size=4,
            ),
        ):
            hal.send_action(action)
    q = dict(zip(names, hal.read_state().position, strict=True))
    for name in left:
        want = 0.1745 if name == "left_joint2" else 0.3
        assert q[name] == pytest.approx(want, abs=0.03), name
    for name in right:
        want = {"right_joint2": -0.1745, "right_joint4": 0.0}.get(name, -0.3)
        assert q[name] == pytest.approx(want, abs=0.03), name
    assert q["left_gripper"] == pytest.approx(0.7, abs=0.03)
    assert q["right_gripper"] == pytest.approx(-0.7, abs=0.03)
