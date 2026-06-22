"""Sim tests for the robot-agnostic ``tabletop_push`` scene (ADR-0033).

The point of this scene is that the robot is a **flag**: the same composer +
rollout drives any position-controlled arm. So the suite runs the full
:class:`openral_sim.SimRollout` Protocol — compose, ``reset``, ``step``,
observation shapes, geometric success — against **four different robots**
(SO-100, SO-101, Franka, UR5e), resolving each robot's base MJCF from its real manifest.

No mocks (CLAUDE.md §1.11): real ``robots/*/robot.yaml`` manifests, the
production MjSpec composer, and a zero / hand-set action.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

# Force EGL (off-screen) rendering so CI hosts without a display don't abort.
os.environ.setdefault("MUJOCO_GL", "egl")

try:
    import mujoco
except Exception as exc:  # mujoco's eager renderer probe can raise non-ImportError
    _MUJOCO_ERROR: str | None = str(exc)
else:
    _MUJOCO_ERROR = None

# Renderer probe — runs in a SUBPROCESS. Creating a renderer on a headless
# host can call abort() at the C level (SIGABRT), which try/except cannot
# catch and which would crash pytest collection. The subprocess isolates
# that so a headless host skips this suite instead of aborting it.
_RENDERER_ERROR: str | None = None
if _MUJOCO_ERROR is None:
    from tests.sim.conftest import mujoco_renderer_probe_error

    _RENDERER_ERROR = mujoco_renderer_probe_error()


# Robots exercised. Each must have an `assets.mjcf` resolvable via
# robot_descriptions; the suite skips a robot whose MJCF can't be fetched
# (offline CI) rather than failing.
_ROBOTS = ("so100_follower", "so101_follower", "franka_panda", "ur5e")


pytestmark = [
    pytest.mark.skipif(_MUJOCO_ERROR is not None, reason=f"mujoco unavailable: {_MUJOCO_ERROR}"),
    pytest.mark.skipif(
        _RENDERER_ERROR is not None, reason=f"mujoco renderer unavailable: {_RENDERER_ERROR}"
    ),
]


def _make_env(robot_id: str, *, backend_options: dict | None = None, cameras=("overhead", "front")):
    """Build a real :class:`SimEnvironment` for the tabletop_push scene."""
    from openral_core.schemas import SceneSpec, SimEnvironment, TaskSpec, VLASpec

    scene = SceneSpec(
        id="tabletop_push",
        backend="mujoco",
        observation_height=64,
        observation_width=64,
        cameras=list(cameras),
        backend_options=dict(backend_options or {}),
    )
    task = TaskSpec(
        id="tabletop_push/push_to_goal",
        scene_id="tabletop_push",
        instruction="push the red cube onto the green goal marker",
        max_steps=20,
        success_key="is_success",
    )
    vla = VLASpec(id="mock-noop", weights_uri="mock-noop", device="cpu")
    return SimEnvironment(robot_id=robot_id, scene=scene, task=task, vla=vla, seed=7)


def _build_or_skip(robot_id: str, **env_kwargs):
    """Build the rollout, skipping if the robot's MJCF can't be fetched offline."""
    from openral_core.exceptions import ROSConfigError
    from openral_sim import SCENES

    try:
        return SCENES.get("tabletop_push")(_make_env(robot_id, **env_kwargs))
    except ROSConfigError as exc:
        if "robot_descriptions" in str(exc) or "could not load" in str(exc):
            pytest.skip(f"{robot_id} MJCF unavailable: {exc}")
        raise


def _build_yaml_or_skip(path: str):
    """Build a rollout from a real SimScene YAML, preserving its backend options."""
    from openral_core import SimEnvironment, SimScene, VLASpec
    from openral_core.exceptions import ROSConfigError
    from openral_sim import SCENES

    cfg = SimScene.from_yaml(path)
    env = SimEnvironment(
        robot_id=cfg.robot_id or "so101_follower",
        scene=cfg.scene,
        task=cfg.task,
        vla=VLASpec(id="mock-noop", weights_uri="mock-noop", device="cpu"),
        base_pose=cfg.base_pose,
        seed=cfg.seed,
        n_episodes=cfg.n_episodes,
    )
    try:
        return cfg, SCENES.get(cfg.scene.id)(env)
    except ROSConfigError as exc:
        if "robot_descriptions" in str(exc) or "could not load" in str(exc):
            pytest.skip(f"{cfg.robot_id} MJCF unavailable: {exc}")
        raise


def test_tabletop_push_registered_free_axis() -> None:
    """The scene is registered WITHOUT a fixed robot (it's a flag)."""
    from openral_sim import SCENES

    assert "tabletop_push" in SCENES
    assert SCENES.fixed_robot("tabletop_push") is None


@pytest.mark.parametrize("robot_id", _ROBOTS)
def test_tabletop_push_composes_for_robot(robot_id: str) -> None:
    """The composer builds a valid model with the task world for each robot."""
    from openral_core import RobotDescription
    from openral_sim.backends.tabletop_push._assets import compose_tabletop_mjcf

    desc = RobotDescription.from_yaml(f"robots/{robot_id}/robot.yaml")
    try:
        model = compose_tabletop_mjcf(desc)
    except Exception as exc:  # offline MJCF fetch
        if "robot_descriptions" in str(exc) or "could not load" in str(exc):
            pytest.skip(f"{robot_id} MJCF unavailable: {exc}")
        raise

    # The robot's actuators are present and the scene added none of its own.
    assert model.nu >= 1
    # Task-world entities exist.
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube") >= 0
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "goal") >= 0
    for cam in ("overhead", "front"):
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam) >= 0
    # The cube freejoint qpos lands AFTER the robot's qpos (so the robot keeps
    # its low actuator/qpos indices — the manifest contract).
    cube = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    cube_qpos = int(model.jnt_qposadr[int(model.body_jntadr[cube])])
    assert cube_qpos >= model.nu


@pytest.mark.parametrize("robot_id", _ROBOTS)
def test_tabletop_push_reset_step_for_robot(robot_id: str) -> None:
    """reset/step run end-to-end and observation shape tracks the robot's DoF."""
    rollout = _build_or_skip(robot_id)
    obs = rollout.reset(seed=1)
    nu = rollout._n_act
    assert set(obs["images"].keys()) == {"overhead", "front"}
    assert obs["images"]["overhead"].shape == (64, 64, 3)
    assert obs["images"]["overhead"].dtype == np.uint8
    assert obs["state"].shape == (nu,)
    assert obs["state"].dtype == np.float32
    assert obs["task"] == "push the red cube onto the green goal marker"

    res = None
    for _ in range(5):
        res = rollout.step(np.zeros(nu, dtype=np.float32))
    assert res is not None
    assert res.info["is_success"] is False
    assert res.terminated is False
    assert isinstance(res.reward, float)
    rollout.close()


@pytest.mark.parametrize("robot_id", _ROBOTS)
def test_tabletop_push_success_geometry(robot_id: str) -> None:
    """Success fires only when the cube rests on the table over the goal."""
    rollout = _build_or_skip(robot_id)
    rollout.reset(seed=0)
    model, data = rollout.mujoco_handles()
    gx, gy = (float(v) for v in model.site_pos[rollout._goal_site_id][:2])
    addr = rollout._cube_qpos_addr

    # On the goal, resting on the table → success.
    data.qpos[addr : addr + 3] = [gx, gy, rollout._resting_cube_z]
    data.qpos[addr + 3 : addr + 7] = [1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(model, data)
    assert rollout._check_on_goal() is True

    # Same XY but lifted well off the table → not success.
    data.qpos[addr + 2] = rollout._resting_cube_z + 0.5
    mujoco.mj_forward(model, data)
    assert rollout._check_on_goal() is False

    # On the table but far from the goal → not success.
    data.qpos[addr : addr + 3] = [gx + 0.5, gy, rollout._resting_cube_z]
    mujoco.mj_forward(model, data)
    assert rollout._check_on_goal() is False
    rollout.close()


def test_tabletop_push_nonzero_action_moves_arm() -> None:
    """A non-zero joint target actually moves the arm (clip uses joint ranges).

    Regression guard mirroring so101_box: position actuators often declare no
    ctrlrange, so clipping to the [0, 0] sentinel would pin every command to
    zero. The rollout clips to the transmission joint's range instead.
    """
    rollout = _build_or_skip("so101_follower")
    q0 = np.asarray(rollout.reset(seed=0)["state"], dtype=np.float64)
    target = np.array([0.4, 0.6, 0.4, 0.3, -0.3, 0.4], dtype=np.float32)
    res = None
    for _ in range(80):
        res = rollout.step(target)
    q1 = np.asarray(res.observation["state"], dtype=np.float64)
    rollout.close()
    assert np.max(np.abs(q1 - q0)) > 0.1, "arm did not move toward the commanded target"


def test_tabletop_push_degree_trained_so101_action_converts_to_radians() -> None:
    """SO-101 LeRobot-degree checkpoints must not be interpreted as radians."""
    offsets = [3.07, 123.16, 124.40, 57.89, -11.04, 9.24]
    rollout = _build_or_skip(
        "so101_follower",
        backend_options={
            "joint_units": "degrees",
            "joint_offsets_deg": offsets,
            "joint_signs": [1, 1, 1, 1, 1, 1],
        },
    )
    try:
        rollout.reset(seed=0)
        target_rad = np.asarray([0.1, 0.4, 0.35, 0.2, -0.15, 0.25], dtype=np.float64)
        action_deg = np.degrees(target_rad) + np.asarray(offsets, dtype=np.float64)
        rollout.step(action_deg.astype(np.float32))
        np.testing.assert_allclose(
            rollout._data.ctrl[: rollout._n_act],
            np.clip(target_rad, rollout._act_clip_ranges[:, 0], rollout._act_clip_ranges[:, 1]),
            atol=1e-6,
        )
    finally:
        rollout.close()


def test_tabletop_push_initial_joint_positions_seed_reset_pose() -> None:
    """A configured policy-unit reset pose seeds qpos and ctrl before the first frame."""
    initial_deg = [-4.90, -100.00, 90.00, 77.11, -96.76, 2.69]
    rollout = _build_or_skip(
        "so101_follower",
        backend_options={
            "joint_units": "degrees",
            "initial_joint_positions": initial_deg,
        },
    )
    try:
        obs = rollout.reset(seed=0)
        expected_rad = np.radians(np.asarray(initial_deg, dtype=np.float64))
        np.testing.assert_allclose(
            rollout._data.ctrl[: rollout._n_act],
            np.clip(expected_rad, rollout._act_clip_ranges[:, 0], rollout._act_clip_ranges[:, 1]),
            atol=1e-6,
        )
        np.testing.assert_allclose(obs["state"], initial_deg, atol=1.0)
    finally:
        rollout.close()


def test_tabletop_push_degree_affine_scales_policy_units() -> None:
    """Degree mode supports servo-degree scale/offset calibration, not just offsets."""
    offsets = np.array([-25.5832, -30.3170, 33.9902, 82.4531, -124.1312, 3.0111])
    scales = np.array([0.3227, 0.8624, 0.8205, 0.2274, 0.2238, 0.2731])
    initial_policy_deg = np.array([-4.90, -103.31, 96.09, 77.11, -96.76, 2.69])
    action_policy_deg = initial_policy_deg + np.array([0.3, -1.0, 2.0, 1.0, -0.5, 0.5])
    rollout = _build_or_skip(
        "so101_follower",
        backend_options={
            "joint_units": "degrees",
            "joint_offsets_deg": offsets.tolist(),
            "joint_scales": scales.tolist(),
            "initial_joint_positions": initial_policy_deg.tolist(),
        },
    )
    try:
        obs = rollout.reset(seed=0)
        expected_initial_rad = np.radians((initial_policy_deg - offsets) / scales)
        np.testing.assert_allclose(
            rollout._data.ctrl[: rollout._n_act],
            np.clip(
                expected_initial_rad,
                rollout._act_clip_ranges[:, 0],
                rollout._act_clip_ranges[:, 1],
            ),
            atol=1e-6,
        )
        np.testing.assert_allclose(obs["state"], initial_policy_deg, atol=1.0)

        rollout.step(action_policy_deg.astype(np.float32))
        expected_action_rad = np.radians((action_policy_deg - offsets) / scales)
        np.testing.assert_allclose(
            rollout._data.ctrl[: rollout._n_act],
            np.clip(
                expected_action_rad,
                rollout._act_clip_ranges[:, 0],
                rollout._act_clip_ranges[:, 1],
            ),
            atol=1e-6,
        )
    finally:
        rollout.close()


def test_tabletop_push_bad_joint_scales_rejected() -> None:
    """Scale calibration is one positive value per actuator."""
    from openral_core.exceptions import ROSConfigError
    from openral_sim import SCENES

    bad_width = _make_env(
        "so101_follower",
        backend_options={
            "joint_units": "degrees",
            "joint_scales": [1.0, 1.0],
        },
    )
    with pytest.raises(ROSConfigError, match="joint_scales"):
        SCENES.get("tabletop_push")(bad_width)

    bad_zero = _make_env(
        "so101_follower",
        backend_options={
            "joint_units": "degrees",
            "joint_scales": [1.0, 1.0, 1.0, 0.0, 1.0, 1.0],
        },
    )
    with pytest.raises(ROSConfigError, match="joint_scales"):
        SCENES.get("tabletop_push")(bad_zero)


def test_tabletop_push_bad_initial_joint_positions_rejected() -> None:
    """The reset pose is one value per actuator; wrong-width configs fail loudly."""
    from openral_core.exceptions import ROSConfigError
    from openral_sim import SCENES

    env = _make_env(
        "so101_follower",
        backend_options={
            "joint_units": "degrees",
            "initial_joint_positions": [0.0, 0.0],
        },
    )
    with pytest.raises(ROSConfigError, match="initial_joint_positions"):
        SCENES.get("tabletop_push")(env)


def test_tabletop_push_random_spawn_varies_with_seed() -> None:
    """Different seeds produce different cube + goal spawns."""
    poses: dict[int, tuple] = {}
    for s in (0, 1, 2):
        rollout = _build_or_skip("so101_follower")
        rollout.reset(seed=s)
        model, data = rollout.mujoco_handles()
        addr = rollout._cube_qpos_addr
        cube_xy = (float(data.qpos[addr]), float(data.qpos[addr + 1]))
        goal_xy = tuple(float(v) for v in model.site_pos[rollout._goal_site_id][:2])
        poses[s] = (cube_xy, goal_xy)
        rollout.close()
    assert len({repr(v) for v in poses.values()}) == 3


def test_tabletop_cube_push_yaml_places_workspace_on_initial_gripper_side() -> None:
    """SO-101 pi0.5 reset pose and cube-push workspace must share the same side."""
    cfg, rollout = _build_yaml_or_skip("scenes/sim/tabletop_cube_push.yaml")
    try:
        rollout.reset(seed=cfg.seed)
        model, data = rollout.mujoco_handles()

        base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
        gripper_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "gripper")
        assert base_id >= 0
        assert gripper_id >= 0

        base_y = float(data.xpos[base_id, 1])
        gripper_y = float(data.xpos[gripper_id, 1])
        cube_y = float(data.qpos[rollout._cube_qpos_addr + 1])
        goal_y = float(model.site_pos[rollout._goal_site_id][1])
        table_y = float(cfg.scene.backend_options["table_center_xy"][1])

        gripper_side = gripper_y - base_y
        assert abs(gripper_side) > 1e-3
        for y in (cube_y, goal_y, table_y):
            assert (y - base_y) * gripper_side > 0.0
        assert abs(goal_y - base_y) > abs(cube_y - base_y)
    finally:
        rollout.close()


def test_tabletop_cube_push_sim_attached_hal_reads_so101_joint_state() -> None:
    """Deploy-sim HAL state must use SO-101 MJCF joint aliases, not zeros."""
    from openral_core import RobotDescription
    from openral_hal.sim_attached import SimAttachedHAL

    cfg, rollout = _build_yaml_or_skip("scenes/sim/tabletop_cube_push.yaml")
    desc = RobotDescription.from_yaml("robots/so101_follower/robot.yaml")
    hal = SimAttachedHAL(rollout, desc, env_reset_seed=cfg.seed)
    try:
        hal.connect()
        state = hal.read_state()
        expected_rad = np.asarray(
            [rollout._data.qpos[addr] for addr in rollout._act_qpos_addrs],
            dtype=np.float64,
        )

        assert state.name == [joint.name for joint in desc.joints]
        np.testing.assert_allclose(state.position, expected_rad, atol=1e-6)
        assert np.linalg.norm(np.asarray(state.position, dtype=np.float64)) > 0.1
    finally:
        hal.disconnect()
        rollout.close()


def test_tabletop_push_wrist_camera_opt_in() -> None:
    """A named mount body adds a wrist camera to the observation."""
    rollout = _build_or_skip(
        "so101_follower",
        backend_options={"wrist_camera_mount_body": "gripper"},
        cameras=("overhead", "front", "wrist"),
    )
    obs = rollout.reset(seed=0)
    assert "wrist" in obs["images"]
    rollout.close()


def test_tabletop_push_wrist_camera_inferred_from_manifest() -> None:
    """Requesting `wrist` auto-resolves the mount body from the robot manifest."""
    rollout = _build_or_skip("so100_follower", cameras=("overhead", "front", "wrist"))
    obs = rollout.reset(seed=0)
    assert "wrist" in obs["images"]
    rollout.close()


def test_tabletop_push_bad_wrist_mount_body_rejected() -> None:
    """An end-effector body absent from the MJCF fails loudly at build time."""
    from openral_core.exceptions import ROSConfigError
    from openral_sim import SCENES

    env = _make_env("so101_follower", backend_options={"wrist_camera_mount_body": "no_such_body"})
    with pytest.raises(ROSConfigError, match="wrist_camera_mount_body"):
        SCENES.get("tabletop_push")(env)


def test_tabletop_push_unknown_backend_option_rejected() -> None:
    """An unknown backend_options key fails loudly (typo guard)."""
    from openral_core.exceptions import ROSConfigError
    from openral_sim import SCENES

    env = _make_env("so101_follower", backend_options={"taable_size_xy": [0.5, 0.5]})
    with pytest.raises(ROSConfigError, match=r"unknown scene\.backend_options"):
        SCENES.get("tabletop_push")(env)


def test_tabletop_push_action_dim_guard() -> None:
    """A wrong-width action is rejected with the robot's actuator count."""
    from openral_core.exceptions import ROSConfigError

    rollout = _build_or_skip("so101_follower")
    rollout.reset(seed=0)
    with pytest.raises(ROSConfigError, match="joint-position action"):
        rollout.step(np.zeros(3, dtype=np.float32))
    rollout.close()
