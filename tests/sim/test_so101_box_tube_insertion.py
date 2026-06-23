"""Sim tests for the ``so101_box`` tube-insertion scene.

Exercises the full :class:`SimRollout` Protocol end-to-end against a
real composed MJCF: model compile, factory build, ``reset()``,
``step()`` with a zero action, observation shape, mujoco_handles
re-entry, and the geometric success check under four synthetic poses
(centred + vertical, off-axis, lifted, horizontal).

No mocks (CLAUDE.md §1.11). The scene is the production composer
output written under ``robot_descriptions`` cache; the policy is the
zero action.
"""

from __future__ import annotations

import math
import os

import numpy as np
import pytest

# Force EGL (off-screen) rendering so CI hosts without a display don't abort.
# The classic renderer calls glXOpenDisplay() and raises SIGABRT on headless
# runners; EGL avoids the display requirement entirely.
os.environ.setdefault("MUJOCO_GL", "egl")

# Lazy GL — every renderer call inside the rollout creates a
# ``mujoco.Renderer`` so the module-level import must succeed before
# the tests run.  Hosts without OSMesa / EGL skip the suite.
try:
    import mujoco
except Exception as exc:  # mujoco's eager renderer probe can raise non-ImportError
    _MUJOCO_ERROR: str | None = str(exc)
else:
    _MUJOCO_ERROR = None

# Renderer probe — runs in a SUBPROCESS. mujoco can be imported even when the
# GL backend is absent; creating a renderer then can call abort() at the C
# level (SIGABRT) on a headless host, which try/except cannot catch and which
# crashes pytest collection. The subprocess turns that abort into a non-zero
# exit we detect, so a headless host skips this suite instead of aborting it.
_RENDERER_ERROR: str | None = None
if _MUJOCO_ERROR is None:
    from tests.sim.conftest import mujoco_renderer_probe_error

    _RENDERER_ERROR = mujoco_renderer_probe_error()

try:
    from robot_descriptions import so_arm101_mj_description as _so101_desc

    _ = _so101_desc.MJCF_PATH  # triggers lazy upstream fetch
    _MJCF_ERROR: str | None = None
except Exception as exc:  # network failure, missing libgit2 wheel, etc.
    _MJCF_ERROR = str(exc)


pytestmark = [
    pytest.mark.skipif(_MUJOCO_ERROR is not None, reason=f"mujoco unavailable: {_MUJOCO_ERROR}"),
    pytest.mark.skipif(
        _RENDERER_ERROR is not None, reason=f"mujoco renderer unavailable: {_RENDERER_ERROR}"
    ),
    pytest.mark.skipif(
        _MJCF_ERROR is not None,
        reason=f"so_arm101_mj_description unavailable: {_MJCF_ERROR}",
    ),
]


@pytest.fixture
def env_cfg():
    """Build a real :class:`SimEnvironment` for the so101_box scene."""
    from openral_core.schemas import (
        SceneSpec,
        SimEnvironment,
        TaskSpec,
        VLASpec,
    )

    scene = SceneSpec(
        id="so101_box",
        backend="mujoco",
        observation_height=64,
        observation_width=64,
        cameras=["oak_top", "wrist"],
    )
    task = TaskSpec(
        id="so101_box/tube_insertion",
        scene_id="so101_box",
        instruction="insert the orange tube into the slotted block",
        max_steps=20,
        success_key="is_success",
    )
    vla = VLASpec(id="mock-noop", weights_uri="mock-noop", device="cpu")
    return SimEnvironment(
        robot_id="so101_follower",
        scene=scene,
        task=task,
        vla=vla,
        seed=7,
    )


def test_so101_box_registered() -> None:
    """The ``so101_box`` scene + ``so101_follower`` robot are both in the registries."""
    from openral_sim import ROBOTS, SCENES

    assert "so101_box" in SCENES
    assert "so101_follower" in ROBOTS
    assert SCENES.fixed_robot("so101_box") == "so101_follower"


def test_so101_box_mjcf_compiles() -> None:
    """The MJCF composer produces a valid model with expected entities."""
    from openral_sim.backends.so101_box._assets import compose_so101_box_mjcf

    xml, path = compose_so101_box_mjcf()
    assert "so101_box_generated.xml" in str(path)
    assert "<mujoco" in xml

    model = mujoco.MjModel.from_xml_path(str(path))
    # 6 arm joints + 2 freejoints (7 qpos each) = 20 qpos / 18 qvel
    assert model.nq == 20
    assert model.nv == 18
    assert model.nu == 6
    # 1 wrist + 1 OAK-D Pro
    assert model.ncam == 2

    for n in ("base", "gripper", "slot_block", "tube"):
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) >= 0, f"missing body {n!r}"
    for n in ("wrist", "oak_top"):
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, n) >= 0, f"missing camera {n!r}"
    for n in ("slot_block_hole", "tube_tip_lo", "tube_tip_hi"):
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, n) >= 0, f"missing site {n!r}"


def test_so101_box_reset_observation_shapes(env_cfg) -> None:
    """``reset`` returns an observation with the expected image keys + shapes."""
    from openral_sim import SCENES

    rollout = SCENES.get("so101_box")(env_cfg)
    obs = rollout.reset(seed=0)
    assert set(obs.keys()) == {"images", "state", "task"}
    assert set(obs["images"].keys()) == {"oak_top", "oak_top_depth", "wrist"}
    assert obs["images"]["oak_top"].shape == (64, 64, 3)
    assert obs["images"]["oak_top"].dtype == np.uint8
    assert obs["images"]["oak_top_depth"].shape == (64, 64)
    assert obs["images"]["oak_top_depth"].dtype == np.float32
    assert obs["images"]["wrist"].shape == (64, 64, 3)
    assert obs["state"].shape == (6,)
    assert obs["state"].dtype == np.float32
    assert obs["task"] == "insert the orange tube into the slotted block"
    rollout.close()


def test_so101_box_step_runs_and_reports_failure(env_cfg) -> None:
    """5 zero-action steps run without raising; success stays False."""
    from openral_sim import SCENES

    rollout = SCENES.get("so101_box")(env_cfg)
    rollout.reset(seed=0)
    action = np.zeros(6, dtype=np.float32)
    for _ in range(5):
        res = rollout.step(action)
    assert res.info[env_cfg.task.success_key] is False
    assert res.terminated is False
    assert isinstance(res.reward, float)
    rollout.close()


def test_so101_box_scene_has_fill_light() -> None:
    """The composer injects a fill headlight so the wrist view isn't black."""
    from openral_sim.backends.so101_box._assets import compose_so101_box_mjcf

    xml, _ = compose_so101_box_mjcf()
    assert "<headlight" in xml, "expected an injected fill headlight in the scene"


def test_so101_box_wrist_camera_sees_lit_scene(env_cfg) -> None:
    """The wrist camera frames a lit scene, not a dark box wall.

    Regression: the wrist camera used to aim along gripper-local -Y (sideways
    into a wall) with no ambient fill, so MolmoAct2 received a near-black,
    near-uniform frame (mean luminance ~45). Re-aimed along the approach axis
    (-X) + fill light, the gripper + workspace are in frame and lit.
    """
    from openral_sim import SCENES

    rollout = SCENES.get("so101_box")(env_cfg)
    obs = rollout.reset(seed=0)
    wrist = np.asarray(obs["images"]["wrist"], dtype=np.float64).mean(axis=2)
    rollout.close()
    assert wrist.mean() > 100.0, (
        f"wrist frame too dark (mean={wrist.mean():.1f}) — unlit / wall-aimed"
    )


def test_so101_box_wrist_camera_matches_tuned_gripper_pose(env_cfg) -> None:
    """The wrist camera matches the static-finger-mounted, inverted phone pose.

    The pose replicates the real lerobot SO-101 wrist rig (Cornito/so101_test2):
    mounted on the static finger face, rolled 180° (``up = (0, 0, -1)``) and
    pitched so the open jaws hang into the bottom ~20% of the frame.
    """
    from openral_sim import SCENES

    rollout = SCENES.get("so101_box")(env_cfg)
    rollout.reset(seed=0)
    model, data = rollout.mujoco_handles()
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist")
    grip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "gripper")
    cam_pos = np.asarray(data.cam_xpos[cam_id], dtype=np.float64)
    cam_xmat = np.asarray(data.cam_xmat[cam_id], dtype=np.float64).reshape(3, 3)
    grip_pos = np.asarray(data.xpos[grip_id], dtype=np.float64)
    grip_xmat = np.asarray(data.xmat[grip_id], dtype=np.float64).reshape(3, 3)
    rollout.close()
    cam_pos_local = grip_xmat.T @ (cam_pos - grip_pos)
    cam_forward_local = grip_xmat.T @ (-cam_xmat[:, 2])
    cam_up_local = grip_xmat.T @ cam_xmat[:, 1]
    np.testing.assert_allclose(cam_pos_local, np.array([-0.0084, 0.0834, -0.0545]), atol=2e-3)
    # forward points down-and-forward toward the grasp; up is rolled (inverted phone).
    expected_forward = np.array([0.0045, -0.7926, -0.6098])
    expected_up = np.array([-0.0035, 0.6098, -0.7926])
    assert float(np.dot(cam_forward_local, expected_forward)) > 0.999
    assert float(np.dot(cam_up_local, expected_up)) > 0.999


def test_so101_box_nonzero_action_moves_arm(env_cfg) -> None:
    """A non-zero joint-position target actually moves the arm.

    Regression: the upstream MJCF's <position> actuators declare no
    ctrlrange, so model.actuator_ctrlrange is the [0, 0] sentinel. The
    env used to clip every action to it (np.clip(a, 0, 0)) which pinned
    all joints to zero — the arm never moved regardless of the policy.
    The clip now uses the joint position limits.
    """
    from openral_sim import SCENES

    rollout = SCENES.get("so101_box")(env_cfg)
    q0 = np.asarray(rollout.reset(seed=0)["state"], dtype=np.float64)
    target = np.array([0.4, 0.6, 0.4, 0.3, -0.3, 0.4], dtype=np.float32)  # rad, within limits
    res = None
    for _ in range(80):
        res = rollout.step(target)
    q1 = np.asarray(res.observation["state"], dtype=np.float64)
    assert np.max(np.abs(q1 - q0)) > 0.1, "arm did not move toward the commanded target"
    rollout.close()


def _env_with_backend_options(env_cfg, **opts):
    """Clone ``env_cfg`` with ``scene.backend_options`` overrides."""
    return env_cfg.model_copy(
        update={"scene": env_cfg.scene.model_copy(update={"backend_options": dict(opts)})},
    )


def test_so101_box_degrees_units_roundtrip(env_cfg) -> None:
    """``joint_units: degrees`` reports state in degrees and accepts degree actions."""
    from openral_sim import SCENES

    rollout = SCENES.get("so101_box")(_env_with_backend_options(env_cfg, joint_units="degrees"))
    obs = rollout.reset(seed=0)
    # Reported state is degrees == degrees(raw radian qpos).
    raw_rad = np.array(
        [float(rollout._data.qpos[a]) for a in rollout._arm_qpos_addrs],
        dtype=np.float64,
    )
    np.testing.assert_allclose(obs["state"], np.degrees(raw_rad), rtol=0, atol=1e-3)
    # A degree target drives the radian qpos toward radians(target).
    target_deg = np.array([20.0, 40.0, 30.0, 20.0, -20.0, 30.0], dtype=np.float32)
    for _ in range(80):
        rollout.step(target_deg)
    raw_rad_after = np.array(
        [float(rollout._data.qpos[a]) for a in rollout._arm_qpos_addrs],
        dtype=np.float64,
    )
    # shoulder_pan (joint 1, limit ±110°) tracks well — assert it moved
    # toward the degree target once converted to radians.
    assert abs(raw_rad_after[0] - math.radians(20.0)) < abs(raw_rad[0] - math.radians(20.0))
    rollout.close()


def test_so101_box_invalid_joint_units_rejected(env_cfg) -> None:
    """An unknown joint_units value fails loudly at build time."""
    from openral_core.exceptions import ROSConfigError
    from openral_sim import SCENES

    bad = _env_with_backend_options(env_cfg, joint_units="gradians")
    with pytest.raises(ROSConfigError, match="joint_units"):
        SCENES.get("so101_box")(bad)


def test_so101_box_calibration_affine_offsets_state(env_cfg) -> None:
    """joint_offsets_deg shifts the reported state by the per-joint offset."""
    from openral_sim import SCENES

    offsets = [3.0, 123.0, 124.0, 58.0, -11.0, 9.0]
    rollout = SCENES.get("so101_box")(
        _env_with_backend_options(env_cfg, joint_units="degrees", joint_offsets_deg=offsets)
    )
    obs = rollout.reset(seed=0)
    raw_deg = np.degrees([float(rollout._data.qpos[a]) for a in rollout._arm_qpos_addrs])
    np.testing.assert_allclose(obs["state"], raw_deg + np.array(offsets), rtol=0, atol=1e-3)
    rollout.close()


def test_so101_box_calibration_affine_action_roundtrip(env_cfg) -> None:
    """An action equal to the reported state holds the pose (affine inverts cleanly).

    state→model uses ``signs*deg+offset``; action→sim must invert it, so
    commanding the current reported state as the target should drive the joints
    back to (approximately) their current qpos rather than somewhere else.
    """
    from openral_sim import SCENES

    rollout = SCENES.get("so101_box")(
        _env_with_backend_options(
            env_cfg,
            joint_units="degrees",
            joint_offsets_deg=[3.0, 123.0, 124.0, 58.0, -11.0, 9.0],
            joint_signs=[1, 1, 1, 1, 1, -1],
        )
    )
    obs = rollout.reset(seed=0)
    q0 = np.array([float(rollout._data.qpos[a]) for a in rollout._arm_qpos_addrs])
    hold = np.asarray(obs["state"], dtype=np.float32)  # command current state as target
    for _ in range(40):
        rollout.step(hold)
    q1 = np.array([float(rollout._data.qpos[a]) for a in rollout._arm_qpos_addrs])
    rollout.close()
    # Holding the reported state must not run the joints away from their pose.
    assert np.max(np.abs(q1 - q0)) < 0.05, (
        f"affine did not invert (drift={np.max(np.abs(q1 - q0)):.3f})"
    )


def test_so101_box_invalid_joint_signs_rejected(env_cfg) -> None:
    """joint_signs entries other than ±1 fail loudly."""
    from openral_core.exceptions import ROSConfigError
    from openral_sim import SCENES

    bad = _env_with_backend_options(env_cfg, joint_signs=[1, 1, 1, 1, 1, 2])
    with pytest.raises(ROSConfigError, match="joint_signs"):
        SCENES.get("so101_box")(bad)


def test_so101_box_random_spawn_varies_with_seed(env_cfg) -> None:
    """Two different seeds produce two different spawn poses."""
    from openral_sim import SCENES

    seeds_to_poses: dict[int, tuple[tuple[float, float], tuple[float, float]]] = {}
    for s in (0, 1, 2):
        rollout = SCENES.get("so101_box")(env_cfg)
        rollout.reset(seed=s)
        model, data = rollout.mujoco_handles()
        block_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "slot_block")
        tube_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "tube")
        seeds_to_poses[s] = (
            (float(data.xpos[block_id][0]), float(data.xpos[block_id][1])),
            (float(data.xpos[tube_id][0]), float(data.xpos[tube_id][1])),
        )
        rollout.close()

    assert len({frozenset(v) for v in seeds_to_poses.values()}) == 3


@pytest.mark.parametrize(
    "case,expected",
    [
        ("vertical_centred", True),
        ("vertical_offcentre", False),
        ("vertical_above_block", False),
        ("horizontal", False),
    ],
)
def test_so101_box_insertion_geometry(env_cfg, case: str, expected: bool) -> None:
    """Geometric success check fires only when the tube is centred, vertical, and deep."""
    from openral_sim import SCENES

    rollout = SCENES.get("so101_box")(env_cfg)
    rollout.reset(seed=0)
    model, data = rollout.mujoco_handles()
    block_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "slot_block")
    block_xy = data.xpos[block_id][:2].copy()
    tube_qpos = rollout._tube_qpos_addr

    if case == "vertical_centred":
        data.qpos[tube_qpos + 0] = block_xy[0]
        data.qpos[tube_qpos + 1] = block_xy[1]
        data.qpos[tube_qpos + 2] = 0.05
        data.qpos[tube_qpos + 3 : tube_qpos + 7] = [1.0, 0.0, 0.0, 0.0]
    elif case == "vertical_offcentre":
        data.qpos[tube_qpos + 0] = block_xy[0] + 0.05
        data.qpos[tube_qpos + 1] = block_xy[1]
        data.qpos[tube_qpos + 2] = 0.05
        data.qpos[tube_qpos + 3 : tube_qpos + 7] = [1.0, 0.0, 0.0, 0.0]
    elif case == "vertical_above_block":
        data.qpos[tube_qpos + 0] = block_xy[0]
        data.qpos[tube_qpos + 1] = block_xy[1]
        data.qpos[tube_qpos + 2] = 0.10
        data.qpos[tube_qpos + 3 : tube_qpos + 7] = [1.0, 0.0, 0.0, 0.0]
    elif case == "horizontal":
        data.qpos[tube_qpos + 0] = block_xy[0]
        data.qpos[tube_qpos + 1] = block_xy[1]
        data.qpos[tube_qpos + 2] = 0.05
        c = math.cos(math.pi / 4.0)
        s = math.sin(math.pi / 4.0)
        data.qpos[tube_qpos + 3 : tube_qpos + 7] = [c, s, 0.0, 0.0]
    mujoco.mj_forward(model, data)
    assert rollout._check_insertion() is expected
    rollout.close()
