"""Sim tests for the ``so101_eraser`` eraser-place scene.

Exercises the :class:`SimRollout` Protocol end-to-end against a real composed
MJCF: registry membership, model compile, factory build, ``reset()`` layout
(the FK-measured 8.4 cm eraser→tape spacing this scene exists to reproduce),
spawn jitter, the lighting presets, both camera renders, ``step()`` with the
home action, and the geometric success check at two synthetic eraser poses.

No mocks (CLAUDE.md §1.11) — the scene is the production composer output and
the shipped ``scenes/sim/so101_eraser_place.yaml`` supplies the config.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import yaml

os.environ.setdefault("MUJOCO_GL", "egl")

try:
    import mujoco
except Exception as exc:  # mujoco's eager renderer probe can raise non-ImportError
    _MUJOCO_ERROR: str | None = str(exc)
else:
    _MUJOCO_ERROR = None

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

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONFIG = _REPO_ROOT / "scenes" / "sim" / "so101_eraser_place.yaml"


def _env_cfg(**backend_overrides: object):
    """Build a :class:`SimEnvironment` from the SHIPPED scene YAML."""
    from openral_core.schemas import SceneSpec, SimEnvironment, TaskSpec, VLASpec

    raw = yaml.safe_load(_CONFIG.read_text())
    raw["scene"].setdefault("backend_options", {}).update(backend_overrides)
    scene = SceneSpec(**raw["scene"]).model_copy(
        update={"observation_height": 64, "observation_width": 64}
    )
    return SimEnvironment(
        robot_id="so101_follower",
        scene=scene,
        task=TaskSpec(**raw["task"]).model_copy(update={"max_steps": 20}),
        vla=VLASpec(id="mock-noop", weights_uri="mock-noop", device="cpu"),
        seed=int(raw.get("seed") or 0),
    )


@pytest.fixture
def env_cfg():
    """The shipped scene, with spawn randomisation pinned off for exact asserts."""
    return _env_cfg(spawn_jitter_m=0.0, spawn_yaw_jitter_deg=0.0)


def test_so101_eraser_registered() -> None:
    """The scene is registered and bound to the SO-101 follower."""
    from openral_sim import ROBOTS, SCENES

    assert "so101_eraser" in SCENES
    assert "so101_follower" in ROBOTS
    assert SCENES.fixed_robot("so101_eraser") == "so101_follower"


def test_mjcf_compiles_with_both_cameras() -> None:
    """The composed MJCF carries the eraser, the tape, and both named cameras."""
    from openral_sim.backends.so101_eraser import compose_so101_eraser_mjcf

    xml, path = compose_so101_eraser_mjcf()
    assert "so101_eraser_generated.xml" in str(path)

    model = mujoco.MjModel.from_xml_path(str(path))
    for name in ("front", "wrist"):
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name) >= 0, name
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "eraser") >= 0
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "tape") >= 0
    # The injected headlight is what gives the preset control of exposure;
    # the upstream <default class="visual"> must not block the injection.
    assert "<headlight" in xml


def test_jaws_are_recoloured_to_match_the_real_rig() -> None:
    """The jaw materials are repainted black, and ``null`` opts out.

    The jaws fill the bottom third of every wrist frame — the view the policy
    grasps from — so an upstream material rename must fail loud rather than
    silently leaving them off-white.
    """
    from openral_core.exceptions import ROSConfigError
    from openral_sim.backends.so101_eraser import (
        EraserSceneOptions,
        _recolour_jaws,
        compose_so101_eraser_mjcf,
    )

    xml, _ = compose_so101_eraser_mjcf()
    for name in ("moving_jaw_so101_v1_material", "wrist_roll_follower_so101_v1_material"):
        line = next(ln for ln in xml.splitlines() if f'name="{name}"' in ln)
        assert 'rgba="0.11 0.11 0.12 1.0"' in line, line

    untouched, _ = compose_so101_eraser_mjcf(EraserSceneOptions(jaw_rgba=None))
    assert 'name="moving_jaw_so101_v1_material" rgba="0.964706' in untouched

    with pytest.raises(ROSConfigError, match="cannot be recoloured"):
        _recolour_jaws("<mujoco/>", (0.0, 0.0, 0.0, 1.0))


def test_eraser_wears_the_wrapper_texture() -> None:
    """One box collider, product dimensions, the label wrapped as a cube texture.

    The texture is appearance only: the single geom keeps its contacts and the
    body its explicit inertial, so physics is exactly the plain block's.
    """
    from openral_sim.backends.so101_eraser import EraserSceneOptions, compose_so101_eraser_mjcf

    _, path = compose_so101_eraser_mjcf()
    model = mujoco.MjModel.from_xml_path(str(path))
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "eraser")
    block = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "eraser_geom")
    assert model.geom_bodyid[block] == body
    assert model.geom_contype[block] != 0

    # The retail product's 40 x 20 x 10 mm block (half-extents).
    assert tuple(model.geom_size[block]) == pytest.approx(
        EraserSceneOptions().eraser_size, abs=1e-6
    )
    assert EraserSceneOptions().eraser_size == (0.020, 0.010, 0.005)

    # The wrapper: a cube texture bound through the eraser material, and no
    # leftover second visual geom from the old two-tone modelling.
    mat = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, "eraser_mat")
    assert mat >= 0 and model.geom_matid[block] == mat
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TEXTURE, "eraser_tex") >= 0
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "eraser_sleeve") < 0


def test_reset_lays_out_the_bench(env_cfg) -> None:
    """Reset reproduces the layout: eraser in front of the arm, tape 8.4 cm right.

    "Right" is the arm's right, which is world -X for an arm facing -Y — the
    same side the overview camera renders on the right of the frame.
    """
    from openral_sim import SCENES

    rollout = SCENES.get("so101_eraser")(env_cfg)
    try:
        rollout.reset(seed=env_cfg.seed)
        model, data = rollout.mujoco_handles()
        eraser = np.asarray(data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "eraser")])
        tape = np.asarray(data.site_xpos[rollout._tape_site_id])

        opts = rollout.options
        assert eraser[1] == pytest.approx(-opts.eraser_offset_y, abs=2e-3)
        assert tape[0] - eraser[0] == pytest.approx(-opts.tape_offset_x, abs=2e-3)
        assert tape[1] - eraser[1] == pytest.approx(opts.tape_offset_y, abs=2e-3)
        # Resting on the desk, not floating or sunk.
        assert eraser[2] == pytest.approx(opts.eraser_size[2], abs=2e-3)
    finally:
        rollout.close()


def test_shipped_layout_matches_the_measured_teleop_distribution() -> None:
    """The shipped defaults are the FK-derived bench layout, not round numbers.

    Guards the numbers the scene exists to reproduce: forward kinematics over
    the 25 teleop episodes of ``makermods/eraser_place_unblurry_real`` puts the
    grasp at (+0.024, -0.258) and the release 8.4 cm to the arm's right, and
    the per-episode grasp spread is ~1.3 cm. A silent edit back to a tidy
    "5 cm" would put the sim off-distribution for the checkpoint.
    """
    from openral_sim.backends.so101_eraser import _options_from_backend_options

    raw = yaml.safe_load(_CONFIG.read_text())
    opts = _options_from_backend_options(raw["scene"]["backend_options"])
    assert opts.eraser_xy == pytest.approx((0.024, -0.258), abs=1e-4)
    assert opts.tape_xy == pytest.approx((-0.060, -0.258), abs=1e-4)
    assert opts.spawn_jitter_m == pytest.approx(0.013, abs=1e-4)


def test_spawn_jitter_moves_the_eraser_and_reseeding_repeats_it() -> None:
    """Jitter is on by default, bounded by its setting, and seed-reproducible."""
    from openral_sim import SCENES

    cfg = _env_cfg()
    rollout = SCENES.get("so101_eraser")(cfg)
    try:
        seen = []
        for seed in (1, 2, 3, 1):
            rollout.reset(seed=seed)
            _, data = rollout.mujoco_handles()
            seen.append(np.asarray(data.xpos[rollout._eraser_body_id])[:2].copy())
        nominal = np.asarray(rollout.options.eraser_xy)
        bound = rollout.options.spawn_jitter_m + 5e-3  # + settle drift
        assert all(np.all(np.abs(p - nominal) <= bound) for p in seen)
        assert not np.allclose(seen[0], seen[1], atol=1e-4)
        assert np.allclose(seen[0], seen[3], atol=1e-6)  # same seed → same layout
    finally:
        rollout.close()


def test_lighting_presets_change_exposure_and_reject_typos() -> None:
    """The lighting knob is real: ``dim`` renders darker than ``bright``."""
    from openral_core.exceptions import ROSConfigError
    from openral_sim import SCENES
    from openral_sim.backends.so101_eraser import _options_from_backend_options

    means = {}
    for preset in ("dim", "bench", "bright"):
        rollout = SCENES.get("so101_eraser")(_env_cfg(lighting=preset))
        try:
            means[preset] = float(rollout.reset(seed=7)["images"]["front"].mean())
        finally:
            rollout.close()
    assert means["dim"] < means["bench"] < means["bright"], means

    with pytest.raises(ROSConfigError, match="unknown lighting preset"):
        _options_from_backend_options({"lighting": "disco"})


def test_observation_carries_both_views_and_the_task_string(env_cfg) -> None:
    """Both cameras render and the instruction is the verbatim training string."""
    from openral_sim import SCENES

    rollout = SCENES.get("so101_eraser")(env_cfg)
    try:
        obs = rollout.reset(seed=env_cfg.seed)
        assert set(obs["images"]) == {"front", "wrist"}
        for img in obs["images"].values():
            assert img.shape == (64, 64, 3)
            assert img.dtype == np.uint8
        assert obs["state"].shape == (6,)
        assert obs["task"] == "place the erase on the blue square"
        # Degrees mode: the home pose is the rSkill's starting_pose, whose
        # shoulder_lift is ~-100 deg. Radians would put it at ~-1.75.
        assert obs["state"][1] < -50.0
    finally:
        rollout.close()


def test_step_holds_the_home_pose(env_cfg) -> None:
    """Commanding the home pose keeps the arm there and reports no success."""
    from openral_sim import SCENES

    rollout = SCENES.get("so101_eraser")(env_cfg)
    try:
        obs = rollout.reset(seed=env_cfg.seed)
        home = np.asarray(obs["state"], dtype=np.float32)
        for _ in range(5):
            result = rollout.step(home)
        assert not result.terminated
        assert result.info["is_success"] is False
        assert np.allclose(result.observation["state"][:5], home[:5], atol=2.0)
    finally:
        rollout.close()


def test_gripper_channel_is_lerobot_normalised_not_degrees(env_cfg) -> None:
    """The gripper obs/action channel is lerobot [0, 100] jaw travel, not degrees.

    The manifest home 0.0195 must read back as ~1.95. The old degrees mapping
    reported it as degrees(qpos) = -7.9 (the MJCF jaw range starts at -10°) and
    landed commanded closes ~10° open — the grasp-critical end of the range.
    """
    from openral_sim import SCENES

    rollout = SCENES.get("so101_eraser")(env_cfg)
    try:
        obs = rollout.reset(seed=env_cfg.seed)
        state = np.asarray(obs["state"], dtype=np.float32)
        assert state[5] == pytest.approx(1.95, abs=0.5)
        cmd = state.copy()
        cmd[5] = 100.0  # fully open on the lerobot scale
        for _ in range(200):
            result = rollout.step(cmd)
        assert result.observation["state"][5] > 80.0
    finally:
        rollout.close()


def test_success_check_fires_only_on_the_tape(env_cfg) -> None:
    """The geometric place check: eraser on the tape passes, beside it fails."""
    from openral_sim import SCENES

    rollout = SCENES.get("so101_eraser")(env_cfg)
    try:
        rollout.reset(seed=env_cfg.seed)
        _, data = rollout.mujoco_handles()
        tape = np.asarray(data.site_xpos[rollout._tape_site_id])

        rollout._write_freejoint(
            qpos_addr=rollout._eraser_qpos_addr,
            xyz=(float(tape[0]), float(tape[1]), rollout.options.eraser_size[2]),
            yaw=0.0,
        )
        mujoco.mj_forward(rollout._model, data)
        assert rollout._check_placed() is True

        off = rollout.options.place_xy_tol_m * 3.0
        rollout._write_freejoint(
            qpos_addr=rollout._eraser_qpos_addr,
            xyz=(float(tape[0]) + off, float(tape[1]), rollout.options.eraser_size[2]),
            yaw=0.0,
        )
        mujoco.mj_forward(rollout._model, data)
        assert rollout._check_placed() is False
    finally:
        rollout.close()


def test_unknown_backend_option_is_rejected(env_cfg) -> None:
    """A YAML typo in ``scene.backend_options`` fails loud, not silently."""
    from openral_core.exceptions import ROSConfigError
    from openral_sim.backends.so101_eraser import _options_from_backend_options

    with pytest.raises(ROSConfigError, match=r"unknown scene\.backend_options"):
        _options_from_backend_options({"eraser_offset_z": 0.1})


def test_eraser_spawns_in_place_without_a_rollout_reset() -> None:
    """The MJCF alone puts the eraser on the desk — deploy has no ``reset``.

    Regression from a live ``openral deploy sim`` run: the eraser body carried
    the MJCF default pose and only ``reset()`` wrote its spawn. Deploy sim
    hands this MJCF to a bare ``MujocoArmHAL`` and never builds a rollout, so
    the freejoint spawned at the world origin — inside the robot base — and
    physics ejected the eraser before the first frame. The operator saw a tape
    square and nothing to pick up.
    """
    from openral_sim.backends.so101_eraser import (
        EraserSceneOptions,
        compose_so101_eraser_deploy_mjcf,
    )

    xml, meshdir = compose_so101_eraser_deploy_mjcf()
    path = meshdir.parent / "so101_eraser_spawn_test.xml"
    path.write_text(xml)

    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "eraser")
    opts = EraserSceneOptions()

    assert data.xpos[body][:2] == pytest.approx(opts.eraser_xy, abs=1e-4)
    # And it STAYS there: a body spawned inside the arm gets ejected.
    for _ in range(500):
        mujoco.mj_step(model, data)
    assert data.xpos[body][:2] == pytest.approx(opts.eraser_xy, abs=5e-3)
    assert data.xpos[body][2] > 0.0
