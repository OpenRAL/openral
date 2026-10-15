"""A sim scene's own control rate and simulated sensors, honoured end to end in deploy sim.

LeIsaac records one action per 1/60 s physics step, so its SmolVLA PickOrange skill
declares ``action_contract.control_freq_hz: 60`` while the SO-101 manifest ticks at
30 Hz. ``scenes/deploy/isaac_leisaac_pick_orange.yaml`` sets
``backend_options.control_freq_hz: 60`` (the one source) and adds a simulated
``top_depth`` camera for the world map (``sim_sensors``);
``openral_core.sim_scene_description`` applies both, and every deploy-sim consumer
calls it. Real fixtures throughout: the shipped scene, the SO-101 manifest, the shipped
rSkill manifest; the HAL leg runs the real factory against the no-Kit sidecar double.
"""

from __future__ import annotations

import copy
import math
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml
from openral_core import (
    DeployScene,
    RobotDescription,
    RSkillManifest,
    scene_control_rate_hz,
    sim_scene_description,
)
from openral_core.exceptions import ROSConfigError

_REPO = Path(__file__).resolve().parents[2]
_SCENE = _REPO / "scenes/deploy/isaac_leisaac_pick_orange.yaml"
_SKILL = _REPO / "rskills/smolvla-so101-leisaac-pick-orange/rskill.yaml"
_FAKE_SIDECAR = _REPO / "tests/unit/fakes/isaac_sidecar_no_kit.py"


def _robot(name: str = "so101_follower") -> RobotDescription:
    return RobotDescription.from_yaml(str(_REPO / "robots" / name / "robot.yaml"))


def _options() -> dict[str, Any]:
    return copy.deepcopy(DeployScene.from_yaml(str(_SCENE)).scene.backend_options)


# ── the rule ──────────────────────────────────────────────────────────────────


def test_scene_rate_wins_else_the_robot_rate() -> None:
    assert scene_control_rate_hz({"control_freq_hz": 60}, 30.0) == 60.0
    assert scene_control_rate_hz({}, 30.0) == 30.0
    assert scene_control_rate_hz(None, None) is None


@pytest.mark.parametrize("bad", [0, -30.0, math.nan, math.inf, "60", True])
def test_a_bad_scene_rate_is_refused(bad: object) -> None:
    with pytest.raises(ROSConfigError, match="control_freq_hz"):
        scene_control_rate_hz({"control_freq_hz": bad}, 30.0)


def test_the_leisaac_deploy_scene_runs_the_so101_at_60hz_with_a_sim_depth_camera() -> None:
    manifest = _robot()
    scene = sim_scene_description(manifest, _options())
    assert manifest.control_rate_hz == 30.0  # the manifest itself is untouched
    assert scene.control_rate_hz == 60.0
    assert [s.name for s in scene.sensors] == ["top", "wrist", "top_depth"]
    depth = scene.sensors[-1]
    assert depth.is_depth_camera and depth.deploy_binding is None
    # Everything else about the robot is the manifest's.
    assert scene.joints == manifest.joints
    assert scene.safety == manifest.safety
    assert scene.collision_geometry == manifest.collision_geometry


def test_a_scene_without_the_keys_returns_the_manifest_itself() -> None:
    manifest = _robot()
    assert sim_scene_description(manifest, {"boot_timeout_s": 1200}) is manifest


def test_a_rate_override_needs_an_action_spec() -> None:
    widowx = _robot("widowx")  # declares no action_spec
    with pytest.raises(ROSConfigError, match="declares no action_spec"):
        sim_scene_description(widowx, {"control_freq_hz": 60.0})


@pytest.mark.parametrize(
    ("patch", "match"),
    [
        ({"modality": "rgb"}, "must be a depth camera"),
        ({"intrinsics": None}, "must be a depth camera"),
        (
            {"deploy_binding": {"backend": "opencv_thread", "backend_params": {"device": 0}}},
            "no deploy_binding",
        ),
        ({"name": "top"}, "defined by the robot manifest"),
        ({"rate_hz": "fast"}, "sim_sensors is invalid"),
    ],
)
def test_a_simulated_sensor_is_a_depth_camera_of_its_own(
    patch: dict[str, object], match: str
) -> None:
    options = _options()
    options["sim_sensors"][0].update(patch)
    with pytest.raises(ROSConfigError, match=match):
        sim_scene_description(_robot(), options)


# ── the deploy scene ──────────────────────────────────────────────────────────


def test_the_deploy_scene_matches_the_sim_scene_it_twins() -> None:
    """Same stage, objects, cameras, gains, spawn and rubric as the sim scene."""
    from openral_sim.backends.isaac_sim import IsaacSimOptions

    sim = yaml.safe_load((_REPO / "scenes/sim/isaac_leisaac_pick_orange.yaml").read_text())
    deploy = DeployScene.from_yaml(str(_SCENE))
    opts = IsaacSimOptions.model_validate(deploy.scene.backend_options)
    sim_opts = IsaacSimOptions.model_validate(sim["scene"]["backend_options"])
    assert opts.model_dump(exclude={"sim_sensors", "camera_mounts"}) == sim_opts.model_dump(
        exclude={"sim_sensors", "camera_mounts"}
    )
    assert {k: opts.camera_mounts[k] for k in ("top", "wrist")} == sim_opts.camera_mounts
    # The simulated depth view is LeIsaac's front camera.
    assert opts.camera_mounts["top_depth"] == opts.camera_mounts["top"]
    assert deploy.robot_id == sim["robot_id"]
    assert deploy.base_pose is not None
    assert list(deploy.base_pose.xyz) == sim["base_pose"]["xyz"]
    assert deploy.runtime is not None and deploy.runtime.enable_octomap is True
    assert deploy.runtime.enable_octomap_kernel_check is True


# ── the runner's trained-rate check ───────────────────────────────────────────


def test_the_runner_runs_the_60hz_skill_at_the_scene_rate_and_refuses_it_at_30hz() -> None:
    import importlib

    runner = importlib.import_module("openral_rskill_ros.rskill_runner_node")
    contract = RSkillManifest.from_yaml(str(_SKILL)).action_contract
    assert contract is not None
    trained = contract.control_freq_hz
    assert trained == 60.0
    manifest = _robot()
    scene_hz = sim_scene_description(manifest, _options()).control_rate_hz
    assert scene_hz is not None
    # The launch forwards the scene rate as `rate_hz`; the runner ticks at it.
    tick, source = runner.resolve_control_rate_source(scene_hz, manifest)
    assert (tick, source) == (60.0, "rate_hz_param")
    runner.check_skill_control_rate(trained, tick, skill_id="leisaac", robot=manifest.name)
    # Without the scene override the same skill on the 30 Hz manifest is refused.
    tick, source = runner.resolve_control_rate_source(0.0, manifest)
    assert (tick, source) == (30.0, "manifest")
    with pytest.raises(ROSConfigError, match="trained at 60 Hz"):
        runner.check_skill_control_rate(trained, tick, skill_id="leisaac", robot=manifest.name)


# ── the CLI ───────────────────────────────────────────────────────────────────


def _resolve(config: Path, **kw: Any) -> Any:
    from openral_cli.deploy_sim import resolve_launch_invocation

    return resolve_launch_invocation(
        config=config,
        robot_override=None,
        dashboard_port=4318,
        reset_to_pose_service=None,
        hal_param_overrides=None,
        enable_object_detector=False,
        **kw,
    )


def test_deploy_sim_maps_the_simulated_depth_and_attaches_the_scene() -> None:
    inv = _resolve(_SCENE, enable_octomap=True, enable_octomap_kernel_check=True)
    assert inv.enable_octomap is True
    argv = " ".join(inv.argv_template)
    assert "enable_octomap_kernel_check:=true" in argv
    assert f"deploy_config:={_SCENE.resolve()}" in argv
    assert inv.hal_params["sim_env_yaml"] == str(_SCENE.resolve())


def test_a_bare_twin_scene_cannot_retime_the_robot(tmp_path: Path) -> None:
    """The bare MuJoCo twin reads the manifest's rate: refuse rather than ignore it."""
    raw = yaml.safe_load((_REPO / "scenes/deploy/so101_box.yaml").read_text())
    raw["scene"]["backend_options"] = {"control_freq_hz": 60.0}
    scene = tmp_path / "so101_box_60hz.yaml"
    scene.write_text(yaml.safe_dump(raw))
    with pytest.raises(ROSConfigError, match="scene-attached sim"):
        _resolve(scene)


# ── the HAL: build_hal holds the Isaac scene to the scene's rate ──────────────


@pytest.fixture
def no_kit_sidecar(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("zmq", reason="isaacsim group (pyzmq) not installed")
    pytest.importorskip("msgpack", reason="isaacsim group (msgpack) not installed")
    monkeypatch.setenv("OPENRAL_ISAAC_SIDECAR_PYTHON", sys.executable)
    monkeypatch.setenv("OPENRAL_ISAAC_SIDECAR_SCRIPT", str(_FAKE_SIDECAR))
    monkeypatch.setenv("OPENRAL_TEST_ISAAC_ARGV_OUT", str(tmp_path / "argv.json"))
    # A completed extraction of the pinned archive, so nothing is downloaded (the
    # sidecar double never opens the stage).
    from openral_sim.backends.isaac_sim import IsaacEnvironmentArchive

    monkeypatch.setenv("OPENRAL_CACHE_DIR", str(tmp_path))
    archive = IsaacEnvironmentArchive.model_validate(_options()["environment_archive"])
    root = tmp_path / "isaac-environments" / archive.sha256[:16]
    (root / archive.usd).parent.mkdir(parents=True)
    (root / archive.usd).write_text("#usda 1.0\n")
    (root / ".extracted").write_text(archive.sha256)


@pytest.mark.usefixtures("no_kit_sidecar")
def test_build_hal_steps_the_scene_at_60hz_and_the_manifest_rate_is_refused() -> None:
    import openral_sim.backends  # noqa: F401 — registers the isaac_sim scene
    from openral_hal import build_hal
    from openral_hal.sim_attached import SimAttachedHAL

    hal = build_hal(_robot(), mode="sim", sim_env_yaml=str(_SCENE))
    assert isinstance(hal, SimAttachedHAL)
    try:
        assert hal.description.control_rate_hz == 60.0
        assert "top_depth" in {s.name for s in hal.description.sensors}
        hal.connect()  # the time-base check passes: one step is 1/60 s
    finally:
        hal.disconnect()
        hal._env.close()
    # The same scene behind the bare manifest (30 Hz): the guard still bites.
    from openral_hal.sim_bringup import build_sim_env_from_yaml

    env, seed = build_sim_env_from_yaml(str(_SCENE))
    try:
        with pytest.raises(ROSConfigError, match="ticks at 30 Hz"):
            SimAttachedHAL(env, _robot(), env_reset_seed=seed).connect()
    finally:
        env.close()
