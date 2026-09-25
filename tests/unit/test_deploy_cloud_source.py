"""Where octomap's cloud comes from, and when a deploy must refuse instead of mapping silence.

Audit findings (generalization of the OpenArm/Thor fixes):

* a real deploy whose manifest declares a depth camera auto-enabled octomap against
  ``/openral/cameras/<depth>/points``, which only the **sim** sensor bridge publishes, so
  the map stayed empty and the kernel dropped every chunk behind healthy nodes; only the
  OpenArm scenes knew to pin the ZED topic;
* the first depth camera silently won on a robot with several;
* a sim twin fed by a real driver's cloud had to remember ``clock_origin: host_wall``;
* the voxel deadline / octree-age bound were Thor-tuned module constants.

Every case runs against real manifests: panda_mobile (sim depth), a franka_panda real-HAL
manifest given a RealSense-style depth camera or a 3D lidar, and so101 (no depth).
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import yaml
from openral_cli.deploy_sim import LaunchInvocation, resolve_launch_invocation
from openral_core import DeployRuntime, RobotDescription, SensorSpec, deploy_cloud_topic
from openral_core.exceptions import ROSConfigError
from pydantic import ValidationError

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PANDA_MOBILE = _REPO_ROOT / "robots" / "panda_mobile" / "robot.yaml"
_REALSENSE_POINTS = "/camera/depth/color/points"
#: With the kernel's world-voxel check on, a real deploy also needs a verified extrinsic
#: for every depth camera (``_preflight_depth_extrinsics``, covered in
#: test_deploy_run_real_resolution.py); these tests are about the cloud wiring only.
_NO_KERNEL_CHECK = "  enable_octomap_kernel_check: false\n"


def _front_depth() -> dict[str, object]:
    """panda_mobile's committed front_depth entry, as the real fixture for a depth camera."""
    data = yaml.safe_load(_PANDA_MOBILE.read_text(encoding="utf-8"))
    (entry,) = [s for s in data["sensors"] if s["name"] == "front_depth"]
    return dict(entry)


def _real_franka_with(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *extra: dict[str, object]
) -> None:
    """franka_panda (a real-HAL manifest) with ``extra`` sensors, served via OPENRAL_ROBOTS_DIR."""
    dst = tmp_path / "robots" / "franka_panda"
    shutil.copytree(_REPO_ROOT / "robots" / "franka_panda", dst)
    manifest = dst / "robot.yaml"
    data = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    data["sensors"] = [*(data.get("sensors") or []), *extra]
    manifest.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    monkeypatch.setenv("OPENRAL_ROBOTS_DIR", str(tmp_path / "robots"))


def _real_scene(tmp_path: Path, runtime: str = "") -> Path:
    scene = tmp_path / "franka_cell.yaml"
    scene.write_text(
        "scene:\n  id: franka_cell\nrobot_id: franka_panda\n"
        + (f"runtime:\n{runtime}" if runtime else ""),
        encoding="utf-8",
    )
    return scene


def _resolve(config: Path | None, hal_mode: str, robot: str | None = None) -> LaunchInvocation:
    return resolve_launch_invocation(
        config=config,
        robot_override=robot,
        dashboard_port=4318,
        reset_to_pose_service=None,
        hal_param_overrides={"viewer_enabled": False} if hal_mode == "sim" else None,
        hal_mode=hal_mode,
    )


# ── openral_core.deploy_cloud_topic ─────────────────────────────────────────────


def test_sim_derives_the_one_depth_cameras_bridge_cloud() -> None:
    sensors = RobotDescription.from_yaml(str(_PANDA_MOBILE)).sensors
    assert deploy_cloud_topic(sensors, pinned=None, hal_mode="sim") == (
        "/openral/cameras/front_depth/points"
    )


def test_real_derives_nothing_and_a_pin_always_wins() -> None:
    """Nothing in-tree publishes a cloud on real hardware: only the driver's pin names one."""
    sensors = RobotDescription.from_yaml(str(_PANDA_MOBILE)).sensors
    assert deploy_cloud_topic(sensors, pinned=None, hal_mode="real") == ""
    for mode in ("sim", "real"):
        assert deploy_cloud_topic(sensors, pinned=_REALSENSE_POINTS, hal_mode=mode) == (
            _REALSENSE_POINTS
        )


def test_several_depth_cameras_need_a_declared_choice() -> None:
    """First-one-wins used to map one camera and ignore the rest; now the scene must pick."""
    front = SensorSpec.model_validate(_front_depth())
    rear = front.model_copy(update={"name": "rear_depth"})
    with pytest.raises(ROSConfigError, match=r"front_depth, rear_depth.*octomap_cloud_topic"):
        deploy_cloud_topic([front, rear], pinned=None, hal_mode="sim")
    pinned = "/openral/cameras/rear_depth/points"
    assert deploy_cloud_topic([front, rear], pinned=pinned, hal_mode="sim") == pinned


def test_a_3d_lidar_is_a_cloud_source_but_not_a_depth_camera() -> None:
    lidar = SensorSpec(name="top_lidar", modality="point_cloud", frame_id="lidar", rate_hz=10.0)
    assert lidar.is_cloud_source and not lidar.is_depth_camera
    front = SensorSpec.model_validate(_front_depth())
    assert front.is_cloud_source and front.is_depth_camera


# ── resolve_launch_invocation: refusal on real hardware ─────────────────────────


def test_real_depth_camera_without_a_pinned_cloud_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The silent-empty-map case: refused before launch, naming the fix."""
    _real_franka_with(tmp_path, monkeypatch, _front_depth())
    with pytest.raises(ROSConfigError, match=r"octomap_cloud_topic.*enable_octomap: false"):
        _resolve(None, "real", robot="franka_panda")


def test_real_3d_lidar_also_needs_its_cloud_pinned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lidar = {"name": "top_lidar", "modality": "point_cloud", "frame_id": "lidar", "rate_hz": 10.0}
    _real_franka_with(tmp_path, monkeypatch, lidar)
    with pytest.raises(ROSConfigError, match="octomap_cloud_topic"):
        _resolve(None, "real", robot="franka_panda")
    # With the kernel's world check on, a lidar is refused: nothing measures its
    # extrinsic yet, and the world check would place obstacles through that pose.
    checked = _real_scene(tmp_path, "  octomap_cloud_topic: /ouster/points\n")
    with pytest.raises(ROSConfigError, match="top_lidar: a robot point_cloud sensor"):
        _resolve(checked, "real")
    scene = _real_scene(tmp_path, f"  octomap_cloud_topic: /ouster/points\n{_NO_KERNEL_CHECK}")
    invocation = _resolve(scene, "real")
    assert invocation.enable_octomap is True
    assert "octomap_cloud_topic:=/ouster/points" in invocation.argv_template


def test_real_depth_camera_with_the_driver_topic_pinned_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _real_franka_with(tmp_path, monkeypatch, _front_depth())
    invocation = _resolve(
        _real_scene(tmp_path, f"  octomap_cloud_topic: {_REALSENSE_POINTS}\n{_NO_KERNEL_CHECK}"),
        "real",
    )
    assert invocation.enable_octomap is True
    assert invocation.clock_origin == "host_wall"
    assert f"octomap_cloud_topic:={_REALSENSE_POINTS}" in invocation.argv_template


def test_real_depth_camera_with_octomap_declared_off_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _real_franka_with(tmp_path, monkeypatch, _front_depth())
    invocation = _resolve(_real_scene(tmp_path, "  enable_octomap: false\n"), "real")
    assert invocation.enable_octomap is False


def test_real_robot_without_a_cloud_source_is_unchanged() -> None:
    """franka_panda as committed has no depth sensor: octomap stays off, nothing refused.

    With no scene, the launch still gets the schema's default voxel freshness pair.
    """
    invocation = _resolve(None, "real", robot="franka_panda")
    assert invocation.enable_octomap is False
    deadline, age = DeployRuntime().voxel_freshness_s
    assert f"world_voxel_deadline_s:={deadline}" in invocation.argv_template
    assert f"max_octree_age_s:={age}" in invocation.argv_template
    rig = DeployRuntime()
    budget = rig.world_voxel_data_age_budget_s
    assert f"world_voxel_data_age_budget_s:={budget}" in invocation.argv_template
    padding = rig.robot_self_filter_padding_m
    assert f"robot_self_filter_padding_m:={padding}" in invocation.argv_template


def test_explicit_octomap_on_a_depthless_sim_robot_refuses_before_launch(tmp_path: Path) -> None:
    text = (_REPO_ROOT / "scenes" / "deploy" / "so101_bench.yaml").read_text(encoding="utf-8")
    scene = tmp_path / "so101_octomap.yaml"
    scene.write_text(text.replace("enable_octomap: false", "enable_octomap: true"), "utf-8")
    with pytest.raises(ROSConfigError, match="so101"):
        _resolve(scene, "sim")


# ── clock origin for a twin fed by a real driver ─────────────────────────────────


def _panda_mobile_scene(tmp_path: Path, runtime: str) -> Path:
    pytest.importorskip("openral_sim")
    text = (_REPO_ROOT / "scenes" / "deploy" / "robocasa_navigate.yaml").read_text("utf-8")
    assert "\nruntime:" not in text, "fixture grew a runtime block; merge instead of appending"
    scene = tmp_path / "navigate_with_real_cloud.yaml"
    scene.write_text(f"{text.rstrip()}\nruntime:\n{runtime}", encoding="utf-8")
    return scene


def test_sim_twin_default_keeps_the_simulator_clock(tmp_path: Path) -> None:
    invocation = _resolve(_panda_mobile_scene(tmp_path, "  enable_reasoner: true\n"), "sim")
    assert invocation.enable_octomap is True
    assert invocation.clock_origin == "simulation"


def test_sim_twin_fed_by_a_real_driver_runs_on_wall_clock(tmp_path: Path) -> None:
    """No ``clock_origin`` pin needed: a cloud outside /openral/cameras is a real driver."""
    scene = _panda_mobile_scene(tmp_path, f"  octomap_cloud_topic: {_REALSENSE_POINTS}\n")
    invocation = _resolve(scene, "sim")
    assert invocation.clock_origin == "host_wall"
    assert "clock_origin:=host_wall" in invocation.argv_template


def test_sim_clock_pinned_with_a_real_driver_cloud_refuses(tmp_path: Path) -> None:
    scene = _panda_mobile_scene(
        tmp_path,
        f"  octomap_cloud_topic: {_REALSENSE_POINTS}\n  clock_origin: simulation\n",
    )
    with pytest.raises(ROSConfigError, match="from the future"):
        _resolve(scene, "sim")


def test_sim_cloud_pinned_inside_openral_cameras_keeps_the_simulator_clock(
    tmp_path: Path,
) -> None:
    scene = _panda_mobile_scene(
        tmp_path, "  octomap_cloud_topic: /openral/cameras/front_depth/points\n"
    )
    assert _resolve(scene, "sim").clock_origin == "simulation"


# ── voxel freshness: a declared, validated rig property ──────────────────────────


def test_voxel_freshness_defaults_and_invariant() -> None:
    """Age defaults to the deadline and may never exceed it, so the kernel fails closed."""
    deadline, age = DeployRuntime().voxel_freshness_s
    assert age <= deadline
    assert DeployRuntime(world_voxel_deadline_s=2.0).voxel_freshness_s == (2.0, 2.0)
    assert DeployRuntime(world_voxel_deadline_s=2.0, max_octree_age_s=1.5).voxel_freshness_s == (
        2.0,
        1.5,
    )
    with pytest.raises(ValidationError, match="must not exceed"):
        DeployRuntime(world_voxel_deadline_s=1.0, max_octree_age_s=1.5)
    with pytest.raises(ValidationError, match="must not exceed"):
        DeployRuntime(max_octree_age_s=deadline + 0.5)  # only the age set, still checked
    with pytest.raises(ValidationError):
        DeployRuntime(world_voxel_deadline_s=0.0)


def test_scene_voxel_freshness_reaches_the_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A slow cloud source's rig declares a longer deadline instead of editing the launch."""
    _real_franka_with(tmp_path, monkeypatch, _front_depth())
    scene = _real_scene(
        tmp_path,
        f"  octomap_cloud_topic: {_REALSENSE_POINTS}\n{_NO_KERNEL_CHECK}"
        "  world_voxel_deadline_s: 2.0\n  max_octree_age_s: 1.5\n",
    )
    argv = _resolve(scene, "real").argv_template
    assert "world_voxel_deadline_s:=2.0" in argv
    assert "max_octree_age_s:=1.5" in argv


def test_rig_perception_values_default_and_are_validated() -> None:
    """The data-age budget and self-filter padding are per-rig ``DeployRuntime`` values.

    Defaults are the Thor-measured budget (1.5 s) and the provisional padding
    (0.02 m, 2026-09-25). A budget must be positive (0 would disable the kernel's check);
    a padding cannot be negative. No relation to the voxel deadline is imposed:
    a smaller budget is only stricter.
    """
    rig = DeployRuntime()
    assert rig.world_voxel_data_age_budget_s == 1.5
    assert rig.robot_self_filter_padding_m == 0.02
    DeployRuntime(world_voxel_data_age_budget_s=0.3, world_voxel_deadline_s=2.0)
    for bad in ({"world_voxel_data_age_budget_s": 0.0}, {"robot_self_filter_padding_m": -0.01}):
        with pytest.raises(ValidationError):
            DeployRuntime.model_validate(bad)


def test_scene_rig_perception_values_reach_the_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A rig declares its measured budget and padding in the scene, not in the launch."""
    _real_franka_with(tmp_path, monkeypatch, _front_depth())
    scene = _real_scene(
        tmp_path,
        f"  octomap_cloud_topic: {_REALSENSE_POINTS}\n{_NO_KERNEL_CHECK}"
        "  world_voxel_data_age_budget_s: 0.8\n  robot_self_filter_padding_m: 0.03\n",
    )
    argv = _resolve(scene, "real").argv_template
    assert "world_voxel_data_age_budget_s:=0.8" in argv
    assert "robot_self_filter_padding_m:=0.03" in argv


# ── hard caps: 2x the pre-change values, refused at load ────────────────────────

_CAPS = {
    "world_voxel_deadline_s": 2.0,
    "world_voxel_data_age_budget_s": 3.0,
    "robot_self_filter_padding_m": 0.10,
}


@pytest.mark.parametrize(("field", "cap"), sorted(_CAPS.items()))
def test_rig_perception_values_have_hard_caps(field: str, cap: float) -> None:
    """A rig value at its cap loads; one above it is refused, naming field, cap and why."""
    at_cap = DeployRuntime.model_validate({field: cap})
    assert getattr(at_cap, field) == cap
    with pytest.raises(ValidationError, match=rf"{field}.*{cap}.*2x"):
        DeployRuntime.model_validate({field: cap + 0.001})


@pytest.mark.parametrize(("field", "cap"), sorted(_CAPS.items()))
def test_a_scene_above_a_cap_is_refused_before_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str, cap: float
) -> None:
    """On a franka_panda (not OpenArm) scene: the cap is enforced where the scene loads."""
    _real_franka_with(tmp_path, monkeypatch, _front_depth())
    base = f"  octomap_cloud_topic: {_REALSENSE_POINTS}\n{_NO_KERNEL_CHECK}"
    argv = _resolve(_real_scene(tmp_path, f"{base}  {field}: {cap}\n"), "real").argv_template
    assert f"{field}:={cap}" in argv
    with pytest.raises((ROSConfigError, ValidationError), match=field):
        _resolve(_real_scene(tmp_path, f"{base}  {field}: {cap * 1.5}\n"), "real")


def test_the_extrinsic_preflight_refuses_a_cloud_source_it_cannot_measure(tmp_path: Path) -> None:
    """A scene-mounted depth camera or a 3D lidar feeds octomap too.

    The preflight used to iterate only the robot manifest's depth cameras with
    intrinsics, so a scene-mounted RealSense or a lidar passed it with nothing checked,
    and the kernel would place obstacles through an unmeasured pose.
    """
    from openral_cli.deploy_sim import _preflight_depth_extrinsics

    robot = RobotDescription.from_yaml(str(_PANDA_MOBILE))
    workcell = SensorSpec(
        name="workcell_lidar", modality="point_cloud", frame_id="workcell_lidar", rate_hz=10.0
    )
    with pytest.raises(ROSConfigError) as err:
        _preflight_depth_extrinsics(robot, tmp_path / "robot.yaml", None, [workcell])
    assert "workcell_lidar: a scene point_cloud sensor without intrinsics" in str(err.value)
