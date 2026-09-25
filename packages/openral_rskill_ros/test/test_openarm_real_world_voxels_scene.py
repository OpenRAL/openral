"""The real-OpenArm world-voxel scene arms the kernel's voxel check at the real margin.

``scenes/deploy/openarm_real_world_voxels.yaml`` is the first scene that turns the kernel's
world-voxel check on against a real camera on a real arm. Driven the way ``openral deploy
run`` drives it: ``resolve_launch_invocation(hal_mode="real")`` produces the argv, and every
``key:=value`` in it is fed to the real ``compose_runtime_graph``. Nothing is launched.

``deploy_config`` is dropped from the real-mode composition only: the scene's ``drivers:``
include resolves ``zed_wrapper`` from the ament path, which exists on the rig overlay, not
here. The kernel/octomap parameters under test come from the argv, not from that include;
the ZED mount (the robot manifest's ``head_zed`` with the Thor unit overlay) is covered by
composing the scene on the sim path, where drivers are ignored. The scene names no
``robot_unit`` (it runs on either cell), so ``OPENRAL_ROBOT_UNIT=thor`` selects the unit.

``deploy run`` refuses this scene until ``head_zed``'s extrinsic is verified (no report is
committed yet), so the real-mode tests resolve against a copy of ``robots/openarm`` served
via ``OPENRAL_ROBOTS_DIR`` with a passing report for the Thor unit's pose (manifest + unit
overlay) at ``calibration/thor/head_zed_extrinsic.json``.

Per CLAUDE.md §1.11: the committed scene, the real ``robots/openarm/robot.yaml``, the real
CLI resolver and launch composition. No mocks.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

import pytest
from _launch_test_common import import_launch_module

pytest.importorskip("launch")
pytest.importorskip("launch_ros")
pytest.importorskip("openral_cli")
pytest.importorskip("mujoco")

pytestmark = pytest.mark.skipif(
    not os.environ.get("ROS_DISTRO"),
    reason="ROS_DISTRO not set — these tests require a sourced ROS 2 installation.",
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_LAUNCH_FILE = _REPO_ROOT / "packages" / "openral_rskill_ros" / "launch" / "deploy_e2e.launch.py"
_SCENE = _REPO_ROOT / "scenes" / "deploy" / "openarm_real_world_voxels.yaml"
_ZED_CLOUD = "/zed/zed_node/point_cloud/cloud_registered"


@pytest.fixture
def calibrated_openarm(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """``robots/openarm`` copied, with a passing head_zed report for the Thor unit's pose."""
    from openral_core import RobotDescription, apply_sensor_overlays, load_robot_unit
    from openral_core.depth_extrinsic import (
        MAX_HEIGHT_ERR_M,
        MAX_MARKER_ERR_M,
        MAX_TILT_DEG,
        MIN_MARKERS,
        extrinsic_report_path,
    )

    robot_dir = tmp_path / "robots" / "openarm"
    shutil.copytree(_REPO_ROOT / "robots" / "openarm", robot_dir)
    desc = RobotDescription.from_yaml(str(robot_dir / "robot.yaml"))
    thor = load_robot_unit(robot_dir / "robot.yaml", "thor").sensors
    (zed,) = [s for s in apply_sensor_overlays(desc.sensors, thor) if s.name == "head_zed"]
    report = extrinsic_report_path(robot_dir / "robot.yaml", "head_zed", "thor")
    report.parent.mkdir(parents=True, exist_ok=True)
    markers = [{"expected_xy": [0.5, y], "error_m": 0.002} for y in (-0.15, 0.15)]
    report.write_text(
        json.dumps(
            {
                "unit": "thor",
                "sensor": zed.name,
                "parent_frame": zed.parent_frame,
                "frame_id": zed.frame_id,
                "base_frame": desc.base_frame,
                "static_transform_xyz_rpy": list(zed.static_transform_xyz_rpy or ()),
                "criteria": {
                    "max_tilt_deg": MAX_TILT_DEG,
                    "max_height_err_m": MAX_HEIGHT_ERR_M,
                    "max_marker_err_m": MAX_MARKER_ERR_M,
                    "min_markers": MIN_MARKERS,
                },
                "residuals": {"tilt_deg": 0.1, "height_err_m": 0.001, "markers": markers},
                "passed": True,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("OPENRAL_ROBOTS_DIR", str(tmp_path / "robots"))
    return robot_dir


@pytest.fixture(autouse=True)
def _thor_unit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENRAL_ROBOT_UNIT", "thor")


def _launch_args(hal_mode: str, scene: Path = _SCENE) -> dict[str, str]:
    """The ``key:=value`` launch arguments ``openral deploy run|sim`` would pass."""
    from openral_cli.deploy_sim import resolve_launch_invocation

    invocation = resolve_launch_invocation(
        config=scene,
        robot_override="openarm",
        dashboard_port=4318,
        reset_to_pose_service=None,
        deploy_config=scene,
        hal_param_overrides={},
        hal_mode=hal_mode,
        enable_dashboard=False,
    )
    args = dict(tok.split(":=", 1) for tok in invocation.argv_template if ":=" in tok)
    args["hal_params_file"] = "/tmp/openral-test-hal-params.yaml"
    return args


def _compose(args: dict[str, str]) -> list[Any]:
    from launch import LaunchContext
    from launch.actions import DeclareLaunchArgument

    module = import_launch_module(_LAUNCH_FILE)
    ctx = LaunchContext()
    ctx.launch_configurations.update(args)
    for entity in module.generate_launch_description().entities:
        if isinstance(entity, DeclareLaunchArgument):
            entity.execute(ctx)
    ctx.launch_configurations.update(args)  # CLI values win over declared defaults
    ctx.launch_configurations["enable_dashboard"] = "false"
    return [ctx, list(module.compose_runtime_graph(ctx))]


def _node(entities: list[Any], package: str, executable: str | None = None) -> Any:
    from launch_ros.actions import Node

    found = [
        e
        for e in entities
        if isinstance(e, Node)
        and getattr(e, "_Node__package", None) == package
        and (executable is None or getattr(e, "_Node__node_executable", None) == executable)
    ]
    assert len(found) == 1, f"expected one {package} node, found {len(found)}"
    return found[0]


def test_deploy_run_refuses_until_the_zed_extrinsic_is_verified() -> None:
    """The committed state: no head_zed report, so the real launch never resolves."""
    from openral_core.exceptions import ROSConfigError

    if (_REPO_ROOT / "robots/openarm/calibration/thor/head_zed_extrinsic.json").exists():
        pytest.skip("a head_zed calibration report is committed")
    with pytest.raises(ROSConfigError, match=r"head_zed: no extrinsic report .*/thor/"):
        _launch_args("real")


def test_deploy_run_reads_only_the_selected_units_report(calibrated_openarm: Path) -> None:
    """A report left at the unit-less path does not clear the selected unit."""
    from openral_core.exceptions import ROSConfigError

    report = calibrated_openarm / "calibration" / "thor" / "head_zed_extrinsic.json"
    report.rename(calibrated_openarm / "calibration" / "head_zed_extrinsic.json")
    with pytest.raises(ROSConfigError, match=r"head_zed: no extrinsic report .*/thor/"):
        _launch_args("real")


@pytest.mark.usefixtures("calibrated_openarm")
def test_deploy_run_resolves_the_zed_cloud_and_the_kernel_check() -> None:
    args = _launch_args("real")

    assert args["hal_mode"] == "real"
    assert args["enable_octomap"] == "true"
    assert args["enable_octomap_kernel_check"] == "true"
    assert args["octomap_cloud_topic"] == _ZED_CLOUD
    assert args["enable_reasoner"] == "false"


@pytest.mark.usefixtures("calibrated_openarm")
def test_the_kernel_gets_world_voxel_enabled_at_the_real_margin() -> None:
    from launch_ros.utilities import evaluate_parameters

    args = _launch_args("real")
    del args["deploy_config"]  # drivers: needs zed_wrapper on the ament path (rig only)
    ctx, entities = _compose(args)

    (kernel_params,) = evaluate_parameters(
        ctx, _node(entities, "openral_safety_kernel")._Node__parameters
    )
    assert kernel_params["world_voxel_enabled"] is True
    assert kernel_params["world_voxel_margin_m"] == 0.02
    assert kernel_params["world_voxel_deadline_ms"] == 1000.0
    # Real hardware has no attachment producer, so payload checking stays off.
    assert kernel_params.get("attached_collision_enabled", False) is False

    def remaps(node: Any) -> dict[str, str]:
        return {
            "".join(s.text for s in k): "".join(s.text for s in v)
            for k, v in node._Node__remappings
        }

    # Real camera: the robot self-filter sits between the ZED cloud and octomap.
    self_filter = _node(entities, "openral_octomap_bridge", "robot_self_filter")
    assert remaps(self_filter)["cloud_in"] == _ZED_CLOUD
    octomap = _node(entities, "octomap_server")
    assert remaps(octomap)["cloud_in"] == remaps(self_filter)["cloud_out"]
    (filter_params,) = evaluate_parameters(ctx, self_filter._Node__parameters)
    # The filter poses the robot from the stream the runtime nodes read: the real
    # ros2_control HAL's rate-limited republish (hal_joint_states_topic).
    assert filter_params["joint_states_topic"] == "/openral_hal_openarm/joint_states"
    (octo_params,) = evaluate_parameters(ctx, octomap._Node__parameters)
    assert octo_params["resolution"] == 0.02
    assert octo_params["frame_id"] == "openarm_base"


def test_the_unit_pose_is_the_only_mount_published_for_the_zed() -> None:
    """The selected unit's head_zed pose reaches /tf_static, once, with the manifest's parent.

    The ZED is bolted to the robot, so its mount is robot geometry: the scene declares no
    ``head_zed`` entry and may not (``check_scene_sensor_overrides``); its frames are the
    manifest's and its calibrated pose the unit overlay's (``robots/openarm/units/thor.yaml``).
    """
    import yaml
    from openral_core import load_robot_unit

    assert "sensors" not in yaml.safe_load(_SCENE.read_text(encoding="utf-8"))
    (zed,) = [
        s
        for s in load_robot_unit(_REPO_ROOT / "robots/openarm/robot.yaml", "thor").sensors
        if s.name == "head_zed"
    ]

    _, entities = _compose(_launch_args("sim"))

    mounts = []
    for e in entities:
        if getattr(e, "_Node__package", None) != "tf2_ros":
            continue
        argv = ["".join(getattr(s, "text", str(s)) for s in a) for a in e._Node__arguments]
        if argv[argv.index("--child-frame-id") + 1] == "zed_camera_link":
            mounts.append(argv)
    assert len(mounts) == 1, mounts
    (argv,) = mounts
    assert argv[argv.index("--frame-id") + 1] == "openarm_base"
    flags = ("--x", "--y", "--z", "--roll", "--pitch", "--yaw")
    assert [float(argv[argv.index(f) + 1]) for f in flags] == pytest.approx(
        list(zed.static_transform_xyz_rpy or ())
    )


def test_deploy_run_refuses_without_a_unit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Both cells run this scene with different ZED mounts: a real run must name its cell."""
    from openral_core.exceptions import ROSConfigError

    monkeypatch.delenv("OPENRAL_ROBOT_UNIT")
    with pytest.raises(ROSConfigError, match="none is selected"):
        _launch_args("real")


@pytest.mark.parametrize("hal_mode", ["real", "sim"])
def test_deploy_refuses_a_scene_that_names_the_zed(tmp_path: Path, hal_mode: str) -> None:
    """``deploy run`` / ``deploy sim`` / ``deploy validate`` refuse before launching.

    All three go through ``resolve_launch_invocation``, which is where the rule is checked
    pre-launch: a deploy scene never touches a robot camera — a pose (or binding) hidden in
    one scene would silently not apply to the robot's others.
    """
    import yaml
    from openral_core.exceptions import ROSConfigError

    data = yaml.safe_load(_SCENE.read_text(encoding="utf-8"))
    data["sensors"] = [
        {
            "name": "head_zed",
            "modality": "depth",
            "frame_id": "zed_camera_link",
            "parent_frame": "openarm_base",
            "rate_hz": 10.0,
            "static_transform_xyz_rpy": [0.013, -0.021, 0.231, 0.004, 0.771, -0.012],
        }
    ]
    scene = tmp_path / "hidden_pose.yaml"
    scene.write_text(yaml.safe_dump(data), encoding="utf-8")

    with pytest.raises(ROSConfigError, match=r"'head_zed'.*defined by the robot manifest"):
        _launch_args(hal_mode, scene)
