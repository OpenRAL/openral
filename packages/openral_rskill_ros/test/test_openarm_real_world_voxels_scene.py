"""The real-OpenArm world-voxel scene arms the kernel's voxel check at the real margin.

``scenes/deploy/openarm_real_world_voxels.yaml`` is the first scene that turns the kernel's
world-voxel check on against a real camera on a real arm. Driven the way ``openral deploy
run`` drives it: ``resolve_launch_invocation(hal_mode="real")`` produces the argv, and every
``key:=value`` in it is fed to the real ``compose_runtime_graph``. Nothing is launched.

``deploy_config`` is dropped from the real-mode composition only: the scene's ``drivers:``
include resolves ``zed_wrapper`` from the ament path, which exists on the rig overlay, not
here. The kernel/octomap parameters under test come from the argv, not from that include;
the scene's own ``sensors:`` merge is covered by composing it on the sim path, where
drivers are ignored.

Per CLAUDE.md §1.11: the committed scene, the real ``robots/openarm/robot.yaml``, the real
CLI resolver and launch composition. No mocks.
"""

from __future__ import annotations

import os
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


def test_deploy_run_resolves_the_zed_cloud_and_the_kernel_check() -> None:
    args = _launch_args("real")

    assert args["hal_mode"] == "real"
    assert args["enable_octomap"] == "true"
    assert args["enable_octomap_kernel_check"] == "true"
    assert args["octomap_cloud_topic"] == _ZED_CLOUD
    assert args["enable_reasoner"] == "false"


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

    octomap = _node(entities, "octomap_server")
    remaps = {
        "".join(s.text for s in k): "".join(s.text for s in v) for k, v in octomap._Node__remappings
    }
    assert remaps["cloud_in"] == _ZED_CLOUD
    (octo_params,) = evaluate_parameters(ctx, octomap._Node__parameters)
    assert octo_params["resolution"] == 0.02
    assert octo_params["frame_id"] == "openarm_base"


def test_the_scene_pose_is_the_only_mount_published_for_the_zed(tmp_path: Path) -> None:
    """The scene's head_zed override reaches /tf_static, once, over the manifest's pose.

    The committed placeholder equals the manifest value, so a calibrated-looking pose is
    swapped in: the assertion must be able to tell the scene from the manifest.
    """
    import yaml

    calibrated = [0.013, -0.021, 0.231, 0.004, 0.771, -0.012]
    data = yaml.safe_load(_SCENE.read_text(encoding="utf-8"))
    (entry,) = [s for s in data["sensors"] if s["name"] == "head_zed"]
    entry["static_transform_xyz_rpy"] = calibrated
    scene = tmp_path / "calibrated.yaml"
    scene.write_text(yaml.safe_dump(data), encoding="utf-8")

    _, entities = _compose(_launch_args("sim", scene))

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
    assert [float(argv[argv.index(f) + 1]) for f in flags] == pytest.approx(calibrated)
