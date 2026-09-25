"""Per-rig ``DeployRuntime`` perception values reach the nodes that enforce them.

``world_voxel_data_age_budget_s`` becomes the safety kernel's
``world_voxel_data_age_budget_ms`` and ``robot_self_filter_padding_m`` the
``robot_self_filter``'s ``padding_m``, whether a scene declares them or not.
Driven the way ``openral deploy`` drives it: ``resolve_launch_invocation``
produces the argv, every ``key:=value`` is fed to the real
``compose_runtime_graph``. Nothing is launched.

The robot is ``franka_panda`` (not OpenArm) with panda_mobile's committed
``front_depth`` camera added, served via ``OPENRAL_ROBOTS_DIR``. The kernel
params come from a sim twin (the kernel check needs no extrinsic report
there); the self-filter only exists on the real path, composed with the
kernel check off (it filters the map's input whether or not the kernel reads
the map). Per CLAUDE.md §1.11: real manifests, real resolver, no mocks.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml
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
_POINTS = "/camera/depth/color/points"


@pytest.fixture
def franka_with_depth(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    data = yaml.safe_load((_REPO_ROOT / "robots/panda_mobile/robot.yaml").read_text())
    (depth,) = [s for s in data["sensors"] if s["name"] == "front_depth"]
    dst = tmp_path / "robots" / "franka_panda"
    shutil.copytree(_REPO_ROOT / "robots" / "franka_panda", dst)
    manifest = yaml.safe_load((dst / "robot.yaml").read_text())
    manifest["sensors"] = [*(manifest.get("sensors") or []), depth]
    (dst / "robot.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))
    monkeypatch.setenv("OPENRAL_ROBOTS_DIR", str(tmp_path / "robots"))
    return tmp_path


def _scene(root: Path, runtime: str) -> Path:
    scene = root / "franka_cell.yaml"
    scene.write_text(
        f"scene:\n  id: franka_cell\nrobot_id: franka_panda\nruntime:\n  enable_octomap: true\n"
        f"{runtime}",
        encoding="utf-8",
    )
    return scene


def _compose(scene: Path, hal_mode: str) -> tuple[Any, list[Any]]:
    from launch import LaunchContext
    from launch.actions import DeclareLaunchArgument
    from openral_cli.deploy_sim import resolve_launch_invocation

    invocation = resolve_launch_invocation(
        config=scene,
        robot_override=None,
        dashboard_port=4318,
        reset_to_pose_service=None,
        deploy_config=scene,
        hal_param_overrides={"viewer_enabled": False} if hal_mode == "sim" else {},
        hal_mode=hal_mode,
        enable_dashboard=False,
    )
    args = dict(tok.split(":=", 1) for tok in invocation.argv_template if ":=" in tok)
    args["hal_params_file"] = "/tmp/openral-test-hal-params.yaml"
    module = import_launch_module(_LAUNCH_FILE)
    ctx = LaunchContext()
    ctx.launch_configurations.update(args)
    for entity in module.generate_launch_description().entities:
        if isinstance(entity, DeclareLaunchArgument):
            entity.execute(ctx)
    ctx.launch_configurations.update(args)
    ctx.launch_configurations["enable_dashboard"] = "false"
    return ctx, list(module.compose_runtime_graph(ctx))


def _params(ctx: Any, entities: list[Any], package: str, executable: str) -> dict[str, Any]:
    from launch_ros.actions import Node
    from launch_ros.utilities import evaluate_parameters

    (node,) = [
        e
        for e in entities
        if isinstance(e, Node)
        and getattr(e, "_Node__package", None) == package
        and getattr(e, "_Node__node_executable", None) == executable
    ]
    (params,) = evaluate_parameters(ctx, node._Node__parameters)
    return dict(params)


@pytest.mark.parametrize(("declared", "expected_ms"), [("", 1500.0), ("0.8", 800.0)])
def test_the_data_age_budget_reaches_the_kernel(
    franka_with_depth: Path, declared: str, expected_ms: float
) -> None:
    runtime = f"  world_voxel_data_age_budget_s: {declared}\n" if declared else ""
    ctx, entities = _compose(_scene(franka_with_depth, runtime), "sim")
    kernel = _params(ctx, entities, "openral_safety_kernel", "safety_kernel_node")
    assert kernel["world_voxel_enabled"] is True
    assert kernel["world_voxel_data_age_budget_ms"] == expected_ms


@pytest.mark.parametrize(("declared", "expected"), [("", 0.05), ("0.03", 0.03)])
def test_the_self_filter_padding_reaches_the_filter(
    franka_with_depth: Path, declared: str, expected: float
) -> None:
    runtime = f"  octomap_cloud_topic: {_POINTS}\n  enable_octomap_kernel_check: false\n" + (
        f"  robot_self_filter_padding_m: {declared}\n" if declared else ""
    )
    ctx, entities = _compose(_scene(franka_with_depth, runtime), "real")
    params = _params(ctx, entities, "openral_octomap_bridge", "robot_self_filter")
    assert params["padding_m"] == expected
