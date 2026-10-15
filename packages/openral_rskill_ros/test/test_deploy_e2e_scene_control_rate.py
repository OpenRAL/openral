"""A sim scene's control rate and simulated depth reach the composed deploy graph.

``scenes/deploy/isaac_leisaac_pick_orange.yaml`` runs the SO-101 at LeIsaac's 60 Hz
(``backend_options.control_freq_hz``) and maps a simulated ``top_depth`` camera
(``backend_options.sim_sensors``). The launch applies both through
``openral_core.sim_scene_description``: the runtime node's ``rate_hz`` (the runner's
tick and the recorder's fps) is the scene's rate, the octomap leg reads the simulated
camera's cloud, and the safety kernel runs its self-collision (the manifest's OBB model
and negative margin) and world-voxel checks. Without a scene the runner keeps the
manifest's rate (``rate_hz`` 0).

Per CLAUDE.md §1.11: real ``RobotDescription``, real ``DeployScene``, real
``LaunchContext``, real ``compose_runtime_graph``. No mocks.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("launch")
pytest.importorskip("launch_ros")
pytest.importorskip("openral_core")
pytest.importorskip("openral_safety")
pytest.importorskip("mujoco")

pytestmark = pytest.mark.skipif(
    not os.environ.get("ROS_DISTRO"),
    reason="ROS_DISTRO not set — these tests require a sourced ROS 2 installation.",
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_LAUNCH_FILE = _REPO_ROOT / "packages" / "openral_rskill_ros" / "launch" / "deploy_e2e.launch.py"
_SCENE = _REPO_ROOT / "scenes" / "deploy" / "isaac_leisaac_pick_orange.yaml"


def _compose(scene: Path | None) -> list[object]:
    from launch import LaunchContext
    from launch.actions import DeclareLaunchArgument

    spec = importlib.util.spec_from_file_location("deploy_e2e_launch_rate", _LAUNCH_FILE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    ctx = LaunchContext()
    cfg = ctx.launch_configurations
    cfg["robot_yaml"] = str(_REPO_ROOT / "robots" / "so101_follower" / "robot.yaml")
    cfg["hal_package"] = "openral_hal_node"
    cfg["hal_executable"] = "lifecycle_node.py"
    cfg["hal_node_name"] = "openral_hal_test"
    cfg["hal_params_file"] = "/tmp/openral-test-hal-params.yaml"
    for entity in module.generate_launch_description().entities:
        if isinstance(entity, DeclareLaunchArgument):
            entity.execute(ctx)
    cfg["hal_mode"] = "sim"
    cfg["enable_slam"] = "false"
    cfg["enable_nav2"] = "false"
    cfg["enable_octomap"] = "true" if scene is not None else "false"
    cfg["enable_octomap_kernel_check"] = "true"
    cfg["enable_object_detector"] = "false"
    cfg["enable_dashboard"] = "false"
    if scene is not None:
        cfg["deploy_config"] = str(scene)
    return list(module.compose_runtime_graph(ctx))


def _text(subs: object) -> str:
    if isinstance(subs, list | tuple):
        return "".join(getattr(sub, "text", str(sub)) for sub in subs)
    return str(subs)


def _node(entities: list[object], executable: str) -> Any:
    from launch_ros.actions import Node

    for entity in entities:
        if isinstance(entity, Node) and getattr(entity, "_Node__node_executable", None) == (
            executable
        ):
            return entity
    raise AssertionError(f"{executable} not in the composed graph")


def _param(node: Any, name: str) -> object:
    """One scalar parameter of ``node``, rendered from launch_ros's normalized params."""
    import yaml

    for params in getattr(node, "_Node__parameters", None) or ():
        for key, value in params.items() if isinstance(params, dict) else ():
            if _text(key) == name:
                return (
                    value if isinstance(value, int | float | bool) else yaml.safe_load(_text(value))
                )
    raise AssertionError(f"{name} not among the node's parameters")


def _remaps(node: Any) -> dict[str, str]:
    return {_text(a): _text(b) for a, b in getattr(node, "_Node__remappings", None) or ()}


def test_the_scene_rate_ticks_the_runner_and_its_depth_feeds_the_kernel_map() -> None:
    graph = _compose(_SCENE)
    assert _param(_node(graph, "runtime_node"), "rate_hz") == 60.0
    assert _remaps(_node(graph, "robot_self_filter"))["cloud_in"] == (
        "/openral/cameras/top_depth/points"
    )
    kernel = _node(graph, "safety_kernel_node")
    assert _param(kernel, "world_voxel_enabled") is True
    assert _param(kernel, "self_collision_enabled") is True
    assert _param(kernel, "self_collision_margin_m") == -0.06  # robots/so101_follower


def test_without_a_scene_the_runner_keeps_the_manifest_rate() -> None:
    assert _param(_node(_compose(None), "runtime_node"), "rate_hz") == 0.0
