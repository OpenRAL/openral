"""A shared-world robot's kernel sees its peers (issue #666), and nothing else does.

``shared_world:=`` (passed by ``openral deploy world`` through ``deploy sim``) turns the
kernel's world check on over the coverage ball, remaps its world grid to the peers'
topic, starts ``openral_hal.peer_voxels`` with that ball, and shuts the graph down if the
producer dies. It is refused where the peers would be dropped or tolerated.

Real ``robots/so101_follower`` manifest, real ``LaunchContext``, real
``compose_runtime_graph``. No mocks (CLAUDE.md §1.11).
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

pytestmark = [
    pytest.mark.skipif(
        not os.environ.get("ROS_DISTRO"),
        reason="ROS_DISTRO not set — these tests require a sourced ROS 2 installation.",
    ),
    pytest.mark.usefixtures("host_confined_ros_graph"),
]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_LAUNCH_FILE = _REPO_ROOT / "packages" / "openral_rskill_ros" / "launch" / "deploy_e2e.launch.py"
_ROBOT = "so101_follower"


def _compose(**extra: str) -> list[Any]:
    """Compose the real sim graph for the SO-101 with ``extra`` launch configurations."""
    from launch import LaunchContext
    from launch.actions import DeclareLaunchArgument

    spec = importlib.util.spec_from_file_location("deploy_e2e_launch_sw", _LAUNCH_FILE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    ctx = LaunchContext()
    cfg = ctx.launch_configurations
    cfg["robot_yaml"] = str(_REPO_ROOT / "robots" / _ROBOT / "robot.yaml")
    cfg["hal_package"] = "openral_hal_node"
    cfg["hal_executable"] = "lifecycle_node.py"
    cfg["hal_node_name"] = f"openral_hal_{_ROBOT}"
    cfg["hal_params_file"] = "/tmp/openral-test-hal-params.yaml"
    for entity in module.generate_launch_description().entities:
        if isinstance(entity, DeclareLaunchArgument):
            entity.execute(ctx)
    cfg["hal_mode"] = "sim"
    cfg["clock_origin"] = "simulation"
    for leg in ("slam", "nav2", "octomap", "object_detector", "dashboard"):
        cfg[f"enable_{leg}"] = "false"
    cfg.update(extra)
    return list(module.compose_runtime_graph(ctx))


def _text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    return "".join(part.text for part in value if hasattr(part, "text"))


def _kernel(entities: list[Any]) -> Any:
    from launch_ros.actions import LifecycleNode

    return next(
        e
        for e in entities
        if isinstance(e, LifecycleNode)
        and _text(getattr(e, "_Node__node_name", "")) == "openral_safety_kernel"
    )


def _kernel_params(kernel: Any) -> dict[str, Any]:
    import yaml

    resolved: dict[str, Any] = {}
    for block in kernel._Node__parameters:  # reason: no public accessor pre-execute
        if hasattr(block, "items"):
            for key, value in block.items():
                resolved[_text(key).strip()] = yaml.safe_load(_text(value))
    return resolved


def _remaps(node: Any) -> dict[str, str]:
    return {_text(a): _text(b) for a, b in (node._Node__remappings or [])}


def _peer_producer(entities: list[Any]) -> list[str] | None:
    from launch.actions import ExecuteProcess

    for entity in entities:
        if isinstance(entity, ExecuteProcess):
            argv = [sub.text for group in entity.cmd for sub in group if hasattr(sub, "text")]
            if "openral_hal.peer_voxels" in argv:
                return argv
    return None


def test_the_kernel_reads_the_peers_over_its_coverage_ball() -> None:
    entities = _compose(shared_world="/tmp/w.sock", shared_world_robot="a")
    kernel = _kernel(entities)
    params = _kernel_params(kernel)
    assert params["world_voxel_enabled"] is True
    # The peers move at arm speed: a peer grid is current for 0.2 s, not the octomap's 1 s.
    assert params["world_voxel_deadline_ms"] == params["world_voxel_data_age_budget_ms"] == 200.0
    assert _remaps(kernel)["/openral/world_voxels"] == "/openral/peer_voxels"
    argv = _peer_producer(entities)
    assert argv is not None
    centre = [float(v) for v in argv[argv.index("--centre") + 1 : argv.index("--centre") + 4]]
    radius = float(argv[argv.index("--radius") + 1])
    resolution = float(argv[argv.index("--resolution") + 1])
    assert len(centre) == 3 and radius > 0.3  # the SO-101 reaches about 0.4 m
    per_axis = int(2 * radius / resolution + 0.5 + 1e-9) + 1
    assert params["world_voxel_max_cells"] == per_axis**3  # what the producer publishes
    assert argv[argv.index("--robot") + 1] == "a"
    assert argv[argv.index("--frame") + 1] == "base"


def test_without_a_shared_world_nothing_changes() -> None:
    entities = _compose()
    kernel = _kernel(entities)
    assert not _kernel_params(kernel).get("world_voxel_enabled", False)
    assert "/openral/world_voxels" not in _remaps(kernel)
    assert _peer_producer(entities) is None


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        ({"enable_octomap": "true"}, "enable_octomap off"),
        ({"shared_world_robot": ""}, "shared_world_robot"),
        ({"clock_origin": "host_wall"}, "clock_origin=simulation"),
        ({"enable_octomap_kernel_check": "false"}, "world check must be on"),
        ({"grasp_allowance_enabled": "true"}, "hand contact with the other robot"),
    ],
)
def test_a_shared_world_is_refused_where_the_peers_would_be_dropped_or_tolerated(
    extra: dict[str, str], message: str
) -> None:
    from openral_core.exceptions import ROSConfigError

    with pytest.raises(ROSConfigError, match=message):
        _compose(**{"shared_world": "/tmp/w.sock", "shared_world_robot": "a", **extra})
