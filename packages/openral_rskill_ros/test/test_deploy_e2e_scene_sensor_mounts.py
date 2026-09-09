"""A workcell-mounted sensor's static transform must reach /tf_static.

``deploy_e2e.launch.py`` turns a sensor's ``parent_frame`` + ``static_transform_xyz_rpy`` into
a ``static_transform_publisher`` so its readings are located by TF instead of mislabelled
into an existing frame. That loop iterated ``description.sensors`` only (the robot
manifest), so a camera declared in ``DeployScene.sensors`` could never get its mount
published.

Not hypothetical: every camera on the OpenArm restock cell is declared at scene level
(``scenes/deploy/openarm_restock_shelf.yaml``, three entries with ``parent_frame:
openarm_base``), as is any ZED feeding the octomap leg. Same silent failure the loop exists
to prevent: `octomap_server` can't resolve the cloud's frame, drops every message, and the
map stays empty while the graph reports healthy.

Per CLAUDE.md §1.11: real ``RobotDescription``, real ``DeployScene``, real
``LaunchContext``, real ``compose_runtime_graph``. No mocks.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

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

# A workcell camera on the OpenArm cell: parented to the manifest's own base
# frame, 0.20 m up and pitched down — the shape of a real head-camera mount.
_MOUNT = (0.0, 0.0, 0.20, 0.0, 0.7853981634, 0.0)

_SCENE_YAML = """
robot_id: "openarm"
scene:
  id: openarm_tabletop_pnp
  backend: mujoco
sensors:
  - name: head_zed
    modality: depth
    frame_id: zed_camera_link
    parent_frame: openarm_base
    static_transform_xyz_rpy: [0.0, 0.0, 0.20, 0.0, 0.7853981634, 0.0]
    rate_hz: 10.0
"""


def _import_launch_module() -> object:
    spec = importlib.util.spec_from_file_location("deploy_e2e_launch_mounts", _LAUNCH_FILE)
    assert spec is not None and spec.loader is not None, f"failed to spec {_LAUNCH_FILE}"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _compose(scene_path: Path | None) -> list[object]:
    """Run the real launch composition, optionally with a DeployScene attached."""
    from launch import LaunchContext
    from launch.actions import DeclareLaunchArgument

    module = _import_launch_module()
    ctx = LaunchContext()
    cfg = ctx.launch_configurations
    cfg["robot_yaml"] = str(_REPO_ROOT / "robots" / "openarm" / "robot.yaml")
    cfg["hal_package"] = "openral_hal_openarm"
    cfg["hal_executable"] = "lifecycle_node.py"
    cfg["hal_node_name"] = "openral_hal_test"
    cfg["hal_params_file"] = "/tmp/openral-test-hal-params.yaml"
    for entity in module.generate_launch_description().entities:  # type: ignore[attr-defined]
        if isinstance(entity, DeclareLaunchArgument):
            entity.execute(ctx)
    cfg["hal_mode"] = "sim"
    cfg["enable_slam"] = "false"
    cfg["enable_nav2"] = "false"
    cfg["enable_octomap"] = "false"
    cfg["enable_object_detector"] = "false"
    cfg["enable_dashboard"] = "false"
    if scene_path is not None:
        cfg["deploy_config"] = str(scene_path)
    return list(module.compose_runtime_graph(ctx))  # type: ignore[attr-defined]


def _static_transform_args(entities: list[object]) -> list[list[str]]:
    """The argument list of every static_transform_publisher in the graph."""
    from launch_ros.actions import Node

    found: list[list[str]] = []
    for entity in entities:
        if not isinstance(entity, Node):
            continue
        if getattr(entity, "_Node__package", None) != "tf2_ros":
            continue
        args = [
            a.perform(LaunchContextStub()) if hasattr(a, "perform") else str(a)
            for a in (getattr(entity, "_Node__arguments", None) or [])
        ]
        found.append(args)
    return found


class LaunchContextStub:
    """Minimal context for rendering the literal string substitutions."""

    launch_configurations: dict[str, str] = {}  # noqa: RUF012

    def perform_substitution(self, sub: object) -> str:
        return str(sub)


def _mount_for(entities: list[object], parent: str, child: str) -> list[str] | None:
    for args in _static_transform_args(entities):
        if "--frame-id" not in args or "--child-frame-id" not in args:
            continue
        if args[args.index("--frame-id") + 1] == parent and (
            args[args.index("--child-frame-id") + 1] == child
        ):
            return args
    return None


def test_a_scene_declared_sensor_gets_its_mount_published(tmp_path: Path) -> None:
    """The regression: this transform did not exist before scene sensors were merged."""
    scene = tmp_path / "workcell.yaml"
    scene.write_text(_SCENE_YAML, encoding="utf-8")

    args = _mount_for(_compose(scene), "openarm_base", "zed_camera_link")

    assert args is not None, (
        "no static_transform_publisher for the scene-declared sensor — the mount "
        "loop is not seeing DeployScene.sensors, so octomap_server / SLAM cannot "
        "resolve the cloud's frame and will silently drop every message"
    )
    for flag, expected in zip(
        ("--x", "--y", "--z", "--roll", "--pitch", "--yaw"), _MOUNT, strict=True
    ):
        assert float(args[args.index(flag) + 1]) == pytest.approx(expected), flag


def test_without_a_scene_the_manifest_mounts_still_publish() -> None:
    """Behaviour-preserving: the URDF-root bridge is unaffected by the merge.

    OpenArm's `base_frame` is not its URDF root, so the manifest declares an
    `assets.urdf` bridge; that transform must exist with or without a scene.
    """
    assert _mount_for(_compose(None), "openarm_base", "world") is not None
