"""Each ``object_tracking`` mode switches only its own consumers (issue #418 matrix A).

``scenes/deploy/isaac_leisaac_pick_orange.yaml`` is composed at ``off``, ``shadow``,
``bind`` and ``on`` the way ``openral deploy`` drives it (``resolve_launch_invocation`` →
the real ``compose_runtime_graph``; nothing is launched). Every node's evaluated
parameters are diffed against ``off``, and the diff must be exactly the consumers the
mode owns:

* ``shadow``: the tracker node is added; the self-filter renders robot masks for the
  tracked camera; the segmenter caches its frames; the runtime and the reasoner are told
  the mode. The HAL and the kernel are untouched.
* ``bind``: adds the HAL's vision bridge giving up ``/openral/attachment_state`` and the
  kernel's jaw-open release (K5).
* ``on``: adds the kernel's tracked grasp exemption (K2), its contact-link list,
  payload ownership and explained-workspace cells.
"""

from __future__ import annotations

import os
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
_SCENE = _REPO_ROOT / "scenes" / "deploy" / "isaac_leisaac_pick_orange.yaml"

_TRACKER = ("openral_world_state", "object_tracker_node")
_FILTER = ("openral_octomap_bridge", "robot_self_filter")
_SEGMENTER = ("openral_perception_ros", "segmenter_node.py")
_REASONER = ("openral_reasoner_ros", "reasoner_node.py")
_RUNTIME = ("openral_rskill_ros", "runtime_node")
_HAL = ("openral_hal_node", "lifecycle_node.py")
_KERNEL = ("openral_safety_kernel", "safety_kernel_node")
_MASK = {
    "robot_mask_enabled",
    "robot_mask_cameras",
    "robot_mask_depth_topics",
    "robot_mask_camera_info_topics",
    "robot_mask_topics",
}
_K5 = {
    "tracked_release_jaw_joints",
    "tracked_release_jaw_open_sign",
    "tracked_release_gap",
    "tracked_release_dwell_s",
}
_SHADOW = {
    _FILTER: _MASK,
    _SEGMENTER: {"cameras", "camera_infos"},
    _REASONER: {"object_tracking"},
    _RUNTIME: {"object_tracking", "grounding_wait_s"},
}
_BIND = {**_SHADOW, _HAL: {"vision_attachment_publish_attachment_state"}, _KERNEL: _K5}
_ON = {
    **_BIND,
    _KERNEL: _K5
    | {
        "tracked_grasp_exemption_enabled",
        "tracked_grasp_extra_contact_links",
        "payload_own_cells_enabled",
        "pinch_payload_own_cells_enabled",
        "workspace_explained_cells_enabled",
    },
}


def _graph(
    tmp: Path, mode: str, *, catalog_json: str | None = None
) -> dict[tuple[str, str], dict[str, Any]]:
    """``(package, executable) -> evaluated parameters`` of every composed node."""
    from launch import LaunchContext
    from launch.actions import DeclareLaunchArgument
    from launch_ros.actions import Node
    from launch_ros.utilities import evaluate_parameters
    from openral_cli.deploy_sim import resolve_launch_invocation

    scene = yaml.safe_load(_SCENE.read_text())
    scene["runtime"] = {**(scene.get("runtime") or {}), "object_tracking": mode}
    if mode != "off":
        scene["runtime"]["grounding_wait_s"] = 5.0
    path = tmp / f"{mode}.yaml"
    path.write_text(yaml.safe_dump(scene))
    invocation = resolve_launch_invocation(
        config=path,
        robot_override=None,
        dashboard_port=4318,
        reset_to_pose_service=None,
        deploy_config=path,
        hal_param_overrides={"viewer_enabled": False},
        hal_mode="sim",
        enable_dashboard=False,
    )
    args = dict(tok.split(":=", 1) for tok in invocation.argv_template if ":=" in tok)
    if catalog_json is not None:
        args["declared_object_catalog_json"] = catalog_json
    args["hal_params_file"] = "/tmp/openral-test-hal-params.yaml"
    module = import_launch_module(_LAUNCH_FILE)
    ctx = LaunchContext()
    ctx.launch_configurations.update(args)
    for entity in module.generate_launch_description().entities:
        if isinstance(entity, DeclareLaunchArgument):
            entity.execute(ctx)
    ctx.launch_configurations.update(args)
    ctx.launch_configurations["enable_dashboard"] = "false"
    nodes: dict[tuple[str, str], dict[str, Any]] = {}
    for entity in module.compose_runtime_graph(ctx):
        if not isinstance(entity, Node):
            continue
        key = (entity._Node__package, entity._Node__node_executable)
        assert key not in nodes, f"two {key} nodes: the diff below would be ambiguous"
        params: dict[str, Any] = {}
        for p in evaluate_parameters(ctx, entity._Node__parameters):
            if isinstance(p, dict):
                params.update(p)
        nodes[key] = params
    return nodes


@pytest.fixture(scope="module")
def graphs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    tmp = tmp_path_factory.mktemp("modes")
    return {mode: _graph(tmp, mode) for mode in ("off", "shadow", "bind", "on")}


@pytest.mark.parametrize(("mode", "expected"), [("shadow", _SHADOW), ("bind", _BIND), ("on", _ON)])
def test_each_mode_switches_only_its_own_consumers(
    graphs: dict[str, Any], mode: str, expected: dict[tuple[str, str], set[str]]
) -> None:
    off, now = graphs["off"], graphs[mode]
    assert set(now) - set(off) == {_TRACKER}, "the tracker is the only node a mode adds"
    assert set(off) <= set(now), "a mode never removes a node"
    changed = {
        key: {k for k in set(off[key]) | set(now[key]) if off[key].get(k) != now[key].get(k)}
        for key in off
    }
    assert {k: v for k, v in changed.items() if v} == expected
    assert now[_TRACKER]["mode"] == mode
    assert now[_RUNTIME]["object_tracking"] == mode
    hal = now[_HAL].get("vision_attachment_publish_attachment_state", True)
    assert hal is (mode == "shadow"), "the HAL publishes attachment_state only in shadow"
    assert now[_KERNEL]["tracked_grasp_exemption_enabled"] is (mode == "on")
    assert now[_KERNEL]["payload_own_cells_enabled"] is (mode == "on")
    assert now[_KERNEL]["pinch_payload_own_cells_enabled"] is (mode == "on")
    assert bool(now[_KERNEL].get("tracked_release_jaw_joints")) is (mode in ("bind", "on"))
    assert now[_FILTER]["robot_mask_enabled"] is True


def test_launch_refuses_external_catalog_when_tracking_is_off(tmp_path: Path) -> None:
    """Direct launch args must not silently lose the external declarations."""
    pytest.importorskip("openral_foxglove_bringup")
    from openral_core.exceptions import ROSConfigError

    with pytest.raises(ROSConfigError, match="declared_object_catalog_json requires"):
        _graph(tmp_path, "off", catalog_json="{}")
