"""Each ``object_tracking`` mode switches only its own consumers (issue #418 matrix A).

``scenes/deploy/isaac_leisaac_pick_orange.yaml`` is composed at ``off``, ``tracking`` and
``on`` the way ``openral deploy`` drives it (``resolve_launch_invocation`` → the real
``compose_runtime_graph``; nothing is launched). Every node's evaluated parameters are
diffed against ``off``, and the diff must be exactly the consumers the mode owns:

* ``tracking``: the tracker node is added; the self-filter renders robot masks for the
  tracked camera; the segmenter caches only the tracked cameras (the HAL's vision bridge
  gives up ``/openral/attachment_state`` and runs only its grasp trigger) and serves them on
  the tracker's frame wait and skew budget, and so does the scene's on-demand locator; the
  runtime and the reasoner are told the mode, the runtime gets the scene's
  ``grounding_required`` and arms its goal-scope place declaration (the tracker measures
  its region, ``TrackerSession.place_region``). The kernel is untouched: no
  ``contact_exemptions.*`` (K2 / K5 / K7 / OT-10 are exemption-provider rules since
  ADR-0121 step 3); its open box rules take the regions the tracker measures.
* ``on`` (refused without an ``exemption_provider``; diffed against ``off`` with the same
  provider): as ``tracking`` plus, for the provider only, under ``contact_exemptions.``, K5,
  the tracked grasp exemption (K2), payload ownership and explained-workspace cells; the
  fixed-jaw contact-link list goes to the kernel, which hands it to the provider.
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
_BANANA = _REPO_ROOT / "scenes" / "deploy" / "isaac_arena_droid_banana_in_bowl.yaml"

_TRACKER = ("openral_world_state", "object_tracker_node")
_FILTER = ("openral_octomap_bridge", "robot_self_filter")
_SEGMENTER = ("openral_perception_ros", "segmenter_node.py")
_LOCATOR = ("openral_perception_ros", "ros_image_detector_node.py")
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
_TRACKING = {
    _FILTER: _MASK,
    # The HAL vision camera is dropped from the segmenter: only the tracked cameras stay.
    _SEGMENTER: {
        "cameras",
        "camera_infos",
        "camera_request_deadlines",
        "camera_max_frame_skews",
        "primary_camera",
    },
    # The scene's committed omdet-turbo-locator also caches the tracked cameras.
    _LOCATOR: {"cameras", "camera_infos"},
    _REASONER: {"object_tracking"},
    # The tracker measures the place region: the goal-scope place declaration is armed.
    _RUNTIME: {
        "object_tracking",
        "grounding_wait_s",
        "grounding_required",
        "place_approach_enabled",
    },
    _HAL: {"vision_attachment_publish_attachment_state"},
}
#: The rules ADR-0121 step 3 removed from the open kernel: provider-only.
_PROVIDER_ONLY = _K5 | {
    "tracked_grasp_exemption_enabled",
    "payload_own_cells_enabled",
    "pinch_payload_own_cells_enabled",
    "workspace_explained_cells_enabled",
}
_PROVIDER = "openral_pro_safety_exemptions/ContactExemptions"
_ON = {
    **_TRACKING,
    _KERNEL: {
        f"contact_exemptions.{k}" for k in _PROVIDER_ONLY | {"tracked_grasp_extra_contact_links"}
    }
    | {"tracked_grasp_extra_contact_links"},
}


def _graph(
    tmp: Path,
    mode: str,
    *,
    catalog_json: str | None = None,
    scene_path: Path = _SCENE,
    real_hal_params: bool = False,
    extra_args: dict[str, str] | None = None,
    provider: str = "",
) -> dict[tuple[str, str], dict[str, Any]]:
    """``(package, executable) -> evaluated parameters`` of every composed node.

    ``real_hal_params`` writes the HAL params ``openral deploy`` resolves into the file the
    launch reads (else a path that does not exist, read as ``{}``).
    """
    from launch import LaunchContext
    from launch.actions import DeclareLaunchArgument
    from launch_ros.actions import Node
    from launch_ros.utilities import evaluate_parameters
    from openral_cli.deploy_sim import resolve_launch_invocation

    scene = yaml.safe_load(scene_path.read_text())
    scene["runtime"] = {**(scene.get("runtime") or {}), "object_tracking": mode}
    if mode != "off":
        scene["runtime"]["grounding_wait_s"] = 5.0
        # Pinned true (the twins commit it) so tracking/on show it forwarded.
        scene["runtime"]["grounding_required"] = True
    if provider:
        scene["runtime"]["exemption_provider"] = provider
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
    args.update(extra_args or {})
    if real_hal_params:
        hal_file = tmp / f"{mode}-hal.yaml"
        hal_file.write_text(yaml.safe_dump({"/**": {"ros__parameters": invocation.hal_params}}))
        args["hal_params_file"] = str(hal_file)
    else:
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
    out = {mode: _graph(tmp, mode) for mode in ("off", "tracking")}
    out["off+provider"] = _graph(tmp, "off", provider=_PROVIDER)
    out["on"] = _graph(tmp, "on", provider=_PROVIDER)
    return out


@pytest.mark.parametrize(("mode", "expected"), [("tracking", _TRACKING), ("on", _ON)])
def test_each_mode_switches_only_its_own_consumers(
    graphs: dict[str, Any], mode: str, expected: dict[tuple[str, str], set[str]]
) -> None:
    off, now = graphs["off+provider" if mode == "on" else "off"], graphs[mode]
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
    assert hal is False, "the tracker is the only attachment_state author"
    assert now[_RUNTIME]["place_approach_enabled"] is True
    assert now[_RUNTIME]["grounding_required"] is True
    va = {e.split("=", 1)[0] for e in now[_SEGMENTER]["cameras"]}
    assert "top" not in va, "only the tracked cameras are cached"
    for key in _PROVIDER_ONLY:
        assert key not in now[_KERNEL], f"{key}: provider-only since ADR-0121 step 3"
    if mode == "on":
        assert now[_KERNEL]["contact_exemptions.tracked_grasp_exemption_enabled"] is True
        assert now[_KERNEL]["contact_exemptions.payload_own_cells_enabled"] is True
        assert now[_KERNEL]["contact_exemptions.tracked_release_jaw_joints"]
    else:
        assert not [k for k in now[_KERNEL] if k.startswith("contact_exemptions.")]
        assert "tracked_grasp_extra_contact_links" not in now[_KERNEL]
    assert now[_FILTER]["robot_mask_enabled"] is True


def test_tracking_on_without_a_provider_is_refused(tmp_path: Path) -> None:
    """ADR-0121 step 3: in open, `on` would stand the HAL region producer down and leave no
    grasp exemption; the scene's `DeployRuntime` refuses it at load."""
    from openral_core.exceptions import ROSConfigError
    from pydantic import ValidationError

    with pytest.raises((ROSConfigError, ValidationError), match="ADR-0121"):
        _graph(tmp_path, "on")


@pytest.mark.parametrize("mode", ["off", "tracking", "on"])
def test_every_mode_arms_the_place_declaration_over_its_producer(tmp_path: Path, mode: str) -> None:
    """The banana scene runs the HAL place-target leg; tracking / on do not build it.

    Off arms the goal-scope place declaration whose region that leg measures. Under
    tracking / on the tracker measures it (`TrackerSession.place_region`): armed, a
    sim-only non-monotone relaxation that awaits Safety-WG review (Spark A/B R388 vs R389:
    only then does the banana reach the bowl).
    """
    nodes = _graph(
        tmp_path,
        mode,
        scene_path=_BANANA,
        real_hal_params=True,
        provider=_PROVIDER if mode == "on" else "",
    )
    assert nodes[_RUNTIME]["place_approach_enabled"] is True


def test_payload_own_cells_get_the_declared_static_cells_with_k7_off(tmp_path: Path) -> None:
    """Spark R374: with K7 (`workspace_explained_cells`) off the kernel got no
    `static_points_file`, so payload ownership (on with `on`) logged `static_cells=0` and
    tolerated shelf cells inside the held box as payload. The declared static cells follow
    every consumer: here K7 off, payload-own cells on. Since ADR-0121 step 3 both are provider
    rules, so this composes `on` with a provider; the kernel takes the cells for its view."""
    geometry = tmp_path / "static.yaml"
    geometry.write_text(
        yaml.safe_dump(
            {
                "static_geometry": [
                    {
                        "kind": "box",
                        "source": "/World/Shelf",
                        "xyz": [0.3, -0.2, 0.0],
                        "quat_xyzw": [0.0, 0.0, 0.0, 1.0],
                        "half_extents": [0.1, 0.02, 0.1],
                    }
                ]
            }
        )
    )
    kernel = _graph(
        tmp_path,
        "on",
        extra_args={
            "static_geometry_file": str(geometry),
            "workspace_explained_cells": "false",
        },
        provider=_PROVIDER,
    )[_KERNEL]
    assert kernel.get("contact_exemptions.workspace_explained_cells_enabled", False) is False
    assert kernel["contact_exemptions.payload_own_cells_enabled"] is True
    cells = Path(kernel["static_points_file"])
    assert cells.name.startswith("openral_static_cells_") and cells.read_text().strip()


def test_launch_refuses_external_catalog_when_tracking_is_off(tmp_path: Path) -> None:
    """Direct launch args must not silently lose the external declarations."""
    pytest.importorskip("openral_foxglove_bringup")
    from openral_core.exceptions import ROSConfigError

    with pytest.raises(ROSConfigError, match="declared_object_catalog_json requires"):
        _graph(tmp_path, "off", catalog_json="{}")
