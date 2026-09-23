"""``detector_available`` must track a real ``locate_in_view`` service, not a phantom one.

Reproduced live: ``openral deploy sim --config scenes/deploy/libero_pnp.yaml
--object-detector --initial-task "..."`` surfaced the ``locate_in_view`` tool
to the LLM, which called it exactly as its system prompt instructs, and got
``/openral/perception/default/locate_in_view not on graph; skipping`` on
every tick — the mission stalled on a tool the reasoner was told existed.

Root cause: the two detector node modes are mutually exclusive at the node
(``detector_node_wiring``, ``detector_factory.py``) — ``continuous``
(``--object-detector``) never constructs the ``LocateInView`` service under
any name; only ``on_demand`` (``--object-detector-locator``) does. But
``reasoner_params["detector_available"]`` was set from
``enable_object_detector or bool(locator_specs)``, so a plain
``--object-detector`` deploy with zero locators flipped it ``True`` anyway —
offering a tool with no backing service and no ``default_on_demand_detector``
to route to.

Real ``LaunchContext`` + real ``compose_runtime_graph`` + the real in-tree
``rskills/omdet-turbo-locator/rskill.yaml`` manifest, per CLAUDE.md §1.11 —
same harness as ``test_deploy_e2e_scene_sensor_mounts.py``.
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
pytest.importorskip("openral_reasoner")
pytest.importorskip("mujoco")

pytestmark = pytest.mark.skipif(
    not os.environ.get("ROS_DISTRO"),
    reason="ROS_DISTRO not set — these tests require a sourced ROS 2 installation.",
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_LAUNCH_FILE = _REPO_ROOT / "packages" / "openral_rskill_ros" / "launch" / "deploy_e2e.launch.py"
_LOCATOR_MANIFEST = _REPO_ROOT / "rskills" / "omdet-turbo-locator" / "rskill.yaml"


def _import_launch_module() -> object:
    spec = importlib.util.spec_from_file_location(
        "deploy_e2e_launch_detector_available", _LAUNCH_FILE
    )
    assert spec is not None and spec.loader is not None, f"failed to spec {_LAUNCH_FILE}"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _compose(*, enable_object_detector: bool, locator_manifest: str) -> list[object]:
    """Run the real launch composition for a franka_panda deploy."""
    from launch import LaunchContext
    from launch.actions import DeclareLaunchArgument

    module = _import_launch_module()
    ctx = LaunchContext()
    cfg = ctx.launch_configurations
    cfg["robot_yaml"] = str(_REPO_ROOT / "robots" / "franka_panda" / "robot.yaml")
    cfg["hal_package"] = "openral_hal_node"
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
    cfg["enable_dashboard"] = "false"
    cfg["enable_reasoner"] = "true"
    cfg["enable_object_detector"] = "true" if enable_object_detector else "false"
    cfg["object_detector_locators"] = locator_manifest
    return list(module.compose_runtime_graph(ctx))  # type: ignore[attr-defined]


def _reasoner_param(entities: list[object], name: str) -> object:
    """Read one parameter's raw value off the composed ``openral_reasoner`` LifecycleNode."""
    from launch_ros.actions import LifecycleNode

    for entity in entities:
        if not isinstance(entity, LifecycleNode):
            continue
        if getattr(entity, "_Node__node_name", None) != "openral_reasoner":
            continue
        # No public parameters accessor before execute() resolves the graph.
        (params_dict,) = entity._Node__parameters
        for key_subs, value in params_dict.items():
            key = "".join(s.text for s in key_subs if hasattr(s, "text"))
            if key == name:
                return value
        pytest.fail(f"openral_reasoner has no {name!r} parameter")
    pytest.fail("no openral_reasoner LifecycleNode found in the composed graph")


def test_continuous_detector_alone_does_not_claim_locate_in_view() -> None:
    """--object-detector with zero locators must not offer a phantom locate_in_view tool."""
    entities = _compose(enable_object_detector=True, locator_manifest="")
    assert _reasoner_param(entities, "detector_available") is False, (
        "detector_available is True with --object-detector alone (no on-demand locator). "
        "The continuous detector node never serves LocateInView under any name "
        "(detector_node_wiring: serve_on_demand=False for CONTINUOUS mode) — this offers the "
        "LLM a tool with no backing service. Reproduced live: every locate_in_view call gets "
        "'not on graph; skipping' and the mission stalls."
    )


def test_an_on_demand_locator_does_grant_locate_in_view() -> None:
    """The positive case: a real on-demand locator manifest does back the tool."""
    entities = _compose(enable_object_detector=False, locator_manifest=str(_LOCATOR_MANIFEST))
    assert _reasoner_param(entities, "detector_available") is True
    assert _reasoner_param(entities, "default_on_demand_detector") is not None


def test_both_legs_together_still_grants_it_correctly() -> None:
    """A continuous detector plus a locator: the locator is what makes this True, not the sum."""
    entities = _compose(enable_object_detector=True, locator_manifest=str(_LOCATOR_MANIFEST))
    assert _reasoner_param(entities, "detector_available") is True
