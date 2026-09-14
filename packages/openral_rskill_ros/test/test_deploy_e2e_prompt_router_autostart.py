"""``prompt_router`` must autostart via the poll-based script, not the racy handlers.

Reproduced live on a real graph: with ``--object-detector`` added
(``openral deploy sim --config scenes/deploy/libero_pnp.yaml --object-detector
--initial-task "..."``), the reasoner reached ACTIVE via
``tools/lifecycle_autostart.py`` and started ticking at 0.2 Hz, but
``openral_prompt_router`` — still wired through ``_autostart_lifecycle``'s
``OnProcessStart``/``OnStateTransition`` launch_ros event handlers — silently
missed its own ``transition_event`` under the heavier graph and stuck in
INACTIVE forever. The CLI accepted ``--initial-task``, threaded it into
``initial_task_prompt``, and the prompt router held it in its
``startup_prompt`` parameter but never reached ``on_activate`` to publish it,
so the reasoner ticked against an empty mission with no diagnostic anywhere
naming the stall — the exact race already documented and fixed for the
reasoner, HAL, and slam_toolbox above this in the same file.

Real ``LaunchContext`` + real ``compose_runtime_graph`` per CLAUDE.md §1.11 —
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
pytest.importorskip("mujoco")

pytestmark = pytest.mark.skipif(
    not os.environ.get("ROS_DISTRO"),
    reason="ROS_DISTRO not set — these tests require a sourced ROS 2 installation.",
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_LAUNCH_FILE = _REPO_ROOT / "packages" / "openral_rskill_ros" / "launch" / "deploy_e2e.launch.py"
_AUTOSTART_SCRIPT = str(_REPO_ROOT / "tools" / "lifecycle_autostart.py")


def _import_launch_module() -> object:
    spec = importlib.util.spec_from_file_location(
        "deploy_e2e_launch_prompt_router_autostart", _LAUNCH_FILE
    )
    assert spec is not None and spec.loader is not None, f"failed to spec {_LAUNCH_FILE}"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _compose() -> list[object]:
    """Run the real launch composition for a plain franka_panda deploy — no scene overrides."""
    from launch import LaunchContext
    from launch.actions import DeclareLaunchArgument

    module = _import_launch_module()
    ctx = LaunchContext()
    cfg = ctx.launch_configurations
    cfg["robot_yaml"] = str(_REPO_ROOT / "robots" / "franka_panda" / "robot.yaml")
    cfg["hal_package"] = "openral_hal_franka"
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
    cfg["enable_reasoner"] = "true"
    return list(module.compose_runtime_graph(ctx))  # type: ignore[attr-defined]


def _autostart_script_targets(entities: list[object]) -> list[str]:
    """The ``--node`` argument of every ``lifecycle_autostart.py`` invocation in the graph."""
    from launch.actions import ExecuteProcess

    targets: list[str] = []
    for entity in entities:
        if not isinstance(entity, ExecuteProcess):
            continue
        parts = [sub.text for group in entity.cmd for sub in group if hasattr(sub, "text")]
        if not parts or _AUTOSTART_SCRIPT not in parts:
            continue
        targets.append(parts[parts.index("--node") + 1])
    return targets


def test_prompt_router_autostarts_via_the_poll_based_script() -> None:
    """Not the launch_ros event handlers that dropped its ACTIVATE on a heavier graph."""
    targets = _autostart_script_targets(_compose())
    assert "/openral_prompt_router" in targets, (
        "openral_prompt_router is not driven by tools/lifecycle_autostart.py — it is back on "
        "_autostart_lifecycle's racy OnProcessStart/OnStateTransition handlers, which silently "
        "drop ACTIVATE under a heavy graph (reproduced live with --object-detector) and leave "
        "--initial-task / /openral/prompt undelivered with no diagnostic."
    )


def test_reasoner_still_autostarts_via_the_poll_based_script() -> None:
    """Companion pin: the reasoner's own poll-based autostart (fixed first) must not regress."""
    targets = _autostart_script_targets(_compose())
    assert "/openral_reasoner" in targets
