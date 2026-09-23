"""The deploy graph must spawn the three independent E-stop sources.

The defect this pins: ``packages/openral_safety_watchdog`` (deadman +
hardware pendant) and ``packages/openral_human_estop`` (the human forwarder)
shipped as complete, tested, "green" lifecycle nodes — and **no launch file
anywhere in the repo ever started them**. Both READMEs claimed they were
"brought up automatically by the ``openral deploy sim`` / ``deploy run``
graph"; ``deploy_e2e.launch.py``, the single launch behind both commands,
mentioned none of them. So the entire defense-in-depth E-stop layer — the one
CLAUDE.md §3 requires so a Python crash cannot leave motors energised — was
documentation, and nothing in the test suite noticed.

A unit-level assertion on the composed graph is the right shape for that:
the failure was structural (a node absent from the launch), not behavioural,
and it has to hold for **both** ``hal_mode`` values.

Real ``LaunchContext`` + real ``compose_runtime_graph`` per CLAUDE.md §1.11 —
same harness as ``test_deploy_e2e_prompt_router_autostart.py``.
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
_AUTOSTART_SCRIPT = str(_REPO_ROOT / "tools" / "lifecycle_autostart.py")

_ESTOP_NODES = {
    "openral_deadman_watchdog": ("openral_safety_watchdog", "deadman_watchdog_node.py"),
    "openral_hardware_estop": ("openral_safety_watchdog", "hardware_estop_node.py"),
    "openral_human_estop_forwarder": ("openral_human_estop", "forwarder_node.py"),
}


def _import_launch_module() -> Any:
    spec = importlib.util.spec_from_file_location("deploy_e2e_launch_estop_sources", _LAUNCH_FILE)
    assert spec is not None and spec.loader is not None, f"failed to spec {_LAUNCH_FILE}"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _compose(hal_mode: str) -> list[Any]:
    """Run the real launch composition for a plain franka_panda deploy."""
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
    for entity in module.generate_launch_description().entities:
        if isinstance(entity, DeclareLaunchArgument):
            entity.execute(ctx)
    cfg["hal_mode"] = hal_mode
    cfg["enable_slam"] = "false"
    cfg["enable_nav2"] = "false"
    cfg["enable_octomap"] = "false"
    cfg["enable_object_detector"] = "false"
    cfg["enable_dashboard"] = "false"
    cfg["enable_reasoner"] = "true"
    return list(module.compose_runtime_graph(ctx))


def _text(value: Any) -> str:
    """Resolve a launch substitution (or an already-plain value) to text."""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    return "".join(part.text for part in value if hasattr(part, "text"))


def _lifecycle_nodes(entities: list[Any]) -> dict[str, Any]:
    """``{node name: LifecycleNode}`` for every lifecycle node in the graph.

    ``LifecycleNode.node_name`` raises before ``execute()``, so read the
    name-mangled constructor field — the same accessor
    ``test_deploy_e2e_detector_available.py`` uses on the composed graph.
    """
    from launch_ros.actions import LifecycleNode

    found: dict[str, Any] = {}
    for entity in entities:
        if isinstance(entity, LifecycleNode):
            found[str(getattr(entity, "_Node__node_name", ""))] = entity
    return found


def _node_field(node: Any, attr: str) -> str:
    """Resolve a mangled ``Node`` constructor field back to plain text."""
    return _text(getattr(node, f"_Node__{attr}"))


def _params(node: Any) -> dict[str, Any]:
    """The literal parameter dict the launch passes to ``node``.

    launch_ros normalises every parameter value into a YAML text
    substitution before ``execute()`` (an empty string arrives as ``"\'\'\\n"``),
    so the text is round-tripped back through ``yaml.safe_load``.
    """
    import yaml

    resolved: dict[str, Any] = {}
    for block in node._Node__parameters:  # reason: no public accessor pre-execute
        if not hasattr(block, "items"):
            continue
        for key, value in block.items():
            resolved[_text(key).strip()] = yaml.safe_load(_text(value))
    return resolved


def _autostart_invocations(entities: list[Any]) -> dict[str, list[str]]:
    """``{--node target: full argv}`` for every ``lifecycle_autostart.py`` call."""
    from launch.actions import ExecuteProcess

    found: dict[str, list[str]] = {}
    for entity in entities:
        if not isinstance(entity, ExecuteProcess):
            continue
        parts = [sub.text for group in entity.cmd for sub in group if hasattr(sub, "text")]
        if not parts or _AUTOSTART_SCRIPT not in parts:
            continue
        found[parts[parts.index("--node") + 1]] = parts
    return found


def _autostart_script_targets(entities: list[Any]) -> list[str]:
    """The ``--node`` argument of every ``lifecycle_autostart.py`` invocation."""
    return list(_autostart_invocations(entities))


@pytest.mark.parametrize("hal_mode", ["sim", "real"])
def test_all_three_estop_sources_are_in_the_graph(hal_mode: str) -> None:
    """Deadman + hardware pendant + human forwarder, on both deploy paths."""
    nodes = _lifecycle_nodes(_compose(hal_mode))
    for node_name, (package, executable) in _ESTOP_NODES.items():
        assert node_name in nodes, (
            f"{node_name} is not in the hal_mode={hal_mode} deploy graph — the "
            f"defense-in-depth E-stop source from {package} is not launched, so a crash in "
            "the in-band safety layer has nothing independent behind it (CLAUDE.md §3)."
        )
        node = nodes[node_name]
        assert _node_field(node, "package") == package
        assert _node_field(node, "node_executable") == executable


@pytest.mark.parametrize("hal_mode", ["sim", "real"])
def test_estop_sources_autostart_via_the_poll_based_script(hal_mode: str) -> None:
    """A dropped ACTIVATE would leave an E-stop source silently in INACTIVE.

    Same Jazzy ``lifecycle_event_manager`` race the safety kernel, reasoner,
    prompt_router and HAL are already routed around: for these nodes it would
    mean an E-stop source that exists on the graph and can never fire.

    The pendant is the deliberate exception — it is only driven when a device
    is actually declared (see
    ``test_the_pendant_is_not_autostarted_when_no_device_is_declared``).
    """
    targets = _autostart_script_targets(_compose(hal_mode))
    for node_name in ("openral_deadman_watchdog", "openral_human_estop_forwarder"):
        assert f"/{node_name}" in targets, (
            f"/{node_name} is not driven by tools/lifecycle_autostart.py on hal_mode="
            f"{hal_mode}; a dropped ACTIVATE would leave it inert with no diagnostic."
        )


@pytest.mark.parametrize("hal_mode", ["sim", "real"])
def test_the_deadman_autostart_is_required_and_shuts_the_graph_down_on_failure(
    hal_mode: str,
) -> None:
    """A deploy that cannot arm its only independent watchdog must not run.

    Two halves, and both are needed. ``--required`` makes an absent node a
    non-zero exit — by default ``lifecycle_autostart.py`` treats "the service
    never appeared" as informational and exits 0, so a colcon overlay built
    without ``openral_safety_watchdog`` would deploy with no out-of-band E-stop
    and one stderr line as evidence. The ``OnProcessExit`` handler is what
    turns that exit code into a refusal to run.
    """
    from launch.actions import ExecuteProcess, RegisterEventHandler
    from launch.event_handlers import OnProcessExit

    entities = _compose(hal_mode)
    argv = _autostart_invocations(entities)["/openral_deadman_watchdog"]
    assert "--required" in argv, (
        "the deadman autostart is not --required, so an absent watchdog exits 0 and the "
        "graph comes up with no independent E-stop source"
    )
    # Find the handler that actually watches the deadman autostart and drive it
    # both ways, through launch's public matches()/handle() API. "Some handler
    # exists" would pass on a handler wired to a different process, or one that
    # logs the failure and lets the graph run on unprotected.
    from launch import LaunchContext
    from launch.actions import Shutdown
    from launch.events.process import ProcessExited

    deadman_action = next(
        e
        for e in entities
        if isinstance(e, ExecuteProcess)
        and "/openral_deadman_watchdog"
        in [sub.text for group in e.cmd for sub in group if hasattr(sub, "text")]
    )

    def _exit_event(returncode: int) -> Any:
        return ProcessExited(
            action=deadman_action,
            name="openral_deadman_autostart",
            cmd=["lifecycle_autostart.py"],
            cwd=None,
            env=None,
            pid=4242,
            returncode=returncode,
        )

    handlers = [
        e.event_handler
        for e in entities
        if isinstance(e, RegisterEventHandler)
        and isinstance(getattr(e, "event_handler", None), OnProcessExit)
    ]
    watching = [h for h in handlers if h.matches(_exit_event(1))]
    assert watching, (
        "no OnProcessExit handler watches the deadman autostart process, so a watchdog that "
        "never activated would leave the graph running with no independent E-stop source"
    )
    handler = watching[0]

    failed = handler.handle(_exit_event(1), LaunchContext()) or []
    assert any(isinstance(a, Shutdown) for a in failed), (
        "a deadman autostart that exited non-zero did not shut the graph down; the deploy "
        "would come up unprotected"
    )

    ok = handler.handle(_exit_event(0), LaunchContext()) or []
    assert not any(isinstance(a, Shutdown) for a in ok), (
        "a successful deadman autostart shut the graph down"
    )


def test_the_pendant_is_not_autostarted_when_no_device_is_declared() -> None:
    """Its expected refusal must not become the noise a real failure hides in.

    With no pendant on the host the node is *supposed* to fail configure.
    Driving it anyway emits that failure on every single deploy, and an
    operator who learns to ignore it also ignores a deadman that failed to
    activate — the two look identical in the log.
    """
    targets = _autostart_script_targets(_compose("sim"))
    assert "/openral_hardware_estop" not in targets, (
        "the pendant is autostarted with no device declared; its routine, expected "
        "configure failure then masks a genuine E-stop source failure"
    )


def test_deadman_is_gated_on_the_runners_own_action_status() -> None:
    """Free-running, the deadman would E-stop every boot within its deadline.

    ``/openral/safe_action`` only carries chunks while an ``ExecuteRskill``
    goal runs, so the gate is what makes launching the watchdog possible at
    all. Losing it does not merely weaken the watchdog — it bricks every
    deploy, and the estop latches.

    It must be the **action status** topic specifically. The advisory
    ``/openral/reward/active_task`` has two publishers, and the reasoner's
    dispatch watchdog clears it precisely when the runner dies after accepting
    a goal — which is the deadman's headline scenario. Gating on that topic
    lets one node silence the watchdog for the case it exists to catch.
    """
    params = _params(_lifecycle_nodes(_compose("sim"))["openral_deadman_watchdog"])
    assert params.get("arm_status_topic") == "/openral/execute_rskill/_action/status", (
        f"deadman arm_status_topic is {params.get('arm_status_topic')!r}, not the runner's "
        "ExecuteRskill action-status topic — an idle deploy publishes no /openral/safe_action, "
        "so an ungated watchdog E-stops the robot seconds after boot, and an advisory gate can "
        "be closed by a second publisher exactly when the runner dies."
    )
    # Not the node's 0.2 s standalone default: one VLA inference separates two
    # consecutive chunks in a real rollout.
    assert float(params["safe_action_deadline_s"]) >= 1.0
    # And the arm -> first-chunk wait must be bounded, or a runner that dies
    # before its first chunk holds the window open forever.
    assert float(params["first_chunk_deadline_s"]) > 0.0


def test_hardware_estop_declares_no_device_by_default() -> None:
    """No pendant driver ships here; the default must not imply one exists."""
    params = _params(_lifecycle_nodes(_compose("sim"))["openral_hardware_estop"])
    assert params["device"] == "", (
        "the deploy graph declares a hardware E-stop device by default — with no vendor "
        "driver in this repo the node would poll a stub that answers 'not pressed' forever, "
        "which is indistinguishable from an armed pendant."
    )
