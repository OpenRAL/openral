# SPDX-License-Identifier: Apache-2.0
"""A hand-run ``deploy_e2e.launch.py hal_mode:=real`` refuses a networked ROS graph.

``openral deploy run`` confines its launch env (``openral_cli._dds_scope``); a
plain ``ros2 launch`` did not, so a robot started by hand ran on SUBNET discovery
where any LAN host could publish ``/openral/safe_action``. The launch now applies
the same decision (``openral_core.ros_graph_scope``) to the environment it runs
under.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import ModuleType

import pytest
import structlog
from openral_core.exceptions import ROSConfigError

_LAUNCH_FILE = Path(__file__).resolve().parent.parent / "launch" / "deploy_e2e.launch.py"
_REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def launch_module() -> ModuleType:
    """Import ``deploy_e2e.launch.py`` by path; skip without a sourced ROS 2 (§1.11)."""
    pytest.importorskip("launch")
    pytest.importorskip("launch_ros")
    pytest.importorskip("openral_foxglove_bringup.topics")
    if not os.environ.get("ROS_DISTRO"):
        pytest.skip("ROS_DISTRO not set — launch_ros requires a sourced ROS 2 install.")
    from _launch_test_common import import_launch_module

    return import_launch_module(_LAUNCH_FILE)


@pytest.mark.parametrize(
    "env",
    [
        {},  # unset range = SUBNET to ROS
        {"ROS_AUTOMATIC_DISCOVERY_RANGE": ""},
        {"ROS_AUTOMATIC_DISCOVERY_RANGE": "SUBNET"},
        {"ROS_AUTOMATIC_DISCOVERY_RANGE": "LOCALHOST", "ROS_STATIC_PEERS": "10.0.0.2"},
        {"ROS_LOCALHOST_ONLY": "0", "ROS_AUTOMATIC_DISCOVERY_RANGE": "SUBNET"},
        {"ROS_AUTOMATIC_DISCOVERY_RANGE": "SUBNET", "OPENRAL_ALLOW_NETWORK_ROS": "true"},
        # Fast DDS discovery server / Zenoh reach other hosts whatever the range says.
        {"ROS_AUTOMATIC_DISCOVERY_RANGE": "LOCALHOST", "ROS_DISCOVERY_SERVER": "10.0.0.9:11811"},
        {"ROS_AUTOMATIC_DISCOVERY_RANGE": "LOCALHOST", "RMW_IMPLEMENTATION": "rmw_zenoh_cpp"},
        {"ROS_LOCALHOST_ONLY": "1", "RMW_IMPLEMENTATION": "rmw_zenoh_cpp"},
    ],
)
def test_a_real_hal_on_a_networked_graph_is_refused(
    launch_module: ModuleType, env: dict[str, str]
) -> None:
    with pytest.raises(ROSConfigError, match="OPENRAL_ALLOW_NETWORK_ROS"):
        launch_module._require_host_confined_real_graph("real", env)


@pytest.mark.parametrize(
    "env",
    [
        {"ROS_AUTOMATIC_DISCOVERY_RANGE": "LOCALHOST"},
        {"ROS_LOCALHOST_ONLY": "1", "ROS_AUTOMATIC_DISCOVERY_RANGE": "SUBNET"},
        {"ROS_AUTOMATIC_DISCOVERY_RANGE": "SUBNET", "OPENRAL_ALLOW_NETWORK_ROS": "1"},
    ],
)
def test_a_host_confined_or_opted_in_real_hal_launches(
    launch_module: ModuleType, env: dict[str, str]
) -> None:
    launch_module._require_host_confined_real_graph("real", env)


def test_a_freshly_sourced_shell_is_refused_and_told_why(launch_module: ModuleType) -> None:
    """``setup.bash`` exports SUBNET when unset; a hand-run launch cannot re-scope its
    own already-started environment, so it refuses and names the cause."""
    with pytest.raises(ROSConfigError, match=r"setup\.bash"):
        launch_module._require_host_confined_real_graph(
            "real", {"ROS_AUTOMATIC_DISCOVERY_RANGE": "SUBNET"}
        )


@pytest.mark.parametrize("hal_mode", ["hardware", "Real", "rael"])
def test_any_non_sim_mode_is_checked(launch_module: ModuleType, hal_mode: str) -> None:
    """Fail closed: the launch composes the hardware path for every non-"sim" value."""
    with pytest.raises(ROSConfigError, match="OPENRAL_ALLOW_NETWORK_ROS"):
        launch_module._require_host_confined_real_graph(hal_mode, {})


def test_discovery_off_is_refused(launch_module: ModuleType) -> None:
    """OFF hides the robot's own nodes from each other: refuse, point at LOCALHOST."""
    with pytest.raises(ROSConfigError, match="use LOCALHOST"):
        launch_module._require_host_confined_real_graph(
            "real", {"ROS_AUTOMATIC_DISCOVERY_RANGE": "off"}
        )


def test_a_sim_is_not_checked(launch_module: ModuleType) -> None:
    """Sims are confined by ``openral deploy sim``; this gate is for real HALs."""
    launch_module._require_host_confined_real_graph(
        "sim", {"ROS_AUTOMATIC_DISCOVERY_RANGE": "SUBNET"}
    )


_HAL_NODE = "openral_hal_scope_test"


def _walk_real_deploy(
    launch_module: ModuleType, started: list[object], *, robot_yaml: Path, hal_params: Path
) -> None:
    """Walk the whole ``deploy_e2e`` description for a ``hal_mode:=real`` deploy.

    Nothing is spawned (``walk_launch``), so this is safe on a host with hardware.
    """
    pytest.importorskip("openral_safety")
    from _launch_test_common import walk_launch
    from launch import LaunchContext

    hal_params.write_text("{}\n")
    context = LaunchContext()
    context.launch_configurations.update(
        {
            "robot_yaml": str(robot_yaml),
            "hal_package": "openral_hal_node",
            "hal_executable": "lifecycle_node.py",
            "hal_node_name": _HAL_NODE,
            "hal_params_file": str(hal_params),
            "hal_mode": "real",
            "enable_dashboard": "false",
        }
    )
    walk_launch(launch_module.generate_launch_description(), context, started)


_FRANKA = _REPO_ROOT / "robots" / "franka_panda" / "robot.yaml"
_SCOPE_VARS = (
    "ROS_LOCALHOST_ONLY",
    "ROS_STATIC_PEERS",
    "ROS_DISCOVERY_SERVER",
    "RMW_IMPLEMENTATION",
    "CYCLONEDDS_URI",
    "FASTRTPS_DEFAULT_PROFILES_FILE",
    "FASTDDS_DEFAULT_PROFILES_FILE",
    "OPENRAL_ALLOW_NETWORK_ROS",
)


@pytest.fixture
def scope_env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    """No inherited variable that widens or narrows the ROS scope."""
    for name in _SCOPE_VARS:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def test_a_refused_launch_starts_no_process(
    launch_module: ModuleType, scope_env: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The whole launch, not the gate alone: a refused scope reaches no node or process (#558).

    Fails if the gate moves after anything that starts a process — later in the
    returned graph, into a timer, or out of the launch.
    """
    scope_env.setenv("ROS_AUTOMATIC_DISCOVERY_RANGE", "SUBNET")
    started: list[object] = []
    with pytest.raises(ROSConfigError, match="OPENRAL_ALLOW_NETWORK_ROS"):
        _walk_real_deploy(
            launch_module, started, robot_yaml=_FRANKA, hal_params=tmp_path / "hal.yaml"
        )
    assert started == []


def test_a_refused_launch_reads_no_manifest(
    launch_module: ModuleType, scope_env: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The refusal comes before any composition work, so it is what the operator sees.

    A launch whose manifest does not exist still answers with the scope refusal: the
    gate runs before ``robot.yaml`` is read, let alone a node composed. Moved later in
    ``compose_runtime_graph``, the manifest (or a missing vendor bringup) fails first
    and hides the security refusal.
    """
    scope_env.setenv("ROS_AUTOMATIC_DISCOVERY_RANGE", "SUBNET")
    started: list[object] = []
    with pytest.raises(ROSConfigError, match="OPENRAL_ALLOW_NETWORK_ROS"):
        _walk_real_deploy(
            launch_module,
            started,
            robot_yaml=tmp_path / "absent" / "robot.yaml",
            hal_params=tmp_path / "hal.yaml",
        )
    assert started == []


def test_the_same_launch_host_confined_starts_the_graph(
    launch_module: ModuleType, scope_env: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Control for the tests above: confined, the same walk reaches the HAL."""
    pytest.importorskip("mujoco")
    scope_env.setenv("ROS_AUTOMATIC_DISCOVERY_RANGE", "LOCALHOST")
    started: list[object] = []
    _walk_real_deploy(launch_module, started, robot_yaml=_FRANKA, hal_params=tmp_path / "hal.yaml")
    assert _HAL_NODE in {str(getattr(entity, "_Node__node_name", "")) for entity in started}


def test_an_opted_in_networked_graph_is_logged_once(launch_module: ModuleType) -> None:
    """#558: the launch gate is the one emitter; ``deploy run``'s scoping logs nothing."""
    dds_scope = pytest.importorskip("openral_cli._dds_scope")
    env = {"ROS_AUTOMATIC_DISCOVERY_RANGE": "SUBNET", "OPENRAL_ALLOW_NETWORK_ROS": "1"}
    with structlog.testing.capture_logs() as logs:
        for _ in range(2):  # orphan reap + launch env, as one deploy run does
            dds_scope.confine_real_scope(dict(env), robot_id="openarm")
        launch_module._require_host_confined_real_graph("real", env)
    warnings = [log for log in logs if log["log_level"] == "warning"]
    assert [log["event"] for log in warnings] == ["deploy_e2e.real_robot_graph_networked"]
    assert "ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET" in warnings[0]["scope"]
