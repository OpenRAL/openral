# SPDX-License-Identifier: Apache-2.0
"""A hand-run ``deploy_e2e.launch.py hal_mode:=real`` refuses a networked ROS graph.

``openral deploy run`` confines its launch env (``openral_cli._dds_scope``); a
plain ``ros2 launch`` did not, so a robot started by hand ran on SUBNET discovery
where any LAN host could publish ``/openral/safe_action``. The launch now applies
the same decision (``openral_core.ros_graph_scope``) to the environment it runs
under.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from types import ModuleType

import pytest
from openral_core.exceptions import ROSConfigError

_LAUNCH_FILE = Path(__file__).resolve().parent.parent / "launch" / "deploy_e2e.launch.py"


@pytest.fixture(scope="module")
def launch_module() -> ModuleType:
    """Import ``deploy_e2e.launch.py`` by path; skip without a sourced ROS 2 (§1.11)."""
    pytest.importorskip("launch")
    pytest.importorskip("launch_ros")
    pytest.importorskip("openral_foxglove_bringup.topics")
    if not os.environ.get("ROS_DISTRO"):
        pytest.skip("ROS_DISTRO not set — launch_ros requires a sourced ROS 2 install.")
    spec = importlib.util.spec_from_file_location("openral_deploy_e2e_scope", _LAUNCH_FILE)
    if spec is None or spec.loader is None:
        pytest.fail(f"failed to build module spec for {_LAUNCH_FILE}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize(
    "env",
    [
        {},  # unset range = SUBNET to ROS
        {"ROS_AUTOMATIC_DISCOVERY_RANGE": ""},
        {"ROS_AUTOMATIC_DISCOVERY_RANGE": "SUBNET"},
        {"ROS_AUTOMATIC_DISCOVERY_RANGE": "LOCALHOST", "ROS_STATIC_PEERS": "10.0.0.2"},
        {"ROS_LOCALHOST_ONLY": "0", "ROS_AUTOMATIC_DISCOVERY_RANGE": "SUBNET"},
        {"ROS_AUTOMATIC_DISCOVERY_RANGE": "SUBNET", "OPENRAL_ALLOW_NETWORK_ROS": "true"},
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
        {"ROS_AUTOMATIC_DISCOVERY_RANGE": "off"},
        {"ROS_LOCALHOST_ONLY": "1", "ROS_AUTOMATIC_DISCOVERY_RANGE": "SUBNET"},
        {"ROS_AUTOMATIC_DISCOVERY_RANGE": "SUBNET", "OPENRAL_ALLOW_NETWORK_ROS": "1"},
    ],
)
def test_a_host_confined_or_opted_in_real_hal_launches(
    launch_module: ModuleType, env: dict[str, str]
) -> None:
    launch_module._require_host_confined_real_graph("real", env)


@pytest.mark.parametrize("hal_mode", ["hardware", "Real", "rael"])
def test_any_non_sim_mode_is_checked(launch_module: ModuleType, hal_mode: str) -> None:
    """Fail closed: the launch composes the hardware path for every non-"sim" value."""
    with pytest.raises(ROSConfigError, match="OPENRAL_ALLOW_NETWORK_ROS"):
        launch_module._require_host_confined_real_graph(hal_mode, {})


def test_a_sim_is_not_checked(launch_module: ModuleType) -> None:
    """Sims are confined by ``openral deploy sim``; this gate is for real HALs."""
    launch_module._require_host_confined_real_graph(
        "sim", {"ROS_AUTOMATIC_DISCOVERY_RANGE": "SUBNET"}
    )


def test_the_gate_runs_before_any_node_is_composed(launch_module: ModuleType) -> None:
    """Wired right after ``hal_mode`` resolves in ``compose_runtime_graph``."""
    source = _LAUNCH_FILE.read_text()
    resolve = source.index('hal_mode = LaunchConfiguration("hal_mode").perform(context)')
    gate = source.index("_require_host_confined_real_graph(hal_mode, os.environ)")
    assert 0 < gate - resolve < 200
