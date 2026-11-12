# SPDX-License-Identifier: Apache-2.0
"""Every shipped vendor ``real_bringup`` launch refuses a networked ROS graph (issue #556).

``deploy_e2e.launch.py hal_mode:=real`` refused a networked graph, but the vendor bringups
it includes are also run by hand, and a hand-run one on ROS's default range (``SUBNET``,
which ``setup.bash`` exports) joined the LAN graph. Each launch now runs
``openral_core.ros_graph_scope.require_host_confined_real_bringup`` before composing any
node. These tests run the real launch files through ``ros2 launch`` under a networked
environment and assert the refusal comes before any process starts; the in-tree launches
are discovered from ``robots/*/robot.yaml`` so a new robot's bringup is covered too.

Hardware can never start here, even if the gate regresses: the launch runs with an empty
``AMENT_PREFIX_PATH`` (no vendor package resolves), with CAN/NIC arguments naming no
interface, in its own process group that is killed whole on timeout.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

_REPO = Path(__file__).resolve().parents[2]
#: Variables that make an env host-confined, opted in, or networked by another branch
#: of ``networked_ros_scope`` than the discovery range under test.
_SCOPE_VARS = frozenset(
    {
        "OPENRAL_ALLOW_NETWORK_ROS",
        "ROS_LOCALHOST_ONLY",
        "ROS_AUTOMATIC_DISCOVERY_RANGE",
        "ROS_STATIC_PEERS",
        "ROS_DISCOVERY_SERVER",
        "RMW_IMPLEMENTATION",
        "CYCLONEDDS_URI",
        "FASTRTPS_DEFAULT_PROFILES_FILE",
        "FASTDDS_DEFAULT_PROFILES_FILE",
    }
)
#: Arguments that point every shipped bringup at hardware that does not exist.
_NO_HARDWARE_ARGS = (
    "left_can_interface:=openral_test_none",
    "right_can_interface:=openral_test_none",
    "network_interface:=openral_test_none",
)


def _in_tree_real_bringups() -> list[Path]:
    launches = set()
    for manifest in sorted((_REPO / "robots").glob("*/robot.yaml")):
        ref = (yaml.safe_load(manifest.read_text()).get("hal") or {}).get("real_bringup")
        if ref:
            package, launch_file = ref.split(":")
            launch = _REPO / "packages" / package / "launch" / launch_file
            # An upstream vendor launch cannot carry our gate; deploy_e2e's covers it.
            if (_REPO / "packages" / package).is_dir():
                launches.add(launch)
    return sorted(launches)


_LAUNCHES = _in_tree_real_bringups()


def test_in_tree_bringups_are_found() -> None:
    assert _LAUNCHES, "no robots/*/robot.yaml declares an in-tree hal.real_bringup"
    for launch in _LAUNCHES:
        assert launch.is_file(), launch


@pytest.mark.parametrize(
    "scope_env",
    [
        {"ROS_AUTOMATIC_DISCOVERY_RANGE": "SUBNET"},  # what setup.bash exports
        {},  # absent = SUBNET to rcl
        {"ROS_AUTOMATIC_DISCOVERY_RANGE": ""},
        {"ROS_AUTOMATIC_DISCOVERY_RANGE": "LOCALHOST", "ROS_STATIC_PEERS": "10.0.0.2"},
    ],
    ids=["subnet", "unset", "empty", "static-peers"],
)
@pytest.mark.parametrize("launch", _LAUNCHES, ids=lambda p: p.parent.parent.name)
def test_a_networked_real_bringup_is_refused_before_any_node(
    launch: Path, scope_env: dict[str, str], tmp_path: Path
) -> None:
    ros2 = shutil.which("ros2")
    if not os.environ.get("ROS_DISTRO") or ros2 is None:
        pytest.skip("needs a sourced ROS 2 install (ros2 launch)")
    pytest.importorskip("launch_ros")
    env = {k: v for k, v in os.environ.items() if k not in _SCOPE_VARS}
    env.update(scope_env, ROS_DOMAIN_ID="156", AMENT_PREFIX_PATH=str(tmp_path))
    # The launch runs under this interpreter, so it imports the same openral_core.
    env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p)
    with subprocess.Popen(
        [sys.executable, ros2, "launch", str(launch), *_NO_HARDWARE_ARGS],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    ) as proc:
        try:
            output, _ = proc.communicate(timeout=120)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            output, _ = proc.communicate()
            pytest.fail(f"launch did not exit within 120 s:\n{output}")
    assert "refusing to start a real robot on a ROS graph that reaches other hosts" in output, (
        output
    )
    assert "ROS_AUTOMATIC_DISCOVERY_RANGE=" in output, output  # the range branch refused
    assert "process started" not in output, output
