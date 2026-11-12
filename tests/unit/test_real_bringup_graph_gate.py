# SPDX-License-Identifier: Apache-2.0
"""Every shipped vendor ``real_bringup`` launch refuses a networked ROS graph (issue #556).

``deploy_e2e.launch.py hal_mode:=real`` refused a networked graph, but the vendor bringups
it includes are also run by hand, and a hand-run one on ROS's default range (``SUBNET``,
which ``setup.bash`` exports) joined the LAN graph. Each launch now runs
``openral_core.ros_graph_scope.require_host_confined_real_bringup`` before composing any
node. These tests run the real launch files through ``ros2 launch`` under a networked
environment and assert the refusal comes before any process starts; the launches are
discovered from ``robots/*/robot.yaml`` so a new robot's bringup is covered without edits.
Only the refusing path is run end to end: a passing gate would start hardware controllers.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from openral_core.ros_graph_scope import require_host_confined_real_bringup

_REPO = Path(__file__).resolve().parents[2]
#: Variables that would make a networked env host-confined or opted in.
_SCOPE_VARS = frozenset(
    {"OPENRAL_ALLOW_NETWORK_ROS", "ROS_LOCALHOST_ONLY", "ROS_STATIC_PEERS", "ROS_DISCOVERY_SERVER"}
)


def _shipped_real_bringups() -> list[Path]:
    launches = set()
    for manifest in sorted((_REPO / "robots").glob("*/robot.yaml")):
        ref = (yaml.safe_load(manifest.read_text()).get("hal") or {}).get("real_bringup")
        if ref:
            package, launch_file = ref.split(":")
            launches.add(_REPO / "packages" / package / "launch" / launch_file)
    return sorted(launches)


_LAUNCHES = _shipped_real_bringups()


def test_every_manifest_bringup_is_in_tree() -> None:
    assert _LAUNCHES, "no robots/*/robot.yaml declares hal.real_bringup"
    for launch in _LAUNCHES:
        assert launch.is_file(), launch


@pytest.mark.parametrize(
    "scope_env",
    [
        {"ROS_AUTOMATIC_DISCOVERY_RANGE": "SUBNET"},  # what setup.bash exports
        {"ROS_AUTOMATIC_DISCOVERY_RANGE": ""},  # unset = SUBNET to rcl
        {"ROS_AUTOMATIC_DISCOVERY_RANGE": "LOCALHOST", "ROS_STATIC_PEERS": "10.0.0.2"},
    ],
    ids=["subnet", "unset", "static-peers"],
)
@pytest.mark.parametrize("launch", _LAUNCHES, ids=lambda p: p.parent.parent.name)
def test_a_networked_real_bringup_is_refused_before_any_node(
    launch: Path, scope_env: dict[str, str]
) -> None:
    ros2 = shutil.which("ros2")
    if not os.environ.get("ROS_DISTRO") or ros2 is None:
        pytest.skip("needs a sourced ROS 2 install (ros2 launch)")
    pytest.importorskip("launch_ros")
    env = {k: v for k, v in os.environ.items() if k not in _SCOPE_VARS}
    env.update(scope_env, ROS_DOMAIN_ID="156")
    # The launch runs under this interpreter, so it imports the same openral_core.
    env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p)
    result = subprocess.run(
        [sys.executable, ros2, "launch", str(launch)],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    output = result.stdout + result.stderr
    assert "refusing to start a real robot on a ROS graph that reaches other hosts" in output, (
        output
    )
    assert "process started" not in output, output


def test_the_gate_passes_a_confined_graph_and_warns_on_an_opted_in_one(
    capsys: pytest.CaptureFixture[str],
) -> None:
    require_host_confined_real_bringup("bringup", {"ROS_AUTOMATIC_DISCOVERY_RANGE": "LOCALHOST"})
    assert capsys.readouterr().out == ""
    require_host_confined_real_bringup(
        "bringup", {"ROS_AUTOMATIC_DISCOVERY_RANGE": "SUBNET", "OPENRAL_ALLOW_NETWORK_ROS": "1"}
    )
    assert "[bringup] WARNING: real robot on a networked ROS graph" in capsys.readouterr().out
