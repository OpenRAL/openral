"""Live ROS coverage for issue #664: sims on adjacent ``ROS_DOMAIN_ID`` values stay apart.

The deploy Fast DDS profile widens localhost discovery with ``maxInitialPeersRange``.
A participant's unicast ports are ``7400 + 250 * domain + 10 + 2 * id`` (+1), so
participant id 125 of domain N is participant id 0 of domain N+1. At the 128 the
profile shipped with, a sim on domain N announced itself to the first participants
of domain N+1, the graphs merged and the second ``openral deploy sim`` was refused
as an occupied graph.

Two real ``rclpy`` nodes, one per domain, under the real sim scope
(``confine_sim_scope``) and the shipped profile; each domain is then read with the
occupancy guard's own probe (``_dds_scope._scan_graph``). The same rig under a copy
of the profile at 128 must show the leak, or the first half proves nothing.

Nodes only — nothing is published, so a robot that happens to live on one of the
two domains sees two extra node names and no data.

Gated on ``OPENRAL_TEST_ROS_LIVE=1``, listed in ``scripts/ros_live_tests.sh``.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
from openral_cli import _dds_scope
from openral_cli.deploy_sim import (
    FASTDDS_LARGE_DATA_PROFILE,
    FASTDDS_PROFILES_ENV,
    apply_fastdds_large_data_profile,
)

_LIVE_ROS = bool(os.getenv("OPENRAL_TEST_ROS_LIVE"))
pytestmark = pytest.mark.skipif(
    not _LIVE_ROS,
    reason="live rclpy graph — set OPENRAL_TEST_ROS_LIVE=1 and source install/setup.bash first.",
)

#: Adjacent, in the robot hash's free range (10-49 minus 42), away from the test domains.
#: ``OPENRAL_TEST_ADJACENT_DOMAIN_LOW`` moves the pair on a host where 46/47 are in use.
_LOW = os.getenv("OPENRAL_TEST_ADJACENT_DOMAIN_LOW", "46")
_HIGH = str(int(_LOW) + 1)

_NODE = """
import sys, time, rclpy
rclpy.init()
node = rclpy.create_node(sys.argv[1])
print("up", flush=True)
deadline = time.monotonic() + 60.0
while time.monotonic() < deadline:
    rclpy.spin_once(node, timeout_sec=0.2)
"""


def _sim_env(domain: str, profile: Path | None) -> dict[str, str]:
    """What ``openral deploy sim`` launches under, on ``domain``, on Fast DDS."""
    env = os.environ.copy()
    for name in ("ROS_LOCALHOST_ONLY", "ROS_STATIC_PEERS", "ROS_DISCOVERY_SERVER"):
        env.pop(name, None)
    env["ROS_AUTOMATIC_DISCOVERY_RANGE"] = "SUBNET"  # what setup.bash exports
    env["ROS_DOMAIN_ID"] = domain
    env["RMW_IMPLEMENTATION"] = "rmw_fastrtps_cpp"  # the profile is Fast DDS's
    env[FASTDDS_PROFILES_ENV] = "" if profile is None else str(profile)
    _dds_scope.confine_sim_scope(env)
    apply_fastdds_large_data_profile(env)
    return env


def _nodes_seen(profile: Path | None) -> tuple[list[str], list[str]]:
    """Node names the guard's probe reads on ``_LOW`` and on ``_HIGH``, one node up on each."""
    low_env, high_env = _sim_env(_LOW, profile), _sim_env(_HIGH, profile)
    procs: list[subprocess.Popen[str]] = []
    try:
        # The higher domain first: its node takes participant id 0 there, the port a
        # too-wide range on the lower domain reaches.
        for env, name in ((high_env, "openral_test_domain_high"), (low_env, "openral_test_low")):
            proc = subprocess.Popen(
                [sys.executable, "-c", _NODE, name], env=env, stdout=subprocess.PIPE, text=True
            )
            procs.append(proc)
            assert proc.stdout is not None
            assert proc.stdout.readline().strip() == "up", f"{name} did not start"
        time.sleep(1.0)
        seen = []
        for env in (low_env, high_env):
            found = _dds_scope._scan_graph(env)
            assert isinstance(found, dict), f"graph probe failed: {found!r}"
            seen.append([str(n) for n in found["nodes"]])
        return seen[0], seen[1]
    finally:
        for proc in procs:
            proc.kill()
            proc.wait()


def test_shipped_profile_keeps_adjacent_domains_apart(tmp_path: Path) -> None:
    low, high = _nodes_seen(None)
    assert low == ["/openral_test_low"]
    assert high == ["/openral_test_domain_high"]

    # The rig can see a leak: the range the profile shipped with before #664.
    wide = tmp_path / "wide.xml"
    text = FASTDDS_LARGE_DATA_PROFILE.read_text()
    assert text.count("<maxInitialPeersRange>120<") == 1
    wide.write_text(text.replace("<maxInitialPeersRange>120<", "<maxInitialPeersRange>128<"))
    low, high = _nodes_seen(wide)
    assert "/openral_test_domain_high" in low or "/openral_test_low" in high
