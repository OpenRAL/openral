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

The nodes carry no publisher or subscriber of their own (only what every ``rclpy``
node has: ``/rosout``, ``/parameter_events``, its parameter services), discovery is
host-only, and the test skips when either domain already has a node on it.

Gated on ``OPENRAL_TEST_ROS_LIVE=1``, listed in ``scripts/ros_live_tests.sh``.
"""

from __future__ import annotations

import os
import re
import select
import shutil
import signal
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

from tests.integration._ros_pkg import ros_pkg_available

_LIVE_ROS = bool(os.getenv("OPENRAL_TEST_ROS_LIVE"))
pytestmark = pytest.mark.skipif(
    not _LIVE_ROS,
    reason="live rclpy graph — set OPENRAL_TEST_ROS_LIVE=1 and source install/setup.bash first.",
)

#: The lower of the two adjacent domains. 33 and 34 are in the robot hash's free range
#: (10-49 minus 42) and no in-tree robot's derived domain. On a shared host
#: ``OPENRAL_TEST_ADJACENT_DOMAIN_LOW`` moves the pair (read in the test, not at import).
_DEFAULT_LOW = "33"
_LOW_ENV = "OPENRAL_TEST_ADJACENT_DOMAIN_LOW"

#: Two nodes and the probe each map the profile's 16 MiB segment.
_SHM_NEEDED_BYTES = 3 * 16 * 1024 * 1024
#: Probes of each domain before the 128 control is taken as "no leak seen" (loaded CI runners).
_CONTROL_PROBES = 3
_READY_TIMEOUT_S = 30.0

_NODE = """
import sys, time, rclpy
from rclpy.executors import ExternalShutdownException
rclpy.init()
node = rclpy.create_node(sys.argv[1])
print("up", flush=True)
deadline = time.monotonic() + 180.0
try:
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.2)
except (KeyboardInterrupt, ExternalShutdownException):
    pass
node.destroy_node()
rclpy.try_shutdown()
"""


def _sim_env(domain: str, profile: Path | None) -> dict[str, str]:
    """What ``openral deploy sim`` launches under, on ``domain``, on Fast DDS."""
    env = os.environ.copy()
    for name in (
        "ROS_LOCALHOST_ONLY",
        "ROS_STATIC_PEERS",
        "ROS_DISCOVERY_SERVER",
        "OPENRAL_ALLOW_NETWORK_ROS",  # with it, SUBNET below would stay SUBNET
    ):
        env.pop(name, None)
    env["ROS_AUTOMATIC_DISCOVERY_RANGE"] = "SUBNET"  # what setup.bash exports
    env["ROS_DOMAIN_ID"] = domain
    env["RMW_IMPLEMENTATION"] = "rmw_fastrtps_cpp"  # the profile is Fast DDS's
    env[FASTDDS_PROFILES_ENV] = "" if profile is None else str(profile)
    _dds_scope.confine_sim_scope(env)
    assert env["ROS_AUTOMATIC_DISCOVERY_RANGE"] == "LOCALHOST"
    apply_fastdds_large_data_profile(env)
    return env


def _nodes_on(env: dict[str, str]) -> list[str]:
    found = _dds_scope._scan_graph(env)
    assert isinstance(found, dict), f"graph probe failed: {found!r}"
    return [str(n) for n in found["nodes"]]


def _nodes_seen(
    low: str, high: str, low_profile: Path | None, *, probes: int = 1
) -> tuple[list[str], list[str]]:
    """Node names the guard's probe reads on each domain, with one node up on each.

    Only the lower domain takes ``low_profile``: the higher one keeps the shipped
    profile, so nothing here reaches past ``high``. With ``probes`` above 1 the
    domains are re-read until a foreign node shows or the probes run out.
    """
    low_env, high_env = _sim_env(low, low_profile), _sim_env(high, None)
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
            ready, _, _ = select.select([proc.stdout], [], [], _READY_TIMEOUT_S)
            assert ready, f"{name} did not start within {_READY_TIMEOUT_S} s"
            assert proc.stdout.readline().strip() == "up", f"{name} did not start"
        time.sleep(1.0)
        for _ in range(probes):
            on_low, on_high = _nodes_on(low_env), _nodes_on(high_env)
            if len(on_low) + len(on_high) > len(procs):
                break
        return on_low, on_high
    finally:
        # SIGINT first: a killed Fast DDS participant leaves its /dev/shm segment behind.
        for proc in procs:
            proc.send_signal(signal.SIGINT)
        for proc in procs:
            try:
                proc.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()


def test_shipped_profile_keeps_adjacent_domains_apart(tmp_path: Path) -> None:
    if not ros_pkg_available("rmw_fastrtps_cpp"):
        pytest.skip("rmw_fastrtps_cpp is not installed; the profile under test is Fast DDS's")
    if shutil.disk_usage("/dev/shm").free < _SHM_NEEDED_BYTES:
        pytest.skip("/dev/shm too small for three Fast DDS participants with 16 MiB segments")
    low = os.getenv(_LOW_ENV, "").strip() or _DEFAULT_LOW
    high = str(int(low) + 1)
    for domain in (low, high):
        if occupants := _nodes_on(_sim_env(domain, None)):
            pytest.skip(f"ROS_DOMAIN_ID={domain} is in use ({occupants[:3]}); set {_LOW_ENV}")

    on_low, on_high = _nodes_seen(low, high, None)
    assert on_low == ["/openral_test_low"]
    assert on_high == ["/openral_test_domain_high"]

    # The rig can see a leak: the range the profile shipped with before #664.
    wide = tmp_path / "wide.xml"
    text, replaced = re.subn(
        r"(<maxInitialPeersRange>)\d+", r"\g<1>128", FASTDDS_LARGE_DATA_PROFILE.read_text()
    )
    assert replaced == 1
    wide.write_text(text)
    on_low, on_high = _nodes_seen(low, high, wide, probes=_CONTROL_PROBES)
    assert "/openral_test_domain_high" in on_low or "/openral_test_low" in on_high
