"""A live Fast-DDS participant survives the ``openral deploy sim`` shm clean.

Regression for Thor 2026-09-24: a ZED driver started before ``deploy sim`` kept
running but published 0 Hz because the clean unlinked its ``/dev/shm``
segments. Real rclpy publisher/subscriber processes on a private domain;
skipped when rclpy is not importable (source ``/opt/ros/jazzy/setup.bash``).
"""

from __future__ import annotations

import os
import select
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
from openral_cli.deploy_sim import _clean_stale_fastrtps_shm

_PUB = r"""
import rclpy
from std_msgs.msg import String
rclpy.init()
node = rclpy.create_node("shm_purge_probe_pub")
pub = node.create_publisher(String, "shm_purge_probe", 10)
node.create_timer(0.1, lambda: pub.publish(String(data="alive")))
print("ready", flush=True)
rclpy.spin(node)
"""

_SUB = r"""
import rclpy
from std_msgs.msg import String
rclpy.init()
node = rclpy.create_node("shm_purge_probe_sub")
got = []
node.create_subscription(String, "shm_purge_probe", lambda m: got.append(m.data), 10)
while not got:
    rclpy.spin_once(node, timeout_sec=0.5)
print(got[0], flush=True)
"""


def _env() -> dict[str, str]:
    env = dict(os.environ)
    env["ROS_DOMAIN_ID"] = str(150 + os.getpid() % 80)
    env["ROS_AUTOMATIC_DISCOVERY_RANGE"] = "LOCALHOST"
    env["RMW_IMPLEMENTATION"] = "rmw_fastrtps_cpp"
    return env


def _rclpy_available() -> bool:
    probe = subprocess.run(
        [sys.executable, "-c", "import rclpy, std_msgs.msg"],
        capture_output=True,
        env=_env(),
        check=False,
        timeout=60,
    )
    return probe.returncode == 0


def _shm_files_of(pid: int) -> set[str]:
    """``/dev/shm/fastrtps_*`` paths the process has open or mapped, per the kernel."""
    paths = {
        os.readlink(fd)
        for fd in Path(f"/proc/{pid}/fd").iterdir()
        if "/fastrtps_" in os.readlink(fd)
    }
    for line in Path(f"/proc/{pid}/maps").read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) == 6 and fields[5].startswith("/dev/shm/fastrtps_"):
            paths.add(fields[5])
    return paths


_READY_TIMEOUT_S = 20.0


@pytest.mark.skipif(not _rclpy_available(), reason="rclpy not importable (ROS 2 not sourced)")
def test_live_participant_keeps_segments_and_still_delivers() -> None:
    pub = subprocess.Popen(
        [sys.executable, "-c", _PUB], stdout=subprocess.PIPE, env=_env(), text=True
    )
    try:
        assert pub.stdout is not None
        # Bounded: a publisher that stalls in rclpy init must fail the test, not hang it.
        ready, _, _ = select.select([pub.stdout], [], [], _READY_TIMEOUT_S)
        assert ready, f"publisher not ready within {_READY_TIMEOUT_S} s"
        assert pub.stdout.readline().strip() == "ready"
        time.sleep(1.0)  # Fast-DDS maps its ports/segments during participant init.
        owned = _shm_files_of(pub.pid)
        assert owned, "publisher created no /dev/shm/fastrtps_* files (SHM transport off?)"

        _purged, kept = _clean_stale_fastrtps_shm()
        assert kept >= len(owned)
        assert all(Path(p).exists() for p in owned)

        sub = subprocess.run(
            [sys.executable, "-c", _SUB],
            capture_output=True,
            env=_env(),
            text=True,
            timeout=60,
            check=False,
        )
        assert sub.stdout.strip() == "alive", sub.stderr
    finally:
        pub.send_signal(signal.SIGKILL)
        pub.wait(timeout=10)

    # Its own segment is now an orphaned leftover; the clean removes it. Port files
    # (``fastrtps_port<N>``) are shared with every participant on the host, so a
    # live one elsewhere may legitimately keep them.
    time.sleep(0.2)
    _clean_stale_fastrtps_shm()
    segments = [p for p in owned if not Path(p).name.startswith("fastrtps_port")]
    assert segments
    assert not [p for p in segments if Path(p).exists()]
