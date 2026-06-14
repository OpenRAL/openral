"""Integration test for the ``probe_dds`` ROS 2 topology probe.

Spawns a real ``ros2 topic pub`` publisher and asserts the probe sees
the resulting topic.  Skipped on hosts without ``ROS_DISTRO``.

Uses the same `ros2 topic pub` subprocess pattern as
``tests/integration/test_autodetect_dds.py`` because in-process rclpy
publishers are unreliable for cross-process DDS discovery in
containers.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time

import pytest
from openral_detect.probes import probe_dds
from openral_detect.report import Ros2TopologyResult

_ROS2_AVAILABLE = bool(os.environ.get("ROS_DISTRO")) and shutil.which("ros2") is not None


def _wait_for_topic(name: str, deadline_s: float = 15.0) -> Ros2TopologyResult:
    deadline = time.monotonic() + deadline_s
    last: Ros2TopologyResult | None = None
    while time.monotonic() < deadline:
        last = probe_dds(timeout_s=3.0, warnings=[])
        if any(t.name == name for t in last.topics):
            return last
        time.sleep(0.5)
    assert last is not None
    return last


@pytest.mark.skipif(not _ROS2_AVAILABLE, reason="ROS_DISTRO / ros2 not available")
def test_probe_dds_finds_chatter_topic() -> None:  # pragma: no cover
    """probe_dds discovers a published topic + captures RMW + domain id."""
    proc = subprocess.Popen(
        [
            "ros2",
            "topic",
            "pub",
            "/bh_detect_test_topic",
            "std_msgs/msg/String",
            '{data: "hello"}',
            "--rate",
            "5",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        result = _wait_for_topic("/bh_detect_test_topic")
        topic_names = {t.name for t in result.topics}
        assert "/bh_detect_test_topic" in topic_names
        # RMW + domain id must be carried through.
        assert isinstance(result.rmw_implementation, str)
        assert isinstance(result.domain_id, int)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=3.0)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
