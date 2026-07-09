"""Real ROS 2 DDS integration test for autodetect.scan_dds_topics.

Requires a sourced ROS 2 installation (``ROS_DISTRO`` env var set).
Skipped automatically in pure-Python CI.

Scenario: spawn ``ros2 topic pub /lowstate std_msgs/msg/String`` in a
separate subprocess (Unitree G1 signature topic), then call
:func:`scan_dds_topics` and verify that :func:`infer_robot_from_topics`
maps the discovered topic to ``"unitree_g1"``.

The publisher runs as a separate ``ros2`` subprocess (instead of an
in-process rclpy node) so its DDS participant is discovered reliably
from the ``ros2 topic list`` subprocess inside containers where FastDDS
multicast discovery between in-process rclpy nodes and external CLI
processes is unreliable.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time

import pytest

_ROS2_AVAILABLE = bool(os.environ.get("ROS_DISTRO")) and shutil.which("ros2") is not None


def _wait_for_topic(name: str, deadline_s: float = 15.0) -> list:  # type: ignore[type-arg]
    """Poll ``scan_dds_topics`` until ``name`` is discovered or the deadline passes."""
    from openral_cli.autodetect import scan_dds_topics

    deadline = time.monotonic() + deadline_s
    topics: list = []  # type: ignore[type-arg]
    while time.monotonic() < deadline:
        topics = scan_dds_topics(timeout_s=3.0)
        if any(t.name == name for t in topics):
            return topics
        time.sleep(0.5)
    return topics


@pytest.mark.skipif(not _ROS2_AVAILABLE, reason="ROS_DISTRO / ros2 not available")
def test_dds_discovery_finds_lowstate_topic() -> None:  # pragma: no cover
    """scan_dds_topics finds /lowstate and maps it to unitree_g1."""
    from openral_cli.autodetect import infer_robot_from_topics

    proc = subprocess.Popen(
        [
            "ros2",
            "topic",
            "pub",
            "-r",
            "10",
            "/lowstate",
            "std_msgs/msg/String",
            "{data: heartbeat}",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    try:
        topics = _wait_for_topic("/lowstate", deadline_s=15.0)
        names = [t.name for t in topics]
        assert "/lowstate" in names, f"Expected /lowstate in topics; got {names}"

        robot = infer_robot_from_topics(topics)
        assert robot == "unitree_g1", f"Expected unitree_g1; inferred {robot!r}"
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=2.0)


@pytest.mark.skipif(not _ROS2_AVAILABLE, reason="ROS_DISTRO / ros2 not available")
def test_dds_discovery_empty_when_no_nodes() -> None:  # pragma: no cover
    """scan_dds_topics returns a list (possibly empty) when only rosout exists."""
    from openral_cli.autodetect import scan_dds_topics

    topics = scan_dds_topics(timeout_s=1.0)
    assert isinstance(topics, list)
