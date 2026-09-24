"""``openral record --profile full`` against a real ``ros2 bag record``.

The full profile records non-camera sensors (2-D lidar, point clouds, IMU) by
message type, whatever topic a manifest gives them, and folds its verbatim
topics into the regex. Mixing explicit topics with ``--regex`` made rosbag2
(Jazzy) stop discovery once as many topics as were listed had been subscribed,
so every topic discovered later was silently dropped. This test starts the
sensor publishers *after* the recorder has subscribed ``/joint_states``, which
is exactly the case that used to be lost.

Real ``ros2 topic pub`` publishers and a real recorder, on an isolated domain.
Skipped on hosts without a sourced ROS 2.
"""

from __future__ import annotations

import os
import random
import shutil
import signal
import subprocess
import time
from pathlib import Path

import pytest
from openral_observability.replay.cli import build_record_command

_ROS2_AVAILABLE = bool(os.environ.get("ROS_DISTRO")) and shutil.which("ros2") is not None

# Topic names a real manifest could pick (panda_mobile's `/scan`, a ZED IMU,
# a Velodyne cloud); none of them is spelled in the profile.
_LATE_SENSORS = {
    "/scan": "sensor_msgs/msg/LaserScan",
    "/zed/imu/data": "sensor_msgs/msg/Imu",
    "/velodyne_points": "sensor_msgs/msg/PointCloud2",
}


def _pub(topic: str, msg_type: str, env: dict[str, str]) -> subprocess.Popen[bytes]:
    return subprocess.Popen(
        ["ros2", "topic", "pub", "-r", "10", topic, msg_type, "{}"],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


@pytest.mark.skipif(not _ROS2_AVAILABLE, reason="ROS_DISTRO / ros2 not available")
def test_full_profile_records_late_lidar_and_imu_topics(tmp_path: Path) -> None:
    env = os.environ.copy()
    env["ROS_DOMAIN_ID"] = str(random.randint(100, 200))  # test isolation, not crypto
    env["ROS_AUTOMATIC_DISCOVERY_RANGE"] = "LOCALHOST"
    bag = tmp_path / "bag"
    procs = [_pub("/joint_states", "sensor_msgs/msg/JointState", env)]
    recorder: subprocess.Popen[bytes] | None = None
    try:
        time.sleep(3.0)
        recorder = subprocess.Popen(
            build_record_command(profile="full", output_dir=bag),
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(4.0)  # the recorder has subscribed /joint_states by now
        procs += [
            _pub(t, ty, env) for t, ty in [*_LATE_SENSORS.items(), ("/x", "std_msgs/msg/String")]
        ]
        time.sleep(8.0)
    finally:
        if recorder is not None:
            recorder.send_signal(signal.SIGINT)
            recorder.wait(timeout=20)
        for p in procs:
            p.terminate()
            p.wait(timeout=10)
    info = subprocess.run(
        ["ros2", "bag", "info", str(bag)], env=env, capture_output=True, text=True, check=True
    ).stdout
    recorded = {
        part.split("Topic:")[1].split("|")[0].strip()
        for part in info.splitlines()
        if "Topic:" in part
    }
    assert recorded >= {"/joint_states", *_LATE_SENSORS}
    assert "/x" not in recorded
