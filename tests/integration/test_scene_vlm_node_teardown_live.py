"""The scene-VLM node ends the sidecar its warm-up spawned when it shuts down (#421 review).

Since the VLM loads at configure (``warm_up``), a node that never served a query still
owns a spawned sidecar; ``destroy_node`` did not close it, so the model process (and its
GPU memory) outlived every ``deploy sim``. The REAL ``scene_vlm_node`` runs as a
subprocess and auto-spawns a fake sidecar at the process boundary
(``tests/integration/fakes/fake_qwen_vlm_sidecar.py``); SIGINT must end both.

Gated on ``OPENRAL_TEST_ROS_LIVE=1`` and pyzmq; listed in ``scripts/ros_live_tests.sh``.
"""

from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_NODE = _REPO / "packages/openral_perception_ros/openral_perception_ros/scene_vlm_node.py"
_FAKE = _REPO / "tests/integration/fakes/fake_qwen_vlm_sidecar.py"
_MANIFEST = _REPO / "rskills/qwen35-4b-nf4/rskill.yaml"

pytestmark = pytest.mark.skipif(
    not os.getenv("OPENRAL_TEST_ROS_LIVE"),
    reason="live rclpy node subprocess: set OPENRAL_TEST_ROS_LIVE=1 and source install/",
)


def _alive(pid: int) -> bool:
    try:
        state = Path(f"/proc/{pid}/stat").read_text().split(")")[-1].split()[0]
    except OSError:
        return False
    return state != "Z"


def test_shutting_down_the_node_ends_the_sidecar_its_warm_up_spawned(tmp_path: Path) -> None:
    pytest.importorskip("zmq")
    pytest.importorskip("rclpy")
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    pid_file = tmp_path / "sidecar.pid"
    env = {
        **os.environ,
        "OPENRAL_QWEN_VLM_SIDECAR": str(_FAKE),
        "FAKE_QWEN_SIDECAR_PID_FILE": str(pid_file),
    }
    log = tmp_path / "node.log"
    with log.open("wb") as out:
        node = subprocess.Popen(
            [
                sys.executable, str(_NODE), "--ros-args",
                "-p", f"manifest_path:={_MANIFEST}",
                "-p", "image_topic:=/test/scene_vlm_teardown/cam",
                "-p", f"sidecar_port:={port}",
            ],
            stdout=out, stderr=subprocess.STDOUT, env=env,
        )  # fmt: skip
    sidecar_pid = None
    try:
        deadline = time.monotonic() + 60.0
        while time.monotonic() < deadline and "(loaded)" not in log.read_text(errors="replace"):
            assert node.poll() is None, log.read_text(errors="replace")
            time.sleep(0.2)
        assert "(loaded)" in log.read_text(errors="replace"), log.read_text(errors="replace")
        sidecar_pid = int(pid_file.read_text())
        assert _alive(sidecar_pid)
        node.send_signal(signal.SIGINT)
        node.wait(timeout=30)
        end = time.monotonic() + 15.0
        while time.monotonic() < end and _alive(sidecar_pid):
            time.sleep(0.2)
        assert not _alive(sidecar_pid), "the warm-up's sidecar outlived the node"
    finally:
        if node.poll() is None:
            node.kill()
            node.wait()
        if sidecar_pid is not None and _alive(sidecar_pid):
            os.kill(sidecar_pid, signal.SIGKILL)
