"""``twist_to_action_relay_node`` puts Nav2's ``/cmd_vel`` under the real safety kernel.

Real components (CLAUDE.md §1.11): the production relay process, the production
``safety_kernel_node`` booted from ``robots/r1pro/robot.yaml`` (``body_twist``, no
``base_joints`` — the case the relay exists for), and real DDS on an isolated
``ROS_DOMAIN_ID``. The test plays Nav2: it publishes ``geometry_msgs/Twist`` on ``/cmd_vel``.

Pins: the candidate chunk's wire content; in-bound twists reach ``/openral/safe_action``;
silence after motion yields exactly one zero twist; an over-speed twist is vetoed + E-stops;
a ``base_joints`` manifest is refused (no double actuation with ``MobileBaseBridge``).

Gated on ``OPENRAL_TEST_ROS_LIVE=1``, listed in ``scripts/ros_live_tests.sh``. Locally::

    source /opt/ros/jazzy/setup.bash && just ros2-build && source install/setup.bash
    OPENRAL_TEST_ROS_LIVE=1 uv run pytest tests/integration/test_twist_to_action_relay_live.py
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import pytest

from tests.sim.safety._kernel_subprocess import (
    activate_kernel_node,
    isolated_domain_id,
    start_kernel,
    terminate_kernel,
)

# Own DDS domain before any rclpy import; the subprocesses inherit it.
os.environ.setdefault("ROS_DOMAIN_ID", str(isolated_domain_id()))

pytestmark = pytest.mark.skipif(
    not os.getenv("OPENRAL_TEST_ROS_LIVE"),
    reason="live rclpy + real safety kernel — set OPENRAL_TEST_ROS_LIVE=1 and source "
    "install/setup.bash first.",
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_RELAY = (
    _REPO_ROOT
    / "packages"
    / "openral_nav2_bringup"
    / "openral_nav2_bringup"
    / "twist_to_action_relay_node.py"
)
_R1PRO = _REPO_ROOT / "robots" / "r1pro" / "robot.yaml"
_PANDA_MOBILE = _REPO_ROOT / "robots" / "panda_mobile" / "robot.yaml"
_BODY_TWIST = 7  # openral_core.CONTROL_MODE_TO_UINT8[BODY_TWIST]
_TIMEOUT_S = 0.3


def _relay(robot_yaml: Path, log_path: Path) -> subprocess.Popen[bytes]:
    return subprocess.Popen(
        [
            sys.executable,
            str(_RELAY),
            "--ros-args",
            "-p",
            f"robot_yaml:={robot_yaml}",
            "-p",
            f"cmd_vel_timeout_s:={_TIMEOUT_S}",
        ],
        stdout=log_path.open("wb"),
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )


def _twist(vx: float, vy: float, wz: float) -> Any:
    from geometry_msgs.msg import Twist

    msg = Twist()
    msg.linear.x, msg.linear.y, msg.angular.z = vx, vy, wz
    return msg


def test_relay_routes_cmd_vel_through_the_kernel(tmp_path: Path) -> None:
    pytest.importorskip("openral_msgs")
    import rclpy
    from openral_core import RobotDescription
    from openral_msgs.msg import ActionChunk
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
    from std_msgs.msg import Empty

    kernel_name = f"relay_kernel_{uuid.uuid4().hex[:8]}"
    kernel = start_kernel(
        RobotDescription.from_yaml(_R1PRO), kernel_name, log_path=tmp_path / "kernel.log"
    )
    relay = _relay(_R1PRO, tmp_path / "relay.log")
    rclpy.init()
    try:
        helper = rclpy.create_node(f"relay_helper_{uuid.uuid4().hex[:6]}")
        assert activate_kernel_node(kernel_name, helper, service_timeout_s=20.0), (
            tmp_path / "kernel.log"
        ).read_text(errors="replace")
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=50,
        )
        candidates: list[Any] = []
        safe: list[Any] = []
        estops: list[Any] = []
        cand_sub = helper.create_subscription(
            ActionChunk, "/openral/candidate_action", candidates.append, qos
        )
        helper.create_subscription(ActionChunk, "/openral/safe_action", safe.append, qos)
        helper.create_subscription(Empty, "/openral/estop", estops.append, 10)
        from geometry_msgs.msg import Twist

        cmd_vel = helper.create_publisher(Twist, "/cmd_vel", 10)
        executor = SingleThreadedExecutor()
        executor.add_node(helper)

        def spin(seconds: float) -> None:
            end = time.monotonic() + seconds
            while time.monotonic() < end:
                executor.spin_once(timeout_sec=0.02)

        deadline = time.monotonic() + 20.0
        while time.monotonic() < deadline and not (
            cmd_vel.get_subscription_count() >= 1 and cand_sub.get_publisher_count() >= 1
        ):
            spin(0.1)
        assert relay.poll() is None, (tmp_path / "relay.log").read_text(errors="replace")
        assert cmd_vel.get_subscription_count() >= 1, "relay never subscribed /cmd_vel"

        # In-bound for r1pro (0.7 m/s, 0.3 rad/s): Nav2-style 10 Hz stream.
        for _ in range(5):
            cmd_vel.publish(_twist(0.4, 0.3, 0.2))
            spin(0.1)
        spin(0.05)
        moving = [c for c in candidates if any(c.flat)]
        assert len(moving) == 5, [list(c.flat) for c in candidates]
        chunk = moving[0]
        assert chunk.control_mode == _BODY_TWIST
        assert (chunk.horizon, chunk.n_dof) == (1, 6)
        assert list(chunk.flat) == pytest.approx([0.4, 0.3, 0.0, 0.0, 0.0, 0.2])
        assert chunk.frame_id == "base_link"
        assert any(any(c.flat) for c in safe), "in-bound twist must reach safe_action"
        assert not estops

        # Nav2 goes silent: exactly one zero, and it too passes the kernel.
        n_before = len(candidates)
        spin(_TIMEOUT_S * 5)
        zeros = candidates[n_before:]
        assert len(zeros) == 1, [list(c.flat) for c in zeros]
        assert list(zeros[0].flat) == [0.0] * 6
        assert list(safe[-1].flat) == [0.0] * 6, "stale zero must reach safe_action"

        # Over-speed (|v| = 0.85 > 0.7): reaches the kernel, never the HAL.
        n_safe = len(safe)
        cmd_vel.publish(_twist(0.6, 0.6, 0.0))
        spin(0.5)
        assert any(list(c.flat)[:2] == [0.6, 0.6] for c in candidates)
        assert all(list(c.flat)[:2] != [0.6, 0.6] for c in safe[n_safe:])
        assert estops, "kernel must E-stop on an over-speed /cmd_vel"
    finally:
        rclpy.shutdown()
        # SIGTERM is what `ros2 launch` escalates to; Jazzy rclpy handles it (no custom handler).
        os.killpg(relay.pid, signal.SIGTERM)
        try:
            relay.wait(timeout=10)
        finally:
            if relay.poll() is None:
                os.killpg(relay.pid, signal.SIGKILL)
            terminate_kernel(kernel)
    assert relay.returncode is not None and relay.returncode >= 0, "relay ignored SIGTERM"


def test_relay_refuses_a_base_joints_manifest(tmp_path: Path) -> None:
    """``panda_mobile``'s ``/cmd_vel`` belongs to ``MobileBaseBridge`` (ADR-0040/0099)."""
    pytest.importorskip("openral_msgs")
    log = tmp_path / "relay.log"
    relay = _relay(_PANDA_MOBILE, log)
    try:
        code = relay.wait(timeout=30)
    finally:
        if relay.poll() is None:
            relay.kill()
    assert code != 0
    assert "declares base_joints" in log.read_text(errors="replace")
