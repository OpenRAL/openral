"""Shared integration fixtures — the live MoveIt panda demo harness.

Used by ``test_moveit_plan_arm_franka.py`` (ADR-0024) and
``test_look_at_franka.py`` (ADR-0044). Real components only: the fixture
spawns the upstream ``moveit_resources_panda_moveit_config`` demo (real
``move_group`` + ``ros2_control`` fake hardware + ``robot_state_publisher``)
and skips — never fakes — when the package or a ROS workspace is absent.
"""

from __future__ import annotations

import shutil
import subprocess
import time
from collections.abc import Iterator
from contextlib import suppress

import pytest


def moveit_panda_demo_available() -> bool:
    """Probe for the upstream MoveIt panda demo launch.

    Resolved via ``ros2 pkg prefix`` rather than importing because the package
    is a pure-ament resource package (no Python entry point).
    """
    if shutil.which("ros2") is None:
        return False
    result = subprocess.run(
        ["ros2", "pkg", "prefix", "moveit_resources_panda_moveit_config"],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    return result.returncode == 0 and result.stdout.strip() != ""


@pytest.fixture(scope="session")
def move_group_subprocess() -> Iterator[None]:
    """Spawn the upstream MoveIt panda demo and wait for ``/move_action``.

    The demo brings up ``move_group``, ``ros2_control`` fake hardware
    (publishes ``/joint_states``), and ``robot_state_publisher``; RViz is
    suppressed. Teardown SIGTERMs the whole process group, escalating to
    SIGKILL after 5 s.
    """
    if not moveit_panda_demo_available():
        pytest.skip(
            "ros-${ROS_DISTRO}-moveit-resources-panda-moveit-config is not installed; "
            "install it (apt) to run the live MoveIt integration tests."
        )

    proc = subprocess.Popen(
        [
            "ros2",
            "launch",
            "moveit_resources_panda_moveit_config",
            "demo.launch.py",
            "use_rviz:=false",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        # New process group so teardown can `killpg` the whole tree
        # (move_group + robot_state_publisher + spawners). Without this,
        # `proc.terminate()` only kills the `ros2 launch` parent and leaves
        # orphans that pollute later test runs.
        start_new_session=True,
    )

    try:
        # Wait for /move_action. ``ros2 action list`` is the simplest
        # cross-distro probe.
        deadline = time.monotonic() + 60.0
        ready = False
        while time.monotonic() < deadline:
            try:
                result = subprocess.run(
                    ["ros2", "action", "list"],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
            except subprocess.TimeoutExpired:
                continue
            if "/move_action" in result.stdout:
                ready = True
                break
            time.sleep(1.0)
        if not ready:
            proc.terminate()
            proc.wait(timeout=10)
            pytest.skip(
                "MoveIt demo launch did not register /move_action within 60s — "
                "host may be too slow or the upstream package may have changed."
            )
        # Settling delay so internal pipelines (planning scene monitor, FK/IK
        # init, controller spawners) finish coming up — MoveIt takes 5-8 s to
        # fully accept goals after /move_action registers.
        time.sleep(8.0)
        yield
    finally:
        import os
        import signal as _signal

        with suppress(ProcessLookupError):
            os.killpg(os.getpgid(proc.pid), _signal.SIGTERM)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            with suppress(ProcessLookupError):
                os.killpg(os.getpgid(proc.pid), _signal.SIGKILL)
            proc.wait(timeout=5)
