"""Shared integration fixtures — the live MoveIt panda demo harness.

Used by ``test_moveit_plan_arm_franka.py`` and
``test_look_at_franka.py``. Real components only: the fixture
spawns the upstream ``moveit_resources_panda_moveit_config`` demo (real
``move_group`` + ``ros2_control`` fake hardware + ``robot_state_publisher``)
and skips — never fakes — when the package or a ROS workspace is absent.
"""

from __future__ import annotations

import shutil
import subprocess
import time
from collections.abc import Callable, Iterator
from contextlib import suppress
from typing import Any

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


# ── Safety-kernel place-* live-ROS test harness (shared by
# test_safety_kernel_place_allowance_band.py and
# test_safety_kernel_place_target_geometry.py, whose 1-DoF carriage rig,
# grid and helper-node wiring are byte-identical) ──────────────────────────
#
# Factories, not fixtures that build the harness themselves: both tests
# construct their own kernel subprocess, rclpy node, publishers and
# subscribers inline (real DDS graph, CLAUDE.md §1.11), and these three
# small pieces are the only closures byte-identical between them. Each
# fixture returns a callable so the test keeps owning its own `helper`
# node / publishers / `spin` and this file never imports ROS at module
# level (only inside the returned callable, matching the tests' own lazy
# `import rclpy` — modules that skip via ``pytestmark`` when ROS isn't
# live must never eagerly import it here).


@pytest.fixture
def publish_occupancy_grid() -> Callable[..., None]:
    """Factory: publish a single-occupied-cell ``OccupancyVoxels`` lattice and settle.

    Returns ``publish_grid(voxel_pub, helper, spin, *, grid_origin_m,
    resolution_m, grid_n, occ_index)``.
    """

    def _publish_grid(
        voxel_pub: Any,
        helper: Any,
        spin: Callable[[float], None],
        *,
        grid_origin_m: float,
        resolution_m: float,
        grid_n: int,
        occ_index: int,
    ) -> None:
        from geometry_msgs.msg import Point, Quaternion
        from openral_msgs.msg import OccupancyVoxels

        grid = OccupancyVoxels()
        grid.header.frame_id = "base"
        grid.header.stamp = helper.get_clock().now().to_msg()
        grid.origin = Point(x=grid_origin_m, y=grid_origin_m, z=grid_origin_m)
        # `OccupancyVoxels` is an oriented grid and its unset orientation is
        # the all-zero quaternion, which every consumer refuses rather than
        # reading as identity — so say identity.
        grid.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
        grid.resolution = resolution_m
        grid.size_x = grid_n
        grid.size_y = grid_n
        grid.size_z = grid_n
        occupancy = [0] * (grid_n**3)
        occupancy[occ_index] = 1
        grid.occupancy = occupancy
        voxel_pub.publish(grid)
        spin(0.4)

    return _publish_grid


@pytest.fixture
def publish_carriage_joint_state() -> Callable[..., None]:
    """Factory: publish one ``JointState`` for the 1-DoF place-* carriage rig and settle.

    Returns ``publish_joint_state(joint_pub, helper, spin, *, joint_names,
    positions)``.
    """

    def _publish_joint_state(
        joint_pub: Any,
        helper: Any,
        spin: Callable[[float], None],
        *,
        joint_names: list[str],
        positions: list[float],
    ) -> None:
        from sensor_msgs.msg import JointState

        js = JointState()
        js.header.stamp = helper.get_clock().now().to_msg()
        js.name = list(joint_names)
        js.position = list(positions)
        joint_pub.publish(js)
        spin(0.2)

    return _publish_joint_state


@pytest.fixture
def reset_kernel_estop() -> Callable[..., None]:
    """Factory: call ``/openral/estop_reset`` and wait for it to clear.

    Returns ``reset_estop(reset_client, executor, spin, estops)`` —
    ``estops`` is the caller's ``/openral/estop`` capture list, cleared
    once the reset lands so a later phase's assertions see only its own
    estop.
    """

    def _reset_estop(
        reset_client: Any,
        executor: Any,
        spin: Callable[[float], None],
        estops: list[Any],
    ) -> None:
        from std_srvs.srv import Trigger

        assert reset_client.wait_for_service(timeout_sec=5.0)
        spin(0.3)  # clear the reset cooldown
        future = reset_client.call_async(Trigger.Request())
        end = time.time() + 5.0
        while time.time() < end and not future.done():
            executor.spin_once(timeout_sec=0.02)
        assert future.done() and future.result().success, "estop reset refused"
        estops.clear()
        spin(0.3)

    return _reset_estop
