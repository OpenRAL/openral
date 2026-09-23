# SPDX-License-Identifier: Apache-2.0
"""Bring up a real `ros2_control` graph with only the motors left out.

Shared by `tests/sim/test_openarm_hal_ros2_control.py` (the actuation path) and
`tests/integration/test_real_hal_estop_ros2_control_live.py` (the e-stop path). Both stand up
the *real* stack — `controller_manager` (`ros2_control_node`), real
`joint_trajectory_controller/JointTrajectoryController`s, a real `joint_state_broadcaster` —
over `mock_components/GenericSystem`, upstream ros2_control's own fake-hardware plugin and the
boundary double CLAUDE.md §1.11 allows. Everything OpenRAL owns is the production article.

Per CLAUDE.md §1.11 the dependency is skipped, never faked: `unavailable()` names why a host
cannot run this, and the callers `pytest.skip` on it.
"""

from __future__ import annotations

import contextlib
import importlib.util
import os
import shutil
import signal
import subprocess
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

#: How long the stack gets to come up. Generous because the first `ros2 run` on a cold host
#: pays for ament index construction; the steady-state cost is a couple of seconds.
BRINGUP_TIMEOUT_S = 60.0

__all__ = [
    "BRINGUP_TIMEOUT_S",
    "bring_up",
    "driver_node",
    "spawn",
    "terminate",
    "unavailable",
    "wait_until",
    "write_controller_config",
    "write_generic_system_urdf",
]


def unavailable() -> str | None:
    """Return why this host cannot run a ros2_control stack, or None if it can.

    A dev host with ROS 2 sourced runs it for real; a CI runner without one skips.
    """
    if shutil.which("ros2") is None:
        return "ros2 CLI not on PATH — source a ROS 2 install"
    for module in ("rclpy", "trajectory_msgs", "sensor_msgs", "controller_manager_msgs"):
        if importlib.util.find_spec(module) is None:
            return f"{module} not importable — source a ROS 2 install"
    from ament_index_python.packages import PackageNotFoundError, get_package_share_directory

    # `mock_components` ships inside `hardware_interface` on Jazzy, so checking that package
    # covers the fake-hardware plugin as well as the resource manager.
    for package in (
        "controller_manager",
        "joint_trajectory_controller",
        "joint_state_broadcaster",
        "robot_state_publisher",
        "hardware_interface",
    ):
        try:
            get_package_share_directory(package)
        except (PackageNotFoundError, ValueError):
            return (
                f"ROS 2 package '{package}' not installed — "
                "`apt install ros-$ROS_DISTRO-ros2-control`"
            )
    return None


def write_generic_system_urdf(path: Path, *, name: str, joints: Sequence[str]) -> Path:
    """Write a minimal serial-chain URDF whose `<ros2_control>` block is `GenericSystem`.

    One revolute joint per name, each with a position command interface and position +
    velocity state interfaces — the exact shape every `JointTrajectoryController` in this
    repo's controller configs claims. The mock echoes commands into state with no dynamics, so
    a joint reads back what it was last told; there is no physics to assert on here.
    """
    links = "".join(f'  <link name="{name}_link_{i}"/>\n' for i in range(len(joints) + 1))
    joint_xml = "".join(
        f'  <joint name="{j}" type="revolute">\n'
        f'    <parent link="{name}_link_{i}"/>\n'
        f'    <child link="{name}_link_{i + 1}"/>\n'
        f'    <axis xyz="0 0 1"/>\n'
        f'    <limit lower="-3.14" upper="3.14" effort="100" velocity="3.0"/>\n'
        f"  </joint>\n"
        for i, j in enumerate(joints)
    )
    control_joints = "".join(
        f'    <joint name="{j}">\n'
        f'      <command_interface name="position"/>\n'
        f'      <state_interface name="position"/>\n'
        f'      <state_interface name="velocity"/>\n'
        f"    </joint>\n"
        for j in joints
    )
    path.write_text(
        f'<?xml version="1.0"?>\n<robot name="{name}">\n'
        f"{links}{joint_xml}"
        f'  <ros2_control name="{name}_hardware" type="system">\n'
        f"    <hardware>\n"
        f"      <plugin>mock_components/GenericSystem</plugin>\n"
        f"    </hardware>\n"
        f"{control_joints}"
        f"  </ros2_control>\n</robot>\n"
    )
    return path


def write_controller_config(
    path: Path, controllers: Mapping[str, Sequence[str]], *, update_rate: int = 100
) -> Path:
    """Write `controller_manager` params: a broadcaster plus one JTC per entry."""
    manager = "".join(
        f"    {name}:\n      type: joint_trajectory_controller/JointTrajectoryController\n"
        for name in controllers
    )
    blocks = "".join(
        f"{name}:\n  ros__parameters:\n    joints:\n"
        + "".join(f"      - {j}\n" for j in joints)
        + "    command_interfaces: [position]\n    state_interfaces: [position, velocity]\n"
        for name, joints in controllers.items()
    )
    path.write_text(
        "controller_manager:\n  ros__parameters:\n"
        f"    update_rate: {update_rate}\n"
        "    joint_state_broadcaster:\n      type: joint_state_broadcaster/JointStateBroadcaster\n"
        f"{manager}\n{blocks}"
    )
    return path


def spawn(argv: list[str], log: Path, env: dict[str, str]) -> subprocess.Popen[bytes]:
    """Start one ROS node in its own process group so teardown can take the whole tree."""
    handle = log.open("wb")
    # Fixed argv, no shell. `start_new_session` is what makes teardown reliable: `ros2 run`
    # execs the node as a child, so signalling the group is the only way to take both.
    return subprocess.Popen(
        argv, stdout=handle, stderr=subprocess.STDOUT, env=env, start_new_session=True
    )


def terminate(process: subprocess.Popen[bytes]) -> None:
    """Stop a node and everything it spawned, escalating to SIGKILL."""
    if process.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGINT)
        process.wait(timeout=10)
    except (subprocess.TimeoutExpired, ProcessLookupError):
        with contextlib.suppress(OSError):
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)


@contextlib.contextmanager
def bring_up(tmp: Path, *, urdf: Path, config: Path, controllers: Sequence[str]) -> Iterator[None]:
    """Run `robot_state_publisher` + `ros2_control_node` and spawn every controller.

    Skips (never fakes) when the host cannot run the stack. The children inherit the caller's
    `ROS_DOMAIN_ID`, which is what keeps this graph off any real robot's domain. Deliberately
    NOT setting `ROS_AUTOMATIC_DISCOVERY_RANGE` — see `tests/sim/conftest.py` for the hang it
    causes across a process boundary the test does not launch into a single environment.
    """
    reason = unavailable()
    if reason is not None:
        pytest.skip(reason)
    env = dict(os.environ)
    processes: list[subprocess.Popen[bytes]] = []
    try:
        # controller_manager takes the description from the /robot_description topic on Jazzy,
        # so robot_state_publisher is a required participant, not a convenience.
        processes.append(
            spawn(
                [
                    "ros2",
                    "run",
                    "robot_state_publisher",
                    "robot_state_publisher",
                    "--ros-args",
                    "-p",
                    f"robot_description:={urdf.read_text()}",
                ],
                tmp / "robot_state_publisher.log",
                env,
            )
        )
        processes.append(
            spawn(
                [
                    "ros2",
                    "run",
                    "controller_manager",
                    "ros2_control_node",
                    "--ros-args",
                    "--params-file",
                    str(config),
                ],
                tmp / "controller_manager.log",
                env,
            )
        )
        deadline = time.monotonic() + BRINGUP_TIMEOUT_S
        for controller in ("joint_state_broadcaster", *controllers):
            remaining = max(1.0, deadline - time.monotonic())
            # The spawner blocks until the controller is configured and active, then exits,
            # so its return code is the readiness signal — no polling needed.
            spawned = subprocess.run(
                ["ros2", "run", "controller_manager", "spawner", controller],
                capture_output=True,
                timeout=remaining,
                env=env,
                check=False,
            )
            if spawned.returncode != 0:
                manager_log = (tmp / "controller_manager.log").read_text(errors="replace")
                pytest.fail(
                    f"Could not activate {controller!r} (exit {spawned.returncode}).\n"
                    f"spawner stderr:\n{spawned.stderr.decode(errors='replace')}\n"
                    f"controller_manager log:\n{manager_log}"
                )
        yield
    finally:
        for process in reversed(processes):
            terminate(process)


@contextlib.contextmanager
def driver_node(name: str) -> Iterator[tuple[Any, Callable[[float], None]]]:
    """Yield `(node, spin)` on a private rclpy context, torn down on the way out.

    A private context (rather than `rclpy.init()`) keeps each test's participant separate and
    leaves the process's default context untouched for the rest of the tier. That forces the
    executor to be created against the same context too — `rclpy.spin_once(node)` reaches for
    the *default* context and fails on a node that does not belong to it.
    """
    import rclpy
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.node import Node

    context = rclpy.Context()
    context.init()
    node = Node(name, context=context)
    executor = SingleThreadedExecutor(context=context)
    executor.add_node(node)

    def spin(seconds: float) -> None:
        """Spin the private executor for ``seconds`` of wall time."""
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            executor.spin_once(timeout_sec=0.02)

    try:
        yield node, spin
    finally:
        executor.remove_node(node)
        node.destroy_node()
        context.shutdown()


def wait_until(
    spin: Callable[[float], None], predicate: Callable[[], bool], timeout_s: float
) -> bool:
    """Spin until `predicate()` holds or the timeout expires; report which happened.

    DDS discovery takes a beat — a fresh participant does not see an existing publisher for
    something on the order of a second — so every wait here is a poll, never a bare sleep.
    """
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        spin(0.1)
    return predicate()
