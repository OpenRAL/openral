# SPDX-License-Identifier: Apache-2.0
"""`RosControlHAL` must actually move a `ros2_control` robot — proven without the robot.

Every other sim test in this tier drives a HAL straight into MuJoCo, which is precisely why
the production actuation path could stay broken for so long: `OpenArmMujocoHAL` and friends
bypass the transport entirely, so no amount of sim coverage said anything about whether a
real arm would move. The two defects fixed on live hardware on 2026-09-08 — a `publish_fn`
that was a `log.debug` no-op, and a staleness watchdog that measured time since `connect()` —
were both invisible here for that reason.

This file closes that gap by standing up the *real* stack and leaving only the motors out:

* real `controller_manager` (`ros2_control_node`), running at 100 Hz,
* real `joint_trajectory_controller/JointTrajectoryController`,
* real `joint_state_broadcaster` publishing `/joint_states`,
* `mock_components/GenericSystem` as the hardware plugin.

Only the last is a stand-in, and it is upstream ros2_control's own fake-hardware component,
not a mock we wrote — the boundary double CLAUDE.md §1.11 allows. Everything OpenRAL owns is
the production article: `RosControlHAL`, `RosControlTransport`, the declared `ControllerKind`,
the QoS profiles, and the by-name joint-state merge. What this cannot prove is physics — the
mock echoes commands into state with no dynamics, so tolerances, following error and timing
still belong to `tests/hil/`.

The robot description is the repo's own vendored `robots/openarm/openarm.urdf`, which already
declares `mock_components/GenericSystem`; the controller config is derived from that file's
`<ros2_control>` blocks so the two cannot drift.
"""

from __future__ import annotations

import contextlib
import importlib.util
import os
import shutil
import signal
import subprocess
import time
import xml.etree.ElementTree as ET
from collections.abc import Iterator
from pathlib import Path

import pytest

pytestmark = pytest.mark.sim

_REPO_ROOT = Path(__file__).resolve().parents[2]
_URDF = _REPO_ROOT / "robots" / "openarm" / "openarm.urdf"

#: The `<ros2_control>` block this test brings up, and the controller spawned over it.
_HARDWARE_COMPONENT = "left_hardware_interface"
_ARM_CONTROLLER = "left_joint_trajectory_controller"

#: How long the stack gets to come up. Generous because the first `ros2 run` on a cold host
#: pays for ament index construction; the steady-state cost is a couple of seconds.
_BRINGUP_TIMEOUT_S = 60.0


def _unavailable() -> str | None:
    """Return why this test cannot run here, or None if it can.

    A dev host with ROS 2 sourced runs it for real; a CI runner without one skips. Per
    CLAUDE.md §1.11 the dependency is skipped, never faked.
    """
    if shutil.which("ros2") is None:
        return "ros2 CLI not on PATH — source a ROS 2 install"
    for module in ("rclpy", "trajectory_msgs", "sensor_msgs"):
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


_SKIP_REASON = _unavailable()


def _controlled_joints(component: str) -> list[str]:
    """Return the joints one `<ros2_control>` block commands, in URDF order.

    Read from the URDF rather than restated here: a controller whose `joints` list disagrees
    with the hardware component fails to activate, and a hand-copied list is exactly how that
    disagreement gets introduced. Mimic joints are excluded — ros2_control drives them from
    their target and rejects a command interface on them.
    """
    root = ET.parse(_URDF).getroot()
    for block in root.findall("ros2_control"):
        if block.get("name") != component:
            continue
        return [
            joint.get("name", "")
            for joint in block.findall("joint")
            if joint.find("command_interface") is not None
        ]
    raise AssertionError(f"{_URDF} declares no <ros2_control> block named {component!r}.")


def _targets_within_limits(joints: list[str]) -> list[float]:
    """Pick one reachable target per joint, from that joint's own URDF limits.

    Derived rather than hardcoded for the same reason the joint list is: the arm joints and
    the gripper finger have very different ranges (the finger's is entirely negative), so a
    literal vector that looks harmless is easy to write and lands outside a limit. The
    fractions differ per index so the assertion would catch a transport that published the
    right values in the wrong order.
    """
    root = ET.parse(_URDF).getroot()
    limits = {
        joint.get("name"): joint.find("limit")
        for joint in root.findall("joint")
        if joint.find("limit") is not None
    }
    targets: list[float] = []
    for i, name in enumerate(joints):
        limit = limits[name]
        lower, upper = float(limit.get("lower", -1.0)), float(limit.get("upper", 1.0))  # type: ignore[union-attr]  # reason: filtered above
        targets.append(lower + (0.25 + 0.04 * i) * (upper - lower))
    return targets


def _write_controller_config(path: Path, joints: list[str]) -> Path:
    """Write the `controller_manager` params for a single-arm trajectory controller."""
    joint_block = "\n".join(f"      - {name}" for name in joints)
    path.write_text(
        "controller_manager:\n"
        "  ros__parameters:\n"
        "    update_rate: 100\n"
        "    joint_state_broadcaster:\n"
        "      type: joint_state_broadcaster/JointStateBroadcaster\n"
        f"    {_ARM_CONTROLLER}:\n"
        "      type: joint_trajectory_controller/JointTrajectoryController\n"
        "\n"
        f"{_ARM_CONTROLLER}:\n"
        "  ros__parameters:\n"
        "    joints:\n"
        f"{joint_block}\n"
        "    command_interfaces: [position]\n"
        "    state_interfaces: [position, velocity]\n"
    )
    return path


def _spawn(argv: list[str], log: Path, env: dict[str, str]) -> subprocess.Popen[bytes]:
    """Start one ROS node in its own process group so teardown can take the whole tree."""
    handle = log.open("wb")
    # Fixed argv, no shell. `start_new_session` is what makes teardown reliable: `ros2 run`
    # execs the node as a child, so signalling the group is the only way to take both.
    return subprocess.Popen(
        argv, stdout=handle, stderr=subprocess.STDOUT, env=env, start_new_session=True
    )


def _terminate(process: subprocess.Popen[bytes]) -> None:
    """Stop a node and everything it spawned, escalating to SIGKILL."""
    if process.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGINT)
        process.wait(timeout=10)
    except (subprocess.TimeoutExpired, ProcessLookupError):
        with contextlib.suppress(OSError):
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)


@pytest.fixture(scope="module")
def ros2_control_stack(tmp_path_factory: pytest.TempPathFactory) -> Iterator[list[str]]:
    """Bring the mock ros2_control graph up once, and yield the commanded joint names.

    Module-scoped: a `controller_manager` bringup costs several seconds and nothing in this
    file mutates the graph in a way the next test would notice — the mock hardware's state is
    whatever the last command left, which each test sets for itself.
    """
    if _SKIP_REASON is not None:
        pytest.skip(_SKIP_REASON)

    tmp = tmp_path_factory.mktemp("ros2_control")
    joints = _controlled_joints(_HARDWARE_COMPONENT)
    config = _write_controller_config(tmp / "controllers.yaml", joints)

    # The conftest pins ROS_DOMAIN_ID per pytest PID; the children inherit it, which is what
    # keeps this graph off any real robot's domain. Deliberately NOT setting
    # ROS_AUTOMATIC_DISCOVERY_RANGE — see tests/sim/conftest.py for the hang it causes across
    # a process boundary the test does not launch into a single environment.
    env = dict(os.environ)

    processes: list[subprocess.Popen[bytes]] = []
    try:
        # controller_manager takes the description from the /robot_description topic on Jazzy,
        # so robot_state_publisher is a required participant, not a convenience.
        processes.append(
            _spawn(
                [
                    "ros2",
                    "run",
                    "robot_state_publisher",
                    "robot_state_publisher",
                    "--ros-args",
                    "-p",
                    f"robot_description:={_URDF.read_text()}",
                ],
                tmp / "robot_state_publisher.log",
                env,
            )
        )
        processes.append(
            _spawn(
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

        deadline = time.monotonic() + _BRINGUP_TIMEOUT_S
        for controller in ("joint_state_broadcaster", _ARM_CONTROLLER):
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
        yield joints
    finally:
        for process in reversed(processes):
            _terminate(process)


def _drive(joints: list[str]):  # type: ignore[no-untyped-def]  # reason: rclpy types are unavailable at lint time
    """Build the production HAL + transport pair against the live graph.

    Deliberately a plain `RosControlHAL`, not `OpenArmRealHAL`: the point is the generic
    ros2_control path every robot in that family inherits, and `OpenArmRealHAL` additionally
    demands live CAN buses and its rig's `openarm_`-prefixed joint names, neither of which
    exists here.
    """
    from openral_core.schemas import (
        ControlMode,
        EmbodimentKind,
        JointSpec,
        JointType,
        RobotCapabilities,
        RobotDescription,
        SafetyEnvelope,
    )
    from openral_hal.ros_control import RosControlHAL
    from openral_hal.ros_control_transport import RosControlTransport

    description = RobotDescription(
        name="openarm_left_ros2_control",
        embodiment_kind=EmbodimentKind.MANIPULATOR,
        joints=[
            JointSpec(
                name=name,
                joint_type=JointType.REVOLUTE,
                parent_link="base" if i == 0 else joints[i - 1],
                child_link=name,
            )
            for i, name in enumerate(joints)
        ],
        capabilities=RobotCapabilities(supported_control_modes=[ControlMode.JOINT_POSITION]),
        safety=SafetyEnvelope(),
    )
    return RosControlHAL(description, controller_name=_ARM_CONTROLLER), RosControlTransport


@contextlib.contextmanager
def _driver_node(name: str):  # type: ignore[no-untyped-def]  # reason: rclpy types are unavailable at lint time
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
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            executor.spin_once(timeout_sec=0.02)

    try:
        yield node, spin
    finally:
        executor.remove_node(node)
        node.destroy_node()
        context.shutdown()


def _wait_until(spin, predicate, timeout_s: float) -> bool:  # type: ignore[no-untyped-def]  # reason: rclpy types are unavailable at lint time
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


def test_a_command_reaches_a_real_controller_and_state_comes_back(
    ros2_control_stack: list[str],
) -> None:
    """The whole actuation path, end to end, with only the motors mocked.

    This is the test that would have caught the no-op `publish_fn`: it asserts the arm's
    measured position *changed to the commanded value*, which is only true if a correctly
    typed message reached a real `JointTrajectoryController`.
    """
    from openral_core.schemas import Action, ControlMode

    joints = ros2_control_stack
    hal, transport_cls = _drive(joints)

    with _driver_node("openral_ros2_control_sim") as (node, spin):
        transport = transport_cls(
            node,
            command_topics=list(hal.command_bindings()),
            joint_names=hal.ros2_control_joint_names(),
            joint_state_topic=hal.joint_state_topic,
            command_kinds=hal.command_bindings(),
        )
        hal.attach_transport(transport.publish, transport.state, transport.last_arrival)
        hal.connect()

        assert _wait_until(spin, lambda: transport.last_arrival() != 0.0, 15.0), (
            f"No sensor_msgs/JointState arrived on {hal.joint_state_topic} — "
            "joint_state_broadcaster is active but the subscriber never matched."
        )
        assert transport.missing_joints() == [], (
            "The state topic never reported these joints, so the by-name merge would "
            f"silently zero-fill them: {transport.missing_joints()}"
        )

        # Freshness is a safety control (CLAUDE.md §1.1): against a live broadcaster this must
        # not raise, and it did before the arrival clock replaced the connect clock.
        spin(1.0)
        hal.read_state()

        target = _targets_within_limits(joints)
        hal.send_action(
            Action(
                control_mode=ControlMode.JOINT_POSITION,
                horizon=1,
                joint_targets=[target],
                stamp_ns=time.time_ns(),
            )
        )

        def _arrived() -> bool:
            reached = hal.read_state()
            return max(abs(a - b) for a, b in zip(reached.position, target, strict=True)) < 1e-3

        assert _wait_until(spin, _arrived, 10.0), (
            f"Commanded {target} but the controller reports "
            f"{list(hal.read_state().position)}. The command did not reach the controller, "
            "or reached it in a form it ignored."
        )
        assert hal.read_state().name == joints


def test_the_joint_state_merge_is_by_name_not_by_arrival_order(
    ros2_control_stack: list[str],
) -> None:
    """`/joint_states` carries both arms; only the commanded seven may be read back.

    `joint_state_broadcaster` publishes every joint the resource manager owns — 18 here, both
    `<ros2_control>` blocks — in an order no publisher guarantees. A positional read would
    therefore return another limb's values under this arm's names.
    """
    from sensor_msgs.msg import JointState as RosJointState

    joints = ros2_control_stack
    hal, transport_cls = _drive(joints)
    seen: list[list[str]] = []

    with _driver_node("openral_ros2_control_names") as (node, spin):
        transport = transport_cls(
            node,
            command_topics=list(hal.command_bindings()),
            joint_names=hal.ros2_control_joint_names(),
            joint_state_topic=hal.joint_state_topic,
            command_kinds=hal.command_bindings(),
        )
        node.create_subscription(
            RosJointState, "/joint_states", lambda m: seen.append(list(m.name)), 10
        )
        hal.attach_transport(transport.publish, transport.state, transport.last_arrival)
        hal.connect()

        assert _wait_until(spin, lambda: bool(seen), 15.0), (
            "joint_state_broadcaster published nothing."
        )
        published = set(seen[-1])
        assert published > set(joints), (
            "This assertion is only meaningful while the broadcaster publishes more joints "
            f"than this controller owns; it published {sorted(published)}."
        )
        state = hal.read_state()
        assert state.name == joints, "read_state must project onto the HAL's own joints only."
