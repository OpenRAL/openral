"""Shared helpers for tests that spawn the real ``safety_kernel_node``.

The kernel binary is an ``rclcpp::spin()`` loop that only exits cleanly on
``SIGINT``. Plain ``proc.terminate()`` sends ``SIGTERM`` which the rclcpp
executor may ignore, leaking the subprocess across pytest sessions and
contaminating DDS discovery for the next test. ``terminate_kernel()``
sends SIGINT first, waits, then escalates to SIGKILL if needed.

Each test that calls these should also set ``ROS_DOMAIN_ID`` to an
isolated value (50..101 hash of PID) so even if a prior process is
still draining, our discovery is on a different domain.

This module also hosts the two helpers every kernel-twin test needs:

* :func:`activate_kernel_node` — drive the safety_kernel_node lifecycle
  from ``unconfigured`` → ``active`` via ``ChangeState``.
* :func:`kernel_param_args` — render the per-field ROS parameter
  ``-p key:=value`` argv list the kernel reads on ``on_configure`` from
  a real :class:`~openral_core.RobotDescription` (via
  :func:`openral_safety.envelope_loader.compute_intersection` +
  :func:`kernel_params_from_envelope`). ADR-0020 PR-K — the kernel has
  no envelope-file path anymore; everything flows through ROS
  parameters.
"""

from __future__ import annotations

import contextlib
import math
import os
import shutil
import signal
import subprocess
import time
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from openral_core import RobotDescription


def isolated_domain_id() -> int:
    """Return a deterministic ROS_DOMAIN_ID isolated per pytest PID."""
    return 50 + (os.getpid() % 50)


def _format_param(value: object) -> str:
    """Render a Python value as a ros2-cli ``-p key:=value`` value string.

    ``ros2 run ... --ros-args -p`` accepts scalars and homogeneous
    arrays. Arrays are passed as the literal Python-list syntax
    (``[1.0, 2.0]``); the ros2 CLI parses them into the matching
    ``double_array`` parameter type. Booleans are case-sensitive
    (``true`` / ``false``).
    """
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple)):
        items: list[str] = []
        for v in value:
            if isinstance(v, float):
                if math.isinf(v):
                    items.append(".inf" if v > 0 else "-.inf")
                else:
                    items.append(repr(float(v)))
            else:
                items.append(repr(v))
        return "[" + ", ".join(items) + "]"
    if isinstance(value, float):
        if math.isinf(value):
            return ".inf" if value > 0 else "-.inf"
        return repr(float(value))
    if isinstance(value, str):
        # ros2-cli parses each `-p key:=value` value as YAML. An empty
        # string renders as a bare `key:=`, which ROS 2 Jazzy's rcl
        # rejects with "Couldn't parse parameter override rule" and the
        # kernel aborts at init. Emit a quoted empty YAML string instead.
        # `compute_intersection(desc, skill=None)` leaves rskill_id /
        # skill_revision empty, so every description-sourced kernel hits
        # this (dict-sourced envelopes simply omit the keys).
        return value if value else "''"
    return str(value)


def kernel_param_args_from_dict(params: dict[str, object]) -> list[str]:
    """Format a dict of kernel parameters as ``-p key:=value`` argv pairs.

    Used by tests that hand-roll specific envelope values (e.g. a 6-DoF
    envelope that exercises a particular violation case) rather than
    drive the kernel from a real ``RobotDescription``.
    """
    argv: list[str] = []
    for key, value in params.items():
        if isinstance(value, (list, tuple)) and len(value) == 0:
            continue  # ros2-cli can't type `-p key:=[]`; the kernel declares it defaulted-empty
        argv += ["-p", f"{key}:={_format_param(value)}"]
    return argv


def kernel_param_args(robot_description: RobotDescription) -> list[str]:
    """Return the ``--ros-args -p key:=value`` argv list for the kernel.

    Mirrors what ``sim_e2e.launch.py`` does in-process: synthesise the
    envelope from the robot manifest, then emit each canonical field as
    a ROS parameter (ADR-0020 PR-K). Callers extend the list with their
    own scalars (e.g. ``estop_reset_cooldown_s``).

    Args:
        robot_description: The robot manifest to load. Skill clamp is
            never applied here (the reasoner picks rSkills at runtime;
            the kernel's boot envelope is the robot ceiling).

    Returns:
        A list of arguments suitable for ``subprocess.Popen([... "--ros-args",
        *kernel_param_args(desc), "-p", "estop_reset_cooldown_s:=0.1"])``.
    """
    from openral_safety.envelope_loader import (  # reason: defer ROS import
        compute_intersection,
        kernel_params_from_envelope,
    )

    intersection = compute_intersection(robot_description, None)
    params = kernel_params_from_envelope(intersection)
    argv: list[str] = []
    for key, value in params.items():
        if isinstance(value, (list, tuple)) and len(value) == 0:
            continue  # ros2-cli can't type `-p key:=[]`; the kernel declares it defaulted-empty
        argv += ["-p", f"{key}:={_format_param(value)}"]
    return argv


def start_kernel(
    source: RobotDescription | dict[str, object],
    node_name: str,
    domain_id: int | None = None,
    *,
    estop_reset_cooldown_s: float = 0.1,
    log_path: os.PathLike[str] | str | None = None,
) -> Any:
    """Launch ``safety_kernel_node`` on an isolated DDS domain.

    Args:
        source: Either a :class:`~openral_core.RobotDescription` (the
            envelope is synthesised from it via
            :func:`kernel_param_args`) or a raw parameter dict (passed
            through :func:`kernel_param_args_from_dict`). Tests that
            need to exercise a specific envelope edge case pass a dict;
            tests that drive a real robot pass the manifest.
        node_name: Unique node name (collisions on the same domain break
            the lifecycle service).
        domain_id: ROS_DOMAIN_ID for the kernel; defaults to
            :func:`isolated_domain_id`.
        estop_reset_cooldown_s: Tests use a short cooldown (≤100 ms).
        log_path: When given, redirects stdout+stderr to this file so
            the parent process can surface the kernel's logs on failure.

    Returns:
        ``subprocess.Popen`` for the kernel; callers should pass it to
        :func:`terminate_kernel` in the test's finally block.
    """
    if shutil.which("ros2") is None:
        pytest.skip("ros2 binary not on PATH; source install/setup.bash first")
    if domain_id is None:
        domain_id = isolated_domain_id()
    os.environ["ROS_DOMAIN_ID"] = str(domain_id)
    env = {**os.environ, "ROS_DOMAIN_ID": str(domain_id)}

    if log_path is not None:
        log_fp = open(log_path, "wb")  # reason: kernel reads after this scope
        stdout: Any = log_fp
        stderr: Any = subprocess.STDOUT
    else:
        stdout = subprocess.DEVNULL
        stderr = subprocess.DEVNULL

    param_args = (
        kernel_param_args_from_dict(source)
        if isinstance(source, dict)
        else kernel_param_args(source)
    )

    return subprocess.Popen(
        [
            "ros2",
            "run",
            "openral_safety_kernel",
            "safety_kernel_node",
            "--ros-args",
            "-r",
            f"__node:={node_name}",
            *param_args,
            "-p",
            f"estop_reset_cooldown_s:={estop_reset_cooldown_s}",
        ],
        env=env,
        stdout=stdout,
        stderr=stderr,
        # Start the kernel in its own process group so we can interrupt
        # the whole group with a single signal, matching how the F7
        # ``openral record`` wrapper handles rosbag2 (CLAUDE.md §1.4).
        start_new_session=True,
    )


def terminate_kernel(proc: Any, *, sigint_grace_s: float = 2.0) -> None:
    """SIGINT the kernel's process group, then escalate to SIGKILL.

    rclcpp::spin() exits on SIGINT but ignores SIGTERM, so plain
    ``proc.terminate()`` leaks the process. We send SIGINT to the
    whole process group (including any subprocesses the kernel may
    have spawned), wait briefly, then SIGKILL if still alive.
    """
    if proc.poll() is not None:
        return
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(proc.pid, signal.SIGINT)
    try:
        proc.wait(timeout=sigint_grace_s)
        return
    except subprocess.TimeoutExpired:
        pass
    # Escalate.
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(proc.pid, signal.SIGKILL)
    with contextlib.suppress(subprocess.TimeoutExpired):
        proc.wait(timeout=2.0)


def activate_kernel_node(
    node_name: str,
    helper: Any,
    *,
    service_timeout_s: float = 10.0,
    transition_timeout_s: float = 5.0,
) -> bool:
    """Drive the safety_kernel_node lifecycle from unconfigured to active.

    The kernel ships as a managed (lifecycle) node, so callers must
    explicitly ``CONFIGURE`` then ``ACTIVATE`` it before its publishers /
    subscribers come online. This helper performs that handshake from a
    plain ``rclpy`` helper node and returns whether activation succeeded
    inside ``transition_timeout_s`` per transition.
    """
    import rclpy
    from lifecycle_msgs.msg import Transition
    from lifecycle_msgs.srv import ChangeState

    client = helper.create_client(ChangeState, f"/{node_name}/change_state")
    if not client.wait_for_service(timeout_sec=service_timeout_s):
        return False
    for t in (Transition.TRANSITION_CONFIGURE, Transition.TRANSITION_ACTIVATE):
        req = ChangeState.Request()
        req.transition.id = t
        fut = client.call_async(req)
        deadline = time.time() + transition_timeout_s
        while time.time() < deadline and not fut.done():
            rclpy.spin_once(helper, timeout_sec=0.05)
        if not fut.done() or not fut.result().success:  # type: ignore[union-attr]
            return False
    return True
