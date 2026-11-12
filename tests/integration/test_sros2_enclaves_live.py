# SPDX-License-Identifier: Apache-2.0
"""Live DDS-Security coverage for the ADR-0130 enclaves.

A keystore written by the real ``openral security init`` (franka_panda: its
manifest adds ``/franka_arm_controller/joint_trajectory``) is put in front of
real participants, one process per enclave, on Cyclone DDS in Enforce mode:

1. every enclave can or cannot create the publishers / service clients the
   ADR's table says, enforced by the DDS access-control plugin (a denied
   writer or reader fails to create);
2. a ``/openral/safe_action`` chunk from the kernel enclave reaches a
   HAL-enclave subscriber, and a participant without keys neither receives it
   nor gets its own chunk through.

Gated on ``OPENRAL_TEST_ROS_LIVE=1``, listed in ``scripts/ros_live_tests.sh``.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys

import pytest

_LIVE_ROS = bool(os.getenv("OPENRAL_TEST_ROS_LIVE"))
pytestmark = pytest.mark.skipif(
    not _LIVE_ROS,
    reason="live rclpy graph — set OPENRAL_TEST_ROS_LIVE=1 and source install/setup.bash first.",
)

#: A per-PID test domain (50–99 is reserved for tests, `RESERVED_DOMAINS`).
_DOMAIN = str(50 + os.getpid() % 50)

#: (kind, name): "pub" = create a publisher, "cli" = a service client, "act" = an action client.
_SURFACES: list[tuple[str, str]] = [
    ("pub", "/openral/safe_action"),
    ("pub", "/openral/estop_cleared"),
    ("pub", "/franka_arm_controller/joint_trajectory"),  # the manifest's command_topic
    ("pub", "/scaled_joint_trajectory_controller/joint_trajectory"),
    ("pub", "/forward_position_controller/commands"),
    ("pub", "/joint_states"),
    ("pub", "/openral_hal_franka_panda/joint_states"),
    ("pub", "/openral/estop"),
    ("pub", "/openral/prompt"),
    ("cli", "/openral/estop_reset"),
    ("cli", "/openral_safety_kernel/change_state"),
    ("cli", "/openral_safety_kernel/set_parameters"),
    ("cli", "/openral_hal_franka_panda/change_state"),
    ("cli", "/openral_deadman_watchdog/set_parameters"),
    ("cli", "/safety_controller/set_parameters"),
    ("cli", "/controller_manager/switch_controller"),
    ("cli", "/openral_segmenter/change_state"),
    ("act", "/arm_controller/follow_joint_trajectory"),
]

_ALL = {name for _, name in _SURFACES}
_COMMANDS = {
    "/openral/safe_action",
    "/openral/estop_cleared",
    "/franka_arm_controller/joint_trajectory",
    "/scaled_joint_trajectory_controller/joint_trajectory",
    "/forward_position_controller/commands",
}
_STATES = {"/joint_states", "/openral_hal_franka_panda/joint_states"}
_LIFECYCLE = {
    "/openral_safety_kernel/change_state",
    "/openral_safety_kernel/set_parameters",
    "/openral_hal_franka_panda/change_state",
    "/openral_deadman_watchdog/set_parameters",
    "/safety_controller/set_parameters",
}
_ACTION = {"/arm_controller/follow_joint_trajectory"}
_CONTROLLER = {"/controller_manager/switch_controller"}
_RESET = {"/openral/estop_reset"}

#: What each enclave may create, per the ADR-0130 table.
_ALLOWED: dict[str, set[str]] = {
    "/openral/safety_kernel": _ALL,
    "/openral/hal": _ALL - {"/openral/safe_action", "/openral/estop_cleared"} - _RESET - _LIFECYCLE,
    "/openral/launch": _ALL - _COMMANDS - _STATES - _ACTION,
    "/openral/operator": (
        _ALL
        - (_COMMANDS - {"/openral/estop_cleared"})
        - _STATES
        - _ACTION
        - _CONTROLLER
        - (_LIFECYCLE - {"/safety_controller/set_parameters"})
    ),
    "/": _ALL - _COMMANDS - _STATES - _ACTION - _CONTROLLER - _RESET - _LIFECYCLE,
}

_CREATE_PROBE = """
import json, sys, rclpy
from rclpy.action import ActionClient
from std_msgs.msg import String
from std_srvs.srv import Trigger
from control_msgs.action import FollowJointTrajectory
rclpy.init()
node = rclpy.create_node("openral_sros2_probe")
result = {}
for kind, name in json.loads(sys.argv[1]):
    try:
        if kind == "pub":
            node.create_publisher(String, name, 10)
        elif kind == "sub":
            node.create_subscription(String, name, lambda _m: None, 10)
        elif kind == "cli":
            node.create_client(Trigger, name)
        elif kind == "srv":
            node.create_service(Trigger, name, lambda _req, res: res)
        else:
            ActionClient(node, FollowJointTrajectory, name)
        result[kind + " " + name] = True
    except Exception:  # rclpy raises RCLError when the access-control plugin refuses
        result[kind + " " + name] = False
print(json.dumps(result))
"""

_TRAFFIC_PROBE = """
import sys, time, rclpy
from openral_msgs.msg import ActionChunk
role, seconds = sys.argv[1], float(sys.argv[2])
rclpy.init()
node = rclpy.create_node("openral_sros2_" + role)
count = [0]
if role == "pub":
    pub = node.create_publisher(ActionChunk, "/openral/safe_action", 10)
else:
    node.create_subscription(
        ActionChunk, "/openral/safe_action", lambda _m: count.__setitem__(0, count[0] + 1), 10
    )
deadline = time.monotonic() + seconds
while time.monotonic() < deadline:
    if role == "pub":
        pub.publish(ActionChunk())
    rclpy.spin_once(node, timeout_sec=0.1)
print(count[0])
"""


@pytest.fixture(scope="module")
def secured_env(tmp_path_factory: pytest.TempPathFactory) -> dict[str, str]:
    """The env of a franka_panda graph under SROS2 Enforce, keystore from the real CLI."""
    pytest.importorskip("rclpy")
    pytest.importorskip("control_msgs.action")
    pytest.importorskip("openral_msgs.msg")
    if shutil.which("ros2") is None:
        pytest.skip("needs a sourced ROS 2 with sros2")
    keystore = tmp_path_factory.mktemp("sros2") / "franka_panda"
    init = subprocess.run(
        [
            sys.executable,
            "-c",
            "from openral_cli.main import app; app()",
            "security",
            "init",
            "franka_panda",
            "--keystore",
            str(keystore),
            "--domain-id",
            _DOMAIN,
        ],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert init.returncode == 0, init.stdout + init.stderr
    env = {k: v for k, v in os.environ.items() if not k.startswith(("ROS_SECURITY_", "ROS_STATIC"))}
    env.update(
        RMW_IMPLEMENTATION="rmw_cyclonedds_cpp",
        ROS_DOMAIN_ID=_DOMAIN,
        ROS_AUTOMATIC_DISCOVERY_RANGE="LOCALHOST",
        ROS_SECURITY_ENABLE="true",
        ROS_SECURITY_STRATEGY="Enforce",
        ROS_SECURITY_KEYSTORE=str(keystore),
    )
    return env


def _run(env: dict[str, str], enclave: str | None, code: str, *args: str) -> str:
    probe_env = dict(env)
    if enclave is None:  # a participant without keys
        probe_env["ROS_SECURITY_ENABLE"] = "false"
    else:
        probe_env["ROS_SECURITY_ENCLAVE_OVERRIDE"] = enclave
    done = subprocess.run(
        [sys.executable, "-c", code, *args],
        env=probe_env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert done.returncode == 0, done.stdout + done.stderr
    return done.stdout.strip().splitlines()[-1]


@pytest.mark.parametrize("enclave", sorted(_ALLOWED))
def test_each_enclave_gets_exactly_its_surfaces(secured_env: dict[str, str], enclave: str) -> None:
    created = json.loads(_run(secured_env, enclave, _CREATE_PROBE, json.dumps(_SURFACES)))
    assert {key.split(" ", 1)[1] for key, ok in created.items() if ok} == _ALLOWED[enclave]


@pytest.mark.parametrize("enclave", sorted(_ALLOWED))
def test_every_enclave_still_reads_and_serves_the_protected_surfaces(
    secured_env: dict[str, str], enclave: str
) -> None:
    """Only writing and calling are denied: the HAL reads safe_action, the watchdog serves
    its own set_parameters. Cyclone refused both until the keystore listed that side first
    (``openral_cli.security._allow_the_other_side``)."""
    probes = [("sub", name) for kind, name in _SURFACES if kind == "pub"]
    probes += [("srv", name) for kind, name in _SURFACES if kind == "cli"]
    created = json.loads(_run(secured_env, enclave, _CREATE_PROBE, json.dumps(probes)))
    assert [key for key, ok in created.items() if not ok] == []


def test_kernel_chunks_reach_the_hal_and_no_keyless_participant(
    secured_env: dict[str, str],
) -> None:
    """Encrypted, authenticated traffic: keyholders talk, a keyless participant is shut out."""

    def start(enclave: str | None, role: str) -> subprocess.Popen[str]:
        env = dict(secured_env)
        if enclave is None:
            env["ROS_SECURITY_ENABLE"] = "false"
        else:
            env["ROS_SECURITY_ENCLAVE_OVERRIDE"] = enclave
        return subprocess.Popen(
            [sys.executable, "-c", _TRAFFIC_PROBE, role, "8"],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

    hal = start("/openral/hal", "sub")
    keyless = start(None, "sub")
    kernel = start("/openral/safety_kernel", "pub")
    received = {}
    for name, proc in (("hal", hal), ("keyless", keyless), ("kernel", kernel)):
        out, err = proc.communicate(timeout=60)
        assert proc.returncode == 0, out + err
        received[name] = int(out.strip().splitlines()[-1])
    assert received["hal"] > 0
    assert received["keyless"] == 0

    # And the other direction: a keyless publisher reaches no secured subscriber.
    hal = start("/openral/hal", "sub")
    intruder = start(None, "pub")
    out, err = hal.communicate(timeout=60)
    intruder.communicate(timeout=60)
    assert hal.returncode == 0, out + err
    assert int(out.strip().splitlines()[-1]) == 0
