"""Live ROS coverage for issue #557: ``openral prompt`` / ``openral estop reset --robot ID``.

A real deploy runs on ``real_domain_id(robot_id)`` with LOCALHOST discovery
(``openral_cli._dds_scope.confine_real_scope``). Run from a fresh terminal — no
``ROS_DOMAIN_ID`` exported, ``ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET`` as ROS's
``setup.bash`` leaves it — both commands used to land on domain 0 and reach
nothing: the prompt was lost and the robot stayed latched. This puts a real
graph on the robot's derived domain (the real ``SafetyPassthroughNode``, which
serves the C++ kernel's ``/openral/estop_reset`` contract, a subscriber on
``/openral/estop_cleared`` and one on the prompt-router's CLI input) and runs
the real CLI as a subprocess from that fresh environment:

1. without ``--robot`` (the old behaviour) neither reaches the graph;
2. with ``--robot`` but no ``openral_hal_<robot_id>`` on the graph, both refuse
   (exit 3): the derived domain names a graph, not a robot, and ids collide;
3. an exported ``ROS_DOMAIN_ID`` still wins over ``--robot`` and is reported as
   overriding the robot's domain;
4. with ``--robot`` and the robot's HAL node up, the prompt arrives, the kernel
   clears and the clear is broadcast.

The HAL is represented by its graph identity only: a node named
``openral_hal_<robot_id>`` (what the deploy names it), which also carries the
test's subscribers. Nothing here actuates.

The "wrong" shell domain is ``_OTHER_DOMAIN`` on LOCALHOST rather than a truly
fresh shell's domain 0 on SUBNET: on a shared host (and its LAN) domain 0 is
where every unconfigured test and tool lands, and the first draft of this test
reset a parallel test's kernel there and broadcast ``/openral/estop_cleared``
onto it — the hazard of #557 in miniature.

Gated on ``OPENRAL_TEST_ROS_LIVE=1``, listed in ``scripts/ros_live_tests.sh``.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from collections.abc import Callable

import pytest

_LIVE_ROS = bool(os.getenv("OPENRAL_TEST_ROS_LIVE"))
pytestmark = pytest.mark.skipif(
    not _LIVE_ROS,
    reason="live rclpy graph — set OPENRAL_TEST_ROS_LIVE=1 and source install/setup.bash first.",
)

#: A real manifest id (robots/ur5e). Its derived domain is what the test listens on.
_ROBOT_ID = "ur5e"

#: The domain the CLI's shell is on when it is not the robot's: in the hash's
#: free range (10-49 minus 42) and no in-tree robot's derived domain.
_OTHER_DOMAIN = "31"


def _wait_until(predicate: Callable[[], object], *, timeout_s: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return bool(predicate())


def _terminal_env(domain: str | None) -> dict[str, str]:
    """Another terminal on the robot host: ROS sourced, nothing exported for the deploy.

    ``domain`` is the shell's own ``ROS_DOMAIN_ID`` (``None``: unset). The range is
    ``SUBNET``, what ``setup.bash`` exports; the CLI confines it under ``--robot``.
    """
    env = os.environ.copy()
    for name in (
        "ROS_DOMAIN_ID",
        "ROS_LOCALHOST_ONLY",
        "ROS_STATIC_PEERS",
        "ROS_DISCOVERY_SERVER",
        "OPENRAL_ALLOW_NETWORK_ROS",
    ):
        env.pop(name, None)
    env["ROS_AUTOMATIC_DISCOVERY_RANGE"] = "SUBNET"
    if domain is not None:
        env["ROS_DOMAIN_ID"] = domain
        env["ROS_AUTOMATIC_DISCOVERY_RANGE"] = "LOCALHOST"  # never onto the LAN
    return env


def _openral(*args: str, domain: str | None = None) -> subprocess.CompletedProcess[str]:
    """Run the real CLI in its own process (its own rclpy context and DDS participant)."""
    return subprocess.run(
        [sys.executable, "-c", "from openral_cli.main import app; app()", *args],
        capture_output=True,
        text=True,
        timeout=90,
        env=_terminal_env(domain),
        check=False,
    )


def _nodes_on(domain: str) -> list[str]:
    """Every node already on ``domain`` (LOCALHOST), seen from a throwaway context."""
    import rclpy
    from rclpy.executors import SingleThreadedExecutor

    context = rclpy.Context()
    rclpy.init(context=context, domain_id=int(domain))
    try:
        probe = rclpy.create_node("openral_issue557_probe", context=context)
        executor = SingleThreadedExecutor(context=context)
        executor.add_node(probe)
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            executor.spin_once(timeout_sec=0.1)
        names = [n for n, _ns in probe.get_node_names_and_namespaces() if n != probe.get_name()]
        executor.shutdown()
        probe.destroy_node()
        return names
    finally:
        rclpy.try_shutdown(context=context)


def test_prompt_and_estop_reset_join_the_robots_derived_domain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rclpy = pytest.importorskip("rclpy")
    pytest.importorskip("std_srvs.srv")
    pytest.importorskip("openral_msgs.msg")
    from openral_cli._dds_scope import hal_node_name, real_domain_id
    from openral_msgs.msg import PromptStamped
    from openral_safety.supervisor_node import SafetyPassthroughNode
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.lifecycle import TransitionCallbackReturn
    from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
    from std_msgs.msg import Empty

    domain = real_domain_id(_ROBOT_ID)
    # The deploy's graph: the derived domain, host-confined (confine_real_scope).
    monkeypatch.setenv("ROS_DOMAIN_ID", domain)
    monkeypatch.setenv("ROS_AUTOMATIC_DISCOVERY_RANGE", "LOCALHOST")
    for name in ("ROS_LOCALHOST_ONLY", "ROS_STATIC_PEERS", "ROS_DISCOVERY_SERVER"):
        monkeypatch.delenv(name, raising=False)
    # A real ur5e deploy or a concurrent run of this test on this host would be
    # reset by the CLI below; never act on a graph this test did not build.
    for probed in (domain, _OTHER_DOMAIN):
        if occupants := _nodes_on(probed):
            pytest.skip(f"ROS_DOMAIN_ID={probed} is in use on this host: {occupants}")

    rclpy.init()
    safety = SafetyPassthroughNode(node_name="openral_issue557_kernel")
    safety.set_parameters(
        [
            rclpy.parameter.Parameter("n_dof", value=6),
            rclpy.parameter.Parameter("estop_reset_cooldown_s", value=0.0),
        ]
    )
    helper = rclpy.create_node("openral_issue557_helper")
    qos = QoSProfile(
        reliability=QoSReliabilityPolicy.RELIABLE,
        durability=QoSDurabilityPolicy.VOLATILE,
        depth=10,
    )
    prompts: list[str] = []
    cleared: list[int] = []
    helper.create_subscription(
        PromptStamped, "/openral/prompt_in/cli", lambda m: prompts.append(m.text), qos
    )
    helper.create_subscription(Empty, "/openral/estop_cleared", lambda _m: cleared.append(1), qos)
    estop_pub = helper.create_publisher(Empty, "/openral/estop", 10)
    executor = MultiThreadedExecutor(num_threads=4)
    for node in (safety, helper):
        executor.add_node(node)
    stop = threading.Event()

    def _spin() -> None:
        while not stop.is_set():
            executor.spin_once(timeout_sec=0.05)

    spinner = threading.Thread(target=_spin, daemon=True)
    spinner.start()
    hal = None
    try:
        assert safety.trigger_configure() == TransitionCallbackReturn.SUCCESS
        assert safety.trigger_activate() == TransitionCallbackReturn.SUCCESS
        assert _wait_until(lambda: (estop_pub.publish(Empty()) or True) and safety._estopped), (
            "the e-stop never latched the kernel"
        )

        # 1. Without --robot: the shell's own scope, not the robot's — the trap of #557.
        lost = _openral(
            "prompt", "--discovery-wait-s", "2", "--wait-s", "0.5", "lost", domain=_OTHER_DOMAIN
        )
        assert lost.returncode == 0, lost.stdout + lost.stderr
        assert "matched_subs=0" in lost.stdout, lost.stdout
        unreached = _openral(
            "estop", "reset", "--service-timeout-s", "3", "--discovery-wait-s", "1",
            domain=_OTHER_DOMAIN,
        )  # fmt: skip
        assert unreached.returncode == 3, unreached.stdout + unreached.stderr

        # 2. --robot on the right domain, but the robot's HAL is not on it: refused.
        for args in (
            ("prompt", "--robot", _ROBOT_ID, "--discovery-wait-s", "2", "no hal"),
            ("estop", "reset", "--robot", _ROBOT_ID, "--discovery-wait-s", "2"),
        ):
            refused = _openral(*args)
            assert refused.returncode == 3, refused.stdout + refused.stderr
            assert f"{hal_node_name(_ROBOT_ID)} is not on ROS_DOMAIN_ID={domain}" in (
                refused.stderr
            )

        # 3. An exported domain wins over --robot, as it does for the deploy, and
        # the override is said out loud; the robot's HAL is not there, so: refused.
        overridden = _openral(
            "estop", "reset", "--robot", _ROBOT_ID, "--discovery-wait-s", "1",
            domain=_OTHER_DOMAIN,
        )  # fmt: skip
        assert overridden.returncode == 3, overridden.stdout + overridden.stderr
        assert f"ROS_DOMAIN_ID={_OTHER_DOMAIN} (exported, overrides robot" in overridden.stderr
        assert f"'s {domain}" in overridden.stderr, overridden.stderr

        time.sleep(0.5)  # anything sent above would have landed by now
        assert prompts == [] and cleared == [] and safety._estopped is True

        # 4. With --robot and the robot's HAL up: no export needed.
        hal = rclpy.create_node(hal_node_name(_ROBOT_ID))
        executor.add_node(hal)
        sent = _openral("prompt", "--robot", _ROBOT_ID, "--wait-s", "0.5", "pick the red cube")
        assert sent.returncode == 0, sent.stdout + sent.stderr
        assert f"ROS_DOMAIN_ID={domain} (robot '{_ROBOT_ID}')" in sent.stderr, sent.stderr
        assert "ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST" in sent.stderr, sent.stderr
        assert _wait_until(lambda: prompts == ["pick the red cube"]), prompts

        reset = _openral("estop", "reset", "--robot", _ROBOT_ID, "--discovery-wait-s", "5")
        assert reset.returncode == 0, reset.stdout + reset.stderr
        assert f"ROS_DOMAIN_ID={domain} (robot '{_ROBOT_ID}')" in reset.stderr, reset.stderr
        assert safety._estopped is False
        assert _wait_until(lambda: len(cleared) > 0), "the clear never reached the graph"
    finally:
        stop.set()
        spinner.join(timeout=5.0)
        executor.shutdown()
        for node in (hal, helper, safety):
            if node is not None:
                node.destroy_node()
        rclpy.try_shutdown()
