"""Defense-in-depth E-stop, proved on a real multi-process graph.

The claim the watchdog packages make — "independent of the in-band safety node
and the C++ safety kernel, so a crash in either still triggers a brake event"
(CLAUDE.md §3 *Safety*: never a path where a Python crash leaves motors
energised) — is a **process-isolation** claim. It cannot be shown in-process:
only a graph where the in-band node is a separate OS process that can be
SIGKILLed says anything about it.

Graph under test, shaped like the one
``packages/openral_rskill_ros/launch/deploy_e2e.launch.py`` now spawns:

* ``openral_safety``'s ``supervisor_node`` — the in-band gate that turns
  ``/openral/candidate_action`` into ``/openral/safe_action``. This is the
  process that gets killed.
* ``openral_safety_watchdog``'s ``deadman_watchdog_node``, gated on the same
  execution-window topic the deploy graph wires.
* ``openral_human_estop``'s ``forwarder_node``.

What the tests prove:

1. With the window open and a live 30 Hz chunk stream through the real in-band
   node, the watchdog is silent. SIGKILL that node and ``/openral/estop`` fires
   from a *different process*, carrying the structured
   ``FailureTrigger(KIND_TIMEOUT, SEVERITY_ABORT)`` + ``TimeoutEvidence`` the
   reasoner needs. The chunk producer keeps publishing across the kill, so the
   only thing that changed is that the in-band safety layer died.
2. The deadman path on its own: ``/openral/safe_action`` goes silent past the
   deadline inside an open window → estop + the same structured trigger.
3. The human forwarder, in the same graph, turns ``/openral/human_estop`` into
   ``/openral/estop`` + ``FailureTrigger(KIND_HUMAN)``.

**Why not ``launch_testing``** (CLAUDE.md §2 names it for this tier): the
workspace pytest config disables the ``launch_testing`` /
``launch_testing_ros`` / ``launch_ros`` plugins globally in
``pyproject.toml``'s ``addopts``, because their
``pytest_launch_collect_makemodule`` hook silently swallows collection of every
ordinary ``test_*.py``. The only lane that could run a ``launch_test`` instead
— ``colcon test`` via ``test-ros2.yml`` — is ``workflow_dispatch``-only while
Actions credits are frozen, so a launch test there would run on no automatic CI
surface at all. This file therefore spawns the same processes the launch
actions would (``get_package_prefix`` → ``lib/<pkg>/<exe>``, ``--ros-args``
remaps and parameters) and runs in the live-ROS lane the ``docker-build``
workflow actually executes. Same processes, same DDS, same IDL, real CI.
``tests/integration/test_sim_idle_stepper_wiring.py`` sets the precedent for
documenting a deliberate deviation from the launch-test form.

Real lifecycle nodes, real ``openral_msgs`` IDL, real DDS, no mocks
(CLAUDE.md §1.11). Gated on ``OPENRAL_TEST_ROS_LIVE=1`` and listed in
``scripts/ros_live_tests.sh``. Locally::

    source /opt/ros/jazzy/setup.bash && just ros2-build
    source install/setup.bash
    just test-ros-live -k estop_watchdog_graph
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from collections.abc import Iterator
from typing import Any

import pytest

_LIVE_ROS = bool(os.getenv("OPENRAL_TEST_ROS_LIVE"))
_LIVE_ROS_REASON = (
    "live multi-process ROS graph — set OPENRAL_TEST_ROS_LIVE=1 in a clean shell "
    "and source install/setup.bash first."
)

pytestmark = pytest.mark.skipif(not _LIVE_ROS, reason=_LIVE_ROS_REASON)

# Node names unique to this test, so a deploy graph running concurrently on the
# same host is never mistaken for — or killed instead of — ours.
_SAFETY_NODE = "openral_safety_estop_graph_test"
_DEADMAN_NODE = "openral_deadman_watchdog_estop_graph_test"
_FORWARDER_NODE = "openral_human_estop_forwarder_estop_graph_test"

# The execution-window topic deploy_e2e.launch.py gates the deadman on: the
# runner's own ExecuteRskill action status. A dead runner cannot publish a
# terminal status, so the window stays open and the silence still fires.
_ARM_STATUS_TOPIC = "/openral/execute_rskill/_action/status"
_SAFETY_STATUS_TOPIC = "/openral/safety_status"

# Well clear of a chunk-publish hiccup on a loaded laptop, small enough to keep
# the test inside the integration tier's budget. The deploy graph runs 8 s
# (sim) / 5 s (real) for reasons packages/openral_safety_watchdog/README.md
# explains; the property under test is identical at either value.
_DEADLINE_S = 1.0

# A fresh rclpy participant needs a couple of seconds of spinning before it
# receives anything; a shorter wait reads as a transport bug when it is only
# DDS discovery latency.
_DISCOVERY_S = 3.0

_CHANNEL_LABEL = "integration_operator"


# ── The graph ────────────────────────────────────────────────────────────────


def _executable(package: str, name: str) -> str:
    """Absolute path of an installed node executable in the colcon overlay."""
    from ament_index_python.packages import get_package_prefix

    return os.path.join(get_package_prefix(package), "lib", package, name)


def _spawn(package: str, executable: str, node_name: str, params: dict[str, str]) -> Any:
    """Start one node as its own OS process, exactly as a launch action would.

    Run through ``sys.executable`` rather than the script shebang so the node
    lands on the workspace venv interpreter (the one that can import
    ``openral_safety`` / ``openral_observability``) regardless of PATH.
    """
    cmd = [sys.executable, _executable(package, executable), "--ros-args", "-r"]
    cmd.append(f"__node:={node_name}")
    for key, value in params.items():
        cmd += ["-p", f"{key}:={value}"]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


@pytest.fixture(scope="module")
def estop_graph() -> Iterator[dict[str, Any]]:
    """Bring up the three-process E-stop graph; tear it down after."""
    procs: dict[str, Any] = {}
    try:
        procs["safety"] = _spawn(
            "openral_safety",
            "supervisor_node",
            _SAFETY_NODE,
            # 3-DoF, no envelope bounds: every well-formed chunk passes through
            # onto /openral/safe_action, which is all this test needs from the
            # in-band node. Envelope enforcement has its own coverage in
            # packages/openral_safety/test/test_supervisor_node.py.
            {"n_dof": "3"},
        )
        procs["deadman"] = _spawn(
            "openral_safety_watchdog",
            "deadman_watchdog_node.py",
            _DEADMAN_NODE,
            {
                "arm_status_topic": _ARM_STATUS_TOPIC,
                "safe_action_deadline_s": str(_DEADLINE_S),
                "first_chunk_deadline_s": str(_DEADLINE_S * 3.0),
                "check_period_s": "0.05",
            },
        )
        procs["forwarder"] = _spawn(
            "openral_human_estop",
            "forwarder_node.py",
            _FORWARDER_NODE,
            {"channel_label": _CHANNEL_LABEL},
        )
        yield procs
    finally:
        for proc in procs.values():
            if proc.poll() is None:
                proc.send_signal(signal.SIGINT)
        for proc in procs.values():
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:  # pragma: no cover - slow teardown
                proc.kill()
                proc.wait(timeout=10)


class _Probe:
    """A single rclpy participant driving and observing the graph."""

    def __init__(self) -> None:
        """Create the probe node + executor and let discovery settle."""
        import rclpy
        from rclpy.executors import SingleThreadedExecutor

        rclpy.init()
        self.node = rclpy.create_node("openral_estop_watchdog_graph_probe")
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.node)
        self.spin_for(_DISCOVERY_S)

    def close(self) -> None:
        """Drop the probe; the fixture owns the node processes."""
        import rclpy

        self.executor.remove_node(self.node)
        self.node.destroy_node()
        rclpy.shutdown()

    def spin_for(self, seconds: float) -> None:
        """Spin the executor for a fixed wall-clock window."""
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            self.executor.spin_once(timeout_sec=0.02)

    def spin_until(self, predicate: Any, *, timeout_s: float) -> bool:
        """Spin until ``predicate()`` holds or the timeout expires."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            self.executor.spin_once(timeout_sec=0.02)
            if predicate():
                return True
        return bool(predicate())

    def call(self, client: Any, request: Any, *, timeout_s: float = 20.0) -> Any:
        """Call a service and spin until it answers.

        Deliberately the probe's OWN executor, not
        ``rclpy.spin_until_future_complete(node, ...)``: that helper builds a
        throwaway executor and attaches the node to it, and tearing it down
        leaves the node detached from this one. Every later ``spin_once`` then
        does nothing, so the probe's subscriptions go silently deaf while its
        publishers keep working — which reads exactly like a transport bug.
        """
        future = client.call_async(request)
        self.executor.spin_until_future_complete(future, timeout_sec=timeout_s)
        return future.result()

    def drive_to_active(self, node_name: str) -> None:
        """CONFIGURE then ACTIVATE ``node_name``; assert it lands ACTIVE.

        The post-call state is the source of truth rather than
        ``response.success`` — Jazzy's first CONFIGURE can answer ``false`` on
        a transition that did happen, which is why
        ``tools/lifecycle_autostart.py`` polls the state too.
        """
        from lifecycle_msgs.msg import Transition
        from lifecycle_msgs.srv import ChangeState, GetState

        change = self.node.create_client(ChangeState, f"/{node_name}/change_state")
        get = self.node.create_client(GetState, f"/{node_name}/get_state")
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline:
            if change.wait_for_service(timeout_sec=1.0) and get.wait_for_service(timeout_sec=1.0):
                break
            self.executor.spin_once(timeout_sec=0.05)
        else:  # pragma: no cover - only on a broken overlay
            raise AssertionError(f"{node_name}: lifecycle services never appeared")

        for transition in (Transition.TRANSITION_CONFIGURE, Transition.TRANSITION_ACTIVATE):
            request = ChangeState.Request()
            request.transition.id = transition
            self.call(change, request)
        state = self.call(get, GetState.Request())
        label = None if state is None else state.current_state.label
        assert label == "active", f"{node_name} did not reach ACTIVE (state={label!r})"


def _status_array(live: bool) -> Any:
    """A GoalStatusArray carrying one live (EXECUTING) or terminal goal."""
    from action_msgs.msg import GoalStatus, GoalStatusArray

    array = GoalStatusArray()
    entry = GoalStatus()
    entry.status = GoalStatus.STATUS_EXECUTING if live else GoalStatus.STATUS_SUCCEEDED
    array.status_list = [entry]
    return array


def _arm_publisher(probe: _Probe) -> Any:
    """Publisher on the action-status topic, at the action server's own QoS."""
    from action_msgs.msg import GoalStatusArray
    from rclpy.qos import qos_profile_action_status_default

    return probe.node.create_publisher(
        GoalStatusArray, _ARM_STATUS_TOPIC, qos_profile_action_status_default
    )


def _close_any_open_window(probe: _Probe, arm_pub: Any) -> None:
    """Publish a terminal goal status, closing any window a prior test left.

    The action-status topic is TRANSIENT_LOCAL, so the last value published by
    an earlier test in this module-scoped graph is delivered to the deadman on
    subscribe. Leaving it EXECUTING means the next test starts inside an open
    window, where the first-chunk budget is already running.
    """
    arm_pub.publish(_status_array(live=False))
    probe.spin_for(0.5)


def _clear_safety_status(probe: _Probe) -> Any:
    """Publish SafetyStatus(latched=False) — the watchdog's real re-arm path."""
    from openral_msgs.msg import SafetyStatus
    from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy

    pub = probe.node.create_publisher(
        SafetyStatus,
        _SAFETY_STATUS_TOPIC,
        QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            depth=1,
        ),
    )
    msg = SafetyStatus()
    msg.latched = False
    msg.drop_reason = SafetyStatus.DROP_NONE
    msg.detail = "integration test recovery"
    pub.publish(msg)
    return pub


def _qos(depth: int) -> Any:
    """RELIABLE + VOLATILE at ``depth`` — the safety/control QoS class."""
    from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy

    return QoSProfile(
        reliability=QoSReliabilityPolicy.RELIABLE,
        durability=QoSDurabilityPolicy.VOLATILE,
        depth=depth,
    )


def _chunk(chunk_cls: Any) -> Any:
    """A well-formed 3-DoF single-step JOINT_POSITION chunk."""
    chunk = chunk_cls()
    chunk.control_mode = 0
    chunk.horizon = 1
    chunk.n_dof = 3
    chunk.flat = [0.0, 0.0, 0.0]
    return chunk


def _assert_timeout_evidence(trigger: Any, failure_cls: Any) -> None:
    """The structured half of a deadman brake."""
    assert trigger.severity == failure_cls.SEVERITY_ABORT
    evidence = json.loads(trigger.evidence_json)
    assert evidence["kind"] == "timeout"
    assert evidence["operation"] == "safe_action"
    assert evidence["deadline_s"] == pytest.approx(_DEADLINE_S)
    assert evidence["elapsed_s"] > _DEADLINE_S


# ── Tests ────────────────────────────────────────────────────────────────────


def test_deadman_fires_after_the_in_band_safety_node_is_killed(
    estop_graph: dict[str, Any],
) -> None:
    """Kill the in-band gate mid-stream; a different process brakes.

    This is the whole reason the watchdog packages exist, and until this PR the
    deploy graph never launched them — so nothing in the repo asserted it end
    to end.
    """
    from openral_msgs.msg import ActionChunk, FailureTrigger
    from std_msgs.msg import Empty

    probe = _Probe()
    try:
        for name in (_SAFETY_NODE, _DEADMAN_NODE, _FORWARDER_NODE):
            probe.drive_to_active(name)

        candidate_pub = probe.node.create_publisher(
            ActionChunk, "/openral/candidate_action", _qos(1)
        )
        arm_pub = _arm_publisher(probe)
        safe_seen: list[Any] = []
        estop_seen: list[Any] = []
        failures: list[Any] = []
        probe.node.create_subscription(
            ActionChunk, "/openral/safe_action", safe_seen.append, _qos(1)
        )
        probe.node.create_subscription(Empty, "/openral/estop", estop_seen.append, _qos(10))
        probe.node.create_subscription(
            FailureTrigger, "/openral/failure/safety", failures.append, _qos(50)
        )
        probe.spin_for(2.0)  # publisher/subscriber matching

        arm_pub.publish(_status_array(live=True))

        # Phase 1 — healthy graph. Chunks flow candidate → in-band node → safe,
        # and the watchdog must stay out of the way for well over a deadline.
        healthy_until = time.monotonic() + (_DEADLINE_S * 2.0)
        while time.monotonic() < healthy_until:
            candidate_pub.publish(_chunk(ActionChunk))
            probe.executor.spin_once(timeout_sec=0.03)
        assert safe_seen, (
            "the in-band safety node never republished on /openral/safe_action — the "
            "graph under test was never healthy, so phase 2 would prove nothing"
        )
        assert not [f for f in failures if f.kind == FailureTrigger.KIND_TIMEOUT], (
            "the deadman fired while the chunk stream was live"
        )
        assert not estop_seen, "an estop fired on a healthy graph"

        # Phase 2 — kill the in-band gate. The producer keeps publishing, so the
        # ONLY thing that changed is that the safety process is gone.
        pid = estop_graph["safety"].pid
        os.kill(pid, signal.SIGKILL)

        fired_by = time.monotonic() + _DEADLINE_S + 10.0
        while time.monotonic() < fired_by and not estop_seen:
            candidate_pub.publish(_chunk(ActionChunk))
            probe.executor.spin_once(timeout_sec=0.03)

        assert estop_seen, (
            f"/openral/estop never fired within {_DEADLINE_S + 10.0:.1f}s of killing the "
            f"in-band safety node (pid {pid}) — the independent watchdog did not brake"
        )
        assert probe.spin_until(
            lambda: any(f.kind == FailureTrigger.KIND_TIMEOUT for f in failures),
            timeout_s=5.0,
        ), "estop fired without the structured FailureTrigger(KIND_TIMEOUT) beside it"
        _assert_timeout_evidence(
            next(f for f in failures if f.kind == FailureTrigger.KIND_TIMEOUT), FailureTrigger
        )
    finally:
        probe.close()


def test_deadman_fires_when_safe_action_goes_silent_inside_an_open_window(
    estop_graph: dict[str, Any],
) -> None:
    """The deadman path proper: chunk stream starts, then stops.

    Distinct from the kill test: here the in-band node is irrelevant — the
    stream is published straight onto ``/openral/safe_action`` and then
    withdrawn, which is the deadline contract in isolation.
    """
    from openral_msgs.msg import ActionChunk, FailureTrigger
    from std_msgs.msg import Empty

    del estop_graph  # the module-scoped graph is already up; nothing to read
    probe = _Probe()
    try:
        # A previous test latched this deadman behind its own estop. Release it
        # the way production does — SafetyStatus(latched=False), i.e. the
        # operator's /openral/estop_reset landing on the kernel. Cycling the
        # node's lifecycle instead would hide the very bug this proves absent.
        arm_pub = _arm_publisher(probe)
        probe.spin_for(1.0)
        _close_any_open_window(probe, arm_pub)
        _clear_safety_status(probe)
        probe.spin_for(1.0)
        safe_pub = probe.node.create_publisher(ActionChunk, "/openral/safe_action", _qos(1))
        estop_seen: list[Any] = []
        failures: list[Any] = []
        probe.node.create_subscription(Empty, "/openral/estop", estop_seen.append, _qos(10))
        probe.node.create_subscription(
            FailureTrigger, "/openral/failure/safety", failures.append, _qos(50)
        )
        probe.spin_for(2.0)

        arm_pub.publish(_status_array(live=True))

        streaming_until = time.monotonic() + (_DEADLINE_S * 1.5)
        while time.monotonic() < streaming_until:
            safe_pub.publish(_chunk(ActionChunk))
            probe.executor.spin_once(timeout_sec=0.03)
        assert not estop_seen, "the deadman braked a live chunk stream"

        # Silence.
        assert probe.spin_until(lambda: bool(estop_seen), timeout_s=_DEADLINE_S + 10.0), (
            f"/openral/estop never fired after {_DEADLINE_S}s of safe_action silence"
        )
        assert probe.spin_until(
            lambda: any(f.kind == FailureTrigger.KIND_TIMEOUT for f in failures),
            timeout_s=5.0,
        ), "estop fired without the structured FailureTrigger(KIND_TIMEOUT) beside it"
        _assert_timeout_evidence(
            next(f for f in failures if f.kind == FailureTrigger.KIND_TIMEOUT), FailureTrigger
        )
    finally:
        probe.close()


def test_human_forwarder_brakes_from_its_own_process(estop_graph: dict[str, Any]) -> None:
    """``/openral/human_estop`` → ``/openral/estop`` + ``KIND_HUMAN``."""
    from openral_msgs.msg import FailureTrigger
    from std_msgs.msg import Empty

    del estop_graph  # the module-scoped graph is already up
    probe = _Probe()
    try:
        human_pub = probe.node.create_publisher(Empty, "/openral/human_estop", _qos(10))
        estop_seen: list[Any] = []
        failures: list[Any] = []
        probe.node.create_subscription(Empty, "/openral/estop", estop_seen.append, _qos(10))
        probe.node.create_subscription(
            FailureTrigger, "/openral/failure/safety", failures.append, _qos(50)
        )
        probe.spin_for(2.0)

        human_pub.publish(Empty())
        assert probe.spin_until(lambda: bool(estop_seen), timeout_s=10.0), (
            "/openral/human_estop was not forwarded onto /openral/estop"
        )
        assert probe.spin_until(
            lambda: any(f.kind == FailureTrigger.KIND_HUMAN for f in failures),
            timeout_s=5.0,
        ), "no FailureTrigger(KIND_HUMAN) accompanied the forwarded estop"
        trigger = next(f for f in failures if f.kind == FailureTrigger.KIND_HUMAN)
        assert trigger.severity == FailureTrigger.SEVERITY_ABORT
        assert json.loads(trigger.evidence_json) == {
            "kind": "human",
            "actor": _CHANNEL_LABEL,
            "reason": "human_estop",
        }
    finally:
        probe.close()


def test_the_deadman_rearms_after_an_estop_is_cleared(estop_graph: dict[str, Any]) -> None:
    """One estop must not silence the watchdog for the rest of the deploy.

    ``/openral/estop`` never auto-clears, and the deadman latches behind any
    estop it sees so it cannot storm the topic. Before this was fixed the ONLY
    thing that released that latch was re-activating the node, which nothing in
    production ever does — so after the first stop-and-reset of a deploy (a
    dashboard button press is enough) the graph had exactly the hole the
    watchdog exists to close, and no way to tell.
    """
    from openral_msgs.msg import ActionChunk, FailureTrigger
    from std_msgs.msg import Empty

    del estop_graph
    probe = _Probe()
    try:
        arm_pub = _arm_publisher(probe)
        safe_pub = probe.node.create_publisher(ActionChunk, "/openral/safe_action", _qos(1))
        human_pub = probe.node.create_publisher(Empty, "/openral/human_estop", _qos(10))
        estop_seen: list[Any] = []
        failures: list[Any] = []
        probe.node.create_subscription(Empty, "/openral/estop", estop_seen.append, _qos(10))
        probe.node.create_subscription(
            FailureTrigger, "/openral/failure/safety", failures.append, _qos(50)
        )
        probe.spin_for(2.0)
        _close_any_open_window(probe, arm_pub)

        # An operator stops the robot. The deadman latches behind that estop.
        human_pub.publish(Empty())
        assert probe.spin_until(lambda: bool(estop_seen), timeout_s=10.0), (
            "the human forwarder never braked, so this test never reached its premise"
        )
        probe.spin_for(1.0)
        timeouts_before = len([f for f in failures if f.kind == FailureTrigger.KIND_TIMEOUT])

        # Operator clears it: the kernel republishes SafetyStatus(latched=False).
        _clear_safety_status(probe)
        probe.spin_for(1.5)

        # A fresh goal, a live stream, then the producer dies. Must fire again.
        arm_pub.publish(_status_array(live=True))
        streaming_until = time.monotonic() + (_DEADLINE_S * 1.5)
        while time.monotonic() < streaming_until:
            safe_pub.publish(_chunk(ActionChunk))
            probe.executor.spin_once(timeout_sec=0.03)

        assert probe.spin_until(
            lambda: (
                len([f for f in failures if f.kind == FailureTrigger.KIND_TIMEOUT])
                > timeouts_before
            ),
            timeout_s=_DEADLINE_S + 10.0,
        ), (
            "the deadman never fired again after the estop was cleared — it is one-shot "
            "per activation, so the first stop of a deploy disarms it for good"
        )
    finally:
        probe.close()
