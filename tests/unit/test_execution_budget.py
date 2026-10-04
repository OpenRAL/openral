"""Execution budgets on the graph clock — ``openral_rskill.execution_budget``.

Regression cover for a slow simulator: with ``use_sim_time=true`` the whole
deploy-sim graph (Nav2, MoveIt, controllers) times itself in sim seconds, but
the rSkill runner's budget and ``ROSActionRskill``'s result wait ran on
``time.monotonic()`` — a 0.09x rendering sim lost a 300 s Nav2 budget after
28 s of sim time. The fix measures both on the graph clock and keeps
wall-clock backstops so a goal still ends when ``/clock`` stops.

Everything here is real: a private rclpy context on an isolated ROS domain, a
real ``rosgraph_msgs/Clock`` publisher (the process boundary the HAL's
``/clock`` thread sits on), real nodes whose ``use_sim_time`` TimeSource
consumes it, real ``rclpy.task.Future``s and a real ``NavigateToPose``
``ActionServer``. ``execution_budget`` is imported inside the tests so the
file also runs against a tree that predates it — the ``_poll_future`` and
runner tests fail there, which is the regression they pin.
"""

from __future__ import annotations

import importlib.util
import itertools
import math
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from types import MethodType, ModuleType
from typing import Any

import pytest

rclpy = pytest.importorskip("rclpy")
pytest.importorskip("rosgraph_msgs")
pytest.importorskip("nav2_msgs")

from rclpy.action import ActionServer, CancelResponse  # noqa: E402
from rclpy.callback_groups import ReentrantCallbackGroup  # noqa: E402
from rclpy.context import Context  # noqa: E402
from rclpy.executors import MultiThreadedExecutor  # noqa: E402
from rclpy.node import Node  # noqa: E402
from rclpy.parameter import Parameter  # noqa: E402
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy  # noqa: E402
from rclpy.task import Future  # noqa: E402
from rosgraph_msgs.msg import Clock  # noqa: E402

from tests.sim.safety._kernel_subprocess import isolated_domain_id  # noqa: E402

_PUBLISH_HZ = 50.0
#: Each ``_Graph`` gets its own ROS domain: the DDS discovery cache outlives
#: ``rclpy.shutdown()``, so two tests on one domain can see each other's
#: ``/navigate_to_pose`` servers.
_DOMAIN_SEQ = itertools.count()
#: Small backstops so every stall / cap test is fast and deterministic (the stall
#: fires long before the cap: 0.5 s vs 3 x budget).
_TEST_STALL_S = 0.5
_TEST_WALL_CAP_FACTOR = 3.0


class _SimClockPublisher:
    """A simulator's ``/clock``: sim time advances at ``rtf`` x wall from ``start_s``.

    Same QoS as ``openral_hal.lifecycle``'s publisher. ``stop()`` is a paused /
    crashed sim: ``/clock`` goes silent and sim time stands still.
    """

    def __init__(self, ctx: Context, *, rtf: float, start_s: float = 1.0) -> None:
        self.node = Node(f"test_sim_clock_{int(start_s)}", context=ctx)
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=10,
        )
        self._pub = self.node.create_publisher(Clock, "/clock", qos)
        self._rtf = rtf
        self._start_s = start_s
        self._wall0 = time.monotonic()
        self._running = threading.Event()
        self._thread = threading.Thread(target=self._run, name="test-sim-clock", daemon=True)

    def start(self) -> None:
        self._running.set()
        self._thread.start()

    def stop(self) -> None:
        self._running.clear()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def _run(self) -> None:
        period = 1.0 / _PUBLISH_HZ
        while self._running.is_set():
            sim_s = self._start_s + (time.monotonic() - self._wall0) * self._rtf
            msg = Clock()
            msg.clock.sec = int(sim_s)
            msg.clock.nanosec = int((sim_s - int(sim_s)) * 1e9)
            self._pub.publish(msg)
            time.sleep(period)


class _Graph:
    """Private rclpy context on its own isolated domain; ``sim_node`` runs on sim time.

    Both consumer nodes declare the runner's two backstop parameters at small
    values, the way ``RskillRunnerNode`` declares them, so ``node_execution_budget``
    picks them up for the runner's goals and the wrapped skills alike.
    """

    def __init__(self) -> None:
        self.ctx = Context()
        domain = 50 + (isolated_domain_id() + next(_DOMAIN_SEQ)) % 50
        rclpy.init(context=self.ctx, domain_id=domain)
        self.sim_node = Node(
            "test_sim_time_consumer",
            context=self.ctx,
            parameter_overrides=[Parameter("use_sim_time", value=True)],
        )
        self.wall_node = Node("test_wall_time_consumer", context=self.ctx)
        for n in (self.sim_node, self.wall_node):
            n.declare_parameter("graph_clock_stall_s", _TEST_STALL_S)
            n.declare_parameter("execution_wall_cap_factor", _TEST_WALL_CAP_FACTOR)
        self.executor = MultiThreadedExecutor(num_threads=4, context=self.ctx)
        self.executor.add_node(self.sim_node)
        self.executor.add_node(self.wall_node)
        self.clocks: list[_SimClockPublisher] = []
        self._spin = threading.Thread(target=self.executor.spin, name="test-spin", daemon=True)
        self._spin.start()

    @property
    def clock(self) -> _SimClockPublisher:
        return self.clocks[-1]

    def start_clock(self, *, rtf: float, start_s: float = 1.0) -> _SimClockPublisher:
        clock = _SimClockPublisher(self.ctx, rtf=rtf, start_s=start_s)
        self.clocks.append(clock)
        self.executor.add_node(clock.node)
        before = self.sim_node.get_clock().now().nanoseconds
        clock.start()
        # Wait until the consumer's TimeSource has taken this publisher's clock.
        deadline = time.monotonic() + 5.0
        while abs(self.sim_node.get_clock().now().nanoseconds - before) < int(0.2e9):
            assert time.monotonic() < deadline, "/clock never reached the sim-time node"
            time.sleep(0.01)
        return clock

    def close(self) -> None:
        for c in self.clocks:
            c.stop()
        self.executor.shutdown(timeout_sec=2.0)
        self._spin.join(timeout=2.0)
        for n in (self.sim_node, self.wall_node, *(c.node for c in self.clocks)):
            n.destroy_node()
        rclpy.shutdown(context=self.ctx)


@pytest.fixture
def graph() -> Iterator[_Graph]:
    g = _Graph()
    try:
        yield g
    finally:
        g.close()


def _poll(budget: Any, wall_s: float) -> Any:
    """Check ``budget`` every 20 ms for ``wall_s``; return the first miss (or ``None``)."""
    end = time.monotonic() + wall_s
    while time.monotonic() < end:
        miss = budget.check()
        if miss is not None:
            return miss
        time.sleep(0.02)
    return None


# ── ExecutionBudget on a real sim-time node ──────────────────────────────────


def test_slow_sim_clock_does_not_lapse_the_budget_early(graph: _Graph) -> None:
    """(a) At 0.1x real time a 1 s budget survives 1.5 s of wall; wall-clock would not."""
    from openral_rskill.execution_budget import ExecutionBudget, graph_clock_fn

    graph.start_clock(rtf=0.1)
    graph_now = graph_clock_fn(graph.sim_node)
    assert graph_now is not None
    sim_budget = ExecutionBudget(1.0, graph_now=graph_now)
    wall_budget = ExecutionBudget(1.0, graph_now=None)
    assert sim_budget.on_graph_clock and not wall_budget.on_graph_clock

    assert _poll(sim_budget, 1.5) is None
    assert 0.05 < sim_budget.elapsed_s() < 0.4  # ~0.15 s of sim time passed
    assert sim_budget.progress() < 0.5
    wall_miss = wall_budget.check()
    assert wall_miss is not None and wall_miss.kind == "deadline_exceeded"
    assert "(wall clock)" in wall_miss.detail


def test_graph_clock_budget_lapses_on_sim_seconds(graph: _Graph) -> None:
    """A fast sim (5x) lapses a 1 s budget after ~0.2 s of wall, reporting graph seconds."""
    from openral_rskill.execution_budget import ExecutionBudget, graph_clock_fn

    graph.start_clock(rtf=5.0)
    budget = ExecutionBudget(1.0, graph_now=graph_clock_fn(graph.sim_node))
    miss = _poll(budget, 2.0)
    assert miss is not None and miss.kind == "deadline_exceeded"
    assert 1.0 < miss.elapsed_s < 1.6
    assert "(graph clock)" in miss.detail


def test_stalled_clock_fails_closed(graph: _Graph) -> None:
    """(b) /clock goes silent mid-goal → ``clock_stalled`` after ``stall_s`` of wall time."""
    from openral_rskill.execution_budget import ExecutionBudget, graph_clock_fn

    clock = graph.start_clock(rtf=0.1)
    budget = ExecutionBudget(300.0, graph_now=graph_clock_fn(graph.sim_node), stall_s=0.5)
    assert _poll(budget, 0.3) is None
    clock.stop()
    t0 = time.monotonic()
    miss = _poll(budget, 3.0)
    assert miss is not None and miss.kind == "clock_stalled"
    assert 0.4 < time.monotonic() - t0 < 1.5
    assert "has not advanced" in miss.detail
    assert miss.elapsed_s < 1.0  # graph time barely moved


def test_clock_not_yet_received_is_not_started(graph: _Graph) -> None:
    """A sim node reads 0 before its first /clock: no instant lapse, and the first
    message (here a jump to t=5000 s) anchors the budget instead of exceeding it."""
    from openral_rskill.execution_budget import ExecutionBudget, graph_clock_fn

    graph_now = graph_clock_fn(graph.sim_node)
    assert graph_now is not None and graph_now() == 0.0
    budget = ExecutionBudget(1.0, graph_now=graph_now, stall_s=5.0)
    assert _poll(budget, 0.3) is None
    assert budget.elapsed_s() == 0.0
    graph.start_clock(rtf=0.1, start_s=5000.0)
    assert budget.check() is None
    assert budget.elapsed_s() < 0.5


def test_clock_that_never_starts_stalls(graph: _Graph) -> None:
    """use_sim_time with no /clock publisher at all still ends the goal."""
    from openral_rskill.execution_budget import ExecutionBudget, graph_clock_fn

    budget = ExecutionBudget(1.0, graph_now=graph_clock_fn(graph.sim_node), stall_s=0.4)
    miss = _poll(budget, 2.0)
    assert miss is not None and miss.kind == "clock_stalled"
    assert "never started" in miss.detail


def test_backward_clock_jump_re_anchors_and_keeps_elapsed(graph: _Graph) -> None:
    """A sim reset (/clock jumps 100 s → 1 s, as a MuJoCo HAL reconnect does) is neither a
    stall nor a reset: the budget banks what elapsed and keeps counting from the new anchor."""
    from openral_rskill.execution_budget import ExecutionBudget, graph_clock_fn

    first = graph.start_clock(rtf=1.0, start_s=100.0)
    budget = ExecutionBudget(10.0, graph_now=graph_clock_fn(graph.sim_node), stall_s=0.5)
    assert _poll(budget, 0.4) is None
    before = budget.elapsed_s()
    assert 0.3 < before < 0.7
    first.stop()
    graph.start_clock(rtf=1.0, start_s=1.0)
    assert _poll(budget, 0.4) is None  # no clock_stalled across the jump
    after = budget.elapsed_s()
    assert after > before, "elapsed must not freeze or rewind over a backward jump"
    assert before + 0.3 < after < before + 1.5


def test_wall_cap_bounds_a_crawling_clock(graph: _Graph) -> None:
    """At 0.01x the wall backstop (2 x 0.3 s budget) fires long before 0.3 sim seconds."""
    from openral_rskill.execution_budget import ExecutionBudget, graph_clock_fn

    graph.start_clock(rtf=0.01)
    budget = ExecutionBudget(
        0.3, graph_now=graph_clock_fn(graph.sim_node), wall_cap_factor=2.0, stall_s=30.0
    )
    miss = _poll(budget, 3.0)
    assert miss is not None and miss.kind == "deadline_exceeded"
    assert "wall elapsed" in miss.detail
    assert miss.elapsed_s < 0.1


def test_default_wall_cap_covers_the_slowest_measured_sim() -> None:
    """The 0.09x rendering graph that motivated this must keep a 300 s budget intact."""
    from openral_rskill.execution_budget import DEFAULT_WALL_CAP_FACTOR

    assert DEFAULT_WALL_CAP_FACTOR >= 1.0 / 0.09


def test_use_sim_time_false_is_monotonic(graph: _Graph) -> None:
    """(c) Without use_sim_time there is no graph clock: budgets are wall-clock as before."""
    from openral_rskill.execution_budget import ExecutionBudget, graph_clock_fn

    assert graph_clock_fn(graph.wall_node) is None
    assert graph_clock_fn(None) is None
    budget = ExecutionBudget(0.2, graph_now=graph_clock_fn(graph.wall_node))
    assert not budget.on_graph_clock
    t0 = time.monotonic()
    miss = _poll(budget, 2.0)
    assert miss is not None and miss.kind == "deadline_exceeded"
    assert 0.15 < time.monotonic() - t0 < 0.6
    assert "(wall clock)" in miss.detail


def test_use_sim_time_flipped_off_at_runtime_returns_to_monotonic(graph: _Graph) -> None:
    """The clock mode is read per budget, not cached at configure."""
    from openral_rskill.execution_budget import node_execution_budget

    graph.start_clock(rtf=0.1)
    assert node_execution_budget(graph.sim_node, 1.0).on_graph_clock
    graph.sim_node.set_parameters([Parameter("use_sim_time", value=False)])
    assert not node_execution_budget(graph.sim_node, 1.0).on_graph_clock


def test_node_budget_reads_the_host_node_backstop_params(graph: _Graph) -> None:
    """``node_execution_budget`` takes the runner's two parameters off the node."""
    from openral_rskill.execution_budget import (
        DEFAULT_CLOCK_STALL_S,
        DEFAULT_WALL_CAP_FACTOR,
        node_execution_budget,
    )

    graph.start_clock(rtf=0.1)
    budget = node_execution_budget(graph.sim_node, 1.0)
    assert budget._stall_s == _TEST_STALL_S
    assert budget._wall_cap_factor == _TEST_WALL_CAP_FACTOR
    # A node that declares nothing gets the module defaults.
    bare = Node("test_bare_consumer", context=graph.ctx)
    try:
        default_budget = node_execution_budget(bare, 1.0)
        assert default_budget._stall_s == DEFAULT_CLOCK_STALL_S
        assert default_budget._wall_cap_factor == DEFAULT_WALL_CAP_FACTOR
    finally:
        bare.destroy_node()


def test_disabled_budget_never_lapses() -> None:
    """``budget_s <= 0`` disables every check, as the runner's contract says."""
    from openral_rskill.execution_budget import ExecutionBudget

    budget = ExecutionBudget(0.0, graph_now=lambda: 0.0, stall_s=0.5)
    assert budget.check() is None and budget.progress() == 0.0


def test_non_finite_budget_is_rejected() -> None:
    """A NaN budget would never lapse (every comparison is false): refuse it up front."""
    from openral_core.exceptions import ROSConfigError
    from openral_rskill.execution_budget import ExecutionBudget

    for bad in (math.nan, math.inf, -math.inf):
        with pytest.raises(ROSConfigError, match="finite"):
            ExecutionBudget(bad)


def test_backstops_cannot_be_disabled_on_a_graph_clock() -> None:
    """Deadline fallback is mandatory: a non-positive / non-finite backstop is a config error."""
    from openral_core.exceptions import ROSConfigError
    from openral_rskill.execution_budget import ExecutionBudget

    for kwargs in ({"stall_s": 0.0}, {"stall_s": math.nan}, {"wall_cap_factor": -1.0}):
        with pytest.raises(ROSConfigError, match="finite value > 0"):
            ExecutionBudget(1.0, graph_now=lambda: 1.0, **kwargs)
    # Irrelevant (and so unchecked) without a graph clock.
    assert ExecutionBudget(1.0, stall_s=0.0, wall_cap_factor=0.0).check() is None


# ── ROSActionRskill._poll_future / result wait ───────────────────────────────


def _nav2_server(ros_node: Node, execute: Callable[[Any], Any]) -> ActionServer:
    """A real ``NavigateToPose`` ``ActionServer`` on ``ros_node`` that accepts cancels."""
    from nav2_msgs.action import NavigateToPose

    return ActionServer(
        ros_node,
        NavigateToPose,
        "/navigate_to_pose",
        execute,
        cancel_callback=lambda _goal: CancelResponse.ACCEPT,
        callback_group=ReentrantCallbackGroup(),
    )


def _succeed_at_once(goal_handle: Any) -> Any:
    from nav2_msgs.action import NavigateToPose

    goal_handle.succeed()
    return NavigateToPose.Result()


def _result_only_skill(ros_node: Node, execute: Callable[[Any], Any] = _succeed_at_once) -> Any:
    """A real ``ROSActionRskill`` (Nav2 result-only manifest) configured on ``ros_node``.

    ``_configure_impl`` waits for the wrapped server, so a real ``NavigateToPose``
    ``ActionServer`` running ``execute`` is stood up on the same node.
    """
    from openral_rskill.ros_action_rskill import ROSActionRskill

    from tests.unit.test_ros_action_rskill import _result_only_manifest

    _nav2_server(ros_node, execute)
    skill = ROSActionRskill(
        manifest=_result_only_manifest(),
        ros_node=ros_node,
        robot_description=None,
        prompt="go to the dock",
        prompt_metadata_json="",
    )
    skill._configure_impl()  # real nav2_msgs ActionClient on the node
    return skill


def _complete_later(future: Future, wall_s: float) -> None:
    def _set() -> None:
        time.sleep(wall_s)
        future.set_result(object())

    threading.Thread(target=_set, daemon=True).start()


def test_poll_future_on_a_slow_sim_clock_waits_in_sim_seconds(graph: _Graph) -> None:
    """(a) A 1 s result deadline at 0.1x no longer raises after 1 s of wall."""
    from openral_core.exceptions import ROSDeadlineMissed

    graph.start_clock(rtf=0.1)
    skill = _result_only_skill(graph.sim_node)
    future = Future()
    _complete_later(future, 1.5)
    skill._poll_future(future, deadline_s=1.0, what="result")  # old code: raised after 1 s
    assert future.done()

    # ... and a stalled /clock still raises the typed deadline failure, naming the
    # stall (b). The node's 0.5 s stall param wins over its 3x wall cap.
    graph.clock.stop()
    t0 = time.monotonic()
    with pytest.raises(ROSDeadlineMissed, match="clock_stalled"):
        skill._poll_future(Future(), deadline_s=1.0, what="result")
    assert 0.4 < time.monotonic() - t0 < 1.5
    skill._shutdown_impl()


def test_poll_future_goal_accept_stays_on_wall_clock(graph: _Graph) -> None:
    """Server acceptance is IPC, not motion: ``graph_clock=False`` times out on wall."""
    from openral_core.exceptions import ROSDeadlineMissed

    graph.start_clock(rtf=0.1)
    skill = _result_only_skill(graph.sim_node)
    t0 = time.monotonic()
    with pytest.raises(ROSDeadlineMissed, match=r"goal-accept did not complete within 0\.3s"):
        skill._poll_future(Future(), deadline_s=0.3, what="goal-accept", graph_clock=False)
    assert time.monotonic() - t0 < 1.0
    skill._shutdown_impl()


def test_poll_future_without_sim_time_is_unchanged(graph: _Graph) -> None:
    """(c) On a wall-clock node the result wait raises after ``deadline_s`` of wall."""
    from openral_core.exceptions import ROSDeadlineMissed

    skill = _result_only_skill(graph.wall_node)
    t0 = time.monotonic()
    with pytest.raises(ROSDeadlineMissed, match=r"result did not complete within 0\.3s"):
        skill._poll_future(Future(), deadline_s=0.3, what="result")
    assert 0.25 < time.monotonic() - t0 < 1.0
    skill._shutdown_impl()


def test_poll_future_non_positive_deadline_fails_closed(graph: _Graph) -> None:
    """``deadline_s <= 0`` lapses at once (as the wall-clock code did), never waits forever."""
    from openral_core.exceptions import ROSDeadlineMissed

    skill = _result_only_skill(graph.wall_node)
    t0 = time.monotonic()
    with pytest.raises(ROSDeadlineMissed, match="non-positive deadline_s"):
        skill._poll_future(Future(), deadline_s=0.0, what="result")
    assert time.monotonic() - t0 < 0.5
    done = Future()
    done.set_result(object())
    skill._poll_future(done, deadline_s=0.0, what="result")  # already done: nothing to wait for
    skill._shutdown_impl()


def test_result_wait_timeout_cancels_the_wrapped_goal(graph: _Graph) -> None:
    """Giving up on a Nav2 result must cancel the goal the server is still executing."""
    from openral_core.exceptions import ROSDeadlineMissed

    cancel_seen = threading.Event()

    def _run_until_cancelled(goal_handle: Any) -> Any:
        from nav2_msgs.action import NavigateToPose

        end = time.monotonic() + 30.0
        while time.monotonic() < end and not goal_handle.is_cancel_requested:
            time.sleep(0.02)
        if goal_handle.is_cancel_requested:
            cancel_seen.set()
            goal_handle.canceled()
        else:
            goal_handle.abort()
        return NavigateToPose.Result()

    skill = _result_only_skill(graph.wall_node, _run_until_cancelled)
    skill._result_deadline_s = 0.5  # the manifest floor is 2 s; keep the test short
    t0 = time.monotonic()
    with pytest.raises(ROSDeadlineMissed, match="result did not complete"):
        skill._send_action_goal_and_await_result()
    assert 0.4 < time.monotonic() - t0 < 3.0
    assert cancel_seen.wait(2.0), "the wrapped server never saw a cancel request"
    skill._shutdown_impl()


# ── RskillRunnerNode._deadline_lapsed on a graph-clock budget ────────────────


def _load_runner_module() -> ModuleType:
    """Load ``rskill_runner_node`` bypassing the ROS2-gated package init."""
    src = (
        Path(__file__).resolve().parents[2]
        / "packages/openral_rskill_ros/openral_rskill_ros/rskill_runner_node.py"
    )
    spec = importlib.util.spec_from_file_location("_test_rskill_runner_graph_clock", src)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _RunnerBudgetHolder:
    """Real carrier for what ``_deadline_lapsed`` touches (see test_skill_runner_deadline)."""

    def __init__(self) -> None:
        import rclpy.logging

        self._logger = rclpy.logging.get_logger("test_execution_budget_runner")
        self._last_deadline_miss: Any = None

    def get_logger(self) -> Any:  # reason: rclpy logger is untyped
        return self._logger


def test_runner_deadline_lapsed_follows_the_graph_clock(graph: _Graph) -> None:
    """The runner's budget predicate: survives a slow sim, fails closed on a stalled one."""
    from openral_rskill.execution_budget import node_execution_budget

    mod = _load_runner_module()
    holder = _RunnerBudgetHolder()
    lapsed = MethodType(mod.RskillRunnerNode._deadline_lapsed, holder)

    clock = graph.start_clock(rtf=0.1)
    budget = node_execution_budget(graph.sim_node, 1.0)
    end = time.monotonic() + 1.5
    while time.monotonic() < end:
        assert lapsed(budget, 3) is False
        time.sleep(0.02)
    assert holder._last_deadline_miss is None

    clock.stop()
    end = time.monotonic() + 3.0
    while time.monotonic() < end and not lapsed(budget, 7):
        time.sleep(0.02)
    miss = holder._last_deadline_miss
    assert miss is not None and miss.kind == "clock_stalled"
