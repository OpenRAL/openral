"""Execution budgets on the graph clock — ``openral_rskill.execution_budget``.

Regression cover for a slow simulator: with ``use_sim_time=true`` the whole
deploy-sim graph (Nav2, MoveIt, controllers) times itself in sim seconds, but
the rSkill runner's budget and ``ROSActionRskill``'s result wait ran on
``time.monotonic()`` — a 0.09x rendering sim lost a 300 s Nav2 budget after
28 s of sim time. The fix measures both on the graph clock and keeps
wall-clock backstops so a goal still ends when ``/clock`` stops.

Everything here is real: a private rclpy context, a real ``rosgraph_msgs/Clock``
publisher (the process boundary the HAL's ``/clock`` thread sits on), real
nodes whose ``use_sim_time`` TimeSource consumes it, and real
``rclpy.task.Future``s. ``execution_budget`` is imported inside the tests so
the file also runs against a tree that predates it — the ``_poll_future`` and
runner tests fail there, which is the regression they pin.
"""

from __future__ import annotations

import importlib.util
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from types import MethodType, ModuleType
from typing import Any

import pytest

rclpy = pytest.importorskip("rclpy")
pytest.importorskip("rosgraph_msgs")

from rclpy.context import Context  # noqa: E402
from rclpy.executors import SingleThreadedExecutor  # noqa: E402
from rclpy.node import Node  # noqa: E402
from rclpy.parameter import Parameter  # noqa: E402
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy  # noqa: E402
from rclpy.task import Future  # noqa: E402
from rosgraph_msgs.msg import Clock  # noqa: E402

_PUBLISH_HZ = 50.0


class _SimClockPublisher:
    """A simulator's ``/clock``: sim time advances at ``rtf`` × wall from ``start_s``.

    Same QoS as ``openral_hal.lifecycle``'s publisher. ``stop()`` is a paused /
    crashed sim: ``/clock`` goes silent and sim time stands still.
    """

    def __init__(self, ctx: Context, *, rtf: float, start_s: float = 1.0) -> None:
        self.node = Node("test_sim_clock", context=ctx)
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
    """Private rclpy context + executor thread; ``sim_node`` runs on sim time."""

    def __init__(self) -> None:
        self.ctx = Context()
        rclpy.init(context=self.ctx)
        self.sim_node = Node(
            "test_sim_time_consumer",
            context=self.ctx,
            parameter_overrides=[Parameter("use_sim_time", value=True)],
        )
        self.wall_node = Node("test_wall_time_consumer", context=self.ctx)
        self.executor = SingleThreadedExecutor(context=self.ctx)
        self.executor.add_node(self.sim_node)
        self.executor.add_node(self.wall_node)
        self.clock: _SimClockPublisher | None = None
        self._spin = threading.Thread(target=self.executor.spin, name="test-spin", daemon=True)
        self._spin.start()

    def start_clock(self, *, rtf: float, start_s: float = 1.0) -> _SimClockPublisher:
        self.clock = _SimClockPublisher(self.ctx, rtf=rtf, start_s=start_s)
        self.executor.add_node(self.clock.node)
        self.clock.start()
        # Wait until the consumer's TimeSource has taken the first /clock.
        deadline = time.monotonic() + 5.0
        while self.sim_node.get_clock().now().nanoseconds == 0:
            assert time.monotonic() < deadline, "/clock never reached the sim-time node"
            time.sleep(0.01)
        return self.clock

    def close(self) -> None:
        if self.clock is not None:
            self.clock.stop()
        self.executor.shutdown(timeout_sec=2.0)
        self._spin.join(timeout=2.0)
        for n in (self.sim_node, self.wall_node, *([self.clock.node] if self.clock else [])):
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


def test_wall_cap_bounds_a_crawling_clock(graph: _Graph) -> None:
    """At 0.01x the wall backstop (2 × 0.3 s budget) fires long before 0.3 sim seconds."""
    from openral_rskill.execution_budget import ExecutionBudget, graph_clock_fn

    graph.start_clock(rtf=0.01)
    budget = ExecutionBudget(
        0.3, graph_now=graph_clock_fn(graph.sim_node), wall_cap_factor=2.0, stall_s=30.0
    )
    miss = _poll(budget, 3.0)
    assert miss is not None and miss.kind == "deadline_exceeded"
    assert "wall elapsed" in miss.detail
    assert miss.elapsed_s < 0.1


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


def test_disabled_budget_never_lapses() -> None:
    """``budget_s <= 0`` disables every check, as the runner's contract says."""
    from openral_rskill.execution_budget import ExecutionBudget

    budget = ExecutionBudget(0.0, graph_now=lambda: 0.0, stall_s=0.0)
    assert budget.check() is None and budget.progress() == 0.0


# ── ROSActionRskill._poll_future ─────────────────────────────────────────────


def _result_only_skill(ros_node: Node) -> Any:
    """A real ``ROSActionRskill`` (Nav2 result-only manifest) configured on ``ros_node``.

    ``_configure_impl`` waits for the wrapped server, so a real (trivially
    succeeding) ``NavigateToPose`` ``ActionServer`` is stood up on the same node.
    """
    from nav2_msgs.action import NavigateToPose
    from openral_rskill.ros_action_rskill import ROSActionRskill
    from rclpy.action import ActionServer
    from rclpy.callback_groups import ReentrantCallbackGroup

    from tests.unit.test_ros_action_rskill import _result_only_manifest

    def _succeed(goal_handle: Any) -> Any:
        goal_handle.succeed()
        return NavigateToPose.Result()

    ActionServer(
        ros_node,
        NavigateToPose,
        "/navigate_to_pose",
        _succeed,
        callback_group=ReentrantCallbackGroup(),
    )
    skill = ROSActionRskill(
        manifest=_result_only_manifest(),
        ros_node=ros_node,
        robot_description=None,
        prompt="go to the dock",
        prompt_metadata_json="",
    )
    skill._configure_impl()  # real nav2_msgs ActionClient; sets _graph_now from the node
    return skill


def _complete_later(future: Future, wall_s: float) -> None:
    def _set() -> None:
        time.sleep(wall_s)
        future.set_result(object())

    threading.Thread(target=_set, daemon=True).start()


def test_poll_future_on_a_slow_sim_clock_waits_in_sim_seconds(graph: _Graph) -> None:
    """(a) A 1 s result deadline at 0.1x no longer raises after 1 s of wall."""
    from openral_core.exceptions import ROSRuntimeError

    graph.start_clock(rtf=0.1)
    skill = _result_only_skill(graph.sim_node)
    future = Future()
    _complete_later(future, 1.5)
    skill._poll_future(future, deadline_s=1.0, what="result")  # old code: ROSRuntimeError
    assert future.done()

    # ... and a stalled /clock still raises, naming the stall (b).
    assert graph.clock is not None
    graph.clock.stop()
    stuck = Future()
    with pytest.raises(ROSRuntimeError, match="clock_stalled"):
        skill._poll_future(stuck, deadline_s=1.0, what="result")
    skill._shutdown_impl()


def test_poll_future_goal_accept_stays_on_wall_clock(graph: _Graph) -> None:
    """Server acceptance is IPC, not motion: ``graph_clock=False`` times out on wall."""
    from openral_core.exceptions import ROSRuntimeError

    graph.start_clock(rtf=0.1)
    skill = _result_only_skill(graph.sim_node)
    t0 = time.monotonic()
    with pytest.raises(ROSRuntimeError, match=r"goal-accept did not complete within 0\.3s"):
        skill._poll_future(Future(), deadline_s=0.3, what="goal-accept", graph_clock=False)
    assert time.monotonic() - t0 < 1.0
    skill._shutdown_impl()


def test_poll_future_without_sim_time_is_unchanged(graph: _Graph) -> None:
    """(c) On a wall-clock node the result wait raises after ``deadline_s`` of wall."""
    from openral_core.exceptions import ROSRuntimeError

    skill = _result_only_skill(graph.wall_node)
    t0 = time.monotonic()
    with pytest.raises(ROSRuntimeError, match=r"result did not complete within 0\.3s"):
        skill._poll_future(Future(), deadline_s=0.3, what="result")
    assert 0.25 < time.monotonic() - t0 < 1.0
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
    from openral_rskill.execution_budget import ExecutionBudget, graph_clock_fn

    mod = _load_runner_module()
    holder = _RunnerBudgetHolder()
    lapsed = MethodType(mod.RskillRunnerNode._deadline_lapsed, holder)

    clock = graph.start_clock(rtf=0.1)
    budget = ExecutionBudget(1.0, graph_now=graph_clock_fn(graph.sim_node), stall_s=0.5)
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
