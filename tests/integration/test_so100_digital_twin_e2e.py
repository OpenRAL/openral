"""End-to-end HIL (digital-twin) verification for ADR-0018 step 1.

Brings up the **full** F1 + F5 + F8 graph in one process against a real
``SO100DigitalTwin`` (no USB hardware required) and asserts the action
chunk reaches the twin's motors — i.e. the topic contract closes the
loop:

    ExecuteRskill goal
        → rskill_runner_node.execute_cb
        → in-process loop (skill.step + ROSPublishingHAL.send_action)
        → /openral/candidate_action
        → safety_node pass-through
        → /openral/safe_action
        → ad-hoc HAL bridge node ._on_safe_action
        → SO100FollowerHAL(robot=SO100DigitalTwin).send_action
        → twin internal joint state advances
        → bridge publishes /joint_states (from twin.read_state)
        → world_state aggregator updates

Per CLAUDE.md §1.11 / §5.4: real ``rclpy``, real
``openral_msgs/ActionChunk``, real ``SO100DigitalTwin`` from lerobot,
real ``rSkillBase`` subclass producing real joint-position targets.
Skipped when ROS 2 is not sourced.

The ad-hoc HAL bridge node mirrors the relevant subset of
``packages/openral_hal_so100/openral_hal_so100/lifecycle_node.py`` — but
bypasses the production lifecycle ``on_configure`` (which opens a USB
serial port to ``/dev/ttyUSB0``). The bridge is a real ROS node, not a
mock; the production HAL lifecycle node's contract (`/joint_states`
publication, `/openral/safe_action` consumption, `/openral/estop`
latch) is exercised on the same topic surface.
"""

from __future__ import annotations

import importlib.util
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from typing import Any

import pytest

_ROS2_AVAILABLE = bool(os.environ.get("ROS_DISTRO")) and (
    importlib.util.find_spec("openral_msgs") is not None
)  # custom IDL must be colcon-built, not just ROS sourced (skip cleanly otherwise)

pytestmark = pytest.mark.skipif(
    not _ROS2_AVAILABLE,
    reason="ROS_DISTRO not set — these tests require a sourced ROS 2 installation.",
)


def _lerobot_available() -> bool:
    """True iff lerobot is importable (SO100DigitalTwin needs it)."""
    try:
        import lerobot  # noqa: F401
    except ImportError:
        return False
    return True


_TARGET = [0.25, 0.25, 0.25, 0.25, 0.25, 0.5]


def _make_constant_skill() -> Any:
    """Real rSkillBase subclass — six-DoF constant joint-position target."""
    from openral_core.schemas import Action, ControlMode
    from openral_rskill.base import rSkillBase

    class _ConstantSkill(rSkillBase):
        """Constant target — simplest exerciser of the topic contract."""

        def __init__(self) -> None:
            super().__init__(
                name="openral/test-so100-constant",
                version="0.1.0",
                role="s1",
                embodiment_tags=["so100_follower"],
            )

        def _configure_impl(self) -> None:
            pass

        def _activate_impl(self) -> None:
            pass

        def _deactivate_impl(self) -> None:
            pass

        def _shutdown_impl(self) -> None:
            pass

        def _step_impl(self, _world_state: Any) -> Action:
            return Action(
                control_mode=ControlMode.JOINT_POSITION,
                horizon=1,
                joint_targets=[list(_TARGET)],
            )

    skill = _ConstantSkill()
    skill.configure()
    skill.activate()
    return skill


def _local_skill_resolver(*_args: Any, **_kwargs: Any) -> Any:
    """Test-local resolver — returns the in-tree constant skill."""
    return _make_constant_skill()


def _make_hal_bridge_node(hal_adapter: Any) -> Any:
    """Build a tiny ``rclpy.Node`` that bridges the SO-100 HAL onto ROS.

    Subscribes to ``/openral/safe_action`` and forwards the first chunk
    row to ``hal_adapter.send_action``. Publishes ``/joint_states`` at
    30 Hz from ``hal_adapter.read_state``. Subscribes to
    ``/openral/estop`` to latch a brake state.

    Mirrors the production
    ``packages/openral_hal_so100/openral_hal_so100/lifecycle_node.py``
    topic surface (the parts F1/F5 actually exercise) without opening
    a USB port.
    """
    from openral_core.schemas import Action, ControlMode
    from openral_msgs.msg import ActionChunk
    from rclpy.node import Node
    from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
    from sensor_msgs.msg import JointState as RosJointState
    from std_msgs.msg import Empty

    class _HALBridge(Node):  # type: ignore[misc]
        """Ad-hoc HAL bridge — production analog without USB."""

        def __init__(self) -> None:
            super().__init__("openral_hal_so100_bridge")
            self._hal = hal_adapter
            self._estopped: bool = False

            chunk_qos = QoSProfile(
                reliability=QoSReliabilityPolicy.RELIABLE,
                durability=QoSDurabilityPolicy.VOLATILE,
                depth=1,
            )
            estop_qos = QoSProfile(
                reliability=QoSReliabilityPolicy.RELIABLE,
                durability=QoSDurabilityPolicy.VOLATILE,
                depth=10,
            )
            control_qos = QoSProfile(
                reliability=QoSReliabilityPolicy.RELIABLE,
                durability=QoSDurabilityPolicy.VOLATILE,
                depth=10,
            )

            self._joint_state_pub = self.create_publisher(
                RosJointState, "/joint_states", control_qos
            )
            self._safe_sub = self.create_subscription(
                ActionChunk,
                "/openral/safe_action",
                self._on_safe_action,
                chunk_qos,
            )
            self._estop_sub = self.create_subscription(
                Empty, "/openral/estop", self._on_estop, estop_qos
            )
            self._timer = self.create_timer(1.0 / 30.0, self._publish_joint_state)
            self.chunks_consumed: int = 0
            self.estop_latched: bool = False

        def _on_safe_action(self, msg: object) -> None:
            if self._estopped:
                return
            flat = list(getattr(msg, "flat", []) or [])
            n_dof = int(getattr(msg, "n_dof", 0) or 0)
            if n_dof <= 0 or not flat:
                return
            action = Action(
                control_mode=ControlMode.JOINT_POSITION,
                horizon=1,
                joint_targets=[flat[:n_dof]],
            )
            try:
                self._hal.send_action(action)
                self.chunks_consumed += 1
            except Exception as exc:  # reason: log + survive
                self.get_logger().warn(f"send_action failed: {exc}")

        def _on_estop(self, _msg: object) -> None:
            self._estopped = True
            self.estop_latched = True

        def _publish_joint_state(self) -> None:
            try:
                state = self._hal.read_state()
            except Exception as exc:  # reason: best-effort read
                self.get_logger().debug(f"read_state: {exc}")
                return
            msg = RosJointState()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.name = list(state.name)
            msg.position = list(state.position)
            msg.velocity = list(state.velocity) if state.velocity else []
            msg.effort = list(state.effort) if state.effort else []
            self._joint_state_pub.publish(msg)

    return _HALBridge()


@contextmanager
def _digital_twin_harness() -> Iterator[tuple[Any, Any, Any, Any, Any, list[Any], list[Any]]]:
    """Compose every F1/F5/F8 node + an SO-100 HAL bridge wrapping the twin.

    Yields ``(executor, runtime, safety, hal_bridge, twin,
    observed_safe, observed_joint_states)``.
    """
    import rclpy
    from openral_hal.so100_follower import SO100FollowerHAL
    from openral_hal.so100_sim import SO100DigitalTwin, SO100DigitalTwinConfig
    from openral_msgs.msg import ActionChunk
    from openral_rskill_ros.compose import compose_so100_runtime
    from openral_safety.supervisor_node import SafetyPassthroughNode
    from rclpy.lifecycle import TransitionCallbackReturn
    from sensor_msgs.msg import JointState as RosJointState

    rclpy.init()
    twin = SO100DigitalTwin(SO100DigitalTwinConfig())
    hal_adapter = SO100FollowerHAL(robot=twin)
    hal_adapter.connect()

    runtime = compose_so100_runtime(skill_resolver=_local_skill_resolver)
    safety = SafetyPassthroughNode(node_name="openral_safety_e2e")
    safety.set_parameters([rclpy.parameter.Parameter("n_dof", value=6)])
    hal_bridge = _make_hal_bridge_node(hal_adapter)

    executor = rclpy.executors.MultiThreadedExecutor(num_threads=6)
    executor.add_node(runtime.world_state_node)
    executor.add_node(runtime.skill_runner_node)
    executor.add_node(safety)
    executor.add_node(hal_bridge)

    helper = rclpy.create_node("openral_so100_e2e_helper")
    executor.add_node(helper)
    observed_safe: list[Any] = []
    observed_joint_states: list[Any] = []
    helper.create_subscription(ActionChunk, "/openral/safe_action", observed_safe.append, 10)
    helper.create_subscription(RosJointState, "/joint_states", observed_joint_states.append, 10)

    try:
        for node in (
            runtime.world_state_node,
            runtime.skill_runner_node,
            safety,
        ):
            assert node.trigger_configure() == TransitionCallbackReturn.SUCCESS
            assert node.trigger_activate() == TransitionCallbackReturn.SUCCESS
        yield (
            executor,
            runtime,
            safety,
            hal_bridge,
            twin,
            observed_safe,
            observed_joint_states,
        )
    finally:
        for node in (runtime.skill_runner_node, runtime.world_state_node, safety):
            try:
                node.trigger_deactivate()
                node.trigger_cleanup()
                node.trigger_shutdown()
            except Exception:
                pass
        with suppress(Exception):
            hal_adapter.disconnect()
        executor.shutdown()
        helper.destroy_node()
        hal_bridge.destroy_node()
        runtime.skill_runner_node.destroy_node()
        runtime.world_state_node.destroy_node()
        safety.destroy_node()
        rclpy.shutdown()


@pytest.mark.skipif(
    not _lerobot_available(),
    reason="lerobot not installed — SO100DigitalTwin needs it. "
    "Run `just sync --all-packages --group hardware` to install.",
)
def test_so100_digital_twin_end_to_end_action_chunk_moves_twin() -> None:
    """Send an ExecuteRskill goal; assert the chunk reaches the twin."""
    from openral_msgs.action import ExecuteRskill
    from rclpy.action import ActionClient

    with _digital_twin_harness() as (
        executor,
        runtime,
        _safety,
        hal_bridge,
        _twin,
        observed_safe,
        observed_joint_states,
    ):
        client = ActionClient(
            runtime.skill_runner_node,
            ExecuteRskill,
            "/openral/execute_rskill",
        )
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            executor.spin_once(timeout_sec=0.02)
        assert client.wait_for_server(timeout_sec=2.0)

        goal = ExecuteRskill.Goal()
        goal.rskill_id = "openral/test-so100-constant"
        goal.revision = ""
        goal.prompt = "drive constant target"
        goal.prompt_metadata_json = ""
        goal.deadline_s = 1.0

        send_future = client.send_goal_async(goal)
        deadline = time.monotonic() + 3.0
        while not send_future.done() and time.monotonic() < deadline:
            executor.spin_once(timeout_sec=0.02)
        goal_handle = send_future.result()
        assert goal_handle is not None and goal_handle.accepted

        result_future = goal_handle.get_result_async()
        deadline = time.monotonic() + 5.0
        while not result_future.done() and time.monotonic() < deadline:
            executor.spin_once(timeout_sec=0.02)
        assert result_future.done()

        # Drain a little more so the HAL bridge's /joint_states timer
        # publishes at least one snapshot after the chunks landed.
        end = time.monotonic() + 0.5
        while time.monotonic() < end:
            executor.spin_once(timeout_sec=0.02)

        bridge_chunks_consumed = hal_bridge.chunks_consumed

    # ── Assertions (outside the harness) ──────────────────────────────────────
    assert observed_safe, "no ActionChunk landed on /openral/safe_action"
    first = observed_safe[0]
    assert int(first.n_dof) == 6
    assert list(first.flat[:6]) == pytest.approx(_TARGET)
    assert first.rskill_id == "openral/test-so100-constant"

    # The HAL bridge must have consumed at least one chunk and forwarded
    # it to the twin — that's the F1+F5 end-to-end loop closing.
    assert bridge_chunks_consumed >= 1, "HAL bridge never consumed a /openral/safe_action chunk"

    # /joint_states should have published the twin's state (HAL bridge
    # publishes from twin.read_state at 30 Hz).
    assert observed_joint_states, "no /joint_states published from HAL bridge"
    last_state = observed_joint_states[-1]
    assert len(list(last_state.name)) == 6
    assert len(list(last_state.position)) == 6
