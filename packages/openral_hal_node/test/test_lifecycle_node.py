"""Lifecycle tests for ``openral_hal_node``, parametrised by robot manifest.

One package hosts the HAL node for every robot, so one test drives the real
``ManifestHALLifecycleNode`` (the class ``lifecycle_node.py`` spins) through
``configure → activate → deactivate → cleanup → shutdown`` for each in-tree
``robots/<id>/robot.yaml`` that declares a sim twin — the same ``robot_yaml`` +
``hal_mode=sim`` parameters ``openral deploy sim`` injects. It replaces the
per-robot copies the retired ``openral_hal_<robot>`` packages each shipped.

Gated on ``rclpy``, ``openral_hal``, ``mujoco`` and ``robot_descriptions``;
missing any causes a clean skip.
"""

from __future__ import annotations

import importlib.util
import time
from pathlib import Path

import pytest

rclpy = pytest.importorskip("rclpy")
pytest.importorskip("openral_hal")
pytest.importorskip("mujoco")
pytest.importorskip("robot_descriptions")

from openral_hal.lifecycle import ManifestHALLifecycleNode
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from rclpy.lifecycle import TransitionCallbackReturn
from rclpy.parameter import Parameter
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import JointState as RosJointState

_ROBOTS_DIR = Path(__file__).resolve().parents[3] / "robots"

# (robot_id, expected joint count of the manifest's RobotDescription)
_SIM_ROBOTS = [
    pytest.param(
        "aloha_bimanual",
        14,
        marks=pytest.mark.skipif(
            importlib.util.find_spec("gym_aloha") is None,
            reason="gym_aloha is not installed; run `just sync --group sim` for the ALOHA twin",
        ),
    ),
    ("franka_panda", 8),
    ("g1", 29),
    ("h1", 19),
    ("openarm", 16),
    ("rizon4", 7),
    ("so100_follower", 6),
    ("ur10e", 6),
    ("ur5e", 6),
]


def _manifest_node(robot_id: str) -> ManifestHALLifecycleNode:
    """The node ``lifecycle_node.py`` spins, parameterised as ``deploy sim`` does."""
    node = ManifestHALLifecycleNode(f"openral_hal_{robot_id}")
    node.set_parameters(
        [
            Parameter("robot_yaml", value=str(_ROBOTS_DIR / robot_id / "robot.yaml")),
            Parameter("hal_mode", value="sim"),
            Parameter("viewer_enabled", value=False),
        ]
    )
    return node


def _teardown(node: ManifestHALLifecycleNode) -> None:
    for transition in ("trigger_deactivate", "trigger_cleanup"):
        try:
            getattr(node, transition)()
        except RuntimeError:
            # RCLError (invalid transition from an already-failed state) /
            # InvalidHandle (rclpy resource already torn down) — both subclass
            # RuntimeError; best-effort teardown.
            pass


def test_lifecycle_entrypoint_imports() -> None:
    from openral_hal_node.lifecycle_node import main

    assert callable(main)


@pytest.mark.parametrize(("robot_id", "n_joints"), _SIM_ROBOTS)
def test_manifest_lifecycle_full_cycle(robot_id: str, n_joints: int) -> None:
    """Every transition succeeds and ``/joint_states`` carries the manifest's DoF."""
    node_name = f"openral_hal_{robot_id}"
    rclpy.init()
    try:
        node = _manifest_node(robot_id)
        executor = rclpy.executors.SingleThreadedExecutor()
        executor.add_node(node)
        helper = rclpy.create_node(f"{node_name}_test_helper")
        executor.add_node(helper)
        received: list[RosJointState] = []
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=10,
        )
        helper.create_subscription(
            RosJointState, f"/{node_name}/joint_states", received.append, qos
        )
        try:
            assert node.trigger_configure() == TransitionCallbackReturn.SUCCESS
            assert node.trigger_activate() == TransitionCallbackReturn.SUCCESS
            assert node._hal is not None
            assert len(node._hal.description.joints) == n_joints

            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline and not received:
                executor.spin_once(timeout_sec=0.02)
            assert received, f"no /{node_name}/joint_states published while ACTIVE"
            assert len(received[-1].position) == n_joints

            assert node.trigger_deactivate() == TransitionCallbackReturn.SUCCESS
            assert node.trigger_cleanup() == TransitionCallbackReturn.SUCCESS
            assert node.trigger_shutdown() == TransitionCallbackReturn.SUCCESS
        finally:
            executor.remove_node(helper)
            executor.remove_node(node)
            helper.destroy_node()
            node.destroy_node()
    finally:
        rclpy.shutdown()


def test_lifecycle_emits_hal_read_state_span(captured_spans: InMemorySpanExporter) -> None:
    """The node emits ``hal.read_state`` spans with the dashboard's Identity attrs.

    Dashboard contract: the per-tick span carries ``openral.hal.adapter``,
    ``openral.hal.robot.model``, ``openral.tick.idx``, plus the per-joint
    reality arrays (``names`` / ``positions``).
    """
    rclpy.init()
    try:
        node = _manifest_node("ur5e")
        executor = rclpy.executors.SingleThreadedExecutor()
        executor.add_node(node)
        try:
            assert node.trigger_configure() == TransitionCallbackReturn.SUCCESS
            assert node.trigger_activate() == TransitionCallbackReturn.SUCCESS
            adapter = type(node._hal).__name__.lower()
            deadline = time.monotonic() + 0.5
            while time.monotonic() < deadline:
                executor.spin_once(timeout_sec=0.02)
        finally:
            _teardown(node)
            executor.remove_node(node)
            node.destroy_node()
    finally:
        rclpy.shutdown()

    spans = [s for s in captured_spans.get_finished_spans() if s.name == "hal.read_state"]
    assert spans, "the HAL node did not emit any hal.read_state spans"
    attrs = dict(spans[0].attributes or {})
    assert attrs.get("openral.hal.adapter") == adapter
    assert str(attrs.get("openral.hal.robot.model"))
    assert attrs.get("openral.tick.idx") == 0
    assert len(list(attrs.get("openral.hal.joint.names") or [])) == 6
    assert len(list(attrs.get("openral.hal.joint.positions") or [])) == 6


def test_lifecycle_emits_hal_send_action_span(captured_spans: InMemorySpanExporter) -> None:
    """A ``/openral/safe_action`` publication produces a ``hal.send_action`` span."""
    pytest.importorskip("openral_msgs.msg", reason="openral_msgs not built; run `just ros2-build`")
    from openral_msgs.msg import ActionChunk

    n_joints = 6
    rclpy.init()
    try:
        node = _manifest_node("ur5e")
        executor = rclpy.executors.MultiThreadedExecutor(num_threads=2)
        executor.add_node(node)
        helper = rclpy.create_node("test_send_action_helper")
        executor.add_node(helper)
        try:
            assert node.trigger_configure() == TransitionCallbackReturn.SUCCESS
            assert node.trigger_activate() == TransitionCallbackReturn.SUCCESS
            adapter = type(node._hal).__name__.lower()
            deadline = time.monotonic() + 0.3
            while time.monotonic() < deadline:
                executor.spin_once(timeout_sec=0.02)
            chunk_qos = QoSProfile(
                reliability=QoSReliabilityPolicy.RELIABLE,
                durability=QoSDurabilityPolicy.VOLATILE,
                depth=1,
            )
            pub = helper.create_publisher(ActionChunk, "/openral/safe_action", chunk_qos)
            chunk = ActionChunk()
            chunk.n_dof = n_joints
            chunk.horizon = 1
            chunk.flat = [0.1] * n_joints
            chunk.rskill_id = "openral/test-generic-hal-span"
            pub.publish(chunk)
            deadline = time.monotonic() + 0.5
            while time.monotonic() < deadline:
                executor.spin_once(timeout_sec=0.02)
        finally:
            _teardown(node)
            executor.remove_node(helper)
            executor.remove_node(node)
            helper.destroy_node()
            node.destroy_node()
    finally:
        rclpy.shutdown()

    spans = [s for s in captured_spans.get_finished_spans() if s.name == "hal.send_action"]
    assert spans, "the HAL node did not emit a hal.send_action span on safe_action"
    attrs = dict(spans[0].attributes or {})
    assert attrs.get("openral.hal.adapter") == adapter
    assert attrs.get("openral.hal.control_mode") == "joint_position"
    assert attrs.get("openral.hal.action.source") == "safe_action"
    assert len(list(attrs.get("openral.hal.action.next") or [])) == n_joints
    assert attrs.get("openral.hal.action.dim") == n_joints
    assert attrs.get("openral.hal.action.applied") is True
