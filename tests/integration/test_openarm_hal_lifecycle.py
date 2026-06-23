"""Lifecycle integration test for ``packages/openral_hal_openarm``.

Exercises the real production lifecycle node end-to-end:

* construct → trigger_configure (loads MJCF, builds MjData)
* trigger_activate (timer + /openral/safe_action sub + /openral/estop sub)
* publish a real ``openral_msgs/ActionChunk`` on ``/openral/safe_action``
  and assert the OpenArm twin's joint state advances toward the target
* publish ``std_msgs/Empty`` on ``/openral/estop`` and assert the latch
  blocks subsequent safe_action commands
* trigger_deactivate + trigger_cleanup

Per CLAUDE.md §1.11 / §5.4: real ``rclpy``, real ``openral_msgs``, real
``OpenArmMujocoHAL`` against the upstream v2 bimanual MJCF. Skipped
when ROS 2 is not sourced or ``mujoco`` is not installed.
"""

from __future__ import annotations

import importlib.util
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Any

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

_OPENARM_YAML = Path(__file__).resolve().parents[2] / "robots" / "openarm" / "robot.yaml"

_ROS2_AVAILABLE = bool(os.environ.get("ROS_DISTRO")) and (
    importlib.util.find_spec("openral_msgs") is not None
)  # custom IDL must be colcon-built, not just ROS sourced (skip cleanly otherwise)

pytestmark = pytest.mark.skipif(
    not _ROS2_AVAILABLE,
    reason="ROS_DISTRO not set — these tests require a sourced ROS 2 installation.",
)

# Gate on mujoco so hosts without the sim group skip cleanly rather
# than crashing on the MJCF load inside on_configure.
pytest.importorskip("mujoco")

# Headless MuJoCo for CI hosts that lack a display server.
os.environ.setdefault("MUJOCO_GL", "egl")


@pytest.fixture
def captured_spans() -> Iterator[InMemorySpanExporter]:
    """Install an in-memory OTel tracer and yield the exporter."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    # opentelemetry-api guards against re-setting the global provider once it
    # has been set; bypass that for the test by zeroing the set-once flag.
    trace._TRACER_PROVIDER_SET_ONCE._done = False  # type: ignore[attr-defined]  # reason: test-only reset
    trace._TRACER_PROVIDER = None  # type: ignore[attr-defined]  # reason: test-only reset
    trace.set_tracer_provider(provider)
    yield exporter
    exporter.clear()


@contextmanager
def _lifecycle_harness() -> Iterator[tuple[Any, Any, dict[str, list[Any]]]]:
    """Bring up the generic manifest node for openarm + a helper observer."""
    import rclpy
    from openral_hal.lifecycle import ManifestHALLifecycleNode
    from rclpy.lifecycle import TransitionCallbackReturn
    from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
    from sensor_msgs.msg import JointState as RosJointState

    rclpy.init()

    # issue #191 Phase 3b — openarm runs on the generic node now. Scene
    # composition + HAL kwargs (gravity_enabled=False, settle_steps=4 so a single
    # safe_action moves qpos visibly) come from the manifest; cameras render via
    # SimSensorBridge + OpenArmMujocoHAL.read_images. viewer off (headless).
    node = ManifestHALLifecycleNode("openral_hal_openarm")
    node.set_parameters(
        [
            rclpy.parameter.Parameter("robot_yaml", value=str(_OPENARM_YAML)),
            rclpy.parameter.Parameter("hal_mode", value="sim"),
            rclpy.parameter.Parameter("publish_rate_hz", value=60.0),
            rclpy.parameter.Parameter("viewer_enabled", value=False),
        ],
    )

    helper = rclpy.create_node("openral_hal_openarm_test_helper")
    observed: dict[str, list[Any]] = {"joint_states": []}
    control_qos = QoSProfile(
        reliability=QoSReliabilityPolicy.RELIABLE,
        durability=QoSDurabilityPolicy.VOLATILE,
        depth=10,
    )
    helper.create_subscription(
        RosJointState,
        "/joint_states",
        observed["joint_states"].append,
        control_qos,
    )

    # SingleThreadedExecutor mirrors how the node actually runs in production
    # (`make_lifecycle_main*` → `rclpy.spin(node)` is single-threaded). It also
    # keeps the OpenArm node's offscreen MuJoCo renderer thread-correct: the
    # `mujoco.Renderer` (EGL context) is created on this thread in
    # `on_configure`, and `_spin_for` below runs the camera-render timer on the
    # SAME thread via `spin_once`. A MultiThreadedExecutor would dispatch that
    # callback to a worker thread, and the thread-affine `eglMakeCurrent` would
    # raise EGL_BAD_ACCESS and abort the process headless.
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(node)
    executor.add_node(helper)

    try:
        assert node.trigger_configure() == TransitionCallbackReturn.SUCCESS
        assert node.trigger_activate() == TransitionCallbackReturn.SUCCESS
        yield executor, node, observed
    finally:
        with suppress(Exception):
            node.trigger_deactivate()
            node.trigger_cleanup()
            node.trigger_shutdown()
        executor.shutdown()
        helper.destroy_node()
        node.destroy_node()
        rclpy.shutdown()


def _spin_for(executor: Any, duration_s: float) -> None:
    """Spin ``executor`` for ``duration_s`` seconds of wall time."""
    deadline = time.monotonic() + duration_s
    while time.monotonic() < deadline:
        executor.spin_once(timeout_sec=0.02)


def test_activate_publishes_sixteen_dof_joint_states() -> None:
    """The HAL's read_state → /joint_states wiring publishes 16-DoF samples after activate."""
    with _lifecycle_harness() as (executor, _node, observed):
        _spin_for(executor, 0.5)

    assert observed["joint_states"], "no /joint_states message was published after activate"
    msg = observed["joint_states"][0]
    assert len(msg.name) == 16, f"expected 16 joint names, got {len(msg.name)}"
    assert len(msg.position) == 16, f"expected 16 positions, got {len(msg.position)}"


def test_cameras_publish_rgb_frames() -> None:
    """SimSensorBridge + OpenArmMujocoHAL.read_images publish the manifest cameras.

    issue #191 Phase 3b — the composed scene's MJCF camera "top" renders the
    "top" RGB sensor (the MJCF and sensor names match per ADR-0070); the frame
    is published on ``/openral/cameras/top/image`` headless (EGL on the
    executor thread).
    """
    import rclpy
    from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
    from sensor_msgs.msg import Image as RosImage

    with _lifecycle_harness() as (executor, _node, _observed):
        sub_node = rclpy.create_node("openarm_camera_test_sub")
        frames: list[Any] = []
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1,
        )
        sub_node.create_subscription(RosImage, "/openral/cameras/top/image", frames.append, qos)
        executor.add_node(sub_node)
        _spin_for(executor, 1.5)
        sub_node.destroy_node()

    assert frames, "no frame published on /openral/cameras/top/image"
    img = frames[0]
    assert img.encoding == "rgb8", img.encoding
    assert img.width == 640 and img.height == 480, (img.width, img.height)


def test_safe_action_drives_hal_send_action() -> None:
    """An ActionChunk on /openral/safe_action moves the digital twin's joints."""
    import rclpy
    from openral_msgs.msg import ActionChunk
    from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy

    with _lifecycle_harness() as (executor, _node, observed):
        _spin_for(executor, 0.3)
        assert observed["joint_states"], "no baseline /joint_states message"
        baseline = list(observed["joint_states"][-1].position)

        pub_node = rclpy.create_node("openral_hal_openarm_test_publisher")
        executor.add_node(pub_node)
        chunk_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1,
        )
        pub = pub_node.create_publisher(ActionChunk, "/openral/safe_action", chunk_qos)

        # Drive every joint to baseline + 0.2 rad and let the v2 position
        # actuators converge for a couple of HAL settle steps.
        target = [float(v) + 0.2 for v in baseline]
        chunk = ActionChunk()
        chunk.n_dof = 16
        chunk.horizon = 1
        chunk.flat = target
        chunk.rskill_id = "openral/test-openarm-lifecycle"
        pub.publish(chunk)

        _spin_for(executor, 1.5)

        post = list(observed["joint_states"][-1].position)
        deltas = [abs(p - b) for p, b in zip(post, baseline, strict=True)]
        # The contract under test is "safe_action reaches the HAL and the
        # twin actuates"; per-joint convergence speed depends on each
        # actuator's PD class (DM8009 / DM4340 / DM4310 / fingers) and is
        # not what this test is gating. Assert that at least one big arm
        # joint clearly moved past the noise floor.
        max_delta = max(deltas)
        assert max_delta > 0.02, (
            f"safe_action did not advance the HAL (max joint delta={max_delta:.4f}, "
            f"deltas={deltas})"
        )


def test_publish_joint_state_emits_hal_read_state_span(
    captured_spans: InMemorySpanExporter,
) -> None:
    """Each timer tick emits an OTel ``hal.read_state`` span with joint reality attrs."""
    with _lifecycle_harness() as (executor, _node, _observed):
        _spin_for(executor, 0.3)
    spans = [s for s in captured_spans.get_finished_spans() if s.name == "hal.read_state"]
    assert spans, "no hal.read_state span emitted by openral_hal_openarm timer"
    attrs = dict(spans[0].attributes or {})
    assert attrs.get("openral.hal.adapter") == "openarmmujocohal"
    assert attrs.get("openral.hal.robot.model"), "openral.hal.robot.model must be set"
    assert attrs.get("openral.tick.idx") == 0
    names = list(attrs.get("openral.hal.joint.names") or [])
    positions = list(attrs.get("openral.hal.joint.positions") or [])
    assert len(names) == 16 and len(positions) == 16, (
        f"hal.read_state span must carry 16-DoF joint vectors (names={len(names)}, "
        f"positions={len(positions)})"
    )


def test_on_safe_action_emits_hal_send_action_span(
    captured_spans: InMemorySpanExporter,
) -> None:
    """A safe_action publication produces an OTel ``hal.send_action`` span."""
    import rclpy
    from openral_msgs.msg import ActionChunk
    from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy

    with _lifecycle_harness() as (executor, _node, observed):
        _spin_for(executor, 0.2)
        baseline = list(observed["joint_states"][-1].position)
        pub_node = rclpy.create_node("openral_hal_openarm_test_span_pub")
        executor.add_node(pub_node)
        chunk_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1,
        )
        pub = pub_node.create_publisher(ActionChunk, "/openral/safe_action", chunk_qos)
        chunk = ActionChunk()
        chunk.n_dof = 16
        chunk.horizon = 1
        chunk.flat = [float(v) + 0.1 for v in baseline]
        chunk.rskill_id = "openral/test-openarm-spans"
        pub.publish(chunk)
        _spin_for(executor, 0.5)

    spans = [s for s in captured_spans.get_finished_spans() if s.name == "hal.send_action"]
    assert spans, "no hal.send_action span emitted after /openral/safe_action publish"
    attrs = dict(spans[0].attributes or {})
    assert attrs.get("openral.hal.adapter") == "openarmmujocohal"
    assert attrs.get("openral.hal.control_mode") == "joint_position"
    next_row = list(attrs.get("openral.hal.action.next") or [])
    assert len(next_row) == 16, f"action.next must carry the 16-DoF row, got {len(next_row)}"
    assert attrs.get("openral.hal.action.dim") == 16
    assert attrs.get("openral.hal.action.applied") is True


def test_estop_latch_blocks_subsequent_safe_action() -> None:
    """An /openral/estop publication latches the HAL against further safe_action."""
    import rclpy
    from openral_msgs.msg import ActionChunk
    from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
    from std_msgs.msg import Empty

    with _lifecycle_harness() as (executor, node, observed):
        _spin_for(executor, 0.3)
        assert observed["joint_states"], "no baseline /joint_states message"

        pub_node = rclpy.create_node("openral_hal_openarm_test_estop_publisher")
        executor.add_node(pub_node)
        estop_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=10,
        )
        chunk_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1,
        )
        estop_pub = pub_node.create_publisher(Empty, "/openral/estop", estop_qos)
        action_pub = pub_node.create_publisher(ActionChunk, "/openral/safe_action", chunk_qos)

        estop_pub.publish(Empty())
        _spin_for(executor, 0.2)
        assert node._estopped is True, "estop latch did not engage"

        baseline = list(observed["joint_states"][-1].position)
        chunk = ActionChunk()
        chunk.n_dof = 16
        chunk.horizon = 1
        chunk.flat = [float(v) + 0.3 for v in baseline]
        chunk.rskill_id = "openral/test-openarm-lifecycle"
        action_pub.publish(chunk)

        _spin_for(executor, 0.5)

        post = list(observed["joint_states"][-1].position)
        deltas = [abs(p - b) for p, b in zip(post, baseline, strict=True)]
        # Estop latch must hold the HAL — no joint should move appreciably.
        max_delta = max(deltas)
        assert max_delta < 0.01, (
            f"estop latch did not block safe_action (max joint delta={max_delta:.4f})"
        )
