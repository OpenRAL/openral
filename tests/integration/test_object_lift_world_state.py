"""E2E integration tests for the object-lift path in the world-state lifecycle node.

ADR-0035 — Task 8: drives the real ``_WorldStateLifecycleNode`` object-lift pipeline
with real components (real node, real VoxelFrustumLifter + ObjectMemory, real
TF2 transforms, real OccupancyVoxels + PromptStamped messages).

Scenario overview
-----------------
All tests share a simple scene: ``base_link`` == ``head_rgb_optical`` == ``map``
(identity transforms).  A 5×5×5 voxel cube at (0,0,2) in base_link projects
to pixel ≈(50,50) on a 100×100 fx=fy=100 cx=cy=50 camera, and lands inside
the detection box (40,40,60,60).

1. **happy_path** — voxels + detection → object remembered in map frame at z≈2.
2. **best_effort_no_voxels** — lift enabled, no voxel grid published → snapshot
   detected_objects stays empty (best-effort proof).
3. **eviction** — happy path object is remembered (topic-driven, so the detection
   camera enters ``_seen_sensor_ids``); then the producer goes silent (the real
   detector publishes NOTHING when it detects nothing) and a memory tick alone —
   no new detection message — evicts the track because the camera FOV (built from
   the camera pose every tick) still covers the object yet it was not re-detected.

All tests skip cleanly when the ROS overlay is not sourced.
"""

from __future__ import annotations

import importlib.util
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import pytest

pytest.importorskip("rclpy")
pytest.importorskip("openral_msgs")
pytest.importorskip("tf2_ros")

_ROS2_AVAILABLE = bool(os.environ.get("ROS_DISTRO")) and (
    importlib.util.find_spec("openral_msgs") is not None
)  # custom IDL must be colcon-built, not just ROS sourced (skip cleanly otherwise)

pytestmark = pytest.mark.skipif(
    not _ROS2_AVAILABLE,
    reason="ROS_DISTRO not set — these tests require a sourced ROS 2 installation.",
)

# ---------------------------------------------------------------------------
# Shared scene constants
# ---------------------------------------------------------------------------
_FRAME_W = 100
_FRAME_H = 100
_FX = _FY = 100.0
_CX = _CY = 50.0
# Voxel cube: origin such that centres span ≈ (0,0,2)
_VOX_ORIGIN = (-0.1, -0.1, 1.9)
_VOX_RES = 0.05
_VOX_SIZE = (5, 5, 5)  # 5×5×5 = 125 voxels, all occupied

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _spin_for(executor: Any, seconds: float) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        executor.spin_once(timeout_sec=0.02)


def _spin_until(executor: Any, predicate: Any, timeout_s: float = 3.0) -> bool:
    """Spin the executor until ``predicate()`` is truthy or ``timeout_s`` expires.

    Returns ``True`` if the predicate became true, ``False`` on timeout.
    """
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        executor.spin_once(timeout_sec=0.02)
        if predicate():
            return True
    return False


def _make_robot_desc() -> Any:
    """Build a minimal RobotDescription with one RGB SensorSpec."""
    from openral_core.schemas import (
        ControlMode,
        EmbodimentKind,
        IntrinsicsPinhole,
        JointSpec,
        JointType,
        RobotCapabilities,
        RobotDescription,
        SafetyEnvelope,
        SensorModality,
        SensorSpec,
    )

    sensor = SensorSpec(
        name="head_rgb",
        modality=SensorModality.RGB,
        frame_id="head_rgb_optical",
        rate_hz=30.0,
        intrinsics=IntrinsicsPinhole(
            width=_FRAME_W,
            height=_FRAME_H,
            fx=_FX,
            fy=_FY,
            cx=_CX,
            cy=_CY,
        ),
    )
    return RobotDescription(
        name="test_robot",
        embodiment_kind=EmbodimentKind.MANIPULATOR,
        joints=[
            JointSpec(
                name="j0",
                joint_type=JointType.REVOLUTE,
                parent_link="base_link",
                child_link="link_0",
            )
        ],
        capabilities=RobotCapabilities(
            supported_control_modes=[ControlMode.JOINT_POSITION],
        ),
        safety=SafetyEnvelope(),
        sensors=[sensor],
    )


def _make_voxels_msg(helper: Any) -> Any:
    """Build an OccupancyVoxels message with a solid 5×5×5 cube at (0,0,2)."""
    from geometry_msgs.msg import Point  # type: ignore[import-untyped]
    from openral_msgs.msg import OccupancyVoxels  # type: ignore[import-untyped]

    msg = OccupancyVoxels()
    msg.header.stamp = helper.get_clock().now().to_msg()
    msg.header.frame_id = "base_link"
    msg.origin = Point(x=_VOX_ORIGIN[0], y=_VOX_ORIGIN[1], z=_VOX_ORIGIN[2])
    msg.resolution = _VOX_RES
    sx, sy, sz = _VOX_SIZE
    msg.size_x = sx
    msg.size_y = sy
    msg.size_z = sz
    msg.occupancy = bytes([1] * (sx * sy * sz))
    return msg


def _make_detection_msg(helper: Any) -> Any:
    """Build a PromptStamped detection event for ``head_rgb`` (one ``cup``).

    Mirrors the real perception_tee output: a single ``ObjectsMetadata`` with
    one detection box. The real detector never publishes an empty-detection
    message — it stays silent when it sees nothing — so this helper only ever
    produces a populated detection.
    """
    from openral_core.schemas import ObjectDetection2D, ObjectsMetadata
    from openral_msgs.msg import PromptStamped  # type: ignore[import-untyped]

    md = ObjectsMetadata(
        sensor_id="head_rgb",
        detections=[ObjectDetection2D(label="cup", confidence=0.9, bbox_xyxy=(40, 40, 60, 60))],
        model_id="test_detector",
        frame_width=_FRAME_W,
        frame_height=_FRAME_H,
    )
    msg = PromptStamped()
    msg.header.stamp = helper.get_clock().now().to_msg()
    msg.header.frame_id = "head_rgb"
    msg.text = ""
    msg.metadata_json = md.model_dump_json()
    return msg


def _publish_identity_tf(helper: Any, executor: Any) -> None:
    """Publish identity static transforms map←base_link and head_rgb_optical←base_link."""
    import geometry_msgs.msg as gm  # type: ignore[import-untyped]
    import tf2_ros  # type: ignore[import-untyped]

    broadcaster = tf2_ros.StaticTransformBroadcaster(helper)

    def _identity_tf(parent: str, child: str) -> gm.TransformStamped:
        ts = gm.TransformStamped()
        ts.header.stamp = helper.get_clock().now().to_msg()
        ts.header.frame_id = parent
        ts.child_frame_id = child
        ts.transform.translation.x = 0.0
        ts.transform.translation.y = 0.0
        ts.transform.translation.z = 0.0
        ts.transform.rotation.x = 0.0
        ts.transform.rotation.y = 0.0
        ts.transform.rotation.z = 0.0
        ts.transform.rotation.w = 1.0
        return ts

    broadcaster.sendTransform(
        [
            _identity_tf("map", "base_link"),
            _identity_tf("base_link", "head_rgb_optical"),
        ]
    )
    # Spin briefly so the node's TransformListener receives the static TF.
    _spin_for(executor, 0.15)


# ---------------------------------------------------------------------------
# Shared lifecycle harness
# ---------------------------------------------------------------------------


@contextmanager
def _object_lift_harness(
    *,
    lift_enabled: bool = True,
    cadence_hz: float = 20.0,
    max_misses: int = 1,
    provide_tf: bool = True,
) -> Iterator[tuple[Any, Any, Any, Any, Any]]:
    """Bring up the world-state lifecycle node in object-lift mode.

    Yields ``(executor, node, helper, vox_pub, det_pub)``.
    """
    import rclpy  # type: ignore[import-untyped]
    from openral_msgs.msg import OccupancyVoxels, PromptStamped  # type: ignore[import-untyped]
    from openral_world_state import WorldStateAggregator
    from openral_world_state_ros.lifecycle_node import _WorldStateLifecycleNode
    from rclpy.lifecycle import TransitionCallbackReturn  # type: ignore[import-untyped]
    from rclpy.qos import (  # type: ignore[import-untyped]
        QoSDurabilityPolicy,
        QoSProfile,
        QoSReliabilityPolicy,
    )

    desc = _make_robot_desc()

    rclpy.init()
    aggregator = WorldStateAggregator(desc)
    node = _WorldStateLifecycleNode(aggregator)

    node.set_parameters(
        [
            rclpy.parameter.Parameter("publish_rate_hz_fast", value=30.0),
            rclpy.parameter.Parameter("publish_rate_hz_slow", value=5.0),
            rclpy.parameter.Parameter("object_lift_enabled", value=lift_enabled),
            rclpy.parameter.Parameter("object_lift_memory_cadence_hz", value=cadence_hz),
            rclpy.parameter.Parameter("object_lift_k_nearest", value=25),
            rclpy.parameter.Parameter("object_lift_min_voxels", value=3),
            rclpy.parameter.Parameter("object_lift_iou_threshold", value=0.3),
            rclpy.parameter.Parameter("object_lift_max_misses", value=max_misses),
            rclpy.parameter.Parameter("object_lift_voxel_staleness_s", value=2.0),
            rclpy.parameter.Parameter("object_lift_map_frame", value="map"),
        ]
    )

    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(node)

    helper = rclpy.create_node("test_object_lift_helper")
    executor.add_node(helper)

    vox_qos = QoSProfile(
        reliability=QoSReliabilityPolicy.RELIABLE,
        durability=QoSDurabilityPolicy.VOLATILE,
        depth=1,
    )
    det_qos = QoSProfile(
        reliability=QoSReliabilityPolicy.BEST_EFFORT,
        durability=QoSDurabilityPolicy.VOLATILE,
        depth=5,
    )
    vox_pub = helper.create_publisher(OccupancyVoxels, "/openral/world_voxels", vox_qos)
    det_pub = helper.create_publisher(PromptStamped, "/openral/perception/objects", det_qos)

    try:
        assert node.trigger_configure() == TransitionCallbackReturn.SUCCESS
        assert node.trigger_activate() == TransitionCallbackReturn.SUCCESS

        if provide_tf:
            _publish_identity_tf(helper, executor)

        yield executor, node, helper, vox_pub, det_pub
    finally:
        try:
            node.trigger_deactivate()
            node.trigger_cleanup()
        except Exception:
            pass
        executor.remove_node(helper)
        helper.destroy_node()
        node.destroy_node()
        rclpy.shutdown()


# ---------------------------------------------------------------------------
# Scenario 1 — happy path
# ---------------------------------------------------------------------------


def test_object_lift_happy_path() -> None:
    """Voxel grid + detection → object remembered in map frame at z≈2.

    Publishes one OccupancyVoxels grid + one PromptStamped detection on
    the real ROS topics, waits for the memory tick, and asserts that
    snapshot().detected_objects contains a cup at z≈2 in the map frame.

    The node is configured with ``max_misses=5`` so a single detection
    survives the assert window: once the camera enters ``_seen_sensor_ids``
    the in-FOV eviction would otherwise drop the track after one missed
    tick, which would race the assertion.
    """
    with _object_lift_harness(max_misses=5) as (executor, node, helper, vox_pub, det_pub):
        # Publish the voxel grid first, then the detection.
        vox_pub.publish(_make_voxels_msg(helper))
        _spin_for(executor, 0.05)
        det_pub.publish(_make_detection_msg(helper))

        def _has_object() -> bool:
            snap = node._aggregator.snapshot()
            return len(snap.detected_objects) >= 1

        ok = _spin_until(executor, _has_object, timeout_s=4.0)
        snap = node._aggregator.snapshot()

        assert ok, (
            f"No detected object within 4 s; snapshot has {len(snap.detected_objects)} objects."
        )
        assert len(snap.detected_objects) == 1
        obj = snap.detected_objects[0]
        assert obj.label == "cup", f"Expected label 'cup', got {obj.label!r}"
        assert obj.pose.frame_id == "map", f"Expected frame_id 'map', got {obj.pose.frame_id!r}"
        assert obj.pose.xyz[2] == pytest.approx(2.0, abs=0.3), (
            f"Expected z≈2.0 m, got z={obj.pose.xyz[2]:.3f}"
        )


# ---------------------------------------------------------------------------
# Scenario 2 — best-effort: no voxels → empty detected_objects
# ---------------------------------------------------------------------------


def test_object_lift_no_voxels_best_effort() -> None:
    """Lift enabled but no voxel grid published → detected_objects stays empty.

    Proves that the best-effort skip path works: the node remains
    functional (produces snapshots) without crashing or blocking.
    """
    with _object_lift_harness() as (executor, node, helper, _vox_pub, det_pub):
        # Publish only a detection, no voxels.
        det_pub.publish(_make_detection_msg(helper))

        # Run for ~1s — memory tick must fire; object should NOT appear.
        _spin_for(executor, 1.0)

        snap = node._aggregator.snapshot()
        assert snap.detected_objects == [], (
            f"Expected empty detected_objects without voxels, got {snap.detected_objects!r}"
        )


# ---------------------------------------------------------------------------
# Scenario 3 — eviction after max_misses
# ---------------------------------------------------------------------------


def test_object_lift_eviction() -> None:
    """A remembered object is evicted when the silent detector stops re-detecting it.

    This models the REAL producer contract: ``perception_tee`` publishes NOTHING
    when ``postprocess_rtdetr`` returns no detections. So once an object leaves
    the scene, zero messages arrive — yet the track must still be evicted because
    the camera that saw it is still pointed at the (now empty) region.

    Approach (no synthetic empty-detection message — the real detector never
    sends one):

    1. **Establish** the object topic-driven (publish voxels + one detection).
       This routes through the real ``_on_objects`` callback, which records
       ``head_rgb`` in ``_seen_sensor_ids``.
    2. **Go silent** — publish no further detections.
    3. **Tick** the real ``_on_memory_tick`` directly (deterministic, no ROS
       timing race). The in-FOV predicate is rebuilt from ``head_rgb``'s current
       pose alone (identity TF → object at z≈2 projects to ≈(50,50), inside the
       100×100 image), so the object is judged in-view, was not re-detected, and
       with ``max_misses=1`` is evicted on this single missed tick.
    """
    with _object_lift_harness(max_misses=1) as (executor, node, helper, vox_pub, det_pub):
        # --- Phase 1: establish the object (topic-driven) ---
        vox_pub.publish(_make_voxels_msg(helper))
        _spin_for(executor, 0.05)
        det_pub.publish(_make_detection_msg(helper))

        ok = _spin_until(
            executor,
            lambda: len(node._aggregator.snapshot().detected_objects) >= 1,
            timeout_s=4.0,
        )
        assert ok, "Object did not appear within 4 s — cannot test eviction."
        # The detection camera is now remembered for FOV-based eviction.
        assert "head_rgb" in node._seen_sensor_ids

        # --- Phase 2: silent detector — no new detection message at all ---
        # Fire the memory tick directly. No `_on_objects` call, no empty message.
        # in_fov fires from the camera pose (object at z≈2 ≈(50,50) in-frame),
        # miss_count reaches 1 == max_misses, so the object is evicted.
        node._on_memory_tick()

        snap = node._aggregator.snapshot()
        assert snap.detected_objects == [], (
            f"Expected eviction after one silent in-view tick, got {snap.detected_objects!r}"
        )
