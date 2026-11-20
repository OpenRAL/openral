"""E2E integration tests for the object-lift path in the world-state lifecycle node.

Task 8: drives the real ``_WorldStateLifecycleNode`` object-lift pipeline with real
components (real node, real VoxelFrustumLifter + ObjectMemory, real TF2 transforms, real
OccupancyVoxels + PromptStamped messages).

All tests share a simple scene: ``base_link`` == ``head_rgb_optical`` == ``map`` (identity
transforms). A 5x5x5 voxel cube at (0,0,2) in base_link projects to pixel ≈(50,50) on a
100x100 fx=fy=100 cx=cy=50 camera, landing inside the detection box (40,40,60,60).

1. happy_path — voxels + detection → object remembered in map frame at z≈2.
2. best_effort_no_voxels — lift enabled, no voxel grid published → detected_objects stays
   empty (best-effort proof).
3. removed → evicted — the object's voxels clear and the wall behind shows through; the
   silent detector is no obstacle (the depth source is the witness), and the track goes
   after a few clear observations, not on the first (issue #636).
4. occluded → kept — an occluder in front, or the object's own voxels still occupied,
   is no evidence however long it lasts.
5. undetected-but-occupied counts only on a tick the detector delivered a batch.
6. persistence — a restarted node restores what it remembered as uncertain, then a fresh
   detection confirms it or a clear view removes it.

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
# Voxel column along the optical axis, z ∈ [1.0, 3.25): the object is the cube at z≈2,
# an occluder slab sits in front of it at z≈1 and a wall behind it at z≈3.
_VOX_ORIGIN = (-0.1, -0.1, 1.0)
_VOX_RES = 0.05
_VOX_SIZE = (5, 5, 45)
_CUBE_Z = range(18, 23)  # centres 1.925 .. 2.125
_OCCLUDER_Z = range(0, 1)  # centre 1.025
_WALL_Z = range(40, 41)  # centre 3.025

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


def _make_voxels_msg(helper: Any, *layers: range) -> Any:
    """Build an OccupancyVoxels message occupying ``layers`` (default: the cube at z≈2)."""
    from geometry_msgs.msg import Point, Quaternion
    from openral_msgs.msg import OccupancyVoxels

    msg = OccupancyVoxels()
    msg.header.stamp = helper.get_clock().now().to_msg()
    msg.header.frame_id = "base_link"
    msg.origin = Point(x=_VOX_ORIGIN[0], y=_VOX_ORIGIN[1], z=_VOX_ORIGIN[2])
    # Synthetic base-aligned lattice: `OccupancyVoxels` is oriented, and its
    # unset orientation is the all-zero quaternion, which the lift refuses
    # rather than reads as identity.
    msg.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
    msg.resolution = _VOX_RES
    sx, sy, sz = _VOX_SIZE
    msg.size_x = sx
    msg.size_y = sy
    msg.size_z = sz
    occupied = {z for layer in (layers or (_CUBE_Z,)) for z in layer}
    msg.occupancy = bytes(int(z in occupied) for z in range(sz) for _ in range(sx * sy))
    return msg


def _publish_voxels(executor: Any, node: Any, vox_pub: Any, msg: Any) -> None:
    """Publish a grid and spin until the node holds it."""
    vox_pub.publish(msg)
    assert _spin_until(
        executor,
        lambda: (
            node._latest_voxels is not None and node._latest_voxels.header.stamp == msg.header.stamp
        ),
    ), "the node never received the voxel grid"


def _observe(executor: Any, node: Any, helper: Any, vox_pub: Any, *layers: range) -> None:
    """One observation: a freshly stamped grid occupying ``layers``, then one memory tick.

    The memory weighs a miss once per new depth observation, so each tick gets its own.
    """
    _publish_voxels(executor, node, vox_pub, _make_voxels_msg(helper, *layers))
    node._on_memory_tick()


def _remember_cup(executor: Any, node: Any, helper: Any, vox_pub: Any, det_pub: Any) -> None:
    """Topic-driven: publish the cube and one ``cup`` detection, then tick once.

    The cube alone: the wall behind it projects onto the same pixels, and the lift's
    K-nearest (by pixel distance to the box centre) would take it into the object's box.
    """
    _publish_voxels(executor, node, vox_pub, _make_voxels_msg(helper, _CUBE_Z))
    det_pub.publish(_make_detection_msg(helper))
    assert _spin_until(executor, lambda: bool(node._candidate_buffer)), "no lifted candidate"
    node._on_memory_tick()
    assert [o.label for o in node._aggregator.snapshot().detected_objects] == ["cup"]


def _make_detection_msg(helper: Any, label: str = "cup") -> Any:
    """Build a PromptStamped detection event for ``head_rgb`` (one ``cup``).

    Mirrors the real perception_tee output: a single ``ObjectsMetadata`` with
    one detection box. The real detector never publishes an empty-detection
    message — it stays silent when it sees nothing — so this helper only ever
    produces a populated detection.
    """
    from openral_core.schemas import ObjectDetection2D, ObjectsMetadata
    from openral_msgs.msg import PromptStamped

    md = ObjectsMetadata(
        sensor_id="head_rgb",
        detections=[ObjectDetection2D(label=label, confidence=0.9, bbox_xyxy=(40, 40, 60, 60))],
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
    import geometry_msgs.msg as gm
    import tf2_ros

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
    provide_tf: bool = True,
    miss_occupied_log_odds: float = -0.1,
    memory_path: str = "",
) -> Iterator[tuple[Any, Any, Any, Any, Any]]:
    """Bring up the world-state lifecycle node in object-lift mode.

    Yields ``(executor, node, helper, vox_pub, det_pub)``.
    """
    import rclpy
    from openral_msgs.msg import OccupancyVoxels, PromptStamped
    from openral_world_state import WorldStateAggregator
    from openral_world_state_ros.lifecycle_node import _WorldStateLifecycleNode
    from rclpy.lifecycle import TransitionCallbackReturn
    from rclpy.qos import (
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
            rclpy.parameter.Parameter(
                "object_lift_miss_occupied_log_odds", value=miss_occupied_log_odds
            ),
            rclpy.parameter.Parameter("object_memory_path", value=memory_path),
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
        except RuntimeError:
            # RCLError/InvalidHandle from a transition attempted after an
            # earlier failure — both subclass RuntimeError; best-effort teardown.
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

    Publishes one OccupancyVoxels + one PromptStamped detection, waits for the memory tick,
    and asserts ``snapshot().detected_objects`` has a cup at z≈2 in the map frame.
    """
    with _object_lift_harness() as (executor, node, helper, vox_pub, det_pub):
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
        assert obj.existence is not None and obj.existence > 0.5  # believed present


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
# Scenarios 3-6 — evidence-based existence belief (issue #636)
#
# The memory timer is parked (cadence 0.1 Hz → one tick per 10 s) and the real
# ``_on_memory_tick`` is called directly (``_observe``), so each observation is counted
# exactly.
# ---------------------------------------------------------------------------

_PARKED_HZ = 0.1


def _labels(node: Any) -> list[str]:
    return [o.label for o in node._aggregator.snapshot().detected_objects]


def test_removed_object_is_evicted_after_a_few_clear_observations() -> None:
    """The object is taken away: its voxels clear and the wall behind shows through.

    Real producer contract: the detector publishes nothing when it sees nothing, so no
    message arrives — the depth source alone witnesses the miss. A once-seen object
    (belief +0.85) survives the first clear observations and is gone at the fourth
    (−0.85 each, absent at −2.0): never on a single miss.
    """
    with _object_lift_harness(cadence_hz=_PARKED_HZ) as (executor, node, helper, vox_pub, det_pub):
        _remember_cup(executor, node, helper, vox_pub, det_pub)

        kept = []
        for _ in range(4):
            _observe(executor, node, helper, vox_pub, _WALL_Z)
            kept.append(_labels(node))
        assert kept == [["cup"], ["cup"], ["cup"], []]


def test_the_same_observation_seen_again_is_not_new_evidence() -> None:
    """A 2 Hz tick over a 0.6 Hz camera: one clear frame is one miss, however often ticked."""
    with _object_lift_harness(cadence_hz=_PARKED_HZ) as (executor, node, helper, vox_pub, det_pub):
        _remember_cup(executor, node, helper, vox_pub, det_pub)
        _observe(executor, node, helper, vox_pub, _WALL_Z)
        once = node._aggregator.snapshot().detected_objects[0].existence
        for _ in range(20):
            node._on_memory_tick()
        assert node._aggregator.snapshot().detected_objects[0].existence == once


def test_occluded_object_is_kept() -> None:
    """Hidden behind the arm for many ticks: no evidence, so nothing changes.

    Two shapes of occlusion: the map still holds the object's voxels (octomap cannot
    clear what it cannot see), and a raw depth view in which only the occluder returns.
    """
    with _object_lift_harness(cadence_hz=_PARKED_HZ) as (executor, node, helper, vox_pub, det_pub):
        _remember_cup(executor, node, helper, vox_pub, det_pub)
        before = node._aggregator.snapshot().detected_objects[0].existence
        for layers in ((_OCCLUDER_Z, _CUBE_Z, _WALL_Z), (_OCCLUDER_Z,), ()):
            for _ in range(20):  # 10 s of observations at the default 2 Hz cadence
                if layers:
                    _observe(executor, node, helper, vox_pub, *layers)
                else:  # no depth source at all
                    node._latest_voxels = None
                    node._on_memory_tick()
            objects = node._aggregator.snapshot().detected_objects
            assert [o.label for o in objects] == ["cup"], f"evicted under layers={layers}"
            assert objects[0].existence == before


def test_an_occupied_miss_counts_only_when_the_detector_ran() -> None:
    """In view, still occupied, undetected: a miss only on a tick with a detector batch."""
    with _object_lift_harness(cadence_hz=_PARKED_HZ, miss_occupied_log_odds=-5.0) as (
        executor,
        node,
        helper,
        vox_pub,
        det_pub,
    ):
        _remember_cup(executor, node, helper, vox_pub, det_pub)
        for _ in range(10):  # the detector published nothing: not a miss
            _observe(executor, node, helper, vox_pub, _CUBE_Z)
        assert _labels(node) == ["cup"]

        # The detector runs and reports something else there: the cup went undetected.
        det_pub.publish(_make_detection_msg(helper, label="bowl"))
        assert _spin_until(executor, lambda: bool(node._candidate_buffer))
        _observe(executor, node, helper, vox_pub, _CUBE_Z)
        assert _labels(node) == ["bowl"]


def test_restart_restores_objects_as_uncertain_then_reconciles(tmp_path: Any) -> None:
    """Opt-in persistence: restored objects are uncertain until seen again or seen gone."""
    from openral_core.schemas import SceneGraph

    path = tmp_path / "object_memory.json"
    with _object_lift_harness(cadence_hz=_PARKED_HZ, memory_path=str(path)) as h:
        _remember_cup(*h)
    saved = SceneGraph.model_validate_json(path.read_text())
    assert [(n.label, n.pose.frame_id) for n in saved.nodes] == [("cup", "map")]

    # Restart 1 — restored uncertain with nothing observed; a fresh detection confirms it.
    with _object_lift_harness(cadence_hz=_PARKED_HZ, memory_path=str(path)) as h:
        executor, node, helper, vox_pub, det_pub = h
        # Published from configure on, before any memory tick: a consumer never
        # reads "nothing remembered" from a node that has just restored its memory.
        restored = node._aggregator.snapshot().detected_objects
        assert [o.label for o in restored] == ["cup"]
        node._on_memory_tick()
        restored = node._aggregator.snapshot().detected_objects
        assert [o.label for o in restored] == ["cup"]
        assert restored[0].existence == 0.5
        assert restored[0].pose.xyz == saved.nodes[0].pose.xyz
        assert f"obj_track_{restored[0].track_id}" == saved.nodes[0].node_id  # id kept
        _remember_cup(*h)
        confirmed = node._aggregator.snapshot().detected_objects
        assert len(confirmed) == 1 and confirmed[0].existence > 0.5

    # Restart 2 — the place is seen clear: the restored object is removed, and forgotten.
    with _object_lift_harness(cadence_hz=_PARKED_HZ, memory_path=str(path)) as h:
        executor, node, helper, vox_pub, det_pub = h
        # A camera is known only once its detector has spoken; the cup is not what it sees.
        _publish_voxels(executor, node, vox_pub, _make_voxels_msg(helper, _WALL_Z))
        det_pub.publish(_make_detection_msg(helper, label="wall"))
        assert _spin_until(executor, lambda: "head_rgb" in node._seen_cameras)
        kept = []
        for _ in range(3):
            _observe(executor, node, helper, vox_pub, _WALL_Z)
            kept.append("cup" in _labels(node))
        assert kept == [True, True, False]
    assert "cup" not in [n.label for n in SceneGraph.model_validate_json(path.read_text()).nodes]


def test_an_unreadable_memory_file_is_left_alone(tmp_path: Any) -> None:
    """A file that does not validate is neither restored nor saved over."""
    path = tmp_path / "object_memory.json"
    path.write_text('{"schema_version": "9.9", "nodes": []}')
    with _object_lift_harness(cadence_hz=_PARKED_HZ, memory_path=str(path)) as h:
        _remember_cup(*h)
    assert path.read_text() == '{"schema_version": "9.9", "nodes": []}'
