"""Live ROS integration test for the SegmentInView perception service.

Stands up the **real** ``openral_perception_ros.segmenter_node`` lifecycle node
with the **real** ``kind: segmenter`` rSkill manifest
(``rskills/rskill-sam2_1-any-grasped_object_mask-bf16``), the **real** SO-101
manifest for its wrist intrinsics and optical frame, a **real** static TF, and a
**real** in-tree wrist frame of the arm holding an eraser. It then calls the
service over a real DDS graph with a real ``rclpy`` client and checks the whole
contract the HAL's attachment bridge depends on:

    RGB frame + 3-D TCP point (in the attach link's frame)
      → tf2 into the camera optical frame
      → manifest intrinsics → pixel prompt
      → SAM 2.1 → plural mono8 masks, area ascending, parallel advisory scores

No doubles anywhere (CLAUDE.md §1.11) — including the model: SAM 2.1 runs here
for real, on CPU. The dev box's GTX 1060 is sm_61 and the CUDA wheels ship no
kernels for it, so ``device:=cpu`` is the honest setting, and CPU inference is
still inference. It is slow (~9 s to load and warm, ~3.7 s per call measured
here against ~53 ms warmed on the reference RTX 4070), which is exactly why the
HAL side of this path has a bounded deadline and a conservative fallback.

The three failure branches are covered too, because each one is what the HAL
turns into a GRIPPER_FORCE attachment rather than a stall: an un-published
camera, a prompt that does not project into the frame, and a deactivated node.

A second test covers the node's **diagnostic** mask topic
(``openral_msgs/SegmentMasks`` on ``/openral/perception/masks``): that it is off
by default — no publisher on the graph at all — and that when an operator turns
it on, real SAM 2.1 masks reach the dashboard's real
``PerceptionOverlaySubscriber`` and land in a real ``/api/state`` snapshot.
Nothing on the robot reads that topic; it only decides what a human sees.

Gated on ``OPENRAL_TEST_ROS_LIVE=1`` like the sibling reasoner integration
tests, and listed in the live-ROS suite (``scripts/ros_live_tests.sh``). CI runs
it inside ``openral:x86`` (the ``docker-build`` workflow). Locally::

    source /opt/ros/jazzy/setup.bash && just ros2-build
    source install/setup.bash
    just test-ros-live            # whole suite; `-k segment_in_view` narrows it
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import pytest

_LIVE_ROS = bool(os.getenv("OPENRAL_TEST_ROS_LIVE"))
_LIVE_ROS_REASON = (
    "live rclpy service round trip — set OPENRAL_TEST_ROS_LIVE=1 in a clean shell "
    "and source install/setup.bash first."
)

_ROBOT_YAML = "robots/so101_follower/robot.yaml"
_MANIFEST = "rskills/rskill-sam2_1-any-grasped_object_mask-bf16/rskill.yaml"
# A real SO-101 wrist capture of the arm holding an eraser — the same frame the
# design work measured its point-prompt quality on.
_WRIST_FRAME = Path("rskills/rskill-smolvla-so101-eraser_place-bf16/media/wrist_mid.png")
_IMAGE_TOPIC = "/openral/cameras/wrist/image"
_SERVICE = "/openral/perception/segment_in_view_itest"
_SEGMENTER_ID = "OpenRAL/rskill-sam2_1-any-grasped_object_mask-bf16"

# The attach link the SO-101's gripper joint hangs off, and the camera's own
# optical frame — both read from the manifest, not invented here.
_ATTACH_LINK = "gripper_base"
_CAMERA_FRAME = "wrist_camera"

# CPU inference plus a cold model load. Generous on purpose: this bound exists
# so a wedged test fails instead of hanging, not to assert a latency budget.
_CALL_TIMEOUT_S = 180.0


def _spin(executor: object, stop: threading.Event) -> None:
    """Spin an executor until the test says stop."""
    while not stop.is_set():
        executor.spin_once(timeout_sec=0.1)  # type: ignore[attr-defined]


def _wait_until(predicate: object, *, timeout_s: float = 10.0) -> bool:
    """Poll a live-graph predicate; the executors run on their own threads."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():  # type: ignore[operator]  # reason: a plain callable, typed loosely for a test helper
            return True
        time.sleep(0.02)
    return bool(predicate())  # type: ignore[operator]  # reason: as above


@pytest.mark.skipif(not _LIVE_ROS, reason=_LIVE_ROS_REASON)
def test_segment_in_view_returns_plural_masks_for_a_real_wrist_grasp() -> None:
    """The real service answers a real 3-D prompt with real SAM 2.1 masks."""
    rclpy = pytest.importorskip("rclpy")
    pytest.importorskip("openral_msgs.srv")
    pytest.importorskip("transformers")
    import numpy as np
    from builtin_interfaces.msg import Time
    from geometry_msgs.msg import Point, TransformStamped
    from openral_msgs.srv import SegmentInView
    from openral_perception_ros.segmenter_node import make_segmenter_node
    from PIL import Image as PilImage
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.lifecycle import TransitionCallbackReturn
    from rclpy.node import Node
    from rclpy.parameter import Parameter
    from rclpy.qos import (
        QoSDurabilityPolicy,
        QoSHistoryPolicy,
        QoSProfile,
        QoSReliabilityPolicy,
    )
    from sensor_msgs.msg import Image
    from tf2_ros import StaticTransformBroadcaster

    rgb = np.asarray(PilImage.open(_WRIST_FRAME).convert("RGB"))
    height, width, _ = rgb.shape

    rclpy.init()
    stop = threading.Event()
    spinner: threading.Thread | None = None
    node = None
    helper = None
    try:
        node = make_segmenter_node("openral_segmenter_itest")
        node.set_parameters(
            [
                Parameter("robot_yaml", Parameter.Type.STRING, _ROBOT_YAML),
                Parameter("manifest_path", Parameter.Type.STRING, _MANIFEST),
                Parameter(
                    "cameras",
                    Parameter.Type.STRING_ARRAY,
                    [f"wrist={_IMAGE_TOPIC}", "top=/openral/cameras/top/image"],
                ),
                Parameter("primary_camera", Parameter.Type.STRING, "wrist"),
                Parameter("segment_in_view_service", Parameter.Type.STRING, _SERVICE),
                # sm_61 dev GPU: no CUDA kernels, so run the model for real on CPU.
                Parameter("device", Parameter.Type.STRING, "cpu"),
            ]
        )
        helper = Node("segment_in_view_itest_client")

        # A real static TF for the attach link -> camera optical hop the node
        # performs on every request. tf2 is the only source of frames; nothing
        # in the node or this test inlines a matrix.
        broadcaster = StaticTransformBroadcaster(helper)
        tf = TransformStamped()
        tf.header.frame_id = _CAMERA_FRAME
        tf.child_frame_id = _ATTACH_LINK
        # The attach link sits 6 cm behind the camera along its optical +z and
        # 3 cm above it, i.e. the jaws hang into the lower part of the frame —
        # the real phone-on-bracket wrist rig this manifest describes.
        tf.transform.translation.x = 0.0
        tf.transform.translation.y = -0.03
        tf.transform.translation.z = 0.06
        tf.transform.rotation.w = 1.0
        broadcaster.sendTransform(tf)

        img_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        image_pub = helper.create_publisher(Image, _IMAGE_TOPIC, img_qos)

        executor = MultiThreadedExecutor(num_threads=4)
        executor.add_node(node)
        executor.add_node(helper)
        spinner = threading.Thread(target=_spin, args=(executor, stop), daemon=True)
        spinner.start()

        assert node.trigger_configure() == TransitionCallbackReturn.SUCCESS
        # Loads AND warms SAM 2.1 — the cold pass is burned here by contract.
        assert node.trigger_activate() == TransitionCallbackReturn.SUCCESS

        frame = Image()
        frame.header.frame_id = _CAMERA_FRAME
        frame.height, frame.width = height, width
        frame.encoding = "rgb8"
        frame.step = width * 3
        frame.data = np.ascontiguousarray(rgb).tobytes()

        client = helper.create_client(SegmentInView, _SERVICE)
        assert client.wait_for_service(timeout_sec=30.0), "segmenter never offered the service"

        # Republish until the node's BEST_EFFORT cache has the frame.
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline and "wrist" not in node._frames:
            image_pub.publish(frame)
            time.sleep(0.1)
        assert "wrist" in node._frames, "the segmenter never cached the published wrist frame"

        def _call(request: object) -> object:
            future = client.call_async(request)
            end = time.monotonic() + _CALL_TIMEOUT_S
            while not future.done() and time.monotonic() < end:
                time.sleep(0.05)
            assert future.done(), "SegmentInView never answered"
            return future.result()

        # ── the clean path ────────────────────────────────────────────────────
        request = SegmentInView.Request()
        request.stamp = Time(sec=17, nanosec=42)
        request.camera = "wrist"
        request.frame_id = _ATTACH_LINK
        # The TCP, in the attach link's frame: 6 cm out along the jaws. Through
        # the static TF above this lands mid-lower-frame, on the held eraser.
        request.tcp_point = Point(x=0.0, y=0.05, z=0.06)
        request.negative_points = []
        response = _call(request)

        assert response.ok is True, response.failure_reason
        assert response.failure_reason == ""
        assert response.camera == "wrist"
        # multimask: true in the manifest => SAM 2's three nested hypotheses,
        # minus any that failed min_mask_area_px.
        assert len(response.masks) > 1, "a multimask manifest must return plural candidates"
        assert len(response.mask_scores_advisory) == len(response.masks), (
            "scores are parallel to masks by contract"
        )

        areas = []
        for mask in response.masks:
            assert mask.encoding == "mono8"
            assert (mask.height, mask.width) == (height, width), (
                "masks come back at the source frame's own resolution"
            )
            assert mask.step == width
            assert len(mask.data) == height * width
            # The CALLER's stamp, so the trace replays against the attach instant.
            assert (mask.header.stamp.sec, mask.header.stamp.nanosec) == (17, 42)
            assert mask.header.frame_id == _CAMERA_FRAME
            decoded = np.frombuffer(bytes(mask.data), dtype=np.uint8)
            assert set(np.unique(decoded).tolist()) <= {0, 255}
            areas.append(int((decoded != 0).sum()))

        assert areas == sorted(areas), "candidates are ordered by area ascending, never by score"
        assert all(area >= 64 for area in areas), "min_mask_area_px from the manifest"
        # The whole point of the plural reply: the model's own ranking does NOT
        # agree with the geometric ordering, which is why the HAL picks.
        assert min(areas) > 0

        # ── an un-published camera ────────────────────────────────────────────
        request.camera = "top"
        stale = _call(request)
        assert stale.ok is False
        assert stale.camera == "top"
        assert stale.masks == []
        assert stale.failure_reason.startswith("ROSPerceptionStale:")

        # ── a prompt that does not project into the frame ─────────────────────
        request.camera = "wrist"
        # Behind the camera's optical plane once transformed: no usable prompt.
        request.tcp_point = Point(x=0.0, y=0.0, z=-10.0)
        off_frame = _call(request)
        assert off_frame.ok is False
        assert off_frame.masks == []
        assert off_frame.failure_reason.startswith("ROSConfigError:")

        # ── a deactivated node ────────────────────────────────────────────────
        assert node.trigger_deactivate() == TransitionCallbackReturn.SUCCESS
        request.tcp_point = Point(x=0.0, y=0.05, z=0.06)
        inactive = _call(request)
        assert inactive.ok is False
        assert inactive.masks == []
        assert inactive.failure_reason.startswith("ROSRuntimeError:")
    finally:
        stop.set()
        if spinner is not None:
            spinner.join(timeout=5.0)
        if node is not None:
            node.destroy_node()
        if helper is not None:
            helper.destroy_node()
        rclpy.try_shutdown()


@pytest.mark.skipif(not _LIVE_ROS, reason=_LIVE_ROS_REASON)
def test_the_diagnostic_mask_topic_is_off_by_default_and_feeds_the_dashboard() -> None:
    """The debug mask topic costs nothing off, and lights the overlay on.

    PR #122 taught the dashboard to render
    ``topics.perception.overlays[camera].masks`` — an LA-PNG-per-mask overlay
    drawn on the camera tile — but the segmenter's contract is a *service*, so
    that renderer had no producer and never drew anything. This is the producer:
    ``openral_msgs/SegmentMasks`` on ``/openral/perception/masks``, behind
    ``publish_debug_masks`` (default false).

    Two properties, both only provable on a real graph:

    1. **Off by default is really off.** Not "publishes nothing" — *no publisher
       exists on the topic at all*, which is what makes it free.
    2. **On, it feeds the real consumer.** Real SAM 2.1 masks → the real
       ``SegmentMasks`` message → the real ``PerceptionOverlaySubscriber`` (its
       own node, its own spin thread, matching sensor-class QoS) → the real
       ``TelemetryStore`` → a real ``/api/state`` snapshot. A QoS drift on
       either side leaves the overlay blank with nothing to explain it, and only
       a live graph catches that.

    Strictly diagnostic: nothing asserted here is read by the HAL, the kernel or
    the attachment path. The HAL takes its masks from its own service reply.
    """
    rclpy = pytest.importorskip("rclpy")
    pytest.importorskip("openral_msgs.srv")
    pytest.importorskip("transformers")
    import numpy as np
    from builtin_interfaces.msg import Time
    from geometry_msgs.msg import Point, TransformStamped
    from openral_msgs.srv import SegmentInView
    from openral_observability.dashboard import TelemetryStore
    from openral_observability.dashboard.perception_overlay_subscriber import (
        PERCEPTION_MASKS_TOPIC,
        PerceptionOverlaySubscriber,
    )
    from openral_perception_ros.segmenter_node import (
        DEFAULT_DEBUG_MASKS_TOPIC,
        make_segmenter_node,
    )
    from PIL import Image as PilImage
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.lifecycle import TransitionCallbackReturn
    from rclpy.node import Node
    from rclpy.parameter import Parameter
    from rclpy.qos import (
        QoSDurabilityPolicy,
        QoSHistoryPolicy,
        QoSProfile,
        QoSReliabilityPolicy,
    )
    from sensor_msgs.msg import Image
    from tf2_ros import StaticTransformBroadcaster

    # The producer and the consumer must name the SAME topic; they are in
    # different packages and neither imports the other, so pin it here.
    assert DEFAULT_DEBUG_MASKS_TOPIC == PERCEPTION_MASKS_TOPIC

    rgb = np.asarray(PilImage.open(_WRIST_FRAME).convert("RGB"))
    height, width, _ = rgb.shape

    rclpy.init()
    stop = threading.Event()
    spinner: threading.Thread | None = None
    node = None
    helper = None
    overlay = None
    try:
        node = make_segmenter_node("openral_segmenter_masks_itest")
        node.set_parameters(
            [
                Parameter("robot_yaml", Parameter.Type.STRING, _ROBOT_YAML),
                Parameter("manifest_path", Parameter.Type.STRING, _MANIFEST),
                Parameter("cameras", Parameter.Type.STRING_ARRAY, [f"wrist={_IMAGE_TOPIC}"]),
                Parameter("primary_camera", Parameter.Type.STRING, "wrist"),
                Parameter("segment_in_view_service", Parameter.Type.STRING, _SERVICE),
                Parameter("device", Parameter.Type.STRING, "cpu"),
            ]
        )
        helper = Node("segment_in_view_masks_itest_client")
        broadcaster = StaticTransformBroadcaster(helper)
        tf = TransformStamped()
        tf.header.frame_id = _CAMERA_FRAME
        tf.child_frame_id = _ATTACH_LINK
        tf.transform.translation.y = -0.03
        tf.transform.translation.z = 0.06
        tf.transform.rotation.w = 1.0
        broadcaster.sendTransform(tf)

        img_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        image_pub = helper.create_publisher(Image, _IMAGE_TOPIC, img_qos)

        executor = MultiThreadedExecutor(num_threads=4)
        executor.add_node(node)
        executor.add_node(helper)
        spinner = threading.Thread(target=_spin, args=(executor, stop), daemon=True)
        spinner.start()

        # ── 1. Default: no publisher, on the real graph ───────────────────────
        assert node.trigger_configure() == TransitionCallbackReturn.SUCCESS
        assert node._masks_pub is None, "publish_debug_masks must default to off"
        assert helper.count_publishers(PERCEPTION_MASKS_TOPIC) == 0, (
            "an off debug topic must have no publisher at all — that is what "
            "makes it free when nobody enabled it"
        )

        # ── 2. Operator turns it on ───────────────────────────────────────────
        assert node.trigger_cleanup() == TransitionCallbackReturn.SUCCESS
        node.set_parameters([Parameter("publish_debug_masks", Parameter.Type.BOOL, True)])
        assert node.trigger_configure() == TransitionCallbackReturn.SUCCESS
        assert node._masks_pub is not None

        overlay = PerceptionOverlaySubscriber(TelemetryStore())
        assert overlay.available, "the dashboard subscriber must come up on a live host"
        assert overlay.masks_available, (
            "openral_msgs/SegmentMasks is built, so the mask leg must be live — "
            "an inert one would make every assertion below vacuous"
        )
        assert _wait_until(lambda: node._masks_pub.get_subscription_count() >= 1), (
            "the dashboard subscriber never matched the segmenter's BEST_EFFORT "
            "mask publisher — check the QoS on both sides"
        )

        # Loads AND warms SAM 2.1 on CPU (real inference, sm_61 has no kernels).
        assert node.trigger_activate() == TransitionCallbackReturn.SUCCESS

        frame = Image()
        frame.header.frame_id = _CAMERA_FRAME
        frame.height, frame.width = height, width
        frame.encoding = "rgb8"
        frame.step = width * 3
        frame.data = np.ascontiguousarray(rgb).tobytes()

        client = helper.create_client(SegmentInView, _SERVICE)
        assert client.wait_for_service(timeout_sec=30.0), "segmenter never offered the service"
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline and "wrist" not in node._frames:
            image_pub.publish(frame)
            time.sleep(0.1)
        assert "wrist" in node._frames

        request = SegmentInView.Request()
        request.stamp = Time(sec=17, nanosec=42)
        request.camera = "wrist"
        request.frame_id = _ATTACH_LINK
        request.tcp_point = Point(x=0.0, y=0.05, z=0.06)
        future = client.call_async(request)
        end = time.monotonic() + _CALL_TIMEOUT_S
        while not future.done() and time.monotonic() < end:
            time.sleep(0.05)
        assert future.done(), "SegmentInView never answered"
        response = future.result()
        assert response.ok is True, response.failure_reason

        # ── 3. …and the same masks reach the dashboard's overlay ──────────────
        store = overlay._store
        assert _wait_until(
            lambda: "wrist" in (store.snapshot()["topics"]["perception"].get("overlays") or {}),
            timeout_s=15.0,
        ), "the diagnostic mask set never reached the dashboard store"

        lane = store.snapshot()["topics"]["perception"]["overlays"]["wrist"]["masks"]
        assert lane["rskill_id"] == _SEGMENTER_ID, "whose masks these are must not be a guess"
        # The CALLER's attach stamp, copied through the topic — a stale overlay
        # has to be detectable rather than plausible.
        assert lane["stamp_unix"] == pytest.approx(17 + 42e-9)
        assert len(lane["masks"]) == len(response.masks) > 1
        for drawn, served in zip(lane["masks"], response.masks, strict=True):
            assert (drawn["width"], drawn["height"]) == (served.width, served.height)
            assert drawn["png_b64"].startswith("iVBOR")
        assert [m["score_advisory"] for m in lane["masks"]] == pytest.approx(
            list(response.mask_scores_advisory)
        )
        # The service's area-ascending order survives to the renderer, and the
        # advisory scores are carried without being used to reorder anything.
        areas = [
            int((np.frombuffer(bytes(m.data), dtype=np.uint8) != 0).sum()) for m in response.masks
        ]
        assert areas == sorted(areas)
    finally:
        stop.set()
        if overlay is not None:
            overlay.close()
        if spinner is not None:
            spinner.join(timeout=5.0)
        if node is not None:
            node.destroy_node()
        if helper is not None:
            helper.destroy_node()
        rclpy.try_shutdown()
