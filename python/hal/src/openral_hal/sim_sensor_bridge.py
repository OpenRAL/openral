# python/hal/src/openral_hal/sim_sensor_bridge.py
"""Shared sim-sensor + viewer bridge for scene-attached HAL lifecycle nodes (ADR-0034).

Republishes whatever a ``SimAttachedHAL`` exposes — RGB camera frames
(``read_images``) on ``/openral/cameras/<n>/image`` and an optional live
``mujoco.viewer`` — for any manifest-driven node, so franka/ur5e/... reach
deploy-sim scene+camera+viewer parity without per-package wiring. Phase 2
adds ``/scan`` + depth ``PointCloud2``. rclpy is imported lazily so the
module stays import-safe in pure-Python CI.
"""

from __future__ import annotations

import contextlib
import time
from typing import TYPE_CHECKING, Any

# Throttle dashboard thumbnail emission to ~1 Hz per camera (1e9 ns).
# The live ROS topic stays at the higher camera_rate_hz; only the OTel
# ``sensors.read_latest`` span is rate-limited to avoid ballooning OTLP
# payload with redundant thumbnails (the dashboard polls at ~1 Hz anyway).
_THUMB_INTERVAL_NS = 1_000_000_000
_IMAGE_DIM = 3  # HWC ndarray
_RGB_CHANNELS = 3

if TYPE_CHECKING:
    from openral_core import RobotDescription

__all__ = ["SimSensorBridge", "constant_scan_no_hit_ranges", "should_idle_step"]


def constant_scan_no_hit_ranges(*, n_beams: int, max_range_m: float) -> list[float]:
    """Return an ``n_beams``-long list with every beam clamped to ``max_range_m``.

    The synthetic ``/scan`` published when no live MuJoCo handle is bound (the
    in-process digital twin has no scene to ray-cast). slam_toolbox and Nav2
    both treat per-beam ``max_range`` as "no hit" rather than ``inf`` / ``NaN``,
    so this is the honest "nothing in front of me" reading. Pure (no rclpy) so
    it is unit-testable in isolation.

    Example:
        >>> constant_scan_no_hit_ranges(n_beams=3, max_range_m=12.0)
        [12.0, 12.0, 12.0]
    """
    return [float(max_range_m)] * int(n_beams)


def should_idle_step(now_ns: int, last_action_ns: int, idle_hold_ns: int) -> bool:
    """Return True iff the sim-only idle stepper should advance the env now.

    The idle stepper yields to active skills: it steps the env with a zero/HOLD
    action ONLY when no real action has arrived within the idle-hold window. So
    it returns True iff ``now_ns - last_action_ns >= idle_hold_ns`` — i.e. the
    last real actuation is at least ``idle_hold_ns`` old (or there has been
    none, ``last_action_ns == 0``).

    Pure (no rclpy / no I/O) so it is unit-testable in isolation. The
    single-threaded rclpy executor guarantees the idle timer and
    ``_on_safe_action`` never run concurrently, so this timestamp comparison
    alone is a sufficient hand-off — no lock is needed.

    Args:
        now_ns: Current monotonic clock in nanoseconds (``time.monotonic_ns()``).
        last_action_ns: Monotonic ns of the last real action through the HAL's
            ``send_action`` (``SimAttachedHAL.last_action_ns``); ``0`` if none.
        idle_hold_ns: Quiet window in ns. A real action within this window of
            ``now_ns`` suppresses the idle tick.

    Returns:
        ``True`` to idle-step now, ``False`` to yield to a recent action.

    Example:
        >>> should_idle_step(now_ns=1_000_000_000, last_action_ns=0, idle_hold_ns=200_000_000)
        True
        >>> should_idle_step(
        ...     now_ns=1_000_000_000, last_action_ns=950_000_000, idle_hold_ns=200_000_000
        ... )
        False
    """
    return now_ns - last_action_ns >= idle_hold_ns


def _obs_key_for_sensor(sensor: Any) -> str:
    """Key into ``read_images()`` for a manifest RGB sensor.

    Scenes key rendered frames by the VLA camera slot (``camera1``,
    ``camera2``, ...): LIBERO emits only those; robocasa emits them as
    aliases alongside the real camera name. So resolve the obs key from the
    sensor's ``vla_feature_key`` suffix (``observation.images.camera1`` ->
    ``camera1``), falling back to the sensor name (robocasa real-name keys, or
    sensors with no ``vla_feature_key``). The published topic stays
    ``/openral/cameras/<sensor.name>/image`` regardless.
    """
    vfk = getattr(sensor, "vla_feature_key", None)
    if vfk:
        return str(vfk).rsplit(".", 1)[-1]
    return str(sensor.name)


def _frame_for_camera(images: dict[str, Any], obs_key: str, name: str) -> Any:
    """Resolve a camera's frame from a ``read_images()`` dict, or ``None``.

    The two sim HALs key their frame dicts by different conventions:
    :class:`~openral_hal.sim_attached.SimAttachedHAL` (scene-attached LIBERO /
    robocasa) keys by the VLA slot (``obs_key`` — ``camera1`` / ``camera2``),
    while :class:`~openral_hal._mujoco_arm.MujocoArmHAL` (bare or composed
    digital twin) keys by the sensor ``name``. Try the slot first, then fall
    back to the name so both conventions resolve.

    Without the fallback, so101's slot ``camera1`` never matched its
    MujocoArmHAL frame keyed ``front`` and no frame ever published (issue #88);
    openarm was unaffected only because its sensor names equal their VLA slots.
    """
    arr = images.get(obs_key)
    if arr is None and name != obs_key:
        arr = images.get(name)
    return arr


def _optical_frame_rgb_cameras(sensors: Any) -> list[Any]:
    """RGB camera specs that own a dedicated ``*_optical_frame`` (ADR-0052).

    These are the cameras :meth:`SimSensorBridge._publish_camera_optical_tfs`
    broadcasts a live ``base_frame -> <camera>_optical_frame`` TF for, so the
    world-state object-lift can project the world voxel map into them. A camera
    whose ``frame_id`` is a robot link (e.g. an eye-in-hand at ``panda_hand``)
    already has TF from ``robot_state_publisher`` and is excluded. Pure (no
    rclpy / MuJoCo) so it is unit-testable in isolation.
    """
    return [
        s
        for s in sensors
        if getattr(s, "modality", None) == "rgb"
        and str(getattr(s, "frame_id", "")).endswith("_optical_frame")
    ]


class SimSensorBridge:
    """Wire + tear down sim-sensor publishers and the viewer on a lifecycle node.

    Args:
        node: the HAL ``LifecycleNode`` (provides ``create_publisher`` /
            ``create_timer`` / ``get_logger``).
        hal: the connected HAL; scene streams activate only when it exposes
            ``read_images`` / ``mujoco_handles``.
        description: host manifest — gates streams on declared sensors.
        viewer_enabled: open ``mujoco.viewer.launch_passive`` (graceful headless).
        camera_rate_hz / viewer_sync_rate_hz: timer rates.
        scan_rate_hz: LaserScan publish rate (Hz). Gated on manifest lidar_2d.
        scan_n_beams: Number of ray-cast beams per scan cycle.
        scan_max_range_m: Sensor max range; "no hit" beams clamp to this.
        scan_min_range_m: Sensor min range (near-field filter).
    """

    def __init__(
        self,
        node: Any,
        hal: Any,
        description: RobotDescription,
        *,
        viewer_enabled: bool = True,
        camera_rate_hz: float = 10.0,
        viewer_sync_rate_hz: float = 30.0,
        scan_rate_hz: float = 10.0,
        scan_n_beams: int = 360,
        scan_max_range_m: float = 12.0,
        scan_min_range_m: float = 0.05,
        depth_rate_hz: float = 10.0,
        depth_max_range_m: float = 5.0,
        depth_pixel_stride: int = 4,
        idle_hold_ms: float = 200.0,
        on_step: Any = None,
    ) -> None:
        """Bind the node + HAL + manifest; opens no publishers until :meth:`setup`.

        ``idle_hold_ms`` is the sim-only idle stepper's quiet window: it
        advances the env with a zero/HOLD action only when no real action has
        arrived within this window (so an active skill always wins). Default
        200 ms — long enough to never race a 30-200 Hz S1 skill stream.

        ``on_step`` (ADR-0049): an optional zero-arg callback invoked after each
        successful ``idle_step`` — the node uses it to refresh the proprio
        snapshot so odom / joint_state stay fresh while the scene idles. It runs
        in this bridge's (default / "sim") callback group, so reading the
        simulator inside it is safe.
        """
        self._node = node
        self._on_step = on_step
        self._hal = hal
        self._description = description
        self._viewer_enabled = viewer_enabled
        self._camera_rate_hz = camera_rate_hz
        self._idle_hold_ns = int(max(idle_hold_ms, 0.0) * 1_000_000)
        self._viewer_sync_rate_hz = viewer_sync_rate_hz
        self._scan_rate_hz = scan_rate_hz
        self._scan_n_beams = scan_n_beams
        self._scan_max_range_m = scan_max_range_m
        self._scan_min_range_m = scan_min_range_m
        self._depth_rate_hz = depth_rate_hz
        self._depth_max_range_m = depth_max_range_m
        self._depth_pixel_stride = depth_pixel_stride
        self._image_pubs: dict[str, Any] = {}
        self._image_obs_key: dict[str, str] = {}
        # Per-camera last thumbnail emit timestamp (ns). Throttles the OTel
        # ``sensors.read_latest`` span to ~1 Hz while the ROS topic publishes
        # at the full camera_rate_hz.
        self._last_thumb_ns: dict[str, int] = {}
        self._image_missing_warned: set[str] = set()
        self._image_timer: Any = None
        self._camera_tf_timer: Any = None
        # ADR-0034 (2026-06-04 idle-stepper amendment) — sim-only free-running
        # stepper timer. Created in setup ONLY when the HAL exposes ``idle_step``
        # AND has live MuJoCo handles (both sim gates); never against a real HAL.
        self._idle_timer: Any = None
        self._viewer: Any = None
        self._viewer_timer: Any = None
        self._scan_pub: Any = None
        self._scan_timer: Any = None
        # ADR-0030 — depth-camera → PointCloud2 publishers, one per depth SensorSpec.
        # Feeds octomap_server → safety kernel world-collision voxel check.
        # Gated on live MuJoCo handles; _depth_disabled prevents repeated warnings.
        self._depth_pubs: dict[str, Any] = {}
        # ADR-0064 — per depth camera, a dense 32FC1 depth image + CameraInfo
        # alongside the PointCloud2, so nvblox's projective depth integrator
        # (which rejects the sparse hit-only cloud) can build a `/map`.
        self._depth_image_pubs: dict[str, Any] = {}
        self._depth_info_pubs: dict[str, Any] = {}
        self._depth_timer: Any = None
        self._depth_disabled: set[str] = set()
        self._depth_base_body: str | None = None
        self._depth_base_body_id: int = -1
        # Robot's own MJCF body ids — dropped from the depth cloud so the
        # base-mounted camera doesn't voxelise the arm into its own world map.
        self._depth_self_bodies: frozenset[int] = frozenset()
        self._tf_broadcaster: Any = None
        # Static world->base_frame TF (ADR-0027 — gives a fixed-base sim arm the
        # world root its TF tree otherwise lacks, so task-space state layouts
        # like ``libero_eef8d`` can read the WORLD-frame EE pose the policy was
        # trained on). Published once from the base body's MuJoCo world pose;
        # skipped for mobile bases (they publish odom->base).
        self._static_tf_broadcaster: Any = None
        self._world_base_published: bool = False
        # ADR-0035 cross-frame lift — RGB cameras whose optical-frame TF failed
        # to resolve (no MJCF camera); warned once, then skipped.
        self._camera_tf_disabled: set[str] = set()
        # Offscreen "cinecam" recorder (website-video capture): when
        # OPENRAL_CINECAM_DIR is set, render the pulled-back free-camera view
        # (same pose as the onscreen viewer) to numbered JPGs each tick. Robust
        # vs the onscreen GLFW window, which the desktop WM can unmap.
        self._cinecam_renderer: Any = None
        self._cinecam_cam: Any = None
        self._cinecam_opt: Any = None
        self._cinecam_timer: Any = None
        self._cinecam_frame: int = 0
        self._cinecam_out_dir: str = ""
        self._cinecam_model: Any = None
        self._cinecam_w: int = 0
        self._cinecam_h: int = 0
        self._cinecam_base_body: Any = None
        self._cinecam_setup_az: float = 0.0
        self._cinecam_setup_el: float = 0.0
        self._cinecam_setup_dist: float = 1.0

    def setup(self) -> None:
        """Activate every stream the manifest + HAL support. Idempotent-safe per activate."""
        self._setup_cameras()
        self._setup_idle_stepper()
        self._setup_cinecam()
        self._setup_viewer()
        self._setup_scan()
        self._setup_depth()

    def teardown(self) -> None:
        """Cancel timers, destroy publishers, and close the viewer (idempotent)."""
        for t in (
            self._image_timer,
            self._camera_tf_timer,
            self._idle_timer,
            self._viewer_timer,
            self._cinecam_timer,
            self._scan_timer,
            self._depth_timer,
        ):
            if t is not None:
                t.cancel()
        self._image_timer = self._idle_timer = self._camera_tf_timer = None
        self._viewer_timer = self._scan_timer = self._depth_timer = None
        self._cinecam_timer = None
        if self._cinecam_renderer is not None:
            with contextlib.suppress(Exception):  # reason: renderer GL ctx may be gone
                self._cinecam_renderer.close()
            self._cinecam_renderer = None
        for pub in self._image_pubs.values():
            self._node.destroy_publisher(pub)
        self._image_pubs.clear()
        self._image_obs_key.clear()
        self._last_thumb_ns.clear()
        self._image_missing_warned.clear()
        if self._scan_pub is not None:
            self._node.destroy_publisher(self._scan_pub)
            self._scan_pub = None
        for pub in (
            *self._depth_pubs.values(),
            *self._depth_image_pubs.values(),
            *self._depth_info_pubs.values(),
        ):
            self._node.destroy_publisher(pub)
        self._depth_pubs.clear()
        self._depth_image_pubs.clear()
        self._depth_info_pubs.clear()
        self._depth_disabled.clear()
        self._depth_base_body = None
        self._depth_base_body_id = -1
        self._depth_self_bodies = frozenset()
        self._camera_tf_disabled.clear()
        self._tf_broadcaster = None
        self._static_tf_broadcaster = None
        self._world_base_published = False
        if self._viewer is not None:
            with contextlib.suppress(Exception):  # reason: viewer already closed
                self._viewer.close()
            self._viewer = None

    # -- RGB cameras --
    def _setup_cameras(self) -> None:
        if not hasattr(self._hal, "read_images"):
            return
        rgb = [s for s in self._description.sensors if s.modality == "rgb"]
        if not rgb:
            return
        from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
        from sensor_msgs.msg import Image as RosImage

        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1,
        )
        for s in rgb:
            self._image_pubs[s.name] = self._node.create_publisher(
                RosImage, f"/openral/cameras/{s.name}/image", qos
            )
            self._image_obs_key[s.name] = _obs_key_for_sensor(s)
        self._image_timer = self._node.create_timer(
            1.0 / max(self._camera_rate_hz, 1.0), self._publish_images
        )
        # ADR-0035 cross-frame lift — broadcast base_frame -> <camera>_optical_frame
        # for every RGB camera that owns a dedicated optical frame, from live
        # MuJoCo poses, so the world-state object-lift can project the world
        # voxel map into any detection camera (generic over robots/camera names).
        from tf2_ros import TransformBroadcaster

        if self._tf_broadcaster is None:
            self._tf_broadcaster = TransformBroadcaster(self._node)
        self._camera_tf_timer = self._node.create_timer(
            1.0 / max(self._camera_rate_hz, 1.0), self._publish_camera_optical_tfs
        )
        self._node.get_logger().info(
            f"SimSensorBridge: publishing {len(rgb)} camera(s): "
            + ", ".join(f"{s.name}<-{self._image_obs_key[s.name]}" for s in rgb)
        )

    def _publish_images(self) -> None:
        """Republish cached camera frames from the HAL as sensor_msgs/Image.

        Reads :meth:`SimAttachedHAL.read_images` (a dict of
        ``camera_name -> HWC uint8 NDArray``) and publishes each frame on
        ``/openral/cameras/<name>/image``.  The obs-key lookup (via
        ``_image_obs_key``) lets LIBERO-style scenes (keyed by VLA slot
        ``camera1`` / ``camera2``) coexist with robocasa real-name keys.

        Encoding handles mono8 / rgb8 / rgba8 arrays automatically. Frame
        data is copied bytewise — no compression hop.

        An OTel ``sensors.read_latest`` span (with JPEG thumbnail) is emitted
        at most once per second per camera so the dashboard Perception card
        updates without ballooning the OTLP payload.
        """
        reader = getattr(self._hal, "read_images", None)
        if reader is None or not self._image_pubs:
            return
        from sensor_msgs.msg import Image as RosImage

        images = reader()  # dict[str, ndarray HWC uint8]
        if not isinstance(images, dict) or not images:
            return
        stamp = self._node.get_clock().now().to_msg()
        from openral_observability.producer import (
            encode_rgb_thumbnail,
            record_sensor_frame_attrs,
        )
        from opentelemetry import trace

        tracer = trace.get_tracer(__name__)
        now_ns = time.monotonic_ns()
        for name, pub in self._image_pubs.items():
            obs_key = self._image_obs_key.get(name, name)
            arr = _frame_for_camera(images, obs_key, name)
            if arr is None:
                if name not in self._image_missing_warned:
                    self._image_missing_warned.add(name)
                    self._node.get_logger().warning(
                        f"SimSensorBridge: no frame for camera '{name}' "
                        f"(expected obs key '{obs_key}' or name '{name}'); "
                        f"available keys: {sorted(images.keys())}. "
                        "Check the scene's --robot override matches sensor layout."
                    )
                continue
            if arr.ndim != _IMAGE_DIM or arr.shape[2] not in (1, _RGB_CHANNELS, 4):
                continue
            h, w, c = arr.shape
            msg = RosImage()
            msg.header.stamp = stamp
            msg.header.frame_id = name
            msg.height = int(h)
            msg.width = int(w)
            msg.encoding = "mono8" if c == 1 else "rgb8" if c == _RGB_CHANNELS else "rgba8"
            msg.is_bigendian = 0
            msg.step = int(w * c)
            msg.data = bytes(arr.astype("uint8").tobytes())
            pub.publish(msg)
            # Emit a ``sensors.read_latest`` span at most once per second per
            # camera (dashboard polls at ~1 Hz; higher rate would balloon OTLP
            # payload with redundant thumbnails).
            last = self._last_thumb_ns.get(name, 0)
            if now_ns - last < _THUMB_INTERVAL_NS:
                continue
            self._last_thumb_ns[name] = now_ns
            thumb = encode_rgb_thumbnail(arr) if c == _RGB_CHANNELS else None
            with tracer.start_as_current_span("sensors.read_latest") as span:
                span.set_attribute("openral.sensors.source", name)
                record_sensor_frame_attrs(
                    span,
                    modality="rgb",
                    encoding=msg.encoding,
                    width=int(w),
                    height=int(h),
                    channels=int(c),
                    age_ms=0.0,
                    thumbnail_bytes=thumb,
                )

    def _publish_camera_optical_tfs(self) -> None:
        """Broadcast ``base_frame -> <camera>_optical_frame`` for every RGB camera.

        ADR-0035 cross-frame object-lift: the world-state lifter projects the
        world voxel map (built from the robot's body-mounted depth sensor) into
        each detection camera using that camera's extrinsics. This publishes
        those extrinsics live from MuJoCo poses — generic over any robot and any
        camera name (the MJCF camera is resolved from each ``SensorSpec``'s
        ``metadata.mjcf_camera``). Only cameras that own a dedicated
        ``*_optical_frame`` are broadcast: a camera whose ``frame_id`` is a robot
        link (e.g. an eye-in-hand at ``panda_hand``) already has TF from
        ``robot_state_publisher`` and must not be clobbered. No-op for non-MuJoCo
        backends or cameras whose MJCF name doesn't resolve (warned once).
        """
        if self._tf_broadcaster is None or not self._image_pubs:
            return
        handle = getattr(self._hal, "mujoco_handles", lambda: None)()
        if handle is None:
            return
        model, data = handle
        if self._depth_base_body is None and self._depth_base_body_id < 0:
            self._resolve_depth_base_body(model)
        if self._depth_base_body is None:
            return

        # ADR-0027 — publish the world root for a fixed-base sim arm (once).
        self._publish_world_base_tf(model, data)

        from geometry_msgs.msg import TransformStamped
        from openral_core.exceptions import ROSConfigError

        from openral_hal.depth_cloud import camera_optical_tf_to_base, mjcf_camera_name

        base_frame_id = getattr(self._description, "base_frame", "base_link")
        stamp = self._node.get_clock().now().to_msg()
        specs = {s.name: s for s in _optical_frame_rgb_cameras(self._description.sensors)}
        for name in self._image_pubs:
            spec = specs.get(name)
            if spec is None or name in self._camera_tf_disabled:
                continue
            try:
                xyz, quat = camera_optical_tf_to_base(
                    model=model,
                    data=data,
                    camera_name=mjcf_camera_name(spec),
                    base_body_name=self._depth_base_body,
                )
            except ROSConfigError as exc:
                self._camera_tf_disabled.add(name)
                self._node.get_logger().warning(
                    f"camera optical TF {name!r} disabled: {exc}; "
                    "check the SensorSpec's mjcf_camera metadata."
                )
                continue
            tf = TransformStamped()
            tf.header.stamp = stamp
            tf.header.frame_id = base_frame_id
            tf.child_frame_id = spec.frame_id
            tf.transform.translation.x = xyz[0]
            tf.transform.translation.y = xyz[1]
            tf.transform.translation.z = xyz[2]
            tf.transform.rotation.x = quat[0]
            tf.transform.rotation.y = quat[1]
            tf.transform.rotation.z = quat[2]
            tf.transform.rotation.w = quat[3]
            self._tf_broadcaster.sendTransform(tf)

    def _publish_world_base_tf(self, model: object, data: object) -> None:
        """Publish a static ``world -> base_frame`` TF from the base body's sim pose.

        A robosuite-attached fixed-base arm (LIBERO franka, ur5e, ...) roots its
        TF tree at ``base_frame`` (panda_link0) with NO parent, yet the robot
        sits at a non-origin world pose (LIBERO mounts the franka at world
        ``[-0.66, 0, 0.912]``, varying by suite). The benchmark feeds the policy
        the WORLD-frame EE pose (robosuite ``robot0_eef_pos``); without this
        transform the ``libero_eef8d`` task-space state layout could only read
        the base-relative EE pose (off by the mount — the policy would see the
        EE ~0.9 m below where it trained). Publishing the base body's live
        MuJoCo world pose as ``world -> base_frame`` makes
        ``tf_lookup("world", "panda_hand_tcp")`` equal robosuite's eef.

        Static (the base is fixed) + latched, so the skill_runner's tf_lookup
        gets it even joining late. Skipped for MOBILE bases — they publish a
        live ``odom -> base`` and a second parent for ``base`` would corrupt the
        tree (detected via ``capabilities.footprint_radius``, which only mobile
        robots declare).
        """
        if self._world_base_published:
            return
        caps = getattr(self._description, "capabilities", None)
        if caps is not None and getattr(caps, "footprint_radius", None) is not None:
            self._world_base_published = True  # mobile: odom owns base->world; nothing to do
            return
        if self._depth_base_body_id < 0:
            return

        from geometry_msgs.msg import TransformStamped
        from tf2_ros import StaticTransformBroadcaster

        bid = self._depth_base_body_id
        pos = data.xpos[bid]  # type: ignore[attr-defined]  # world position of the base body
        quat_wxyz = data.xquat[bid]  # type: ignore[attr-defined]  # MuJoCo quaternion is wxyz
        if self._static_tf_broadcaster is None:
            self._static_tf_broadcaster = StaticTransformBroadcaster(self._node)
        base_frame_id = getattr(self._description, "base_frame", "base_link")
        tf = TransformStamped()
        tf.header.stamp = self._node.get_clock().now().to_msg()
        tf.header.frame_id = "world"
        tf.child_frame_id = base_frame_id
        tf.transform.translation.x = float(pos[0])
        tf.transform.translation.y = float(pos[1])
        tf.transform.translation.z = float(pos[2])
        tf.transform.rotation.w = float(quat_wxyz[0])
        tf.transform.rotation.x = float(quat_wxyz[1])
        tf.transform.rotation.y = float(quat_wxyz[2])
        tf.transform.rotation.z = float(quat_wxyz[3])
        self._static_tf_broadcaster.sendTransform(tf)
        self._world_base_published = True
        self._node.get_logger().info(
            f"published static world->{base_frame_id} at "
            f"[{float(pos[0]):.3f}, {float(pos[1]):.3f}, {float(pos[2]):.3f}] "
            "(fixed-base sim world root, ADR-0027)"
        )

    # -- Sim-only free-running idle stepper (ADR-0034 amendment) --
    def _setup_idle_stepper(self) -> None:
        """Create the sim-only idle-step timer, gated on a callable ``idle_step``.

        Gate (the PRIMARY safety gate): the HAL exposes a callable ``idle_step``,
        defined ONLY on :class:`~openral_hal.sim_attached.SimAttachedHAL`. A real
        HAL never defines it, so the timer is never created against real
        hardware. This is the real guarantee, not "zero is harmless" (a zero
        vector is a HOLD in sim but "drive to 0 rad" — violent — on a real
        absolute-position arm).

        No MuJoCo-handle gate (dropped in the ADR-0034 amendment): idle-stepping
        is valid for any wrapped SimRollout, so a non-MuJoCo backend (Isaac Sim
        sidecar, ManiSkill3) keeps its cameras live when idle too. ``idle_step``
        itself returns ``False`` for a non-sim HAL, and the
        catch-once-and-disable guard below contains any per-tick fault.

        The timer runs at ``camera_rate_hz`` so step-then-publish stays matched
        (the existing camera timer republishes the freshened ``_last_obs``); no
        separate rate param is introduced. The single-threaded rclpy executor
        ensures the idle callback and ``_on_safe_action`` never run
        concurrently, so :func:`should_idle_step`'s timestamp check is a
        sufficient hand-off (no lock).
        """
        if not callable(getattr(self._hal, "idle_step", None)):
            return
        # ADR-0048 Phase 2 — drive the idle stepper on WALL time, never the
        # node's clock. Under ``use_sim_time`` (simulation clock authority) a node-clock
        # timer fires off ``/clock`` — but the idle step is what ADVANCES
        # ``/clock`` (it steps the sim), so a sim-time timer here deadlocks: no
        # step → no /clock → no fire. A SYSTEM_TIME clock breaks the cycle so the
        # sim keeps stepping and bootstraps /clock. Harmless on wall-clock runs
        # (SYSTEM_TIME == the node clock there).
        from rclpy.clock import Clock, ClockType

        self._idle_timer = self._node.create_timer(
            1.0 / max(self._camera_rate_hz, 1.0),
            self._idle_step_tick,
            clock=Clock(clock_type=ClockType.SYSTEM_TIME),
        )
        self._node.get_logger().info(
            f"SimSensorBridge: sim-only idle stepper @ {self._camera_rate_hz:.1f} Hz "
            f"(idle_hold={self._idle_hold_ns / 1e6:.0f} ms) — keeps cameras live when idle."
        )

    def _idle_step_tick(self) -> None:
        """Advance the sim one HOLD tick when no recent real action has arrived.

        Yields to active skills via :func:`should_idle_step`; the camera-publish
        timer then republishes the freshened ``_last_obs`` (publish path
        unchanged).

        Containment: ``idle_step`` fires autonomously on this timer (with
        ``last_action_ns == 0`` an idle scene starts stepping immediately). If
        it raises — most likely an ``env.step`` action-dim mismatch on a native
        backend whose true width was not probed (the documented probe gap;
        ``so101_box`` wants 6-D but the fallback is 11-D) — we log ONE loud
        warning and cancel/disable this timer so it cannot crash-loop the graph
        every tick. We do NOT swallow silently: one warning, then stop.
        """
        # Already disabled (e.g. by a prior error) — a callback queued before
        # the cancel must be a no-op, never re-trigger the disabled path.
        if self._idle_timer is None:
            return
        idle_step = getattr(self._hal, "idle_step", None)
        if not callable(idle_step):
            return
        last_action_ns = int(getattr(self._hal, "last_action_ns", 0))
        if not should_idle_step(time.monotonic_ns(), last_action_ns, self._idle_hold_ns):
            return
        try:
            idle_step()
        except Exception as exc:  # reason: contain a per-tick crash-loop; warn once + disable
            self._node.get_logger().warning(
                f"SimSensorBridge: idle stepper disabled after error: {exc}. "
                "Possibly an env action-dim mismatch (an explicit env_action_dim "
                "override that disagrees with the backend's step width; native "
                "backends now expose their own action_dim so the probe resolves it). "
                "Cameras will only refresh while a skill is actively stepping the env."
            )
            if self._idle_timer is not None:
                self._idle_timer.cancel()
                self._idle_timer = None
            return
        # ADR-0049 — the env advanced; refresh the proprio snapshot so the
        # control group's odom/joint_state publishers stay fresh while idle.
        if self._on_step is not None:
            self._on_step()

    # -- 2-D LiDAR / LaserScan --
    def _setup_scan(self) -> None:
        """Create the ``/scan`` publisher + timer, gated only on manifest lidar_2d.

        Gate: the manifest declares a ``lidar_2d`` sensor
        (``RobotDescription.lidar_sensor is not None``); franka/ur5e/so100 nodes
        without one never advertise a scan topic.

        The publisher is created whenever a lidar is declared, regardless of
        whether the HAL has live MuJoCo handles. :meth:`_compute_scan_ranges`
        ray-casts against the scene when handles are bound (``SimAttachedHAL``)
        and emits a constant ``max_range`` no-hit fan otherwise (the in-process
        digital twin has no scene to ray-cast, so "no hit everywhere" is the
        honest reading slam_toolbox / Nav2 boot on). This makes the bridge the
        single owner of ``/scan`` — issue #191 Phase 3 removed the panda_mobile
        node's separate digital-twin no-hit publisher.
        """
        if self._description.lidar_sensor is None:
            return
        from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
        from sensor_msgs.msg import LaserScan

        # `/scan` is BEST_EFFORT (sensor-class data per CLAUDE.md §2 ROS QoS
        # table). slam_toolbox and Nav2 both subscribe BEST_EFFORT by default.
        scan_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=5,
        )
        self._scan_pub = self._node.create_publisher(LaserScan, "/scan", scan_qos)
        self._scan_timer = self._node.create_timer(
            1.0 / max(self._scan_rate_hz, 1.0), self._publish_scan
        )
        lidar = self._description.lidar_sensor
        self._node.get_logger().info(
            f"SimSensorBridge: publishing /scan @ {self._scan_rate_hz:.1f} Hz "
            f"frame={lidar.frame_id} beams={self._scan_n_beams} "
            f"range=[{self._scan_min_range_m}, {self._scan_max_range_m}] m."
        )

    def _publish_scan(self) -> None:
        """Construct + publish a /scan message (live MJCF ray-cast or synthetic no-hit).

        Lifted from ``openral_hal_panda_mobile.lifecycle_node._publish_scan`` /
        ``_compute_scan_ranges`` (ADR-0034 Phase 2). Uses the same
        :func:`openral_sim.backends.robocasa.synthesize_laser_scan_2d` call
        and identical no-hit fallback so nav-stack behaviour is bit-identical
        to the panda_mobile node.
        """
        if self._scan_pub is None:
            return
        import math  # reason: stdlib defer

        from sensor_msgs.msg import LaserScan

        n_beams = int(self._scan_n_beams)
        max_range = float(self._scan_max_range_m)
        min_range = float(self._scan_min_range_m)
        scan_rate = float(self._scan_rate_hz)

        ranges = self._compute_scan_ranges(n_beams=n_beams, max_range_m=max_range)

        lidar = self._description.lidar_sensor
        frame_id = lidar.frame_id if lidar is not None else "base_link"

        msg = LaserScan()
        msg.header.stamp = self._node.get_clock().now().to_msg()
        msg.header.frame_id = frame_id
        msg.angle_min = float(-math.pi)
        msg.angle_max = float(math.pi)
        msg.angle_increment = float(2.0 * math.pi / max(n_beams, 1))
        msg.scan_time = float(1.0 / max(scan_rate, 1.0))
        msg.time_increment = 0.0
        msg.range_min = min_range
        msg.range_max = max_range
        msg.ranges = list(ranges)
        msg.intensities = []
        self._scan_pub.publish(msg)

    def _compute_scan_ranges(self, *, n_beams: int, max_range_m: float) -> list[float]:
        """Return scan ranges — live MJCF ray-cast if handles are bound, else no-hit fan.

        Mirrors ``openral_hal_panda_mobile.lifecycle_node._compute_scan_ranges``
        exactly: tries ``hal.mujoco_handles()``, falls back to a
        ``max_range_m``-clamped no-hit list so slam_toolbox / Nav2 treat every
        beam as "nothing in front of me" rather than NaN-poisoning their grids.
        """
        handle = getattr(self._hal, "mujoco_handles", lambda: None)()
        if handle is None:
            # Non-MuJoCo backend: use the ranges the HAL surfaces (Isaac lidar
            # ray-cast, ADR-0045), else an honest no-hit fan.
            read = getattr(self._hal, "read_scan", None)
            if callable(read):
                scan = read()
                if scan is not None and len(scan) == n_beams:
                    return [float(r) for r in scan]
            return constant_scan_no_hit_ranges(n_beams=n_beams, max_range_m=max_range_m)
        model, data = handle
        from openral_core import (  # reason: scoped to scan synthesis
            extract_base_sim_joint_names,
        )
        from openral_sim.backends.robocasa import (  # reason: optional dep
            synthesize_laser_scan_2d,
        )

        # Pull MJCF joint names from the HAL's description so the
        # sim-side helper doesn't depend on hardcoded robosuite /
        # robocasa naming conventions. ADR-0025.
        base_names: tuple[str, str, str] | None = None
        description = getattr(self._hal, "description", None)
        if description is not None:
            base_names = extract_base_sim_joint_names(description)

        ranges = synthesize_laser_scan_2d(
            model=model,
            data=data,
            base_joint_names=base_names,
            n_beams=n_beams,
            max_range_m=max_range_m,
        )
        return [float(r) for r in ranges]

    # -- Depth PointCloud2 (ADR-0030 / ADR-0034 Phase 2) --
    def _setup_depth(self) -> None:
        """Create a PointCloud2 publisher + timer per depth SensorSpec.

        Gate conditions (both must hold to publish):
        1. The manifest declares ≥1 depth/point_cloud SensorSpec with intrinsics
           (``depth_cloud.is_depth_sensor(s)`` is True for at least one sensor).
        2. The HAL exposes live MuJoCo handles
           (``hal.mujoco_handles()`` returns a non-``None`` pair).

        When either gate fails the method returns silently — no publisher,
        no timer, no TF broadcaster. This lets arm-only robots use the bridge
        without advertising any depth topics.

        Lifted from ``openral_hal_panda_mobile.lifecycle_node._setup_depth_publishers``
        (ADR-0034 Phase 2). QoS matches panda_mobile's BEST_EFFORT depth QoS.
        """
        # Publish if the HAL ray-casts depth (MuJoCo) OR surfaces ready clouds in
        # obs (non-MuJoCo, e.g. the Isaac scene — ADR-0045). Otherwise no topics.
        has_mujoco = getattr(self._hal, "mujoco_handles", lambda: None)() is not None
        has_obs_depth = callable(getattr(self._hal, "read_depth_clouds", None))
        if not (has_mujoco or has_obs_depth):
            return
        from openral_hal import depth_cloud

        depth_specs = [s for s in self._description.sensors if depth_cloud.is_depth_sensor(s)]
        if not depth_specs:
            return

        from rclpy.qos import (
            QoSDurabilityPolicy,
            QoSProfile,
            QoSReliabilityPolicy,
        )
        from sensor_msgs.msg import CameraInfo, Image, PointCloud2
        from tf2_ros import TransformBroadcaster

        if self._tf_broadcaster is None:  # may already exist (RGB camera TFs)
            self._tf_broadcaster = TransformBroadcaster(self._node)

        depth_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=5,
        )
        # CameraInfo is low-rate + near-static: RELIABLE + TRANSIENT_LOCAL so any
        # subscriber QoS (nvblox's included) matches and late joiners get it.
        info_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            depth=1,
        )
        for spec in depth_specs:
            base = f"/openral/cameras/{spec.name}"
            self._depth_pubs[spec.name] = self._node.create_publisher(
                PointCloud2, f"{base}/points", depth_qos
            )
            # ADR-0064 — dense depth image + CameraInfo for nvblox's depth integrator.
            self._depth_image_pubs[spec.name] = self._node.create_publisher(
                Image, f"{base}/depth/image", depth_qos
            )
            self._depth_info_pubs[spec.name] = self._node.create_publisher(
                CameraInfo, f"{base}/depth/camera_info", info_qos
            )
        self._depth_timer = self._node.create_timer(
            1.0 / max(self._depth_rate_hz, 1.0), self._publish_depth_clouds
        )
        self._node.get_logger().info(
            f"SimSensorBridge: publishing {len(depth_specs)} depth camera(s) "
            "(PointCloud2 + 32FC1 depth image + CameraInfo): "
            + ", ".join(s.name for s in depth_specs)
            + f" @ {self._depth_rate_hz:.1f} Hz"
        )

    def _resolve_depth_base_body(self, model: object) -> None:
        """Resolve + cache the MJCF base body name (TF parent + self-exclusion).

        Mirrors panda_mobile's ``_resolve_depth_base_body``: strips the
        ``_joint_*`` tail off the first base joint name and appends ``_base``
        (``mobilebase0_base`` under a composed kitchen), falling back to the
        bare ``"base"`` for synthetic MJCFs.

        Also populates ``_depth_self_bodies`` — the robot's own MJCF body ids
        for the depth self-filter, derived from the manifest's sim_joint_name
        prefixes (arm + base + gripper).
        """
        import mujoco  # reason: defer optional sim dep

        from openral_hal.depth_cloud import resolve_base_body_name, robot_self_body_ids

        description = getattr(self._hal, "description", None)
        base_body = resolve_base_body_name(model, description=description)
        self._depth_base_body = base_body
        self._depth_base_body_id = (
            int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, base_body))
            if base_body is not None
            else -1
        )
        # Robot self-body set (arm + base + gripper) for the depth self-filter.

        sim_names = (
            [j.sim_joint_name for j in description.joints] if description is not None else []
        )
        self._depth_self_bodies = robot_self_body_ids(model, sim_names)

    def _render_size(self) -> tuple[int, int] | None:
        """Resolution the scene actually rendered its RGB frames at, or ``None``.

        deploy-sim scenes render the same MuJoCo camera at
        ``scene.observation_width``/``height`` (e.g. 128 or 640), which can
        differ from the manifest's nominal intrinsics resolution. The depth
        synth must back-project at the render resolution so its cloud lines up
        with the RGB the detector ran on; this reads the live rendered RGB frame
        shape (the ground truth of what robosuite rendered) and returns
        ``(width, height)`` so :func:`depth_synth_kwargs` can rescale the
        intrinsics. Returns ``None`` when no frame is available yet (the synth
        then falls back to the manifest's nominal intrinsics).
        """
        reader = getattr(self._hal, "read_images", None)
        if reader is None:
            return None
        images = reader()
        if not isinstance(images, dict):
            return None
        for arr in images.values():
            shape = getattr(arr, "shape", None)
            if shape is not None and len(shape) == _IMAGE_DIM:
                h, w = int(shape[0]), int(shape[1])
                if w > 0 and h > 0:
                    return (w, h)
        return None

    def _publish_depth_clouds(self) -> None:
        """Ray-cast + publish a PointCloud2 (+ ADR-0064 depth image) per camera, and its TF.

        ADR-0030 — the deploy-sim source for octomap_server. Each depth
        ``SensorSpec`` is synthesised with
        :func:`openral_sim.backends.depth_camera.synthesize_depth_pointcloud`
        (camera-optical frame), packed into ``sensor_msgs/PointCloud2``,
        and published with a live ``base_link -> <camera>_optical_frame``
        TF so octomap_server can lift it into the world map. A camera
        whose MJCF name doesn't resolve is disabled after one warning
        (sim sensor, not a safety path).

        Lifted from ``openral_hal_panda_mobile.lifecycle_node._publish_depth_clouds``
        (ADR-0034 Phase 2). Logic is faithfully preserved — the ray-cast, the
        self-body exclusion, the point filtering, and the TF broadcast are
        unchanged so octomap_server sees bit-identical clouds.
        """
        if not self._depth_pubs:
            return
        handle = getattr(self._hal, "mujoco_handles", lambda: None)()
        if handle is None:
            # Non-MuJoCo backend: publish the base_link clouds the HAL surfaces.
            self._publish_depth_clouds_from_obs()
            return
        model, data = handle

        from geometry_msgs.msg import (
            TransformStamped,
        )
        from openral_core.exceptions import ROSConfigError
        from openral_sim.backends.depth_camera import (
            synthesize_depth_image,
            synthesize_depth_pointcloud,
        )

        from openral_hal.depth_cloud import (
            camera_info_from_intrinsics,
            camera_optical_tf_to_base,
            depth_image_from_grid,
            depth_synth_kwargs,
            is_depth_sensor,
            pointcloud2_from_points_xyz,
        )

        if self._depth_base_body is None and self._depth_base_body_id < 0:
            self._resolve_depth_base_body(model)
        exclude_id = self._depth_base_body_id if self._depth_base_body_id >= 0 else None

        max_range_default = float(self._depth_max_range_m)
        stride = max(int(self._depth_pixel_stride), 1)
        stamp = self._node.get_clock().now().to_msg()

        base_frame_id = getattr(self._description, "base_frame", "base_link")
        specs = {s.name: s for s in self._description.sensors if is_depth_sensor(s)}
        for name, pub in self._depth_pubs.items():
            if name in self._depth_disabled:
                continue
            spec = specs.get(name)
            if spec is None:
                continue
            try:
                kwargs = depth_synth_kwargs(
                    spec,
                    max_range_default=max_range_default,
                    render_size=self._render_size(),
                )
                points = synthesize_depth_pointcloud(
                    model=model,
                    data=data,
                    stride=stride,
                    exclude_body_id=exclude_id,
                    exclude_body_ids=self._depth_self_bodies or None,
                    **kwargs,
                )
                cloud = pointcloud2_from_points_xyz(points, frame_id=spec.frame_id, stamp=stamp)
                pub.publish(cloud)
                # ADR-0064 — dense 32FC1 depth image + CameraInfo for nvblox.
                # Same pinhole ray-cast, but every pixel (0.0 = no return); the
                # CameraInfo intrinsics scale by 1/stride to match the raster.
                depth_grid = synthesize_depth_image(
                    model=model,
                    data=data,
                    stride=stride,
                    exclude_body_id=exclude_id,
                    exclude_body_ids=self._depth_self_bodies or None,
                    **kwargs,
                )
                h_eff, w_eff = (int(depth_grid.shape[0]), int(depth_grid.shape[1]))
                self._depth_image_pubs[name].publish(
                    depth_image_from_grid(depth_grid, frame_id=spec.frame_id, stamp=stamp)
                )
                self._depth_info_pubs[name].publish(
                    camera_info_from_intrinsics(
                        width=w_eff,
                        height=h_eff,
                        fx=kwargs["fx"] / stride,
                        fy=kwargs["fy"] / stride,
                        cx=kwargs["cx"] / stride,
                        cy=kwargs["cy"] / stride,
                        frame_id=spec.frame_id,
                        stamp=stamp,
                    )
                )
                if self._depth_base_body is not None and self._tf_broadcaster is not None:
                    xyz, quat = camera_optical_tf_to_base(
                        model=model,
                        data=data,
                        camera_name=kwargs["camera_name"],
                        base_body_name=self._depth_base_body,
                    )
                    tf = TransformStamped()
                    tf.header.stamp = stamp
                    tf.header.frame_id = base_frame_id
                    tf.child_frame_id = spec.frame_id
                    tf.transform.translation.x = xyz[0]
                    tf.transform.translation.y = xyz[1]
                    tf.transform.translation.z = xyz[2]
                    tf.transform.rotation.x = quat[0]
                    tf.transform.rotation.y = quat[1]
                    tf.transform.rotation.z = quat[2]
                    tf.transform.rotation.w = quat[3]
                    self._tf_broadcaster.sendTransform(tf)
            except ROSConfigError as exc:
                self._depth_disabled.add(name)
                self._node.get_logger().warning(
                    f"depth camera {name!r} disabled: {exc}; "
                    "check the SensorSpec's mjcf_camera metadata."
                )

    def _publish_depth_clouds_from_obs(self) -> None:
        """Publish the HAL's ready ``base_link`` clouds as ``PointCloud2`` (ADR-0045).

        Non-MuJoCo path: the backend (Isaac scene) already deprojected each depth
        camera to a ``(N, 3)`` cloud in ``base_link`` (Isaac owns the camera
        convention), surfaced via ``hal.read_depth_clouds()``. We just wrap each in
        a ``PointCloud2`` stamped ``base_link`` — no ray-cast, no per-camera optical
        TF (``base_link`` is already on /tf via /odom). octomap lifts it into the
        world map exactly as it does the MuJoCo clouds.
        """
        read = getattr(self._hal, "read_depth_clouds", None)
        if not callable(read):
            return
        clouds = read()
        if not clouds:
            return
        from openral_hal.depth_cloud import pointcloud2_from_points_xyz

        base_frame_id = getattr(self._description, "base_frame", "base_link")
        stamp = self._node.get_clock().now().to_msg()
        for name, pub in self._depth_pubs.items():
            pts = clouds.get(name)
            if pts is None or pts.size == 0:
                continue
            cloud = pointcloud2_from_points_xyz(pts, frame_id=base_frame_id, stamp=stamp)
            pub.publish(cloud)

    # -- Viewer --
    def _setup_viewer(self) -> None:
        handles = getattr(self._hal, "mujoco_handles", lambda: None)()
        if not self._viewer_enabled or handles is None:
            return
        model, data = handles
        try:
            import mujoco.viewer  # reason: optional dep — robosuite/mujoco ships it

            # Hide both side panels (left settings / right info) so the window
            # shows only the simulation render — the panels are still reachable
            # at runtime via Tab / Shift+Tab.
            self._viewer = mujoco.viewer.launch_passive(
                model, data, show_left_ui=False, show_right_ui=False
            )
            # Mirror RoboCasa's offscreen camera render config: hide geom group 0
            # (the collision shell, which RoboCasa colours dark red by convention)
            # and show group 1 (the textured visual geoms). The passive viewer
            # renders ALL groups by default, so without this the kitchen shows up
            # as a red collision box even though the /camera streams (group 1
            # only) render correctly.
            with contextlib.suppress(Exception):
                self._viewer.opt.geomgroup[0] = 0  # collision — hide
                self._viewer.opt.geomgroup[1] = 1  # visual — show
            # Open the viewer on a 3rd-person scene camera (agentview/top/…),
            # falling back to the base-aligned free camera for camera-less twins.
            self._aim_viewer_camera(model, data)
        except Exception as exc:  # reason: GL/DISPLAY failures are non-fatal (headless)
            self._node.get_logger().warning(
                f"SimSensorBridge: viewer launch failed ({exc!s}); continuing headless. "
                "Common causes: no DISPLAY, MUJOCO_GL=egl (use 'glfw'), missing libglfw/libGL."
            )
            self._viewer = None
            return
        self._viewer_timer = self._node.create_timer(
            1.0 / max(self._viewer_sync_rate_hz, 1.0), self._sync_viewer
        )
        self._node.get_logger().info(
            f"SimSensorBridge: MuJoCo viewer open @ {self._viewer_sync_rate_hz:.1f} Hz."
        )

    def _aim_viewer_camera(self, model: Any, data: Any) -> None:
        """Set the viewer's opening **free-camera** pose (mouse stays live).

        Sets the initial viewpoint via
        :func:`openral_hal.depth_cloud.initial_viewer_camera` — eye at the
        authored overview camera (``agentview`` / ``top`` / …) with the orbit
        pivot on the robot base, else the base-aligned default. The camera stays
        ``mjCAMERA_FREE`` so the user can drag to orbit and scroll to zoom; we
        only set the initial view. Best effort: any failure leaves the default.
        """
        if self._viewer is None:
            return
        with contextlib.suppress(Exception):
            import mujoco  # reason: optional sim dep

            from openral_hal.depth_cloud import initial_viewer_camera

            lookat, distance, azimuth, elevation = initial_viewer_camera(
                model=model, data=data, description=getattr(self._hal, "description", None)
            )
            with self._viewer.lock():
                cam = self._viewer.cam
                cam.type = mujoco.mjtCamera.mjCAMERA_FREE
                cam.lookat[:] = lookat
                cam.distance = distance
                cam.azimuth = azimuth
                cam.elevation = elevation

    def _sync_viewer(self) -> None:
        if self._viewer is None:
            return
        try:
            self._viewer.sync()
        except Exception as exc:  # reason: viewer closed by user
            self._node.get_logger().warning(f"viewer sync failed; closing: {exc!s}")
            if self._viewer_timer is not None:
                self._viewer_timer.cancel()
                self._viewer_timer = None
            self._viewer = None

    # -- offscreen cinecam recorder (website-video capture) --
    def _configure_cinecam_camera(self, mujoco: Any, model: Any, data: Any) -> Any:
        """Build the free-camera pose from the viewer default + env overrides.

        Resolves the opening pose via :func:`initial_viewer_camera`, then applies
        absolute overrides (``OPENRAL_CINECAM_AZ_DEG`` / ``_EL_DEG`` / ``_DIST_M``)
        and deltas (``_AZ_OFFSET_DEG`` / ``_EL_OFFSET_DEG`` / ``_DIST_DELTA_M``),
        resolves the base body for the follow-cam, and snapshots the final pose
        as the baseline for the live ``OPENRAL_CINECAM_TUNE`` deltas.
        """
        import os  # reason: env-gated capture feature

        from openral_hal.depth_cloud import initial_viewer_camera, resolve_base_body_name

        cam = mujoco.MjvCamera()
        lookat, distance, azimuth, elevation = initial_viewer_camera(
            model=model, data=data, description=getattr(self._hal, "description", None)
        )
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = lookat
        cam.distance = distance
        cam.azimuth = azimuth
        cam.elevation = elevation
        for env_key, attr in (
            ("OPENRAL_CINECAM_AZ_DEG", "azimuth"),
            ("OPENRAL_CINECAM_EL_DEG", "elevation"),
            ("OPENRAL_CINECAM_DIST_M", "distance"),
        ):
            val = os.environ.get(env_key)
            if val:
                setattr(cam, attr, float(val))
        az_off = os.environ.get("OPENRAL_CINECAM_AZ_OFFSET_DEG")
        if az_off:
            cam.azimuth = float(cam.azimuth) + float(az_off)
        el_off = os.environ.get("OPENRAL_CINECAM_EL_OFFSET_DEG")  # +ve = less top-down
        if el_off:
            cam.elevation = float(cam.elevation) + float(el_off)
        dist_delta = os.environ.get("OPENRAL_CINECAM_DIST_DELTA_M")  # -ve = closer
        if dist_delta:
            cam.distance = max(0.3, float(cam.distance) + float(dist_delta))
        self._cinecam_base_body = resolve_base_body_name(
            model, description=getattr(self._hal, "description", None)
        )
        self._cinecam_setup_az = float(cam.azimuth)
        self._cinecam_setup_el = float(cam.elevation)
        self._cinecam_setup_dist = float(cam.distance)
        return cam

    def _setup_cinecam(self) -> None:
        """Render the pulled-back free-camera view to JPGs when OPENRAL_CINECAM_DIR is set.

        Offscreen (EGL) render — robust against the onscreen GLFW viewer being
        unmapped/throttled by the desktop WM. Frame pose matches the viewer
        (:func:`openral_hal.depth_cloud.initial_viewer_camera`); collision shells
        are hidden so RoboCasa textures show. ``OPENRAL_CINECAM_SIZE`` (``WxH``,
        default ``1280x960``) and ``OPENRAL_CINECAM_FPS`` (default ``12``) tune it.
        """
        import os  # reason: env-gated debug/capture feature

        out_dir = os.environ.get("OPENRAL_CINECAM_DIR")
        if not out_dir:
            return
        handles = getattr(self._hal, "mujoco_handles", lambda: None)()
        if handles is None:
            return
        model, data = handles
        try:
            import mujoco  # reason: optional sim dep

            from openral_hal.depth_cloud import apply_robosuite_visual_geomgroups

            size = os.environ.get("OPENRAL_CINECAM_SIZE", "1280x960")
            width, height = (int(v) for v in size.lower().split("x"))
            fps = float(os.environ.get("OPENRAL_CINECAM_FPS", "12"))
            os.makedirs(out_dir, exist_ok=True)
            self._cinecam_out_dir = out_dir
            # The MJCF offscreen framebuffer defaults to 640x480; enlarge it so
            # the cinecam can render at the requested (higher) resolution.
            with contextlib.suppress(Exception):
                model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), width)
                model.vis.global_.offheight = max(int(model.vis.global_.offheight), height)
            try:
                self._cinecam_renderer = mujoco.Renderer(model, height=height, width=width)
            except Exception:  # reason: offscreen framebuffer smaller than request
                width = int(model.vis.global_.offwidth)
                height = int(model.vis.global_.offheight)
                self._cinecam_renderer = mujoco.Renderer(model, height=height, width=width)
            self._cinecam_model = model
            self._cinecam_w = width
            self._cinecam_h = height
            self._cinecam_opt = mujoco.MjvOption()
            apply_robosuite_visual_geomgroups(self._cinecam_opt, model)
            self._cinecam_cam = self._configure_cinecam_camera(mujoco, model, data)
        except Exception as exc:  # reason: GL/render failure is non-fatal
            self._node.get_logger().warning(
                f"SimSensorBridge: cinecam setup failed ({exc!s}); no offscreen capture."
            )
            self._cinecam_renderer = None
            return
        self._cinecam_timer = self._node.create_timer(1.0 / max(fps, 1.0), self._render_cinecam)
        self._node.get_logger().info(
            f"SimSensorBridge: cinecam recording {width}x{height} @ {fps:.0f} Hz → {out_dir}"
        )

    def _render_cinecam(self) -> None:
        if self._cinecam_renderer is None or self._cinecam_cam is None:
            return
        handles = getattr(self._hal, "mujoco_handles", lambda: None)()
        if handles is None:
            return
        model, data = handles
        try:
            import mujoco  # reason: optional sim dep
            from PIL import Image  # reason: optional dep — present in the sim env

            # robosuite rebuilds its sim (a fresh MjModel) on env reset, so the
            # renderer bound to the setup-time model would render a frozen scene.
            # Rebuild it (same size + scene opts) whenever the live model changes.
            if model is not self._cinecam_model:
                with contextlib.suppress(Exception):
                    self._cinecam_renderer.close()
                self._cinecam_renderer = mujoco.Renderer(
                    model, height=self._cinecam_h, width=self._cinecam_w
                )
                from openral_hal.depth_cloud import apply_robosuite_visual_geomgroups

                apply_robosuite_visual_geomgroups(self._cinecam_opt, model)
                self._cinecam_model = model
            # Ensure derived kinematics (geom_xpos) reflect the latest qpos.
            mujoco.mj_forward(model, data)
            # Follow-cam: re-pin lookat to the live base position so a navigating
            # robot stays centred (OPENRAL_CINECAM_FOLLOW=1). Lift the pivot to
            # ~torso height for a nicer frame.
            import os  # reason: env-gated

            if os.environ.get("OPENRAL_CINECAM_FOLLOW") and self._cinecam_base_body:
                bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, self._cinecam_base_body)
                if bid >= 0:
                    self._cinecam_cam.lookat[0] = float(data.xpos[bid][0])
                    self._cinecam_cam.lookat[1] = float(data.xpos[bid][1])
                    self._cinecam_cam.lookat[2] = float(data.xpos[bid][2]) + 0.5
            # Live tuning: re-read "az_delta el_delta dist_delta" from the tune
            # file each tick so framing can be dialed in without a relaunch.
            tune_path = os.environ.get("OPENRAL_CINECAM_TUNE")
            if tune_path and os.path.exists(tune_path):
                with contextlib.suppress(Exception):
                    with open(tune_path) as _tf:
                        parts = _tf.read().split()
                    az_d, el_d, dist_d = (float(parts[0]), float(parts[1]), float(parts[2]))
                    self._cinecam_cam.azimuth = self._cinecam_setup_az + az_d
                    self._cinecam_cam.elevation = self._cinecam_setup_el + el_d
                    self._cinecam_cam.distance = max(0.3, self._cinecam_setup_dist + dist_d)
            self._cinecam_renderer.update_scene(
                data, camera=self._cinecam_cam, scene_option=self._cinecam_opt
            )
            rgb = self._cinecam_renderer.render()
            self._cinecam_frame += 1
            path = f"{self._cinecam_out_dir}/f_{self._cinecam_frame:05d}.jpg"
            Image.fromarray(rgb).save(path, quality=88)
        except Exception as exc:  # reason: a dropped frame must not crash the HAL
            with contextlib.suppress(Exception):
                self._node.get_logger().warning(f"cinecam frame failed: {exc!s}")
