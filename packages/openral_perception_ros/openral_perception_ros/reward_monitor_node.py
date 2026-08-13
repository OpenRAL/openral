#!/usr/bin/env python3
"""Reward-monitor query service node.

Subscribes the co-active VLA's camera ``sensor_msgs/Image`` stream(s), buffers
recent frames in a rolling time window, and serves
``/openral/perception/query_task_progress`` (``openral_msgs/srv/QueryTaskProgress``):
a read-only, on-demand "how is the task progressing / succeeding over the last
N seconds?" backed by a ``kind: "reward"`` rSkill (Robometer-4B NF4 by default).

Driven by the reasoner's ``query_task_progress`` tool: the reasoner co-activates
this monitor with a VLA and queries it to decide whether to continue, escalate
to ``query_scene``, advance, or enter the replanning ladder. The signal is
**advisory** — it never actuates (CLAUDE.md §1.1).

This is the reward counterpart of the scene-VLM node
(:mod:`openral_perception_ros.scene_vlm_node`, which serves ``query_scene``).
The rolling buffer lives here, node-side; reward backends score on demand.

**Frame-source agnostic.** It subscribes the same camera image topic the VLA
consumes — fed by the GStreamer tee on real hardware or the sim HAL camera
publisher in ``deploy-sim`` (no GStreamer). Frame timestamps use the node clock,
so eviction/staleness behave identically against a sim clock and a real clock.

Parameters:
    cameras (str[]): logical cameras as ``"id=topic"`` entries. Empty = a single
        camera ``primary_camera`` on ``image_topic``.
    primary_camera (str): id of the default camera the monitor scores.
    image_topic (str): single-camera fallback topic.
    manifest_path (str): rSkill manifest path (``kind: "reward"``). Required.
    task (str): default task instruction (used when a request leaves ``task`` empty).
    score_period_s (float): reward-score heartbeat cadence — how often the timer
        scores the recent window and emits the ``reward.score`` span that drives the
        dashboard reward bar. Default 2.0 s. The timer runs while a VLA is executing
        (``gate_scoring_on_execution``) or the critic leg is on.
    score_window_s (float): seconds of RECENT footage the heartbeat scores. Kept
        small (default 2.0 s ≈ 6 frames) so the reward forward fits beside a VLA on
        an 8 GB card; the full 40 s buffer would OOM. Raise on a bigger GPU.
    enable_critic_score (bool): additionally publish a generic
        ``openral_msgs/CriticScore`` per heartbeat to feed the Tier-C
        critic producer. Default False. Gates ONLY the publish — the scoring +
        dashboard bar run regardless.
    critic_score_topic (str): topic for the CriticScore stream. Default
        ``/openral/critic/score``.
    critic_score_threshold (float): pass bar stamped on each CriticScore (the
        producer's watchdog fires when progress stays below it). Default 0.8.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from typing import Any

from openral_perception_ros.camera_topics import resolve_camera_topics


def _camera_label(topic: str) -> str:
    """Derive a short, human-readable camera name from a camera image topic.

    Convention topics are ``/openral/cameras/<name>/image`` (see
    ``DeployScene.sensors`` / the ROS 2 sensor bringup) — this pulls out
    ``<name>`` (e.g. ``"top"``) so the dashboard can show which camera the
    reward monitor actually attends to instead of a full topic string or the
    uninformative default ``primary_camera`` id (usually just ``"default"``).
    Falls back to the raw topic when it doesn't match the convention.

    Example:
        >>> _camera_label("/openral/cameras/top/image")
        'top'
        >>> _camera_label("/some/other/topic")
        '/some/other/topic'
    """
    parts = topic.strip("/").split("/")
    if len(parts) >= 3 and parts[0] == "openral" and parts[1] == "cameras":
        return parts[2]
    return topic


def main(args: Any = None) -> None:
    """Entry point: init ROS, spin the reward-monitor node, shut down cleanly."""
    import rclpy
    from openral_runner.backends.reward.frame_source import Frame, RollingFrameBuffer
    from openral_runner.backends.reward.robometer_reward import critic_score_from_assessment
    from rclpy.executors import ExternalShutdownException
    from rclpy.node import Node
    from rclpy.qos import (
        QoSDurabilityPolicy,
        QoSHistoryPolicy,
        QoSProfile,
        QoSReliabilityPolicy,
    )
    from sensor_msgs.msg import Image

    from openral_perception_ros.image_convert import ImageConvertError, image_to_bgr_bytes

    class RewardMonitorNode(Node):  # type: ignore[misc]
        """Subscribe camera Image(s), buffer frames, serve query_task_progress."""

        def __init__(self) -> None:  # noqa: PLR0915  # reason: node ctor wires cameras + reward backend + critic + scoring-gate in one place
            super().__init__("openral_reward_monitor")
            self.declare_parameter("cameras", [""])
            self.declare_parameter("primary_camera", "default")
            self.declare_parameter("image_topic", "/openral/cameras/agentview_left/image")
            self.declare_parameter("manifest_path", "")
            self.declare_parameter("task", "")
            # Opt-in: also publish a generic openral_msgs/CriticScore per
            # window so the Tier-C critic producer (critic_producer_node) can fire a
            # /openral/failure/critic on a progress stall. Off by default — the node
            # stays query-only unless asked.
            self.declare_parameter("enable_critic_score", False)
            self.declare_parameter("critic_score_topic", "/openral/critic/score")
            self.declare_parameter("critic_score_threshold", 0.8)
            # Reward-score heartbeat: how often to score (period) and over how much
            # RECENT footage (window). The window is deliberately small — the full
            # 40 s buffer subsampled to max_frames=8 OOMs the reward forward when
            # co-resident with a VLA on an 8 GB card; a ~2 s slice (≈6 frames at
            # 3 fps) fits and still tracks the live progress arc. Raise on a bigger
            # GPU. (The reasoner's mission-verify query keeps the full-buffer window
            # for the calibrated completion verdict — a separate path.)
            self.declare_parameter("score_period_s", 2.0)
            self.declare_parameter("score_window_s", 2.0)

            gp = self.get_parameter
            manifest_path = gp("manifest_path").get_parameter_value().string_value
            if not manifest_path:
                raise ValueError("reward_monitor_node requires a manifest_path (kind: 'reward')")

            self._default_task = gp("task").get_parameter_value().string_value
            self._cameras = self._resolve_cameras()
            self._primary_id = next(iter(self._cameras))
            # Surfaced on the ``reward.score`` span (dashboard rSkill card) so an
            # operator can see which camera the reward monitor is actually
            # scoring — the primary_camera id is usually the uninformative
            # single-camera fallback "default", the topic tail is not.
            self._camera_name = _camera_label(self._cameras[self._primary_id])
            self._monitor, window_s, fps = self._build_monitor(manifest_path)
            # One rolling buffer per camera id (the primary is what we score).
            self._buffers: dict[str, RollingFrameBuffer] = {
                cid: RollingFrameBuffer(window_s=window_s) for cid in self._cameras
            }
            self._target_dt_ns = int(1e9 / fps) if fps > 0 else 0
            self._last_push_ns: dict[str, int] = {cid: 0 for cid in self._cameras}

            img_qos = QoSProfile(
                history=QoSHistoryPolicy.KEEP_LAST,
                depth=1,
                reliability=QoSReliabilityPolicy.BEST_EFFORT,
                durability=QoSDurabilityPolicy.VOLATILE,
            )
            self._subs = [
                self.create_subscription(Image, topic, self._make_cache_cb(cid), img_qos)
                for cid, topic in self._cameras.items()
            ]

            self._srv = None
            try:
                from openral_msgs.srv import QueryTaskProgress

                self._srv = self.create_service(
                    QueryTaskProgress,
                    "/openral/perception/query_task_progress",
                    self._on_query_task_progress,
                )
            except ImportError:
                self.get_logger().warning(
                    "openral_msgs/srv/QueryTaskProgress not built; "
                    "query_task_progress service disabled"
                )

            # 2026-06-29 — gate continuous scoring to VLA-execution windows AND score
            # the instruction the policy is actually running. A VLA's reward only
            # means something while it acts, and only against the task it is doing —
            # scoring the collective mission goal ("put ALL objects…") while the VLA
            # picks one object can never read as progress (and wastes the GPU on an
            # idle scene). When `gate_scoring_on_execution` is set, the node scores
            # only while `/openral/reward/active_task` (std_msgs/String, the exact
            # prompt the reasoner sent the VLA) is non-empty, and scores THAT task.
            # Default off = score the default task continuously (legacy; the on-demand
            # query_task_progress service is unaffected either way).
            self.declare_parameter("gate_scoring_on_execution", False)
            self._gate_scoring = gp("gate_scoring_on_execution").get_parameter_value().bool_value
            self._vla_active = not self._gate_scoring  # ungated → always "active"
            self._active_task = ""  # the VLA's current instruction (gated mode)
            if self._gate_scoring:
                from std_msgs.msg import String as _String

                self.create_subscription(
                    _String,
                    "/openral/reward/active_task",
                    self._on_reward_active_task,
                    QoSProfile(
                        history=QoSHistoryPolicy.KEEP_LAST,
                        depth=1,
                        reliability=QoSReliabilityPolicy.RELIABLE,
                        durability=QoSDurabilityPolicy.VOLATILE,
                    ),
                )

            enable_critic = gp("enable_critic_score").get_parameter_value().bool_value
            self._score_window_s = gp("score_window_s").get_parameter_value().double_value
            score_period = gp("score_period_s").get_parameter_value().double_value

            # Optional CriticScore PUBLISHER leg (Tier-C producer feed).
            # `enable_critic_score` gates ONLY this publish; the scoring itself runs
            # on the heartbeat timer below regardless, so the dashboard reward bar
            # never depends on the critic being wired.
            self._critic_pub = None
            self._critic_msg_cls: Any = None
            self._critic_threshold = 0.0
            if enable_critic:
                self._critic_threshold = (
                    gp("critic_score_threshold").get_parameter_value().double_value
                )
                critic_topic = gp("critic_score_topic").get_parameter_value().string_value
                try:
                    from openral_msgs.msg import CriticScore

                    critic_qos = QoSProfile(
                        history=QoSHistoryPolicy.KEEP_LAST,
                        depth=10,
                        reliability=QoSReliabilityPolicy.RELIABLE,
                        durability=QoSDurabilityPolicy.VOLATILE,
                    )
                    self._critic_msg_cls = CriticScore
                    self._critic_pub = self.create_publisher(CriticScore, critic_topic, critic_qos)
                    self.get_logger().info(
                        f"critic_score: publishing progress on {critic_topic!r} "
                        f"(threshold={self._critic_threshold})"
                    )
                except ImportError:
                    self.get_logger().warning(
                        "openral_msgs/CriticScore not built; critic_score publishing disabled"
                    )

            # Reward-score heartbeat: scores a bounded recent window on a timer so the
            # dashboard rSkill card shows a live progress/success bar the whole time a
            # skill runs — INDEPENDENT of the CriticScore leg and of whether a reasoner
            # is polling query_task_progress. `_score_tick` self-limits to
            # `_vla_active` (the runner/reasoner publishes /openral/reward/active_task
            # around each execute_rskill), so it never grinds the GPU on an idle scene.
            # Created only when execution-gating bounds it, or the critic leg needs it;
            # ungated + no-critic keeps the legacy on-demand-only behaviour.
            self._score_timer = None
            if self._gate_scoring or enable_critic:
                self._score_timer = self.create_timer(score_period, self._score_tick)

            heartbeat = (
                f"every {score_period}s/win {self._score_window_s}s" if self._score_timer else "off"
            )
            self.get_logger().info(
                f"reward_monitor: cameras={self._cameras} primary={self._primary_id!r} "
                f"window_s={window_s} fps={fps} manifest={manifest_path}, "
                f"query_task_progress={'on' if self._srv else 'off'}, "
                f"score_heartbeat={heartbeat}, critic_score={'on' if self._critic_pub else 'off'}"
            )

        def _resolve_cameras(self) -> dict[str, str]:
            """Resolve the camera-id -> topic map (camera-agnostic)."""
            gp = self.get_parameter
            return resolve_camera_topics(
                list(gp("cameras").get_parameter_value().string_array_value),
                primary_camera=gp("primary_camera").get_parameter_value().string_value,
                image_topic=gp("image_topic").get_parameter_value().string_value,
            )

        def _build_monitor(self, manifest_path: str) -> tuple[Any, float, float]:
            """Build the reward backend; return (monitor, frame_window_s, target_fps)."""
            from openral_core.schemas import RSkillManifest
            from openral_runner.backends.reward.robometer_reward import build_reward_monitor

            manifest = RSkillManifest.from_yaml(manifest_path)
            if manifest.reward is None:
                raise ValueError(
                    f"manifest {manifest.name!r} is kind:reward but has no reward block"
                )
            monitor = build_reward_monitor(manifest)
            self.get_logger().info(f"reward backend model={manifest.name}")
            self._critic_id = manifest.name
            return monitor, manifest.reward.frame_window_s, manifest.reward.target_fps

        def _make_cache_cb(self, cid: str) -> Callable[[Any], None]:
            def _cb(msg: Any) -> None:
                now_ns = self.get_clock().now().nanoseconds
                # Downsample to the model's target fps — buffering every camera
                # frame (30-200 Hz) would waste memory; the monitor is S2-rate.
                if self._target_dt_ns and (now_ns - self._last_push_ns[cid]) < self._target_dt_ns:
                    return
                try:
                    bgr, w, h = image_to_bgr_bytes(msg)
                except ImageConvertError as exc:
                    self.get_logger().debug(f"cache_frame({cid}): convert failed: {exc}")
                    return
                self._buffers[cid].push(Frame(stamp_ns=now_ns, bgr=bgr, width=w, height=h))
                self._last_push_ns[cid] = now_ns

            return _cb

        def _emit_score_span(self, a: dict[str, Any], task: str) -> None:
            """Trace one assessment as a ``reward.score`` span.

            The dashboard's rSkill card renders the latest score as a live
            progress/success bar and the ``reward.camera`` attribute as the
            camera the monitor is attending to; without this span a scored run
            leaves no reward record in the trace stream. No-op when OTel is
            unconfigured.
            """
            from openral_observability import semconv
            from opentelemetry import trace

            tracer = trace.get_tracer("openral_perception_ros")
            with tracer.start_as_current_span(
                semconv.SPAN_REWARD_SCORE,
                attributes={
                    semconv.REWARD_PROGRESS: float(a["progress_now"]),
                    semconv.REWARD_SUCCESS: float(a["success_now"]),
                    semconv.REWARD_STALLED: bool(a["stalled"]),
                    semconv.REWARD_SUCCEEDED: bool(a["succeeded"]),
                    semconv.REWARD_FRAMES: int(a["frames_seen"]),
                    semconv.REWARD_TASK: task,
                    semconv.REWARD_CAMERA: self._camera_name,
                },
            ):
                pass

        def _on_query_task_progress(self, request: Any, response: Any) -> Any:
            """Service: assess task progress/success over a window."""
            task = request.task.strip() or self._default_task
            window_s = request.window_s if request.window_s > 0.0 else 1e9
            buf = self._buffers[self._primary_id]
            now_ns = self.get_clock().now().nanoseconds

            if buf.is_stale(now_ns) or not task:
                response.ok = False
                response.stale = True
                response.frames_seen = len(buf)
                self.get_logger().warning(
                    f"query_task_progress: stale/no-task (frames={len(buf)}, task={task!r})"
                )
                return response

            frames = buf.window(window_s)
            try:
                a = self._monitor.assess(frames, task)
            except Exception as exc:  # best-effort; never crash the service
                self.get_logger().warning(f"query_task_progress failed: {exc}")
                response.ok = False
                response.stale = False
                response.frames_seen = len(frames)
                return response

            response.ok = True
            response.progress_now = float(a["progress_now"])
            response.success_now = float(a["success_now"])
            response.progress_trend = float(a["progress_trend"])
            response.success_trend = float(a["success_trend"])
            response.stalled = bool(a["stalled"])
            response.succeeded = bool(a["succeeded"])
            response.frames_seen = int(a["frames_seen"])
            response.stale = False
            self._emit_score_span(a, task)
            self.get_logger().info(
                f"query_task_progress: progress={response.progress_now:.3f} "
                f"success={response.success_now:.3f} stalled={response.stalled} "
                f"succeeded={response.succeeded} frames={response.frames_seen}"
            )
            return response

        def _on_reward_active_task(self, msg: Any) -> None:
            """Gate + retarget scoring on the reasoner's active VLA instruction (2026-06-29).

            ``/openral/reward/active_task`` (std_msgs/String) carries the exact prompt
            the VLA is running while an ``execute_rskill`` goal is in flight, and empty
            on result. The continuous critic leg scores only while non-empty, and
            scores THAT instruction (not the collective default). Only subscribed when
            ``gate_scoring_on_execution``.
            """
            task = (msg.data or "").strip()
            self._active_task = task
            self._vla_active = bool(task)

        def _score_tick(self) -> None:
            """Timer: score a bounded recent window → reward.score span (+ optional CriticScore).

            Drives the dashboard rSkill card's live reward bar via the span it emits;
            the CriticScore publish is an optional add-on when the Tier-C
            producer leg is wired. Best-effort and advisory — skips quietly when gated
            off / no task / stale-or-empty buffer, and never crashes the timer on an
            assess error.
            """
            if not self._vla_active:
                return  # gated: no VLA executing → don't score an idle scene
            # Score the instruction the VLA is actually running (gated mode), not the
            # collective default — a single-object pick must be judged as that pick.
            task = self._active_task or self._default_task
            buf = self._buffers[self._primary_id]
            now_ns = self.get_clock().now().nanoseconds
            if not task or buf.is_stale(now_ns) or len(buf) == 0:
                return
            try:
                # Bounded RECENT window (not the full buffer) so the reward forward
                # fits beside the VLA on an 8 GB card — see score_window_s.
                a = self._monitor.assess(buf.window(self._score_window_s), task)
            except Exception as exc:  # best-effort; never crash the timer
                self.get_logger().debug(f"reward score_tick assess failed: {exc}")
                return
            self._emit_score_span(a, task)  # dashboard reward bar — ALWAYS
            # Tier-C CriticScore — only when the publisher leg is enabled.
            if self._critic_pub is not None:
                from openral_observability.propagation import current_traceparent

                score, threshold = critic_score_from_assessment(a, threshold=self._critic_threshold)
                msg = self._critic_msg_cls()
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.header.frame_id = self._critic_id
                msg.critic_id = self._critic_id
                msg.score = score
                msg.threshold = threshold
                msg.trace_id = current_traceparent() or ""
                self._critic_pub.publish(msg)
            # INFO: advisory operator signal at the heartbeat cadence, not a hot loop.
            self.get_logger().info(
                f"reward score: progress={float(a['progress_now']):.3f} "
                f"success={float(a['success_now']):.3f} "
                f"trend={float(a['progress_trend']):+.3f} frames={int(a['frames_seen'])}"
                + (" critic=on" if self._critic_pub is not None else "")
            )

        def destroy_node(self) -> None:
            """Release the reward backend before tearing down.

            Reward backends release their VLM and CUDA cache.
            """
            with contextlib.suppress(Exception):
                self._monitor.close()
            super().destroy_node()

    from openral_observability import configure_observability

    # Idempotent + no-op when OTEL_EXPORTER_OTLP_ENDPOINT is unset. Lets the
    # ``reward.score`` spans reach the dashboard's rSkill card.
    configure_observability(service_name="openral.reward_monitor")

    rclpy.init(args=args)
    node = RewardMonitorNode()
    try:
        try:
            rclpy.spin(node)
        except (KeyboardInterrupt, ExternalShutdownException):
            pass
        finally:
            node.destroy_node()
    finally:
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
