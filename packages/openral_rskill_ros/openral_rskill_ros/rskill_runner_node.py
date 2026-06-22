#!/usr/bin/env python3
"""ADR-0018 F1 — `rskill_runner_node` lifecycle node.

Owns the ``openral_msgs/action/ExecuteRskill`` action server and the
in-process :class:`openral_runner.HardwareRunner` mandated by ADR-0018
§F1. One node per robot.

Action-goal lifecycle (CLAUDE.md §6.4 + ADR-0018 §F1):

1. **goal_accept_cb** — resolve the rskill against a callable resolver
   passed at construction time (defaults to
   :func:`openral_rskill.rSkill.from_pretrained`); license + capability
   gate; envelope gate against the configured ``RobotDescription``.
   Reject with :class:`ROSCapabilityMismatch` /
   :class:`ROSConfigError` (CLAUDE.md §10) surfaced as
   ``ExecuteRskill.Result(success=False, failure_reason=<typed>)``.
2. **execute_cb** — instantiate / reuse a
   :class:`openral_runner.HardwareRunner` with the
   :class:`openral_runner.ROSPublishingHAL` sink and the shared
   :class:`openral_world_state.WorldStateAggregator`; run until the
   rskill signals completion or the goal's ``deadline_s`` lapses.
3. **cancel_cb** — drain the in-flight chunk (≤100 ms), then idle-hold
   the last commanded joint state. Runner stays ``active``, ready for
   the next goal.
4. **/openral/estop subscription** — defense in depth alongside
   ``safety_node`` (CLAUDE.md §1.5). Aborts the goal with
   ``failure_reason="safety_estop"`` and transitions to ``inactive``.

The action-goal path is the **only** way an external client triggers a
rskill; the legacy CLI / single-process invocation continues to exist as
a thin wrapper that sends a goal to this server.
"""

from __future__ import annotations

import contextlib
import math
import os
import sys
import threading
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import structlog

if TYPE_CHECKING:
    from openral_core.schemas import RobotDescription
    from openral_rskill.base import rSkillBase
    from openral_world_state import WorldStateAggregator

__all__ = ["RskillRunnerNode", "main"]

log = structlog.get_logger(__name__)


# Type for the injected skill resolver. Takes the goal's rskill_id /
# revision / prompt / prompt_metadata_json and returns a *configured +
# activated* rSkill ready to receive `step` calls. Production deployments
# pass `openral_rskill.rSkill.from_pretrained`-shaped callables;
# integration tests pass a real local-only resolver.
SkillResolver = Callable[..., "rSkillBase"]

# 100 ms cancel drain (ADR-0018 §F1).
_CANCEL_DRAIN_S = 0.1

# Runaway guard for the ADR-0053 MoveIt approach replay; real MoveGroup
# trajectories are far smaller (hundreds of points).
_MAX_APPROACH_WAYPOINTS = 100_000


def _commercial_deployment() -> bool:
    """Return whether the running deployment is commercial.

    Convention: ``OPENRAL_COMMERCIAL_DEPLOYMENT=1`` flips the flag on.
    Anything else (unset / ``0`` / empty) keeps the deployment in
    non-commercial mode. ADR-0018 §F1 calls this the second of two
    license gates (the first is ``ral skill install``).
    """
    return os.environ.get("OPENRAL_COMMERCIAL_DEPLOYMENT", "").strip() in (
        "1",
        "true",
        "True",
        "yes",
    )


try:
    import rclpy
    from openral_observability import log_lifecycle_errors
    from rclpy.action import ActionServer, CancelResponse, GoalResponse
    from rclpy.action.server import ServerGoalHandle
    from rclpy.executors import ExternalShutdownException
    from rclpy.lifecycle import LifecycleNode, LifecycleState, TransitionCallbackReturn

    _ROS2_AVAILABLE = True
except ImportError:
    _ROS2_AVAILABLE = False


if _ROS2_AVAILABLE:

    class RskillRunnerNode(LifecycleNode):  # type: ignore[misc]  # reason: rclpy untyped
        """ADR-0018 F1 — lifecycle node + ExecuteRskill action server.

        Args:
            node_name: ROS 2 node name (default ``"openral_rskill_runner"``).
            robot_description: The :class:`RobotDescription` for the
                robot this runner targets. Must already be loaded by the
                caller — the runner does not consult the registry.
            aggregator: The single shared
                :class:`WorldStateAggregator` constructed by
                :func:`compose_so100_runtime` (ADR-0018 §3). The runner
                calls ``.snapshot()`` against it in-process; never
                subscribes to ``/world_state`` over ROS.
            skill_resolver: Callable that resolves the goal's
                ``rskill_id`` / ``revision`` into a configured + active
                :class:`rSkillBase`. Defaults to a thin wrapper around
                :meth:`openral_rskill.rSkill.from_pretrained` ; tests
                inject a local-only resolver to avoid HF Hub access.
        """

        def __init__(
            self,
            *,
            node_name: str = "openral_skill_runner",
            robot_description: RobotDescription | None = None,
            aggregator: WorldStateAggregator | None = None,
            skill_resolver: SkillResolver | None = None,
        ) -> None:
            """Store references; opens no ROS resources until ``on_configure``."""
            super().__init__(node_name)
            self.declare_parameter("rate_hz", 30.0)
            self.declare_parameter("estop_topic", "/openral/estop")
            # Optional HAL service that snaps qpos to a manifest's
            # ``starting_pose`` before the first inference tick.
            # Empty string disables the call — useful for HALs that
            # don't expose the service, or for tests. The OpenArm e2e
            # launch overrides this to
            # ``/openral/openarm/reset_to_pose``.
            self.declare_parameter("reset_to_pose_service", "")
            # ADR-0053: MoveIt approach to the manifest ``starting_pose``. When
            # set, the runner dispatches this rSkill (the rskill-moveit-joints
            # MoveGroup wrapper) retargeted at the next skill's starting_pose,
            # preferred over ``reset_to_pose_service``; a failure ABORTS the goal
            # (vs. the best-effort snap). ``openral deploy sim`` / ``deploy run``
            # set it to ``rskills/rskill-moveit-joints``. Empty = legacy snap.
            self.declare_parameter("approach_skill_id", "")
            self.declare_parameter("approach_skill_revision", "main")
            self._description: RobotDescription | None = robot_description
            self._aggregator: WorldStateAggregator | None = aggregator
            self._skill_resolver: SkillResolver | None = skill_resolver

            self._hal: Any = None
            self._action_server: Any = None
            self._estop_sub: Any = None
            self._episode_pub: Any = None
            self._episode_counter: int = 0
            # ADR-0019 — 1-based inference-tick index stamped onto every
            # ActionChunk via the HAL's tick_index_getter (0 = no goal running).
            self._current_tick_index: int = 0
            self._heartbeat: Any = None
            # ADR-0027 — state-adapter wiring. Populated by
            # ``_init_tf_lookup`` at on_configure; ``None`` until then.
            self._tf_lookup: Any = None
            self._tf_buffer: Any = None
            self._tf_listener: Any = None

            # Per-goal state. Held across the goal's execute_cb.
            self._goal_lock = threading.RLock()
            self._active_goal: Any = None
            self._active_skill: Any = None
            self._active_skill_id: str = ""
            self._active_skill_revision: str = ""
            # ADR-0050 — single GPU-resident skill. The runner keeps exactly one
            # resolved skill loaded, keyed by (rskill_id, revision, prompt).
            # Dispatching a different key evicts (``shutdown()`` → frees VRAM)
            # the resident skill before loading the next; re-dispatching the
            # same key reuses it (no reload, no double-load).
            self._resident_skill: Any = None
            self._resident_key: tuple[str, str, str] = ("", "", "")
            self._chunks_published: int = 0
            self._estop_latched: bool = False
            self._cancel_requested: bool = False

        # ── Lifecycle ────────────────────────────────────────────────────────

        @log_lifecycle_errors
        def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:
            """Open the action server, /openral/estop sub, and heartbeat."""
            del state
            from openral_core.exceptions import ROSConfigError
            from openral_msgs.action import ExecuteRskill
            from openral_msgs.msg import Episode
            from openral_observability import DiagnosticsHeartbeat, Level
            from openral_runner import ROSPublishingHAL
            from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
            from std_msgs.msg import Empty

            if self._description is None:
                self.get_logger().error(
                    "robot_description was not supplied; rskill_runner_node "
                    "needs a RobotDescription to construct the HAL "
                    "(use compose_so100_runtime)."
                )
                return TransitionCallbackReturn.FAILURE
            if self._aggregator is None:
                self.get_logger().error(
                    "aggregator was not supplied; the runner consumes a "
                    "shared WorldStateAggregator via compose_so100_runtime."
                )
                return TransitionCallbackReturn.FAILURE
            # ADR-0027 — wire a tf2_ros buffer + lookup callable into
            # the runner so wrapped-task-space rSkills (state_contract.layout
            # ∈ WRAPPED_TASK_SPACE_LAYOUTS) can have their per-checkpoint
            # state vector assembled from live /tf at each step. Cheap to
            # spin up; only consulted by ``_PolicyAdapterSkill._step_impl``
            # when the manifest has a layout+bindings AND that layout's
            # assembler is registered. Joint-space rSkills (today's
            # default) are unaffected.
            self._tf_lookup = self._init_tf_lookup()

            if self._skill_resolver is None:
                # Default production resolver — captures `self` so that
                # wrapped-ROS rSkills (kind: ros_action / ros_service) can
                # build their ActionClient / service client on the same
                # lifecycle node that hosts the ExecuteRskill action
                # server. Empty `search_paths` keeps the VLA branch
                # falling through to `_default_skill_resolver`'s HF Hub
                # path; deployments that ship in-tree rSkill bundles
                # pass `skill_resolver=make_default_skill_resolver(node,
                # search_paths=[...])` from their compose factory.
                self._skill_resolver = make_default_skill_resolver(
                    self,
                    tf_lookup=self._tf_lookup,
                )

            # ADR-0018 F1 — ROSPublishingHAL replaces the motor-driving HAL.
            self._hal = ROSPublishingHAL(
                node=self,
                description=self._description,
                skill_id_getter=lambda: self._active_skill_id,
                skill_revision_getter=lambda: self._active_skill_revision,
                tick_index_getter=lambda: self._current_tick_index,
            )
            try:
                self._hal.connect()
            except ROSConfigError as exc:
                self.get_logger().error(f"ROSPublishingHAL.connect failed: {exc}")
                return TransitionCallbackReturn.FAILURE

            # ExecuteRskill action server.
            self._action_server = ActionServer(
                self,
                ExecuteRskill,
                "/openral/execute_rskill",
                execute_callback=self._execute_cb,
                goal_callback=self._goal_cb,
                cancel_callback=self._cancel_cb,
            )

            # /openral/estop defense in depth (CLAUDE.md §1.5). Subscribe
            # so the runner aborts even if safety_node already brakes the
            # HAL.
            estop_qos = QoSProfile(
                reliability=QoSReliabilityPolicy.RELIABLE,
                durability=QoSDurabilityPolicy.VOLATILE,
                depth=10,
            )
            estop_topic: str = self.get_parameter("estop_topic").get_parameter_value().string_value
            self._estop_sub = self.create_subscription(
                Empty, estop_topic, self._on_estop, estop_qos
            )

            # ADR-0019 — Episode boundary markers on the bus. A dataset
            # recorder (openral_runner.DatasetRecorderBridge, attached by
            # compose_runtime when --dataset-out is set) and `openral record`
            # both consume these to segment a deploy session into episodes.
            # Sparse events → RELIABLE / VOLATILE / KEEP_LAST=10.
            episode_qos = QoSProfile(
                reliability=QoSReliabilityPolicy.RELIABLE,
                durability=QoSDurabilityPolicy.VOLATILE,
                depth=10,
            )
            self._episode_pub = self.create_publisher(Episode, "/openral/episode", episode_qos)

            # F8 heartbeat.
            robot_name = self._description.name

            def _status() -> tuple[int, str, dict[str, str]]:
                if self._estop_latched:
                    return (
                        Level.ERROR,
                        "estop latched",
                        {
                            "robot": robot_name,
                            "chunks_published": str(self._chunks_published),
                        },
                    )
                if self._active_goal is not None:
                    return (
                        Level.OK,
                        "goal active",
                        {
                            "robot": robot_name,
                            "rskill_id": self._active_skill_id,
                            "chunks_published": str(self._chunks_published),
                        },
                    )
                return (
                    Level.OK,
                    "idle",
                    {
                        "robot": robot_name,
                        "chunks_published": str(self._chunks_published),
                    },
                )

            self._heartbeat = DiagnosticsHeartbeat(
                self,
                hardware_id=f"openral_rskill_runner:{robot_name}",
                component_name="openral_rskill_runner",
                status_fn=_status,
            )
            self._heartbeat.create_publisher()
            self.get_logger().info(f"rskill_runner_node configured (robot={robot_name}).")
            return TransitionCallbackReturn.SUCCESS

        def _init_tf_lookup(self) -> Any:
            """Build the ADR-0027 ``TfLookup`` callable from tf2_ros.

            Subscribes to ``/tf`` + ``/tf_static`` once at on_configure
            (no extra subscription per skill), returns a closure that
            converts ``tf2_ros.Buffer.lookup_transform`` results into
            :class:`openral_state_adapter.TransformView` instances.
            The buffer + listener stay alive for the node's lifetime;
            ``on_cleanup`` releases them.

            Returns:
                A ``TfLookup``-shaped callable, or ``None`` when
                ``tf2_ros`` isn't importable (non-ROS unit-test paths
                still build the node via ``compose_runtime``).
            """
            try:
                import tf2_ros  # type: ignore[import-untyped]
            except ImportError:
                return None
            import rclpy.time  # type: ignore[import-untyped]
            from openral_state_adapter import TransformView  # type: ignore[import-untyped]

            self._tf_buffer = tf2_ros.Buffer()
            self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

            def _lookup(target_frame: str, source_frame: str) -> TransformView:
                # rclpy.time.Time() = "latest available" — matches
                # ``robot_state_publisher`` + slam_toolbox's emission cadence
                # without picking a specific stamp. Per-call lookup; the
                # buffer handles caching + interpolation.
                tf = self._tf_buffer.lookup_transform(
                    target_frame,
                    source_frame,
                    rclpy.time.Time(),
                )
                t = tf.transform.translation
                r = tf.transform.rotation
                return TransformView(
                    position=(float(t.x), float(t.y), float(t.z)),
                    quaternion_xyzw=(float(r.x), float(r.y), float(r.z), float(r.w)),
                )

            return _lookup

        @log_lifecycle_errors
        def on_activate(self, state: LifecycleState) -> TransitionCallbackReturn:
            """Start the diagnostics heartbeat."""
            del state
            if self._heartbeat is not None:
                self._heartbeat.start()
            return TransitionCallbackReturn.SUCCESS

        def on_deactivate(self, state: LifecycleState) -> TransitionCallbackReturn:
            """Stop the heartbeat. Cancels any in-flight goal."""
            del state
            with self._goal_lock:
                self._cancel_requested = True
            if self._heartbeat is not None:
                self._heartbeat.stop()
            return TransitionCallbackReturn.SUCCESS

        def on_cleanup(self, state: LifecycleState) -> TransitionCallbackReturn:
            """Release ROS resources."""
            del state
            # ADR-0050 — free the GPU-resident skill's VRAM on teardown.
            self._evict_resident_skill()
            if self._heartbeat is not None:
                self._heartbeat.destroy()
                self._heartbeat = None
            if self._estop_sub is not None:
                self.destroy_subscription(self._estop_sub)
                self._estop_sub = None
            if self._episode_pub is not None:
                self.destroy_publisher(self._episode_pub)
                self._episode_pub = None
            if self._action_server is not None:
                self._action_server.destroy()
                self._action_server = None
            if self._hal is not None:
                self._hal.disconnect()
                self._hal = None
            # Release tf2 buffer + listener so a re-configure rebuilds
            # them cleanly. The TransformListener owns a subscription
            # to /tf + /tf_static; dropping the reference is enough.
            self._tf_lookup = None
            self._tf_buffer = None  # type: ignore[assignment]
            self._tf_listener = None  # type: ignore[assignment]
            return TransitionCallbackReturn.SUCCESS

        def on_shutdown(self, state: LifecycleState) -> TransitionCallbackReturn:
            """Force cleanup."""
            return self.on_cleanup(state)

        def _acquire_skill(
            self,
            *,
            rskill_id: str,
            revision: str,
            prompt: str,
            prompt_metadata_json: str,
            goal_params_json: str,
        ) -> rSkillBase:
            """ADR-0050 — return the GPU-resident skill for this dispatch key.

            Keyed by ``(rskill_id, revision, prompt)``: a differing key evicts
            the resident skill (``shutdown()`` → frees VRAM) before loading the
            next; an exact match reuses it (no reload, no double-load); a miss
            resolves + caches. Resolve failures propagate to the caller's abort
            path unchanged.
            """
            req_key = (rskill_id, revision, prompt)
            if self._resident_skill is not None and self._resident_key != req_key:
                self._evict_resident_skill()
            if self._resident_skill is not None and self._resident_key == req_key:
                return cast("rSkillBase", self._resident_skill)
            skill = self._resolve_and_check_skill(
                rskill_id=rskill_id,
                revision=revision,
                prompt=prompt,
                prompt_metadata_json=prompt_metadata_json,
                goal_params_json=goal_params_json,
            )
            self._resident_skill = skill
            self._resident_key = req_key
            return skill

        def _evict_resident_skill(self) -> None:
            """ADR-0050 — shut down the GPU-resident skill, freeing its VRAM.

            ``shutdown()`` drives ``on_unload_weights`` per the rSkill lifecycle
            contract. Best-effort: a resolver that returns a non-lifecycle handle
            (the HF-Hub production path) or a skill whose teardown raises must
            not block the next dispatch.
            """
            skill = self._resident_skill
            self._resident_skill = None
            self._resident_key = ("", "", "")
            if skill is None:
                return
            shutdown = getattr(skill, "shutdown", None)
            if not callable(shutdown):
                return
            try:
                shutdown()
            except Exception as exc:  # reason: eviction must never block the next goal
                self.get_logger().warning(f"rskill_runner.evict_failed: {exc!s}")

        # ── Action callbacks ─────────────────────────────────────────────────

        def _goal_cb(self, _goal_request: Any) -> Any:
            """Accept every well-formed goal; rejection happens in execute_cb.

            The action-server contract makes goal_callback a fast yes/no;
            heavy work (skill download, capability check) belongs in the
            executor where we can surface a typed failure_reason on the
            Result message instead of refusing to even start the goal.
            """
            if self._estop_latched:
                return GoalResponse.REJECT
            return GoalResponse.ACCEPT

        def _cancel_cb(self, _goal_handle: Any) -> Any:
            """Acknowledge a cancel; the executor drains."""
            with self._goal_lock:
                self._cancel_requested = True
            return CancelResponse.ACCEPT

        def _execute_cb(self, goal_handle: ServerGoalHandle) -> Any:  # noqa: PLR0915  # reason: sequential goal-lifecycle handler — acquire → starting-pose → run, each with a typed failure branch that sets failure_reason + aborts; splitting the linear flow hurts readability
            """Run a single ExecuteRskill goal end-to-end (synchronously)."""
            from openral_core.exceptions import (
                ROSCapabilityMismatch,
                ROSConfigError,
                ROSError,
                ROSEStopRequested,
                ROSSafetyViolation,
            )
            from openral_msgs.action import ExecuteRskill
            from openral_observability import rskill_span

            req = goal_handle.request
            rskill_id = req.rskill_id
            revision = req.revision
            deadline_s = float(req.deadline_s)
            with self._goal_lock:
                self._active_goal = goal_handle
                self._active_skill_id = rskill_id
                self._active_skill_revision = revision
                self._chunks_published = 0
                self._cancel_requested = False

            result = ExecuteRskill.Result()
            with rskill_span("rskill.execute", rskill_id=rskill_id) as span:
                # Stamp the trace id on the result before any early-return
                # path so the caller can correlate even on failure.
                from openral_observability import propagation

                result.trace_id = propagation.current_traceparent() or ""

                try:
                    # ADR-0050 — single GPU-resident skill: evict-on-switch,
                    # reuse-on-match, else resolve + cache (see _acquire_skill).
                    skill = self._acquire_skill(
                        rskill_id=rskill_id,
                        revision=revision,
                        prompt=req.prompt,
                        prompt_metadata_json=req.prompt_metadata_json,
                        # ADR-0026 — empty string when the goal carries no
                        # structured params (today's default; PR3 wires
                        # the LLM to populate it).
                        goal_params_json=getattr(req, "goal_params_json", ""),
                    )
                except (ROSConfigError, ROSCapabilityMismatch) as exc:
                    span.record_exception(exc)
                    self.get_logger().error(
                        f"rskill_runner.goal_rejected: kind={type(exc).__name__} reason={exc!s}"
                    )
                    result.success = False
                    result.failure_reason = f"{type(exc).__name__}: {exc!s}"
                    goal_handle.abort()
                    self._reset_active_goal()
                    return result

                self._active_skill = skill

                # Move the HAL to the manifest's in-distribution ``starting_pose``
                # before the first inference tick (without this a checkpoint
                # trained from a specific pose sees an out-of-distribution state
                # and drifts joints into their stops). ADR-0053: prefer the MoveIt
                # approach skill (collision-free MoveGroup plan to starting_pose) —
                # a failure there is FATAL (abort the goal, never start the policy
                # from an unreachable/colliding state). The legacy ResetToPose snap
                # stays best-effort (a failure only warns).
                # Record the wall-clock instant just before the starting-pose
                # reset. /joint_states is timer-published (~30 Hz, ADR-0049) and
                # the world_state node re-stamps each JointState with its
                # wall-clock ARRIVAL time, so the aggregator cache the first
                # inference reads can still hold the PRE-reset pose for up to one
                # publish period + DDS + cross-node latency after ResetToPose
                # returns → the policy's first observation would be stale (OOD →
                # self-collision wedge). Gate the first tick on a joint state
                # stamped at/after this instant.
                reset_wall_ns = time.time_ns()
                if self._apply_starting_pose_or_abort(skill, goal_handle, result):
                    return result
                self._wait_for_post_reset_joint_state(skill, reset_wall_ns)

                # ADR-0019 — frame the episode on the bus so a dataset
                # recorder (DatasetRecorderBridge) / `openral record` can
                # segment a deploy session. PHASE_END is published in the
                # ``finally`` so every exit path (estop / error / cancel /
                # success / safety re-raise) closes the episode with the
                # resolved success flag.
                episode_task = req.prompt or rskill_id
                self._publish_episode_start(task_string=episode_task)
                try:
                    try:
                        self._run_until_done_or_deadline(
                            goal_handle=goal_handle,
                            skill=skill,
                            deadline_s=deadline_s,
                        )
                    except ROSEStopRequested as exc:
                        # Latched by /openral/estop or HAL.estop().
                        span.record_exception(exc)
                        result.success = False
                        result.failure_reason = f"safety_estop:{exc!s}"
                        goal_handle.abort()
                        self._reset_active_goal()
                        return result
                    except ROSSafetyViolation:
                        # Never convert a safety violation into a soft failure_reason
                        # (CLAUDE.md §1) — let it reach the safety-supervisor boundary.
                        raise
                    except ROSError as exc:
                        # A typed runtime failure during execution (e.g. a wrapped
                        # ros_action skill whose goal could not be built from
                        # malformed goal_params_json, an inference timeout, a planning
                        # error). Previously these escaped the callback and the goal
                        # aborted with an EMPTY failure_reason, so the reasoner could
                        # not replan. Surface the typed reason so the replanning ladder
                        # can act on it.
                        span.record_exception(exc)
                        _kind = type(exc).__name__
                        self.get_logger().error(
                            f"rskill_runner.execute_failed: kind={_kind} reason={exc!s}"
                        )
                        result.success = False
                        result.failure_reason = f"{_kind}: {exc!s}"
                        goal_handle.abort()
                        self._reset_active_goal()
                        return result

                    # Honour cancel — drain then idle-hold (ADR-0018 §F1).
                    if self._cancel_requested or goal_handle.is_cancel_requested:
                        self._drain_and_idle_hold(skill)
                        result.success = False
                        result.failure_reason = "cancelled"
                        goal_handle.canceled()
                        self._reset_active_goal()
                        return result

                    result.success = True
                    result.failure_reason = ""
                    goal_handle.succeed()
                    self._reset_active_goal()
                    return result
                finally:
                    self._publish_episode_end(
                        task_string=episode_task, success=bool(result.success)
                    )

        # ── Internal helpers ─────────────────────────────────────────────────

        def _resolve_and_check_skill(
            self,
            *,
            rskill_id: str,
            revision: str,
            prompt: str,
            prompt_metadata_json: str,
            goal_params_json: str = "",
        ) -> rSkillBase:
            """Resolve the skill via the configured resolver + license / capability gates."""
            from openral_core.exceptions import ROSCapabilityMismatch, ROSConfigError

            assert self._skill_resolver is not None  # invariant set in on_configure
            # `ros_node=self` lets resolvers that build ROS-wrapped skills
            # (kind: ros_action / ros_service) create ActionClient / service
            # client handles on the same lifecycle node that hosts the
            # ExecuteRskill action server. Resolvers that don't need it (the
            # legacy local + HF Hub VLA paths) accept the kwarg and ignore
            # it via ``**kwargs`` / ``del`` — preserving every existing
            # injected fake-resolver signature.
            skill = self._skill_resolver(
                rskill_id=rskill_id,
                revision=revision,
                prompt=prompt,
                prompt_metadata_json=prompt_metadata_json,
                goal_params_json=goal_params_json,
                description=self._description,
                commercial_deployment=_commercial_deployment(),
                ros_node=self,
            )

            # Embodiment gate — skill must declare this robot's
            # embodiment tags. Skip when the resolver returned a Skill
            # with no declared tags (test harness path).
            assert self._description is not None  # invariant from on_configure
            tags = list(getattr(skill, "info", None).embodiment_tags or [])
            if tags:
                allowed = set(self._description.capabilities.embodiment_tags or [])
                if allowed and not any(t in allowed for t in tags):
                    raise ROSCapabilityMismatch(
                        f"skill embodiment_tags {tags!r} disjoint from "
                        f"robot embodiment_tags {sorted(allowed)!r}"
                    )

            # Hand-validate that the skill is in a runnable state. The
            # resolver is expected to have driven `configure` + `activate`;
            # anything else is a contract violation we surface as a typed
            # error so the goal fails fast.
            from openral_core.schemas import RSkillState

            if skill.info.state is not RSkillState.ACTIVE:
                raise ROSConfigError(
                    f"resolver returned skill in state {skill.info.state!r}; "
                    "expected ACTIVE — drive configure() + activate() first"
                )
            return skill

        def _resolve_inference_labels(self, skill: rSkillBase) -> tuple[str, str | None]:
            """Resolve the inference engine + device labels for the chunk span.

            Engine comes from the manifest runtime (torch / onnx / tensorrt);
            device from the policy adapter (lerobot ``.device`` convention).
            Both best-effort — a missing attribute renders "—" on the dashboard.
            """
            _manifest = getattr(skill, "manifest", None)
            _runtime = getattr(_manifest, "runtime", None)
            engine = str(getattr(_runtime, "value", _runtime) or "") or "torch"
            _adapter = getattr(skill, "_adapter", None)
            _device = getattr(_adapter, "device", None) or getattr(skill, "device", None)
            return engine, (str(_device) if _device is not None else None)

        def _run_until_done_or_deadline(
            self,
            *,
            goal_handle: ServerGoalHandle,
            skill: rSkillBase,
            deadline_s: float,
        ) -> None:
            """Drive ``skill.step(snapshot)`` until done / cancelled / deadline.

            Mirrors the inner loop of
            :meth:`openral_runner.HardwareRunner._tick_impl` but trims
            back the observability + per-stage timing to keep this PR
            focused on the topic-shape contract. The full HardwareRunner
            integration lands when F2 ships ``WorldStateStamped`` so the
            in-process snapshot has the typed staleness array; for now
            the lightweight loop here is sufficient to publish chunks on
            ``/openral/candidate_action``.
            """
            from openral_core.exceptions import ROSEStopRequested, ROSRskillGoalSatisfied
            from openral_msgs.action import ExecuteRskill
            from openral_observability import inference_span

            assert self._aggregator is not None  # invariant set in on_configure
            assert self._hal is not None

            rate_hz: float = self.get_parameter("rate_hz").get_parameter_value().double_value
            period_s = 1.0 / max(rate_hz, 1.0)
            start = time.monotonic()
            chunk_index = 0
            skill_info = getattr(skill, "info", None)
            skill_role = str(getattr(skill_info, "role", "")) if skill_info is not None else ""
            # Resolve inference engine + device once so the dashboard's Inference
            # card + the `inference.engine`/`inference.device` Identity latches
            # populate (best-effort — omitted attrs render "—").
            inference_engine, inference_device = self._resolve_inference_labels(skill)
            while True:
                if self._estop_latched:
                    raise ROSEStopRequested("/openral/estop received during goal")
                if self._cancel_requested or goal_handle.is_cancel_requested:
                    return
                if deadline_s > 0.0 and time.monotonic() - start > deadline_s:
                    return

                snapshot = self._aggregator.snapshot()
                # Wrap inference so the Inference card and the rskill.id /
                # rskill.role identity latches both populate. The store's
                # `_HEADLINE_FAMILIES` maps `rskill.chunk_inference`
                # (semconv.SPAN_RSKILL_CHUNK_INFERENCE) → the
                # Inference card; `_IDENTITY_KEYS` latches `rskill.id` /
                # `rskill.role` regardless of which span carries them.
                # `inference_span(**attrs)` prefixes kwargs with
                # ``inference.`` — set the literal rskill.* keys directly
                # via `set_attribute` so they keep their dotted names.
                _inf_attrs: dict[str, str] = {"engine": inference_engine}
                if inference_device:
                    _inf_attrs["device"] = inference_device
                with inference_span(chunk_index=chunk_index, **_inf_attrs) as inf_span:
                    if self._active_skill_id:
                        inf_span.set_attribute("rskill.id", self._active_skill_id)
                    if skill_role:
                        inf_span.set_attribute("rskill.role", skill_role)
                    try:
                        step_result = skill.step(snapshot)
                    except ROSRskillGoalSatisfied as completion:
                        # Wrapped-ROS rSkills (kind: ros_action / ros_service)
                        # raise this AFTER the last waypoint has been emitted
                        # (trajectory mode) or AFTER the wrapped server's
                        # result has been awaited (result-only mode, e.g.
                        # Nav2). It is a success signal, not an error — break
                        # the loop and let the caller close the goal with
                        # success=True. Logged so traces show why the loop
                        # exited.
                        self.get_logger().info(
                            f"rskill_runner.rskill_goal_satisfied: {completion!s}"
                        )
                        inf_span.set_attribute("rskill.completion", "goal_satisfied")
                        return
                    # ADR-0028b — ``step()`` may return a single ``Action``
                    # (legacy single-surface rskills) or ``list[Action]``
                    # (slot-dispatched multi-surface output, e.g. the
                    # RoboCasa pi0.5 cartesian-delta + gripper + body-twist
                    # split). Normalise to a list and emit one ActionChunk
                    # per entry; they inherit the same OTel trace_id by
                    # construction (the span context is set in
                    # ``inference_span``, which envelops every
                    # ``send_action`` call below).
                    actions = list(step_result) if isinstance(step_result, list) else [step_result]
                    inf_span.set_attribute("inference.actions_emitted", len(actions))
                    if actions and actions[0].horizon:
                        inf_span.set_attribute("inference.chunk_size", int(actions[0].horizon))
                # ADR-0019 — stamp every slot chunk of THIS tick with the same
                # 1-based tick index (read by ROSPublishingHAL via its
                # tick_index_getter) so the dataset recorder groups them.
                self._current_tick_index = chunk_index + 1
                for action in actions:
                    self._hal.send_action(action)
                    self._chunks_published += 1
                chunk_index += 1

                feedback = ExecuteRskill.Feedback()
                feedback.progress = (
                    min((time.monotonic() - start) / deadline_s, 1.0) if deadline_s > 0.0 else 0.0
                )
                feedback.state = "executing"
                feedback.chunk_index = chunk_index
                feedback.chunks_total = 0  # unknown — rskills are open-loop
                goal_handle.publish_feedback(feedback)

                # Sleep until the next tick boundary — the loop's
                # rate-limiting is intentionally minimal here; production
                # use composes the full HardwareRunner via the compose
                # factory when F2's typed WorldStateStamped lands.
                time.sleep(period_s)

        def _apply_starting_pose_or_abort(self, skill: Any, goal_handle: Any, result: Any) -> bool:
            """Move to ``starting_pose``; abort the goal on a fatal failure (ADR-0053).

            Returns ``True`` when the goal was aborted (the caller returns
            ``result`` immediately), ``False`` to proceed with execution.
            """
            reason = self._apply_starting_pose(skill)
            if reason is None:
                return False
            self.get_logger().error(f"rskill_runner.approach_failed: {reason}")
            result.success = False
            result.failure_reason = reason
            goal_handle.abort()
            self._reset_active_goal()
            return True

        def _wait_for_post_reset_joint_state(self, skill: Any, reset_wall_ns: int) -> None:
            """Block until the aggregator's joint state is newer than the reset.

            Closes the cross-node staleness race after a ``starting_pose`` reset:
            the HAL refreshes its proprio snapshot inside the ResetToPose handler,
            but ``/joint_states`` is only re-published on the next publisher-thread
            tick (~30 Hz, ADR-0049) and must transit DDS + the world_state node's
            ``_on_joint_state`` callback before it lands in the
            ``WorldStateAggregator`` cache the first inference reads.
            ``WorldState.joint_state.stamp_ns`` is the world_state node's
            wall-clock ARRIVAL time, so any value ``>= reset_wall_ns`` was
            published from the reset snapshot. No-op unless a ``starting_pose``
            reset actually fired; bounded by a short deadline → best-effort
            (mirrors the reset's own posture; never wedges the goal).
            """
            manifest = getattr(skill, "manifest", None)
            pose = getattr(manifest, "starting_pose", None) if manifest is not None else None
            if not pose:
                return
            if not self.get_parameter("reset_to_pose_service").get_parameter_value().string_value:
                return
            if self._aggregator is None:
                return
            deadline = time.monotonic() + 1.0
            while time.monotonic() < deadline:
                js = self._aggregator.snapshot().joint_state
                if js is not None and int(js.stamp_ns) >= reset_wall_ns:
                    self.get_logger().info(
                        "rskill_runner.post_reset_joint_state_fresh "
                        f"(age_ms={(time.time_ns() - int(js.stamp_ns)) / 1e6:.1f})"
                    )
                    return
                time.sleep(0.005)
            self.get_logger().warning(
                "rskill_runner.post_reset_joint_state_timeout: no joint state newer "
                "than the reset within 1.0 s; proceeding with current cache."
            )

        def _apply_starting_pose(self, skill: Any) -> str | None:
            """Move the HAL to the manifest ``starting_pose`` (ADR-0053 dispatch).

            Prefers the MoveIt approach skill (``approach_skill_id``) over the
            legacy ``ResetToPose`` snap. Returns a failure reason **only** when a
            fatal (approach) attempt failed — the caller then aborts the
            ExecuteSkill goal. The best-effort reset path always returns ``None``
            (a failure only warns), preserving pre-ADR-0053 behaviour.
            """
            from openral_rskill_ros._starting_pose import resolve_starting_pose_action

            manifest = getattr(skill, "manifest", None)
            starting_pose = (
                getattr(manifest, "starting_pose", None) if manifest is not None else None
            )
            action = resolve_starting_pose_action(
                approach_skill_id=self.get_parameter("approach_skill_id")
                .get_parameter_value()
                .string_value,
                reset_to_pose_service=self.get_parameter("reset_to_pose_service")
                .get_parameter_value()
                .string_value,
                starting_pose=starting_pose,
            )
            if action.mode == "approach":
                return self._dispatch_moveit_approach(action.pose)
            if action.mode == "reset":
                self._maybe_reset_hal_to_starting_pose(skill)
            return None

        def _dispatch_moveit_approach(self, pose: list[float]) -> str | None:
            """Run the MoveIt approach rSkill retargeted at ``pose`` (ADR-0053 §D2).

            Resolves the ``approach_skill_id`` rSkill (``rskill-moveit-joints``)
            with a ``goal_params_json`` that overrides the MoveGroup goal's
            ``joint`` block positions with ``pose`` (the next skill's ``starting_pose``),
            then runs it through the standard skill loop — ``ROSActionRskill`` sends
            the MoveGroup goal, MoveIt plans a collision-free trajectory (self +
            planning-scene/world), and each waypoint replays through
            ``/openral/candidate_action`` (the kernel checks every step).

            Returns ``None`` on success; otherwise a typed failure reason the caller
            surfaces as the aborted goal's ``failure_reason`` — the policy never
            starts from an unreachable / colliding state (ADR-0053 §D4).
            """
            from openral_core.exceptions import ROSError
            from openral_rskill.loader import load_rskill_manifest

            from openral_rskill_ros._starting_pose import (
                joint_names_from_goal_json,
                moveit_joint_goal_override,
            )

            skill_id = self.get_parameter("approach_skill_id").get_parameter_value().string_value
            revision = (
                self.get_parameter("approach_skill_revision").get_parameter_value().string_value
                or "main"
            )
            # Build the retarget override from the approach manifest's declared
            # planning-group joint names + the target starting_pose.
            try:
                manifest = load_rskill_manifest(skill_id)
                integration = manifest.ros_integration
                if integration is None:
                    return (
                        f"ROSConfigError: approach skill {skill_id!r} declares no "
                        "ros_integration (expected a kind: ros_action MoveGroup wrapper)."
                    )
                joint_names = joint_names_from_goal_json(integration.default_goal_json)
                goal_params_json = moveit_joint_goal_override(joint_names, pose)
            except (ROSError, ValueError) as exc:
                return f"ROSConfigError: cannot build MoveIt approach goal: {exc}"

            approach: rSkillBase | None = None
            try:
                approach = self._resolve_and_check_skill(
                    rskill_id=skill_id,
                    revision=revision,
                    prompt="approach_to_starting_pose",
                    prompt_metadata_json="",
                    goal_params_json=goal_params_json,
                )
                self._run_approach_skill(approach)
            except ROSError as exc:
                return f"ROSPlanningError: MoveIt approach failed: {type(exc).__name__}: {exc!s}"
            finally:
                if approach is not None:
                    with contextlib.suppress(Exception):
                        approach.shutdown()
            self.get_logger().info(
                f"MoveIt approach reached {len(pose)}-D starting_pose via {skill_id!r}."
            )
            return None

        def _run_approach_skill(self, approach: rSkillBase) -> None:
            """Tick the approach skill to completion, publishing each waypoint.

            Mirrors the inner loop of :meth:`_run_until_done_or_deadline` but for
            the pre-skill approach: ``ROSActionRskill`` plans on the first
            ``step()`` (blocking, with its own result deadline) then emits one
            ``JOINT_POSITION`` waypoint per ``step()`` — each replayed by
            ``self._hal.send_action`` onto ``/openral/candidate_action`` — and
            raises ``ROSRskillGoalSatisfied`` once the planned trajectory is
            exhausted.

            Raises:
                ROSRuntimeError: If the trajectory exceeds ``_MAX_APPROACH_WAYPOINTS``
                    (a runaway guard; real MoveIt trajectories are far smaller).
            """
            from openral_core.exceptions import ROSRskillGoalSatisfied, ROSRuntimeError

            assert self._hal is not None
            assert self._aggregator is not None
            try:
                for _ in range(_MAX_APPROACH_WAYPOINTS):
                    snapshot = self._aggregator.snapshot()
                    step_result = approach.step(snapshot)
                    actions = list(step_result) if isinstance(step_result, list) else [step_result]
                    for action in actions:
                        self._hal.send_action(action)
            except ROSRskillGoalSatisfied:
                return
            raise ROSRuntimeError(
                f"MoveIt approach exceeded {_MAX_APPROACH_WAYPOINTS} waypoints without "
                "completing — aborting (runaway guard)."
            )

        def _maybe_reset_hal_to_starting_pose(self, skill: Any) -> None:
            """Call the HAL's ResetToPose service if the manifest declares one.

            The service name is configurable via the
            ``reset_to_pose_service`` ROS parameter (defaults to
            empty, i.e. disabled). The OpenArm e2e launch sets it to
            ``/openral/openarm/reset_to_pose``; HALs that don't expose
            a pose-reset service leave it empty. Likewise, a manifest
            with no ``starting_pose`` (or one whose length doesn't
            match the robot's DoF count) is a no-op — only an explicit
            maintainer-declared pose triggers a reset.
            """
            service_name: str = (
                self.get_parameter("reset_to_pose_service").get_parameter_value().string_value
            )
            if not service_name:
                return
            manifest = getattr(skill, "manifest", None)
            pose = getattr(manifest, "starting_pose", None) if manifest is not None else None
            if not pose:
                return
            try:
                from openral_msgs.srv import ResetToPose
            except ImportError as exc:  # reason: build mismatch, surface and continue
                self.get_logger().warning(
                    f"ResetToPose IDL not importable ({exc!s}); skipping pose reset."
                )
                return

            client = self.create_client(ResetToPose, service_name)
            try:
                if not client.wait_for_service(timeout_sec=1.0):
                    self.get_logger().info(
                        f"ResetToPose service {service_name!r} not available; "
                        "HAL likely doesn't support pose reset. Continuing."
                    )
                    return
                req = ResetToPose.Request()
                req.pose = [float(v) for v in pose]
                future = client.call_async(req)
                # Block until the service responds. The action server's
                # execute_cb already runs on a worker thread, so the
                # node's main rclpy spin keeps servicing callbacks.
                deadline = time.monotonic() + 5.0
                while not future.done() and time.monotonic() < deadline:
                    time.sleep(0.02)
                if not future.done():
                    self.get_logger().warning(
                        "ResetToPose call timed out after 5 s; continuing with current HAL state."
                    )
                    return
                resp = future.result()
                if resp is None or not resp.success:
                    reason = "unknown" if resp is None else resp.failure_reason
                    self.get_logger().warning(
                        f"ResetToPose failed: {reason}; continuing with current HAL state."
                    )
                else:
                    self.get_logger().info(
                        f"ResetToPose applied {len(req.pose)}-D manifest starting_pose."
                    )
            finally:
                self.destroy_client(client)

        def _drain_and_idle_hold(self, skill: rSkillBase) -> None:
            """Honour ADR-0018 §F1 cancel semantics: ≤100 ms drain + idle-hold.

            For Day-1 we wait the configured drain window with a short
            sleep — the runner does not currently re-publish a hold
            chunk because the HAL lifecycle node (commit 3 — added in
            this PR's HAL consumer wiring) latches the last `safe_action`
            and brakes on `/openral/estop` independently. When the C++
            kernel lands and we route through it, this method republishes
            a single ``ActionChunk`` with the last commanded row.
            """
            del skill
            time.sleep(_CANCEL_DRAIN_S)

        def _publish_episode_start(self, *, task_string: str) -> None:
            """Publish an Episode(PHASE_START) marker (ADR-0019). No-op if unconfigured."""
            self._publish_episode_marker(phase=0, task_string=task_string, success=False)

        def _publish_episode_end(self, *, task_string: str, success: bool) -> None:
            """Publish an Episode(PHASE_END) marker (ADR-0019). No-op if unconfigured."""
            self._publish_episode_marker(phase=1, task_string=task_string, success=success)

        def _publish_episode_marker(self, *, phase: int, task_string: str, success: bool) -> None:
            """Publish one ``openral_msgs/Episode`` boundary marker; bump idx on END."""
            if self._episode_pub is None:
                return
            from openral_msgs.msg import Episode

            msg = Episode()
            msg.stamp = self.get_clock().now().to_msg()
            msg.episode_idx = self._episode_counter
            msg.task_string = task_string
            msg.phase = int(phase)
            msg.success = bool(success)
            self._episode_pub.publish(msg)
            if int(phase) == Episode.PHASE_END:
                self._episode_counter += 1

        def _reset_active_goal(self) -> None:
            """Clear the per-goal state under the lock."""
            with self._goal_lock:
                self._active_goal = None
                self._active_skill = None
                self._active_skill_id = ""
                self._active_skill_revision = ""
                self._cancel_requested = False
                self._current_tick_index = 0

        def _on_estop(self, _msg: object) -> None:
            """``/openral/estop`` callback: latch + abort the active goal."""
            self._estop_latched = True
            self.get_logger().error(
                "rskill_runner.estop_received; "
                f"aborting in-flight goal (rskill_id={self._active_skill_id!r})"
            )


def _default_skill_resolver(
    *,
    rskill_id: str,
    revision: str,
    prompt: str,
    prompt_metadata_json: str,
    goal_params_json: str = "",
    description: RobotDescription | None,
    commercial_deployment: bool,
    ros_node: Any = None,
) -> rSkillBase:
    """Production resolver: pull from HF Hub via ``rSkill.from_pretrained``.

    Kept as a module-level callable so tests can swap it via
    ``RskillRunnerNode(..., skill_resolver=local_resolver)`` without
    monkey-patching.

    ``ros_node`` is accepted but unused on this path: HF-Hub VLA skills
    do not need a node handle. Wrapped-ROS skills go through
    :func:`make_default_skill_resolver` instead, which captures the
    runner's lifecycle node so ``ROSActionRskill`` can build action /
    service clients on it.

    ``goal_params_json`` (ADR-0026) is accepted but unused — VLA
    skills consume the ``prompt`` as their structured input.
    """
    del prompt, prompt_metadata_json, goal_params_json, description, ros_node
    from openral_rskill.loader import rSkill

    handle = rSkill.from_pretrained(
        repo_id=rskill_id,
        revision=revision or None,
        commercial_use=commercial_deployment,
    )
    # ``rSkill.from_pretrained`` returns a packaging-format handle, not
    # the runtime ``rSkillBase``. Production use will route through the
    # loader's instantiation helpers (the ADR-0018 §F1 design defers the
    # exact runtime-binding to a follow-up that lands alongside the
    # reasoner — F4 — when the loader-to-runtime seam is finalised).
    # For now we expose the handle as the resolver's return value and
    # let the production deployment configuration provide a richer
    # wrapper.
    return handle  # type: ignore[return-value]  # see comment above


def _ros_action_adapter_cls(builder: str | None) -> type:
    """Map a manifest ``goal_builder`` to its ``ROSActionRskill`` subclass.

    ADR-0044 / ADR-0054 — ``ros_integration.goal_builder`` selects a
    goal-lowering adapter over the base verbatim-``default_goal_json``
    engine: ``look_at`` → gaze pose, ``pose`` → generic Cartesian EEF,
    ``joint`` → joint-space goal. ``None`` keeps the base engine.
    """
    from openral_rskill.ros_action_rskill import ROSActionRskill

    if builder == "look_at":
        from openral_rskill.look_at_rskill import LookAtRskill

        return LookAtRskill
    if builder == "pose":
        from openral_rskill.pose_goal_rskill import PoseGoalRskill

        return PoseGoalRskill
    if builder == "joint":
        from openral_rskill.joint_goal_rskill import JointGoalRskill

        return JointGoalRskill
    return ROSActionRskill


def make_default_skill_resolver(
    ros_node: Any,
    *,
    search_paths: Sequence[str | Path] = (),
    scene_cameras: Sequence[str] = (),
    tf_lookup: Any = None,
    tf_lookup_getter: Any = None,
) -> SkillResolver:
    """Build the production resolver that knows about wrapped-ROS rSkills.

    Replacement for using :func:`_default_skill_resolver` directly. The
    factory captures the runner's lifecycle node so that resolved
    :class:`~openral_rskill.ros_action_rskill.ROSActionRskill` instances
    can create their wrapped ``ActionClient`` / service client on the
    same node — futures share the runner's existing rclpy spin and the
    polling pattern used by ``_maybe_reset_hal_to_starting_pose`` keeps
    working.

    Behaviour per ``manifest.kind`` of the resolved manifest:

    * ``"vla"`` — when ``search_paths`` are provided AND the manifest is
      indexed there, route through the in-tree
      :func:`make_local_skill_resolver` path (same shim used by
      ``openral sim run``). Otherwise fall back to
      :func:`_default_skill_resolver` (HF Hub).
    * ``"ros_action"`` / ``"ros_service"`` — build a
      :class:`~openral_rskill.ros_action_rskill.ROSActionRskill` against
      the captured ``ros_node``. Requires a local in-tree manifest in
      ``search_paths`` because wrapped skills carry no HF Hub weights
      to download — the manifest itself is the entire on-disk artefact.
    * ``"wam"`` — rejected with :class:`ROSConfigError`; the WAM
      resolver branch is not implemented in this PR (tracked separately).

    Args:
        ros_node: The host :class:`rclpy.lifecycle.LifecycleNode` (the
            ``RskillRunnerNode`` instance itself in production).
        search_paths: In-tree manifest search paths. The wrapped-ROS
            branch needs at least one entry; the VLA branch falls
            through to HF Hub when empty.
        scene_cameras: Forwarded into the VLA local-resolver path; see
            :func:`make_local_skill_resolver`.
        tf_lookup: ADR-0027 — forwarded into the VLA local-resolver
            path so wrapped-task-space layouts (``human300_16d`` etc.)
            assemble ``observation.state`` from live TF at step time.
            None preserves the joint-space path.
        tf_lookup_getter: ADR-0027 — zero-arg callable returning the
            current ``tf_lookup`` (or None). Lets the resolver pick up a
            TF buffer that is wired after the resolver is built; forwarded
            to :func:`make_local_skill_resolver`.
    """
    local_resolver = make_local_skill_resolver(
        search_paths=search_paths,
        scene_cameras=scene_cameras,
        tf_lookup=tf_lookup,
        tf_lookup_getter=tf_lookup_getter,
    )
    # Capture the canonical node handle in the closure scope BEFORE the
    # inner resolver is defined so the body can reference it as a free
    # variable. The inner resolver also accepts a `ros_node` kwarg (the
    # per-call handle the runner passes) and ignores it — the captured
    # handle wins. This keeps the production wiring authoritative even
    # if a caller forgets the kwarg.
    ros_node_captured = ros_node

    def _resolver(
        *,
        rskill_id: str,
        revision: str,
        prompt: str,
        prompt_metadata_json: str,
        goal_params_json: str = "",
        description: RobotDescription | None,
        commercial_deployment: bool,
        ros_node: Any = None,  # unused here; the captured `ros_node` wins
    ) -> rSkillBase:
        del ros_node  # closure captures the canonical node above
        from openral_core import RSkillManifest as _RSkillManifest
        from openral_core.exceptions import ROSConfigError

        # The local resolver tries each search_path; we duplicate just the
        # lookup so we can dispatch on `manifest.kind` BEFORE delegating to
        # the right branch. Keeps the wrapped-ROS path closed against
        # accidentally hitting the VLA `make_policy` path.
        manifest: _RSkillManifest | None = None
        import pathlib

        for root in (pathlib.Path(p) for p in search_paths if str(p)):
            if not root.exists():
                continue
            for yaml_path in sorted(root.glob("*/rskill.yaml")):
                try:
                    candidate = _RSkillManifest.from_yaml(str(yaml_path))
                except Exception:  # reason: skip unloadable manifests
                    continue
                if candidate.name == rskill_id:
                    manifest = candidate
                    break
            if manifest is not None:
                break

        if manifest is None:
            # No in-tree match → must be a VLA on HF Hub. Wrapped-ROS
            # skills have no HF-Hub fallback path (no weights to fetch).
            return _default_skill_resolver(
                rskill_id=rskill_id,
                revision=revision,
                prompt=prompt,
                prompt_metadata_json=prompt_metadata_json,
                goal_params_json=goal_params_json,
                description=description,
                commercial_deployment=commercial_deployment,
                ros_node=ros_node_captured,
            )

        if manifest.kind == "vla":
            return local_resolver(
                rskill_id=rskill_id,
                revision=revision,
                prompt=prompt,
                prompt_metadata_json=prompt_metadata_json,
                description=description,
                commercial_deployment=commercial_deployment,
                ros_node=ros_node_captured,
            )
        if manifest.kind in {"ros_action", "ros_service"}:
            # ADR-0044 / ADR-0054 — ros_integration.goal_builder selects a
            # goal-lowering adapter subclass instead of the verbatim
            # default_goal_json path.
            builder = (
                manifest.ros_integration.goal_builder
                if manifest.ros_integration is not None
                else None
            )
            adapter_cls = _ros_action_adapter_cls(builder)
            skill = adapter_cls(
                manifest=manifest,
                ros_node=ros_node_captured,
                robot_description=description,
                prompt=prompt,
                prompt_metadata_json=prompt_metadata_json,
                goal_params_json=goal_params_json,
            )
            skill.configure()
            skill.activate()
            return skill
        if manifest.kind == "wam":
            raise ROSConfigError(
                f"rSkill {rskill_id!r} declares kind='wam'; the WAM resolver "
                "branch is not implemented yet (tracked separately). "
                "VLA / ros_action / ros_service kinds are supported today."
            )
        raise ROSConfigError(
            f"rSkill {rskill_id!r} declares unknown kind={manifest.kind!r}; "
            "expected one of 'vla', 'wam', 'ros_action', 'ros_service'."
        )

    return _resolver


def make_local_skill_resolver(
    search_paths: Sequence[str | Path],
    *,
    scene_cameras: Sequence[str] = (),
    tf_lookup: Any = None,
    tf_lookup_getter: Any = None,
) -> SkillResolver:
    """Build a resolver that loads rSkills strictly from in-tree manifests.

    Walks each search path once and indexes every ``*/rskill.yaml`` by
    the manifest's ``name:`` field. On each resolve call:

    * If ``rskill_id`` is in the index, builds the runtime policy
      adapter via :func:`openral_sim.factory.make_policy` (the same
      path ``openral sim run`` takes) and wraps it in a thin
      :class:`rSkillBase` shim so the skill_runner's lifecycle +
      embodiment-gate contract is satisfied. No HF Hub fallback.
    * Otherwise raises ``ROSConfigError`` listing the known skill ids.
      The reasoner's tool palette is built from the same search paths,
      so anything the reasoner can pick MUST be resolvable here.

    The manifest index is built lazily on first resolve.

    Args:
        search_paths: Iterable of directories containing
            ``<id>/rskill.yaml`` files. The same parameter the reasoner
            uses to seed its palette (``rskill_search_paths``).
        scene_cameras: Camera-key tuple forwarded to the policy
            factory's ``_SimpleEnvCfg.scene.cameras`` so adapters like
            pi05 wire ``observation.images.<cam>`` correctly. Empty
            tuple is fine for adapters that fall back to manifest
            aliases.
        tf_lookup: ADR-0027 — forwarded to the inner
            ``_PolicyAdapterSkill`` so wrapped-task-space VLAs can
            assemble ``observation.state`` from live TF + bindings at
            step time. ``None`` preserves the joint-space path.
        tf_lookup_getter: Lazy alternative to ``tf_lookup``. Called at
            dispatch time (not factory time) to read the current TF
            lookup — required when the lookup is initialised in
            ``on_configure`` AFTER this resolver factory runs (the
            ``compose_runtime`` path). When set, takes precedence over
            ``tf_lookup``.
    """
    import pathlib

    paths = [pathlib.Path(p) for p in search_paths if str(p)]
    scene_cameras_t = tuple(str(c) for c in scene_cameras)

    index: dict[str, pathlib.Path] = {}
    index_built = False

    def _build_index() -> None:
        from openral_core import RSkillManifest

        nonlocal index_built
        if index_built:
            return
        for root in paths:
            if not root.exists():
                continue
            for yaml_path in sorted(root.glob("*/rskill.yaml")):
                try:
                    manifest = RSkillManifest.from_yaml(str(yaml_path))
                except Exception:  # reason: skip unloadable manifests
                    continue
                index[manifest.name] = yaml_path
        index_built = True

    def _resolver(
        *,
        rskill_id: str,
        revision: str,
        prompt: str,
        prompt_metadata_json: str,
        description: RobotDescription | None,
        commercial_deployment: bool,
        ros_node: Any = None,
    ) -> rSkillBase:
        # `ros_node` is accepted so the local resolver shares the
        # SkillResolver signature with `make_default_skill_resolver` —
        # in-tree VLA shims don't need the node handle, but matching
        # signatures keeps the runner's call site uniform.
        del revision, prompt_metadata_json, commercial_deployment, ros_node
        from openral_core.exceptions import ROSConfigError

        _build_index()
        yaml_path = index.get(rskill_id)
        if yaml_path is None:
            raise ROSConfigError(
                f"rskill_id {rskill_id!r} not in local search paths "
                f"({[str(p) for p in paths]!r}); known ids: {sorted(index)!r}. "
                "The reasoner picked a skill the runner cannot resolve — "
                "either add it under one of the search paths or fix the "
                "reasoner's palette so it doesn't surface unresolvable ids.",
            )
        # Resolve tf_lookup lazily — at dispatch time, AFTER on_configure
        # has wired ``_init_tf_lookup``. Without this the resolver would
        # capture None at factory time and the wrapped-task-space
        # assembler in ``_step_impl`` would silently fall back to the
        # raw 11-D joint-state slice.
        resolved_tf_lookup = tf_lookup_getter() if tf_lookup_getter is not None else tf_lookup
        return _build_runtime_skill_from_manifest(
            yaml_path=yaml_path,
            prompt=prompt,
            scene_cameras=scene_cameras_t,
            description=description,
            tf_lookup=resolved_tf_lookup,
        )

    return _resolver


def _sensor_name_to_vla_slot(description: RobotDescription | None) -> dict[str, str]:
    """Map each RGB sensor's NAME to its VLA observation slot.

    Deploy-sim keys ``WorldState.image_frames`` (and the topic basename
    ``/openral/cameras/<name>/image``) by the manifest sensor NAME, but
    VLA adapters look up ``obs["images"]`` by the VLA slot — ``camera1``
    / ``camera2`` / ... — the LIBERO convention ``openral sim run`` and the
    rldx adapter already use. This map realigns the two so a manifest
    whose RGB sensors are descriptively named (franka: ``agentview`` /
    ``wrist``) still feeds the adapter ``camera1`` / ``camera2``.

    The slot is the ``vla_feature_key`` suffix
    (``observation.images.camera1`` -> ``camera1``); sensors without a
    ``vla_feature_key`` fall back to their own name (robocasa real-name
    keys, where the sensor name already IS the slot). Mirrors
    ``openral_hal.sim_sensor_bridge._obs_key_for_sensor`` — kept local
    because a Layer-3 skill package must not import the Layer-0 HAL
    (CLAUDE.md §3).
    """
    if description is None:
        return {}
    out: dict[str, str] = {}
    for sensor in description.sensors:
        if getattr(sensor, "modality", None) != "rgb":
            continue
        vfk = getattr(sensor, "vla_feature_key", None)
        out[sensor.name] = str(vfk).rsplit(".", 1)[-1] if vfk else sensor.name
    return out


def _vla_camera_slots(description: RobotDescription | None) -> tuple[str, ...]:
    """RGB sensor VLA slots (``camera1`` / ``camera2`` / ...) in manifest order.

    The values of :func:`_sensor_name_to_vla_slot`, used as the adapter's
    ``scene_cameras`` so ``resolve_camera_keys`` -> ``_camera_keys`` lands
    on the slots the checkpoint's ``cam_alias`` maps (``camera1 ->
    image``). Empty when the manifest declares no RGB sensors — callers
    then keep their existing ``scene_cameras``.
    """
    return tuple(_sensor_name_to_vla_slot(description).values())


def _decode_image_frames(
    image_frames: dict[str, Any],
    sensor_to_slot: dict[str, str],
) -> dict[str, Any]:
    """Decode ``WorldState.image_frames`` into a VLA-slot-keyed ``obs["images"]``.

    Each :class:`~openral_core.schemas.SensorFrame` with inline ``data``
    is decoded into an ``HxWxC`` uint8 array and stored under its VLA slot
    (:func:`_sensor_name_to_vla_slot`). Sensors absent from
    ``sensor_to_slot`` pass through under their own name. Frames without
    inline pixels (``data is None`` — topic / handle delivery) are
    skipped.
    """
    import numpy as np

    images: dict[str, Any] = {}
    for name, frame in image_frames.items():
        if frame.data is None:
            continue
        arr = np.frombuffer(frame.data, dtype=np.uint8).reshape(
            int(frame.height),
            int(frame.width),
            int(frame.channels),
        )
        images[sensor_to_slot.get(name, name)] = arr
    return images


def _build_runtime_skill_from_manifest(
    *,
    yaml_path: Path,
    prompt: str,
    scene_cameras: tuple[str, ...] = (),
    description: RobotDescription | None = None,
    tf_lookup: Any = None,
) -> rSkillBase:
    """Mirror ``openral sim run`` end-to-end: manifest → VLASpec → make_policy → rSkillBase shim.

    Bridges the openral_sim ``PolicyAdapter`` Protocol (used by the
    eval runner) to the openral_rskill ``rSkillBase`` ABC (used by the
    F1 skill_runner action server). The shim:

    * exposes manifest fields through ``rSkillBase.info`` so the
      runner's embodiment / role / license gates pass;
    * drives ``configure → active`` so the runner sees an active
      skill;
    * forwards ``_step_impl(world_state)`` to ``adapter.step(obs, task)``
      after building an ``Observation`` from
      ``world_state.image_frames`` + ``joint_state``.
    """
    from openral_core import RSkillManifest, VLASpec
    from openral_core.exceptions import ROSConfigError, ROSRuntimeError
    from openral_sim.factory import make_policy
    from openral_sim.policy_deps import (
        model_family_install_hint,
        purge_partial_imports,
    )

    manifest = RSkillManifest.from_yaml(str(yaml_path))
    # Defensive guard: this helper builds a VLA policy adapter shim;
    # wrapped-ROS rSkills (kind: ros_action / ros_service) must NOT come
    # through here because they have no model weights to bind. The
    # production dispatch (``make_default_skill_resolver``) branches on
    # ``manifest.kind`` and routes those to ``ROSActionRskill`` directly;
    # this branch protects the legacy call sites that still go via
    # ``make_local_skill_resolver`` from accidentally invoking
    # ``make_policy`` on a manifest that doesn't carry a model_family.
    if manifest.kind != "vla":
        raise ROSConfigError(
            f"_build_runtime_skill_from_manifest({manifest.name!r}): "
            f"kind={manifest.kind!r} is not a VLA policy. Wrapped-ROS skills "
            "are resolved via make_default_skill_resolver(), which branches "
            "on manifest.kind before reaching this helper."
        )
    vla = VLASpec(
        id=manifest.model_family,
        # Pass the absolute local directory so the same resolver that
        # `openral sim run` uses can find the manifest on disk.
        weights_uri=str(yaml_path.parent),
        device="auto",
        extra={},
    )
    # The policy factories only access `env_cfg.vla`; a SimpleNamespace
    # wrapper is enough to avoid building a full Pydantic SimEnvironment
    # (which requires scene + task fields the runtime path does not
    # care about). Mirrors what the sim CLI does internally.
    #
    # Deploy-sim's runtime_node forwards the manifest SENSOR NAMES as
    # `scene_cameras` (its `camera_names`), but the adapter needs the VLA
    # slots (camera1/camera2/...) so `resolve_camera_keys` -> `_camera_keys`
    # lands where the checkpoint's `cam_alias` maps them (camera1 -> image).
    # When the manifest declares RGB sensors the derived slots INTENTIONALLY
    # supersede the passed `scene_cameras` (deploy-sim only ever populates it
    # with sensor names); only fall back to the passed value when the
    # description declares no RGB sensors. Slot-level overrides belong in the
    # rSkill manifest's `extra["camera_keys"]`, which `resolve_camera_keys`
    # still honours first. The obs-image keys are realigned to match in
    # `_PolicyAdapterSkill._step_impl`.
    effective_scene_cameras = _vla_camera_slots(description) or scene_cameras
    env_stub = _SimpleEnvCfg(vla=vla, cameras=effective_scene_cameras)
    try:
        adapter = make_policy(env_stub)  # type: ignore[arg-type]
    except ImportError as exc:
        # Most policy factories (smolvla / pi05 / act / diffusion) live
        # behind opt-in extras groups (``sim`` / ``libero`` / ``metaworld``
        # / ``robocasa``) — ``transformers``, ``bitsandbytes``,
        # ``lerobot[…]``, etc. When that group isn't installed the
        # factory raises ``ImportError`` (or ``ModuleNotFoundError``)
        # deep inside lerobot. We translate to ``ROSRuntimeError`` with
        # an actionable install hint so the action server reports a
        # clean failure_reason instead of a stack-trace through three
        # layers of lerobot imports. The reasoner's pre-flight palette
        # filter (``openral_sim.policy_deps.filter_importable_manifests``)
        # normally drops such skills at boot, so this branch is a
        # belt-and-braces safety net for skills that snuck through
        # (out-of-tree families, drift in the family→imports map, …).
        # NOTE: ``torch`` is intentionally NOT purged — its C++ side
        # holds process-global state that breaks if the Python module
        # is removed from ``sys.modules`` mid-process.
        purge_partial_imports(("lerobot", "transformers"))
        family = manifest.model_family
        install_hint = model_family_install_hint(family)
        raise ROSRuntimeError(
            f"failed to build {family!r} policy for rSkill "
            f"{manifest.name!r}: {type(exc).__name__}: {exc}. "
            f"{install_hint}"
        ) from exc
    return _make_policy_adapter_skill(
        manifest=manifest,
        adapter=adapter,
        prompt=prompt,
        description=description,
        tf_lookup=tf_lookup,
    )


class _SimpleSceneCfg:
    """Minimal `env_cfg.scene` carrier — only `.cameras` is read by pi05/SmolVLA."""

    def __init__(self, *, cameras: list[str] | tuple[str, ...]) -> None:
        self.cameras = tuple(cameras)


class _SimpleEnvCfg:
    """Minimal `env_cfg` carrier — `.vla` and `.scene.cameras` are read by policy factories."""

    def __init__(self, *, vla: object, cameras: list[str] | tuple[str, ...] = ()) -> None:
        self.vla = vla
        self.scene = _SimpleSceneCfg(cameras=cameras)


def _detect_joint_units_are_degrees(adapter: object) -> bool:
    """Inspect the loaded normalizer's state stats to infer the checkpoint's joint units.

    Different OpenArm pi05 checkpoints use different conventions:

    * ``yuto-urushima/openarm_pickplace_*`` /
      ``OpenRAL/rskill-pi05-openarm-pickplace-*`` record state +
      action in DEGREES (the canonical lerobot OpenArm SDK
      convention — ``logger.debug(f"Clipped {motor_name} from
      {position:.2f}° to {clipped_position:.2f}°")`` lives in
      ``lerobot/robots/openarm_follower/openarm_follower.py:282``).
    * ``mddoai/pi05_openarm_*`` records in RADIANS (state quantiles
      align with ``robots/openarm/robot.yaml``'s ``position_limits``
      in radians — e.g. ``L_j4.min/max = 0.299/2.438`` rad ↔ yaml
      ``[0.0, 2.44346]``).

    Decoded by walking the preprocessor pipeline to find the
    ``normalizer_processor`` step's loaded ``observation.state.q99``
    tensor. If any non-gripper arm-joint q99 exceeds ``π``
    (3.14) — physically impossible in radians for a manipulator
    joint — the checkpoint is in degrees. Defaults to radians on any
    introspection failure (the safer default — a missed deg→rad
    conversion sends large numbers; a spurious one quietly compresses
    them).
    """
    try:
        pipeline = adapter._preprocessor  # type: ignore[attr-defined]
        for step in getattr(pipeline, "steps", []):
            stats = getattr(step, "stats", None) or getattr(step, "_stats", None)
            if stats is None:
                continue
            q99 = stats.get("observation.state.q99")
            if q99 is None:
                continue
            # The 16-vector layout is checkpoint-specific, but the
            # *peak* magnitude across all 16 channels is enough:
            # any radians-encoded arm joint stays below π regardless
            # of position, while a degrees-encoded q99 hits 90+ for
            # the elbows. Use 5 rad ≈ 286° as the threshold so
            # gripper outliers (custom motor unit, can be > 1 in
            # either convention) don't trip the heuristic.
            peak = float(abs(q99).max())
            return peak > 5.0
    except Exception:  # reason: introspection across processor versions; never fatal
        pass
    return False


def _build_joint_permutation(
    *,
    adapter: object,
    description: RobotDescription | None,
) -> tuple[list[int] | None, list[bool]]:
    """Map ``description.joints`` order onto ``policy.config.action_feature_names``.

    Different checkpoints can publish their state / action vectors in a
    different joint order than the robot's URDF / RobotDescription. For
    OpenArm bimanual this is the classic case: ``robots/openarm/robot.yaml``
    lists ``[left_joint1..7, left_gripper, right_joint1..7, right_gripper]``
    (left-first), but the pi05-openarm-pickplace-120ep checkpoint's
    ``config.json`` declares
    ``[right_joint_1.pos, ..., right_gripper.pos, left_joint_1.pos, ..., left_gripper.pos]``
    (right-first). Without a reorder the safety kernel correctly rejects
    each chunk because it validates ``policy_action[i]`` against the
    envelope's ``robot_joint_position_max[i]`` — different joints.

    Returns:
        A list ``robot_to_policy`` of length ``len(description.joints)``
        where ``robot_to_policy[i] = j`` such that
        ``policy_names[j] == robot_names[i]`` (after normalising both
        sides). Or ``None`` when:

        * the adapter does not expose ``policy.config.action_feature_names``
          (e.g. ACT / DiffusionPolicy adapters, or a non-pi05 backbone);
        * the policy's joint count does not match the robot's
          (single-arm checkpoint on a bimanual robot or vice-versa);
        * any robot joint name is missing from the policy's list
          (incompatible embodiments — surface the mismatch loudly instead
          of silently flopping bytes around).

        A ``None`` return is interpreted by the shim as "pass through";
        the kernel then enforces correctness downstream.
    """
    if description is None:
        return None, []
    try:
        policy = adapter._policy  # type: ignore[attr-defined]  # reason: documented Protocol-internal field
        names = list(policy.config.action_feature_names)
    except (AttributeError, TypeError):
        return None, []
    if not names:
        return None, []

    def _normalize(s: str) -> str:
        # Map LeRobot feature keys to robot.yaml joint names.
        # Handles three known conventions:
        #   ``right_joint_1.pos`` → ``right_joint1`` (yuto / AdrianLlopart)
        #   ``openarm_left_joint1`` → ``left_joint1`` (mddoai)
        #   ``left_joint1``        → ``left_joint1`` (canonical)
        # All three end up matching ``robots/openarm/robot.yaml``'s
        # joint name list.
        s = s.replace(".pos", "").replace("_joint_", "_joint")
        if s.startswith("openarm_"):
            s = s[len("openarm_") :]
        return s

    policy_names = [_normalize(n) for n in names]
    robot_names = [j.name for j in description.joints]
    if len(policy_names) != len(robot_names):
        return None, []
    name_to_pidx = {n: i for i, n in enumerate(policy_names)}
    try:
        perm = [name_to_pidx[n] for n in robot_names]
    except KeyError:
        return None, []
    # `policy_is_gripper[j] = True` iff policy slot j is a gripper feature.
    # Decoded from the (normalised) feature name (`*_gripper`). The LeRobot
    # OpenArm dataset records arm joints in DEGREES but grippers in a
    # custom motor unit (state distribution centres around -1 with a long
    # tail to -50 — neither radians nor degrees), so the shim must apply
    # rad↔deg conversion to the 7 arm joints per side and pass the
    # grippers through untouched. Mis-converting the gripper produces
    # the same "every joint slammed to limit" symptom as before.
    policy_is_gripper = ["gripper" in n.lower() for n in policy_names]
    return perm, policy_is_gripper


def _pad_joint_payload(
    slice_values: list[float],
    joint_names: list[str],
    name_to_idx: dict[str, int],
    n_dof_total: int,
) -> list[float]:
    """Pad a sub-slot JOINT_* slice to full-dof so the kernel n_dof check passes.

    ADR-0028d — the C++ safety kernel enforces ``chunk.n_dof ==
    envelope.n_dof`` for any JOINT_* mode (per-joint validation indexes
    into ``envelope.joint_*_max[]``). A slot-dispatched chunk that
    targets only a few joints (e.g. the rldx-rc365 3-D base
    JOINT_VELOCITY) therefore needs to be expanded into a full-dof
    vector with zeros at non-target joints. Zeros are within all
    velocity bounds, so the kernel's per-joint validation still runs
    correctly on the active joints.

    Falls back to the raw slice when ``joint_names`` is empty or no
    description is available (legacy single-surface joint dispatch
    where the caller already produced the full-dof payload).
    """
    if not joint_names or n_dof_total == 0:
        return slice_values
    if len(slice_values) != len(joint_names):
        raise ValueError(
            f"_pad_joint_payload: slice len {len(slice_values)} does not match "
            f"joint_names len {len(joint_names)}"
        )
    padded = [0.0] * n_dof_total
    for i, name in enumerate(joint_names):
        idx = name_to_idx.get(name)
        if idx is None:
            raise ValueError(
                f"_pad_joint_payload: joint_names[{i}]={name!r} not in robot "
                f"description (have: {sorted(name_to_idx.keys())})"
            )
        padded[idx] = slice_values[i]
    return padded


def _dispatch_slots(  # noqa: PLR0912  # reason: one branch per ActionSlot control mode; flat dispatch mirrors the manifest's slot list
    slots: list,
    policy_action: Any,
    *,
    description: Any | None = None,
) -> list:
    """Build one typed :class:`Action` per non-discard :class:`ActionSlot`.

    ADR-0028b — the manifest's ``action_contract.slots`` block declares
    how the policy's flat action vector splits into typed sub-actions.
    The :class:`openral_core.ActionContract` validator already enforced
    coverage (no gaps / overlaps / over-range) and per-mode field
    requirements at fixture load, so this loop trusts its inputs and
    only does the byte-routing.

    Args:
        slots: ``manifest.action_contract.slots``, a list of
            :class:`openral_core.ActionSlot`.
        policy_action: The raw 1-D ``np.float32`` policy vector from
            ``adapter.step()``. Indexed directly per slot range — no
            permutation / clamp, since per-mode safety bounds live on
            the supervisor side (ADR-0028b step 5).
        description: Optional ``RobotDescription`` used to pad
            sub-slot JOINT_* chunks to full-dof per ADR-0028d (so the
            C++ safety kernel's ``chunk.n_dof == envelope.n_dof``
            check accepts them). When ``None``, JOINT_* chunks pass
            through with the raw slice width.

    Returns:
        ``list[Action]`` — one per non-discard slot, all carrying
        ``horizon=1``. The runner inherits the same OTel trace_id for
        all returned actions via the enclosing ``inference_span``.
    """
    from openral_core.schemas import Action, ControlMode

    joint_name_to_idx: dict[str, int] = {}
    n_dof_total: int = 0
    if description is not None:
        n_dof_total = len(description.joints)
        joint_name_to_idx = {j.name: i for i, j in enumerate(description.joints)}

    out: list = []
    for slot in slots:
        if slot.discard:
            continue
        lo, hi = slot.range
        sl = [float(v) for v in policy_action[lo : hi + 1].tolist()]
        mode = slot.control_mode
        if mode is ControlMode.JOINT_POSITION:
            payload = _pad_joint_payload(sl, slot.joint_names, joint_name_to_idx, n_dof_total)
            out.append(Action(control_mode=mode, horizon=1, joint_targets=[payload]))
        elif mode is ControlMode.JOINT_VELOCITY:
            payload = _pad_joint_payload(sl, slot.joint_names, joint_name_to_idx, n_dof_total)
            out.append(Action(control_mode=mode, horizon=1, joint_velocities=[payload]))
        elif mode is ControlMode.JOINT_TORQUE:
            payload = _pad_joint_payload(sl, slot.joint_names, joint_name_to_idx, n_dof_total)
            out.append(Action(control_mode=mode, horizon=1, joint_torques=[payload]))
        elif mode is ControlMode.CARTESIAN_DELTA:
            out.append(
                Action(
                    control_mode=mode,
                    horizon=1,
                    cartesian_delta=[tuple(sl)],
                    ee_name=slot.ee,
                    frame_id=slot.frame,
                )
            )
        elif mode is ControlMode.CARTESIAN_TWIST:
            out.append(
                Action(
                    control_mode=mode,
                    horizon=1,
                    cartesian_twist=[tuple(sl)],
                    ee_name=slot.ee,
                    frame_id=slot.frame,
                )
            )
        elif mode is ControlMode.BODY_TWIST:
            # ``Action.body_twist`` is a list of 6-tuples
            # ``(vx, vy, vz, wx, wy, wz)``. A 3-D planar twist slice
            # (RoboCasa: forward, side, yaw) pads the 4 missing
            # components with 0.0 — the convention the panda_mobile HAL
            # already consumes on /cmd_vel (linear.z, angular.x,
            # angular.y stay zero on a holonomic planar base).
            if len(sl) == 3:
                vx, vy, wz = sl
                twist = (vx, vy, 0.0, 0.0, 0.0, wz)
            elif len(sl) == 6:
                twist = tuple(sl)  # type: ignore[assignment]  # reason: length 6 checked
            else:
                raise ValueError(
                    f"_dispatch_slots: BODY_TWIST slot must be 3-D (planar base) "
                    f"or 6-D (full twist); got width {len(sl)} on slot range "
                    f"[{lo}, {hi}]"
                )
            out.append(
                Action(control_mode=mode, horizon=1, body_twist=[twist], frame_id=slot.frame)
            )
        elif mode in (ControlMode.GRIPPER_BINARY, ControlMode.GRIPPER_POSITION):
            out.append(Action(control_mode=mode, horizon=1, gripper=sl, ee_name=slot.ee))
        elif mode is ControlMode.COMPOSITE_MODE:
            # ADR-0028d — slot width is 1 (validated by ActionSlot).
            out.append(Action(control_mode=mode, horizon=1, composite_mode=sl))
        else:
            raise ValueError(
                f"_dispatch_slots: unsupported control_mode {mode!r} on slot "
                f"range [{lo}, {hi}]. Supported: joint_*, cartesian_delta, "
                "cartesian_twist, body_twist, gripper_*."
            )
    return out


def _make_policy_adapter_skill(
    *,
    manifest: object,
    adapter: object,
    prompt: str,
    description: RobotDescription | None = None,
    tf_lookup: Any = None,
) -> rSkillBase:
    """Instantiate the rSkillBase shim around a `PolicyAdapter`.

    Imports `openral_rskill.base.rSkillBase` lazily because the import
    transitively pulls torch / lerobot — we only pay that cost on a
    real skill resolve.

    Args:
        manifest: ``RSkillManifest`` whose ``embodiment_tags`` /
            ``role`` / ``latency_budget`` / ``state_contract`` the
            shim exposes through ``rSkillBase.info``.
        adapter: Built ``openral_sim.policy.PolicyAdapter`` — driven
            by ``adapter.step(obs, prompt)`` per tick.
        prompt: Operator-supplied natural-language instruction.
            Routed into ``obs["task"]`` every step.
        description: ``RobotDescription`` used to build the
            robot-order ↔ policy-order joint permutation. Optional
            — when absent the runner's joint permutation is skipped.
        tf_lookup: Optional :class:`openral_state_adapter.TfLookup`
            callable. When set AND the manifest declares a
            ``state_contract.layout`` registered in the
            ``openral_state_adapter`` registry, ``_step_impl``
            assembles ``obs["state"]`` via that layout's assembler
            instead of the raw joint-state slice. None preserves the
            joint-space path (every VLA shipped before ADR-0027).
    """
    import numpy as np
    from openral_core.schemas import Action, ControlMode
    from openral_rskill.base import rSkillBase

    robot_to_policy, policy_is_gripper = _build_joint_permutation(
        adapter=adapter,
        description=description,
    )
    # Sensor-name -> VLA-slot map (camera1/camera2/...) so `_step_impl`
    # rekeys `obs["images"]` to what the adapter looks up. Built once at
    # skill-build time; see `_sensor_name_to_vla_slot`.
    sensor_to_slot = _sensor_name_to_vla_slot(description)
    joint_units_are_degrees = _detect_joint_units_are_degrees(adapter)
    # Print to stderr so the diagnostic shows up in the launch's
    # stitched-together stdout (structlog's OTel sink doesn't surface
    # there). One-time event at build-time — keeps the per-step
    # console quiet.
    print(
        f"[rskill_runner_node] policy_adapter.skill_built "
        f"skill={getattr(manifest, 'name', '?')!r} "
        f"joint_units={'degrees' if joint_units_are_degrees else 'radians'} "
        f"perm={robot_to_policy} "
        f"is_gripper={policy_is_gripper}",
        file=sys.stderr,
        flush=True,
    )
    # Per-joint absolute clamping bounds taken straight from the
    # RobotDescription (URDF / MJCF) joint limits — same limits the
    # safety_kernel's envelope encodes. Without this, an out-of-distribution
    # checkpoint can emit a joint target a few degrees past the mechanical
    # range; the kernel correctly rejects + estops, the robot never moves,
    # and the operator sees nothing happen. In hardware the motors / firmware
    # would also clamp (or refuse) such commands; in MuJoCo the <position>
    # actuator's ctrlrange clamps too. Pre-clamping here lets the kernel see
    # an in-range chunk + the HAL apply it, while staying *strictly tighter*
    # than the envelope (so any real envelope violation still surfaces).
    if description is not None:
        joint_limits: list[tuple[float, float] | None] = [
            (
                (float(j.position_limits[0]), float(j.position_limits[1]))
                if j.position_limits is not None
                else None
            )
            for j in description.joints
        ]
    else:
        joint_limits = []

    class _PolicyAdapterSkill(rSkillBase):  # type: ignore[misc, valid-type]
        """``rSkillBase`` shim over an :class:`openral_sim.policy.PolicyAdapter`.

        The adapter (built via :func:`openral_sim.factory.make_policy`)
        is the same runtime object ``openral sim run`` drives — same
        weights, same preprocessor, same action contract. This shim
        exposes the manifest's contract through ``info`` and routes
        the runner's ``step(world_state)`` calls onto
        ``adapter.step(observation, instruction)``.
        """

        def __init__(self) -> None:
            super().__init__(
                name=manifest.name,  # type: ignore[attr-defined]
                version=manifest.version,  # type: ignore[attr-defined]
                role=manifest.role,  # type: ignore[attr-defined]
                embodiment_tags=list(manifest.embodiment_tags),  # type: ignore[attr-defined]
                latency_budget_ms=(
                    manifest.latency_budget.per_chunk_ms  # type: ignore[attr-defined]
                    if manifest.latency_budget is not None  # type: ignore[attr-defined]
                    else None
                ),
            )
            self._adapter = adapter
            self._prompt = prompt
            # Hold the full manifest so the F1 skill_runner can read
            # fields the rSkillBase ABC doesn't expose (e.g.
            # ``starting_pose`` for the HAL ResetToPose call before
            # the first inference tick).
            self.manifest = manifest
            # ADR-0027 — when the manifest declares a wrapped-task-space
            # layout AND its assembler is registered, _step_impl
            # substitutes the assembled vector for the raw joint-state
            # slice. None = preserve the joint-space path.
            self._tf_lookup = tf_lookup
            # Debug-only obs/action capture. When ``OPENRAL_DUMP_OBS_TICK``
            # is set to a comma-separated list of policy-tick indices
            # (e.g. ``1,50,200``), ``_step_impl`` pickles the full
            # (obs, raw_policy_action) tuple to
            # ``OPENRAL_DUMP_OBS_PATH``/``<rskill>_tickNN.pkl`` for the
            # listed ticks and a sibling ``...tickNN_camera<key>.npy``
            # per camera key. Default unset → zero overhead.
            # Lets a deploy_sim vs sim_run side-by-side compare the exact
            # bytes the rldx adapter sees on each path without writing a
            # production code path.
            import os  # reason: defer to keep import light

            _dump_env = os.environ.get("OPENRAL_DUMP_OBS_TICK", "").strip()
            self._dump_ticks: set[int] = set()
            if _dump_env:
                for raw_tok in _dump_env.split(","):
                    tok = raw_tok.strip()
                    if tok.isdigit():
                        self._dump_ticks.add(int(tok))
            self._dump_path = os.environ.get("OPENRAL_DUMP_OBS_PATH", "/tmp/openral_obs_dump")

        def _configure_impl(self) -> None:
            """No-op — `make_policy` already built the adapter."""

        def _activate_impl(self) -> None:
            """Reset the adapter's per-episode state (action queue, RNG)."""
            if hasattr(self._adapter, "reset"):
                self._adapter.reset()  # type: ignore[attr-defined]

        def _deactivate_impl(self) -> None:
            """No-op — the adapter stays loaded for the next activate."""

        def _shutdown_impl(self) -> None:
            """Release GPU memory / file handles owned by the adapter."""
            if hasattr(self._adapter, "close"):
                self._adapter.close()  # type: ignore[attr-defined]

        def _dump_obs_to_disk(
            self,
            *,
            tick: int,
            obs: dict[str, object],
            raw_policy_action: Any,
            robot_action_pre_clamp: Any,
        ) -> None:
            """Pickle a snapshot of one policy tick for offline diff.

            File layout under ``self._dump_path``:
              * ``<rskill>_tick<NN>.pkl`` — dict with keys ``obs_state``
                (numpy), ``raw_policy_action`` (numpy),
                ``robot_action_pre_clamp`` (numpy), ``prompt`` (str),
                ``image_keys`` (list[str]), ``image_shapes`` (dict),
                ``manifest_name``, ``manifest_version``, ``tick``.
              * ``<rskill>_tick<NN>_camera<key>.npy`` — one file per
                camera, full uint8 HWC array.

            Camera arrays go to ``.npy`` rather than into the pickle so
            you can also `np.load(...)` and PIL-show them without
            unpickling Pydantic objects, and so the pickle stays small
            and grep-friendly. Failures here are swallowed — the dump
            is debug instrumentation, never load-bearing.
            """
            import pickle  # reason: defer; debug-only
            from pathlib import Path

            try:
                root = Path(self._dump_path)
                root.mkdir(parents=True, exist_ok=True)
                stem = f"{getattr(self.manifest, 'name', 'skill').replace('/', '_')}_tick{tick:04d}"
                images = obs.get("images") or {}
                image_shapes: dict[str, tuple[int, ...]] = {}
                if isinstance(images, dict):
                    for k, v in images.items():
                        arr = np.asarray(v)
                        image_shapes[str(k)] = tuple(arr.shape)
                        np.save(root / f"{stem}_camera{k}.npy", arr)
                payload = {
                    "tick": tick,
                    "manifest_name": getattr(self.manifest, "name", "?"),
                    "manifest_version": getattr(self.manifest, "version", "?"),
                    "prompt": self._prompt,
                    "obs_state": np.asarray(obs.get("state"))
                    if obs.get("state") is not None
                    else None,
                    "raw_policy_action": np.asarray(raw_policy_action),
                    "robot_action_pre_clamp": np.asarray(robot_action_pre_clamp),
                    "image_keys": sorted(image_shapes.keys()),
                    "image_shapes": image_shapes,
                }
                with (root / f"{stem}.pkl").open("wb") as f:
                    pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
                print(
                    f"[rskill_runner_node] obs_dump tick={tick} wrote "
                    f"{root / f'{stem}.pkl'} + {len(image_shapes)} camera npy(s)",
                    file=sys.stderr,
                    flush=True,
                )
            except Exception as exc:  # reason: debug dump never load-bearing
                print(
                    f"[rskill_runner_node] obs_dump tick={tick} failed: {exc!r}",
                    file=sys.stderr,
                    flush=True,
                )

        def _step_impl(self, world_state: Any) -> Action | list[Action]:  # noqa: PLR0912, PLR0915  # reason: layout/unit/clamp branches per openarm pi0.5 checkpoint are intentionally inline; splitting them obscures the per-step contract
            obs: dict[str, object] = {"task": self._prompt}
            js = world_state.joint_state
            robot_state = np.asarray(list(js.position), dtype=np.float32)
            # ADR-0027 — when the manifest declares a wrapped-task-space
            # ``state_contract.layout`` whose assembler is registered AND
            # a ``tf_lookup`` is wired, substitute the assembled vector
            # for the raw joint-state slice. The assembler reads live
            # ``/tf`` + the per-robot bindings the manifest declares;
            # the joint-permutation path below is skipped because the
            # layout's field order is the contract.
            state_assembled = False
            sc = getattr(self.manifest, "state_contract", None)
            layout = getattr(sc, "layout", None) if sc is not None else None
            bindings = getattr(sc, "bindings", None) if sc is not None else None
            if self._tf_lookup is not None and layout is not None and bindings is not None:
                # Deferred import keeps the runner module load light
                # (openral_state_adapter pulls numpy + the layout
                # registry); the registry lookup is a single dict
                # check per step.
                from openral_state_adapter import assemble_state, registered_layouts

                if layout in registered_layouts():
                    joint_positions = dict(zip(js.name, js.position, strict=False))
                    obs["state"] = assemble_state(
                        layout,
                        bindings,
                        joint_positions,
                        self._tf_lookup,
                    )
                    state_assembled = True
            # Reorder robot-order state → policy-order state so the
            # checkpoint sees its training-distribution joint layout
            # (see `_build_joint_permutation`). When the permutation is
            # None the policy's order matches the robot's already (or we
            # don't have enough metadata to safely reorder).
            #
            # ALSO: convert rad → deg for the 7 arm joints per side.
            # The LeRobot OpenArm dataset's state + action features are
            # in DEGREES (decoded from
            # ``policy_preprocessor_step_3_normalizer_processor.safetensors`` —
            # ``observation.state.q50[left_joint4]`` is 96.4, which is
            # the bent-elbow home pose in degrees, ≈ π/2 rad). Sending
            # 1.57 (radians) to a policy that's seen 90 (degrees) puts
            # every joint deep in the lower tail of the QUANTILES
            # normalizer and triggers the "all joints slam to max"
            # symptom. Grippers (``policy_is_gripper[j] == True``) are
            # kept untouched — their state distribution centres around
            # ``-1`` in a custom motor unit that isn't a rad↔deg conversion.
            if not state_assembled:
                if robot_to_policy is not None and robot_state.shape[0] == len(robot_to_policy):
                    policy_state = np.empty_like(robot_state)
                    for i, j in enumerate(robot_to_policy):
                        val = float(robot_state[i])
                        if joint_units_are_degrees and not (
                            policy_is_gripper and policy_is_gripper[j]
                        ):
                            val = math.degrees(val)
                        policy_state[j] = val
                    obs["state"] = policy_state
                else:
                    obs["state"] = robot_state
            # Deploy-sim keys `world_state.image_frames` by the manifest
            # sensor NAME; VLA adapters look up `obs["images"]` by the VLA
            # slot (camera1/camera2/...). `sensor_to_slot` realigns the two
            # so the adapter + `openral sim run` agree (see
            # `_sensor_name_to_vla_slot` / `_decode_image_frames`).
            obs["images"] = (
                _decode_image_frames(world_state.image_frames, sensor_to_slot)
                if world_state.image_frames
                else {}
            )

            action_array = self._adapter.step(obs, self._prompt)  # type: ignore[attr-defined]
            # Reorder policy-order action → robot-order action so the
            # safety_kernel + HAL apply each value to the joint the
            # envelope's per-index limits describe. PolicyAdapter returns
            # a 1-D float32 per-step action; wrap as a single-horizon
            # JOINT_POSITION Action. The adapter's action_contract may
            # be cartesian_delta — the safety_kernel + HAL chain
            # interprets the bytes as joint targets; a downstream OSC /
            # IK shim translates if the adapter's output semantics
            # differ.
            policy_action = np.asarray(action_array, dtype=np.float32)
            if robot_to_policy is not None and policy_action.shape[0] == len(robot_to_policy):
                robot_action = np.empty_like(policy_action)
                for i, j in enumerate(robot_to_policy):
                    val = float(policy_action[j])
                    # Symmetric to the state path: convert action back
                    # from degrees only when the checkpoint declares
                    # joints in degrees, and only for non-gripper
                    # channels.
                    if joint_units_are_degrees and not (policy_is_gripper and policy_is_gripper[j]):
                        val = math.radians(val)
                    robot_action[i] = val
            else:
                robot_action = policy_action
            # One-shot stderr diagnostic so the launch's stdout shows
            # what's actually being commanded. Print the FIRST step
            # (or every 50th) to catch policy saturation without spam.
            self._step_count = getattr(self, "_step_count", 0) + 1
            if self._dump_ticks and self._step_count in self._dump_ticks:
                self._dump_obs_to_disk(
                    tick=self._step_count,
                    obs=obs,
                    raw_policy_action=policy_action,
                    robot_action_pre_clamp=robot_action,
                )
            if self._step_count == 1 or self._step_count % 50 == 0:
                obs_state_v = obs.get("state")
                obs_state_dump = (
                    [f"{float(v):+.3f}" for v in np.asarray(obs_state_v).tolist()]
                    if obs_state_v is not None
                    else "?"
                )
                print(
                    f"[rskill_runner_node] policy_step "
                    f"step={self._step_count} "
                    f"|act|max={float(np.abs(robot_action).max()):.3f} "
                    f"state_to_policy={obs_state_dump} "
                    f"raw_policy_action={[f'{float(v):+.3f}' for v in policy_action.tolist()]} "
                    f"robot_action_pre_clamp={[f'{float(v):+.3f}' for v in robot_action.tolist()]}",
                    file=sys.stderr,
                    flush=True,
                )
            # Pre-clamp to the per-joint mechanical range. The safety
            # kernel uses the same limits in its envelope; any value
            # inside [min, max] passes through, any value the policy
            # emits beyond range would otherwise trip an estop. In
            # hardware the motors / firmware clamp here too; in MuJoCo
            # the actuator's ctrlrange clamps; doing it explicitly in
            # the shim makes the safety kernel + the simulator agree
            # on what "in-range" means, so the operator sees motion
            # instead of an immediate estop on out-of-distribution
            # checkpoints.
            # ADR-0028b — when the manifest declares an
            # ``action_contract.slots`` block, the runner dispatches
            # slices of the RAW policy vector onto typed ``Action``
            # objects per the slot's declared ``control_mode``. The
            # joint-permutation + joint-limit clamp path above only
            # applies to legacy single-surface joint_position output;
            # multi-surface slots route each slice to its own HAL
            # channel (cartesian → OSC controller, body twist →
            # /cmd_vel, gripper → gripper actuator).
            ac = getattr(self.manifest, "action_contract", None)
            slots = getattr(ac, "slots", None) if ac is not None else None
            if (
                not slots
                and ac is not None
                and getattr(ac, "representation", None) is not None
                and description is not None
            ):
                # ADR-0036 — a skill that declares only ``representation``
                # (no explicit ``slots``) gets the canonical slot layout
                # for its action space, so the runner dispatches
                # cartesian_delta + gripper instead of defaulting the
                # whole vector to JOINT_POSITION (which the joint-space
                # envelope rejects on franka: ``n_dof 7 != 8``). Joint
                # representations return ``None`` and fall through to the
                # legacy whole-vector JOINT_POSITION path below. The
                # ``description is not None`` guard keeps a no-manifest
                # resolve (cartesian slots need the robot's primary EE) on
                # the legacy path instead of raising in
                # ``canonical_slots_for_representation``.
                from openral_core.schemas import canonical_slots_for_representation

                slots = canonical_slots_for_representation(
                    ac.representation, dim=ac.dim, description=description
                )
            if slots:
                # ``description`` is the closure var from
                # ``_make_policy_adapter_skill``; used to pad sub-slot
                # JOINT_* chunks to full-dof per ADR-0028d.
                return _dispatch_slots(slots, policy_action, description=description)
            if joint_limits and robot_action.shape[0] == len(joint_limits):
                # Strictly INSIDE the envelope — the safety_kernel
                # validates ``value > limit_max`` / ``value < limit_min``
                # (open intervals), so clamping to the exact limit still
                # trips a violation on the boundary. Pull in by a small
                # epsilon (well under any sensor / control precision)
                # to stay safe across float round-trips.
                clamp_eps = 1e-3
                for i, lims in enumerate(joint_limits):
                    if lims is None:
                        continue
                    lo, hi = lims
                    lo_safe = lo + clamp_eps
                    hi_safe = hi - clamp_eps
                    if robot_action[i] < lo_safe:
                        robot_action[i] = lo_safe
                    elif robot_action[i] > hi_safe:
                        robot_action[i] = hi_safe
            return Action(
                control_mode=ControlMode.JOINT_POSITION,
                horizon=1,
                joint_targets=[list(map(float, robot_action))],
            )

    skill = _PolicyAdapterSkill()
    skill.configure()
    skill.activate()
    return skill  # type: ignore[return-value]


def main(args: list[str] | None = None) -> int:
    """Entry point for ``ros2 run openral_rskill_ros rskill_runner_node``.

    Bootstraps the standalone-mode rskill_runner_node: a freshly
    constructed :class:`RobotDescription` stub plus a fresh
    :class:`WorldStateAggregator`. Production launches use
    :func:`openral_rskill_ros.compose.compose_so100_runtime` to share the
    aggregator with the colocated ``world_state_node`` per ADR-0018 §3.
    """
    if not _ROS2_AVAILABLE:
        print("rclpy not found — cannot start rskill_runner_node without ROS 2.", file=sys.stderr)
        return 1

    from openral_core import (
        ControlMode,
        EmbodimentKind,
        JointSpec,
        JointType,
        RobotCapabilities,
        RobotDescription,
        SafetyEnvelope,
    )
    from openral_world_state import WorldStateAggregator

    rclpy.init(args=args)
    description = RobotDescription(
        name="robot",
        embodiment_kind=EmbodimentKind.MANIPULATOR,
        joints=[
            JointSpec(
                name=f"j{i}",
                joint_type=JointType.REVOLUTE,
                parent_link=f"link_{i}",
                child_link=f"link_{i + 1}",
            )
            for i in range(6)
        ],
        capabilities=RobotCapabilities(
            supported_control_modes=[ControlMode.JOINT_POSITION],
        ),
        safety=SafetyEnvelope(),
    )
    aggregator = WorldStateAggregator(description)
    node = RskillRunnerNode(
        robot_description=description,
        aggregator=aggregator,
    )
    try:
        try:
            rclpy.spin(node)
        except (KeyboardInterrupt, ExternalShutdownException):
            # Normal teardown path. rclpy installs a SIGINT handler at
            # `rclpy.init()` that shuts down the context AND raises
            # KeyboardInterrupt out of `rclpy.spin()` on Jazzy. On
            # ROS 2 Rolling / a manual `rclpy.shutdown()` from another
            # thread, spin instead raises ExternalShutdownException.
            # Either way the context is already shut down by the time we
            # reach the `finally` below, so the bare `rclpy.shutdown()`
            # we used to call there raised
            # `RCLError: rcl_shutdown already called` — the
            # `try_shutdown()` switch below is the corresponding fix.
            pass
        finally:
            node.destroy_node()
    finally:
        # Idempotent — no-op when the SIGINT handler (or whoever fired
        # ExternalShutdownException) already shut down the context.
        rclpy.try_shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
