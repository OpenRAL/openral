"""ROS-wrapping rSkill — wraps any ROS 2 action / service as an :class:`rSkillBase`.

Bridges between OpenRAL's :class:`~openral_core.schemas.RSkillManifest` /
:meth:`rSkillBase.step` lifecycle and an arbitrary upstream ROS 2 action
server (MoveIt's :class:`moveit_msgs.action.MoveGroup`, Nav2's
:class:`nav2_msgs.action.NavigateToPose`, …). One adapter, two operating
modes selected by :attr:`RosIntegration.result_trajectory_field`:

* **Trajectory mode** (``result_trajectory_field`` set, e.g. MoveIt):
  on the first :meth:`~rSkillBase.step` call the adapter sends the goal,
  blocks on the result, extracts a
  :class:`trajectory_msgs.msg.JointTrajectory` from the result, reorders
  its joints into the host :class:`RobotDescription`'s joint order, and
  emits one waypoint per subsequent ``step()`` as an
  :class:`~openral_core.schemas.Action` chunk. After the last waypoint
  raises :class:`~openral_core.exceptions.ROSRskillGoalSatisfied`.
* **Result-only mode** (``result_trajectory_field is None``, e.g. Nav2):
  the wrapped action server drives actuators itself (Nav2 publishes
  ``/cmd_vel`` via its behaviour tree). The adapter just awaits the
  result and raises ``ROSRskillGoalSatisfied`` on success on the first
  ``step()`` call.

ROS imports (``rclpy``, the IDL package named in
:attr:`RosIntegration.package`) are deferred to lifecycle hooks so that
schema-level tests and tooling can import this module without an
``ament``-built workspace on ``$PYTHONPATH``.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from importlib import import_module
from typing import TYPE_CHECKING, Any

import structlog
from openral_core.exceptions import (
    ROSConfigError,
    ROSRskillGoalSatisfied,
    ROSRuntimeError,
)
from openral_core.schemas import (
    Action,
    ComputeSpec,
    ControlMode,
    RobotDescription,
    RosIntegration,
    RSkillManifest,
    WorldState,
)

from openral_rskill.base import rSkillBase

if TYPE_CHECKING:
    # `rclpy` and the wrapped IDL package are not importable on every host
    # (schema-only tests, CI without an ament workspace). Type-check time
    # uses `Any` to avoid the hard import.
    pass

__all__ = [
    "CUMOTION_PIPELINE_ID",
    "ROSActionRskill",
    "build_joint_permutation_from_names",
    "maybe_inject_cumotion_pipeline",
]

log = structlog.get_logger(__name__)


# Oversize factor on the manifest's per-chunk latency budget when waiting
# for the wrapped action's result. The budget describes the per-chunk
# inference cost; the *first* call additionally pays planning latency,
# which for MoveIt can be 5-10x the budget on a complex scene. A small
# multiplier keeps the wait bounded without painting the operator into
# a corner.
_RESULT_DEADLINE_MULTIPLIER = 5.0

# Minimum wall-clock floor when no latency budget is declared on the
# manifest (the schema requires one, but a defensive floor protects
# against `per_chunk_ms == 0`).
_MIN_RESULT_DEADLINE_S = 2.0

# Cadence at which we poll rclpy futures while the wrapped server runs.
# Matches the cadence used by `rskill_runner_node._maybe_reset_hal_to_starting_pose`.
_FUTURE_POLL_INTERVAL_S = 0.02

# Time we wait for the wrapped server to come up at configure time
# and to accept the goal. MoveIt's cold-start path includes a planning-
# scene-monitor + FK/IK initialisation that takes several seconds on a
# fresh boot; the safety_kernel + Nav2 are similar. Set conservatively
# at 15 s so a slow first-boot doesn't false-fail; subsequent goals
# accept in <1 s once the server is warm.
_WAIT_FOR_SERVER_TIMEOUT_S = 15.0

# Mirror of ``action_msgs.msg.GoalStatus.STATUS_*`` for human-readable
# failure messages. Constants are integers per the IDL; we don't import
# the IDL at module top because rclpy is optional in some test paths.
# Numbers come from
# ``action_msgs/msg/GoalStatus.msg`` in the ROS 2 source.
_GOAL_STATUS_LABELS: dict[int, str] = {
    0: "STATUS_UNKNOWN",
    1: "STATUS_ACCEPTED",
    2: "STATUS_EXECUTING",
    3: "STATUS_CANCELING",
    4: "STATUS_SUCCEEDED",
    5: "STATUS_CANCELED",
    6: "STATUS_ABORTED",
}


def _merge_nested(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Deep-merge ``overrides`` onto ``base``; return a new dict.

    ADR-0026 — used by :class:`ROSActionRskill._configure_impl` to merge
    the LLM's ``goal_params_json`` over the manifest's
    ``ros_integration.default_goal_json``. Semantics:

    * Both values at a key are dicts → recurse.
    * Otherwise → ``overrides[key]`` replaces ``base[key]`` verbatim.
    * Keys only in ``base`` are preserved.
    * Keys only in ``overrides`` are added.

    Arrays are NOT element-wise merged — the LLM specifies either the
    full array or none of it. Element-wise merge would surprise a
    palette designer expecting position-by-index semantics.
    """
    out: dict[str, Any] = dict(base)  # shallow copy; recursive copies happen below
    for key, override_val in overrides.items():
        base_val = out.get(key)
        if isinstance(base_val, dict) and isinstance(override_val, dict):
            out[key] = _merge_nested(base_val, override_val)
        else:
            out[key] = override_val
    return out


# The cuMotion MoveIt planning-pipeline id (ADR-0065). Selecting it is a per-request
# ``MotionPlanRequest.pipeline_id`` — OpenRAL does not bring up ``move_group``; the
# user's moveit_config must load this pipeline (a `ros_dependency` + config snippet).
CUMOTION_PIPELINE_ID = "isaac_ros_cumotion"


def maybe_inject_cumotion_pipeline(
    goal_dict: dict[str, Any],
    *,
    interface_type: str,
    compute: ComputeSpec | None,
) -> dict[str, Any]:
    """Return ``goal_dict`` with ``request.pipeline_id`` set to cuMotion when gated on.

    cuMotion is a MoveIt planning-pipeline plugin selected per request via
    ``moveit_msgs/MotionPlanRequest.pipeline_id`` (ADR-0065 D1). When the host
    clears the cuMotion GPU floor (:meth:`ComputeSpec.supports_cumotion`)
    and the wrapped action is ``MoveGroup``, this injects the cuMotion pipeline id
    into the goal's ``request`` block; otherwise the goal is returned unchanged so
    MoveIt uses its default pipeline (OMPL).

    Never overwrites an explicit ``pipeline_id`` (a manifest / LLM choice wins),
    never mutates the input, and is a no-op when there is no ``request`` block.
    """
    if interface_type != "MoveGroup":
        return goal_dict
    if compute is None or not compute.supports_cumotion():
        return goal_dict
    request = goal_dict.get("request")
    if not isinstance(request, dict) or request.get("pipeline_id"):
        return goal_dict
    return {**goal_dict, "request": {**request, "pipeline_id": CUMOTION_PIPELINE_ID}}


def build_joint_permutation_from_names(
    *,
    source_names: Sequence[str],
    target_names: Sequence[str],
) -> tuple[list[int], list[int]]:
    """Build the permutation that maps a wrapped server's joints onto the host robot.

    The wrapped action (MoveIt) returns a
    :class:`trajectory_msgs.msg.JointTrajectory` whose ``joint_names``
    list orders the per-point ``positions`` array. The
    :class:`~openral_core.schemas.RobotDescription`'s ``joints`` list
    orders the safety supervisor's envelope check and the HAL's wire
    ``ActionChunk``. The two are not guaranteed to match — MoveIt uses
    its ``JointModelGroup`` ordering, which is configuration-driven,
    and the robot description typically carries additional joints the
    planner doesn't move (gripper, head pan, …). Without a reorder
    the supervisor checks the wrong joint against the wrong envelope
    limit (``packages/openral_safety/openral_safety/supervisor_node.py``);
    silently mis-applied joint targets would be a safety-critical bug.

    The wrapped server's joint list MUST be a (non-strict) subset of
    the host's joint list. Slots in ``target_names`` that don't appear
    in ``source_names`` are returned in the second tuple element so
    the caller (typically
    :meth:`ROSActionRskill._dispatch_and_cache_result`) knows which
    slots to backfill from the host's current
    :attr:`~openral_core.schemas.WorldState.joint_state` rather than
    leaving them undefined.

    Args:
        source_names: Joint names in the wrapped server's order.
        target_names: Joint names in the host's
            :class:`RobotDescription`'s order.

    Returns:
        A pair ``(perm, unmoved_indices)``:

        * ``perm`` — length ``len(target_names)``. ``perm[i]`` is the
          index into ``source_names`` corresponding to
          ``target_names[i]``, or ``-1`` if ``target_names[i]`` is not
          in ``source_names``.
        * ``unmoved_indices`` — sorted list of every ``i`` for which
          ``perm[i] == -1``. The host robot's joints at those indices
          stay at their current value (read from the live
          ``WorldState.joint_state``) — the wrapped planner did not
          schedule motion for them.

    Raises:
        ROSConfigError: If any joint in ``source_names`` is missing
            from ``target_names`` (set-extra on the source side).
            Surfaced loudly so an operator catches a planner that's
            commanding a joint the host robot doesn't know about
            instead of silently swapping bytes.
    """
    source_set = set(source_names)
    target_set = set(target_names)
    extra_in_source = source_set - target_set
    if extra_in_source:
        raise ROSConfigError(
            "Wrapped server commands joints the host robot does not have. "
            f"source={list(source_names)!r} target={list(target_names)!r} "
            f"extra_in_source={sorted(extra_in_source)!r}"
        )
    source_index = {n: i for i, n in enumerate(source_names)}
    perm: list[int] = [source_index.get(n, -1) for n in target_names]
    unmoved_indices: list[int] = [i for i, j in enumerate(perm) if j == -1]
    return perm, unmoved_indices


def _resolve_dotted(obj: Any, dotted: str) -> Any:  # noqa: ANN401  # reason: walks arbitrary IDL slots
    """Walk ``obj.a.b.c`` from a dotted accessor string.

    Used to extract the per-action ``JointTrajectory`` slot from a
    wrapped server's result message (e.g.
    ``planned_trajectory.joint_trajectory`` for MoveIt's
    ``MoveGroup``).
    """
    cur = obj
    for part in dotted.split("."):
        cur = getattr(cur, part)
    return cur


def _import_action_or_service(integration: RosIntegration) -> tuple[type, str]:
    """Lazy-import the IDL named in :attr:`RosIntegration.package`.

    Returns the type plus the import kind (``"action"`` or ``"service"``).
    Raises :class:`ROSConfigError` quoting ``ros_dependencies`` when the
    import fails.
    """
    for sub in ("action", "srv"):
        module_name = f"{integration.package}.{sub}"
        try:
            module = import_module(module_name)
        except ImportError:
            continue
        type_obj = getattr(module, integration.interface_type, None)
        if type_obj is not None:
            return type_obj, "action" if sub == "action" else "service"
    deps = ", ".join(integration.ros_dependencies) or "<none declared>"
    raise ROSConfigError(
        f"Cannot import {integration.package!r}.action.{integration.interface_type} "
        f"or {integration.package!r}.srv.{integration.interface_type}. "
        f"Install the required ROS package(s): {deps}."
    )


def _set_message_fields(msg: Any, data: dict[str, Any]) -> None:  # noqa: ANN401  # reason: walks arbitrary IDL slots
    """Apply a JSON dict to a ROS 2 message instance, recursively.

    Prefers ``rosidl_runtime_py.set_message_fields`` (the canonical
    ROS 2 utility) when available; falls back to a recursive
    setattr walk so this module can be unit-tested against synthetic
    message classes that don't sit under the ``rosidl_runtime_py``
    introspection surface.
    """
    try:
        from rosidl_runtime_py import set_message_fields  # noqa: PLC0415
    except ImportError:
        for key, value in data.items():
            current = getattr(msg, key, None)
            if isinstance(value, dict) and current is not None and hasattr(current, "__slots__"):
                _set_message_fields(current, value)
            else:
                setattr(msg, key, value)
        return
    set_message_fields(msg, data)


class ROSActionRskill(rSkillBase):
    """rSkill adapter that wraps a ROS 2 action or service server.

    Constructed by the local skill resolver (``rskill_runner_node.
    make_default_skill_resolver``) when ``manifest.kind in
    {"ros_action", "ros_service"}``. The adapter is owned by the
    ``RskillRunnerNode``'s ``execute_cb`` worker thread; ``ros_node`` is
    the same lifecycle node that hosts the ``ExecuteSkill`` action
    server, so the wrapped action client lives in the same rclpy
    executor and its futures are serviced by the node's existing spin.

    Args:
        manifest: The validated rSkill manifest. ``manifest.kind`` must
            be ``"ros_action"`` or ``"ros_service"``;
            ``manifest.ros_integration`` must be set.
        ros_node: The host ``rclpy.lifecycle.LifecycleNode`` (or plain
            ``rclpy.node.Node`` in tests). Used to create the wrapped
            ``ActionClient`` / service client so its futures share the
            node's executor.
        robot_description: The host robot's :class:`RobotDescription`.
            Optional in test paths that skip joint reordering; in
            production this is the description the
            ``RskillRunnerNode`` was constructed with. Used to align
            the wrapped server's joint order with the supervisor's
            envelope index.
        prompt: The ``ExecuteSkill`` goal's free-form prompt. Logged
            with the request but not consumed by v1 — the goal payload
            comes from ``ros_integration.default_goal_json``.
        prompt_metadata_json: The ``ExecuteSkill`` goal's structured
            payload. Reserved for the follow-up structured-prompt path;
            currently logged but not consumed.
    """

    def __init__(
        self,
        *,
        manifest: RSkillManifest,
        ros_node: Any,  # noqa: ANN401  # reason: rclpy.node.Node not importable on hosts without a sourced ROS 2 workspace; the only consumer is `_configure_impl` which lazy-imports rclpy
        robot_description: RobotDescription | None,
        prompt: str,
        prompt_metadata_json: str,
        goal_params_json: str = "",
    ) -> None:
        """Initialise; defers all ROS-side work to :meth:`_configure_impl`."""
        if manifest.ros_integration is None:
            raise ROSConfigError(
                f"ROSActionRskill requires manifest.ros_integration (kind={manifest.kind!r}); "
                "this manifest declares no integration block."
            )
        super().__init__(
            name=manifest.name,
            version=manifest.version,
            role=manifest.role,
            embodiment_tags=list(manifest.embodiment_tags),
            latency_budget_ms=(
                manifest.latency_budget.per_chunk_ms
                if manifest.latency_budget is not None
                else None
            ),
        )
        self.manifest = manifest
        self._integration: RosIntegration = manifest.ros_integration
        self._node = ros_node
        self._description = robot_description
        self._prompt = prompt
        self._prompt_metadata_json = prompt_metadata_json
        # ADR-0026 — per-dispatch JSON object merged over
        # ``ros_integration.default_goal_json`` at configure-time. Empty
        # = today's behaviour (the manifest default is sent verbatim).
        self._goal_params_json = goal_params_json

        # Filled in by `_configure_impl` / first `_step_impl`.
        self._interface_kind: str = ""  # "action" | "service"
        self._interface_type: type | None = None
        self._client: Any = None
        self._goal_dict: dict[str, Any] = {}
        # Cached trajectory in robot-order. Filled on the first step in
        # trajectory mode; empty in result-only mode.
        self._waypoints: list[list[float]] = []
        self._waypoint_index: int = 0
        # Set once the wrapped action's result has been awaited.
        self._result_consumed: bool = False
        # Per-slot values used to populate joints the wrapped planner
        # doesn't move (e.g. a gripper slot when MoveIt plans for an
        # arm-only `JointModelGroup`). Captured from the live
        # WorldState on the first step before the goal is sent; all
        # zeroes if no description / no live snapshot is available
        # (test-only path).
        n_target = len(robot_description.joints) if robot_description is not None else 0
        self._unmoved_joint_padding: list[float] = [0.0] * n_target
        # Result-only deadline budget (s) — sourced from latency budget.
        budget_ms = (
            manifest.latency_budget.per_chunk_ms if manifest.latency_budget is not None else None
        )
        if budget_ms is None or budget_ms <= 0.0:
            self._result_deadline_s = _MIN_RESULT_DEADLINE_S
        else:
            self._result_deadline_s = max(
                _MIN_RESULT_DEADLINE_S,
                (budget_ms / 1000.0) * _RESULT_DEADLINE_MULTIPLIER,
            )

    # ── Lifecycle hooks ──────────────────────────────────────────────────────

    def _configure_impl(self) -> None:
        """Build the wrapped ActionClient / service client; parse the goal JSON.

        ROS imports happen here so the schema-only test surface can
        construct ``ROSActionRskill`` without ``rclpy`` on the path.
        """
        type_obj, kind = _import_action_or_service(self._integration)
        self._interface_type = type_obj
        self._interface_kind = kind

        try:
            self._goal_dict = json.loads(self._integration.default_goal_json)
        except json.JSONDecodeError as exc:
            # The schema validator catches this at manifest-load time; this
            # branch is defence in depth for callers who synthesise a manifest
            # outside the Pydantic validator (test fixtures, mostly).
            raise ROSConfigError(
                f"ROSActionRskill({self.name!r}): default_goal_json is not valid JSON: {exc}"
            ) from exc

        # ADR-0026 — deep-merge per-dispatch goal_params_json over the
        # manifest's default_goal_json. Overrides win at leaves; nested
        # dicts recurse; arrays + scalars replace verbatim (no
        # element-wise merge — too surprising). Empty string = no merge.
        if self._goal_params_json:
            try:
                overrides = json.loads(self._goal_params_json)
            except json.JSONDecodeError as exc:
                raise ROSConfigError(
                    f"ROSActionRskill({self.name!r}): goal_params_json from the "
                    f"reasoner / action goal is not valid JSON: {exc}"
                ) from exc
            if not isinstance(overrides, dict):
                raise ROSConfigError(
                    f"ROSActionRskill({self.name!r}): goal_params_json must "
                    f"decode to a JSON object; got {type(overrides).__name__}."
                )
            self._goal_dict = _merge_nested(self._goal_dict, overrides)

        # ADR-0065 D1 — on a host that clears the cuMotion GPU floor, select the
        # cuMotion MoveIt pipeline for MoveGroup goals (per-request `pipeline_id`).
        # No-op for non-MoveGroup actions and on CPU/low-VRAM hosts (MoveIt then
        # uses its default OMPL pipeline). An explicit pipeline_id still wins.
        self._goal_dict = maybe_inject_cumotion_pipeline(
            self._goal_dict,
            interface_type=self._integration.interface_type,
            compute=(self._description.compute_edge or self._description.compute_local)
            if self._description is not None
            else None,
        )

        if kind == "action":
            try:
                from rclpy.action import ActionClient  # noqa: PLC0415
            except ImportError as exc:
                raise ROSConfigError(
                    f"ROSActionRskill({self.name!r}): rclpy.action is unavailable. "
                    "Build/source a ROS 2 workspace before resolving wrapped skills."
                ) from exc
            client = ActionClient(self._node, type_obj, self._integration.interface_name)
            if not client.wait_for_server(timeout_sec=_WAIT_FOR_SERVER_TIMEOUT_S):
                client.destroy()
                deps = ", ".join(self._integration.ros_dependencies) or "<none declared>"
                raise ROSConfigError(
                    f"ROSActionRskill({self.name!r}): action server "
                    f"{self._integration.interface_name!r} did not come up within "
                    f"{_WAIT_FOR_SERVER_TIMEOUT_S}s. Required ROS packages: {deps}."
                )
            self._client = client
        else:  # service
            client = self._node.create_client(type_obj, self._integration.interface_name)
            if not client.wait_for_service(timeout_sec=_WAIT_FOR_SERVER_TIMEOUT_S):
                self._node.destroy_client(client)
                deps = ", ".join(self._integration.ros_dependencies) or "<none declared>"
                raise ROSConfigError(
                    f"ROSActionRskill({self.name!r}): service "
                    f"{self._integration.interface_name!r} did not come up within "
                    f"{_WAIT_FOR_SERVER_TIMEOUT_S}s. Required ROS packages: {deps}."
                )
            self._client = client

        log.info(
            "ros_action_rskill.configured",
            name=self.name,
            interface_kind=kind,
            interface_name=self._integration.interface_name,
            interface_type=self._integration.interface_type,
            trajectory_field=self._integration.result_trajectory_field,
        )

    def _activate_impl(self) -> None:
        """No-op — the wrapped action is dispatched on the first ``step``."""

    def _deactivate_impl(self) -> None:
        """Pause execution. The client stays connected for the next ``activate``."""

    def _shutdown_impl(self) -> None:
        """Release the wrapped ActionClient / service client."""
        if self._client is None:
            return
        try:
            if self._interface_kind == "action":
                self._client.destroy()
            else:
                self._node.destroy_client(self._client)
        finally:
            self._client = None

    # ── Hot path ─────────────────────────────────────────────────────────────

    def _step_impl(self, world_state: WorldState) -> Action:
        """Return one ``Action`` chunk per call until ``ROSRskillGoalSatisfied``.

        Trajectory mode emits one cached waypoint per call; result-only
        mode raises ``ROSRskillGoalSatisfied`` on the first call (after
        awaiting the wrapped action's result), since no joint targets
        flow through OpenRAL's actuation path in that case.

        Joints in the host :class:`RobotDescription` that the wrapped
        planner does NOT move (e.g. the panda_gripper slot when MoveIt
        plans for the ``panda_arm`` group) are backfilled with their
        current positions from ``world_state.joint_state`` on the
        first call, then held constant across every emitted waypoint.
        """
        is_trajectory_mode = self._integration.result_trajectory_field is not None

        if not self._result_consumed:
            # Capture the live joint positions for the slots the wrapped
            # planner won't move. This MUST happen before we send the
            # goal so an in-flight motion doesn't change the gripper
            # value mid-cache.
            self._capture_unmoved_joint_padding(world_state)
            self._dispatch_and_cache_result()
            self._result_consumed = True
            if not is_trajectory_mode:
                # Result-only mode: nothing to actuate via OpenRAL.
                raise ROSRskillGoalSatisfied(
                    f"{self.name}: wrapped result-only action completed (no trajectory)."
                )

        if self._waypoint_index >= len(self._waypoints):
            raise ROSRskillGoalSatisfied(
                f"{self.name}: emitted all {len(self._waypoints)} waypoints."
            )

        waypoint = self._waypoints[self._waypoint_index]
        self._waypoint_index += 1
        return Action(
            control_mode=ControlMode.JOINT_POSITION,
            horizon=1,
            joint_targets=[waypoint],
        )

    # ── Internals ────────────────────────────────────────────────────────────

    def _capture_unmoved_joint_padding(self, world_state: Any) -> None:  # noqa: ANN401
        """Snapshot current joint positions for slots the wrapped planner won't move.

        Reads ``world_state.joint_state.position`` (a list of floats in
        :attr:`RobotDescription.joints` order) when available; falls
        back to all-zeroes when ``world_state`` is ``None`` (test path
        only — production always supplies a live snapshot).
        """
        if self._description is None or world_state is None:
            return  # init-time zeros already in place
        try:
            positions = list(world_state.joint_state.position)
        except AttributeError:
            return
        if len(positions) != len(self._description.joints):
            log.warning(
                "ros_action_rskill.padding_size_mismatch",
                name=self.name,
                got=len(positions),
                expected=len(self._description.joints),
            )
            return
        self._unmoved_joint_padding = [float(v) for v in positions]

    def _dispatch_and_cache_result(self) -> None:
        """Send the wrapped goal and block on the result.

        Trajectory mode caches the reordered waypoint list in
        ``self._waypoints``. Result-only mode just confirms success and
        returns. Any failure surfaces as :class:`ROSRuntimeError`.
        """
        if self._interface_kind == "action":
            result = self._send_action_goal_and_await_result()
        else:
            result = self._call_service_and_await_response()

        if self._integration.result_trajectory_field is None:
            return  # result-only — nothing to extract

        try:
            jtraj = _resolve_dotted(result, self._integration.result_trajectory_field)
        except AttributeError as exc:
            raise ROSRuntimeError(
                f"ROSActionRskill({self.name!r}): cannot resolve "
                f"result.{self._integration.result_trajectory_field} on the wrapped "
                f"result ({type(result).__name__}): {exc}"
            ) from exc

        # `trajectory_msgs/JointTrajectory` carries `joint_names: list[str]`
        # and `points: list[JointTrajectoryPoint]`, each with a
        # `positions: list[float]`. Reorder positions into the host
        # robot's joint order so the safety supervisor's per-index
        # envelope check looks at the right joint.
        source_names: list[str] = list(getattr(jtraj, "joint_names", []) or [])
        points = list(getattr(jtraj, "points", []) or [])
        if not source_names or not points:
            # Surface every diagnostic the wrapped server provided — for
            # MoveIt this includes ``result.error_code.val`` (MoveItErrorCodes
            # enum) which is the difference between "planner failed" and
            # "no motion needed". Without it the operator sees only "empty
            # trajectory" and can't tell which.
            ec = getattr(result, "error_code", None)
            ec_val = getattr(ec, "val", None) if ec is not None else None
            planning_time = getattr(result, "planning_time", None)
            raise ROSRuntimeError(
                f"ROSActionRskill({self.name!r}): wrapped trajectory is empty "
                f"(joint_names={source_names!r}, n_points={len(points)}, "
                f"error_code={ec_val!r}, planning_time={planning_time!r}). "
                f"Inspect the full result with `ros2 action send_goal "
                f"{self._integration.interface_name} {self._integration.package}/"
                f"action/{self._integration.interface_type}` against the same goal "
                "for the operator-facing error string."
            )

        if self._description is not None:
            target_names = [j.name for j in self._description.joints]
            perm, unmoved_indices = build_joint_permutation_from_names(
                source_names=source_names,
                target_names=target_names,
            )
            # Build each waypoint by:
            #   1. starting from the cached padding (current positions
            #      for unmoved slots; 0.0 by default for moved slots);
            #   2. overwriting the moved slots from the trajectory.
            # Unmoved slots stay at their captured value across every
            # emitted waypoint — the wrapped planner did not schedule
            # motion for them, so we MUST NOT command motion either,
            # or the safety supervisor's per-row check sees a spurious
            # delta on those joints.
            self._waypoints = []
            for p in points:
                wp = list(self._unmoved_joint_padding)
                for i, j in enumerate(perm):
                    if j >= 0:
                        wp[i] = float(p.positions[j])
                self._waypoints.append(wp)
            if unmoved_indices:
                log.info(
                    "ros_action_rskill.unmoved_joints",
                    name=self.name,
                    unmoved_count=len(unmoved_indices),
                    unmoved_names=[target_names[i] for i in unmoved_indices],
                )
        else:
            # No RobotDescription to align against — pass positions through
            # unchanged. Downstream consumers MUST verify the joint order
            # matches their HAL's expectation. Production use always
            # supplies a description; this branch exists only for unit
            # tests that exercise the adapter without a host description.
            self._waypoints = [[float(v) for v in p.positions] for p in points]

        log.info(
            "ros_action_rskill.trajectory_cached",
            name=self.name,
            n_waypoints=len(self._waypoints),
            n_joints=len(self._waypoints[0]) if self._waypoints else 0,
        )

    def _send_action_goal_and_await_result(self) -> Any:  # noqa: ANN401  # reason: arbitrary IDL result type
        """Send the cached goal, poll the goal handle + result futures."""
        assert self._interface_type is not None
        goal_msg = self._interface_type.Goal()  # type: ignore[attr-defined]  # reason: dynamically imported action IDL class — every rosidl action type carries a Goal nested class
        try:
            _set_message_fields(goal_msg, self._goal_dict)
        except Exception as exc:  # reason: IDL setattr can raise various TypeError shapes
            # Almost always a goal_params_json that doesn't match the IDL nesting
            # (e.g. an LLM flattening NavigateToPose's pose.pose.{position,
            # orientation} up to the PoseStamped). Echo the merged goal keys so
            # the propagated failure_reason lets the reasoner correct on replan.
            raise ROSConfigError(
                f"ROSActionRskill({self.name!r}): failed to apply goal params "
                f"to {self._interface_type.__name__}.Goal — {exc}. The merged goal "
                f"must match the IDL field nesting; got top-level keys "
                f"{sorted(self._goal_dict)}. Check goal_params_json against the "
                f"manifest's goal_params_schema (each level must nest, not flatten)."
            ) from exc

        goal_future = self._client.send_goal_async(goal_msg)
        self._poll_future(goal_future, deadline_s=_WAIT_FOR_SERVER_TIMEOUT_S, what="goal-accept")
        goal_handle = goal_future.result()
        if goal_handle is None or not getattr(goal_handle, "accepted", False):
            raise ROSRuntimeError(
                f"ROSActionRskill({self.name!r}): wrapped server rejected the goal."
            )

        result_future = goal_handle.get_result_async()
        self._poll_future(result_future, deadline_s=self._result_deadline_s, what="result")
        wrapper = result_future.result()
        if wrapper is None:
            raise ROSRuntimeError(
                f"ROSActionRskill({self.name!r}): wrapped server returned a null result."
            )
        # ``wrapper`` is a ``GetResult_Response`` with both ``.result``
        # (the IDL action result) AND ``.status`` (an ``action_msgs.msg.
        # GoalStatus`` enum). Without this check, a Nav2 goal that
        # aborts (planner can't reach the target, costmap rejects the
        # pose, controller times out, …) is indistinguishable from
        # success: the wrapper still completes, just with
        # ``status=STATUS_ABORTED=6``. Result-only mode would then
        # raise ``ROSRskillGoalSatisfied`` and the reasoner would log
        # ``execute_rskill succeeded`` on a failed navigation.
        # ``action_msgs`` is a runtime-only IDL; load lazily and fall
        # back to integer literals when rclpy isn't sourced (the
        # production path always has it, this branch only protects
        # unit tests where ``action_msgs`` may be missing).
        try:
            from action_msgs.msg import GoalStatus  # noqa: PLC0415

            status_succeeded = GoalStatus.STATUS_SUCCEEDED
            status_unknown = GoalStatus.STATUS_UNKNOWN
        except ImportError:
            status_succeeded = 4
            status_unknown = 0
        status = getattr(wrapper, "status", status_unknown)
        if status != status_succeeded:
            label = _GOAL_STATUS_LABELS.get(status, f"unknown({status!r})")
            # The Nav2 result IDL carries an ``error_code`` + ``error_msg``;
            # MoveIt's ``MoveGroup.Result`` carries ``error_code.val``. Try
            # both shapes so the failure message is actionable.
            err_msg = getattr(wrapper.result, "error_msg", "") or ""
            err_code = getattr(wrapper.result, "error_code", None)
            err_code_str = ""
            err_code_val: Any = None
            if err_code is not None:
                err_code_val = getattr(err_code, "val", err_code)
                err_code_str = f" error_code={err_code_val!r}"
            # A generic abort with no error_code/message (Nav2 BT abort, the
            # common case) is almost always operational, not a bug: the goal is
            # unreachable (in furniture / outside the known free space), the
            # costmap is not yet populated, or the controller could not make
            # progress. Spell that out so the log is actionable instead of an
            # opaque "status=STATUS_ABORTED error_code=0".
            hint = ""
            if not err_msg and err_code_val in (None, 0):
                hint = (
                    " (no error_code/message from the server — typically an unreachable goal, "
                    "an unpopulated/too-small costmap, or the controller failing to make progress)"
                )
            raise ROSRuntimeError(
                f"ROSActionRskill({self.name!r}): wrapped server returned "
                f"status={label}{err_code_str}"
                + (f" error_msg={err_msg!r}" if err_msg else "")
                + hint
            )
        return wrapper.result

    def _call_service_and_await_response(self) -> Any:  # noqa: ANN401  # reason: arbitrary IDL response
        """Send the cached service request, poll the response future."""
        assert self._interface_type is not None
        request = self._interface_type.Request()  # type: ignore[attr-defined]  # reason: dynamically imported service IDL class — every rosidl srv type carries a Request nested class
        try:
            _set_message_fields(request, self._goal_dict)
        except Exception as exc:  # reason: IDL setattr can raise various TypeError shapes
            raise ROSConfigError(
                f"ROSActionRskill({self.name!r}): failed to apply default_goal_json "
                f"to {self._interface_type.__name__}.Request — check field names / types: "
                f"{exc}"
            ) from exc
        future = self._client.call_async(request)
        self._poll_future(future, deadline_s=self._result_deadline_s, what="service-response")
        response = future.result()
        if response is None:
            raise ROSRuntimeError(f"ROSActionRskill({self.name!r}): wrapped service returned null.")
        return response

    def _poll_future(self, future: Any, *, deadline_s: float, what: str) -> None:  # noqa: ANN401  # reason: rclpy.task.Future is untyped
        """Block (without spinning) until ``future.done()`` or deadline.

        Mirrors the pattern in
        ``rskill_runner_node._maybe_reset_hal_to_starting_pose``: the host
        node's main rclpy spin services callbacks; we just poll
        ``done()`` here. Avoids re-entering ``rclpy.spin_until_future_complete``
        from a worker thread, which is unsafe with the default
        single-threaded executor.
        """
        deadline = time.monotonic() + deadline_s
        while not future.done():
            if time.monotonic() >= deadline:
                raise ROSRuntimeError(
                    f"ROSActionRskill({self.name!r}): wrapped {what} did not "
                    f"complete within {deadline_s:.1f}s."
                )
            time.sleep(_FUTURE_POLL_INTERVAL_S)
