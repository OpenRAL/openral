#!/usr/bin/env python3
"""ADR-0018 F4 — ``reasoner_node`` lifecycle wrapper.

Subscribes to:

* ``/openral/world_state_slow``  — ``openral_msgs/WorldStateStamped``, 5 Hz
* ``/openral/failure/{hal,sensor,rskill,safety,wam,critic}``  — ``openral_msgs/FailureTrigger``
* ``/openral/perception/{motion,objects,ocr,scene_change}``  — ``openral_msgs/PromptStamped``
* ``/openral/prompt``            — ``openral_msgs/PromptStamped`` (operator)

Heartbeat tick at ``tick_hz`` (default 0.2 Hz = one every 5 s; was 5 Hz
pre-2026-05-25 amendment to ADR-0018). The event bus is the primary
trigger: an incoming :class:`FailureTrigger` with
``severity>=SEVERITY_FAIL`` (or ``>=SEVERITY_WARN`` on
``/openral/failure/safety`` — Tier A), or a new ``/openral/prompt``
arrival, forces an out-of-band tick (subject to the
:class:`~openral_reasoner.ReasonerCore` 100 ms min-interval per
ADR-0018 §4). Heartbeat ticks that see no new event since the last
successful tick are short-circuited inside ``ReasonerCore`` with
``suppressed_reason="heartbeat_idle"``.

Dispatches the selected :data:`~openral_core.ReasonerToolCall`:

* :class:`ExecuteRskillTool` → action goal on
  ``/openral/execute_rskill`` (the F1 ``rskill_runner_node`` server).
  Streams feedback to the structlog warning channel, and emits a
  :class:`~openral_msgs.msg.FailureTrigger` on
  ``/openral/failure/rskill`` (``KIND_CONTROLLER`` for
  rejection/abort/server-unavailable; ``KIND_TIMEOUT`` when the
  ``deadline_s`` elapses before the server returns a result).
* :class:`LifecycleTransitionTool` → service call on
  ``<node>/change_state`` (``lifecycle_msgs/srv/ChangeState``). The
  ``"configure"`` / ``"activate"`` / ``"deactivate"`` / ``"cleanup"``
  strings are mapped to the matching ``Transition.TRANSITION_*``
  constants; ``"shutdown"`` is deliberately absent from the palette
  per CLAUDE.md §6 Layer 6.
* :class:`ReloadGstPipelineTool` → service call on
  ``/openral/sensors/<sensor_id>/reload_pipeline``. **Deferred** — the
  F6 sensor-package service IDL is not yet on disk; this branch logs
  a warning and acknowledges the call. Wired in a follow-up PR once
  the F6 sensor packages land.
* :class:`EmitPromptTool` → republish on the target ``PromptStamped``
  topic.

The reasoner **never** publishes ``openral_msgs/ActionChunk`` (ADR-0018
§4 "Holds no authority over actuation").
"""

from __future__ import annotations

import contextlib
import datetime
import json
import pathlib
import sys
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from openral_world_state import SpatialMemory

import rclpy
from openral_core import (
    SIM_EXECUTABLE_CONTROL_MODES,
    ControllerEvidence,
    ControlMode,
    DecomposeMissionTool,
    EmitPromptTool,
    ExecuteRskillTool,
    LifecycleTransitionTool,
    LocateInViewTool,
    MemorySearchTool,
    MemoryWriteTool,
    ObjectsMetadata,
    QuerySceneTool,
    QueryTaskProgressTool,
    RecallObjectTool,
    ReloadGstPipelineTool,
    ResolvePlaceTool,
    RewardContract,
    RobotCapabilities,
    RobotDescription,
    RSkillManifest,
    TimeoutEvidence,
    control_modes_for_representation,
    is_collective_target,
)
from openral_core.exceptions import ROSConfigError, ROSReasonerInvalidPlan
from openral_observability import log_lifecycle_errors
from openral_reasoner.active_search import SearchBudget, SearchProgress
from openral_reasoner.completion import (
    COMPLETION_QUESTION as _COMPLETION_QUESTION,
)
from openral_reasoner.completion import (
    image_msg_to_jpeg as _image_msg_to_jpeg,
)
from openral_reasoner.completion import (
    is_reward_wake as _is_reward_wake,
)
from openral_reasoner.completion import (
    parse_yes_no as _parse_yes_no,
)
from openral_reasoner.completion import (
    resolve_band_edges as _resolve_band_edges,
)
from openral_reasoner.completion import (
    resolve_patience_s as _resolve_patience_s,
)
from openral_reasoner.context import (
    ContextRenderer,
    ExecutionEventRecord,
    FailureEventRecord,
    PerceptionEventRecord,
    PromptRecord,
    reflect_on_failure,
    reflect_on_invalid_plan,
    reflect_on_retry_cap,
    reflect_on_reward_plateau,
    render_playbooks_block,
    render_robot_self_model,
)
from openral_reasoner.core import ReasonerCore
from openral_reasoner.memory import MemoryEntry, MemoryStore
from openral_reasoner.mission import (
    DEFAULT_MAX_SUBDIVIDE_DEPTH,
    MissionState,
    TaskState,
    evaluate_task_verdict,
)
from openral_reasoner.palette import (
    ToolPalette,
    build_tool_palette,
    locate_in_view_service,
    task_space_disagreement,
)
from openral_reasoner.spatial_query import SpatialMemoryQuerier, run_spatial_query_detailed
from openral_reasoner.tool_use import (
    ToolUseClient,
    build_tool_use_client_from_env,
    resolve_reasoner_system_prompt,
)
from pydantic import ValidationError
from rclpy.executors import ExternalShutdownException
from rclpy.lifecycle import LifecycleNode, LifecycleState, TransitionCallbackReturn
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)

# Imports below are pinned because the ROS-generated IDL is a runtime
# dep — the openral_msgs Python module only exists after a colcon
# build. Tests construct ``ReasonerNode`` after sourcing ``install/``.
try:  # pragma: no cover — gated by colcon-built artifact
    # The action was renamed ExecuteSkill → ExecuteRskill in the skill→rskill
    # rename (#262); the runner serves it as `/openral/execute_rskill`. Import
    # the current name so on_configure does not abort with a misleading
    # "openral_msgs not on PYTHONPATH" when only this symbol moved.
    from openral_msgs.action import ExecuteRskill as IDLExecuteRskill
    from openral_msgs.msg import FailureTrigger as IDLFailureTrigger
    from openral_msgs.msg import PromptStamped as IDLPromptStamped
    from openral_msgs.msg import WorldStateStamped as IDLWorldStateStamped
except ImportError:  # pragma: no cover — only firing when openral_msgs absent
    IDLExecuteRskill = None  # type: ignore[assignment, misc]
    IDLFailureTrigger = None  # type: ignore[assignment, misc]
    IDLPromptStamped = None  # type: ignore[assignment, misc]
    IDLWorldStateStamped = None  # type: ignore[assignment, misc]

# lifecycle_msgs ships with ROS 2; the LifecycleTransitionTool dispatcher
# uses srv/ChangeState + Transition.TRANSITION_* constants.
try:  # pragma: no cover — gated by sourced ROS install
    from lifecycle_msgs.msg import Transition as IDLTransition
    from lifecycle_msgs.srv import ChangeState as IDLChangeState
except ImportError:  # pragma: no cover
    IDLChangeState = None  # type: ignore[assignment, misc]
    IDLTransition = None  # type: ignore[assignment, misc]

# std_msgs ships with ROS 2 Jazzy; this is the empty payload the
# ``ral skill install`` / ``ral skill remove`` CLI fires on
# ``/openral/skill_registry_changed`` to invalidate the reasoner's
# palette (ADR-0018 §4 "palette ... refreshed on
# /openral/skill_registry_changed").
try:  # pragma: no cover — gated by sourced ROS install
    from std_msgs.msg import Empty as IDLEmpty
except ImportError:  # pragma: no cover
    IDLEmpty = None  # type: ignore[assignment, misc]


__all__ = ["ReasonerNode"]

# QoS profiles per ADR-0018 §1 + CLAUDE.md §5.3
_QOS_WORLD_STATE = QoSProfile(
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.VOLATILE,
)
_QOS_FAILURE = QoSProfile(
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=50,
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.VOLATILE,
)
_QOS_PERCEPTION = QoSProfile(
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=10,
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    durability=QoSDurabilityPolicy.VOLATILE,
)
_QOS_PROMPT = QoSProfile(
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=10,
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.VOLATILE,
)
# /openral/skill_registry_changed is a rare event (a ral skill install /
# remove fires it once) — RELIABLE+TRANSIENT_LOCAL so a late-subscribing
# reasoner doesn't miss the most recent invalidation.
_QOS_REGISTRY_CHANGED = QoSProfile(
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
)
# ADR-0044 Phase 4 — the slam_toolbox map is latched (description/static QoS
# class): RELIABLE + TRANSIENT_LOCAL so a late-joining reasoner still receives
# the current grid snapshot.
_QOS_MAP = QoSProfile(
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
)

# ADR-0074 §5 — BEST_EFFORT sensor QoS for the completion-camera frame cache.
# One-frame keep-last: the adjudicator always sees the most recent frame; older
# frames are dropped rather than queued. VOLATILE means no history replay (the
# verification window is the present moment, not a historical one).
_QOS_COMPLETION_CAMERA = QoSProfile(
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    durability=QoSDurabilityPolicy.VOLATILE,
)

# Closed sets from ADR-0018 §3 / capability review §3.
# `rskill` was renamed from `skill` on 2026-05-25 (ADR-0018 amendment §5)
# for consistency with the carried `rskill_id` field.
_FAILURE_SOURCES: tuple[str, ...] = ("hal", "sensor", "rskill", "safety", "wam", "critic")
_PERCEPTION_KINDS: tuple[str, ...] = ("motion", "objects", "ocr", "scene_change")

# ADR-0073 §1 — prompt frame_ids the reasoner re-publishes onto /openral/prompt
# for its OWN cascade (advisory query responses + spatial-memory re-prompts).
# These are not new operator goals, so they must NOT (re)build the mission queue
# — only a genuine operator/cli/dashboard prompt does. Self-emits (frame_id ==
# the node name) are already dropped earlier in `_on_prompt`.
_CASCADE_PROMPT_SOURCES: frozenset[str] = frozenset(
    {"spatial_memory", "detector", "scene_vlm", "reward_monitor", "memory", "mission"}
)

# ADR-0073 §2 — reward window (s) for the automatic post-skill task verification.
# Matches the reward_monitor's default rolling window so the windowed score
# reflects the end-of-attempt state, not a single frame.
_MISSION_VERIFY_WINDOW_S: float = 8.0

# ADR-0074 Decision 5 — three-tier verdict band edges + the patience ceiling.
# These are the SYSTEM FALLBACK in the authority stack (system < reward-model
# calibrated default < LLM per-task override): used only when no reward manifest
# is wired (``reward_manifest_path`` unset). When a reward model is active the
# node reads the live ``RewardContract`` (``_reward_contract``) instead — see
# ``_band_edges`` / ``_effective_patience_s``. The values mirror the robometer
# rskill.yaml reward block so a fallback matches today's deploy default.
_DEFAULT_SUCCESS_THRESHOLD: float = 0.8
_DEFAULT_CHECK_FLOOR: float = 0.5

# FailureTrigger constants — IDL-mirror per openral_observability.failure_bus
# (kept inline rather than importing the helper so the reasoner_node can
# emit a FailureTrigger without dragging the rate-limiter into the
# dispatch path; the reasoner publishes O(1) events per skill goal, not
# a stream).
_KIND_TIMEOUT: int = 0
_KIND_CONTROLLER: int = 5
_SEVERITY_WARN: int = 1
_SEVERITY_FAIL: int = 2

# Brief, non-blocking probe used before sending an ExecuteSkill goal — if
# the F1 server isn't on the graph yet we emit a KIND_CONTROLLER
# FailureTrigger instead of blocking the executor thread.
_EXECUTE_SKILL_SERVER_PROBE_S: float = 0.1
_LIFECYCLE_SERVER_PROBE_S: float = 0.1

# ADR-0018 2026-05-25 amendment — trigger taxonomy. Maps each failure
# source to its tier so the reasoner_node stamps a ``reasoner.tier``
# attribute on the OTel span (observability only — the preemption
# threshold per source is decided inline in :meth:`_on_failure`). Tier
# labels: A=safety, B=replan-class (hal/sensor/rskill/wam), C=critic,
# D=operator/perception (handled in their own callbacks).
_FAILURE_TIER_FOR_SOURCE: dict[str, str] = {
    "safety": "A",
    "hal": "B",
    "sensor": "B",
    "rskill": "B",
    "wam": "B",
    "critic": "C",
}

# Wrapped task-space layouts pack non-joint quantities — eef pose,
# base pose, gripper qpos — composed by a sim adapter, not derivable
# from raw JointState. Dropping these for a ``deploy sim`` (which
# feeds JointState) is informational, not an error: the rSkill is
# fine, this robot path just doesn't expose the wrapped observation
# it expects. Joint-space layouts (``smolvla_9d``, ``libero``, etc.)
# ARE joint-count contracts, so a dim mismatch there IS a real
# incompatibility worth a WARN. ADR-0025 amendment 2026-05-27.
# Canonical source is ``openral_core.WRAPPED_TASK_SPACE_LAYOUTS``
# (ADR-0027 — single source of truth so the schema validator and the
# reasoner filter stay in lockstep).
from openral_core import WRAPPED_TASK_SPACE_LAYOUTS as _WRAPPED_TASK_SPACE_LAYOUTS  # noqa: E402

# ADR-0036 (amended 2026-06-04) — deploy-path-aware action-mode palette gate.
#
# The state-contract filter above gates a VLA's *input* (state dim vs
# joint count); the ``hal_mode == "sim"`` executable set gates a VLA's
# *output* (the ControlMode s its action vector drives). A sim env brought
# up with a robosuite OSC / composite controller can execute joint modes
# AND a cartesian-EE + gripper + base-twist set, even when the *physical*
# robot only advertises ``joint_position`` — the OSC layer synthesises
# joint commands from the cartesian goal. So under ``sim`` a cartesian
# skill (e.g. pi05/smolvla LIBERO with a delta-EEF representation) is
# admissible; under ``real`` only the robot's declared
# ``supported_control_modes`` are.
#
# The canonical set is ``openral_core.SIM_EXECUTABLE_CONTROL_MODES`` — the
# single source of truth pinned to the actual sim HAL action-packers in
# ``python/hal/src/openral_hal/sim_attached.py`` by the lockstep test
# ``tests/unit/test_sim_executable_modes_match_packers.py`` (both
# directions). Importing it here (rather than re-declaring it) is what
# stops the gate and the packers from drifting: a mode the gate admits but
# no packer executes would boot-pass and then E-stop mid-run.


def _required_control_modes(manifest: RSkillManifest) -> set[ControlMode]:
    """The :class:`ControlMode` s a skill's ``action_contract`` demands (ADR-0036).

    Pure helper (no ROS spin) so the deploy-path palette gate is unit
    testable. The contract is read in order of specificity:

    * No ``action_contract`` → empty set (the skill declares no action
      constraint, so it is admitted by :func:`_action_executable`).
    * ``representation`` set → :func:`control_modes_for_representation`.
    * ``slots`` set → every slot's ``control_mode`` (discard slots carry
      ``None`` and are skipped).
    * Bare ``dim`` only (legacy ADR-0019 contract) → ``{JOINT_POSITION}``;
      the skill_runner dispatches a bare-dim vector as one whole-vector
      joint-position Action.

    Args:
        manifest: The rSkill manifest to inspect.

    Returns:
        The set of control modes the skill's action vector drives.
    """
    contract = manifest.action_contract
    if contract is None:
        return set()
    if contract.representation is not None:
        return control_modes_for_representation(contract.representation)
    if contract.slots is not None:
        return {slot.control_mode for slot in contract.slots if slot.control_mode is not None}
    return {ControlMode.JOINT_POSITION}


def _action_executable(
    manifest: RSkillManifest,
    description: RobotDescription,
    hal_mode: str,
) -> bool:
    """Whether the deploy path can execute a skill's action modes (ADR-0036).

    Pure helper (no ROS spin). The executable set depends on ``hal_mode``:

    * ``"sim"`` → :data:`openral_core.SIM_EXECUTABLE_CONTROL_MODES` (a robosuite
      OSC / composite controller synthesises cartesian + gripper + base
      goals into joint commands).
    * anything else (``"real"``) → the robot's declared
      :attr:`RobotCapabilities.supported_control_modes`.

    ``supported_control_modes`` deserializes as :class:`ControlMode`
    enum members (``RobotCapabilities`` does not set
    ``use_enum_values``); both sides are coerced to :class:`ControlMode`
    so the comparison is robust even if a hand-built description carries
    raw ``"joint_position"`` strings.

    Args:
        manifest: The rSkill manifest.
        description: The target robot description.
        hal_mode: ``"sim"`` or ``"real"``.

    Returns:
        ``True`` when every required mode is executable on the deploy
        path (or the skill declares no action constraint).
    """
    required = _required_control_modes(manifest)
    if not required:
        return True
    if hal_mode == "sim":
        executable: set[ControlMode] = set(SIM_EXECUTABLE_CONTROL_MODES)
    else:
        executable = {ControlMode(m) for m in description.capabilities.supported_control_modes}
    return {ControlMode(m) for m in required} <= executable


def _resets_search_episode(call: Any) -> bool:
    """True when dispatching ``call`` should end the active-search episode (ADR-0039 §3).

    The cascade bound counts only *consecutive* spatial-search queries, so any
    non-search dispatch resets ``_spatial_search`` + ``_locate_escalated``. The
    search actions that must NOT reset are ``recall_object`` / ``resolve_place``
    (remembered objects) and ``locate_in_view`` (live detector) — the latter is
    the regression this guards: if a directly-emitted ``locate_in_view`` reset
    the budget, a ``recall → locate → recall`` loop against an undetectable
    object would zero the counter every cycle and never hand off.
    """
    return not isinstance(call, RecallObjectTool | ResolvePlaceTool | LocateInViewTool)


def _should_offer_subdivision(
    active: TaskState,
    offered: set[str],
    max_depth: int,
) -> bool:
    """True when a blocked task may be offered subdivision before abandonment (#123).

    Bounded two ways so a task that refuses to decompose still terminates in
    human-handoff rather than looping: **once per task id** (the ``offered`` set —
    a second abandon of the same task falls through to the normal abandon/advance
    ladder) and only while the task is **below** the re-decomposition depth bound
    (a task already split to ``max_depth`` is handed off, not split again).
    """
    return active.depth < max_depth and active.task_id not in offered


def _resolve_execute_prompt(call_prompt: str, active_text: str | None) -> str:
    """Fall back to the active mission task's text when the LLM omits the prompt.

    A VLA conditions on this string (SmolVLA writes it into ``observation["task"]``),
    so an empty ``ExecuteRskillTool.prompt`` (the field defaults to ``""`` with no
    ``min_length``) gives the policy no instruction — it cannot know which object to
    manipulate. The active mission task *is* the instruction, so use it whenever the
    LLM leaves the prompt empty/whitespace; otherwise pass the LLM prompt through.
    Returns ``""`` when neither is available (the runner/manifest default applies).
    """
    if call_prompt.strip():
        return call_prompt
    return active_text or ""


# The "collective target" predicate (ADR-0075) is the single source of truth in
# ``openral_core`` (`is_collective_target`) — shared by the `GroundedSubtask`
# schema validator and this node's runtime execute gate so a skill never acts on
# a quantified/plural set ("all the objects"); it must be enumerated from the live
# ``scene_objects`` context and decomposed into one grounded subtask per object.


class ReasonerNode(LifecycleNode):
    """ROS 2 lifecycle wrapper around :class:`ReasonerCore` (ADR-0018 F4).

    Args:
        node_name: ROS node name. Default ``openral_reasoner``.
        tick_hz: Heartbeat tick rate in Hz. Default 0.2 (one every
            5 s). Per ADR-0018 amendment 2026-05-25, the reasoner is
            event-driven: failure/prompt arrivals preempt with
            ``force=True``, and the periodic timer is the safety net
            for "task is not making progress but nothing has fired".
            Heartbeat ticks that see no new event since the last
            successful tick are short-circuited inside
            :class:`ReasonerCore` with
            ``suppressed_reason="heartbeat_idle"``.
        client: Optional pre-built :class:`ToolUseClient`. When ``None``
            :meth:`on_configure` builds one from the
            ``OPENRAL_REASONER_LLM_*`` env vars via
            :func:`build_tool_use_client_from_env`. Tests pass a
            :class:`FakeToolUseClient` here.
        palette: Optional pre-built :class:`ToolPalette`. When ``None``
            :meth:`on_configure` builds an empty palette (the
            ``skill_registry_changed`` topic populates it). Tests
            inject a palette directly.
        robot_capabilities: The active robot's capabilities. Required
            for the ``/openral/skill_registry_changed`` refresh path
            to rebuild the palette; ``None`` leaves the palette fixed
            at the constructor-injected value and logs a warning on
            each refresh event.
        commercial_deployment: Forwarded to
            :func:`build_tool_palette` on every refresh — when
            ``True``, skills whose
            :attr:`RSkillManifest.is_commercial_use_allowed` is
            ``False`` are filtered out (defense-in-depth against a
            cached non-commercial weights repo in a commercial
            deployment, CLAUDE.md §1.9).
    """

    def __init__(  # noqa: PLR0915  # reason: lifecycle node wires many subsystems in one ctor; one attr over the 50-statement threshold
        self,
        *,
        node_name: str = "openral_reasoner",
        tick_hz: float = 0.2,
        client: ToolUseClient | None = None,
        palette: ToolPalette | None = None,
        robot_capabilities: RobotCapabilities | None = None,
        commercial_deployment: bool = False,
        spatial_memory: SpatialMemoryQuerier | None = None,
    ) -> None:
        """Initialise without rclpy I/O; resources opened in on_configure.

        ``spatial_memory`` (ADR-0039 Phase 2b) is an optional read-only
        scene-graph query backend (an ADR-0038 ``SpatialMemory``). When
        provided, the ``recall_object`` / ``resolve_place`` tools are offered to
        the LLM and dispatched against it; the result is republished as a
        ``PromptStamped`` so the next tick sees it (the prompt cascade). When
        ``None`` the query tools are never offered.
        """
        super().__init__(node_name)
        if tick_hz <= 0:
            raise ValueError(f"ReasonerNode.tick_hz must be > 0; got {tick_hz!r}")
        self._tick_hz = tick_hz
        self._injected_client = client
        self._injected_palette = palette
        self._robot_capabilities = robot_capabilities
        self._commercial_deployment = commercial_deployment
        self._spatial_memory = spatial_memory
        # ADR-0038 live dynamic memory — when the reasoner *owns* the backend
        # (preloaded from disk, or auto-created for `spatial_memory_ingest`),
        # this concrete handle lets `_on_tick` fold each WorldState.detected_objects
        # snapshot into it. Stays None for an externally-injected read-only
        # querier (we don't mutate a backend we don't own).
        self._spatial_memory_writer: SpatialMemory | None = None
        # ADR-0044 Phase 4 — latest decoded occupancy grid (an
        # ``openral_world_state.grid.OccupancyGridIndex``), from the latched
        # ``occupancy_map_topic`` subscription. ``None`` until a map arrives;
        # ``_dispatch_spatial_query`` then refines every recall_object approach
        # viewpoint through it (grid absent → geometric viewpoints pass
        # through unchanged).
        self._occupancy_grid: Any = None
        # ADR-0039 §3 — bound the find→re-prompt cascade so a query that keeps
        # missing terminates in human-handoff instead of looping forever.
        self._spatial_search = SearchProgress(SearchBudget())
        # ADR-0043/0056 — recall_object queries already escalated to a live
        # locate_in_view this search streak (one escalation per query term, so a
        # repeated miss doesn't re-fire the detector every tick). Reset whenever
        # the active-search bound resets (new operator goal / non-search action).
        self._locate_escalated: set[str] = set()
        # #123 — task ids already offered one subdivision before being abandoned.
        # One offer per task so a task that declines to decompose still terminates
        # in human-handoff; cleared when a new operator goal rebuilds the mission.
        self._subdivide_offered: set[str] = set()

        # ROS parameters: when both are set, on_configure walks
        # `rskill_search_paths` for `*/rskill.yaml`, loads the
        # `RobotCapabilities` from `robot_yaml`, and seeds the palette
        # via `build_tool_palette`. Either parameter being empty leaves
        # the palette at the constructor-supplied value (or empty),
        # preserving the existing `/openral/skill_registry_changed`
        # refresh path for HF-Hub-installed skills.
        self.declare_parameter("robot_yaml", "")
        self.declare_parameter("rskill_search_paths", [""])
        # ADR-0025 — additional lifecycle peer node names to surface in
        # the LLM tool palette's `node_ids` slot so the Reasoner can
        # emit `LifecycleTransitionTool(node=..., transition=...)` against
        # background services like `/openral_slam_toolbox`. Defaults to
        # empty; deploy launches set this via `reasoner_lifecycle_peers:=
        # [openral_slam_toolbox]` when the corresponding `--enable-<svc>`
        # CLI flag was passed.
        self.declare_parameter("lifecycle_peer_node_ids", [""])
        # ADR-0050 — GPU lifecycle peers (the object-detector LifecycleNode is
        # the canonical one) to DEACTIVATE before dispatching a GPU-heavy
        # ``execute_rskill`` and REACTIVATE once it finishes, so their VRAM is
        # freed for the policy. Without this the detector (~1.3 GB) co-resident
        # with a VLA (~4.5 GB) OOMs an 8 GB card at load. Default empty; the
        # deploy launch sets it to the detector node id when
        # ``--enable-object-detector``. Distinct from ``lifecycle_peer_node_ids``
        # (which only surfaces peers to the LLM tool palette, not auto-managed).
        self.declare_parameter("vram_lifecycle_peers", [""])
        # ADR-0039 Phase 2b deployment wiring — absolute path to a persisted
        # ADR-0038 scene graph (``SceneGraph`` JSON written by
        # ``SpatialMemory.save``). When set (and no ``spatial_memory`` backend
        # was injected), ``on_configure`` loads it into a ``SpatialMemory`` and
        # wires it as the read-only query backend, enabling the
        # ``recall_object`` / ``resolve_place`` tools against a preloaded map.
        # Empty = disabled.
        self.declare_parameter("spatial_memory_path", "")
        # ADR-0072 §3 / Phase 4b — path to the self-maintained MEMORY.md (read at
        # configure into the `## MEMORY` context block). Empty omits the section.
        self.declare_parameter("memory_md_path", "")
        # ADR-0072 §3 / Phase 5 — retrieval-under-cap: render at most this many
        # memory entries in the always-on `## MEMORY` block (top by importance then
        # recency; the tail stays searchable via memory_search). 0 = no cap.
        self.declare_parameter("memory_context_cap", 0)
        # ADR-0038 live dynamic memory — when true, ``on_configure`` ensures a
        # ``SpatialMemory`` backend exists (auto-creating an empty one if no
        # ``spatial_memory_path`` was loaded and none injected) and ``_on_tick``
        # folds each ``/openral/world_state_slow`` ``WorldState.detected_objects``
        # snapshot into it — accumulating the durable scene graph from the
        # ADR-0035 perception object-lift producer so ``recall_object`` recalls
        # what the robot has actually seen. Default false (preloaded-map only).
        self.declare_parameter("spatial_memory_ingest", False)
        # ADR-0044 Phase 4 — occupancy-grid refinement of recall approach
        # poses. The reasoner subscribes the latched slam_toolbox map on this
        # topic and validates/snaps every ``recall_object`` approach viewpoint
        # (free under ``approach_inflation_m`` + line-of-sight) before the LLM
        # sees it. Empty string disables the subscription; no map received →
        # geometric viewpoints pass through unchanged.
        self.declare_parameter("occupancy_map_topic", "/map")
        self.declare_parameter("approach_inflation_m", 0.25)
        # ADR-0036 — deploy-path selector for the action-mode palette gate.
        # ``"sim"`` (default; deploy sim is the common path) admits skills
        # whose action modes a robosuite OSC / composite controller can
        # synthesise; ``"real"`` admits only the robot's declared
        # ``supported_control_modes``. The deploy launch sets this
        # explicitly to match the HAL it brings up (a later task).
        self.declare_parameter("hal_mode", "sim")
        # ADR-0043 — when true, offer the read-only ``locate_in_view`` tool (ask a
        # live VLM detector if an object is in the current frame, via the
        # ``/openral/perception/locate_in_view`` service). The deploy launch sets
        # this when it brings up an object detector. Default false (no hidden tool).
        self.declare_parameter("detector_available", False)
        self._detector_available: bool = (
            self.get_parameter("detector_available").get_parameter_value().bool_value
        )
        # ADR-0056 — the default on-demand locator alias used when a locate_in_view
        # call leaves ``detector`` empty (e.g. "omdet-turbo-locator"). Empty = the
        # legacy single-detector service /openral/perception/locate_in_view. Set by
        # the deploy launch to the default locator it brings up.
        self.declare_parameter("default_on_demand_detector", "")
        self._default_on_demand_detector: str = (
            self.get_parameter("default_on_demand_detector").get_parameter_value().string_value
        )
        # ADR-0056 — locate_in_view clients cached per resolved service name (one
        # per on-demand locator the reasoner has routed to), created lazily.
        self._locate_in_view_clients: dict[str, Any] = {}
        # ADR-0047 — when true, offer the read-only ``query_scene`` tool (ask a
        # scene VLM an open-ended question about the current view, via the
        # ``/openral/perception/query_scene`` service). The deploy launch sets
        # this when it brings up a scene VLM. Default false (no hidden tool).
        self.declare_parameter("scene_query_available", False)
        self._scene_query_available: bool = (
            self.get_parameter("scene_query_available").get_parameter_value().bool_value
        )
        # Cached client for the query_scene service; created lazily on first use.
        self._query_scene_client: Any = None
        # ADR-0057 — when true, offer the read-only ``query_task_progress`` tool
        # (ask the Robometer reward monitor for a windowed progress/success
        # assessment of the current task, via the
        # ``/openral/perception/query_task_progress`` service). The deploy launch
        # sets this when it brings up a reward monitor. Default false.
        self.declare_parameter("task_progress_available", False)
        self._task_progress_available: bool = (
            self.get_parameter("task_progress_available").get_parameter_value().bool_value
        )
        # Cached client for the query_task_progress service; created lazily.
        self._query_task_progress_client: Any = None

        # ADR-0074 §1/§3 — the active reward model's manifest (same path the
        # reward_monitor_node loads). When set, the node reads its
        # ``RewardContract`` calibration (band edges + default patience) and uses
        # it in place of the module-level system fallbacks. A bad path degrades
        # to the fallbacks (logged) rather than failing node construction.
        self._reward_contract: RewardContract | None = None
        self.declare_parameter("reward_manifest_path", "")
        reward_manifest_path = (
            self.get_parameter("reward_manifest_path").get_parameter_value().string_value
        )
        if reward_manifest_path:
            try:
                self._reward_contract = RSkillManifest.from_yaml(reward_manifest_path).reward
            except Exception as exc:  # reason: bad manifest must not block startup
                self.get_logger().warning(
                    f"reward_manifest_path={reward_manifest_path!r} failed to load "
                    f"({type(exc).__name__}: {exc}); using system-default band edges + patience",
                )
            else:
                if self._reward_contract is not None:
                    self.get_logger().info(
                        "reward calibration: success_threshold="
                        f"{self._reward_contract.success_threshold:.2f} "
                        f"check_floor={self._reward_contract.check_floor:.2f} "
                        f"default_patience_s={self._reward_contract.default_patience_s:.0f}",
                    )

        # ADR-0074 §5 — completion-camera topic for VLM adjudication.
        # When set to a non-empty string, on_configure subscribes sensor_msgs/Image
        # on this topic (BEST_EFFORT, VOLATILE, depth=1) and caches the latest frame
        # as JPEG bytes in `_latest_completion_frame` for `_adjudicate_completion`.
        # Empty string disables the subscription (no hidden camera subscription).
        self.declare_parameter("completion_camera_topic", "/openral/cameras/top/image")

        # ADR-0074 §5 — tool-use client handle (mirrors the one held by ReasonerCore)
        # so the completion gate can call describe_image without reaching into the core.
        # Set in on_configure; cleared in on_cleanup.
        self._tool_use_client: ToolUseClient | None = None
        # Latest completion frame as JPEG bytes; None until the first camera message.
        self._latest_completion_frame: bytes | None = None

        # Populated by on_configure.
        self._renderer: ContextRenderer = ContextRenderer()
        self._world_state_msg: Any = None
        self._core: ReasonerCore | None = None
        # Log a `retry_cap` suppression only ONCE per streak — without this it
        # re-warns every heartbeat tick and floods the log. Cleared the moment a
        # non-retry_cap tick happens (a different tool, a dispatch, an error, or
        # a new operator prompt that resets the streak).
        self._retry_cap_warned: bool = False
        # ADR-0072 Phase 3 — the rendered `## PLAYBOOKS` system-prompt block,
        # collected from installed capability-matched playbook rSkills at seed time.
        self._playbooks_block: str = ""
        # ADR-0072 §3 — the self-maintained MEMORY.md store (Phase 4b read path:
        # loaded at configure + rendered as the `## MEMORY` context block; Phase 4c
        # write path: `memory_write` edits + `memory_search` archival recall). The
        # archive is the append-only log of superseded/deleted entries that left the
        # live file (MemGPT recall storage); persisted as `<MEMORY.md>.archive.jsonl`.
        self._memory_store: MemoryStore | None = None
        self._memory_md_path: pathlib.Path | None = None
        self._memory_archive: list[MemoryEntry] = []
        self._palette: ToolPalette = palette or ToolPalette(execute_rskill_ids=frozenset())
        # ADR-0039 — offer the read-only query tools only when a backend is wired.
        if spatial_memory is not None and not self._palette.spatial_memory_available:
            self._palette = self._palette.model_copy(update={"spatial_memory_available": True})
        self._tick_timer: Any = None
        self._prompt_pub: Any = None
        self._failure_pub: Any = None  # /openral/failure/rskill
        self._execute_rskill_client: Any = None  # rclpy_action.ActionClient
        # Lifecycle clients are cached per target node — one
        # ``<node>/change_state`` client per peer.
        self._lifecycle_clients: dict[str, Any] = {}
        # ADR-0050 — GPU lifecycle peers to free before a VLA dispatch (read
        # from ``vram_lifecycle_peers`` at configure) and the subset actually
        # deactivated for the in-flight skill (reactivated on its result).
        self._vram_lifecycle_peers: list[str] = []
        self._deactivated_vram_peers: list[str] = []
        # Pending skill-goal deadline timers, keyed by goal-uuid bytes so
        # the result callback can cancel the deadline timer when the
        # action server returns before deadline_s elapses.
        self._pending_skill_deadlines: dict[bytes, Any] = {}
        # ADR-0074 §2 — the in-flight execute_rskill goal so a reward-watcher
        # wake can cancel it (stop the VLA now, verify on the reward signal,
        # not at the deadline clock). Tuple of (goal_handle, call, traceparent);
        # set on goal-accept, cleared on the terminal result. ``cancel_reason``
        # is ``"reward"`` while a reward-driven cancel is in flight so the
        # canceled result re-enters the verify gate (vs an operator/estop
        # cancel, which stays a no-op).
        self._active_rskill_goal: tuple[Any, ExecuteRskillTool, str | None] | None = None
        self._rskill_cancel_reason: str | None = None
        self._dispatched_calls: list[Any] = []  # for tests/observability

    # ── lifecycle transitions ───────────────────────────────────────────────

    @log_lifecycle_errors
    def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Build the tool-use client + subscribers; no ticking yet."""
        del state
        if (
            IDLPromptStamped is None
            or IDLFailureTrigger is None
            or IDLWorldStateStamped is None
            or IDLExecuteRskill is None
        ):
            self.get_logger().error(
                "openral_msgs not on PYTHONPATH — colcon-build openral_msgs and source install/",
            )
            return TransitionCallbackReturn.FAILURE
        if IDLChangeState is None or IDLTransition is None:
            self.get_logger().error(
                "lifecycle_msgs not on PYTHONPATH — source the ROS 2 install first",
            )
            return TransitionCallbackReturn.FAILURE

        try:
            client = self._injected_client or build_tool_use_client_from_env()
        except ROSConfigError as exc:
            self.get_logger().error(f"on_configure: {exc}")
            return TransitionCallbackReturn.FAILURE

        # ADR-0074 §5 — hold the client on the node so the VLM adjudication gate
        # can call describe_image without reaching into ReasonerCore internals.
        self._tool_use_client = client

        # NOTE: ``self._core`` is built *after* the palette seed below, so the
        # robot-context system prompt (option B) reflects the capabilities
        # loaded from ``robot_yaml``. Nothing between here and then dispatches
        # a tick (callbacks only run once the executor spins, after configure
        # returns), so the late construction is safe.

        # Subscriptions.
        self.create_subscription(
            IDLWorldStateStamped,
            "/openral/world_state_slow",
            self._on_world_state,
            _QOS_WORLD_STATE,
        )
        for source in _FAILURE_SOURCES:
            topic = f"/openral/failure/{source}"
            self.create_subscription(
                IDLFailureTrigger,
                topic,
                lambda msg, _source=source: self._on_failure(_source, msg),
                _QOS_FAILURE,
            )
        for kind in _PERCEPTION_KINDS:
            topic = f"/openral/perception/{kind}"
            self.create_subscription(
                IDLPromptStamped,
                topic,
                lambda msg, _kind=kind: self._on_perception(_kind, msg),
                _QOS_PERCEPTION,
            )
        self.create_subscription(
            IDLPromptStamped,
            "/openral/prompt",
            self._on_prompt,
            _QOS_PROMPT,
        )

        # ADR-0074 §5 — completion-camera subscription (BEST_EFFORT sensor QoS).
        # sensor_msgs/Image ships with every ROS 2 install but is gated like
        # nav_msgs above so a stripped environment degrades to "no frame cache"
        # instead of failing configure. An empty topic param disables the sub.
        completion_camera_topic = (
            self.get_parameter("completion_camera_topic").get_parameter_value().string_value
        )
        if completion_camera_topic:
            try:
                from sensor_msgs.msg import (
                    Image as _IDLImage,  # reason: ROS IDL import gated like the others above
                )
            except ImportError:
                self.get_logger().warning(
                    "sensor_msgs is unavailable; completion-camera adjudication disabled"
                )
            else:
                self.create_subscription(
                    _IDLImage,
                    completion_camera_topic,
                    self._on_completion_camera,
                    _QOS_COMPLETION_CAMERA,
                )
                self.get_logger().info(
                    f"on_configure: completion-camera subscribed on {completion_camera_topic!r}"
                )

        # ADR-0044 Phase 4 — latched occupancy grid for approach refinement.
        # nav_msgs ships with every ROS 2 base install, but gate like the
        # other IDL imports so a stripped environment degrades to "no grid"
        # instead of failing configure.
        map_topic = self.get_parameter("occupancy_map_topic").get_parameter_value().string_value
        if map_topic:
            try:
                from nav_msgs.msg import (
                    OccupancyGrid,  # reason: ROS IDL import gated like the others above
                )
            except ImportError:
                self.get_logger().warning(
                    "nav_msgs is unavailable; occupancy-grid approach refinement disabled"
                )
            else:
                self.create_subscription(OccupancyGrid, map_topic, self._on_map, _QOS_MAP)

        # ADR-0018 §4: palette is rebuilt on every
        # /openral/skill_registry_changed event (fired by
        # `ral skill install|remove`). Empty payload — the topic is
        # the signal. std_msgs/Empty may be absent on hosts without
        # a sourced ROS install; that's the same gate as the IDL
        # imports above, so we re-check here.
        if IDLEmpty is not None:
            self.create_subscription(
                IDLEmpty,
                "/openral/skill_registry_changed",
                self._on_skill_registry_changed,
                _QOS_REGISTRY_CHANGED,
            )

        # Publisher for EmitPromptTool dispatch.
        self._prompt_pub = self.create_publisher(
            IDLPromptStamped,
            "/openral/prompt",
            _QOS_PROMPT,
        )

        # FailureTrigger publisher on /openral/failure/rskill — the
        # reasoner is the consumer of skill outcomes, so failed
        # ExecuteSkill goals are reported under the rskill-source bus
        # (kind=KIND_CONTROLLER for rejection/abort, kind=KIND_TIMEOUT
        # for deadline_s expiry). QoS matches the failure-bus profile.
        # The `rskill` suffix replaced `skill` on 2026-05-25 (ADR-0018
        # amendment §5).
        self._failure_pub = self.create_publisher(
            IDLFailureTrigger,
            "/openral/failure/rskill",
            _QOS_FAILURE,
        )

        # ExecuteRskill action client (F1 rskill_runner_node server). The
        # client is opened in on_configure so wait_for_server can pre-
        # negotiate without paying connect cost on the dispatch path. The
        # type + topic were renamed skill→rskill in #262; the runner serves
        # `/openral/execute_rskill`.
        from rclpy.action import ActionClient

        self._execute_rskill_client = ActionClient(
            self,
            IDLExecuteRskill,
            "/openral/execute_rskill",
        )

        # ADR-0039 — load a persisted scene graph into the query backend
        # before the palette seed, so the rebuilt palette offers the query
        # tools when a map is preloaded.
        self._maybe_load_spatial_memory()
        self._maybe_load_memory()

        # ADR-0050 — GPU lifecycle peers to deactivate before a VLA dispatch and
        # reactivate after (the object detector is the canonical one). Read
        # unconditionally so it is honoured regardless of whether the palette
        # seed path runs. Empty entries skipped.
        self._vram_lifecycle_peers = [
            p
            for p in self.get_parameter("vram_lifecycle_peers")
            .get_parameter_value()
            .string_array_value
            if p
        ]

        # Seed the palette from the `rskills/` search paths + the
        # robot's manifest if both ROS parameters are set. This lets a
        # demo launch ship with a populated palette out of the box;
        # without it the palette stays empty until
        # `/openral/skill_registry_changed` fires.
        self._maybe_seed_palette_from_search_paths()

        # Option B (ADR-0018 F4): give the reasoner LLM standing knowledge of
        # the body it drives. ``self._robot_capabilities`` is now finalised
        # (from the constructor or the ``robot_yaml`` loaded during the seed),
        # so the system prompt carries a ``## THIS ROBOT`` block; ``None``
        # leaves the robot-agnostic brief unchanged. The base brief honours
        # the ``OPENRAL_REASONER_SYSTEM_PROMPT`` deployment override.
        base_prompt = resolve_reasoner_system_prompt(self._robot_capabilities)
        # ADR-0072 Phase 3 — append installed playbooks (empty block = no-op).
        system_prompt = (
            f"{base_prompt}\n\n{self._playbooks_block}" if self._playbooks_block else base_prompt
        )
        self._core = ReasonerCore(
            client=client,
            system_prompt=system_prompt,
        )

        self.get_logger().info(
            f"on_configure: reasoner ready at {self._tick_hz} Hz "
            f"({len(self._palette.execute_rskill_ids)} skills in palette)",
        )
        return TransitionCallbackReturn.SUCCESS

    @log_lifecycle_errors
    def on_activate(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Arm the periodic tick timer."""
        del state
        period_s = 1.0 / self._tick_hz
        self._tick_timer = self.create_timer(period_s, self._on_tick)
        self.get_logger().info("on_activate: ticking")
        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Stop the tick timer (subscriptions remain attached)."""
        del state
        if self._tick_timer is not None:
            self._tick_timer.cancel()
            self._tick_timer = None
        self.get_logger().info("on_deactivate: stopped")
        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Drop state; subscriptions are auto-cleaned by rclpy."""
        del state
        self._core = None
        self._occupancy_grid = None
        self._renderer = ContextRenderer()
        # ADR-0074 §5 — clear the VLM client handle and frame cache on cleanup.
        self._tool_use_client = None
        self._latest_completion_frame = None
        for timer in list(self._pending_skill_deadlines.values()):
            with contextlib.suppress(Exception):
                timer.cancel()
        self._pending_skill_deadlines.clear()
        if self._execute_rskill_client is not None:
            self._execute_rskill_client.destroy()
            self._execute_rskill_client = None
        self._lifecycle_clients.clear()
        self.get_logger().info("on_cleanup: state cleared")
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Final shutdown."""
        del state
        self.get_logger().info("on_shutdown")
        return TransitionCallbackReturn.SUCCESS

    # ── topic callbacks ─────────────────────────────────────────────────────

    def _on_world_state(self, msg: Any) -> None:
        """Cache the latest WorldStateStamped snapshot."""
        self._world_state_msg = msg

    def _on_map(self, msg: Any) -> None:
        """Decode the latched occupancy grid for approach refinement (ADR-0044).

        Keeps only the latest snapshot; slam_toolbox republishes the latched
        map as it grows, so the refiner always sees the current grid.
        """
        # Layer-2 import deferred like SpatialMemory in _maybe_load_spatial_memory.
        from openral_world_state.grid import OccupancyGridIndex

        first = self._occupancy_grid is None
        try:
            self._occupancy_grid = OccupancyGridIndex.from_msg(msg)
        except (ValueError, AttributeError) as exc:
            self.get_logger().warning(f"occupancy map decode failed: {exc}")
            return
        if first:
            self.get_logger().info(
                f"occupancy grid online ({msg.info.width}x{msg.info.height} @ "
                f"{msg.info.resolution:.3f} m) — recall_object approaches are now grid-refined"
            )

    def _on_completion_camera(self, msg: Any) -> None:
        """Cache the latest camera frame as JPEG bytes for VLM adjudication (ADR-0074 §5).

        Converts ``sensor_msgs/Image`` to JPEG using numpy + PIL (no cv_bridge).
        Supports ``"rgb8"`` and ``"bgr8"`` encodings. On any decode failure the
        cache is left unchanged so the next successful frame recovers silently —
        a decode error must not raise into the rclpy executor.
        """
        try:
            jpeg = _image_msg_to_jpeg(
                data=bytes(msg.data),
                height=int(msg.height),
                width=int(msg.width),
                encoding=str(msg.encoding),
            )
        except Exception as exc:  # reason: never raise in a topic callback
            self.get_logger().debug(f"completion-camera: decode failed, cache unchanged: {exc}")
            return
        self._latest_completion_frame = jpeg

    def _adjudicate_completion(self, task_text: str) -> bool | None:
        """Ask the VLM whether ``task_text`` is complete in the latest camera frame.

        Returns ``True`` (complete), ``False`` (not complete), or ``None``
        (could not adjudicate — no frame cached or no multimodal client).
        ``None`` degrades to the ladder (ADR-0074 §6 no-VLM path). A provider
        or transport error is logged and returns ``None`` — never a false ``True``.

        Args:
            task_text: The active task description shown to the VLM.
        """
        if self._latest_completion_frame is None:
            self.get_logger().debug(
                "adjudicate_completion: no frame cached — cannot adjudicate; treating as no"
            )
            return None
        if self._tool_use_client is None:
            self.get_logger().debug(
                "adjudicate_completion: no VLM client — cannot adjudicate; treating as no"
            )
            return None
        question = _COMPLETION_QUESTION.format(task=task_text)
        try:
            answer = self._tool_use_client.describe_image(
                image_jpeg=self._latest_completion_frame,
                question=question,
            )
        except Exception as exc:  # reason: provider errors must not block the loop
            self.get_logger().warning(
                f"adjudicate_completion: describe_image raised — treating as no: {exc}"
            )
            return None
        result = _parse_yes_no(answer)
        self.get_logger().debug(
            f"adjudicate_completion: answer={answer!r} → {'yes' if result else 'no'}"
        )
        return result

    def _complete_active_and_advance(
        self,
        active: TaskState,
        verdict: str,
        *,
        traceparent: str | None,
    ) -> None:
        """Mark the active task done and advance the mission queue (DRY helper).

        Shared by the native ``"complete"`` verdict branch and the
        VLM-confirmed ``"vlm_check"`` branch of
        :meth:`_on_mission_verify_response`. Calls
        ``advance_mission(done=True)``, resets the per-kind tick streak when
        a next task is activated, emits the mission-complete summary when the
        queue drains, and forces a Tier-C tick.
        """
        mission = self._renderer.mission
        if mission is None:
            return
        nxt = self._renderer.advance_mission(done=True, verdict=verdict)
        if nxt is not None:
            if self._core is not None:
                self._core.reset_kind_streak()
            self.get_logger().info(
                f"mission: task {active.task_id} done ✓ ({verdict}); "
                f"advancing → {nxt.task_id}={nxt.text[:60]!r}",
            )
        else:
            self.get_logger().info(
                f"mission: task {active.task_id} done ✓ ({verdict}); MISSION COMPLETE",
            )
            self._emit_mission_complete(mission, traceparent=traceparent)
        self._on_tick(force=True, tier="C")

    def _on_failure(self, source: str, msg: Any) -> None:
        """Append a failure event; preempt per the ADR-0018 trigger taxonomy.

        Tier A (``source == "safety"``) preempts on
        ``severity >= SEVERITY_WARN`` (=1) — a safety WARN means the
        C++ kernel (or F5 pass-through) saw a near-miss and the LLM
        needs to be in the loop before the next chunk lands.

        Tier B (``hal`` / ``sensor`` / ``rskill`` / ``wam``) and Tier C
        (``critic``) preempt on ``severity >= SEVERITY_FAIL`` (=2);
        WARN/INFO are buffered without preemption.

        See ADR-0018 amendment 2026-05-25 §3 for the full taxonomy.
        """
        record = FailureEventRecord(
            source=source,
            kind=int(msg.kind),
            severity=int(msg.severity),
            evidence_json=msg.evidence_json,
            rskill_id=msg.rskill_id,
            trace_id=msg.trace_id,
            stamp_ns=int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec),
        )
        self._renderer.append_failure(record)
        # ADR-0074 §2 — a reward-watcher wake (critic FAIL) while a VLA is in
        # flight is the *primary* stop: cancel the attempt now so the verify
        # gate runs on the reward signal rather than burning the rest of the
        # deadline clock. The canceled result re-enters
        # ``_maybe_verify_active_mission_task`` (the three-tier / VLM gate),
        # which forces the next full-palette tick. When no goal is in flight
        # (e.g. between tasks) the wake falls through to the ordinary preempt.
        if (
            _is_reward_wake(source=source, severity=record.severity, severity_fail=_SEVERITY_FAIL)
            and self._active_rskill_goal is not None
            and self._rskill_cancel_reason != "reward"
        ):
            self._cancel_inflight_rskill_for_reward()
            return
        preempt_threshold = _SEVERITY_WARN if source == "safety" else _SEVERITY_FAIL
        if record.severity >= preempt_threshold:
            self._on_tick(force=True, tier=_FAILURE_TIER_FOR_SOURCE.get(source, "B"))

    def _band_edges(self) -> tuple[float, float]:
        """Three-tier verdict band edges from the active reward calibration (ADR-0074 §1/§5).

        Thin adapter over :func:`openral_reasoner.completion.resolve_band_edges`
        — the live ``RewardContract`` when wired, else the system fallback.
        """
        c = self._reward_contract
        return _resolve_band_edges(
            contract_threshold=c.success_threshold if c is not None else None,
            contract_floor=c.check_floor if c is not None else None,
            fallback_threshold=_DEFAULT_SUCCESS_THRESHOLD,
            fallback_floor=_DEFAULT_CHECK_FLOOR,
        )

    def _effective_patience_s(self, call: ExecuteRskillTool) -> float:
        """Patience ceiling for a dispatch (ADR-0074 §2/§3).

        Thin adapter over :func:`openral_reasoner.completion.resolve_patience_s`
        (LLM ``patience_s`` override > reward-model ``default_patience_s`` >
        legacy ``deadline_s``). The result is sent as the goal's ``deadline_s``
        (the runner's backstop) and arms the reasoner-side timer; the
        reward-watcher is the usual stop.
        """
        c = self._reward_contract
        return _resolve_patience_s(
            override=call.patience_s,
            contract_default=c.default_patience_s if c is not None else None,
            legacy_deadline_s=call.deadline_s,
        )

    def _cancel_inflight_rskill_for_reward(self) -> None:
        """Cancel the in-flight execute_rskill goal on a reward-watcher wake (ADR-0074 §2).

        Sets ``_rskill_cancel_reason = "reward"`` so the canceled result runs
        the verify gate (a reward-ended attempt), then requests the cancel. A
        failed cancel request is non-fatal — the deadline timer is still the
        backstop and the result callback fires regardless, with the reason
        already latched.
        """
        assert self._active_rskill_goal is not None
        goal_handle, call, _traceparent = self._active_rskill_goal
        self._rskill_cancel_reason = "reward"
        self.get_logger().info(
            f"reward wake: cancelling in-flight execute_rskill {call.rskill_id!r} "
            "to verify on the reward signal (ADR-0074 §2)",
        )
        try:
            goal_handle.cancel_goal_async()
        except Exception as exc:  # reason: cancel is best-effort; deadline backstops it
            self.get_logger().error(
                f"reward-cancel cancel_goal_async failed: {type(exc).__name__}: {exc}",
            )

    def _on_perception(self, kind: str, msg: Any) -> None:
        """Append a perception event; no preemption — perception is informational.

        ADR-0076: an ``objects`` event also refreshes the reasoner's camera-space
        ``in_view`` enumeration (the continuous detector's 2D detections with stable
        det_ids), so the LLM can ground a goal noun / decompose a collective task
        even when the 3D lift (``scene_objects``) cannot run (RGB-only / no depth).
        """
        self._renderer.append_perception(
            PerceptionEventRecord(
                kind=kind,
                text=msg.text,
                metadata_json=msg.metadata_json,
                stamp_ns=int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec),
            ),
        )
        if kind == "objects":
            try:
                self._renderer.set_in_view(ObjectsMetadata.model_validate_json(msg.metadata_json))
            except ValidationError as exc:
                self.get_logger().debug(f"in_view: dropping malformed objects metadata: {exc!s}")

    def _on_prompt(self, msg: Any) -> None:
        """Append an operator prompt; preempt the tick to handle it quickly.

        Filters out prompts the reasoner itself just emitted — both the
        reasoner subscriber and the EmitPromptTool dispatcher are on
        ``/openral/prompt``, so a self-emit without this guard creates
        an infinite feedback loop ("system ready, please provide a
        task" → reasoner sees it as a new prompt → forces a tick →
        model picks emit_prompt again → ...). frame_id is stamped to
        ``openral_reasoner`` on every outbound EmitPrompt; we drop
        inputs that carry that tag.

        Resets the core's consecutive-tool streak before forcing the
        tick. The retry-cap gate exists to prevent the model from
        looping on the same failure mode against a static context;
        a fresh operator prompt is a new situation, so the previous
        streak carries no information — without the reset it would
        silently suppress the very tick this prompt triggered.
        """
        # frame_id is the canonical "who sent this"; the prompt_router
        # rewrites it to the source name (cli / dashboard / auto) for
        # external sources, but our own EmitPromptTool dispatcher
        # writes "openral_reasoner". The router preserves frame_id
        # when fanning out to /openral/prompt so the filter is robust
        # against routing.
        if str(getattr(msg.header, "frame_id", "") or "") == self.get_name():
            return
        source = str(getattr(msg.header, "frame_id", "") or "")
        self._renderer.append_prompt(
            PromptRecord(
                text=msg.text,
                metadata_json=msg.metadata_json,
                stamp_ns=int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec),
            ),
        )
        # ADR-0073 §1 — a genuine operator goal (re)builds the mission queue:
        # the operator goal seeds one task so the reasoner sequences and the
        # goal survives the pull-once prompt drain. Cascade re-prompts
        # (advisory query responses, spatial-memory) are NOT new goals and must
        # not reset the mission. The operator goal seeds one task; the
        # `decompose-mission` playbook decomposes/repopulates the queue.
        if source not in _CASCADE_PROMPT_SOURCES:
            mission = MissionState.from_prompt(msg.text)
            if not mission.is_empty():
                self._renderer.set_mission(mission)
                self._subdivide_offered.clear()  # #123 — fresh goal, fresh offers
                self.get_logger().info(
                    f"mission: {len(mission)} task(s) — active={mission.active().text[:80]!r}",
                )
        if self._core is not None:
            self._core.reset_kind_streak()
        # A fresh *operator* prompt is a new goal — reset the active-search bound.
        # The cascade's own "spatial_memory" re-prompts must NOT reset it, or the
        # bound never accumulates.
        if str(getattr(msg.header, "frame_id", "") or "") != "spatial_memory":
            self._spatial_search.reset()
            self._locate_escalated.clear()
        self._on_tick(force=True, tier="D")

    def _on_skill_registry_changed(self, msg: Any) -> None:
        """ADR-0018 §4 — rebuild the tool palette from the local rSkill registry.

        Fired by ``ral skill install|remove``. Walks the on-disk
        registry, loads each :class:`~openral_core.RSkillManifest`, and
        runs :func:`build_tool_palette` against the active
        :attr:`robot_capabilities`. Calls :meth:`set_palette` with the
        result.

        Without ``robot_capabilities`` set on the constructor the
        callback logs a warning and leaves the palette alone — the
        reasoner_node has no way to know which embodiment tags to
        match, so producing an unfiltered palette would risk
        dispatching a skill onto an incompatible robot.
        """
        del msg  # Empty payload; the topic is the signal.
        # Two refresh sources exist:
        #   (a) ``rskill_search_paths`` was set on the constructor params
        #       (the deploy_sim path) — re-run the full seed pipeline so
        #       in-tree manifests + the wrapped-ROS graph-availability
        #       filter re-evaluate against the (now-richer) ROS graph.
        #   (b) Only the installed-skills registry exists (the
        #       ``ral skill install`` path) — fall back to
        #       :meth:`_rebuild_palette_from_registry`.
        # Without this branch the wrapped-ROS rSkills shipped via
        # ``rskills/*/rskill.yaml`` never re-enter the palette when
        # Nav2 / MoveIt finish bringing up, because
        # ``rSkill.list_installed()`` only sees globally-installed skills.
        search_paths: list[str] = list(
            self.get_parameter("rskill_search_paths").get_parameter_value().string_array_value,
        )
        old_count = len(self._palette.execute_rskill_ids)
        if any(p for p in search_paths):
            try:
                self._maybe_seed_palette_from_search_paths()
            except Exception as exc:  # reason: surface seed pipeline issues
                self.get_logger().error(
                    f"palette refresh (search-paths) failed: {type(exc).__name__}: {exc}",
                )
                return
            new_count = len(self._palette.execute_rskill_ids)
            self.get_logger().info(
                f"palette refreshed (search-paths): {old_count} → {new_count} skills",
            )
            return
        if self._robot_capabilities is None:
            self.get_logger().warning(
                "/openral/skill_registry_changed fired but robot_capabilities is None — "
                "palette refresh skipped. Pass robot_capabilities to the ReasonerNode "
                "constructor to enable refreshes.",
            )
            return
        try:
            new_palette = self._rebuild_palette_from_registry()
        except Exception as exc:  # reason: surface registry load issues
            self.get_logger().error(
                f"palette refresh failed: {type(exc).__name__}: {exc}",
            )
            return
        new_count = len(new_palette.execute_rskill_ids)
        self.set_palette(new_palette)
        self.get_logger().info(
            f"palette refreshed: {old_count} → {new_count} skills",
        )

    def _rebuild_palette_from_registry(self) -> ToolPalette:
        """Load the installed rSkill manifests and run :func:`build_tool_palette`.

        ``openral_rskill`` is a heavy dep (pulls torch); lazy-imported
        here so the reasoner_node module stays cheap to import.
        """
        from openral_core import RSkillManifest
        from openral_rskill.loader import rSkill

        installed = rSkill.list_installed()
        manifests: list[RSkillManifest] = []
        for entry in installed:
            try:
                manifests.append(RSkillManifest.from_yaml(entry.manifest_path))
            except (OSError, ValueError) as exc:
                self.get_logger().warning(
                    f"skipping unloadable rSkill {entry.repo_id!r}: {exc}",
                )
        assert self._robot_capabilities is not None  # caller-guarded
        return build_tool_palette(
            installed_skills=manifests,
            robot_capabilities=self._robot_capabilities,
            sensor_ids=self._palette.sensor_ids,
            node_ids=self._palette.node_ids,
            commercial_deployment=self._commercial_deployment,
            spatial_memory_available=self._spatial_memory is not None,
            detector_available=self._detector_available,
            scene_query_available=self._scene_query_available,
            task_progress_available=self._task_progress_available,
        )

    def _maybe_load_memory(self) -> None:
        """Load the self-maintained ``MEMORY.md`` into the ``## MEMORY`` block (ADR-0072 §3).

        Read path (Phase 4b): when the ``memory_md_path`` ROS parameter is set, parse
        the file (or start empty if absent) and render it as the reasoner's persistent
        ``## MEMORY`` context section. Write path (Phase 4c): record the path + load the
        ``<MEMORY.md>.archive.jsonl`` recall log, and advertise the ``memory_write`` /
        ``memory_search`` tools by flipping ``memory_available`` on the palette.
        Advisory only — never gates the safety kernel.
        """
        path = self.get_parameter("memory_md_path").get_parameter_value().string_value
        if not path:
            return
        p = pathlib.Path(path)
        try:
            text = p.read_text(encoding="utf-8") if p.exists() else ""
        except OSError as exc:
            self.get_logger().warning(f"memory: failed to read {path!r}: {exc}; starting empty")
            text = ""
        self._memory_store = MemoryStore.from_markdown(text)
        self._memory_md_path = p
        self._memory_archive = self._load_memory_archive(p)
        self._renderer.set_memory_block(self._render_memory_block())
        if not self._palette.memory_available:
            self._palette = self._palette.model_copy(update={"memory_available": True})
        self.get_logger().info(
            f"memory: loaded {len(self._memory_store.entries)} entries "
            f"(+{len(self._memory_archive)} archived) from {path!r}; write tools enabled",
        )

    def _render_memory_block(self) -> str:
        """Render the ``## MEMORY`` block under the ``memory_context_cap`` param (Phase 5)."""
        assert self._memory_store is not None
        cap = self.get_parameter("memory_context_cap").get_parameter_value().integer_value
        return self._memory_store.to_context_block(cap=cap if cap > 0 else None)

    @staticmethod
    def _memory_archive_path(memory_md_path: pathlib.Path) -> pathlib.Path:
        """The append-only recall log beside the ``MEMORY.md`` (MemGPT archival store)."""
        return memory_md_path.with_name(memory_md_path.name + ".archive.jsonl")

    def _load_memory_archive(self, memory_md_path: pathlib.Path) -> list[MemoryEntry]:
        """Parse the archival JSONL recall log (one entry per line); ``[]`` if absent/bad."""
        archive_path = self._memory_archive_path(memory_md_path)
        if not archive_path.exists():
            return []
        entries: list[MemoryEntry] = []
        try:
            for line in archive_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                entries.append(
                    MemoryEntry(
                        section=rec["section"],
                        content=rec["content"],
                        importance=float(rec.get("importance", 0.5)),
                        timestamp=rec.get("timestamp", ""),
                        status=rec.get("status", "stale"),
                    )
                )
        except (OSError, ValueError, KeyError) as exc:
            self.get_logger().warning(
                f"memory: failed to read archive {archive_path!r}: {exc}; starting empty",
            )
            return []
        return entries

    def _maybe_load_spatial_memory(self) -> None:
        """Wire the ADR-0038 spatial-memory backend at ``on_configure`` (ADR-0039 / ADR-0038).

        No-op when a ``spatial_memory`` backend was injected at construction.
        Otherwise: if ``spatial_memory_path`` is set, load that persisted scene
        graph; else if ``spatial_memory_ingest`` is set, start an empty memory
        that ``_on_tick`` accumulates from live ``WorldState.detected_objects``.
        A path load failure degrades gracefully to no backend (logged at WARNING)
        — never a fabricated map (CLAUDE.md §1.2). When the reasoner owns the
        backend (either case) it keeps a concrete ``_spatial_memory_writer`` so
        the tick can fold detections in; an injected read-only querier is left
        un-owned and unmutated.
        """
        if self._spatial_memory is not None:
            return
        path = self.get_parameter("spatial_memory_path").get_parameter_value().string_value
        ingest = self.get_parameter("spatial_memory_ingest").get_parameter_value().bool_value
        if not path and not ingest:
            return
        from openral_world_state import SpatialMemory

        if path:
            try:
                memory = SpatialMemory.load(path)
            except (OSError, ValueError) as exc:
                self.get_logger().warning(
                    f"spatial_memory_path={path!r} failed to load; query tools disabled: {exc}",
                )
                return
            origin = f"loaded spatial memory from {path!r}"
        else:
            memory = SpatialMemory()
            origin = "started empty spatial memory for live ingest"
        self._spatial_memory = memory
        self._spatial_memory_writer = memory
        if not self._palette.spatial_memory_available:
            self._palette = self._palette.model_copy(update={"spatial_memory_available": True})
        node_count = len(memory.to_scene_graph().nodes)
        self.get_logger().info(
            f"on_configure: {origin} ({node_count} nodes; ingest={ingest}); "
            "recall_object / resolve_place tools enabled",
        )
        # Publish the (possibly empty) map once now so the dashboard shows it
        # before the first heartbeat tick (which re-emits on the 0.2 Hz cadence).
        self._emit_scene_objects_span()

    def _emit_scene_objects_span(self) -> None:
        """Publish the remembered objects as a ``world.scene_objects`` span (ADR-0038).

        Advisory dashboard telemetry only (never a safety input). No-op without a
        spatial-memory backend; any failure is swallowed at DEBUG so a telemetry
        hiccup can never disturb the reasoning loop. Today the backend is the
        preloaded ``spatial_memory_path`` map; post-producer (ADR-0035 / PR #229)
        the World-State node becomes the canonical emitter of the same span.
        """
        if self._spatial_memory is None:
            return
        try:
            from openral_world_state import emit_scene_objects_span

            emit_scene_objects_span(
                self._spatial_memory.to_scene_graph(),
                source_node=self.get_name(),
            )
        except Exception as exc:  # reason: telemetry must never break the tick
            self.get_logger().debug(f"scene-objects span emit failed: {exc!s}")

    def _ingest_detected_objects(self, world_state: Any) -> None:
        """Fold a snapshot's ``detected_objects`` into the owned SpatialMemory (ADR-0038).

        No-op unless the reasoner owns a writable backend (``spatial_memory_ingest``
        or a preloaded map) and the snapshot carries detections. Accrual is
        advisory: failures degrade at DEBUG so a hiccup never disturbs the tick.
        Uses the snapshot's ``stamp_ns`` for recency (deterministic in sim),
        falling back to wall-clock.
        """
        writer = self._spatial_memory_writer
        if writer is None or world_state is None:
            return
        objects = getattr(world_state, "detected_objects", None)
        if not objects:
            return
        try:
            now_ns = int(getattr(world_state, "stamp_ns", 0)) or time.time_ns()
            touched = writer.ingest_detected_objects(objects, now_ns=now_ns)
            self.get_logger().debug(
                f"spatial-memory ingest: {len(touched)} node(s) from {len(objects)} detection(s)",
            )
        except Exception as exc:  # reason: memory accrual must never break the tick
            self.get_logger().debug(f"spatial-memory ingest failed: {exc!s}")

    def _collect_playbooks_block(
        self, manifests: list[RSkillManifest], paths: list[pathlib.Path]
    ) -> str:
        """Render the ``## PLAYBOOKS`` block from installed, matched playbook rSkills.

        ADR-0072 Phase 3: for each ``kind: playbook`` manifest this robot satisfies
        (embodiment + capability flags), read its ``PLAYBOOK.md`` body and render it
        for the system prompt. Returns ``""`` when none match.
        """
        from openral_core.exceptions import ROSCapabilityMismatch
        from openral_rskill.loader import rSkill

        entries: list[tuple[str, str]] = []
        for manifest, path in zip(manifests, paths, strict=True):
            if manifest.kind != "playbook" or manifest.playbook is None:
                continue
            if self._robot_capabilities is not None:
                try:
                    rSkill.check_capabilities(manifest, self._robot_capabilities)
                except ROSCapabilityMismatch as exc:
                    self.get_logger().info(f"playbook {manifest.name!r} not installed: {exc}")
                    continue
            body_path = (path.parent / manifest.playbook.body_uri).resolve()
            try:
                body = body_path.read_text(encoding="utf-8")
            except OSError as exc:
                self.get_logger().warning(
                    f"playbook {manifest.name!r}: cannot read body {body_path}: {exc}",
                )
                continue
            # Label with the bare playbook name (strip the ``<org>/rskill-`` prefix):
            # the full machine id reads like an executable skill id and tempts the
            # LLM to call ``execute_rskill`` on the playbook itself (ADR-0072 — a
            # playbook is an SOP to follow, never a dispatch target).
            label = manifest.name.split("/")[-1].removeprefix("rskill-")
            entries.append((f"{label} — {manifest.playbook.trigger}", body))
        if entries:
            self.get_logger().info(f"playbooks: injected {len(entries)} into the system prompt")
        return render_playbooks_block(entries)

    def _maybe_seed_palette_from_search_paths(self) -> None:  # noqa: PLR0912, PLR0915  # reason: linear palette-seed pipeline (load → capability filter → ros-server probe → state-contract probe → import-deps probe → build); splitting hides the filter order
        """Populate the palette from in-tree ``rskills/<id>/rskill.yaml`` files.

        Triggered once at lifecycle ``on_configure``. Inspects two ROS
        parameters set by the launch:

        * ``robot_yaml`` — absolute path to ``robots/<id>/robot.yaml``;
          loaded via :meth:`RobotDescription.from_yaml`. The
          :attr:`RobotDescription.capabilities` is then the filter
          basis for :func:`build_tool_palette`, replacing the
          constructor-supplied :attr:`_robot_capabilities` if it was
          ``None``.
        * ``rskill_search_paths`` — list of directory paths (each a
          glob root for ``*/rskill.yaml``). Empty / unset means "skip
          the seed step and leave the palette where it is".

        Failure of either path is non-fatal — it falls back to the
        existing :meth:`/openral/skill_registry_changed` refresh path.
        Per-file errors are warned, not raised, so a single broken
        manifest doesn't block the bring-up.
        """
        from openral_core import RobotDescription, RSkillManifest

        robot_yaml: str = self.get_parameter("robot_yaml").get_parameter_value().string_value
        search_paths_raw: list[str] = list(
            self.get_parameter("rskill_search_paths").get_parameter_value().string_array_value,
        )
        search_paths = [p for p in search_paths_raw if p]
        if not robot_yaml or not search_paths:
            return

        try:
            description = RobotDescription.from_yaml(robot_yaml)
        except (OSError, ValueError) as exc:
            self.get_logger().warning(
                f"palette seed skipped: failed to load robot_yaml={robot_yaml!r}: {exc}",
            )
            return
        self._robot_capabilities = description.capabilities
        # ADR-0072 Decision 2.1 — render the static robot self-model once and
        # surface it as the reasoner's `## ROBOT` context section so the LLM can
        # judge reach/view feasibility before dispatching a skill.
        self._renderer.set_robot_model(render_robot_self_model(description))

        manifests: list[RSkillManifest] = []
        manifest_paths: list[pathlib.Path] = []
        for root_str in search_paths:
            root = pathlib.Path(root_str)
            if not root.exists():
                self.get_logger().warning(
                    f"palette seed: rskill_search_path {root_str!r} does not exist; skipping",
                )
                continue
            manifest_paths.extend(sorted(root.glob("*/rskill.yaml")))

        loaded_paths: list[pathlib.Path] = []
        for path in manifest_paths:
            try:
                manifests.append(RSkillManifest.from_yaml(str(path)))
                loaded_paths.append(path)
            except (OSError, ValueError) as exc:
                self.get_logger().warning(
                    f"palette seed: skipping unloadable rskill {path!s}: {exc}",
                )

        # ADR-0072 Phase 3 — collect installed, capability-matched `kind: playbook`
        # rSkills and render their PLAYBOOK.md bodies into the `## PLAYBOOKS`
        # system-prompt block. Playbooks are role:s2 (excluded from the ExecuteSkill
        # palette); they reach the LLM as authored decision-procedure *content*.
        self._playbooks_block = self._collect_playbooks_block(manifests, loaded_paths)

        # ADR-0025 — merge any deploy-time lifecycle peer node ids
        # (e.g. /openral_slam_toolbox when --enable-slam was passed) into
        # the palette's `node_ids` set so the Reasoner's LLM can target
        # them via LifecycleTransitionTool. The seed list comes from the
        # `lifecycle_peer_node_ids` ROS parameter; empty entries skipped.
        peer_ids: list[str] = list(
            self.get_parameter("lifecycle_peer_node_ids").get_parameter_value().string_array_value
        )
        merged_node_ids = self._palette.node_ids | frozenset(p for p in peer_ids if p)

        # Capability-filter FIRST: drop manifests whose embodiment_tags
        # / sensors_required / actuators_required / role / license don't
        # match this robot. Then probe import-deps on just the survivors.
        # The opposite order (deps first, capability second) generates
        # noisy warnings for manifests that would never have been in the
        # palette anyway — e.g. when running ``deploy sim`` on
        # ``panda_mobile``, the ``xvla-libero`` rSkill targets
        # ``franka_panda`` so it's filtered out by embodiment, but the
        # deps-first ordering emits a spurious
        # "dropping rSkill 'xvla-libero': No module named 'xvla'"
        # warning even though the user never needed xvla installed.
        capability_palette = build_tool_palette(
            installed_skills=manifests,
            robot_capabilities=self._robot_capabilities,
            sensor_ids=self._palette.sensor_ids,
            node_ids=merged_node_ids,
            commercial_deployment=self._commercial_deployment,
        )
        capability_matched_ids = capability_palette.execute_rskill_ids
        capability_matched = [m for m in manifests if m.name in capability_matched_ids]

        # Wrapped-ROS server availability filter: drop ``ros_action`` /
        # ``ros_service`` rSkills whose ``ros_integration.interface_name``
        # isn't currently advertised on the ROS graph. Without this,
        # the reasoner LLM dispatches the nav2 / moveit / look-at
        # wrapper skills against absent backends and the
        # adapter raises ``ROSConfigError: action server X did not
        # come up within 15.0s`` per dispatch — a 15s ERROR per
        # autonomous tick. We can't fix the missing backend from this
        # process (the operator has to bring up MoveIt / Nav2 /
        # gripper controllers separately), so we filter at boot. The
        # check is best-effort by design: action servers that come
        # up later won't auto-re-enter the palette until the next
        # ``/openral/skill_registry_changed`` refresh, which is fine
        # for the deploy-sim use case (the launcher brings up the
        # backends or it doesn't; mid-run additions are rare).
        topic_names_and_types = self.get_topic_names_and_types()
        graph_topics = {name for name, _ in topic_names_and_types}
        ros_server_available: list = []
        for m in capability_matched:
            if m.kind not in {"ros_action", "ros_service"}:
                ros_server_available.append(m)
                continue
            integration = m.ros_integration
            if integration is None:
                # Manifest declares wrapped-ROS but no integration —
                # the schema enforces this, so this is defensive.
                self.get_logger().warning(
                    f"palette: dropping rSkill {m.name!r} (kind={m.kind!r}): "
                    f"manifest is missing required ros_integration block."
                )
                continue
            interface_name = integration.interface_name
            # Action servers advertise ``<name>/_action/feedback``,
            # ``..goal``, etc. Services advertise ``<name>`` as a service
            # not a topic, so check ``get_service_names_and_types`` too.
            action_present = any(t.startswith(f"{interface_name}/_action/") for t in graph_topics)
            service_present = False
            if m.kind == "ros_service":
                service_names = {s for s, _ in self.get_service_names_and_types()}
                service_present = interface_name in service_names
            if not (action_present or service_present):
                self.get_logger().warning(
                    f"palette: dropping rSkill {m.name!r} (kind={m.kind!r}): "
                    f"interface {interface_name!r} is not advertised on the "
                    f"ROS graph. The wrapped server isn't running in this "
                    f"deployment — bring it up (e.g. via the matching "
                    f"controller / Nav2 / MoveIt launch include) and "
                    f"retrigger the palette via "
                    f"/openral/skill_registry_changed, or pick a different "
                    f"rSkill."
                )
                continue
            ros_server_available.append(m)
        capability_matched = ros_server_available

        # State-contract compatibility filter: drop VLA rSkills whose
        # ``state_contract.dim`` is incompatible with the robot's
        # joint count. The deploy_sim observation pipeline feeds the
        # HAL's raw ``JointState`` (one float per joint) into the
        # adapter; VLA rSkills with a wrapped state layout
        # (``rc365``/``human300_16d``/``libero``/``gr1``) expect the
        # SIM ADAPTER's composed state shape, not the raw joint
        # vector. When the LLM autonomously dispatches such a skill
        # the rldx / pi05 adapter raises ``ROSConfigError: expects
        # a 16-D state for state_layout=..., got 10-D`` mid-run. Pre-
        # filtering at palette seed turns a 5 Hz dispatch failure
        # into a single ``palette: dropping...`` warning at boot.
        # Wrapped-ROS skills (``kind: ros_action`` / ``ros_service``)
        # bypass this — they don't consume ``observation.state`` at
        # all, so any state_contract on them is informational.
        n_joints = len(description.joints)
        state_compatible: list[RSkillManifest] = []
        for m in capability_matched:
            sc = m.state_contract
            if m.kind == "vla" and sc is not None and sc.dim != n_joints:
                if sc.layout in _WRAPPED_TASK_SPACE_LAYOUTS:
                    # ADR-0027 — admit-with-adapter when the layout's
                    # assembler is registered in the openral_state_adapter
                    # registry. The skill_runner injects a live TF lookup
                    # at step time so the manifest-declared bindings
                    # resolve against the real /tf graph.
                    # Defer the import — keeps the reasoner_node
                    # module-load path off the openral_state_adapter
                    # tree until we actually consult it.
                    from openral_state_adapter import registered_layouts

                    if sc.layout in registered_layouts():
                        self.get_logger().info(
                            f"palette: admitting rSkill {m.name!r} "
                            f"(model_family={m.model_family!r}): "
                            f"wrapped task-space layout {sc.layout!r} "
                            f"(dim={sc.dim}) has a registered assembler "
                            "in openral_state_adapter (ADR-0027). "
                            "The skill_runner will assemble observation."
                            "state from live /tf at step time."
                        )
                        state_compatible.append(m)
                        continue
                    # Informational drop: the layout is a task-space
                    # composite the in-tree deploy_sim path doesn't
                    # synthesise — no assembler is registered. Register
                    # one under python/state_adapter/src/openral_state_adapter
                    # /layouts/<layout>.py to admit this rSkill.
                    self.get_logger().info(
                        f"palette: skipping rSkill {m.name!r} "
                        f"(model_family={m.model_family!r}): targets "
                        f"wrapped task-space layout {sc.layout!r} "
                        f"(dim={sc.dim}); no assembler registered "
                        "in openral_state_adapter for this layout. "
                        "Add one or run via "
                        "``openral sim run --vla ...``."
                    )
                else:
                    self.get_logger().warning(
                        f"palette: dropping rSkill {m.name!r} "
                        f"(model_family={m.model_family!r}): "
                        f"state_contract.dim={sc.dim} (layout={sc.layout!r}) "
                        f"is incompatible with this robot's joint count "
                        f"({n_joints}). Pick a state-compatible rSkill "
                        f"for ``deploy sim``."
                    )
                continue
            state_compatible.append(m)

        # Action-mode executability filter (ADR-0036): drop VLA rSkills
        # whose action vector drives a ControlMode the deploy path can't
        # execute. The state-contract filter above gates the *input*
        # (state dim vs joint count); this gates the *output*. Without it
        # a cartesian/OSC skill gets offered to the LLM on a joint-only
        # robot and fails at runtime — the n_dof / control-mode mismatch
        # surfaces as a mid-run estop instead of a single boot-time
        # warning. ``hal_mode`` selects the executable set: ``"sim"``
        # admits the robosuite-OSC default set even on a joint-only
        # physical robot; ``"real"`` admits only the robot's declared
        # ``supported_control_modes``. Non-vla skills (``ros_action`` /
        # ``ros_service``) pass through — they don't emit an
        # ``ActionChunk`` from a learned action vector.
        hal_mode = self.get_parameter("hal_mode").get_parameter_value().string_value or "sim"
        if hal_mode == "sim":
            executable_modes = set(SIM_EXECUTABLE_CONTROL_MODES)
        else:
            executable_modes = {
                ControlMode(x) for x in description.capabilities.supported_control_modes
            }
        executable_repr = sorted(c.value for c in executable_modes)
        action_executable: list[RSkillManifest] = []
        for m in state_compatible:
            if m.kind == "vla":
                legacy_ok = _action_executable(m, description, hal_mode)
                # ADR-0071 Phase 2 — shadow the canonical TaskSpace gate alongside
                # the legacy mode check (warn-only; the legacy verdict still
                # decides the drop). Surfaces cross-layer mismatches the
                # ``_action_executable`` mode-set check misses — an EE-addressed
                # slot naming an end-effector the robot does not declare, or a
                # joint segment wider than the robot's joint count. Phase 4 makes
                # ``task_space_compatible`` authoritative.
                ts_warning = task_space_disagreement(m, description, hal_mode, legacy_ok)
                if ts_warning is not None:
                    self.get_logger().warning(ts_warning)
                if not legacy_ok:
                    required_repr = sorted(c.value for c in _required_control_modes(m))
                    self.get_logger().warning(
                        f"palette: dropping rSkill {m.name!r} "
                        f"(model_family={m.model_family!r}): requires control modes "
                        f"{required_repr} which are not executable on this deployment "
                        f"(hal_mode={hal_mode!r}; executable={executable_repr}). "
                        f"Pick an action-compatible rSkill or bring up a controller "
                        f"that executes these modes."
                    )
                    continue
            action_executable.append(m)

        # Import-deps filter on capability-matched survivors only.
        # Skills whose family is unknown to
        # ``policy_deps._FAMILY_REQUIRED_IMPORTS`` survive the filter —
        # better to surface a clearer factory-side error at dispatch
        # time than to silently drop a skill an out-of-tree family
        # registered. We probe ONCE at on_configure so the operator
        # sees a single warning per dropped skill with the actionable
        # install command instead of every ``execute_rskill`` dispatch
        # failing at goal-execute time with a confusing stack trace
        # through three layers of lerobot imports.
        from openral_sim.policy_deps import filter_importable_manifests

        importable = filter_importable_manifests(
            action_executable,
            log_fn=self.get_logger().warning,
        )
        n_dropped = len(capability_matched) - len(importable)

        new_palette = build_tool_palette(
            installed_skills=importable,
            robot_capabilities=self._robot_capabilities,
            sensor_ids=self._palette.sensor_ids,
            node_ids=merged_node_ids,
            commercial_deployment=self._commercial_deployment,
            # ADR-0039 — preserve the read-only query tools when a spatial-memory
            # backend is wired; `_maybe_load_spatial_memory` runs before this seed
            # and a rebuild without the flag would silently drop recall_object /
            # resolve_place.
            spatial_memory_available=self._spatial_memory is not None,
            detector_available=self._detector_available,
            scene_query_available=self._scene_query_available,
            task_progress_available=self._task_progress_available,
        )
        self._palette = new_palette
        self.get_logger().info(
            f"palette seeded from {len(manifest_paths)} manifest(s) "
            f"across {len(search_paths)} path(s): "
            f"{len(new_palette.execute_rskill_ids)} match robot capabilities"
            + (
                f" ({n_dropped} dropped by import-deps filter — see warnings above)"
                if n_dropped
                else ""
            ),
        )

    # ── tick + dispatch ─────────────────────────────────────────────────────

    def _handle_suppressed_tick(self, result: Any) -> None:
        """Log a suppressed tick and reflect on an exhausted retry streak.

        `min_interval` fires every fractional second and would spam at INFO;
        `heartbeat_idle` is the steady-state on a quiet system (one suppression
        per heartbeat period); both stay at DEBUG. Everything else is rare and
        operationally important — `retry_cap` in particular used to be silent
        and left operators wondering why their prompt did nothing.
        """
        if result.suppressed_reason in ("min_interval", "heartbeat_idle"):
            self.get_logger().debug(f"tick suppressed: {result.suppressed_reason}")
        elif result.suppressed_reason == "retry_cap":
            # Warn once per streak, not every heartbeat — otherwise this
            # floods the log while the model keeps re-picking the same tool.
            if not self._retry_cap_warned:
                self._retry_cap_warned = True
                cap = self._core._retry_cap if self._core is not None else "N"
                self.get_logger().warning(
                    f"tick suppressed: retry_cap — same tool kind {cap}+ ticks in a row. "
                    "A new operator prompt resets the streak; otherwise it self-clears "
                    "when the model picks a different tool. (Repeats logged at debug.)",
                )
                # ADR-0072 §2.3 — inject a Reflexion strategy hint into context
                # (once per streak) so the NEXT tick changes approach instead of
                # looping. Appending bumps `seq`, so the next heartbeat runs
                # rather than being suppressed as idle.
                if self._core is not None:
                    tool = self._core._kind_streak[0]
                    self._renderer.append_execution(
                        ExecutionEventRecord(
                            rskill_id="(ladder)",
                            outcome="failed",
                            summary=f"retry ladder exhausted for {tool!r}",
                            reflection=reflect_on_retry_cap(tool, self._core._retry_cap),
                            stamp_ns=self.get_clock().now().nanoseconds,
                        )
                    )
            else:
                self.get_logger().debug("tick suppressed: retry_cap (ongoing streak)")
            # An ongoing retry_cap streak keeps the one-shot latch set.
            return
        else:
            self.get_logger().info(f"tick suppressed: {result.suppressed_reason}")
        # Any suppression other than an ongoing retry_cap streak clears the
        # one-shot latch so the next streak warns again.
        self._retry_cap_warned = False

    def _on_tick(self, *, force: bool = False, tier: str = "heartbeat") -> None:
        """Run one orchestrator pass and dispatch the selected tool call.

        Args:
            force: Bypasses :class:`ReasonerCore`'s ``min_interval`` and
                ``heartbeat_idle`` gates. Set by callbacks that
                preempt — Tier A safety + operator prompts.
            tier: Trigger tier driving this call — ``"A"``/``"B"``/
                ``"C"``/``"D"`` for the four event tiers, or
                ``"heartbeat"`` (default) when the periodic timer fired
                with no preempting callback. Recorded on the OTel span
                as ``reasoner.tier`` for trace-filtering.
        """
        # Decode the latest /openral/world_state_slow IDL message into a
        # Pydantic `WorldState` once — used both for live spatial-memory ingest
        # (below) and, when the core is ready, the LLM context. Without it the
        # WORLD_STATE block in the LLM context just reads "(no snapshot yet)"
        # and the model keeps asking for state instead of dispatching a skill.
        world_state: Any = None
        if self._world_state_msg is not None:
            try:
                from openral_world_state_ros.lifecycle_node import (
                    world_state_from_idl,
                )

                world_state = world_state_from_idl(self._world_state_msg)
            except Exception as exc:  # reason: decode failures stay non-fatal
                self.get_logger().warning(
                    f"world_state_from_idl failed; ticking without snapshot: {exc!s}",
                )
                world_state = None
        # ADR-0038 — fold the snapshot's detected_objects into the durable memory
        # we own, then refresh the dashboard's scene-objects view. Both run on
        # every heartbeat, independent of LLM readiness (a preloaded/accumulating
        # map is worth maintaining even before the tool-use client is built).
        self._ingest_detected_objects(world_state)
        self._emit_scene_objects_span()
        if self._core is None:
            return
        result = self._core.tick(
            world_state=world_state,
            renderer=self._renderer,
            palette=self._palette,
            force=force,
            tier=tier,
        )
        if result.suppressed_reason:
            self._handle_suppressed_tick(result)
            return
        # A tick that was not suppressed (dispatch, error, or no-op) breaks any
        # retry_cap streak — clear the latch so a future streak warns again.
        self._retry_cap_warned = False
        if result.error is not None:
            self.get_logger().warning(f"tick error: {result.error!s}")
            # ADR-0072 §2.2/§2.3 — an invalid plan is the model's *own* mistake
            # (malformed JSON args, a non-object payload, a field/rskill_id the
            # palette rejects). Feed it back into the `## EXECUTION` section with
            # a Reflexion hint so the NEXT tick emits a valid call instead of
            # re-issuing the same broken one. (Provider/transport errors —
            # timeout, 403 — are not the model's fault, so they only log.)
            # Appending bumps `seq`, so the next heartbeat runs rather than being
            # suppressed as idle.
            if isinstance(result.error, ROSReasonerInvalidPlan):
                self._renderer.append_execution(
                    ExecutionEventRecord(
                        rskill_id="(invalid-plan)",
                        outcome="failed",
                        summary=f"undecodable tool call: {result.error!s}",
                        reflection=reflect_on_invalid_plan(str(result.error)),
                        stamp_ns=self.get_clock().now().nanoseconds,
                    )
                )
            return
        if result.tool_call is None:
            return
        self._dispatched_calls.append(result.tool_call)
        self._dispatch(result.tool_call, traceparent=result.traceparent)

    def _dispatch(self, call: Any, *, traceparent: str | None = None) -> None:  # noqa: PLR0911  # reason: one return per tool variant — a flat dispatch table is clearer than collapsing the isinstance branches
        """Route a typed tool call onto the ROS graph.

        :class:`EmitPromptTool` publishes inline. :class:`ExecuteRskillTool`
        sends an action goal on ``/openral/execute_rskill`` and wires
        feedback/result/timeout into ``/openral/failure/rskill``.
        :class:`LifecycleTransitionTool` calls ``<node>/change_state``.
        :class:`ReloadGstPipelineTool` remains a log-and-acknowledge
        stub pending the F6 sensor-package service IDL.
        """
        # ADR-0039 §3 — any non-search dispatch ends the search episode, so the
        # cascade bound counts only *consecutive* spatial queries (incl. the live
        # locate_in_view — see _resets_search_episode).
        if _resets_search_episode(call):
            self._spatial_search.reset()
            self._locate_escalated.clear()
        if isinstance(call, EmitPromptTool):
            self._dispatch_emit_prompt(call, traceparent=traceparent)
            return
        if isinstance(call, ExecuteRskillTool):
            self._dispatch_execute_rskill(call, traceparent=traceparent)
            return
        if isinstance(call, LifecycleTransitionTool):
            self._dispatch_lifecycle_transition(call)
            return
        if isinstance(call, RecallObjectTool | ResolvePlaceTool):
            self._dispatch_spatial_query(call, traceparent=traceparent)
            return
        if isinstance(call, LocateInViewTool):
            self._dispatch_locate_in_view(call, traceparent=traceparent)
            return
        if isinstance(call, QuerySceneTool):
            self._dispatch_query_scene(call, traceparent=traceparent)
            return
        if isinstance(call, QueryTaskProgressTool):
            self._dispatch_query_task_progress(call, traceparent=traceparent)
            return
        if isinstance(call, MemoryWriteTool):
            self._dispatch_memory_write(call, traceparent=traceparent)
            return
        if isinstance(call, MemorySearchTool):
            self._dispatch_memory_search(call, traceparent=traceparent)
            return
        if isinstance(call, DecomposeMissionTool):
            self._dispatch_decompose_mission(call)
            return
        if isinstance(call, ReloadGstPipelineTool):
            # F6 sensor-package service IDL (e.g.
            # openral_sensor_msgs/srv/ReloadGstPipeline) is not yet on
            # disk; the client harness is a one-liner once the schema
            # lands. Logged at warning so it surfaces in operator logs
            # without spamming when the reasoner picks the tool
            # repeatedly.
            self.get_logger().warning(
                f"dispatch: reload_gst_pipeline sensor_id={call.sensor_id!r} ignored — "
                "F6 sensor-package service IDL not yet on disk (see GH-126).",
            )
            return
        self.get_logger().warning(f"dispatch: unhandled tool call {type(call).__name__}")

    def _dispatch_emit_prompt(
        self,
        call: EmitPromptTool,
        *,
        traceparent: str | None,
    ) -> None:
        """Publish a :class:`PromptStamped` on the target topic.

        ADR-0018 §6 — the active OTel traceparent (captured by
        :meth:`ReasonerCore.tick` while the ``reasoner.tick`` span is
        open) is stamped into ``metadata_json`` so the F7 bag↔OTel
        correlator can join the published prompt back to the reasoner
        span that produced it.
        """
        assert self._prompt_pub is not None
        msg = IDLPromptStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "openral_reasoner"
        msg.text = call.text
        try:
            base_metadata = json.loads(call.metadata_json) if call.metadata_json else {}
            if not isinstance(base_metadata, dict):
                base_metadata = {"_inbound": base_metadata}
        except json.JSONDecodeError:
            base_metadata = {"_inbound_raw": call.metadata_json}
        base_metadata.setdefault("source", "openral_reasoner")
        base_metadata.setdefault("rationale", call.rationale)
        if traceparent is not None:
            base_metadata["traceparent"] = traceparent
        msg.metadata_json = json.dumps(base_metadata, sort_keys=True)
        self._prompt_pub.publish(msg)
        self.get_logger().info(
            f"dispatch: emit_prompt → {call.target_topic} text={call.text!r}",
        )

    def _dispatch_spatial_query(
        self,
        call: RecallObjectTool | ResolvePlaceTool,
        *,
        traceparent: str | None,
    ) -> None:
        """Run a read-only spatial-memory query and re-prompt with the result (ADR-0039).

        The query runs against the injected ADR-0038 ``SpatialMemory`` backend and
        the rendered result is republished as a ``PromptStamped`` with frame_id
        ``"spatial_memory"`` (so ``_on_prompt`` consumes it rather than filtering
        it as a reasoner self-emit), feeding the answer into the next tick — the
        prompt cascade. Read-only: no actuation, no ``FailureTrigger``.

        ADR-0039 §3 bound: consecutive queries are counted against a
        ``SearchBudget``; once exhausted the result is published with the
        reasoner's own frame_id (so ``_on_prompt`` filters it — no further tick),
        terminating the search in human-handoff instead of looping forever.
        """
        if self._spatial_memory is None:
            self.get_logger().warning(
                f"dispatch: {call.tool} received but no SpatialMemory backend is wired",
            )
            return
        assert self._prompt_pub is not None
        now_ns = self.get_clock().now().nanoseconds
        # ADR-0044 Phase 4 — when a slam map is online, every recall_object
        # approach viewpoint is validated/snapped against it (free under the
        # robot footprint + line-of-sight) before the LLM sees it; a match
        # with no reachable viewpoint is rendered BLOCKED, never fabricated.
        refiner = None
        if self._occupancy_grid is not None:
            # Layer-2 import deferred like SpatialMemory in _maybe_load_spatial_memory.
            from openral_world_state.grid import refine_approach_pose

            grid = self._occupancy_grid
            inflation_m = (
                self.get_parameter("approach_inflation_m").get_parameter_value().double_value
            )

            # ApproachRefiner protocol; openral_core types resolved at the call site.
            def refiner(viewpoint: Any, target_xyz: tuple[float, float, float]) -> Any:
                return refine_approach_pose(grid, viewpoint, target_xyz, inflation_m=inflation_m)

        outcome = run_spatial_query_detailed(
            call, self._spatial_memory, now_ns=now_ns, refine_approach=refiner
        )
        result_text = outcome.text

        # ADR-0043/0056 — a recall_object MISS escalates to a live locate_in_view
        # (open-vocab, same query) BEFORE the search budget runs out and we hand
        # off. The on-demand detector grounds objects the spatial map never
        # ingested, and matches the goal term verbatim even when the stored label
        # differs (e.g. recall "baguette" vs ingested "bread"). This is policy —
        # it does not depend on the LLM choosing locate_in_view. One escalation
        # per query term per search streak so a repeated miss can't spam the
        # detector; if locate also misses, the normal budget/handoff path resumes.
        if (
            isinstance(call, RecallObjectTool)
            and not outcome.found
            and self._detector_available
            and call.query not in self._locate_escalated
        ):
            self._locate_escalated.add(call.query)
            self.get_logger().info(
                f"dispatch: recall_object miss for {call.query!r} → escalating to "
                "locate_in_view (live detector) before handoff",
            )
            self._dispatch_locate_in_view(
                LocateInViewTool(query=call.query, detector=self._default_on_demand_detector),
                traceparent=traceparent,
            )
            return

        within_budget = self._spatial_search.record_attempt()
        msg = IDLPromptStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        if within_budget:
            # Re-prompt so the next tick sees the answer (cascade continues).
            msg.header.frame_id = "spatial_memory"
            msg.text = result_text
        else:
            # Budget exhausted → hand off. Use the reasoner's own frame_id so
            # _on_prompt filters it (no further tick): the loop stops here.
            msg.header.frame_id = self.get_name()
            msg.text = (
                f"{result_text}\nactive_search: query budget exhausted after "
                f"{self._spatial_search.attempts} consecutive lookups — handing off to a human."
            )
            self.get_logger().warning(
                f"dispatch: {call.tool} search budget exhausted "
                f"({self._spatial_search.attempts} queries) — handing off",
            )
        metadata: dict[str, Any] = {"source": "spatial_memory", "tool": call.tool}
        if traceparent is not None:
            metadata["traceparent"] = traceparent
        msg.metadata_json = json.dumps(metadata, sort_keys=True)
        self._prompt_pub.publish(msg)
        if within_budget:
            self.get_logger().info(
                f"dispatch: {call.tool} → re-prompt ({len(result_text)} chars)",
            )

    def _dispatch_locate_in_view(
        self,
        call: LocateInViewTool,
        *,
        traceparent: str | None,
    ) -> None:
        """Ask a live VLM detector if an object is in view; re-prompt with the answer (ADR-0043).

        The complement to :meth:`_dispatch_spatial_query` (remembered objects): this
        calls the detector node's ``/openral/perception/locate_in_view`` service to
        look at the CURRENT frame now. The call is async (``call_async`` +
        done-callback) so the ~1-2 s VLM inference never blocks the reasoner's
        executor; the rendered answer is republished as a ``PromptStamped`` with
        frame_id ``"detector"`` (consumed by ``_on_prompt``, feeding the next tick —
        the prompt cascade). Read-only: no actuation, no ``FailureTrigger``.
        """
        try:
            from openral_msgs.srv import LocateInView
        except ImportError:
            self.get_logger().warning(
                "dispatch: locate_in_view — openral_msgs/srv/LocateInView not built; skipping",
            )
            return
        # ADR-0056 — route to the chosen on-demand locator's namespaced service;
        # empty ``detector`` falls back to the deployment default (or the legacy
        # single-detector service). One cached client per resolved service name.
        service = locate_in_view_service(call.detector, default=self._default_on_demand_detector)
        client = self._locate_in_view_clients.get(service)
        if client is None:
            client = self.create_client(LocateInView, service)
            self._locate_in_view_clients[service] = client
        if not client.service_is_ready() and not client.wait_for_service(
            timeout_sec=_LIFECYCLE_SERVER_PROBE_S,
        ):
            self.get_logger().warning(
                f"dispatch: locate_in_view query={call.query!r} camera={call.camera!r} "
                f"detector={call.detector!r} — {service} not on graph; skipping",
            )
            return
        req = LocateInView.Request()
        req.query = call.query
        req.camera = call.camera
        req.detector = call.detector
        future = client.call_async(req)
        future.add_done_callback(
            lambda fut: self._on_locate_in_view_response(call, fut, traceparent=traceparent),
        )
        self.get_logger().info(
            f"dispatch: locate_in_view query={call.query!r} camera={call.camera!r} "
            f"detector={call.detector!r} → {service}",
        )

    def _on_locate_in_view_response(
        self,
        call: LocateInViewTool,
        future: Any,
        *,
        traceparent: str | None,
    ) -> None:
        """Render a ``LocateInView`` response as a re-prompt (ADR-0043 prompt cascade)."""
        try:
            resp = future.result()
        except Exception as exc:  # best-effort; a failed lookup must not kill the tick
            self.get_logger().warning(f"dispatch: locate_in_view response failed: {exc}")
            return
        assert self._prompt_pub is not None
        cam = resp.camera or call.camera or "default"
        # A live locate counts as one spatial-search step (ADR-0039 §3): a miss
        # consumes budget so a repeated "not visible" terminates in handoff
        # instead of looping; a hit ends the search streak so the next find
        # starts fresh.
        frame_id = "detector"  # consumed by _on_prompt → feeds the next tick
        if resp.found:
            self._spatial_search.reset()
            self._locate_escalated.clear()
            text = (
                f"locate_in_view: {call.query!r} IS visible in camera {cam!r} right now. "
                f"detections={resp.metadata_json}"
            )
        elif self._spatial_search.record_attempt():
            text = f"locate_in_view: {call.query!r} is NOT visible in camera {cam!r} right now."
        else:
            # Budget exhausted → hand off. The reasoner's own frame_id makes
            # _on_prompt filter it (no further tick): the cascade stops here.
            frame_id = self.get_name()
            text = (
                f"locate_in_view: {call.query!r} is NOT visible in camera {cam!r} right now.\n"
                f"active_search: query budget exhausted after {self._spatial_search.attempts} "
                "consecutive lookups — handing off to a human."
            )
            self.get_logger().warning(
                f"dispatch: locate_in_view {call.query!r} budget exhausted "
                f"({self._spatial_search.attempts} lookups) — handing off",
            )
        msg = IDLPromptStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = frame_id
        msg.text = text
        metadata: dict[str, Any] = {"source": "detector", "tool": call.tool}
        if traceparent is not None:
            metadata["traceparent"] = traceparent
        msg.metadata_json = json.dumps(metadata, sort_keys=True)
        self._prompt_pub.publish(msg)
        self.get_logger().info(
            f"dispatch: locate_in_view → re-prompt found={resp.found} ({len(text)} chars)",
        )

    def _dispatch_query_scene(
        self,
        call: QuerySceneTool,
        *,
        traceparent: str | None,
    ) -> None:
        """Ask a scene VLM an open-ended question; re-prompt with the answer (ADR-0047).

        The complement to :meth:`_dispatch_locate_in_view` (object localization): this
        calls the perception node's ``/openral/perception/query_scene`` service to ask
        the scene VLM about the CURRENT frame's state ("has the robot grasped the
        mug?", "is the task complete?"). The call is async (``call_async`` +
        done-callback) so the multi-second VLM inference never blocks the reasoner's
        executor; the answer is republished as a ``PromptStamped`` with frame_id
        ``"scene_vlm"`` (consumed by ``_on_prompt``, feeding the next tick — the
        prompt cascade). Read-only: no actuation, no ``FailureTrigger``.
        """
        try:
            from openral_msgs.srv import QueryScene
        except ImportError:
            self.get_logger().warning(
                "dispatch: query_scene — openral_msgs/srv/QueryScene not built; skipping",
            )
            return
        if self._query_scene_client is None:
            self._query_scene_client = self.create_client(
                QueryScene, "/openral/perception/query_scene"
            )
        client = self._query_scene_client
        if not client.service_is_ready() and not client.wait_for_service(
            timeout_sec=_LIFECYCLE_SERVER_PROBE_S,
        ):
            self.get_logger().warning(
                f"dispatch: query_scene question={call.question!r} camera={call.camera!r} — "
                "/openral/perception/query_scene not on graph; skipping",
            )
            return
        req = QueryScene.Request()
        req.question = call.question
        req.camera = call.camera
        future = client.call_async(req)
        future.add_done_callback(
            lambda fut: self._on_query_scene_response(call, fut, traceparent=traceparent),
        )
        self.get_logger().info(
            f"dispatch: query_scene question={call.question!r} camera={call.camera!r}",
        )

    def _on_query_scene_response(
        self,
        call: QuerySceneTool,
        future: Any,
        *,
        traceparent: str | None,
    ) -> None:
        """Render a ``QueryScene`` response as a re-prompt (ADR-0047 prompt cascade)."""
        try:
            resp = future.result()
        except Exception as exc:  # best-effort; a failed query must not kill the tick
            self.get_logger().warning(f"dispatch: query_scene response failed: {exc}")
            return
        assert self._prompt_pub is not None
        cam = resp.camera or call.camera or "default"
        if resp.ok:
            text = f"query_scene[{call.question!r} | camera {cam!r}]: {resp.answer}"
        else:
            text = (
                f"query_scene[{call.question!r} | camera {cam!r}]: no answer "
                "(no frame available or the scene VLM errored)."
            )
        msg = IDLPromptStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "scene_vlm"  # consumed by _on_prompt → feeds the next tick
        msg.text = text
        metadata: dict[str, Any] = {"source": "scene_vlm", "tool": call.tool}
        if traceparent is not None:
            metadata["traceparent"] = traceparent
        msg.metadata_json = json.dumps(metadata, sort_keys=True)
        self._prompt_pub.publish(msg)
        self.get_logger().info(
            f"dispatch: query_scene → re-prompt ok={resp.ok} ({len(text)} chars)",
        )

    def _dispatch_query_task_progress(
        self,
        call: QueryTaskProgressTool,
        *,
        traceparent: str | None,
    ) -> None:
        """Ask the reward monitor for a windowed progress/success assessment (ADR-0057).

        Calls ``/openral/perception/query_task_progress`` (served by the
        reward_monitor_node, backed by the Robometer NF4 sidecar). Async
        (``call_async`` + done-callback) so the multi-hundred-ms reward inference
        never blocks the reasoner executor; the quantitative result is republished
        as a ``PromptStamped`` (frame_id ``"reward_monitor"``) feeding the next
        tick — the prompt cascade that drives the replanning ladder. Read-only:
        the reward signal is advisory, no actuation, no ``FailureTrigger``.
        """
        try:
            from openral_msgs.srv import QueryTaskProgress
        except ImportError:
            self.get_logger().warning(
                "dispatch: query_task_progress — openral_msgs/srv/QueryTaskProgress "
                "not built; skipping",
            )
            return
        if self._query_task_progress_client is None:
            self._query_task_progress_client = self.create_client(
                QueryTaskProgress, "/openral/perception/query_task_progress"
            )
        client = self._query_task_progress_client
        if not client.service_is_ready() and not client.wait_for_service(
            timeout_sec=_LIFECYCLE_SERVER_PROBE_S,
        ):
            self.get_logger().warning(
                f"dispatch: query_task_progress window_s={call.window_s} — "
                "/openral/perception/query_task_progress not on graph; skipping",
            )
            return
        req = QueryTaskProgress.Request()
        req.window_s = call.window_s
        req.task = call.task
        future = client.call_async(req)
        future.add_done_callback(
            lambda fut: self._on_query_task_progress_response(call, fut, traceparent=traceparent),
        )
        self.get_logger().info(
            f"dispatch: query_task_progress window_s={call.window_s} task={call.task!r}",
        )

    def _on_query_task_progress_response(
        self,
        call: QueryTaskProgressTool,
        future: Any,
        *,
        traceparent: str | None,
    ) -> None:
        """Render a ``QueryTaskProgress`` response as a re-prompt (ADR-0057 cascade).

        Surfaces the quantitative assessment in plain language so the LLM can act
        on it — continue, escalate to ``query_scene``, advance, or replan when the
        task has ``stalled`` or success is low.
        """
        try:
            resp = future.result()
        except Exception as exc:  # best-effort; a failed query must not kill the tick
            self.get_logger().warning(f"dispatch: query_task_progress response failed: {exc}")
            return
        assert self._prompt_pub is not None
        if not resp.ok:
            reason = "no fresh camera frames" if resp.stale else "the reward monitor errored"
            text = f"query_task_progress[window {call.window_s:.0f}s]: no assessment ({reason})."
        else:
            verdict = (
                "SUCCEEDED"
                if resp.succeeded
                else ("STALLED — consider replanning" if resp.stalled else "in progress")
            )
            text = (
                f"query_task_progress[window {call.window_s:.0f}s, {resp.frames_seen} frames]: "
                f"progress={resp.progress_now:.2f} (trend {resp.progress_trend:+.3f}/frame), "
                f"success={resp.success_now:.2f} (trend {resp.success_trend:+.3f}/frame) — "
                f"{verdict}."
            )
        msg = IDLPromptStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "reward_monitor"  # consumed by _on_prompt → next tick
        msg.text = text
        metadata: dict[str, Any] = {"source": "reward_monitor", "tool": call.tool}
        if traceparent is not None:
            metadata["traceparent"] = traceparent
        msg.metadata_json = json.dumps(metadata, sort_keys=True)
        self._prompt_pub.publish(msg)
        self.get_logger().info(
            f"dispatch: query_task_progress → re-prompt ok={resp.ok} ({len(text)} chars)",
        )

    # ── ADR-0073 §2 — automatic reward-gated task verification ──────────────

    def _maybe_verify_active_mission_task(
        self, call: ExecuteRskillTool, *, traceparent: str | None
    ) -> None:
        """Verify the active mission task against the reward signal after a skill returns.

        Only acts when a reward monitor is available (``task_progress_available``):
        a VLA never self-terminates, so its runner "success" cannot confirm the
        task. Without a reward monitor the task stays active and the LLM/playbook
        drives — never an auto-complete on deadline alone (no fake success). Issues
        a windowed ``query_task_progress`` for the active task; the gate runs in
        :meth:`_on_mission_verify_response`.
        """
        mission = self._renderer.mission
        if mission is None:
            return
        active = mission.active()
        if active is None:
            return
        if not self._task_progress_available:
            return
        # Only verify the skill that was dispatched for this task.
        if active.last_rskill_id is not None and active.last_rskill_id != call.rskill_id:
            return
        mission.mark_verifying()
        try:
            from openral_msgs.srv import QueryTaskProgress
        except ImportError:
            return
        if self._query_task_progress_client is None:
            self._query_task_progress_client = self.create_client(
                QueryTaskProgress, "/openral/perception/query_task_progress"
            )
        client = self._query_task_progress_client
        if not client.service_is_ready() and not client.wait_for_service(
            timeout_sec=_LIFECYCLE_SERVER_PROBE_S,
        ):
            self.get_logger().warning(
                "mission verify: query_task_progress not on graph; active task stays active",
            )
            return
        req = QueryTaskProgress.Request()
        req.window_s = _MISSION_VERIFY_WINDOW_S
        req.task = active.text
        future = client.call_async(req)
        future.add_done_callback(
            lambda fut: self._on_mission_verify_response(active.text, fut, traceparent=traceparent),
        )
        self.get_logger().info(
            f"mission verify: querying reward for active task {active.text[:60]!r} "
            f"(attempt {active.attempts})",
        )

    def _on_mission_verify_response(  # noqa: PLR0911, PLR0912  # reason: one return per verdict branch — a flat dispatch table is clearer than collapsing the branches
        self, task_text: str, future: Any, *, traceparent: str | None
    ) -> None:
        """Apply the reward gate (ADR-0073 §2): complete / abandon / retry.

        Runs ``evaluate_task_verdict`` on the reward response and the active task's
        attempt count, then advances the deterministic queue. ``complete`` →
        advance to the next task; ``abandon`` (ladder exhausted) → mark abandoned,
        advance, and on mission end emit an honest handoff; ``retry`` → keep the
        task active. A forced tick wakes the reasoner to act on the new state.
        """
        try:
            resp = future.result()
        except Exception as exc:  # best-effort; a failed verify must not kill the loop
            self.get_logger().warning(f"mission verify: query failed: {exc}")
            return
        mission = self._renderer.mission
        if mission is None:
            return
        active = mission.active()
        if active is None or active.text != task_text:
            return  # the mission advanced or changed under us; stale verdict
        # ADR-0074 §1/§5 — band edges from the active reward model's calibration
        # (or the system fallback when none is wired).
        success_threshold, check_floor = self._band_edges()
        action, verdict = evaluate_task_verdict(
            ok=bool(resp.ok),
            success_now=float(resp.success_now),
            success_threshold=success_threshold,
            check_floor=check_floor,
            attempts=active.attempts,
        )
        if action == "vlm_check":
            # ADR-0074 §5 — ambiguous reward band: ask the VLM whether the task is
            # visually complete. True → advance as if "complete"; False/None → degrade
            # to the ladder (same code path as action == "retry"). Never false-complete:
            # None (no frame / no client) is treated as "not done".
            verdict_vlm = self._adjudicate_completion(active.text)
            if verdict_vlm is True:
                self.get_logger().info(
                    f"mission verify: VLM confirmed complete (success={resp.success_now:.2f})"
                )
                self._complete_active_and_advance(active, verdict, traceparent=traceparent)
                return
            if verdict_vlm is False:
                self.get_logger().info(
                    f"mission verify: VLM says not complete ({verdict}) — falling to ladder"
                )
            else:
                self.get_logger().info(
                    f"mission verify: could not adjudicate ({verdict}) — falling to ladder"
                )
            # Degrade to the attempts ladder. A reward stuck in the ambiguous band
            # that the VLM cannot confirm complete must still be *bounded*: re-run
            # the verdict with ok=False to skip tiers 1/2 and apply the attempts
            # ladder (abandon once attempts >= max), then fall through to the
            # retry / abandon handlers below. Without this an ambiguous-band task
            # retries forever — never abandons, never hands off (CLAUDE.md §3
            # bounded ladder). ``attempts`` is monotonic thanks to the subdivide
            # guard, so this terminates.
            action, verdict = evaluate_task_verdict(
                ok=False,
                success_now=float(resp.success_now),
                success_threshold=success_threshold,
                check_floor=check_floor,
                attempts=active.attempts,
            )
            # fall through — no return; the action == "retry" / "abandon" blocks below apply
        if action == "retry":
            self.get_logger().info(f"mission verify: {verdict} — retrying active task")
            # ADR-0074 — surface the reward-plateau FAILURE to the LLM. The reward
            # verify path otherwise records nothing on a retry, so the LLM only sees
            # "task still active" and blindly re-issues the identical instruction (a
            # direct replanning probe confirmed: no signal → repeat; the timeout hint
            # → subdivide; a reward-plateau hint → replan with a different approach).
            # Append a failed execution outcome with a reward-plateau-specific
            # Reflexion hint so the next (forced) tick replans tactic instead of
            # repeating. last_rskill_id is set by MissionState.record_attempt.
            self._renderer.append_execution(
                ExecutionEventRecord(
                    rskill_id=active.last_rskill_id or "",
                    outcome="failed",
                    summary=(
                        f"reward says NOT done: success={float(resp.success_now):.2f} below the "
                        f"success bar (attempt {active.attempts}); the policy executed without a "
                        "fault but did not accomplish the task"
                    ),
                    reflection=reflect_on_reward_plateau(float(resp.success_now)),
                    stamp_ns=self.get_clock().now().nanoseconds,
                )
            )
            self._on_tick(force=True, tier="C")
            return
        if action == "complete":
            self._complete_active_and_advance(active, verdict, traceparent=traceparent)
            return
        if action == "abandon" and _should_offer_subdivision(
            active, self._subdivide_offered, DEFAULT_MAX_SUBDIVIDE_DEPTH
        ):
            # #123 — before abandoning a blocked task, give the LLM ONE chance to
            # decompose it into finer subtasks (depth-bounded; one offer per task
            # id via `_subdivide_offered` so a task that declines to decompose
            # still terminates in human-handoff rather than looping). Re-arm the
            # task to `active` so the normal dispatch / decompose_mission cycle
            # resumes, then nudge the reasoner with an explicit invite tick.
            self._subdivide_offered.add(active.task_id)
            mission.rearm_active()
            if self._core is not None:
                self._core.reset_kind_streak()
            self.get_logger().info(
                f"mission: task {active.task_id} blocked ({verdict}); offering subdivision "
                f"(depth {active.depth} < {DEFAULT_MAX_SUBDIVIDE_DEPTH})",
            )
            self._emit_subdivision_invite(active, verdict, traceparent=traceparent)
            self._on_tick(force=True, tier="C")
            return
        # action == "abandon" (no subdivision offer): advance with done=False.
        nxt = self._renderer.advance_mission(done=False, verdict=verdict)
        if nxt is not None:
            # A new active task is a fresh goal — clear the per-kind tick streak so
            # the next task isn't suppressed by `retry_cap` for re-using the same
            # tool kind (e.g. execute_rskill) the just-finished task ended on.
            # Mirrors the reset on a new operator prompt (see _on_prompt).
            if self._core is not None:
                self._core.reset_kind_streak()
            self.get_logger().info(
                f"mission: task {active.task_id} abandoned ✗ ({verdict}); "
                f"advancing → {nxt.task_id}={nxt.text[:60]!r}",
            )
        else:
            self.get_logger().info(
                f"mission: task {active.task_id} abandoned ✗ ({verdict}); MISSION COMPLETE",
            )
            self._emit_mission_complete(mission, traceparent=traceparent)
        self._on_tick(force=True, tier="C")

    def _emit_mission_complete(self, mission: MissionState, *, traceparent: str | None) -> None:
        """Emit an honest operator-facing mission summary (self-prompt, ADR-0073 §2).

        Frame_id ``openral_reasoner`` so it reaches operator surfaces but the
        reasoner's own subscriber filters it (no feedback loop). A new operator
        goal supersedes the finished mission via :meth:`_on_prompt`.
        """
        if self._prompt_pub is None:
            return
        done = sum(1 for t in mission.tasks if t.status == "done")
        abandoned = sum(1 for t in mission.tasks if t.status == "abandoned")
        ledger = "; ".join(
            f"{t.task_id} {t.status}" + (f" [{t.last_verdict}]" if t.last_verdict else "")
            for t in mission.tasks
        )
        text = (
            f"Mission finished: {done} completed, {abandoned} abandoned. {ledger}. "
            "Awaiting the next goal."
        )
        msg = IDLPromptStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "openral_reasoner"
        metadata: dict[str, Any] = {"source": "openral_reasoner", "mission": "complete"}
        if traceparent is not None:
            metadata["traceparent"] = traceparent
        msg.text = text
        msg.metadata_json = json.dumps(metadata, sort_keys=True)
        self._prompt_pub.publish(msg)
        self.get_logger().info(f"mission: {text}")

    def _dispatch_decompose_mission(self, call: DecomposeMissionTool) -> None:
        """Apply an LLM mission decomposition to the typed task queue (#123).

        Two modes by ``target_task_id`` (ADR-0073 amendment / ADR-0072
        ``decompose-mission``):

        * **subdivide** (id set) — flat-splice the named *active* blocked task in
          place with finer children via :meth:`MissionState.subdivide_active`
          (depth-bounded). Only the active task may be subdivided; a stale /
          non-active id is logged and ignored.
        * **populate** (id empty) — replace the whole queue with a better
          decomposition of the operator goal, but only before any task has been
          attempted (:meth:`MissionState.has_started`) so a refinement never
          discards in-flight progress.

        Edits the S2 ledger only — no actuation. A forced Tier-C tick wakes the
        reasoner to act on the new active task.
        """
        mission = self._renderer.mission
        if call.target_task_id:
            if mission is None:
                self.get_logger().warning(
                    f"decompose_mission: target {call.target_task_id!r} but no active mission",
                )
                return
            active = mission.active()
            if active is None or active.task_id != call.target_task_id:
                self.get_logger().warning(
                    f"decompose_mission: target {call.target_task_id!r} is not the active task "
                    f"(active={active.task_id if active else None!r}) — ignored",
                )
                return
            child = mission.subdivide_active(call.rendered_subtasks())
            if child is None:
                # Depth bound reached (or empty) — fall through to the existing
                # attempt-cap → abandon → human-handoff ladder; do not loop.
                self.get_logger().info(
                    f"decompose_mission: refused to subdivide {call.target_task_id!r} "
                    f"(depth {active.depth} ≥ {DEFAULT_MAX_SUBDIVIDE_DEPTH}) — will hand off",
                )
                return
            self._renderer.set_mission(mission)  # bump seq so the new active task wakes a tick
            self.get_logger().info(
                f"decompose_mission: subdivided {call.target_task_id!r} into "
                f"{len(call.subtasks)} subtask(s) → active {child.task_id}={child.text[:60]!r}",
            )
        else:
            if mission is not None and mission.has_started():
                self.get_logger().warning(
                    "decompose_mission: populate ignored — the mission has already started "
                    "(use target_task_id to subdivide the active task instead)",
                )
                return
            new_mission = MissionState(call.rendered_subtasks())
            if new_mission.is_empty():
                return
            self._renderer.set_mission(new_mission)
            self._subdivide_offered.clear()
            self.get_logger().info(
                f"decompose_mission: populated mission with {len(new_mission)} task(s) — "
                f"active={new_mission.active().text[:60]!r}",
            )
        if self._core is not None:
            self._core.reset_kind_streak()
        self._on_tick(force=True, tier="C")

    def _emit_subdivision_invite(
        self,
        task: TaskState,
        verdict: str,
        *,
        traceparent: str | None,
    ) -> None:
        """Self-prompt inviting the LLM to subdivide a blocked task (#123).

        frame_id ``mission`` so the reasoner *consumes* it on the next tick (it is
        a cascade source, unlike ``openral_reasoner`` operator summaries) without
        rebuilding the deterministic mission queue.
        """
        if self._prompt_pub is None:
            return
        text = (
            f"Task {task.task_id} ({task.text!r}) is blocked: {verdict}. It is too coarse "
            "for one skill — call decompose_mission with target_task_id="
            f"{task.task_id!r} and an ordered list of finer subtasks to break it down and "
            "continue, or it will be handed off to the operator."
        )
        msg = IDLPromptStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "mission"
        metadata: dict[str, Any] = {"source": "mission", "task_id": task.task_id}
        if traceparent is not None:
            metadata["traceparent"] = traceparent
        msg.text = text
        msg.metadata_json = json.dumps(metadata, sort_keys=True)
        self._prompt_pub.publish(msg)

    def _emit_enumeration_invite(
        self,
        task: TaskState,
        *,
        traceparent: str | None,
    ) -> None:
        """Self-prompt forcing a collective task to be enumerated + decomposed.

        Emitted by the execute gate (:meth:`_dispatch_execute_rskill`) when the
        active task targets a collective/quantified set ("put ALL the objects in
        the basket"). A skill acts on one specific object, so the LLM must look at
        the live ``scene_objects`` list (already in its context every tick) and
        split the task into one concrete subtask per object before any actuation.
        frame_id ``mission`` so the reasoner consumes it next tick (cascade source)
        without rebuilding the deterministic queue — same channel as
        :meth:`_emit_subdivision_invite`.
        """
        if self._prompt_pub is None:
            return
        text = (
            f"REFUSED to run a skill on task {task.task_id} ({task.text!r}): it targets a "
            "COLLECTIVE/quantified set ('all', 'every', 'the objects', …), not a single "
            "object. A skill acts on exactly ONE specific object. Look at the "
            "`in_view` line in your context (every object the detector sees, with a "
            "stable id and camera-space pixel centre; `scene_objects` adds 3D positions "
            "when depth is available) — and call decompose_mission(target_task_id="
            f"{task.task_id!r}, subtasks=[…]) to split this into one grounded subtask per "
            "specific object. Each subtask is {object_ref: <ONE specific object>, text: "
            "<instruction naming that object>}, e.g. {object_ref: 'milk', text: 'pick up "
            "the milk and put it in the basket'}, {object_ref: 'ketchup', text: 'pick up "
            "the ketchup and put it in the basket'}. object_ref must be ONE concrete "
            "object (never 'all'/'objects'/'the first batch'). Do NOT call execute_rskill "
            "until the active task names a single specific object."
        )
        msg = IDLPromptStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "mission"
        metadata: dict[str, Any] = {"source": "mission", "task_id": task.task_id}
        if traceparent is not None:
            metadata["traceparent"] = traceparent
        msg.text = text
        msg.metadata_json = json.dumps(metadata, sort_keys=True)
        self._prompt_pub.publish(msg)

    def _memory_now(self) -> str:
        """An ISO-8601 timestamp from the ROS clock (sim-time-aware) for memory entries."""
        secs = self.get_clock().now().nanoseconds / 1e9
        return datetime.datetime.fromtimestamp(secs, tz=datetime.UTC).isoformat(timespec="seconds")

    def _persist_memory(self) -> None:
        """Write the live store back to ``MEMORY.md`` (advisory — a failure logs, never raises)."""
        if self._memory_store is None or self._memory_md_path is None:
            return
        try:
            self._memory_md_path.write_text(self._memory_store.to_markdown(), encoding="utf-8")
        except OSError as exc:
            self.get_logger().warning(f"memory: failed to persist {self._memory_md_path!r}: {exc}")

    def _archive_memory_entry(self, entry: MemoryEntry) -> None:
        """Append one superseded/deleted entry to the JSONL recall log (best-effort)."""
        self._memory_archive.append(entry)
        if self._memory_md_path is None:
            return
        archive_path = self._memory_archive_path(self._memory_md_path)
        record = {
            "section": entry.section,
            "content": entry.content,
            "importance": entry.importance,
            "timestamp": entry.timestamp,
            "status": entry.status,
        }
        try:
            with archive_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, sort_keys=True) + "\n")
        except OSError as exc:
            self.get_logger().warning(f"memory: failed to append archive {archive_path!r}: {exc}")

    def _reprompt_memory(self, text: str, *, traceparent: str | None) -> None:
        """Re-prompt with a memory result so the next tick sees it (frame_id ``memory``)."""
        assert self._prompt_pub is not None
        msg = IDLPromptStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "memory"  # consumed by _on_prompt → next tick
        msg.text = text
        metadata: dict[str, Any] = {"source": "memory"}
        if traceparent is not None:
            metadata["traceparent"] = traceparent
        msg.metadata_json = json.dumps(metadata, sort_keys=True)
        self._prompt_pub.publish(msg)

    def _dispatch_memory_write(
        self,
        call: MemoryWriteTool,
        *,
        traceparent: str | None,
    ) -> None:
        """Apply one explicit MEMORY.md edit, persist it, and confirm (ADR-0072 §3 / Phase 4c).

        The reasoner's first **write-capable** tool: an ``add``/``update``/``supersede``/
        ``delete`` op over a typed :class:`~openral_core.MemorySection` — never a
        free-form rewrite (the writer half of the Statler reader/writer split). The
        edit is applied to the live store, any entry that *left* the file (an
        ``update``-replaced or ``delete``-removed prior) is appended to the archival
        recall log, the file is persisted, the ``## MEMORY`` context block is
        re-rendered (so the next tick reads the new fact), and a short confirmation is
        re-prompted. Advisory only — a wrong memory yields a bad plan the C++ safety
        kernel still vetoes (CLAUDE.md §1.1).
        """
        if self._memory_store is None:
            self.get_logger().warning(
                "dispatch: memory_write received but no MEMORY.md backend is wired",
            )
            return
        archived = self._memory_store.apply(
            op=call.op,
            section=call.section,
            content=call.content,
            importance=call.importance,
            target=call.target,
            now=self._memory_now(),
        )
        if archived is not None:
            self._archive_memory_entry(archived)
        # ADR-0072 Phase 5 — consolidate: merge any exact-duplicate facts the write
        # may have introduced, paging the removed copies to the archive (Mem0).
        for dup in self._memory_store.consolidate():
            self._archive_memory_entry(dup)
        self._persist_memory()
        self._renderer.set_memory_block(self._render_memory_block())
        detail = f"{call.op} [{call.section}]"
        body = call.target if call.op == "delete" else call.content
        self.get_logger().info(f"dispatch: memory_write {detail} {body!r} → persisted")
        self._reprompt_memory(
            f"memory updated: {detail} — {body!r}. Continue the task.",
            traceparent=traceparent,
        )

    def _dispatch_memory_search(
        self,
        call: MemorySearchTool,
        *,
        traceparent: str | None,
    ) -> None:
        """Recall archived memory entries by keyword and re-prompt (ADR-0072 §3 / Phase 4c).

        Read-only (the reader half): current memory is already in the ``## MEMORY``
        context block every tick, so this searches only the **archive** — superseded /
        deleted entries that left the live file — ranked by importance then recency
        (MemGPT recall). No actuation, no file write.
        """
        hits = MemoryStore.search(
            self._memory_archive,
            query=call.query,
            section=call.section,
            limit=call.limit,
        )
        if hits:
            lines = "\n".join(h.render_line() for h in hits)
            text = f"memory_search {call.query!r} — {len(hits)} archived match(es):\n{lines}"
        else:
            text = f"memory_search {call.query!r} — no archived memory matches."
        self.get_logger().info(f"dispatch: memory_search {call.query!r} → {len(hits)} hit(s)")
        self._reprompt_memory(text, traceparent=traceparent)

    def _dispatch_execute_rskill(
        self,
        call: ExecuteRskillTool,
        *,
        traceparent: str | None,
    ) -> None:
        """Send an :class:`ExecuteRskill.Goal` to ``/openral/execute_rskill``.

        Feedback streams via :meth:`_on_execute_rskill_feedback` (warning
        channel — visible to the operator). Goal-response and result
        futures attach :meth:`_on_execute_rskill_goal_response` and
        :meth:`_on_execute_rskill_result`; both paths emit a
        :class:`FailureTrigger` on ``/openral/failure/rskill`` on
        rejection/abort. A one-shot deadline timer fires
        :meth:`_on_execute_rskill_deadline` when ``call.deadline_s`` is
        positive, producing a ``KIND_TIMEOUT`` event.
        """
        # Grounding gate: a skill acts on ONE specific object, so refuse to
        # actuate while the active mission task targets a collective/quantified
        # set ("put ALL the objects in the basket"). Self-prompt the LLM to
        # enumerate (scene_objects is in its context) and decompose into one
        # subtask per object; once it does, the active task names a single
        # object and this gate passes. No attempt is recorded — a refused
        # actuation is not a try at the task.
        mission = self._renderer.mission
        active = mission.active() if mission is not None else None
        if active is not None and is_collective_target(active.text):
            self.get_logger().info(
                f"execute gate: refusing execute_rskill on collective task "
                f"{active.task_id} ({active.text[:60]!r}) — inviting per-object decomposition",
            )
            self._emit_enumeration_invite(active, traceparent=traceparent)
            self._on_tick(force=True, tier="C")
            return
        assert self._execute_rskill_client is not None
        # Non-blocking single probe: ActionClient.wait_for_server
        # spins the executor; passing a short timeout keeps the tick
        # bounded if the F1 server is not yet on the graph.
        if (
            not self._execute_rskill_client.server_is_ready()
            and not self._execute_rskill_client.wait_for_server(
                timeout_sec=_EXECUTE_SKILL_SERVER_PROBE_S,
            )
        ):
            self.get_logger().warning(
                "dispatch: execute_rskill server /openral/execute_rskill not on graph; "
                f"emitting KIND_CONTROLLER FailureTrigger for rskill_id={call.rskill_id!r}",
            )
            self._publish_skill_failure(
                kind=_KIND_CONTROLLER,
                rskill_id=call.rskill_id,
                evidence=ControllerEvidence(
                    controller_name=call.rskill_id,
                    state="unavailable",
                    detail="action server /openral/execute_rskill not on graph",
                ),
                traceparent=traceparent,
            )
            return

        # ADR-0073 §2 — count this dispatch as an attempt at the active mission
        # task so the reward gate can bound retries (abandon + hand off after the
        # cap). execute_rskill is the actuation tool; locate/query are separate
        # tools, so a dispatch here is a manipulation attempt at the active task.
        mission = self._renderer.mission
        if mission is not None and mission.active() is not None:
            mission.record_attempt(rskill_id=call.rskill_id, trace_id=traceparent)

        # ADR-0050 — free GPU lifecycle peers (the object detector) before the
        # policy loads, then reactivate when the skill finishes. Sequenced so
        # the peer's VRAM is released before the goal reaches the runner; an
        # 8 GB card OOMs if the ~1.3 GB detector co-resides with the VLA.
        if self._vram_lifecycle_peers:
            self._free_vram_peers_then_send(call, list(self._vram_lifecycle_peers), traceparent)
        else:
            self._send_execute_rskill_goal(call, traceparent)

    def _send_execute_rskill_goal(
        self,
        call: ExecuteRskillTool,
        traceparent: str | None,
    ) -> None:
        """Build and send the ``ExecuteRskill.Goal`` (the VLA dispatch itself)."""
        assert self._execute_rskill_client is not None
        goal = IDLExecuteRskill.Goal()
        goal.rskill_id = call.rskill_id
        goal.revision = ""
        # An empty LLM prompt leaves the VLA with no task-conditioning; fall back
        # to the active mission task's text (the actual instruction).
        mission = self._renderer.mission
        active = mission.active() if mission is not None else None
        goal.prompt = _resolve_execute_prompt(
            call.prompt, active.text if active is not None else None
        )
        # The reasoner does not yet construct a SkillPrompt payload —
        # F4 stays on the text path; the structured-prompt route is
        # wired in a later ADR-0018 follow-up.
        goal.prompt_metadata_json = ""
        # ADR-0026 — forward the LLM's per-skill structured params, if
        # any. Wrapped-ROS adapters merge ``goal_params_json`` over
        # their manifest's ``default_goal_json`` at configure-time.
        goal.goal_params_json = call.goal_params_json
        # ADR-0074 §2/§3 — the goal's ``deadline_s`` slot now carries the resolved
        # patience ceiling (LLM override > reward-model default > legacy deadline_s);
        # it is the runner's backstop, not the usual stop (the reward-watcher is).
        patience_s = self._effective_patience_s(call)
        goal.deadline_s = patience_s
        sent_at = time.monotonic()
        send_future = self._execute_rskill_client.send_goal_async(
            goal,
            feedback_callback=lambda fb: self._on_execute_rskill_feedback(call.rskill_id, fb),
        )
        send_future.add_done_callback(
            lambda fut: self._on_execute_rskill_goal_response(call, sent_at, fut, traceparent),
        )
        self.get_logger().info(
            f"dispatch: execute_rskill rskill_id={call.rskill_id!r} prompt={goal.prompt!r} "
            f"patience_s={patience_s:.0f} (backstop; reward-watcher is the usual stop)",
        )

    def _free_vram_peers_then_send(
        self,
        call: ExecuteRskillTool,
        peers: list[str],
        traceparent: str | None,
    ) -> None:
        """Deactivate GPU peers, then send the goal once they have all released.

        ADR-0050. Each peer's ``change_state`` is async; the goal is sent only
        after every in-flight deactivation has returned, so the freed VRAM is
        available before the runner loads the policy. Peers whose service isn't
        on the graph are skipped (best-effort — the dispatch still proceeds).
        The deactivated subset is recorded for reactivation on the skill result.
        """
        self._deactivated_vram_peers = []
        futures: list[tuple[str, Any]] = []
        for peer in peers:
            future = self._change_state_async(peer, "deactivate")
            if future is None:
                self.get_logger().warning(
                    f"vram: peer {peer!r} change_state not on graph; "
                    f"dispatching execute_rskill {call.rskill_id!r} without freeing it",
                )
                continue
            self.get_logger().info(
                f"vram: deactivating GPU peer {peer!r} before execute_rskill {call.rskill_id!r}",
            )
            futures.append((peer, future))
        if not futures:
            self._send_execute_rskill_goal(call, traceparent)
            return
        remaining = {"n": len(futures)}

        def _after_one(peer: str, fut: Any) -> None:
            try:
                ok = bool(fut.result().success)
            except Exception as exc:  # reason: a failed change_state must not strand dispatch
                ok = False
                self.get_logger().warning(f"vram: peer {peer!r} deactivate errored: {exc}")
            if ok:
                self._deactivated_vram_peers.append(peer)
            else:
                self.get_logger().warning(
                    f"vram: peer {peer!r} did not deactivate cleanly; proceeding",
                )
            remaining["n"] -= 1
            if remaining["n"] == 0:
                self._send_execute_rskill_goal(call, traceparent)

        for peer, future in futures:
            future.add_done_callback(lambda fut, p=peer: _after_one(p, fut))

    def _reactivate_vram_peers(self) -> None:
        """Reactivate the GPU peers deactivated for a now-finished skill (ADR-0050).

        Idempotent: clears the tracked set, so repeated terminal callbacks
        reactivate at most once.
        """
        peers = self._deactivated_vram_peers
        self._deactivated_vram_peers = []
        for peer in peers:
            future = self._change_state_async(peer, "activate")
            if future is None:
                self.get_logger().warning(
                    f"vram: peer {peer!r} change_state gone; cannot reactivate",
                )
                continue
            self.get_logger().info(f"vram: reactivating GPU peer {peer!r} after execute_rskill")
            future.add_done_callback(lambda fut, p=peer: self._on_reactivate_result(p, fut))

    def _on_reactivate_result(self, peer: str, future: Any) -> None:
        """Log the reactivation ``change_state`` outcome (best-effort, ADR-0050)."""
        try:
            ok = bool(future.result().success)
        except Exception as exc:  # reason: surface rclpy errors
            self.get_logger().warning(f"vram: peer {peer!r} reactivate errored: {exc}")
            return
        if not ok:
            self.get_logger().warning(f"vram: peer {peer!r} reactivate rejected by the node")

    def _change_state_async(self, node: str, transition: str) -> Any | None:
        """Call ``<node>/change_state`` for ``transition``; return the future.

        Returns ``None`` if the ``change_state`` service is not on the graph.
        Lifecycle clients are cached per peer node — a typical reasoner flips a
        handful of peers (HAL, perception, dispatcher), not a long tail, so a
        dict is cheaper than rebuilding the client on every call.
        """
        client = self._lifecycle_clients.get(node)
        if client is None:
            assert IDLChangeState is not None
            client = self.create_client(IDLChangeState, f"{node}/change_state")
            self._lifecycle_clients[node] = client
        if not client.service_is_ready() and not client.wait_for_service(
            timeout_sec=_LIFECYCLE_SERVER_PROBE_S,
        ):
            return None
        assert IDLTransition is not None
        transition_id = {
            "configure": IDLTransition.TRANSITION_CONFIGURE,
            "activate": IDLTransition.TRANSITION_ACTIVATE,
            "deactivate": IDLTransition.TRANSITION_DEACTIVATE,
            "cleanup": IDLTransition.TRANSITION_CLEANUP,
        }[transition]
        req = IDLChangeState.Request()
        req.transition.id = transition_id
        req.transition.label = transition
        return client.call_async(req)

    def _dispatch_lifecycle_transition(self, call: LifecycleTransitionTool) -> None:
        """Call ``<call.node>/change_state`` with the matching ``Transition.TRANSITION_*``."""
        future = self._change_state_async(call.node, call.transition)
        if future is None:
            self.get_logger().warning(
                f"dispatch: lifecycle_transition node={call.node!r} "
                f"transition={call.transition!r} — service "
                f"{call.node}/change_state not on graph; skipping",
            )
            return
        future.add_done_callback(
            lambda fut: self._on_lifecycle_response(call, fut),
        )
        self.get_logger().info(
            f"dispatch: lifecycle_transition node={call.node!r} transition={call.transition!r}",
        )

    # ── ExecuteSkill action callbacks ───────────────────────────────────────

    def _on_execute_rskill_feedback(self, rskill_id: str, feedback_msg: Any) -> None:
        """Forward action feedback to the operator log at warning level.

        Feedback is rare (chunk_index advances) so a warning-channel
        log is fine — the operator wants visibility, and structlog/
        OTel will route this to the dashboard.
        """
        fb = feedback_msg.feedback
        self.get_logger().warning(
            f"execute_rskill feedback rskill_id={rskill_id!r} state={fb.state!r} "
            f"progress={fb.progress:.2f} chunk={fb.chunk_index}/{fb.chunks_total}",
        )

    def _on_execute_rskill_goal_response(
        self,
        call: ExecuteRskillTool,
        sent_at: float,
        future: Any,
        traceparent: str | None,
    ) -> None:
        """Goal-response done callback.

        On rejection emits a ``KIND_CONTROLLER`` FailureTrigger; on
        acceptance attaches a result-future callback and arms the
        deadline timer (if ``call.deadline_s > 0``).
        """
        try:
            goal_handle = future.result()
        except Exception as exc:  # reason: surface any rclpy error path
            self.get_logger().error(
                f"execute_rskill send_goal failed rskill_id={call.rskill_id!r}: "
                f"{type(exc).__name__}: {exc}",
            )
            self._publish_skill_failure(
                kind=_KIND_CONTROLLER,
                rskill_id=call.rskill_id,
                evidence=ControllerEvidence(
                    controller_name=call.rskill_id,
                    state="error",
                    detail=f"{type(exc).__name__}: {exc}",
                ),
                traceparent=traceparent,
            )
            # ADR-0050 — the goal never reached the runner; restore the GPU
            # peers we froze for it so perception resumes.
            self._reactivate_vram_peers()
            return
        if not goal_handle.accepted:
            self.get_logger().warning(
                f"execute_rskill goal rejected rskill_id={call.rskill_id!r}",
            )
            self._publish_skill_failure(
                kind=_KIND_CONTROLLER,
                rskill_id=call.rskill_id,
                evidence=ControllerEvidence(
                    controller_name=call.rskill_id,
                    state="rejected",
                    detail="action server rejected goal",
                ),
                traceparent=traceparent,
            )
            # ADR-0050 — goal rejected (skill won't run); restore the GPU peers.
            self._reactivate_vram_peers()
            return
        goal_id = bytes(goal_handle.goal_id.uuid)
        # ADR-0074 §2 — remember the in-flight goal so a reward-watcher wake can
        # cancel it. Cleared on the terminal result. A new dispatch overwrites a
        # stale handle (the runner serves one goal at a time).
        self._active_rskill_goal = (goal_handle, call, traceparent)
        self._rskill_cancel_reason = None
        # ADR-0074 §2/§3 — arm the reasoner-side backstop at the resolved patience
        # (matches the goal's deadline_s sent to the runner). 0 → the runner owns
        # the ceiling (manifest latency budget); no reasoner-side timer.
        patience_s = self._effective_patience_s(call)
        if patience_s > 0:
            self._pending_skill_deadlines[goal_id] = self.create_timer(
                patience_s,
                lambda: self._on_execute_rskill_deadline(
                    call=call,
                    sent_at=sent_at,
                    goal_handle=goal_handle,
                    traceparent=traceparent,
                ),
            )
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(
            lambda fut: self._on_execute_rskill_result(call, goal_id, fut, traceparent),
        )

    def _on_execute_rskill_result(
        self,
        call: ExecuteRskillTool,
        goal_id: bytes,
        future: Any,
        traceparent: str | None,
    ) -> None:
        """Result done callback. Cancels deadline timer; emits on abort."""
        # ADR-0050 — the skill is terminal (success/abort/cancel/error), so the
        # policy's VRAM is released; restore the GPU peers (detector) we froze
        # for it. Runs before any early return below so it always fires.
        self._reactivate_vram_peers()
        # ADR-0074 §2 — the goal is terminal: drop the in-flight handle and read
        # (then clear) the cancel reason so a reward-driven cancel verifies below
        # while an operator/estop cancel stays a no-op.
        self._active_rskill_goal = None
        cancel_reason = self._rskill_cancel_reason
        self._rskill_cancel_reason = None
        timer = self._pending_skill_deadlines.pop(goal_id, None)
        if timer is not None:
            timer.cancel()
        try:
            wrapped = future.result()
        except Exception as exc:  # reason: surface any rclpy error path
            self.get_logger().error(
                f"execute_rskill result fetch failed rskill_id={call.rskill_id!r}: "
                f"{type(exc).__name__}: {exc}",
            )
            self._publish_skill_failure(
                kind=_KIND_CONTROLLER,
                rskill_id=call.rskill_id,
                evidence=ControllerEvidence(
                    controller_name=call.rskill_id,
                    state="error",
                    detail=f"{type(exc).__name__}: {exc}",
                ),
                traceparent=traceparent,
            )
            return
        # action_msgs/GoalStatus: STATUS_SUCCEEDED=4, STATUS_ABORTED=6,
        # STATUS_CANCELED=5. We treat aborted/canceled as controller
        # failures; succeeded passes through silently.
        result = wrapped.result
        status = int(wrapped.status)
        now_ns = self.get_clock().now().nanoseconds
        if status == 4 and result.success:
            self.get_logger().info(
                f"execute_rskill succeeded rskill_id={call.rskill_id!r} "
                f"trace_id={result.trace_id!r}",
            )
            # ADR-0072 §2.2 — surface SUCCESS to the LLM (Inner Monologue). The
            # failure path already reaches the FAILURES buffer; success used to
            # pass through silently, leaving the reasoner blind to "it worked".
            self._renderer.append_execution(
                ExecutionEventRecord(
                    rskill_id=call.rskill_id,
                    outcome="ok",
                    summary=f"trace={result.trace_id[:8] if result.trace_id else '-'}",
                    reflection=None,
                    stamp_ns=now_ns,
                )
            )
            # ADR-0073 §2 — runner "success" for a VLA means "ran to its deadline
            # without a controller fault", NOT "task accomplished". Verify the
            # active mission task against the reward signal before advancing.
            self._maybe_verify_active_mission_task(call, traceparent=traceparent)
            return
        # ADR-0074 §2 — a reward-driven cancel (status 5, reason "reward") is an
        # intentional stop, not a controller fault: the reward-watcher decided
        # the attempt was over (success/plateau/patience). Verify on the reward
        # signal — the three-tier / VLM gate completes or advances the ladder —
        # and skip the KIND_CONTROLLER failure path (no fault to report).
        if status == 5 and cancel_reason == "reward":
            self.get_logger().info(
                f"execute_rskill reward-cancelled rskill_id={call.rskill_id!r} — verifying",
            )
            self._maybe_verify_active_mission_task(call, traceparent=traceparent)
            return
        self.get_logger().warning(
            f"execute_rskill failed rskill_id={call.rskill_id!r} status={status} "
            f"reason={result.failure_reason!r}",
        )
        outcome_state = "aborted" if status == 6 else "canceled" if status == 5 else "failed"
        detail = result.failure_reason or f"GoalStatus={status}"
        # ADR-0072 §2.2/§2.3 — execution feedback + a Reflexion strategy hint so
        # the next tick advances the ladder instead of blindly retrying.
        self._renderer.append_execution(
            ExecutionEventRecord(
                rskill_id=call.rskill_id,
                outcome="failed",
                summary=detail,
                reflection=reflect_on_failure(outcome_state, detail),
                stamp_ns=now_ns,
            )
        )
        self._publish_skill_failure(
            kind=_KIND_CONTROLLER,
            rskill_id=call.rskill_id,
            evidence=ControllerEvidence(
                controller_name=call.rskill_id,
                state=outcome_state,
                detail=detail,
            ),
            traceparent=traceparent,
            trace_id=result.trace_id or None,
        )
        # ADR-0073 §2 — an aborted (terminal) episode still ran the policy, so it
        # is a real attempt at the active task; verify so a repeatedly-aborting
        # task is bounded by the attempt cap (abandon → hand off) rather than
        # looping forever. Canceled (status 5) is operator-driven, not an attempt.
        if status == 6:
            self._maybe_verify_active_mission_task(call, traceparent=traceparent)

    def _on_execute_rskill_deadline(
        self,
        *,
        call: ExecuteRskillTool,
        sent_at: float,
        goal_handle: Any,
        traceparent: str | None,
    ) -> None:
        """Patience-ceiling backstop (ADR-0074 §2): cancel goal + emit ``KIND_TIMEOUT``.

        Fires only when the reward-watcher did not stop the attempt first — the
        resolved patience ceiling elapsed. ``deadline_s`` in the log/evidence is
        that resolved patience (LLM override > reward default > legacy).
        """
        goal_id = bytes(goal_handle.goal_id.uuid)
        timer = self._pending_skill_deadlines.pop(goal_id, None)
        if timer is not None:
            timer.cancel()
        elapsed = time.monotonic() - sent_at
        patience_s = self._effective_patience_s(call)
        self.get_logger().warning(
            f"execute_rskill patience ceiling patience_s={patience_s:.0f} "
            f"elapsed_s={elapsed:.3f} — emitting KIND_TIMEOUT FailureTrigger and cancelling goal",
        )
        try:
            goal_handle.cancel_goal_async()
        except Exception as exc:  # reason: cancel is best-effort
            self.get_logger().error(
                f"execute_rskill cancel_goal_async failed: {type(exc).__name__}: {exc}",
            )
        self._publish_skill_failure(
            kind=_KIND_TIMEOUT,
            rskill_id=call.rskill_id,
            evidence=TimeoutEvidence(
                operation=f"skill.{call.rskill_id}",
                deadline_s=patience_s,
                elapsed_s=elapsed,
            ),
            traceparent=traceparent,
        )

    # ── Lifecycle service callback ──────────────────────────────────────────

    def _on_lifecycle_response(
        self,
        call: LifecycleTransitionTool,
        future: Any,
    ) -> None:
        """Log the ``ChangeState`` result.

        Failure is logged but not re-published as a FailureTrigger —
        lifecycle clients are operator-driven and the failure surface
        lives in the target node's own logs.
        """
        try:
            resp = future.result()
        except Exception as exc:  # reason: surface rclpy errors
            self.get_logger().error(
                f"lifecycle_transition node={call.node!r} transition={call.transition!r} "
                f"call failed: {type(exc).__name__}: {exc}",
            )
            return
        if resp.success:
            self.get_logger().info(
                f"lifecycle_transition node={call.node!r} transition={call.transition!r} ok",
            )
        else:
            self.get_logger().warning(
                f"lifecycle_transition node={call.node!r} transition={call.transition!r} "
                "rejected by the target node",
            )

    # ── FailureTrigger emit helper ──────────────────────────────────────────

    def _publish_skill_failure(
        self,
        *,
        kind: int,
        rskill_id: str,
        evidence: Any,
        traceparent: str | None,
        trace_id: str | None = None,
    ) -> None:
        """Publish a :class:`FailureTrigger` on ``/openral/failure/rskill``.

        ``trace_id`` (when supplied — e.g. propagated by the action
        server's result) takes precedence; otherwise the reasoner's
        active ``traceparent`` is used so a downstream replanner / F7
        correlator can join the failure event to the producing tick.
        """
        if self._failure_pub is None:
            return
        msg = IDLFailureTrigger()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "rskill"
        msg.kind = int(kind)
        msg.severity = _SEVERITY_FAIL
        msg.evidence_json = evidence.model_dump_json()
        msg.rskill_id = rskill_id
        msg.trace_id = trace_id or traceparent or ""
        self._failure_pub.publish(msg)

    # ── public helpers for tests ────────────────────────────────────────────

    @property
    def renderer(self) -> ContextRenderer:
        """Direct read access for tests asserting buffer state."""
        return self._renderer

    @property
    def dispatched_calls(self) -> tuple[Any, ...]:
        """Snapshot of tool calls the reasoner has dispatched (in order)."""
        return tuple(self._dispatched_calls)

    def set_palette(self, palette: ToolPalette) -> None:
        """Replace the active palette (rebuilt on ``/openral/skill_registry_changed``)."""
        self._palette = palette


def main(args: list[str] | None = None) -> int:
    """Entry point for ``ros2 run openral_reasoner_ros reasoner_node``."""
    from openral_observability import configure_observability

    # Idempotent + no-op when OTEL_EXPORTER_OTLP_ENDPOINT is unset.
    # The launch passes the dashboard endpoint via additional_env so
    # `reasoner.tick` spans + metrics land on the live UI.
    configure_observability(service_name="openral.reasoner")

    rclpy.init(args=args)
    try:
        node = ReasonerNode()
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
