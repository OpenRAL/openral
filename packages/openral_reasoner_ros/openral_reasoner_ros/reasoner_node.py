#!/usr/bin/env python3
"""ROS 2 reasoner + supervisor graph — ``reasoner_node`` lifecycle wrapper.

Subscribes to:

* ``/openral/world_state_slow``  — ``openral_msgs/WorldStateStamped``, 5 Hz
* ``/openral/failure/{hal,sensor,rskill,safety,wam,critic}``  — ``openral_msgs/FailureTrigger``
* ``/openral/perception/{motion,objects,ocr,scene_change}``  — ``openral_msgs/PromptStamped``
* ``/openral/prompt``            — ``openral_msgs/PromptStamped`` (operator)

Heartbeat tick at ``tick_hz`` (default 0.2 Hz = one every 5 s; was 5 Hz
pre-2026-05-25 amendment to this design). The event bus is the primary
trigger: an incoming :class:`FailureTrigger` with
``severity>=SEVERITY_FAIL`` (or ``>=SEVERITY_WARN`` on
``/openral/failure/safety`` — Tier A), or a new ``/openral/prompt``
arrival, forces an out-of-band tick (subject to the
:class:`~openral_reasoner.ReasonerCore` 100 ms min-interval per
the reasoner+supervisor design §4). Heartbeat ticks that see no new event since the last
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

The reasoner **never** publishes ``openral_msgs/ActionChunk`` (per the
reasoner+supervisor design §4 "Holds no authority over actuation").
"""

from __future__ import annotations

import contextlib
import datetime
import json
import pathlib
import subprocess
import sys
import time
from collections import deque
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
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
    SpatialNodeKind,
    TimeoutEvidence,
    WaitTool,
    assert_vla_reward_fits,
    control_modes_for_representation,
    is_collective_target,
)
from openral_core.exceptions import ROSConfigError, ROSGPUMemoryError, ROSReasonerInvalidPlan
from openral_observability import log_lifecycle_errors
from openral_reasoner.active_search import SearchBudget, SearchProgress
from openral_reasoner.completion import (
    COMPLETION_QUESTION as _COMPLETION_QUESTION,
)
from openral_reasoner.completion import (
    image_msg_to_jpeg as _image_msg_to_jpeg,
)
from openral_reasoner.completion import (
    is_frame_fresh as _is_frame_fresh,
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
    RewardStateRecord,
    reflect_on_failure,
    reflect_on_invalid_plan,
    reflect_on_retry_cap,
    reflect_on_reward_plateau,
    render_playbooks_block,
    render_robot_self_model,
)
from openral_reasoner.core import PreparedTick, ReasonerCore, ReasonerTickResult
from openral_reasoner.memory import MemoryEntry, MemoryStore
from openral_reasoner.mission import (
    DEFAULT_MAX_SUBDIVIDE_DEPTH,
    MissionState,
    TaskLocateBudget,
    TaskState,
    evaluate_task_verdict,
)
from openral_reasoner.node_policy import (
    CASCADE_PROMPT_SOURCES,
    should_rebuild_mission,
)
from openral_reasoner.palette import (
    ToolPalette,
    build_tool_palette,
    locate_in_view_service,
    task_space_disagreement,
)
from openral_reasoner.persistence import (
    ReasonerLadderState,
    load_ladder_state,
    save_ladder_state,
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
from rclpy.time import Time

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
# palette ("palette ... refreshed on
# /openral/skill_registry_changed").
try:  # pragma: no cover — gated by sourced ROS install
    from std_msgs.msg import Empty as IDLEmpty
except ImportError:  # pragma: no cover
    IDLEmpty = None  # type: ignore[assignment, misc]

# std_msgs/String — the reward monitor's active-task signal (2026-06-29). The
# reasoner publishes the EXACT instruction the VLA is running (the active subtask)
# so the reward model scores the same question the policy is acting on, not the
# collective mission goal; empty = no VLA acting (gates continuous scoring off).
# Gated like IDLEmpty above.
try:  # pragma: no cover — gated by sourced ROS install
    from std_msgs.msg import String as IDLString
except ImportError:  # pragma: no cover
    IDLString = None  # type: ignore[assignment, misc]


__all__ = ["ReasonerNode"]

# QoS profiles per the reasoner+supervisor design §1 + CLAUDE.md §5.3
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
# The occupancy-grid-refined approach phase — the slam_toolbox map is latched (description/static
# QoS class): RELIABLE + TRANSIENT_LOCAL so a late-joining reasoner still receives the current grid
# snapshot.
_QOS_MAP = QoSProfile(
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
)

# VLM-adjudicated completion (§5) — BEST_EFFORT sensor QoS for the completion-camera frame cache.
# One-frame keep-last: the adjudicator always sees the most recent frame; older
# frames are dropped rather than queued. VOLATILE means no history replay (the
# verification window is the present moment, not a historical one).
_QOS_COMPLETION_CAMERA = QoSProfile(
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    durability=QoSDurabilityPolicy.VOLATILE,
)

# Closed sets from the reasoner+supervisor design §3 / capability review §3.
# `rskill` was renamed from `skill` on 2026-05-25 (reasoner+supervisor design amendment §5)
# for consistency with the carried `rskill_id` field.
_FAILURE_SOURCES: tuple[str, ...] = ("hal", "sensor", "rskill", "safety", "wam", "critic")
_PERCEPTION_KINDS: tuple[str, ...] = ("motion", "objects", "ocr", "scene_change")

# Reward-gated task verification §2 / VLM-adjudicated completion amendment — reward window (s) for
# the automatic post-skill task verification. Robometer scores a trajectory from its START, so the
# verify must request the WHOLE attempt (start→now), not a trailing slice: an 8 s tail missed the
# completion arc and under-scored progress to ~0.70 (vlm_check/ladder) on real successes whose
# full-attempt progress was ~0.85. The request uses the active reward model's ``frame_window_s``
# (the buffer retention = the attempt horizon) when a contract is wired; this constant is the
# fallback when none is. Sized to span the default 30 s patience ceiling + margin.
_MISSION_VERIFY_WINDOW_S: float = 40.0

# VLM-adjudicated completion, Decision 5 — three-tier verdict band edges + the patience ceiling.
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

# Fallback failure-state names for the dashboard EVENT_SKILL_FAILURE event when
# the evidence carries no ``state`` field (e.g. KIND_TIMEOUT's TimeoutEvidence).
# ControllerEvidence already names its state (``vram_insufficient`` /
# ``unavailable`` / ``aborted``), so this only covers the kinds without one.
_SKILL_FAILURE_KIND_NAMES: dict[int, str] = {
    _KIND_TIMEOUT: "timeout",
    _KIND_CONTROLLER: "controller",
}

# Brief, non-blocking probe used before sending an ExecuteSkill goal — if
# the F1 server isn't on the graph yet we emit a KIND_CONTROLLER
# FailureTrigger instead of blocking the executor thread.
_EXECUTE_SKILL_SERVER_PROBE_S: float = 0.1
_LIFECYCLE_SERVER_PROBE_S: float = 0.1

# Reasoner+supervisor design, 2026-05-25 amendment — trigger taxonomy. Maps each failure
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
# incompatibility worth a WARN (amendment 2026-05-27).
# Canonical source is ``openral_core.WRAPPED_TASK_SPACE_LAYOUTS``
# (single source of truth so the schema validator and the
# reasoner filter stay in lockstep).
from openral_core import WRAPPED_TASK_SPACE_LAYOUTS as _WRAPPED_TASK_SPACE_LAYOUTS  # noqa: E402

# Deploy-path-aware action-mode palette gate (amended 2026-06-04).
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


def _query_gpu_gb(field: str) -> float:
    """One ``nvidia-smi --query-gpu=<field>`` value for GPU 0 in GB, or ``0.0``.

    Deliberately torch-free (the reasoner_node stays cheap to import — torch is
    only pulled lazily for the skill loader). Any failure (no nvidia-smi, no
    GPU, parse error) returns ``0.0`` → callers skip their check rather than
    blocking dispatch on a host where the value can't be read.
    """
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={field}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5.0,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return 0.0
    first = out.stdout.strip().splitlines()
    if not first:
        return 0.0
    try:
        return float(first[0].strip()) / 1024.0  # MiB → GiB
    except ValueError:
        return 0.0


def _detect_gpu_total_vram_gb() -> float:
    """Total VRAM (GB) of GPU 0, or ``0.0`` when unavailable.

    Used by the VLA/reward VRAM-fit pre-dispatch pair check when the
    ``gpu_total_vram_gb`` param is unset.
    """
    return _query_gpu_gb("memory.total")


def _detect_gpu_free_vram_gb() -> float:
    """*Free* VRAM (GB) on GPU 0 right now, or ``0.0`` when unavailable.

    The static pair check above budgets against the card's TOTAL, which is
    blind to what other processes hold at dispatch time — observed live
    (2026-07-20): an external vLLM server held 4.7 GB of an 8 GB card, the
    palette's molmoact2 (declared 4.0 GB) passed every static gate, and the
    dispatch burned ~30 s in a CUDA OOM abort. Probed per dispatch (the
    dispatch path is slow-path; one nvidia-smi call is ~100 ms).
    """
    return _query_gpu_gb("memory.free")


def _required_control_modes(manifest: RSkillManifest) -> set[ControlMode]:
    """The :class:`ControlMode` s a skill's ``action_contract`` demands.

    Pure helper (no ROS spin) so the deploy-path palette gate is unit
    testable. The contract is read in order of specificity:

    * No ``action_contract`` → empty set (the skill declares no action
      constraint, so it is admitted by :func:`_action_executable`).
    * ``representation`` set → :func:`control_modes_for_representation`.
    * ``slots`` set → every slot's ``control_mode`` (discard slots carry
      ``None`` and are skipped).
    * Bare ``dim`` only (legacy rosbag2↔LeRobotDataset bridge contract) → ``{JOINT_POSITION}``;
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
    """Whether the deploy path can execute a skill's action modes.

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


# Minimum seconds between operator-visible (WARNING) execute_rskill feedback
# log lines; the raw per-chunk stream continues at DEBUG.
_FEEDBACK_LOG_PERIOD_S: float = 1.0


def _resets_search_episode(call: Any) -> bool:
    """True when dispatching ``call`` should end the active-search episode.

    The cascade bound (active object search §3) counts only *consecutive*
    spatial-search queries, so any
    non-search dispatch resets ``_spatial_search`` + ``_locate_escalated``. The
    search actions that must NOT reset are ``recall_object`` / ``resolve_place``
    (remembered objects) and ``locate_in_view`` (live detector) — the latter is
    the regression this guards: if a directly-emitted ``locate_in_view`` reset
    the budget, a ``recall → locate → recall`` loop against an undetectable
    object would zero the counter every cycle and never hand off. ``wait`` is
    exempt for the same reason: a deliberate no-op makes no progress on the
    search, so a ``locate → wait → locate`` alternation must keep charging the
    budget rather than zeroing it every other tick (caught live by
    ``test_cascade_reprompt_does_not_reset_search_budget``).
    """
    return not isinstance(call, RecallObjectTool | ResolvePlaceTool | LocateInViewTool | WaitTool)


def _search_term(call: RecallObjectTool | ResolvePlaceTool) -> str:
    """Return the term a spatial-search miss should escalate to the detector on.

    The two search variants name it differently — ``recall_object`` carries
    ``query``, ``resolve_place`` carries ``reference`` — which is why the
    open-vocab ``locate_in_view`` escalation used to be gated on
    ``isinstance(call, RecallObjectTool)`` and skipped places entirely. A
    ``resolve_place`` miss then fell through to a re-prompt saying only "not in
    memory", the LLM read that as "try again", and it looped on the same tool
    until the search budget handed off to a human — with an open-vocab detector
    sitting idle in the same graph.

    Example:
        >>> _search_term(RecallObjectTool(query="baguette"))
        'baguette'
        >>> _search_term(ResolvePlaceTool(reference="the counter"))
        'the counter'
    """
    return call.query if isinstance(call, RecallObjectTool) else call.reference


def _palette_after_rskill_failure(
    palette: ToolPalette,
    rskill_id: str,
    detail: str,
) -> ToolPalette:
    """Drop a skill after a typed, session-persistent availability failure."""
    if not detail.startswith(("ROSConfigError:", "ROSCapabilityMismatch:")):
        return palette
    return palette.model_copy(
        update={
            "skills": tuple(skill for skill in palette.skills if skill.rskill_id != rskill_id),
            "execute_rskill_ids": palette.execute_rskill_ids - {rskill_id},
        }
    )


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


# The "collective target" predicate (grounded task decomposition) is the single source of truth in
# ``openral_core`` (`is_collective_target`) — shared by the `GroundedSubtask`
# schema validator and this node's runtime execute gate so a skill never acts on
# a quantified/plural set ("all the objects"); it must be enumerated from the live
# ``scene_objects`` context and decomposed into one grounded subtask per object.


class ReasonerNode(LifecycleNode):
    """ROS 2 lifecycle wrapper around :class:`ReasonerCore` (reasoner + supervisor graph F4).

    Args:
        node_name: ROS node name. Default ``openral_reasoner``.
        tick_hz: Heartbeat tick rate in Hz. Default 0.2 (one every
            5 s). Per the reasoner+supervisor design amendment 2026-05-25, the reasoner is
            event-driven: failure/prompt arrivals preempt with
            ``force=True``, and the periodic timer is the safety net
            for "task is not making progress but nothing has fired".
            Heartbeat ticks that see no new event since the last
            successful tick are short-circuited inside
            :class:`ReasonerCore` with
            ``suppressed_reason="heartbeat_idle"``.
        client: Optional pre-built :class:`ToolUseClient`. When ``None``
            :meth:`on_configure` builds one from the
            ``OPENRAL_REASONER_*`` env vars via
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

        ``spatial_memory`` (active object search, Phase 2b) is an optional read-only
        scene-graph query backend (a persistent-spatial-memory ``SpatialMemory``). When
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
        # Persistent spatial memory, live dynamic memory — when the reasoner *owns* the backend
        # (preloaded from disk, or auto-created for `spatial_memory_ingest`),
        # this concrete handle lets `_on_tick` fold each WorldState.detected_objects
        # snapshot into it. Stays None for an externally-injected read-only
        # querier (we don't mutate a backend we don't own).
        self._spatial_memory_writer: SpatialMemory | None = None
        # Last emitted semantic scene key and keepalive anchor.
        self._scene_objects_key: tuple[object, ...] | None = None
        self._scene_objects_emitted_monotonic: float = 0.0
        # Occupancy-grid-refined approach phase — latest decoded occupancy grid (an
        # ``openral_world_state.grid.OccupancyGridIndex``), from the latched
        # ``occupancy_map_topic`` subscription. ``None`` until a map arrives;
        # ``_dispatch_spatial_query`` then refines every recall_object approach
        # viewpoint through it (grid absent → geometric viewpoints pass
        # through unchanged).
        self._occupancy_grid: Any = None
        # Active object search §3 — bound the find→re-prompt cascade so a query that keeps
        # missing terminates in human-handoff instead of looping forever.
        self._spatial_search = SearchProgress(SearchBudget())
        # locate_in_view / on-demand detectors — recall_object queries already escalated to a live
        # locate_in_view this search streak (one escalation per query term, so a
        # repeated miss doesn't re-fire the detector every tick). Reset whenever
        # the active-search bound resets (new operator goal / non-search action).
        self._locate_escalated: set[str] = set()
        # #123 — task ids already offered one subdivision before being abandoned.
        # One offer per task so a task that declines to decompose still terminates
        # in human-handoff; cleared when a new operator goal rebuilds the mission.
        self._subdivide_offered: set[str] = set()
        # VLM-adjudicated completion amendment — PER-TASK locate budget. The ``_spatial_search``
        # bound only counts locate MISSES and resets on a HIT, so a live
        # locate-loop where ``locate_in_view`` keeps HITTING (found=True) but the
        # reasoner never dispatches an ``execute_rskill`` never terminates. This
        # counts locate cycles spent on the *active mission task* (regardless of
        # hit/miss); once exhausted the active subtask is abandoned via the
        # mission ladder with a displayed reason so the next pick proceeds. Reset
        # on task advance and on a successful execute dispatch.
        self._task_locate_budget = TaskLocateBudget()
        # VLM-adjudicated completion amendment 2026-06-29 — locate-budget × ground-before-decompose:
        # on a COLLECTIVE goal, locating to confirm objects IS the legitimate path to
        # `decompose_mission` (which is not a skill dispatch), so a budget-hit there
        # must NOT abandon the mission — it nudges decompose and resets the budget.
        # Bounded by this per-task nudge cap so a goal that genuinely can't be
        # grounded still terminates (abandon) instead of looping nudge↔locate.
        self._collective_decompose_nudges: dict[str, int] = {}
        self._max_collective_decompose_nudges = 2

        # ROS parameters: when both are set, on_configure walks
        # `rskill_search_paths` for `*/rskill.yaml`, loads the
        # `RobotCapabilities` from `robot_yaml`, and seeds the palette
        # via `build_tool_palette`. Either parameter being empty leaves
        # the palette at the constructor-supplied value (or empty),
        # preserving the existing `/openral/skill_registry_changed`
        # refresh path for HF-Hub-installed skills.
        self.declare_parameter("robot_yaml", "")
        self.declare_parameter("rskill_search_paths", [""])
        # Reasoner-managed background services — additional lifecycle peer node names to surface in
        # the LLM tool palette's `node_ids` slot so the Reasoner can
        # emit `LifecycleTransitionTool(node=..., transition=...)` against
        # background services like `/openral_slam_toolbox`. Defaults to
        # empty; deploy launches set this via `reasoner_lifecycle_peers:=
        # [openral_slam_toolbox]` when the corresponding `--enable-<svc>`
        # CLI flag was passed.
        self.declare_parameter("lifecycle_peer_node_ids", [""])
        # Single-resident-skill VRAM eviction — GPU lifecycle peers (the object-detector
        # LifecycleNode is the canonical one) to DEACTIVATE before dispatching a GPU-heavy
        # ``execute_rskill`` and REACTIVATE once it finishes, so their VRAM is freed for the policy.
        # Without this the detector (~1.3 GB) co-resident with a VLA (~4.5 GB) OOMs an 8 GB card at
        # load. Default empty; the deploy launch sets it to the detector node id when
        # ``--enable-object-detector``. Distinct from ``lifecycle_peer_node_ids`` (which only
        # surfaces peers to the LLM tool palette, not auto-managed).
        self.declare_parameter("vram_lifecycle_peers", [""])
        # Active object search, Phase 2b deployment wiring — absolute path to a persisted
        # persistent-spatial-memory scene graph (``SceneGraph`` JSON written by
        # ``SpatialMemory.save``). When set (and no ``spatial_memory`` backend
        # was injected), ``on_configure`` loads it into a ``SpatialMemory`` and
        # wires it as the read-only query backend, enabling the
        # ``recall_object`` / ``resolve_place`` tools against a preloaded map.
        # Empty = disabled.
        self.declare_parameter("spatial_memory_path", "")
        # Reasoner playbooks + self-maintained memory §3 / Phase 4b — path to the self-maintained
        # MEMORY.md (read at configure into the `## MEMORY` context block). Empty omits the section.
        self.declare_parameter("memory_md_path", "")
        # Reasoner playbooks + self-maintained memory §3 / Phase 5 — retrieval-under-cap: render at
        # most this many memory entries in the always-on `## MEMORY` block (top by importance then
        # recency; the tail stays searchable via memory_search). 0 = no cap.
        self.declare_parameter("memory_context_cap", 0)
        # Crash-safe ladder resume — absolute path where the mission ledger +
        # every replanning-ladder bound (subdivision offers, decompose nudges,
        # the per-task locate budget) is snapshotted after each mutation and
        # reloaded at configure, so a reasoner restart RESUMES the ladder
        # instead of resetting every cap mid-mission (CLAUDE.md §1.8
        # replayability). Empty disables persistence (previous behaviour).
        self.declare_parameter("ladder_state_path", "")
        # Dispatch-phase watchdog ceiling (s) for the send→goal-response
        # window (see `_dispatch_watchdog`). Generous vs. the ~1 s accept
        # latency observed live so executor starvation (issue #21) cannot
        # produce false wedge verdicts; <= 0 disables the watchdog.
        self.declare_parameter("dispatch_watchdog_s", 30.0)
        # Persistent spatial memory, live dynamic memory — when true, ``on_configure`` ensures a
        # ``SpatialMemory`` backend exists (auto-creating an empty one if no
        # ``spatial_memory_path`` was loaded and none injected) and ``_on_tick``
        # folds each ``/openral/world_state_slow`` ``WorldState.detected_objects``
        # snapshot into it — accumulating the durable scene graph from the
        # perception → spatial-memory object-lift producer so ``recall_object`` recalls
        # what the robot has actually seen. Default false (preloaded-map only).
        self.declare_parameter("spatial_memory_ingest", False)
        # Occupancy-grid-refined approach phase — occupancy-grid refinement of recall approach
        # poses. The reasoner subscribes the latched slam_toolbox map on this
        # topic and validates/snaps every ``recall_object`` approach viewpoint
        # (free under ``approach_inflation_m`` + line-of-sight) before the LLM
        # sees it. Empty string disables the subscription; no map received →
        # geometric viewpoints pass through unchanged.
        self.declare_parameter("occupancy_map_topic", "/map")
        self.declare_parameter("approach_inflation_m", 0.25)
        # Deploy-path-aware action-mode palette gate — deploy-path selector for the palette gate.
        # ``"sim"`` (default; deploy sim is the common path) admits skills
        # whose action modes a robosuite OSC / composite controller can
        # synthesise; ``"real"`` admits only the robot's declared
        # ``supported_control_modes``. The deploy launch sets this
        # explicitly to match the HAL it brings up (a later task).
        self.declare_parameter("hal_mode", "sim")
        # locate_in_view: on-demand live-detector query — when true, offer the read-only
        # ``locate_in_view`` tool (ask a live VLM detector if an object is in the current frame, via
        # the ``/openral/perception/locate_in_view`` service). The deploy launch sets this when it
        # brings up an object detector. Default false (no hidden tool).
        self.declare_parameter("detector_available", False)
        self._detector_available: bool = (
            self.get_parameter("detector_available").get_parameter_value().bool_value
        )
        # On-demand detectors as prompt-able reasoner tools — the default on-demand locator alias
        # used when a locate_in_view call leaves ``detector`` empty
        # (e.g. "omdet_turbo-any-locator-fp16").
        # Empty = the legacy single-detector service /openral/perception/locate_in_view. Set by the
        # deploy launch to the default locator it brings up.
        self.declare_parameter("default_on_demand_detector", "")
        self._default_on_demand_detector: str = (
            self.get_parameter("default_on_demand_detector").get_parameter_value().string_value
        )
        # On-demand detectors as prompt-able reasoner tools — locate_in_view clients cached per
        # resolved service name (one per on-demand locator the reasoner has routed to), created
        # lazily.
        self._locate_in_view_clients: dict[str, Any] = {}
        # vlm rSkill kind — when true, offer the read-only ``query_scene`` tool (ask a
        # scene VLM an open-ended question about the current view, via the
        # ``/openral/perception/query_scene`` service). The deploy launch sets
        # this when it brings up a scene VLM. Default false (no hidden tool).
        self.declare_parameter("scene_query_available", False)
        self._scene_query_available: bool = (
            self.get_parameter("scene_query_available").get_parameter_value().bool_value
        )
        # Cached client for the query_scene service; created lazily on first use.
        self._query_scene_client: Any = None
        # kind: reward rSkills — when true, offer the read-only ``query_task_progress`` tool
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

        # VLM-adjudicated completion §1/§3 — the active reward model's manifest (same path the
        # reward_monitor_node loads). When set, the node reads its
        # ``RewardContract`` calibration (band edges + default patience) and uses
        # it in place of the module-level system fallbacks. A bad path degrades
        # to the fallbacks (logged) rather than failing node construction.
        self._reward_contract: RewardContract | None = None
        # VLA/reward VRAM-fit pairing — the full reward manifest (not just its contract) so the
        # pre-dispatch VLA+reward VRAM check has the reward model's `min_vram_gb`.
        self._reward_manifest: RSkillManifest | None = None
        # VLA/reward VRAM-fit pairing — VLA manifests keyed by rskill_id, loaded lazily on first
        # dispatch (the palette path discards them); used for the pair fit check.
        self._manifests_by_id: dict[str, RSkillManifest] = {}
        # VLA/reward VRAM-fit pairing — total GPU VRAM (GB) for the pair fit check. The deploy may
        # pin it via the `gpu_total_vram_gb` param; else probe nvidia-smi once.
        # 0.0 = unknown → the check is skipped (cannot verify what we can't read).
        self.declare_parameter("gpu_total_vram_gb", 0.0)
        self._gpu_total_vram_gb = float(
            self.get_parameter("gpu_total_vram_gb").get_parameter_value().double_value
        )
        if self._gpu_total_vram_gb <= 0.0:
            self._gpu_total_vram_gb = _detect_gpu_total_vram_gb()
        self.declare_parameter("reward_manifest_path", "")
        reward_manifest_path = (
            self.get_parameter("reward_manifest_path").get_parameter_value().string_value
        )
        if reward_manifest_path:
            try:
                self._reward_manifest = RSkillManifest.from_yaml(reward_manifest_path)
                self._reward_contract = self._reward_manifest.reward
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

        # Make the pair fit check's state explicit in the logs (§1.4):
        # armed only when a reward model is active AND the GPU total is known.
        if self._reward_manifest is not None and self._gpu_total_vram_gb > 0.0:
            self.get_logger().info(
                f"VLA+reward VRAM fit check ARMED — reward="
                f"{self._reward_manifest.name!r} "
                f"({self._reward_manifest.active_min_vram_gb()} GB), "
                f"gpu_total={self._gpu_total_vram_gb:.2f} GB",
            )
        else:
            why = (
                "no reward model active"
                if self._reward_manifest is None
                else "GPU total unreadable"
            )
            self.get_logger().info(f"VLA+reward VRAM fit check SKIPPED ({why})")

        # VLM-adjudicated completion §5 — completion-camera topic for VLM adjudication.
        # When set to a non-empty string, on_configure subscribes sensor_msgs/Image
        # on this topic (BEST_EFFORT, VOLATILE, depth=1) and caches the latest frame
        # as JPEG bytes in `_latest_completion_frame` for `_adjudicate_completion`.
        # Empty string disables the subscription (no hidden camera subscription).
        self.declare_parameter("completion_camera_topic", "/openral/cameras/top/image")
        # VLM-adjudicated completion §5 — the HAL publishes LIBERO/MuJoCo frames bottom-up (the
        # topic is raw; OPENRAL_DASHBOARD_FLIP_180 flips only the dashboard thumbnail —
        # sim_sensor_bridge). Rotate the cached completion frame 180° so the VLM judges an upright
        # scene (the dashboard and the VLA apply the same flip).
        self.declare_parameter("completion_camera_flip_180", False)
        # VLM-adjudicated completion §5 — reject a completion frame older than this many seconds (a
        # stale frame from a prior attempt would make the VLM judge the wrong
        # scene → false verdict). 0 disables the guard. Frames stream continuously
        # on real hardware; in deploy-sim the sim-clock is frozen during verify so
        # the end-of-execution frame reads age≈0.
        self.declare_parameter("completion_frame_max_age_s", 2.0)

        # VLM-adjudicated completion §5 — tool-use client handle (mirrors the one held by
        # ReasonerCore) so the completion gate can call describe_image without reaching into the
        # core. Set in on_configure; cleared in on_cleanup.
        self._tool_use_client: ToolUseClient | None = None
        # Latest completion frame as JPEG bytes; None until the first camera message.
        self._latest_completion_frame: bytes | None = None
        # ROS-clock receipt time of the cached frame (for the freshness guard).
        self._latest_completion_frame_time: Time | None = None
        # Cached at on_configure from the params above.
        self._completion_flip_180: bool = False
        self._completion_frame_max_age_s: float = 0.0

        # Populated by on_configure.
        self._renderer: ContextRenderer = ContextRenderer()
        self._world_state_msg: Any = None
        self._core: ReasonerCore | None = None
        # Log a `retry_cap` suppression only ONCE per streak — without this it
        # re-warns every heartbeat tick and floods the log. Cleared the moment a
        # non-retry_cap tick happens (a different tool, a dispatch, an error, or
        # a new operator prompt that resets the streak).
        self._retry_cap_warned: bool = False
        # Reasoner playbooks + self-maintained memory, Phase 3 — the rendered `## PLAYBOOKS`
        # system-prompt block, collected from installed capability-matched playbook rSkills at seed
        # time.
        self._playbooks_block: str = ""
        # Reasoner playbooks + self-maintained memory §3 — the self-maintained MEMORY.md store
        # (Phase 4b read path: loaded at configure + rendered as the `## MEMORY` context block;
        # Phase 4c write path: `memory_write` edits + `memory_search` archival recall). The archive
        # is the append-only log of superseded/deleted entries that left the live file (MemGPT
        # recall storage); persisted as `<MEMORY.md>.archive.jsonl`.
        self._memory_store: MemoryStore | None = None
        self._memory_md_path: pathlib.Path | None = None
        self._memory_archive: list[MemoryEntry] = []
        self._palette: ToolPalette = palette or ToolPalette(execute_rskill_ids=frozenset())
        # Active object search — offer the read-only query tools only when a backend is wired.
        if spatial_memory is not None and not self._palette.spatial_memory_available:
            self._palette = self._palette.model_copy(update={"spatial_memory_available": True})
        self._tick_timer: Any = None
        self._prompt_pub: Any = None
        # EmitPromptTool publishers keyed by target topic (beyond the standing
        # /openral/prompt cascade publisher), created lazily per topic.
        self._emit_prompt_pubs: dict[str, Any] = {}
        self._failure_pub: Any = None  # /openral/failure/rskill
        # Single-flight tick trampoline — several dispatch/verify handlers
        # force a follow-up tick synchronously; running it *nested* inside the
        # in-flight tick stacks blocking LLM calls on the executor thread and
        # recurses _on_tick → _dispatch → _on_tick. The trampoline queues the
        # strongest requested (force, tier) instead and replays it after the
        # outer tick returns — same semantics, flat stack, one LLM call at a
        # time.
        self._tick_in_flight: bool = False
        self._queued_tick: tuple[bool, str] | None = None
        # Consecutive queued-tick replays (backstop counter for the
        # trampoline; reset whenever a tick finishes with nothing queued).
        self._tick_replays: int = 0
        # Async LLM phase (#21): the blocking select_tool / describe_image
        # round-trips run on this single worker so the rclpy executor stays
        # free for goal results, patience timers, and Tier-A preemptions.
        # One worker = one outstanding LLM call, preserving the trampoline's
        # single-flight semantics. Built in on_configure, torn down in
        # on_cleanup.
        self._llm_pool: ThreadPoolExecutor | None = None
        # Separate single worker for the VLM completion gate (describe_image)
        # so a Tier-A tick's select_tool is never queued behind an in-flight
        # adjudication call (both can run for the full LLM timeout budget).
        self._vlm_pool: ThreadPoolExecutor | None = None
        # Marshals worker completions back onto the executor thread: the
        # worker appends a callable and triggers the guard condition, whose
        # callback drains the inbox on the next executor spin.
        self._executor_inbox: deque[Callable[[], None]] = deque()
        self._inbox_guard: Any = None
        # Bumped on deactivate/cleanup so an LLM round-trip that lands after
        # a lifecycle transition is dropped instead of dispatching onto a
        # stopped node.
        self._llm_generation: int = 0
        # An execute_rskill goal is being dispatched or is running (covers the
        # send→accept window `_active_rskill_goal` cannot). Gates a second
        # dispatch — the runner serves one goal at a time, and a forced tick
        # mid-execution used to double-dispatch blind.
        self._rskill_inflight: bool = False
        # Dispatch-phase watchdog: bounds the send→goal-response window the
        # busy latch opens. rclpy futures never time out on their own, so a
        # runner (or VRAM peer) that dies AFTER the readiness probe but before
        # its response leaves the latch set forever — wedging all future
        # dispatches. One-shot timer armed with the latch, cancelled when the
        # goal response arrives (any outcome); its callback releases the latch
        # and emits a KIND_CONTROLLER FailureTrigger so the ladder handles it.
        self._dispatch_watchdog: Any = None  # rclpy Timer
        # The send_goal_async future of the current dispatch (None during the
        # peer-eviction phase / when idle). The watchdog checks .done() before
        # declaring a wedge so a merely-starved executor (issue #21: callbacks
        # observed ~70 s late) is not misread as a dead runner.
        self._dispatch_send_future: Any = None
        # A watchdog expiry invalidates its generation before releasing the
        # busy latch, so late callbacks cannot mutate a newer dispatch.
        self._dispatch_generation: int = 0
        self._active_dispatch_generation: int | None = None
        # rskill_id of the VLA the runner most recently ACCEPTED (and keeps
        # warm after the goal ends). The live free-VRAM probe skips a
        # re-dispatch of this exact skill: the resident policy's own VRAM is
        # why free is low, and re-dispatching it needs ~0 new VRAM — probing
        # would falsely refuse every large-VLA dispatch after the first.
        self._resident_vla_id: str | None = None
        # monotonic() of the last operator-visible feedback WARNING (throttle).
        self._last_feedback_warn_s: float = -float("inf")
        # Crash-safe ladder persistence (`ladder_state_path` param): the
        # snapshot file, or None when persistence is disabled.
        self._ladder_state_path: pathlib.Path | None = None
        self._execute_rskill_client: Any = None  # rclpy.action.ActionClient
        # Lifecycle clients are cached per target node — one
        # ``<node>/change_state`` client per peer.
        self._lifecycle_clients: dict[str, Any] = {}
        # Single-resident-skill VRAM eviction — GPU lifecycle peers to free before a VLA dispatch
        # (read from ``vram_lifecycle_peers`` at configure) and the subset actually deactivated for
        # the in-flight skill (reactivated on its result).
        self._vram_lifecycle_peers: list[str] = []
        self._deactivated_vram_peers: list[str] = []
        # Pending skill-goal deadline timers, keyed by goal-uuid bytes so
        # the result callback can cancel the deadline timer when the
        # action server returns before deadline_s elapses.
        self._pending_skill_deadlines: dict[bytes, Any] = {}
        # VLM-adjudicated completion §2 — the in-flight execute_rskill goal so a reward-watcher
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
    def on_configure(  # noqa: PLR0915  # reason: linear lifecycle setup — declare params, build the tool-use client, and open each gated subscriber in sequence; splitting hurts readability
        self, state: LifecycleState
    ) -> TransitionCallbackReturn:
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

        # VLM-adjudicated completion §5 — hold the client on the node so the VLM adjudication gate
        # can call describe_image without reaching into ReasonerCore internals.
        self._tool_use_client = client

        # Async LLM phase (#21) — single worker for the blocking LLM/VLM
        # round-trips + guard condition to marshal completions back onto the
        # executor thread.
        self._llm_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="reasoner-llm")
        self._vlm_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="reasoner-vlm")
        self._inbox_guard = self.create_guard_condition(self._drain_executor_inbox)

        # Start a managed LLM sidecar NOW rather than on the first tick.
        # Clients that own one (the Cosmos 3 Edge local vLLM) otherwise boot
        # it lazily from ``select_tool``, i.e. after the whole graph is up
        # and an operator is already waiting on a decision — a vLLM model
        # load at best, a venv provision plus a ~9 GB download on a cold
        # host. Bringup has minutes of unrelated work (HAL ``on_configure``,
        # MuJoCo, camera first-frame gating) to overlap it with.
        #
        # On the LLM pool, so ``on_configure`` still returns promptly and
        # the lifecycle transition is not held open by a model load. Cloud
        # clients expose no ``warm`` and are untouched. Failure is
        # non-fatal: the lazy path in ``select_tool`` remains the source of
        # truth for whether the server is actually usable, and it will
        # report the real error at the point it matters.
        self._submit_client_warmup(client)

        # NOTE: ``self._core`` is built *after* the palette seed below, so the
        # robot-context system prompt (option B) reflects the capabilities
        # loaded from ``robot_yaml``. Nothing between here and then dispatches
        # a tick (callbacks only run once the executor spins, after configure
        # returns), so the late construction is safe.

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
        # Recovery signal: the operator cleared the safety e-stop (broadcast by
        # the kernel-reset path, same topic the HAL + runner un-latch on). The
        # reasoner is otherwise BLIND to the clear — the e-stop abort stays in its
        # failure context, so after a reset it keeps refusing to retry ("please
        # clear the e-stop") instead of re-dispatching. On clear, drop that stale
        # failure context so the next operator prompt starts fresh. QoS matches
        # the estop publishers (RELIABLE / VOLATILE / depth 10).
        _estop_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=10,
        )
        self.create_subscription(
            IDLEmpty,
            "/openral/estop_cleared",
            self._on_estop_cleared,
            _estop_qos,
        )

        # VLM-adjudicated completion §5 — completion-camera subscription (BEST_EFFORT sensor QoS).
        # sensor_msgs/Image ships with every ROS 2 install but is gated like
        # nav_msgs above so a stripped environment degrades to "no frame cache"
        # instead of failing configure. An empty topic param disables the sub.
        completion_camera_topic = (
            self.get_parameter("completion_camera_topic").get_parameter_value().string_value
        )
        self._completion_flip_180 = (
            self.get_parameter("completion_camera_flip_180").get_parameter_value().bool_value
        )
        self._completion_frame_max_age_s = (
            self.get_parameter("completion_frame_max_age_s").get_parameter_value().double_value
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

        # Occupancy-grid-refined approach phase — latched occupancy grid for approach refinement.
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

        # Palette is rebuilt on every
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
        # The `rskill` suffix replaced `skill` on 2026-05-25 (reasoner+supervisor
        # design amendment §5).
        self._failure_pub = self.create_publisher(
            IDLFailureTrigger,
            "/openral/failure/rskill",
            _QOS_FAILURE,
        )

        # 2026-06-29 — reward monitor's active-task signal. Published with the exact
        # instruction the VLA is running on each execute_rskill dispatch, and empty
        # on its result. The reward leg only scores while non-empty (the user's
        # "reward only runs when the VLA runs") AND scores against THAT instruction,
        # not the collective mission goal (so a single-object pick is judged as the
        # single-object task the policy is actually doing). RELIABLE +
        # TRANSIENT_LOCAL so a late-joining monitor sees the latest state. None when
        # std_msgs is unavailable (degrades to no gate / monitor's default task).
        self._reward_active_pub = (
            self.create_publisher(
                IDLString,
                "/openral/reward/active_task",
                QoSProfile(
                    history=QoSHistoryPolicy.KEEP_LAST,
                    depth=1,
                    reliability=QoSReliabilityPolicy.RELIABLE,
                    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
                ),
            )
            if IDLString is not None
            else None
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

        # Active object search — load a persisted scene graph into the query backend
        # before the palette seed, so the rebuilt palette offers the query
        # tools when a map is preloaded.
        self._maybe_load_spatial_memory()
        self._maybe_load_memory()
        self._maybe_restore_ladder_state()

        # Single-resident-skill VRAM eviction — GPU lifecycle peers to deactivate before a VLA
        # dispatch and reactivate after (the object detector is the canonical one). Read
        # unconditionally so it is honoured regardless of whether the palette seed path runs. Empty
        # entries skipped.
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

        # Option B (reasoner+supervisor design F4): give the reasoner LLM standing knowledge of
        # the body it drives. ``self._robot_capabilities`` is now finalised
        # (from the constructor or the ``robot_yaml`` loaded during the seed),
        # so the system prompt carries a ``## THIS ROBOT`` block; ``None``
        # leaves the robot-agnostic brief unchanged. The base brief honours
        # the ``OPENRAL_REASONER_SYSTEM_PROMPT`` deployment override.
        base_prompt = resolve_reasoner_system_prompt(self._robot_capabilities)
        # Reasoner playbooks, Phase 3 — append installed playbooks (empty block = no-op).
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

    def _submit_client_warmup(self, client: object) -> None:
        """Kick a managed LLM sidecar's boot onto the LLM pool, if it has one.

        No-op for clients without a ``warm()`` — every cloud provider.
        Runs off the executor thread so ``on_configure`` returns promptly
        and the lifecycle transition is not held open by a model load.

        A failure here is not the deploy's problem: the lazy path in
        ``select_tool`` still owns whether the server is usable and
        surfaces the real error where an operator can act on it. Logging at
        warning keeps a silent sidecar failure visible during bringup
        rather than only at the first tick.
        """
        warm = getattr(client, "warm", None)
        if not callable(warm):
            return

        def _run() -> None:
            try:
                warm()
            except Exception as exc:  # reason: pre-warm is an optimisation, never a gate
                self.get_logger().warning(f"on_configure: LLM sidecar pre-warm failed: {exc!s}")

        self._llm_pool.submit(_run)

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
        # Drop any in-flight async LLM round-trip: its completion callback
        # compares generations and discards itself instead of dispatching
        # onto a deactivated node (the pre-async design could not be
        # mid-tick across a lifecycle transition; this preserves that).
        self._llm_generation += 1
        self._tick_in_flight = False
        self._queued_tick = None
        self._tick_replays = 0
        self.get_logger().info("on_deactivate: stopped")
        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Drop state; subscriptions are auto-cleaned by rclpy."""
        del state
        self._core = None
        self._occupancy_grid = None
        self._renderer = ContextRenderer()
        # VLM-adjudicated completion §5 — clear the VLM client handle and frame cache on cleanup.
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
        for publisher in self._emit_prompt_pubs.values():
            self.destroy_publisher(publisher)
        self._emit_prompt_pubs.clear()
        self._locate_in_view_clients.clear()
        self._query_scene_client = None
        self._query_task_progress_client = None
        self._active_rskill_goal = None
        self._rskill_cancel_reason = None
        self._rskill_inflight = False
        self._active_dispatch_generation = None
        self._cancel_dispatch_watchdog()
        self._resident_vla_id = None
        self._tick_in_flight = False
        self._queued_tick = None
        self._tick_replays = 0
        # Async LLM phase (#21) teardown — a still-running worker call is not
        # interruptible (provider SDKs have their own timeouts); shutdown
        # without waiting and let the generation bump drop its completion.
        self._shutdown_worker_pools()
        if self._inbox_guard is not None:
            self.destroy_guard_condition(self._inbox_guard)
            self._inbox_guard = None
        self._executor_inbox.clear()
        self.get_logger().info("on_cleanup: state cleared")
        return TransitionCallbackReturn.SUCCESS

    def _shutdown_worker_pools(self) -> None:
        """Drop the LLM / VLM worker pools and abandon any in-flight call.

        A still-running worker call is not interruptible (provider SDKs own
        their timeouts), so shut down without waiting and let the generation
        bump drop its completion.

        Called from BOTH ``on_cleanup`` and ``on_shutdown``. ``on_cleanup``
        alone is not enough: ``_submit_client_warmup`` starts the managed vLLM
        sidecar early in ``on_configure``, and ``@log_lifecycle_errors`` turns a
        later raise in that same transition into ``FAILURE`` — which leaves the
        node ``unconfigured``, a state ``on_cleanup`` is never entered from. The
        pool's non-daemon thread and the child it spawned would then outlive the
        node with nothing left to reap them.
        """
        self._llm_generation += 1
        if self._llm_pool is not None:
            self._llm_pool.shutdown(wait=False, cancel_futures=True)
            self._llm_pool = None
        if self._vlm_pool is not None:
            self._vlm_pool.shutdown(wait=False, cancel_futures=True)
            self._vlm_pool = None

    def on_shutdown(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Final shutdown.

        Reaps the worker pools unconditionally — this is the only teardown hook
        a node that failed ``on_configure`` will ever reach.
        """
        del state
        self._shutdown_worker_pools()
        self.get_logger().info("on_shutdown")
        return TransitionCallbackReturn.SUCCESS

    # ── topic callbacks ─────────────────────────────────────────────────────

    def _on_world_state(self, msg: Any) -> None:
        """Cache the latest WorldStateStamped snapshot."""
        self._world_state_msg = msg

    def _on_map(self, msg: Any) -> None:
        """Decode the latched occupancy grid for approach refinement.

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
        """Cache the latest camera frame as JPEG bytes for VLM adjudication (§5).

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
                flip_180=self._completion_flip_180,
            )
        except Exception as exc:  # reason: never raise in a topic callback
            self.get_logger().debug(f"completion-camera: decode failed, cache unchanged: {exc}")
            return
        self._latest_completion_frame = jpeg
        self._latest_completion_frame_time = self.get_clock().now()

    def _adjudicate_completion(self, task_text: str) -> bool | None:
        """Ask the VLM whether ``task_text`` is complete in the latest camera frame.

        Returns ``True`` (complete), ``False`` (not complete), or ``None``
        (could not adjudicate — no frame cached or no multimodal client).
        ``None`` degrades to the ladder (§6 no-VLM path). A provider
        or transport error is logged and returns ``None`` — never a false ``True``.

        Args:
            task_text: The active task description shown to the VLM.
        """
        if self._latest_completion_frame is None:
            self.get_logger().debug(
                "adjudicate_completion: no frame cached — cannot adjudicate; treating as no"
            )
            return None
        if self._latest_completion_frame_time is not None:
            age_s = (self.get_clock().now() - self._latest_completion_frame_time).nanoseconds / 1e9
            if not _is_frame_fresh(age_s=age_s, max_age_s=self._completion_frame_max_age_s):
                self.get_logger().info(
                    f"adjudicate_completion: frame is stale ({age_s:.1f}s > "
                    f"{self._completion_frame_max_age_s:.1f}s) — cannot adjudicate; treating as no"
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

    def _adjudicate_completion_async(
        self, task_text: str, done: Callable[[bool | None], None]
    ) -> None:
        """Run :meth:`_adjudicate_completion` off-executor; deliver the verdict back (#21).

        ``describe_image`` shares the LLM timeout budget (10 s cloud / 60 s
        local) and used to run inside a service done-callback, starving the
        executor exactly like ``select_tool``. The whole sync helper is safe
        on the worker: it only reads immutable snapshot refs (frame bytes,
        client handle) and the network call. ``done(verdict)`` runs on the
        executor thread; it is dropped when a lifecycle transition landed
        first (generation mismatch). With no pool (pre-configure), degrades
        to the synchronous call. Runs on its OWN single worker (not the tick
        pool) so a Tier-A tick's select_tool is never queued behind an
        in-flight adjudication call.
        """
        if self._vlm_pool is None:
            done(self._adjudicate_completion(task_text))
            return
        generation = self._llm_generation

        def _worker() -> None:
            try:
                verdict = self._adjudicate_completion(task_text)
            except Exception as exc:  # reason: a worker crash must not vanish silently
                self.get_logger().warning(f"adjudicate_completion (async) raised: {exc}")
                verdict = None

            def _deliver() -> None:
                if generation != self._llm_generation:
                    return  # lifecycle transitioned mid-flight — stale verdict
                done(verdict)

            self._post_to_executor(_deliver)

        self._vlm_pool.submit(_worker)

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
        self._persist_ladder_state()
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
        """Append a failure event; preempt per the reasoner+supervisor design trigger taxonomy.

        Tier A (``source == "safety"``) preempts on
        ``severity >= SEVERITY_WARN`` (=1) — a safety WARN means the
        C++ kernel (or F5 pass-through) saw a near-miss and the LLM
        needs to be in the loop before the next chunk lands.

        Tier B (``hal`` / ``sensor`` / ``rskill`` / ``wam``) and Tier C
        (``critic``) preempt on ``severity >= SEVERITY_FAIL`` (=2);
        WARN/INFO are buffered without preemption.

        See the reasoner+supervisor design amendment 2026-05-25 §3 for the full taxonomy.
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
        # VLM-adjudicated completion §2 — a reward-watcher wake (critic FAIL) while a VLA is in
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
        """Three-tier verdict band edges from the active reward calibration (§1/§5).

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
        """Patience ceiling for a dispatch (§2/§3).

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
        """Cancel the in-flight execute_rskill goal on a reward-watcher wake (§2).

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
            "to verify on the reward signal",
        )
        try:
            goal_handle.cancel_goal_async()
        except Exception as exc:  # reason: cancel is best-effort; deadline backstops it
            self.get_logger().error(
                f"reward-cancel cancel_goal_async failed: {type(exc).__name__}: {exc}",
            )

    def _on_perception(self, kind: str, msg: Any) -> None:
        """Append a perception event; no preemption — perception is informational.

        Detection-time object identity + camera-space enumeration: an ``objects``
        event also refreshes the reasoner's camera-space
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

        For a genuine external prompt (not one of the reasoner's own
        cascade sources) this also resets the core's consecutive-call
        streak and the active-search bounds before forcing the tick: a
        fresh operator prompt is a new situation, so the previous streak
        carries no information. Cascade re-prompts reset **nothing** —
        they are responses the bounds are counting.
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
        # Reward-gated task verification §1 — a genuine operator goal (re)builds the mission queue:
        # the operator goal seeds one task so the reasoner sequences and the
        # goal survives the pull-once prompt drain. Cascade re-prompts
        # (advisory query responses, spatial-memory) are NOT new goals and must
        # not reset the mission — and while a mission is IN PROGRESS an
        # external prompt only replaces it when its metadata carries
        # `"new_goal": true` (an operator *reply* to a reasoner question must
        # not silently discard the in-flight queue; it reaches the LLM via
        # the PROMPTS section). Policy is pure + single-sourced in
        # `openral_reasoner.node_policy` (unit-tested without rclpy).
        if should_rebuild_mission(source, msg.metadata_json, self._renderer.mission):
            mission = MissionState.from_prompt(msg.text)
            if not mission.is_empty():
                self._renderer.set_mission(mission)
                self._subdivide_offered.clear()  # #123 — fresh goal, fresh offers
                self._reset_task_locate_budget()  # fresh goal, fresh budget
                self._collective_decompose_nudges.clear()  # fresh goal, fresh nudge cap
                # Sticky open-vocab grounding from the previous goal is stale
                # authority for a new one (objects may have been moved).
                self._renderer.clear_located()
                self._persist_ladder_state()
                self.get_logger().info(
                    f"mission: {len(mission)} task(s) — active={mission.active().text[:80]!r}",
                )
        # A fresh *operator* prompt is a new situation — reset the retry-cap
        # call streak and the active-search bounds. The reasoner's own cascade
        # re-prompts (detector / spatial_memory / reward_monitor / mission /
        # memory / scene_vlm) must reset NEITHER: a locate-miss re-prompt that
        # reset the search budget it had just charged could never exhaust it
        # (the pre-fix bug: only "spatial_memory" was excluded here), and a
        # cascade response resetting the call streak disabled the retry cap
        # for every query loop.
        if source not in CASCADE_PROMPT_SOURCES:
            if self._core is not None:
                self._core.reset_kind_streak()
            self._spatial_search.reset()
            self._locate_escalated.clear()
        self._on_tick(force=True, tier="D")

    def _on_estop_cleared(self, _msg: Any) -> None:
        """Operator cleared the safety e-stop — drop stale failure context.

        The e-stop aborts the in-flight skill and records a ``safety_estop``
        failure + a failed execution. Those are stale the moment the operator
        resets, but the reasoner otherwise keeps them in context and refuses to
        retry ("the e-stop aborted the motion / please clear the e-stop") even
        after the HAL + runner have un-latched. Clearing them (and the retry-cap
        streak) lets the NEXT operator prompt dispatch cleanly. We deliberately
        do NOT auto-dispatch: the operator re-prompts when ready (they may want
        to reposition the arm or the object first).
        """
        self._renderer.clear_failures()
        if self._core is not None:
            self._core.reset_kind_streak()
        self.get_logger().info(
            "reasoner.estop_cleared; dropped stale e-stop failure context "
            "— ready to retry on the next prompt.",
        )

    def _on_skill_registry_changed(self, msg: Any) -> None:
        """Rebuild the tool palette from the local rSkill registry.

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
            # A rebuild must not silently drop the memory tools a wired
            # MEMORY.md enabled (the pre-fix bug: this flag was omitted, so
            # every registry refresh removed memory_write / memory_search).
            memory_available=self._memory_store is not None,
        )

    def _persist_ladder_state(self) -> None:
        """Snapshot the mission ledger + ladder bounds to ``ladder_state_path``.

        Called after every ledger mutation (mission set/advance/subdivide,
        attempt recorded, subdivision offered, nudge charged). Best-effort:
        a write failure logs and never disturbs the loop. No-op when the
        param is unset.
        """
        if self._ladder_state_path is None:
            return
        try:
            save_ladder_state(
                self._ladder_state_path,
                ReasonerLadderState(
                    mission=self._renderer.mission,
                    subdivide_offered=set(self._subdivide_offered),
                    collective_nudges=dict(self._collective_decompose_nudges),
                    locate_task_id=self._task_locate_budget.task_id,
                    locate_count=self._task_locate_budget.count,
                ),
            )
        except OSError as exc:
            self.get_logger().warning(
                f"ladder persistence: failed to write {self._ladder_state_path}: {exc}",
            )

    def _maybe_restore_ladder_state(self) -> None:
        """Restore a persisted mission + ladder bounds at configure (crash resume).

        A missing/corrupt/finished snapshot starts fresh (logged). Restoring
        sets the mission on the renderer — the next tick resumes on the
        restored active task instead of idling for a new operator goal.
        """
        path = self.get_parameter("ladder_state_path").get_parameter_value().string_value
        self._ladder_state_path = pathlib.Path(path) if path else None
        if self._ladder_state_path is None:
            return
        restored = load_ladder_state(self._ladder_state_path)
        if restored is None:
            self.get_logger().info(
                f"ladder persistence: no usable snapshot at {path!r}; starting fresh",
            )
            return
        self._subdivide_offered = set(restored.subdivide_offered)
        self._collective_decompose_nudges = dict(restored.collective_nudges)
        self._task_locate_budget.restore(restored.locate_task_id, restored.locate_count)
        if (
            restored.mission is not None
            and not restored.mission.is_empty()
            and not restored.mission.is_complete()
        ):
            self._renderer.set_mission(restored.mission)
            active = restored.mission.active()
            active_txt = active.text[:60] if active is not None else "(no active task)"
            self.get_logger().info(
                f"ladder persistence: restored mission ({len(restored.mission)} task(s), "
                f"active={active_txt!r})",
            )
        elif restored.mission is not None and restored.mission.is_complete():
            # A finished mission restarts fresh (per the docstring contract) —
            # restoring it would idle the reasoner on a completed queue.
            self.get_logger().info(
                "ladder persistence: snapshot mission is complete; starting fresh",
            )

    def _maybe_load_memory(self) -> None:
        """Load the self-maintained ``MEMORY.md`` into the ``## MEMORY`` block (§3).

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
        """Wire the persistent-spatial-memory backend at ``on_configure``.

        Used by active object search. No-op when a ``spatial_memory``
        backend was injected at construction.
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

    # Repopulates collectors that restart while the scene stays unchanged.
    _SCENE_OBJECTS_KEEPALIVE_S: float = 60.0

    def _emit_scene_objects_span(self) -> None:
        """Publish the remembered objects as a ``world.scene_objects`` span.

        Advisory dashboard telemetry only (never a safety input). No-op without a
        spatial-memory backend; any failure is swallowed at DEBUG so a telemetry
        hiccup can never disturb the reasoning loop. Today the backend is the
        preloaded ``spatial_memory_path`` map; post-producer (once the
        perception → spatial-memory object lift lands, PR #229) the World-State
        node becomes the canonical emitter of the same span.

        Emit-on-change: this is called on every 0.2 Hz heartbeat, but an
        unchanged scene re-emits only every ``_SCENE_OBJECTS_KEEPALIVE_S`` —
        a static map produced 720 zero-information event rows per hour, each
        dragging the full object list JSON along. The change key is SEMANTIC
        (id / label / pose to 1 cm / is_container): detector ingest bumps
        ``last_seen_ns`` / ``observation_count`` and jitters ``pose`` on every
        snapshot, so an exact-payload hash would never suppress anything.
        """
        if self._spatial_memory is None:
            return
        try:
            import time as _time

            from openral_world_state import emit_scene_objects_span

            graph = self._spatial_memory.to_scene_graph()
            key: tuple[object, ...] = tuple(
                sorted(
                    (
                        node.node_id,
                        node.label,
                        *(round(value, 2) for value in node.pose.xyz),
                        node.is_container,
                    )
                    for node in graph.nodes
                    if node.kind is SpatialNodeKind.OBJECT
                )
            )
            now = _time.monotonic()
            if (
                key == self._scene_objects_key
                and now - self._scene_objects_emitted_monotonic < self._SCENE_OBJECTS_KEEPALIVE_S
            ):
                return
            emit_scene_objects_span(graph, source_node=self.get_name())
            self._scene_objects_key = key
            self._scene_objects_emitted_monotonic = now
        except Exception as exc:  # reason: telemetry must never break the tick
            self.get_logger().debug(f"scene-objects span emit failed: {exc!s}")

    def _ingest_detected_objects(self, world_state: Any) -> None:
        """Fold a snapshot's ``detected_objects`` into the owned SpatialMemory.

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

        Reasoner playbooks, Phase 3: for each ``kind: playbook`` manifest this robot satisfies
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
            # LLM to call ``execute_rskill`` on the playbook itself (a
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
        # Reasoner playbooks, Decision 2.1 — render the static robot self-model once and
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
                manifest = RSkillManifest.from_yaml(str(path))
            except (OSError, ValueError) as exc:
                self.get_logger().warning(
                    f"palette seed: skipping unloadable rskill {path!s}: {exc}",
                )
                continue
            manifests.append(manifest)
            loaded_paths.append(path)
            # Feed the pre-dispatch VRAM gate from the SAME manifests the LLM is
            # being offered. `_manifest_for_rskill` otherwise falls back to the
            # install registry (`~/.local/share/openral/rskills.json`), which a
            # search-path-seeded palette never populates: on the reference host
            # the palette carried 48 manifests while the registry held 3 detector
            # skills and no VLA at all, so `_refuse_unfittable_vla` early-returned
            # `False` for EVERY VLA and both its tiers were dead. Observed live
            # 2026-08-04: `molmoact2-multi-so101-nf4` (4.0 GB + 5.5 GB reward on
            # an 8 GB card) was named "will be refused at dispatch" by the CLI
            # preflight, then dispatched, loaded its processor, and burned the
            # full 20 s patience ceiling before being cancelled as KIND_TIMEOUT —
            # a VRAM refusal reported as a timeout.
            self._manifests_by_id.setdefault(manifest.name, manifest)

        # Reasoner playbooks, Phase 3 — collect installed, capability-matched `kind: playbook`
        # rSkills and render their PLAYBOOK.md bodies into the `## PLAYBOOKS`
        # system-prompt block. Playbooks are role:s2 (excluded from the ExecuteSkill
        # palette); they reach the LLM as authored decision-procedure *content*.
        self._playbooks_block = self._collect_playbooks_block(manifests, loaded_paths)

        # Reasoner-managed background services — merge any deploy-time lifecycle peer node ids
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
                    # admit-with-adapter when the layout's
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
                            "in openral_state_adapter. "
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

        # Action-mode executability filter (deploy-path-aware action-mode palette gate): drop VLA
        # rSkills whose action vector drives a ControlMode the deploy path can't execute. The
        # state-contract filter above gates the *input* (state dim vs joint count); this gates the
        # *output*. Without it a cartesian/OSC skill gets offered to the LLM on a joint-only robot
        # and fails at runtime — the n_dof / control-mode mismatch surfaces as a mid-run estop
        # instead of a single boot-time warning. ``hal_mode`` selects the executable set: ``"sim"``
        # admits the robosuite-OSC default set even on a joint-only physical robot; ``"real"``
        # admits only the robot's declared ``supported_control_modes``. Non-vla skills
        # (``ros_action`` / ``ros_service``) pass through — they don't emit an ``ActionChunk`` from
        # a learned action vector.
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
                # Shared TaskSpace action-space contract, Phase 2 — shadow the canonical TaskSpace
                # gate alongside the legacy mode check (warn-only; the legacy verdict still decides
                # the drop). Surfaces cross-layer mismatches the ``_action_executable`` mode-set
                # check misses — an EE-addressed slot naming an end-effector the robot does not
                # declare, or a joint segment wider than the robot's joint count. Phase 4 makes
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
            # Active object search — preserve the read-only query tools when a spatial-memory
            # backend is wired; `_maybe_load_spatial_memory` runs before this seed
            # and a rebuild without the flag would silently drop recall_object /
            # resolve_place.
            spatial_memory_available=self._spatial_memory is not None,
            detector_available=self._detector_available,
            scene_query_available=self._scene_query_available,
            task_progress_available=self._task_progress_available,
            # Same preservation for the memory tools: `_maybe_load_memory`
            # runs before this seed, and a rebuild omitting the flag silently
            # dropped memory_write / memory_search in every deploy that set
            # both `memory_md_path` and `rskill_search_paths`.
            memory_available=self._memory_store is not None,
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
        if result.suppressed_reason in ("min_interval", "heartbeat_idle", "retry_cap_hold"):
            # retry_cap_hold is the steady-state after a capped streak whose
            # context has not moved (the LLM call is skipped entirely) —
            # debug-level like the other quiet suppressions. None of these
            # three touch the streak (no LLM call ran), so the one-shot
            # retry_cap warn latch must survive them: falling through to the
            # clear below re-warned (and re-appended a duplicate ladder
            # reflection) for the SAME unbroken streak after every hold.
            self.get_logger().debug(f"tick suppressed: {result.suppressed_reason}")
            return
        elif result.suppressed_reason == "retry_cap":
            # Warn once per streak, not every heartbeat — otherwise this
            # floods the log while the model keeps re-picking the same call.
            if not self._retry_cap_warned:
                self._retry_cap_warned = True
                cap = self._core.retry_cap if self._core is not None else "N"
                self.get_logger().warning(
                    f"tick suppressed: retry_cap — the identical tool call {cap}+ ticks in "
                    "a row. A new operator prompt resets the streak; otherwise it "
                    "self-clears when the model changes its call. (Repeats logged at debug.)",
                )
                # Reasoner playbooks §2.3 — inject a Reflexion strategy hint into context
                # (once per streak) so the NEXT tick changes approach instead of
                # looping. Appending bumps `seq`, so the next heartbeat runs
                # rather than being suppressed as idle (and releases the
                # pre-call retry_cap_hold, which holds on the capped seq).
                if self._core is not None:
                    tool = self._core.streak_tool
                    self._renderer.append_execution(
                        ExecutionEventRecord(
                            rskill_id="(ladder)",
                            outcome="failed",
                            summary=f"retry ladder exhausted for {tool!r}",
                            reflection=reflect_on_retry_cap(tool, self._core.retry_cap),
                            stamp_ns=self.get_clock().now().nanoseconds,
                        )
                    )
            else:
                self.get_logger().debug("tick suppressed: retry_cap (ongoing streak)")
            # An ongoing retry_cap streak keeps the one-shot latch set.
            return
        else:
            self.get_logger().info(f"tick suppressed: {result.suppressed_reason}")
        # Loud suppressions (palette_empty, …) reach the LLM-facing flow, so a
        # later cap is a fresh event worth re-warning about. Quiet suppressions
        # return above with the latch intact — the streak they interleave with
        # is unbroken. Unsuppressed ticks clear it at the _on_tick call site.
        self._retry_cap_warned = False

    #: Max queued-tick replays one trampoline entry will run. The caps
    #: (retry-cap identity streak, invite/nudge bounds) terminate the
    #: dispatch→tick ping-pong well before this; the constant is a backstop
    #: so a future bug degrades to a logged skip, not an unbounded loop.
    _MAX_TICK_REPLAYS: int = 4

    def _on_tick(self, *, force: bool = False, tier: str = "heartbeat") -> None:
        """Start one orchestrator pass (single-flight, queued-replay trampoline).

        Several dispatch/verify handlers force a follow-up tick synchronously
        from *inside* a tick's own dispatch. Running it nested stacked
        blocking LLM calls on the executor thread and recursed
        ``_on_tick → _dispatch → _on_tick``; instead, a request that arrives
        while a tick is in flight is coalesced (``force`` wins) and replayed
        after the in-flight pass finishes — same semantics, flat stack, one
        LLM call at a time.

        Since #21 the blocking LLM phase runs on a worker thread:
        ``_tick_in_flight`` now spans prepare → worker round-trip →
        :meth:`_finish_llm_tick`, and the executor stays free in between —
        goal results, patience timers, and Tier-A preemptions run instead of
        queueing behind the LLM call (their forced ticks coalesce here
        exactly as before).

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
        if self._tick_in_flight:
            prev = self._queued_tick
            if prev is None or (force and not prev[0]):
                self._queued_tick = (force, tier)
            return
        self._tick_in_flight = True
        try:
            llm_in_flight = self._start_tick(force=force, tier=tier)
        except BaseException:
            self._tick_in_flight = False
            self._queued_tick = None
            self._tick_replays = 0
            raise
        if not llm_in_flight:
            self._release_tick_and_maybe_replay()

    def _release_tick_and_maybe_replay(self) -> None:
        """End the single-flight window; replay the coalesced queued tick, bounded.

        The replay counter accumulates across *consecutive* chained replays
        (the async analog of the old in-loop counter) and resets whenever a
        tick finishes with nothing queued, so the ``_MAX_TICK_REPLAYS``
        backstop still degrades a future ping-pong bug to a logged skip.
        """
        self._tick_in_flight = False
        queued = self._queued_tick
        self._queued_tick = None
        if queued is None:
            self._tick_replays = 0
            return
        self._tick_replays += 1
        if self._tick_replays > self._MAX_TICK_REPLAYS:
            self.get_logger().warning(
                f"tick trampoline: dropped a queued tick after "
                f"{self._MAX_TICK_REPLAYS} replays (tier={queued[1]})",
            )
            self._tick_replays = 0
            return
        self._on_tick(force=queued[0], tier=queued[1])

    def _post_to_executor(self, fn: Callable[[], None]) -> None:
        """Marshal ``fn`` onto the rclpy executor thread. Thread-safe.

        Appends to the inbox and wakes the executor via the guard
        condition; :meth:`_drain_executor_inbox` runs ``fn`` on the next
        spin. After cleanup (guard destroyed) the callable is dropped —
        the generation check in the callables makes that safe.
        """
        self._executor_inbox.append(fn)
        guard = self._inbox_guard
        if guard is not None:
            guard.trigger()

    def _drain_executor_inbox(self) -> None:
        """Guard-condition callback: run every marshaled callable in order."""
        while True:
            try:
                fn = self._executor_inbox.popleft()
            except IndexError:
                return
            try:
                fn()
            except BaseException:
                # Preserve crash semantics (an unexpected error still raises
                # into the executor) but don't strand the remaining inbox.
                if self._executor_inbox and self._inbox_guard is not None:
                    self._inbox_guard.trigger()
                raise

    def _start_tick(self, *, force: bool, tier: str) -> bool:
        """One orchestrator pass: render context → hand the LLM phase to the worker.

        Returns ``True`` when the blocking LLM phase went to the worker
        thread (finish + single-flight release deferred to
        :meth:`_finish_llm_tick`); ``False`` when the tick completed
        synchronously (suppressed by a gate, or no core yet).
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
        # Persistent spatial memory — fold the snapshot's detected_objects into the durable memory
        # we own, then refresh the dashboard's scene-objects view. Both run on
        # every heartbeat, independent of LLM readiness (a preloaded/accumulating
        # map is worth maintaining even before the tool-use client is built).
        self._ingest_detected_objects(world_state)
        self._emit_scene_objects_span()
        if self._core is None:
            return False
        prep = self._core.prepare_tick(
            world_state=world_state,
            renderer=self._renderer,
            palette=self._palette,
            force=force,
            tier=tier,
        )
        if isinstance(prep, ReasonerTickResult):
            # A suppression gate fired — no LLM call, finish synchronously.
            self._handle_tick_result(prep)
            return False
        if self._llm_pool is None:  # unreachable: pool is built with _core in on_configure
            raise ROSConfigError("reasoner tick with no LLM worker pool — configure first")
        core = self._core
        generation = self._llm_generation

        def _llm_worker() -> None:
            # Worker thread: ONLY the blocking client round-trip. Everything
            # stateful (retry-cap, renderer drain, dispatch) is marshaled
            # back onto the executor thread.
            call: Any = None
            error: BaseException | None = None
            try:
                call = core.run_prepared_llm(prep)
            except Exception as exc:  # reason: marshaled to finish_tick, the single handler
                error = exc
            self._post_to_executor(
                lambda: self._finish_llm_tick(prep, call=call, error=error, generation=generation)
            )

        self._llm_pool.submit(_llm_worker)
        return True

    def _finish_llm_tick(
        self,
        prep: PreparedTick,
        *,
        call: Any,
        error: BaseException | None,
        generation: int,
    ) -> None:
        """Executor-thread continuation of :meth:`_start_tick` (worker done).

        Runs :meth:`ReasonerCore.finish_tick` (bookkeeping + span close),
        handles the result exactly as the synchronous path did, then
        releases the single-flight window and replays the coalesced queued
        tick, if any. A round-trip that lands after deactivate/cleanup
        (generation mismatch) is dropped — the span is closed, nothing
        dispatches.
        """
        if generation != self._llm_generation or self._core is None:
            prep.span.end()
            self.get_logger().info("tick: dropping stale LLM result (lifecycle transitioned)")
            return
        try:
            result = self._core.finish_tick(prep, call=call, error=error)
            self._handle_tick_result(result)
        except BaseException:
            self._tick_in_flight = False
            self._queued_tick = None
            self._tick_replays = 0
            raise
        self._release_tick_and_maybe_replay()

    def _handle_tick_result(self, result: ReasonerTickResult) -> None:
        """Shared post-tick handling: suppression, error feedback, dispatch."""
        if result.suppressed_reason:
            self._handle_suppressed_tick(result)
            return
        # A tick that was not suppressed (dispatch, error, or no-op) breaks any
        # retry_cap streak — clear the latch so a future streak warns again.
        self._retry_cap_warned = False
        if result.error is not None:
            self.get_logger().warning(f"tick error: {result.error!s}")
            # Reasoner playbooks §2.2/§2.3 — an invalid plan is the model's *own* mistake
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

    def _dispatch(self, call: Any, *, traceparent: str | None = None) -> None:  # noqa: PLR0911, PLR0912  # reason: one return/branch per tool variant — a flat dispatch table is clearer than collapsing the isinstance branches
        """Route a typed tool call onto the ROS graph.

        :class:`EmitPromptTool` publishes inline. :class:`ExecuteRskillTool`
        sends an action goal on ``/openral/execute_rskill`` and wires
        feedback/result/timeout into ``/openral/failure/rskill``.
        :class:`LifecycleTransitionTool` calls ``<node>/change_state``.
        :class:`ReloadGstPipelineTool` remains a log-and-acknowledge
        stub pending the F6 sensor-package service IDL.
        """
        # Active object search §3 — any non-search dispatch ends the search episode, so the
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
        if isinstance(call, WaitTool):
            # Deliberate no-op: the LLM chose to observe. The rationale is
            # already on the reasoner span (ReasonerCore logs it); nothing to
            # dispatch, no state change, no re-prompt.
            self.get_logger().debug(f"dispatch: wait — {call.rationale or 'no rationale'}")
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

    def _emit_prompt_publisher(self, target_topic: str) -> Any:
        """The ``PromptStamped`` publisher for ``target_topic`` (cached per topic).

        ``/openral/prompt`` reuses the standing cascade publisher; any other
        topic the LLM targets gets its own publisher created on first use and
        cached — the schema and system prompt document emit_prompt as
        "publish onto another topic", and silently rerouting every call to
        ``/openral/prompt`` (the pre-fix behaviour) dropped cross-topic
        cascades on the floor while the log claimed delivery.
        """
        if target_topic == "/openral/prompt":
            return self._prompt_pub
        pub = self._emit_prompt_pubs.get(target_topic)
        if pub is None:
            pub = self.create_publisher(IDLPromptStamped, target_topic, _QOS_PROMPT)
            self._emit_prompt_pubs[target_topic] = pub
        return pub

    def _dispatch_emit_prompt(
        self,
        call: EmitPromptTool,
        *,
        traceparent: str | None,
    ) -> None:
        """Publish a :class:`PromptStamped` on ``call.target_topic``.

        The active OTel traceparent (captured by
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
        try:
            publisher = self._emit_prompt_publisher(call.target_topic)
        except Exception as exc:  # reason: an invalid LLM-supplied topic must not kill the tick
            self.get_logger().warning(
                f"dispatch: emit_prompt target_topic={call.target_topic!r} rejected "
                f"({type(exc).__name__}: {exc}); dropping the prompt",
            )
            return
        publisher.publish(msg)
        self.get_logger().info(
            f"dispatch: emit_prompt → {call.target_topic} text={call.text!r}",
        )

    def _dispatch_spatial_query(
        self,
        call: RecallObjectTool | ResolvePlaceTool,
        *,
        traceparent: str | None,
    ) -> None:
        """Run a read-only spatial-memory query and re-prompt with the result.

        The query runs against the injected persistent-spatial-memory ``SpatialMemory`` backend and
        the rendered result is republished as a ``PromptStamped`` with frame_id
        ``"spatial_memory"`` (so ``_on_prompt`` consumes it rather than filtering
        it as a reasoner self-emit), feeding the answer into the next tick — the
        prompt cascade. Read-only: no actuation, no ``FailureTrigger``.

        Active object search §3 bound: consecutive queries are counted against a
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
        # Occupancy-grid-refined approach phase — when a slam map is online, every recall_object
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

        # locate_in_view / on-demand detectors — a recall_object MISS escalates to a live
        # locate_in_view (open-vocab, same query) BEFORE the search budget runs out and we hand off.
        # The on-demand detector grounds objects the spatial map never ingested, and matches the
        # goal term verbatim even when the stored label differs (e.g. recall "baguette" vs ingested
        # "bread"). This is policy — it does not depend on the LLM choosing locate_in_view. One
        # escalation per query term per search streak so a repeated miss can't spam the detector; if
        # locate also misses, the normal budget/handoff path resumes.
        #
        # ``resolve_place`` escalates on the same policy — see :func:`_search_term`
        # for why it used to be excluded and what that cost.
        search_term = _search_term(call)
        if (
            not outcome.found
            and self._detector_available
            and search_term not in self._locate_escalated
        ):
            self._locate_escalated.add(search_term)
            self.get_logger().info(
                f"dispatch: {call.tool} miss for {search_term!r} → escalating to "
                "locate_in_view (live detector) before handoff",
            )
            self._dispatch_locate_in_view(
                LocateInViewTool(query=search_term, detector=self._default_on_demand_detector),
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

    def _reset_task_locate_budget(self) -> None:
        """Reset the per-task locate budget (VLM-adjudicated completion amendment).

        Called when the active task makes real progress (an ``execute_rskill``
        dispatch) so locate cycles only count toward abandonment while the task
        has produced no skill dispatch.
        """
        self._task_locate_budget.reset()

    def _charge_task_locate_budget(
        self, call: LocateInViewTool, *, traceparent: str | None
    ) -> bool:
        """Charge a locate cycle against the active task; abandon it if exhausted.

        Returns ``True`` when the active subtask was abandoned on the budget — the
        caller must then NOT dispatch the locate. Without an active mission task
        there is no per-task budget (a standalone ``locate_in_view`` is unbounded
        here; the :class:`SearchProgress` miss budget still applies in the
        response handler), so this is a no-op returning ``False``.

        On exhaustion: append the displayed reason to ``## EXECUTION`` (so the
        *why* is explicit), abandon the active task via the mission ladder (the
        reason becomes its ``✗`` ledger verdict), reset the budget + per-kind
        streak, and force a tick so the next pick proceeds on the new active task.
        """
        mission = self._renderer.mission
        if mission is None:
            return False
        active = mission.active()
        if active is None:
            return False
        exhausted = self._task_locate_budget.charge(active.task_id)
        self._persist_ladder_state()
        if not exhausted:
            return False
        # VLM-adjudicated completion amendment — a COLLECTIVE goal locating to confirm objects is on
        # the path to `decompose_mission`, not stuck. On a budget-hit, nudge it to
        # decompose NOW (objects are confirmed in `located` by this point) and reset
        # the budget, rather than abandoning the whole mission mid-grounding. Bounded
        # by `_max_collective_decompose_nudges`: if it still won't decompose after
        # the cap, fall through to abandon (genuinely ungroundable).
        if is_collective_target(active.text):
            nudges = self._collective_decompose_nudges.get(active.task_id, 0)
            if nudges < self._max_collective_decompose_nudges:
                self._collective_decompose_nudges[active.task_id] = nudges + 1
                self._reset_task_locate_budget()
                self._persist_ladder_state()
                self.get_logger().info(
                    f"mission: locate budget hit on collective task {active.task_id} — "
                    f"nudging decompose ({nudges + 1}/{self._max_collective_decompose_nudges}) "
                    "instead of abandoning; resetting locate budget",
                )
                # The invite prompt forces its own tick when it arrives via
                # _on_prompt — forcing one HERE raced the DDS delivery and ran
                # the LLM against a context that did not contain the invite yet.
                self._emit_enumeration_invite(active, traceparent=traceparent)
                return True  # skip this locate; the next tick should decompose
            self.get_logger().warning(
                f"mission: collective task {active.task_id} could not be decomposed after "
                f"{nudges} nudge(s) — abandoning",
            )
        reason = self._task_locate_budget.reason(call.query)
        self.get_logger().warning(
            f"mission: task {active.task_id} abandoned on locate budget — {reason}",
        )
        # Surface the reason in ## EXECUTION (the ledger carries it as the task's
        # ✗ verdict; this makes the *why* explicit for the next reasoner pick).
        self._renderer.append_execution(
            ExecutionEventRecord(
                rskill_id="",
                outcome="failed",
                summary=reason,
                reflection=(
                    "repeated locate attempts confirmed nothing actionable — do NOT keep "
                    "locating this object; move on to the next mission object."
                ),
                stamp_ns=self.get_clock().now().nanoseconds,
            )
        )
        nxt = self._renderer.advance_mission(done=False, verdict=reason)
        self._reset_task_locate_budget()
        self._persist_ladder_state()
        if self._core is not None:
            self._core.reset_kind_streak()
        if nxt is not None:
            self.get_logger().info(
                f"mission: locate-budget abandon ✗ → advancing to {nxt.task_id}={nxt.text[:60]!r}",
            )
        else:
            self._emit_mission_complete(mission, traceparent=traceparent)
        self._on_tick(force=True, tier="C")
        return True

    def _dispatch_locate_in_view(
        self,
        call: LocateInViewTool,
        *,
        traceparent: str | None,
    ) -> None:
        """Ask a live VLM detector if an object is in view; re-prompt with the answer.

        The complement to :meth:`_dispatch_spatial_query` (remembered objects): this
        calls the detector node's ``/openral/perception/locate_in_view`` service to
        look at the CURRENT frame now. The call is async (``call_async`` +
        done-callback) so the ~1-2 s VLM inference never blocks the reasoner's
        executor; the rendered answer is republished as a ``PromptStamped`` with
        frame_id ``"detector"`` (consumed by ``_on_prompt``, feeding the next tick —
        the prompt cascade). Read-only: no actuation, no ``FailureTrigger``.

        VLM-adjudicated completion amendment — before dispatching, charge this cycle against the
        per-task locate budget (:class:`TaskLocateBudget`). If the active mission
        task has now spent its locate budget without an ``execute_rskill``
        dispatch, the subtask is abandoned (with a displayed reason) instead of
        locating again — the live locate-loop persists otherwise because
        ``locate_in_view`` keeps HITTING and never consumes the miss budget.
        """
        if self._charge_task_locate_budget(call, traceparent=traceparent):
            return  # active subtask abandoned on the locate budget; do not locate
        try:
            from openral_msgs.srv import LocateInView
        except ImportError:
            self.get_logger().warning(
                "dispatch: locate_in_view — openral_msgs/srv/LocateInView not built; skipping",
            )
            return
        # On-demand detectors as prompt-able reasoner tools — route to the chosen on-demand
        # locator's namespaced service; empty ``detector`` falls back to the deployment default (or
        # the legacy single-detector service). One cached client per resolved service name.
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
        """Render a ``LocateInView`` response as a re-prompt (prompt cascade)."""
        try:
            resp = future.result()
        except Exception as exc:  # best-effort; a failed lookup must not kill the tick
            self.get_logger().warning(f"dispatch: locate_in_view response failed: {exc}")
            return
        assert self._prompt_pub is not None
        cam = resp.camera or call.camera or "default"
        # A live locate counts as one spatial-search step (active object search §3): a miss
        # consumes budget so a repeated "not visible" terminates in handoff
        # instead of looping; a hit ends the search streak so the next find
        # starts fresh.
        frame_id = "detector"  # consumed by _on_prompt → feeds the next tick
        if resp.found:
            self._spatial_search.reset()
            self._locate_escalated.clear()
            # Detection-time object identity + camera-space enumeration — fold the open-vocab hit
            # into the sticky ``located`` line so the goal noun (which the fixed-vocab continuous
            # in_view mislabels) survives the next clobber and the LLM grounds / decomposes against
            # it instead of re-locating it every tick (the deploy locate-loop).
            try:
                self._renderer.note_located(ObjectsMetadata.model_validate_json(resp.metadata_json))
            except ValidationError as exc:
                self.get_logger().debug(f"located: dropping malformed locate metadata: {exc!s}")
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
        """Ask a scene VLM an open-ended question; re-prompt with the answer.

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
        """Render a ``QueryScene`` response as a re-prompt (prompt cascade)."""
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
        """Ask the reward monitor for a windowed progress/success assessment.

        Calls ``/openral/perception/query_task_progress`` (served by the
        reward_monitor_node, backed by the Robometer NF4 scorer). Async
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
        """Render a ``QueryTaskProgress`` response as a re-prompt (cascade).

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
            # VLM-adjudicated completion amendment — surface BOTH heads in the persistent `##
            # REWARD` context section (the re-prompt above is a one-shot; this keeps the latest
            # assessment visible to every subsequent tick, labelled).
            self._renderer.set_reward_state(
                RewardStateRecord(
                    progress=float(resp.progress_now),
                    success=float(resp.success_now),
                    progress_trend=float(resp.progress_trend),
                    success_trend=float(resp.success_trend),
                    task=call.task,
                    stamp_ns=self.get_clock().now().nanoseconds,
                )
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

    # ── §2 — automatic reward-gated task verification ──────────────

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
        # VLM-adjudicated completion amendment — score the whole attempt (start→now), not a trailing
        # slice. Request the reward model's full buffer span (``frame_window_s``)
        # when a contract is wired; the monitor clamps to its retained horizon.
        req.window_s = (
            self._reward_contract.frame_window_s
            if self._reward_contract is not None
            else _MISSION_VERIFY_WINDOW_S
        )
        req.task = active.text
        future = client.call_async(req)
        # Carry BOTH the task id and text into the response guard: two tasks
        # frequently share text after a decompose ("pick up the milk…" twice),
        # so a text-only staleness check could apply an old verdict to the
        # wrong task instance.
        future.add_done_callback(
            lambda fut: self._on_mission_verify_response(
                active.task_id, active.text, fut, traceparent=traceparent
            ),
        )
        self.get_logger().info(
            f"mission verify: querying reward for active task {active.text[:60]!r} "
            f"(attempt {active.attempts})",
        )

    def _on_mission_verify_response(
        self, task_id: str, task_text: str, future: Any, *, traceparent: str | None
    ) -> None:
        """Apply the reward gate (§2): complete / abandon / retry.

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
        if active is None or active.task_id != task_id or active.text != task_text:
            return  # the mission advanced or changed under us; stale verdict
        # VLM-adjudicated completion §1/§5 — band edges from the active reward model's calibration
        # (or the system fallback when none is wired).
        success_threshold, check_floor = self._band_edges()
        # VLM-adjudicated completion amendment — gate the band on the PROGRESS head (task closeness,
        # reaches ~0.80–0.86 on a real success and separates well); the bars
        # (0.8/0.5) were calibrated against progress, not the compressed success
        # head (~0.56–0.79 even on a genuine success). ``success_now`` is threaded
        # as a secondary corroborating signal surfaced in the verdict + the
        # vlm_check tier (it never overrides the progress band).
        progress_now = float(resp.progress_now)
        success_now = float(resp.success_now)
        if resp.ok:
            # VLM-adjudicated completion amendment — keep both heads visible in `## REWARD` for the
            # next tick's persist-vs-replan decision (the verify gate is internal).
            self._renderer.set_reward_state(
                RewardStateRecord(
                    progress=progress_now,
                    success=success_now,
                    progress_trend=float(resp.progress_trend),
                    success_trend=float(resp.success_trend),
                    task=active.text,
                    stamp_ns=self.get_clock().now().nanoseconds,
                )
            )
        action, verdict = evaluate_task_verdict(
            ok=bool(resp.ok),
            progress_now=progress_now,
            success_now=success_now,
            success_threshold=success_threshold,
            check_floor=check_floor,
            attempts=active.attempts,
        )
        if action == "vlm_check":
            # VLM-adjudicated completion §5 — ambiguous reward band: ask the VLM whether the task is
            # visually complete. True → advance as if "complete"; False/None → degrade
            # to the ladder (same code path as action == "retry"). Never false-complete:
            # None (no frame / no client) is treated as "not done". The VLM is the
            # primary adjudicator here; the success head (``success_now``) is the
            # secondary corroborating cue already folded into ``verdict``.
            # The describe_image round-trip runs off-executor (#21); the
            # continuation re-checks staleness before acting.
            self._adjudicate_completion_async(
                active.text,
                lambda verdict_vlm: self._on_vlm_completion_verdict(
                    task_id=task_id,
                    task_text=task_text,
                    verdict=verdict,
                    verdict_vlm=verdict_vlm,
                    progress_now=progress_now,
                    success_now=success_now,
                    success_threshold=success_threshold,
                    check_floor=check_floor,
                    traceparent=traceparent,
                ),
            )
            return
        self._apply_mission_verdict(
            action,
            verdict,
            mission=mission,
            active=active,
            progress_now=progress_now,
            success_now=success_now,
            traceparent=traceparent,
        )

    def _on_vlm_completion_verdict(
        self,
        *,
        task_id: str,
        task_text: str,
        verdict: str,
        verdict_vlm: bool | None,
        progress_now: float,
        success_now: float,
        success_threshold: float,
        check_floor: float,
        traceparent: str | None,
    ) -> None:
        """Executor-thread continuation of the ``vlm_check`` branch (#21).

        Re-fetches the active task (the mission may have advanced while the
        VLM call was in flight — same stale-verdict guard as the verify
        response itself), then applies the pre-async semantics: True →
        complete; False/None → degrade to the attempts ladder.
        """
        mission = self._renderer.mission
        if mission is None:
            return
        active = mission.active()
        if active is None or active.task_id != task_id or active.text != task_text:
            return  # the mission advanced or changed under us; stale verdict
        if verdict_vlm is True:
            self.get_logger().info(
                "mission verify: VLM confirmed complete "
                f"(progress={progress_now:.2f}, success={success_now:.2f})"
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
        # ladder (abandon once attempts >= max). Without this an ambiguous-band
        # task retries forever — never abandons, never hands off (CLAUDE.md §3
        # bounded ladder). ``attempts`` is monotonic thanks to the subdivide
        # guard, so this terminates.
        action, verdict = evaluate_task_verdict(
            ok=False,
            progress_now=progress_now,
            success_now=success_now,
            success_threshold=success_threshold,
            check_floor=check_floor,
            attempts=active.attempts,
        )
        self._apply_mission_verdict(
            action,
            verdict,
            mission=mission,
            active=active,
            progress_now=progress_now,
            success_now=success_now,
            traceparent=traceparent,
        )

    def _apply_mission_verdict(
        self,
        action: str,
        verdict: str,
        *,
        mission: MissionState,
        active: TaskState,
        progress_now: float,
        success_now: float,
        traceparent: str | None,
    ) -> None:
        """Apply a resolved verify verdict: retry / complete / abandon (±subdivision).

        Shared tail of :meth:`_on_mission_verify_response` and its async VLM
        continuation :meth:`_on_vlm_completion_verdict`.
        """
        if action == "retry":
            self.get_logger().info(f"mission verify: {verdict} — retrying active task")
            # VLM-adjudicated completion — surface the reward-plateau FAILURE to the LLM. The reward
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
                        f"reward says NOT done: progress={progress_now:.2f} below the "
                        f"progress bar (success={success_now:.2f}, attempt {active.attempts}); "
                        "the policy executed without a fault but did not accomplish the task"
                    ),
                    reflection=reflect_on_reward_plateau(progress_now),
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
            # No inline forced tick: the invite's own arrival forces one with
            # the invite actually present in context (see _on_prompt).
            self._emit_subdivision_invite(active, verdict, traceparent=traceparent)
            self._persist_ladder_state()
            return
        # action == "abandon" (no subdivision offer): advance with done=False.
        nxt = self._renderer.advance_mission(done=False, verdict=verdict)
        self._persist_ladder_state()
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
        """Emit an honest operator-facing mission summary (self-prompt, §2).

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

        Two modes by ``target_task_id`` (reward-gated task verification /
        reasoner playbooks ``decompose-mission``):

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
            self._persist_ladder_state()
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
            self._persist_ladder_state()
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
        """Apply one explicit MEMORY.md edit, persist it, and confirm (§3 / Phase 4c).

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
        # Reasoner playbooks, Phase 5 — consolidate: merge any exact-duplicate facts the write
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
        """Recall archived memory entries by keyword and re-prompt (§3 / Phase 4c).

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

    def _manifest_for_rskill(self, rskill_id: str) -> RSkillManifest | None:
        """The :class:`RSkillManifest` for ``rskill_id`` (cached), or None.

        VLA/reward VRAM-fit pairing — the pre-dispatch gate needs the VLA's
        ``min_vram_gb``. The cache is primed by :meth:`_seed_palette` from the
        very manifests the LLM is offered, so **every** palette skill resolves;
        the ``rSkill.list_installed()`` fallback below covers only ids that
        reached the graph some other way (a Hub-installed skill not on any
        ``rskill_search_paths`` root). ``rSkill`` pulls torch, so it stays lazy.

        Returning ``None`` disables the VRAM gate for that dispatch, which is
        why the priming matters: the registry alone left the gate dead for
        every VLA on a search-path-seeded palette.
        """
        if rskill_id in self._manifests_by_id:
            return self._manifests_by_id[rskill_id]
        from openral_rskill.loader import rSkill  # heavy (torch) — lazy

        for entry in rSkill.list_installed():
            if entry.repo_id != rskill_id:
                continue
            try:
                manifest = RSkillManifest.from_yaml(entry.manifest_path)
            except (OSError, ValueError) as exc:
                self.get_logger().warning(f"manifest load failed for {rskill_id!r}: {exc}")
                return None
            self._manifests_by_id[rskill_id] = manifest
            return manifest
        return None

    def _refuse_unfittable_vla(
        self,
        call: ExecuteRskillTool,
        *,
        traceparent: str | None,
    ) -> bool:
        """VRAM pre-dispatch gate: refuse a VLA that cannot fit the GPU.

        On a miss we refuse + publish a ``FailureTrigger`` (the reasoner sees
        it and bounds retries → handoff) rather than OOM mid-run or run the
        VLA blind. Returns ``True`` when the dispatch was refused (caller must
        not send the goal). Two tiers, both refusing with the same
        ``vram_insufficient`` evidence:

        1. **Live free-VRAM probe** (needs only the VLA's declared
           ``active_min_vram_gb`` + a readable ``nvidia-smi``): the static pair
           check budgets against the card's TOTAL, which is blind to what
           *other* processes hold right now — observed live (2026-07-20): an
           external vLLM server held 4.7 GB of an 8 GB card, molmoact2
           (declared 4.0 GB) passed the static gates and burned ~30 s in a
           CUDA OOM abort. Skipped when the dispatched skill is already
           resident in the runner (``_resident_vla_id`` — its own residency
           is why free is low; re-dispatch needs ~0 new VRAM), when
           ``vram_lifecycle_peers`` are configured (their eviction frees VRAM
           the pre-eviction probe cannot see — a false refusal would block a
           dispatch eviction would have enabled), when the manifest declares
           no size, or when free VRAM is unreadable.
        2. **Static VLA+reward pair fit** (unchanged): when a reward model is
           active and the GPU total is known, the pair must fit the card.

        Skipped entirely — dispatch proceeds — when the dispatched skill is
        not a VLA / not installed.
        """
        vla = self._manifest_for_rskill(call.rskill_id)
        if vla is None or vla.kind != "vla":
            return False
        vla_min_gb = vla.active_min_vram_gb()
        if call.rskill_id == self._resident_vla_id:
            # Re-dispatch of the skill the runner already holds warm: its own
            # residency is why free VRAM is low, and re-running it needs ~0
            # new VRAM — the live probe would falsely refuse every large-VLA
            # dispatch after the first. Tier-2 (static pair fit) still applies.
            self.get_logger().info(
                f"dispatch: skill {call.rskill_id!r} already resident in the runner; "
                "skipping the live free-VRAM probe",
            )
        elif vla_min_gb is not None and vla_min_gb > 0.0 and not self._vram_lifecycle_peers:
            free_gb = _detect_gpu_free_vram_gb()
            if 0.0 < free_gb < vla_min_gb:
                detail = (
                    f"VLA {call.rskill_id!r} declares {vla_min_gb:.1f} GB at its active "
                    f"quantization but only {free_gb:.1f} GB of GPU VRAM is free right now "
                    f"(total {self._gpu_total_vram_gb:.1f} GB) — another process is likely "
                    "holding the difference. Refusing before dispatch instead of OOMing "
                    "mid-load; free VRAM or pick a smaller skill."
                )
                self.get_logger().error(f"dispatch: refusing execute_rskill — {detail}")
                self._publish_skill_failure(
                    kind=_KIND_CONTROLLER,
                    rskill_id=call.rskill_id,
                    evidence=ControllerEvidence(
                        controller_name=call.rskill_id,
                        state="vram_insufficient",
                        detail=detail[:480],
                    ),
                    traceparent=traceparent,
                )
                return True
        if self._reward_manifest is None or self._gpu_total_vram_gb <= 0.0:
            return False
        if vla.reward_rskill_name and vla.reward_rskill_name != self._reward_manifest.name:
            self.get_logger().warning(
                f"reward pairing mismatch: VLA {call.rskill_id!r} names "
                f"{vla.reward_rskill_name!r} but the active reward model is "
                f"{self._reward_manifest.name!r}",
            )
        try:
            assert_vla_reward_fits(vla, self._reward_manifest, self._gpu_total_vram_gb)
        except (ROSGPUMemoryError, ROSConfigError) as exc:
            self.get_logger().error(
                f"dispatch: refusing execute_rskill {call.rskill_id!r} — VLA + reward "
                f"model cannot co-reside on GPU: {exc}",
            )
            self._publish_skill_failure(
                kind=_KIND_CONTROLLER,
                rskill_id=call.rskill_id,
                evidence=ControllerEvidence(
                    controller_name=call.rskill_id,
                    state="vram_insufficient",
                    detail=str(exc)[:480],
                ),
                traceparent=traceparent,
            )
            return True
        return False

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
            # No inline forced tick: the invite's own arrival forces one with
            # the invite actually present in context (see _on_prompt).
            self._emit_enumeration_invite(active, traceparent=traceparent)
            return
        # Busy gate: the runner serves ONE goal at a time, and a forced tick
        # mid-execution (failure event, operator prompt) used to double-
        # dispatch blind — silently overwriting the in-flight goal handle and
        # orphaning the VRAM peers frozen for it. Refuse, tell the LLM why
        # (the in_flight context line plus this feedback), and let it wait or
        # poll instead.
        if self._rskill_inflight:
            inflight = self._renderer.inflight_skill or "(dispatching)"
            self.get_logger().info(
                f"dispatch: refusing execute_rskill {call.rskill_id!r} — "
                f"goal {inflight!r} is already in flight",
            )
            self._renderer.append_execution(
                ExecutionEventRecord(
                    rskill_id=call.rskill_id,
                    outcome="failed",
                    summary=f"refused: skill {inflight} is already in flight",
                    reflection=(
                        "a skill goal is already running — do NOT dispatch another; "
                        "wait for its result or poll query_task_progress."
                    ),
                    stamp_ns=self.get_clock().now().nanoseconds,
                )
            )
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

        # VLA/reward VRAM-fit pairing — a VLA runs only with its reward model resident alongside it.
        # Refuse (and notify) before the goal is sent if the pair can't co-reside
        # on the GPU, rather than running the VLA with no progress signal / OOMing
        # mid-run. No attempt is recorded — a refused dispatch is not a try.
        if self._refuse_unfittable_vla(call, traceparent=traceparent):
            return

        # Reward-gated task verification §2 — count this dispatch as an attempt at the active
        # mission task so the reward gate can bound retries (abandon + hand off after the cap).
        # execute_rskill is the actuation tool; locate/query are separate tools, so a dispatch here
        # is a manipulation attempt at the active task.
        mission = self._renderer.mission
        if mission is not None and mission.active() is not None:
            mission.record_attempt(rskill_id=call.rskill_id, trace_id=traceparent)
            # VLM-adjudicated completion amendment — an execute dispatch is real progress on the
            # active task, so the per-task locate budget resets (locate cycles only
            # count toward abandonment while no skill has been dispatched).
            self._reset_task_locate_budget()
            self._persist_ladder_state()

        # Latch the busy gate for the whole send→terminal window (the
        # accepted-goal handle alone leaves the send→accept gap unguarded).
        # Cleared on rejection / send failure / terminal result. The renderer
        # mirrors the phase ("dispatching" now, "running" on accept) so the
        # LLM can tell a cold-loading goal from a stalled one instead of
        # escalating to the operator mid-load.
        self._rskill_inflight = True
        self._dispatch_generation += 1
        generation = self._dispatch_generation
        self._active_dispatch_generation = generation
        self._renderer.set_inflight_skill(
            call.rskill_id,
            stamp_ns=self.get_clock().now().nanoseconds,
            state="dispatching",
        )
        # Arm the dispatch-phase watchdog with the latch: if neither the peer
        # eviction nor the goal response resolves within the ceiling, the
        # runner/peer died post-probe and nothing else would ever release the
        # latch (the deadline timer only arms on ACCEPT).
        self._dispatch_send_future = None
        watchdog_s = self.get_parameter("dispatch_watchdog_s").get_parameter_value().double_value
        if watchdog_s > 0:
            self._dispatch_watchdog = self.create_timer(
                watchdog_s,
                lambda: self._on_dispatch_watchdog(call, generation, traceparent),
            )

        # Single-resident-skill VRAM eviction — free GPU lifecycle peers (the object detector)
        # before the policy loads, then reactivate when the skill finishes. Sequenced so the peer's
        # VRAM is released before the goal reaches the runner; an 8 GB card OOMs if the ~1.3 GB
        # detector co-resides with the VLA.
        if self._vram_lifecycle_peers:
            self._free_vram_peers_then_send(
                call,
                list(self._vram_lifecycle_peers),
                generation,
                traceparent,
            )
        else:
            self._send_execute_rskill_goal(call, generation, traceparent)

    def _set_reward_task(self, task: str) -> None:
        """Publish the instruction the reward monitor should score (2026-06-29).

        ``/openral/reward/active_task`` carries the EXACT prompt the VLA is running
        (the active subtask) on dispatch, and empty on result. The reward leg only
        scores while non-empty AND scores that instruction — so a single-object pick
        is judged as the task the policy is actually doing, not the collective goal.
        No-op when std_msgs is unavailable.
        """
        if self._reward_active_pub is None:
            return
        msg = IDLString()
        msg.data = task
        self._reward_active_pub.publish(msg)

    def _send_execute_rskill_goal(
        self,
        call: ExecuteRskillTool,
        generation: int,
        traceparent: str | None,
    ) -> None:
        """Build and send the ``ExecuteRskill.Goal`` (the VLA dispatch itself)."""
        if generation != self._active_dispatch_generation:
            return
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
        # 2026-06-29 — open the reward-scoring window with the SAME instruction the
        # VLA gets, so the reward model scores the task the policy is actually doing.
        self._set_reward_task(goal.prompt)
        # The reasoner does not yet construct a SkillPrompt payload —
        # F4 stays on the text path; the structured-prompt route is
        # wired in a later follow-up.
        goal.prompt_metadata_json = ""
        # rSkill structured goal parameters — forward the LLM's per-skill structured params, if
        # any. Wrapped-ROS adapters merge ``goal_params_json`` over
        # their manifest's ``default_goal_json`` at configure-time.
        goal.goal_params_json = call.goal_params_json
        # VLM-adjudicated completion §2/§3 — the goal's ``deadline_s`` slot now carries the resolved
        # patience ceiling (LLM override > reward-model default > legacy deadline_s);
        # it is the runner's backstop, not the usual stop (the reward-watcher is).
        patience_s = self._effective_patience_s(call)
        goal.deadline_s = patience_s
        sent_at = time.monotonic()
        send_future = self._execute_rskill_client.send_goal_async(
            goal,
            feedback_callback=lambda fb: self._on_execute_rskill_feedback(call.rskill_id, fb),
        )
        # Visible to the dispatch watchdog: .done() distinguishes "response
        # arrived but its callback is starved" from a genuinely dead server.
        self._dispatch_send_future = send_future
        send_future.add_done_callback(
            lambda fut: self._on_execute_rskill_goal_response(
                call,
                generation,
                sent_at,
                fut,
                traceparent,
            ),
        )
        self.get_logger().info(
            f"dispatch: execute_rskill rskill_id={call.rskill_id!r} prompt={goal.prompt!r} "
            f"patience_s={patience_s:.0f} (backstop; reward-watcher is the usual stop)",
        )

    def _free_vram_peers_then_send(
        self,
        call: ExecuteRskillTool,
        peers: list[str],
        generation: int,
        traceparent: str | None,
    ) -> None:
        """Deactivate GPU peers, then send the goal once they have all released.

        Single-resident-skill VRAM eviction. Each peer's ``change_state`` is
        async; the goal is sent only after every in-flight deactivation has
        returned, so the freed VRAM is
        available before the runner loads the policy. Peers whose service isn't
        on the graph are skipped (best-effort — the dispatch still proceeds).
        The deactivated subset is recorded for reactivation on the skill result.
        """
        if self._deactivated_vram_peers:
            # Defensive: never clobber a peer set another dispatch deactivated
            # and has not yet reactivated (the busy gate should prevent this;
            # losing the record would leave perception peers off for good).
            self.get_logger().warning(
                f"vram: {len(self._deactivated_vram_peers)} peer(s) still deactivated "
                "from a prior dispatch — keeping them tracked",
            )
        else:
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
            self._send_execute_rskill_goal(call, generation, traceparent)
            return
        remaining = {"n": len(futures)}

        def _after_one(peer: str, fut: Any) -> None:
            try:
                ok = bool(fut.result().success)
            except Exception as exc:  # reason: a failed change_state must not strand dispatch
                ok = False
                self.get_logger().warning(f"vram: peer {peer!r} deactivate errored: {exc}")
            if generation != self._active_dispatch_generation:
                if ok:
                    reactivate = self._change_state_async(peer, "activate")
                    if reactivate is not None:
                        reactivate.add_done_callback(
                            lambda done, p=peer: self._on_reactivate_result(p, done)
                        )
                return
            if ok:
                self._deactivated_vram_peers.append(peer)
            else:
                self.get_logger().warning(
                    f"vram: peer {peer!r} did not deactivate cleanly; proceeding",
                )
            remaining["n"] -= 1
            if remaining["n"] == 0:
                self._send_execute_rskill_goal(call, generation, traceparent)

        for peer, future in futures:
            future.add_done_callback(lambda fut, p=peer: _after_one(p, fut))

    def _reactivate_vram_peers(self) -> None:
        """Reactivate the GPU peers deactivated for a now-finished skill.

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
        """Log the reactivation ``change_state`` outcome (best-effort)."""
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
        """Forward action feedback to the operator log, throttled to ~1 Hz.

        Feedback is NOT rare in practice — a VLA goal streams one message per
        action chunk (600+ per goal observed live, 2026-07-20), and logging
        each at WARNING drowned the operator log. Keep the operator-visible
        warning but emit at most one per :data:`_FEEDBACK_LOG_PERIOD_S`
        (suppressed messages go to DEBUG so a trace-level investigation still
        has every chunk).
        """
        fb = feedback_msg.feedback
        line = (
            f"execute_rskill feedback rskill_id={rskill_id!r} state={fb.state!r} "
            f"progress={fb.progress:.2f} chunk={fb.chunk_index}/{fb.chunks_total}"
        )
        now_s = time.monotonic()
        if now_s - self._last_feedback_warn_s >= _FEEDBACK_LOG_PERIOD_S:
            self._last_feedback_warn_s = now_s
            self.get_logger().warning(line)
        else:
            self.get_logger().debug(line)

    def _cancel_dispatch_watchdog(self, generation: int | None = None) -> None:
        """Disarm the dispatch-phase watchdog (goal response arrived / cleanup)."""
        if generation is not None and generation != self._active_dispatch_generation:
            return
        timer = self._dispatch_watchdog
        self._dispatch_watchdog = None
        self._dispatch_send_future = None
        if timer is not None:
            timer.cancel()
            self.destroy_timer(timer)

    def _on_dispatch_watchdog(
        self,
        call: ExecuteRskillTool,
        generation: int,
        traceparent: str | None,
    ) -> None:
        """Dispatch-phase watchdog expiry — release a wedged busy latch.

        Fires only when the send→goal-response window never closed: a VRAM
        peer's ``change_state`` future or ``send_goal_async``'s goal-response
        future that never resolves (server died AFTER the readiness probe —
        rclpy futures have no timeout of their own) would otherwise leave
        ``_rskill_inflight`` latched forever, wedging every future dispatch
        while the context tells the LLM not to dispatch. A completed-but-
        starved response future (issue #21) is NOT a wedge — its callback is
        already queued on this same executor and runs next.
        """
        if generation != self._active_dispatch_generation:
            return
        send_future = self._dispatch_send_future
        self._cancel_dispatch_watchdog(generation)
        if not self._rskill_inflight or self._active_rskill_goal is not None:
            return  # dispatch resolved normally; stale timer
        if send_future is not None and send_future.done():
            # Response arrived; the goal-response callback is queued behind
            # this one on the executor — let it handle accept/reject.
            return
        phase = "goal response" if send_future is not None else "VRAM peer eviction"
        detail = (
            f"dispatch watchdog: no {phase} for execute_rskill {call.rskill_id!r} "
            "within the ceiling — the runner or a VRAM peer likely died after "
            "the readiness probe. Releasing the busy latch."
        )
        self.get_logger().error(detail)
        self._active_dispatch_generation = None
        self._rskill_inflight = False
        self._renderer.set_inflight_skill(None)
        self._reactivate_vram_peers()
        self._set_reward_task("")
        self._publish_skill_failure(
            kind=_KIND_CONTROLLER,
            rskill_id=call.rskill_id,
            evidence=ControllerEvidence(
                controller_name=call.rskill_id,
                state="dispatch_timeout",
                detail=detail,
            ),
            traceparent=traceparent,
        )

    def _on_execute_rskill_goal_response(
        self,
        call: ExecuteRskillTool,
        generation: int,
        sent_at: float,
        future: Any,
        traceparent: str | None,
    ) -> None:
        """Goal-response done callback.

        On rejection emits a ``KIND_CONTROLLER`` FailureTrigger; on
        acceptance attaches a result-future callback and arms the
        deadline timer (if ``call.deadline_s > 0``).
        """
        if generation != self._active_dispatch_generation:
            try:
                stale_handle = future.result()
            except Exception:
                return
            if stale_handle.accepted:
                stale_handle.cancel_goal_async()
            return
        # The dispatch window is over regardless of outcome below.
        self._cancel_dispatch_watchdog(generation)
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
            # Single-resident-skill VRAM eviction — the goal never reached the runner; restore the
            # GPU peers we froze for it so perception resumes.
            self._active_dispatch_generation = None
            self._rskill_inflight = False
            self._renderer.set_inflight_skill(None)
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
            # Single-resident-skill VRAM eviction — goal rejected (skill won't run); restore the GPU
            # peers.
            self._active_dispatch_generation = None
            self._rskill_inflight = False
            self._renderer.set_inflight_skill(None)
            self._reactivate_vram_peers()
            return
        goal_id = bytes(goal_handle.goal_id.uuid)
        # The runner will load (or already holds) this policy; remember it so
        # the live free-VRAM probe never falsely refuses re-dispatching the
        # skill whose own residency is why free VRAM is low.
        self._resident_vla_id = call.rskill_id
        # VLM-adjudicated completion §2 — remember the in-flight goal so a reward-watcher wake can
        # cancel it. Cleared on the terminal result. The busy gate in
        # _dispatch_execute_rskill prevents a second dispatch overwriting it.
        self._active_rskill_goal = (goal_handle, call, traceparent)
        self._rskill_cancel_reason = None
        # Surface "a skill is running" to the LLM's context (the in_flight
        # line in ## EXECUTION) and keep the heartbeat live for mid-run
        # reward polling (ReasonerCore's heartbeat-idle gate reads it).
        self._renderer.set_inflight_skill(
            call.rskill_id, stamp_ns=self.get_clock().now().nanoseconds, state="running"
        )
        # VLM-adjudicated completion §2/§3 — arm the reasoner-side backstop at the resolved patience
        # (matches the goal's deadline_s sent to the runner). 0 → the runner owns
        # the ceiling (manifest latency budget); no reasoner-side timer.
        patience_s = self._effective_patience_s(call)
        if patience_s > 0:
            self._pending_skill_deadlines[goal_id] = self.create_timer(
                patience_s,
                lambda: self._on_execute_rskill_deadline(
                    call=call,
                    generation=generation,
                    sent_at=sent_at,
                    goal_handle=goal_handle,
                    traceparent=traceparent,
                ),
            )
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(
            lambda fut: self._on_execute_rskill_result(
                call,
                generation,
                goal_id,
                fut,
                traceparent,
            ),
        )

    def _on_execute_rskill_result(
        self,
        call: ExecuteRskillTool,
        generation: int,
        goal_id: bytes,
        future: Any,
        traceparent: str | None,
    ) -> None:
        """Result done callback. Cancels deadline timer; emits on abort."""
        if generation != self._active_dispatch_generation:
            return
        # Single-resident-skill VRAM eviction — the skill is terminal (success/abort/cancel/error),
        # so the policy's VRAM is released; restore the GPU peers (detector) we froze for it. Runs
        # before any early return below so it always fires.
        self._reactivate_vram_peers()
        # 2026-06-29 — close the reward-scoring window: no VLA is acting now.
        self._set_reward_task("")
        # VLM-adjudicated completion §2 — the goal is terminal: drop the in-flight handle and read
        # (then clear) the cancel reason so a reward-driven cancel verifies below
        # while an operator/estop cancel stays a no-op.
        self._active_dispatch_generation = None
        self._active_rskill_goal = None
        self._rskill_inflight = False
        self._renderer.set_inflight_skill(None)
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
            # Reasoner playbooks §2.2 — surface SUCCESS to the LLM (Inner Monologue). The
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
            # Reward-gated task verification §2 — runner "success" for a VLA means "ran to its
            # deadline without a controller fault", NOT "task accomplished". Verify the active
            # mission task against the reward signal before advancing.
            self._maybe_verify_active_mission_task(call, traceparent=traceparent)
            return
        # VLM-adjudicated completion §2 — a reasoner-driven cancel (status 5, reason "reward" from
        # the reward-watcher wake, or "patience" from the patience-ceiling
        # timer) is an intentional end-of-attempt, not a controller fault: the
        # policy DID run, so the attempt must be verified — the three-tier /
        # VLM gate completes or advances the ladder — and the KIND_CONTROLLER
        # failure path is skipped (no fault to report; the patience path
        # already emitted its KIND_TIMEOUT). Only an operator/e-stop cancel
        # (reason None) is not an attempt.
        if status == 5 and cancel_reason in ("reward", "patience"):
            self.get_logger().info(
                f"execute_rskill {cancel_reason}-cancelled rskill_id={call.rskill_id!r} "
                "— verifying",
            )
            self._maybe_verify_active_mission_task(call, traceparent=traceparent)
            return
        self.get_logger().warning(
            f"execute_rskill failed rskill_id={call.rskill_id!r} status={status} "
            f"reason={result.failure_reason!r}",
        )
        outcome_state = "aborted" if status == 6 else "canceled" if status == 5 else "failed"
        detail = result.failure_reason or f"GoalStatus={status}"
        # Reasoner playbooks §2.2/§2.3 — execution feedback + a Reflexion strategy hint so
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
        updated_palette = _palette_after_rskill_failure(self._palette, call.rskill_id, detail)
        if updated_palette != self._palette:
            self._palette = updated_palette
            self.get_logger().warning(
                f"execute_rskill removing unavailable rskill_id={call.rskill_id!r} "
                "from the palette until it is rebuilt",
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
        # Reward-gated task verification §2 — an aborted (terminal) episode still ran the policy, so
        # it is a real attempt at the active task; verify so a repeatedly-aborting task is bounded
        # by the attempt cap (abandon → hand off) rather than looping forever. Canceled (status 5)
        # is operator-driven, not an attempt.
        if status == 6:
            self._maybe_verify_active_mission_task(call, traceparent=traceparent)

    def _on_execute_rskill_deadline(
        self,
        *,
        call: ExecuteRskillTool,
        generation: int,
        sent_at: float,
        goal_handle: Any,
        traceparent: str | None,
    ) -> None:
        """Patience-ceiling backstop (§2): cancel goal + emit ``KIND_TIMEOUT``.

        Fires only when the reward-watcher did not stop the attempt first — the
        resolved patience ceiling elapsed. ``deadline_s`` in the log/evidence is
        that resolved patience (LLM override > reward default > legacy).
        """
        if generation != self._active_dispatch_generation:
            return
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
        # A patience-expired attempt is a REAL attempt: latch the cancel
        # reason so the canceled result runs the reward verify gate exactly
        # like a reward-watcher stop, instead of being misread as an
        # operator cancel and skipping verification.
        self._rskill_cancel_reason = "patience"
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
        # Mirror the failure onto the OTLP span path so the dashboard tallies it
        # on the "skill failures" counter + surfaces the state.
        # The ROS FailureTrigger bus is invisible to the dashboard (it ingests
        # OTLP, not ROS topics); this event is the only thing that reaches it.
        self._emit_skill_failure_event(kind=kind, rskill_id=rskill_id, evidence=evidence)

    def _emit_skill_failure_event(self, *, kind: int, rskill_id: str, evidence: Any) -> None:
        """Stamp an ``EVENT_SKILL_FAILURE`` span event for the live dashboard.

        Adds the event to the active ``reasoner.tick`` span when one is recording
        (the synchronous dispatch-gate paths — vram_insufficient, server
        unavailable); otherwise opens a transient ``reasoner.skill_failure`` span
        so the async action-callback paths (abort / timeout) are tallied too. The
        concrete state (``evidence.state`` when present, else a kind-derived name)
        rides on ``SKILL_FAILURE_STATE`` so the operator sees *why* a skill failed.
        """
        from openral_observability import semconv
        from opentelemetry import trace

        state = getattr(evidence, "state", None)
        if not state:
            state = _SKILL_FAILURE_KIND_NAMES.get(int(kind), "failed")
            # KIND_TIMEOUT carries TimeoutEvidence (no ``state`` field) — fold the
            # elapsed patience ceiling into the reason so the dashboard shows WHY
            # the skill was cut, not just the bare "timeout".
            deadline_s = getattr(evidence, "deadline_s", None)
            if deadline_s is not None:
                state = f"{state} after {float(deadline_s):.0f}s patience ceiling"
        attrs: dict[str, str] = {
            semconv.SKILL_FAILURE_STATE: str(state),
            semconv.REASONER_RSKILL_ID: rskill_id,
        }
        span = trace.get_current_span()
        if span.is_recording():
            span.add_event(semconv.EVENT_SKILL_FAILURE, attributes=attrs)
            return
        tracer = trace.get_tracer("openral.reasoner")
        with tracer.start_as_current_span("reasoner.skill_failure") as transient:
            transient.add_event(semconv.EVENT_SKILL_FAILURE, attributes=attrs)

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
            pass  # context already shut down by the SIGINT handler
        finally:
            node.destroy_node()
    finally:
        rclpy.try_shutdown()  # idempotent — no-op if context already shut down
    return 0


if __name__ == "__main__":
    sys.exit(main())
