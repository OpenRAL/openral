"""ADR-0018 F4 — :class:`ContextRenderer`.

Builds the structured **text** context the reasoner LLM consumes each
tick. Per ADR-0018 §4 "No pixels in v1": the context is a rolling text
digest of

* the latest :class:`~openral_core.WorldState` snapshot (joint state,
  EE pose, staleness, control mode);
* the recent :class:`FailureTrigger` events received per source bus
  (FIFO buffers, one per ``/openral/failure/<source>``);
* the recent ``/openral/perception/<kind>`` events;
* any pending operator prompts.

Output is deterministic, byte-stable for a given input, and bounded in
size — small enough for a 4k context window even when every buffer is
full.
"""

from __future__ import annotations

import dataclasses
import json
from collections import deque

from openral_core import (
    FailureEvidence,
    ObjectDetection2D,
    ObjectsMetadata,
    PerceptionEventMetadata,
    RobotDescription,
    WorldState,
)
from pydantic import TypeAdapter

from openral_reasoner.mission import MissionState, TaskState

__all__ = [
    "DEFAULT_BUFFER_SIZE",
    "DEFAULT_PROMPT_PRIORITY",
    "ContextRenderer",
    "ExecutionEventRecord",
    "FailureEventRecord",
    "PerceptionEventRecord",
    "PromptRecord",
    "RewardStateRecord",
    "reflect_on_failure",
    "reflect_on_invalid_plan",
    "reflect_on_retry_cap",
    "render_playbooks_block",
    "render_robot_self_model",
]

# Each rolling buffer is sized so an entire window fits in ~1 KB of
# text — the LLM context budget is dominated by the WorldState fields,
# not these buffers.
DEFAULT_BUFFER_SIZE: int = 8

# Default operator-prompt priority — matches the auto-cascade priority
# in ``openral_prompt_router.DEFAULT_SOURCES`` (ADR-0018 §3.F10). Human
# sources (CLI, dashboard) get 100; cascades stay at 10.
DEFAULT_PROMPT_PRIORITY: int = 10

_FAILURE_ADAPTER: TypeAdapter[FailureEvidence] = TypeAdapter(FailureEvidence)
_PERCEPTION_ADAPTER: TypeAdapter[PerceptionEventMetadata] = TypeAdapter(PerceptionEventMetadata)


@dataclasses.dataclass(frozen=True, slots=True)
class FailureEventRecord:
    """One entry in the reasoner's rolling failure buffer.

    Constructed by the reasoner_node on each
    :class:`openral_msgs/FailureTrigger` arrival.
    """

    source: str  # /openral/failure/<source>
    kind: int  # uint8 KIND_* from openral_msgs/FailureTrigger
    severity: int  # uint8 SEVERITY_*
    evidence_json: str
    rskill_id: str
    trace_id: str
    stamp_ns: int


@dataclasses.dataclass(frozen=True, slots=True)
class PerceptionEventRecord:
    """One entry in the reasoner's rolling perception event buffer."""

    kind: str  # /openral/perception/<kind>
    text: str  # PromptStamped.text
    metadata_json: str  # PromptStamped.metadata_json (Pydantic PerceptionEventMetadata)
    stamp_ns: int


@dataclasses.dataclass(frozen=True, slots=True)
class PromptRecord:
    """One entry in the reasoner's rolling operator-prompt buffer.

    Attributes:
        text: ``PromptStamped.text``.
        metadata_json: ``PromptStamped.metadata_json``. The F10
            ``prompt_router_node`` stamps a ``{"source": "...",
            "priority": <int>}`` field into this JSON; the reasoner's
            :meth:`ContextRenderer.append_prompt` reads ``priority`` to
            order the drain (human-source prompts override queued
            auto-prompts per ADR-0018 §3.F10).
        stamp_ns: Arrival timestamp in nanoseconds.
        priority: Drain priority (higher = drained first). Default
            ``10`` matches the auto-cascade priority documented on
            ``PromptRouterNode.DEFAULT_SOURCES``; the router fills
            ``100`` for human-source prompts. Tests construct
            :class:`PromptRecord` directly with the desired priority;
            production code paths leave the default and let
            ``append_prompt`` parse it from ``metadata_json``.
    """

    text: str
    metadata_json: str
    stamp_ns: int
    priority: int = DEFAULT_PROMPT_PRIORITY


@dataclasses.dataclass(frozen=True, slots=True)
class ExecutionEventRecord:
    """One entry in the reasoner's rolling **execution-feedback** buffer.

    ADR-0072 Decision 2.2 (Inner Monologue): a typed one-line outcome appended
    after every dispatched skill — on *success as well as failure*, so the LLM
    reasons on what actually happened (closed loop) instead of only seeing
    failures. On failure it also carries a :attr:`reflection` strategy hint
    (Decision 2.3, Reflexion).
    """

    rskill_id: str
    outcome: str  # "ok" | "failed"
    summary: str  # short NL outcome ("trace=abc12345", "object not in gripper")
    reflection: str | None  # Decision 2.3 strategy hint (failures only)
    stamp_ns: int


@dataclasses.dataclass(frozen=True, slots=True)
class RewardStateRecord:
    """Latest reward-model assessment surfaced to the LLM (ADR-0074 amendment).

    Robometer-4B emits **two distinct heads** with different meanings, and the
    LLM must use each for the right decision:

    * :attr:`progress` — task *closeness* (0=untouched, 1=done). This is the
        head the verdict gates on (it reaches ~0.80-0.86 on a genuine success);
        the LLM should weigh it for *persist-vs-replan*.
    * :attr:`success` — done-*confidence* (a separate probability that is
        empirically compressed, ~0.56-0.79 even on a real success); the LLM
        should weigh it for *done-ness*, not as the primary completion bar.

    Rendered as the ``## REWARD`` section so both heads are labelled and never
    blurred. Set by the node from each ``query_task_progress`` / mission-verify
    response; ``None`` omits the section.

    Attributes:
        progress: Latest progress (closeness) score in [0, 1].
        success: Latest success (done-confidence) score in [0, 1].
        progress_trend: Per-frame progress slope (+rising / -falling).
        success_trend: Per-frame success slope.
        task: The task text the assessment was scored against.
        stamp_ns: Arrival timestamp in nanoseconds.
    """

    progress: float
    success: float
    progress_trend: float
    success_trend: float
    task: str
    stamp_ns: int


def reflect_on_failure(outcome_state: str, detail: str) -> str:
    """One-line strategy hint from a terminal skill outcome (ADR-0072 §2.3).

    Reflexion-style: convert a raw failure into a *next-step* hint so the
    replanning ladder advances instead of blindly retrying. Deterministic — no
    LLM call.

    Args:
        outcome_state: The terminal state — ``"aborted"`` / ``"canceled"`` /
            ``"failed"`` / ``"error"``.
        detail: Free-text failure reason (matched for ``timeout`` / ``deadline``).

    Example:
        >>> "infeasible" in reflect_on_failure("aborted", "joint limit")
        True
    """
    state = outcome_state.lower()
    if "timeout" in detail.lower() or "deadline" in detail.lower():
        return (
            "the skill timed out — it may be stuck; try a shorter-horizon step "
            "or a different skill."
        )
    if state == "aborted":
        return (
            "the controller aborted mid-execution — the action is likely infeasible from "
            "here; reposition or substitute a different skill rather than retrying."
        )
    if state == "canceled":
        return (
            "the action was canceled — re-check the goal and preconditions before re-dispatching."
        )
    return (
        "the skill failed — don't repeat the same call; try a different skill "
        "or replan the subgoal."
    )


def reflect_on_reward_plateau(progress_now: float) -> str:
    """One-line strategy hint for a reward-plateau failure (ADR-0074).

    Distinct from :func:`reflect_on_failure`: there the *controller* faulted
    (timeout / abort) so "shorten the horizon or substitute a skill" is right.
    Here the skill executed **without a fault** — it just didn't accomplish the
    task (the reward signal says the object was not picked / placed). The wrong
    move is to subdivide the same action or re-issue the identical instruction
    (a direct LLM probe showed both: the timeout hint → subdivide; no signal →
    blind repeat). The right move is to change *tactic*.

    Args:
        progress_now: The reward model's **progress** (closeness) score for the
            attempt — the gated head (ADR-0074 amendment), below the contract's
            ``check_floor`` (a genuine failure).

    Example:
        >>> "different approach" in reflect_on_reward_plateau(0.48)
        True
    """
    return (
        f"the policy executed but the reward says the task was NOT completed "
        f"(progress={progress_now:.2f}) — this exact approach is not working. Do NOT "
        "re-issue the same instruction or subdivide it into the same action: "
        "re-dispatch with a DIFFERENT approach (a different grasp / angle / "
        "strategy). If several different approaches keep failing, this object is "
        "abandoned and the mission moves on to the next one."
    )


def reflect_on_invalid_plan(detail: str) -> str:
    """Strategy hint when the model's own tool call was malformed (ADR-0072 §2.3).

    The previous tick produced a tool call the reasoner could not decode —
    malformed JSON arguments, a non-object payload, a wrong/missing field, or an
    rskill_id outside the palette. Feed it straight back so the *next* tick fixes
    the call instead of re-emitting the same broken one. Deterministic — no LLM
    call.

    Args:
        detail: The decode/validation error text (carried verbatim so the model
            sees exactly what was wrong).

    Example:
        >>> "valid tool call" in reflect_on_invalid_plan("malformed JSON arguments")
        True
    """
    return (
        "your previous tool call could not be decoded — emit one valid tool call with a "
        "single well-formed JSON arguments object, using only fields and rskill_ids the "
        f"palette allows. Decode error: {detail}"
    )


def reflect_on_retry_cap(tool: str, cap: int) -> str:
    """Strategy hint when the per-kind retry ladder is exhausted (ADR-0072 §2.3).

    Upgrades the bare retry counter into an explicit "stop repeating, change
    approach" reflection the next tick can act on.
    """
    return (
        f"'{tool}' was selected {cap}+ ticks in a row with no progress — the retry ladder "
        "is exhausted; substitute a different skill, adjust parameters, or replan the goal."
    )


def render_playbooks_block(entries: list[tuple[str, str]]) -> str:
    r"""Render the ``## PLAYBOOKS`` system-prompt block (ADR-0072 Decision 1 / Phase 3).

    Each entry is ``(header, body_markdown)`` — the playbook's ``name — trigger``
    header and its hand-authored ``PLAYBOOK.md`` SOP. The reasoner appends this to
    its system prompt at configure time so the LLM follows an installed decision
    procedure when its trigger matches the goal. The playbook guides *decisions*
    only — every motion still goes through ``execute_rskill`` and the C++ safety
    kernel (CLAUDE.md §1.1).

    Returns ``""`` when no playbooks are installed, so appending it to a system
    prompt is a no-op (the block is omitted entirely).

    Example:
        >>> render_playbooks_block([])
        ''
        >>> block = render_playbooks_block([("find-object — locate X", "## Steps\\n1. ...")])
        >>> "## PLAYBOOKS" in block
        True
    """
    if not entries:
        return ""
    parts = [
        "## PLAYBOOKS",
        (
            "Installed decision procedures (SOPs). When a goal matches a playbook's "
            "trigger, follow its steps, verify its done predicate, and use its "
            "fallbacks. Playbooks guide your decisions; every motion still goes "
            "through execute_rskill and the safety kernel. A playbook is NOT a "
            "skill: never pass a playbook name to execute_rskill. The only "
            "executable skills are the ones in your tool list; a playbook only "
            "tells you which of those to dispatch and in what order."
        ),
    ]
    for header, body in entries:
        parts.append(f"\n--- playbook: {header} ---\n{body.strip()}")
    return "\n".join(parts)


def render_robot_self_model(description: RobotDescription) -> str:
    """Render the static **robot self-model** ("robot resume") text block.

    A one-time, deterministic summary of what the robot *is and can do* —
    embodiment, DOF, end-effectors, locomotion, payload, capability flags, and
    cameras (with field-of-view) — derived from the :class:`RobotDescription`.
    The reasoner injects this as the ``## ROBOT`` context section (computed once
    at configure time) so the LLM can judge feasibility — "is the target in
    reach / in view?" — before dispatching a skill, instead of guessing (ADR-0072
    Decision 2.1, the EMOS "Robot Resume" idea). Pixel-free (ADR-0018 §4).

    Args:
        description: The robot manifest loaded from ``robots/<id>/robot.yaml``.

    Returns:
        A deterministic multi-line block (no trailing newline).

    Example:
        >>> from openral_core import RobotDescription
        >>> d = RobotDescription.from_yaml("robots/so100_follower/robot.yaml")
        >>> "name: so100_follower" in render_robot_self_model(d)
        True
    """
    caps = description.capabilities
    lines: list[str] = [
        f"name: {description.name} (embodiment={description.embodiment_kind.value})",
        f"dof: {len(description.joints)} joints",
    ]
    if description.end_effectors:
        ees = ", ".join(f"{e.name}({e.kind})" for e in description.end_effectors)
        lines.append(f"end_effectors: {ees}")
    if caps.locomotion:
        # LocomotionKind is a Literal[str], so the items are already strings.
        lines.append(f"locomotion: {', '.join(sorted(caps.locomotion))}")
    if caps.can_lift_kg > 0.0:
        lines.append(f"payload_kg: {caps.can_lift_kg:.1f}")
    flags = [
        name
        for name, present in (
            ("vision", caps.has_vision),
            ("force_control", caps.has_force_control),
            ("tactile", caps.has_tactile),
            ("dexterous_hands", caps.has_dexterous_hands),
            ("lidar", caps.has_lidar),
            ("audio", caps.has_audio),
            ("bimanual", caps.bimanual),
        )
        if present
    ]
    if flags:
        lines.append(f"capabilities: {', '.join(flags)}")
    # Cameras = sensors carrying pinhole intrinsics or a declared FOV. The FOV
    # (frustum) is the LLM's "what can I see and how wide" cue for view feasibility.
    cameras: list[str] = []
    for sensor in description.sensors:
        if sensor.intrinsics is None and sensor.fov_h_deg is None:
            continue
        if sensor.fov_h_deg is not None and sensor.fov_v_deg is not None:
            cameras.append(f"{sensor.name}(fov {sensor.fov_h_deg:.0f}x{sensor.fov_v_deg:.0f}deg)")
        else:
            cameras.append(sensor.name)
    if cameras:
        lines.append(f"cameras: {', '.join(cameras)}")
    if caps.supported_control_modes:
        lines.append(f"control_modes: {', '.join(m.value for m in caps.supported_control_modes)}")
    return "\n".join(lines)


class ContextRenderer:
    """Stateful structured-text builder for the reasoner LLM.

    Rolling buffers retain the most recent
    :data:`DEFAULT_BUFFER_SIZE` items per category by default; older
    events fall off. The :meth:`render` method produces a deterministic
    text block given the current world state plus the buffer contents.

    Args:
        buffer_size: Per-category retention. Smaller values keep the
            prompt cheap; larger values give the LLM more history.
            Default :data:`DEFAULT_BUFFER_SIZE`.

    Example:
        >>> r = ContextRenderer()
        >>> r.append_prompt(PromptRecord(text="pick the cube", metadata_json="", stamp_ns=0))
        >>> "pick the cube" in r.render(world_state=None)
        True
    """

    def __init__(
        self, *, buffer_size: int = DEFAULT_BUFFER_SIZE, robot_model: str | None = None
    ) -> None:
        """Stash buffer capacity, the static robot self-model, and arm empty FIFOs.

        Args:
            buffer_size: Per-category rolling-buffer retention.
            robot_model: Pre-rendered robot self-model text (ADR-0072 Decision
                2.1, from :func:`render_robot_self_model`), rendered as the
                ``## ROBOT`` section. ``None`` omits the section (e.g. before the
                robot description is loaded).
        """
        if buffer_size < 1:
            raise ValueError(
                f"ContextRenderer.buffer_size must be >= 1; got {buffer_size!r}",
            )
        self._buffer_size = buffer_size
        self._robot_model = robot_model
        # ADR-0072 §3 / Phase 4b — the rendered `## MEMORY` block (the self-
        # maintained MEMORY.md), set via `set_memory_block`. None omits the section.
        self._memory_block: str | None = None
        # ADR-0073 §1 — the active mission (ordered task queue). Set via
        # `set_mission`, advanced via `advance_mission`; rendered as `## MISSION`.
        # None (or an empty mission) omits the section.
        self._mission: MissionState | None = None
        # ADR-0076 — latest continuous-detector enumeration (camera-space 2D
        # detections with stable det_ids). Set via `set_in_view`; rendered as the
        # `in_view[<camera>]` line in WORLD_STATE. None omits it. Depth-free, so it
        # populates even when the 3D lift / `scene_objects` cannot.
        self._in_view: ObjectsMetadata | None = None
        # ADR-0076 — sticky open-vocab locate hits, keyed by lowercased label
        # (latest bbox wins, insertion-ordered, capped). The continuous detector
        # overwrites `_in_view` every frame with its fixed vocabulary; a goal noun
        # the reasoner confirmed via `locate_in_view` (open-vocab) would otherwise
        # vanish on the next clobber. Persisting it here keeps the grounded object
        # in the `located[<cam>]` line so the LLM can decompose / dispatch instead
        # of re-locating it every tick. Fed via `note_located`.
        self._located: dict[str, ObjectDetection2D] = {}
        self._located_sensor: str | None = None
        # ADR-0074 amendment — latest reward-model assessment (both heads). Set
        # via `set_reward_state`; rendered as the `## REWARD` section. None omits
        # it. Kept as a single latest snapshot (not a buffer): the reward is a
        # current-state readout, not an event stream.
        self._reward_state: RewardStateRecord | None = None
        self._failures: deque[FailureEventRecord] = deque(maxlen=buffer_size)
        self._executions: deque[ExecutionEventRecord] = deque(maxlen=buffer_size)
        self._perception: deque[PerceptionEventRecord] = deque(maxlen=buffer_size)
        # Prompt buffer is a list (not a deque) because we order it by
        # priority on insert (ADR-0018 §3.F10 — human-source prompts
        # override queued auto-prompts). Capacity is enforced by
        # ``append_prompt`` evicting the oldest lowest-priority entry.
        self._prompts: list[PromptRecord] = []
        # Monotonic mutation counter — increments on every successful
        # append_*. ReasonerCore reads ``seq`` to decide whether a
        # heartbeat tick can be suppressed: if nothing has arrived
        # since the last successful tick the LLM call is wasted.
        # ADR-0018 amendment 2026-05-25 §2 ("heartbeat_idle").
        self._seq: int = 0

    # ── static robot self-model ─────────────────────────────────────────────

    def set_robot_model(self, robot_model: str | None) -> None:
        """Set (or clear) the static robot self-model rendered as ``## ROBOT``.

        Called once after the ``RobotDescription`` is loaded (ADR-0072 Decision
        2.1). Static config, not an event: it does not touch the rolling buffers
        or bump :attr:`seq`, so it is safe to call on a live renderer.
        """
        self._robot_model = robot_model

    def set_memory_block(self, memory_block: str | None) -> None:
        """Set (or clear) the ``## MEMORY`` block — the self-maintained MEMORY.md.

        ADR-0072 §3 / Phase 4b. Re-set after each ``memory_write`` so the LLM sees
        the updated memory next tick. Static config — does not bump :attr:`seq`.
        """
        self._memory_block = memory_block

    # ── mission (ADR-0073 §1 — sequential task queue) ───────────────────────

    def set_mission(self, mission: MissionState | None) -> None:
        """Set (or clear) the active mission rendered as ``## MISSION``.

        A new mission is a new goal — an **event** — so this bumps :attr:`seq`
        to wake an otherwise-idle heartbeat (unlike the static
        :meth:`set_robot_model` / :meth:`set_memory_block`). The active task's
        text is the goal the reasoner pursues until it is verified and the queue
        advances.
        """
        self._mission = mission
        self._seq += 1

    def set_in_view(self, objects: ObjectsMetadata | None) -> None:
        """Set (or clear) the camera-space ``in_view`` enumeration (ADR-0076).

        The latest continuous-detector :class:`ObjectsMetadata` — 2D detections
        with stable ``det_id``s — rendered as the ``in_view[<camera>]`` line in
        WORLD_STATE. A new perception snapshot is an **event**, so this bumps
        :attr:`seq` to wake an otherwise-idle heartbeat. Depth-free: it grounds
        a goal noun onto a concrete object even when the 3D ``scene_objects`` line
        is empty (RGB-only / no lift).
        """
        self._in_view = objects
        self._seq += 1

    #: Max distinct labels retained in the sticky ``located`` store (ADR-0076).
    _LOCATED_CAP = 12

    def note_located(self, objects: ObjectsMetadata | None) -> None:
        """Persist open-vocab ``locate_in_view`` hits into the sticky ``located`` line.

        The complement to :meth:`set_in_view`: that holds the *continuous*
        detector's latest (fixed-vocabulary) frame, which is overwritten every
        tick. When the reasoner confirms a goal noun with the open-vocab
        ``locate_in_view`` detector (e.g. ``basket``, ``ketchup`` — labels the
        fixed indoor vocabulary mislabels as ``tray`` / ``bottle``), the hit is
        kept here keyed by lowercased label (latest bbox wins, capped at
        :attr:`_LOCATED_CAP`) so it survives the next continuous clobber and the
        LLM can ground / decompose instead of re-locating it. A confirmed
        detection is an **event**, so this bumps :attr:`seq`. ``None`` / empty is
        a no-op.

        Example:
            >>> from openral_core import ObjectDetection2D, ObjectsMetadata
            >>> r = ContextRenderer()
            >>> r.note_located(ObjectsMetadata(sensor_id="top", model_id="omdet",
            ...     frame_width=256, frame_height=256, detections=[
            ...     ObjectDetection2D(label="basket", confidence=0.6,
            ...                       bbox_xyxy=(10, 20, 30, 40))]))
            >>> "located[top]: basket" in r.render(world_state=None)
            True
        """
        if objects is None or not objects.detections:
            return
        self._located_sensor = objects.sensor_id or self._located_sensor
        for det in objects.detections:
            key = det.label.strip().lower()
            if not key:
                continue
            self._located.pop(key, None)  # re-insert so newest sorts last / evicts last
            self._located[key] = det
        while len(self._located) > self._LOCATED_CAP:
            self._located.pop(next(iter(self._located)))  # evict oldest
        self._seq += 1

    def set_reward_state(self, reward: RewardStateRecord | None) -> None:
        """Set (or clear) the latest reward assessment rendered as ``## REWARD``.

        ADR-0074 amendment — surfaces **both** reward heads (progress closeness +
        success done-confidence), distinctly labelled, so the LLM uses progress
        for persist-vs-replan and success for done-ness. A fresh assessment is an
        **event**, so this bumps :attr:`seq` to wake an otherwise-idle heartbeat.

        Example:
            >>> r = ContextRenderer()
            >>> r.set_reward_state(RewardStateRecord(progress=0.81, success=0.45,
            ...     progress_trend=0.04, success_trend=0.01, task="pick the bowl",
            ...     stamp_ns=0))
            >>> "progress=0.81 (closeness" in r.render(world_state=None)
            True
            >>> "success=0.45 (done-confidence" in r.render(world_state=None)
            True
        """
        self._reward_state = reward
        self._seq += 1

    @property
    def mission(self) -> MissionState | None:
        """The active :class:`MissionState`, or ``None``.

        The node mutates it in place for non-waking bookkeeping
        (:meth:`MissionState.record_attempt` / :meth:`MissionState.mark_verifying`);
        completion/abandonment go through :meth:`advance_mission` so the next
        active task wakes the reasoner.
        """
        return self._mission

    def advance_mission(self, *, done: bool, verdict: str) -> TaskState | None:
        """Terminate the active task and activate the next, bumping :attr:`seq`.

        ``done=True`` marks the active task ``done`` (verified complete);
        ``done=False`` marks it ``abandoned`` (ladder exhausted / unverifiable).
        Advancing the queue is an event — the new active task must wake the
        reasoner to dispatch it — so this bumps :attr:`seq` whenever a mission is
        present. Returns the newly-active :class:`TaskState`, or ``None`` when the
        mission is finished. A no-op (no mission / no active task) does not bump.
        """
        if self._mission is None:
            return None
        nxt = (
            self._mission.complete_active(verdict)
            if done
            else self._mission.abandon_active(verdict)
        )
        self._seq += 1
        return nxt

    # ── rolling buffer mutators ─────────────────────────────────────────────

    def append_failure(self, record: FailureEventRecord) -> None:
        """Push a failure event onto the rolling buffer."""
        self._failures.append(record)
        self._seq += 1

    def append_execution(self, record: ExecutionEventRecord) -> None:
        """Push a skill execution outcome onto the rolling buffer (ADR-0072 §2.2).

        A completed skill is a meaningful event, so this bumps :attr:`seq` —
        the success/failure feedback should wake an otherwise-idle heartbeat.
        """
        self._executions.append(record)
        self._seq += 1

    def append_perception(self, record: PerceptionEventRecord) -> None:
        """Push a perception event onto the rolling buffer."""
        self._perception.append(record)
        self._seq += 1

    def append_prompt(self, record: PromptRecord) -> None:
        """Push an operator prompt onto the rolling buffer.

        Priority resolution (ADR-0018 §3.F10):

        * If ``record.priority`` is anything other than the
          default sentinel (10), it wins verbatim.
        * Otherwise, ``metadata_json`` is parsed and its
          top-level ``priority`` field (if present and ``int``) is
          honoured — that's the field the F10
          ``prompt_router_node`` writes on every fan-out.
        * Otherwise the default of 10 is kept.

        After resolution, the record is inserted so the buffer stays
        ordered by priority descending then arrival ascending. When
        the buffer would exceed :attr:`_buffer_size`, the oldest entry
        in the **lowest-priority** band is evicted (high-priority
        prompts never get dropped to make room for a low-priority
        arrival).
        """
        resolved = record
        if record.priority == DEFAULT_PROMPT_PRIORITY:
            parsed = _extract_priority(record.metadata_json)
            if parsed != DEFAULT_PROMPT_PRIORITY:
                resolved = dataclasses.replace(record, priority=parsed)
        # Find the insertion point: keep the list ordered by
        # ``-priority`` (descending) and stable arrival order.
        insert_at = len(self._prompts)
        for i, existing in enumerate(self._prompts):
            if existing.priority < resolved.priority:
                insert_at = i
                break
        self._prompts.insert(insert_at, resolved)
        # Enforce capacity by evicting the trailing (lowest-priority,
        # oldest) entry. This preserves the documented contract that
        # a high-priority prompt never gets dropped on a buffer
        # overflow caused by lower-priority arrivals.
        while len(self._prompts) > self._buffer_size:
            self._prompts.pop()
        self._seq += 1

    # ── rendering ───────────────────────────────────────────────────────────

    def render(self, *, world_state: WorldState | None) -> str:
        """Return the deterministic text snapshot for an LLM tick.

        Args:
            world_state: Latest WorldState snapshot, or ``None`` when
                the aggregator has not yet produced one.

        Returns:
            A multi-section text block: optional ``## ROBOT`` self-model,
            ``## MEMORY``, ``## MISSION`` (the active task queue, when a
            non-empty mission is set), and ``## REWARD`` (the latest two-head
            reward assessment, when set) followed by ``## WORLD_STATE``,
            ``## EXECUTION``, ``## FAILURES``, ``## PERCEPTION``, ``## PROMPTS``.
        """
        sections: list[str] = []
        if self._robot_model is not None:
            sections += ["## ROBOT", self._robot_model, ""]
        if self._memory_block is not None:
            sections += [self._memory_block, ""]
        if self._mission is not None and not self._mission.is_empty():
            sections += ["## MISSION", self._mission.render(), ""]
        if self._reward_state is not None:
            sections += ["## REWARD", self._render_reward(), ""]
        sections += [
            "## WORLD_STATE",
            self._render_world_state(world_state),
            "",
            "## EXECUTION",
            self._render_executions(),
            "",
            "## FAILURES",
            self._render_failures(),
            "",
            "## PERCEPTION",
            self._render_perception(),
            "",
            "## PROMPTS",
            self._render_prompts(),
        ]
        return "\n".join(sections).rstrip() + "\n"

    def _render_reward(self) -> str:
        """Render the two-head reward assessment (ADR-0074 amendment).

        Both heads carry distinct meanings, so they are labelled in line: progress
        is *closeness* (the gated head, drives persist-vs-replan), success is
        *done-confidence* (compressed; secondary). Trends are per-frame slopes so a
        rising progress (``+``) says "persist", a flat/falling one says "replan".
        """
        r = self._reward_state
        assert r is not None  # render() guards `is not None` before calling
        return (
            f"reward[{r.task}]: progress={r.progress:.2f} (closeness, "
            f"trend {r.progress_trend:+.3f}/frame), "
            f"success={r.success:.2f} (done-confidence, trend {r.success_trend:+.3f}/frame). "
            "Gate on progress for persist-vs-replan; success is a secondary done-ness cue."
        )

    def _render_world_state(self, world_state: WorldState | None) -> str:
        """Render the WorldState block — deterministic key order, no pixels."""
        if world_state is None:
            # ADR-0076: the camera-space in_view enumeration is depth-free and may
            # arrive before the first WorldState snapshot — surface it regardless.
            in_view = self._render_in_view()
            return f"(no snapshot yet)\n{in_view}" if in_view else "(no snapshot yet)"
        lines: list[str] = [f"stamp_ns: {world_state.stamp_ns}"]
        if world_state.joint_state is not None:
            js = world_state.joint_state
            joints = ", ".join(
                f"{name}={pos:+.3f}"
                for name, pos in sorted(zip(js.name, js.position, strict=False))
            )
            lines.append(f"joint_state: {joints}")
        if world_state.ee_poses:
            for ee_name, pose in sorted(world_state.ee_poses.items()):
                x, y, z = pose.xyz
                qx, qy, qz, qw = pose.quat_xyzw
                lines.append(
                    f"ee_pose[{ee_name}]: xyz=({x:+.3f},{y:+.3f},{z:+.3f}) "
                    f"quat_xyzw=({qx:+.3f},{qy:+.3f},{qz:+.3f},{qw:+.3f}) "
                    f"frame={pose.frame_id}",
                )
        if world_state.battery_pct is not None:
            lines.append(f"battery_pct: {world_state.battery_pct:.1f}")
        if world_state.diagnostics:
            diag = ", ".join(
                f"{name}={status}" for name, status in sorted(world_state.diagnostics.items())
            )
            lines.append(f"diagnostics: {diag}")
        if world_state.detected_objects:
            # ADR-0035/0051 (#14) — surface the lifted scene objects (label +
            # frame-centre) so the LLM can map a goal noun onto a detected label
            # with its own semantics (e.g. a "baguette" goal onto the detected
            # "bread") instead of only learning a name is "not in memory". The
            # open-vocab detector emits overlapping boxes, so dedupe by label
            # (first-seen pose) and render in sorted order for a stable context.
            first_seen: dict[str, tuple[float, float, float]] = {}
            for obj in world_state.detected_objects:
                first_seen.setdefault(obj.label, obj.pose.xyz)
            frame = world_state.detected_objects[0].pose.frame_id
            items = ", ".join(
                f"{label}@({xyz[0]:+.2f},{xyz[1]:+.2f},{xyz[2]:+.2f})"
                for label, xyz in sorted(first_seen.items())
            )
            lines.append(f"scene_objects[{frame}]: {items}")
        in_view = self._render_in_view()
        if in_view:
            lines.append(in_view)
        return "\n".join(lines)

    def _render_in_view(self) -> str:
        """Render the camera-space ``in_view`` enumeration (ADR-0076), or ``""``.

        ``in_view[<camera>]: #<det_id> <label> @px(<cx>,<cy>), …`` — one entry per
        live 2D detection, ``@px`` the pixel centre in the detector's frame
        (image space, **not** a 3D pose). Ordered by ``det_id`` for a stable
        context. Distinct from the 3D ``scene_objects[<map>]:@(x,y,z)`` line so the
        two coordinate spaces never blur; it populates without the 3D lift.
        """
        lines: list[str] = []
        md = self._in_view
        if md is not None and md.detections:
            items = ", ".join(
                f"#{d.det_id} {d.label} @px({(d.bbox_xyxy[0] + d.bbox_xyxy[2]) // 2},"
                f"{(d.bbox_xyxy[1] + d.bbox_xyxy[3]) // 2})"
                for d in sorted(md.detections, key=lambda d: d.det_id)
            )
            lines.append(f"in_view[{md.sensor_id}]: {items}")
        # ADR-0076 — sticky open-vocab locate hits. These carry the goal nouns the
        # fixed-vocabulary in_view line mislabels, so they are the authoritative
        # grounding for decomposing / dispatching against the mission objects.
        if self._located:
            loc = ", ".join(
                f"{d.label} @px({(d.bbox_xyxy[0] + d.bbox_xyxy[2]) // 2},"
                f"{(d.bbox_xyxy[1] + d.bbox_xyxy[3]) // 2})"
                for d in self._located.values()
            )
            lines.append(f"located[{self._located_sensor or 'view'}]: {loc}")
        return "\n".join(lines)

    def _render_failures(self) -> str:
        """Render the failure buffer; one line per event, oldest first."""
        if not self._failures:
            return "(none)"
        lines: list[str] = []
        for rec in self._failures:
            evidence_summary = _summarise_evidence_json(rec.evidence_json)
            lines.append(
                f"[{rec.source}] kind={rec.kind} severity={rec.severity} "
                f"skill={rec.rskill_id or '-'} trace={rec.trace_id[:8] or '-'} {evidence_summary}",
            )
        return "\n".join(lines)

    def _render_executions(self) -> str:
        """Render the execution-feedback buffer; one line per outcome, oldest first.

        ``[ok] skill=<id>: <summary>`` for successes; failures append the
        Reflexion strategy hint: ``[failed] skill=<id>: <summary> — reflect: <hint>``.
        """
        if not self._executions:
            return "(none)"
        lines: list[str] = []
        for rec in self._executions:
            line = f"[{rec.outcome}] skill={rec.rskill_id or '-'}: {rec.summary}"
            if rec.reflection:
                line += f" — reflect: {rec.reflection}"
            lines.append(line)
        return "\n".join(lines)

    def _render_perception(self) -> str:
        """Render the perception buffer; one line per event, oldest first."""
        if not self._perception:
            return "(none)"
        return "\n".join(f"[{rec.kind}] {rec.text}" for rec in self._perception)

    def _render_prompts(self) -> str:
        """Render the pending-prompt buffer; one line per prompt, oldest first."""
        if not self._prompts:
            return "(none)"
        return "\n".join(rec.text for rec in self._prompts)

    # ── readers used by tests and the action dispatcher ────────────────────

    @property
    def failures(self) -> tuple[FailureEventRecord, ...]:
        """Return a snapshot of the failure buffer (oldest first)."""
        return tuple(self._failures)

    @property
    def executions(self) -> tuple[ExecutionEventRecord, ...]:
        """Return a snapshot of the execution-feedback buffer (oldest first)."""
        return tuple(self._executions)

    @property
    def perception_events(self) -> tuple[PerceptionEventRecord, ...]:
        """Return a snapshot of the perception buffer (oldest first)."""
        return tuple(self._perception)

    @property
    def prompts(self) -> tuple[PromptRecord, ...]:
        """Return a snapshot of the prompt buffer (oldest first)."""
        return tuple(self._prompts)

    @property
    def seq(self) -> int:
        """Monotonic mutation counter.

        Increments on every successful :meth:`append_failure`,
        :meth:`append_perception`, or :meth:`append_prompt`. Used by
        :class:`~openral_reasoner.core.ReasonerCore` to short-circuit a
        heartbeat tick when no event has arrived since the last
        successful tick (ADR-0018 amendment 2026-05-25 §2).

        Not reset by :meth:`drain_prompts` — the buffer is empty
        afterwards but the renderer has still "seen" the prompt.
        """
        return self._seq

    def drain_prompts(self) -> tuple[PromptRecord, ...]:
        """Return and clear the prompt buffer.

        Prompts are pull-once events — once the reasoner has seen one
        on a tick it should not see the same one again, so the
        reasoner_node calls :meth:`drain_prompts` after each
        successful :meth:`render`. Records come back ordered by
        priority descending (then arrival ascending) per ADR-0018
        §3.F10.
        """
        drained = tuple(self._prompts)
        self._prompts.clear()
        return drained


def _extract_priority(metadata_json: str) -> int:
    """Pull a top-level ``priority`` field out of ``metadata_json``.

    Returns ``10`` (the default auto-cascade priority) when the JSON
    is empty / malformed / does not contain a top-level integer
    ``priority`` field. Mirrors the field
    :func:`PromptRouterNode._on_inbound` writes on every fan-out.
    """
    if not metadata_json:
        return DEFAULT_PROMPT_PRIORITY
    try:
        parsed = json.loads(metadata_json)
    except json.JSONDecodeError:
        return DEFAULT_PROMPT_PRIORITY
    if not isinstance(parsed, dict):
        return DEFAULT_PROMPT_PRIORITY
    value = parsed.get("priority")
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return DEFAULT_PROMPT_PRIORITY


def _summarise_evidence_json(payload: str) -> str:
    """Return a one-line summary of a FailureEvidence payload.

    Falls back to ``"evidence=<raw-truncated>"`` when the payload does
    not decode against the discriminator (a malformed publisher should
    be visible in the prompt, not silently swallowed).
    """
    if not payload:
        return ""
    try:
        evidence = _FAILURE_ADAPTER.validate_json(payload)
    except Exception:  # reason: defensive — malformed publisher
        try:
            return f"evidence={json.dumps(json.loads(payload), sort_keys=True)[:120]}"
        except Exception:
            return f"evidence={payload[:120]!r}"
    summary_fields = {k: v for k, v in evidence.model_dump().items() if k not in {"kind"}}
    return f"evidence={json.dumps(summary_fields, sort_keys=True)[:120]}"


# Suppress unused-import — TypeAdapter import for PerceptionEventMetadata is
# reserved for the next iteration that will summarise perception payloads
# alongside their text. For now the text field carries enough; the adapter
# is left exposed so downstream consumers can decode metadata_json themselves.
_ = _PERCEPTION_ADAPTER
