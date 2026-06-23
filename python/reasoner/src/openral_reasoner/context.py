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
    PerceptionEventMetadata,
    RobotDescription,
    WorldState,
)
from pydantic import TypeAdapter

__all__ = [
    "DEFAULT_BUFFER_SIZE",
    "DEFAULT_PROMPT_PRIORITY",
    "ContextRenderer",
    "ExecutionEventRecord",
    "FailureEventRecord",
    "PerceptionEventRecord",
    "PromptRecord",
    "reflect_on_failure",
    "reflect_on_retry_cap",
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

    ADR-0071 Decision 2.2 (Inner Monologue): a typed one-line outcome appended
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


def reflect_on_failure(outcome_state: str, detail: str) -> str:
    """One-line strategy hint from a terminal skill outcome (ADR-0071 §2.3).

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


def reflect_on_retry_cap(tool: str, cap: int) -> str:
    """Strategy hint when the per-kind retry ladder is exhausted (ADR-0071 §2.3).

    Upgrades the bare retry counter into an explicit "stop repeating, change
    approach" reflection the next tick can act on.
    """
    return (
        f"'{tool}' was selected {cap}+ ticks in a row with no progress — the retry ladder "
        "is exhausted; substitute a different skill, adjust parameters, or replan the goal."
    )


def render_robot_self_model(description: RobotDescription) -> str:
    """Render the static **robot self-model** ("robot resume") text block.

    A one-time, deterministic summary of what the robot *is and can do* —
    embodiment, DOF, end-effectors, locomotion, payload, capability flags, and
    cameras (with field-of-view) — derived from the :class:`RobotDescription`.
    The reasoner injects this as the ``## ROBOT`` context section (computed once
    at configure time) so the LLM can judge feasibility — "is the target in
    reach / in view?" — before dispatching a skill, instead of guessing (ADR-0071
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
            robot_model: Pre-rendered robot self-model text (ADR-0071 Decision
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

        Called once after the ``RobotDescription`` is loaded (ADR-0071 Decision
        2.1). Static config, not an event: it does not touch the rolling buffers
        or bump :attr:`seq`, so it is safe to call on a live renderer.
        """
        self._robot_model = robot_model

    # ── rolling buffer mutators ─────────────────────────────────────────────

    def append_failure(self, record: FailureEventRecord) -> None:
        """Push a failure event onto the rolling buffer."""
        self._failures.append(record)
        self._seq += 1

    def append_execution(self, record: ExecutionEventRecord) -> None:
        """Push a skill execution outcome onto the rolling buffer (ADR-0071 §2.2).

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
            A multi-section text block: an optional ``## ROBOT`` self-model
            (when provided at construction) followed by ``## WORLD_STATE``,
            ``## FAILURES``, ``## PERCEPTION``, ``## PROMPTS`` sections.
        """
        sections: list[str] = []
        if self._robot_model is not None:
            sections += ["## ROBOT", self._robot_model, ""]
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

    def _render_world_state(self, world_state: WorldState | None) -> str:
        """Render the WorldState block — deterministic key order, no pixels."""
        if world_state is None:
            return "(no snapshot yet)"
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
