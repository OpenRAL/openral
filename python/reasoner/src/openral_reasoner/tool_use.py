"""ADR-0018 F4 — typed LLM tool-use client + concrete provider implementations.

Replaces the BT-XML-oriented :class:`~openral_reasoner.LLMClient` /
:class:`~openral_reasoner.Plan` flow with direct LLM tool dispatch:
every reasoner tick the LLM picks exactly one of the four
:data:`~openral_core.ReasonerToolCall` variants (ExecuteSkill,
ReloadGstPipeline, LifecycleTransition, EmitPrompt) and the reasoner
node routes it onto the ROS graph. No BT v4 XML in v1
(CLAUDE.md §6.2 / §7.6 amendment in this PR).

This module ships:

- :class:`ToolUseClient` — structural Protocol every provider satisfies.
- :class:`AnthropicToolUseClient` — wraps the Anthropic Python SDK's
  tool-use API. Lazy-imported; gated by ``OPENRAL_REASONER_LLM_PROVIDER=anthropic``.
- :class:`OpenAICompatibleToolUseClient` — wraps the OpenAI Python SDK
  pointed at any OpenAI-compatible endpoint (cloud OpenAI, local vLLM,
  Ollama-OpenAI, etc.). Lazy-imported; gated by
  ``OPENRAL_REASONER_LLM_PROVIDER=openai-compatible``.
- :class:`build_tool_use_client_from_env` — factory that reads the
  deployment env and returns the right client. No cloud lock-in:
  the open-core path defaults to "no provider configured"; the user
  picks one explicitly via env. ``PROVIDER=openrouter`` is a
  shortcut that pre-fills the OpenRouter base URL on top of the
  generic ``openai-compatible`` client.

Per CLAUDE.md §1.11 a deterministic :class:`FakeToolUseClient` lives
under :mod:`tests.integration.fakes.fake_llm` — it is the only test
double we allow at this process boundary, named explicitly, and used
exclusively in tests.
"""

from __future__ import annotations

import base64
import json
import os
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import structlog
from openral_core import ReasonerToolCall, RobotCapabilities
from openral_core.exceptions import (
    ROSConfigError,
    ROSPlanningError,
    ROSReasonerInvalidPlan,
)
from pydantic import BaseModel, TypeAdapter, ValidationError

if TYPE_CHECKING:
    from openral_reasoner.palette import RSkillToolEntry, ToolPalette

__all__ = [
    "DEEPSEEK_BASE_URL",
    "DEFAULT_SYSTEM_PROMPT",
    "GEMINI_BASE_URL",
    "OLLAMA_BASE_URL",
    "OPENROUTER_BASE_URL",
    "SYSTEM_PROMPT_ENV_VAR",
    "VLLM_BASE_URL",
    "XAI_BASE_URL",
    "AnthropicToolUseClient",
    "OpenAICompatibleToolUseClient",
    "ToolUseClient",
    "build_tool_use_client_from_env",
    "render_robot_context_prompt",
    "resolve_reasoner_system_prompt",
]

log = structlog.get_logger(__name__)

# Default system prompt — the robot-agnostic operating brief. Factual and
# intentionally non-anthropomorphising: the reasoner is a scheduler, not a
# chatbot. ``ReasonerNode`` composes the live prompt via
# :func:`resolve_reasoner_system_prompt`, which lets a deployment replace
# this base via the ``OPENRAL_REASONER_SYSTEM_PROMPT`` env var and appends a
# per-robot ``## THIS ROBOT`` block (:func:`render_robot_context_prompt`).
# ``ReasonerCore(system_prompt=...)`` can also override it directly.
DEFAULT_SYSTEM_PROMPT: str = (
    # ── Role ──────────────────────────────────────────────────────────
    "You are the OpenRAL S2 reasoner: the slow, deliberative control "
    "layer for a PHYSICAL robot operating in the real world. Your "
    "decisions move real hardware, so they must be deliberate and "
    "conservative. On every tick you receive a structured text snapshot "
    "with four sections: WORLD_STATE (the robot's joints, end-effector "
    "poses, battery, and diagnostics), FAILURES (recent faults, "
    "including safety-kernel and e-stop events), PERCEPTION (recent "
    "perception events, e.g. detected objects), and PROMPTS (pending "
    "operator instructions — this is where the task/goal arrives). "
    # ── One tool per tick ─────────────────────────────────────────────
    "Pick EXACTLY ONE tool from the provided palette to advance the "
    "task; do not narrate, do not chain. You are called repeatedly: "
    "multi-step tasks are sequenced across ticks, not within one call. "
    "Each tick, re-read the snapshot and choose the single best next "
    "action given what has changed. "
    # ── Follow the operator's goal faithfully ─────────────────────────
    "Follow the operator's instruction in PROMPTS as literally and "
    "faithfully as you can. Do not substitute your own objective, do "
    "not expand scope, and do not skip steps the instruction implies. "
    # ── Smallest actionable unit: one specific object per skill ───────
    "A skill acts on EXACTLY ONE specific, concrete object. Before you "
    "dispatch execute_rskill, the active task MUST name a single specific "
    "object you can point to in PERCEPTION — two lines enumerate what the "
    "detector currently sees: `in_view[<camera>]` lists every object with a "
    "stable id and a camera-space pixel centre (`#3 milk @px(412,233)`; always "
    "present), and `scene_objects[<map>]` adds 3D world positions when depth is "
    "available. Use EITHER to enumerate — prefer `in_view` when `scene_objects` "
    "is absent. A collective or quantified target is NEVER directly actionable: a "
    "quantifier (all / every / each / both / everything) or a bare generic "
    "plural (the objects / items / things) means 'look and enumerate "
    "FIRST'. When the goal targets such a set, do NOT call execute_rskill "
    "— read the concrete objects from PERCEPTION and call decompose_mission "
    "to split the goal into one subtask per specific object, each phrased "
    "as a verb + that specific object (+ destination), e.g. 'pick up the "
    "milk and put it in the basket', 'pick up the ketchup and put it in "
    "the basket'. Each subtask names ONE concrete object — never 'the "
    "first batch of objects', never 'the remaining items'. Only once the "
    "active task names a single specific object may you dispatch a skill "
    "on it. The same rule generalises beyond picking: a coarse goal "
    "('bring me a glass of wine') decomposes into specific verb+object "
    "steps (navigate to the kitchen, open the fridge, pick up the wine "
    "bottle, …), each naming a concrete object, never a vague category. "
    # ── Ground (confirm) before you decompose a collective goal ────────
    # ADR-0075/0076 — the continuous detector's `in_view` line is a FIXED
    # vocabulary that mislabels goal nouns (teapot→bottle, basket→box). A
    # direct probe (glm-5.2, .goals/.../probe_reasoner_decompose_gate.py)
    # showed the LLM building a mission straight from those raw clutter
    # labels (0/3) unless told `located` is authoritative; with this block it
    # locates-to-confirm first and decomposes only on confirmed objects (3/3).
    "GROUND BEFORE YOU DECOMPOSE (collective goals). The `in_view[<camera>]` "
    "line comes from a FIXED-VOCABULARY continuous detector that frequently "
    "MISLABELS the goal objects (a teapot read as 'bottle', a basket as "
    "'box'); do NOT build a mission directly from raw `in_view` labels. The "
    "authoritative grounding is the `located[<camera>]` line — open-vocab "
    "locate_in_view confirmations of the actual goal nouns. Procedure for a "
    "collective/quantified goal (all / every / the objects / things): (1) if "
    "`located` does NOT yet name the goal objects, call locate_in_view to "
    "CONFIRM which visible objects are the ones to act on (this populates "
    "`located`). Phrase that query as concrete object NOUNS — a single noun or "
    "a comma-separated list of the candidate objects (read likely nouns from "
    "the raw `in_view` labels even when mislabeled: e.g. query "
    "`cup, bowl, bottle, basket`). NEVER query the collective phrase itself "
    "(`the objects on the table`, `everything`): the fast locator matches each "
    "term as one object class, so a phrase matches nothing and wastes a "
    "look. (2) the MOMENT `located` names one or more concrete goal "
    "objects, call decompose_mission (one grounded subtask per located "
    "object) and do NOT locate_in_view / recall_object again for an object "
    "already in `located` — re-locating a confirmed object makes no progress. "
    # ── Skill selection (robot- and scene-matched) ────────────────────
    "The palette includes one execute_rskill__<skill> tool per installed "
    "rSkill. The palette has ALREADY been filtered to THIS robot's body "
    "(embodiment tags, capabilities) and to the skills installed for "
    "THIS scene, so every skill tool you see is physically compatible "
    "with the robot — you do not need to re-check the body. Pick the "
    "skill whose description matches the current goal step by its "
    "objects, scenes, and action verbs; when several match, prefer the "
    "most specific. "
    # ── The go-see-then-act ladder ────────────────────────────────────
    "For a goal that targets a specific object, work the ladder in order: "
    "(1) recall where it is, (2) navigate to a reachable approach pose, "
    "(3) aim a camera at it, (4) verify it is actually in view, then "
    "(5) manipulate. Each rung is a separate tick; only descend a rung "
    "once the one above it has succeeded. Skip a rung only when the "
    "snapshot already shows it satisfied (e.g. the object is already "
    "framed in PERCEPTION). "
    # ── (1) Locate before you manipulate ──────────────────────────────
    "Before acting on a target object, confirm it is actually present: "
    "look for it in WORLD_STATE / PERCEPTION. If the object the goal "
    "needs is not currently known and the read-only recall_object tool is "
    "in the palette, call recall_object to recall it from spatial memory "
    "(it returns the object's pose, a camera-facing approach viewpoint, "
    "and any occluding container to open first). The approach viewpoint is "
    "validated against the live map: if a match reads 'approach BLOCKED', "
    "do NOT navigate to it — pick another match or a different vantage. If "
    "the object is still unknown, dispatch a search / exploration skill to "
    "go look for it rather than blindly dispatching a manipulation skill at "
    "a target you cannot locate. "
    # ── (1b) Seen-but-not-lifted: in view is enough to try ─────────────
    "EXCEPTION — seen but not yet lifted: recall_object returns a pose only "
    "for objects already lifted into spatial memory (which needs near-field "
    "depth + detection overlap). If recall_object cannot resolve a pose YET, "
    "but a live-perception check (locate_in_view) or the PERCEPTION snapshot "
    "shows the target is visible RIGHT NOW, it is acceptable to dispatch a "
    "manipulation skill anyway — a mobile-manipulator skill drives the base "
    "AND arm straight from the live camera, so 'in view' is enough to attempt "
    "it. Prefer recall_object's pose for a precise approach when you have it, "
    "but do NOT stall on the search ladder (or escalate to human-handoff) on a "
    "missing 3-D lift while the object is plainly in view. "
    # ── (2) Navigate to get within reach ──────────────────────────────
    "To act on an object that is out of reach, APPROACH it first: "
    "dispatch the navigation (Nav2-backed) skill to drive the base "
    "to the recalled approach pose (or close enough to interact), then "
    "continue on a later tick. When the read-only resolve_place tool is "
    "in the palette, use it to turn a place / room / agent reference "
    "(e.g. 'the kitchen', 'where I was standing') into a navigation goal "
    "pose plus a traversable path. Do not dispatch a manipulation skill at "
    "an object the robot has not yet reached. "
    # ── (3+4) Frame the target, then verify it is in view ─────────────
    "Once within reach, FRAME the target before grasping it: when a "
    "camera-aiming (look-at) skill is in the palette, dispatch it with the "
    "object's 3-D position so the wrist (or named) camera points at it. "
    "Then VERIFY: when the read-only locate_in_view tool is in the palette, "
    "call it to confirm the object is visible in the CURRENT frame right "
    "now — unlike recall_object, which only recalls what was remembered, "
    "locate_in_view checks live perception this instant. Only dispatch a "
    "manipulation skill once the target is confirmed in view; if "
    "locate_in_view reports it is NOT visible, re-aim (look-at a corrected "
    "position) or re-approach rather than grasping blind. "
    # ── Evaluate progress; adapt, don't repeat ────────────────────────
    "Each tick, judge whether the last action is achieving the goal by "
    "comparing WORLD_STATE and PERCEPTION against what the task expects. "
    "If progress stalled or regressed, change tactic — tweak the goal "
    "params, substitute a different skill, or re-plan the approach — "
    "rather than re-issuing the same call against unchanged context. "
    # ── Break a stuck task into finer steps (decompose_mission, #123) ──
    "The MISSION section shows the ordered task queue (one task active at a "
    "time); it advances only when the active task is verified. When a task is "
    "too coarse for one skill, or query_task_progress shows it stalling and a "
    "skill swap / param tweak is not helping, use decompose_mission to break it "
    "into finer subtasks instead of burning attempts until it is abandoned: "
    "call it with target_task_id set to the active task's id (e.g. 't2') and "
    "subtasks = the ordered finer steps. Each subtask is grounded: "
    "{object_ref: <ONE specific object, e.g. 'milk'>, text: <the instruction, "
    "naming that object>} — object_ref must name exactly one concrete object "
    "(never 'all'/'the objects'/'the first batch'), and text must name it. The "
    "active task is flat-spliced in place by its children and you continue on "
    "the first. Subdivision is "
    "depth-bounded; a task already broken down twice is handed to the operator "
    "rather than split again. You may also call decompose_mission with an EMPTY "
    "target_task_id once at the very start to replace a coarse operator goal "
    "with a better ordered decomposition (only before any task has been "
    "attempted). decompose_mission only edits the task ledger — it never moves "
    "the robot. "
    # ── Poll the reward monitor to judge a running skill (ADR-0057) ────
    "When the read-only query_task_progress tool is in the palette, a "
    "reward monitor is running IN PARALLEL with the executing skill (e.g. a "
    "VLA), scoring the live camera against the task. Use it to judge HOW a "
    "running skill is doing — call query_task_progress (with the active "
    "task text and a window_s over the last few seconds) to get a "
    "normalized per-frame progress (0-1), a success probability, and "
    "whether progress has stalled. Poll it WHEN YOU SEE FIT — typically a "
    "few seconds into a long manipulation/VLA dispatch, or whenever you "
    "must decide between letting the skill continue, advancing to the next "
    "step (success high / rising), or entering the replanning ladder "
    "(stalled or success not improving). It is unlike recall_object / "
    "locate_in_view (which answer WHERE an object is): query_task_progress "
    "answers WHETHER THE TASK IS SUCCEEDING. It is ADVISORY and read-only — "
    "it never moves the robot; you act on its signal by choosing the next "
    "tool. Do not poll it every tick for its own sake; query when the "
    "answer would change your next decision. "
    # ── Bound long dispatches so you can poll between them ─────────────
    "IMPORTANT — to be ABLE to poll a running skill, you must hand control "
    "back to yourself: an execute_rskill goal dispatched with deadline_s=0 "
    "runs to completion and you cannot act (or poll) until it returns. So "
    "when query_task_progress is in the palette and you dispatch a long "
    "policy (e.g. a VLA), set a BOUNDED deadline_s (a few seconds) — the "
    "goal returns control to you at the deadline, you poll "
    "query_task_progress to judge it, then either re-dispatch the same "
    "skill to continue (progress rising), advance, or replan (stalled / not "
    "succeeding). When no reward monitor is present, deadline_s=0 (run to "
    "completion) remains fine. "
    # ── Safety: observe and work around, NEVER bypass ─────────────────
    "Safety is enforced below you by the C++ safety kernel: you PROPOSE, "
    "the kernel DISPOSES. You observe safety in the FAILURES section "
    "(e-stop requests, workspace / force-limit violations, kernel "
    "faults) but you hold NO authority to enforce or override it. When "
    "a safety or e-stop failure appears, do NOT re-dispatch the action "
    "that triggered it — the kernel may have de-energised the motors. "
    "Work AROUND the failure by changing your plan: pick a safer or "
    "different skill, reduce scope, navigate clear of the obstacle, or "
    "hand the situation to the operator. NEVER attempt to clear an "
    "e-stop, disable or weaken a safety check, or repeat a motion that "
    "just violated a workspace or force limit. If the same skill keeps "
    "failing, stop retrying it and escalate. "
    # ── Other always-present tools ────────────────────────────────────
    "The palette always also includes: reload_gst_pipeline (swap a "
    "sensor's GStreamer pipeline at runtime), lifecycle_transition "
    "(drive a peer ROS node through configure / activate / deactivate / "
    "cleanup), and emit_prompt (publish a PromptStamped onto another "
    "topic — used to message the operator or trigger a cascade). "
    # ── When nothing fits ─────────────────────────────────────────────
    "If no skill tool is appropriate — the task is ambiguous, the "
    "target cannot be found, a search budget is exhausted, or you are "
    "blocked by an unrecoverable failure — pick emit_prompt with "
    "target_topic=/openral/prompt and a brief, specific operator-facing "
    "text explaining why and what you need. "
    # Field-name discipline — small models (gemma, qwen) hallucinate
    # close-but-wrong field names (e.g. `rational` instead of
    # `rationale`, `skill` instead of `rskill_id`); the schema is
    # `extra=forbid` so any typo'd field rejects the whole tool call
    # and the tick is wasted.
    "Use the EXACT field names from each tool's input_schema — "
    "spelling and case must match. The optional explanation field is "
    "spelled `rationale` (with the trailing -e); do not abbreviate it "
    "to `rational`. Do not invent fields the schema does not declare."
)

_TOOL_ADAPTER: TypeAdapter[ReasonerToolCall] = TypeAdapter(ReasonerToolCall)


def render_robot_context_prompt(
    capabilities: RobotCapabilities | None,
    *,
    base_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> str:
    """Append a ``## THIS ROBOT`` body-awareness block to the system prompt.

    The base :data:`DEFAULT_SYSTEM_PROMPT` is robot-agnostic. At reasoner
    lifecycle ``configure`` time the active robot's
    :class:`~openral_core.RobotCapabilities` is known (from the
    ``robot_yaml`` manifest or the constructor), so the node calls this
    to give the LLM standing knowledge of the body it is driving: its
    embodiment tags, whether it can locomote (which gates the base
    prompt's "navigate to approach" rule), what manipulation / sensing
    hardware it has, its payload, and its control modes.

    The block is deterministic (fixed field order, sorted tag lists) so a
    given robot always renders the same prompt — reproducibility per
    CLAUDE.md §8. ``capabilities is None`` returns ``base_prompt``
    unchanged (the reasoner has no robot wired yet).

    Args:
        capabilities: The active robot's capabilities, or ``None`` when
            no robot manifest has been loaded.
        base_prompt: The system prompt to extend. Defaults to
            :data:`DEFAULT_SYSTEM_PROMPT`.

    Returns:
        ``base_prompt`` with a trailing ``## THIS ROBOT`` section, or
        ``base_prompt`` verbatim when ``capabilities is None``.

    Example:
        >>> from openral_core import RobotCapabilities
        >>> caps = RobotCapabilities(
        ...     embodiment_tags=["panda_mobile"],
        ...     locomotion=["wheeled"],
        ...     has_force_control=True,
        ... )
        >>> prompt = render_robot_context_prompt(caps)
        >>> "## THIS ROBOT" in prompt
        True
        >>> "embodiment_tags: panda_mobile" in prompt
        True
        >>> render_robot_context_prompt(None) == DEFAULT_SYSTEM_PROMPT
        True
    """
    if capabilities is None:
        return base_prompt

    lines: list[str] = [
        "## THIS ROBOT",
        "You are embodied as this specific robot. Only pick actions its body supports.",
    ]

    tags = ", ".join(sorted(capabilities.embodiment_tags)) or "(none declared)"
    lines.append(f"embodiment_tags: {tags}")

    # Locomotion drives the base prompt's navigate-to-approach rule, so be
    # explicit either way. ``["none"]`` (the default) / empty means fixed base.
    locomotion = sorted(k for k in capabilities.locomotion if k != "none")
    if locomotion:
        lines.append(
            f"locomotion: {', '.join(locomotion)} — the base can move, so you "
            "may dispatch a navigation skill to approach a target or search "
            "for a missing object.",
        )
    else:
        lines.append(
            "locomotion: none — this robot has no mobile base; it cannot "
            "drive to objects. Do not pick a navigation skill to approach; "
            "if a target is out of reach, hand off to the operator.",
        )

    # Manipulation hardware — only list what's present, deterministic order.
    manip: list[str] = []
    manip.append("dual-arm" if capabilities.bimanual else "single-arm")
    if capabilities.has_dexterous_hands:
        manip.append("dexterous hands")
    if capabilities.has_force_control:
        manip.append("force/impedance control")
    if capabilities.has_tactile:
        manip.append("tactile sensing")
    lines.append(f"manipulation: {', '.join(manip)}")

    sensing: list[str] = []
    if capabilities.has_vision:
        sensing.append("vision")
    if capabilities.has_lidar:
        sensing.append("lidar")
    if capabilities.has_audio:
        sensing.append("audio")
    lines.append(f"sensing: {', '.join(sensing) or '(none declared)'}")

    if capabilities.can_lift_kg > 0:
        lines.append(
            f"payload: up to {capabilities.can_lift_kg:g} kg — do not attempt "
            "to lift heavier loads.",
        )

    if capabilities.supported_control_modes:
        modes = ", ".join(m.value for m in capabilities.supported_control_modes)
        lines.append(f"control_modes: {modes}")

    return base_prompt.rstrip() + "\n\n" + "\n".join(lines) + "\n"


# Env var that overrides the base system prompt (the operating brief). When
# set non-empty it replaces :data:`DEFAULT_SYSTEM_PROMPT`; the per-robot
# ``## THIS ROBOT`` block is still appended on top, so a custom brief keeps
# the factual body description it cannot hardcode. Honoured by
# :func:`resolve_reasoner_system_prompt` (called from ``ReasonerNode``).
SYSTEM_PROMPT_ENV_VAR: str = "OPENRAL_REASONER_SYSTEM_PROMPT"


def resolve_reasoner_system_prompt(
    capabilities: RobotCapabilities | None,
    *,
    env: Mapping[str, str] | None = None,
) -> str:
    """Compose the reasoner system prompt from the env override + robot block.

    The base operating brief is :data:`DEFAULT_SYSTEM_PROMPT`, unless the
    deployment sets :data:`SYSTEM_PROMPT_ENV_VAR`
    (``OPENRAL_REASONER_SYSTEM_PROMPT``) to a non-empty value, which
    replaces it. The per-robot ``## THIS ROBOT`` block
    (:func:`render_robot_context_prompt`) is then appended to whichever
    base is in effect, so a custom brief still carries the factual body
    description (embodiment, locomotion, payload, …) it cannot know
    ahead of time.

    Args:
        capabilities: The active robot's capabilities, or ``None`` when
            no robot manifest has been loaded (no ``## THIS ROBOT``
            block is appended).
        env: Environment mapping to read; defaults to ``os.environ``.
            Injectable so tests don't mutate the process environment.

    Returns:
        The fully-composed system prompt string.

    Example:
        >>> from openral_core import RobotCapabilities
        >>> caps = RobotCapabilities(embodiment_tags=["so100_follower"])
        >>> custom = resolve_reasoner_system_prompt(
        ...     caps, env={"OPENRAL_REASONER_SYSTEM_PROMPT": "Custom brief."}
        ... )
        >>> custom.startswith("Custom brief.")
        True
        >>> "## THIS ROBOT" in custom
        True
        >>> default = resolve_reasoner_system_prompt(caps, env={})
        >>> default.startswith(DEFAULT_SYSTEM_PROMPT.rstrip())
        True
    """
    environ = os.environ if env is None else env
    override = environ.get(SYSTEM_PROMPT_ENV_VAR, "").strip()
    base_prompt = override or DEFAULT_SYSTEM_PROMPT
    return render_robot_context_prompt(capabilities, base_prompt=base_prompt)


@runtime_checkable
class ToolUseClient(Protocol):
    """Wire-level Protocol for an LLM provider that supports tool use.

    Concrete implementations call into Anthropic's ``tools`` API, an
    OpenAI-compatible ``tool_calls`` payload, or any other provider
    that can emit a typed tool selection. The reasoner consumes a
    fully-validated :data:`~openral_core.ReasonerToolCall`; parsing /
    validation lives behind this Protocol.

    Attributes:
        model_id: Provider-specific identifier (``claude-opus-4-7``,
            ``gpt-4o``, ``qwen2.5-7b-instruct``, ...). Recorded on the
            reasoner span.
    """

    model_id: str

    def select_tool(
        self,
        *,
        context_text: str,
        palette: ToolPalette,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ) -> ReasonerToolCall:
        """Pick exactly one :data:`ReasonerToolCall` for ``context_text``.

        Args:
            context_text: The reasoner-built structured text snapshot
                (no pixels). Built by
                :class:`~openral_reasoner.context.ContextRenderer`.
            palette: The currently-valid tool palette built from the
                local rSkill registry filtered by
                :class:`RobotCapabilities`. The LLM is restricted to
                skill ids that appear in
                :attr:`ToolPalette.execute_rskill_ids`.
            system_prompt: System message; defaults to
                :data:`DEFAULT_SYSTEM_PROMPT`.

        Returns:
            A validated :data:`ReasonerToolCall` variant.

        Raises:
            ROSReasonerInvalidPlan: When the provider returns a payload
                that fails Pydantic discriminator validation, or names
                a rskill_id outside the palette.
            ROSPlanningError: For any other provider-side failure
                (timeout, transport error, malformed response).
        """

    def describe_image(self, *, image_jpeg: bytes, question: str) -> str:
        """Ask the model a free-text question about a single camera frame.

        Encodes ``image_jpeg`` as base64 and sends ONE chat/messages
        request with the image + ``question`` as user content; returns
        the model's text answer (stripped). No tool-use, no streaming.

        Args:
            image_jpeg: JPEG-encoded image bytes.
            question: Natural-language question about the frame (e.g.
                "Is the cup placed on the coaster?").

        Returns:
            The model's text answer, stripped of leading/trailing
            whitespace.  For reasoning models that surface their answer
            in a ``reasoning`` field with an empty ``content``, the
            ``reasoning`` text is returned instead.

        Raises:
            ROSConfigError: When the SDK is not installed.
            ROSPlanningError: On transport / provider failure.
        """


# OpenRouter's OpenAI-compatible base URL; pre-filled by the
# ``OPENRAL_REASONER_LLM_PROVIDER=openrouter`` shortcut so users don't
# have to memorise it. An explicit ``OPENRAL_REASONER_LLM_BASE_URL``
# still wins (so a proxy / staging gateway can be substituted).
OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"

# Local Ollama's OpenAI-compatible base URL; pre-filled by the
# ``OPENRAL_REASONER_LLM_PROVIDER=ollama`` shortcut. Ollama does not
# enforce auth by default, so the ``ollama`` provider also drops the
# API-key requirement that ``anthropic`` / ``openrouter`` insist on.
# A hosted Ollama-compatible gateway with auth can still set
# ``OPENRAL_REASONER_LLM_API_KEY`` and ``OPENRAL_REASONER_LLM_BASE_URL``
# to override both defaults.
OLLAMA_BASE_URL: str = "http://localhost:11434/v1"

# Local vLLM's OpenAI-compatible base URL; pre-filled by the
# ``OPENRAL_REASONER_LLM_PROVIDER=vllm`` shortcut (``vllm serve`` listens
# on ``:8000`` by default). Like Ollama it is a self-hosted endpoint that
# does not enforce auth unless started with ``--api-key``; the preset
# therefore drops the API-key requirement, and an explicit
# ``OPENRAL_REASONER_LLM_{BASE_URL,API_KEY}`` still overrides both.
VLLM_BASE_URL: str = "http://localhost:8000/v1"

# Named cloud presets reachable through Google / xAI / DeepSeek's own
# OpenAI-compatible endpoints. Each is a thin convenience on top of the
# generic ``openai-compatible`` client: the ``PROVIDER=<name>`` shortcut
# pre-fills the base URL below so users don't hand-configure it. All
# three enforce auth, so the factory requires
# ``OPENRAL_REASONER_LLM_API_KEY`` (an explicit
# ``OPENRAL_REASONER_LLM_BASE_URL`` still wins for a proxy / gateway).
# Gemini's OpenAI-compat shim lives under a ``/v1beta/openai/`` path.
GEMINI_BASE_URL: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
XAI_BASE_URL: str = "https://api.x.ai/v1"
DEEPSEEK_BASE_URL: str = "https://api.deepseek.com"
# Hugging Face's OpenAI-compatible inference router (serverless / provider-routed
# models, e.g. ``Qwen/Qwen3-8B``). Auth via an HF access token.
HUGGINGFACE_BASE_URL: str = "https://router.huggingface.co/v1"

# Auth-required named presets that wrap :class:`OpenAICompatibleToolUseClient`
# with a pre-filled base URL. ``openrouter`` is the original member;
# ``gemini`` / ``xai`` / ``deepseek`` / ``huggingface`` mirror it for the direct
# vendor endpoints. Keyed by PROVIDER value → default base URL.
_OPENAI_COMPATIBLE_PRESETS: dict[str, str] = {
    "openrouter": OPENROUTER_BASE_URL,
    "gemini": GEMINI_BASE_URL,
    "xai": XAI_BASE_URL,
    "deepseek": DEEPSEEK_BASE_URL,
    "huggingface": HUGGINGFACE_BASE_URL,
}

# Providers whose endpoint rejects ``tool_choice="required"`` and only honours
# ``"auto"``/``"none"`` (the HF router returns 400 INVALID_TOOL_CHOICE on
# ``required``). The client falls back to ``"auto"`` for these and retries once
# with an explicit nudge if the model answers in prose instead of a tool call.
_AUTO_TOOL_CHOICE_PROVIDERS: frozenset[str] = frozenset({"huggingface"})

# Local self-hosted OpenAI-compatible presets. Same wrapper as the cloud
# presets above, but these point at a loopback daemon and do NOT require
# an API key (the endpoint enforces none by default). Keyed by PROVIDER
# value → default base URL; the factory also gives this class a longer
# default timeout to absorb a cold-model first call.
_LOCAL_OPENAI_COMPATIBLE_PRESETS: dict[str, str] = {
    "ollama": OLLAMA_BASE_URL,
    "vllm": VLLM_BASE_URL,
}

# All accepted PROVIDER values; surfaced in error messages and reused by
# the ``openral doctor`` reasoner check so the two stay in sync.
_KNOWN_PROVIDERS: frozenset[str] = frozenset(
    {
        "anthropic",
        "openai-compatible",
        *_LOCAL_OPENAI_COMPATIBLE_PRESETS,
        *_OPENAI_COMPATIBLE_PRESETS,
    },
)


def build_tool_use_client_from_env() -> ToolUseClient:
    """Read ``OPENRAL_REASONER_LLM_*`` env and build the matching client.

    Recognised env vars:

    * ``OPENRAL_REASONER_LLM_PROVIDER`` — one of ``anthropic`` /
      ``openai-compatible`` / ``openrouter`` / ``ollama`` / ``vllm`` /
      ``gemini`` / ``xai`` / ``deepseek`` / ``huggingface``. Required.
    * ``OPENRAL_REASONER_LLM_MODEL`` — provider-specific model id.
      Required.
    * ``OPENRAL_REASONER_LLM_API_KEY`` — provider API key. Required for
      ``anthropic`` and the auth-required cloud presets (``openrouter`` /
      ``gemini`` / ``xai`` / ``deepseek`` / ``huggingface``); ignored by
      local ``openai-compatible`` / ``ollama`` / ``vllm`` endpoints that
      don't enforce it.
    * ``OPENRAL_REASONER_LLM_BASE_URL`` — for ``openai-compatible`` and
      every preset. For ``openai-compatible`` it defaults to the OpenAI
      cloud; for the local presets it defaults to
      :data:`OLLAMA_BASE_URL` (``http://localhost:11434/v1``) /
      :data:`VLLM_BASE_URL` (``http://localhost:8000/v1``); for
      ``openrouter`` / ``gemini`` / ``xai`` / ``deepseek`` /
      ``huggingface`` it defaults to that vendor's OpenAI-compatible
      endpoint (:data:`OPENROUTER_BASE_URL`, :data:`GEMINI_BASE_URL`,
      :data:`XAI_BASE_URL`, :data:`DEEPSEEK_BASE_URL`,
      :data:`HUGGINGFACE_BASE_URL`). ``huggingface`` uses
      ``tool_choice="auto"`` (its router rejects ``"required"``).

    Returns:
        A constructed :class:`ToolUseClient`.

    Raises:
        ROSConfigError: When the required env vars are not set or the
            provider is unknown.

    Example:
        >>> # Real usage requires API credentials; see tests for the
        >>> # FakeToolUseClient counterpart that needs no env.
        >>> import os
        >>> _ = os.environ  # placeholder for doctest discovery
    """
    provider = os.environ.get("OPENRAL_REASONER_LLM_PROVIDER", "").strip().lower()
    if not provider:
        msg = (
            "OPENRAL_REASONER_LLM_PROVIDER is unset; "
            "set to one of 'anthropic' / 'openai-compatible' / 'openrouter' / 'ollama' / "
            "'vllm' / 'gemini' / 'xai' / 'deepseek' / 'huggingface' to enable the reasoner. "
            "The open-core path has no default — tests use FakeToolUseClient."
        )
        raise ROSConfigError(msg)
    model = os.environ.get("OPENRAL_REASONER_LLM_MODEL", "").strip()
    if not model:
        raise ROSConfigError(
            "OPENRAL_REASONER_LLM_MODEL is unset; required to construct a ToolUseClient.",
        )
    api_key = os.environ.get("OPENRAL_REASONER_LLM_API_KEY", "").strip() or None
    # Local self-hosted models do tool-use slowly on a cold first call
    # (qwen3:0.6b under Ollama needs ~30 s warm-up; vLLM can be mid-load);
    # the cloud providers stay tight on the 10 s default. An explicit env
    # wins for either side.
    timeout_env = os.environ.get("OPENRAL_REASONER_LLM_TIMEOUT_S", "").strip()
    # The HF router can cold-start a serverless model on the first call, so it
    # gets the same generous default as the local self-hosted presets.
    slow_first_call = provider in _LOCAL_OPENAI_COMPATIBLE_PRESETS or provider == "huggingface"
    default_timeout_s = 60.0 if slow_first_call else 10.0
    timeout_s = float(timeout_env) if timeout_env else default_timeout_s
    # Optional completion-token cap (OpenAI-compatible providers only). Unset →
    # the endpoint default; set it to fit a low-balance metered key or to bound
    # cost/latency (a tick needs just one tool call). The Anthropic path keeps
    # its own 1024 default.
    max_tokens_env = os.environ.get("OPENRAL_REASONER_LLM_MAX_TOKENS", "").strip()
    max_tokens = int(max_tokens_env) if max_tokens_env else None
    if provider == "anthropic":
        if api_key is None:
            raise ROSConfigError(
                "OPENRAL_REASONER_LLM_API_KEY is unset; required for provider=anthropic.",
            )
        return AnthropicToolUseClient(model_id=model, api_key=api_key, timeout_s=timeout_s)
    if provider == "openai-compatible":
        base_url = os.environ.get("OPENRAL_REASONER_LLM_BASE_URL", "").strip() or None
        return OpenAICompatibleToolUseClient(
            model_id=model,
            api_key=api_key,
            base_url=base_url,
            timeout_s=timeout_s,
            max_tokens=max_tokens,
        )
    if provider in _OPENAI_COMPATIBLE_PRESETS:
        # openrouter / gemini / xai / deepseek — thin auth-required presets
        # that pre-fill the vendor's OpenAI-compatible base URL. An explicit
        # OPENRAL_REASONER_LLM_BASE_URL still wins (proxy / staging gateway).
        if api_key is None:
            raise ROSConfigError(
                f"OPENRAL_REASONER_LLM_API_KEY is unset; required for provider={provider}.",
            )
        base_url = (
            os.environ.get("OPENRAL_REASONER_LLM_BASE_URL", "").strip()
            or _OPENAI_COMPATIBLE_PRESETS[provider]
        )
        return OpenAICompatibleToolUseClient(
            model_id=model,
            api_key=api_key,
            base_url=base_url,
            timeout_s=timeout_s,
            tool_choice="auto" if provider in _AUTO_TOOL_CHOICE_PROVIDERS else "required",
            max_tokens=max_tokens,
        )
    if provider in _LOCAL_OPENAI_COMPATIBLE_PRESETS:
        # ollama / vllm — local self-hosted OpenAI-compatible servers whose
        # endpoint does not enforce auth by default; an explicit env value
        # still passes through for users who front them with a gateway (or
        # `vllm serve --api-key`) that does.
        base_url = (
            os.environ.get("OPENRAL_REASONER_LLM_BASE_URL", "").strip()
            or _LOCAL_OPENAI_COMPATIBLE_PRESETS[provider]
        )
        return OpenAICompatibleToolUseClient(
            model_id=model,
            api_key=api_key,
            base_url=base_url,
            timeout_s=timeout_s,
            max_tokens=max_tokens,
        )
    raise ROSConfigError(
        f"OPENRAL_REASONER_LLM_PROVIDER={provider!r} is unknown; "
        f"expected one of {sorted(_KNOWN_PROVIDERS)!r}.",
    )


# ── Concrete provider clients ─────────────────────────────────────────────────


_PER_SKILL_TOOL_PREFIX: str = "execute_rskill__"
# Anthropic + OpenAI tool names match ^[a-zA-Z0-9_-]{1,64}$ — keep the slug at or
# under this length so the LLM API accepts it.
_LLM_TOOL_NAME_MAX_LEN: int = 64


def _skill_id_to_tool_name(rskill_id: str) -> str:
    """Slugify a HF Hub skill id into a 64-char-max LLM tool name.

    HF Hub ids are ``<owner>/<repo>``; ``/`` and ``.`` are the only chars
    outside the Anthropic / OpenAI tool-name regex in canonical ids.
    Long ids get an 8-char sha1 suffix so the slug stays unique after
    truncation.

    >>> _skill_id_to_tool_name("OpenRAL/rskill-act-aloha")
    'execute_rskill__OpenRAL__rskill-act-aloha'
    """
    import hashlib  # noqa: PLC0415  # reason: stdlib, only used on the slow palette-build path

    slug = rskill_id.replace("/", "__").replace(".", "_")
    candidate = f"{_PER_SKILL_TOOL_PREFIX}{slug}"
    if len(candidate) <= _LLM_TOOL_NAME_MAX_LEN:
        return candidate
    # Reserve 9 chars for "_" + 8-char sha1 suffix to disambiguate post-truncation.
    h = hashlib.sha1(rskill_id.encode("utf-8")).hexdigest()[:8]
    max_slug = _LLM_TOOL_NAME_MAX_LEN - len(_PER_SKILL_TOOL_PREFIX) - 9
    return f"{_PER_SKILL_TOOL_PREFIX}{slug[:max_slug]}_{h}"


def _format_skill_tool_description(entry: RSkillToolEntry) -> str:
    """Render a skill's metadata into an LLM-facing tool description.

    The LLM scores tools primarily on description text. Lead with the
    canonical id (so the LLM has a stable handle), follow with the
    skill's NL description, then the structured action verbs + objects
    + scenes so the LLM can disambiguate similar skills.
    """
    parts = [f"Execute rSkill `{entry.rskill_id}`.", entry.description.strip()]
    if entry.actions:
        parts.append("Actions: " + ", ".join(a.value for a in entry.actions) + ".")
    if entry.objects:
        parts.append("Objects: " + ", ".join(entry.objects) + ".")
    if entry.scenes:
        parts.append("Scenes: " + ", ".join(entry.scenes) + ".")
    return " ".join(parts)


def _tool_palette_to_anthropic_tools(palette: ToolPalette) -> list[dict[str, object]]:  # noqa: PLR0912  # reason: a flat one-branch-per-optional-tool-group table is clearer than nesting
    """Render the palette as Anthropic ``tools`` schemas.

    ADR-0022: when ``palette.skills`` is populated, the LLM gets one
    ``execute_rskill__<slug>`` tool per skill — each with the skill's NL
    description + action verbs + objects + scenes. The ``rskill_id``
    field is dropped from the per-skill ``input_schema`` because the
    tool name already identifies the skill; the decoder fills
    ``rskill_id`` back in by reverse-lookup before validation.

    Palettes that only carry ``execute_rskill_ids`` (no per-skill
    metadata — synthetic test palettes, the default empty palette in
    ``reasoner_node``) fall back to a single ``execute_rskill`` tool
    with the ids embedded in its description.
    """
    # Lazy import — pydantic is already a hard dep; importing here keeps
    # ``ToolPalette`` symbol-free in non-reasoner code paths.
    from openral_core import (  # noqa: PLC0415
        EmitPromptTool,
        ExecuteRskillTool,
        LifecycleTransitionTool,
        ReloadGstPipelineTool,
    )

    tools: list[dict[str, object]] = []

    if palette.skills:
        execute_schema = ExecuteRskillTool.model_json_schema()
        per_skill_schema = _drop_property(execute_schema, "rskill_id")
        for entry in palette.skills:
            # ADR-0026 — when the manifest declares ``goal_params_schema``,
            # replace the per-skill tool's ``goal_params_json`` property
            # (default ``{"type": "string"}``) with the skill's actual
            # JSON Schema. The provider's structured-output / tool-use
            # path then guides the LLM to emit a well-formed object;
            # ``_decode_tool_payload`` JSON-stringifies it back to the
            # wire-format ``str`` field before constructing
            # :class:`ExecuteRskillTool`. Skills without a schema fall
            # through to the freeform string surface (today's behaviour).
            tool_schema = per_skill_schema
            if entry.goal_params_schema is not None:
                tool_schema = _replace_property_schema(
                    per_skill_schema,
                    "goal_params_json",
                    entry.goal_params_schema,
                )
            tools.append(
                {
                    "name": _skill_id_to_tool_name(entry.rskill_id),
                    "description": _format_skill_tool_description(entry),
                    "input_schema": tool_schema,
                },
            )
    elif palette.execute_rskill_ids:
        # Palettes carrying only ids (no per-skill metadata — synthetic
        # test palettes, the default empty palette) collapse to a
        # single execute_rskill tool with the ids enumerated in its
        # description. reason: keeps the LLM tool surface non-empty
        # for tests that don't load real RSkillManifest fixtures.
        tools.append(
            {
                "name": "execute_rskill",
                "description": (
                    "Invoke an installed rSkill. Allowed rskill_id values: "
                    f"{sorted(palette.execute_rskill_ids)!r}."
                ),
                "input_schema": ExecuteRskillTool.model_json_schema(),
            },
        )

    for kind, cls, description in (
        (
            "reload_gst_pipeline",
            ReloadGstPipelineTool,
            "Swap a sensor's GStreamer pipeline at runtime. "
            "pipeline_yaml must be a valid SensorReaderConfig YAML body.",
        ),
        (
            "lifecycle_transition",
            LifecycleTransitionTool,
            "Drive a peer ROS node through a lifecycle transition.",
        ),
        (
            "emit_prompt",
            EmitPromptTool,
            "Publish a PromptStamped onto another topic for cascades / operator messaging.",
        ),
    ):
        tools.append(
            {"name": kind, "description": description, "input_schema": cls.model_json_schema()},
        )

    # ADR-0039 — the two read-only spatial-memory query tools are surfaced only
    # when the reasoner_node has a SpatialMemory query backend wired (Phase 2);
    # they read the ADR-0038 scene graph and hold no actuation authority.
    if palette.spatial_memory_available:
        from openral_core import RecallObjectTool, ResolvePlaceTool  # noqa: PLC0415

        query_tools: tuple[tuple[str, type[BaseModel], str], ...] = (
            (
                "recall_object",
                RecallObjectTool,
                "Read-only: recall a remembered object from spatial memory by name/description. "
                "Returns its map-frame pose, a camera-facing approach viewpoint, and any "
                "occluding container that must be opened first. No actuation.",
            ),
            (
                "resolve_place",
                ResolvePlaceTool,
                "Read-only: resolve a place/room/agent reference (e.g. 'the kitchen', "
                "'where I was standing') to a navigation goal pose plus a traversable path. "
                "No actuation.",
            ),
        )
        for q_kind, q_cls, q_description in query_tools:
            tools.append(
                {
                    "name": q_kind,
                    "description": q_description,
                    "input_schema": q_cls.model_json_schema(),
                },
            )

    # ADR-0043 — the read-only live-detector query tool is surfaced only when a
    # detector exposes the /openral/perception/locate_in_view service. Unlike
    # recall_object (which recalls *remembered* objects from spatial memory), this
    # asks a live VLM detector to look at the current frame now. No actuation.
    if palette.detector_available:
        from openral_core import LocateInViewTool  # noqa: PLC0415

        description = (
            "Read-only: ask a live camera-mounted VLM detector whether an object is "
            "visible in the CURRENT frame right now (open-vocabulary). Complements "
            "recall_object (which only recalls remembered objects): use locate_in_view to "
            "verify what the robot can see this instant. Optionally name a 'camera' to "
            "pick a viewpoint; empty uses the default camera. No actuation."
        )
        # ADR-0056 — when several on-demand locators are in the graph, let the LLM
        # choose one by alias via the 'detector' field (empty = the default).
        if palette.on_demand_detectors:
            options = "; ".join(
                f"{d.alias} ({d.description})" if d.description else d.alias
                for d in palette.on_demand_detectors
            )
            description += (
                " Choose a locator with the optional 'detector' field (empty = default): "
                f"{options}. Prefer a light real-time locator for simple 'find X' and a "
                "grounding VLM for complex / attribute-qualified referring expressions."
            )
        # ADR-0051 — when continuous background detectors are running, tell the LLM
        # what they already cover so it reserves this on-demand locator for the long
        # tail (novel / attribute-qualified objects outside the always-on bank).
        if palette.continuous_detectors:
            coverage = "; ".join(
                f"{d.rskill_id} ({d.num_labels} classes — {', '.join(d.objects)})"
                if d.objects
                else f"{d.rskill_id} ({d.num_labels} classes)"
                for d in palette.continuous_detectors
            )
            description += (
                " These object classes are ALREADY tracked continuously in world state by: "
                f"{coverage}. For objects within that coverage, prefer reading world state / "
                "recall_object instead of calling locate_in_view; use locate_in_view for "
                "objects outside it (novel, highly specific, or attribute-qualified)."
            )

        tools.append(
            {
                "name": "locate_in_view",
                "description": description,
                "input_schema": LocateInViewTool.model_json_schema(),
            },
        )

    # ADR-0047 — the read-only scene-VLM query tool is surfaced only when a scene
    # VLM exposes the /openral/perception/query_scene service. Unlike
    # locate_in_view (which localizes an object and returns boxes), this answers
    # open-ended questions about the scene's state — task progress and
    # success/failure verification for the replanning ladder. No actuation.
    if palette.scene_query_available:
        from openral_core import QuerySceneTool  # noqa: PLC0415

        tools.append(
            {
                "name": "query_scene",
                "description": (
                    "Read-only: ask a scene vision-language model an open-ended QUESTION about "
                    "the CURRENT camera view and get a free-text answer. Use this to verify task "
                    "state and progress — e.g. 'Has the robot grasped the red mug?', 'Is the bowl "
                    "on the shelf?', 'Did we drop the object?', 'Is the table clear?'. This is NOT "
                    "a localizer: to find WHERE an object is, use locate_in_view instead. "
                    "Optionally name a 'camera' to pick a viewpoint; empty uses the default "
                    "camera. No actuation."
                ),
                "input_schema": QuerySceneTool.model_json_schema(),
            },
        )

    # ADR-0057 — the read-only reward-monitor tool is surfaced only when a reward
    # rSkill (Robometer-4B NF4) exposes the /openral/perception/query_task_progress
    # service. Where query_scene returns free text, this returns a quantitative
    # windowed progress/success assessment of the current task. No actuation.
    if palette.task_progress_available:
        from openral_core import QueryTaskProgressTool  # noqa: PLC0415

        tools.append(
            {
                "name": "query_task_progress",
                "description": (
                    "Read-only: ask the reward monitor how the CURRENT task is going. Returns a "
                    "quantitative assessment over the last 'window_s' seconds: normalized "
                    "progress (0-1) and success probability (0-1) now, their trends, and a "
                    "'stalled' flag. Use this to decide whether the policy is succeeding, "
                    "stalling (replan), or done. For open-ended scene questions use query_scene "
                    "instead. Optionally override 'task'; empty reuses the active goal. "
                    "No actuation."
                ),
                "input_schema": QueryTaskProgressTool.model_json_schema(),
            },
        )

    # ADR-0072 §3 — the self-maintained MEMORY.md tools are surfaced only when the
    # reasoner_node has a MEMORY.md wired (memory_md_path param). The reasoner
    # already READS current memory every tick (the ## MEMORY context block); these
    # add the WRITE path (memory_write — the reasoner's first actuation-free
    # write-capable tool) and archival recall (memory_search). Advisory only — a
    # wrong memory yields a bad plan the C++ safety kernel still vetoes.
    if palette.memory_available:
        from openral_core import MemorySearchTool, MemoryWriteTool  # noqa: PLC0415

        tools.append(
            {
                "name": "memory_write",
                "description": (
                    "Persist a durable fact to the robot's self-maintained semantic memory "
                    "(MEMORY.md). Use SPARINGLY for facts useful across tasks/sessions: a user "
                    "PREFERENCE ('clothes go in the bedroom drawer'), a learned LESSON / "
                    "correction ('grasp mugs by the handle'), a durable HOME-MAP fact, a "
                    "long-lived OBJECT-LOCATION, or an OPEN-TASK commitment. Pick 'section'. Ops: "
                    "'add' a new fact; 'update' to replace a 'target' fact's text in place; "
                    "'supersede' when a fact CHANGED (keeps the old one as a stale search hint); "
                    "'delete' a wrong/obsolete 'target'. Set 'importance' (0-1). Do NOT store "
                    "transient world state (live poses, battery) — that lives in world state. "
                    "No actuation."
                ),
                "input_schema": MemoryWriteTool.model_json_schema(),
            },
        )
        tools.append(
            {
                "name": "memory_search",
                "description": (
                    "Read-only: recall facts from the memory ARCHIVE (superseded / deleted "
                    "entries no longer in the live ## MEMORY block) by keyword. Current memory is "
                    "always in context already; use this only to retrieve an older fact you lost — "
                    "e.g. where an object USED to be. Optionally restrict to a 'section'. "
                    "No actuation."
                ),
                "input_schema": MemorySearchTool.model_json_schema(),
            },
        )

    # ADR-0073 amendment (#123) — the typed path for the decompose-mission
    # playbook to write the ## MISSION task queue. Always available: it is a core
    # reasoner capability with no resident-resource dependency (unlike the
    # reward/scene/memory tools above). Edits the S2 ledger only — no actuation.
    from openral_core import DecomposeMissionTool  # noqa: PLC0415

    tools.append(
        {
            "name": "decompose_mission",
            "description": (
                "Write the reasoner's task queue (the ## MISSION ledger). Two modes by "
                "'target_task_id': set it to the ACTIVE task's id (e.g. 't2') to break a "
                "too-coarse or STALLED task into finer ordered 'subtasks' — the task is "
                "flat-spliced in place by its children and you continue on the first child. "
                "Leave 'target_task_id' empty to replace the WHOLE queue with a better "
                "decomposition of the operator goal (only before any task has been attempted). "
                "Subdivision is depth-bounded (a task split twice is handed off, not split "
                "again). Use this instead of burning execute_rskill attempts on a task one "
                "skill cannot finish. No actuation — it only edits the task ledger."
            ),
            "input_schema": DecomposeMissionTool.model_json_schema(),
        },
    )
    return tools


def _drop_property(schema: dict[str, object], name: str) -> dict[str, object]:
    """Return a copy of a JSON Schema dict with ``name`` removed from properties + required."""
    out: dict[str, object] = dict(schema)
    props = out.get("properties")
    if isinstance(props, dict) and name in props:
        out["properties"] = {k: v for k, v in props.items() if k != name}
    required = out.get("required")
    if isinstance(required, list) and name in required:
        out["required"] = [r for r in required if r != name]
    return out


def _replace_property_schema(
    schema: dict[str, object],
    name: str,
    replacement: dict[str, Any],
) -> dict[str, object]:
    """Return a copy of ``schema`` with ``properties[name]`` swapped for ``replacement``.

    ADR-0026 — used to splice each rSkill's manifest-declared
    ``goal_params_schema`` into the per-skill LLM tool's
    ``goal_params_json`` slot. When the rSkill's manifest carries no
    schema (the common case for VLAs), this helper is not called and
    the LLM sees the freeform ``"type": "string"`` default. The
    replacement schema is forwarded verbatim — the manifest is the
    source of truth.
    """
    out: dict[str, object] = dict(schema)
    props = out.get("properties")
    if not isinstance(props, dict) or name not in props:
        return out
    new_props = dict(props)
    new_props[name] = replacement
    out["properties"] = new_props
    return out


def _decode_tool_payload(
    *,
    tool_name: str,
    arguments: dict[str, object],
    palette: ToolPalette,
) -> ReasonerToolCall:
    """Validate a provider's tool-call payload against the union + palette.

    ADR-0022: per-skill tool names (``execute_rskill__<slug>``) are
    resolved back to the canonical ``execute_rskill`` discriminator
    here, with ``rskill_id`` looked up from
    :attr:`ToolPalette.skills`. The LLM's own ``rskill_id`` (if any) is
    overridden by the lookup result so the tool name is the authority.
    """
    resolved_name = tool_name
    resolved_args = dict(arguments)
    if tool_name.startswith(_PER_SKILL_TOOL_PREFIX) and palette.skills:
        slug_lookup = {_skill_id_to_tool_name(s.rskill_id): s.rskill_id for s in palette.skills}
        if tool_name not in slug_lookup:
            raise ROSReasonerInvalidPlan(
                f"LLM returned per-skill tool {tool_name!r} but no matching skill "
                f"in palette (allowed: {sorted(slug_lookup)!r})",
            )
        resolved_args["rskill_id"] = slug_lookup[tool_name]
        resolved_name = "execute_rskill"

    # ADR-0026 — when the manifest declared ``goal_params_schema``, the
    # per-skill LLM tool's ``goal_params_json`` slot carries a structured
    # JSON Schema and the provider returns a parsed object (dict). The
    # Pydantic field is typed as ``str`` so the wire payload stays JSON;
    # re-serialise here before constructing ``ExecuteRskillTool``. When
    # the LLM already emitted a string (provider's choice, or no schema
    # was declared), leave it untouched.
    if isinstance(resolved_args.get("goal_params_json"), dict):
        resolved_args["goal_params_json"] = json.dumps(
            resolved_args["goal_params_json"], separators=(",", ":"), sort_keys=True
        )

    payload = dict(resolved_args)
    payload["tool"] = resolved_name
    try:
        call = _TOOL_ADAPTER.validate_python(payload)
    except ValidationError as exc:
        raise ROSReasonerInvalidPlan(
            f"LLM returned an invalid {tool_name!r} payload: {exc}",
        ) from exc
    if call.tool == "execute_rskill" and call.rskill_id not in palette.execute_rskill_ids:
        raise ROSReasonerInvalidPlan(
            f"LLM picked rskill_id={call.rskill_id!r} which is not in the palette "
            f"(allowed: {sorted(palette.execute_rskill_ids)!r})",
        )
    return call


class AnthropicToolUseClient:
    """Anthropic SDK-backed :class:`ToolUseClient` (ADR-0018 F4).

    Lazy-imports the ``anthropic`` Python SDK at first use so this
    module is importable on hosts without the SDK installed. Pulls
    structured tool selections via ``Anthropic.messages.create`` with
    a ``tools=[...]`` payload derived from the active
    :class:`ToolPalette`.

    Args:
        model_id: Anthropic model identifier (e.g. ``claude-opus-4-7``,
            ``claude-sonnet-4-6``).
        api_key: Anthropic API key. Required.
        max_tokens: Hard cap on response tokens; defaults to 1024
            (tool-call payloads are small).
        timeout_s: Per-call wall-clock timeout in seconds. Defaults to
            10 s, matching the F4 tick budget.

    Raises:
        ROSConfigError: When ``api_key`` is empty.
    """

    def __init__(
        self,
        *,
        model_id: str,
        api_key: str,
        max_tokens: int = 1024,
        timeout_s: float = 10.0,
    ) -> None:
        """Stash configuration; no SDK import until :meth:`select_tool`."""
        if not api_key:
            raise ROSConfigError("AnthropicToolUseClient: api_key is required")
        self.model_id = model_id
        self._api_key = api_key
        self._max_tokens = max_tokens
        self._timeout_s = timeout_s

    def select_tool(
        self,
        *,
        context_text: str,
        palette: ToolPalette,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ) -> ReasonerToolCall:
        """Call Anthropic and decode the resulting tool payload."""
        try:
            import anthropic  # noqa: PLC0415  # reason: optional cloud dep
        except ImportError as exc:
            raise ROSConfigError(
                "AnthropicToolUseClient requires the `anthropic` SDK; "
                "install with `uv add anthropic --package openral-reasoner` "
                "or pick OPENRAL_REASONER_LLM_PROVIDER=openai-compatible.",
            ) from exc
        client = anthropic.Anthropic(api_key=self._api_key, timeout=self._timeout_s)
        tools = _tool_palette_to_anthropic_tools(palette)
        try:
            response = client.messages.create(  # type: ignore[call-overload]  # reason: provider SDK boundary — tools/messages/tool_choice are heterogeneous TypedDicts that mypy cannot reconcile through dict literals; runtime payload is validated by the SDK
                model=self.model_id,
                system=system_prompt,
                max_tokens=self._max_tokens,
                tools=tools,
                tool_choice={"type": "any"},  # force exactly one tool call
                messages=[{"role": "user", "content": context_text}],
            )
        except Exception as exc:  # reason: provider SDK boundary
            raise ROSPlanningError(f"Anthropic call failed: {exc!s}") from exc
        for block in response.content:
            if getattr(block, "type", None) == "tool_use":
                return _decode_tool_payload(
                    tool_name=block.name,
                    arguments=dict(block.input),
                    palette=palette,
                )
        raise ROSReasonerInvalidPlan(
            "Anthropic response did not contain a tool_use block",
        )

    def describe_image(self, *, image_jpeg: bytes, question: str) -> str:
        """Ask the Anthropic model a free-text question about a camera frame.

        Sends one ``messages.create`` request with a base64-encoded image
        content block and the question; returns the text answer.  No tool-use,
        no streaming.

        Args:
            image_jpeg: JPEG-encoded image bytes.
            question: Natural-language question about the frame.

        Returns:
            Model's text answer, stripped.  Falls back to the ``reasoning``
            field when ``content`` is empty (thinking-mode models).

        Raises:
            ROSConfigError: When the ``anthropic`` SDK is not installed.
            ROSPlanningError: On transport / provider failure.
        """
        try:
            import anthropic  # noqa: PLC0415  # reason: optional cloud dep
        except ImportError as exc:
            raise ROSConfigError(
                "AnthropicToolUseClient requires the `anthropic` SDK; "
                "install with `uv add anthropic --package openral-reasoner`.",
            ) from exc
        client = anthropic.Anthropic(api_key=self._api_key, timeout=self._timeout_s)
        b64 = base64.b64encode(image_jpeg).decode()
        try:
            response = client.messages.create(
                model=self.model_id,
                max_tokens=self._max_tokens,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": b64,
                                },
                            },
                            {"type": "text", "text": question},
                        ],
                    }
                ],
            )
        except Exception as exc:  # reason: provider SDK boundary
            raise ROSPlanningError(f"Anthropic describe_image failed: {exc!s}") from exc
        # Extract text; fall back to reasoning field for thinking/reasoning models.
        text: str = ""
        if response.content:
            text = getattr(response.content[0], "text", "") or ""
        if not text:
            text = getattr(response, "reasoning", "") or ""
        return text.strip()


class OpenAICompatibleToolUseClient:
    """OpenAI-compatible SDK-backed :class:`ToolUseClient`.

    Wraps the ``openai`` Python SDK pointed at any
    OpenAI-protocol-compatible endpoint — cloud OpenAI, vLLM, Ollama
    (with the OpenAI shim), llama-server, etc. The choice of endpoint
    is a deployment config knob; this client itself imposes no cloud
    lock-in.

    Args:
        model_id: Model identifier as understood by the target endpoint
            (e.g. ``gpt-4o`` for cloud OpenAI, ``qwen2.5-7b-instruct``
            for a local vLLM).
        api_key: API key, or ``None`` for endpoints that don't enforce
            auth (the openai SDK still requires a non-empty string —
            we substitute ``"local"``).
        base_url: Endpoint base URL. Defaults to
            ``https://api.openai.com/v1`` when ``None``.
        timeout_s: Per-call wall-clock timeout in seconds. Defaults to 10 s.
        tool_choice: OpenAI ``tool_choice`` mode. ``"required"`` (the default)
            forces exactly one tool call per tick — the reasoner's contract.
            Some endpoints (the HF router) reject ``"required"`` and only honour
            ``"auto"``; pass ``"auto"`` for those and the client retries once
            with an explicit nudge if the model replies in prose instead.
        max_tokens: Optional hard cap on completion tokens per call. ``None``
            (the default) sends no cap, letting the endpoint apply its own —
            but reasoning models (GPT-5.x) default that to their full window
            (65k), which a metered gateway (OpenRouter) *reserves* up front and
            may reject on a low-balance key (HTTP 402). Set a cap (env
            ``OPENRAL_REASONER_LLM_MAX_TOKENS``) to bound cost/latency; a tick
            only needs one tool call, so a few thousand tokens (plus reasoning
            headroom) suffices.
    """

    def __init__(
        self,
        *,
        model_id: str,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout_s: float = 10.0,
        tool_choice: str = "required",
        max_tokens: int | None = None,
    ) -> None:
        """Stash configuration; no SDK import until :meth:`select_tool`."""
        self.model_id = model_id
        self._api_key = api_key
        self._base_url = base_url
        self._timeout_s = timeout_s
        self._tool_choice = tool_choice
        self._max_tokens = max_tokens

    def select_tool(
        self,
        *,
        context_text: str,
        palette: ToolPalette,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ) -> ReasonerToolCall:
        """Call the OpenAI-compatible endpoint and decode the tool call."""
        try:
            from openai import OpenAI  # noqa: PLC0415  # reason: optional cloud dep
        except ImportError as exc:
            raise ROSConfigError(
                "OpenAICompatibleToolUseClient requires the `openai` SDK; "
                "install with `uv add openai --package openral-reasoner`.",
            ) from exc
        client = OpenAI(
            api_key=self._api_key or "local",
            base_url=self._base_url,
            timeout=self._timeout_s,
        )
        tools = [
            {"type": "function", "function": {**spec, "parameters": spec.pop("input_schema")}}
            for spec in _tool_palette_to_anthropic_tools(palette)
        ]
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": context_text},
        ]
        # Omit ``max_tokens`` entirely when unset so the endpoint keeps its own
        # default; only a configured cap is sent (bounds a metered gateway's
        # up-front token reservation — see __init__).
        cap: dict[str, int] = {} if self._max_tokens is None else {"max_tokens": self._max_tokens}
        try:
            response = client.chat.completions.create(  # type: ignore[call-overload]  # reason: provider SDK boundary — tools/messages are heterogeneous TypedDicts (ChatCompletionMessageParam, ChatCompletionToolParam) that mypy cannot reconcile through dict literals; runtime payload is validated by the SDK
                model=self.model_id,
                messages=messages,
                tools=tools,
                tool_choice=self._tool_choice,
                **cap,
            )
            choices = list(response.choices)
            no_tool_call = not choices or not choices[0].message.tool_calls
            if no_tool_call and self._tool_choice != "required":
                # ``tool_choice="auto"`` endpoints (HF router) may answer in prose;
                # nudge once before giving up so the tick is not wasted.
                messages.append(
                    {"role": "user", "content": "You must respond with exactly one tool call now."},
                )
                response = client.chat.completions.create(  # type: ignore[call-overload]  # reason: provider SDK boundary (see above)
                    model=self.model_id,
                    messages=messages,
                    tools=tools,
                    tool_choice=self._tool_choice,
                    **cap,
                )
                choices = list(response.choices)
        except Exception as exc:  # reason: provider SDK boundary
            raise ROSPlanningError(f"OpenAI-compatible call failed: {exc!s}") from exc
        if not choices or not choices[0].message.tool_calls:
            raise ROSReasonerInvalidPlan(
                "OpenAI-compatible response did not contain a tool_calls block",
            )
        call = choices[0].message.tool_calls[0]
        raw_arguments = call.function.arguments or "{}"
        # A weak / cheap model can emit tool-call ``arguments`` that aren't a
        # single clean JSON object (trailing tokens, two concatenated objects,
        # a bare list). Surface it as ROSReasonerInvalidPlan — core.tick catches
        # ROSPlanningError and the reasoner_node feeds the hint back into the
        # next prompt's ``## EXECUTION`` section — instead of letting a raw
        # JSONDecodeError (or a downstream dict() TypeError) crash the node.
        try:
            arguments = json.loads(raw_arguments)
        except json.JSONDecodeError as exc:
            raise ROSReasonerInvalidPlan(
                f"tool {call.function.name!r} returned malformed JSON arguments "
                f"({exc!s}); emit a single valid JSON object: {raw_arguments!r}",
            ) from exc
        if not isinstance(arguments, dict):
            raise ROSReasonerInvalidPlan(
                f"tool {call.function.name!r} returned JSON arguments of type "
                f"{type(arguments).__name__}; emit a single JSON object: {raw_arguments!r}",
            )
        return _decode_tool_payload(
            tool_name=call.function.name,
            arguments=arguments,
            palette=palette,
        )

    def describe_image(self, *, image_jpeg: bytes, question: str) -> str:
        """Ask an OpenAI-compatible model a free-text question about a frame.

        Sends one ``chat.completions.create`` request with the question as a
        text block and the image as an ``image_url`` data-URI block; returns
        the text answer.  No tool-use, no streaming.

        Args:
            image_jpeg: JPEG-encoded image bytes.
            question: Natural-language question about the frame.

        Returns:
            Model's text answer, stripped.  Falls back to the ``reasoning``
            field when ``message.content`` is empty (observed with reasoning
            models such as ``z-ai/glm-5.2`` on OpenRouter).

        Raises:
            ROSConfigError: When the ``openai`` SDK is not installed.
            ROSPlanningError: On transport / provider failure.
        """
        try:
            from openai import OpenAI  # noqa: PLC0415  # reason: optional cloud dep
        except ImportError as exc:
            raise ROSConfigError(
                "OpenAICompatibleToolUseClient requires the `openai` SDK; "
                "install with `uv add openai --package openral-reasoner`.",
            ) from exc
        client = OpenAI(
            api_key=self._api_key or "local",
            base_url=self._base_url,
            timeout=self._timeout_s,
        )
        b64 = base64.b64encode(image_jpeg).decode()
        messages: list[dict[str, object]] = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                    },
                ],
            }
        ]
        try:
            response = client.chat.completions.create(
                model=self.model_id,
                messages=messages,  # type: ignore[arg-type]  # reason: provider SDK boundary — image_url content blocks use heterogeneous TypedDicts that mypy cannot reconcile through dict literals; runtime payload is validated by the SDK
            )
        except Exception as exc:  # reason: provider SDK boundary
            raise ROSPlanningError(f"OpenAI-compatible describe_image failed: {exc!s}") from exc
        message = response.choices[0].message
        text: str = message.content or ""
        if not text:
            text = getattr(message, "reasoning", "") or ""
        return text.strip()
