"""Typed tool-call dispatch — typed LLM tool-use client + concrete provider implementations.

Every reasoner tick the LLM picks exactly one of the four
``ReasonerToolCall`` variants (ExecuteRskill,
ReloadGstPipeline, LifecycleTransition, EmitPrompt) and the reasoner
node routes it onto the ROS graph.

Ships:

- ``ToolUseClient`` — structural Protocol every provider satisfies.
- ``AnthropicToolUseClient`` — wraps the Anthropic SDK's tool-use
  API (curated model, ``dialect="anthropic"``); lazy-imported.
- ``OpenAICompatibleToolUseClient`` — wraps the OpenAI SDK against
  any OpenAI-compatible endpoint (cloud OpenAI, local vLLM,
  Ollama-OpenAI, …); curated ``dialect="openai"`` or the uncurated
  escape hatch; lazy-imported.
- ``build_tool_use_client_from_env`` — factory over the
  model-first deployment env and ``openral_core.REASONER_MODELS``;
  defaults to "no model configured", endpoint independently overrideable.

Per CLAUDE.md §1.11, ``FakeToolUseClient``
(``tests.integration.fakes.fake_llm``) is the only test double
allowed at this process boundary.
"""

from __future__ import annotations

import base64
import functools
import hashlib
import json
import os
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Protocol, cast, runtime_checkable

import structlog
from openral_core import (
    ANTHROPIC_BASE_URL,
    DEEPSEEK_BASE_URL,
    GEMINI_BASE_URL,
    HUGGINGFACE_BASE_URL,
    OLLAMA_BASE_URL,
    OPENAI_BASE_URL,
    OPENROUTER_BASE_URL,
    REASONER_ENDPOINT_PRESETS,
    REASONER_MANAGED_ENDPOINT,
    REASONER_MODELS,
    VLLM_BASE_URL,
    XAI_BASE_URL,
    ReasonerEndpointPreset,
    ReasonerModel,
    ReasonerToolCall,
    RobotCapabilities,
    WaitTool,
    parse_reasoner_tool_choice,
)
from openral_core.exceptions import (
    ROSConfigError,
    ROSPlanningError,
    ROSReasonerInvalidPlan,
)
from pydantic import BaseModel, TypeAdapter, ValidationError
from pydantic.json_schema import GenerateJsonSchema, JsonSchemaValue
from pydantic_core import core_schema

if TYPE_CHECKING:
    from openral_reasoner.palette import RSkillToolEntry, ToolPalette

__all__ = [
    "ANTHROPIC_BASE_URL",
    "DEEPSEEK_BASE_URL",
    "DEFAULT_SYSTEM_PROMPT",
    "GEMINI_BASE_URL",
    "HUGGINGFACE_BASE_URL",
    "OLLAMA_BASE_URL",
    "OPENAI_BASE_URL",
    "OPENROUTER_BASE_URL",
    "SYSTEM_PROMPT_ENV_VAR",
    "VLLM_BASE_URL",
    "XAI_BASE_URL",
    "AnthropicToolUseClient",
    "OpenAICompatibleToolUseClient",
    "ToolUseClient",
    "build_tool_use_client_from_env",
    "render_robot_context_prompt",
    "resolve_max_skill_tools",
    "resolve_reasoner_system_prompt",
]

log = structlog.get_logger(__name__)

# Default system prompt — the robot-agnostic operating brief. Factual and
# intentionally non-anthropomorphising: the reasoner is a scheduler, not a
# chatbot. ``ReasonerNode`` composes the live prompt via
# ``resolve_reasoner_system_prompt``, which lets a deployment replace
# this base via the ``OPENRAL_REASONER_SYSTEM_PROMPT`` env var and appends a
# per-robot ``## THIS ROBOT`` block (``render_robot_context_prompt``).
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
    # The continuous detector's `in_view` line is a FIXED
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
    # ── Poll the reward monitor to judge a running skill ────
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
    "cleanup), emit_prompt (publish a PromptStamped onto another "
    "topic — used to message the operator or trigger a cascade), and "
    "wait (deliberately do nothing this tick). "
    # ── When the right action is no action ────────────────────────────
    "When the snapshot shows a dispatched skill still in flight and "
    "progressing normally, or the mission is finished and no new goal has "
    "arrived, pick wait — do not invent busywork, re-query what you "
    "already know, or message the operator without new information. "
    # ── Grasp / place targets (reasoner names, perception grounds) ─────
    "grasp_target is OPTIONAL: the skill's policy decides which object it "
    "picks, so leave grasp_target unset unless you must pin one object. To pin "
    "one, name it by its perception label, and when memory has it, "
    "recall_object first and pass the instance's node_id as object_id "
    "(required when several objects share the label). Where to set an object "
    "down is the skill's job too: the surface under the carried payload is "
    "measured from the map, so place_target is optional — set it only to "
    "restrict the place to one labelled surface (a shelf, a table). Perception "
    "grounds what you "
    "name; a target that grounds to nothing or to several objects refuses the "
    "dispatch — an ungroundable place_target never falls back to placing "
    "anywhere, so omit it unless you can name a surface perception sees. "
    "Naming a target arms nothing by itself: the safety kernel only "
    "trusts the region its producer measures. "
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

    The base ``DEFAULT_SYSTEM_PROMPT`` is robot-agnostic. At reasoner
    lifecycle ``configure`` time the active robot's
    ``RobotCapabilities`` is known (from the
    ``robot_yaml`` manifest or the constructor), so the node calls this
    to give the LLM standing knowledge of the body it is driving: its
    embodiment tags, whether it can locomote (which gates the base
    prompt's "navigate to approach" rule), what manipulation / sensing
    hardware it has, its payload, and its control modes.

    The block is deterministic (fixed field order, sorted tag lists) so a
    given robot always renders the same prompt — reproducibility per
    CLAUDE.md §1.8. ``capabilities is None`` returns ``base_prompt``
    unchanged (the reasoner has no robot wired yet).

    Args:
        capabilities: The active robot's capabilities, or ``None`` when
            no robot manifest has been loaded.
        base_prompt: The system prompt to extend. Defaults to
            ``DEFAULT_SYSTEM_PROMPT``.

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
# set non-empty it replaces ``DEFAULT_SYSTEM_PROMPT``; the per-robot
# ``## THIS ROBOT`` block is still appended on top, so a custom brief keeps
# the factual body description it cannot hardcode. Honoured by
# ``resolve_reasoner_system_prompt`` (called from ``ReasonerNode``).
SYSTEM_PROMPT_ENV_VAR: str = "OPENRAL_REASONER_SYSTEM_PROMPT"


def resolve_reasoner_system_prompt(
    capabilities: RobotCapabilities | None,
    *,
    env: Mapping[str, str] | None = None,
) -> str:
    """Compose the reasoner system prompt from the env override + robot block.

    The base operating brief is ``DEFAULT_SYSTEM_PROMPT``, unless the
    deployment sets ``SYSTEM_PROMPT_ENV_VAR``
    (``OPENRAL_REASONER_SYSTEM_PROMPT``) to a non-empty value, which
    replaces it. The per-robot ``## THIS ROBOT`` block
    (``render_robot_context_prompt``) is then appended to whichever
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
    fully-validated ``ReasonerToolCall``; parsing /
    validation lives behind this Protocol.

    Attributes:
        model_id: Provider-specific identifier (``claude-opus-4-7``,
            ``gpt-4o``, ``qwen2.5-7b-instruct``, ...). Recorded on the
            reasoner span.

    Optional (not part of the Protocol, read via ``getattr``): an
    implementation that can see provider usage may expose
    ``last_prompt_tokens: int | None`` — the prompt tokens of its most recent
    ``select_tool`` call — and the reasoner records it on the tick span.
    """

    model_id: str

    def select_tool(
        self,
        *,
        context_text: str,
        palette: ToolPalette,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ) -> ReasonerToolCall:
        """Pick exactly one ``ReasonerToolCall`` for ``context_text``.

        Args:
            context_text: The reasoner-built structured text snapshot
                (no pixels). Built by
                ``ContextRenderer``.
            palette: The currently-valid tool palette built from the
                local rSkill registry filtered by
                ``RobotCapabilities``. The LLM is restricted to
                skill ids that appear in
                ``ToolPalette.execute_rskill_ids``.
            system_prompt: System message; defaults to
                ``DEFAULT_SYSTEM_PROMPT``.

        Returns:
            A validated ``ReasonerToolCall`` variant.

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


# Base URLs + named-endpoint presets live in ``openral_core`` next to
# ``REASONER_MODELS`` (imported at the top of this module and re-exported via
# ``__all__`` for back-compat — this module was their public home), so the
# factory and ``openral doctor`` consume the SAME table. Doctor used to
# hand-mirror it and the mirror drifted twice (commits 2fe732a, 131a489).


# ── Model-first env contract (ADR-0088) ───────────────────────────────────────
# Selection pivots from the location-leaking PROVIDER enum to the *model*; where
# it runs is the orthogonal ENDPOINT axis.
REASONER_MODEL_ENV: str = "OPENRAL_REASONER_MODEL"
REASONER_ENDPOINT_ENV: str = "OPENRAL_REASONER_ENDPOINT"
REASONER_API_KEY_ENV: str = "OPENRAL_REASONER_API_KEY"
REASONER_DIALECT_ENV: str = "OPENRAL_REASONER_DIALECT"
REASONER_MAX_TOKENS_ENV: str = "OPENRAL_REASONER_MAX_TOKENS"
REASONER_TIMEOUT_ENV: str = "OPENRAL_REASONER_TIMEOUT_S"
# Per-tick skill-tool cap override (``resolve_max_skill_tools``); ``0`` = no cap.
REASONER_MAX_SKILL_TOOLS_ENV: str = "OPENRAL_REASONER_MAX_SKILL_TOOLS"
# Escape hatch for an uncurated model on a bare URL (openai dialect) or on the
# anthropic dialect: some models refuse forced tool use outright (HTTP 400) —
# a "thinking mode" model behind an OpenAI-compatible gateway ("Thinking mode
# does not support this tool_choice", e.g. DeepSeek-V4.1-Flash), or the Claude 5
# generation ("tool_choice: type \"tool\" and \"any\" are not supported for
# this model", #522). The clients also downgrade to ``auto`` on that 400 at
# runtime; setting ``auto`` here skips the failed first call. Named openai
# endpoints and curated models carry their own value and ignore this var.
REASONER_TOOL_CHOICE_ENV: str = "OPENRAL_REASONER_TOOL_CHOICE"

# Completion cap for an uncurated anthropic-dialect model with no
# ``OPENRAL_REASONER_MAX_TOKENS``: a current Claude may think before it calls a
# tool, and thinking tokens count against ``max_tokens`` (a ceiling, not a
# reservation — Anthropic bills what is generated).
_ANTHROPIC_UNCURATED_MAX_TOKENS: int = 16000

# The status a provider answers a refused request parameter with.
_HTTP_BAD_REQUEST: int = 400

# Sent once when a ``tool_choice="auto"`` model answers in prose instead of
# calling a tool.
_NO_TOOL_CALL_NUDGE: str = (
    "You must respond with exactly one tool call now. "
    "If nothing should happen this tick, call `wait`."
)
# A prose reply that survives the nudge becomes a ``wait`` whose rationale is
# that text, cut to this many characters (it is a one-line trace field).
_NO_TOOL_CALL_RATIONALE_MAX_CHARS: int = 500


def _rejects_forced_tool_choice(exc: BaseException) -> bool:
    """True for a provider 400 that refuses the requested ``tool_choice``.

    Matches on the HTTP status and the parameter name, which every provider
    seen so far puts in the message (Anthropic: ``tool_choice: type "tool" and
    "any" are not supported for this model``; DeepSeek gateways: ``Thinking
    mode does not support this tool_choice``; HF router: ``INVALID_TOOL_CHOICE``).

    Example:
        >>> class _E(Exception):
        ...     status_code = 400
        >>> _rejects_forced_tool_choice(_E('tool_choice: type "any" is not supported'))
        True
        >>> _rejects_forced_tool_choice(_E("max_tokens too large"))
        False
    """
    return (
        getattr(exc, "status_code", None) == _HTTP_BAD_REQUEST and "tool_choice" in str(exc).lower()
    )


def _rejects_max_tokens(exc: BaseException) -> bool:
    """True for an OpenAI 400 that wants ``max_completion_tokens`` instead.

    OpenAI's own API refuses ``max_tokens`` on its reasoning models
    (``Unsupported parameter: 'max_tokens' is not supported with this model.
    Use 'max_completion_tokens' instead.``).

    Example:
        >>> class _E(Exception):
        ...     status_code = 400
        >>> _rejects_max_tokens(_E("Use 'max_completion_tokens' instead."))
        True
    """
    return getattr(
        exc, "status_code", None
    ) == _HTTP_BAD_REQUEST and "max_completion_tokens" in str(exc)


def _wait_for_prose_reply(text: str, *, model_id: str) -> WaitTool:
    """The no-op tick for a ``tool_choice="auto"`` reply with no tool call (#522).

    The model chose not to act; recording that as an explicit ``wait`` (its
    text as rationale) keeps the tick traceable instead of failing it.
    """
    rationale = " ".join(text.split())[:_NO_TOOL_CALL_RATIONALE_MAX_CHARS]
    log.warning(
        "reasoner.no_tool_call",
        model=model_id,
        detail="model replied without a tool call after one nudge; recorded as wait",
        text=rationale,
    )
    return WaitTool(rationale=rationale or "model replied without a tool call")


def _prompt_tokens(response: object) -> int | None:
    """Provider-reported prompt-token count, or ``None`` when unreported.

    Reads whichever usage fields the provider actually sent and sums them:
    OpenAI-compatible endpoints report ``prompt_tokens``; Anthropic splits the
    same quantity across ``input_tokens`` + the two ``cache_*_input_tokens``
    counters, and the reasoner enables prompt caching, so summing is what
    keeps the number comparable across providers and across ticks.

    ``ReasonerCore`` picks this up off the client
    (``last_prompt_tokens``) to record how prompt size tracks tick latency.
    """
    usage = getattr(response, "usage", None)
    total: int | None = None
    for name in (
        "prompt_tokens",
        "input_tokens",
        "cache_read_input_tokens",
        "cache_creation_input_tokens",
    ):
        value = getattr(usage, name, None)
        if isinstance(value, int):
            total = value if total is None else total + value
    return total


def _openai_choices(response: object) -> list[Any]:
    """``response.choices``, or a legible error when the provider sent none.

    An OpenAI-compatible *gateway* (e.g. OpenRouter) that fails upstream —
    rate-limited, 5xx, timeout — still answers HTTP 200 with ``choices:
    null`` and the real reason under a non-standard ``error`` key; the SDK
    models that as ``choices=None``, so a bare ``list(response.choices)``
    raises ``TypeError: 'NoneType' object is not iterable`` instead of
    surfacing "rate-limited upstream" (observed live on a free OpenRouter
    model, 2026-08-04).
    """
    choices = getattr(response, "choices", None)
    if choices is not None:
        return list(choices)
    detail = getattr(response, "error", None)
    raise ROSPlanningError(
        f"provider returned no choices — upstream error: {detail or '<none reported>'}"
    )


def _openai_tool_calls(choices: list[Any]) -> list[Any]:
    """The first choice's tool calls (empty when it answered in prose)."""
    if not choices:
        return []
    return list(getattr(choices[0].message, "tool_calls", None) or [])


def _openai_may_nudge(choices: list[Any]) -> bool:
    """Whether a tool-less reply is worth one nudge (not cut off or filtered)."""
    if not choices:
        return True
    return getattr(choices[0], "finish_reason", None) not in {"length", "content_filter"}


# Back-compat aliases: the class and table moved to openral_core (single
# source shared with `openral doctor`); every internal reference below keeps
# its historical name.
_EndpointPreset = ReasonerEndpointPreset
_ENDPOINT_PRESETS: dict[str, ReasonerEndpointPreset] = REASONER_ENDPOINT_PRESETS


def build_tool_use_client_from_env() -> ToolUseClient:
    """Build a ``ToolUseClient`` from the model-first env (ADR-0088).

    Env vars:

    * ``OPENRAL_REASONER_MODEL`` (**required**) — a curated registry key
      (``REASONER_MODELS``: ``claude-opus-4-8`` / ``claude-opus-5-5`` /
      ``claude-sonnet-5-5`` / ``claude-haiku-5-5`` / ``gpt-5.5`` / ``gpt-5.6``
      / ``cosmos3-edge``), or a raw model id for the uncurated
      escape hatch.
    * ``OPENRAL_REASONER_ENDPOINT`` (optional) — a named endpoint
      (``_ENDPOINT_PRESETS``: ``openai`` / ``openrouter`` / ``ollama`` / ``vllm`` /
      ``gemini`` / ``xai`` / ``deepseek`` / ``huggingface`` / ``anthropic``), a
      URL, the ``managed`` sentinel, or unset (the model's registry default).
      The only axis that carries local-vs-cloud, and it does so literally.
    * ``OPENRAL_REASONER_API_KEY`` (conditional) — required iff the resolved
      endpoint needs auth.
    * ``OPENRAL_REASONER_DIALECT`` (escape hatch only) — ``anthropic`` |
      ``openai``. Needed only for an uncurated model on a **bare URL**; a
      named endpoint already knows its wire format.
    * ``OPENRAL_REASONER_MAX_SKILL_TOOLS`` (optional) — read by
      ``resolve_max_skill_tools``, not by this builder: the per-tick skill-tool
      cap ``ReasonerCore`` applies.
    * ``OPENRAL_REASONER_{MAX_TOKENS,TIMEOUT_S}`` (optional) — override the
      registry defaults.
    * ``OPENRAL_REASONER_TOOL_CHOICE`` (escape hatch only) — overrides the
      ``tool_choice`` of an uncurated model on a bare URL (openai dialect) or
      on the anthropic dialect (``auto``; ``required`` is sent as Anthropic
      ``any``). Some models refuse forced tool use outright (Claude 5, thinking-
      mode gateways); the clients also downgrade on that 400 at runtime.
      Ignored for a named openai endpoint or a curated model — both already
      carry the correct value.

    A raw model id that is not in the registry takes the escape hatch: it
    requires ``OPENRAL_REASONER_ENDPOINT`` + ``OPENRAL_REASONER_DIALECT`` and
    logs an ``reasoner.model.uncurated`` warning (untested for robotics tool
    calling).

    Returns:
        A constructed ``ToolUseClient``.

    Raises:
        ROSConfigError: When no model is selected, an uncurated model lacks its
            endpoint/dialect, or a required API key is missing.

    Example:
        >>> # Real usage requires env + credentials; see tests for the
        >>> # FakeToolUseClient counterpart that needs no env.
        >>> import os
        >>> _ = os.environ  # placeholder for doctest discovery
    """
    model_key = os.environ.get(REASONER_MODEL_ENV, "").strip()
    if not model_key:
        raise ROSConfigError(
            f"{REASONER_MODEL_ENV} is unset; set it to a curated model "
            f"({sorted(REASONER_MODELS)}), or to a raw model id plus "
            f"{REASONER_ENDPOINT_ENV} (a named endpoint "
            f"{sorted(_ENDPOINT_PRESETS)}, or a URL plus {REASONER_DIALECT_ENV}) "
            "for an uncurated endpoint. The open-core path has no default — tests use "
            "FakeToolUseClient."
        )
    # Validated on every path, so a bad value fails at build even where it
    # would be ignored.
    tool_choice = parse_reasoner_tool_choice(os.environ.get(REASONER_TOOL_CHOICE_ENV))
    entry = REASONER_MODELS.get(model_key)
    if entry is None:
        return _build_uncurated_model(model_key, tool_choice)
    if tool_choice is not None:
        _warn_tool_choice_ignored(tool_choice, f"curated model {entry.id!r}")
    return _build_curated_model(entry)


def resolve_max_skill_tools(client: ToolUseClient) -> int | None:
    """The per-tick skill-tool cap for the reasoner model ``client`` talks to (#494).

    ``OPENRAL_REASONER_MAX_SKILL_TOOLS`` wins when set (``0`` = no cap, for
    benchmarking a curated model on its full palette, or capping an uncurated
    small model); otherwise the ``ReasonerModel.max_skill_tools`` of the curated
    entry whose ``served_model_id`` is ``client.model_id``; otherwise ``None``
    (every capability-matched skill is offered). Keyed on the client actually
    built (or injected), not on ``OPENRAL_REASONER_MODEL``, so the two cannot
    disagree.

    Args:
        client: The tool-use client the reasoner will call.

    Returns:
        The cap, or ``None`` for no cap.

    Raises:
        ROSConfigError: When the env value is not an integer or is negative.

    Example:
        >>> client = OpenAICompatibleToolUseClient(model_id="nvidia/Cosmos3-Edge")
        >>> import os
        >>> saved = os.environ.pop(REASONER_MAX_SKILL_TOOLS_ENV, None)
        >>> resolve_max_skill_tools(client)
        4
        >>> resolve_max_skill_tools(OpenAICompatibleToolUseClient(model_id="openai/gpt-5.5"))
        >>> if saved is not None:
        ...     os.environ[REASONER_MAX_SKILL_TOOLS_ENV] = saved
    """
    env = _env_int(REASONER_MAX_SKILL_TOOLS_ENV)
    if env is not None:
        if env < 0:
            raise ROSConfigError(f"{REASONER_MAX_SKILL_TOOLS_ENV}={env} must be >= 0 (0 = no cap).")
        return env or None
    return next(
        (
            entry.max_skill_tools
            for entry in REASONER_MODELS.values()
            if entry.served_model_id == client.model_id
        ),
        None,
    )


def _warn_tool_choice_ignored(tool_choice: str, why: str) -> None:
    """Log that ``OPENRAL_REASONER_TOOL_CHOICE`` is set but has no effect here."""
    log.warning(
        "reasoner.tool_choice_ignored",
        tool_choice=tool_choice,
        reason=f"{REASONER_TOOL_CHOICE_ENV} only applies to an uncurated model on a bare-URL "
        f"openai endpoint or the anthropic dialect; {why} carries its own value",
    )


def _env_float(name: str) -> float | None:
    """A float env override, or ``None`` when unset/empty.

    Raises:
        ROSConfigError: When set but not parseable — a typo in an env var is
            a configuration error, not an untyped ``ValueError`` traceback.
    """
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError as exc:
        raise ROSConfigError(f"{name}={raw!r} is not a number.") from exc


def _env_int(name: str) -> int | None:
    """An int env override, or ``None`` when unset/empty.

    Raises:
        ROSConfigError: When set but not parseable (see ``_env_float``).
    """
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError as exc:
        raise ROSConfigError(f"{name}={raw!r} is not an integer.") from exc


def _reasoner_timeout_s(hosting: str) -> float:
    """Per-call timeout: env override, else a hosting-aware default.

    ``managed_local`` gets 120 s (an on-device model's first inference after
    server boot compiles kernels); cloud/byo get the tight 10 s default.
    """
    env = _env_float(REASONER_TIMEOUT_ENV)
    if env is not None:
        return env
    return 120.0 if hosting == "managed_local" else 10.0


def _reasoner_max_tokens(entry: ReasonerModel) -> int | None:
    """Completion cap: env override, else the model's registry default."""
    env = _env_int(REASONER_MAX_TOKENS_ENV)
    if env is not None:
        return env
    return entry.max_tokens_default


def _build_curated_model(entry: ReasonerModel) -> ToolUseClient:
    """Build the client for a registry model from its resolved properties."""
    api_key = os.environ.get(REASONER_API_KEY_ENV, "").strip() or None
    raw_endpoint = os.environ.get(REASONER_ENDPOINT_ENV, "").strip() or None
    # A named endpoint resolves the same way as on the uncurated path
    # (otherwise `ENDPOINT=ollama` would be handed to the SDK as a literal
    # URL). It contributes its ENDPOINT properties — url, tool_choice,
    # cold-start timeout, auth posture — while the registry keeps the MODEL
    # properties: served id, dialect, token cap ("the endpoint is the
    # orthogonal axis"). Using only `preset.url` drops the other four: e.g.
    # `gpt-5.5` on the HF router built with `tool_choice="required"` (a 400)
    # and a cold `ollama` on the 10 s cloud timeout instead of 60 s.
    preset = _ENDPOINT_PRESETS.get(raw_endpoint.lower()) if raw_endpoint else None
    if preset is not None and preset.dialect != entry.dialect:
        # e.g. claude-opus-4-8 (anthropic) on `ollama` (openai): building an
        # Anthropic client against an OpenAI-only server configures cleanly
        # and then fails on every single tick.
        raise ROSConfigError(
            f"{REASONER_ENDPOINT_ENV}={raw_endpoint!r} speaks the "
            f"{preset.dialect!r} dialect but model {entry.id!r} speaks "
            f"{entry.dialect!r}; a named endpoint cannot re-dialect a curated "
            f"model. Use a URL for a proxy that translates."
        )
    endpoint_override = preset.url if preset is not None else raw_endpoint
    env_timeout = _env_float(REASONER_TIMEOUT_ENV)
    timeout_s = (
        env_timeout
        if env_timeout is not None
        else (preset.timeout_s if preset is not None else _reasoner_timeout_s(entry.hosting))
    )
    max_tokens = _reasoner_max_tokens(entry)

    if entry.dialect == "anthropic":
        if api_key is None:
            raise ROSConfigError(
                f"{REASONER_API_KEY_ENV} is unset; required for model {entry.id!r}."
            )
        return AnthropicToolUseClient(
            model_id=entry.served_model_id,
            api_key=api_key,
            base_url=endpoint_override,
            max_tokens=max_tokens or 1024,
            timeout_s=timeout_s,
            tool_choice=entry.tool_choice,
        )

    if entry.hosting == "managed_local":
        return _build_managed_local(entry, api_key, endpoint_override, timeout_s, max_tokens)

    # openai dialect, cloud or byo_local. A named endpoint states its own auth
    # posture; a bare URL means the operator owns the endpoint, so auth is not
    # forced there.
    base_url = endpoint_override or entry.default_endpoint
    auth_required = preset.auth_required if preset is not None else entry.auth_required
    if auth_required and api_key is None and (preset is not None or raw_endpoint is None):
        raise ROSConfigError(
            f"{REASONER_API_KEY_ENV} is unset; required for model {entry.id!r} "
            f"on endpoint {base_url}."
        )
    return OpenAICompatibleToolUseClient(
        # `openai/gpt-5.5` on OpenRouter is `gpt-5.5` on OpenAI's own API.
        model_id=(
            preset.served_model_id(entry.served_model_id)
            if preset is not None
            else entry.served_model_id
        ),
        api_key=api_key,
        base_url=base_url,
        timeout_s=timeout_s,
        tool_choice=preset.tool_choice if preset is not None else entry.tool_choice,
        max_tokens=max_tokens,
        max_tokens_param=preset.max_tokens_param if preset is not None else "max_tokens",
    )


def _build_managed_local(
    entry: ReasonerModel,
    api_key: str | None,
    endpoint_override: str | None,
    timeout_s: float,
    max_tokens: int | None,
) -> ToolUseClient:
    """Build the managed-local (autostart) client for a ``managed_local`` model.

    Today the only such model is ``cosmos3-edge`` → ``Cosmos3ToolUseClient``.
    An explicit endpoint URL (anything but the ``managed`` sentinel) is a
    bring-your-own server: autostart disengages inside the client for
    non-loopback / explicit URLs.
    """
    from openral_reasoner.cosmos3 import (  # noqa: PLC0415  # reason: avoid a module-level import cycle (cosmos3 subclasses OpenAICompatibleToolUseClient)
        AUTOSTART_ENV,
        BOOT_TIMEOUT_ENV,
        COSMOS3_BASE_URL,
        DEFAULT_BOOT_TIMEOUT_S,
        Cosmos3ToolUseClient,
    )

    if endpoint_override and endpoint_override != REASONER_MANAGED_ENDPOINT:
        base_url = endpoint_override
    else:
        base_url = COSMOS3_BASE_URL
    auto_start = os.environ.get(AUTOSTART_ENV, "").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }
    boot_env = os.environ.get(BOOT_TIMEOUT_ENV, "").strip()
    return Cosmos3ToolUseClient(
        model_id=entry.served_model_id,
        api_key=api_key,
        base_url=base_url,
        timeout_s=timeout_s,
        max_tokens=max_tokens,
        auto_start=auto_start,
        boot_timeout_s=float(boot_env) if boot_env else DEFAULT_BOOT_TIMEOUT_S,
    )


def _build_uncurated_model(model_key: str, tool_choice: str | None) -> ToolUseClient:
    """Escape hatch: a non-registry model id + a named or raw endpoint.

    ``OPENRAL_REASONER_ENDPOINT`` takes either a ``_ENDPOINT_PRESETS`` name
    (``openrouter`` / ``ollama`` / ``vllm`` / ``anthropic`` / …), which carries
    its own URL, dialect, auth posture and cold-start timeout, or a bare URL —
    which nothing can classify, so that spelling still needs
    ``OPENRAL_REASONER_DIALECT``. An explicit dialect always wins, so a preset
    fronted by a translating proxy stays configurable.

    Warns that the model is untested for robotics tool calling (ADR-0088);
    nothing is blocked, the curated menu is simply first-class.
    """
    raw_endpoint = os.environ.get(REASONER_ENDPOINT_ENV, "").strip()
    dialect = os.environ.get(REASONER_DIALECT_ENV, "").strip().lower()
    preset = _ENDPOINT_PRESETS.get(raw_endpoint.lower())
    endpoint: str | None
    if preset is not None:
        endpoint, dialect = preset.url, dialect or preset.dialect
    else:
        endpoint = raw_endpoint or None
    if endpoint is None or dialect not in {"anthropic", "openai"}:
        raise ROSConfigError(
            f"{REASONER_MODEL_ENV}={model_key!r} is not a curated model "
            f"({sorted(REASONER_MODELS)}); for an uncurated endpoint set "
            f"{REASONER_ENDPOINT_ENV} to a named endpoint "
            f"({sorted(_ENDPOINT_PRESETS)}) or to a URL plus "
            f"{REASONER_DIALECT_ENV} (anthropic|openai)."
        )
    log.warning(
        "reasoner.model.uncurated",
        model=model_key,
        detail=(
            f"{model_key!r} is not on the curated robotics tool-calling list "
            f"({sorted(REASONER_MODELS)}); tool-call reliability is unverified."
        ),
    )
    api_key = os.environ.get(REASONER_API_KEY_ENV, "").strip() or None
    if preset is not None and preset.auth_required and api_key is None:
        raise ROSConfigError(
            f"{REASONER_API_KEY_ENV} is unset; required for "
            f"{REASONER_ENDPOINT_ENV}={raw_endpoint!r}."
        )
    env_timeout = _env_float(REASONER_TIMEOUT_ENV)
    timeout_s = (
        env_timeout
        if env_timeout is not None
        else (preset.timeout_s if preset is not None else _reasoner_timeout_s("byo_local"))
    )
    max_tokens = _env_int(REASONER_MAX_TOKENS_ENV)
    if dialect == "anthropic":
        if api_key is None:
            raise ROSConfigError(
                f"{REASONER_API_KEY_ENV} is unset; required for dialect=anthropic."
            )
        # The anthropic preset cannot know whether this model accepts forced
        # tool use (Claude 5 does not, #522), so the env override applies here.
        return AnthropicToolUseClient(
            model_id=model_key,
            api_key=api_key,
            base_url=endpoint,
            max_tokens=max_tokens or _ANTHROPIC_UNCURATED_MAX_TOKENS,
            timeout_s=timeout_s,
            tool_choice=tool_choice or (preset.tool_choice if preset is not None else "required"),
        )
    if preset is not None and tool_choice is not None:
        _warn_tool_choice_ignored(tool_choice, f"named endpoint {raw_endpoint!r}")
    return OpenAICompatibleToolUseClient(
        model_id=model_key,
        api_key=api_key,
        base_url=endpoint,
        timeout_s=timeout_s,
        tool_choice=(preset.tool_choice if preset is not None else (tool_choice or "required")),
        max_tokens=max_tokens,
        max_tokens_param=preset.max_tokens_param if preset is not None else "max_tokens",
    )


# ── Concrete provider clients ─────────────────────────────────────────────────


_PER_SKILL_TOOL_PREFIX: str = "execute_rskill__"
# Anthropic + OpenAI tool names match ^[a-zA-Z0-9_-]{1,64}$ — keep the slug at or
# under this length so the LLM API accepts it.
_LLM_TOOL_NAME_MAX_LEN: int = 64


def _skill_id_to_tool_name(rskill_id: str) -> str:
    """Map a HF Hub skill id to a collision-resistant LLM tool name.

    HF Hub ids are ``<owner>/<repo>``; ``/`` and ``.`` are the only chars
    outside the Anthropic / OpenAI tool-name regex in canonical ids. Every
    name gets an 8-char sha1 suffix because the readable character mapping is
    not injective (``a.b`` and ``a_b`` both slug to ``a_b``).

    >>> _skill_id_to_tool_name("OpenRAL/rskill-qwen35_4b-any-general-nf4")
    'execute_rskill__OpenRAL__rskill-qwen35_4b-any-general-n_9bf15b4a'
    """
    slug = rskill_id.replace("/", "__").replace(".", "_")
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


# The argument type of ``GenerateJsonSchema.field_title_should_be_set``; pydantic
# defines it only in a private module, so it is spelled out here.
_CoreSchemaOrField = (
    core_schema.CoreSchema
    | core_schema.ModelField
    | core_schema.DataclassField
    | core_schema.TypedDictField
    | core_schema.ComputedField
)


class _LLMSchemaGenerator(GenerateJsonSchema):
    """Emit tool schemas without the parts only a Python reader needs.

    Drops the auto-generated ``title`` of every field, model and enum, and every
    model / enum docstring ``description``; field descriptions are untouched. An
    explicit ``Field(title=...)`` and dataclass / TypedDict schemas keep theirs —
    the reasoner tools use neither, and a unit test fails any reasoner tool
    variant whose schema carries a ``title``.
    """

    def field_title_should_be_set(self, schema: _CoreSchemaOrField) -> bool:
        """Never title a field."""
        return False

    def model_schema(self, schema: core_schema.ModelSchema) -> JsonSchemaValue:
        """A model's schema without its title or docstring."""
        out = super().model_schema(schema)
        out.pop("title", None)
        out.pop("description", None)
        return out

    def enum_schema(self, schema: core_schema.EnumSchema) -> JsonSchemaValue:
        """An enum's schema without its title or docstring."""
        out = super().enum_schema(schema)
        out.pop("title", None)
        out.pop("description", None)
        return out


@functools.cache
def _llm_schema_json(model: type[BaseModel]) -> str:
    """The stripped schema as JSON, built once per class; a str cannot be mutated."""
    return json.dumps(model.model_json_schema(schema_generator=_LLMSchemaGenerator))


def _llm_schema(model: type[BaseModel]) -> dict[str, Any]:
    """``model.model_json_schema()`` minus what only a Python reader needs.

    Pydantic copies each model's docstring into the schema as its
    ``description`` and adds a ``title`` to every node. Both ride along on
    every tool of every tick, and a docstring is written for developers
    (``Attributes:``, ``Example:``, design-note references), not the LLM.
    Shipping them made each per-skill tool ~9.4K characters, a 44K-token
    ``franka_panda`` palette, and Cosmos3-Edge tool-call runaways (issue #440).
    What the LLM needs to know about a field lives in its
    ``Field(description=...)``, which is kept; every field of a reasoner tool
    must carry one (enforced in ``tests/unit/test_tool_use_wire_payload.py``).

    Built once per model class and cached as an immutable JSON string; each
    call returns a fresh dict, since callers splice per-skill properties into it.

    >>> from openral_core import ExecuteRskillTool
    >>> s = _llm_schema(ExecuteRskillTool)
    >>> "title" in s or "description" in s or "description" in s["$defs"]["GraspTargetRef"]
    False
    >>> "description" in s["properties"]["patience_s"]
    True
    >>> s is _llm_schema(ExecuteRskillTool)
    False
    """
    return cast("dict[str, Any]", json.loads(_llm_schema_json(model)))


def _tool_palette_to_anthropic_tools(palette: ToolPalette) -> list[dict[str, object]]:  # noqa: PLR0912  # reason: a flat one-branch-per-optional-tool-group table is clearer than nesting
    """Render the palette as Anthropic ``tools`` schemas.

    When ``palette.skills`` is populated, the LLM gets one
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
        WaitTool,
    )

    tools: list[dict[str, object]] = []

    if palette.skills:
        execute_schema = _llm_schema(ExecuteRskillTool)
        per_skill_schema = _drop_property(execute_schema, "rskill_id")
        for entry in palette.skills:
            # When the manifest declares ``goal_params_schema``,
            # replace the per-skill tool's ``goal_params_json`` property
            # (default ``{"type": "string"}``) with the skill's actual
            # JSON Schema. The provider's structured-output / tool-use
            # path then guides the LLM to emit a well-formed object;
            # ``_decode_tool_payload`` JSON-stringifies it back to the
            # wire-format ``str`` field before constructing
            # ``ExecuteRskillTool``. Skills without a schema fall
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
                "input_schema": _llm_schema(ExecuteRskillTool),
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
        (
            "wait",
            WaitTool,
            "Do nothing this tick — deliberately observe and wait. Pick this when no "
            "action is warranted right now: a dispatched skill is mid-execution and "
            "progress is nominal, the mission is finished and no new goal has arrived, "
            "or acting would just repeat a call whose outcome has not changed. Never "
            "pick it to avoid a decision the snapshot clearly demands.",
        ),
    ):
        schema = _llm_schema(cls)
        text = description
        if kind == "lifecycle_transition":
            # The decoder refuses any node outside ``palette.node_ids``, so the LLM is
            # given that set instead of guessing names from a free-text description.
            if palette.node_ids:
                schema["properties"]["node"]["enum"] = sorted(palette.node_ids)
            else:
                text += " No lifecycle peers are configured here: every call is refused."
        tools.append({"name": kind, "description": text, "input_schema": schema})

    # The two read-only spatial-memory query tools are surfaced only
    # when the reasoner_node has a SpatialMemory query backend wired (Phase 2);
    # they read the spatial-memory scene-graph and hold no actuation authority.
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
                    "input_schema": _llm_schema(q_cls),
                },
            )

    # The read-only live-detector query tool is surfaced only when a
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
            "pick a viewpoint, or leave it blank for the primary camera. No actuation."
        )
        # When several on-demand locators are in the graph, let the LLM
        # choose one by alias via the 'detector' field. "empty = the default"
        # reads as an instruction to type the word "default" — observed live
        # with an uncurated model, which then routed to a service literally
        # named /openral/perception/default/locate_in_view that never
        # existed. Spelled out unambiguously as "leave blank" instead; the
        # dispatcher (reasoner_node._dispatch_locate_in_view) also treats any
        # unrecognised alias as blank, so this wording fix reduces the
        # frequency of the mistake rather than being the only thing standing
        # between it and a stalled mission.
        if palette.on_demand_detectors:
            options = "; ".join(
                f"{d.alias} ({d.description})" if d.description else d.alias
                for d in palette.on_demand_detectors
            )
            description += (
                " Choose a locator by its alias in the optional 'detector' field, or leave it "
                f"blank to use the deployment's default: {options}. Prefer a light real-time "
                "locator for simple 'find X' and a grounding VLM for complex / "
                "attribute-qualified referring expressions."
            )
        # When continuous background detectors are running, tell the LLM
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
                "input_schema": _llm_schema(LocateInViewTool),
            },
        )

    # The read-only scene-VLM query tool is surfaced only when a scene
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
                    "Optionally name a 'camera' to pick a viewpoint, or leave it blank for the "
                    "primary camera. No actuation."
                ),
                "input_schema": _llm_schema(QuerySceneTool),
            },
        )

    # The read-only reward-monitor tool is surfaced only when a reward
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
                "input_schema": _llm_schema(QueryTaskProgressTool),
            },
        )

    # The self-maintained MEMORY.md tools are surfaced only when the
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
                "input_schema": _llm_schema(MemoryWriteTool),
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
                "input_schema": _llm_schema(MemorySearchTool),
            },
        )

    # The typed path for the decompose-mission
    # playbook to write the ## MISSION task queue (see #123). Always available: it is a core
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
            "input_schema": _llm_schema(DecomposeMissionTool),
        },
    )
    return tools


def _tool_palette_to_openai_tools(palette: ToolPalette) -> list[dict[str, object]]:
    """Render the palette as OpenAI ``tools`` function schemas.

    Same tool surface as ``_tool_palette_to_anthropic_tools``, re-keyed to
    the OpenAI wire shape: Anthropic's ``input_schema`` becomes the function's
    ``parameters`` and MUST NOT also ride along under its original key —
    strict endpoints 400 on the unknown field and lenient ones silently
    double every tool schema's token cost per call.

    >>> from openral_reasoner.palette import ToolPalette
    >>> palette = ToolPalette(execute_rskill_ids=frozenset())
    >>> specs = _tool_palette_to_openai_tools(palette)
    >>> all(set(t["function"]) == {"name", "description", "parameters"} for t in specs)
    True
    """
    tools: list[dict[str, object]] = []
    for spec in _tool_palette_to_anthropic_tools(palette):
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": spec["name"],
                    "description": spec["description"],
                    "parameters": spec["input_schema"],
                },
            }
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

    Used to splice each rSkill's manifest-declared
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

    Per-skill tool names (``execute_rskill__<slug>``) are
    resolved back to the canonical ``execute_rskill`` discriminator
    here, with ``rskill_id`` looked up from
    ``ToolPalette.skills``. The LLM's own ``rskill_id`` (if any) is
    overridden by the lookup result so the tool name is the authority.
    """
    resolved_name = tool_name
    resolved_args = dict(arguments)
    if tool_name.startswith(_PER_SKILL_TOOL_PREFIX) and palette.skills:
        slug_lookup = {
            _skill_id_to_tool_name(entry.rskill_id): entry.rskill_id for entry in palette.skills
        }
        if tool_name not in slug_lookup:
            raise ROSReasonerInvalidPlan(
                f"LLM returned per-skill tool {tool_name!r} but no matching skill "
                f"in palette (allowed: {sorted(slug_lookup)!r})",
            )
        resolved_args["rskill_id"] = slug_lookup[tool_name]
        resolved_name = "execute_rskill"

    # When the manifest declared ``goal_params_schema``, the
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
    if call.tool == "lifecycle_transition" and call.node not in palette.node_ids:
        raise ROSReasonerInvalidPlan(
            f"LLM picked lifecycle node={call.node!r} which is not in the palette "
            f"(allowed: {sorted(palette.node_ids)!r})",
        )
    return call


def _anthropic_response_text(response: Any) -> str:  # noqa: ANN401  # reason: provider SDK boundary — anthropic Message object, duck-typed
    """Concatenated text of every ``text`` content block, stripped.

    A thinking-enabled model leads with a ``thinking`` block, so reading only
    ``content[0].text`` returns ``""`` on exactly the responses that carry an
    answer later in the list — scan all blocks and keep the ``text`` ones.

    >>> class _Block:
    ...     def __init__(self, type_: str, text: str = "") -> None:
    ...         self.type, self.text = type_, text
    >>> class _Resp:
    ...     content = [_Block("thinking"), _Block("text", "Yes, it is done. ")]
    >>> _anthropic_response_text(_Resp())
    'Yes, it is done.'
    """
    parts = [
        getattr(block, "text", "") or ""
        for block in getattr(response, "content", None) or []
        if getattr(block, "type", None) == "text"
    ]
    return "\n".join(p for p in parts if p).strip()


def _anthropic_tool_use_block(response: Any) -> Any:  # noqa: ANN401  # reason: provider SDK boundary — anthropic content block, duck-typed
    """The first ``tool_use`` content block, or ``None``."""
    for block in getattr(response, "content", None) or []:
        if getattr(block, "type", None) == "tool_use":
            return block
    return None


def _anthropic_may_nudge(response: Any) -> bool:  # noqa: ANN401  # reason: provider SDK boundary — anthropic Message object
    """Whether a tool-less reply is worth one nudge (not a refusal or a cut-off)."""
    return getattr(response, "stop_reason", None) not in {"refusal", "max_tokens"}


class AnthropicToolUseClient:
    """Anthropic SDK-backed ``ToolUseClient`` (typed tool-call dispatch).

    Lazy-imports the ``anthropic`` Python SDK at first use so this
    module is importable on hosts without the SDK installed. Pulls
    structured tool selections via ``Anthropic.messages.create`` with
    a ``tools=[...]`` payload derived from the active
    ``ToolPalette``.

    Args:
        model_id: Anthropic model identifier (e.g. ``claude-opus-4-7``,
            ``claude-sonnet-4-6``).
        api_key: Anthropic API key. Required.
        base_url: Optional Anthropic-compatible endpoint override.
        max_tokens: Hard cap on response tokens; defaults to 1024
            (tool-call payloads are small).
        timeout_s: Per-call wall-clock timeout in seconds. Defaults to
            10 s, matching the F4 tick budget.
        tool_choice: ``"required"`` (the default) sends ``{"type": "any"}`` —
            exactly one tool call per tick. ``"auto"`` for models that refuse
            forced tool use (the Claude 5 generation, #522): a prose reply is
            nudged once, then recorded as a ``wait`` tick. A 400 that names
            ``tool_choice`` downgrades ``"required"`` to ``"auto"`` for the
            rest of the client's life, logged as
            ``reasoner.tool_choice_downgraded``.

    Raises:
        ROSConfigError: When ``api_key`` is empty or ``tool_choice`` is not
            ``"required"`` / ``"auto"``.
    """

    def __init__(
        self,
        *,
        model_id: str,
        api_key: str,
        base_url: str | None = None,
        max_tokens: int = 1024,
        timeout_s: float = 10.0,
        tool_choice: str = "required",
    ) -> None:
        """Stash configuration; no SDK import until ``select_tool``."""
        if not api_key:
            raise ROSConfigError("AnthropicToolUseClient: api_key is required")
        if tool_choice not in {"required", "auto"}:
            raise ROSConfigError(
                f"AnthropicToolUseClient: tool_choice={tool_choice!r}; expected required|auto"
            )
        self.model_id = model_id
        self._api_key = api_key
        self._base_url = base_url
        self._max_tokens = max_tokens
        self._timeout_s = timeout_s
        self._tool_choice = tool_choice
        self.last_prompt_tokens: int | None = None
        """Prompt tokens (cache reads included) the provider reported for the
        most recent ``select_tool`` call; read by
        ``ReasonerCore.run_prepared_llm``."""
        # SDK client built once on first use and reused across ticks —
        # rebuilding per call throws away the HTTP connection pool and pays
        # a fresh TLS handshake on every reasoner tick.
        self._sdk_client: Any = None
        # The rendered tool list for the last palette seen. The palette only
        # changes on a registry refresh (``set_palette`` installs a new object),
        # so rendering once per palette object instead of once per tick is exact.
        self._rendered: tuple[ToolPalette, list[dict[str, object]]] | None = None

    def _tools(self, palette: ToolPalette) -> list[dict[str, object]]:
        """Anthropic tool schemas for ``palette``, rendered once per palette object."""
        # Read the cache once and return the local: re-reading ``self._rendered``
        # after the write could hand back another palette's tools when two
        # ``select_tool`` calls overlap.
        rendered = self._rendered
        if rendered is None or rendered[0] is not palette:
            tools = _tool_palette_to_anthropic_tools(palette)
            # Prompt caching: the tool schemas + system brief are large and
            # static between ticks; a cache breakpoint after them lets Anthropic
            # reuse the prefix across the 0.2 Hz tick stream instead of
            # re-ingesting it every call. Only the per-tick context stays uncached.
            if tools:
                tools[-1]["cache_control"] = {"type": "ephemeral"}
            rendered = (palette, tools)
            self._rendered = rendered
        return rendered[1]

    def _client(self) -> Any:  # noqa: ANN401  # reason: provider SDK boundary — anthropic.Anthropic, lazy-imported
        """The cached ``anthropic.Anthropic`` instance (lazy import + construct)."""
        if self._sdk_client is None:
            try:
                import anthropic  # noqa: PLC0415  # reason: optional cloud dep
            except ImportError as exc:
                raise ROSConfigError(
                    "AnthropicToolUseClient requires the `anthropic` SDK; "
                    "install with `uv add anthropic --package openral-reasoner` "
                    "or select an openai-dialect ReasonerModel.",
                ) from exc
            self._sdk_client = anthropic.Anthropic(
                api_key=self._api_key,
                base_url=self._base_url,
                timeout=self._timeout_s,
            )
        return self._sdk_client

    def _create(self, client: Any, **kwargs: Any) -> Any:  # noqa: ANN401  # reason: provider SDK boundary — anthropic Message object
        """``messages.create`` with this client's ``tool_choice``, downgraded on refusal.

        A model that refuses forced tool use (Claude 5: HTTP 400 ``tool_choice:
        type "tool" and "any" are not supported for this model``) is retried
        once with ``auto``, and the client stays on ``auto`` from then on.
        """
        try:
            return client.messages.create(tool_choice=self._wire_tool_choice(), **kwargs)
        except Exception as exc:  # reason: provider SDK boundary; re-raised unless tool_choice
            if self._tool_choice == "auto" or not _rejects_forced_tool_choice(exc):
                raise
            log.warning(
                "reasoner.tool_choice_downgraded",
                model=self.model_id,
                tool_choice_from="required",
                tool_choice_to="auto",
                error=str(exc),
            )
            self._tool_choice = "auto"
            return client.messages.create(tool_choice=self._wire_tool_choice(), **kwargs)

    def _wire_tool_choice(self) -> dict[str, object]:
        """The Anthropic ``tool_choice`` block: ``any`` (forced) or ``auto``, one call."""
        return {
            "type": "any" if self._tool_choice == "required" else "auto",
            "disable_parallel_tool_use": True,
        }

    def select_tool(
        self,
        *,
        context_text: str,
        palette: ToolPalette,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ) -> ReasonerToolCall:
        """Call Anthropic and decode the resulting tool payload.

        Under ``tool_choice="auto"`` a reply with no ``tool_use`` block is
        nudged once; a second prose reply becomes a ``wait`` tick with that
        text as rationale (#522). A refusal or a reply cut off at
        ``max_tokens`` is a ``ROSPlanningError`` instead — neither is the
        model choosing to wait.
        """
        client = self._client()
        tools = self._tools(palette)
        system_blocks = [
            {
                "type": "text",
                "text": system_prompt,
                "cache_control": {"type": "ephemeral"},
            }
        ]
        content: list[dict[str, str]] = [{"type": "text", "text": context_text}]
        try:
            response = self._create(
                client,
                model=self.model_id,
                system=system_blocks,
                max_tokens=self._max_tokens,
                tools=tools,
                messages=[{"role": "user", "content": content}],
            )
            block = _anthropic_tool_use_block(response)
            if block is None and self._tool_choice == "auto" and _anthropic_may_nudge(response):
                response = self._create(
                    client,
                    model=self.model_id,
                    system=system_blocks,
                    max_tokens=self._max_tokens,
                    tools=tools,
                    messages=[
                        {
                            "role": "user",
                            "content": [*content, {"type": "text", "text": _NO_TOOL_CALL_NUDGE}],
                        }
                    ],
                )
                block = _anthropic_tool_use_block(response)
        except Exception as exc:  # reason: provider SDK boundary
            raise ROSPlanningError(f"Anthropic call failed: {exc!s}") from exc
        self.last_prompt_tokens = _prompt_tokens(response)
        if block is not None:
            return _decode_tool_payload(
                tool_name=block.name,
                arguments=dict(block.input),
                palette=palette,
            )
        stop_reason = getattr(response, "stop_reason", None)
        if stop_reason == "refusal":
            raise ROSPlanningError(
                f"Anthropic model {self.model_id!r} refused the tick "
                f"(stop_details={getattr(response, 'stop_details', None)!r})"
            )
        if stop_reason == "max_tokens":
            raise ROSPlanningError(
                f"Anthropic reply hit max_tokens={self._max_tokens} before a tool call; "
                f"raise {REASONER_MAX_TOKENS_ENV}"
            )
        if self._tool_choice == "auto":
            return _wait_for_prose_reply(_anthropic_response_text(response), model_id=self.model_id)
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
        client = self._client()
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
        return _anthropic_response_text(response)


class OpenAICompatibleToolUseClient:
    """OpenAI-compatible SDK-backed ``ToolUseClient``.

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
            with an explicit nudge if the model replies in prose instead, then
            records a second prose reply as a ``wait`` tick (#522). A 400 that
            names ``tool_choice`` downgrades ``"required"`` to ``"auto"`` for
            the rest of the client's life (``reasoner.tool_choice_downgraded``).
        max_tokens: Optional hard cap on completion tokens per call. ``None``
            (the default) sends no cap, letting the endpoint apply its own —
            but reasoning models (GPT-5.x) default that to their full window
            (65k), which a metered gateway (OpenRouter) *reserves* up front and
            may reject on a low-balance key (HTTP 402). Set a cap (env
            ``OPENRAL_REASONER_MAX_TOKENS``) to bound cost/latency; a tick
            only needs one tool call, so a few thousand tokens (plus reasoning
            headroom) suffices.
        max_tokens_param: Request field the cap is sent as — ``max_tokens``, or
            ``max_completion_tokens`` for OpenAI's own API, whose reasoning
            models refuse ``max_tokens``. A 400 asking for
            ``max_completion_tokens`` switches it at runtime
            (``reasoner.max_tokens_param_switched``).
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
        max_tokens_param: str = "max_tokens",
    ) -> None:
        """Stash configuration; no SDK import until ``select_tool``."""
        self.model_id = model_id
        self._api_key = api_key
        self._base_url = base_url
        self._timeout_s = timeout_s
        self._tool_choice = tool_choice
        self._max_tokens = max_tokens
        self._max_tokens_param = max_tokens_param
        self.last_prompt_tokens: int | None = None
        """Prompt tokens the endpoint reported for the most recent
        ``select_tool`` call; read by ``ReasonerCore.run_prepared_llm``."""
        # SDK client built once on first use and reused across ticks (see
        # AnthropicToolUseClient._client for the rationale).
        self._sdk_client: Any = None
        # Rendered tools for the last palette seen (see AnthropicToolUseClient._tools).
        self._rendered: tuple[ToolPalette, list[dict[str, object]]] | None = None

    def _tools(self, palette: ToolPalette) -> list[dict[str, object]]:
        """OpenAI tool schemas for ``palette``, rendered once per palette object."""
        rendered = self._rendered
        if rendered is None or rendered[0] is not palette:
            rendered = (palette, _tool_palette_to_openai_tools(palette))
            self._rendered = rendered
        return rendered[1]

    def _client(self) -> Any:  # noqa: ANN401  # reason: provider SDK boundary — openai.OpenAI, lazy-imported
        """The cached ``openai.OpenAI`` instance (lazy import + construct)."""
        if self._sdk_client is None:
            try:
                from openai import OpenAI  # noqa: PLC0415  # reason: optional cloud dep
            except ImportError as exc:
                raise ROSConfigError(
                    "OpenAICompatibleToolUseClient requires the `openai` SDK; "
                    "install with `uv add openai --package openral-reasoner`.",
                ) from exc
            self._sdk_client = OpenAI(
                api_key=self._api_key or "local",
                base_url=self._base_url,
                timeout=self._timeout_s,
            )
        return self._sdk_client

    def select_tool(
        self,
        *,
        context_text: str,
        palette: ToolPalette,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ) -> ReasonerToolCall:
        """Call the OpenAI-compatible endpoint and decode the tool call.

        Under ``tool_choice="auto"`` a reply with no tool call is nudged once;
        a second prose reply becomes a ``wait`` tick with that text as
        rationale (#522). A reply cut off at the token cap or by a content
        filter is a ``ROSPlanningError`` instead.
        """
        client = self._client()
        tools = self._tools(palette)
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": context_text},
        ]
        try:
            choices, response = self._complete(client, messages, tools)
            if (
                not _openai_tool_calls(choices)
                and self._tool_choice != "required"
                and _openai_may_nudge(choices)
            ):
                # ``tool_choice="auto"`` endpoints may answer in prose; nudge
                # once before giving up so the tick is not wasted.
                messages.append({"role": "user", "content": _NO_TOOL_CALL_NUDGE})
                choices, response = self._complete(client, messages, tools)
        except Exception as exc:  # reason: provider SDK boundary
            raise ROSPlanningError(f"OpenAI-compatible call failed: {exc!s}") from exc
        self.last_prompt_tokens = _prompt_tokens(response)
        if not _openai_tool_calls(choices):
            finish_reason = getattr(choices[0], "finish_reason", None) if choices else None
            if finish_reason == "length":
                raise ROSPlanningError(
                    f"OpenAI-compatible reply hit the token cap ({self._max_tokens}) "
                    f"before a tool call; raise {REASONER_MAX_TOKENS_ENV}"
                )
            if finish_reason == "content_filter":
                raise ROSPlanningError(
                    f"OpenAI-compatible model {self.model_id!r} refused the tick (content_filter)"
                )
            if self._tool_choice != "required" and choices:
                return _wait_for_prose_reply(
                    getattr(choices[0].message, "content", None) or "", model_id=self.model_id
                )
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

    def _complete(
        self,
        client: Any,  # noqa: ANN401  # reason: provider SDK boundary — openai.OpenAI
        messages: list[dict[str, str]],
        tools: list[dict[str, object]],
    ) -> tuple[list[Any], Any]:
        """One ``chat.completions.create``, adapting to what the endpoint refuses.

        Up to two runtime adaptations, each logged and kept for later ticks: a
        400 naming ``tool_choice`` downgrades ``"required"`` to ``"auto"``; a
        400 asking for ``max_completion_tokens`` sends the cap under that name.

        Returns:
            The response's choices and the response itself.
        """
        for _ in range(3):
            # Omit the cap entirely when unset so the endpoint keeps its own
            # default; only a configured cap is sent (bounds a metered
            # gateway's up-front token reservation — see __init__).
            cap: dict[str, int] = (
                {} if self._max_tokens is None else {self._max_tokens_param: self._max_tokens}
            )
            try:
                response = client.chat.completions.create(
                    model=self.model_id,
                    messages=messages,
                    tools=tools,
                    tool_choice=self._tool_choice,
                    **cap,
                )
            except Exception as exc:  # reason: provider SDK boundary; re-raised unless adaptable
                if self._tool_choice == "required" and _rejects_forced_tool_choice(exc):
                    log.warning(
                        "reasoner.tool_choice_downgraded",
                        model=self.model_id,
                        tool_choice_from="required",
                        tool_choice_to="auto",
                        error=str(exc),
                    )
                    self._tool_choice = "auto"
                    continue
                if cap and self._max_tokens_param == "max_tokens" and _rejects_max_tokens(exc):
                    log.warning(
                        "reasoner.max_tokens_param_switched",
                        model=self.model_id,
                        param_from="max_tokens",
                        param_to="max_completion_tokens",
                        error=str(exc),
                    )
                    self._max_tokens_param = "max_completion_tokens"
                    continue
                raise
            return _openai_choices(response), response
        raise ROSPlanningError(  # pragma: no cover — each adaptation fires at most once
            "OpenAI-compatible call: endpoint kept refusing after adapting tool_choice/max_tokens"
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
        client = self._client()
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
                messages=messages,
            )
        except Exception as exc:  # reason: provider SDK boundary
            raise ROSPlanningError(f"OpenAI-compatible describe_image failed: {exc!s}") from exc
        message = _openai_choices(response)[0].message
        text: str = message.content or ""
        if not text:
            text = getattr(message, "reasoning", "") or ""
        return text.strip()
