"""ADR-0018 F4 / ADR-0022 — :class:`ToolPalette` + builder.

The palette is the *closed set* of choices the LLM sees on every
:meth:`ToolUseClient.select_tool` call. It is built at reasoner
lifecycle ``configure`` time from the local rSkill registry filtered
by the active robot's :class:`~openral_core.RobotCapabilities`, and
refreshed when ``/openral/skill_registry_changed`` fires (fired by
``ral skill install|remove``).

Per ADR-0018 §4 "Tool palette built at lifecycle configure ... LLM
cannot dispatch a skill that isn't installed, isn't capability-matched,
or isn't licensed for the deployment." This module enforces the
"installed + capability-matched" half; license posture is checked
downstream by the action server (defense in depth) when the goal is
accepted.

ADR-0022 amendment: the palette carries per-skill metadata
(:class:`RSkillToolEntry`) so the LLM tool schema can present each
skill as its own tool with a real description + action verbs + object
/ scene discriminators, instead of a single ``execute_rskill`` tool
with an opaque list of ids.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import (
    Any,  # reason: pydantic model_validator(mode="before") receives untyped input
    Literal,
)

from openral_core import (
    DetectorMode,
    RobotCapabilities,
    RobotDescription,
    RSkillAction,
    RSkillManifest,
    TaskSpace,
    task_space_compatible,
)
from pydantic import BaseModel, ConfigDict, model_validator

__all__ = [
    "ContinuousDetectorEntry",
    "OnDemandDetectorEntry",
    "RSkillToolEntry",
    "ToolPalette",
    "build_tool_palette",
    "detector_alias",
    "detector_service_segment",
    "locate_in_view_service",
    "task_space_disagreement",
]


def task_space_disagreement(
    manifest: RSkillManifest,
    description: RobotDescription,
    hal_mode: str,
    legacy_ok: bool,
) -> str | None:
    """Compare the ADR-0071 ``task_space_compatible`` gate to the legacy verdict.

    Phase 2 of ADR-0071 (warn-only). The reasoner's deploy palette filter and
    ``rskill_publisher`` both run the legacy mode check (``_action_executable`` /
    ``control_modes_for_representation``) to decide whether a VLA skill is
    offered / publishable. This helper runs the canonical
    :func:`task_space_compatible` gate over the same (skill, robot, hal_mode)
    and returns a human-readable warning **only when the two disagree** — so the
    caller can surface cross-layer mismatches the mode-set check misses (an
    EE-addressed slot naming an end-effector the robot does not declare; a joint
    segment wider than the robot's joint count) without changing any drop /
    publish decision. Phase 4 makes ``task_space_compatible`` authoritative.

    Pure (no ROS) so it is unit-testable without ``rclpy``; the caller owns the
    drop/publish decision and the logging sink.

    Args:
        manifest: The rSkill manifest under consideration.
        description: The target robot description.
        hal_mode: ``"sim"`` (default-sim OSC packers) or anything else (treated
            as ``"real"`` — robot's advertised modes).
        legacy_ok: The legacy gate's verdict (``True`` = the caller would
            offer / publish the skill).

    Returns:
        A warning message when the canonical gate disagrees with ``legacy_ok``,
        else ``None``. Skills without an ``action_contract`` (detectors, VLMs,
        rewards, ros_action) return ``None`` — they carry no task space.

    Example:
        >>> # A skill whose verdict matches the legacy gate produces no warning.
        >>> # (Full worked examples live in tests/unit/test_task_space_phase2.py.)
        >>> task_space_disagreement.__name__
        'task_space_disagreement'
    """
    if manifest.action_contract is None:
        return None
    mode: Literal["sim", "real"] = "sim" if hal_mode == "sim" else "real"
    space = TaskSpace.from_action_contract(manifest.action_contract, description)
    match = task_space_compatible(space, description, hal_mode=mode)
    if match.ok == legacy_ok:
        return None
    return (
        f"ADR-0071 task_space_compatible disagrees with the legacy action gate "
        f"for rSkill {manifest.name!r} on robot {description.name!r} "
        f"(hal_mode={mode!r}): task_space_compatible.ok={match.ok}, "
        f"legacy_ok={legacy_ok}; reasons={match.reasons or ['<compatible>']}. "
        f"Warn-only (ADR-0071 Phase 2) — not yet enforced."
    )


class ContinuousDetectorEntry(BaseModel):
    """A ``mode: continuous`` detector's coverage, surfaced to the reasoner (ADR-0051).

    Continuous detectors are *not* ExecuteSkill tools and the reasoner never
    prompts them — they stream ``ObjectsMetadata`` into
    ``WorldState.detected_objects`` every frame. But the LLM still needs to know
    *what they cover* so it can decide between reading world state for an object
    that is already tracked vs prompting the on-demand ``locate_in_view`` locator
    for one that is not. This record carries that coverage characterisation
    (deliberately a compact summary, not the full label list).

    Attributes:
        rskill_id: The detector rSkill's :attr:`RSkillManifest.name`.
        description: The manifest ``description`` — the coverage summary.
        objects: Free-form object/category keywords from the manifest.
        scenes: Free-form scene keywords from the manifest.
        num_labels: Size of the detector's fixed class vocabulary.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    rskill_id: str
    description: str
    objects: tuple[str, ...] = ()
    scenes: tuple[str, ...] = ()
    num_labels: int = 0


def detector_alias(rskill_name: str) -> str:
    """Short, LLM- and operator-facing id for a detector rSkill (ADR-0056).

    Strips the ``OpenRAL/`` org prefix and the ``rskill-`` kind prefix, so
    ``"OpenRAL/rskill-omdet-turbo-locator"`` → ``"omdet-turbo-locator"``. This is
    the value the reasoner passes in ``LocateInViewTool.detector`` and the basis
    for the per-detector service namespace (see :func:`detector_service_segment`).
    """
    short = rskill_name.rsplit("/", 1)[-1]
    return short[len("rskill-") :] if short.startswith("rskill-") else short


def detector_service_segment(alias: str) -> str:
    """ROS-safe service-namespace segment for a detector alias (ADR-0056).

    ROS 2 names allow only ``[A-Za-z0-9_]`` per token, so hyphens in the alias
    become underscores: ``"omdet-turbo-locator"`` → ``"omdet_turbo_locator"``. The
    detector node's locate service lives at
    ``/openral/perception/<segment>/locate_in_view``.
    """
    return alias.replace("-", "_")


def locate_in_view_service(detector: str, *, default: str = "") -> str:
    """Resolve the ``locate_in_view`` service for a (possibly empty) detector selector (ADR-0056).

    Single source of truth shared by the reasoner dispatch (resolves
    ``LocateInViewTool.detector``) and the deploy launch (names each on-demand
    locator node's service). An empty ``detector`` falls back to ``default``; an
    empty resolved alias yields the legacy single-detector service
    ``/openral/perception/locate_in_view`` (back-compat for deployments that bring
    up exactly one on-demand detector). Otherwise the service is namespaced by the
    ROS-safe alias segment.
    """
    alias = detector or default
    if not alias:
        return "/openral/perception/locate_in_view"
    return f"/openral/perception/{detector_service_segment(alias)}/locate_in_view"


class OnDemandDetectorEntry(BaseModel):
    """A ``mode: on_demand`` open-vocab locator, surfaced as a locate_in_view option (ADR-0056).

    On-demand locators are prompt-able **read-only** tools: the reasoner picks one
    by :attr:`alias` in ``LocateInViewTool.detector`` and asks it "is object X in
    view right now?". Unlike continuous detectors they are not background
    producers; unlike VLAs they carry no actuation authority.

    Attributes:
        rskill_id: The detector rSkill's :attr:`RSkillManifest.name`.
        alias: Short selector the reasoner passes as ``detector``
            (see :func:`detector_alias`).
        description: The manifest ``description`` — the capability hint the LLM
            scores when choosing between locators.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    rskill_id: str
    alias: str
    description: str


class RSkillToolEntry(BaseModel):
    """Per-skill metadata surfaced to the reasoner LLM as one tool.

    ADR-0022. The reasoner constructs one tool per :class:`RSkillToolEntry`
    in :attr:`ToolPalette.skills` so the LLM can pick a skill by what it
    does (description + actions + objects + scenes) rather than by
    inferring meaning from a slug.

    Attributes:
        rskill_id: HF Hub id, e.g. ``"OpenRAL/rskill-pi05-..."``.
            Matches :attr:`RSkillManifest.name`.
        description: Short NL summary, mirrored from
            :attr:`RSkillManifest.description`. Primary signal the LLM
            scores tools on; keep specific (objects, scenes, task type).
        actions: Action verbs the skill performs
            (:class:`~openral_core.RSkillAction`). At least one entry.
        objects: Free-form object keywords (``"cube"``, ``"pipe"``, …).
        scenes: Free-form scene keywords (``"tabletop"``, ``"kitchen"``).
        goal_params_schema: ADR-0026 — per-skill JSON-Schema 7 / OpenAPI
            shape describing the ``goal_params_json`` payload the LLM may
            attach. ``None`` (the common case for VLAs) means the LLM
            sees no structured params for this skill — only the flat
            ``(rskill_id, prompt, deadline_s)`` surface. When set,
            ``ToolUseClient`` (the LLM tool-use shim) merges this into
            the per-skill tool's ``parameters.properties.goal_params_json``
            so the provider's structured-output path generates a
            well-formed payload.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    rskill_id: str
    description: str
    actions: tuple[RSkillAction, ...]
    objects: tuple[str, ...] = ()
    scenes: tuple[str, ...] = ()
    goal_params_schema: dict[str, Any] | None = None


class ToolPalette(BaseModel):
    """Closed-set tool palette presented to the reasoner's LLM each tick.

    Three of the four :data:`~openral_core.ReasonerToolCall` variants
    are always available (``reload_gst_pipeline``,
    ``lifecycle_transition``, ``emit_prompt``); only
    :class:`~openral_core.ExecuteRskillTool` is gated, because it can
    actually drive actuators via the ``rskill_runner_node`` action
    server (F1).

    ADR-0022: when :attr:`skills` is populated the reasoner emits one
    LLM tool per skill (named ``execute_rskill__<slug>``) carrying the
    skill's description + actions + objects + scenes. When only
    :attr:`execute_rskill_ids` is populated (e.g. synthetic test
    palettes or the default empty palette in ``reasoner_node``), the
    reasoner falls back to a single ``execute_rskill`` tool with the id
    set as an enum constraint.

    Attributes:
        skills: Per-skill metadata records. The LLM sees one tool per
            entry. The primary surface (ADR-0022).
        execute_rskill_ids: Set of skill ids the LLM may pass to
            ``ExecuteRskillTool.rskill_id``. Auto-derived from
            :attr:`skills` when ``skills`` is non-empty; may also be
            set directly for palettes without per-skill metadata
            (synthetic test palettes; the default empty palette).
        sensor_ids: Set of sensor ids the LLM may pass to
            ``ReloadGstPipelineTool.sensor_id``. Populated from the
            active runtime's sensor catalog at palette-build time. Used
            by the reasoner_node for client-side validation before the
            service call is made.
        node_ids: Set of fully-qualified ROS node names the reasoner
            may target with ``LifecycleTransitionTool``. Populated
            from the deployment YAML's known lifecycle peers.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    skills: tuple[RSkillToolEntry, ...] = ()
    execute_rskill_ids: frozenset[str] = frozenset()
    sensor_ids: frozenset[str] = frozenset()
    node_ids: frozenset[str] = frozenset()
    continuous_detectors: tuple[ContinuousDetectorEntry, ...] = ()
    """ADR-0051 — ``mode: continuous`` detectors installed for the active robot.
    Not tools (the reasoner never prompts them); surfaced so the LLM knows which
    object classes are already tracked in world state for free, and can reserve
    the on-demand ``locate_in_view`` locator for objects outside that coverage."""
    spatial_memory_available: bool = False
    """ADR-0039 — when ``True`` the LLM additionally sees the two **read-only**
    spatial-memory query tools (``recall_object`` / ``resolve_place``). Set by the
    reasoner_node only when an ADR-0038 ``SpatialMemory`` query backend is wired
    (Phase 2); off by default so the tools never appear without a dispatcher."""
    detector_available: bool = False
    """ADR-0043 — when ``True`` the LLM additionally sees the **read-only**
    ``locate_in_view`` tool (ask a live VLM detector whether an object is in the
    current camera frame). Set by the reasoner_node only when a detector exposes a
    ``/openral/perception/<detector>/locate_in_view`` service; off by default so the
    tool never appears without a dispatcher."""
    on_demand_detectors: tuple[OnDemandDetectorEntry, ...] = ()
    """ADR-0056 — ``mode: on_demand`` open-vocab locators installed for the active
    robot. Surfaced as the selectable ``detector`` options of the read-only
    ``locate_in_view`` tool so the LLM can choose the model (e.g. a light real-time
    locator vs a high-quality grounding VLM). Gated by ``detector_available`` like
    the tool itself; empty = the tool keeps its single default-locator behaviour."""
    scene_query_available: bool = False
    """ADR-0047 — when ``True`` the LLM additionally sees the **read-only**
    ``query_scene`` tool (ask a scene VLM an open-ended question about the current
    view — task progress / success-failure verification). Set by the reasoner_node
    only when a scene VLM exposes the ``/openral/perception/query_scene`` service;
    off by default so the tool never appears without a dispatcher. Distinct from
    ``detector_available``: localization (``locate_in_view``) and scene-state
    reasoning (``query_scene``) are independently provisioned backends."""
    task_progress_available: bool = False
    """ADR-0058 — when ``True`` the LLM additionally sees the **read-only**
    ``query_task_progress`` tool (ask the Robometer reward monitor for a
    quantitative windowed progress/success assessment of the current task). Set
    by the reasoner_node only when a reward monitor exposes the
    ``/openral/perception/query_task_progress`` service; off by default. Distinct
    from ``scene_query_available``: ``query_scene`` returns free text, this
    returns normalized progress/success scalars + trends."""
    memory_available: bool = False
    """ADR-0072 §3 — when ``True`` the LLM additionally sees the self-maintained
    semantic-memory tools: the write-capable ``memory_write``
    (``add``/``update``/``supersede``/``delete`` over a ``MemorySection``) and the
    read-only ``memory_search`` (archival recall). Set by the reasoner_node only
    when a ``MEMORY.md`` is wired (``memory_md_path`` param); off by default so the
    tools never appear without a dispatcher. The reasoner already *reads* current
    memory every tick as the ``## MEMORY`` context block — these tools let it edit
    that file explicitly and recall superseded/archived entries (reader/writer
    split)."""

    @model_validator(mode="before")
    @classmethod
    def _derive_execute_rskill_ids(cls, data: Any) -> Any:  # noqa: ANN401  # reason: untyped pydantic raw input
        """Auto-fill ``execute_rskill_ids`` from ``skills`` when only the latter is provided.

        Lets ``ToolPalette(skills=(...))`` work without forcing every
        caller to also pass ``execute_rskill_ids``. Call sites that pass
        only ``execute_rskill_ids`` (synthetic test palettes; the
        default empty palette in ``reasoner_node``) keep working
        unchanged.
        """
        if not isinstance(data, dict):
            return data
        if "skills" in data and "execute_rskill_ids" not in data:
            skills = data["skills"]
            ids: list[str] = []
            for entry in skills:
                if isinstance(entry, RSkillToolEntry):
                    ids.append(entry.rskill_id)
                elif isinstance(entry, dict):
                    ids.append(entry["rskill_id"])
            data = {**data, "execute_rskill_ids": frozenset(ids)}
        return data

    @model_validator(mode="after")
    def _check_skills_match_ids(self) -> ToolPalette:
        """When ``skills`` is set, every entry's id must appear in ``execute_rskill_ids``.

        Catches the case where a caller passes both and they disagree.
        """
        if self.skills:
            skill_ids = {s.rskill_id for s in self.skills}
            if skill_ids != set(self.execute_rskill_ids):
                raise ValueError(
                    f"ToolPalette.skills ids {sorted(skill_ids)!r} do not match "
                    f"execute_rskill_ids {sorted(self.execute_rskill_ids)!r}; pass one "
                    "or the other, or make them consistent."
                )
        return self


def build_tool_palette(
    *,
    installed_skills: Iterable[RSkillManifest],
    robot_capabilities: RobotCapabilities,
    sensor_ids: Iterable[str] = (),
    node_ids: Iterable[str] = (),
    commercial_deployment: bool = False,
    spatial_memory_available: bool = False,
    detector_available: bool = False,
    scene_query_available: bool = False,
    task_progress_available: bool = False,
    memory_available: bool = False,
) -> ToolPalette:
    """Build a :class:`ToolPalette` from the installed-skill registry.

    A skill is included in the palette iff:

    1. Every flag in ``skill.capabilities_required`` is set on
       ``robot_capabilities``.
    2. ``skill.embodiment_tags`` intersects
       ``robot_capabilities.embodiment_tags``.
    3. ``role == "s1"`` — only S1 skills are dispatchable via
       ``ExecuteRskillTool`` per CLAUDE.md §6.2 (S0/S2 slots are
       reserved and have separate dispatch paths) — **and**
       ``kind != "detector"``: detector rSkills are S1-rate perception
       producers (ADR-0035/0037), not ExecuteSkill-dispatchable; they
       activate as the perception ROS node / GStreamer tee consumer.
    4. If ``commercial_deployment`` is ``True``, the skill's license
       posture allows commercial use
       (:attr:`RSkillManifest.is_commercial_use_allowed`). Defense in
       depth: ``ral skill install`` gates at install time too
       (CLAUDE.md §1.9), but the palette filter prevents a smuggled
       weights cache from reaching production.

    Each included skill is materialised as a :class:`RSkillToolEntry`
    carrying the manifest's ``description`` + ``actions`` + ``objects``
    + ``scenes`` (ADR-0022), so the reasoner LLM sees one tool per
    skill with a real description.

    Args:
        installed_skills: Iterable of every installed
            :class:`RSkillManifest`.
        robot_capabilities: The active robot's capabilities.
        sensor_ids: Sensor ids known to the active runtime; forwarded
            verbatim into :attr:`ToolPalette.sensor_ids`.
        node_ids: Lifecycle-peer node ids known to the deployment;
            forwarded verbatim.
        commercial_deployment: When ``True`` filters out non-commercial
            skills (e.g. NVIDIA GR00T weights). Defaults to ``False``
            (research / lab deployment).
        spatial_memory_available: When ``True`` the palette advertises the
            read-only ``recall_object`` / ``resolve_place`` query tools (ADR-0039);
            set by the reasoner when a ``SpatialMemory`` backend is wired.
        detector_available: When ``True`` the palette advertises the read-only
            ``locate_in_view`` query tool (ADR-0043); set by the reasoner when an
            object detector exposes ``/openral/perception/locate_in_view``.
        scene_query_available: When ``True`` the palette advertises the read-only
            ``query_scene`` tool (ADR-0047); set by the reasoner when a scene VLM
            exposes ``/openral/perception/query_scene``.
        task_progress_available: When ``True`` the palette advertises the
            read-only ``query_task_progress`` tool (ADR-0057); set by the reasoner
            when a reward monitor exposes
            ``/openral/perception/query_task_progress``.
        memory_available: When ``True`` the palette advertises the ``memory_write``
            + ``memory_search`` tools (ADR-0072 §3); set by the reasoner when a
            ``MEMORY.md`` is wired via the ``memory_md_path`` param.

    Returns:
        A frozen :class:`ToolPalette`.

    Example:
        >>> from openral_core import RobotCapabilities
        >>> palette = build_tool_palette(
        ...     installed_skills=[],
        ...     robot_capabilities=RobotCapabilities(embodiment_tags=["so100_follower"]),
        ... )
        >>> palette.execute_rskill_ids
        frozenset()
        >>> palette.skills
        ()
    """
    robot_tags = set(robot_capabilities.embodiment_tags)
    entries: list[RSkillToolEntry] = []
    continuous_detectors: list[ContinuousDetectorEntry] = []
    on_demand_detectors: list[OnDemandDetectorEntry] = []
    for skill in installed_skills:
        if skill.role != "s1":
            continue
        # ``detector`` rSkills are S1-rate perception producers (RT-DETR →
        # ObjectsMetadata, ADR-0035/0037), not ExecuteSkill-dispatchable
        # policies — they are activated as the perception ROS node / GStreamer
        # tee consumer, never via ExecuteRskillTool. Admitting them here would
        # let the reasoner dispatch a detector as if it actuated the robot.
        # A ``mode: continuous`` detector (ADR-0051) is still surfaced — not as a
        # tool, but as coverage the LLM reads from world state — so it can reserve
        # the on-demand ``locate_in_view`` locator for objects outside that bank.
        if skill.kind == "detector":
            if skill.detector is not None and skill.detector.mode is DetectorMode.CONTINUOUS:
                continuous_detectors.append(
                    ContinuousDetectorEntry(
                        rskill_id=skill.name,
                        description=skill.description,
                        objects=tuple(skill.objects),
                        scenes=tuple(skill.scenes),
                        num_labels=len(skill.detector.labels),
                    )
                )
            elif skill.detector is not None and skill.detector.mode is DetectorMode.ON_DEMAND:
                # ADR-0056 — on-demand locators are prompt-able read-only tools, not
                # ExecuteSkill policies: surfaced as selectable ``detector`` options
                # of locate_in_view, never admitted to the actuating palette.
                on_demand_detectors.append(
                    OnDemandDetectorEntry(
                        rskill_id=skill.name,
                        alias=detector_alias(skill.name),
                        description=skill.description,
                    )
                )
            continue
        skill_tags = set(skill.embodiment_tags)
        if not (skill_tags & robot_tags):
            continue
        required = set(skill.capabilities_required or [])
        capabilities_set: set[str] = {
            field for field, value in robot_capabilities.model_dump().items() if value is True
        }
        if not required.issubset(capabilities_set):
            continue
        if commercial_deployment and not skill.is_commercial_use_allowed:
            continue
        entries.append(
            RSkillToolEntry(
                rskill_id=skill.name,
                description=skill.description,
                actions=tuple(skill.actions),
                objects=tuple(skill.objects),
                scenes=tuple(skill.scenes),
                goal_params_schema=skill.goal_params_schema,
            ),
        )
    # Stable order so tool schemas are deterministic across builds.
    entries.sort(key=lambda e: e.rskill_id)
    continuous_detectors.sort(key=lambda d: d.rskill_id)
    on_demand_detectors.sort(key=lambda d: d.alias)
    return ToolPalette(
        skills=tuple(entries),
        sensor_ids=frozenset(sensor_ids),
        node_ids=frozenset(node_ids),
        continuous_detectors=tuple(continuous_detectors),
        on_demand_detectors=tuple(on_demand_detectors),
        spatial_memory_available=spatial_memory_available,
        detector_available=detector_available,
        scene_query_available=scene_query_available,
        task_progress_available=task_progress_available,
        memory_available=memory_available,
    )
