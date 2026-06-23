# ADR-0071: Reasoner playbooks (`kind: "playbook"`) + a self-maintained `MEMORY.md`

- **Status:** Proposed
- **Date:** 2026-06-23
- **Author:** Adrian Llopart
- **Related:**
  - [ADR-0018](0018-ros2-reasoner-supervisor.md) — the S2 Reasoner, the **closed**
    `ReasonerToolCall` palette this ADR extends, the bounded replanning ladder,
    and the event-driven heartbeat. **Extended here.**
  - [ADR-0038](0038-persistent-semantic-spatial-memory.md) — the Layer-2
    *geometric* scene-graph world model (`recall_object` / `resolve_place`). This
    ADR's `MEMORY.md` is the **complementary semantic/narrative** memory; the two
    are deliberately separate stores (see §Decision-3). **Companion, not superseded.**
  - [ADR-0039](0039-llm-task-planning-active-search.md) — task decomposition +
    active object search **as reasoner Python**. This ADR repackages those proven
    behaviors as **authored, versioned playbook content** and gives them a
    persistent memory to draw on. **0039 is the engine; playbooks are the content.**
  - [ADR-0047](0047-vlm-rskill-kind.md) / [ADR-0057](0057-robometer-reward-rskill.md)
    — precedent for adding a non-actuating `RSkillKind` with a per-kind manifest
    contract and a read-only reasoner tool. The `playbook` kind follows that mould.
  - [ADR-0043](0043-locate-in-view-reasoner-tool.md) / ADR-0056 — read-only
    reasoner tools (`locate_in_view`, on-demand detectors) the playbooks compose.
  - [ADR-0044](0044-look-at-skill-grid-refined-approach.md) — the
    `OccupancyGridIndex` approach-refinement grid, the **consumer** of the 2D map
    in the deploy memory bundle (§Decision-3b); nav2 `map_server` is the loader.
  - [ADR-0024](0024-ros-wrapped-rskills.md) / [ADR-0022](0022-rskill-action-vocabulary.md)
    — the wrapped Nav2 / MoveIt skills + the `RSkillAction` verb vocabulary the
    playbooks sequence.
  - [ADR-0030](0030-geometric-safety-collision-checking.md) — the C++ safety
    kernel that still **disposes** every motion. Nothing here relaxes it.
  - CLAUDE.md §1.1 (safety beats helpfulness), §1.2 (truth over plausibility —
    no "signed/verified" claims), §1.4 (explicit, no hidden fallback), §1.6
    (schemas evolve, never silently), §1.8 (reproducibility), §3 (Reasoner &
    dispatch; dual-system pattern; WAM is a *separate* Layer-5 concept).
- **Literature grounding** (full citations in §Appendix A): EMOS (robot resume,
  execution history), OpenMind OM1 (NL data bus + memory buffer), SayCan
  (affordance gating), Inner Monologue (closed-loop feedback), Reflexion
  (verbal self-reflection), SayPlan (collapsed-graph expand/contract), ESC / L3MVN
  (commonsense object search), MemGPT/Letta (tiered self-editing memory),
  Generative Agents (importance×recency×relevance + reflection), Mem0/Zep
  (ADD/UPDATE/DELETE/NOOP + temporal supersession), TidyBot (distilled
  preference rules), DROC (distilled human corrections), Statler (reader/writer
  split), Voyager (verified-skill promotion), COME-robot (failure-cause
  attribution), ReMEmbR / Episodic-Memory-Verbalization (object-location log +
  consolidation).

---

## Context

### Where the S2 reasoner is today

The reasoner's **mechanism** is solid; its **content and surrounding loop** are thin.

- **Mechanism (good).** Tool calls are a Pydantic discriminated union
  (`ReasonerToolCall`, `python/core/src/openral_core/schemas.py:6775`) validated
  through the provider's native tool-use API — no free-form JSON, schema-enforced,
  provider-agnostic (Anthropic / OpenAI-compatible / OpenRouter,
  `tool_use.py:988`/`:1067`). Per-skill tools carry the skill's
  `goal_params_schema` (ADR-0026). Read-only introspection tools already exist:
  `recall_object`, `resolve_place`, `locate_in_view`, `query_scene`,
  `query_task_progress`.
- **Content (thin).** The LLM sees four flat text sections —
  `WORLD_STATE / FAILURES / PERCEPTION / PROMPTS` (`context.py:206`). It is **not
  told what the robot can physically do** (no reach/FOV/footprint), it gets
  **execution feedback only on failure** (the FAILURES buffer; successes are
  invisible), and the "bounded replanning ladder" is in practice a **per-kind
  retry counter** (`core.py:266`, cap = 3) with **no reflection** on *why* a step
  failed.
- **No persistence.** Between tasks the reasoner remembers nothing. ADR-0038 gives
  it a *geometric* scene graph ("where is the mug"), but there is no store for
  **user preferences, human corrections, learned lessons, or durable home facts**
  ("the user keeps mugs in the top-left cabinet"; "approach mugs by the handle —
  bare-body grasp slipped twice"). Every episode re-derives this from scratch.
- **No reusable decision procedures.** ADR-0039 proved that decomposition and
  active search *work*, but they live as bespoke Python in `reasoner_node`. There
  is no way to **author, version, install, or share** a high-level decision
  procedure ("how to find an object you can't see") the way we package a policy or
  a detector. The `RSkillKind` enum is entirely *neural/actuating* (`vla`,
  `ros_action`, `detector`, `vlm`, `reward`) plus the reserved-but-unimplemented
  `wam`; there is **no home for symbolic decision logic**.

### What the competitors and the literature do better

The throughline of the survey (§Appendix A): the leaders do **not** have smarter
LLMs. They (a) feed the LLM a **structured, embodiment-grounded situation report**,
(b) wrap tool-selection in a **search → verify → reflect loop with memory**, and
(c) persist **distilled knowledge** (preferences, corrections, object locations)
across episodes through an **explicitly-edited, size-bounded memory**. Concretely:

- **EMOS "Robot Resume"** auto-generates each robot's capability description
  (reach hull, camera frustum, footprint) from its URDF and feeds *that* to the
  planner — and each agent runs a **feasibility self-check** before accepting a
  subtask. We feed joints/EE but never "what can I reach / see."
- **OM1's NL data bus + memory** captions every sensor into timestamped NL, fuses
  it into one situation report, and keeps a recent-history buffer. We hand the LLM
  four raw typed blobs.
- **MemGPT/Letta** make the agent own its memory hygiene: a small always-loaded
  **core** block plus paged **archival** storage, edited only through function
  calls. **Generative Agents** retrieve by `importance × recency × relevance` and
  periodically **reflect** raw observations into durable insights. **Mem0/Zep**
  edit memory through explicit `ADD/UPDATE/DELETE/NOOP` with **temporal
  supersession** rather than blind append. **TidyBot/DROC** store **distilled
  generalizable rules** (preferences, corrections), not raw events.

These three gaps — *grounded context*, *persistent self-maintained memory*, and
*authored decision procedures* — are one theme: **closing the gap between the
current repo and a robot that operates usefully in a real household.** This ADR
addresses all three because they are tightly coupled: playbooks read memory;
memory is written during playbook execution; both need the grounded context to
make good decisions. (§Phasing lands them independently; §Alternatives explains
why one ADR rather than three.)

---

## Decision

This ADR makes **four** changes, all in **Layer 4 (Reasoning)** plus the Layer-0
schema. None shifts actuation authority: the reasoner still only *proposes*; the
ADR-0030 C++ kernel still *disposes* (§Safety).

### Decision 1 — `kind: "playbook"`: decision procedures as packaged rSkill content

Add `"playbook"` to `RSkillKind`
(`python/core/src/openral_core/schemas.py:3761`). A **playbook** is a
human-readable **standard-operating-procedure** — a structured Markdown document
describing *how the S2 reasoner should approach a class of task* (its trigger,
preconditions, the ordered decision steps it composes from existing tools, its
fallbacks, and its **acceptance / done predicate**). It is **content the reasoner
reads, never code it executes**. It carries **no weights, no actuators, no ROS
server, no Action contract** — it is the symbolic counterpart to a `vla` policy.

> **Why a new kind and not `wam`.** A WAM (Layer 5) is a *learned* mental-simulation /
> failure-anticipation model (CLAUDE.md §3). A playbook is a *symbolic, authored*
> procedure interpreted by the existing S2 LLM. Overloading `wam` would muddle the
> layer semantics exactly as ADR-0047 rejected overloading it for VLMs.

**Manifest contract** (enforced by `RSkillManifest._check_kind_consistency`,
mirroring the `vlm`/`reward` precedent):

| Field | Constraint |
|---|---|
| `kind` | `"playbook"` |
| `role` | MUST be `"s2"` |
| `playbook` (new `PlaybookContract` block) | REQUIRED |
| `actions` | REQUIRED (`min_length=1`); MUST be `[PLAN]` — a **new `RSkillAction.PLAN` verb this ADR adds**, mirroring ADR-0047 adding `QUERY` for `vlm` and `MONITOR` for `reward`. It is registry/discovery metadata only — a playbook is `role: s2` so it is never an `ExecuteSkill` dispatch verb. |
| `chunk_size` | MUST be `1`. `chunk_size` is a **required** manifest field (`Field(gt=0)`, no default); a playbook emits no `Action` rows, so it is pinned to `1` exactly like the ROS-wrapper and perception (`vlm`/`reward`) kinds. *(Earlier draft wrongly listed this FORBIDDEN.)* |
| `latency_budget` | REQUIRED — declares the S2 planning budget, e.g. `{per_chunk_ms: 5000}` (~0.2 Hz tick). CI enforces it on the reference host (§2 latency budgets). |
| `description` | REQUIRED (1–500 chars), as for every manifest. |
| `weights_uri`, `model_family`, `min_vram_gb` | FORBIDDEN — a playbook downloads no weights and reserves no VRAM. |
| `runtime`, `quantization` | Left at their defaults and **ignored** — the loader instantiates **no** inference runtime for a playbook (these fields have defaults, so the manifest stays valid; the loader's `playbook` branch never reads them). |
| `actuators_required` | MUST be empty (non-actuating, like `vlm`/`reward`). |
| `sensors_required` | SHOULD be empty — the playbook needs no sensors itself; the tools it composes declare their own. |
| `action_contract`, `state_contract`, `envelope` | FORBIDDEN. |
| `detector`, `ros_integration`, `processors`, `image_preprocessing`, `n_action_steps`, `starting_pose` | FORBIDDEN. |
| `capabilities_required` | OPTIONAL — composing-tool gate (e.g. a playbook that opens containers may require `manipulation`); the loader filters a playbook out of the active set when the robot lacks a required capability. |
| `embodiment_tags` | OPTIONAL — empty = embodiment-agnostic (most playbooks are). |
| `objects`, `scenes` | OPTIONAL metadata — the object/scene domains the playbook applies to (e.g. `scenes: [kitchen, indoor]`), surfaced in the discovery view. |

**`PlaybookContract`** (new Pydantic model in `schemas.py`, `extra="forbid"`,
`frozen=True`):

```text
PlaybookContract:
  trigger:            str            # NL description of when this playbook applies
                                     #   (used for retrieval scoring — see §loading)
  body_uri:           str            # relative path inside the rSkill repo to the
                                     #   Markdown SOP (e.g. "PLAYBOOK.md")
  composes_tools:     list[str]      # the ReasonerToolCall discriminators it uses
                                     #   (execute_rskill, recall_object, locate_in_view,
                                     #    query_scene, query_task_progress, memory_*, …)
                                     #   — validated against the known tool set; a
                                     #   capability/skill gate the loader can pre-check
  done_predicate:     str            # NL acceptance test ("the target object is in the
                                     #   gripper", "all checklist items are done")
  max_steps:          int            # explicit bound on tool calls before the playbook
                                     #   must terminate (no hidden default; §1.4)
  fallback_skill_id:  str | None     # terminal escalation (usually a human-handoff
                                     #   playbook or emit_prompt), reusing the existing
                                     #   manifest field semantics
```

The **SOP body** (`body_uri`) is plain Markdown with a fixed lightweight skeleton
(`## Trigger`, `## Preconditions`, `## Steps`, `## Verify (done predicate)`,
`## Fallbacks`) so it reads like an agent `SKILL.md` and so
`tools/generate_rskill_skillmd.py` can keep emitting the discovery view.

**Packaging.** Identical to every other rSkill: one HF Hub repo with `rskill.yaml`
+ `PLAYBOOK.md` + `README.md` + a generated `SKILL.md`. No weights, so publishing
is text-only; the existing provenance gates (`OPENRAL_REQUIRE_SIGNED_SKILLS`,
`*.pt` refusal) apply unchanged — there are no weights to refuse, but the manifest
still flows through `rSkill.from_pretrained` and emits the
`rskill.unverified_provenance` warning until ADR-0006 signing lands (do **not**
describe playbooks as "verified", §1.2).

**Three text files, three distinct roles — do not conflate them** (this is the
one subtlety unique to a content-only kind):

| File | Authored? | Read by | Role |
|---|---|---|---|
| `rskill.yaml` | hand | loader + **reasoner palette / `load_playbook` retrieval** | source of truth; **selection** metadata (`trigger`, `description`, `actions`, `scenes`) |
| `PLAYBOOK.md` (`body_uri`) | **hand** | **the S2 reasoner** (injected into the system prompt) | the **SOP content** the LLM follows — a playbook's "runtime" |
| `SKILL.md` | **generated** (`generate_rskill_skillmd.py`; never hand-edit) | external agent harnesses + the `--check` CI gate | discovery-only **mirror of the manifest**, exactly as for every other rSkill |

Note the in-process reasoner **never reads `SKILL.md`** — nothing at runtime does;
it is the standard external-discovery view (CLAUDE.md §1.3). The reasoner discovers
playbooks by the manifest (selection) and acts on the `PLAYBOOK.md` body (content).
In the **default loading path** all installed, capability-matched playbooks' bodies
are injected, so the LLM always sees them; the manifest `trigger` only gates the
deferred `load_playbook` retrieval at scale.

**Loading & surfacing to the reasoner** (the lazy default first, the scale path
behind a flag):

1. **Default — inject installed playbooks into the system prompt.** At
   `on_configure`, the reasoner loads the `PLAYBOOK.md` bodies of all installed,
   capability-matched playbooks into a fenced `## PLAYBOOKS` block of its system
   prompt. For the handful of playbooks a single robot runs, this is the whole
   mechanism — no new runtime, no retriever, no tool. The LLM reads the SOP and
   follows it using tools it already has.
2. **Scale path (behind `playbook_retrieval_available`, off by default).** When
   the installed set grows past a size budget, expose a read-only
   **`LoadPlaybookTool`** (`ReasonerToolCall` variant, discriminator
   `"load_playbook"`) that retrieves the top-`k` playbook bodies by embedding
   similarity of the current goal against each `trigger` (Voyager-style
   retrieval), and injects only those. This is additive and deferred (§Phasing).

Playbooks are **never** dispatched through `ExecuteSkill` (they are `role: s2`,
non-actuating) — exactly the exclusion `vlm`/`reward` already get in
`build_tool_palette`. A playbook **changes what the LLM *decides*, not what it is
*authorized* to actuate**: every motion it triggers is still an `execute_rskill`
→ Action chunk → C++ kernel. This is the key safety property (§Safety).

#### Worked example: the `find-object` playbook

A complete package is one HF Hub repo: `rskill.yaml` + `PLAYBOOK.md` + `README.md`.

**`rskill.yaml`** (modelled on the real `rskills/qwen35-4b-nf4` manifest; the
`playbook` block and `actions: [plan]` are the kind-specific parts):

```yaml
name: rskill-find-object
version: 0.1.0
license: apache-2.0
role: s2
kind: playbook
actions: [plan]                 # RSkillAction.PLAN — registry/discovery metadata only
chunk_size: 1                   # required field; pinned (no Action rows)
latency_budget: {per_chunk_ms: 5000.0}
description: >-
  S2 decision procedure: locate a named object, falling back to bounded
  commonsense active search and finally human handoff. Composes recall_object,
  resolve_place, locate_in_view and NAVIGATE/OPEN skills. ADR-0071.
embodiment_tags: []             # embodiment-agnostic
capabilities_required: {has_vision: true}   # real RobotCapabilities flag (camera needed to locate)
scenes: [kitchen, indoor]
objects: [open-vocabulary object]
playbook:
  trigger: "the goal names a physical object whose location is not given"
  body_uri: ./PLAYBOOK.md
  composes_tools: [recall_object, resolve_place, locate_in_view, execute_rskill, memory_search]
  done_predicate: "the target object is confirmed in view at a known pose"
  max_steps: 12
  fallback_skill_id: null       # terminal escalation is emit_prompt (human handoff)
```

**`PLAYBOOK.md`** (the SOP the reasoner reads — fixed skeleton):

```markdown
# find-object

## Trigger
The goal names an object (e.g. "bring the water bottle") whose pose is unknown.

## Preconditions
A spatial-memory backend is available; the robot has a mobile base.

## Steps
1. `recall_object(target)`. If a current pose is returned, go to **Verify**.
2. On a miss, consult `MEMORY.md` Object-Location Log via `memory_search(target)`
   for the last-seen (possibly stale) location — use it as the top search prior.
3. Rank candidate places: scene-graph regions + commonsense priors
   ("a water bottle is usually in the kitchen / fridge"). Containers whose
   contents are occluded come first.
4. For each candidate in rank order, within `max_steps`:
   `resolve_place` → NAVIGATE → if it is an occluding container, `execute_rskill(OPEN)`
   → `locate_in_view(target)`. Stop on a hit.

## Verify (done predicate)
`locate_in_view` confirms the target in view at a known pose. On success,
`memory_write(op=supersede, section="Object-Location Log", target, content=place)`.

## Fallbacks
Budget exhausted with no hit → `emit_prompt` to the operator ("I can't find the
water bottle — where should I look?"). Never loop past `max_steps`.
```

This is the **find-object** row below; the other six follow the same shape.

#### The seven launch playbooks (household, S2)

Each composes existing tools; none introduces new actuation. The first two
package ADR-0039's proven behaviors; the rest are new SOPs. (Per the design
review, the earlier human-handover / resource-guard / safe-abort drafts are
**dropped** from this ADR — they touch the safety/HAL boundary harder and belong
in a dedicated safety-WG ADR, not this Layer-4 content ADR.)

| # | Playbook (`rskill-…`) | Decision logic (SOP summary) | Composes | Terminal fallback |
|---|---|---|---|---|
| 1 | **find-object** | `recall_object` → on miss, rank likely rooms via scene-graph regions + commonsense priors (ESC/L3MVN) → `resolve_place`+navigate → `locate_in_view` → open occluding container → re-look; bounded by `max_steps` | recall_object, resolve_place, execute_rskill(NAVIGATE/OPEN), locate_in_view, memory_search | human-handoff |
| 2 | **decompose-mission** ("internal TODO.md") | Split an NL goal into an **ordered checklist of subtasks, each with a `done` predicate**; execute one subtask/tick; mark done/blocked; re-plan the tail on a blocked item (LLM-Planner re-grounding). The checklist is held in working memory and surfaced in the trace | execute_rskill, query_task_progress, subtask-with-goal (#7), memory_* | human-handoff |
| 3 | **clarify-ambiguity** | If a referent maps to >1 candidate or confidence < τ, ask **one** targeted question *before* any actuation; proceed on the answer | query_scene, locate_in_view, emit_prompt | proceed with best guess only if human unreachable |
| 4 | **verify-outcome** | After each manipulation skill, `query_task_progress` / `query_scene` to confirm the subtask's `done` predicate; on mismatch, write a **diagnostic reflection** (Reflexion) and re-enter the ladder with a strategy hint instead of blind retry | query_task_progress, query_scene, memory_write | substitute-skill, then human-handoff |
| 5 | **preflight-reach** | Before dispatch: is the target inside the robot's **reach hull** and **camera frustum**, and is a path known? (SayCan affordance × EMOS self-check, reads the §Decision-2 self-model). If not, reposition / `look_at` first | resolve_place, execute_rskill(look_at/NAVIGATE), recall_object | abort skill + request re-staging |
| 6 | **stage-for-manipulation** | Compute *where to stand* to both see and reach the object (approach-viewpoint, refined on the occupancy grid — extends ADR-0044), then navigate there before the manip skill | recall_object, resolve_place, execute_rskill(NAVIGATE) | nearest reachable viewpoint, else "unreachable" |
| 7 | **subtask-with-goal** (generalized) | The **recursive contract every subtask satisfies**: define a subtask as `(precondition, ordered steps, verifiable done-predicate)`; execute the steps; **verify the predicate (via #4) before returning success**; may itself spawn sub-subtasks the same way. Generalizes "open-the-container-then-act" — opening is one instantiation where the done-predicate is "container open". Gives #2's decomposition a **definition-of-done at every node**, so a parent only advances when a child *provably* succeeded | execute_rskill, verify-outcome (#4), query_task_progress, query_scene, memory_* | escalate the failing node to its parent's replan, then human-handoff |

> #2 builds the *tree*; #7 is the *acceptance contract each node of the tree must
> satisfy*. Together they are "an internal TODO list where every item has a
> checkable definition of done", which is the real-deployment robustness the
> request asked for.

### Decision 2 — Ground the reasoner's context (the situation report)

Mechanism changes in `context.py` / `core.py` (not packaged content). These are
the cheap, high-leverage upgrades the playbooks rely on.

1. **Robot self-model block** (EMOS Robot Resume). Render, from
   `RobotDescription` / `RobotCapabilities` / `ComputeSpec`, a `## ROBOT`
   context section with computed **reach hull bounds, camera FOV/frustum,
   base footprint, gripper width, payload, DOF**. This is derived once at
   `on_configure` and cached. `preflight-reach` (#5) reads it.
2. **Execution feedback on success *and* failure** (Inner Monologue). Add an
   `## EXECUTION` section fed by `rskill_runner` outcomes: one typed NL line per
   completed skill — `grasp ok: bottle in gripper` / `grasp failed: object not in
   gripper`. Today only failures reach the LLM (via FAILURES); making **success**
   visible closes the loop so the next tick reasons on reality, not stale belief.
3. **Reflection on the replanning ladder** (Reflexion). On failure, before the
   next rung, the reasoner writes a one-line diagnostic NL reflection ("failed:
   target out of reach → reposition base") into context — converting a raw error
   into a *strategy hint*. This upgrades the bare per-kind counter at
   `core.py:266` without removing its bound.
4. **(Deferred, §Phasing) One fused situation report** (OM1 NLDB). A "data fuser"
   step caption-merges the sections into a single timestamped NL report. Improves
   selection *and* makes traces human-readable (§1.8). Deferred because it
   restructures context assembly and the four-section form is adequate for the
   launch playbooks.
5. **(Deferred) Scene-graph expand/contract tools** (SayPlan). For multi-room
   homes, give the reasoner `expand_region` / `contract` tools over the ADR-0038
   collapsed graph so it pulls only task-relevant subgraphs instead of serializing
   the whole world. Deferred until scale demands it.

### Decision 3 — `MEMORY.md`: a self-maintained, persistent semantic memory

A new **per-robot persistent memory**, **complementary to** (not replacing) the
ADR-0038 scene graph. The split is deliberate:

- **ADR-0038 scene graph** = *geometric* memory: object poses, places, traversability. Answers **"where is the mug?"**
- **`MEMORY.md`** = *semantic / narrative* memory: preferences, corrections, lessons, durable home facts, open commitments. Answers **"how does this household like things done, and what have I learned?"**

**Storage model — MemGPT tiered, deliberately lazy.**
- **Core = `MEMORY.md`** (one human-readable Markdown file per robot). Always
  loaded into the S2 prompt as a fenced `## MEMORY` block, **strictly per-section
  size-capped**. This is "RAM".
- **Archival = `memory/*.jsonl`** (append-only log of every superseded/evicted
  entry, with timestamps). Unbounded "cold storage", paged in on demand by a
  search tool. *Plain files, no DB* — the lazy choice that works; revisit only if
  retrieval latency measurably hurts (`ponytail:` upgrade path = sqlite/FTS).

**Where it lives (runtime vs fixture).** The **mutable** `MEMORY.md` + `memory/`
archival live in a **writable runtime state directory**, addressed by a new
`memory_md_path` ROS parameter on `reasoner_node` — exactly mirroring ADR-0039's
`spatial_memory_path` (declared at `on_configure`, exposed in `sim_e2e.launch.py`).
It is **not** written into the checked-in `robots/<id>/` tree (read-only manifest
fixtures). A robot ships an optional **seed** `robots/<id>/MEMORY.seed.md` that is
copied to the runtime path on first boot; tests validate against that seed fixture
(§1.11 — a real file, no `"foo"` placeholders).

**File structure** (`MEMORY.md`, carries its own `schema_version`):

```markdown
# MEMORY.md  (schema_version: "0.1", last_consolidated: <iso8601>)

## Home Map / Places          # stable; EDIT in place (region-connectivity, EMOS L1)
- kitchen — ground floor, north. Contains: fridge, sink, counter.

## User Preferences           # distilled RULES, not events; EDIT in place (TidyBot)
- [imp:0.9] Clothes go in the bedroom drawer, not the shelf.
- [imp:0.7] Quiet operation after 22:00.

## Learned Lessons / Corrections   # distilled corrections; APPEND then consolidate (DROC)
- [imp:0.8, 2026-06-20] Grasp mugs by the handle — bare-body grasp slipped twice.

## Object-Location Log         # APPEND-mostly, timestamped; SUPERSEDE on re-observation
- bottle → fridge   (last_seen: 2026-06-22T10:04, conf:0.9, status:current)
- ~~bottle → counter (last_seen: 2026-06-19, status:stale)~~

## Open Tasks / Commitments    # EDIT; remove on completion
- [ ] Water the plants daily ~09:00 (recurring).
```

**Read policy (start of each task).** Load the whole bounded file into the
`## MEMORY` prompt block, fenced from live world state. If a section is at its cap,
load only entries scored highest by **`recency × importance × relevance`**
(Generative Agents) to the current goal, and leave the rest pageable via
`memory_search`.

**Write policy — explicit operations, never free-form rewrite** (Mem0
`ADD/UPDATE/DELETE/NOOP` + Zep temporal supersession). The reasoner edits memory
through **one typed tool** with an `op` discriminator (lazy: one tool, not five).
Two new `ReasonerToolCall` variants — both extend `_ReasonerToolBase`, so they
carry the shared optional `rationale` field like every existing variant:

```text
MemoryWriteTool   (discriminator "memory_write")          # the only write surface
  op:         Literal["add", "update", "supersede", "delete"]
  section:    Literal["home_map", "preferences", "lessons",
                      "object_locations", "open_tasks"]
  content:    str                       # the fact, as a distilled NL rule/line
  importance: float  (0.0–1.0)          # LLM-assigned at write time (Generative Agents)
  target:     str | None                # the existing entry to update/supersede/delete

MemorySearchTool  (discriminator "memory_search")         # read-only, archival paging
  query:      str
  section:    Literal[...] | None        # optional section filter
  limit:      int                        # bounded result count
```

Op semantics:
- *Home map / preferences / open tasks* → `update` / `delete` in place (durable
  facts, never duplicated).
- *Lessons / corrections* → `add` with timestamp + importance.
- *Object locations* → `supersede`: the new sighting becomes `status:current`;
  the prior same-object row is stamped `status:stale` with its old timestamp
  (validity-window, **evict don't delete** — a stale location is still a useful
  search prior for `find-object`).
- **`NOOP`** is not a tool call — it is the default: most ticks emit no
  `memory_write` at all.

Like the ADR-0039 spatial-query and ADR-0043 detector tools, both dispatch via
`reasoner_node` (here against the `MEMORY.md` file + archival JSONL, not a ROS
service) and return their result to the LLM's next step as a `PromptStamped`
(frame_id `"memory"`). `MemoryWriteTool` is the reasoner's **first write-capable
tool**; every other variant is read-only or actuation-proposing. It writes only
to the advisory memory file — it **cannot** actuate (§Safety).

**What is worth writing (decision rule the SOP gives the reasoner).** Write **iff**
the fact is *persistent across tasks* **and** *generalizable* **and**
*non-redundant* (checked against existing memory — Mem0 compare-before-write)
**and** `importance ≥ threshold`. Write: preferences, human corrections, stable
home/object facts, recurring commitments, hard-won lessons. **Never** write:
transient world state (live poses, battery), one-off observations, or anything
re-derivable from sensors next time. (Distil first: a preference/correction is
written as a *generalized rule*, TidyBot/DROC — not the raw event.)

**Reader/writer separation** (Statler) — phased. *Launch path:* the planner tick
**reads** the `## MEMORY` block and **commits** edits by emitting a
`MemoryWriteTool` call, which is itself a discrete, traced event — no whole-file
rewrite is ever possible, so the planner cannot silently corrupt memory. *Deferred
refinement (Phase 5+):* route every `MemoryWriteTool` through a dedicated
**memory-writer** LLM step that runs *after* a subtask verifies (a writer distinct
from the planner), for stricter auditability on long-horizon tasks. The launch
path already gives the audit trail (§1.8); the separate writer is a hardening, not
a prerequisite.

**Consolidation / staleness** (Generative Agents reflection + Episodic-Memory
Verbalization). Every *K* tasks, or when a section hits its cap, a `consolidate`
pass: merge duplicate sightings, fold repeated corrections into one generalized
rule, drop entries past a TTL to archival, and stamp `last_consolidated`. Keeps
the file bounded, human-readable, and free of contradiction.

> **Provenance / safety of memory.** `MEMORY.md` is **advisory only** — like the
> scene graph, a wrong memory yields a bad *plan* the C++ kernel still vetoes;
> memory never gates motors (§1.1). It is plain text the operator can read and
> hand-edit. No PII beyond what the operator consents to (§3); the writer SOP is
> instructed not to persist identifying detail without consent.

### Decision 3b — The deploy memory bundle (three modalities, loaded at deploy start)

`MEMORY.md` is **text** (narrative/semantic). The **2D nav occupancy grid** is a
binary image and the **3D world-state scene graph** is structured JSON — neither
belongs *inside* a Markdown file. Instead, generalize "what the robot remembers"
into a **per-robot memory bundle** directory, where each modality is persisted in
its native format and loaded at `on_configure` by its **correct consumer**:

```
<memory_dir>/                  # the memory_md_path dir from Decision 3
  MEMORY.md          # narrative/semantic   → the S2 reasoner reads (this ADR)
  scene_graph.json   # 3D world-state graph → SpatialMemory / recall_object (ADR-0038)
  map.png + map.yaml # 2D occupancy grid    → nav2 map_server → costmap + ADR-0044
  memory/*.jsonl     # archival (Decision 3)
```

- **Scene graph — already shipped, just standardize the path.** `SceneGraph`
  (`schemas.py:2265`) is a versioned JSON contract; `SpatialMemory.save()` /
  `SceneGraph.model_validate_json` persist and load it
  (`world_state/spatial_memory.py:465`/`:471`); `reasoner_node` already loads it at
  `on_configure` via the **`spatial_memory_path`** ROS parameter
  (`reasoner_node.py:966`, `_maybe_load_spatial_memory`), exposed as
  `spatial_memory_path:=<path>` in `sim_e2e.launch.py`. **No new persistence code**
  — the bundle simply places `scene_graph.json` at a conventional path the existing
  param points to (ADR-0038/0039 Phase 2c).
- **Occupancy grid — ROS-standard, add a `map_path` param.** A
  `nav_msgs/OccupancyGrid` is persisted by `nav2_map_server`'s `map_saver` as
  `map.pgm`/`map.png` + a `map.yaml` sidecar (resolution, origin, occupied/free
  thresholds) and reloaded at startup by `map_server`. Our `OccupancyGridIndex`
  (`world_state/grid.py`, the ADR-0044 approach-refinement grid) is already the
  *consumer* of a live grid; this ADR adds a deploy **`map_path`** ROS parameter
  (parallel to `spatial_memory_path`) that seeds the nav stack's `map_server` so a
  saved map is available from the first tick.
- **Consumer layering (do not blur it).** The reasoner **never reads map pixels**
  (ADR-0018 §4 "no pixels in v1"). The occupancy grid is consumed by **Layer-1 nav**
  (`map_server` → costmap) and the ADR-0044 approach refinement; the LLM only ever
  sees *derived* facts — regions, "kitchen reachable", traversability — via the
  scene graph and `resolve_place`. So the grid is a memory *artifact* whose consumer
  is the nav stack, not the reasoner.
- **`MEMORY.md` links, never embeds.** The `## Home Map / Places` section carries a
  pointer line per artifact — a **relative path + `last_mapped` timestamp** — to
  `scene_graph.json` and `map.yaml` (A-MEM cross-modal link). The binaries stay out
  of the text; the narrative memory just references them.
- **Staleness — prior, not ground truth.** A saved map/graph is a **seed**: at
  deploy start the robot re-localizes against the 2D map (AMCL) and SLAM
  (`openral_slam_bringup`) + the ADR-0038 Phase-2 builder keep correcting it live —
  the same supersede-on-re-observation discipline as the object-location log
  (Decision 3). `last_mapped` makes staleness decidable.
- **Safety unchanged.** Both artifacts stay **advisory** (the `SceneGraph`
  docstring: "advisory only — never a safety input"). The C++ kernel keeps its own
  **separate, ephemeral** ADR-0030 collision grid; this nav map never feeds it
  (§1.1).

`ponytail:` no new serializer (scene-graph JSON exists), no new map format (nav2
PGM/PNG+YAML is the ROS standard), no binary inlined into Markdown — the only new
code is the `map_path` param + seeding `map_server`, mirroring the existing
`spatial_memory_path` wiring.

### Decision 4 — CLAUDE.md & docs

§3 (Reasoner & dispatch) gains: a note that the reasoner's **read surface** now
includes `MEMORY.md` and installed playbooks, and its **write surface** includes
`memory_write` against `MEMORY.md` — **actuation authority unchanged**. The
`RSkillKind` list, `docs/methods/` (Layer-3/4 files), the repo state map
(`SCHEMAS` += `PlaybookContract`; `ReasonerToolCall` variants), and
`rskills/template/` (a `playbook` template) are updated in the same PR (§1.13/§1.14).

---

## Safety

- **No actuation authority added.** Playbooks and memory only change what the LLM
  *decides*; every motion is still an `execute_rskill` → Action chunk → ADR-0030
  C++ kernel, which disposes (§1.1, §3). A wrong playbook or stale memory produces
  a bad plan the kernel still vetoes — never a relaxed check.
- **Everything is bounded.** Each playbook has an explicit `max_steps`; active
  search keeps ADR-0039's `SearchBudget`; the replanning ladder keeps its per-kind
  cap. No new unbounded loop (§1.4 — bounds are explicit config, no hidden default).
- **Fully traced** (§1.8). Playbook selection, each composed tool call, every
  `memory_write` op, and each consolidation are OTel span events, so a run is
  replayable from the trace + the `MEMORY.md` snapshot at task start.
- **The dropped playbooks** (human-handover, resource-guard/recharge, safe-abort)
  sequence motions near the HAL/safety boundary and are **out of scope** here;
  they require a safety-WG ADR + hazard-log entry and are deferred deliberately.

---

## Schema & versioning

- **`RSkillKind += "playbook"`**, **`RSkillAction += PLAN ("plan")`**, and the new
  `PlaybookContract` are **additive, backward-compatible** schema changes (existing
  manifests stay valid) — no on-disk migration; evolves in place (§1.6).
  `_check_kind_consistency` gains the `playbook` branch (the per-kind contract
  table above). A playbook manifest still satisfies the **required** base fields
  (`actions ≥ 1` → `[PLAN]`, `chunk_size = 1`, `latency_budget`, `description`);
  `TestInTreeManifests` must stay green, and a real `rskills/find-object/` manifest
  is added as the first in-tree fixture.
- **`ReasonerToolCall`** grows by up to three variants — `memory_write` (write),
  `memory_search` (read), and the deferred `load_playbook` (read) — additive union
  members. `hypothesis` round-trip + discriminator-decode tests required (incl. the
  `rationale`/`rational` small-model-robustness alias the base already handles).
- **`MEMORY.md`** is a **new on-disk artifact** with its own `schema_version`
  starting at `"0.1"`; a backward-incompatible change to its layout bumps it and
  ships a migrator (§1.6). A real fixture (`robots/<id>/MEMORY.md`) validates in
  tests (§1.11 — no `"foo"` placeholders).

---

## Phasing

Each phase ships tests against real fixtures (no mocks, §1.11) and updates all
affected docs in the same PR (§1.14). Phases are independently landable.

1. **`playbook` kind + contract (schema-only).** `RSkillKind`,
   `RSkillAction.PLAN`, `PlaybookContract`, `_check_kind_consistency`,
   `rskills/template/` playbook template (with a hand-authored `PLAYBOOK.md`), and
   `generate_rskill_skillmd.py` support (a `playbook` entry in its `_KIND_NOUN`
   map so the generated discovery `SKILL.md` + `--check` gate cover playbooks like
   every other kind). Fuzz + in-tree-manifest tests. *No runtime behavior yet.*
2. **Context grounding (Decision 2.1–2.3).** Robot self-model block, success/failure
   `## EXECUTION` feedback, ladder reflection line. Pure-Python + a reasoner
   integration test on a real robot fixture. *Highest leverage, no new kind needed —
   could land first.*
3. **Playbook loading (default path).** `on_configure` injects installed,
   capability-matched `PLAYBOOK.md` bodies into the system prompt. Author
   **find-object** and **decompose-mission** (+ the **subtask-with-goal** contract)
   as the first three; sim test on the home fixture (reuses ADR-0039's wine task).
4. **`MEMORY.md` core (Decision 3, read + `memory_write` add/update/supersede).**
   File schema + loader + the `memory_write`/`memory_search` tools + reader/writer
   split. Author **verify-outcome** and **clarify-ambiguity** (they drive most
   writes). Live reasoner test: a correction persists and is recalled next task.
4b. **Deploy memory bundle (Decision 3b).** Standardize the bundle layout: place
   `scene_graph.json` at the conventional path (**reuses** the shipped
   `spatial_memory_path` loader — no new code) and add the `map_path` ROS param +
   `map_server` seeding for the 2D occupancy grid, exposed in `sim_e2e.launch.py`
   alongside `spatial_memory_path`. Integration test: a deploy boots with a seeded
   `scene_graph.json` + `map.yaml` and the reasoner answers `recall_object` while
   the nav costmap is populated from the saved map.
5. **Consolidation + remaining playbooks.** The `consolidate` reflection pass;
   **preflight-reach**, **stage-for-manipulation**. Importance×recency×relevance
   retrieval under cap; archival paging.
6. **Deferred / next-iteration.** Fused situation report (2.4), scene-graph
   expand/contract (2.5), `load_playbook` retrieval at scale, Voyager-style
   **verified-playbook promotion** (the agent proposes a new playbook only after a
   verified success), COME-robot failure-cause attribution written back as lessons.

---

## Alternatives considered

1. **Three separate ADRs** (playbook kind / context grounding / memory). Rejected
   as the primary framing: the three are one theme and tightly coupled (playbooks
   read memory and the self-model; memory is written during playbook execution).
   One ADR keeps the design coherent; **§Phasing lands them independently**, so the
   coupling costs nothing operationally. They may be *split during review* if a
   reviewer prefers — the phase boundaries are the natural cut lines.
2. **Hard-code the playbooks as Python BTs / a DSL** (extend ADR-0039's approach).
   Rejected. S2 is already an LLM; a Markdown SOP it *interprets* degrades
   gracefully and adapts to context, where a BT is rigid and a new DSL is a runtime
   to build and maintain. `ponytail:` Markdown-in-prompt is the version that ships;
   add structure only when a playbook provably needs control flow the LLM can't
   follow.
3. **Put memory in the ADR-0038 scene graph** (one store). Rejected. The scene
   graph is a typed *geometric* world model; preferences/corrections/lessons are
   *narrative* and benefit from being a human-readable, hand-editable file the
   operator can inspect (MemGPT core-memory model). Different access patterns,
   different audiences. They cross-link (a lesson may name an object/place) but
   stay separate stores.
4. **A vector DB / Mem0 / Zep service for memory now.** Rejected as premature.
   A bounded Markdown core + JSONL archival covers the launch need with zero new
   infra; the `op`-based write API is forward-compatible with swapping the archival
   tier for sqlite/FTS or a graph store later (`ponytail:` upgrade path noted in
   Decision 3).
5. **Inject playbooks/memory as static context every tick instead of tools.**
   Partially adopted: playbooks and `MEMORY.md` core *are* injected as context
   (they're small and always relevant); but the **iterative** parts (memory search,
   playbook retrieval at scale, active search) are tool calls, because — as ADR-0039
   established — iterative query/look/re-query is expressed naturally by the
   tool-call loop and badly by a static dump.
6. **Reuse `wam` for playbooks.** Rejected — WAM is a learned Layer-5 simulator,
   not an authored Layer-4 procedure (mirrors ADR-0047's rejection for VLMs).

---

## Consequences

### Positive
- The reasoner gains a **grounded situation report** (knows its own reach/FOV,
  sees success feedback, reflects on failure) — the bulk of what EMOS/OM1 do better,
  at low cost (Phase 2 is pure-Python, no new kind).
- High-level decision procedures become **first-class, authored, versioned,
  shareable rSkill content** — installable per robot, discoverable in the registry,
  refinable over time — instead of bespoke `reasoner_node` Python.
- The robot **persists distilled knowledge** (preferences, corrections, object
  locations, lessons) across episodes in a human-readable, operator-editable file,
  closing the single biggest real-deployment gap vs. a memoryless reasoner.
- "An internal TODO list where every subtask has a verifiable definition of done"
  (#2 + #7) gives long-horizon tasks robustness: a parent advances only when a
  child provably succeeded.

### Negative / risks
- **More to get right in the prompt.** A larger context (self-model + memory +
  playbooks) costs tokens and can distract a small local model. Mitigation:
  per-section caps, importance/recency retrieval, and `playbook_retrieval_available`
  for scale; budget validated on the reference host (§2 latency budgets).
- **Memory can be wrong or go stale.** Mitigated by temporal supersession,
  consolidation, advisory-only status (kernel still vetoes), and operator
  hand-editability. Never gates safety.
- **Authoring burden.** Each playbook is a human-written SOP; quality varies.
  Mitigated by the fixed skeleton, the publish gate (`rskill_publisher`), and
  deferring auto-authoring (Voyager promotion) to Phase 6 behind a verified-success
  check.
- **A new `RSkillKind` + union variants are schema surface.** Covered by the
  additive-only versioning above and the in-tree manifest test suite.

---

## Appendix A — literature map (sources)

**Context grounding:** EMOS — Robot Resume + feasibility self-check + execution
history (arXiv:2410.22662; github.com/SgtVincent/EMOS). OpenMind OM1 — NL data bus
+ Perception→Memory→Planning→Action, "Memory" = timestamped NL-fragment buffer
(github.com/OpenMind/OM1; docs.openmind.com). SayCan — affordance × LLM-preference
gating (say-can.github.io). Inner Monologue — closed-loop NL feedback
(arXiv:2207.05608). Reflexion — verbal self-reflection stored to memory
(arXiv:2303.11366). SayPlan — collapsed-graph expand/contract + classical path
offload (arXiv:2307.06135). ESC / L3MVN — commonsense object-search priors
(arXiv:2301.13166; survey arXiv:2403.09971). COME-robot — feasibility verify +
failure-cause attribution (arXiv:2404.10220). Statler — world-model reader/writer
split (arXiv:2306.17840). LLM-Planner — re-grounding on replan (arXiv:2212.04088).

**Memory:** MemGPT/Letta — tiered core/recall/archival, self-edit via function
calls (arXiv:2310.08560; letta.com). Generative Agents — memory stream,
importance×recency×relevance retrieval, reflection (arXiv:2304.03442). Mem0 —
extract-then-`ADD/UPDATE/DELETE/NOOP` (arXiv:2504.19413). Zep/Graphiti — temporal
validity windows (LongMemEval). A-MEM — linked notes (Zettelkasten). TidyBot —
distilled preference rules from few examples (arXiv:2305.05658). DROC — distilled,
retrievable human corrections (arXiv:2311.10678). ReMEmbR — spatio-temporal
object-location memory `{caption, position, timestamp}` (arXiv:2409.13682).
Episodic Memory Verbalization — hierarchical NL consolidation of life-long
experience (arXiv:2409.17702). RoboMemory — brain-inspired multi-memory taxonomy
(arXiv:2508.01415). Voyager — verified-skill library, embedding retrieval
(arXiv:2305.16291).

*Accessed 2026-06-23 via web research; arXiv IDs are primary where available.*
