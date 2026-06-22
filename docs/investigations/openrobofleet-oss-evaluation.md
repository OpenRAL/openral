# Evaluation: OpenRoboFleet ideas → OpenRAL (open-source side only)

> Investigation note, 2026-06-22. Scope: **open-source OpenRAL only.** This
> triages the ideas in the external *OpenRoboFleet* vision proposal, keeping the
> ones that improve the OSS harness and explicitly excluding the monetizable
> fleet/cloud backend. On-disk format (YAML vs TOML vs JSON) is out of scope.
> Every code claim below was checked against the tree at the time of writing
> (line numbers drift — `grep` the symbol, don't trust the number).

## Context

*OpenRoboFleet* reframes OpenRAL as an open robotics-harness SDK + "fleet-as-code"
standard with a proprietary backend (QDS / Pipeline / RLinf) underneath. The
question this note answers: **which ideas genuinely improve the open-source
OpenRAL**, excluding anything about fleet management or cloud dispatch (the
monetizable layer). The output is a prioritized, independently-adoptable list —
some items are PR-sized, some are ADR-first multi-month efforts. Sequencing notes
call out dependencies.

### Grounding (verified against the codebase)

- **`rskill` conflates several concerns** in one word/manifest: the executable
  artifact (`RSkillManifest.kind` ∈ `vla | wam | ros_action | ros_service |
  detector | vlm | reward` — `schemas.py` `RSkillKind`), the reasoning slot
  (`role` ∈ `s0 | s1 | s2`), the capability doc (auto-generated `SKILL.md`), and
  the runtime node (`rSkillBase`). `RSkillManifest` (`RobotDescription` sibling in
  `python/core/src/openral_core/schemas.py`), loader
  (`python/rskill/src/openral_rskill/loader.py` — `rSkill.from_pretrained` /
  `from_yaml`), tool call `ExecuteRskillTool` (`schemas.py`), palette
  (`python/reasoner/src/openral_reasoner/palette.py`, `build_tool_palette`).
- **LLM access is hand-rolled, embedded in the reasoner.** A custom
  `ToolUseClient` Protocol plus `AnthropicToolUseClient` /
  `OpenAICompatibleToolUseClient` in
  `python/reasoner/src/openral_reasoner/tool_use.py`; `anthropic` *and* `openai`
  are unconditional deps of `openral_reasoner` (its `pyproject.toml`). **No
  LiteLLM anywhere.** Tool-use is Pydantic structured output (every tool emits
  `model_json_schema()` to the provider, validated against the `ReasonerToolCall`
  discriminated union) — *not* free-form JSON; CLAUDE.md §"Reasoner & dispatch"
  requires that any change preserve it. Env contract: `OPENRAL_REASONER_LLM_*`
  (`PROVIDER` ∈ `anthropic | openai-compatible | openrouter | ollama`).
- **`RobotDescription` is monolithic** (`schemas.py`): sensors are an inline
  `list[SensorSpec]`; `safety: SafetyEnvelope` is an inline field; compute is
  capability flags plus `RobotCapabilities.onboard_compute_tops` (on
  `RobotCapabilities`, *not* `RobotDescription` — the manifest only carries an
  untyped `onboard_compute: dict`); **no "rig"** (multi-robot composite). Note
  `SensorSpec` already exists as a standalone, reusable Pydantic *class* (with its
  own `frame_id` / `parent_frame` / `static_transform_xyz_rpy`) — what's missing
  is authoring a sensor as a standalone *record resolved by id* and referencing
  it. URDF/MJCF/SRDF are *referenced*, not re-authored (`AssetRefs` + `UrdfAsset`
  + `resolve_asset`, `python/core/src/openral_core/assets.py`, ADR-0058). tf2
  frames already use `static_transform_xyz_rpy` (sensors) /
  `base_to_root_xyz_rpy` (`UrdfAsset`, ADR-0027).
- **`SafetyEnvelope` already carries the human-in-loop concept.**
  `SafetyEnvelope.human_in_loop_required: list[str]` (rSkill names requiring
  supervision). A standalone-envelope item must *keep* this field, not invent it.
- **No hierarchical/composite policy.** `COMPOSITE_MODE` (`schemas.py`) is an
  unrelated robosuite control-mux flag, *not* policy composition.
- **Safety: gate exists, kernel partial, sandbox missing.** Gate =
  `packages/openral_safety/openral_safety/supervisor_node.py`
  (`SafetyPassthroughNode`); C++ kernel scaffold = `cpp/openral_safety_kernel/src/`
  (envelope / collision / validator / lifecycle); **no isolation sandbox** for
  untrusted weights/code, despite existing provenance gates
  (`OPENRAL_ALLOW_UNSAFE_PICKLE`, `OPENRAL_ALLOW_REMOTE_CODE`,
  unverified-provenance warnings). Out-of-process isolation already exists for
  perception/eval/VLA backends (`tools/rldx_sidecar.py`, `isaac_sidecar.py`,
  `robometer_sidecar.py`, … ; GR00T ZMQ sidecar per ADR-0046) — but as ad-hoc
  infra, not a declared safety boundary.
- **World-state store already matches the doc**: transient `WorldStateAggregator`
  (`python/world_state/src/openral_world_state/aggregator.py`) + persistent
  `SceneGraph` spatial memory (`schemas.py`, ADR-0038). No work needed.
- **Already aligned with the doc's conventions**: root `AGENTS.md`, per-rskill
  `SKILL.md` (`tools/generate_rskill_skillmd.py`, ADR-0055), referenced (not
  re-authored) kinematics, and a declarative MJCF composer seam (ADR-0029
  `SceneDefaults.composition` → `SceneComposition.composer` = `module:fn → MJCF`;
  extension seams `HalEntrypoints` + `SCENES.register`).

---

## Tier 1 — High ROI, contained, low risk

### 1. Untangle `rskill` → Policy / Skill / tool-call invocation

- **Why.** "Skill" means several different things in code today (artifact,
  runtime node, kind, role, palette entry, SKILL.md). It blocks hierarchy
  (item 6), confuses every reader, and the doc's split is clean: **Skills
  describe, Policies execute, tool-calls invoke.**
- **What.** Make the *executable artifact* a **Policy** (`kind` = vla / diffusion
  / classical / composite); keep **Skill** = the `SKILL.md` discovery doc the
  reasoner reads; keep the tool-call (`execute_rskill` → `execute_policy`).
  `role` stays as the s0/s1/s2 slot. This is the conceptual keystone for item 6.
- **How.** ADR first (crosses layer 3↔4). `schema_version` bump + migrator
  (the manifest is published now). Stage it: introduce `Policy`/`PolicyManifest`
  as the canonical name with `rSkill` kept as a deprecated alias for one release,
  rename `ExecuteRskillTool`→`ExecutePolicyTool`, update `docs/methods/*`,
  `tools/generate_rskill_skillmd.py` (ADR-0055), and the repo-state map. **Risk:
  very large rename surface** — high value but its own epic, not bundled. (Note:
  the tool discriminator has *always* been `execute_rskill`; only the ROS action
  topic is `/openral/execute_skill`. There is no prior manifest-field rename to
  cite as precedent.)

### 2. Dedicated LiteLLM-based LLM package, separate from the policy plane

- **Why.** The reasoner hand-rolls per-provider clients and bundles two SDKs;
  LiteLLM gives one interface for any provider with far less maintenance. It is
  also the **prerequisite for sharing S2 across composite policies** (item 6).
  The two planes (chat-completion reasoning vs S1 action streaming) are
  conceptually distinct and should not share an interface.
- **What.** New `openral_llm` package wrapping LiteLLM behind the *existing*
  `ToolUseClient` Protocol, preserving the `OPENRAL_REASONER_LLM_*` env contract
  and Pydantic structured tool-use. Reasoner depends on it; composite policies
  reuse it.
- **How.** Implement as a drop-in adapter behind `ToolUseClient` so the reasoner
  change is minimal; verify tool-use parity (Anthropic `tool_use` ↔ OpenAI
  `tools`) against a real provider, not a mock (CLAUDE.md §1.11). ADR (contained,
  reasoner-scoped). Medium size, low risk.

### 3. JSON-Schema emission + an `openral check` graph validator  *(implemented — see PR)*

- **Why.** Format-agnostic win: the *contract* (not the file syntax) becomes
  legible to non-Python consumers (dashboard, third parties, coding agents
  reviewing a diff) and a single command validates a whole robot+skill+scene set
  instead of discovering errors at runtime. The doc's core deliverable is "the
  schema is the standard" — this realizes it cheaply.
- **What.** Two halves, one already existing: (a) **JSON-Schema emission already
  exists** — `tools/schema_export.py` emits `model_json_schema()` for the public
  `openral_core` models into `docs/reference/schemas/`, drift-gated by
  `quality.yml`. It was simply **missing `RSkillManifest`** (the most important
  external contract) — now added, so the rSkill manifest schema is published. (b)
  The genuinely new piece is **`openral check`**, a static, host-independent
  validator that cross-checks the declarative set in one pass — every manifest
  parses, `file:`/`ros2://` asset refs resolve, scene `robot_id`s resolve to a
  real robot dir, each rSkill's embodiment tags reach at least one in-repo robot,
  and each sensor `parent_frame` is a declared tf2 frame.
- **How.** Pure reuse: `RobotDescription.from_yaml`, the scene tiers' loaders,
  and `resolve_asset` (`assets.py`) — no parallel validation logic, and no
  duplication of the existing `tools/schema_export.py` emitter or the
  host-specific compatibility report (`openral_detect.check_installed_rskills`,
  which wraps the production `rSkill.check_compatibility`). Lightweight, low risk,
  no schema change. Good first PR — it also de-risks items 1/4/5 by giving them a
  validation harness, and immediately surfaced a real latent issue (the SO-100/101
  wrist camera's `parent_frame` names a joint, not a link frame).

---

## Tier 2 — Meaningful capability, medium effort

### 4. First-class reusable **Sensor** + standalone deny-by-default **Safety envelope**

- **Why.** Sensors are inline today, so a camera calibration can't be *shared*
  across robots/rigs by reference; the safety envelope is buried inside the robot
  manifest, so it isn't independently reviewable/diffable. The doc makes both
  first-class, referenced by id — sensors get versioned on their own, and the
  envelope becomes a deny-by-default artifact that actuation must pass through.
- **What.** Allow a `SensorSpec` and a `SafetyEnvelope` to be *authored as
  standalone records* resolved by `type`+`id` and referenced from the robot/rig —
  keeping inline as sugar for the simple case. The envelope keeps its existing
  `human_in_loop_required` field (no new permission concept needed — it's already
  there).
- **How.** Discriminated-union loader keyed on `type`+`id` (the doc's
  Kubernetes/Terraform model), built on the ADR-0058 asset-ref grammar.
  `schema_version` bump + migrator; ADR. The contained, valuable slice of the
  doc's "component model." Safety-envelope extraction needs safety-WG review
  (CLAUDE.md §3).

### 5. **Rig**: a composite multi-robot / multi-embodiment record

- **Why.** Bimanual is faked today as one `RobotDescription`
  (`embodiment_kind=bimanual`); true multi-arm / multi-robot cells (the doc's
  `bimanual_so101_cell`) have no representation. A rig composes robots + sensors +
  compute by id with tf2 frames, covers sim and real uniformly, and becomes the
  unit a harness, safety envelope, and RL rollout (item 8) drive.
- **What.** New `Rig` schema referencing member component ids + a frame tree;
  the embodiment the harness drives.
- **How.** ADR (new layer-spanning entity). Reuse existing tf2 conventions
  (`static_transform_xyz_rpy`, `base_to_root_xyz_rpy`). Largest schema lift in
  the contained set; depends on item 4's component loader. This is the heart of
  "rig as code" that is *not* fleet management — it's local embodiment
  composition.

### 6. Hierarchical / composite policies (S2 below top-level)

- **Why.** Absent today. A composite policy that carries its own S2 reasoner and
  sequences child policies (tidy-table → pick-place) enables reusable task-level
  behaviors without forcing everything into the single top-level reasoner.
- **What.** `kind: "composite"` with `children: [policy ids]` + an embedded
  reasoner config; recursive tool dispatch.
- **How.** ADR (layer 3↔4). Reuse `ReasonerCore` + the shared LLM package
  (item 2). **Sequenced after items 1 and 2** (needs the Policy naming and the
  shared LLM interface). Large.

### 7. Safety **sandbox** for untrusted extension / policy code

- **Why.** The one missing canonical safety module. It aligns directly with
  OpenRAL's existing-but-ad-hoc posture: untrusted `*.pt`, `trust_remote_code`
  models, and third-party backends currently load in-process behind only env-var
  gates. OpenRAL already isolates some backends out-of-process (the rldx / GR00T
  ZMQ sidecars, the isolated NF4 venvs) — this generalizes that into a *declared*
  boundary.
- **What.** A process/resource isolation boundary (the doc's "typed ports are
  the boundary you ship across") for loading untrusted weights/backends.
- **How.** Safety-WG reviewer + hazard-log + TDD (CLAUDE.md §3). Build by
  promoting the existing sidecar isolation pattern (`tools/*_sidecar.py`,
  ADR-0046) into a first-class, manifest-declarable sandbox. Medium-large;
  safety-gated.

---

## Tier 3 — Big bets (high value, multi-month, research-y)

### 8. Standardize a rollout/env Protocol + reference single-node RL loop (open part only)

- **Why.** OpenRAL has `SimRollout` (`python/sim/src/openral_sim/rollout.py`) but
  no standard reset/step/trajectory + reward-hook interface. The doc is explicit
  that the **single-node reference loop and the rollout/env interface are the
  open piece** — the *scaled, distributed* learner (RLinf, weight server) is the
  proprietary part and is **out of scope**. The open interface is still useful
  for local RL/finetuning.
- **What.** A rollout/env Protocol over the sim/real rig abstraction, a
  reward-input hook (reward-model ref or inline spec — OpenRAL already has
  `kind:reward` rSkills / Robometer, ADR-0057, and the critic-score topic
  ADR-0064), and a reference local loop. Explicitly exclude the distributed
  weight-sync server.
- **How.** Reuse `SimRollout` + `kind:reward`. ADR. Clearly fence the scope:
  edge/single-node interface only.

### 9. Minimal-core + extension registry (Pi-inspired)

- **Why.** The 8-layer stack is largely hardwired; HAL adapters, sensor drivers
  and sim backends are named in code. Seeds of a registry already exist
  (`HalEntrypoints` import strings, `SCENES.register`, the backend import
  convention, the rSkill registry of ADR-0055) — a real registry would let third
  parties extend without patching core.
- **What.** Formalize one extension registry (entry-points based) for HAL
  adapters / sensor drivers / sim backends / reasoners / observability; shrink
  core to run loop + config loader + world-state + dispatch + registry.
- **How.** ADR-heavy, touches all layers — exactly the "refactor across all 8
  layers" CLAUDE.md §6 says STOP on without an ADR. **Stage it**: first promote
  the existing seams (`HalEntrypoints`, `SCENES`, ADR-0055) into one documented
  registry; do *not* rip up the layer boundaries in one PR. Highest risk/effort.

### 10. Component IR with pluggable renderers + procedural sim ("sim curriculum as code")

- **Why.** The doc's "React for robots" (one composition tree → URDF/MJCF/USD).
  OpenRAL already references (not re-authors) kinematics and already has a
  declarative MJCF composer (ADR-0029). The genuinely reachable, valuable slice
  is the **procedural-sim payoff**: sweep props (poses/friction/lighting) → emit
  N randomized MJCF stages → domain-randomized RL/eval curriculum.
- **What.** A procedural/domain-randomization layer on the existing composer
  seam. Treat a fully symmetric multi-renderer IR — especially USD — as
  long-horizon: USD needs Isaac, which is py3.12-incompatible and proprietary
  (per prior feasibility work, ADR-0045), so it is not practically reachable now.
- **How.** Extend the ADR-0029 composer with parameterized sweeps + seed; reuse
  the existing composition machinery. MJCF domain randomization is achievable
  OSS; the USD renderer is gated on Isaac feasibility and should not be promised.

---

## Explicitly out of scope (per the brief)

These are the monetizable / backend / fleet-ops parts of the document and were
deliberately excluded: `fleet.toml` topology placing rigs on compute, the GitOps
reconciler, the QDS git-native registry, the Pipeline DAG-as-code compiler +
typed-op executors, the RLinf distributed learner + weight server,
cloud/edge/split dispatch, the SaaS telemetry dashboard, and op-memoization. The
**rename to OpenRoboFleet** is a strategic/branding decision, not an OSS-technical
improvement, so it is noted but not recommended as an engineering item — though
its underlying framing ("the harness config is version-controlled code you can
diff, review, and roll back") is already largely true of OpenRAL's git-tracked
manifests.

---

## Recommended sequencing

1. **Item 3** (validator/JSON-Schema) — cheap, de-risks everything else.
   **Implemented in the PR that adds this note (`openral check`).**
2. **Item 2** (LiteLLM package) — contained, unblocks item 6.
3. **Item 1** (Policy/Skill split) — keystone rename; its own epic.
4. **Item 4** (Sensor + Safety envelope as components) — then **Item 5** (Rig).
5. **Item 7** (sandbox) — safety-gated, parallelizable.
6. **Item 6** (composite policies) — after 1 + 2.
7. **Items 8–10** — big bets; ADR-first, scope-fenced, only if prioritized.

## Verification approach (for whichever items are picked)

- Each schema change validates against a **real fixture** in `robots/` /
  `rskills/` / `scenes/` (CLAUDE.md §1.11 — no mocks) and ships a `hypothesis`
  round-trip + JSON-Schema test on the Pydantic model.
- Item 2: run the LiteLLM adapter against a real provider and confirm a live
  structured tool-call round-trips (parity with the current client).
- Items 1/4/5/6: bump `schema_version` + ship a migrator; run `openral check`
  (item 3) across the full robot/skill/scene set; re-run a deploy-sim dispatch
  end-to-end to confirm no regression.
- Item 7: TDD + safety-WG sign-off + hazard-log; prove the boundary is at least
  as conservative as today.
- Every picked item updates `docs/methods/*`, the repo-state map, and the
  relevant ADR in the same PR (CLAUDE.md §1.13–1.14).
