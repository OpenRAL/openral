# ADR-0022: rSkill action vocabulary for the reasoner LLM tool palette

- Status: **Proposed**
- Date: 2026-05-24
- Amended: 2026-05-24 (see Amendments below)
- Related: [ADR-0018](0018-ros2-reasoner-supervisor.md) §4 (the tool
  palette this ADR enriches); [ADR-0013](0013-rskill-manifest-actuators-and-processors.md)
  (V1 in-place extension precedent);
  CLAUDE.md §6.4 (rSkill packaging), §7.6 (reasoner palette).

## Context

ADR-0018 F4 builds the reasoner's LLM tool palette from the installed
rSkill registry, filtered by the active robot's `RobotCapabilities`.
The shipping shape (`ToolPalette.execute_rskill_ids: frozenset[str]`)
gives the LLM a list of opaque HF Hub ids and **nothing else** — the
single Anthropic tool the palette emits today has the literal text

> "Invoke an installed rSkill. Allowed skill_id values:
>  ['OpenRAL/rskill-pi05-openarm-bimanual-pick-pipe-nf4',
>   'OpenRAL/rskill-smolvla-libero', ...]"

as its description (`python/reasoner/src/openral_reasoner/tool_use.py:218`).
The LLM has to *infer from slugs* what each skill does. That works
when the slug is descriptive (`pick-pipe-nf4`) and breaks down for
foundation checkpoints or fine-tunes whose slug names datasets, not
tasks (`rldx1-ft-rc365-nf4`).

The rSkill manifest already carries a free-form `description: str |
None` (max 500 chars) used by `ral skill list` — but the palette
never propagates it into the LLM tool schema, and there's no
structured signal at all about what task verbs the skill performs.

PR #142 (OpenArm bimanual scene) adds a palette-seed path that loads
rskill.yaml files at reasoner `on_configure`, making this gap more
visible: a fully-populated palette still gives the LLM nothing more
than a list of slugs.

## Decision

Add a closed-vocabulary **action verb** field to every rSkill, plus
two free-form discriminator fields, and surface them to the LLM as
**one tool per skill** (replacing the single-`execute_skill`-with-enum
pattern when the palette carries metadata).

### Schema (additive)

New enum in `openral_core`:

```python
class RSkillAction(str, Enum):
    PICK; PLACE; PICK_AND_PLACE; TRANSFER; GRASP; RELEASE
    OPEN; CLOSE; PUSH; PULL; SLIDE; INSERT; POUR; WIPE; ROTATE
    REACH
    NAVIGATE
    WAVE; SHAKE
    GENERALIST
```

`RSkillManifest` extensions:

- `description: str` — promoted from optional (`str | None = None`)
  to required (`min_length=1, max_length=500`). All 19 in-tree
  manifests already populate it; this tightens the schema so HF-Hub
  authors can't ship an undescribed skill.
- `actions: list[RSkillAction]` — REQUIRED, `min_length=1`. Closed
  vocabulary so the palette can pre-filter on verb (`PICK`) before
  hitting the LLM, the schema is unit-testable with hypothesis, and
  the enum can grow additively without breaking V1 manifests.
- `objects: list[str] = []` — free-form discriminative keywords
  (`cube`, `pipe`, `drawer`). Long tail (RoboCasa-365 alone has
  hundreds of object categories) makes a closed enum impractical;
  authors get to add whatever discriminators help the LLM.
- `scenes: list[str] = []` — free-form scene/environment keywords
  (`tabletop`, `kitchen`, `tabletop_2d`).

No `schema_version` bump (V1 has not been published yet — same
precedent as ADR-0013).

### Reasoner palette

`RSkillToolEntry` carries the per-skill metadata; `ToolPalette` gains
a `skills: tuple[RSkillToolEntry, ...]` field (the new primary
surface) while keeping `execute_rskill_ids` as a back-compat field
auto-derived via a `mode="before"` model-validator. `build_tool_palette`
populates `RSkillToolEntry` records in stable id-sorted order so the
LLM tool schema is deterministic (CLAUDE.md operating principle 8).

### LLM tool schema

`_tool_palette_to_anthropic_tools` emits one `execute_rskill__<slug>`
tool per `RSkillToolEntry`, where each tool's:

- **`description`** is the skill's manifest `description` + a
  structured `Actions: …. Objects: …. Scenes: ….` suffix. This is
  the field LLM tool-use APIs score against; it carries the real
  semantic signal.
- **`input_schema`** is `ExecuteRskillTool.model_json_schema()` with
  `skill_id` stripped from `properties` and `required` — the tool
  name itself is the authority on which skill to run, so the LLM
  only fills `prompt` / `deadline_s` / `rationale`.

`_decode_tool_payload` resolves the per-skill tool name back to the
canonical `execute_skill` discriminator via a palette lookup, and
overrides any LLM-supplied `skill_id` with the lookup result (so a
LLM that decides to also fill in `skill_id` can't smuggle in a
different skill than the one it picked).

`_skill_id_to_tool_name` slugifies `<owner>/<repo>` into the 64-char
Anthropic / OpenAI tool-name budget; long ids get an 8-char sha1
suffix to stay unique post-truncation.

### Back-compat path

Palettes built from only `execute_rskill_ids=frozenset(...)` (legacy
construction or call sites that haven't yet been migrated) keep
working: the LLM gets the original single `execute_skill` tool with
the id list embedded in its description. No call-site changes are
forced by this ADR; the new metadata path only kicks in once a
caller populates `skills`.

## Alternatives considered

**A. Add named macro-actions to `RobotCapabilities` instead (issue #46).**
That ADR — the OM1-style robot-native action vocabulary — is
complementary, not a substitute. Issue #46 lets the LLM dispatch
robot-native actions like `shake_paw` alongside skills; ADR-0022
tells the LLM what each *skill* can do. We'll likely do issue #46
later as a separate ADR.

**B. Free-form `tasks: list[str]` instead of a closed enum.**
Lower friction for authors, but the enum lets us write hypothesis
tests, pre-filter the palette before the LLM call, and grow the
vocabulary deliberately. The `objects` / `scenes` fields stay
free-form precisely because their long tail can't be enumerated.

**C. Keep the single-tool-with-enum schema, just enrich its description.**
The LLM tool-use APIs (Anthropic + OpenAI) are explicitly designed
around per-tool scoring; a single tool with a long description
forces the LLM to do internal text matching that the API's
tool-selection layer is built to handle natively. One tool per
skill is the idiomatic shape and the recommended pattern in
Anthropic's cookbook.

**D. Make `description` stay optional.** Then a freshly-published
HF-Hub skill could land with no LLM-readable signal at all. Promoting
it to required is the cheapest correctness gate — every in-tree
manifest already had one.

## Consequences

- Every in-tree rSkill carries curated `actions` / `objects` / `scenes`
  values committed alongside this ADR. HF-Hub-published skills that
  pre-date the ADR will fail to load until they add the required
  fields (loader raises `ROSConfigError` via Pydantic `ValidationError`).
  We accept this because V1 is unpublished — there are no
  externally-pinned manifests today.
- The reasoner LLM context budget grows roughly linearly with the
  number of installed skills (one tool description each, ~150–300
  chars), which is fine for the current `O(15)` in-tree set but
  bears watching as the catalog grows. Capping the tool list with
  pre-LLM action-verb filtering is the natural next step and is
  enabled cheaply by the closed `RSkillAction` enum.
- The back-compat shim on `ToolPalette` adds a small amount of
  validator logic. The mid-term plan is to drop the
  `execute_rskill_ids`-only path once every in-tree call site
  constructs `ToolPalette(skills=…)`. Out of scope for this ADR.

## Amendments

### 2026-05-24 — ADR renumbered 0021 → 0022

This document was originally filed as ADR-0021 in draft. A numbering
collision with the curl-installer ADR was resolved by reassigning the
next free slot: ADR-0022 (this document) for rSkill action vocabulary,
ADR-0023 for data-driven MuJoCo HAL. All internal cross-references
updated in the same renumbering commit.
