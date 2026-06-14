# ADR-0026: rSkill structured goal parameters (`goal_params_json`)

- Status: **Proposed**
- Date: 2026-05-27
- Related: [ADR-0018](0018-ros2-reasoner-supervisor.md) §4 (the typed
  tool-call palette this ADR extends); [ADR-0022](0022-rskill-action-vocabulary.md)
  (the action / verb vocabulary surfaced to the LLM); [ADR-0024](0024-ros-wrapped-rskills.md)
  (wrapped-ROS rSkills whose `default_goal_json` motivated this work);
  CLAUDE.md §3 (types are the contract), §1.4 (explicit beats implicit),
  §1.6 (schemas evolve, never silently).

## Context

The Reasoner's `ExecuteRskillTool` (the LLM-visible dispatcher) today
carries only three fields:

```python
class ExecuteRskillTool(_ReasonerToolBase):
    tool: Literal["execute_rskill"] = "execute_rskill"
    rskill_id: str
    prompt: str = ""
    deadline_s: float = 0.0
```

The matching `ExecuteRskill.action` IDL mirrors this:

```
string rskill_id
string revision
string prompt
string prompt_metadata_json
float64 deadline_s
```

`prompt` is the *only* per-dispatch free-form input. For **VLA skills**
(SmolVLA, π₀.₅, ACT, …) this is sufficient by design — the policy
adapter writes the prompt directly into the observation dict under the
`"task"` key (`python/rskill/src/openral_rskill/smolvla.py:175`) and the
underlying VLA consumes it as the natural-language instruction signal
that grounds its action distribution. **There is no need for structured
parameters on a VLA skill**: the VLA itself is the natural-language
parser. "Pick up the red mug" already routes through the VLA's
language conditioning.

For **wrapped-ROS skills** (ADR-0024 — `kind: ros_action` /
`ros_service`) the situation is reversed. MoveIt, Nav2, FoundationPose
are **deterministic planners**: they need a numeric goal (target joint
positions, target pose in the map frame, target object id) the moment
the action goal is sent. The wrapped-action adapter currently builds
its goal from `manifest.ros_integration.default_goal_json`
(`ros_action_rskill.py:345`) — a per-manifest constant. The
``prompt`` field IS stored on the adapter but never consumed; the
``default_goal_json`` is sent verbatim every dispatch.

This created the user-visible bug that motivated this ADR:

```
operator prompt:   "move back 1 meter"
reasoner dispatch: execute_skill rskill_id='OpenRAL/rskill-nav2-navigate-to-pose'
nav2 received:     {pose: {position: {0.0, 0.0, 0.0}}}  ← hardcoded
nav2 result:       Goal Coordinates of (0.0, 0.0) was outside bounds
```

The LLM has the linguistic capability to compute "(11.52, -8.21) =
(12.52, -8.21) − (1, 0)" from the prompt + current robot pose. It
does not have the *contract surface* to pass that result through to
Nav2.

This ADR adds that surface — generically, so the same field works for
every wrapped-ROS skill (and any future structured rSkill), and so
backward-compat is a no-op for VLAs that never need it.

## Decision

Add an optional `goal_params_json: str = ""` field at four layers, and
an optional `goal_params_schema: dict | None = None` field on
`RSkillManifest` that the Reasoner exposes to the LLM tool palette so
the LLM knows what fields each skill accepts.

### 1. Pydantic — `ExecuteRskillTool` (LLM-visible)

```python
# python/core/src/openral_core/schemas.py
class ExecuteRskillTool(_ReasonerToolBase):
    tool: Literal["execute_rskill"] = "execute_rskill"
    rskill_id: str
    prompt: str = ""
    goal_params_json: str = ""   # NEW — serialised JSON object
    deadline_s: float = 0.0
```

The LLM emits this via the provider's tool-use API. Field is `""` by
default so existing rSkill dispatches stay byte-identical.

### 2. ROS IDL — `ExecuteRskill.action`

```
string rskill_id
string revision
string prompt
string prompt_metadata_json
string goal_params_json    # NEW — serialised JSON object
float64 deadline_s
---
bool success
string failure_kind
string failure_evidence_json
---
ExecuteRskillFeedback feedback
```

ROS IDL doesn't carry default values; consumers treat empty string as
"no params". `openral_msgs` `schema_version` stays `"0.1"` —
adding a `string` field at the end is wire-compatible with existing
producers (the deserializer fills empty when the wire payload is
shorter) per the CDR add-field-at-end rule. **`openral_msgs/CHANGELOG.md`
notes the additive field**; no version bump pre-publish (CLAUDE.md
§1.6 — schemas evolve in place pre-publish).

### 3. `rSkillBase.__init__` + resolver dispatch

```python
# python/rskill/src/openral_rskill/base.py
class rSkillBase:
    def __init__(
        self,
        *,
        manifest: RSkillManifest,
        prompt: str = "",
        prompt_metadata_json: str = "",
        goal_params_json: str = "",   # NEW
        # … existing params unchanged
    ) -> None:
```

The rskill_runner_node's `_resolve_and_check_skill` threads
`req.goal_params_json` into the resolver call alongside `prompt`.
Resolvers that don't accept it (out-of-tree adapters predating this
ADR) raise the standard `TypeError` at construction — operators see a
loud "kwarg unexpected" trace rather than silent drop.

### 4. `ROSActionRskill._configure_impl` — merge over `default_goal_json`

```python
# python/rskill/src/openral_rskill/ros_action_rskill.py
def _configure_impl(self, ...):
    base = json.loads(self._integration.default_goal_json or "{}")
    overrides = json.loads(self._goal_params_json or "{}")
    self._goal_dict = _merge_nested(base, overrides)   # overrides win
```

`_merge_nested` is a recursive dict merge — leaves replace, dicts
recurse. JSON arrays replace verbatim (no element-wise merge — too
surprising). The base + overrides shapes are both validated against
`manifest.goal_params_schema` when set, at configure time, so a
malformed LLM payload fails loud at goal-send time rather than mid-
inference. Validation uses `jsonschema` (already an indirect dep
through several other packages); if the schema field is absent, no
validation runs (today's behavior).

### 5. `RSkillManifest.goal_params_schema` — declared per-skill

```yaml
# rskills/rskill-nav2-navigate-to-pose/rskill.yaml
goal_params_schema:
  type: object
  properties:
    target_x:    { type: number, description: "Map-frame x (m)" }
    target_y:    { type: number, description: "Map-frame y (m)" }
    target_yaw:  { type: number, description: "Map-frame yaw (rad)" }
    frame_id:    { type: string, enum: ["map", "base_link"], default: "map" }
  required: ["target_x", "target_y", "target_yaw"]
```

The reasoner's `build_tool_palette` exposes this schema in the LLM
tool definition (per-rskill `parameters` JSON Schema), so the LLM's
structured output for the `execute_skill` tool call carries
well-formed `goal_params_json` for each skill it might pick.

A skill that **omits** `goal_params_schema` still works — the LLM
sees the same flat `(rskill_id, prompt, deadline_s)` surface for that
skill, exactly as today.

### 6. VLA backward-compat

VLA adapters (SmolVLA, π₀.₅, ACT, …) accept the new constructor
kwarg but ignore it. The prompt remains the sole natural-language
control signal — that's what the VLA was trained on. Their manifests
omit `goal_params_schema`, the LLM doesn't see structured params for
them, and the adapter doesn't need them. **No behavioural change.**

This is the answer to the user's design question: today the VLA
*receives the prompt* through `prompt: str` on `rSkillBase.__init__`
and writes it into the observation dict's `"task"` key. That hook is
already structured for VLAs — the LLM speaks language, the VLA
understands language. Wrapped-ROS skills are deterministic planners
that don't speak language, so they need a parallel typed channel.

## Consequences

**Positive:**

- Nav2's `"move back 1 meter"` works end-to-end. The LLM reads the
  prompt + current pose (from world_state) and emits
  `goal_params_json='{"target_x": 11.52, ...}'`. The adapter merges
  it onto `default_goal_json` and Nav2 receives the right pose.
- MoveIt becomes parameterisable from the LLM: `"reach to (0.5, 0,
  0.3)"` → `goal_params_json='{"ee_position": [0.5, 0, 0.3]}'`.
- FoundationPose and other future structured rSkills get a typed
  surface at zero additional tool-palette cost (still one
  `execute_skill` tool variant).
- The LLM tool definition reflects each skill's actual contract —
  the model gets a per-skill JSON Schema and can fill it
  deterministically rather than hoping the prompt-string survives
  freeform parsing on the adapter side.

**Negative / risks:**

- One more layer of optional indirection. Mitigated by:
  - The field defaulting to `""` everywhere.
  - Schema validation at configure-time (loud failure rather than
    mid-action mystery).
  - Mandatory unit tests for `_merge_nested` + per-skill schema
    enforcement.
- LLM tool definition grows. For 24 in-tree rSkills, each with a
  10-line schema, ~250 extra tokens per system prompt. Mitigated by
  only including `goal_params_schema` when the manifest declares
  it — VLAs add zero overhead.
- `jsonschema` dep becomes load-bearing. Already transitively
  present via several extras; promotion to a first-class dep is
  contained.

## Implementation plan

Three PRs, each independently shippable:

**PR1 — schema + IDL (no behaviour change):**

- Add `goal_params_json: str = ""` to `ExecuteRskillTool` in
  `python/core/src/openral_core/schemas.py`.
- Add `string goal_params_json` to `packages/msgs/action/ExecuteRskill.action`.
- Add optional `goal_params_schema: dict[str, Any] | None = None` to
  `RSkillManifest`.
- Hypothesis round-trip + JSON-Schema fuzz on `ExecuteRskillTool` and
  `RSkillManifest`.

**PR2 — thread the field through (no LLM-side change yet):**

- `rSkillBase.__init__` accepts `goal_params_json: str = ""`.
- rskill_runner_node forwards `req.goal_params_json` to the resolver.
- `ROSActionRskill._configure_impl` merges `goal_params_json` over
  `default_goal_json` via `_merge_nested`. Validates against
  `manifest.goal_params_schema` when set.
- VLA adapters accept + ignore the kwarg.
- Reasoner dispatcher stamps `goal.goal_params_json =
  call.goal_params_json` (today's reasoner emits `""` until PR3).
- Unit tests:
  - `_merge_nested` semantics (overrides win, arrays replace,
    deep-nested keys preserved).
  - Schema-validation rejection at configure.
  - Hermetic round-trip: ExecuteRskillTool → action goal → rSkill
    constructor preserves the field.

**PR3 — surface `goal_params_schema` to the LLM tool palette:**

- `build_tool_palette` includes per-skill JSON Schema in the LLM
  tool definition when the manifest declares it.
- Anthropic / OpenAI-compat tool-use clients accept the schema and
  pass through to the provider's `tools=[…]` argument.
- Update `rskills/rskill-nav2-navigate-to-pose/rskill.yaml` and
  `rskills/rskill-moveit-joints/rskill.yaml` (then `openral-moveit-plan-arm`,
  renamed under ADR-0054) with their
  per-skill schemas.
- Integration test: `tests/integration/test_reasoner_node_end_to_end.py`
  feeds a `"move back 1 meter"` prompt + a Cyclonix-mocked LLM that
  returns a structured `goal_params_json`, asserts the wrapped
  action receives the merged goal.

## Alternatives considered

**(A) Per-skill `ExecuteRskillTool` variants** — e.g.
`NavigateToPoseTool(target_x, target_y, target_yaw)`,
`MoveItPlanArmTool(joint_targets)`. Strongly typed, no JSON merging.
**Rejected**: a closed Pydantic union over every rSkill on every host
is a maintenance nightmare. ADR-0018 §4 deliberately keeps the
discriminated union small (four variants); a per-skill explosion
defeats that.

**(B) Stuff structured params into `prompt_metadata_json`.**
**Rejected**: `prompt_metadata_json` is the F10 router's source/priority
fan-in channel — overloading it with goal params conflates two
distinct contracts and breaks the prompt router's invariants. A
separate field is honest.

**(C) Parse the prompt string in the adapter (LLM-side or
adapter-side).** **Rejected**: the adapter is C-fast and deterministic;
asking it to do natural-language parsing in the wrapped-action path
is a layering violation and a reliability hazard. The Reasoner LLM
already has the linguistic context — let it produce structured output
once, then deliver it as structured data.

**(D) Status quo: edit each manifest's `default_goal_json` per
deployment.** **Rejected**: that's the v1 workaround documented in
the Nav2 manifest, and it's why this ADR exists. The whole point of a
reasoner that selects skills from natural language is that the
operator doesn't pre-pin every goal at manifest-edit time.

## Verification

- `just lint` clean (ruff + mypy --strict).
- `just test` clean: ~10 new hermetic unit tests across the three PRs
  (constructor wiring, merge semantics, schema rejection, LLM tool
  palette inclusion).
- Integration: `tests/integration/test_reasoner_node_end_to_end.py`
  adds a "navigate to (x, y)" path with a FakeToolUseClient returning
  structured params; asserts the wrapped Nav2 action receives the
  merged goal.
- End-to-end live: `openral deploy sim --config
  scenes/deploy/robocasa_pnp.yaml`, send `"move back
  1 meter"` from the dashboard prompt, observe Nav2 receives
  `(11.52, -8.21)` and the robot drives backward. Add as a manual
  test recipe under `docs/runbooks/` (not gated in CI — needs a live
  robocasa kitchen).

## Out of scope

- A general-purpose pose-frame transformer (`base_link` ↔ `map` ↔
  `odom`) that the LLM can call from a tool. The current ADR keeps
  pose-frame conversion the LLM's responsibility (it sees TF
  via WorldState). A future ADR can promote frame-transform to its
  own tool if the LLM proves unreliable at it.
- Streaming structured params mid-action (the LLM updating Nav2's
  target while it's en route). Today's `ExecuteRskill` action is
  one-goal-per-dispatch; mid-action updates would need a different
  ROS interaction model (e.g. service calls to the wrapped server's
  config socket) and is out of scope.
- Cross-skill parameter linkage (LLM emits `goal_params_json` for
  skill A whose result is consumed by skill B). The replanning
  ladder (ADR-0018 F4 §"bounded retry counter per failure kind")
  is the right surface for this — it's a future ADR.

## Amendment 2026-06-08 — three-tier scene paths

ADR-0041 split `scenes/` into deploy/sim/benchmark tiers. The end-to-end
live demo above (`openral deploy sim --config …`) now points at
`scenes/deploy/robocasa_pnp.yaml` because there is no DeployScene sibling
for the old `scenes/benchmarks/panda_mobile_navigate_kitchen.yaml` and
`openral deploy sim` rejects non-DeployScene tiers strictly. The
substrate (`panda_mobile` in a robocasa kitchen) is unchanged; the
robocasa scene id is `PickPlaceCounterToCabinet` rather than
`NavigateKitchen`, but the Nav2 + structured-params behavior exercised
by the demo is layer-on-top and independent of the task block. See
ADR-0041 and [`scenes/README.md`](https://github.com/OpenRAL/openral/blob/master/scenes/README.md) for the
per-tier strict-CLI matrix.
