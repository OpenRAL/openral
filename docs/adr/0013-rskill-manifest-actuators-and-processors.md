# ADR-0013: rSkill manifest — actuators, custom-embodiment hatch, deferred processors

- Status: Accepted
- Date: 2026-05-24
- Amended: 2026-05-24 (see Amendments below)
- Related: ADR-0012 (licensing), CLAUDE.md §1.6 (schemas evolve
  never silently), §1.9 (license lineage), §6.4 (rSkill packaging),
  §7.4 (VLA license matrix); ADR-0009 (separate sim and benchmarking)
- Supersedes: none (extends the manifest surface introduced earlier in
  this PR series in-place — the schema was never tagged or published, so
  no version bump; ``schema_version`` stays at ``"0.1"``)

## Context

The V1 RSkillManifest, introduced in the same PR series that adopts this
ADR, hardened the manifest's surface across 11 axes — closed
`embodiment_tags` Literal, SemVer + HF Hub regex, `weights_uri`
discriminator, derived `is_commercial_use_allowed`, typed `benchmarks`
dict, required `chunk_size`, required `model_family`. After V1 landed in
draft, two gaps surfaced in design review:

1. **Asymmetric compat check.** V1 declares what a skill needs on the
   *input* side via `sensors_required: list[SensorRequirement]`; the
   loader matches each entry against `RobotDescription.sensors`. But
   there is no symmetric `actuators_required` on the *output* side.
   Action-space compatibility is implicitly trusted via the
   `embodiment_tags` intersection — a π0.5 manifest tagged `franka_panda`
   is *trusted* to emit a 7-DoF joint-position action, with nothing in
   the manifest to catch the mismatch if the wrapped checkpoint emits,
   say, cartesian deltas. This is a real failure mode: lerobot
   checkpoints that predate the PolicyProcessorPipeline migration (the
   ACT-ALOHA pair shipped in tree, for example) carry no embedded
   action-space metadata, and a misconfigured `model_family` plus a
   matching `embodiment_tag` would silently dispatch wrong-shape actions
   to the HAL.

2. **No story for non-in-tree embodiments.** The V1 closed Literal of 9
   canonical embodiments (`so100_follower`, `franka_panda`, `ur5e`,
   `ur10e`, `sawyer`, `aloha`, `pusht`, `google_robot`, `widowx`)
   protects against typos (`lerobot`, `libero` etc were silently
   rejected) but locks third parties out of declaring custom embodiments
   without a schema bump. There is no published escape hatch.

A third concern was tabled: **pre / post-processing declarations on the
manifest** (e.g. an `image_normalize: imagenet` or an opaque
`processor_pipeline_uri` pointing at a stored
`lerobot.PolicyProcessorPipeline` artifact). The lerobot ecosystem
currently handles this *inside* the policy checkpoint via the embedded
processor pipeline (`policy.preprocess` / `policy.postprocess` invoked
from `select_action`). The shape of that pipeline file is in flux
upstream — adding a manifest field for it today would either invent a
parallel vocabulary or pin to a moving target. Per CLAUDE.md §1.6
("schemas evolve, but never silently"), shipping a field the loader
cannot enforce is debt.

## Decision

Extend the rSkill manifest surface in place with **actuators +
custom-embodiment escape hatch**; **defer processors** until lerobot's
pipeline format stabilises (or until we need an enforcer for non-lerobot
policies). ``schema_version`` stays at ``"0.1"`` because the schema has
not been published — the entire in-flight design lands as one
pre-publish baseline; a real bump is reserved for the first
published-after-1.0 shape change.

### 1. New `ActuatorRequirement` model

```python
class ActuatorRequirement(BaseModel):
    kind: ControlMode                # e.g. JOINT_POSITION, CARTESIAN_DELTA
    n_dof: int | None = None         # auto-filled for predefined embodiments
    vla_action_key: str | None = None  # auto-filled for predefined embodiments
```

`kind` reuses the existing `ControlMode` enum (`joint_position`,
`joint_velocity`, `joint_torque`, `joint_trajectory`, `cartesian_pose`,
`cartesian_delta`, `cartesian_twist`, `body_twist`, `foot_placement`,
`gripper_binary`, `gripper_position`, `dex_hand_joint`). This is the
single source of truth for action-space typing across the repo
(`Action.control_mode`, `RobotDescription.action_spec.control_mode`,
HAL action emission); using a separate enum would invent drift.

`n_dof` and `vla_action_key` are optional on the manifest. For
predefined embodiments, the loader auto-fills them from
`RobotDescription.action_spec` at compatibility-check time. For custom
embodiments (see §2), they MUST be set on the manifest — the loader
has no canonical robot YAML to crib from.

### 2. `"custom"` embodiment escape hatch

`EmbodimentTag` Literal gains `"custom"` as the tenth allowed value.
When `"custom"` appears in `embodiment_tags`, the manifest MUST set a
new `embodiment_extra: EmbodimentExtra | None` field:

```python
class EmbodimentExtra(BaseModel):
    """Declares the sensor + actuator surface for a 'custom' embodiment.

    Required when 'custom' appears in RSkillManifest.embodiment_tags;
    forbidden otherwise (model_validator cross-check).
    """
    sensors: list[SensorRequirement] = Field(min_length=1)
    actuators: list[ActuatorRequirement] = Field(min_length=1)
```

Sensors reuse `SensorRequirement`; actuators use `ActuatorRequirement`.
Cross-validators on `RSkillManifest`:

- `"custom" in embodiment_tags` ↔ `embodiment_extra is not None`
- When `"custom"` is present, every entry in `actuators_required` must
  carry both `n_dof` and `vla_action_key` (the loader has no auto-fill
  source).

The closed Literal stays in place for the 9 canonical embodiments
(typo guard for the 95% case); `"custom"` is the explicit
"I know what I'm doing" door.

### 3. Keep `schema_version` at `"0.1"`

`schema_version: Literal["0.1"] = "0.1"` — unchanged from the initial
shape. The schema has not been tagged or published; the entire
in-flight design lands as one pre-publish baseline. CLAUDE.md §1.6's
"schemas evolve, but never silently" rule applies at the *published*
boundary — pre-release iteration does not bump. A real bump is
reserved for the first published-after-1.0 shape change. The reading
is: **baseline (final) ≡ baseline (initial) + actuators +
custom-embodiment hatch**.

### 4. `actuators_required` is required, `min_length=1`

Every in-tree skill emits at least one action (otherwise it is not a
policy). Making the field `min_length=1` ensures the symmetric guard
is loud — an empty list would silently revert to the asymmetric V1
behaviour this ADR is fixing. The
authoritative values per in-tree manifest:

| skill | kind | n_dof |
|---|---|---|
| smolvla-base / smolvla-libero / pi05-libero-nf4 / xvla-libero | `joint_position` | (auto-fill from robot) |
| smolvla-metaworld | `joint_position` | (auto-fill from sawyer manifest) |
| act-aloha / act-aloha-insertion | `joint_position` | (auto-fill from aloha bimanual: 14) |
| diffusion-pusht | `cartesian_delta` | (auto-fill from pusht_2d: 2) |

### 5. Processors: explicitly out of scope for this iteration

No `preprocessing` / `postprocessing` / `processor_pipeline_uri` field
ships in V1. Rationale:

- The lerobot checkpoint format embeds preprocessing inside the
  policy artifact (`policy.preprocess(batch)` / `policy.postprocess(out)`
  invoked from `select_action`). For lerobot-wrapped skills (100% of
  in-tree skills today), the manifest does not need to redeclare it.
- The lerobot `PolicyProcessorPipeline` file format is still being
  iterated upstream — adding a manifest field that points at it would
  pin a moving target.
- Adding a field with no enforcer at the loader level violates
  CLAUDE.md §1.6 ("schemas evolve, but never silently"). The enforcer
  is the right time to add the field, not before.

When non-lerobot policies (OpenVLA, GR00T, Cosmos-driven WAMs)
become first-class, that PR ships its own ADR with the enforcer and
the manifest field together.

## Consequences

**Wins**

- Symmetric input/output compat check: `sensors_required` /
  `actuators_required` both validated against the robot before any
  motor command is dispatched.
- The closed embodiment Literal stays loud for the canonical case while
  unblocking third-party / one-off custom rigs via `"custom"` +
  `embodiment_extra`.
- Reuses `ControlMode`, `SensorRequirement` — no parallel enums or
  duplicate dataclasses.
- `n_dof` / `vla_action_key` stay implicit for the 9 canonical
  embodiments → in-tree manifests stay terse (skill author writes
  `kind: joint_position`, the loader fills the rest from the robot
  YAML).

**Costs**

- All 9 in-tree `rskills/*/rskill.yaml` files need a one-line
  `actuators_required` block (done in the same PR). No
  `schema_version` change.
- The `is_commercial_use_allowed` derivation, the initial-V1
  validators, and every test fixture / hypothesis strategy that
  constructs a `RSkillManifest` keep working unchanged — this
  iteration is strictly additive.
- Migration tool: not needed. The schema sits at the pre-publish
  baseline (`schema_version: "0.1"`) and evolves in place until a
  real post-1.0 bump is required (see CLAUDE.md §1.6).

**Risks**

- **Auto-fill ambiguity**: if a canonical robot's `action_spec`
  declares more than one `ControlMode` (e.g. supports both
  `joint_position` and `joint_torque`), the loader needs a rule to
  pick which one fills `n_dof` / `vla_action_key`. The rule:
  match the `kind` declared on the manifest against the robot's
  `action_spec.control_mode`; raise `ROSCapabilityMismatch` if the
  robot does not advertise the requested kind. Multi-mode robots
  are an open question — none of the 9 in-tree robots are multi-mode
  today.
- **`"custom"` embodiment compatibility**: a manifest tagged
  `"custom"` cannot intersect any canonical robot's `RobotCapabilities.embodiment_tags`
  unless that robot ALSO tags itself `"custom"`. Intentional — custom
  manifests are run-it-yourself; the user wires the rig.

## Alternatives considered

1. **Open `embodiment_tags` to free strings + add a `kind` discriminator**.
   Rejected: walks back the loud-typo-rejection that V1 added; the 95%
   case (a typo) silently always-misses. The `"custom"` literal + extra
   block gives the same flexibility while preserving the V1 safety net.

2. **Make `actuators_required` optional with a default empty list**.
   Rejected: an empty list is silently the same as no check, which
   defeats the purpose. Making it `min_length=1` ensures every skill
   declares at least one action shape.

3. **Introduce a parallel `ActuatorKind` enum instead of reusing
   `ControlMode`**. Rejected: drift. `ControlMode` already lives at
   the layer boundary between Skill → Safety → HAL and is the
   normative wire type on `Action`; using a different enum on the
   manifest side would force a mapping table.

4. **Ship `preprocessing` / `postprocessing` fields now, even without
   an enforcer**. Rejected per CLAUDE.md §1.6 ("schemas evolve, but
   never silently") — see Decision §5.

5. **Bake `n_dof` and `vla_action_key` into the schema as required**.
   Rejected: redundant with `RobotDescription.action_spec` for the 9
   canonical embodiments, and a maintenance burden on skill authors
   (every fork of an embodiment would need to redeclare the same
   numbers). The auto-fill rule keeps in-tree manifests one line and
   forces explicitness only when there is no canonical robot to crib
   from.

## Migration plan

The same PR that adopts this ADR:

1. Adds `ActuatorRequirement` + `EmbodimentExtra` to
   `python/core/src/openral_core/schemas.py`.
2. Adds `"custom"` to the `EmbodimentTag` Literal.
3. Adds `actuators_required` (required, `min_length=1`) and
   `embodiment_extra: EmbodimentExtra | None = None` to
   `RSkillManifest`.
4. Adds the cross-validator: `"custom" in embodiment_tags` ↔
   `embodiment_extra is not None`; custom actuators must have
   `n_dof` and `vla_action_key` set.
5. Keeps `schema_version: Literal["0.1"] = "0.1"` (schema has not been
   published — no bump until the first published-after-1.0 shape).
6. Adds `actuators_required: [{kind: ...}]` to all 9
   `rskills/*/rskill.yaml` files.
7. Updates `tests/unit/test_rskill_manifest.py` /
   `test_rskill_loader.py` / `test_schemas_fuzz.py` /
   `test_rskill_eval_validation.py` / `test_cli_skill.py`.
8. Updates `docs/METHODS.md` and the repo state map.

A future PR will add the actual auto-fill logic in
`rSkill.check_compatibility` (today the manifest declares the
contract; the loader's compat check is what consumes
`actuators_required`'s `n_dof` / `vla_action_key` against
`RobotDescription.action_spec`).
