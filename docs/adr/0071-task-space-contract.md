# ADR-0071 — `TaskSpace`: one shared action-space contract across robots, rSkills, and scenes

- **Status:** Draft 2026-06-24. Proposes a layer-neutral `TaskSpace` value object
  and a single cross-layer compatibility check, replacing today's three implicit
  encodings (robot `supported_control_modes` set, rSkill
  `ActionContract`/`StateContract`, scene runtime-probed `action_dim`).
- **Date:** 2026-06-24
- **ADR number:** `0071`. The integer is not load-bearing — cross-refs use
  filenames.
- **Related:**
  - The task-space audit that motivates this ADR — a sweep of every declared
    action/observation space across `robots/`, `rskills/`, and `scenes/`. Its
    findings are summarized inline in the Context + Findings sections below; this
    ADR is its proposal, made concrete. (Tracked separately from this PR.)
  - ADR-0028b — `ActionSlot` / `ActionContract.slots`: the typed per-rSkill
    action-vector segments. **This ADR builds on them**, it does not replace
    them — a `TaskSpace` is the normalized, layer-neutral *view* over a slot
    layout (and the robot/scene equivalents that have no slot layout today).
  - ADR-0036 — `control_modes_for_representation()` + `canonical_slots_for_representation()`:
    the representation→ControlMode bridge. `TaskSpace.from_action_contract` reuses
    both so a skill that declares only a `representation` still expands to typed
    segments.
  - ADR-0027 / ADR-0014 — `StateContract` + `StateContractBindings`: the state
    side. This ADR is action-first; the state side is rolled into the same view
    but its layout/binding machinery is unchanged.
  - ADR-0007 — robot/sim split: the same Franka emits 7-D delta-EEF on LIBERO vs
    8-D joint-pos on a hardware deploy. `TaskSpace` is therefore a *per-deployment*
    join of (rSkill × robot × scene), never a field frozen on the robot.

## Context

The task-space audit pulled
every declared action/observation space out of `robots/`, `rskills/`, and
`scenes/` and found there is **no shared task-space contract**. Each layer
encodes it differently:

- **Robots** declare a flat `capabilities.supported_control_modes` set (e.g.
  `[joint_position]`) plus joints (some tagged `role: gripper`) and
  `end_effectors`. There is no notion of *how wide* each mode is or *which EE* it
  drives — just "this robot can, in principle, accept joint-position commands".
- **rSkills** declare `action_contract.dim` + optional `representation` + optional
  typed `slots` (ADR-0028b), and `state_contract.dim` + optional `layout`. The
  rich slot machinery exists but **most manifests don't use it**: `representation`
  is unset on the majority of actuating skills, so the ADR-0036 gate can't fire,
  and matching silently degrades to `embodiment_tags` string equality + raw dim
  equality.
- **Scenes** declare **nothing** about the task space. `action_dim` is probed at
  rollout time from the live sim env (`robosuite` robots / gym `action_space` /
  Isaac sidecar). A robot/skill/scene dimension mismatch is only discoverable by
  running the rollout.

The three layers connect only through (a) `embodiment_tags` string matching,
(b) dimension numbers happening to line up, and (c) translation buried in the sim
/ HAL adapters that is invisible in every manifest. Concrete fallout documented
in the audit: `actuators_required: joint_position` coexisting with
`representation: delta_ee_6d_plus_gripper` on every LIBERO skill (the two fields
describe different things and nothing reconciles them); gripper-bearing robots
that never advertise a `gripper_position` mode; `so100` vs `so101` modelling the
same gripper two different ways; `gr1` advertising an empty
`supported_control_modes` while a 29-D skill targets it.

We already have the right primitives — `ControlMode`, `ActionSlot`,
`ActionContract`, `StateContract`, `control_modes_for_representation()`. What is
missing is **one object that all three layers can produce and that a single
validator can compare.**

## Decision

Introduce a layer-neutral **`TaskSpace`** value object in `openral_core.schemas`,
plus one cross-layer compatibility function. `TaskSpace` is *derived*, never a new
hand-authored field on the robot manifest — it is computed from primitives each
layer already owns, so it cannot drift from them.

### 1. `TaskSpace` = ordered list of typed `TaskSpaceSegment`s

A `TaskSpace` describes an action interface as an ordered sequence of segments,
each tagging a `ControlMode`, a width, and (for EE-addressed modes) a target
end-effector. The gripper is therefore *always* an explicit segment — answering
the audit's central question ("is the gripper a dimension?") structurally instead
of by convention.

```python
class TaskSpaceFamily(str, Enum):
    JOINT = "joint"          # joint_position / joint_velocity / joint_torque
    CARTESIAN = "cartesian"  # cartesian_pose / cartesian_delta / cartesian_twist
    GRIPPER = "gripper"      # gripper_binary / gripper_position
    BASE = "base"            # body_twist / foot_placement
    DEX_HAND = "dex_hand"    # dex_hand_joint
    COMPOSITE = "composite"  # composite_mode multiplexer flag

class TaskSpaceSegment(BaseModel):
    family: TaskSpaceFamily
    control_mode: ControlMode
    width: int                       # > 0
    target: str | None = None        # EE name for cartesian/gripper/dex; None otherwise

class TaskSpace(BaseModel):
    segments: list[TaskSpaceSegment]
    representation: ActionRepresentation | None = None
    # total_dim is a derived property = sum(segment.width)
```

`family` is redundant with `control_mode` (derivable) but is stored so the object
reads cleanly in logs / dashboards and so the family↔mode consistency is
validated once at construction (reusing the existing `_JOINT_MODES` /
`_CARTESIAN_MODES` / `_GRIPPER_MODES` frozensets).

### 2. Each layer *produces* a `TaskSpace`

- **rSkill** → `TaskSpace.from_action_contract(ac, robot)`:
  - if `ac.slots` is set, map each non-`discard` slot → one segment (family from
    mode, width from `range`, target from `ee`);
  - elif `ac.representation` is set, expand via the ADR-0036
    `canonical_slots_for_representation()` and map those;
  - else fall back to a single `JOINT_POSITION` segment of width `ac.dim` (the
    legacy whole-vector path).
- **Robot** → read **directly by the matcher**, not coerced into a `TaskSpace`. A
  robot's supported space is a *menu* of independent control modes (each mode is a
  separate command interface), not one concatenated vector — so a `from_robot`
  that summed arm + cartesian + gripper widths would double-count. The matcher
  therefore inspects `robot.capabilities.supported_control_modes`, `robot.joints`
  (`role`), and `robot.end_effectors` directly. (A future `from_robot` capability
  descriptor is possible but deferred — it is not needed for the check.)
- **Scene** → the executed task space is declared once per scene-adapter family in
  `SCENE_FAMILY_TASK_SPACE` (a `SceneTaskSpace` of `modes` + optional fixed
  `action_dim`), keyed by the `evaluated_tasks` vocabulary. `scene_task_space_
  compatible(family, skill_ts)` is the matcher. (See §5 — this supersedes the
  originally-planned per-scene `expected_action_dim` field, which would have
  drifted across 40 YAMLs.)

### 3. One validator, three layers — sim vs real aware

`task_space_compatible(skill_ts, robot, *, hal_mode) -> TaskSpaceMatch` returns a
structured result (`ok: bool`, `reasons: list[str]`). It **mirrors the existing
reasoner deploy gate** `_action_executable` (ADR-0036) rather than inventing a
second rule: the control-mode check uses

- `hal_mode="sim"` → `SIM_EXECUTABLE_CONTROL_MODES` (the default-sim robosuite OSC
  / composite packers synthesise cartesian + gripper + base goals from joint
  commands, so a Franka advertising only `joint_position` still runs a
  `cartesian_delta` LIBERO skill in sim);
- `hal_mode="real"` → the robot's advertised `supported_control_modes` (real
  hardware needs the actual controller — no hidden OSC translation).

Both modes additionally require every EE-addressed segment to name a real
`end_effector` and the joint segments to fit the robot's joint count (physical
facts independent of HAL mode). This *subsumes and makes explicit* the three-way
implicit wiring. The reasoner's deploy palette gate and the `rskill_publisher`
validator are the intended callers; the scene gate additionally asserts
`skill_ts.total_dim == scene.expected_action_dim`.

The `hal_mode` split is essential: without it the gate flags every sim-only
checkpoint (all LIBERO/SIMPLER/robocasa skills) as incompatible with its own
robot, because those robots advertise only the real-hardware modes. The sweep in
Phase 1 confirmed this — 14/24 pairs fail in `real` mode (correct: they need an
OSC controller real HW doesn't declare) but 20/24 pass in `sim` mode (correct:
they run under `openral deploy sim` today).

### 4. Migration is additive and staged

- **Phase 1 (this ADR + draft schema):** land `TaskSpace`, `TaskSpaceSegment`,
  `task_space_compatible`, and the `from_*` producers in `openral_core`, with
  unit tests against real fixtures plus a **repo-wide compliance sweep**
  (`tests/unit/test_task_space_sweep.py`) that asserts every shipped robot +
  rSkill loads and pins the sim-mode compatibility of all 24 actuating
  skill×robot pairs. Nothing in the runtime calls it yet — pure, side-effect-free.
- **Phase 2:** wire the reasoner palette gate + `rskill_publisher` to call
  `task_space_compatible` *in addition to* (not instead of) the current
  `embodiment_tags` check; log disagreements as warnings to surface the
  inconsistencies without breaking existing rollouts.
- **Phase 3:** the manifest cleanups, including the four cross-layer gaps the
  Phase-1 sweep surfaced and recorded in `KNOWN_SIM_GAPS` (each ships as its own
  `fix(...)` commit per CLAUDE.md §1.15, then drops out of the set):
  - `3d-diffuser-actor-rlbench` slots name `ee="panda_gripper"`; franka_panda
    declares `panda_hand` (and its `cartesian_pose` runs via the RLBench sidecar,
    not the default-sim packers);
  - `pi05-robocasa365-human300-nf4` + `rldx1-ft-rc365-nf4` slots name
    `ee="panda_hand"`; panda_mobile declares its gripper EE as `panda_gripper`;
  - the `gr1` robot manifest models 17 joints but the RLDX-1 GR1 checkpoint
    drives a 29-DOF waist+arms+hands body — the manifest under-models the hands.

  Plus the broader cleanups (set `representation` on joint-space skills, add
  `gripper_position` to gripper-bearing robots, normalize `so101`'s gripper
  joint, give `gr1` real `supported_control_modes`). Add `scene.expected_action_dim`.
- **Phase 4:** make `task_space_compatible` blocking; retire the raw-dim-equality
  fallback.

`schema_version` on the rSkill/robot manifests does **not** bump in Phase 1: no
on-disk field changes (the `TaskSpace` is derived). It bumps in Phase 3 only for
manifests that gain a `representation` / `slots` they lacked.

### 5. Phase 4 amendment — the scene leg, completed (implemented)

A follow-up exhaustive audit (robots × rSkills × scenes × backends, across every
benchmark / sim / deploy-sim scene) confirmed the rSkill↔robot leg was sound but
surfaced that the **scene leg was still unconnected**: nothing statically tied an
rSkill's `TaskSpace` to the scene it runs in, and two rSkills *mislabelled* their
action space and only passed the rSkill↔robot gate by coincidence of dimension.
Phase 4 closes both.

**Scene task space — a central family registry, not a per-scene field.** The
originally-planned `scene.expected_action_dim` is **superseded** by
`SCENE_FAMILY_TASK_SPACE: dict[str, SceneTaskSpace]` in `openral_core`. The
executed control interface is a property of the *scene-adapter family* (LIBERO
OSC, RoboCasa composite, gym-aloha joints, the RLBench PyRep sidecar), **not** of
the coarse `PhysicsBackend` (one `MUJOCO` value spans LIBERO / MetaWorld /
RoboCasa / aloha, each driving a different interface) and **not** of the
individual scene instance (the four LIBERO suites and RoboCasa's ~100 prebuilt
tasks share one adapter). So the task space is declared **once per family**, keyed
by the same vocabulary rSkills already use in `evaluated_tasks` (the leading token
before any `/`). This is the same "derive / declare once, don't hand-author per
asset" principle that rejected a per-manifest `task_space:` block (Alternatives
§1): a per-scene `expected_action_dim` on 40 YAMLs would immediately drift.
`SceneTaskSpace` carries `(modes, action_dim | None, runs_via_default_packers)`;
the `runs_via_default_packers=False` flag marks the dedicated-controller adapters
(RLBench `CARTESIAN_POSE` motion planner; RoboCasa-GR1 whole-body BASIC composite)
— the scene-leg analogue of `KNOWN_SIM_GAPS`.

`scene_task_space_compatible(family, skill_ts) -> TaskSpaceMatch` is the third leg
of the gate: every mode the skill emits must be in the family's executed set, and
— when the family fixes a width — the dims must match. Together with
`task_space_compatible` (rSkill × robot) this closes the rSkill × robot × scene
triangle the audit found open. The repo-wide sweep gains
`test_scene_families_are_declared` (a new task family must record what it drives)
and `test_scene_executability_matches_known_state` (exact, like the sim sweep;
`KNOWN_SCENE_GAPS` is empty after the fixes below).

**Two mislabels fixed — every rSkill `TaskSpace` is now honest.**

- `smolvla-metaworld` declared a bare `dim=4`, which fell through to the legacy
  `JOINT_POSITION(4)` path and passed only because Sawyer has ≥4 joints. MetaWorld
  actually drives a 3-D EE translation delta + 1 gripper. Phase 4 adds
  `ActionRepresentation.DELTA_EE_3D_PLUS_GRIPPER` (and a 3-wide cartesian slice in
  `canonical_slots_for_representation`, vs the existing 6-wide), and the manifest
  now declares it → `cartesian_delta(3) + gripper_position(1)`, executable in sim,
  correctly *not* on real Sawyer hardware (which advertises no OSC controller).
- `diffusion-pusht` / `pusht_2d`: the robot advertised `cartesian_pose` while its
  two DoF are literal prismatic `tip_x`/`tip_y` joints and the gym-pusht env
  commands their absolute position. The robot mode is now `joint_position` and the
  skill declares `joint_positions(2)`, making it real-executable (was a real-mode
  contradiction) — no new "planar" enum needed.

The `DELTA_EE_3D_PLUS_GRIPPER` addition retires the `smolvla-metaworld` entry from
`_KNOWN_DEFERRED` in `test_rskill_action_contracts_executable.py` (the self-policing
guard forced its removal the moment the representation existed).

## Alternatives considered

1. **A new hand-authored `task_space:` block on every manifest** (the audit's
   first-cut sketch). Rejected as the *primary* representation: it duplicates
   information already in `joints` / `end_effectors` / `action_contract.slots` and
   would immediately drift. `TaskSpace` is derived instead. (A manifest may still
   *override* the derivation later if a checkpoint needs it, but the default is
   computed.)
2. **Extend `ActionContract` to cover the robot + scene sides.** Rejected:
   `ActionContract` is explicitly per-checkpoint (ADR-0007 / ADR-0019) and carries
   absolute `[start, end]` ranges into a specific policy vector. A robot's
   capability and a scene's expectation are not policy vectors; forcing them
   through `ActionContract` would overload its meaning.
3. **Do nothing; rely on `embodiment_tags` + runtime probing.** Rejected: this is
   the status quo the audit indicts. Mismatches surface only at rollout, and the
   `representation`/`actuators_required` contradiction stays unresolved.

## Consequences

- **Positive:** one comparable object across the three layers; the gripper is a
  first-class dimension; mismatches caught at config-load; the audit's cleanups
  become mechanically checkable (each flips a warning green); no new drift-prone
  hand-authored field.
- **Negative / cost:** a derivation layer to maintain (`from_robot` must track
  morphology conventions); Phase 2 will surface many existing warnings (that is
  the point, but it is noise until Phase 3 lands); `from_robot` for humanoids
  (`gr1` dex hands, `g1`/`h1` whole-body) needs care — drafted conservatively and
  flagged where `n_dof: None` blocks width arithmetic.
- **Follow-ups:** `docs/methods/` entries for the new public symbols; repo state
  map `SCHEMAS` update once the symbols land; the Phase 3 manifest cleanups each
  get their own `fix(...)` commit per CLAUDE.md §1.15.
