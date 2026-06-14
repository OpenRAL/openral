# ADR-0028: rSkill action-contract slot dispatch + robot.yaml gripper-joint convention

- Status: **Proposed (split: 0028a + 0028b + 0028c)**
- Date: 2026-05-27
- Related: [ADR-0007](0007-robot-sim-split.md) (per-checkpoint
  action contract — the field this ADR extends);
  [ADR-0013](0013-rskill-manifest-actuators-and-processors.md) (closed `EmbodimentTag` literal —
  the same closure model applies to action `control_mode`);
  [ADR-0018](0018-ros2-reasoner-supervisor.md) §F1 (typed `ActionChunk`
  wire format already carries `control_mode` — this ADR makes it actually
  load-bearing);
  [ADR-0019](0019-rosbag2-lerobot-dataset-bridge.md)
  (`ActionContract.dim` as dataset-bridge contract);
  [ADR-0024](0024-ros-wrapped-rskills.md) (skill_runner is the dispatch
  layer);
  [ADR-0025](0025-reasoner-managed-background-services.md) (panda_mobile
  HAL accepts `JOINT_POSITION` + `BODY_TWIST` — this ADR adds the
  remaining surfaces);
  [ADR-0027](0027-rskill-state-contract-bindings.md) (state-side layout
  adapters — this ADR is the symmetric action-side pattern);
  CLAUDE.md §1.3 (types are the contract), §1.4 (explicit beats implicit),
  §3 (layer boundaries — Skill ↔ Safety ↔ HAL).

> **Forward-reference (2026-06-14):** the Status line above predates the final split. This
> decision was realised across four files — this one (the "0028a" foundation),
> [ADR-0028b](0028b-rskill-action-contract-slots-dispatch.md) (slot dispatch),
> [ADR-0028c](0028c-panda-mobile-hal-cartesian-gripper-handlers.md) (cartesian/gripper
> handlers), and [ADR-0028d](0028d-panda-mobile-hal-joint-velocity-torso-handlers.md)
> (joint-velocity + composite-mode handlers). See the clustered
> [ADR index](README.md#b-rskill-packaging-manifest-action-contracts) for the full family.

## Context

The trace below E-stops within 22 seconds of dispatch:

```
[reasoner] dispatch: execute_skill rskill_id='OpenRAL/rskill-pi05-robocasa365-human300-nf4' prompt='pick up kettle'
[skill_runner] policy_adapter.skill_built skill='…pi05-robocasa365-human300-nf4' joint_units=radians perm=None is_gripper=[]
[skill_runner] policy_step step=1 raw_policy_action=['+0.014', '+0.000', '-0.003', '+0.001', '-0.000', '+0.000', '-0.989', '+0.001', '-0.000', '+0.000', '+0.000', '-0.991']
[safety_kernel] safety.envelope_violation kind=controller field=n_dof joint=65535 step=65535 value=12 limit=10
[hal_panda_mobile] openral_hal.estop_received; ignoring further commands until reset.
```

The policy emits a **12-D action vector**. The robot manifest declares
**10 joints** (3 holonomic base + 7 Franka arm). The safety kernel
correctly rejects the mismatch.

The 12-D layout is verified from `python/sim/src/openral_sim/backends/robocasa.py:244-264`
and matches the per-channel unnormalizer stats across four RoboCasa-family
pi0.5 checkpoints (`OpenRAL/rskill-pi05-robocasa365-human300-nf4`,
`RoMALab/pi05_robocasa-MG_300`, `RoMALab/pi05_robocasa-MG_1000_v1`,
`ruiname/pi05-robocasa-10tasks-200k`):

```
[arm_osc(6) + gripper(2) + base(3) + torso(1)]
   ↑           ↑           ↑          ↑
 cartesian   gripper      body      placeholder
 delta       width        twist     (always -1)
```

This is a *mixed-semantics* action vector — not 12 joint positions. Two
gaps are responsible:

1. **The rSkill manifest cannot describe the layout.** Today
   `action_contract` carries only `dim` (and an optional opaque
   `representation` literal). The runner has no field to read that says
   "slot 0–5 is cartesian delta, slot 6 is gripper width, slot 8–10 is
   body twist, drop the rest." `actuators_required` declares
   `kind: joint_position` — *factually wrong* for this checkpoint —
   and there is no validation against ground truth.

2. **The HAL has no slot dispatch.** Even with a correct layout
   declared, `python/runner/src/openral_runner/ros_publishing_hal.py:240-257`
   rejects every non-joint `ControlMode` with `ROSConfigError`, and
   `packages/openral_hal_panda_mobile/lifecycle_node.py:752` only
   whitelists `JOINT_POSITION` + `BODY_TWIST`. `CARTESIAN_DELTA` and
   `GRIPPER_POSITION` exist as enum values in
   `openral_core.ControlMode` but have no consumer.

Adjacent to (1), **the robot.yaml fleet is inconsistent about whether
the gripper is a joint**:

| Robot | Gripper as joint | Gripper in `end_effectors:` | Status |
|---|---|---|---|
| `franka_panda`, `widowx`, `so100_follower`, `google_robot`, `aloha_bimanual`, `openarm` | ✓ (with `position_limits`) | ✓ (`parallel_gripper`) | canonical |
| `panda_mobile` | ✗ (only in `end_effectors`) | ✓ (`parallel_gripper`) | outlier |
| `sawyer` | ✗ (only in `end_effectors`) | ✓ (`parallel_gripper`) | outlier |

Two robots haven't been ported to the convention the rest of the fleet
already follows. This blocks any future joint-position-with-gripper VLA
on `panda_mobile` and `sawyer`, and creates latent footguns in
`rskills/pi05-robocasa365-human300-nf4/rskill.yaml:91` which binds
`gripper_qpos_joints: ["panda_finger_joint1", "panda_finger_joint2"]`
— neither joint declared in `robots/panda_mobile/robot.yaml`. The bind
only works today because the state assembler reads from live
`JointState` (which the MJCF emits), not from the robot.yaml manifest.

Without a structural way to identify the gripper channel, the runner
also resorts to `"gripper" in name.lower()` substring sniffing
(`packages/openral_rskill_ros/openral_rskill_ros/rskill_runner_node.py:1258`)
— fragile across naming conventions, breaks on any joint with "gripper"
in the name that isn't the gripper.

## Decision (split into three sub-ADRs)

### 0028a — robot.yaml gripper-joint convention + `JointSpec.role` (THIS ADR)

The cheap, independently-valuable foundation. Lands first.

1. **Schema (`openral_core.JointSpec`):** add optional
   `role: Literal["arm","base","gripper","torso","leg","head","neck","wheel","unknown"] = "unknown"`.
   Structural identification of joint purpose without name-substring
   matching. Default `"unknown"` so existing manifests load unchanged.
2. **Schema (`openral_core.EndEffectorSpec`):** formalize the previously
   silently-ignored `actuated: bool = True`. Two existing yamls already
   set it; Pydantic's `extra="forbid"` would have rejected them on a
   strict load.
3. **Robot manifests:** add a `panda_gripper` joint to `panda_mobile`
   (1-DoF prismatic, `[0.0, 1.0]`, matching `franka_panda` exactly).
   Add a `right_gripper` joint to `sawyer`. Result: every parallel-gripper
   embodiment in the fleet declares the gripper as a joint.
4. **Role annotations:** tag every existing gripper joint across the
   fleet with `role: "gripper"`. Tag base joints on `panda_mobile` and
   `pusht_2d` with `role: "base"`. Other joints stay `"unknown"` (the
   default) — explicit re-annotation deferred per CLAUDE.md §6 ("don't
   refactor across all layers in one PR").
5. **Invariant test:** `tests/unit/test_rskill_action_dim_invariant.py`
   asserts that for every rskill manifest with `actuators_required` of
   `kind: joint_position`, `action_contract.dim == len(robot.joints)`
   for every embodiment_tag it claims (or the rskill carries a slot
   layout — see 0028b). The check would have caught the panda_mobile
   gap at fixture load.

This ADR does **not**:

- Fix the failing demo trace. The 12-D RoboCasa pi0.5 dispatch still
  E-stops after 0028a — that's 0028b's job. 0028a only makes the
  fleet uniform so 0028b's slot dispatch has a clean target.
- Change the rskill manifest schema beyond what's needed to declare
  the invariant the test enforces.
- Touch the runner, the safety kernel, or any HAL.

### 0028b — `action_contract.slots` schema + generic slot dispatch (LATER)

Sketch only — full ADR follows after 0028a merges.

```yaml
# rskills/pi05-robocasa365-human300-nf4/rskill.yaml
action_contract:
  dim: 12
  slots:
    - {range: [0, 5],  control_mode: "cartesian_delta", ee: "panda_hand",  frame: "panda_link0"}
    - {range: [6, 6],  control_mode: "gripper_position", ee: "panda_gripper"}
    - {range: [7, 7],  discard: true}    # paired gripper channel (training artifact)
    - {range: [8, 10], control_mode: "body_twist",     frame: "base_link"}
    - {range: [11, 11], discard: true}   # torso placeholder (dataset always -1)
```

Runner reads `slots`, emits one typed `ActionChunk` per non-discard slot
(all sharing the same `trace_id`), `ros_publishing_hal._action_to_chunk`
serialises any `ControlMode`, safety kernel grows per-mode envelopes.
Robots whose rskills emit pure joint_position write a one-slot block —
zero behavioural change for them.

### 0028c — `panda_mobile` HAL grows `CARTESIAN_DELTA` + `GRIPPER_POSITION` handlers (LATER)

Routes the typed chunks 0028b emits onto robosuite's OSC controller
(for arm cartesian deltas) and the gripper actuator (for gripper width).
The `BODY_TWIST` path is already wired (Nav2 uses it). Single-robot
PR.

After 0028a → 0028b → 0028c the trace at the top of this document runs
to completion: 12-D vector splits into one cartesian-delta arm chunk,
one gripper width chunk, one base-twist chunk; safety validates each;
HAL applies each on its native channel; the kettle gets picked up.

## Schema (this ADR — 0028a only)

```python
# python/core/src/openral_core/schemas.py

JointRole: TypeAlias = Literal[
    "arm",     # manipulator joint contributing to EE pose
    "base",    # planar base DoF (x/y/yaw for holonomic; theta for diff-drive)
    "gripper", # parallel-gripper width / single mimicked DoF
    "torso",   # trunk / waist on humanoids
    "leg",     # locomotion joint on humanoids / quadrupeds
    "head",
    "neck",
    "wheel",   # rotational wheel joint on diff-drive bases
    "unknown",
]


class JointSpec(BaseModel):
    ...
    role: JointRole = "unknown"   # NEW — structural tag, not derived from name


class EndEffectorSpec(BaseModel):
    ...
    actuated: bool = True   # NEW (formalised) — False for passive tools / inert flanges
```

`role` is `"unknown"` by default so existing manifests load without
edits. Annotating is a follow-up commit per CLAUDE.md §1.15 (drive-by
fixes get their own commit).

## Robot manifest updates (this ADR — 0028a only)

`robots/panda_mobile/robot.yaml` — add to `joints:`:

```yaml
- name: "panda_gripper"
  joint_type: "prismatic"
  parent_link: "panda_link7"
  child_link: "panda_finger_pair"
  position_limits: [0.0, 1.0]
  velocity_limit: 0.1
  effort_limit: 70.0
  has_torque_sensor: false
  actuator_kind: "servo"
  role: "gripper"
  # MJCF expands this 1-D abstraction into the franka two-finger mimic
  # at the simulator layer; the safety kernel + the rskill action
  # contract see a single width DoF.
  sim_joint_name: "gripper0_finger_joint1"
```

`robots/sawyer/robot.yaml` — add to `joints:`:

```yaml
- name: "right_gripper"
  joint_type: "prismatic"
  parent_link: "right_hand"
  child_link: "right_finger_pair"
  position_limits: [0.0, 0.041]
  velocity_limit: 0.1
  effort_limit: 35.0
  actuator_kind: "servo"
  role: "gripper"
```

Base joints on `panda_mobile` get `role: "base"`. Existing gripper
joints on `franka_panda` / `widowx` / `so100_follower` / `google_robot`
/ `aloha_bimanual` / `openarm` get `role: "gripper"`. Arm joints on
all manipulators get `role: "arm"`. Annotations only; no value changes.

## Invariant test

```python
# tests/unit/test_rskill_action_dim_invariant.py

def test_joint_position_rskills_action_dim_matches_robot_joints() -> None:
    """An rskill declaring only joint_position actuators must have
    action_contract.dim == len(robot.joints) for every embodiment_tag.

    This is the structural check that would have caught the
    panda_mobile / RoboCasa pi0.5 dim mismatch at fixture load
    (the trace at the top of ADR-0028).

    rskills carrying mixed control surfaces declare an
    action_contract.slots block (ADR-0028b) and are exempted.
    """
```

Reads every `rskills/*/rskill.yaml`, looks up each `embodiment_tag` in
`openral_sim.registry.ROBOTS`, validates the dim invariant. Fails loudly
with the manifest path + the expected/observed dims.

## Consequences

**Positive**

- `panda_mobile` and `sawyer` match the fleet convention. Any future
  joint-position-with-gripper VLA on either robot lands without
  E-stopping on the n_dof envelope check.
- `JointSpec.role` removes the `"gripper" in name.lower()` heuristic in
  `rskill_runner_node.py:1258`. Structural identification across naming
  conventions (RoboCasa `gripper0_…`, ALOHA `*_gripper`, openarm
  `*_gripper`, future fleet additions).
- The latent state-side bug in
  `rskills/pi05-robocasa365-human300-nf4/rskill.yaml:91` (binding to
  joints not declared in robot.yaml) is documented; the binding stays
  pointed at MJCF JointState names by design (state observations live
  on a different layer than control surfaces — the binding records
  sim/HW JointState names, not robot.yaml joint names).
- The invariant test makes the structural contract enforceable. Adding
  a new rskill whose action_contract doesn't match the embodiment's
  joint count fails at fixture load — not at runtime via E-stop.
- 0028b lands on a uniform fleet, no per-robot exceptions in the slot
  dispatcher.

**Negative / cost**

- One new schema field on the most-touched Pydantic model in the
  codebase (`JointSpec`). Default `"unknown"` keeps the change additive;
  no existing fixture rewrite required.
- Two robot.yaml files gain one joint each. The base-frame URDFs for
  `panda_mobile` and `sawyer` already model the gripper at the MJCF
  layer — the change is metadata, not kinematics.
- The `EndEffectorSpec.actuated` formalisation reveals two existing
  yamls (`panda_mobile`, `gr1`) that already set the field; their
  current behaviour was a silent no-op. After this ADR the field is
  load-bearing — future readers can rely on it (e.g. for safety bounds
  on un-actuated tool flanges that should never receive grip
  commands).

**Out of scope**

- The full slot-dispatch machinery. 0028b. Mentioned here only because
  ADR splits need to be telegraphed up-front — a reader of 0028a in
  isolation would not understand why we add `role` without using it
  yet.
- Dexterous-hand finger joints on `gr1`. Different problem (>5 DoF
  per hand, separate per-finger control). Separate ADR when the GR-1
  rskills land.
- Real-hardware HAL ports. 0028c only updates the sim HAL.
- Per-mode safety envelope declarations on robot.yaml. Required by
  0028b, deferred until then.

## Implementation sequence

Per CLAUDE.md §4.2 (smallest viable PR; each commit independently
reviewable):

1. **`docs(adr): ADR-0028`** — this file only.
2. **`feat(schemas): JointSpec.role + EndEffectorSpec.actuated`** —
   Pydantic fields, `JointRole` literal, docstrings. `docs/METHODS.md`
   updated in the same commit (per CLAUDE.md §1.13). Round-trip + Hypothesis
   tests on each model.
3. **`feat(robots): panda_mobile + sawyer gripper joints`** — add
   `panda_gripper` and `right_gripper`. Update the panda_mobile robot
   description comment block ("10-D…plus 1-D gripper" → "11-D"). No
   other yaml changes.
4. **`feat(robots): annotate role: tags across fleet`** — gripper /
   base / arm tags on existing joints. Pure metadata.
5. **`test(unit): action_contract.dim matches robot.joints invariant`** —
   new test asserting the structural contract. Fails today on
   panda_mobile + sawyer rskills; passes after step 3.
6. **`docs: repo-state-map + METHODS.md`** — reflect the schema field
   addition + ADR-0028a status. CLAUDE.md §4.3.

Each commit independently reviewable; full sequence merges as one PR.
