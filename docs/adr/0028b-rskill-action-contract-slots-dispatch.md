# ADR-0028b: rSkill action-contract slot dispatch (`action_contract.slots`)

- Status: **Proposed**
- Date: 2026-05-27
- Related: [ADR-0028](0028-rskill-action-contract-slots.md) (the
  parent ADR — context, problem statement, sub-ADR split, the
  invariant test 0028a installed which xfails the manifests this
  ADR exists to fix); [ADR-0027](0027-rskill-state-contract-bindings.md)
  (symmetric state-side layout machinery); [ADR-0018](0018-ros2-reasoner-supervisor.md)
  §F1 (typed `ActionChunk` wire format already carries `control_mode`
  per-message; this ADR makes the field actually load-bearing across
  the runner → safety → HAL chain); [ADR-0024](0024-ros-wrapped-rskills.md)
  (skill runner is the dispatch layer); CLAUDE.md §1.4 (explicit beats
  implicit), §3 (Skill ↔ Safety ↔ HAL layer boundaries).

## Context

ADR-0028a brought the robot manifest fleet into uniform shape:
every parallel-gripper embodiment now declares the gripper as a joint
with structural role tagging, the invariant test guards
`action_contract.dim <= len(robot.joints)`, and four mis-declared
manifests are pinned as `xfail` pointing here.

The failing trace at the top of [ADR-0028](0028-rskill-action-contract-slots.md)
is still red. The pi05-robocasa365-human300 checkpoint emits

```
[arm_osc(6) + gripper(2) + base(3) + torso(1)]   = 12-D
```

— a *mixed-semantics* action vector, not 12 joint positions. The
fleet uniformity 0028a delivered is the foundation, but the runner
still has no way to read the layout from the manifest and dispatch
the slices to the right control surfaces. Today
`packages/openral_rskill_ros/.../rskill_runner_node.py:1466-1543` wraps
the raw policy vector as one `Action(control_mode=JOINT_POSITION, …)`
and ships it; the safety kernel correctly E-stops on the
`n_dof` mismatch.

ADR-0028a deliberately scoped slot dispatch out. This ADR brings it in.

## Decision

Extend `openral_core.ActionContract` with an optional `slots` block
that declares — per checkpoint, in the manifest, by the rSkill
author — how the policy's flat action vector splits into typed
sub-actions. The runner reads the block and emits one
`openral_msgs/ActionChunk` per non-discard slot, each carrying its
own `control_mode`, joined to the same parent step by a shared
`trace_id`. The wire format does not change. The HAL routes by
`control_mode` (the panda_mobile HAL already does this for
`JOINT_POSITION` + `BODY_TWIST`; 0028c adds `CARTESIAN_DELTA` +
`GRIPPER_POSITION`). The safety kernel grows per-mode envelopes so
each typed chunk is validated against bounds appropriate to its
control surface.

### Schema (`python/core/src/openral_core/schemas.py`)

```python
class ActionSlot(BaseModel):
    """Declarative description of one contiguous slice of an rSkill's
    action vector (ADR-0028b).

    The skill_runner reads ``ActionContract.slots`` and emits one
    typed ``ActionChunk`` per non-discard slot per step. All chunks
    share the parent step's ``trace_id`` so the safety supervisor
    and downstream telemetry can join them post-hoc.
    """

    model_config = ConfigDict(extra="forbid")

    range: tuple[int, int]
    """Inclusive ``[start, end]`` slice into the flat policy action
    vector. ``range[0]`` must be ≤ ``range[1]``; both bounds must
    fall within ``[0, ActionContract.dim)``."""

    control_mode: ControlMode | None = None
    """The :class:`ControlMode` the slice's bytes are routed to. The
    HAL whitelist on the target robot must include this mode (the
    palette filter rejects the rSkill at install time otherwise).
    ``None`` only when :attr:`discard` is True."""

    discard: bool = False
    """When True, the slice is dropped silently (used for dataset
    artefacts like RoboCasa365's `torso` placeholder dim or the
    paired gripper channel). The slot still occupies its range so
    coverage validation works; no ActionChunk is emitted for it.
    ``control_mode`` must be None when discard is True."""

    ee: str | None = None
    """End-effector name from the robot's :attr:`RobotDescription.end_effectors`
    or :attr:`RobotDescription.joints`. Required for
    ``cartesian_*`` modes (the EE pose is computed in the named EE's
    frame) and for ``gripper_*`` modes (names the actuator). Forbidden
    for ``body_twist`` and ``joint_position``."""

    frame: str | None = None
    """tf2 frame name. Required for cartesian + body_twist modes
    (the slice's bytes are expressed in this frame). Forbidden for
    joint_position + gripper."""

    joint_names: list[str] = Field(default_factory=list)
    """Robot joint names the slice targets when ``control_mode is
    ControlMode.JOINT_POSITION``. Length must match
    ``range[1] - range[0] + 1``. Forbidden for non-joint modes.

    When omitted on a single all-joints slot covering the whole
    vector, the runner defaults to ``robot.joints`` in declaration
    order — the back-compat path for legacy joint-position rSkills."""


class ActionContract(BaseModel):
    ...
    dim: int = Field(gt=0)
    representation: ActionRepresentation | None = None
    slots: list[ActionSlot] | None = None  # NEW
```

**Cross-field validator** on `ActionContract`:

- When `slots is None`, the existing behaviour stands (one implicit
  joint_position slot covering the whole vector — backward
  compatible).
- When `slots` is set:
  - Every index in `[0, dim)` is covered by exactly one slot
    (`ROSConfigError` on gap or overlap).
  - Each slot's `control_mode` ↔ `ee` / `frame` / `joint_names`
    requirements are honoured (see per-mode constraints above).
  - `discard` slots have `control_mode is None`; non-discard slots
    have `control_mode` set.

**Per-mode field requirements:**

| `control_mode` | `ee` required | `frame` required | `joint_names` required |
|---|---|---|---|
| `JOINT_POSITION` / `JOINT_VELOCITY` / `JOINT_TORQUE` | no | no | yes (length = slot width) |
| `CARTESIAN_POSE` / `CARTESIAN_DELTA` / `CARTESIAN_TWIST` | yes | yes | no |
| `BODY_TWIST` | no | yes | no |
| `GRIPPER_POSITION` / `GRIPPER_BINARY` | yes (gripper joint name) | no | no |
| `discard: true` | no | no | no |

### Skill runner (`packages/openral_rskill_ros/.../rskill_runner_node.py:_step_impl`)

When `manifest.action_contract.slots` is set, replace the single
`Action(JOINT_POSITION, joint_targets=[...])` build with a slot loop:

```python
def _slot_to_action(slot: ActionSlot, vec: np.ndarray) -> Action | None:
    if slot.discard:
        return None
    lo, hi = slot.range
    slice_ = vec[lo : hi + 1]
    mode = slot.control_mode  # validated non-None when discard is False
    if mode is ControlMode.JOINT_POSITION:
        return Action(control_mode=mode, horizon=1,
                      joint_targets=[list(map(float, slice_))],
                      joint_names=slot.joint_names)
    if mode is ControlMode.CARTESIAN_DELTA:
        return Action(control_mode=mode, horizon=1,
                      cartesian_deltas=[list(map(float, slice_))],
                      ee_name=slot.ee, frame_id=slot.frame)
    if mode is ControlMode.BODY_TWIST:
        return Action(control_mode=mode, horizon=1,
                      body_twists=[list(map(float, slice_))],
                      frame_id=slot.frame)
    if mode is ControlMode.GRIPPER_POSITION:
        return Action(control_mode=mode, horizon=1,
                      gripper_targets=[float(slice_[0])],
                      ee_name=slot.ee)
    raise ROSConfigError(f"slot dispatcher: unsupported control_mode {mode!r}")

# In _step_impl, after `policy_action = np.asarray(action_array, …)`:
if manifest.action_contract is not None and manifest.action_contract.slots is not None:
    return [a for s in manifest.action_contract.slots
              if (a := _slot_to_action(s, policy_action)) is not None]
# else: legacy joint_position single-Action path stays verbatim
```

`_step_impl`'s return type widens to `Action | list[Action]`. The
wrapping `ROSPublishingHAL.act()` is the layer that splits a list
into per-mode `ActionChunk`s (see next section). Per-step latency is
dominated by the policy forward — slot dispatch is microseconds of
slicing.

### Action serialisation (`python/runner/src/openral_runner/ros_publishing_hal.py`)

Two changes:

1. **`_action_to_chunk`** (currently `:240-257`) lifts the per-mode
   rejection. Each `ControlMode` gets its own field mapping:
   - `JOINT_*` → `flat = row_major(joint_*targets)`, `n_dof = arity`.
   - `CARTESIAN_*` → `flat = [x,y,z,rx,ry,rz]` (or 7 for pose-quat),
     `n_dof = 6` (or 7), `ee_name` + `frame_id` populated.
   - `BODY_TWIST` → `flat = [vx,vy,vz,wx,wy,wz]`, `n_dof = 6`,
     `frame_id` populated.
   - `GRIPPER_*` → `flat = [width]` or `[binary]`, `n_dof = 1`,
     `ee_name` populated.

2. **`act()`** grows a list overload: when the runner returns
   `list[Action]`, build one `ActionChunk` per action, publish them
   onto `/openral/action_chunk` with a shared `trace_id` (from the
   active OTel span — already the source).

### Safety supervisor (`packages/openral_safety/openral_safety/supervisor_node.py:_envelope_violation`)

Today's check (lines `301-356`) is joint-position-specific: it
validates `n_dof` against the launch parameter and each row of
`flat` against `min_joint` / `max_joint`. Extend by dispatching on
the incoming chunk's `control_mode`:

| `control_mode` | check |
|---|---|
| `JOINT_*` | existing path (n_dof + per-joint bounds). |
| `CARTESIAN_DELTA` | `|delta_xyz| <= safety.max_cartesian_step_m`, `|delta_rotvec| <= safety.max_cartesian_step_rad`. |
| `CARTESIAN_TWIST` | `|linear| <= safety.max_ee_speed_m_s` (already on `SafetyEnvelope`!), `|angular| <= safety.max_ee_angular_speed_rad_s`. |
| `BODY_TWIST` | `|linear_xy| <= robot.base_velocity_limit`, `|angular_z| <= robot.base_angular_velocity_limit`. |
| `GRIPPER_*` | `0 <= width <= robot.gripper.position_limits` (read straight from the JointSpec the slot's `ee` resolves to). |

Per-mode bounds come from the existing `SafetyEnvelope` fields where
they exist; the rest land as new fields on `SafetyEnvelope` (additive,
defaults preserve current behaviour). The supervisor heartbeat emits
per-mode pass/drop counters for observability.

### Per-robot envelope additions (`robots/*/robot.yaml::safety:`)

Additive. Defaults left as `None` mean "no per-mode bound declared,
skip the check" — back-compat. Robots that exercise the new modes
declare bounds; everyone else stays unchanged.

```yaml
safety:
  ...                                 # existing fields stay verbatim
  max_cartesian_step_m: 0.05          # CARTESIAN_DELTA per-step bound
  max_cartesian_step_rad: 0.2         # CARTESIAN_DELTA orientation bound
  max_ee_angular_speed_rad_s: 1.0     # CARTESIAN_TWIST angular bound
  max_base_linear_speed_m_s: 1.0      # BODY_TWIST linear bound
  max_base_angular_speed_rad_s: 1.5   # BODY_TWIST angular bound
```

panda_mobile gets all five for the RoboCasa rSkills; other robots
add them as their first slot-using rSkill arrives.

### Manifest update — the four xfailed rSkills

`rskills/pi05-robocasa365-human300-nf4/rskill.yaml`,
`rskills/rldx1-ft-rc365-nf4/rskill.yaml`:

```yaml
action_contract:
  dim: 12
  slots:
    - {range: [0, 5],  control_mode: "cartesian_delta", ee: "panda_hand",  frame: "panda_link0"}
    - {range: [6, 6],  control_mode: "gripper_position", ee: "panda_gripper"}
    - {range: [7, 7],  discard: true}
    - {range: [8, 10], control_mode: "body_twist",     frame: "base_link"}
    - {range: [11, 11], discard: true}
```

After this lands, the invariant-test `_PENDING_SLOT_LAYOUT`
entries for those three pairs go away — the test recognises that a
manifest with `slots` is internally consistent by construction (the
ActionContract validator already enforces coverage), so it skips the
raw `dim <= len(joints)` check.

The GR-1 case (`rldx1-ft-gr1-nf4` × `gr1`) remains xfailed for a
separate ADR — `robots/gr1/robot.yaml` lacks the per-finger DoFs the
29-D action commands, which is a dexterous-hand declaration gap, not
a slot-dispatch gap.

## Consequences

**Positive**

- The trace at the top of ADR-0028 runs end-to-end on `panda_mobile`
  with the RoboCasa pi0.5 / rldx1 checkpoints, once 0028c lands the
  matching HAL handlers. The four xfails in
  `tests/unit/test_rskill_action_dim_invariant.py` collapse to one
  (the orthogonal GR-1 case).
- The runner becomes the slot interpreter; no per-checkpoint Python
  registry, no `openral_action_adapter` package. Each rSkill's
  action layout is fully described in its own YAML.
- Symmetric to ADR-0027's state side: contract lives in the manifest,
  the runner runs a generic engine, no manifest-author-vs-Python-author
  drift.
- The safety kernel learns per-mode reasoning. A future cartesian-only
  policy on a UR5e can declare `safety.max_cartesian_step_m` and the
  supervisor enforces it before any motion command reaches the
  controller.
- Robots that command via composite controllers (robosuite OSC,
  Franka FCI, UR's `script_server`) finally have an honest declaration
  surface for what they actually consume — instead of the runner
  pretending OSC deltas are joint targets.

**Negative / cost**

- `ActionContract` schema grows. Existing manifests without `slots`
  keep working (back-compat by default). New manifests pay an
  authoring cost — 5 lines for a simple two-slot arm+gripper layout,
  ~10 for a panda_mobile mixed surface.
- The safety supervisor's complexity grows. Per-mode dispatch is
  modular but adds ~150 LoC and a corresponding test surface. Per
  CLAUDE.md §3, this is a layer-6 change that requires safety-WG
  review.
- A list-of-actions return widens the `Skill.step()` Protocol. Today
  `step()` returns one `Action`; after this ADR it returns
  `Action | list[Action]`. Every concrete skill that returns a list
  is one of: a slot-using VLA rSkill (runner-internal), a BT-style
  composite action (future). The Protocol bump is small; existing
  consumers (skills returning a single Action) keep working.

**Out of scope**

- The HAL handlers themselves. ADR-0028c (`panda_mobile` HAL grows
  `CARTESIAN_DELTA` + `GRIPPER_POSITION` routing — robosuite's OSC
  controller + gripper actuator). The HAL whitelist in
  `lifecycle_node.py:752` stays at `{JOINT_POSITION, BODY_TWIST}` in
  this ADR's PR; 0028c extends it.
- IK / FK transforms inside the runner. The runner is a pure
  byte-router; if a checkpoint emits cartesian deltas the HAL is
  responsible for inverse kinematics (or for talking to a controller
  that does — robosuite OSC, Franka cartesian impedance).
- Dexterous-hand finger joints on `gr1`. The 29-D rldx1-ft-gr1
  checkpoint needs the GR-1 robot.yaml to declare its finger DoFs;
  separate ADR.
- Real-hardware HAL ports. 0028c only updates the sim HAL.

## Implementation sequence

Per CLAUDE.md §4.2 (smallest viable PR; each commit independently
reviewable):

1. **`docs(adr): ADR-0028b`** — this file only.
2. **`feat(schemas): ActionSlot + ActionContract.slots`** — Pydantic
   model + cross-field validator. Hypothesis fuzz on `ActionContract`
   (coverage, gap, overlap, per-mode field requirements). `docs/METHODS.md`
   updated.
3. **`feat(skill_runner): slot dispatch when action_contract.slots is set`** —
   `_step_impl` widens its return type to `Action | list[Action]`;
   the slot loop builds one typed Action per non-discard slot. Unit
   test against synthetic 12-D RoboCasa-shaped vectors.
4. **`feat(ros_publishing_hal): serialise cartesian/twist/gripper Actions`** —
   `_action_to_chunk` per-mode dispatch; `act()` accepts
   `list[Action]` and publishes one chunk per action with the shared
   trace_id.
5. **`feat(safety): per-mode envelope dispatch`** — supervisor reads
   chunk `control_mode` and dispatches to the matching check. New
   `SafetyEnvelope` fields (additive, defaulted). Per-mode pass/drop
   counters in the diagnostics heartbeat.
6. **`feat(robots): per-mode safety bounds on panda_mobile`** — add
   the five new fields to `robots/panda_mobile/robot.yaml::safety`.
7. **`feat(rskills): action_contract.slots on RoboCasa pi0.5 / rldx1 manifests`** —
   the five-slot block on the three rskills; remove their
   `_PENDING_SLOT_LAYOUT` entries from the invariant test (which now
   exempts slot-bearing manifests by design).
8. **`test(integration): panda_mobile + pi05-robocasa365-human300-nf4 dispatches without E-stop`** —
   launches the full stack (skill_runner + safety_kernel + HAL +
   reasoner), drives one step, asserts (a) 3 typed chunks emitted
   (arm cartesian, gripper, base twist), (b) all 3 pass safety,
   (c) HAL routes each one. This is the test the original trace
   would have passed.

Step 8 depends on ADR-0028c's HAL handlers being in place; ADR-0028c
ships in the same PR as ADR-0028b step 7 + 8, or as the
immediately-following PR. Either order works because the new HAL
handlers are additive (a panda_mobile run without the RoboCasa
rSkills keeps working today's joint_position + body_twist paths).
