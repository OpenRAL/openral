# ADR-0028d: panda_mobile HAL JOINT_VELOCITY + COMPOSITE_MODE handlers (mobile base + mux flag)

- Status: **Proposed**
- Date: 2026-05-28
- Related: [ADR-0028](0028-rskill-action-contract-slots.md) (parent);
  [ADR-0028b](0028b-rskill-action-contract-slots-dispatch.md) (slot dispatch);
  [ADR-0028c](0028c-panda-mobile-hal-cartesian-gripper-handlers.md)
  (sibling — added CARTESIAN_DELTA + GRIPPER_POSITION); CLAUDE.md §3
  (HAL is layer 0).

## Context

ADR-0028c gave the panda_mobile HAL handlers for the arm
(`CARTESIAN_DELTA`) and gripper (`GRIPPER_POSITION`) slots that the
RoboCasa pi0.5 / rldx1 manifests emit. The remaining four dims of the
12-D `PandaOmron + HybridMobileBase` action vector — slots `[7, 10]`
plus the trailing composite-mode flag at slot 11 — were declared
`discard: true` in the rldx-rc365 / rldx-robocasa / pi05-human300
manifests with the explanation:

> base_motion is 4-D and cannot be expressed as a single 6-D
> `BODY_TWIST` slot (schema requires width=6 for
> `[vx,vy,vz,wx,wy,wz]`). Pending a non-standard-base ControlMode
> extension, indices 7-10 are declared discard.
> — `rskills/rldx1-ft-rc365-nf4/rskill.yaml:189-194`

Three things are wrong with that framing:

1. **It's not a 4-D base.** Runtime introspection of robosuite's
   `HybridMobileBase._action_split_indexes` for `PandaOmron` shows
   ```
   'right'          (0, 6)   arm OSC_POSE
   'right_gripper'  (6, 7)   gripper
   'base'           (7, 10)  3-D JOINT_VELOCITY (forward, side, yaw)
   'torso'          (10, 11) 1-D JOINT_POSITION (torso slider)
   ```
   The contiguous `[7, 10]` discard range was conflating the 3-D
   planar-base channel with the 1-D torso slider. They are two
   separate composite parts.

2. **The torso channel was never used during training.** The RC365
   `general_embodiment.action.base_motion` normalizer in
   `RLWRLD/RLDX-1-FT-RC365/statistics.json` reports the 4th dim with
   `mean=0, std=0, min=0, max=0`; the same shape holds in pi05's
   `action.std[10] = 0`. Live deploy_sim captures confirm
   `raw_policy_action[10] = +0.000` at every step. So slot 10 is
   padding, not torso control — discarding it leaves robosuite's
   torso controller integrating zero (the JOINT_POSITION controller
   defaults to delta semantics: input=0 ⇒ "hold current position"),
   which is exactly what the policy emitted during training. **No
   torso declaration is required.**

3. **No new ControlMode is needed.** The robosuite training-time
   controller for the *active* parts maps directly onto an existing
   `openral_core.ControlMode`:
   ```
   robosuite controller     ←→  openral ControlMode
   ────────────────────────────────────────────────
   OSC_POSE (delta)          ↔  CARTESIAN_DELTA   ✓ ADR-0028c
   GRIP                      ↔  GRIPPER_POSITION  ✓ ADR-0028c
   JOINT_VELOCITY (base)     ↔  JOINT_VELOCITY    (this ADR)
   JOINT_POSITION (torso)    ↔  (discard — zero in training)
   ```
   `JOINT_VELOCITY` is already in the enum (`schemas.py:84`); the
   HAL just lacks the routing rule mapping a 3-D chunk onto
   `_part_slot('base')`.

The cost of the misframing is concrete: the rldx-rc365 policy was
trained on `PickPlaceCounterToCabinet`, a task that requires
**mobile manipulation** — the cabinet and the counter are at
different physical locations, the robot must drive its base + raise
its torso to reach both. In the diagnostic run captured 2026-05-28
the policy emitted non-trivial base channels at multiple ticks (e.g.
step 200 `[7:9] = [+0.468, +0.132, -0.014]`) which `deploy_sim`
silently dropped. The arm reached down toward the counter (50 cm
descent in 600 steps — a real consequence of the obs-side fix in
commit `f52eb7c`) but never closed the geometry to the bread because
the base + torso never moved.

The training controller's slot 11 — a continuous-valued sign flag
that `HybridMobileBase.set_goal` reads as `all_action[-1]` and uses
to multiplex between "manipulate" (arm OSC tracks the achieved pose;
base velocity gets directly applied) and "navigate" (arm freezes,
base drives) — is **not a per-joint actuator command**. The HAL
already hardcodes `out[-1] = -1.0` ("manipulate") at
`sim_attached.py:623-624`. That is a deliberate limitation of the
slot-dispatch path: per-slot ControlMode chunks can't carry a
sim-internal multiplexer flag without leaking robosuite specifics
into the wire ABI. We accept this limitation for now (this ADR);
the rare case where the policy wants to disable arm tracking to
drive the base is left on the table and revisited if the closed-loop
benchmark numbers demand it.

## Decision

Add two handlers to the `panda_mobile` HAL's `_pack_with_composite_split`
helper so the `'base'` composite part AND the trailing multiplexer flag
can be written from slot-dispatched `ActionChunk`s:

1. **`JOINT_VELOCITY` (3-D base)** — looks up `_part_slot('base')` at
   runtime via the same `_action_split_indexes` introspection
   ADR-0028c already uses for `'right'` / `'right_gripper'`. Writes
   the chunk's `joint_velocities[0]` directly into those slots.
   The handler accepts the manifest-declared `joint_names`
   (`["base_x", "base_y", "base_yaw"]`) and validates that they
   resolve to a single composite part (so a future skill that
   names only `["base_yaw"]` would route just that one channel
   into the correct sub-slot via the same `_action_split_indexes`).

2. **Update the three RoboCasa rSkill manifests'
   `action_contract.slots`** to split the `[7, 10]` discard into
   one routed slot + one (still-discarded) torso slot:
   ```yaml
   - {range: [7,  9],  control_mode: "joint_velocity",
      joint_names: ["base_x", "base_y", "base_yaw"]}
   - {range: [10, 10], discard: true}   # torso — std=0 in training
   - {range: [11, 11], discard: true}   # composite mode flag
   ```
   Applies to: `rskills/rldx1-ft-rc365-nf4/rskill.yaml`. The pi05 manifest
   (`rskills/pi05-robocasa365-human300-nf4/rskill.yaml`) **also
   needs this exact layout** — its current
   `[7] discard, [8,10] body_twist, [11] discard` is misaligned by
   one dim against the actual `_action_split_indexes` ordering, so
   today pi05 silently routes `base_y_vel` into BODY_TWIST's `vx`
   slot, `base_yaw_vel` into `vy`, and the always-zero torso into
   `wz`. The 1-dim shift plus the velocity-vs-delta semantic
   mismatch explains the "base wanders sideways and never rotates"
   symptom.

3. **`supported_control_modes`** in `robots/panda_mobile/robot.yaml`
   adds `"joint_velocity"` so the palette filter doesn't reject
   manifests that declare a `JOINT_VELOCITY` slot. The safety
   supervisor already enforces per-joint `velocity_limit`
   declarations via ADR-0028b §5 — no new global envelope keys.

4. **`COMPOSITE_MODE` (slot 11)** — the trailing dim of the rldx /
   pi05 12-D action is robosuite's `HybridMobileBase.set_goal`
   multiplexer flag (`action[-1] > 0` ⇒ arm OSC tracks the
   commanded delta; `≤ 0` ⇒ arm OSC tracks the achieved pose and is
   effectively frozen, while base velocity passes through). The
   policy emits this dynamically; hardcoding it in the HAL (as
   the pre-ADR-0028d code did with `out[-1] = -1.0`) freezes the
   arm and gives the visual symptom "only the base moves." We
   promote this to a first-class `COMPOSITE_MODE` `ControlMode`
   (1-D, value ∈ [-1, +1], sim-only) carried end-to-end through
   the typed-Action ABI:
   - `openral_core.ControlMode.COMPOSITE_MODE`, wire uint8 = 12,
     mirrored in `cpp/openral_safety_kernel/include/.../validator.hpp`.
   - `Action.composite_mode: list[float] | None` field +
     `ActionSlot` validator branch (width=1, no ee/frame/joints).
   - C++ kernel routes `kCompositeMode` through the same
     per-mode passthrough as cartesian/twist/gripper (structural
     validation only — no per-joint or workspace bounds apply).
   - `SimAttachedHAL._pack_with_composite_split` writes the value
     to `out[-1]` and early-returns (skipping the legacy
     `out[-1] = -1.0` override below).
   - Real-HW HAL adapters (Franka FCI + Omron Nav2 + Franka
     gripper) accept the mode but treat it as a no-op: their
     independent arm + base controllers don't need the mux.
   - rldx-rc365 / rldx-robocasa / pi05-human300 manifests'
     slot 11 changes from `discard: true` to
     `control_mode: composite_mode`.

5. **Torso joint declaration is deferred.** Both rldx-rc365 and
   pi05 emit slot 10 = 0 by training-distribution construction;
   the existing `discard` semantics (zero in the env_action vector
   ⇒ JOINT_POSITION delta=0 ⇒ "hold current torso position")
   replays training fidelity for free. A future skill that
   actually exercises the torso would lift this — add a `torso_z`
   joint to `robot.yaml` with
   `sim_joint_name: "mobilebase0_joint_torso_height"` and a
   targeted `JOINT_POSITION` HAL handler (mirror of the
   `JOINT_VELOCITY` pattern). Out of scope here.

### Why "leverage robocasa's controller" doesn't mean call it directly

Tempting alternative: in `SimAttachedHAL`, pass the raw 12-D action
vector straight through to `env.step()` — let robosuite's
`HybridMobileBase` do its own dispatch. This is essentially what
`sim_run` does today (`_assemble_rc365_chunk` → `env.step(action)`).

Rejected because:

- **HAL is layer 0** (CLAUDE.md §3) — the typed-Action boundary is
  the contract that survives the move to real hardware. A
  "passthrough" mode hides the action semantics behind a sim-specific
  black box and re-introduces the dim-12-vs-11 skew the ADR-0028
  family was written to eliminate.
- **The safety supervisor needs typed chunks** to envelope-check
  per-mode (ADR-0028b §5). A passthrough bypasses that.
- **The mapping is trivial.** Each composite part is already a
  one-line lookup against `_part_slot(name)` and writes a 1-D or 3-D
  slice. ADR-0028c established this pattern for `'right'` and
  `'right_gripper'`; extending to `'base'` / `'torso'` is the same
  shape.

We *do* leverage robosuite's controller — but at the level of "use
its `_action_split_indexes` to know where to write," not "hand it
raw bytes." The actual MJCF actuators are still driven by
robosuite's `JOINT_VELOCITY` / `JOINT_POSITION` part controllers
once we've written into the right slots.

## Consequences

**Positive**

- `rldx1-ft-rc365-nf4` can now actuate its full 12-D output in
  `deploy_sim`. Closed-loop reach-to-grasp tasks on
  `PickPlaceCounterToCabinet` (and other mobile-manipulation
  RoboCasa scenes) become viable.
- pi05's "base wanders" symptom — observed in the same deploy_sim
  failure that motivated `f52eb7c` — should resolve, because the
  semantic mismatch (BODY_TWIST = velocity expecting m/s vs the
  policy emitting normalized [-1, +1]) is replaced with the
  correct JOINT_VELOCITY routing whose normalization robosuite's
  controller already handles internally.
- The slot-dispatch contract stays clean: every manifest dim is
  either explicitly routed or explicitly discarded. No
  passthrough escape hatch.
- The torso joint becomes a first-class entry in
  `RobotDescription.joints`, addressable by name from any future
  state assembler that needs it (e.g. a sit-to-stand skill on a
  different scene).

**Negative / accepted limitations**

- **Composite mode flag (slot 11) remains hardcoded to -1
  ("manipulate")** at the HAL — the policy can't switch modes.
  Acceptable because (a) the trained head emits negative slot-11
  values during arm-active phases (which is the only phase where
  the rest of the chunk's arm channels are meaningful), and (b)
  the alternative is a sim-only ControlMode that pollutes the
  wire ABI.
- **Real-hardware port for the OmronMobileBase + torso slider is
  out of scope** for this ADR. The HAL handler is gated on the
  presence of a robosuite composite controller (i.e.
  `SimAttachedHAL`); a future `PandaMobileHAL` (real-HW) would
  need its own JOINT_VELOCITY / JOINT_POSITION implementations,
  driven from the appropriate ros2_control hardware interface.
- pi05 may still have a residual closed-loop gap if its mode-flag
  expectations differ from the hardcoded -1; we benchmark and
  revisit if so.

### Implementation note — kernel n_dof gate

The C++ safety kernel (`cpp/openral_safety_kernel/src/validator.cpp:51-61`)
enforces `chunk.n_dof == envelope.n_dof` for any JOINT_* mode (the
per-joint validator at line 111-128 indexes `envelope.joint_velocity_max[j]`
for `j ∈ [0, envelope.n_dof)`). A slot-dispatched 3-D JOINT_VELOCITY
chunk for the base alone trips `kNdofMismatch` and e-stops the robot
before the HAL ever sees it.

Two responses considered:
- (a) Extend the kernel to treat JOINT_VELOCITY as a per-mode width
  when accompanied by `joint_names` (mirroring the existing per-mode
  short-circuit for CARTESIAN / GRIPPER / BODY_TWIST). Requires IDL
  change to carry `joint_names` on the wire.
- (b) Pad the chunk to full-dof at dispatch time, with zeros at
  non-target joints. Preserves the kernel's per-joint validation and
  doesn't touch the IDL.

We picked (b): the dispatcher (`_dispatch_slots` in
`packages/openral_rskill_ros/openral_rskill_ros/rskill_runner_node.py`)
uses the slot's `joint_names` + the robot's joint ordering to
construct a length-`n_dof_total` payload. The HAL's
`_pack_with_composite_split` then extracts the base values via
`description.base_joints` and writes them to `_part_slot('base')`.

Side effect on the kernel's per-joint velocity check: the policy
emits NORMALIZED values in [-1, +1] (robosuite's HybridMobileBase
controller does the physical scaling internally), so the base joints'
`velocity_limit` in `robot.yaml` must be ≥ 1.0 for the kernel to
accept the full normalized range. We bump the three base joints
(`base_x`, `base_y`, `base_yaw`) to `velocity_limit: 1.0` and add a
comment explaining the normalized-vs-physical distinction. Real-HW
deployment would need a scaling step at the publisher (or restore
physical limits and re-scale at the HAL boundary).

## Implementation sequence

1. `robots/panda_mobile/robot.yaml`: add `"joint_velocity"` to
   `capabilities.supported_control_modes`; bump base joints'
   `velocity_limit` to 1.0 (normalized contract).
2. `packages/openral_rskill_ros/openral_rskill_ros/rskill_runner_node.py`:
   add `_pad_joint_payload` helper + extend `_dispatch_slots` to
   accept `description` and pad JOINT_* slices.
3. `python/hal/src/openral_hal/sim_attached.py:_pack_with_composite_split`:
   add a `JOINT_VELOCITY` branch that extracts base values from the
   full-dof payload via `description.base_joints` and writes to
   `_part_slot('base')`.
4. Update the three RoboCasa rSkill manifests' `action_contract`
   to replace the `discard: true` for slots 7-10 with the
   3-routed-to-`joint_velocity` + 1-discarded-torso pattern.
4. Tests:
   - Unit (`python/hal/tests/test_sim_attached_pack.py` — extend or
     create): assert a 3-D `JOINT_VELOCITY` chunk lands in slots
     `[7, 10)` of the packed env vector; assert misaligned widths
     raise `ROSConfigError`.
   - Integration: confirm the env-side action vector carries
     non-zero base velocity channels when the policy emits them
     (e.g. by logging the pre-`env.step` action in deploy_sim).
   - End-to-end (`tests/sim/`): `obj_eef_dist_m` must drop below
     0.10 m by step 220 (yaml `max_steps`) on at least 1 of 5
     seeds — RC365's documented success rate is 31.5%.
5. Docs: update the manifest comment blocks to replace the
   "4-D non-standard-base ControlMode pending" lament with a
   pointer to this ADR.

## Rejected alternatives

- **Add a `MOBILE_BASE_DELTA` ControlMode (4-D).** Considered first
  because the manifest comments anticipated it. Rejected once the
  live `_action_split_indexes` plus the normalizer stats showed the
  trailing dim is unused padding — the active surface is just 3-D
  base velocity, already covered.
- **Declare `torso_z` and route slot 10 through JOINT_POSITION.**
  Considered for "schema completeness." Rejected because (a) the
  trained policies never exercise it (std=0) so we'd be wiring a
  dead channel, and (b) `discard ⇒ out[10]=0 ⇒ delta=0 ⇒ hold` is
  semantically identical to training behaviour. Trivially added
  later if a torso-active skill ships.
- **Passthrough mode.** Violates the typed-Action boundary,
  sidesteps the safety supervisor.
- **Treat slot 7-9 as `BODY_TWIST` (3-D planar).** Rejected because
  BODY_TWIST is semantically m/s velocity that the HAL integrates
  as `velocity * dt` (sim_attached.py:290-295), whereas robosuite's
  `'base'` part already exposes a `JOINT_VELOCITY` controller that
  handles the per-joint normalization internally. Routing through
  BODY_TWIST introduces a ~20× scaling pitfall (the policy emits
  normalized [-1, +1], BODY_TWIST expects m/s) — exactly the
  symptom pi05 exhibits today with its misaligned manifest.
