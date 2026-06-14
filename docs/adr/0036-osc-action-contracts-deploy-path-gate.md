# ADR-0036: Cartesian/OSC action contracts + deploy-path-aware palette gate

- Status: **Accepted**
- Date: 2026-06-03
- Related: extends [ADR-0028b](0028b-rskill-action-contract-slots-dispatch.md) (action-contract slot
  dispatch); [ADR-0028c](0028c-panda-mobile-hal-cartesian-gripper-handlers.md) (HAL cartesian/gripper
  handlers); [ADR-0027](0027-rskill-state-contract-bindings.md) (state-contract compatibility);
  [ADR-0029](0029-unified-hal-lifecycle-node.md); [ADR-0034](0034-deploy-sim-scene-attach-for-arms.md)
  (deploy-sim scene-attach — prerequisite to reach the action path).
- Safety: does **not** modify `packages/openral_safety/` or `cpp/openral_safety_kernel/`. The new
  palette gate is the real-hardware exclusion boundary, so it gets a hazard-log entry + a safety-aware
  reviewer (CLAUDE.md §3 spirit) before merge, even though no safety-package code changes.

## Context

A VLA checkpoint's action representation (joint-space vs cartesian/OSC) is a per-checkpoint property
carried by `RSkillManifest.action_contract` (ADR-0007: it lives on the rSkill, not the
`RobotDescription` — the same Franka emits 7-D delta-EEF on LIBERO and 8-D joint pos on a hardware
deploy). LIBERO / SIMPLER / MetaWorld / PushT checkpoints emit a cartesian end-effector action
(LIBERO = 6-D OSC pose delta + 1-D gripper = 7).

When such a skill declares only `action_contract.dim` (no `slots`, no `representation`),
`rskill_runner_node._step_impl` falls back to the legacy path and labels the **whole vector
`JOINT_POSITION`**. The C++ safety kernel validates it against the joint-space envelope and rejects it
(`n_dof 7 ≠ 8` for franka) → E-stop. Observed live on `openral deploy sim` of `pi05-libero-nf4`: the skill
was picked and stepped once with camera frames, then E-stopped on
`envelope_violation field=n_dof value=7 limit=8`.

The robocasa skills (dim=12) already declare `slots` and work end-to-end, proving the slot-dispatch
mechanism (ADR-0028b), the kernel's cartesian routing, the supervisor's per-mode bounds, and the
manifest-derived HAL packing. Two gaps remain: (1) cartesian skills that under-declare their contract
are silently treated as joint actions; (2) the reasoner palette gates only `state_contract.dim` vs
joint count — it never checks whether the *action* is executable on the target robot / deploy path, so
an unexecutable skill is offered to the LLM and fails at runtime.

## Decision

### 1. `ActionRepresentation` → canonical slots in the skill_runner

Define one mapping from `action_contract.representation` to a `ControlMode` + canonical
`ActionSlot` layout (`joint_positions` → JOINT_POSITION whole-vector; `delta_ee_6d_plus_gripper` →
`[cartesian_delta 0–5, gripper_position last]`; `delta_ee_6d` → `[cartesian_delta 0–5]`;
`cartesian_pose` → cartesian-pose slot). EE/frame for cartesian/gripper slots derive from the robot's
primary `end_effectors` entry. Precedence: explicit `action_contract.slots` win verbatim (robocasa
unchanged); else `representation` expands to canonical slots; else the legacy whole-vector
`JOINT_POSITION` path is preserved (joint-space skills unaffected). Non-canonical layouts (MetaWorld
3-D EE delta + gripper; PushT 2-D) use a new representation enum entry or explicit `slots`.

### 2. Deploy-path-aware palette gate (drop at seed)

`reasoner_node` gains an action-compatibility filter after the state-contract filter: map each
candidate's `action_contract` → required `ControlMode`s and drop (with a boot warning, mirroring the
state-dim drop) any whose modes aren't executable for the deployment.

- `hal_mode == "real"` → require modes ∈ `description.capabilities.supported_control_modes`.
- `hal_mode == "sim"` → admit modes the attached scene env's controller executes; a robosuite OSC
  arm scene admits `cartesian_delta` + `gripper_*` in addition to joint modes.

The robot manifest's `supported_control_modes` stays real-hardware truth. The reasoner node gains a
`hal_mode` ROS parameter, set by the launch (`deploy sim` → `sim`, `deploy run` → `real`).

### 3. Safety posture (no safety-package change)

The Python supervisor (`openral_safety/supervisor_node.py`, which owns the per-step
`max_cartesian_step_m/rad` enforcement) is **not** spawned in `sim_e2e.launch.py` — only the C++
`safety_kernel` is. So robocasa cartesian skills already run in deploy-sim bounded only by the kernel's
structural / NaN / self-collision checks + the robosuite OSC controller + MuJoCo `ctrlrange`. Enabling
LIBERO/SIMPLER cartesian skills in deploy-sim is therefore consistent with the already-accepted
robocasa path — a digital twin, no real motor to protect — not a new unbounded real-hardware path.

- **deploy-sim:** no `safety_kernel` / supervisor / envelope change. Cartesian is structurally
  validated by the C++ kernel (unchanged) and physically bounded by OSC + MuJoCo clamps.
- **Real-hardware boundary:** the deploy-path-aware palette gate (§2) **drops** cartesian skills on
  real joint-only robots, so no cartesian action ever reaches a real motor — the at-least-as-conservative
  guarantee for `deploy run`.
- This work touches neither `packages/openral_safety/` nor `cpp/openral_safety_kernel/`. It still gets
  a hazard-log entry referencing this ADR and a safety-aware reviewer on the gate.
- **Deferred:** real-hardware cartesian execution — an OSC→joint IK shim + the Python supervisor in the
  `deploy run` graph + populated `max_cartesian_step_m/rad`. Separate effort with its own safety profile.

### 4. Contract validator (regression safeguard)

A validator/test asserting every VLA rSkill's `action_contract` is executable on each declared
embodiment (the action maps to an executable control mode, or declares slots/representation that do),
so a future cartesian rSkill cannot silently reintroduce the joint-default bug.

## Consequences

- Cartesian/OSC skills (LIBERO, SIMPLER, MetaWorld, PushT) run under `openral deploy sim` without the
  spurious n_dof E-stop; the arm executes via the scene's OSC controller.
- The reasoner never offers a skill it cannot execute on the current deploy path; `deploy run` on a
  joint-only robot correctly drops cartesian skills (no IK shim yet — deferred).
- Joint-space skills (act-aloha, so100/so101, openarm, gr1, maniskill) are unchanged.
- `supported_control_modes` semantics are clarified: real-hardware truth, with sim executability
  derived from the scene's controller — no per-robot sim/real mode duplication.
- `schema_version` stays `"0.1"` (no migrators; CLAUDE.md §6).
- **Known follow-up — `actuators_required` vs `action_contract.representation`.** The swept cartesian
  manifests keep their pre-existing `actuators_required[].kind: joint_position` (the field
  `check_capabilities` resolves against the robot's advertised modes). It is not changed to
  `cartesian_delta` here because `check_capabilities` is **not** deploy-path-aware — a joint-only robot
  (franka real) does not advertise `cartesian_delta`, so flipping the field would make
  `check_capabilities` reject the skill and break `openral sim run`. The new reasoner palette gate already
  enforces the correct deploy-path executability (and is the runtime authority); reconciling
  `actuators_required` to reflect the true output mode requires extending `check_capabilities` with the
  same sim/real deploy-path awareness — tracked as a separate change. Until then `actuators_required`
  describes the robot's physical actuator class (joint) while `action_contract.representation` is the
  authoritative policy-output contract for dispatch.

## Hazard log

- **HZ-0036-1 — deploy-sim now executes cartesian/OSC VLA actions.** Previously such actions were
  rejected (`n_dof` mismatch → E-stop) and never ran. They now execute against the scene's robosuite
  OSC controller in the MuJoCo digital twin. *Mitigation / residual risk:* a digital twin has no real
  motor to protect; the action is structurally validated by the C++ kernel (shape / NaN / self-collision,
  unchanged) and physically clamped by the OSC controller + MuJoCo `ctrlrange`. This is identical to the
  already-accepted robocasa cartesian path. **Verified** 2026-06-03: `openral deploy sim` of `pi05-libero-nf4`
  on franka executed `cartesian_delta` (`env_dim=7`) for 1400+ ticks with zero envelope violations,
  E-stops, dimension errors, or crashes.
- **HZ-0036-2 — a cartesian skill must never reach a real joint-only motor.** *Mitigation:* the
  deploy-path-aware palette gate (§2) drops cartesian skills at seed when `hal_mode="real"` and the
  robot's `supported_control_modes` excludes the action's modes. No IK shim exists, so `deploy run` on a
  joint-only robot cannot dispatch a cartesian skill. The contract validator (§4) prevents a future
  rSkill from silently regressing the action contract.
- **Review:** safety-aware reviewer required on the palette gate before PR merge. No
  `packages/openral_safety/` or `cpp/openral_safety_kernel/` code changes in this work.

## Alternatives considered

- **Add cartesian to franka `supported_control_modes`.** Rejected: conflates the real joint-only HAL
  with the sim OSC controller; would wrongly admit cartesian for `deploy run`.
- **Robot-modes-only gate (no sim/real distinction).** Rejected: drops cartesian skills for franka in
  sim too, defeating the use case.
- **Hand-written `slots` on every skill.** Rejected as the primary path: error-prone across ~12
  skills; `representation`-driven canonical slots is one line and backward-compatible (explicit slots
  still win).
- **OSC→joint IK shim for real deploy.** Deferred: lets cartesian skills drive real joint-only arms,
  but is a separate, larger effort with its own safety profile.

## Amendment 2026-06-04 — `COMPOSITE_MODE` is sim-executable

The original `_SIM_EXECUTABLE_CONTROL_MODES` set excluded `COMPOSITE_MODE`, lumping it with the
genuinely sim-unsupported modes (`CARTESIAN_TWIST`, `FOOT_PLACEMENT`, `DEX_HAND_JOINT`). That was a
regression: `COMPOSITE_MODE` is, by definition (ADR-0028d), the **sim-only robosuite-composite
(HybridMobileBase) multiplexer** — precisely the controller the `sim` deploy path runs. The path is
purpose-built to execute it: `openral_hal.lifecycle` decodes a `COMPOSITE_MODE` chunk into
`Action.composite_mode`, and `SimAttachedHAL` merges the per-slot chunks (arm `cartesian_delta` +
`gripper_position` + base `joint_velocity` + the composite flag) via its composite-split packer
(ADR-0028c) before `env.step`.

Excluding it dropped every RoboCasa mobile-manipulator VLA at boot — pi05 / rldx robocasa, whose
action contracts carry a `composite_mode` slot — even though the sim executes them fine. With no
admissible skill the runner never stepped the env, so in `openral deploy sim` the cameras never
re-rendered and any RGB-consuming node (e.g. the ADR-0035 object detector) stayed idle.

**Fix:** `COMPOSITE_MODE` is added to `_SIM_EXECUTABLE_CONTROL_MODES`. `hal_mode="real"` is unchanged —
it still gates on the robot's declared `supported_control_modes`, so a real joint-only robot
(no `composite_mode`) correctly cannot dispatch a composite skill. Covered by
`tests/unit/test_reasoner_palette_action_gate.py::test_composite_skill_executable_on_sim` and
`::test_composite_skill_not_executable_on_real_joint_robot`.

## Amendment 2026-06-04 — single source of truth + trim to packer-implemented modes

The sim-executable set was a frozenset local to `reasoner_node.py`, hand-maintained against the HAL
action-packers in `python/hal/src/openral_hal/sim_attached.py` — so the two could drift, and they had:
the gate admitted **ten** modes while the packers implement only **six**. The four extra modes
(`JOINT_TORQUE`, `JOINT_TRAJECTORY`, `CARTESIAN_POSE`, `GRIPPER_BINARY`) were latent false-admits — a
skill demanding one would pass the boot-time palette gate and then **E-stop mid-run** when its first
chunk hit a packer's unsupported-mode `else` branch. (`openral_hal.lifecycle.decode_action_chunk`
decodes some of those modes, but decoding ≠ pack-executing — the gate must mirror the *packers*.)

**Fix (this amendment):**

1. The set is promoted to a canonical module-level constant
   `openral_core.SIM_EXECUTABLE_CONTROL_MODES` (next to `CONTROL_MODE_TO_UINT8`; `openral_core` is
   already a dependency of both the reasoner and the HAL, so it is the correct shared home).
2. It is **trimmed to the six modes the default sim packers actually implement**:
   `JOINT_POSITION`, `JOINT_VELOCITY`, `CARTESIAN_DELTA`, `GRIPPER_POSITION`, `BODY_TWIST`,
   `COMPOSITE_MODE`. `BODY_TWIST` executes via the direct base-qpos path in
   `SimAttachedHAL.send_action` (`_apply_body_twist_to_qpos`), not through a packer slot.
3. Both `reasoner_node.py` (the gate) and `sim_attached.py` (the packers) import the one constant; the
   lockstep is pinned **in both directions** by
   `tests/unit/test_sim_executable_modes_match_packers.py`, which drives every `ControlMode` through
   both packers and asserts the handled-mode union equals the constant — and that the four removed
   plus three never-admitted modes (`CARTESIAN_TWIST`, `FOOT_PLACEMENT`, `DEX_HAND_JOINT`) are both
   absent from the constant and rejected by both packers. Drift can no longer ship silently.

## Amendment 2026-06-05 — episode-terminal auto-reset must catch *raised* terminals too

`openral deploy sim` drives the wrapped `SimRollout` continuously through `SimAttachedHAL` — there is
no `SimRunner` owning episode boundaries (that exists only on the `openral sim run` path). To make an
episodic backend behave like a continuous digital twin, `SimAttachedHAL._step_and_cache` auto-resets on
episode termination: the prior step's `StepResult.terminated/truncated` is latched as `_episode_done`,
and the next step resets the env before stepping. This handles backends that **return** a terminal.

**Gap this amendment closes.** The raw-robosuite backends — `franka_libero_custom_bddl` (custom-BDDL
milk/soup scenes) and `so100_robosuite` — construct `robosuite.OffScreenRenderEnv` with
`ignore_done=False`. Such envs do not *return* a terminal forever; once `done` is set (task success or
`horizon == task.max_steps`), the **next** `env.step` **HARD-RAISES**
`ValueError("executing action in terminated episode")` (`robosuite/environments/base.py`). Because the
terminal arrives as a *raise*, not a returned flag, the `_episode_done` latch never fires, the deferred
reset never runs, and **every** subsequent `send_action`/`idle_step` re-raises — the arm freezes and the
log spams `send_action (safe_action) failed: … env.step failed: executing action in terminated episode`
(observed live on `scenes/sim/franka_libero_pnp.yaml`). This was invisible to the existing
returned-terminal test, which forces `_episode_done = True` and so never exercises the raise.

This is **not** the same as the robocasa path: `sim_bringup._maybe_force_ignore_done` injects
`ignore_done=True` for `robocasa*` scenes only (robocasa is the sole backend that *reads* the option;
LIBERO/so100 hardcode it), so robocasa never raises. An audit of all `openral_sim.SCENES` backends found
the gymnasium-wrapped (`libero` suite, `metaworld`, `aloha`, `pusht`, `maniskill3`, `simpler_env`) and
native-MuJoCo (`so101_box`, `tabletop_push`, `openarm_robosuite`) backends all *return* terminals and
never raise — so only the two raw-robosuite backends were exposed.

**Fix (this amendment):** the recovery is moved into `SimAttachedHAL._step_and_cache` itself, so it is
**backend-agnostic** rather than a per-scene `ignore_done` patch. The reset block is extracted to
`_reset_terminated_episode(source, *, trigger)` and called from both terminal paths:

1. *returned* terminal — the existing `_episode_done` latch (`trigger="returned-terminal"`); and
2. *raised* terminal — `env.step` is wrapped; if it throws and
   `openral_hal.sim_attached.is_terminated_episode_error(exc)` matches robosuite's guard, the env is
   reset once and the action re-stepped (`trigger="raised-terminal"`). robosuite's `reset` clears
   `done`, so the re-step cannot re-raise the same guard.

A non-terminal `step` failure (bad action width, NaN, contact blow-up) is **never** swallowed — the
predicate returns `False` and the original `ROSRuntimeError` propagates (CLAUDE.md §1.4 observability).
Both `trigger` values are surfaced in the `[sim_attached.<source>] episode terminated (<trigger>);
auto-reset` stdout line so the two paths stay distinguishable in deploy-sim output. Because the fix sits
at the single env-stepping choke point, any future raw-robosuite backend is covered automatically.

Tests (`python/hal/tests/test_sim_attached_action_dim.py`, real LIBERO twin, no mocks): a new test puts
the real robosuite env into its terminal state with the latch clear (the genuine desync) and asserts
`send_action` recovers instead of raising; a pure-predicate test pins `is_terminated_episode_error` to
robosuite's message only (real faults propagate). The pre-existing returned-terminal test is unchanged.

## Amendment 2026-06-08 — three-tier scene paths (ADR-0041)

ADR-0041 split `scenes/` into deploy/sim/benchmark tiers and stripped
rSkill names from filenames. The bug-observation note in the 2026-06-05
amendment ("observed live on …") now points at
`scenes/sim/franka_libero_pnp.yaml`, the renamed successor to the
pre-refactor `scenes/native/pi05_libero_custom_milk.yaml` (the file
content — custom-BDDL milk-pick scene against `franka_libero_custom_bddl`
— is the same; only the on-disk path was renamed). The raised-terminal
bug, the predicate, and the auto-reset fix are unchanged. See ADR-0041
and [`scenes/README.md`](https://github.com/OpenRAL/openral/blob/master/scenes/README.md) for the tier hierarchy.
