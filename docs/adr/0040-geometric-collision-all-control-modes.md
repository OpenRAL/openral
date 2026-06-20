# ADR-0040: Geometric collision checking for every control mode (not just joint-position)

- Status: **Proposed**
- Date: 2026-06-06
- Deciders: safety-WG (required), TSC
- Related: [ADR-0030](0030-geometric-safety-collision-checking.md) (self/world/voxel
  collision in the kernel — this ADR completes the non-position-mode phase it
  explicitly deferred); [ADR-0020](0020-cpp-safety-kernel.md) (the C++ kernel +
  allocation-free `validate()` contract); [ADR-0018](0018-ros2-reasoner-supervisor.md)
  §5 (the `ActionChunk` / `/openral/candidate_action` topic contract);
  [ADR-0036](0036-osc-action-contracts-deploy-path-gate.md) (OSC/Cartesian action
  contracts on the deploy path — the modes this ADR must cover);
  [ADR-0024](0024-ros-wrapped-rskills.md) (wrapped MoveIt/Nav2 planning-time checks
  this does not replace); CLAUDE.md §1.1 (safety beats helpfulness), §1.4
  (explicit / reject-not-clamp), §1.5 (Python proposes, C++ disposes; hot path is
  C++ and bounded), §3 Layer 6 Safety (safety-WG review + hazard-log update +
  at-least-as-conservative tests).

## Context

ADR-0030 added an allocation-free geometric collision check (self + world-primitive
+ world-voxel) to the C++ safety kernel: for each row of a candidate `ActionChunk`
the kernel runs forward kinematics, poses the per-link capsules, and rejects +
E-stops the chunk if any capsule penetrates another link, a world primitive, or an
occupied voxel within the configured margin.

That check, as shipped, runs **only for `ControlMode::kJointPosition` chunks**.
`cpp/openral_safety_kernel/src/lifecycle_kernel.cpp` gates the entire geometric
block on the control mode:

```cpp
// Runs only for absolute joint-position chunks (the rows are full joint
// configs FK can place).
if (geom_enabled &&
    view.control_mode == static_cast<std::uint8_t>(ControlMode::kJointPosition) &&
    view.n_dof >= collision_required_dof_ && view.flat_data != nullptr) {
  // ... self / world / voxel checks ...
}
```

ADR-0030 §"Forward kinematics" anticipated this limit:

> `JOINT_POSITION` / `JOINT_TRAJECTORY` rows are joint configurations directly.
> `CARTESIAN_*` / `*_TWIST` modes require seeding from the latest `/joint_states`
> and integrating forward (a later phase that adds a kernel joint-state
> subscription).

That later phase never landed. The result is a **silent safety hole that affects
most rSkills**, because joint-position is the *least* common actuation mode in the
fleet:

| rSkill / source | Control mode it commands | Geometric collision today |
| --- | --- | --- |
| `pi05` / `pi0.5` robocasa policies | `JOINT_VELOCITY` (OSC-lowered) | **none** |
| `smolvla`, `act`, `dp/dp3` (delta-EE) | `CARTESIAN_DELTA` | **none** |
| `rldx`, composite robocasa | `COMPOSITE_MODE` (twist + delta + gripper) | **none** |
| teleop / jog streams | `CARTESIAN_TWIST` / `BODY_TWIST` | **none** |
| MoveIt-planned waypoints (ADR-0024) | `JOINT_TRAJECTORY` | partial (planner-side; kernel only if rows are absolute joint configs) |
| direct joint goals | `JOINT_POSITION` | ✅ covered |

This was observed concretely: in `openral deploy sim`
(`pi05_robocasa_pnp_nf4`), with the world-voxel map correctly populated
(`/openral/world_voxels` ~985 occupied cells after the octomap clock fix, PR #268)
and `world_voxel_enabled=true`, the arm still drove into the kitchen counter
**uncaught** — every candidate chunk was `JOINT_VELOCITY`, so the geometric block
was skipped. A hand-injected `JOINT_POSITION` chunk at the same configuration *was*
rejected + E-stopped, confirming the only difference is the mode gate.

The fix belongs in the kernel and applies to **every embodiment and every rSkill**,
so it is an ADR, not a patch.

### Non-goals

- This ADR does **not** change the scalar envelope check, the world-model ingestion
  contract, or the fail-closed degradation policy of ADR-0030 — those are reused
  verbatim.
- It does **not** add collision geometry for the gripper/fingers. The collision
  model deliberately excludes the end-effector contact surfaces (so the gripper can
  reach targets); intentional gripper-surface contact remains out of scope and is
  governed by force/torque limits in the envelope, not geometry.
- It does **not** replace planning-time collision checking for wrapped MoveIt/Nav2
  skills (ADR-0024); the kernel remains the last-line reactive gate.

## Decision

**Make the kernel's geometric collision check mode-complete: for every control mode
an rSkill can emit, reconstruct the candidate future configuration(s) the chunk
would produce, seeded from the latest measured joint state, and run the existing
self/world/voxel checks over them. No control mode is silently exempt.**

### 1. Kernel subscribes to the measured joint state

Add a double-buffered `/joint_states` (or `world_state_fast`) subscription to the
kernel (the subscription ADR-0030 deferred). It maintains the latest measured joint
configuration `q_meas` in a lock-free latest-wins slot, plus its stamp for a
freshness gate. `q_meas` is **only** read on the hot path; ingestion never allocates.

Freshness is fail-closed: if `q_meas` is older than `collision_state_deadline_ms`
when a mode that *requires* seeding arrives, the chunk is rejected as
`state_unavailable` (drop, no latch — motion resumes when a fresh state lands),
mirroring the existing `world_unavailable` / `voxel_unavailable` semantics.

### 2. Per-mode reconstruction of the checked configuration(s)

For each chunk the kernel derives the sequence of joint configurations to FK + check.
`dt` is the chunk period (`1 / rate_hz`); substeps `S` keep the swept check
continuous as in ADR-0030.

| Control mode | Configuration(s) checked | Seed |
| --- | --- | --- |
| `JOINT_POSITION` / `JOINT_TRAJECTORY` | rows verbatim (current behavior) | none |
| `JOINT_VELOCITY` | `q_k = q_meas + Σ_{i≤k} v_i · dt` integrated over rows, with `S` substeps between configs | `q_meas` |
| `JOINT_TORQUE` | conservative: check `q_meas` plus a bounded reachable shell `q_meas ± v_max · dt` (torque→config needs full dynamics; we do not integrate it on the hot path) | `q_meas` |
| `CARTESIAN_POSE` | IK-free: FK `q_meas`, then check the **swept tool/wrist capsule** from the current EE pose to the commanded pose (capsule-cast), not a re-solved `q` | `q_meas` + EE FK |
| `CARTESIAN_DELTA` | same as `CARTESIAN_POSE` with target = current EE pose ∘ delta | `q_meas` + EE FK |
| `CARTESIAN_TWIST` / `BODY_TWIST` | swept capsule along `twist · dt` from the current EE / base pose | `q_meas` (+ base pose from TF) |
| `COMPOSITE_MODE` | decompose into its sub-actions; apply this table to each and reject if **any** sub-check trips | per sub-action |
| `GRIPPER_*` | no arm-link motion → geometric check skipped (envelope force limits apply) | — |

The Cartesian/twist rows are checked with a **capsule-cast** (swept EE/wrist capsule
between the seeded pose and the predicted pose) rather than a re-solved IK
configuration: it is allocation-free, requires no IK in the hot path, and is
**conservative** (the cast volume is a superset of the true link path for the
distal links that actually approach obstacles). Whole-arm Jacobian prediction is a
possible refinement (Phasing) but is not required for the conservative guarantee.

### 3. Conservatism is the contract, not accuracy

Every per-mode reconstruction must be **at least as conservative** as reality: it
may reject a chunk that would not have collided, but it must never pass a chunk that
would. Where the reconstruction is uncertain (torque mode, IK-ambiguous Cartesian),
the kernel widens the checked volume (reachable shell / capsule-cast) rather than
narrowing it. This is the invariant the safety-WG and the test suite verify.

### 4. Degradation and observability (reused from ADR-0030, extended)

- Fail-closed default unchanged: no collision model, stale/over-capacity world,
  **stale measured state for a seed-requiring mode**, or per-chunk budget overrun →
  reject + E-stop (or `state_unavailable` drop without latch for the freshness gate).
- `safety.check` spans gain `safety.collision_mode` (the control mode that was
  reconstructed) so the dashboard and traces show *which* path rejected.
- The `warn`-only sim/bring-up mode (explicit param + loud log + `severity=warn`)
  extends to the new modes; it never disables the check and is never the default.

### 5. No mode left silently unchecked

The mode gate is inverted: instead of "check only if `JOINT_POSITION`", the kernel
checks **every** geometry-relevant mode and maintains an explicit allow-list of
modes that legitimately need no arm-geometry check (`GRIPPER_*`, `FOOT_PLACEMENT`
pending its own ADR). An unknown / unhandled mode with geometry enabled is
**rejected** (fail-closed), not passed.

## Alternatives considered

1. **Require all rSkills to emit `JOINT_POSITION`.** Rejected: VLA policies emit
   velocity/delta chunks by construction (ADR-0036); forcing position output would
   require re-integrating in every adapter, duplicating logic and losing the
   policies' native action space. Safety must adapt to the fleet's modes, not the
   reverse.
2. **Planning-time collision only (MoveIt/Nav2).** Rejected: VLAs and teleop do not
   plan; the kernel is the only gate they cross (ADR-0030 §Context).
3. **Check only the current measured configuration each tick (reactive, no seeding).**
   Partially adopted as the *torque* fallback, but rejected as the general design:
   it catches a collision one tick *after* the command that caused it (the arm is
   already penetrating), whereas seeding + integration is predictive and rejects the
   colliding command *before* it is applied. The predictive form is strictly more
   conservative for non-zero-latency actuation.
4. **Full IK per Cartesian row.** Rejected for the hot path: IK is iterative,
   allocates, and is ambiguous (multiple solutions). The conservative swept
   capsule-cast gives the safety guarantee without IK.

## Consequences

- **Every rSkill gains world/self collision coverage**, not just joint-position
  ones — closing the hole that let the pi05 robocasa arm hit the table.
- The kernel gains a `/joint_states` subscription and per-mode reconstruction code;
  both are bounded and allocation-free per ADR-0020. The reconstruction adds a
  small, bounded cost to the per-chunk budget (measured against the deadline in the
  HIL/sim suite).
- **Safety-WG gating (CLAUDE.md §3):** this touches `cpp/openral_safety_kernel/` and
  changes safety behavior. It requires (a) a safety-WG reviewer, (b) a hazard-log
  entry (the velocity-mode-uncaught-collision hazard this closes), and (c) tests
  proving the new behavior is **at least as conservative** as the position-mode path
  for the same geometry — including a regression that reproduces the deploy-sim
  table collision and asserts the kernel now rejects + E-stops the velocity chunk.
- Tests: per-mode reconstruction unit tests (each mode vs a known-colliding and a
  known-clear geometry), a property test that the reconstructed swept volume is a
  superset of the integrated link path, and a sim test
  (`tests/sim/test_panda_mobile_pi05_robocasa.py`) asserting a counter approach is
  vetoed.
- Latency: the swept/substep count is the tunable; the budget is enforced on the
  reference host as in ADR-0030.

## Phasing

1. **Contracts + joint-state subscription** — add the kernel `/joint_states` seed +
   freshness gate + the `safety.collision_mode` span attribute; no behavior change
   yet (position mode still the only checked mode), so it is a safe, reviewable base.
2. **Joint-velocity coverage** — integrate `v·dt` and run the existing checks;
   ship the deploy-sim regression. This alone closes the observed pi05 hole.
3. **Cartesian / twist coverage** — swept capsule-cast for `CARTESIAN_*` /
   `*_TWIST`, and `COMPOSITE_MODE` decomposition.
4. **Torque coverage + refinements** — conservative reachable-shell for
   `JOINT_TORQUE`; optional Jacobian whole-arm prediction for tighter (still
   conservative) Cartesian bounds.

Each phase is independently safety-WG-reviewed and ships its own tests; the
fail-closed default means a not-yet-implemented mode is **rejected**, never silently
passed, from Phase 1 onward.

## Fleet control-mode audit (what is actually emitted today)

Before implementing, the in-tree rSkills + robots were audited for the
`ControlMode` they emit on `/openral/candidate_action`:

| ControlMode | Emitted by | Through kernel? | Build now? |
| --- | --- | --- | --- |
| `JOINT_POSITION` | act-aloha, molmoact2-so101, smolvla-maniskill/metaworld, pi05-openarm, rldx1-gr1, ACT/diffusion legacy | yes | already (ADR-0030) |
| `CARTESIAN_DELTA` (+`GRIPPER_POSITION`) | **12 rSkills** — all LIBERO (act/smolvla/pi05/xvla/rldx), SIMPLER (widowx/google), DROID, **+ the robocasa arm chunk** (rldx-rc365/robocasa, pi05-robocasa365) | yes | **YES — Phase 3 (reactive floor + Jacobian predictive look-ahead)** |
| `JOINT_VELOCITY` | robocasa **base** chunk (rldx-rc365/robocasa, pi05-robocasa365 slot 7:9) | yes | **YES — Phase 2** |
| `COMPOSITE_MODE` | robocasa mux flag (same 3 skills, slot 11, n_dof=1 scalar) | yes | **no geom** — scalar flag, no arm geometry; passes the `n_dof≥required` gate; the arm is the companion CARTESIAN_DELTA chunk |
| `GRIPPER_POSITION`/`GRIPPER_BINARY` | LIBERO/robocasa gripper slot (n_dof=1) | yes | **no geom** — opening/closing does not move arm-link capsules |
| `BODY_TWIST` | nav2 `cmd_vel` bridge | **NO** — bypasses the kernel (ADR-0024) | n/a |
| `JOINT_TORQUE`, `JOINT_TRAJECTORY`, `CARTESIAN_POSE`, `CARTESIAN_TWIST`, `FOOT_PLACEMENT`, `DEX_HAND_JOINT` | **none** | — | defer (unused) |

**Verdict applied:** (a) Cartesian — **built**, both the reactive floor *and* the
Jacobian predictive look-ahead (largest footprint, and the actual robocasa-arm
path that hit the table); (c) COMPOSITE_MODE — **no work needed** (scalar
passthrough; arm covered by its CARTESIAN_DELTA companion); (d) JOINT_VELOCITY
predictive-dt — **moot for the only emitter** (the robocasa base velocity dofs are
zeroed by the base-relative frame fix, so integrating them is a no-op; reactive
covers it); (b) JOINT_TORQUE shell — **skipped** (no robot/rSkill emits it).

## Implementation status (this branch)

- **Phase 1** (joint-state seed + fail-closed gate) — done.
- **Phase 2** (joint-velocity: reactive measured config + predictive `v·dt`, dt-gated) — done.
- **Phase 3** (Cartesian) — done, in two layers:
  - **Reactive** measured-config check for `CARTESIAN_*`/`BODY_TWIST` (rejects an
    arm already in/at an obstacle) — the guaranteed floor.
  - **Predictive** look-ahead for `CARTESIAN_DELTA` (the robocasa arm + all
    LIBERO/SIMPLER/DROID arms): the kernel reconstructs where the chunk's EE
    deltas drive the **whole arm** via a damped-least-squares Jacobian
    (`jacobian_dls_step`, `dq = Jᵀ(JJᵀ+λ²I)⁻¹·dx`), integrates the trajectory
    forward, and runs the full capsule self/world/voxel check at each step. The
    **last step is always verified**; earlier steps up to `collision_predict_max_steps`
    (0 = all). Per-step margin inflation (`collision_predict_margin_growth_m`)
    bounds the linearization/DLS residual. On a mobile base the base dofs are
    **blocked** from the arm Jacobian (the EE delta is realised by arm joints; the
    base is checked in its own frame). The EE control link is the deepest
    collision link (`ee_link_index_from_collision_params`); `collision_ee_link_index<0`
    disables predictive (reactive only). **Safety invariant:** predictive is
    purely *additive* early warning on top of the reactive floor — an imperfect
    Jacobian/frame reconstruction can only add rejections (conservative) or
    degrade to reactive (catch at contact), never make the kernel less safe.
    Allocation-free (fixed 6×6 DLS solve + stack Jacobian scratch, capacity
    `kMaxJacobianDof`); proven by `NoAlloc.JacobianDlsStepIsAllocationFree`.
  - **Assumption:** `CARTESIAN_DELTA` rows are EE twists in the **base frame**
    (robosuite OSC convention). A future non-base `frame_id` would fall back to
    reactive (the floor), not mis-predict unsafely.
  - **Still deferred:** predictive `CARTESIAN_TWIST`/`CARTESIAN_POSE` (no in-tree
    emitter) stay reactive/fail-safe.
- **Mobile-base frame fix** — the base-relative FK zeroes the manifest `base_joints`
  so a mobile manipulator's arm is checked in the `base_link` frame the voxel grid
  lives in (also repairs ADR-0030's position path for mobile bases).
- **Deferred (unused today):** `JOINT_TORQUE` reachable-shell, predictive Cartesian
  capsule-cast, `CARTESIAN_TWIST`. These remain **fail-closed** when a future
  rSkill emits them with the seed plumbed.

## Hazard log

### HZ-0040-1 — mobile-base dof zeroing silently disabled (dangling reference)

- **Severity:** high. On a mobile manipulator (`panda_mobile`/robocasa) the
  base-relative FK frame fix was **silently a no-op**, so the arm capsules were
  placed at the base's *world* pose (metres outside the base_link-relative
  world/voxel grid) and **no world/voxel collision could ever be caught** — the
  exact "robot drove into the table and the kernel did not stop it" symptom.
- **Root cause:** `on_configure` iterated a range-based `for` directly over
  `get_parameter("collision_base_dofs").as_integer_array()`. `as_integer_array()`
  returns a reference into the temporary `rclcpp::Parameter`; the range-based for
  does **not** lifetime-extend that temporary, so the loop read freed stack
  memory. `collision_base_dofs_` ended up empty (or garbage) depending on stack
  reuse — i.e. it *appeared* to work under some builds/logging and failed under
  others. (ASAN: `stack-use-after-scope`, `lifecycle_kernel.cpp:246`.)
- **Fix:** bind the parameter array to a **named local** before iterating
  (`const std::vector<int64_t> base_dofs = ...; for (d : base_dofs)`).
- **Detection / regression:** found via AddressSanitizer; locked down by
  `LifecycleKernelTest.MobileBaseArmCaughtAgainstVoxelWall` — a deterministic,
  geometry-verified test that seeds the base far out in the world, requires the
  base dof to be zeroed for the arm capsule to land in an occupied voxel wall,
  and asserts e-stop. Verified deterministic (10/10 runs) and ASAN-clean.
- **Conservativeness:** strictly more protective — the fix *enables* a check
  that was previously dead; no path is made less conservative.

### HZ-0040-2 — predictive Cartesian reconstruction is an approximation

- **Concern:** the predictive `CARTESIAN_DELTA` check reconstructs the future arm
  configuration from EE deltas using a *damped-least-squares Jacobian* (a local
  linearization) and assumes deltas are in the **base frame**. Both are
  approximations: DLS can undershoot near singularities; a wrong frame would
  predict the wrong trajectory.
- **Why it is safe anyway (design invariant):** prediction is **purely additive**
  on top of the reactive measured-config check, which always runs first and
  e-stops if the *current* config collides. The predictive pass can therefore
  only (a) add an *earlier* rejection, or (b) fail to predict and fall back to
  the reactive floor (catch at contact). It can **never** suppress the reactive
  check or pass a config the reactive check would reject. Per-step margin
  inflation (`collision_predict_margin_growth_m`) further biases prediction
  toward early rejection to bound the linearization/DLS residual. Worst case =
  reactive-only behavior, which is the pre-Phase-3 contract.
- **Mitigations in code:** base dofs blocked from the arm Jacobian (mobile base);
  EE-link mis-id bounded by the whole-arm capsule check; `collision_ee_link_index<0`
  and non-`CARTESIAN_DELTA` modes fall back to reactive; allocation-free
  (no hot-path surprises).
- **Tests:** `CartesianDeltaPredictiveCatchesChunkDrivingEeIntoWall` (start clear,
  predicted trajectory enters a wall → estop) and
  `CartesianDeltaPredictivePassesWhenTrajectoryStaysClear` (no false positive),
  plus `JacobianDls.*` unit coverage (twist realization, fail-safe on bad EE
  link, damping bound, blocked-dof, allocation-free).

## Out of scope — base navigation is not kernel-protected

Mobile-base *navigation* via the `rskill-nav2-navigate-to-pose` rSkill does **not**
flow through this kernel at all, and the geometric collision check here does not
cover it:

- The skill is **result-only** (ADR-0024): it forwards a `nav2_msgs/NavigateToPose`
  goal and emits **no** `ActionChunk` on `/openral/candidate_action`.
- Nav2's own behaviour tree publishes `cmd_vel` **directly** to the base
  controller, bypassing the safety supervisor. Base collision avoidance therefore
  relies **entirely on Nav2's costmap**, not on our occupancy/voxel check.
- No in-tree HAL advertises `body_twist` in `supported_control_modes` today, so
  there is no velocity stream for the kernel to gate even if we wanted to.

**Follow-up (tracked, unscheduled):** when a mobile-base HAL exposes `body_twist`
under the supervisor, route the base velocity stream through the kernel and reuse
the same base-relative voxel check (the base capsules, not the arm, against the
local occupancy grid). Until then, the kernel's protection is the **manipulator**,
and the costmap's is the **base** — this split is intentional and documented so it
is not mistaken for full base-collision coverage.
