# External joint-torque observer — the force/current collision tier (issue #373)

The safety kernel estimates the external torque on every joint from what `/joint_states`
already carries (position, velocity, effort) and stops the arm when a joint's estimate
exceeds its ceiling. It is the kernel's first check that reads a measured force, not a
geometry. It only adds stops: an invalid estimate stops nothing and relaxes nothing, so
every geometric check behaves exactly as it does without this layer.

Status: implemented and tested in simulation. **No shipped manifest or scene sets a
ceiling.** The real OpenArm runs the observer for disclosure only, until the load-cell
go/no-go (below). ADR, hazard-log rows and Safety-WG review are owed (§6).

## 1. What runs

`cpp/openral_safety_kernel/src/force_observer.cpp`, allocation-free, no external
dependency (like the collision core). One sample per `/joint_states` message:

- Dynamics over the kernel's own kinematic tree (the `CollisionModel` the self-collision FK
  walks), with per-link inertials and per-joint friction from the manifest:
  `M(q) q̈ + C(q, q̇) q̇ + g(q) + τ_f(q̇) = τ_m + τ_ext`.
- `M` from n RNEA passes with unit accelerations; `g + C q̇` from one pass; `Cᵀq̇ = Ṁq̇ − C q̇`
  with `Ṁ` by finite difference. Mimic links (the OpenArm's follower finger) project
  through their multiplier.
- Friction: `τ_f = offset + coulomb · tanh(q̇ / v_c) + viscous · q̇`; `armature` adds to
  `M`'s diagonal.
- Generalized-momentum observer `ṙ = K (τ_ext − r)`, discretised exactly so it is stable
  at every sample period (the real OpenArm publishes at ~733 Hz, a deploy-sim HAL at its
  control rate):
  `ŵ = Δp/dt − ½(τ_k + τ_{k−1}) + ½(β_k + β_{k−1})`, `r ← r + (1 − e^{−K dt})(ŵ − r)`,
  with `p = M q̇` and `β = g − Cᵀq̇`.

The time base is the message header stamp; a repeated stamp is ignored and a gap longer
than the state deadline restarts the estimate.

## 2. Validity gate

The estimate is reset (and judged invalid) when any of these holds:

| condition | why |
|---|---|
| a joint the tree moves has no fresh position or velocity | the dynamics couple every column |
| a monitored joint has no fresh effort (`effort=[]` means not measured, #388) | nothing to compare |
| a monitored joint is within `limit_margin_rad` of a position limit | a hard stop pushes back like a contact |
| a non-finite input or `dt ≤ 0` | |

Freshness uses the kernel's `collision_state_deadline_ms`. The observer is refused at
configure for a mobile base (`collision_base_dofs`): it models a fixed root.

## 3. The collision tier

Over `SafetyEnvelope.max_external_joint_torque_nm[joint]` (N·m), on a valid estimate, the
kernel latches like any other stop: `FailureTrigger(KIND_FORCE)` with
`ForceEvidence{joint_or_ee, measured_n, limit_n, unit: "Nm"}`, `SafetyStatus` drop reason
`KIND_FORCE`, `/openral/estop`, and the log line `safety.force_limit`. `/openral/estop_reset`
restarts the estimate; a contact that is still there trips again within a few `1/K`.

The bound is **per joint**, not a hand force. Mapping the residual to the hand through
`pinv(Jᵀ)` blows up near a singularity: the real OpenArm reads 45–100 N there with no
contact (#373 step 2). `SafetyEnvelope.max_force_n` therefore stays declared, not enforced.

There is **no contact tier** (tolerating contact under a force bound). The real
OpenArm's signal does not support one: the reported effort is the PD law (correlation
≥ 0.99 with `Kp(cmd − q) − Kd q̇`), so it adds nothing beyond the position lead the
#349 workspace rule already bounds; the J3/J4 stiction band (±0.6–0.8 N·m at J4) makes
forces below ~2–3 N at the hand unobservable at rest; and the held-out no-contact floor is
already 4–12 N at the hand (p50). Under the #349 workspace rule this layer can only add a
stop, never tolerate one.

## 4. Configuration

- `RobotDescription.force_observer` (`ForceObserverSpec`): inertials (`LinkInertial`, in the
  manifest's link frames) and joint terms (`JointFriction`), plus `gain_per_s`,
  `coulomb_velocity_rad_s`, `limit_margin_rad`, `gravity_xyz_m_s2`. Lowered from the MJCF by
  `tools/lower_force_observer.py` (`openral_safety.dynamics_lowering`), which refuses unless
  every manifest link's zero-configuration pose is its MJCF body's; never hand-written.
  `tests/sim/safety/test_force_observer_lowering.py` fails on drift.
- `SafetyEnvelope.max_external_joint_torque_nm`: `{joint: N·m}`. A robot, deploy scene or
  rSkill may set it; a deploy scene or rSkill may add a joint or lower a value, never raise
  or drop one (merged per joint). A ceiling with no model is refused by the schema, the
  loader and the kernel.
- The kernel always publishes its estimate on `/openral/safety/external_torque`
  (`sensor_msgs/JointState`, `effort` = estimate in the action's dof order, 50 Hz, sensor
  QoS) and reports `force_observer` (`valid`/`invalid`) and `force_peak_ratio` (max
  |estimate| / ceiling since the last tick) in `/diagnostics`.

To enable on a deploy scene:

```yaml
safety:
  max_external_joint_torque_nm: {right_joint1: 3.0, right_joint2: 3.0}
```

## 5. Evidence

| source | result |
|---|---|
| gtest `test_force_observer` (pendulum, closed form) | RNEA = closed form to 1e-9; a 1.5 N·m step read to 0.05 N·m in 5/K; stable and convergent at 30 Hz with K·dt = 1.7; modelled Coulomb friction not read as contact (unmodelled: > 0.5 N·m) |
| gtest `test_force_observer_kernel` (real lifecycle node) | stops on 3 N·m over a 2 N·m ceiling and names the joint; no stop on gravity-consistent effort, an unmonitored joint, `effort=[]`, or a joint in its limit margin; refuses a ceiling with no model |
| `test_safety_kernel_force_observer_openarm.py` (real kernel binary, MuJoCo OpenArm v2, 250 Hz `/joint_states`, ceiling 3 N·m on right J1–J4) | free sinusoidal motion peaks at 0.52 N·m; a 40 N push on the moving hand stops in 8 ms; an arm swung into a fixed box stops on the first contact sample (pre-contact peak 0.40 N·m) |
| real OpenArm, 9 Oct recordings + 16 Oct policy run (#373 step 2) | no-contact residual 3–5 N·m on J1–J3 at policy speed (CAD model, no friction calibration): a ceiling below ~5 N·m on the shoulders would false-trip there |

The simulation rows use the plant's own model (the lowering reads the same MJCF), so they
prove the wiring and the dynamics, not a real-arm floor.

## 6. Owed before a real-hardware ceiling

1. **Real-cell load-cell go/no-go** (#373 step 2) on the OpenArm: sweep known forces at
   three or more poses including one near a singularity, read `/openral/safety/external_torque`,
   and set per-joint ceilings above the measured no-contact floor at policy speed.
2. **A real friction model**: the manifest's joint terms are the MJCF's (the simulator's
   drives). Per-joint offsets and stiction from the real recordings go into
   `JointFriction` (a re-lowering keeps them only if the tool is taught to; today it
   overwrites them).
3. **ADR** in `OpenRAL/management` `adr/`: the observer's output crosses Layer 0
   (effort) → Layer 5 (kernel).
4. **Hazard-log rows**: an observer false negative (a contact absorbed by stiction, or a
   joint within its limit margin); effort dropout (the estimate goes invalid and stops
   nothing); model or friction drift (a false stop, or a raised floor).
5. **Safety-WG review** (CLAUDE.md §3), with #374's real-hardware bounds.
