# openral_safety_kernel — C++ safety kernel

Layer 5 (Safety). A separate-process, real-time validator on the chunk-rate boundary. It sits
between the rSkill runner and the HAL: every `ActionChunk` on `/openral/candidate_action` is
checked, and only accepted chunks are republished on `/openral/safe_action`. It replaces the
Python pass-through (`SafetyPassthroughNode`, `packages/openral_safety/openral_safety/supervisor_node.py`)
behind the same topic contract. **Python proposes, C++ disposes** (CLAUDE.md §3 "Safety").

The kernel is **deny by default**. A chunk is dropped unless every enabled check passes, an input
the kernel needs (envelope, measured state, world grid, payload set) that is missing or stale
drops the chunk, and any non-finite or malformed input is refused (see
[Fail-closed inputs](#fail-closed-inputs)). The open kernel contains only checks that can refuse
motion. Rules that *relax* a verdict live outside this tree, in a private exemption provider
(OpenRAL Pro); see [Exemption provider](#exemption-provider-adr-0121).

## Order of checks

For each candidate chunk (`on_candidate_action`, `src/lifecycle_kernel.cpp`):

1. **Fault latch** — latched: drop (`estop_latched`).
2. **Envelope loaded** — no envelope: drop (`envelope_unconfigured`), not latched.
3. **Per-mode magnitude bounds** — `validate()` in `src/validator.cpp`, allocation-free.
4. **Input availability** (geometric checks only) — fresh, complete measured state; fresh world
   grid (receipt deadline and sensor data-age budget); fresh peer grid when enabled; verifiable
   attached-payload set. Missing or stale: drop, not latched.
5. **Geometric collision** over the chunk horizon, for position, velocity and Cartesian modes:
   self-collision, then arm links vs the world grid, then the carried payload vs the world grid
   (support-witness and attach-residue exemptions only), then the contact-force gate, then the
   robot and payload vs the peer grid. The first hit stops the sweep.
6. **Accept**, optionally slowed by the distance-graded velocity scale (off by default).

Independently of chunks, the **external-torque observer** runs on `/joint_states` and can latch.
The base is not checked here (see [Base motion](#what-the-kernel-does-not-check-base-motion)).

## Interfaces

### Topics and services

| Direction | Topic / service | Type | QoS |
| --- | --- | --- | --- |
| sub | `/openral/candidate_action` | `openral_msgs/ActionChunk` | RELIABLE, VOLATILE, KL=50 |
| sub | `/openral/estop` | `std_msgs/Empty` | RELIABLE, VOLATILE, KL=10 |
| sub | `/joint_states` | `sensor_msgs/JointState` | BEST_EFFORT, VOLATILE, KL=1 (only with a geometric check or the observer on) |
| sub | `/openral/world_voxels` | `openral_msgs/OccupancyVoxels` | RELIABLE, VOLATILE, KL=1 (`world_voxel_enabled`) |
| sub | `/openral/peer_voxels` | `openral_msgs/OccupancyVoxels` | RELIABLE, VOLATILE, KL=1 (`peer_voxel_enabled`) |
| sub | `/openral/world_state_fast` | `openral_msgs/WorldStateStamped` | RELIABLE, VOLATILE, KL=1 (attached checking or grasp allowance on) |
| pub | `/openral/safe_action` | `openral_msgs/ActionChunk` | RELIABLE, VOLATILE, KL=50 |
| pub | `/openral/estop` | `std_msgs/Empty` | RELIABLE, VOLATILE, KL=10 |
| pub | `/openral/failure/safety` | `openral_msgs/FailureTrigger` | RELIABLE, VOLATILE, KL=50 |
| pub | `/openral/safety_status` | `openral_msgs/SafetyStatus` | RELIABLE, **TRANSIENT_LOCAL**, KL=1 |
| pub | `/openral/attachment_state_applied` | `std_msgs/UInt64` | RELIABLE, TRANSIENT_LOCAL, KL=1 (`attached_collision_enabled`) |
| pub | `/openral/safety/external_torque` | `sensor_msgs/JointState` (`effort` = estimate, N·m) | BEST_EFFORT, VOLATILE, KL=5 (`force_observer_enabled`) |
| pub | `/diagnostics` | `diagnostic_msgs/DiagnosticArray`, 1 Hz | KL=1 |
| srv | `/openral/estop_reset` | `std_srvs/Trigger` | — |

`/openral/candidate_action` and `/openral/safe_action` deliberately use `KEEP_LAST=50`, not
CLAUDE.md §2's control profile (`KEEP_LAST=1`): several slots (arm, gripper, base) are published
per tick, and `KEEP_LAST=1` would coalesce them (`chunk_qos()` in `lifecycle_kernel.cpp`).

### `/openral/safety_status` — current state, not events (ADR-0096)

The only latched topic. `/openral/estop` and `/openral/failure/safety` are volatile event
streams, so a late subscriber sees nothing. `SafetyStatus` carries the current state (`latched`,
`drop_reason`, `detail`, `rskill_id`, `trace_id`, `header.stamp`), so a dashboard opened
mid-mission or a runner reconnecting after a crash reads the truth immediately. Publication is
transition-gated on `(latched, drop_reason)` and observability only; it changes no decision.

| Path | `latched` | `drop_reason` |
| --- | --- | --- |
| lifecycle activation | current latch | `DROP_NONE` when clear |
| envelope violation | `true` | `KIND_FORCE` / `KIND_WORKSPACE` / `KIND_CONTROLLER` |
| geometric collision | `true` | `KIND_COLLISION` |
| external `/openral/estop` | `true` | `DROP_EXTERNAL_ESTOP` |
| no envelope | `false` | `DROP_ENVELOPE_UNCONFIGURED` |
| voxel / state unavailable or overflow | `false` | `DROP_VOXEL_UNAVAILABLE`, `DROP_STATE_UNAVAILABLE`, `DROP_VOXEL_OVERFLOW` |
| attached payload unverifiable | `false` | `DROP_ATTACHED_UNAVAILABLE`, `DROP_ATTACHED_OVERFLOW` |
| arrived place refused in the advisory band | `false` | `DROP_PLACE_ADVISORY` |
| chunk accepted after a drop, or `/openral/estop_reset` succeeded | `false` | `DROP_NONE` |

Values `101` and `104` are reserved and never reused (retired by ADR-0109; they still appear in
old bags). Rules that make the durable value trustworthy:

1. **Publish on every activation**, so a restarted kernel overwrites a stale sample.
2. **Re-stamp at 1 Hz** on the `/diagnostics` heartbeat, so `header.stamp` is evidence the
   publisher is alive. Consumers treat a status older than their liveness window as *unknown,
   not safe*.

Every non-latching drop also publishes a `FailureTrigger` (`SEVERITY_ABORT`). A drop because an
input was too old or never arrived (`voxel_unavailable`, `voxel_stale`, `peer_voxel_*`,
`attached_unavailable`, `state_unavailable`) is `KIND_PERCEPTION` with `PerceptionStaleEvidence`
(input topic, age, threshold crossed). An overflow, an unstamped grid or an unmapped slot is
`KIND_CONTROLLER`.

### Parameters

Every limit is a ROS parameter; there is no envelope file. The launch loads `robot.yaml` through
Pydantic, synthesises the envelope (`openral_safety.envelope_loader.compute_intersection` +
`kernel_params_from_envelope`) and passes each field as a parameter (`kernel_only.launch.py` for
tests, `packages/openral_rskill_ros/launch/deploy_e2e.launch.py` in a deployment).
**Parameters are fixed once configured**: an `on_set_parameters` guard refuses every change to an
existing parameter while the node is not `UNCONFIGURED`, and for ever after a failed configure or
its first cleanup, so a manifest limit cannot be relaxed over the ROS graph. Launch overrides at
declaration and new declarations (a provider's own) are unaffected. To change a limit, edit
`robot.yaml` and relaunch. Threat model:
[`ros-graph-security.md`](../../docs/reference/ros-graph-security.md).

Key parameters (all declared in the `SafetyKernelLifecycleNode` constructor,
`src/lifecycle_kernel.cpp`; collision-model arrays — `collision_*`, `force_observer_*` — are
lowered from the manifest by the launch and not listed):

| Parameter | Default | Meaning |
| --- | --- | --- |
| `n_dof` | `0` | Joint count; `0` (no envelope) refuses configure |
| `joint_position_min/max`, `joint_velocity_max`, `joint_torque_max` | empty | Per-joint bounds; length must equal `n_dof` |
| `workspace_box_min_xyz` / `_max_xyz` | empty | Cartesian workspace box |
| `max_ee_speed_m_s`, `max_ee_angular_speed_rad_s`, `max_ee_accel_m_s2`, `max_force_n`, `max_torque_nm`, `contact_force_threshold_n`, `max_base_linear_speed_m_s`, `max_base_angular_speed_rad_s`, `max_cartesian_step_m`, `max_cartesian_step_rad` | `+inf` (no-op) | Scalar bounds; `+inf` = undeclared |
| `gripper_ee_names`, `gripper_command_min/max` | empty | Per-end-effector gripper command range |
| `estop_reset_cooldown_s` | `0.5` | Minimum time since the last estop before reset is accepted |
| `chunk_validation_deadline_us` | `1000` | Declared; no check reads it today (unverified use) |
| `self_collision_enabled`, `self_collision_margin_m` | `false`, `0.0` | Self-collision check |
| `world_voxel_enabled`, `world_voxel_margin_m` | `false`, `0.0` | Arm links vs `/openral/world_voxels` |
| `world_voxel_deadline_ms` | `500` | Receipt deadline; configure refuses outside (0, 2000] |
| `world_voxel_data_age_budget_ms` | `1500` | Max age of the world in the grid (from `source_stamp`); refused outside (0, 3000]; `0` is refused, never "off" |
| `world_voxel_max_cells` | `614125` | Lattice cap (85³ at 25 mm); a larger grid is refused |
| `peer_voxel_enabled` | `false` | Peer-grid check; needs `world_voxel_enabled` |
| `peer_voxel_deadline_ms`, `peer_voxel_data_age_budget_ms` | `300`, `300` | Same caps as the world grid |
| `attached_collision_enabled`, `attached_collision_margin_m` | `false`, `0.0` | Carried-payload check |
| `attached_collision_deadline_ms` | `500` | Freshness of `/openral/world_state_fast` |
| `attached_contact_tolerance_m` | `0.001` | Physical slack (FK/pose noise) on an attested support depth |
| `attached_max_objects`, `_max_primitives`, `_max_touch_links` | `8`, `16`, `32` | Payload capacity; over capacity fails closed |
| `support_witness_max_patch_radius_m`, `_max_penetration_m` | `0.5`, `0.01` | Caps on what a support attestation may claim |
| `place_advisory_max_consecutive` | `3` | Advisory refusals before the next latches; `0` disables the band |
| `collision_scale_proximity_m`, `_k`, `_min` | `0.0`, `20.0`, `0.1` | Graded velocity scale; `0` proximity = off |
| `collision_state_deadline_ms` | `200` | Max age of each measured joint position |
| `collision_seed_dt_s`, `collision_predict_lambda`, `collision_predict_margin_growth_m`, `collision_predict_max_steps` | `0.0`, `0.05`, `0.01`, `0` | Velocity/Cartesian prediction (damped-least-squares integration) |
| `grasp_allowance_enabled`, `grasp_contact_links`, `grasp_region_max_age_s` | `false`, empty, `0` | Grasp-target exemption (ADR-0115) |
| `place_region_max_age_s` | `0` | `0` = 2 × `world_voxel_deadline_ms`; resolved bound must be in (0, 4 s] while the matching allowance is on |
| `workspace_contact_enabled` and `workspace_contact_*` | `false` | Hand-only workspace contact (sim-only) |
| `station_motion_translation_m`, `station_motion_rotation_rad` | `0.02`, `0.02` | Base-motion retirement thresholds; refused outside (0, 0.05] |
| `force_observer_enabled`, `joint_external_torque_max` | `false`, empty | External-torque tier |
| `hal_mode` | `real` | `sim` or `real`; a provider loads only on `sim` |
| `exemption_provider`, `exemption_provider_links` | empty | Pluginlib class and relaxable-link mask |

`deploy_e2e.launch.py` reads `OPENRAL_COLLISION_SCALE_PROXIMITY_M` / `_K` / `_MIN` so the velocity
band can be swept without a rebuild. The shipped launches do not pass `station_motion_*`.

**World-voxel freshness is capped in the kernel, not only in the schema.** With
`world_voxel_enabled`, `on_configure` returns FAILURE for a deadline or data-age budget outside
the ranges above, so a kernel started with `ros2 run` cannot run looser than `DeployRuntime`
(hazard log Entries 033/034; `tests/unit/test_perception_caps_mirror.py` pins both sides). A grid
without a `source_stamp` always drops as `voxel_stale`.

## Run it

```bash
colcon build --packages-select openral_safety_kernel
source install/setup.bash
# OPENRAL_VENV_SITE (the venv's site-packages) lets the launch import
# openral_core / openral_safety under the system Python.
ros2 launch openral_safety_kernel kernel_only.launch.py \
    robot_yaml:=robots/so100_follower/robot.yaml
```

## Lifecycle

`unconfigured → configure → activate → deactivate → cleanup → shutdown`

- **configure**: loads the envelope from parameters; refuses to advance when `n_dof` is 0, a
  joint array disagrees with `n_dof`, or a bound is NaN/negative. Opens publishers, subscribers
  and the service; starts the 1 Hz `/diagnostics` timer; loads the exemption provider last.
- **activate**: activates managed publishers so messages flow.
- **deactivate**: stops outbound publication; subscribers keep receiving (the fault latch still
  trips on external estop).
- **cleanup**: releases resources; clears the fault latch and counters.

## Fault latch and recovery

On an envelope or collision violation:

1. Drop the candidate chunk (no republish on `/openral/safe_action`).
2. Publish `FailureTrigger` on `/openral/failure/safety` (`KIND_FORCE | KIND_WORKSPACE |
   KIND_CONTROLLER | KIND_COLLISION`, `SEVERITY_ABORT`, Pydantic-deserialisable
   `evidence_json`, `rskill_id` and `trace_id` from the chunk).
3. Publish `std_msgs/Empty` on `/openral/estop`.
4. Publish the latched `SafetyStatus` (`latched=true`, the same kind).
5. Set `fault_latch=true`. All further candidates drop with `estop_latched`.

An external `/openral/estop` runs only steps 4–5 (`DROP_EXTERNAL_ESTOP`). Recovery is manual:
`ros2 service call /openral/estop_reset std_srvs/srv/Trigger`. The service refuses until
`estop_reset_cooldown_s` has passed since the last estop publish. `ROSEStopRequested` is never
auto-cleared (CLAUDE.md §3 "Safety"). The kernel has no flag that disables a check and no debug
path that bypasses E-stop.

Non-latching drops (unavailable inputs, advisory refusal) publish a `FailureTrigger` but no
E-stop; motion resumes when a fresh input lands.

## Checks

### Per-control-mode magnitude bounds

`validate()` bounds every row of the chunk's horizon. A per-mode chunk's row stride is its own
`n_dof` (6 twist/delta, 7 pose, 1 scalar), never the robot's joint count. Bounds come from the
manifest's `safety:` block; an undeclared bound is `+inf`, so the check is a no-op for it.

| Mode | Bound | Manifest field | Evidence |
|---|---|---|---|
| `JOINT_POSITION` / `JOINT_TRAJECTORY` | per-joint position range | `joints[].position_limits` | `workspace` |
| `JOINT_VELOCITY` / `JOINT_TORQUE` | per-joint cap | `velocity_limit × max_joint_speed_factor` / `effort_limit`, `max_torque_nm` | `workspace` / `force` |
| `CARTESIAN_POSE` | position inside the workspace box | `workspace_box_*` | `workspace` |
| `CARTESIAN_TWIST` | `‖v‖`, `‖ω‖` | `max_ee_speed_m_s`, `max_ee_angular_speed_rad_s` | `force` |
| `BODY_TWIST` | `‖v‖`, `‖ω‖` | `max_base_linear_speed_m_s`, `max_base_angular_speed_rad_s` | `force` |
| `CARTESIAN_DELTA` | `‖Δxyz‖`, `‖Δrot‖` after `clip(raw,−1,1)·cartesian_delta_scale` | `max_cartesian_step_m`, `max_cartesian_step_rad` | `force` |
| `COMPOSITE_MODE` | flag in `[−1, 1]` | — | `workspace` |
| `GRIPPER_BINARY` / `GRIPPER_POSITION` | value inside the addressed end effector's command range; a chunk resolving to no declared end effector is refused (`gripper_unresolved`) | `end_effectors[].command_convention` / `command_range`, or the sim scene's `gripper_convention` | `workspace` / `controller` |

The norms are Euclidean, so a controller with a per-axis output limit `a` needs a bound of at
least `a·√3`. Gripper bounds are per end effector, in its own encoding (`[0,1]` fraction,
robosuite `[−1,1]`, jaw radians); the kernel resolves each chunk by `ee_name` (an unnamed chunk
binds to the robot's only gripper). Matching the skill's encoding is the loader's job
(`rSkill.check_gripper_conventions`). `tests/sim/safety/test_kernel_per_mode_envelopes.py` drives
the real node from the real `r1pro`, `panda_mobile` and `openarm` manifests.

### Collision model, frame and joint state

Geometric checks forward-kinematics each link from the manifest's per-joint origins starting at
`base_frame` (ADR-0095). The kernel applies **no transforms**: the world grid's `frame_id` and
`origin` are taken at face value, so the grid and the FK root must be measured against the same
body. A half-applied frame change blinds the kernel by the offset; the producer is fixed at
source (hazard HZ-0095-1).

`JOINT_TRAJECTORY` is collision-checked exactly like `JOINT_POSITION`. Velocity and Cartesian
chunks are reconstructed from the measured state (reactive check, plus predictive integration
when `collision_predict_max_steps` > 0). The kernel reads `/joint_states` and folds it by
`collision_joint_names`; the stream must spell joints the manifest's way, so `deploy_e2e` remaps
the kernel's `/joint_states` to the topic the runner acts on.

**Slot rows (ADR-0102).** A multi-slot rSkill sends each `JOINT_POSITION` arm slot zero-padded,
naming its joints in `ActionChunk.joint_names`. A row that leaves FK dofs uncommanded is checked
(1) with the other joints at their **measured** value, from a fresh, complete state, fresh *per
joint* within `collision_state_deadline_ms` (else dropped `state_unavailable`, not latched; an
unplaceable name drops `slot_joints_unmapped`); (2) again with earlier accepted slots of the same
`(runner_session_id, tick_index)` at their targets; (3) for an ungrouped slot
(`tick_group_size <= 1`), also as published. Checks 2 and 3 only add refusals.

**Mimic joints and gripper curves.** A `<mimic>` follower finger is its own link
(`collision_joint_kind` 3/4 on the leader's dof column, `collision_mimic_multiplier/offset`), and a
`role: gripper` joint with `urdf_curve` is lowered to `collision_curve_*`. Configure refuses
malformed mimic or curve arrays. Design: [`finger-mimic-geometry.md`](../../docs/reference/finger-mimic-geometry.md).

### Self-collision

`check_self_collision` checks link vs link over the horizon, with an allowed-pair matrix
(`collision_allowed_pairs`). Where **both** links ship a stage-2 hull, a pair the boxes cannot
clear is re-asked with the exact hulls (`hull_hull_distance`); the box stays broad phase and
fallback. When GJK proves overlap it cannot size it, so the box bound is reported and
`CollisionHit::depth_is_box_bound` is set. That flag is disclosure only and never a gate.

### World-voxel check: staged hull narrow phase

Arm links vs occupied cells of `/openral/world_voxels`, at `world_voxel_margin_m`. For a link whose
manifest declares `tight_geometry` the narrow phase is staged:

1. **Stage 1**, every occupied cell: 16-axis separating-axis lower bound against the link's 26-DOP.
2. **Stage 2**, only cells stage 1 cannot clear: GJK on the exact convex hull.
3. `box_box_distance` is folded in (the maximum of the two lower bounds) for any cell about to
   stop the robot, so **the staged path can never stop where the shipped box path would not**.

The broad-phase window is sized from the box alone; `hull ⊆ 26-DOP ⊆ OBB` is proved at
`on_configure` by `validate_tight_geometry`, fail-closed. Bounds that fail conservative:
`kGjkMaxIterations` 24, `kMaxStage2PerCheck` 32 (shared per call; falls back to stage 1 + box),
`kMaxTightHullVertices` 320 (that link runs stage 1 only). Capsule-lowered robots and links with
no tight geometry keep the primitive path. Design and measurements:
[`collision-hull-narrow-phase.md`](../../docs/reference/collision-hull-narrow-phase.md);
geometry generation: [`collision-geometry-review.md`](../../docs/reference/collision-geometry-review.md).

### Peer voxels (ADR-0137)

The other robots of a workcell arrive as a second grid on `/openral/peer_voxels`
(`peer_voxel_enabled`; requires `world_voxel_enabled` and the same frame and cell cap). The robot
and anything it carries are checked against it at `world_voxel_margin_m` with **no exemption of any
kind**: the peer grid has no grasp or place region and no provider is consulted, so a peer cell
trips wherever it is, even inside a grasp region. Links no chunk moves (fixed to the root or moved
only by base dofs) are skipped; base clearance is Nav2's. The grid is never assumed: missing,
overflowing, late (`peer_voxel_deadline_ms`), stale or from the future beyond
`peer_voxel_data_age_budget_ms` (either sign), or in another frame than the world grid drops the
chunk (`peer_voxel_unavailable|overflow|stale|from_future|frame`, non-latching). Configure refuses
the deadline or budget outside the world grid's caps. Evidence names the cell `peer_voxel_<n>`.

### Distance-graded velocity scaling — off by default

An **accepted** chunk near an obstacle can be republished slower.
`collision_scale_proximity_m` (`0.0` = off, reproducing the unscaled republish exactly),
`collision_scale_k` (20), `collision_scale_min` (0.1). The scale is `exp(k·(slack − proximity))`
clamped to `[collision_scale_min, 1]`, where `slack` is the smallest clearance above its own gate
margin anywhere in the sweep. The band also widens the voxel broad-phase window (never the trip
margin).

- **It never changes a verdict.** It is applied after acceptance and cannot turn an accept into a
  drop or the reverse; scaling a rate down preserves every bound the chunk satisfied.
- **Only rate-shaped modes are scaled** (`JOINT_VELOCITY`, `CARTESIAN_DELTA`, `CARTESIAN_TWIST`).
  Absolute targets, gripper, composite and `BODY_TWIST` rows are never scaled.
- **Exempted pairs do not slow the robot**: a pair below its margin that did not trip is ignored.
  A payload resting against a surface still sees its neighbouring cells, so it runs reduced.
- It is disclosed by the `safety.velocity_scale` / `safety.scale_slack_m` span attributes, a
  transition-gated `safety.collision_scaled` log line and a `scaled` count on `/diagnostics`.

**Operator precondition, not enforced by the kernel:** the kernel does not know which policy
produced a chunk, and a scaled chunk changes what a policy that observes joint velocity sees on
the next tick. Do not arm the band for the `xvla` model family (it feeds `qvel`). Multi-frame pose
histories (`xr1`, `rldx`) encode velocity implicitly, and that coupling is not audited. It ships
disabled; a non-zero band is earned by a five-round A/B battery per arm. Background:
[`collision-safety-alternatives-survey.md`](../../docs/reference/collision-safety-alternatives-survey.md).

### Attached payload (ADR-0092)

A grasped payload on `/openral/world_state_fast` is checked as robot geometry against the world
(and peer) grid and against the robot's own links (except the attach link and touch links). An
unverifiable set (never received, stale, over the `attached_max_*` caps, unknown link, non-finite
pose, malformed) drops chunks fail-closed (`DROP_ATTACHED_UNAVAILABLE` / `_OVERFLOW`). The kernel
exempts exactly two things inside `check_attached_voxel_collision`; an exempted cell reaches
`sweep_min_distance` only and **can never be the cell an E-stop names**.

**1. Support-contact witness (ADR-0092 D6).** World State attests, at attach, one support contact
(identity, contact point and normal in the object frame, patch radius, physical depth). The kernel
exempts a cell only if it is inside the patch laterally *and* no higher above the attested support
plane than `half_resolution·(|n.x|+|n.y|+|n.z|) + attested depth + slack(1 mm) + resolution`. That
accounts exactly for voxel-cube overshoot of the real surface and one co-planar voxel of adjacent
structure (height only, never reach). Solid above that bound, deepening into the support, or
contact against any surface the witness does not name still stops the robot.

- The witness is **latched and dies on separation**: it stays live only while some *occupied* cell
  it would exempt is within margin; it re-arms only on a new attestation keyed
  `(object_id, support_id, stamp_ns)`. Lift and re-contact, regrasp, and contact elsewhere are new
  violations. No attestation (the vision producer's honest output) means no exemption.
- Caps: `support_witness_max_patch_radius_m` 0.5 and `support_witness_max_penetration_m` 0.01; a
  larger claim fails the whole attachment message closed.
- `openral_octomap_bridge`'s payload clearing and this predicate are a deliberate mirror
  (`withheld ⊆ exempt`): every cell near the payload is either cleared there or exempted here,
  never neither and never both. They must move together
  ([`packages/openral_octomap_bridge/README.md`](../../packages/openral_octomap_bridge/README.md)).
- **Place (ADR-0097/0098).** A typed `PlaceDeclaration` (dispatch → HAL → World State) makes the
  *producer* attest a second witness for measured contact on the declared target. The kernel has
  no place-specific exemption. It gains one input: `VoxelGrid::place_region`, an oriented box from
  the producer (sim only; no allowance is applied on real hardware today). Inside it, for the
  declared payload only, cells are gated at `margin − min(1.5 × resolution, 40 mm)` (an absolute
  cap, `kMaxPlaceApproachAllowanceM`). **This is a margin change, not an exemption**: a deeper cell
  still stops the robot and the reported distance is the true one (`place_allowance_active`
  discloses it). A malformed region (non-finite, > 1.5 m a side, > 8 m³, wrong frame, older than
  `place_region_max_age_s`) grants nothing. It dies with the declaration (retraction, timeout,
  detach, failed ingest, re-framed grid). With optional declared-target `geometry` (≤ 64
  primitives, `kMaxPlaceTargetPrimitives`) the pair is adjudicated against the declared body at
  zero clearance, only while `d_target <= d_cell + allowance`; a malformed shape refuses the whole
  region.
- **Advisory refusal (arrived place).** A payload-only hit inside a declared region and no deeper
  than `min(0.2 × voxel, 5 mm)` past the allowance is *refused without latching or E-stop*
  (`DROP_PLACE_ADVISORY`; the action is still dropped). At most `place_advisory_max_consecutive`
  (3) in a row, then it latches; any accepted chunk resets the run. A robot link is never
  advisory; severity outranks depth when folding pairs. `0` restores the latching behaviour.

**2. Embedded attach-time residue.** The payload's own occupancy already in the map at attach (a
cell at least half a voxel inside the payload). The attach-time snapshot is retaken on an
`attachment_revision` change only when the set of carried objects changed, or a provider is
loaded; a carried payload keeps its own baseline when another payload joins or leaves. In the
open kernel a tracker growing a payload's geometry cannot widen the residue allowance. There is no
index-keyed non-deepening baseline and no `attached_contact_allow_new_shallow`.
`attached_contact_tolerance_m` (1 mm) is physical slack only.

Staged hulls also apply to a carried payload that ships a proved-contained refinement
(`validate_tight_hull`; a refinement that cannot be proved is *dropped* and the primitive checked as
its plain box, never truncated). `check_attached_self_collision` and the place-target adjudication
keep reading the shipped box.

### Grasp-target exemption (ADR-0115) — default off

Off unless `grasp_allowance_enabled`; with it false the region is never set and the kernel
behaves as without the feature. Draft for Safety-WG review. A gripper finger contains the grasp
target before any attachment can exist, so a producer-measured `GraspDeclaration` (on
`/openral/world_state_fast`) arms a base-frame oriented box. While valid, a cell whose centre is
inside the box does not trip **for a link in the exempt mask only**. The mask is the manifest's
gripper links and mimic followers (`grasp_contact_links`); enabled with an empty or unknown
allowlist, configure fails. Every other link, cells outside the box, self-collision, the attached
checks and the force gate are unchanged. An exempt pair never supplies the reported identity or
distance and reaches `sweep_min_distance` at its own depth.

Bounds and fail-closed rules:
- Region: finite, unit-quaternion pose, half-extents ≤ `kMaxGraspRegionHalfExtentM` 0.20 m, volume
  ≤ `kMaxGraspRegionVolumeM3` 0.03 m³, empty `geometry`, in the grid's frame, backstop `timeout_s`
  in (0, 120 s]. Caps are Safety-WG placeholders. A refusal means no exemption, with its own
  `reason=`.
- Every `contact_links` entry must be allowlisted and all must belong to **one hand**
  (`links_span_hands` refuses otherwise).
- Freshness: the region's `stamp_ns` must be within `grasp_region_max_age_s` (default `0` = 2 ×
  `world_voxel_deadline_ms`; configure refuses outside (0, 4 s]), and the world-state stream
  within `attached_collision_deadline_ms`. Stale means no exemption, never a drop by itself.
- Handover: only a payload attached on the declaring gripper's chain can be the handover. The
  region **latches at the handover edge** and a re-measured region cannot extend it; a mismatched
  object, hand or latch retires it. While handed over the exemption lives only while the payload
  origin stays inside the box.
- Retirement: retraction, timeout, detach, release, a rejected attachment set or a grid-frame
  change retire the identity, and a retired identity never re-arms; only a new declaration can
  (a fixed ring of `kGraspRetiredCapacity` 16, allocation-free).
- Tracked objects (`TrackedObjectSet`, `kMaxTrackedObjects` 32): a malformed set (wrong frame, oversize box or hull, duplicate id, going-back revision, future stamp) is refused whole. A `tracked_object` attachment is "at the hand" only if it names a live, non-`provisional` track whose box meets the carrying links within one voxel at the measured configuration; otherwise it is still checked as a payload but arms no witness and is never a handover. Tracked geometry is a monotone union (never shrinks, <= 16 primitives), so staleness can cost a false stop, never a missed one.
- **Release of a tracked payload.** The release *decision* belongs to the exemption provider.
  Without one the open kernel never releases a payload; it rides the hand until the producer
  detaches it. After a provider releases it, the kernel freezes it in the base frame and opens a
  window in which only the release hand and touch links are exempt against it, closed by
  separation > `tracked_release_clear_m` (0.04, ≤ 0.1) or `tracked_release_timeout_s` (3.0, ≤ 10).
  That decision is not monotone (a payload still in the hand is carried unmodelled until the map
  re-marks it, hazard OT-7) and is therefore Pro-side.
- Disclosure: `/diagnostics` key `grasp_region`, span attribute `safety.grasp_exemption_active`,
  and `grasp_exemption_active=` / `grasp_target=` / `grasp_exemption_source=` (`region`,
  `provider:<name>`, `none`) on every `safety.collision` stop line.

Spec: [`real-pick-place-design.md`](../../docs/reference/real-pick-place-design.md) §2.1; drafts:
[`real-pick-place-adr-drafts.md`](../../docs/reference/real-pick-place-adr-drafts.md).

### Manipulation-workspace contact — default off, sim-only

Off unless `workspace_contact_enabled`; `openral deploy run` and a non-sim `deploy_e2e` refuse it.
Draft for Safety-WG review (real-hardware values are a WG decision, issue #374). A learned pick in
a packed tote touches neighbouring cartons with the hand only. After the grasp-target exemption, a
(link, occupied cell, row) pair does **not** trip iff all hold:

1. the link is a **hand link** (`workspace_contact_links`, derived from the manifest by the launch);
2. the cell centre is inside the live **workspace box** (`GraspDeclaration.workspace_region`,
   confirmed by the HAL against a fresh grid, capped at half-extent 0.50 m / volume 0.25 m³,
   confirmed no longer ago than `grasp_region_max_age_s`);
3. the **hand is slow**: `hand_speed_mps ≤ workspace_contact_max_speed_mps`, measured per candidate
   from FK at the measured state and `/joint_states` velocity (clamped to the joint's own
   Δposition/Δstamp), falling back to the (faster) commanded speed unless every FK dof has a fresh
   measured velocity;
4. the **hand does not lead far**: `hand_lead_m ≤ workspace_contact_max_lead_m` (≤
   `kMaxWorkspaceContactLeadM` 0.10 m, a WG placeholder), jaw joints held at measured.

A carried payload's cells inside the workspace are exempt by the same predicate on its attach
link. **Fail-closed:** feature off; no live declaration or workspace; wrong frame, geometry or
non-unit quaternion; over the caps; stale workspace or world-state stream; stale or incomplete
measured state; any mode other than `JOINT_POSITION`; speed or lead above bound or non-finite.
Arm links, cells outside the box, self-collision and the attached checks are untouched. Configure
refuses enabling it without `grasp_allowance_enabled`, a non-finite or non-positive speed or
period, a lead not in (0, 0.10], an empty link list or an unknown link.

**Workspace brake** (`workspace_contact_brake_enabled`): when a `JOINT_POSITION` chunk's *only*
reasons to stop are the measured hand speed and/or lead, the kernel publishes a hold (non-jaw
joints at a hold anchor, falling back to measured) or a creep toward the command instead of an
E-stop; `workspace_contact_hold_max_consecutive` > 0 bounds consecutive holds, then the latched
stop. Every hold is re-checked by every open check.

Disclosure is never silent: `safety.workspace_contact` WARN (≤ 1 Hz per link), armed/dropped/
rejected logs, span attributes `safety.workspace_*`, and workspace fields on the stop line. Design
and calibration: [`manipulation-workspace-contact.md`](../../docs/reference/manipulation-workspace-contact.md).

### Station mode: base motion retires every base-frame allowance

On a mobile manipulator (`collision_base_dofs` non-empty) every allowance (grasp region, place
region, workspace, tracked grasp key, open release window) is declared in the base frame and valid
only at the pose it was declared at. `update_station` (from `/joint_states`) anchors the base dofs;
once the base leaves the anchor by more than `station_motion_translation_m` (planar) or
`station_motion_rotation_rad` (per revolute dof, wrapped), every allowance is retired
(`reason=base_moving`), release windows close, and any declaration stamped at or before the motion
is refused (`base_moved`). Only a fresh declaration re-arms. The anchor re-anchors when the first
allowance goes live; a clock that runs backwards retires everything (`clock_reset`). Fixed-base
robots never run the rule. Thresholds must be in (0, 0.05]; tunable, never off. Sim-only pending
Safety-WG review.

### Contact-force gate (ADR-0100)

Position checks cannot tell "set down" from "crush". `check_contact_force_gate` **only ever adds a
refusal**: it runs after the self, world and attached checks have passed, has no way to widen a
margin or create an exemption, and a measured force never licenses contact the geometry refused.
It arms only when all hold, each failing toward geometry-only behaviour:
a live `PlaceDeclaration` with `contact_force_threshold_n` > 0 (≤ `kMaxContactForceThresholdN`
140 N, ISO/TS 15066 Table A.2); the declared payload; a witness on the declared target; a fresh
(non-future, within the declaration's lifetime and `attached_collision_deadline_ms`), **calibrated**
magnitude (`contact_force_calibrated` and a named calibration; uncalibrated values are never read as
newtons). The effective threshold is the smaller of the declaration's and the envelope's
`contact_force_threshold_n`; a task can lower the ceiling, not raise it. The comparison is
inclusive and the deepest overrun is reported. No calibrated producer exists today (sim helper
`probe_contact_force` has no runtime caller), so the gate is dormant and geometry stays
authoritative. `/joint_states.effort` can never arm it.

### External joint-torque observer (issue #373)

`force_observer.{hpp,cpp}`: a generalized-momentum observer over the collision model's kinematic
tree, fed by `/joint_states` position, velocity and effort, estimates the external torque per
joint. Over a per-joint ceiling `joint_external_torque_max` (from
`SafetyEnvelope.max_external_joint_torque_nm`) the kernel latches with `FailureTrigger(KIND_FORCE)`
and `ForceEvidence{unit:"Nm"}` and fires `/openral/estop`. It only adds stops: a stale or missing
input, or a monitored joint inside its limit margin, resets the estimate and stops nothing. A
ceiling with no model fails configure. No shipped manifest or scene sets a ceiling yet; the real
OpenArm publishes the estimate for disclosure only. Design:
[`external-torque-observer.md`](../../docs/reference/external-torque-observer.md).

## Exemption provider (ADR-0121)

Default none, sim-only. The open kernel keeps every check that can only refuse motion. A rule that
*relaxes* a verdict may live in a private provider (OpenRAL Pro) that plugs in through
[`exemption_provider.hpp`](include/openral_safety_kernel/exemption_provider.hpp) and a `pluginlib`
class deriving `openral_safety_kernel::ExemptionProvider`. **No provider ships in this repo, and
real providers are OpenRAL Pro.** The tracker-based rules (tracked grasp exemption, release
decision, explained cells, contact hold, payload own cells) exist only in the provider; the open
kernel has none of them and a stale launch parameter for one (`tracked_grasp_exemption_enabled`,
`workspace_explained_cells_enabled`, `payload_own_cells_enabled`,
`pinch_payload_own_cells_enabled`, `tracked_release_jaw_joints`) **refuses configure**, because
running on without a requested rule would be silent. A payload on its own measured cell stops in
open.

- **Loading** happens once, at configure, after the collision model, and **never fails configure**.
  `exemption_provider` empty (default) = no provider. A provider is refused (ERROR
  `safety.exemption_provider refused`, `/diagnostics exemption_provider=refused: <why>`, kernel
  unchanged) on: `hal_mode` other than `sim`; an empty or unknown `exemption_provider_links`
  mask; a pluginlib load failure; `api_version()` other than `kExemptionProviderApiVersion` (5); a
  throwing or false `initialize`; or an unsound `workspace_contact_config()`.
- **Fail closed.** Providers are consulted only after every open check and open exemption, only for
  a pair that would otherwise trip, only with **every robot link of the pair** in the mask, and for
  a payload only while carried (never a frozen one). A tolerated pair does not trip, reaches
  `sweep_min_distance` at its own depth, is counted in `CollisionHit::provider_pairs` and logged
  (`safety.exemption_provider_contact`, ≤ 1 Hz). The limits, the envelope, the lead bound and the
  watchdog are never offered. Hooks are `noexcept` and must not allocate.
- **Read-only view.** Each candidate gets a preallocated `CandidateView` (tracked set, FK at the
  measured state, the kernel's own ADR-0115 and #349 verdicts quoted as data, hands, freshness,
  grid sequence). The provider reads verdicts; ingest of frames, ages and retirement stays in the
  kernel. API v5 quotes the #349 verdict (`workspace_exempt`) so a provider never re-derives it.
- **A loaded provider takes over** #349's application (its in-tree switch off) and, if
  `replaces_grasp_region()`, the grasp exemption source. Without it the open ADR-0115 region applies.
- **Stricter, never looser, under a provider:** a refined box payload is checked as its shipped
  padded box, and row 0's hand speed reads the measured state.


## Fail-closed inputs

NaN fails every ordered comparison, so the kernel refuses non-finite geometry at every input and
treats any NaN that reaches a distance as a hard collision (`fold_pair`).

- **Measured state:** a non-finite `/joint_states` position is no measurement.
- **Attachments:** a non-finite `pose_in_link` or `pose_in_object` refuses the whole message
  (`safety.attachment_fail_closed reason=non_finite_pose`, `DROP_ATTACHED_UNAVAILABLE`).
- **World and peer voxels:** a grid whose size overflows or exceeds `world_voxel_max_cells`, a
  non-finite or non-positive `resolution`, or a non-finite `origin` is `voxel_overflow`.
- **Stamps:** a negative `/joint_states` stamp reads as unstamped and a negative `source_stamp` as
  unknown (refused by the data-age gate), instead of throwing out of the executor.

## What the kernel does NOT check: base motion

The kernel checks the **arm's link geometry and the attached payload** (and peers) against the
voxel grids. It does not check where the base drives; Nav2's 2-D costmap does, against its own
world model (ADR-0040; boundary table in
[`openral_nav2_bringup`](../../packages/openral_nav2_bringup/README.md)). `/cmd_vel` never reaches
`/openral/candidate_action`: `MobileBaseBridge` applies Nav2's `Twist` through the HAL directly, and
Nav2's `velocity_smoother` caps it. (A `BODY_TWIST` chunk a policy publishes *is* bounded here.) The
kernel's single authority over base motion is the **E-stop latch**: the bridge drops every
`/cmd_vel` while the HAL node is latched. A carried payload is not in Nav2's footprint (ADR-0099);
`payload_scan_filter_node` only drops its returns from the scan.

## Observability

One OTel `safety.check` span per candidate chunk over OTLP/HTTP (the dashboard ingests on
`/v1/traces`, default port 4318; endpoint from `OTEL_EXPORTER_OTLP_ENDPOINT`, falling back to
`http://localhost:4318`). The W3C `traceparent` in `ActionChunk.trace_id` is the parent context.
`BatchSpanProcessor` ferries spans off the chunk thread so the validator stays allocation-free.

| Attribute | Value |
| --- | --- |
| `safety.check_name` | `"envelope"` |
| `safety.kernel` | `"cpp"` |
| `safety.severity` | `"info"` (pass), `"warn"` (latched / unconfigured), `"violation"` |
| `safety.drop_reason` | `estop_latched`, `envelope_unconfigured`, `force`, `workspace`, `controller`, `collision`, `collision_advisory`, or an input-unavailable reason (`state_unavailable`, `slot_joints_unmapped`, `voxel_unavailable`, `voxel_overflow`, `voxel_stale`, `peer_voxel_*`, `attached_unavailable`, `attached_overflow`) |
| `safety.violation_{reason,joint,value,limit}` | on `violation` |
| `safety.sweep_min_distance_m` | collision stops only |
| `rskill.id` | `ActionChunk.rskill_id` |

A violation also fires an `openral.event.safety_violation` event.

**Collision evidence is one pair, one distance.** `CollisionEvidence.link_a`, `link_b_or_object` and
`min_distance_m` describe the **same** pair, the deepest that actually tripped; never the
sweep-wide minimum. That minimum is reported separately (`sweep_min_distance_m`, always ≤
`min_distance_m`); a large gap means something deep is being tolerated on purpose (an exempted
payload contact). The payload also carries `joint_positions_rad`, the configuration FK ran on for
the reported `horizon_step` (for a predicted step, the kernel's own integration of the chunk), all
numbers serialized at `max_digits10`. `FailureTrigger.evidence_json` may also carry joint names,
`workspace_denial_reason`, workspace/grasp flags, `depth_is_box_bound` and bounded payload
provenance. These fields report the check; they neither authorize contact nor identify the physical
object behind an anonymous voxel.

## Real-time guarantees

- `validate()` is allocation-free, pinned in CI by `test_no_alloc.cpp` via a counting global
  `operator new`; the staged narrow phase (stack-only GJK simplex) and the provider query run under
  the same counter.
- C++17, no exceptions across the kernel boundary (`Result<void, Violation>`).
- No real-time scheduling knobs: no `SCHED_FIFO`, no CPU pinning; policy and affinity are whatever
  the launching process inherits.

## Build and test

```bash
colcon build --packages-select openral_safety_kernel
just safety-kernel-test
```

Three tiers, all driving the real `safety_kernel_node` (no mocks):

- **C++ unit** (`test/`): `test_validator.cpp`, `test_collision.cpp`, `test_force_observer.cpp`,
  `test_lifecycle_kernel.cpp`, `test_force_observer_kernel.cpp`,
  `test_exemption_provider_kernel.cpp`, and the allocation pin `test_no_alloc.cpp`.
  `LifecycleKernelTest` sets `ROS_AUTOMATIC_DISCOVERY_RANGE=OFF` so no outside participant (another
  run, a live `openral deploy sim`) can assert `/openral/estop` into the node under test; keep it.
- **Launch** (`test/launch/`): `test_e2e_passthrough.py`, `test_e2e_violation.py`,
  `test_e2e_otel.py`.
- **Sim** (`tests/sim/safety/`): the kernel subprocess gated against a MuJoCo oracle (envelope in
  `test_kernel_with_*_twin.py`, collision in `test_kernel_*_collision*.py`).

Geometric collision arms automatically in `openral deploy sim` / `deploy run`: the launch lowers the
robot's collision model (`openral_safety.mjcf_lowering` preferred, manifest fallback) and the kernel
logs `self-collision check enabled: N links`. Joints are matched by movable-joint order, not name.

## Related

- Design history and measurements: [Investigations and evidence](../../docs/reference/investigations.md)
  (hull narrow phase, geometry review, finger mimic, torque observer, workspace contact, validation
  ledger [`collision-validation-evidence.md`](../../docs/reference/collision-validation-evidence.md)).
- `cpp/opentelemetry_cpp_vendor` — vendor package for `opentelemetry-cpp` at colcon-build time.
- [`packages/openral_nav2_bringup/README.md`](../../packages/openral_nav2_bringup/README.md) — Nav2's
  base-only footprint, the `/cmd_vel` boundary and the payload scan filter.
- [`packages/openral_safety/README.md`](../../packages/openral_safety/README.md) — Python envelope
  loader, collision lowering and the pass-through supervisor.
- `python/observability/src/openral_observability/dashboard/store.py` — consumes `safety.check` spans.
