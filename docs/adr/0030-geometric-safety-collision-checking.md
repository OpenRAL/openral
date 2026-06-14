# ADR-0030: Geometric safety — self- and world-collision checking in the kernel

- Status: **Proposed**
- Date: 2026-05-29
- Related: [ADR-0020](0020-cpp-safety-kernel.md) (the C++ kernel and its
  allocation-free `validate()` contract this ADR extends);
  [ADR-0018](0018-ros2-reasoner-supervisor.md) §5 (the chunk-rate topic
  contract); [ADR-0024](0024-ros-wrapped-rskills.md) (wrapped MoveIt/Nav2
  skills — the planning-time collision checks this ADR does **not**
  replace); CLAUDE.md §1.1 (safety beats helpfulness), §1.4 (explicit /
  reject-not-clamp), §1.5 (Python proposes, C++ disposes; the hot path is
  C++ and bounded), §1.9 (license lineage), §3 Layer 0 / 2 / 6, §3 Safety
  (safety-WG review + hazard log).

## Context

The C++ safety kernel (`cpp/openral_safety_kernel/`, ADR-0020) is the single
gate every actuation path crosses: an rSkill emits an `Action`, the skill
runner publishes it as `ActionChunk` on `/openral/candidate_action`, the
kernel's `validate()` either republishes it on `/openral/safe_action` or
drops it with an E-stop + `FailureTrigger`. The HAL only ever drives motors
from `/openral/safe_action`.

That gate is **kinematics-agnostic**. `validate()`
(`src/validator.cpp`) and the `EnvelopeIntersection` it consumes
(`include/openral_safety_kernel/envelope.hpp`) carry only **scalar bounds**:
per-joint position/velocity/torque limits, a Cartesian AABB workspace box,
EE speed/force caps, plus shape and NaN/Inf checks. There is **no robot
geometry, no forward kinematics, and no world model** anywhere in the kernel.

Consequently the kernel cannot detect either failure that defines "safe
motion" geometrically:

- a **self-collision** — two links intersecting while every joint is
  individually inside its limits; and
- a **world collision** — a link or the EE entering an obstacle that happens
  to sit inside the (necessarily coarse) AABB workspace box.

Today the *only* collision checking in the system lives inside the
`rskill-moveit-joints` rSkill (the MoveGroup wrapper; `openral-moveit-plan-arm`
before its ADR-0054 rename), whose own manifest is explicit: *"the
OpenRAL safety supervisor does NOT do collision checking today … we trust
MoveIt's pass."* That covers **only motions MoveIt itself planned**. A VLA
policy emitting action chunks, a teleop stream, a `cartesian_delta` jog, or a
learned navigation skill produces motion MoveIt never sees and therefore
never checks. Mobile-base collision is handled separately by the Nav2 costmap
(`inflation_radius` derived from `RobotDescription.footprint_radius`), on a
`/cmd_vel` path that bypasses the kernel entirely. The repository ships no
URDF and no SRDF.

The requirement — *validate **any** movement produced by **any** rSkill
against the robot itself and the world* — is therefore a property of **where
the gate sits**, not of which geometry engine runs. The gate must be the one
mandatory checkpoint every path already crosses: the kernel.

## Decision

### 1. Collision validation is a first-class check inside the C++ kernel

Self- and world-collision checking become part of the kernel's `validate()`
path, using **convex analytic primitives**: each robot link is modelled as a
**capsule or sphere**; the world (later phase) as a bounded set of primitives
and/or a fixed-capacity occupancy grid. On a collision the kernel **rejects**
— drops the chunk, publishes `/openral/estop`, and emits a
`FailureTrigger(KIND_COLLISION)` — exactly like every other violation today.
It **never clamps** to a nearby collision-free pose (§1.4; clamping is a
future ADR).

This keeps the universal-gate property (every path is checked) and routes
collision faults through the same flag-and-trace machinery (`FailureTrigger`,
E-stop latch, OTel `safety.check` span) that operators already consume.

### 2. Why primitives, and why this stays real-time

The kernel's defining guarantee is that `validate()` is **allocation-free and
time-bounded** — pinned in CI by `test/test_no_alloc.cpp` (a counting
`operator new` that must observe zero allocations across 10,000 calls) and
run under a single-threaded executor inside the chunk deadline
(`src/safety_kernel_node.cpp`).

Mesh / BVH collision (FCL, MoveIt's `CollisionEnv`, Bullet) heap-allocates and
is not time-bounded; it **cannot** live on this path. Convex analytic
primitives **can**:

- **Geometry** is loaded once at `on_configure` into pre-sized buffers — the
  same pattern the joint-limit vectors already use — so nothing allocates in
  the hot loop.
- **Forward kinematics** over the chunk is a chain of fixed-size 4×4
  homogeneous transforms (fixed-size `Eigen`, stack-allocated; no heap).
- **Self-collision** is closed-form capsule–capsule (segment-to-segment)
  distance against a per-pair clearance; the pair set is fixed at configure
  from the allowed-collision matrix and capped at a compile-time maximum.
- **Swept / continuous** checking is a fixed substep count between chunk rows.

For a 16-DOF arm (≈16 capsules, horizon 16, a few substeps) this is on the
order of tens to low-hundreds of microseconds of pure floating-point work —
comfortably inside a 1 ms budget even at 1 kHz. The `test_no_alloc.cpp`
harness extends directly to cover the collision path.

Primitives are a **conservative over-approximation**: a capsule bounds the
true link, so the check is safe (never misses a real collision of the bounded
volume) but may occasionally false-positive near an obstacle. That is the
correct bias for a safety gate. Mesh-*accurate* collision — needed for
threading an EE through tight geometry — remains a **planning-layer** concern
(MoveIt stays the planning-time checker for arm skills that route through it);
it is explicitly not a safety guarantee.

### 3. Geometry source — pluggable adapters lowered to one form

The kernel never parses URDF / SRDF / MJCF. A Python launch/offline
**lowering tool** (out of scope for the first PR; see Phasing) produces the
kernel's primitive parameters from whichever source a robot has:

- **MJCF** (`SimDescription.mjcf_uri`; the default for sim-first robots) —
  already carries native **capsule geoms** and a built-in allowed-collision
  matrix (`contype` / `conaffinity`, `<contact><exclude>`). No mesh reduction.
- **URDF + SRDF** (real robots that ship them) — `RobotDescription.urdf_path`
  already exists; this ADR adds **`srdf_path`**. The SRDF `disable_collisions`
  block is **consumed directly as the allowed-collision matrix** (it is
  exactly that, and is auto-generated by MoveIt's setup assistant). URDF
  `<collision>` is usually meshes, so a **mesh→capsule reduction** step is
  required regardless of source.
- **Hand-authored** primitives in `robot.yaml` — fallback, and the cache the
  lowering tool emits.

**Avoiding a dual source of truth for kinematics:** `RobotDescription.joints`
stays **normative** for the kinematic chain (§1.3). URDF/SRDF contribute
**geometry + ACM only**; the lowering tool validates that URDF link names
match `joints` and fails closed on mismatch.

### 4. Forward kinematics — hand-rolled in the kernel, reused offline

In the hot path FK is hand-rolled (fixed-size `Eigen`, ~50 lines, provably
allocation-free, no new dependency). We deliberately do **not** use:

- **KDL (`orocos_kdl`)** — LGPL, so §1.9 requires TSC review, and its solvers
  allocate; wrong on both counts for the no-alloc kernel.
- **Pinocchio inside the kernel** — BSD and fast, but a heavy dependency whose
  no-alloc behaviour would need separate vetting.
- **`robot_state_publisher` / TF2** — unusable here: it FKs only the *current
  measured* joint state and publishes TF, whereas the kernel must FK the
  *candidate future* configurations carried in the chunk.

The **offline lowering tool** is free to reuse `mujoco.mj_kinematics` or
Pinocchio (BSD) to compute and verify link frames at authoring time.

`JOINT_POSITION` / `JOINT_TRAJECTORY` rows are joint configurations directly.
`CARTESIAN_*` / `*_TWIST` modes require seeding from the latest
`/joint_states` and integrating forward (a later phase that adds a kernel
joint-state subscription).

### 5. World model and degradation

World-collision (a later phase) checks link capsules against a **bounded**
world surface added to `WorldState` — a capped `collision_primitives` list
and/or an `occupancy_grid` reference (mirroring the `nav_msgs/OccupancyGrid`
shape `openral_runner/slam_bridge.py` already decodes). Ingestion is
double-buffered so a perception update never allocates on the hot path; a
world snapshot older than a configured age, or one exceeding capacity, is
treated as **unavailable**.

Degradation is **fail-closed by default** and never hidden (§1.4): no
collision model loaded, a stale/over-capacity world, or a per-chunk budget
overrun → reject + E-stop. A `warn`-only mode exists for sim/bring-up but is
gated behind an explicit parameter, a loud log, and OTel
`safety.severity=warn`; it is never the default, and **no parameter disables
the check** (§3 Safety, §6).

### 6. Contract surface added by this ADR (contracts-only first PR)

- **Pydantic (`openral_core.schemas`)**: `CapsuleShape`, `SphereShape`, the
  `CollisionShape` union, `LinkCollisionGeometry`; `OccupancyGridRef`;
  `RobotDescription.collision_geometry`, `.allowed_collision_pairs`,
  `.srdf_path`; `WorldState.collision_primitives`, `.occupancy_grid`;
  `CollisionEvidence` added to the `FailureEvidence` union.
- **IDL**: `openral_msgs/FailureTrigger` gains `uint8 KIND_COLLISION = 10`
  (the next free constant). The evidence variant and the IDL constant land
  together — the coupling is normative.
- **C++**: `ViolationKind::kCollision = 10` in `validator.hpp` plus the
  matching arms in the `lifecycle_kernel.cpp` evidence switch (mapping only;
  the kernel does not yet *emit* it in the contracts-only PR).
- **Exception**: `ROSCollisionImminent(ROSSafetyViolation)`.

`schema_version` stays `"0.1"` and all new fields default empty / `None`, so
every existing manifest and fixture loads unchanged (§1.6, no migrators).

### 7. Read-only visualization leg (dashboard pointcloud card)

The world map this ADR builds (the OctoMap whose occupied voxel centers are
published on `/octomap_point_cloud_centers`, and whose lowered grid the kernel
rasterizes on `/openral/world_voxels`) is otherwise invisible to an operator —
only the kernel's pass/clamp verdicts reveal it. To make "what the robot
believes is occupied" inspectable, a **read-only** bridge,
`openral_runner.world_cloud_bridge.WorldCloudBridge`, subscribes to
`/octomap_point_cloud_centers`, transforms the points into the robot
`base_link` frame via TF2, crops them to a local box, renders an oblique
"chase-cam" perspective PNG colored by distance from the robot, and emits one
`world.pointcloud` OTel span (throttled to 1 Hz). The dashboard store routes
that span into `_topics["pointcloud"]`, and the UI shows it in a
`card-world-cloud` card beside the SLAM 2D occupancy card.

This leg is **outside the real-time kernel** (Python proposes; the kernel's
`/openral/world_voxels` path and `validate()` are untouched) and adds no
safety surface: it only subscribes to a perception topic and emits telemetry,
raises no `ROSSafetyViolation`, and degrades by warn-and-skip when TF is not
yet available. It is wired through `compose_runtime(enable_world_cloud_bridge=…)`
and enabled by the same `--enable-octomap` deploy-sim flag that brings up
octomap_server (the centers topic exists exactly when octomap is on). The
mirror of ADR-0025's `SlamMapBridge → slam.occupancy_grid → SLAM card`.

## Alternatives considered

- **MoveIt / FCL as the gate.** Only checks motions it plans (not VLA, teleop,
  jogs, or learned nav); FCL/BVH allocates and is not real-time, so it cannot
  live in the kernel; it needs per-robot URDF+SRDF the repo does not ship; and
  it is arm-centric (no mobile base, no humanoid whole-body). Retained as a
  *planning-time* checker only — this ADR does not remove it.
- **A non-real-time Python pre-filter node** (wrapping MoveIt's `CollisionEnv`
  or MuJoCo) in front of the kernel. Mesh-accurate and universal, but it moves
  the authoritative gate out of the real-time kernel and adds a second
  estop-producing node. Rejected: the kernel should remain the single,
  auditable, allocation-free last gate.
- **Collision in the Layer-5 WAM.** The WAM is optional, best-effort planning
  with deadline-fallback semantics — unsuitable as a hard safety veto.
- **Clamp to the nearest collision-free configuration.** Violates
  reject-not-clamp (§1.4); a future ADR if ever justified.

## Consequences

- The kernel gains a robot collision model (and, for the world phase, a
  read-only world subscription) — more logic in the safety-critical component,
  hence a **mandatory safety-WG reviewer and a hazard-log update** on the
  implementing PRs (§3 Safety).
- Collision faults become first-class `FailureTrigger(KIND_COLLISION)` events,
  traceable and rate-limited like every other safety event.
- Robots gain an authoring path for link primitives + ACM (MJCF-derived,
  URDF/SRDF-derived, or hand-authored); the MJCF-first sim fleet needs no new
  files.
- Conservative primitives may reject motions that pass very close to
  obstacles; mesh-accurate planning remains a separate, non-safety concern.

## Phasing

1. **Contracts + this ADR (no behaviour change).** Schemas, `KIND_COLLISION`,
   `kCollision`, `ROSCollisionImminent`, a real openarm fixture, and tests.
2. Kernel runtime: hand-rolled FK + closed-form capsule **self-collision** for
   `JOINT_POSITION`, allocation-free, with sim tests (real MuJoCo, no mocks).
3. The **lowering tool** (MJCF / URDF+SRDF / hand-authored adapters).
4. **World-collision** via the bounded `WorldState` surface; fail-closed on
   stale world.
5. **Swept / CCD** over the chunk horizon.
6. **Mobile-base / Nav2** path — route `/cmd_vel`-derived base motion through
   the supervised pipeline so the footprint is checked against the world;
   amends ADR-0024's out-of-scope note.

Each runtime phase ships with sim tests and, where a HAL changes, HIL tests
(§2), and re-confirms the `test_no_alloc.cpp` guarantee.

---

## Amendment 2026-06-10: the offline URDF/SRDF lowering tool (Phase 3)

Phase 3's "lowering tool" is now implemented for the **URDF/SRDF** source as
`packages/openral_safety/openral_safety/urdf_lowering.py` plus the
`openral collision lower|check` CLI. It populates the manifest path
(`robot.yaml`'s `collision_geometry` + `allowed_collision_pairs`, consumed by
`collision_params_from_description`) — the version-controlled, hand-reviewable
form — and is distinct from `mjcf_lowering.py` (the runtime MJCF path).

### Geometry
One conservative capsule/sphere per URDF `<collision>`: primitives map by exact
analytic bounds (box→8 corners, cylinder→cap rims, sphere→exact `SphereShape`);
meshes are PCA-fit to a bounding capsule that **contains every vertex** (a
conservative over-approximation, so the kernel never under-covers). A missing
mesh file warns rather than silently dropping the link (§1.4).

### ACM — computed against the geometry the kernel loads
The kernel checks collisions with **capsules**, but a MoveIt SRDF is
**mesh**-based. So the ACM is computed against the same capsule geometry the
kernel will use (`acm_for_geometry`):

```
ACM = adjacent ∪ always-colliding(capsule) ∪ [ SRDF-disabled   if an SRDF exists
                                              | never-colliding(capsule) otherwise ]
```

- **adjacent** — directly joint-connected (always disabled).
- **always-colliding(capsule)** — pairs whose capsules overlap in *every* sampled
  pose. These are the **capsule-junction artifacts a mesh SRDF omits**: a short
  link (e.g. panda `link6`) makes its skip-one neighbours' capsules
  (`link5`↔`link7`, 32 mm penetration in 100 % of poses) permanently overlap.
  They **must** be disabled or the kernel false-E-stops every step.
- **SRDF-disabled** — the mesh-proven "never collide" set (`disable_collisions`),
  authoritative where an SRDF is vendored.
- **never-colliding(capsule)** — the no-SRDF fallback: a MoveIt-Setup-Assistant-
  style random-pose sweep, disabling pairs that overlap in *no* sampled pose.

Collision is tested with the kernel's own `mjcf_lowering._seg_seg_distance`, so
the matrix matches exactly what the kernel sees. The sweep is **deterministic**
(pinned `_RNG_SEED = 20260610`, `_N_SAMPLES = 2000`) — the basis of
`openral collision check`. The sampling fallback is **conservative** against
URDF-lowered (mesh-bounding) capsules: its disabled set is a subset of the
precise-mesh SRDF's, never false-permissive.

### Workflow & safety posture (CLAUDE.md §3)
- `openral collision lower --robot <yaml>` prints a unified **diff by default**;
  it mutates only with explicit `--write`. A regenerated ACM — a safety input —
  never changes silently. `--acm-only` / `--geometry-only` scope the output so
  hand-tuned safety geometry isn't churned when only the ACM needs refreshing.
- `openral collision check (--robot | --all)` exits non-zero on drift — a
  fleet-wide guard (`tests/unit/test_collision_lowering_fleet.py`) replacing the
  earlier single hand-pinned panda test.
- Every regenerated ACM is reviewed by the safety WG before merge; the manifest
  splice preserves all surrounding comments and emits a `# GENERATED (source:
  srdf|sampling)` provenance header so the block's origin is auditable.

### First application
`panda_mobile` regenerated from a vendored Franka MoveIt SRDF
(`robots/panda_mobile/panda_mobile.srdf`): the tool restores the four SRDF
"Never" pairs the hand-authored ACM had dropped — including `link1`↔`link4`, the
false self-collision that **E-stopped a live pi05 episode** — and retains the
`link5`↔`link7` capsule junction (now auto-derived, not hand-listed). The result
is identical to the hand-aligned ACM of 247cfb5, now reproducible from source.

### Per-robot source policy
A robot is onboarded onto self-collision checking individually (each is a safety
decision): vendor its upstream MoveIt SRDF to `robots/<name>/<name>.srdf` and set
`srdf_path` where one exists (UR, SO-ARM, Flexiv, Interbotix/WidowX-ALOHA,
Sawyer, Franka); otherwise the sampling fallback applies (the humanoids, openarm —
openarm is MJCF-only and stays on the `mjcf_lowering` path until it gains a URDF).
Onboarding a robot adds a new kernel input and ships with the safety-WG review and
hazard-log update this ADR's §Safety requires.

### Amendment 2026-06-10b: onboarding new robots (geometry + ACM + joint FK)

The tool now onboards a robot that had **no** collision model. Beyond geometry +
ACM it lowers the per-joint **forward kinematics** the kernel needs to place
capsules:

- `lower_joint_fk` computes each manifest joint's fixed `origin` as the URDF
  transform from its ``parent_link`` to its ``child_link`` at the zero
  configuration (via the URDF's own FK), and takes ``axis`` from the matching URDF
  joint. Using FK-composite — not a single joint origin — is correct even when the
  URDF inserts non-identity intermediate links (e.g. UR's ``base_link_inertia``,
  rotated 180° about Z). Joints whose parent/child aren't both in the URDF (a
  synthetic gripper, a base DoF) are omitted.
- Generated geometry is **scoped to the manifest's kinematic chain**, so an orphan
  URDF link (e.g. ``panda_leftfinger`` on a manifest that models a single
  ``panda_finger_pair``) can't reach the kernel.
- `urdf_path` accepts a ``robot_descriptions:<module>`` form for **xacro-only**
  robots (UR / Flexiv); `xacrodoc` processes the xacro in-process (no ament
  workspace needed). The CLI writer injects FK into the joints block
  (comment-preserving, idempotent) and appends the collision blocks when absent.

**Onboarded (this is a per-robot safety decision; each ships with safety-WG
review):** `franka_panda` (plain URDF + the Franka SRDF), `ur5e` / `ur10e`
(xacro URDF + vendored ros-industrial UR SRDFs). Each carries conservative
capsules, a mesh-authoritative ACM (+ capsule-junction pairs), and joint FK
verified by composing the manifest chain against the URDF's FK. The fleet guard
checks the **full** model (geometry + ACM + FK) for tool-generated robots and
ACM-only for hand-tuned ones (panda_mobile). Still pending: the humanoids
(g1/h1/gr1 — sampling fallback, no SRDF) and openarm (MJCF-only, no URDF).

### Amendment 2026-06-10c: conservative sampling, MJCF backend, openarm/rizon4

**Conservative sampling ACM.** A random-pose sweep cannot *prove* a pair never
collides — it can miss the tail, especially between independent kinematic branches
on a bimanual / humanoid robot. So the no-SRDF fallback now disables **only**
adjacent + always-colliding (capsule-junction) pairs; every other pair stays
**checked**. (Mesh-authoritative SRDFs still contribute their proven "Never" set,
so SRDF robots stay efficient.) This regenerated h1/g1/so100/rizon4 to smaller,
safer ACMs (e.g. h1 137→19 pairs) and is the safe default for bimanual cross-arm
pairs, which now always stay checked.

**MJCF backend** (`lower_robot_from_mjcf`). For MJCF-native robots with no URDF
whose collision geoms are meshes (which `mjcf_lowering`'s primitive path skips) —
e.g. bimanual `openarm`. It keeps the manifest's hand-authored capsules and lowers
**joint FK** (the MJCF parent→child transform at the rest pose) + the conservative
ACM (mujoco FK sweep). Manifest link names that diverge from the MJCF
(`openarm` `link0`/`link7` vs MJCF `base_link`/`ee_base_link`) are reconciled via
the `sim_joint_name` ↔ MJCF-joint correspondence, not by guessing names. The CLI
writer injects FK and writes the ACM while leaving the hand-authored geometry block
untouched.

**openarm + rizon4 onboarded.** `rizon4` via its xacro URDF (sampling ACM); `openarm`
via the MJCF backend (FK for all 14 arm joints, verified by composing the manifest
chain against the MJCF; 0 cross-arm pairs disabled — all checked).

**Launch fix.** `sim_e2e.launch.py` only replaces the manifest self-collision model
with `mjcf_lowering`'s params when those actually carry a model
(`self_collision_enabled`). A mesh-only MJCF (openarm) lowers to *disabled*; using
that silently turned bimanual self-collision **off** at sim deploy. The launch now
keeps the manifest's hand-authored capsules + (newly lowered) FK + ACM instead —
re-enabling openarm self-collision. Safety-WG reviewable.
