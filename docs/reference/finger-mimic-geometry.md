# Gripper fingers at their commanded opening: the follower finger as a rigid mimic-joint link

**Status:** implemented on `feat/349-finger-stroke-geometry`; Safety-WG review, the hazard-log row,
the ADR and the Isaac deploy-sim validation are owed (§7). Issue #349.

## 1. Problem (measured)

Isaac deploy-sim runs of a VLA restock policy on the bimanual OpenArm stopped on world-voxel
collisions of `openarm_left_finger_pair` / `openarm_right_finger_pair` while PhysX ground truth
said the fingers touched nothing: 0 mm penetration in 23 of 24 stops, the nearest surfaces 7-10 mm
or more away. 10 mm octomap cells did not remove the stops, so cell quantisation was not the main
term.

The main term was the finger model. `finger_pair` was the leader finger's link with the
equality-coupled follower finger folded in (`urdf_lowering.fit_collision_geometry_from_mjcf`), the
follower swept over the whole gripper stroke at 9 samples into **one** box + 26-DOP + exact hull.
The volume between the fingers was solid at every opening.

Measured with `hull_overhang_upper_bound_m` (the largest distance from a point of the kernel's hull
surface to the real finger meshes at that opening — how far the kernel invents finger), at 49
openings over the stroke, both arms (identical to 0.1 mm):

| geometry the kernel checks | max | mean |
|---|---|---|
| one hull, both fingers, whole stroke (before #349) | 84.2 mm | 66.8 mm |
| leader rigid + follower in 8 stroke bins (first implementation, rejected in review) | 26.3 mm | 21.9 mm |
| leader rigid + follower rigid on a mimic joint (**implemented**) | **16.9 mm** | **16.9 mm** |

The 16.9 mm that remains is each finger's own non-convexity (the hull of one finger overhangs its
mesh by 19.9-20.2 mm at the worst facet sample, `hull_overhang_m` in the manifest); it is the same
at every opening because nothing is swept any more. At full open the midpoint between the two
finger centroids (125 mm apart) is **28.5 mm clear** of the kernel's finger geometry (bins: 21.6 mm;
before: 22.1 mm inside it).

## 2. Design

**Rejected in review: stroke bins.** The first implementation kept the follower folded into
`finger_pair` and carried 8 per-sub-range hulls selected per check from the gripper's
`[measured, commanded]` interval. Every other joint in the kernel is checked at its commanded value
only; the bins added 8 hulls per finger (+5600 manifest lines), a selection pass, four fail-closed
fallbacks tied to measured-state freshness and a between-sample residual — machinery for one link,
where a rigid follower is what MoveIt and cuRobo do.

**Implemented: the follower finger is its own rigid link, posed by a mimic joint.**

- **Manifest.** `RobotDescription.mimic_joints` (new, default empty, backward compatible): a
  `MimicJoint` is the URDF `<mimic>` / MuJoCo joint-equality relation
  `q = multiplier * q[mimics] + offset` — `name`, `parent_link`, `child_link`, `mimics` (a
  `joints` name), `multiplier`, `offset`, `joint_type`, `axis_xyz`, `origin_xyz`/`origin_rpy`,
  `sim_joint_name`. It is the third edge kind of the collision tree beside `joints` and
  `fixed_attachments`: moving, but no command channel, so invisible to every `joints` consumer
  (action width, qpos map, state contract, runner permutation). The OpenArm declares
  `openarm_<side>_finger_joint2` → `openarm_<side>_finger_follower`, mimicking `<side>_gripper`
  with multiplier 1 / offset 0 on axis `[1, 0, 0]` (the MJCF spelling; the URDF's `multiplier="-1"`
  on axis `[-1, 0, 0]` is the same motion). The schema refuses a zero multiplier, a fixed type, a
  child defined by another edge and a `mimics` not in `joints`. **`finger_pair` keeps its name,
  its joint and its geometry role as the leader finger**, so every `GraspDeclaration.contact_links`
  producer, `jaw_point`, the attached-payload touch links and the tutorials are unchanged.
- **Lowering** (`openral collision lower`, `urdf_lowering`). A follower the manifest declares is
  **not** folded into its leader: both fingers are fitted rigidly in their own MJCF body frames, one
  box + exact hull each (`fit_link_with_tight_geometry`), and `robots/openarm/robot.yaml` was
  regenerated (`collision check` reproduces it). The lowering checks the manifest's
  `multiplier`/`offset` against the MJCF equality and refuses a disagreement
  (`_declared_mimic_followers`); the mimic joint's `origin_*`/`axis_xyz` are re-injected from the
  MJCF like every joint's. The ACM sweep poses followers from the equality at every sample (MuJoCo's
  `mj_kinematics` does not enforce equalities), and exempts a mimic child against its parent (adjacent)
  and against its leader's child — the two jaws of one gripper, rigidly related by the one
  coordinate they share and designed to meet (their hulls overlap 2.9 mm at closed). An undeclared
  follower (no manifest in-tree has one) still takes the whole-stroke sweep.
- **Envelope loader.** A mimic edge lowers to `collision_joint_kind` 3 (revolute) / 4 (prismatic)
  on the **mimicked** joint's `collision_dof_index` column, with the relation in the per-link
  `collision_mimic_multiplier` / `collision_mimic_offset` arrays (emitted only when the tree has a
  mimic edge; identity for every other link).
- **Kernel.** `JointKind::kMimicRevolute` / `kMimicPrismatic` (`is_mimic`): `forward_kinematics`
  reads the leader's column and moves the link by `multiplier * q + offset`; no qpos slot of its own,
  no Jacobian column (`jacobian_dls_step` skips it; configure refuses a mimic on the EE chain, which
  no finger is on), counted once in the FK-dof set. Configure refuses a mimic link without a dof
  column, with a non-finite relation or a zero multiplier, and mis-shaped mimic arrays. The octomap
  bridge's `robot_self_filter` mirrors the same kinds (`SelfJointKind`), pinned to the kernel's FK to
  1e-12.
- **Grasp-region exemption (ADR-0115).** The launch allowlist `grasp_contact_links` is now
  `RobotDescription.grasp_contact_links()`: every `role: gripper` jaw **and** its mimic follower. A
  `GraspDeclaration` still names the leader (`openarm_<side>_finger_pair`); the kernel widens the
  declaration's mask to every allowlisted link driven by the same joint column — the other jaw of
  the same gripper — so the follower is exempt with the leader. A follower off the allowlist stays
  checked (and `grasp_links_one_hand` still sees one hand: both jaws hang off `link7`).

## 3. Soundness (measured, `tests/unit/test_collision_finger_mimic.py`)

- **Enclosure.** Each finger's exact hull holds its own MJCF mesh: 0 vertices outside (to 1 nm
  rounding), once, because the geometry is rigid in its link frame — the same argument as for every
  other link; the 1 mm swept-link tolerance no longer applies to the fingers.
- **Placement.** The kernel's follower frame (`origin · R(axis, multiplier · q + offset)` from
  `link7`) against MuJoCo's follower body frame, on every follower mesh vertex, at 25 openings over
  the stroke: **0.0000 mm** (asserted ≤ 1 mm).
- **Conservatism.** Less conservative only between the fingers when the jaw is open — the point —
  and nowhere else: a cell a finger occupies at the commanded opening trips
  (`MimicJoint.ACellOnTheFollowerSurfaceTripsAtThatOpening`), a cell between closed fingers trips
  (`MimicJoint.ACellBetweenTheFingersClearsWhenOpenAndTripsWhenClosed`). The kernel checks discrete
  configurations, as for every link: a finger's motion between a chunk's rows is not swept, before or
  after this change. The follower is checked at the **commanded** value like every other link; the
  measured-state interval of the rejected design is gone with the bins.
- **Self-collision.** Follower ↔ leader and follower ↔ `link7` are exempt (above). Follower ↔
  `link5` / `link6` are **checked**: over 500 seeded in-limit poses the kernel's hull-refined gap
  never trips (`link5`: min +0.1 / +0.6 mm left/right, mesh-hull clearance ≥ 16.3 mm; `link6`: min
  +3.0 / +2.5 mm, mesh-hull clearance ≥ 21.5 mm; rest pose +39.4 / +7.3 mm). No new SRDF row. The
  leader's existing `finger_pair ↔ link5/link6` rows were measured against the old swept hull and
  may now be retirable; retiring a row is a Safety-WG decision (§7).

## 4. Cost

Per check: one more link (one box + a 276-282-vertex hull per follower) in the broad phase and the
staged narrow phase; one affine op in FK. No selection pass, no scratch beyond `link_world`, no
measured-state dependency. `robots/openarm/robot.yaml` is ~5600 lines shorter than with bins and
~400 longer than before #349 (two more hulls).

## 5. Tests

- C++ `test_collision`: `MimicJoint.*` — FK places the follower at `multiplier · q + offset` against
  a hand-computed pose (and its mirror relation to the leader); the identity relation when the mimic
  arrays are empty; a cell between the fingers clears open and trips closed; a cell on the follower
  surface trips at every opening; the grasp-region exemption covers a follower in its mask; leader ↔
  follower self-collision only when not exempt; no Jacobian column; allocation-free FK + voxel check.
- C++ `test_lifecycle_kernel`: end to end through the node — open clears the cell between the jaws,
  closed stops, a follower-only cell stops; a declaration naming only the leader exempts the follower
  (and not a non-allowlisted one); mis-shaped arrays, a zero multiplier, a mimic without a column, a
  mimic on the EE chain and an unknown kind fail configure.
- C++ `test_self_filter` (octomap bridge): a mimic link matches the kernel FK to 1e-12; refusals.
- Python `tests/unit/test_collision_finger_mimic.py`: rigid enclosure, follower placement vs the MJCF
  at 25 openings, the open-jaw gap, the MJCF-equality check and its refusal, kernel params (kind 3
  on the gripper column, relation arrays), `grasp_contact_links`, the generated ACM rows, schema
  refusals. `test_collision_geometry_enclosure.py` holds each finger to its own body (a declared
  follower is no longer the leader's); `test_collision_geometry_zero_pose.py`,
  `test_collision_fleet_static_links.py`, `test_collision_geometry_contracts.py` walk the mimic edge;
  `test_schemas_fuzz.py` fuzzes `MimicJoint`.

## 6. ADR draft (for `OpenRAL/management/adr/`; number assigned there)

> **ADR — The follower finger as a rigid mimic-joint link of the collision model**
>
> *Status:* proposed. *Deciders:* Safety WG. *Date:* 2026-10-11.
>
> *Context.* The OpenArm's second finger is equality-coupled to the first and was folded into
> `finger_pair` swept over the whole stroke (ADR-0114, §8.3 "Follower-joint finger" of
> `collision-geometry-review.md`). The kernel therefore checked a solid jaw at every opening: the
> union hull invented up to 84 mm (mean 67 mm) of finger, and Isaac deploy-sim runs stopped on
> cells PhysX says the fingers never touched (23 of 24 stops at 0 mm penetration). A stroke-binned
> follower (8 hulls selected from the `[measured, commanded]` interval) was implemented and rejected
> in review as machinery for one link that every other joint does without.
>
> *Decision.* `RobotDescription.mimic_joints` declares the URDF `<mimic>` relation as a third edge
> kind of the collision tree; the kernel gains `JointKind::kMimicRevolute` / `kMimicPrismatic`,
> posed from the mimicked joint's column at `multiplier * q + offset`, no column of its own. Each
> finger is lowered as one rigid box + exact hull and checked at the commanded value like every
> other link. The grasp-region allowlist is every gripper jaw and its follower; the kernel widens a
> declaration's mask to the allowlisted links on the same joint column. The lowering auto-exempts a
> mimic child against its parent and its leader's child. Schema addition is backward compatible.
>
> *Consequences.* Kernel finger excess 84.2 → 16.9 mm max, 66.8 → 16.9 mm mean (each finger's own
> non-convexity); the open-jaw gap midpoint 28.5 mm clear. Enclosure is rigid (0 mm), placement
> matches the MJCF to < 1 µm. Less conservative only between open fingers; never adds a stop. Two new
> exempt pairs per arm (jaw ↔ jaw, follower ↔ mount); follower ↔ link5/link6 checked and non-tripping
> over 500 poses. Every FK consumer (kernel, octomap self-filter, launch coverage FK, test FKs) learns
> the two kinds.
>
> *Alternatives rejected.* One hull per bin of both fingers (a convex hull of two fingers fills the
> gap: 22-64 mm per bin). Stroke bins on the follower (26.3 / 21.9 mm; rejected for the machinery and
> the measured-state dependency). Retiring `finger_pair` for two new links (`finger_leader`,
> `finger_follower`): every contact-link producer, tutorial and test would learn a new name for the
> same leader jaw, for no geometric gain.

## 7. Owed before relying on it

- **Safety-WG review** of this change (CLAUDE.md §3: touches `cpp/openral_safety_kernel` and the
  ACM: two new exempt pairs per arm).
- **Hazard-log row** (private `OpenRAL/management`): the residual of a rigid follower — the kernel
  checks the commanded opening (discrete configurations, as for every link), the manifest relation is
  trusted after the lowering's MJCF check, and the two auto-exempt pairs. Not assigned an id here.
- **ADR** (§6) recorded in the management decision log.
- **WG decision** on the leader's `finger_pair ↔ link5/link6` SRDF rows, measured against the old
  swept hull: the rigid leader's hull may no longer overlap them (the follower's does not).
- **Isaac deploy-sim validation** on the restock policy: the stop count and the PhysX ground truth
  against the 24-stop baseline.
