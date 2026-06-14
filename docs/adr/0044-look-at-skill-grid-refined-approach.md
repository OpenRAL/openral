# ADR-0044 — `look_at` skill + occupancy-grid-refined approach poses

- **Status:** Accepted — Phases 1–3 + 4a on `master`; Phase 4b implemented on the #282 stack (2026-06-10)
- **Date:** 2026-06-10

> **2026-06-10 Phase-3 amendment — `plan_only: true`.** The skill ships with
> MoveGroup's `plan_only` set: OpenRAL's actuation path is the per-waypoint
> replay through `/openral/candidate_action` (safety kernel checks every step,
> HAL actuates). Letting `move_group` *also* execute on its own
> `ros2_control` controllers would bypass the kernel and double-drive the arm.
> (Surfaced by the live franka run: the demo host lacked
> `joint_trajectory_controller` and returned `CONTROL_FAILED` — execution we
> never needed.) The integration test asserts the gaze via MoveIt's
> `/compute_fk` on the plan's final waypoint instead of executed TF. The
> selection mechanism is a new `RosIntegration.goal_builder` field
> (`Literal["look_at"] | None`) rather than a manifest-name match, so future
> goal-lowering adapters slot in the same way.
>
> **2026-06-12 amendment — see [ADR-0054](0054-moveit-goal-builder-library.md).**
> `goal_builder` is generalised into a `{None, "joint", "pose", "look_at"}`
> builder library over `ROSActionRskill`: a new `PoseGoalRskill` adds generic
> Cartesian-EEF goals, and `look_at` is re-expressed as a specialisation of it
> (the gaze step computes a pose, then delegates to the shared
> `build_pose_constraints` lowering this skill's `build_look_at_constraints`
> becomes). The two MoveGroup skills stay separate capabilities — only the
> builders unify. ADR-0054 also **renames** the skills: this `look_at` skill
> ships as `OpenRAL/rskill-moveit-look-at` (dir `rskills/rskill-moveit-look-at/`)
> and the joint-space sibling as `rskill-moveit-joints`. The historical
> `openral-look-at` / `openral-moveit-plan-arm` names below are retained as the
> design-time names; the loadable paths are the renamed ones.
- **Related:** ADR-0038 (spatial memory; `ApproachViewpoint` / `compute_approach_viewpoint`),
  ADR-0039 (`recall_object` / `resolve_place` query tools + active search),
  ADR-0043 (`locate_in_view` live-detector query — PR #282, reserves the 0043 slot),
  ADR-0030 (geometric safety; `OccupancyGridRef` on `WorldState`, Phase 1 landed),
  ADR-0024/0025 (Nav2 as reasoner-managed background service; `/cmd_vel` out of
  supervisor scope), ADR-0026 (`goal_params_json` structured skill goals)

## Context

The reasoner can now answer *where an object is* two ways: `recall_object`
(ADR-0038/0039 scene-graph memory → map-frame pose + a camera-facing
`ApproachViewpoint`) and `locate_in_view` (ADR-0043, live VLM detector). What it
**cannot** yet do is complete the physical follow-through:

1. **The standoff pose is not grid-validated.** `compute_approach_viewpoint`
   places the viewpoint `standoff_m` from the object in the x,y plane, yawed to
   face it — pure geometry. Nothing checks the chosen cell is *free* in the
   slam_toolbox occupancy grid or has line-of-sight to the object. If the ideal
   standoff lands inside a counter, the LLM dispatches a navigation goal the
   robot can only approximately honour (Nav2 goal tolerance), with the camera
   facing who-knows-what. Nav2 plans collision-free *paths*; it does not pick a
   *better goal*. The safety kernel (ADR-0030) enforces envelopes and (future
   phases) collision primitives — it is a last-resort limiter, **not** a
   planner, and pose refinement is explicitly out of its scope.

2. **There is no way to aim a camera at a point.** The approach yaw is planar —
   it never pitches a wrist camera down at a tabletop object. No `look_at`
   primitive exists; the gaze math (`_look_at_quat`: eye, target, up →
   quaternion) is **triplicated** in sim asset composers
   (`openral_sim/backends/{so101_box,openarm_robosuite,tabletop_push}/_assets.py`)
   where it only orients MJCF scene cameras. Camera aiming on a real robot is
   **actuation** — it must flow through `ExecuteRskillTool` → rskill_runner →
   `/openral/candidate_action` → safety kernel, never through a read-only
   reasoner tool.

Target behaviour (the "go see the object" chain):

```
recall_object ──▶ approach pose (grid-refined: free cell + line-of-sight)
      │
      ▼
ExecuteSkill(rskill-nav2-navigate-to-pose, goal=refined approach)   # base
      │
      ▼
ExecuteSkill(openral-look-at, target_xyz=object, camera="wrist")      # aim
      │
      ▼
locate_in_view(query, camera="wrist")                                 # verify
      │
      ▼
ExecuteSkill(<manipulation VLA>)                                      # act
```

## Decision

Four phases; 1–3 land on this branch (off `master`), Phase 4 lands after
PR #282 merges (it needs `recall_object` / `locate_in_view`).

### Phase 1 — shared gaze geometry (pure, no ROS)

Promote the triplicated `_look_at_quat` into a single public helper
(CLAUDE.md §1.13):

- `look_at_quat(eye_xyz, target_xyz, *, up=(0,0,1)) -> (w,x,y,z)` — canonical
  look-at rotation with the degenerate-case handling the sim copies already
  have (zero norm → downward fallback; near-parallel up → alternate up).
- `compute_gaze_pose(camera_xyz, target_xyz, *, up=(0,0,1)) -> Pose6D` — the
  full 6-DOF pose whose camera view axis points at the target.

Home: `openral_world_state` next to `compute_approach_viewpoint` (Layer 2 owns
spatial geometry; `openral_core` stays schemas-only). The three sim asset
files are refactored onto it in a separate `refactor(sim):` commit. Property
tests: for randomized eye/target pairs, rotating the view axis by the returned
quaternion points at the target within tolerance; degenerate cases match the
documented fallbacks.

### Phase 2 — occupancy-grid approach refinement (Layer 2)

New `openral_world_state.grid` module:

- `OccupancyGridIndex.from_msg(nav_msgs/OccupancyGrid)` — decode the row-major
  int8 grid (`-1` unknown / `0..100` occupancy) + `resolution_m` / `origin`
  into a queryable index; `is_free(x, y, *, inflation_m)` treats unknown as
  occupied (conservative) and requires the inflation disc free.
- `line_of_sight(grid, a_xy, b_xy) -> bool` — Bresenham ray over occupied
  cells (unknown blocks sight too).
- `refine_approach_pose(grid, viewpoint: ApproachViewpoint, target_xyz, *,
  max_radius_m, min_standoff_m, max_standoff_m) -> ApproachViewpoint | None` —
  if the ideal viewpoint cell is free **and** sees the target, return it
  unchanged; otherwise spiral outward (ring search at grid resolution) for the
  nearest cell that (a) is free under inflation, (b) keeps standoff within
  `[min, max]` of the target, (c) has line-of-sight to the target's x,y; re-yaw
  the pose at the snapped position via the existing yaw math. `None` when no
  cell qualifies inside `max_radius_m` — the caller reports "no reachable
  viewpoint" honestly (never fabricates, mirroring `ROSObjectNotInMemory`
  posture).

Tests drive scenario grids that encode the live situations (ideal standoff
overlapping furniture under footprint inflation, a wall breaking
line-of-sight, a target sealed beyond admissible standoff → `None`) at a
realistic slam resolution — the same fixture style the ADR-0030 Phase 6
kernel nav-goal sim test uses; a slam-captured map fixture upgrade rides with
Phase 4 (which wires the live `/map` anyway). One semantic worth naming: the
**target's own cell is exempt** from line-of-sight (an object shares its 2-D
footprint cell with whatever it sits on).

**Relationship to ADR-0030 Phase 6** (`/openral/check_nav_goal`, on its own
branch): that gate is kernel-side *enforcement* — "is this goal safe?". This
module is planning-layer *proposal* — "which nearby pose is free and sees the
object?". Complementary, not overlapping: a refined pose still crosses the
Phase 6 gate (and every other safety check) when it lands.

### Phase 3 — `rskills/rskill-moveit-look-at` (the actuation skill)

New `kind: ros_action`, `role: s1` rSkill wrapping **MoveGroup pose-goal
constraints** (`/move_action`), reusing `ROSActionRskill` unchanged — the
`goal_params_json` deep-merge already carries arbitrary goal structure; the
existing `openral-moveit-plan-arm` stays joint-space (its manifest says
pose-goal variants ship as a separate rSkill, which this is).

- `goal_params_schema`: `{target_xyz: [x,y,z] (map frame), camera: str
  (sensor name from RobotDescription.sensors; default `"wrist"`), standoff_m?:
  number}`. A robot with no sensor named `wrist` and no explicit `camera`
  fails at configure with `ROSConfigError` listing the available sensor names
  (explicit beats implicit — no silent guess at which camera to aim).
- At configure: resolve the named sensor's `frame_id` from the robot manifest
  (camera-agnostic, ADR-0043 posture — never a hardcoded frame); TF-lookup the
  camera→EE static offset; `compute_gaze_pose` (Phase 1) for the camera; compose
  to an EE goal; submit MoveGroup `position_constraints` +
  `orientation_constraints`.
- Trajectory replay waypoint-per-chunk through `/openral/candidate_action` —
  **every step crosses the safety kernel** (unlike Nav2's result-only mode,
  ADR-0024). No new safety surface.
- Integration test beside `tests/integration/test_moveit_plan_arm_franka.py`
  (real MoveIt, real franka description): plan a gaze at a known tabletop
  point; assert the final EE pose's camera axis hits the target within
  tolerance.

### Phase 4 — reasoner wiring (4a landed; 4b after PR #282 merges)

**Phase 4a (implemented 2026-06-10, this branch).** `reasoner_node` subscribes
the latched `/map` (`TRANSIENT_LOCAL`, matching slam_toolbox; params
`occupancy_map_topic` default `/map` — empty disables — and
`approach_inflation_m` default 0.25) and holds the latest
`OccupancyGridIndex`; the spatial-query dispatch refines each `recall_object`
match's `ApproachViewpoint` through `refine_approach_pose` before rendering,
so the LLM only ever sees grid-valid approach poses (an explicit "approach
BLOCKED on the occupancy grid" when refinement fails — never a fabricated
pose). The hook is a duck-typed `ApproachRefiner` callback on
`run_spatial_query`, keeping the L4 bridge module free of L2 imports (same
pattern as `SpatialMemoryQuerier`). Grid absent → the geometric viewpoint
passes through unchanged, silently — a caveat would be noise on deployments
that don't run SLAM at all (refinement of the originally-drafted "rendered
caveat" wording).

**Phase 4b (on the #282 stack).** `DEFAULT_SYSTEM_PROMPT` gains the ladder:
recall → navigate(refined approach) → `look_at` → verify with `locate_in_view`
→ manipulate. It was deferred off `master` because the ladder names
`locate_in_view`, which only exists on the #282 branch (ADR-0043); building it
there avoids advertising a tool the LLM cannot call. When #282 was rebased onto
the merged Phase-4a `master`, its `find_object → recall_object` rename swept
the Phase-4a wiring too, so 4a and 4b speak one vocabulary on this stack.

Layer note: the reasoner (L4) reading L2 world-state data mirrors the existing
injected-`SpatialMemory` pattern (ADR-0039 Phase 2b) — no new boundary, but
recorded here because the grid is a second L2 read surface.

## Consequences

- **Duplication removed:** three `_look_at_quat` copies collapse into one
  public helper (METHODS.md entry; sim asset files import it).
- **New rSkill** `openral-look-at` is a classical (non-VLA) skill —
  installable/dispatchable wherever MoveIt runs; embodiments without an arm
  planner simply don't install it (capability gating, ADR-0018 §4).
- **The kernel is untouched.** Refinement is planning-layer; ADR-0030's
  phases proceed independently. Nav2 still owns path planning; we only pick a
  better goal.
- **Honest failure modes:** `refine_approach_pose` returning `None` and the
  "grid stale" caveat both surface in the LLM prompt — no silent fallbacks.
- **Phase 4 dependency:** explicitly sequenced after PR #282; until then the
  refined-approach path is exercised only by unit/integration tests, not the
  live reasoner.

## Testing

- Phase 1: property tests (randomized gaze poses), degenerate-case tests,
  sim-asset refactor covered by existing sim suites.
- Phase 2: real captured slam map fixture + hand-authored edge grids
  (occupied ideal cell, no-line-of-sight, no-free-cell-in-radius → `None`).
- Phase 3: `tests/integration/` with real MoveIt + franka description
  (skip path: MoveIt absent); manifest validation against the real
  `rskill.yaml`; hypothesis round-trip on any new schema surface.
- Phase 4 (later PR): reasoner integration test — canned `RecallObjectTool`
  with a grid where the ideal viewpoint is occupied; assert the re-prompt
  carries the snapped pose.
