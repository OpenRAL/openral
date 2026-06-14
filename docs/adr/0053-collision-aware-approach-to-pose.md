# ADR-0053 — collision-aware "approach to pose" before rSkill activation

- **Status:** Accepted — design + runner wiring landed 2026-06-12; real-HW actuation safety-WG-gated. **Supersedes the 2026-06-12 first draft of this ADR**, which proposed a bespoke MuJoCo capsule/voxel planner on a false premise (see §History).
- **Date:** 2026-06-12 (revised same day)
- **Issue:** #222 (`feat(hal): collision-aware "approach to pose" before rSkill activation (sim + real)`)
- **Related:**
  - **`rskills/rskill-moveit-joints/`** (ADR-0024, `kind: ros_action`; `openral-moveit-plan-arm` before its ADR-0054 rename) — wraps `moveit_msgs/action/MoveGroup`; plans a **collision-free joint-space motion** (MoveIt FCL: self-collision **and** planning-scene/world collision) and replays the returned `trajectory_msgs/JointTrajectory` one waypoint per `step()` onto `/openral/candidate_action` (the safety kernel checks every waypoint). **This is the planner + executor this ADR reuses.**
  - **`rskills/rskill-moveit-look-at/`** (ADR-0044; `openral-look-at` before ADR-0054) — same MoveGroup wrapper but `goal_builder: look_at` lowers a target into `position_constraints` + `orientation_constraints` (`plan_only: true`); the Cartesian-pose-goal precedent if approach ever needs a pose (not joint) target. ADR-0054 also adds `rskill-moveit-eef-pose` as the generic Cartesian-EEF sibling.
  - `python/rskill/src/openral_rskill/ros_action_rskill.py` — `ROSActionRskill` + `build_joint_permutation_from_names`: the trajectory-mode adapter (goal build, joint reorder into `RobotDescription.joints` order, per-waypoint replay) both MoveIt rSkills use.
  - ADR-0044 (`refine_approach_pose`, `OccupancyGridIndex`) — the **base** 2-D standoff viewpoint; still the base leg's job.
  - ADR-0024/0025 (`rskill-nav2-navigate-to-pose`; Nav2 as a reasoner-managed background service; `/cmd_vel` out of supervisor scope).
  - ADR-0030 (geometric safety; the kernel's capsule/voxel limiter — stays a *last-resort limiter*, not the approach planner).
  - ADR-0026 (`goal_params_json` structured skill goals — how the runner retargets MoveIt at the next skill's `starting_pose`).
  - ADR-0027 (`robot.yaml`; `urdf_path`). **NB:** ADR-0027 leaves `urdf_path` unset for **OpenArm specifically**, not repo-wide (see §Context).

## Context

Today an rSkill's in-distribution `starting_pose` is applied as a **hard `qpos` teleport**:

- `openral sim run --rskill …` snaps it unconditionally inside `env.reset()`.
- `openral deploy sim` snaps it when an `ExecuteSkill` goal begins:
  `rskill_runner_node._maybe_reset_hal_to_starting_pose`
  → `/openral/<robot>/reset_to_pose` (reflective `ResetToPose`) →
  `MujocoArmHAL.reset_to_pose`, which sets `qpos` directly.

The teleport is unphysical, **ignores collisions** (it can place links inside obstacles), and has **no real-HW analogue**. We want a smooth, collision-checked motion from the robot's *current* configuration to `starting_pose` so the policy's first observation is in-distribution **and** the motion is safe — across `sim run`, `deploy sim`, and `deploy run` (real HW).

**Two facts shape the design — and correct the first draft of this ADR:**

1. **MoveIt is already integrated; URDF/SRDF already exist.** The repo ships **SRDFs** for `franka_panda`, `ur5e`, `ur10e`, `rizon4`, `panda_mobile`, and sets `urdf_path` for franka/ur5e/ur10e/rizon4/g1/so100. Two production rSkills already drive MoveIt: `rskill-moveit-joints` (joint-goal) and `rskill-moveit-look-at` (pose-goal). MoveIt's planner does **self-collision + planning-scene (world, incl. octomap) collision** during planning — mesh-accurate, strictly better than a capsule/voxel approximation. The first draft's premise ("no URDF/SRDF, no MoveIt, MoveIt would be greenfield") was **false** — it over-generalised ADR-0027's *OpenArm-specific* unset `urdf_path` to the whole repo. We do **not** build a parallel motion planner.
2. **The motion has two physically distinct parts.** For a *mobile* robot: drive the **base** to a standoff (2-D floor plan), then move the **arm** to `starting_pose` (3-D volume). Different collision models, different existing infrastructure. Conflating them is the trap the original sim-only sketch fell into.

## Decision

Split along the base/arm seam and **reuse the existing MoveIt + Nav2 rSkills on each side**. There is **no new planner, no new HAL method, and no new IDL.**

### D1 — Base approach: reuse `rskill-nav2-navigate-to-pose` (unchanged)

For mobile robots the **base standoff** is reached by the existing
`rskill-nav2-navigate-to-pose` rSkill (ADR-0024) over the **2-D slam_toolbox / Nav2 costmap**, given the grid-refined viewpoint from **ADR-0044** (`refine_approach_pose`). This ADR adds nothing here; it records the ordering:

```
recall_object / locate_in_view ─▶ refine_approach_pose (ADR-0044, 2-D grid)
        ─▶ ExecuteSkill(rskill-nav2-navigate-to-pose)   # BASE → standoff (Nav2 costmap)
        ─▶ approach-to-pose (THIS ADR: MoveIt)            # ARM  → starting_pose (3-D MoveIt scene)
        ─▶ ExecuteSkill(<manipulation VLA>)               # ACT
```

(Honest caveat inherited from ADR-0024: Nav2's `/cmd_vel` is not under the OpenRAL supervisor today — ADR-0024's gap, not this one's.)

### D2 — Arm approach IS a MoveGroup plan, dispatched at the next skill's `starting_pose`

"Approach to pose" = run the **`rskill-moveit-joints` rSkill retargeted at the upcoming skill's `starting_pose`**. Concretely, before executing skill *S*, the runner:

1. Reads `S.manifest.starting_pose` (joint positions, `RobotDescription.joints` order).
2. Resolves the configured **approach skill** (`approach_skill_id`, default `rskills/rskill-moveit-joints`) via the existing `_resolve_and_check_skill`, passing a `goal_params_json` (ADR-0026) that overrides the `rskill-moveit-joints` `joint` block's `positions` with `starting_pose` (joint names read from the approach manifest's `default_goal_json`).
3. Runs it through the existing `_run_until_done_or_deadline` loop — `ROSActionRskill` sends the `MoveGroup` goal, MoveIt plans a **collision-free** trajectory (self + planning-scene/world), and the adapter replays each waypoint as a `JOINT_POSITION` `Action` onto `/openral/candidate_action`.

MoveIt does the planning **and** the collision check. OpenRAL's actuation path (`/openral/candidate_action` → kernel → `safe_action` → HAL) is the executor — there is **no separate "JointTrajectory executor"** to build; MoveIt produces the `JointTrajectory` and `ROSActionRskill` already replays it.

### D3 — Collision model: MoveIt's planning scene (self + world)

MoveIt's FCL planner checks the candidate trajectory against:

- **Self-collision** — from the robot's URDF + SRDF allowed-collision matrix (mesh-accurate; the SRDFs already in `robots/<robot>/`).
- **World collision** — the live `/planning_scene`, which a MoveIt **octomap plugin** populates from the same depth → octomap feed `packages/openral_octomap_bridge/` produces. So the **3-D world avoidance** #222 asks for is MoveIt's, fed by the existing perception bridge.

The **base's 2-D** avoidance (D1, Nav2 costmap) and the **arm's 3-D** avoidance (D3, MoveIt planning scene) are therefore *both* enforced, at the layer each belongs to. The ADR-0030 kernel capsule/voxel check remains the independent **last-resort limiter** on `/openral/candidate_action` — it re-checks every replayed waypoint, so a MoveIt/kernel scene divergence fails closed, not open.

### D4 — Failure contract: typed abort, never blind motion

If MoveIt cannot plan (`MoveGroup` returns a non-success `error_code`, empty trajectory, or the action aborts/times out), `ROSActionRskill` raises a typed `ROSError`; the runner maps it to a `ROSPlanningError` `failure_reason`, **aborts the `ExecuteSkill` goal**, and never falls back to the teleport or starts the policy from an unreachable/colliding state. This is the key difference from `ResetToPose` (best-effort, warns-then-continues).

### D5 — Runner wiring (the only new code)

- **`rskill_runner_node`** gains `approach_skill_id` (rSkill URI; empty = disabled) and `approach_skill_revision` params. When set **and** the next skill declares a `starting_pose`, the runner dispatches the approach skill (D2) and aborts on failure (D4). When unset it keeps the legacy best-effort `ResetToPose` snap — **opt-in, no regression**. Dispatch precedence (`approach` ▸ `reset` ▸ `none`) is the pure `resolve_starting_pose_action`.
- **`deploy_sim.py` / `deploy run`** auto-wire `approach_skill_id:=…` alongside `reset_to_pose_service:=…`; `sim_e2e.launch.py` forwards it.
- **No HAL method, no lifecycle service, no new IDL.** The whole mechanism is the existing `kind: ros_action` path.

## Safety

- **Real (`deploy run`):** the trajectory is replayed **one waypoint at a time through `/openral/candidate_action`**, so the safety kernel checks every step (velocity / workspace / force envelope + the ADR-0030 capsule/voxel limiter) — the same actuation path ADR-0044's `plan_only` MoveIt replay uses, and the inverse of the HAL-internal `ResetToPose` snap. `MoveGroup.plan_only` MUST be true (as in look_at) so move_group does not *also* drive its own `ros2_control` controllers and bypass the kernel.
- **Gating.** Standing up `move_group` + controllers on real HW, and any HIL run, pulls in (a) a safety-WG reviewer, (b) a hazard-log entry, (c) tests proving the new behaviour is **at least as conservative** as today's teleport (which ignores collisions → trivially less safe). **No flag disables the kernel limiter. No path leaves motors energized on a Python crash** — a dispatch/plan exception aborts the goal; the deadman watchdog owns the energized state.

## Implementation plan

1. **Runner dispatch** *(landed)* — `approach_skill_id` param; `resolve_starting_pose_action` precedence; `_dispatch_moveit_approach` builds the `goal_params_json` joint override from `starting_pose` and runs the approach skill via `_resolve_and_check_skill` + `_run_until_done_or_deadline`; abort on failure. Pure pieces (`resolve_starting_pose_action`, the goal-override builder, the manifest joint-name extractor) are unit-tested; the ROS dispatch is gated like every other ROS-graph path here (needs a colcon-built `openral_msgs`/`moveit_msgs` + a running `move_group`).
2. **CLI / launch auto-wiring** *(landed)* — `deploy_sim.py` + `sim_e2e.launch.py` forward `approach_skill_id`.
3. **Per-robot approach manifests** — `rskill-moveit-joints` ships a Franka `panda_arm` default; each target arm needs a manifest copy with its planning group + joint names (mechanical; tracked per robot).
4. **MoveIt bring-up in the launch graph** **[safety-WG-gated for real]** — `move_group` + the robot's MoveIt config (SRDF already present) + the octomap planning-scene plugin, with `plan_only`. Sim first (deploy-sim), then HIL on a real arm (UR5e / Franka — see Q2) with an e-stop teardown hook.

## Consequences

- **No parallel motion planner.** We reuse MoveIt's mesh-accurate FCL planning (self + world) and the `ROSActionRskill` replay — `grep`-before-helper / don't-duplicate (CLAUDE.md §1.13) honoured. The bespoke capsule/voxel + interp→RRT planner from the first draft is **removed**.
- **The "JointTrajectory executor" was never needed.** MoveIt emits the `JointTrajectory`; `/openral/candidate_action` → kernel → HAL is the executor. "Real vs sim" is just whether `move_group` + controllers are up, not different OpenRAL code.
- **Two-level avoidance is honest and layered:** base 2-D via the nav rSkill + Nav2 costmap; arm 3-D via MoveIt's planning scene (octomap-fed) with the kernel limiter as backstop.
- **The teleport stays for bring-up** (`approach_skill_id` unset) — opt-in, no regression.
- **Robots without a MoveIt config get no approach** (they keep the snap) — e.g. OpenArm (no URDF/SRDF, ADR-0027). Stated, not hidden; resolving it = ship an OpenArm MoveIt config, not a fallback planner.

## Open questions (need a human decision)

- **Q1 — sim move_group cost.** Standing up `move_group` per deploy-sim run adds cold-start latency (MoveIt 0.5–2 s first plan) and a planning-scene monitor. Acceptable for the pre-skill approach? (The latency budget multiplier in `ROSActionRskill` already accommodates it.)
- **Q2 — real-executor validation target.** Validate phase 4 on UR5e / Franka (SRDF present) first; OpenArm-real is `null` and has no MoveIt config, so it stays on the snap until both land.
- **Q3 — octomap → planning scene wiring.** Confirm the MoveIt octomap-updater is fed from `packages/openral_octomap_bridge/` (vs. a second depth subscription) so the kernel limiter and MoveIt see the *same* world.

## History

The first draft of this ADR (same day) proposed a bespoke planner: a pure-Python capsule/voxel `is_valid(q)` predicate mirroring the C++ kernel, an interp→RRT-Connect `plan_approach`, a `MujocoArmHAL.approach_pose` sim executor, an `ApproachToPose.srv`, and a reflective lifecycle service — phases "1–8 landed". That design rested on the **false** premise that the repo had no MoveIt/URDF/SRDF. It was **reverted** in favour of reusing `rskill-moveit-joints` (then `openral-moveit-plan-arm`). Lesson (CLAUDE.md §1.2): verify "X doesn't exist in the repo" with a grep before building a parallel X.
