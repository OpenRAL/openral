# ADR-0054 — `goal_builder` as a joint/pose/look_at library over `ROSActionRskill`

- **Status:** Accepted 2026-06-12 — names confirmed (`rskill-moveit-joints` / `-eef-pose` / `-look-at`), Q1–Q3 resolved. Implementation landing phase-by-phase; HF Hub create/delete (phase 7) gated on explicit go-ahead. Amends/extends **ADR-0044** (which introduced `RosIntegration.goal_builder`).
- **Date:** 2026-06-12
- **ADR number:** `0054`. Renumbered from `0052` on merge with `master`, which had since claimed `0050` (skill VRAM eviction), `0051` (detector invocation mode), and `0052` (cross-frame object lift); the approach-to-pose ADR is `0053`, the registry ADR `0055`. The integer is not load-bearing — cross-refs use filenames.
- **Related:**
  - ADR-0044 — introduced `goal_builder: Literal["look_at"] | None` and `LookAtRskill`; this ADR generalises that seam.
  - ADR-0024 — `kind: ros_action` + `ROSActionRskill`: the shared MoveGroup/action engine (goal build → plan → joint-reorder → per-waypoint replay through `/openral/candidate_action`).
  - ADR-0026 — `goal_params_json` deep-merge over `default_goal_json`; the per-dispatch override path a builder consumes.
  - ADR-0053 — approach-to-pose dispatches the MoveGroup rSkill at a **joint-space** `starting_pose`; a future Cartesian approach would use the `pose` builder defined here.
  - ADR-0022 — reasoner LLM tool palette (per-skill `goal_params_schema`).

## Context

Two production rSkills wrap `moveit_msgs/MoveGroup`, and a third path exists:

| Skill | `goal_builder` | Goal type | Adapter |
| --- | --- | --- | --- |
| `openral-moveit-plan-arm` | *(unset)* | **joint-space** (`joint_constraints`) | base `ROSActionRskill` (verbatim `default_goal_json`) |
| `openral-look-at` | `"look_at"` | **Cartesian gaze** (camera +Z → target; `position_constraints` + `orientation_constraints`) | `LookAtRskill(ROSActionRskill)` |

They already **share the entire engine** — `ROSActionRskill` does the goal send, `build_joint_permutation_from_names` reorder, and the per-waypoint replay; the C++ kernel checks every replayed waypoint. **The only thing that differs is how the MoveGroup goal is constructed**, and that variation is already abstracted by `RosIntegration.goal_builder` (`python/core/.../schemas.py`: `Literal["look_at"] | None`).

Today the Cartesian path is reachable *only* as the gaze specialisation. There is no generic "plan to this end-effector pose" goal, and the joint path is an implicit "unset builder" rather than a named one. The question (raised reviewing ADR-0053): should "approach/reach" skills be able to take **either a joint-space or a Cartesian end-effector goal**, and should `look_at` and `moveit-plan-arm` be unified?

**Finding:** the engine is already unified. `LookAtRskill` already splits into a gaze-specific step (`compute_gaze_pose(goal_xyz, target_xyz, view_axis="+z")`) and a **generic** pose→constraints step (`build_look_at_constraints(camera_goal, link_name, link_t_cam, tolerances)` → a MoveGroup `goal_constraints` entry with position + orientation constraints, accounting for a link→target mount offset). The generic half is exactly a Cartesian-pose builder; gaze is a pose *source*.

## Decision

Promote `goal_builder` from a single flag into a small **builder library** over the one `ROSActionRskill` engine. Do **not** merge the two skill manifests.

### D1 — `goal_builder` enum: `{None, "joint", "pose", "look_at"}`

Widen `RosIntegration.goal_builder` to `Literal["joint", "pose", "look_at"] | None` (additive; `None` stays the default — back-compatible, `schema_version` stays `"0.1"`, no migrator):

- **`None`** — verbatim `default_goal_json` + ADR-0026 overrides (the raw-IDL escape hatch; kept for arbitrary wrapped actions, e.g. Nav2).
- **`"joint"`** *(ship it — Q1 resolved)* — `JointGoalRskill`: consumes a `joint` block (`{group_name, positions: [...], joint_names?: [...], tolerances}`) and emits `joint_constraints`. The clean, LLM-facing replacement for `openral-moveit-plan-arm`'s hand-written `joint_constraints` JSON (positions, not raw constraint dicts). `joint_names` default to the group's manifest-declared order.
- **`"pose"`** *(the headline addition)* — `PoseGoalRskill`: consumes a `pose` block and lowers it into `position_constraints` + `orientation_constraints`.
- **`"look_at"`** — unchanged behaviour; re-expressed as a *specialisation of* `"pose"` (D3).

### D2 — `PoseGoalRskill(ROSActionRskill)`: the generic Cartesian EEF builder

New adapter selected by `goal_builder: "pose"`. Its `pose` block:

```json
{ "pose": {
    "frame_id": "panda_link0",
    "position": [x, y, z],
    "orientation": [a, b, c, d],
    "quaternion_order": "xyzw",
    "position_tolerance_m": 0.01,
    "orientation_tolerance_rad": 0.05
} }
```

- **Orientation is a quaternion (Q2 resolved).** Supplied as a 4-float array whose component order is declared by the manifest's `quaternion_order` (`Literal["xyzw","wxyz"]`, default `"xyzw"`) — so each rSkill fixes its own convention and the builder maps it to `geometry_msgs/Quaternion` unambiguously. (Named-field `{x,y,z,w}` is the unambiguous alternative; we follow the declared-order array per the answer to Q2.)
- **The constrained link + tool offset come from the `RobotDescription` (Q3 resolved).** Default: constrain the planning group's tip link with an **identity** offset. When the manifest/robot declares an end-effector/tool frame with a mount transform, the builder re-expresses the goal for that frame — exactly as `LookAtRskill` sources the camera mount from `SensorSpec` via `_camera_mount` (`goal_link = goal_target @ inv(link_t_target)`). If no such frame is declared → identity. *(Caveat: `EndEffectorSpec` carries no transform today — see Implementation phase 6 / Q-new.)*

It lowers the pose into one MoveGroup `goal_constraints` entry via a shared `build_pose_constraints(pose, *, link_name, link_t_target=identity, position_tolerance_m, orientation_tolerance_rad)` helper — extracted by generalising the existing `build_look_at_constraints` (which already does pose→constraints with a link offset). No TF, no gaze, no camera. `plan_only` honoured as for look_at (replay through `/openral/candidate_action`; the kernel is the limiter).

### D3 — `LookAtRskill` becomes a `pose` specialisation

Refactor so the inheritance is `ROSActionRskill ◀ PoseGoalRskill ◀ LookAtRskill`. `LookAtRskill` keeps only the gaze-specific work — resolve the camera sensor + mount, read its current pose from TF, `compute_gaze_pose(...)` — then hands the resulting pose to the inherited `build_pose_constraints` lowering. `build_look_at_constraints` collapses into `build_pose_constraints` (the `link_t_cam` offset is the generic `link_t_target`). Net: one pose→constraints implementation, two goal *sources* (explicit pose, computed gaze).

### D4 — Keep the skills separate; unify the *builders*, not the *manifests*

The three goal types stay **distinct rSkills / distinct reasoner capabilities** (not one overloaded "move-arm (mode=…)" skill). Rationale:

- The reasoner palette (ADR-0022/0026) is clearer with single-purpose tools + tight `goal_params_schema` than one overloaded tool — overloading muddies LLM tool selection.
- rSkill packaging (ADR-0024) treats one capability = one manifest; that is the unit the registry and reasoner reason about.
- Gaze is not "just a pose goal" — its camera/TF/standoff geometry belongs in the `look_at` source, kept out of the generic `pose` path.

### D5 — Rename the skills for unambiguous reasoner selection

Today's names mix conventions (`openral-moveit-plan-arm`, `openral-look-at`). Rename to a uniform `rskill-moveit-<goal-type>` scheme (HF repo `OpenRAL/rskill-moveit-<goal-type>`) so name, `goal_builder`, and intent line up:

| New rSkill (HF: `OpenRAL/…`) | `goal_builder` | `actions` | Intent | Replaces |
| --- | --- | --- | --- | --- |
| `rskill-moveit-joints` | `"joint"` | `reach` | move the arm to a **joint configuration** | `rskill-moveit-plan-arm` |
| `rskill-moveit-eef-pose` | `"pose"` | `reach` | move the **end-effector to a 6-DOF Cartesian pose** | *(new)* |
| `rskill-moveit-look-at` | `"look_at"` | `look` | **aim the camera** at a 3-D point | `rskill-look-at` |

**Recommended over the first proposal** (`-eef` / `-joints` / `-look`): `-eef-pose` reads as "a pose target" (vs. a joint "pose"); `-look-at` keeps the action verb (matches `actions: look`); `-joints` is already clear. `RSkillAction` is a **closed enum** — both joint and Cartesian arm moves are the `reach` verb; the slug + `goal_builder` + `goal_params_schema` disambiguate joints-vs-EEF, not a bespoke action. **The heaviest disambiguation for the LLM is the `description` + `goal_params_schema`, not the bare id** — those are what the palette renders (ADR-0022/0026), so they matter more than the slug. *(Names are a recommendation pending your confirmation — they become HF repo ids, which are awkward to change once published.)*

### D6 — Per-robot manifests unchanged in shape

Each builder still needs the robot's planning-group (`group_name`) and (for `pose`/`look_at`) the constrained link / tool frame; these stay per-robot manifest copies (as today). The builder library does not remove that, but it does mean a new goal *type* is a new builder + a thin manifest, never a new engine.

## Consequences

- **"Joint-space or Cartesian EEF" becomes a manifest choice**, not a code fork: pick `goal_builder` ∈ `{None/"joint", "pose", "look_at"}`. ADR-0053's approach-to-pose stays on joint (`starting_pose` is joints); a Cartesian approach is a `"pose"` dispatch through the same runner path — no new mechanism.
- **One pose→constraints implementation.** `build_look_at_constraints` is absorbed into `build_pose_constraints`; less surface, one place to get the link-offset math right (safety-relevant — a wrong constraint frame mis-aims the arm).
- **No skill merge / no palette regression.** Distinct capabilities stay distinct.
- **Additive schema change.** `goal_builder` widening is back-compatible; existing manifests (unset / `"look_at"`) are unaffected; `schema_version` stays `"0.1"`.

## Implementation plan (phased; each independently testable)

1. **Schema:** widen `RosIntegration.goal_builder` to `Literal["joint","pose","look_at"] | None`; add `quaternion_order` to the `pose`-block contract; update the field docstring + hypothesis round-trip; `docs/methods` + repo-state-map (`RosIntegration` note). Additive, `schema_version` stays `"0.1"`.
2. **Extract `build_pose_constraints`** from `build_look_at_constraints` (pure; unit-test pose→constraints incl. the `link_t_target` offset + `quaternion_order` mapping, against the existing look_at fixtures so behaviour is provably unchanged).
3. **`PoseGoalRskill(ROSActionRskill)`** + `goal_builder: "pose"` resolver branch (mirror the `look_at` branch in `make_default_skill_resolver`). Unit-test the `pose`-block parse + lowering with a synthetic manifest (no ROS — the look_at adapter test pattern).
4. **`JointGoalRskill`** + `goal_builder: "joint"` (Q1) — `joint`-block → `joint_constraints`. Unit-test parse + emission.
5. **Refactor `LookAtRskill` → `PoseGoalRskill`** subclass; keep its tests green (no behaviour change).
6. **`RobotDescription` tool-frame offset (Q3)** — source `link_t_target` from a declared EEF/tool frame when present (mirroring `_camera_mount` from `SensorSpec`), else identity. `EndEffectorSpec` carries no transform today, so this is a small schema add (an optional `mount_xyz_quat` / tool-frame ref) — gate it behind its own test; until then `pose` constrains the group tip link with identity.
7. **New rSkills + HF migration (D5)** — see §Migration; gated on name confirmation + acceptance.

## Migration & publishing (HF Hub)

rSkills are HF Hub repos (ADR-0024; the manifest is the artifact). Renaming + adding the `pose` skill is a Hub migration, **outward-facing** — it does not run until the ADR is accepted and the names (D5) are confirmed:

1. **Create** the new local `rskills/rskill-moveit-{joints,eef-pose,look-at}/` dirs (manifest + README per `rskills/template/README.md`; `rskill_publisher` validator must pass), each with its `goal_builder`, `actions` verb, and `goal_params_schema`. Per-robot copies as today (planning group + frames).
2. **Publish** to `OpenRAL/rskill-moveit-{joints,eef-pose,look-at}` (capital-`O` org namespace — HF ownership is case-sensitive) via `tools/rskill_publisher.py … --publish` under the authenticated org. Note CLAUDE.md §3: provenance is **unverified** (sigstore not implemented) — do **not** describe these as "signed".
3. **Update all references** in one PR: `deploy_sim.py` `approach_skill_id` default (ADR-0053), `sim_e2e.launch.py`, embodiment/robot manifests that name the old skills, the resolver search paths, ADR-0053 + ADR-0044 cross-refs, `docs/methods`, the repo-state-map, and any eval fixtures.
4. **Remove** the old `openral/rskill-moveit-plan-arm` + `openral/rskill-look-at` Hub repos (and local dirs) **only after** the new ones are live and all references are switched — deletion is destructive + outward-facing, so it needs explicit go-ahead and a grep proving zero remaining references.

The Hub create/delete steps (2, 4) are external side effects — I will not run them without an explicit instruction at that point.

## Non-goals

- Merging the three skills into one overloaded skill (D4).
- Replacing the joint-space approach-to-pose of ADR-0053 (joint remains correct for a joint `starting_pose`).
- Adding a MoveIt dependency or `move_group` bring-up (that is ADR-0053 phase 4 / the deploy graph's concern).

## Open questions

- **Q-new — `EndEffectorSpec` transform (from Q3).** Sourcing `link_t_target` from the `RobotDescription` needs a tool-frame *transform*, which `EndEffectorSpec` lacks today. Add an optional `tool_frame` + `mount_xyz_quat` to `EndEffectorSpec` (mirrors `SensorSpec`'s camera mount), or look the frame up via TF at dispatch (like look_at does for the camera)? The former is static + testable without TF; the latter handles runtime-reconfigurable tools. Defaulting to identity until decided is safe.

*Resolved by review:* **Q1** — ship `"joint"` (D1). **Q2** — quaternion array with manifest-declared `quaternion_order`, default `"xyzw"` (D2). **Q3** — offset from the `RobotDescription` tool frame when declared, else identity (D2/D6/phase 6).
