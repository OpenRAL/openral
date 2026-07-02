# ADR-0065 — cuMotion: CUDA-accelerated MoveIt planning behind a GPU capability gate

- **Status:** Proposed 2026-06-21. Addresses the *manipulation* half of issue #76 ("Evaluate GPU-accelerated navigation (cuMotion / GPU-MPPI)"); the *navigation* half (GPU-MPPI for the mobile/humanoid base + the `cmd_vel` safety-supervisor bypass) is explicitly **split out** to a separate ADR (see §Out of scope). Seeded by an evaluation doc (`docs/evaluations/issue-76-cumotion-moveit.md`, since removed; findings folded into this ADR).
- **Date:** 2026-06-21
- **ADR number:** `0065` (next free; 0064 is the highest accepted). The integer is not load-bearing — cross-refs use filenames.
- **Related:**
  - ADR-0024 — `kind: ros_action` + `ROSActionRskill`: the shared MoveGroup/action engine (goal build → plan → joint-reorder → per-waypoint replay through `/openral/candidate_action`). cuMotion plugs in **below** this, at the MoveIt planner level.
  - ADR-0054 — `goal_builder` library (`joint`/`pose`/`look_at`) over `ROSActionRskill`; the `rskill-moveit-*` family that transparently inherits cuMotion planning.
  - ADR-0053 — collision-aware approach-to-pose; composes the MoveGroup rSkill before policy activation — also inherits cuMotion planning for free.
  - ADR-0030 / ADR-0040 — geometric self/world/voxel collision checking **in the C++ safety kernel**, every control mode. cuMotion plan-time collision is *additive* to this, never a replacement.
  - ADR-0046 — NVIDIA GR00T as an out-of-process GPU backend; precedent for GPU-stack isolation and license-posture handling.
  - ADR-0016 — multi-platform support (x86 CUDA + CPU, L4T/Jetson); the capability-gate + CPU-fallback discipline this ADR follows.
  - ADR-0012 — uniform Apache-2.0 for OpenRAL code; cuRobo is Apache-2.0 so no posture guard is needed on our code.

## Context

OpenRAL plans arm motion through MoveIt 2. The `rskill-moveit-{joints,eef-pose,look-at}` family (ADR-0024 + ADR-0054, all `kind: ros_action`) sends a `moveit_msgs/MoveGroup` goal to `/move_action`, extracts `planned_trajectory.joint_trajectory`, reorders it into `RobotDescription.joints`, and replays it **one waypoint per step** (`chunk_size: 1`, enforced for `ros_action`) through `/openral/candidate_action` → Python supervisor → C++ safety kernel → HAL → `ros2_control`. The planner today is **OMPL (CPU)** with MoveIt's **FCL** collision check at plan time. ADR-0053's approach-to-pose and any future Cartesian approach reuse the same engine.

Issue #76 asks whether GPU-accelerated planning can improve this. **NVIDIA Isaac ROS cuMotion** (`isaac_ros_cumotion`) is a CUDA manipulation stack built on **cuRobo** (Apache-2.0; PyTorch + CUDA + Warp). It ships **`isaac_ros_cumotion_moveit`, a MoveIt 2 planning-pipeline plugin** — i.e. it slots in at the *planner* level, **below** the rSkill, leaving the `/move_action` → `JointTrajectory` contract unchanged. cuRobo plans in tens of ms (vs OMPL's hundreds of ms–seconds), produces smooth dynamics-aware **collision-free interpolants** (not just collision-free waypoints), and can ingest depth-derived ESDF (nvblox) for live obstacle avoidance. Supporting packages: `isaac_ros_cumotion_object_attachment`, `isaac_ros_cumotion_robot_segmenter`.

**Distro/OS fit (verified).** cuMotion (x86) requires Ubuntu 24.04, an Ampere+ GPU, CUDA 13.0+, driver 580+ (or Jetson Thor / JetPack 7.1). OpenRAL's default distro is **ROS 2 Jazzy on Ubuntu 24.04 (Noble)** (`scripts/bootstrap_ubuntu.sh`, `Dockerfile.dev`/`Dockerfile.runtime` default `ROS_DISTRO=jazzy`), and the canonical inference image `docker/inference/Dockerfile.x86` is already **x86_64 + CUDA 13 + ROS 2 Jazzy + Ubuntu 24.04** — an exact match. Humble/22.04 is the secondary supported path and is **not** a cuMotion target (falls back to OMPL).

**Truth-over-plausibility note.** cuMotion is an *arm/manipulator* planner. It does **not** address the issue's title-level concern of *base navigation* (Nav2 MPPI for mobile/humanoid). That is a separate workstream (§Out of scope). This ADR is deliberately scoped to the manipulation path, which is OpenRAL's core strength.

## Decision

Adopt cuMotion as an **optional, capability-gated MoveIt planning pipeline** behind the existing `/move_action` server. Do **not** change the rSkill contract, and do **not** change the safety path.

### D1 — cuMotion is selected per-request via `pipeline_id`, gated by capability, OMPL fallback

cuMotion is a MoveIt **planning-pipeline plugin** (`isaac_ros_cumotion_moveit`), selected per planning request by `moveit_msgs/MotionPlanRequest.pipeline_id`. OpenRAL **does not own the `move_group` bring-up** — the per-robot upstream `*_moveit_config` is already an rSkill `ros_dependency` and provides `/move_action`; the `rskill-moveit-*` family only *consumes* it. So adopting cuMotion is **not** a new OpenRAL MoveIt package or a re-implemented bring-up. It is two thin pieces:

1. **Make the pipeline available to `move_group`** — add the cuMotion `ros_dependency` (`ros-${ROS_DISTRO}-isaac-ros-cumotion-moveit`, or built-from-source) and ship a documented `cumotion_planning.yaml` snippet to add to the user's existing moveit_config `planning_pipelines`. Config + docs, no new package.
2. **Select the pipeline** — set `request.pipeline_id = "isaac_ros_cumotion"` on the MoveGroup goal, **gated by `RobotCapabilities.supports_cumotion()`** (which already encodes the Ampere+/CUDA≥13/VRAM floor). When the gate is false (CPU-only, low VRAM, non-CUDA, Humble) the field is left unset and MoveIt uses its default pipeline (**OMPL — today's behaviour, unchanged**).

The selection (point 2) is a gate-driven injection into the goal via the existing ADR-0026 goal-merge — **one rSkill works on GPU and CPU hosts**, no parallel manifest. Because the `request` block already flows from `default_goal_json`/`goal_params_json`, **`rskill-moveit-{joints,eef-pose,look-at}` and ADR-0053 approach-to-pose inherit GPU planning transparently** — no new rSkill, adapter, or `goal_builder`.

*(Correction, 2026-06-22: an earlier draft of D1 proposed a new `openral_moveit_config` package + `move_group.launch.py`. That was over-scoped and inconsistent with OpenRAL's existing MoveIt integration, which deliberately defers `move_group` bring-up to the upstream per-robot moveit_config. Selection is a `pipeline_id` field, not a bespoke launch.)*

### D2 — Trajectory-only (`plan_only`); the safety path is unchanged

cuMotion **must not drive controllers**. The `move_group` goal is `plan_only` (as the `rskill-moveit-*` family already requires); the planned `JointTrajectory` is replayed waypoint-by-waypoint through `/openral/candidate_action`. The Python supervisor (`openral_safety/supervisor_node.py`) and the C++ kernel (`cpp/openral_safety_kernel`) validate **every waypoint** exactly as for OMPL: n_dof, per-joint position/velocity limits, workspace box, force/torque caps, NaN/Inf, and ADR-0030/0040 capsule self/world/voxel collision — **reject, never clamp**, E-stop + `FailureTrigger` on violation. "Python proposes; C++ disposes" holds verbatim. No safety-kernel code change is required to adopt cuMotion. *(Touching `openral_safety` / the kernel is therefore explicitly out of scope for D1–D2; if any kernel change is later proposed it follows CLAUDE.md §3 — safety-WG reviewer, hazard-log update, at-least-as-conservative tests.)*

### D3 — cuMotion's plan-time collision is additive defense in depth

cuRobo's GPU collision checking (sphere/ESDF) is a **plan-time** filter, orthogonal to the kernel — the same relationship MoveIt FCL has today (ADR-0030: "planning-time and in-kernel checks are orthogonal"). It does not gain any new authority over actuation. Net safety effect is **positive**: the kernel checks discrete waypoints, while cuRobo optimizes continuous collision-free, in-envelope interpolants ⇒ fewer kernel rejections / E-stops and smoother motion.

### D4 — Per-robot cuRobo config is generated from the existing collision-lowering tool

cuRobo needs a per-embodiment kinematics + sphere/capsule config. The kernel already needs per-embodiment capsule collision models, produced by `openral collision lower` (ADR-0030). Extend that tool to **also** emit the cuRobo robot config from the **same** URDF/SRDF source of truth, so plan-time and kernel-time geometry stay consistent (a divergence here is safety-relevant). Do **not** bundle NVIDIA's example robot assets (separate, non-Apache licenses, ADR-0012); author configs from OpenRAL robot manifests.

### D5 — Process / packaging isolation

cuMotion is already a native ROS 2 node, so its torch/Warp/CUDA stack is isolated **by process** simply by running it as its own node — cleaner than the ZMQ sidecar of ADR-0046, which existed only to bridge a Python-version gap. If a version/ABI conflict with VLA envs nonetheless appears, containerize the `move_group` + cuMotion node (the `docker/inference/Dockerfile.x86` Jazzy+CUDA-13 image is the natural base). Declare the dependency, never vendor it.

### D6 — Optional dedicated `rskill-cumotion-*` only for cuMotion-exclusive features

A separate rSkill is warranted **only** for capabilities `MoveGroup`/OMPL can't express: nvblox depth-ESDF live obstacle avoidance (via `isaac_ros_cumotion_robot_segmenter`), batch/multi-seed IK, or cuMotion-native goal types. Express as `kind: ros_action` with a `RosIntegration` block pointing at cuMotion's action server; add a `goal_builder: "cumotion"` adapter **only if** its goal IDL differs from `MoveGroup`. Declare GPU need via `min_vram_gb` + `capabilities_required`; set `fallback_skill_id` to the OMPL `rskill-moveit-*`. This is a **follow-up**, not part of the initial landing.

## Consequences

- **GPU arm planning becomes a deployment capability, not a code fork.** Same rSkills, same reasoner palette, same safety path; the only difference is which MoveIt pipeline `move_group` loads, chosen from `RobotCapabilities`.
- **Lower planning latency** tightens the S1/S2 loop and leaves more headroom in the replanning ladder and per-skill latency budgets.
- **Fewer E-stops / smoother motion** from collision-free, dynamics-aware interpolants.
- **Clean license posture.** cuRobo is Apache-2.0; no posture guard on OpenRAL's Apache-2.0 code (contrast GR00T weights, ADR-0046). NVIDIA example *assets* are excluded (D4).
- **New hardware/runtime dependency**, fully optional and gated — CPU/Humble hosts keep OMPL with zero behaviour change. The 8 GB reference host must budget VRAM for cuMotion **alongside** a VLA; the gate + fallback make low-VRAM hosts degrade gracefully.
- **One more geometry artifact per robot** (cuRobo config), generated by an extended ADR-0030 tool from the existing source of truth.

## Implementation plan (phased; each independently testable)

1. **Capability gate + `pipeline_id` selection (D1).** ✅ *gate done* — `RobotCapabilities.supports_cumotion()` (Ampere+/CUDA≥13/VRAM floor; unit-tested). *Remaining:* inject `request.pipeline_id="isaac_ros_cumotion"` into the MoveGroup goal via the ADR-0026 goal-merge when the gate is true, else leave unset (OMPL). Plus the `cumotion_planning.yaml` snippet + `ros_dependency` for the user's moveit_config. **No new package / no `move_group` launch** (OpenRAL doesn't own bring-up).
2. **cuRobo config generation (D4).** ✅ done — `openral collision lower --emit-cumotion` emits a cuRobo robot config (collision spheres + ACM + cspace joints) from the same lowered geometry; unit-tested + verified for real against franka_panda. (`retract_config`/accel limits deferred to phase 3.)
3. **End-to-end plan-and-replay (D2/D3)** on a GPU host: `rskill-moveit-joints`/`-eef-pose` planning via cuMotion, trajectory replayed through `/openral/candidate_action`, **kernel validates every waypoint** — verify no bypass and that an out-of-envelope plan is still rejected + E-stopped. Real env per the project's testing norms (or sim where no GPU/robot is available, with `pytest.skip(reason=...)` on CPU CI).
4. **Latency + quality benchmark.** Compare cuMotion vs OMPL plan time + kernel-rejection rate on a fixed set of goals; record numbers in the eval doc.
5. **Docs in the same PR(s).** ADR status → Accepted on ratification; `docs/methods` for any new public symbol; repo-state-map if a package/flag is added; README/toolchain notes for the GPU dependency.
6. **(Follow-up) `rskill-cumotion-*` (D6)** for nvblox/batch-IK — separate ADR amendment or PR, gated on a real depth-perception use case.

## Non-goals / Out of scope

- **GPU navigation (the issue's headline).** GPU-MPPI for the mobile/humanoid base, the humanoid S0 cerebellar layer, and **closing the `cmd_vel` safety-supervisor bypass** (Nav2 publishes `BODY_TWIST` straight to HAL, documented as out-of-scope of ADR-0024, relying on Nav2's `velocity_smoother`) are a **separate ADR + issue**. cuMotion is arm-only and does not touch them. Recommend splitting issue #76 accordingly.
- **Any change to the safety kernel or supervisor** (D2 keeps them untouched).
- **Replacing OMPL** — OMPL stays the CPU/Humble fallback (D1).
- **Bundling NVIDIA example robot assets** (D4; ADR-0012).

## Validation (2026-06-22, RTX 4070 Laptop / Ada, CUDA 13.2, ROS Jazzy)

- **Install (Phase 3, partial).** `ros-jazzy-isaac-ros-cumotion{,-moveit,-interfaces,-robot-description}` 4.4.0 install cleanly from NVIDIA's Isaac apt repo. The package is **self-contained C++/CUDA** — `libcumotion_planner_lib.so` links a bundled native `libcumotion.so.1` + `libcudart.so.13`; **no Python cuRobo** is needed (correcting an earlier draft assumption). The MoveIt plugin registers as `isaac_ros_cumotion_moveit/CumotionPlanner` in moveit pluginlib. NVIDIA ships `franka.xrdf` / `ur5e.xrdf` / `ur10e.xrdf`; `--emit-cumotion` (D4) covers the rest of the fleet.
- **Live GPU plan.** The cuMotion planner node loaded the panda config (`panda.urdf` + `franka.xrdf`) and brought up its IK solver + trajectory optimizer + `MotionPlan` action server on the GPU. A joint-to-joint `MotionPlan` goal returned **`success: true`, `MoveItErrorCodes.SUCCESS`, in ~0.12 s** — a concrete datapoint for the latency benefit (vs OMPL's hundreds of ms–seconds). (Benign quirk: NVIDIA's `franka.xrdf` expects two finger joints while the `moveit_resources` panda URDF has `panda_finger_joint2` as a mimic → a logged-but-non-fatal warning; arm planning unaffected.)
- **Remaining for full e2e (Phase 3 cont.).** Drive cuMotion through `move_group` via `pipeline_id` (the injection is unit-tested) and route the planned trajectory through `/openral/candidate_action` → supervisor → C++ kernel, confirming an out-of-envelope plan is still rejected + E-stopped. Needs the colcon-built OpenRAL workspace + a `move_group` bring-up with the cuMotion pipeline in `planning_pipelines`.

## Open questions

- **Q1 — VRAM co-residency.** Can cuMotion's working set + a quantized VLA co-exist on the 8 GB reference host, or does cuMotion need eviction coordination with the GPU-resident skill slot (ADR-0050)? Measure in phase 4.
- **Q2 — `move_group` lifecycle.** Is cuMotion warm-loaded once per deploy, or per-skill? Warm-load avoids per-call CUDA init cost but holds VRAM (interacts with Q1).
- **Q3 — nvblox source.** If/when D6 lands, does the depth-ESDF come from the object-detection/octomap perception bus already in-tree, or a dedicated cuMotion segmenter pipeline?
