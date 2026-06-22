# Evaluation — Issue #76: cuMotion-backed GPU motion planning for MoveIt

> Status: **evaluation / pre-ADR** (not an accepted decision). Seeds a future ADR (next free number ≈ **ADR-0065**).
> Scope: answers "can we make an rSkill from NVIDIA Isaac ROS cuMotion that improves MoveIt, how does it interact with the safety kernel, and what are the benefits?"
> Branch: `worktree-eval+issue-76-gpu-nav-cumotion`.

---

## 0. TL;DR

- **cuMotion is an arm/manipulator motion planner**, not a mobile-base navigation stack. It ships `isaac_ros_cumotion_moveit`, a **MoveIt 2 planning-pipeline plugin** backed by **cuRobo** (CUDA motion generation). Issue #76 is titled around *GPU navigation* (Nav2 MPPI, mobile + humanoid base). **cuMotion does not address that headline** — it improves the *manipulation* path, which is OpenRAL's core strength.
- **Recommendation: split Issue #76 into two workstreams.**
  - **(A) cuMotion ⇒ MoveIt arm planning** — well-scoped, high value, low new surface (this document).
  - **(B) GPU-MPPI navigation + the `cmd_vel` safety-supervisor bypass** — the issue's actual headline, a separate design (own ADR).
- **For (A), the cheapest integration is not a new rSkill at all**: swap the MoveIt *planner plugin* under the existing `/move_action` server to `isaac_ros_cumotion_moveit`. The existing `rskill-moveit-{joints,eef-pose,look-at}` family (ADR-0024 + ADR-0054) then gets GPU planning **transparently**, behind a capability flag. A dedicated `rskill-cumotion-*` is only warranted for cuMotion-specific features (nvblox depth obstacle avoidance, batch IK).
- **Safety fit is strong and unchanged**: cuMotion plans; the kernel still disposes. Plan with `plan_only`, replay the trajectory waypoint-by-waypoint through `/openral/candidate_action`, and the Python supervisor + C++ kernel validate every waypoint exactly as they do for OMPL today. cuMotion's GPU collision checking is *plan-time defense in depth*, not a kernel replacement.
- **License is clean**: cuRobo is **Apache-2.0** (unlike GR00T weights), so no license-posture guard on OpenRAL code. Real cost is **hardware** (Ampere+ GPU, CUDA, VRAM budget vs. VLAs) and **per-robot cuRobo config authoring**.

---

## 1. What cuMotion actually is (verified)

Source: <https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_cumotion/index.html> and <https://github.com/NVlabs/curobo>.

`isaac_ros_cumotion` provides **CUDA-accelerated manipulation** — collision-free trajectory generation for **robot arms**, integrated into **MoveIt 2**. Four packages:

| Package | Role |
|---|---|
| `isaac_ros_cumotion` | Core planner node — ROS 2 IK + motion-generation interfaces |
| `isaac_ros_cumotion_moveit` | **MoveIt 2 planning plugin** exposing cuMotion as an external planner |
| `isaac_ros_cumotion_object_attachment` | Attach/detach objects on the gripper for planning |
| `isaac_ros_cumotion_robot_segmenter` | Segments the robot out of depth streams (for nvblox obstacle maps) |

**Backend: cuRobo** — "CUDA-accelerated library for robot motion generation, built on PyTorch, CUDA and Warp." **License: Apache-2.0** (example robot assets have separate licenses). Capabilities: GPU-parallel FK/IK, collision checking, trajectory optimization, geometric planning, and **GPU-native ESDF perception** (signed-distance fields from depth, i.e. nvblox).

**Hardware requirements** (from the docs):
- x86_64 + **Ampere or newer** NVIDIA GPU, **8 GB+**, Ubuntu 24.04, **CUDA 13.0+, driver 580+**; or
- **Jetson Thor** (T5000/T4000, JetPack 7.1); or DGX Spark.

> **Truth-over-plausibility note.** The issue body and title say "GPU-accelerated navigation (cuMotion / GPU-MPPI) for mobile + humanoid." cuMotion is for **manipulator arms via MoveIt**. It is the right tool for *arm* planning and a poor fit for *base* navigation. The two are conflated in the issue; this evaluation separates them.

---

## 2. How OpenRAL does MoveIt arm planning today

The `rskill-moveit-*` family (ADR-0024 ros-wrapped rSkills + ADR-0054 goal-builder library):

| rSkill | `goal_builder` | Target |
|---|---|---|
| `rskill-moveit-joints` | `joint` | joint-space goal |
| `rskill-moveit-eef-pose` | `pose` | Cartesian EE pose |
| `rskill-moveit-look-at` | `look_at` | camera gaze pose |

All three are `kind: ros_action`, talk to **`moveit_msgs/MoveGroup` at `/move_action`**, extract `planned_trajectory.joint_trajectory`, and **replay one waypoint per step** through the safety path (`chunk_size: 1`, enforced for `ros_action`). The planner is whatever MoveIt's configured pipeline uses — today **OMPL (CPU)** with MoveIt's **FCL** collision checking at plan time.

Call path (verified):
```
ExecuteRskill goal
  → RskillRunnerNode._execute_cb
  → make_default_skill_resolver  (manifest.kind == ros_action; pick adapter by goal_builder)
  → {Joint,Pose,LookAt}GoalRskill (lowers goal → MoveGroup constraints)
  → /move_action  (MoveIt: OMPL plan + FCL collision)
  → result.planned_trajectory.joint_trajectory  (reordered into RobotDescription.joints)
  → per-waypoint Action → /openral/candidate_action
  → Python supervisor → C++ safety kernel → HAL → ros2_control
```
Key files: `python/rskill/src/openral_rskill/ros_action_rskill.py`, `joint_goal_rskill.py`, `pose_goal_rskill.py`, `look_at_rskill.py`; resolver in `packages/openral_rskill_ros/openral_rskill_ros/rskill_runner_node.py`; `RosIntegration` schema in `python/core/src/openral_core/schemas.py`.

**Consequence:** cuMotion plugs in at the **MoveIt planner level**, *below* the rSkill. The rSkill contract (`/move_action` → JointTrajectory) is unchanged.

---

## 3. Integration options (cheapest first)

### Option A — cuMotion as the MoveIt planner plugin (recommended primary)
Configure `isaac_ros_cumotion_moveit` as the planning pipeline behind the **same** `/move_action` server, selected by a **capability flag** (GPU present → cuMotion pipeline; else OMPL).
- **No new rSkill, no new adapter.** `rskill-moveit-{joints,eef-pose,look-at}` get GPU planning for free.
- Lowest surface; matches the issue's step 2 ("alternate plugin behind a capability flag").
- cuMotion runs as its **own ROS 2 node** — natural process isolation for its torch/Warp/CUDA stack (cleaner than the GR00T ZMQ sidecar of ADR-0046, because it is already ROS-native).

### Option B — dedicated `rskill-cumotion-*` for cuMotion-only features
Only when we want capabilities OMPL/MoveGroup can't express:
- **nvblox depth obstacle avoidance** (live ESDF from cameras via `robot_segmenter`),
- **batch IK / multi-seed** reachability,
- cuMotion-native goal types not covered by `MoveGroup`.

Express as `kind: ros_action` with a `RosIntegration` block pointing at cuMotion's action server; add a `goal_builder: "cumotion"` adapter only if the goal IDL differs from `MoveGroup`. Declare GPU need via `min_vram_gb` + `capabilities_required` (the `RobotCapabilities` schema already carries `gpu_vram_gb`, `cuda_compute_capability`, `cuda_toolkit_version`). Set `fallback_skill_id` to the OMPL `rskill-moveit-*` for CPU-only/non-CUDA hosts.

**Recommendation:** ship **A** first (broad, transparent benefit), add **B** later for nvblox/batch-IK.

---

## 4. Interaction with the safety kernel

The architecture is **planner-agnostic by design** — "Python proposes; C++ disposes." cuMotion changes *who plans*, not *who validates*.

```
cuMotion (GPU plan, plan_only) ── trajectory ──▶ replay 1 waypoint/step
   ▼
/openral/candidate_action
   ▼  Python supervisor (openral_safety/supervisor_node.py)
       n_dof, per-joint position limits, per-control-mode bounds → drop + /openral/estop on violation
   ▼  C++ safety kernel (cpp/openral_safety_kernel)
       position/velocity/workspace/force limits, NaN/Inf, ADR-0030/0040 capsule self/world/voxel collision
       → reject (never clamp) + /openral/estop + FailureTrigger on violation
   ▼  HAL → ros2_control → arm
```

What this means for cuMotion:
1. **Same gates, no new bypass.** A cuMotion trajectory is validated exactly like an OMPL one. No safety code changes are *required* to adopt it.
2. **`plan_only` is mandatory.** cuMotion must **not** drive controllers itself; OpenRAL's HAL + safety path is the only actuation route (same constraint as today's `plan_only` MoveIt rSkills). This preserves "no path where a Python crash leaves motors energized."
3. **cuMotion's GPU collision checking is plan-time defense in depth** — it *complements*, never *replaces*, the kernel (exactly as MoveIt FCL does today, per ADR-0030's "planning-time and in-kernel checks are orthogonal").
4. **It can actually *improve* safety outcomes.** The kernel checks **discrete waypoints**; cuRobo optimizes **continuous, dynamics-aware, collision-free interpolants**. Smoother, in-envelope trajectories ⇒ **fewer kernel rejections / E-stops** and less jerk at execution.
5. **Per-robot collision parity.** cuRobo needs a sphere/capsule approximation of the robot; OpenRAL's kernel needs capsule collision models (ADR-0030). The existing **collision-lowering tool (`openral collision lower`, ADR-0030)** is the natural place to *also* emit the cuRobo robot config, keeping plan-time and kernel-time geometry consistent.
6. **Out of scope here:** the **nav `cmd_vel` bypass** the issue flags (Nav2 publishes `BODY_TWIST` straight to HAL, documented as out-of-scope of ADR-0024, relying on Nav2's `velocity_smoother`). cuMotion is arm-only and does **not** touch this. It belongs to workstream (B).

---

## 5. Benefits

| Benefit | Why it matters for OpenRAL |
|---|---|
| **Planning latency** | cuRobo plans in ~tens of ms vs OMPL's hundreds of ms–seconds. Tightens the S1/S2 loop, leaves far more headroom in the replanning ladder and per-skill latency budgets. |
| **Trajectory quality** | Smooth, optimized, dynamics-aware, **collision-free interpolant** (not just collision-free waypoints) → fewer safety-kernel rejections, smoother motion. |
| **Reachability robustness** | GPU batch IK / multi-seed solves succeed where single-seed OMPL+IK stalls. |
| **Perception-aware planning** | nvblox ESDF from depth → live obstacle avoidance in the planning scene; synergizes with the object-detection / octomap epic and ADR-0030 world collision. |
| **Minimal new surface (Option A)** | Drops in under existing `rskill-moveit-*`; no new rSkill, adapter, or safety code for the common path. |
| **Clean license** | cuRobo Apache-2.0 — no posture guard on OpenRAL's Apache-2.0 code (contrast GR00T weights). |
| **Strategic** | Delivers the "GPU planning differentiator" the issue calls for, aimed at OpenRAL's manipulation-heavy core rather than its weaker mobile/humanoid base story. |

---

## 6. Costs, risks, blockers

| Item | Detail | Mitigation |
|---|---|---|
| **GPU hardware** | Needs Ampere+ , CUDA 13/driver 580, Ubuntu 24.04, or Jetson Thor. Reference dev host is an 8 GB Ada — cuMotion's 8 GB floor is **tight alongside a VLA**. | Capability-gate (Option A flag); `min_vram_gb` + `capabilities_required`; OMPL `fallback_skill_id` on CPU/low-VRAM/non-CUDA. |
| **Env / ABI** | cuRobo's torch+Warp+CUDA may clash with VLA envs. | cuMotion is its own ROS node → process isolation is natural (cleaner than ADR-0046's sidecar). |
| **ROS distro / OS** | cuMotion targets Ubuntu 24.04 + recent ROS 2; CUDA 13/driver 580 are new. | Verify against OpenRAL's pinned ROS distro before committing; may need a containerized node. |
| **Per-robot cuRobo config** | Each embodiment needs a kinematics + sphere config. | Extend the ADR-0030 collision-lowering tool to emit cuRobo config from the same URDF/SRDF source of truth. |
| **Asset licenses** | cuRobo *example robot assets* have separate (non-Apache) licenses. | Author OpenRAL robot configs from our own manifests; don't bundle NVIDIA example assets. |
| **Wrong-axis expectation** | Stakeholders may expect this to solve *navigation* per the issue title. | This doc + issue split (workstream A vs B). |

---

## 7. Recommendation

1. **Write an ADR (≈0065)** — required: new external dependency + cross-layer interaction (Skill/Safety/HAL), per issue step 1.
2. **Scope the ADR to Option A** (cuMotion as a capability-gated MoveIt planning plugin under the existing `/move_action` + `rskill-moveit-*`), with **Option B** (`rskill-cumotion-*` for nvblox/batch-IK) as a follow-up.
3. **Keep `plan_only` + the existing safety path** — no kernel changes; cuMotion plan-time collision is additive defense in depth.
4. **Reuse ADR-0030 collision lowering** to generate cuRobo robot configs alongside kernel capsule models (single geometry source).
5. **Split the issue:** move **GPU-MPPI navigation + `cmd_vel` supervised-path** into a separate issue/ADR — that is the original headline and is independent of cuMotion.
6. **Verify before commit:** OpenRAL ROS distro vs cuMotion's Ubuntu 24.04/CUDA 13 requirement; 8 GB co-residency with a VLA on the reference host.

### Precedent ADRs
ADR-0024 (ros-wrapped rSkills) · ADR-0054 (MoveIt goal-builder library) · ADR-0053 (collision-aware approach-to-pose / composing MoveIt with policies) · ADR-0030 + ADR-0040 (geometric collision in the kernel) · ADR-0046 (NVIDIA GPU out-of-process integration precedent).
