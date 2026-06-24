# Task-Space Definitions Audit — robots × rSkills × scenes

> Audit date: 2026-06-24. Scope: how the **action space** (joint vs end-effector
> vs gripper vs base) and **state/observation space** are declared across the
> three asset layers — `robots/`, `rskills/`, `scenes/` — and whether those
> declarations actually connect.
>
> **TL;DR.** The three layers do not share a single task-space contract. A robot
> declares `supported_control_modes` + end-effectors + (sometimes) a gripper
> joint; an rSkill declares `state_contract.dim` + `action_contract.dim` +
> (sometimes) a `representation`; a scene declares *nothing* about the task space
> and probes `action_dim` from the live sim env at rollout time. They are wired
> together only by (a) `embodiment_tags` string matching, (b) dimension numbers
> happening to line up, and (c) runtime translation inside the sim/HAL adapters
> that is invisible in every manifest. This document pulls out every defined
> task space and proposes one generalized contract to populate everywhere.

---

## 1. The canonical vocabulary (already in `openral_core.schemas`)

These enums exist and are the right primitives — they are just under-used by the
manifests.

**`ControlMode`** (action interface; `CONTROL_MODE_TO_UINT8` wire encoding):

| # | mode | space | width |
|---|------|-------|-------|
| 0 | `joint_position` | joint | n_joints |
| 1 | `joint_velocity` | joint | n_joints |
| 2 | `joint_torque` | joint | n_joints |
| 3 | `joint_trajectory` | joint | n_joints |
| 4 | `cartesian_pose` | EE | 6 (abs pose) |
| 5 | `cartesian_delta` | EE | 6 (delta) |
| 6 | `cartesian_twist` | EE | 6 (velocity) |
| 7 | `body_twist` | base | 6 |
| 8 | `foot_placement` | base | discrete |
| 9 | `gripper_binary` | gripper | 1 |
| 10 | `gripper_position` | gripper | 1 |
| 11 | `dex_hand_joint` | hand | n_finger_dof |
| 12 | `composite_mode` | multiplexer | 1 |

**`StateRepresentation`**: `joint_positions`, `eef_pos_axisangle`,
`eef_pos_euler`, `eef_pos_quat`, `eef_pos_axisangle_gripper`.

**`ActionRepresentation`**: `joint_positions`, `joint_velocities`,
`delta_ee_6d_plus_gripper`, `delta_ee_6d`, `cartesian_pose`.

**The bridge** is `control_modes_for_representation()` (ADR-0036): it maps an
rSkill's `ActionRepresentation` → the set of `ControlMode`s the target robot must
advertise. This is the *only* code path that statically connects an rSkill's task
space to a robot's task space. It can only fire when `representation` is set.

```
joint_positions            → {joint_position}
joint_velocities           → {joint_velocity}
delta_ee_6d                → {cartesian_delta}
delta_ee_6d_plus_gripper   → {cartesian_delta, gripper_position}
cartesian_pose             → {cartesian_pose}
```

---

## 2. Robots — declared task space (17 robots)

| robot | kind | control_modes | arm J | grip J | end-effectors | bimanual | locomotion |
|-------|------|---------------|:----:|:------:|---------------|:--------:|------------|
| aloha_agilex | bimanual | `joint_position` | 12 | 2 | 2× parallel_gripper | ✔ | none |
| aloha_bimanual | bimanual | `joint_position` | 12 | 2 | 2× parallel_gripper | ✔ | none |
| franka_panda | manipulator | `joint_position` | 7 | 1 | parallel_gripper | ✘ | none |
| g1 | humanoid | `joint_position` | 29 | 0 | — | ✔ | bipedal |
| google_robot | mobile_manip | `joint_position`, `cartesian_pose` | 7 | 1 | parallel_gripper | ✘ | wheeled |
| **gr1** | humanoid | **`[]` (empty)** | 17 | 0 | 2× dexterous_hand (n_dof=**None**) | **None** | bipedal |
| h1 | humanoid | `joint_position` | 19 | 0 | — | ✔ | bipedal |
| openarm | bimanual | `joint_position` | 14 | 2 | 2× parallel_gripper | ✔ | none |
| panda_mobile | mobile_manip | `body_twist`, `joint_position`, `joint_velocity`, `composite_mode` | 10 | 1 | parallel_gripper (n_dof=**None**) | **None** | wheeled |
| pusht_2d | manipulator | `cartesian_pose` | 2 | 0 | — | ✘ | none |
| rizon4 | manipulator | `joint_position` | 7 | 0 | tool (n_dof=0) | ✘ | none |
| sawyer | manipulator | `joint_position` | 7 | 1 | parallel_gripper | ✘ | none |
| so100_follower | manipulator | `joint_position`, `gripper_position` | 5 | 1 | parallel_gripper | ✘ | none |
| so101_follower | manipulator | `joint_position`, `gripper_position` | 6 | **0** | parallel_gripper | ✘ | none |
| ur10e | manipulator | `joint_position` | 6 | 0 | tool (n_dof=0) | ✘ | none |
| ur5e | manipulator | `joint_position` | 6 | 0 | tool (n_dof=0) | ✘ | none |
| widowx | manipulator | `joint_position`, `cartesian_pose` | 5 | 1 | parallel_gripper | ✘ | none |

---

## 3. rSkills — declared task space (32 packages)

### 3a. Actuating policies (VLA / ros_action)

| rSkill | family | state dim | action dim | representation | actuators_required | embodiment_tags |
|--------|--------|:---------:|:----------:|----------------|--------------------|-----------------|
| act-aloha | act | 14 | 14 | **None** | joint_position | aloha |
| act-aloha-insertion | act | 14 | 14 | **None** | joint_position | aloha |
| smolvla-robotwin | smolvla | 14 | 14 | **None** | joint_position | aloha_agilex |
| act-libero | act | 8 | 7 | delta_ee_6d_plus_gripper | joint_position | franka_panda |
| smolvla-libero | smolvla | 8 | 7 | delta_ee_6d_plus_gripper | joint_position | franka_panda |
| pi05-libero-nf4 | pi05 | 8 | 7 | delta_ee_6d_plus_gripper | joint_position | franka_panda |
| xvla-libero | xvla | 8 | 7 | delta_ee_6d_plus_gripper | joint_position | franka_panda |
| rldx1-ft-libero-nf4 | rldx | 8 | 7 | delta_ee_6d_plus_gripper | joint_position | franka_panda |
| molmoact2-libero-nf4 | molmoact2 | 8 | 7 | delta_ee_6d_plus_gripper | joint_position | franka_panda |
| gr00t-n17-libero | gr00t | 8 | 7 | delta_ee_6d_plus_gripper | joint_position | franka_panda |
| openvla-oft-simpler-widowx-nf4 | openvla | 8 | 7 | delta_ee_6d_plus_gripper | joint_position | widowx |
| rldx1-ft-simpler-widowx-nf4 | rldx | 8 | 7 | delta_ee_6d_plus_gripper | joint_position | widowx |
| smolvla-maniskill-franka | smolvla | 9 | 8 | **None** | joint_position | franka_panda |
| smolvla-metaworld | smolvla | 4 | 4 | **None** | joint_position | sawyer |
| molmoact2-so101-nf4 | molmoact2 | 6 | 6 | **None** | joint_position | so100/so101_follower |
| diffusion-pusht | diffusion | 2 | 2 | **None** | cartesian_pose | pusht |
| 3d-diffuser-actor-rlbench | diffuser_actor | None | 8 | **None** | cartesian_pose | franka_panda |
| pi05-robocasa365-human300-nf4 | pi05 | 16 | 12 | **None** | joint_position | panda_mobile |
| rldx1-ft-rc365-nf4 | rldx | 16 | 12 | **None** | joint_position | panda_mobile |
| rldx1-ft-gr1-nf4 | rldx | 29 | 29 | **None** | joint_position | gr1 |
| template | pi05 | 8 | 8 | **None** | joint_position | franka_panda |
| rskill-moveit-joints | — | — | — | — | joint_position | franka_panda, ur5e, ur10e, so100_follower, openarm, rizon4, sawyer, widowx |
| rskill-moveit-eef-pose | — | — | — | — | joint_position | (same 8) |
| rskill-moveit-look-at | — | — | — | — | joint_position | (same 8) |
| rskill-nav2-navigate-to-pose | — | — | — | — | body_twist | mobile_base |

### 3b. Non-actuating skills (no task space — perception / reasoning / reward)

`locateanything-3b-nf4`, `omdet-turbo-indoor`, `omdet-turbo-locator`,
`rtdetr-coco-r18`, `rtdetr-v2-r50vd` (detectors); `qwen35-4b-nf4` (VLM/S2);
`robometer-4b` (reward/S2). These have no `state_contract` / `action_contract` /
`actuators_required` and are correctly out of scope for task-space matching.

---

## 4. Scenes — declared task space (none)

Scene YAMLs (`scenes/sim/`, `scenes/benchmark/`, `scenes/deploy/`) declare
**zero** task-space fields. A scene carries `scene.id`, `backend`, observation
height/width/cameras, and a `task` block (instruction, success_key, max_steps),
plus an optional `robot_id`. The action/observation dimensionality is resolved
**at runtime** by the backend adapter, e.g.:

- `libero.py::action_dim` → `sum(r.action_dim for r in robosuite robots)`
- `maniskill3.py::action_dim` → `_action_dim_from_space(env.action_space)`
- `isaac_sim.py::action_dim` → probed from the sidecar (`env.action_dim`)
- `aloha.py` → `obs["agent_pos"]` shape from the gym env

So the scene's task space is whatever the live simulator hands back. There is no
static field a robot or rSkill manifest can be checked against before a rollout.

---

## 5. The connection map — what actually links the three layers

```
   rSkill ──embodiment_tags (string)──► Robot.capabilities.embodiment_tags
   rSkill ──actuators_required.kind───► Robot.supported_control_modes   (string ∈ set)
   rSkill ──action_contract.representation──► control_modes_for_representation() ──► Robot.supported_control_modes
   rSkill ──state_contract.dim / action_contract.dim──► (numeric) ──► Scene env action_dim   [runtime only]
   Scene  ──robot_id──► Robot                                          (sim physics)
   Scene  ──(nothing)──► task space                                    [probed at runtime]
```

There is **no shared, named task-space object**. The closest thing to a
cross-layer contract is `control_modes_for_representation()`, and it is bypassed
whenever `representation` is unset (which is the majority — see §6.1).

---

## 6. Findings — where the connection breaks

### 6.1 `action_contract.representation` is unset on most actuating rSkills
Only the 9 `delta_ee_6d_plus_gripper` checkpoints (LIBERO + SIMPLER-widowx
family) set it. Every joint-space skill (aloha ×3, maniskill, metaworld, so101,
gr1, robocasa ×2, template) and pusht/rlbench leave `representation: None`. With
it unset, ADR-0036's `control_modes_for_representation()` gate cannot run, so the
deploy-path palette can't verify the robot advertises the right control modes —
matching silently falls back to `embodiment_tags` + raw dim equality.

### 6.2 `actuators_required.kind` contradicts `representation`
Every LIBERO skill declares `actuators_required: joint_position` **and**
`representation: delta_ee_6d_plus_gripper`. But that representation maps to
`{cartesian_delta, gripper_position}` — *not* `joint_position`. The two fields
describe different things (one matches the physical robot, the other the
checkpoint's native vector) and nothing reconciles them. This is the single
biggest source of confusion in the audit: the manifest asserts both "I need a
joint-position robot" and "I emit EE deltas + gripper," and the conversion only
exists inside the sim adapter.

### 6.3 Robots with grippers don't advertise gripper control modes
9 robots have a gripper end-effector; only **so100** and **so101** list a
`gripper_position` control mode. franka_panda, both aloha, openarm, google_robot,
sawyer, widowx, panda_mobile all expose a `parallel_gripper` EE (and most a
gripper joint) but advertise **no** gripper mode. So a skill whose representation
implies `gripper_position` cannot be matched to them through the palette gate.

| robot | gripper joint | gripper EE | declares gripper mode |
|-------|:---:|:---:|:---:|
| franka_panda, aloha ×2, openarm, google_robot, sawyer, widowx, panda_mobile | ✔ (most) | ✔ | ✘ |
| so100_follower | ✔ | ✔ | ✔ |
| so101_follower | ✘ | ✔ | ✔ |
| g1, h1, ur5e, ur10e, rizon4, pusht_2d | ✘ | ✘ / tool | ✘ |
| gr1 | ✘ | dexterous_hand | ✘ |

### 6.4 The gripper dimension is modeled three different ways
1. **Tagged joint** (`role: gripper`): franka, aloha ×2, openarm, panda_mobile, so100, sawyer, widowx, google_robot.
2. **Folded into the arm joint count, untagged**: so101 (6 joints incl. gripper, 0 tagged) — directly inconsistent with its near-twin so100.
3. **Absent**: g1/h1 (no EE at all); gr1 (dexterous_hand, `n_dof: None`).

### 6.5 `gr1` robot is internally broken for matching
`supported_control_modes: []` and `supported_vla_embodiments: None`, yet
`rldx1-ft-gr1-nf4` targets embodiment tag `gr1` with a 29-D `joint_position`
action. The skill side is fine; the robot advertises nothing to match against.

### 6.6 `n_dof: None` on actuated end-effectors
`gr1` dexterous hands and `panda_mobile` gripper carry `n_dof: None` instead of
an int. `EndEffectorSpec.n_dof` defaults to 1; an explicit `None` defeats any
width arithmetic that sums EE DoF into the action vector.

### 6.7 `state_contract.layout` present on only 5 skills
Only openvla-oft, pi05-rc365, rldx1 (gr1/rc365/widowx) declare a `layout`.
Without it, `StateContractBindings` (the per-slice → control-mode routing,
symmetric to the action side) can't be derived, so state assembly stays a magic
flat vector keyed only by `dim`.

### 6.8 Scenes can't be statically validated against either layer
Because the scene task space is runtime-probed (§4), a robot/skill/scene
mismatch (e.g. a 7-D skill against an 8-D env) is only discoverable by running
the rollout, not by loading manifests.

---

## 7. Proposal — one generalized task-space contract to populate everywhere

The primitives already exist; the fix is to make the task space an **explicit,
shared object** on all three layers and require it.

### 7.1 Canonical `TaskSpace` block (new shared schema)
A named, composable description of an action/observation vector built from typed
*segments*, each tagging a `ControlMode` + width + target end-effector/frame:

```yaml
task_space:
  action:
    - segment: arm        # 6× cartesian_delta on panda_hand
      mode: cartesian_delta
      width: 6
      target: panda_hand
    - segment: gripper
      mode: gripper_position
      width: 1
      target: panda_hand
    total_dim: 7
  state:
    representation: eef_pos_axisangle_gripper
    total_dim: 8
```

This generalizes uniformly:
- **joint** robots → one `joint_position` segment of width `n_joints` (+ gripper segment).
- **EE** policies (LIBERO) → `cartesian_delta` (6) + `gripper_position` (1) = 7.
- **bimanual** (aloha) → two arm segments + two gripper segments = 14.
- **mobile** (panda_mobile) → `body_twist` (3) + arm + gripper + `composite_mode` (1) = 12.
- **humanoid** (gr1) → `dex_hand_joint` segments per hand + body.
- **pusht** → one `cartesian_pose` segment of width 2.

`total_dim` is the existing `action_contract.dim`; the segments are what's
missing — they make "is the gripper a dimension?" and "joint vs EE?" explicit and
machine-checkable instead of implied.

### 7.2 Make the three layers share it
- **Robots**: derive the *supported* `TaskSpace` segments from joints + EE +
  control modes; require gripper-bearing robots to advertise `gripper_position`.
- **rSkills**: replace the `representation`/`actuators_required` split with the
  segmented `task_space.action` (keep `representation` as a derived shorthand).
  Reconcile §6.2 by stating once, per segment, what the checkpoint emits.
- **Scenes**: declare the expected `task_space` so the runtime-probed env dim can
  be asserted against it at load time (§6.8), not at rollout.

### 7.3 One validator, three layers
A single check: `skill.task_space.action` segments ⊆ robot supported segments,
and `== scene.task_space.action`. This replaces the current three-way implicit
wiring (string tags + dim equality + adapter magic) with one explicit contract.

### 7.4 Immediate cleanups (independent of the schema work)
1. Set `representation` on every actuating rSkill (§6.1).
2. Fix `gr1` robot: real `supported_control_modes` + `embodiment_tags` (§6.5).
3. Add `gripper_position` to gripper-bearing robots (§6.3).
4. Make so101 model its gripper like so100 (tagged joint) (§6.4).
5. Replace `n_dof: None` with real ints on gr1 / panda_mobile (§6.6).
6. Resolve the `actuators_required` vs `representation` contradiction (§6.2).

---

## 8. Quick reference — task-space families observed

| dim | space | examples |
|:---:|-------|----------|
| 2 | 2-D cartesian (planar pusher) | pusht |
| 4 | 3 EE-delta + 1 gripper | metaworld (sawyer) |
| 6 | 6 joint (incl. gripper) | so101 |
| 7 | 6 EE-delta + 1 gripper | LIBERO ×7, SIMPLER-widowx ×2 |
| 8 | 7 joint + 1 gripper / 7 cartesian_pose+grip | maniskill-franka, rlbench, template |
| 12 | base + arm + gripper + composite | robocasa (panda_mobile) ×2 |
| 14 | 2× (6 joint + 1 gripper) | aloha ×3 |
| 29 | full-body joint | gr1 humanoid |
