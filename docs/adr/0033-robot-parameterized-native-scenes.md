# ADR-0033: Robot-parameterized native sim scenes (Effort 3)

- Status: **Proposed**
- Date: 2026-06-01
- Related: [ADR-0031](0031-sim-real-hal-separation.md) (`build_hal` / manifest `sim.mjcf_uri`);
  [ADR-0023](0023-data-driven-mujoco-hal.md) (`MujocoArmHAL.from_description`,
  `resolve_mjcf_uri`); CLAUDE.md §1.11 (real fixtures, no mocks).

## Context

`openral sim run` drives a `SimRollout` scene; the scene owns physics + `step()` and the
runner is scene-agnostic. There are two scene families:

- **External benchmark** scenes (LIBERO, RoboCasa, MetaWorld, ManiSkill3, gym-aloha,
  gym-pusht, SimplerEnv) own their robot + reward and exist to reproduce published numbers —
  they must stay exactly as they are (the `fixed_robot` CLI guard already isolates them).
- **Native MuJoCo** scenes we author (`so101_box`, `openarm`) compose a task world around a
  robot. Today they **hardcode the robot**: `compose_so101_box_mjcf` calls `_resolve_so101_mjcf()`
  (always the SO-101 MJCF) and splices the arena into it via fixed anchors `<body name="base">`
  + `<body name="gripper">`. Iterating a different robot in the same task needs a new scene file.

Goal: let a native scene take the robot as a **flag** (`robot_id`), resolving the robot MJCF
from the manifest — so creating/iterating a robot + rSkill in a custom scene needs only a
manifest + the scene's task geometry.

## Decision (Option A — robot-parameterized scene; the scene keeps owning physics)

The native scene composer resolves its robot from the **manifest**, not a hardcode, and the
`SimRollout` / `SimRunner` contract is unchanged. Concretely:

1. **Robot MJCF from the manifest.** Replace `_resolve_so101_mjcf()` with resolution of the
   robot's `description.sim.mjcf_uri` (the same source `build_hal(mode="sim")` /
   `MujocoArmHAL.from_description` use, via `openral_hal.resolve_mjcf_uri`). The robot MJCF is
   the base document; the task world is spliced in as today.
2. **Splice anchors from the manifest.** The base-body re-anchor and the wrist-camera mount
   take their body names from the manifest (`base_frame` / the end-effector body) with the
   current `"base"` / `"gripper"` values as defaults — so a robot whose MJCF names differ is
   supported by declaring them, not by forking the composer.
3. **`robot_id` flag.** The scene factory reads `env_cfg.robot_id`, resolves the
   `RobotDescription`, and passes it to the composer. The task world (arena / tube / block /
   overhead camera), the success criterion, and action sizing (from the manifest joints) are
   robot-agnostic; only the robot base + end-effector differ.
4. **`build_hal` is *not* on this path.** `sim run` drives the `SimRollout` directly (no ROS,
   no safety kernel), so the scene owning physics is correct; routing through a HAL would add a
   layer with no payoff here. (`deploy sim` already wraps scenes behind `SimAttachedHAL` for the
   ROS path — that stays.) The shared contract with `build_hal` is the **manifest** (`sim.mjcf_uri`),
   not the HAL object.

### Scope: `so101_box` first (proof of concept)

This ADR first landed the manifest-MJCF resolution on `so101_box` (PoC). "Robot as a flag"
realistically means **any compatible arm** — the flag swaps *which arm*, the task stays. The PoC
finding (below) showed `so101_box` is too coupled to be that vehicle, so a greenfield scene
(`tabletop_push`, see "Greenfield scene" below) now carries the flag. `openarm` (bimanual, per-arm
OSC controllers) remains a follow-up.

## Alternatives considered

- **Option B — scene over `build_hal(mode="sim")`** (HAL owns physics; scene splices objects
  onto the HAL's MuJoCo model via `mjcf_path_override`): truer to "over the sim HAL", but
  inverts physics ownership and adds a HAL layer with no benefit on the no-ROS `sim run` path.
  Rejected for `sim run`; the ROS path already has `SimAttachedHAL`.
- **Keep hardcoded per-robot scene files** — rejected; the whole point is to stop forking a
  scene per robot.

## Implementation finding (PoC outcome)

The composer separation landed: `compose_so101_box_mjcf` resolves the base arm MJCF from the
manifest's `sim.mjcf_uri` (default `None` → SO-101, byte-for-byte unchanged; verified: nq=20,
nu=6 + 17 scene tests pass). **But the PoC surfaced that the scene is coupled to the *so_arm101*
MJCF *schema*, not merely "an arm":** the splice relies on `<body name="base" pos=… quat=…>` +
`<body name="gripper">` + actuators named `"1"`..`"6"`. The SO-100 (`so_arm100`) MJCF — the
nominal "sibling" — has **none of these** (no `base` body with pos/quat, no `gripper` body, no
`"1".."6"` actuators), so it fails the anchor splice. So the only manifest whose `sim.mjcf_uri`
composes cleanly today is `so101_follower` itself.

Conclusion: a *usable* robot flag on an existing tightly-coupled scene needs the splice anchors
+ actuator naming parameterised from the manifest (non-trivial, and per-robot). `so101_box`
therefore **stays `fixed_robot="so101_follower"`** for now — exposing a flag that only accepts
so101 would be a footgun. The manifest-driven MJCF resolution is kept as the foundation.

## Greenfield scene (follow-up landed): `tabletop_push`

The PoC finding above said the true robot-flag vehicle is a greenfield scene built robot-agnostic
from the start, not a retrofit of so_arm101-coupled `so101_box`. That scene now exists:
`python/sim/src/openral_sim/backends/tabletop_push/`, registered **free-axis**
(`@SCENES.register("tabletop_push")`, no `fixed_robot`). It is the realisation of "robot as a flag":

- **MjSpec composition, not regex.** `compose_tabletop_mjcf(description, options, base_pose)` loads
  the robot's manifest MJCF into a `mujoco.MjSpec` and **appends** the task world (table, cube, goal
  disc, overhead + front cameras, light) to its `worldbody`. Appending never reorders the robot's
  joints/actuators, so the composed model keeps the robot's low actuator/qpos indices — exactly the
  1:1 contract `MujocoArmHAL._sim_kwargs_for` already relies on. The free objects' qpos land *after*
  the robot's, so driving the robot by its low actuator indices is correct for any robot.
- **No body-name lookup.** The robot base is re-anchored by mutating the spec's root body
  (`worldbody.bodies[0]`) — an SO-ARM `base`, a Franka `link0` and a UR `base` are handled
  identically. This is the anchor coupling `so101_box` could not escape.
- **Robot-agnostic task + success.** The action/state dim is the compiled model's actuator count
  `nu` (read, not hardcoded); success is geometric on the cube + goal poses only, so it makes no
  gripper/end-effector assumption. Verified end-to-end (compose → reset → step → success) for
  **SO-101, Franka and UR5e** in `tests/sim/test_tabletop_push_scene.py`.
- **Full 6-DOF mounting.** `base_pose:` (honoured by free-axis scenes) sets the root body pos+quat;
  a yaw-only `robot_base_xyz` / `robot_base_yaw_deg` backend-option fallback keeps a minimal YAML
  composable. (Improves on `openarm_robosuite`'s translation-only base mount.)

`sim run` still drives the `SimRollout` directly (Decision §4 stands): the scene owns physics; the
shared contract with `build_hal` is the manifest (`sim.mjcf_uri`), not a HAL object.

## Consequences

- The manifest is now the single source for a robot's sim MJCF across `build_hal`, `deploy sim`,
  and native `sim run` scene composition — the architectural foundation for a robot flag.
- `so101_box` default behavior is unchanged; external benchmark scenes are untouched.
- The robot flag is **realised by `tabletop_push`** (free-axis, any compatible arm). `so101_box`
  intentionally **stays `fixed_robot="so101_follower"`**: making *it* free-axis would still require
  parameterising its so_arm101 splice anchors + actuator naming per robot (a separate, lower-value
  follow-up now that a clean robot-agnostic scene exists).
- New robot-flexible native scenes should follow the `tabletop_push` MjSpec-append pattern rather
  than the `so101_box` regex-splice pattern.
