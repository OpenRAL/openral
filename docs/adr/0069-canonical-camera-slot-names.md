# ADR-0069 — Canonical Camera Slot Names

**Status:** Accepted  
**Date:** 2026-06-22  
**Author:** OpenRAL engineering

---

## Context

Robot manifests (`robots/<id>/robot.yaml`) name their RGB camera sensors with the
`sensors[].name` field.  Before this ADR the names were a mix of:

* Simulation-backend-specific aliases that leaked into the HAL layer:
  `agentview`, `agentview_left`, `agentview_right`, `eye_in_hand`
* Opaque ordinal slots: `camera1`, `camera2`, `camera3`
* Non-uniform semantic names: `overhead`, `topdown`, `corner`

These names appeared throughout scene YAMLs (`cameras: [...]`), rSkill manifests
(`image_preprocessing.aliases`), sim backend code, and tests — making it
impossible to write robot-agnostic tooling and confusing anyone reading a scene
config without first opening the corresponding `robot.yaml`.

Two orthogonal naming spaces exist in the codebase:

| Space | Field | Owner | Example |
|-------|-------|-------|---------|
| **HAL / robot-facing** | `SensorSpec.name` | robot manifest | `top` |
| **Checkpoint-facing** | `SensorSpec.vla_feature_key` | rSkill checkpoint | `observation.images.camera1` |

The `vla_feature_key` is **frozen per trained checkpoint** and must not change.
Only the `name` field is standardised here.

---

## Decision

### Canonical vocabulary

| Name | Meaning |
|------|---------|
| `front` | Fixed front/third-person workspace overview (e.g. MetaWorld sawyer corner view) |
| `top` | Overhead / top-down view, world-fixed or head-mounted (preferred for cameras placed above the workspace looking down) |
| `head` | Head-mounted camera on humanoid robots |
| `shoulder_left` | Body-mounted camera on the left shoulder of a mobile manipulator (anatomical placement; preferred over `front_left`) |
| `shoulder_right` | Body-mounted camera on the right shoulder of a mobile manipulator |
| `wrist` | Wrist / end-effector camera, single arm |
| `wrist_left` | Wrist camera, left arm (bimanual) |
| `wrist_right` | Wrist camera, right arm (bimanual) |
| `side_left` | Left-lateral workspace view (reserved; no current robot) |
| `side_right` | Right-lateral workspace view (reserved; no current robot) |

**Rule:** Every `SensorSpec.name` for an RGB or depth camera in a
`robots/*/robot.yaml` must be drawn from this table or be a qualified extension
(`<base>_<qualifier>`, e.g. `front_depth`).

**Rationale — `top` over `front` for overhead cams.** `top` is used when the
camera is positioned overhead looking at the workspace (e.g. so100/so101
overhead OAK-D, franka_panda LIBERO agentview at z≈1.35 m). `front` is reserved
for genuine front-facing workspace views (e.g. MetaWorld sawyer corner).

**Rationale — `shoulder_left/right` over `front_left/right` for mobile bases.**
Body-mounted cameras on a mobile manipulator (PR2, Fetch, TIAGo, panda_mobile)
are described by anatomical position (`shoulder_*`) rather than direction
(`front_*`). This (a) generalises across robot orientations — a turned base
doesn't change a "shoulder" camera into a "rear" one, (b) matches real
mobile-manipulator vocabulary, and (c) avoids the ambiguity between "camera
facing the front-left direction" and "left-side camera on the front of the
robot".

### Changes to robot manifests

| Robot | Old name | New name |
|-------|----------|----------|
| `aloha_agilex` | `camera1` | `top` |
| `aloha_agilex` | `camera2` | `wrist_left` |
| `aloha_agilex` | `camera3` | `wrist_right` |
| `so100_follower` | `front` | `top` |
| `so101_follower` | `front` | `top` |
| `franka_panda` | `agentview` | `top` |
| `panda_mobile` | `agentview_left` | `shoulder_left` |
| `panda_mobile` | `agentview_right` | `shoulder_right` |
| `widowx` | `overhead` | `top` |
| `sawyer` | `corner` | `front` |
| `pusht_2d` | `topdown` | `top` |
| `openarm` | `base` | `top` (removes redundant `sim_camera_name: "top"`) |
| `openarm` | `left_wrist` | `wrist_left` |
| `openarm` | `right_wrist` | `wrist_right` |

Already-canonical names (no change): `wrist` (so100, so101, franka_panda wrist),
`front` (sawyer / MetaWorld corner view), `top` (aloha_bimanual), `head` (gr1,
google_robot).

### Scene `cameras` lists

`SceneSpec.cameras` lists **robot sensor names** (HAL-facing, now canonical).  
Every scene YAML is updated to match the renamed sensors in the relevant robot
manifest.

### Sim backends

Backends that previously hardcoded `camera1`/`camera2`/… as observation image
keys now key their output by `scene.cameras[i]`, falling back to `f"camera{i+1}"`
when the scene declares no explicit cameras.  The existing
`SimSensorBridge._frame_for_camera` fallback (try `vla_feature_key` slot first,
then sensor `name`) ensures the HAL bridge is unaffected regardless of backend
key convention.

### rSkill `image_preprocessing.aliases`

Alias keys name the *scene* camera (now canonical).  All aliases that previously
referenced `camera1`, `camera2`, `camera3` are renamed to match the canonical
sensor names of the target embodiment.

### RoboCasa / RLDX policy adapters

The RLDX adapter's `camera_keys` resolution is updated to fall back to
`scene.cameras[:2]` when `vla.extra.camera_keys` is not overridden.  This keeps
scene YAMLs as the single source of truth for camera order.

---

## Consequences

### Positive

* Robot-agnostic tooling can name sensors without robot-specific knowledge.
* Scene YAMLs are self-documenting (`cameras: ["front", "wrist"]` vs
  `cameras: ["camera1", "camera2"]`).
* Eliminates the simulation-backend-specific vocabulary from the HAL API surface.

### Negative / risks

* Wide diff (~50 files).  Mitigated by the mechanical nature and comprehensive
  test coverage updated in the same PR.
* Any rSkill published to HF Hub **before this ADR** with
  `image_preprocessing.aliases: {camera1: …}` must be re-published or supplied
  with a compatibility shim.  All in-tree rSkills are updated in the same commit.

### Invariants (non-changes)

* `SensorSpec.vla_feature_key` is **unchanged** — it is checkpoint-owned.
* `sensors_required[].vla_feature_key` in rSkill manifests is unchanged.
* The `sim_camera_name` escape hatch remains in `SensorSpec` for robots whose
  MJCF camera name cannot match the canonical sensor name for legacy reasons.
