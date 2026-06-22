# ADR-0066 — DeployScene owns its MJCF composition (robot / scene / rSkill separation)

- **Status:** Accepted 2026-06-22. Separates the OpenArm deploy environment into
  three independent concerns — the robot, the scene, the rSkill — by moving the
  tabletop-arena composition off the robot manifest.
- **Date:** 2026-06-22
- **ADR number:** `0066`. The integer is not load-bearing — cross-refs use
  filenames. (`0065` is taken by a separate in-flight branch; numbering around it
  is fine.)
- **Related:**
  - issue #191 Phase 3b — `scene_defaults.composition` on the robot manifest:
    the manifest-driven HAL node composes the MJCF and threads it in as the HAL's
    `mjcf_path`. This ADR keeps that node mechanism but moves *where the
    composition is declared* from the robot manifest to the scene.
  - ADR-0044 — `scene_defaults.top_camera`: the overview-camera pose. Moves to
    the scene alongside the composition (it is a scene camera, not a robot
    sensor mount).
  - ADR-0034 — scene-attach (`SimAttachedHAL`). Orthogonal: a scene-attach robot
    builds the scene's `SimRollout` directly; this ADR is about the
    *compose-a-bare-twin* path.

## Context

`openral deploy sim --config scenes/deploy/openarm_tabletop.yaml` brings up the
OpenArm on a tabletop with three cubes and a drawer. That manipulation arena was
declared in the **robot manifest** (`robots/openarm/robot.yaml`
`scene_defaults.composition`, mirrored in the `OPENARM_DESCRIPTION` HAL
constant), so the robot description carried a specific *scene*: a table, cubes, a
drawer, and an overview camera. That conflates three things that should be
independent:

- the **robot** (kinematics, sensors, safety, HAL) — reusable across any scene;
- the **scene** (the table + props + overview camera) — one environment among
  many the robot could be dropped into;
- the **rSkill** (the policy, e.g. pi05-openarm) — already separate, referenced
  by id and declaring only its `sensors_required`.

Because the arena lived on the robot, every consumer of the OpenArm manifest
inherited the tabletop, and the same `scene_defaults` had to serve both the
deploy path and the sim/eval path. A second OpenArm scene would have had to fight
the manifest's baked-in arena.

## Decision

1. **`DeployScene` gains an optional `composition: SceneComposition`.** A deploy
   scene declares the composer that builds its environment. `openral deploy sim`
   reads it and forwards it to the manifest-driven HAL node as a
   `scene_composition_json` ROS parameter; the node composes the MJCF and builds
   a bare twin off the result. The scene's composition takes **precedence** over
   the robot manifest's `scene_defaults.composition` (kept as a back-compat
   fallback, now unused by any in-tree robot).

2. **The OpenArm tabletop arena moves to the scenes that own it.** The deploy
   scene (`scenes/deploy/openarm_tabletop.yaml`) carries the `composition`
   block (table + cubes + drawer + overview camera, including the `top_camera_*`
   pose). The sim/eval scene (`scenes/sim/openarm_tabletop.yaml`) carries the
   `top_camera_*` pose in its `backend_options` (the path the
   `openarm_robosuite` env already reads). `robots/openarm/robot.yaml` and
   `OPENARM_DESCRIPTION` drop `scene_defaults` entirely.

3. **The robot manifest describes only the robot; the rSkill stays separate.**
   The OpenArm manifest now carries no scene config. The pi05-openarm rSkill is
   already independent (external `rskill://` id, `sensors_required` only) — no
   change. Robot / scene / rSkill are three separate artifacts.

## Consequences

**Positive**
- A robot manifest is scene-agnostic and reusable; adding a second OpenArm scene
  is a new scene file, not a manifest edit.
- The deploy scene is self-contained — its arena travels with it.
- `DeployScene.composition` is generic: any robot's deploy scene can compose its
  own environment without touching the robot manifest.

**Negative / costs**
- Additive change to the core `DeployScene` schema (new optional field) — JSON
  Schema export + repo-state map + fuzz coverage updated; `schema_version`
  unchanged (backward-compatible).
- Two ways to declare composition coexist (scene `composition` wins, manifest
  `scene_defaults.composition` is the fallback). The fallback is retained for
  back-compat but is now used by no in-tree robot; it can be removed once no
  manifest relies on it.

## Alternatives considered

- **Keep `scene_defaults.composition` on the robot manifest (status quo).**
  Rejected: couples the robot to one scene; a second scene can't override the
  baked-in arena cleanly; conflates robot and scene.
- **Encode the scene path and have the node load the `DeployScene`.** Rejected
  in favour of forwarding the `SceneComposition` as JSON — the node stays
  decoupled from scene-file loading and the parameter is self-describing.
