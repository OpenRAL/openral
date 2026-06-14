# Robot Bring-Up Blueprint

Use this reference when the task involves a real or simulated robot integration. It is a checklist and context map, not a replacement for `CLAUDE.md` or the schema files.

## Intake Checklist

Collect these before editing robot, HAL, or deployment files:

| Area | Required facts |
| --- | --- |
| Identity | robot ID, display name, vendor, embodiment kind, intended OpenRAL layer surfaces |
| Kinematics | base frame, joints, parent/child links, axes, limits, mimic/passive joints |
| Actuation | actuators, control modes, command units, update rates, gripper semantics |
| Safety | workspace, speed/force/torque limits, payload, deadman, E-stop path, brake behavior |
| Sensors | camera/depth/IMU/F/T/tactile IDs, frames, calibration, VLA feature keys |
| Transport | serial port, IP/FCI port, SDK, ROS driver, permissions, udev needs |
| Simulation | MuJoCo/robosuite/other asset path, digital-twin status, `sim:` block support |
| rSkills | compatible embodiment tags, sensors required, action/state contracts, license gates |
| Validation | sim command, deploy-sim command, HIL availability, dashboard/trace path |

Unknown safety, limits, frame, or control-mode facts must remain explicit blockers.

## Manifest Blueprint

`robots/<id>/robot.yaml` should cover:

- `name`, `embodiment_kind`, and `base_frame`.
- `joints[]` with limits, links, axes, and actuators.
- `end_effectors[]` with gripper/hand semantics and limits.
- `sensors[]` with modalities, frames, and VLA feature keys.
- `capabilities` including control modes and `embodiment_tags`.
- `safety` including workspace, speed, force/torque, and deadman requirements.
- `observation_spec` and `action_spec` matching runtime contracts.
- Optional `sim:` block for MuJoCo digital-twin wiring.

Use nearby robot examples before inventing a new structure:

- `robots/so100_follower/robot.yaml` for compact single-arm + gripper patterns.
- `robots/franka_panda/robot.yaml` for 7-DoF arm patterns.
- `robots/aloha_bimanual/robot.yaml` for bimanual patterns.
- `robots/openarm/robot.yaml` for bimanual/passive-finger simulation details.
- `robots/g1/robot.yaml` and `robots/h1/robot.yaml` for humanoid/floating-base patterns.

## HAL Decision Tree

1. Manifest-only is enough if existing sim/HAL machinery can consume the robot.
2. Extend an existing HAL family if the transport and control semantics match.
3. Add a new HAL adapter only when the hardware SDK, ROS driver, or control interface requires it.
4. Use ROS 2 lifecycle nodes for stateful hardware integration.
5. Keep E-stop, deadman, and safe-action subscription behavior present at every actuation boundary.

Never let rSkill, scene, reasoner, or dashboard code call hardware APIs directly.

## Validation Ladder

Run the narrowest safe check first:

```bash
openral detect
openral rskill check <rskill-id> --robot robots/<id>/robot.yaml
openral sim list
openral deploy sim --config scenes/<scene>.yaml --rskill rskills/<skill>
openral deploy list
openral deploy run --config deployments/<robot_task>.yaml
```

Only run real-hardware commands when the user explicitly asks, the robot is powered and clear, and E-stop is physically available.

## rSkill Compatibility Checklist

- Robot `embodiment_tags` intersect rSkill tags.
- Sensors satisfy `sensors_required`, including feature keys and camera geometry.
- Actuators satisfy required control modes and action semantics.
- `state_contract` and `action_contract` match the robot observation/action specs.
- Runtime can fit local GPU/device constraints.
- License gates are acceptable for the intended use.
- Sim-only tags are not treated as real-hardware proof.

## Tests and Docs Checklist

- Manifest load test uses the real YAML fixture.
- HAL lifecycle test uses real interfaces or HIL gating, not mocks.
- Sim registration test covers new `sim:` or robot registry behavior.
- HIL tests are under `tests/hil/` and gated by lab labels.
- Robot README explains setup, sensors, safety, and compatible rSkills.
- `docs/reference/robots.md` updated for newly supported robots.
- the matching `docs/methods/` file updated for public symbols.
- Repo state map updated when the package/surface changes.

## Hard Stops

- Unknown safety limit or E-stop behavior.
- Missing control-mode semantics.
- Missing physical dimensions or frame conventions needed for safety or sim.
- Request to bypass E-stop, deadman, watchdog, capability, or safety checks.
- Real-hardware command without explicit user intent and physical readiness.