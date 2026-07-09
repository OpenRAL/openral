---
name: robot-bring-up-guide
description: 'Use when adding or deploying a new robot in OpenRAL: RobotDescription, robots/<id>/robot.yaml, HAL adapter, sensors, safety envelope, sim block, MuJoCo digital twin, deploy run config, HIL tests, rSkill compatibility, openral detect, openral deploy, or dashboard bring-up.'
argument-hint: 'Robot name/spec, existing manifest path, HAL target, sensors, or deployment goal'
---

# Robot Bring-Up Guide

Guide a new robot from hardware facts to an OpenRAL-compatible manifest, sim path, deployment config, tests, and docs. Keep the work typed, safety-first, and fixture-backed.

## When to Use

- Adding `robots/<id>/robot.yaml` or a robot README.
- Bringing up a HAL adapter, digital twin, HIL path, or deployment YAML.
- Mapping sensors, joints, end-effectors, control modes, safety limits, and embodiment tags.
- Checking whether an existing rSkill can run on a robot.
- Debugging `openral detect`, `openral deploy sim`, `openral deploy run`, or dashboard bring-up for a robot.

## Required Context

Read only what matches the target:

1. `CLAUDE.md` for layer boundaries, safety, tests, ROS 2, and docs rules.
2. `docs/tutorials/deploy/deploy-run-and-dashboard.md` for real-hardware deployment flow.
3. `docs/tutorials/sim/create-a-sim-environment.md` for robot manifests and sim registration.
4. `docs/reference/robots.md`, `docs/reference/sensors_landscape.md`, and `docs/reference/vla_compatibility.md`.
5. Existing robot examples under `robots/`, especially `so100_follower`, `franka_panda`, `ur5e`, `aloha_bimanual`, `openarm`, `g1`, and `h1`.
6. Existing HAL packages under `packages/openral_hal_*` and Python HAL helpers before adding new code.
7. `python/core/src/openral_core/schemas.py` for `RobotDescription`, capabilities, safety, observation, action, and sim schemas.

## References

- Load [bring-up-blueprint.md](./references/bring-up-blueprint.md) when you need the intake checklist, manifest blueprint, HAL decision tree, validation ladder, rSkill compatibility checklist, or tests/docs checklist.
- Do not create a `scripts/` directory unless there is a new helper that is specific to this skill and not already a canonical OpenRAL command.

## Workflow

1. Gather robot facts from real sources.
   - Joints, links, limits, actuators, end-effectors, payload, force/torque limits, frames, base frame, sensors, transport, SDK, ROS driver, control modes, and E-stop behavior.
   - Mark unknown physical constants as blockers. Never invent frame IDs, ports, dimensions, torque limits, workspace limits, camera intrinsics, or serial IDs.

2. Create or update `RobotDescription`.
   - Add `robots/<id>/robot.yaml` and `robots/<id>/README.md`.
   - Fill identity, joints, end-effectors, sensors, capabilities, safety, observation spec, action spec, and optional `sim` block.
   - Choose canonical embodiment tags from existing registry/docs where possible.
   - Ensure safety limits are conservative and traceable.

3. Decide the HAL path.
   - Reuse an existing HAL family if the robot fits.
   - Add a HAL package only when required by hardware or SDK boundaries.
   - Keep HAL in layer 0 and do not let rSkill or reasoner code talk directly to hardware.
   - Stateful HAL components should be ROS 2 lifecycle nodes with E-stop and safe-action behavior preserved.

4. Wire sensors and world state.
   - Map every sensor to a typed `SensorSpec` and VLA feature key where needed.
   - Use the sensor catalog and existing adapters before creating a new one.
   - Confirm timestamps, frames, QoS, calibration, and staleness expectations.

5. Add sim support before hardware when possible.
   - Use the manifest `sim:` block for robots supported by the shared MuJoCo HAL path.
   - For custom scenes, coordinate with the Scene Creator skill.
   - Verify registration with `openral sim list` and a small `openral deploy sim` or `openral sim run` path.

6. Create deployment config guidance.
   - Deployment YAMLs under `deployments/` encode lab-specific ports, serials, camera IDs, task, rSkill, and safety overrides.
   - Do not commit secrets, private IP assumptions, or lab-specific values unless the repo convention allows that fixture.
   - Use `openral deploy sim` before `openral deploy run`.

7. Check rSkill compatibility.
   - Use `openral rskill check <id> --robot <robot.yaml>` where available.
   - Match embodiment tags, sensors, actuators, control modes, state/action contracts, runtime, GPU, and license gates.
   - Do not claim compatibility from a tag match alone if camera geometry or action semantics differ.

8. Add tests and docs.
   - Add real manifest load tests, HAL lifecycle tests, sim tests, and HIL gates as appropriate.
   - No mocks for robot manifests or actuation paths; unavailable hardware should be HIL-gated or skipped with a concrete reason.
   - Update `docs/reference/robots.md`, relevant package READMEs, the matching `docs/methods/` file, and repo state map when the surface changes.

## Safe Command Patterns

```bash
openral detect
openral rskill check <rskill-id> --robot robots/<id>/robot.yaml
openral sim list
openral deploy sim --config scenes/<scene>.yaml --rskill rskills/<skill>
openral deploy list
openral deploy run --config deployments/<robot_task>.yaml
```

Run real-hardware commands only when the user explicitly asks and the robot is physically ready with E-stop access.

## Output Checklist

Report the robot ID, manifest path, HAL path, sensors, control modes, safety assumptions, sim support, deployment config, compatible rSkills, tests/docs changed, validation commands, and unresolved physical facts.

## Stop Conditions

Stop before hardware execution or code generation if safety limits, E-stop behavior, required SDK licensing, physical dimensions, or control mode semantics are unknown.