---
name: scene-creator
description: 'Use when creating OpenRAL simulation scenes, SceneEnvironment YAML, task configs, BDDL custom LIBERO scenes, MuJoCo adapters, scene registry entries, benchmark configs, openral sim run configs, robot-scene-task tuples, policy/rSkill pairing, or sim validation.'
argument-hint: 'Scene goal, robot id, simulator/backend, task, BDDL idea, or config path'
---

# Scene Creator

Create or review OpenRAL simulation scenes and task configs that can run through `openral sim run` with an rSkill. A scene is the on-disk robot, scene, and task tuple; the rSkill is supplied separately at runtime.

## When to Use

- Creating `scenes/<...>.yaml` or benchmark configs.
- Authoring a `SceneEnvironment` for an existing robot, scene, task, and rSkill pairing.
- Creating custom pi0.5/LIBERO BDDL tasks.
- Adding a new scene adapter or simulator backend.
- Pairing a scene with a robot manifest and rSkill for eval.
- Debugging scene registration, task IDs, fixed/free robot axes, success keys, or sim config validation.

## Required Context

Read only what matches the scene type:

1. `CLAUDE.md` for architecture, tests, docs, and no-mocks rules.
2. `docs/tutorials/sim/create-a-sim-environment.md` for the canonical scene workflow.
3. `scenes/README.md` and existing YAMLs under `scenes/`.
4. `benchmarks/README.md` and existing benchmark YAMLs when adding eval suites.
5. `docs/reference/sim-environments.md` and `docs/reference/vla_compatibility.md`.
6. `python/sim/src/openral_sim/registry.py`, scene adapters, and policy adapters when adding Python registry entries.
7. `python/core/src/openral_core/schemas.py` for `SceneEnvironment`, `SceneSpec`, `TaskSpec`, and `SimEnvironment`.

## References

- Load [scene-blueprint.md](./references/scene-blueprint.md) when you need the scene path decision table, YAML blueprint, BDDL checklist, adapter checklist, rSkill pairing checklist, validation commands, or output checklist.
- Do not create a `scripts/` directory unless there is a scene-specific helper that is not already supported by `openral sim`, `openral benchmark`, or existing repo tools.

## Workflow

1. Choose the scene path.
   - Existing registry scene plus new YAML is preferred.
   - For custom Franka/Panda LIBERO tasks, prefer BDDL with `franka_libero_custom_bddl` over a raw MuJoCo adapter.
   - Use a new Python adapter only when the task cannot fit an existing backend, BDDL, or parameterized scene.

2. Author the YAML correctly.
   - Include `scene`, `task`, `seed`, `n_episodes`, and optional video/metadata fields according to existing examples.
   - Add `robot_id` only for free-axis scenes; do not set it for fixed-robot scenes that reject overrides.
   - Do not add a `vla:` block to `SceneEnvironment` YAML. The rSkill is supplied via `--rskill` and composed into `SimEnvironment` at runtime.
   - Set `task.success_key` to a real key produced by the adapter's `info` dict.

3. Pair with an rSkill.
   - Check the rSkill manifest for embodiment tags, sensors, state/action contracts, camera aliases, preprocessing, and `policy_extras`.
   - Ensure robot capability tags and scene action semantics match the rSkill.
   - Do not rely on a language instruction that conflicts with the success predicate.

4. For custom LIBERO BDDL scenes.
   - Copy from upstream LIBERO BDDL patterns.
   - Keep `:objects`, `:obj_of_interest`, `:init`, `:goal`, and `:language` consistent.
   - Remember that CLI `--instruction` changes only the policy prompt, not the BDDL success goal.
   - Use this route for pi0.5-LIBERO fidelity whenever the task stays inside the Panda + LIBERO + HOPE-object envelope.

5. For new Python scene adapters.
   - Register with `@SCENES.register("<id>")` and import the module so registration fires.
   - Implement the `SimRollout` protocol shape used by existing adapters.
   - Return observations with conventional `images`, `state`, and `task` fields where possible.
   - Ensure `StepResult.info` contains the declared success key.
   - Expose `mujoco_handles()` only when a real MuJoCo model/data pair exists.

6. Validate with real runs.
   - First check registry visibility with `openral sim list`.
   - Run the smallest relevant sim command with the intended rSkill.
   - Save video or JSON summaries when useful for reviewing task behavior.
   - Do not use mocks or fake success metrics; use real simulator assets or skip with a concrete missing dependency reason.

7. Update tests and docs.
   - Add scene YAML load tests, adapter tests, sim tests, and benchmark docs as appropriate.
   - Update `docs/reference/sim-environments.md`, `scenes/README.md`, benchmark docs, the matching `docs/methods/` file, and repo state map when the surface changes.

## Command Patterns

```bash
openral sim list
openral sim run --config scenes/<path>.yaml --rskill rskills/<skill>
openral sim run --config scenes/<path>.yaml --rskill rskills/<skill> --task <task-id> --instruction "<prompt>"
openral benchmark run --suite <suite-id> --vla <vla-id>:rskills/<skill>
```

## Output Checklist

Report the scene config path, scene ID, task ID, robot axis behavior, backend, rSkill pairing, success key, validation command, generated artifacts, docs/tests changed, and any limitations such as simulator licenses, missing assets, or unreproduced metrics.

## Stop Conditions

Stop before claiming a scene is valid if the adapter is not registered, success key is unknown, robot/rSkill action semantics differ, BDDL goal conflicts with the instruction, required simulator assets are unavailable, or eval metrics would need to be invented.