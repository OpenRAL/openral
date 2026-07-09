# Scene Creation Blueprint

Use this reference when creating or reviewing OpenRAL simulation scenes, BDDL tasks, benchmark configs, or scene adapters.

## Scene Path Decision

| Need | Recommended path |
| --- | --- |
| New task in an existing registered backend | Add a `SceneEnvironment` YAML |
| Custom Panda + LIBERO object/task variant | Write BDDL and use `franka_libero_custom_bddl` |
| New arena, non-LIBERO robot, custom physics, or new simulator | Add a Python scene adapter |
| Reproducible evaluation suite | Add benchmark config plus docs/eval command |

Prefer YAML or BDDL before adding Python code.

## Scene YAML Blueprint

Minimal shape:

```yaml
scene:
  id: <registered_scene_id>
  backend: mujoco
  cameras: []

task:
  id: <task_id>
  scene_id: <same_as_scene.id>
  instruction: "<task instruction>"
  success_key: <info_key>

seed: 42
n_episodes: 1
record_video: false

metadata:
  honest_scope: "what this scene does and does not prove"
```

Rules:

- Do not include `vla:` in scene YAML.
- Pass the rSkill separately with `--rskill`.
- Add `robot_id` only for free-axis scenes.
- Keep `task.scene_id` equal to `scene.id`.
- `success_key` must be produced by the adapter's `StepResult.info`.
- Use `metadata` for honest scope, paper links, and reproduction notes.

## BDDL Checklist

For custom LIBERO scenes:

- Start from an upstream LIBERO BDDL file.
- Keep `:objects`, `:obj_of_interest`, `:init`, `:goal`, and `:language` coherent.
- Use object names that exist in LIBERO's object registry.
- Remember that `--instruction` changes the prompt only; success still follows `:goal`.
- Prefer this path for pi0.5-LIBERO when staying inside Panda + robosuite + HOPE object assumptions.
- Record the BDDL path and scene YAML path together.

## Adapter Checklist

Add Python scene adapters only when needed:

- Register with `@SCENES.register("<id>")`.
- Ensure module import triggers registration.
- Return an object satisfying the `SimRollout` protocol.
- Provide conventional observation keys: `images`, `state`, and `task` where possible.
- Ensure `StepResult.info` includes the declared success key.
- Expose `mujoco_handles()` only with a real MuJoCo model/data pair.
- Add real adapter tests and sim tests.

## rSkill Pairing Checklist

- rSkill `embodiment_tags` match the robot/scene.
- Camera keys and preprocessing match what the adapter emits.
- State/action dimensions match the rollout observation and action space.
- `policy_extras` or adapter knobs are in the rSkill manifest when policy-specific.
- Language instruction does not contradict success predicate.
- License gates allow intended local eval/deployment.

## Validation Commands

```bash
openral sim list
openral sim run --config scenes/<path>.yaml --rskill rskills/<skill> --n-episodes 1
openral sim run --config scenes/<path>.yaml --rskill rskills/<skill> --task <task-id>
openral sim run --config scenes/<path>.yaml --rskill rskills/<skill> --instruction "<prompt>"
openral benchmark run --suite <suite-id> --vla <vla-id>:rskills/<skill>
```

If optional simulator dependencies are missing, tests may skip only with a concrete dependency reason. Do not replace a sim run with a mock success.

## Output Checklist

- Scene config path.
- Scene ID and backend.
- Robot ID and fixed/free-axis behavior.
- Task ID, instruction, success key, and success predicate source.
- rSkill used for validation.
- Commands run and artifacts produced.
- Docs/tests updated.
- Known limitations, missing assets, or unreproduced metrics.

## Hard Stops

- Adapter is not registered.
- Success key or predicate is unknown.
- Robot/rSkill action semantics differ.
- BDDL goal conflicts with the instruction.
- Required simulator assets or licenses are unavailable.
- Eval metrics would need to be invented.