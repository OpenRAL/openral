# Create a sim environment

This tutorial walks you through authoring a **`SimScene` YAML** — the
on-disk `(robot × scene × task)` tuple — that `openral sim run` consumes
together with an rSkill (`--rskill rskills/<id>`), so you can test a VLA /
rSkill against a task in simulation without touching hardware. The
runtime form the adapters see is the composed **`SimEnvironment`**
(`SimScene` + `RSkillManifest`); the YAML on disk never carries a
`vla:` block.

`SimScene` is the middle tier of the [ADR-0041](../../adr/0041-scene-three-tier-hierarchy.md)
scene hierarchy:

```
DeployScene  ⊆  SimScene  ⊆  BenchmarkScene
  (deploy run)    (sim run)    (benchmark scene / benchmark run)
```

This page focuses on the **`SimScene`** tier — the ad-hoc, single-rollout
shape consumed by `openral sim run`. Adding `metadata: {paper, honest_scope}`
+ non-`None` `seed` and `n_episodes` to a `SimScene` YAML turns it into a
**`BenchmarkScene`** that `openral benchmark scene` accepts; dropping the
`task:` block turns it into a **`DeployScene`** that `openral deploy sim`
accepts. Per-tier loaders refuse wrong-tier YAMLs at parse time.

It covers six things, in increasing depth:

1. The `openral sim run` flag surface and what is registry-resolved (i.e. **not**
   hardcoded).
2. Authoring a `SimScene` YAML for an **existing** robot, scene, task,
   and pairing it with an rSkill.
3. Bringing a **new** robot manifest (`robots/<id>/robot.yaml`) into sim.
4. **Recommended path for pi0.5-LIBERO custom scenes**: write a BDDL file
   and drive it through the `franka_libero_custom_bddl` adapter.
5. Writing a **new scene adapter** (a new task suite or simulator wrapper)
   in Python, for cases that don't fit BDDL (custom robot, custom physics,
   non-LIBERO backend).
6. Writing a **new policy adapter** (a new VLA backend) and matching it to an
   rSkill.

The companion cookbook is [`scenes/README.md`](https://github.com/OpenRAL/openral/blob/master/scenes/README.md);
the reference schemas are documented in
[ADR-0002](../../adr/0002-eval-and-sim-environments.md),
[ADR-0009](../../adr/0009-separate-sim-and-benchmarking.md), and
[ADR-0041](../../adr/0041-scene-three-tier-hierarchy.md).

---

## 1. What `openral sim run` accepts

`openral sim run` is defined in
[`python/sim/src/openral_sim/cli.py`](https://github.com/OpenRAL/openral/blob/master/python/sim/src/openral_sim/cli.py).
Every axis — robot, scene, task, VLA, physics backend, device — is resolved at
runtime through one of three registries (no hardcoded IDs):

| Registry | Defined in | Populated by |
|---|---|---|
| `SCENES` | `python/sim/src/openral_sim/registry.py` | `@SCENES.register("<id>")` decorators in `adapters/*.py` |
| `POLICIES` | same | `@POLICIES.register("<id>")` decorators in `adapters/*.py` |
| `ROBOTS` | same | Auto-discovered from `robots/<id>/robot.yaml` at import time |

List everything that is currently registered on your install:

```bash
openral sim list
```

`openral sim list` is a sibling subcommand to `openral sim run`; it prints the three
registries (scenes / policies / robots) and exits without touching OTel or the
runtime path.

### Flags (`openral sim run`)

```
--config PATH                Path to a SimScene YAML (REQUIRED; strict —
                             DeployScene / BenchmarkScene YAMLs are rejected
                             with a redirect message pointing at
                             `openral deploy sim` / `openral benchmark scene`).
--rskill <weights_uri>       rSkill reference: rskills/<id>, bare name,
                             or HF repo id  (REQUIRED).
--robot ID                   robot_id override for free-axis scenes
                             (rejected on scenes that hard-fix a robot,
                              and on YAMLs that already set robot_id:).
--task  ID                   Override task.id (e.g. libero_spatial/3).
--instruction TEXT           Override the natural-language task instruction.
                             Wins over a scene's per-episode language (a
                             custom BDDL `:language` clause, a RoboCasa
                             sampled-object string) — see §4.
--max-steps  N               Override task.max_steps.
--n-episodes N               Override SimScene.n_episodes.
--seed       N               Override the global seed.
--device     {cpu, cuda:0, mps, auto}
                             Torch device for the policy.
--save-dir   DIR             Where to write the JSON summary.
--save-video [PATH]          Write the 3-panel debug MP4 (also enables frame capture).
--view / --no-view           Open a passive mujoco.viewer.
--verbose / -v               DEBUG logging.
```

### Canonical invocation

```bash
openral sim run --config scenes/sim/libero_spatial.yaml \
            --rskill rskills/smolvla-libero
```

Both `--config` and `--rskill` are **required** — `openral sim run` always
composes a runtime `SimEnvironment` from a `SimScene` (the YAML)
and an `RSkillManifest` (the rSkill). There is no bare-CLI invocation
path; supply the scene + task in the YAML.

### Per-axis overrides on top of `--config`

Beyond the two required flags, the remaining options — `--robot` (only
on free-axis scenes), `--task`, `--instruction`, `--max-steps`,
`--n-episodes`, `--seed`, `--device`, `--save-dir`, `--save-video` —
**overlay** the loaded config (see `_load_or_build_env` in
[`cli.py`](https://github.com/OpenRAL/openral/blob/master/python/sim/src/openral_sim/cli.py)),
so a single YAML can drive an entire task suite:

```bash
# Same VLA + scene; iterate through tasks.
for i in 0 1 2 3; do
    openral sim run --config scenes/sim/libero_spatial.yaml --rskill smolvla-libero --task "libero_spatial/$i"
done
```

If you need to swap an axis that *is* baked in, copy the YAML and edit it —
that is the supported pattern.

### Startup performance (`activate()` parallelisation)

`SimRunner.activate()` builds the env (MuJoCo XML compile / dataset
prefetch) and the policy (PaliGemma / NF4 quantization on π0.5, weights
load on SmolVLA) **concurrently** on a 2-worker `ThreadPoolExecutor`.
Both factories only read the immutable `SimEnvironment` and share no
mutable state, so the wall-clock for `activate` collapses to
`max(env_ms, policy_ms)` instead of their sum — a meaningful win on
LIBERO / RoboCasa / GR1-tabletop where each side is 30–150 s.

On every invocation the runner logs a structured `sim_init_parallel`
record with `env_ms`, `policy_ms`, `total_ms`, and `saved_ms`, so you
can confirm the win for your specific (scene, VLA) combination.

To force the legacy sequential path (e.g. when interleaved logs would
obscure a profiling investigation), set
`OPENRAL_SIM_SEQUENTIAL_INIT=1`:

```bash
OPENRAL_SIM_SEQUENTIAL_INIT=1 openral sim run --config scenes/sim/libero_spatial.yaml --rskill pi05-libero-nf4
```

See [GH-134](https://github.com/OpenRAL/openral/issues/134).

---

## 2. Author a SimScene YAML

The on-disk shape is a [`SimScene`](https://github.com/OpenRAL/openral/blob/master/python/core/src/openral_core/schemas.py)
— `(robot × scene × task)` — defined at `python/core/src/openral_core/schemas.py:4728`.
At runtime the CLI composes it with the rSkill manifest (`--rskill`) into a
[`SimEnvironment`](https://github.com/OpenRAL/openral/blob/master/python/core/src/openral_core/schemas.py)
(`schemas.py:4598`) that adapter factories consume. Loading a YAML that
carries a `vla:` block raises `ROSConfigError` — policy *always* travels
on the CLI, not in the YAML.

Take an existing config as your starting point — for example,
[`scenes/sim/libero_spatial.yaml`](https://github.com/OpenRAL/openral/blob/master/scenes/sim/libero_spatial.yaml):

```yaml
# SimScene = scene + task. Policy is supplied at the CLI via
#   --rskill rskills/<id>

# robot_id: free-axis scenes only (LIBERO/MetaWorld/RoboCasa hard-fix the
# robot in the registry; setting it here on those scenes is a ROSConfigError).
# robot_id: franka_panda

scene:
  id: libero_spatial              # key into SCENES
  backend: mujoco                 # PhysicsBackend enum
  observation_height: 256
  observation_width: 256
  cameras: ["agentview", "wrist"]  # optional; SceneSpec.cameras defaults to []

task:
  id: libero_spatial/0            # adapter splits on "/" to resolve
  scene_id: libero_spatial        # MUST equal scene.id (validated post-init)
  instruction: ""                 # LIBERO overrides from suite metadata
  max_steps: 100
  success_key: is_success         # which info[] key marks success

seed: 42                          # optional on SimScene; defaults to 0
n_episodes: 1                     # optional on SimScene; defaults to 1
record_video: false               # optional; defaults to false
```

To promote this same scene to **paper-comparable** form, add a
`metadata: BenchmarkMetadata` block + set `seed` and `n_episodes` to
canonical values, then move the file to `scenes/benchmark/`. The
benchmark-tier sibling lives at
[`scenes/benchmark/libero_spatial.yaml`](https://github.com/OpenRAL/openral/blob/master/scenes/benchmark/libero_spatial.yaml)
and adds:

```yaml
seed: 0                           # required (no default)
n_episodes: 500                   # required (no default; paper protocol)
metadata:
  paper: "https://arxiv.org/abs/2309.11500"
  honest_scope: "..."             # honest scope statement; required
```

`openral sim run` loads the sim-tier sibling; `openral benchmark scene` loads
the benchmark-tier sibling. The per-tier loader (`load_scene_strict`)
rejects wrong-tier YAMLs — a `BenchmarkScene` YAML passed to `openral sim
run` returns a redirect message pointing at `openral benchmark scene`, and
vice versa.

### The required blocks

| Block | Schema | Required keys |
|---|---|---|
| `scene` | [`SceneSpec`](https://github.com/OpenRAL/openral/blob/master/python/core/src/openral_core/schemas.py) (`schemas.py:4285`) | `id`; `backend` defaults to `mujoco` |
| `task` | [`TaskSpec`](https://github.com/OpenRAL/openral/blob/master/python/core/src/openral_core/schemas.py) (`schemas.py:4500`) | `id`, `scene_id` (must equal `scene.id`) |
| `robot_id` | string, key into `ROBOTS` | Only on **free-axis** scenes — and only if you want to bake the robot into the YAML rather than passing `--robot`. Forbidden on `fixed_robot` scenes (LIBERO/MetaWorld/RoboCasa). |

Policy is **not** a YAML block; it is supplied at the CLI as
`--rskill rskills/<id>` → resolves to an `RSkillManifest`
(`schemas.py` — search for `class RSkillManifest`) and is composed onto the
runtime `SimEnvironment.vla` by `_load_or_build_env` in
[`cli.py`](https://github.com/OpenRAL/openral/blob/master/python/sim/src/openral_sim/cli.py).
The Pydantic registries reject unknown ids with a list of valid ones, so
typos surface immediately. Run it:

```bash
openral sim run --config my_config.yaml --rskill rskills/<your_skill>
```

You will see a per-episode summary line and a `0` exit code on success.
The same `InferenceRunner` Protocol underneath also drives
`openral benchmark run`. (`openral deploy run` is the hardware sibling — it
consumes a separate `RobotEnvironment` YAML with the policy bundled
in-file, and is not interchangeable with `openral sim run`.)

---

## 3. Add a new robot manifest

Robots are auto-registered from `robots/<id>/robot.yaml` at import time — no
Python edit required. The discovery loop lives at
[`python/sim/src/openral_sim/policies/robots.py:67-118`](https://github.com/OpenRAL/openral/blob/master/python/sim/src/openral_sim/policies/robots.py).
The search path is, in order:

1. `$OPENRAL_ROBOTS_DIR/<id>/robot.yaml` (if the env var is set)
2. `<repo_root>/robots/<id>/robot.yaml`

Use [`robots/so100_follower/robot.yaml`](https://github.com/OpenRAL/openral/blob/master/robots/so100_follower/robot.yaml)
as a small (6-DoF + gripper) template; use
[`robots/franka_panda/robot.yaml`](https://github.com/OpenRAL/openral/blob/master/robots/franka_panda/robot.yaml)
for a 7-DoF arm. The required top-level blocks are:

| Block | Purpose |
|---|---|
| `name`, `embodiment_kind`, `base_frame` | Identity + URDF root frame |
| `joints[]` | Per-joint: name, type, parent/child links, axis, limits, actuator |
| `end_effectors[]` | Gripper(s) / hand(s); kind, DoF, force/payload limits |
| `sensors[]` | Cameras / IMUs / etc.; each maps to a `vla_feature_key` |
| `capabilities` | Control modes, embodiment tags, lift / dexterity flags |
| `safety` | Workspace box, speed/force/torque limits, deadman flag |
| `observation_spec`, `action_spec` | State / action shapes and representations |
| `sim` (optional) | MuJoCo wiring consumed by `MujocoArmHAL.from_description` — see below |

### The `sim:` block (ADR-0023)

For any robot that should drive a MuJoCo digital twin through the shared
`MujocoArmHAL` base, declare a `sim:` block. The runner reads it directly;
no per-robot Python file is required — pass the loaded manifest into
`MujocoArmHAL.from_description(desc)`.

```yaml
sim:
  # One of:
  #   robot_descriptions:<module>   — vendored menagerie MJCF
  #   gym_aloha:<scene>              — gym-aloha package asset
  #   openarm_v2:bimanual            — Enactic OpenArm v2 (fetched on first use)
  #   file:/abs/path.xml | /abs/path — explicit override
  mjcf_uri: "robot_descriptions:ur5e_mj_description"

  # Floating-base humanoids (G1, H1): qpos offset 7, qvel offset 6 are
  # derived automatically.  Single-arm robots can omit.
  floating_base: false

  # Explicit joint→qpos / →actuator overrides — only needed when the
  # MJCF declares joints in an order other than ``description.joints`` or
  # leaves passive follower qpos slots in between (OpenArm).
  # joint_qpos_addr:
  #   joint_a: 0
  #   joint_b: 1
  # actuator_index:
  #   joint_a: 0

  grippers:                                    # zero, one, or two entries
    - joint: "panda_gripper"                   # name from joints[]
      ctrl_range: [0.0, 255.0]
      qpos_addrs: [7, 8]                       # finger qpos indices
      qpos_scale: 0.08                         # 2 * 0.04 m max extent
      read_mode: "sum_over_scale"              # | "affine_low_high" | "passthrough"
      write_mode: "normalised"                 # | "passthrough"
      # actuator_index: 7                      # override; defaults to actuator_index map
      # mirror_actuator_index: 15              # Aloha: writes -ctrl to the negative finger

  # Connect-time hooks:
  # keyframe_index: 0          # mj_resetDataKeyframe(model, data, idx) — Aloha
  # seed_ctrl_from_qpos: true  # ctrl = qpos on connect — OpenArm v2
```

**Single source of truth.** When you add a `<ROBOT>_DESCRIPTION` Python
constant (e.g. `UR5e_DESCRIPTION` in `openral_hal/ur.py`), mirror the
`sim` block via `sim=SimDescription(...)` so the manifest-vs-YAML drift
guard (`tests/unit/test_robot_manifests_match_hal_constants.py`, plus
`tests/sim/test_data_driven_mujoco_hal.py::test_python_description_matches_yaml`)
stays green.

**Worked examples in tree** (use as templates):

| Robot | Pattern |
|---|---|
| `so100_follower` | single-arm + revolute Jaw (`read_mode: affine_low_high`) |
| `franka_panda` | single-arm + parallel gripper (`read_mode: sum_over_scale`) |
| `ur5e`, `ur10e`, `rizon4` | single-arm, no gripper, no overrides |
| `g1`, `h1` | floating-base humanoid (`floating_base: true`) |
| `aloha_bimanual` | bimanual + two passthrough grippers with `mirror_actuator_index` + `keyframe_index: 0` |
| `openarm` | bimanual + two passthrough grippers + explicit `joint_qpos_addr` skipping passive follower fingers + `seed_ctrl_from_qpos: true` |

### Drop a new robot in

```bash
mkdir -p robots/my_arm
$EDITOR robots/my_arm/robot.yaml        # copy & adapt so100_follower/robot.yaml
$EDITOR robots/my_arm/README.md         # pair the manifest with adapter notes
```

### Verify it registered

```bash
openral sim list | grep "robots:"
# should now include `my_arm`
```

You can also confirm the manifest loads cleanly from Python:

```python
from openral_sim import ROBOTS
robot = ROBOTS.get("my_arm")()         # invokes the cached factory
print(robot.name, len(robot.joints))
```

### Match the manifest to a sim scene

Every scene adapter expects a specific embodiment. LIBERO assumes a 7-DoF arm
with a parallel gripper; MetaWorld assumes the Sawyer. For your new robot to
run end-to-end you also need either (a) a scene adapter that knows how to drive
it, or (b) the `mock` scene, which accepts any action dimensionality (see §4).

---

## 4. Custom pi0.5-LIBERO scenes via BDDL (recommended)

If you want **pi0.5-LIBERO** to drive a custom Franka Panda pick-and-place
scene — different objects, different start positions, a different goal
predicate, different language instruction — the cleanest path is to
**write a BDDL file** and drive it through the
[`franka_libero_custom_bddl`](https://github.com/OpenRAL/openral/blob/master/python/sim/src/openral_sim/backends/libero_custom_bddl.py)
adapter. That adapter routes through robosuite + LIBERO's
`OffScreenRenderEnv`, so the controller (OSC_POSE), the renderer, and the
state encoding are bit-identical to what pi0.5-LIBERO was trained on. No
custom Python adapter needed.

This path is **strongly preferred** for pi0.5 use cases over writing a
raw-mujoco scene adapter (Section 5). The pi0.5 vision tower is highly
sensitive to pixel-level rendering details (sRGB framebuffer, panda mesh
appearance, lighting model) that are non-trivial to replicate outside
robosuite's pipeline.

### Anatomy of a custom BDDL file

LIBERO's BDDL is a Lisp-style task definition. The four blocks you'll edit:

```lisp
(define (problem LIBERO_Floor_Manipulation)
  (:domain robosuite)
  (:language Pick the alphabet soup and place it in the basket)

  (:regions
    ;; Named spawn regions on the floor — (min_x min_y max_x max_y).
    (bin_region (:target floor)
        (:ranges ((-0.01 0.25 0.01 0.27))))
    (target_object_region (:target floor)
        (:ranges ((-0.145 -0.265 -0.095 -0.215))))
    ;; ... additional regions for distractors
  )

  (:fixtures
    main_floor - floor      ;; the workspace plane
  )

  (:objects
    ;; Any object registered under libero/envs/objects/. Common ones:
    ;;   alphabet_soup, basket, salad_dressing, cream_cheese, milk,
    ;;   tomato_sauce, butter, bbq_sauce, ketchup, ...
    alphabet_soup_1 - alphabet_soup
    basket_1         - basket
    ;; ... distractors if you want them visible
  )

  (:init
    ;; Where each object spawns relative to a region.
    (On alphabet_soup_1 floor_target_object_region)
    (On basket_1         floor_bin_region)
  )

  (:goal
    ;; The success predicate. Common forms:
    ;;   (On X Y)  → X ends up resting on Y
    ;;   (In X Y)  → X ends up contained in Y
    (On alphabet_soup_1 basket_1)
  )
)
```

The existing `libero/bddl_files/libero_object/*.bddl` files (shipped with the
`libero` PyPI package) are the canonical reference — copy one as a starting
point and edit.

### Wiring the BDDL into a YAML

```yaml
# scenes/sim/my_custom_task.yaml
robot_id: franka_panda

scene:
  id: franka_libero_custom_bddl
  backend: mujoco
  observation_height: 256
  observation_width: 256
  backend_options:
    # Absolute path to your authored BDDL file.
    bddl_file: "/abs/path/to/my_task.bddl"
    # Optional — path to a .pruned_init file with hand-tuned starting
    # qpos (a (N, ?) numpy array, torch.save-pickled). Omit to let
    # robosuite use the BDDL's default randomised spawn.
    init_state_file: "/abs/path/to/my_task.pruned_init"
    init_state_index: 0   # which row of init_state_file to use

task:
  id: my_task/0
  scene_id: franka_libero_custom_bddl
  instruction: ""           # the adapter reads from the BDDL's :language clause
  max_steps: 300
  success_key: is_success

# Policy is supplied at the CLI via --rskill rskills/<id>.
# Adapter-specific knobs (e.g. n_action_steps, flip_images_180, camera_keys)
# live in the rSkill manifest's `policy_extras:` block, not the YAML.
```

#### Instruction precedence (what the policy is actually prompted with)

Each step the policy is prompted with the first non-blank of, in order:

1. an explicit `--instruction "<text>"` on the CLI,
2. the scene's per-episode language — the BDDL `:language` clause for this
   adapter (exposed as `env.language_instruction` → `obs["task"]`),
3. the static YAML `task.instruction`.

So leaving `task.instruction: ""` defers to the `:language` clause, but
passing `--instruction` overrides it — useful for probing how the policy
reacts to a different command without re-authoring the BDDL.

> **Note — instruction vs. success.** `--instruction` only changes what the
> policy is *told*; the success predicate is still the BDDL `:goal`. Telling
> the policy to "pick the orange juice" on a BDDL whose `:goal` is
> `(On milk_1 basket_1)` will steer the arm toward the juice but the episode
> can only succeed on the milk. To change the *task*, edit `:obj_of_interest`
> / `:goal` (and the `:language`) in the BDDL itself.

Run it:

```bash
openral sim run --config scenes/sim/my_custom_task.yaml \
            --rskill rskills/<your-skill>
```

### Worked examples in the repo

A minimal demo lives at `scenes/sim/`:

| YAML | What it customises |
|---|---|
| [`franka_libero_pnp.yaml`](https://github.com/OpenRAL/openral/blob/master/scenes/sim/franka_libero_pnp.yaml) (+ sibling [`franka_libero_pnp.bddl`](https://github.com/OpenRAL/openral/blob/master/scenes/sim/franka_libero_pnp.bddl)) | Custom BDDL routed through `franka_libero_custom_bddl` → robosuite `OffScreenRenderEnv` — picks `milk_1` into a basket from a HOPE-library distractor mix (`cream_cheese`, `tomato_sauce`, `butter`, `orange_juice`, `chocolate_pudding`). |

The customisation is entirely in the choice of target + distractor
objects assembled from LIBERO's HOPE library; the policy generalises
across these combinations because it was trained on many similar
permutations. (Two near-identical sibling demos — salad-dressing and
bbq-sauce — were removed as replications of the same target-swap
concept; author your own `:obj_of_interest` / `:objects` to make new
ones.) The `pi05-libero-nf4` rSkill is nf4-quantised, so this runs on a
CUDA device (nf4 has no CPU path) — invoke with
`openral sim run --config scenes/sim/franka_libero_pnp.yaml --rskill pi05-libero-nf4`.

### When BDDL is **not** enough

Reach for the Python adapter path (Section 5) when:

- You want a completely different arena (no LIBERO floor / table).
- You want a different robot (LIBERO's BDDL is panda-only).
- You want to add objects that don't exist in LIBERO's
  `envs/objects/` registry — adding new HOPE / scanned objects is
  upstream LIBERO work.
- You want physics independent of robosuite (e.g. a mock scene or a
  non-MuJoCo backend).

For everything else inside the "panda + floor + HOPE objects" envelope, the
BDDL path gives you full pi0.5 fidelity for ~50 lines of Lisp.

---

## 5. Write a custom scene adapter

A scene adapter is a function decorated with `@SCENES.register("<id>")` that
returns an object satisfying the
[`SimRollout`](https://github.com/OpenRAL/openral/blob/master/python/sim/src/openral_sim/rollout.py)
Protocol:

```python
class SimRollout(Protocol):
    scene: SceneSpec
    task:  TaskSpec
    def reset(self, seed: int | None = ...) -> Observation: ...
    def step(self, action: NDArray[np.float32]) -> StepResult: ...
    def render(self) -> NDArray[np.uint8] | None: ...
    def close(self) -> None: ...
```

`Observation` is a free-form dict; adapters SHOULD include `"images"`
(dict of HWC uint8 RGB frames), `"state"` (1-D float32), and `"task"`
(natural-language instruction). `StepResult` is a 5-tuple-shaped dataclass
(`observation`, `reward`, `terminated`, `truncated`, `info`) — the runner
reads success from `info[task.success_key]`.

The smallest working scene adapter lives in
[`python/sim/src/openral_sim/policies/mock.py`](https://github.com/OpenRAL/openral/blob/master/python/sim/src/openral_sim/policies/mock.py).
It is the recommended reference because it has no physics and no external
dependencies. The realistic reference is `adapters/libero.py` (wraps the
LIBERO gymnasium env).

### Skeleton

Place new adapters under `python/sim/src/openral_sim/{policies,backends}/` so they are
imported by the package's `__init__.py` (which is what triggers the
`@register` side-effect):

```python
# python/sim/src/openral_sim/{policies,backends}/my_scene.py
from __future__ import annotations
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from openral_sim.registry import SCENES
from openral_sim.rollout import Observation, StepResult


@dataclass
class _MyScene:
    scene: "SceneSpec"
    task:  "TaskSpec"
    _step: int = 0

    def reset(self, seed: int | None = None) -> Observation:
        self._step = 0
        return self._observe()

    def step(self, action: NDArray[np.float32]) -> StepResult:
        self._step += 1
        done = self._step >= self.task.max_steps
        return StepResult(
            observation=self._observe(),
            reward=0.0,
            terminated=done,
            truncated=False,
            info={self.task.success_key: done},
        )

    def render(self) -> NDArray[np.uint8] | None:
        return np.zeros(
            (self.scene.observation_height, self.scene.observation_width, 3),
            dtype=np.uint8,
        )

    def close(self) -> None:
        return None

    def _observe(self) -> Observation:
        return {
            "images": {"camera1": self.render()},
            "state":  np.zeros(8, dtype=np.float32),
            "task":   self.task.instruction,
        }


@SCENES.register("my_scene")
def _build(env_cfg: "SimEnvironment") -> _MyScene:
    return _MyScene(scene=env_cfg.scene, task=env_cfg.task)
```

Then wire it into the package's import set so `@register` actually fires.
The simplest way is a one-line import in
`python/sim/src/openral_sim/{policies,backends}/__init__.py`:

```python
from . import my_scene  # noqa: F401  # reason: register-by-import
```

Confirm it shows up in `openral sim list` under **scenes:**.

### Optional: `mujoco_handles` for `openral sim run --view`

If your adapter wraps a MuJoCo model and exposes
`mujoco_handles(self) -> tuple[mujoco.MjModel, mujoco.MjData] | None`,
`--view` will open a passive viewer. The method is intentionally **not** part
of the `SimRollout` Protocol — the runner uses `getattr(env,
"mujoco_handles", None)`, so non-MuJoCo adapters need not stub it.

### Worked example — `so101_box` (custom arena + custom objects + non-Panda robot)

[`python/sim/src/openral_sim/backends/so101_box/`](https://github.com/OpenRAL/openral/blob/master/python/sim/src/openral_sim/backends/so101_box/)
is a reference for the "custom arena + custom robot + custom task"
shape this section targets. It registers `so101_box`
(fixed_robot=`so101_follower`) and ships an SO-101 in a configurable
box arena with an OAK-D Pro RGB-D overhead camera, a wrist camera
parented to the gripper, a slotted target block + cylindrical tube as
the task, and a geometric tube-insertion success check.

Two design points worth lifting into your own adapter:

* **Every scene parameter is YAML, no geometry is hard-coded.** The
  composer is fed a single typed `BoxSceneOptions` dataclass that
  carries every dimension, pose and threshold. The CLI's
  `scene.backend_options` block populates it via
  `_options_from_backend_options` (which rejects unknown keys
  loudly). Once the adapter is registered, the next "SO-101 in a
  similar arena" scene is a pure YAML edit — no Python change. See
  [`scenes/sim/so101_tube_insertion.yaml`](https://github.com/OpenRAL/openral/blob/master/scenes/sim/so101_tube_insertion.yaml)
  for the full surface.
* **The MJCF is composed by reading the upstream robot MJCF, then
  rewriting + appending.** `compose_so101_box_mjcf` reads
  `robot_descriptions:so_arm101_mj_description` (the SO-101 MJCF
  shipped with `TheRobotStudio/SO-ARM100`), re-anchors its
  `<body name="base">` to the configured world pose via a regex
  rewrite, splices a `<camera>` into the gripper body, and appends
  the arena + objects + overhead camera to the worldbody just before
  `</worldbody>`. The result is written next to the upstream MJCF so
  `meshdir="assets"` resolves at compile time without copying any
  STLs. Same pattern as `openarm_robosuite`.

If your custom scene also needs RGB + depth from the same camera, the
`so101_box` rollout shows the convention: two `mujoco.Renderer`
instances on the same model, one with `enable_depth_rendering()` set,
both updating against the same camera by name. The depth array is
in metres directly (no normalisation) so the policy / dataset bridge
sees the same units the real OAK-D Pro driver emits.

---

## 6. Write a custom policy adapter

A policy adapter is a function decorated with `@POLICIES.register("<id>")`
that returns an object satisfying the
[`PolicyAdapter`](https://github.com/OpenRAL/openral/blob/master/python/sim/src/openral_sim/policy.py)
Protocol:

```python
class PolicyAdapter(Protocol):
    spec: VLASpec
    device: str
    def reset(self) -> None: ...
    def step(self, observation: Observation, instruction: str) -> NDArray[np.float32]: ...
    def close(self) -> None: ...
```

The simplest reference is `_ZeroPolicy` / `_RandomPolicy` in
[`adapters/mock.py`](https://github.com/OpenRAL/openral/blob/master/python/sim/src/openral_sim/policies/mock.py)
(no weights, CPU-only, end-to-end test on any machine). The realistic
reference is `adapters/smolvla.py` (loads a `SmolVLAPolicy`, normalises the
observation dict, caches action chunks).

### Skeleton

```python
# python/sim/src/openral_sim/{policies,backends}/my_policy.py
from __future__ import annotations
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from openral_sim.registry import POLICIES
from openral_sim.rollout import Observation


@dataclass
class _MyPolicy:
    spec: "VLASpec"
    device: str
    action_dim: int = 7

    def reset(self) -> None:
        # load weights / reset action queue / re-seed RNG here
        return None

    def step(self, observation: Observation, instruction: str) -> NDArray[np.float32]:
        # adapter is responsible for mapping observation → its input format
        del observation, instruction
        return np.zeros(self.action_dim, dtype=np.float32)

    def close(self) -> None:
        return None


@POLICIES.register("my_policy")
def _build(env_cfg: "SimEnvironment") -> _MyPolicy:
    return _MyPolicy(spec=env_cfg.vla, device=env_cfg.vla.device)
```

Wire it in via the same `adapters/__init__.py` import pattern as §4.

### Pair it with an rSkill manifest

The runner enforces that `vla.weights_uri` is a valid skill reference and that
the rSkill manifest's `embodiment_tags` overlap the robot's (see
`openral_sim.runner._check_rskill_compatibility`). To use your policy
with real weights, create `rskills/<my-skill>/rskill.yaml` declaring the
target embodiment(s), then reference it as
`rskills/<my-skill>`. The skills directory layout is documented
under `rskills/README.md`; existing manifests (e.g. `rskills/smolvla-libero/`)
are the practical templates.

If your rollout needs to write a LeRobotDataset v3 via the bridge
(ADR-0019, `openral sim run --dataset-out`), declare BOTH `state_contract`
AND `action_contract` on the manifest:

```yaml
# Per-checkpoint proprioception layout — bridge's observation.state shape.
state_contract:
  dim: 8        # e.g. 7 joint + 1 gripper for Franka+LIBERO
# Per-checkpoint action vector — bridge's action shape.
action_contract:
  dim: 7        # e.g. 7-D delta-EEF for pi05/smolvla/xvla/act on LIBERO
```

The dataset bridge reads these contracts (not the robot manifest)
because the same physical robot can host many checkpoints with
different I/O dims (Franka emits 7-D delta-EEF on LIBERO, 12-D for
RoboCasa, 8-D joint+gripper on real hardware). The scene's
`observation_height/width` flows through as the bridge's per-camera
shape.

### Optional: `last_input_frame` for video capture

Visuomotor adapters MAY also implement
`last_input_frame(self) -> NDArray[np.uint8] | None`. When present and
`record_video` is `True`, the runner stitches the frame the VLA actually saw
into the debug MP4. Scripted / mock policies should omit it.

### End-to-end smoketest

Compose your new scene and policy with an existing robot to confirm
everything is wired:

```bash
openral sim run \
    --config  scenes/<your-config>.yaml \
    --rskill  rskills/<my-skill> \
    --n-episodes 1 \
    --max-steps  10
```

(Substitute `<my-skill>` for the rSkill manifest you wrote alongside
your `<my-config>.yaml`; the runner refuses to load a manifest whose
``embodiment_tags`` do not overlap the resolved robot's, so the smoketest
fails loud on mismatches.)

---

## Level 6: a custom MuJoCo environment via RoboCasa (ADR-0011)

[ADR-0015](../../adr/0015-robocasa-isolated-backend-lazy-assets.md)
adds **RoboCasa** as a `openral sim` backend so you can run kitchen
scenarios with custom robots, tasks, and rSkills against real MuJoCo
physics.

### One-time setup (auto-installed on first use)

`openral_sim._deps.ensure_backend_deps` handles the install chain for
you. On the **first** `openral sim run` against a `robocasa/<task>` scene
id, you see a Rich banner listing every subprocess step plus the
license posture, then a `typer.confirm()` prompt:

```bash
openral sim run --config scenes/sim/robocasa_pnp.yaml \
            --rskill rskills/pi05-robocasa365-human300-nf4 \
            --max-steps 200
```

Banner steps (kitchen variant):

1. `uv sync --all-packages --group robocasa`
2. `mkdir -p ~/.cache/openral/repos`
3. `git clone https://github.com/ARISE-Initiative/robosuite.git ~/.cache/openral/repos/robosuite` (idempotent, master branch — kitchen needs the master tip, not the 1.5.2 PyPI wheel. Master adds `make mink optional` (commit `95743f6`, 3 commits past the v1.5.2 tag) so `mink` stays in `extras_require`; without that, `mink==0.0.5`'s `numpy<2` pin would wedge the workspace. The lockfile-honest `[tool.uv.sources] robosuite = { git = "…", rev = "…" }` entry in `pyproject.toml` pins the exact commit `_robocasa_kitchen_plan` reinstalls editable here.)
4. patch the clone: `touch robosuite/examples/__init__.py`, `touch robosuite/examples/third_party_controller/__init__.py`, `touch robosuite/macros_private.py` (upstream `find_packages()` drops the `examples/` directory + the macros nag is silenced by an empty `macros_private.py`)
5. `uv pip install --force-reinstall --no-deps -e ~/.cache/openral/repos/robosuite`
6. `uv pip install --no-deps "robosuite-models @ git+..."`
7. `uv pip install h5py>=3.16 lxml>=5 llvmlite numba qpsolvers pyopengl-accelerate`
8. `uv pip install --no-deps "mink==0.0.5"` (newer mink relocated `mink.tasks.exceptions.TargetNotSet` away from where robosuite's `mink_controller.py` imports it; `--no-deps` avoids the numpy<2 downgrade)
9. `uv pip install --no-deps "robocasa @ git+..."`

Confirm with `y` and walk away — total install runs in ~30 s on
fibre. Skip the prompt in CI / Dockerfiles with `OPENRAL_AUTO_INSTALL_DEPS=1`.

After the deps land, the same command auto-fetches the ~11 GB
CC-BY-4.0 kitchen asset bundle into `~/.cache/openral/robocasa/`
behind a second Rich license banner (gated by
`typer.confirm()` or `OPENRAL_ALLOW_ROBOCASA_ASSETS=1`).

The `libero` extras-group conflict means **you cannot share a venv
with LIBERO**. The workspace's `[tool.uv].conflicts` block declares
this so `uv sync` refuses an impossible mix; if you need both
backends, use two clones (or two venvs against the same clone).

The auto-installer also handles a number of upstream-bug workarounds
transparently: `robocasa/__init__.py`'s hardcoded
`mujoco==3.3.1` / `numpy==2.2.5` / `robosuite>=1.5.2` assertions (upstream
kitchen 1.0.1) — or `mujoco==3.2.6` / `numpy in {1.23.x, 1.26.4}` /
`robosuite in {1.5.0, 1.5.1}` (robocasa-gr1-tabletop-tasks 0.2.0 fork) — are
bypassed at adapter import-time via `_spoof_robocasa_version_pins`;
the controller config's missing-actuator entries are stripped before
they reach robosuite. See "Known constraints" below for the full set.

### Running with a π₀.₅ checkpoint

`rskills/pi05-robocasa365-human300-nf4` (manifest: `rskills/pi05-robocasa365-human300-nf4/rskill.yaml`)
wraps the **OpenRAL/rskill-pi05-robocasa365-human300-nf4**
Apache-2.0 checkpoint — Physical Intelligence's π₀.₅ (3.4 B params,
16-D state, chunk_size=50) fine-tuned on RoboCasa365 Human-300
(300 atomic+composite tasks, 100 demos each) against the
**PandaMobile** robot, pre-quantized to nf4 so the prequant fast-path
in `openral_sim._quantization` loads the policy in ~20 s instead
of the ~150 s `from_pretrained` walk.

```bash
OPENRAL_ALLOW_ROBOCASA_ASSETS=1 \
  uv run openral sim run --config scenes/sim/robocasa_pnp.yaml \
                    --rskill rskills/pi05-robocasa365-human300-nf4 \
                    --view --max-steps 200
```

Three extra one-time dependencies are needed alongside the RoboCasa
setup above (the `robocasa` extras group drops `transformers` and
`bitsandbytes` because of its `libero`-group conflict):

```bash
# lerobot pi05's tested transformers pin (5.3.0) -- newer versions
# break with `'Tensor' object has no attribute 'pooler_output'` because
# SiglipVisionModel's output type changed in transformers >=4.50.
uv pip install "transformers==5.3.0"

# nf4 quantization (essential for 8 GiB consumer cards -- bf16 OOMs
# the moment robosuite's offscreen renderer pins GL textures).
uv pip install "bitsandbytes>=0.45"

# Authenticate to download google/paligemma-3b-pt-224 (gated -- accept
# the license at https://huggingface.co/google/paligemma-3b-pt-224
# first):
hf auth login --token <YOUR_HF_TOKEN>
```

The first launch downloads the prequantized nf4 safetensors from
`OpenRAL/rskill-pi05-robocasa365-human300-nf4` plus
`google/paligemma-3b-pt-224`'s tokenizer.model into
`~/.cache/huggingface/hub/`. The pi05 adapter detects the
`quantization_metadata.json` sentinel and overlays the prequant state
via `install_prequantized_linears`, skipping the bf16->nf4 conversion
entirely. Total VRAM after warmup: ~4-5 GiB.

What that config wires together:

- `robot_id: panda_mobile` -- the `robots/panda_mobile/robot.yaml`
  manifest (Franka Panda on a 3-DoF holonomic base).
- `scene.id: robocasa/PickPlaceCounterToCabinet` -- one of the curated
  atomic tasks registered with `SCENES.register(...,
  fixed_robot="panda_mobile")` so an accidental `--robot franka_panda`
  fails fast with `ROSConfigError`.
- `scene.backend_options.mode: prebuilt` -- validated through
  :class:`openral_core.RoboCasaBackendOptions` (prebuilt-vs-procedural
  XOR).
- `--rskill rskills/pi05-robocasa365-human300-nf4` -- the
  prequantized π₀.₅ Apache-2.0 manifest that declares the embodiment
  tags / sensor requirements the runner validates.

The first invocation fetches the ~11 GB CC-BY-4.0 kitchen asset bundle
under `~/.cache/openral/robocasa/`. The download is gated by
`typer.confirm()`; `OPENRAL_ALLOW_ROBOCASA_ASSETS=1` is the CI
bypass. The fetch script also needs the mujoco-version spoof, which the
asset helper does in a subprocess `-c` wrapper.

### Authoring a procedural kitchen

For free-axis authoring, pass `--scene robocasa` (no slash) plus a
procedural `backend_options` block:

```yaml
scene:
  id: robocasa
  backend: mujoco
  backend_options:
    mode: procedural
    kitchen_style: 3      # 0..9, one of robocasa's 10 aesthetic packs
    layout_id: 7          # 0..9, one of robocasa's 10 floor plans
    fixtures: ["sink", "stovetop", "microwave"]
    spawn_objects: ["coffee_cup", "apple"]
    task_verb: pnp        # pnp | open | close | press | navigate
    robots: ["PandaMobile"]
    controller: BASIC
    horizon: 500
```

`task_verb` resolves to the matching atomic env (e.g. `pnp` →
`PickPlaceCounterToCabinet`); the remaining keys are validated by
`RoboCasaBackendOptions`'s `model_validator` to enforce the
prebuilt-vs-procedural XOR.

### Known constraints

- **mujoco / numpy / robosuite assertion.** Both robocasa variants
  hard-assert exact micro versions of mujoco, numpy, and robosuite at
  import time even though all three work fine on newer versions in
  practice. The RoboCasa adapter (`openral_sim/backends/robocasa.py:
  _spoof_robocasa_version_pins`) monkey-patches the `__version__`
  strings *only* across the robocasa import block and restores them
  immediately afterwards so the rest of the workspace is unaffected.
- **`examples/` / `macros_private.py` patches.** The auto-installer
  clones robosuite + the robocasa fork to
  `~/.cache/openral/repos/`, drops the missing `__init__.py` files
  under `robosuite/examples/`, and writes empty `macros_private.py`
  stubs into both packages. Without these patches every run emits a
  `Could not load the mink-based whole-body IK` WARN (robosuite's own
  `__init__.py:37` import) + 3 `No private macro file found` lines
  per package. See the `_robocasa_*_plan` install steps in
  `openral_sim/_deps.py` for the full list.
- **No LIBERO in the same venv.** `[tool.uv].conflicts` enforces this
  cleanly when both groups would be active; swap venvs (or clones)
  when you need both backends.
- **One informational print remains.**
  `robocasa.__init__` does `try: import mimicgen` + `print()`
  unconditionally; installing mimicgen IS a real fix but mimicgen's
  own `__init__` imports robosuite paths the current master has
  renamed and prints 2 new warning lines, so the net noise is worse
  with mimicgen than without. We leave the one print.

---

## Level 7: NVIDIA GR-1 tabletop tasks (RoboCasa GR1 fork)

The [RoboCasa GR1 Tabletop Tasks](https://github.com/robocasa/robocasa-gr1-tabletop-tasks)
fork — a soft fork of robocasa that NVIDIA shipped alongside the
**GR00T N1** open foundation model
([arXiv:2503.14734](https://arxiv.org/abs/2503.14734)) — adds 24 PnP
tabletop tasks on the Fourier GR-1 humanoid (the
`GR1ArmsAndWaistFourierHands` composition: 7-DoF right arm + 7-DoF
left arm + 3-DoF waist + two 6-DoF Fourier dex hands, leg and head
actuation disabled). The bot-harness sim layer exposes them as
`robocasa/gr1/<TaskName>` scene ids pinned to the `gr1` robot
manifest (`robots/gr1/robot.yaml`).

The two `robocasa`-named python packages (kitchen + GR1 fork) **share
the python package name**, so a host installs ONE or the OTHER -- the
auto-installer picks the variant matching the scene id you requested.
The adapter still registers both task families regardless; the
unavailable one fails at `robosuite.make()` with a clean "unknown
env_name" rather than at import time.

### One-shot run

Drive the GR1 tabletop scene with the in-tree RLDX-1-FT-GR1 nf4 rSkill
via the auto-managed sidecar:

```bash
OPENRAL_AUTO_INSTALL_DEPS=1 \
OPENRAL_ALLOW_ROBOCASA_ASSETS=1 \
OPENRAL_ALLOW_NONCOMMERCIAL=1 \
  openral sim run --config scenes/sim/robocasa_gr1_pnp_cup_to_drawer.yaml \
              --rskill rskills/rldx1-ft-gr1-nf4 --max-steps 30
```

This drives a real `robosuite.make(env_name="PnPCupToDrawerClose",
robots=["GR1ArmsAndWaistFourierHands"])` rollout on the local GPU.

Drop the env-var bypasses on first run for the interactive prompts:

1. `robocasa_gr1` deps banner → `y` (clones robosuite + the GR1 fork
   to `~/.cache/openral/repos/`, installs both editable with the
   `examples/__init__.py` + `macros_private.py` patches dropped in).
2. `RoboCasa GR1 tabletop assets` license banner → `y` (~750 MB
   download to the cloned fork's `robocasa/models/assets/`).

### Sweeping to the other 23 tasks

Swap `scene.id` + `scene.backend_options.prebuilt_task` + `task.scene_id`
in `scenes/sim/robocasa_gr1_pnp_cup_to_drawer.yaml` to any of the 24
tabletop env class names (see `_GR1_TABLETOP_TASKS` in
`openral_sim/backends/robocasa.py`):

- `PnPCupToDrawerClose` / `PnPPotatoToMicrowaveClose` /
  `PnPMilkToMicrowaveClose` / `PnPBottleToCabinetClose` /
  `PnPWineToCabinetClose` / `PnPCanToDrawerClose` — the 6 canonical
  PnP atomic tasks.
- 18 `Posttrain*` variants (Cuttingboard → Basket / Cardboardbox /
  Pan / Pot / Tieredbasket / Plate / etc.) — the post-training task
  set from the GR00T-N1 paper.

### Benchmark protocol

`benchmarks/gr1_tabletop.yaml` mirrors the paper's eval protocol
(scaled to 10 episodes for a ~30 min single-GPU run; bump
`n_episodes` + `seeds` together for the 50-ep paper reproduction):

```bash
openral benchmark run --suite gr1_tabletop --rskill rskills/<gr1-skill>
```

The auto-install prompts fire from the benchmark runner's path too —
`openral benchmark run` and `openral sim run` share the same scene factory.

---

## Where to go next

- The full list of registered IDs on your machine: `openral sim list`.
- The cookbook of existing configs and a per-backend ID table:
  [`scenes/README.md`](https://github.com/OpenRAL/openral/blob/master/scenes/README.md).
- ADRs that explain the design:
  [ADR-0002](../../adr/0002-eval-and-sim-environments.md) (the original
  scene/eval design — the `SceneEnvironment` → `SimScene` rename and the
  three-tier split landed in ADR-0041),
  [ADR-0009](../../adr/0009-separate-sim-and-benchmarking.md) (`openral sim
  run` vs `openral benchmark run`),
  [ADR-0041](../../adr/0041-scene-three-tier-hierarchy.md) (the
  `DeployScene ⊆ SimScene ⊆ BenchmarkScene` hierarchy + per-tier loader
  strictness), and
  [ADR-0015](../../adr/0015-robocasa-isolated-backend-lazy-assets.md)
  (RoboCasa as a free-axis MuJoCo backend with custom robots + tasks —
  rolling out in five PRs per
  [issue #88](https://github.com/OpenRAL/openral/issues/88);
  the Pydantic `RoboCasaBackendOptions` validator and the
  `[dependency-groups].robocasa` extras group already ship today, the
  adapter and a Level-6 procedural-kitchen walkthrough land in later
  PRs).
- The public-symbol inventory for the sim layer:
  [`docs/METHODS.md`](../../METHODS.md), section **Eval (sim)**.
