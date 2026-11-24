# Create a sim environment

**Goal:** write a `SimScene` YAML for `openral sim run`, and, when no registered scene fits, add a robot manifest, a scene backend or a policy adapter.

A `SimScene` YAML is the `(robot × scene × task)` tuple. `openral sim run` pairs it with an rSkill (`--rskill`) to test a policy in simulation without hardware. The YAML never carries a `vla:` block; the policy always comes from the CLI.

```
DeployScene  ⊆  SimScene  ⊆  BenchmarkScene
 deploy sim      sim run      benchmark scene / benchmark run
```

Add `metadata: {paper, honest_scope}` plus non-`None` `seed` and `n_episodes` and it is a `BenchmarkScene`. Drop the `task:` block and it is a `DeployScene`. Each loader refuses a wrong-tier YAML and names the right command.

## Prerequisites

- You have run [your first sim rollout](first-rollout.md).
- `just sync --group sim`, plus the group your backend needs ([Sim backend installs](../../reference/sim-backend-installs.md)).
- Paste-able halves: `uv run openral sim list` (scene configs) and `uv run openral rskill list` (rSkills). Neither prints the `SCENES` / `POLICIES` / `ROBOTS` registries; import them from Python to inspect those.

---

## 1. Author a SimScene YAML

Start from a shipped file, such as [`scenes/sim/libero_spatial.yaml`](https://github.com/OpenRAL/openral/blob/master/scenes/sim/libero_spatial.yaml):

```yaml
scene:
  id: libero_spatial              # key into SCENES
  backend: mujoco                 # PhysicsBackend; defaults to mujoco
  observation_height: 256
  observation_width: 256

task:
  id: libero_spatial/0            # the adapter splits on "/" to resolve the task
  scene_id: libero_spatial        # must equal scene.id
  instruction: ""                 # LIBERO takes it from the suite metadata
  max_steps: 100
  success_key: is_success         # the info[] key that marks success
```

| Block | Schema | Rules |
| --- | --- | --- |
| `scene` | `SceneSpec` | `id` required |
| `task` | `TaskSpec` | `id` and `scene_id` required; `scene_id` must equal `scene.id` |
| `robot_id` | key into `ROBOTS` | Required on free-axis scenes unless you pass `--robot`. On a scene that fixes its robot (LIBERO, MetaWorld, RoboCasa) it is optional and must name an allowed robot. |
| `seed`, `n_episodes`, `record_video`, `save_dir` | optional | Defaults: `0`, `1`, `false` |

`scene.backend_options` is an opaque dict. Each backend validates it with its own `options_model`, so a misspelled key fails at load with a `ROSConfigError` naming the scene id and the field.

Run it:

```bash
uv run openral sim run --config my_config.yaml --rskill rskills/smolvla-libero
```

It prints one summary line per episode and exits `0` regardless of success. Flags: `uv run openral sim run --help`. The ones you need most are `--task`, `--instruction`, `--max-steps`, `--n-episodes`, `--seed`, `--device`, `--robot` (free-axis scenes only), `--save-dir`, `--save-video`, `--view` and `--dry-run`. A suite file works too: `--config benchmarks/libero_10.yaml --task libero_10/3` (see [Run a benchmark](../benchmark/run-a-benchmark.md)).

To make the YAML paper-comparable, add `robot_id`, `seed`, `n_episodes` and `metadata: {paper, honest_scope}` and move it to `scenes/benchmark/`.

Isaac Sim scenes also take objects, stage objects and camera mounts. Those fields are in [Sim environments: Isaac Sim scenes](../../reference/sim-environments.md#isaac-sim-scenes).

---

## 2. Add a robot manifest

Robots register from `robots/<id>/robot.yaml` at import time, with no Python edit, for `sim run` and `deploy sim|run`. The search path is `$OPENRAL_ROBOTS_DIR/<id>/robot.yaml`, then `<repo>/robots/<id>/robot.yaml`. Templates: [`so100_follower`](https://github.com/OpenRAL/openral/blob/master/robots/so100_follower/robot.yaml) (6 DoF + gripper) and [`franka_panda`](https://github.com/OpenRAL/openral/blob/master/robots/franka_panda/robot.yaml) (7 DoF).

```bash
mkdir -p robots/my_arm
$EDITOR robots/my_arm/robot.yaml       # copy and adapt so100_follower
```

Required: `name`, `embodiment_kind`, `joints`, `capabilities`, `safety`. Optional blocks you will usually want:

| Block | Purpose |
| --- | --- |
| `base_frame` | URDF root frame (default `base_link`) |
| `end_effectors[]` | Grippers and hands, with `command_convention` (and `command_range` for radians or metres). The safety kernel bounds gripper values by it and skills must match it. |
| `sensors[]` | Cameras, IMUs; each maps to a `vla_feature_key` |
| `observation_spec`, `action_spec` | State and action shapes; `action_spec.control_freq_hz` sets the control rate |
| `assets` | URDF, MJCF, SRDF references |
| `sim` | MuJoCo joint-to-qpos wiring for `MujocoArmHAL.from_description` |
| `hal` | Names the sim and real HAL classes (see below) |

### `assets:`

References use the `resolve_asset` grammar:

| Prefix | Meaning |
| --- | --- |
| `rd:<module>` | `robot_descriptions` package, fetched on first use |
| `gym_aloha:<scene>`, `openarm:<variant>`, `unitree:<model>`, `makermods:<variant>`, `galaxea:<model>`, `robotwin:<model>`, `robosuite:<robot>`, `behavior:<model>` | Pinned upstream assets, fetched or derived on first use |
| `file:<relpath>` | Manifest-relative file |
| `ros2://robot_description` | The live ROS parameter (URDF only) |

```yaml
assets:
  mjcf: "rd:ur5e_mj_description"
  # urdf: {ref: "file:ur5e.urdf", root_frame: "base_link"}
  # srdf: "file:ur5e.srdf"        # seeds allowed_collision_pairs
```

### `sim:`

```yaml
sim:
  floating_base: false           # humanoids: qpos offset 7, qvel offset 6 are derived
  # joint_qpos_addr: {joint_a: 0, joint_b: 1}   # only when the MJCF order differs
  # actuator_index: {joint_a: 0}
  # joint_qvel_addr, settle_steps_default       # see SimDescription
  grippers:                      # zero, one or two entries
    - joint: "panda_gripper"
      ctrl_range: [0.0, 255.0]
      qpos_addrs: [7, 8]
      qpos_scale: 0.08
      read_mode: "sum_over_scale"   # | affine_low_high | passthrough
      write_mode: "normalised"      # | passthrough
  # keyframe_index: 0              # mj_resetDataKeyframe on connect (ALOHA)
  # seed_ctrl_from_qpos: true      # ctrl = qpos on connect (OpenArm)
```

In-tree examples: `so100_follower` (revolute jaw), `franka_panda` (parallel gripper), `ur5e` / `ur10e` / `rizon4` (no gripper), `g1` / `h1` (floating base), `aloha_bimanual` (two normalised grippers), `openarm` (passthrough grippers, explicit `joint_qpos_addr`).

If the robot also has a Python `<ROBOT>_DESCRIPTION` constant, mirror the `sim` block there. `tests/unit/test_robot_manifests_match_hal_constants.py` and `test_python_description_matches_yaml` fail when the two drift.

### `hal:`

Leave `hal.sim: null` and `build_hal` builds `MujocoArmHAL.from_description` from your `sim:` block. Name a class in `hal.sim` only when the robot needs behaviour beyond data (a walking controller, a torque hook); it must accept `description=`. `hal.real` names the real-hardware HAL; its constructor kwargs come from `hal.parameters.defaults`. A vendor ros2_control bringup goes in `hal.real_bringup: "<pkg>:<file>.launch.py"` and must call `openral_core.ros_graph_scope.require_host_confined_real_bringup` ([ROS graph security](../../reference/ros-graph-security.md)).

`openral deploy sim --config <scene>` then works with no ROS package: one manifest-driven `openral_hal_node` hosts every robot.

Verify it registered:

```python
from openral_sim import ROBOTS
print(ROBOTS.get("my_arm")().name)
```

### Before this robot touches real hardware

Sim runs on hand-authored geometry; a real deploy does not. Two fits are not forced by `deploy sim`:

- **Self-collision geometry.** `uv run openral collision lower --robot robots/my_arm/robot.yaml --write` fits the collision capsules to the real meshes. An unfitted template capsule can miss its mesh by centimetres and make the kernel's collision checks non-conservative.
- **Depth-camera extrinsic.** For a robot-mounted depth camera, calibrate the mount yourself (a hand-eye calibration) and declare it as `static_transform_xyz_rpy` in `robots/my_arm/units/<unit>.yaml`. `deploy run` refuses a real world-voxel launch on the nominal mount. See [the world-voxel check tutorial](../deploy/openarm-real-world-voxel-check.md), step 2.

`uv run openral detect` prints both reminders when it writes a manifest.

---

## 3. Write a scene backend

Write one when the catalogue cannot express your task: a different arena, objects the existing adapters cannot spawn, or a non-robosuite physics engine. A backend is a function decorated with `@SCENES.register("<id>")` that returns a [`SimRollout`](https://github.com/OpenRAL/openral/blob/master/python/sim/src/openral_sim/rollout.py):

```python
class SimRollout(Protocol):
    scene: SceneSpec
    task: TaskSpec
    def reset(self, seed: int | None = ...) -> Observation: ...
    def step(self, action: NDArray[np.float32]) -> StepResult: ...
    def render(self) -> NDArray[np.uint8] | None: ...
    def close(self) -> None: ...
```

`Observation` is a dict. Include `"images"` (HWC uint8 frames by camera name), `"state"` (1-D float32) and `"task"` (the instruction). `StepResult` carries `observation`, `reward`, `terminated`, `truncated` and `info`; the runner reads success from `info[task.success_key]`.

The smallest reference is the `mock` scene in [`policies/mock.py`](https://github.com/OpenRAL/openral/blob/master/python/sim/src/openral_sim/policies/mock.py) (no physics; accepts any action size). A realistic one is `backends/libero.py`; the arena-composing one is `backends/so101_box/`.

### Skeleton

Create `python/sim/src/openral_sim/backends/my_scene.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict

from openral_sim.registry import SCENES
from openral_sim.rollout import Observation, StepResult

if TYPE_CHECKING:
    from openral_core import SceneSpec, SimEnvironment, TaskSpec


class MySceneOptions(BaseModel):
    """What ``scene.backend_options`` may carry for this backend."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    table_height: float = 0.75


@dataclass
class _MyScene:
    scene: SceneSpec
    task: TaskSpec
    _step: int = 0

    def reset(self, seed: int | None = None) -> Observation:
        self._step = 0
        return self._observe()

    def step(self, action: NDArray[np.float32]) -> StepResult:
        self._step += 1
        done = self._step >= (self.task.max_steps or 1)  # max_steps may be None
        return StepResult(
            observation=self._observe(),
            reward=0.0,
            terminated=done,
            truncated=False,
            info={self.task.success_key: done},
        )

    def render(self) -> NDArray[np.uint8] | None:
        return np.zeros(
            (self.scene.observation_height, self.scene.observation_width, 3), dtype=np.uint8
        )

    def close(self) -> None:
        return None

    def _observe(self) -> Observation:
        return {
            "images": {"camera1": self.render()},
            "state": np.zeros(8, dtype=np.float32),
            "task": self.task.instruction,
        }


@SCENES.register("my_scene", options_model=MySceneOptions)
def _build(env_cfg: SimEnvironment) -> _MyScene:
    return _MyScene(scene=env_cfg.scene, task=env_cfg.task)
```

Then add the module name to the import tuple in `_register_backends()` in `python/sim/src/openral_sim/backends/__init__.py`. That import runs the `@register` decorator. Confirm:

```python
from openral_sim import SCENES
print("my_scene" in SCENES.names())
```

### Registration options

| Argument | Effect |
| --- | --- |
| `options_model=` | Validates `scene.backend_options` at every load site (`make_env`, `sim run`, `benchmark scene`, `deploy validate`). Ids under a registered prefix inherit it (`robocasa/<Task>` uses `robocasa`'s). Read the validated instance back with `SCENES.validate_options`. |
| `fixed_robot=` | One robot id or a set. A `--robot` or `robot_id:` outside it raises `ROSConfigError`; omitting it takes the scene's default. Absent means free-axis. |
| `gripper_convention=` | The gripper encoding your environment consumes, for example `"normalized_close_symmetric"` for a robosuite env. It replaces the robot's own encoding in this scene; the safety kernel bounds gripper chunks by it and the runner refuses a skill that emits another. |
| `sequential_init=True` | Builds the env and the policy one after the other. Set it only if the env thread races the policy thread; see [Sim backend installs](../../reference/sim-backend-installs.md#parallel-versus-sequential-startup). |
| `provision=` | A hook `deploy sim` runs before `ros2 launch`, so a slow install does not land inside the HAL's 300 s configure deadline. Make it idempotent. |

### Optional: `mujoco_handles` for `--view`

If your adapter wraps a MuJoCo model, add `mujoco_handles(self) -> tuple[MjModel, MjData] | None` and `--view` opens a passive viewer. It is not part of the `SimRollout` protocol (the runner uses `getattr`), so other adapters skip it.

---

## 4. Write a policy adapter

A policy adapter is a function decorated with `@POLICIES.register("<id>")` that returns a [`PolicyAdapter`](https://github.com/OpenRAL/openral/blob/master/python/sim/src/openral_sim/policy.py):

```python
class PolicyAdapter(Protocol):
    spec: VLASpec
    device: str
    def reset(self) -> None: ...
    def step(self, observation: Observation, instruction: str) -> NDArray[np.float32]: ...
    def close(self) -> None: ...
```

`step` returns one flat per-step action; chunk caching belongs in the adapter. References: `_ZeroPolicy` in `policies/mock.py` (no weights) and `policies/smolvla.py` (a real checkpoint). Skeleton:

```python
@dataclass
class _MyPolicy:
    spec: VLASpec
    device: str
    action_dim: int = 7

    def reset(self) -> None: ...                    # load weights, clear the action queue

    def step(self, observation: Observation, instruction: str) -> NDArray[np.float32]:
        return np.zeros(self.action_dim, dtype=np.float32)

    def close(self) -> None: ...


@POLICIES.register("my_policy")
def _build(env_cfg: SimEnvironment) -> _MyPolicy:
    return _MyPolicy(spec=env_cfg.vla, device=env_cfg.vla.device)
```

Add the module to the import tuple in `_register_policies()` in `policies/__init__.py`. A policy package outside the repo can register through the `openral.policies` entry-point group instead.

### Pair it with an rSkill

The adapter id is the manifest's `model_family`. Create `rskills/<my-skill>/rskill.yaml` declaring its `embodiment_tags`. The runner refuses a pairing whose tags do not overlap the robot's capabilities (`_check_rskill_compatibility`). Layout: [`rskills/README.md`](https://github.com/OpenRAL/openral/blob/master/rskills/README.md); a template: `rskills/smolvla-libero/`.

To write a LeRobotDataset v3 with `--dataset-out`, the manifest needs both contracts. The bridge reads them, not the robot manifest, because one robot can host checkpoints with different I/O sizes:

```yaml
state_contract: {dim: 8}     # observation.state size
action_contract: {dim: 7}    # action size
```

The `--dataset-out` path must not exist yet; lerobot v3 refuses a populated root.

A visuomotor adapter may also implement `last_input_frame() -> NDArray[np.uint8] | None`; with `--save-video` the runner stitches the frame the policy saw into the debug video.

### Smoke test

```bash
uv run openral sim run --config scenes/<your-config>.yaml \
  --rskill rskills/<my-skill> --n-episodes 1 --max-steps 10
```

---

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `ROSConfigError` about `scene_id` | `task.scene_id` must equal `scene.id` |
| `ROSConfigError` about a `vla:` block | Delete it; pass `--rskill` |
| Wrong-tier redirect message | Use the command it names (`deploy sim`, `benchmark scene`) |
| `--robot` rejected | The scene fixes its robot; the error lists the allowed ones |
| Bad `backend_options` key | The error names the scene id and the field; check the backend's options model |
| `TypeError` comparing `max_steps` | `TaskSpec.max_steps` can be `None`; guard it as in the skeleton |
| `ROSCapabilityMismatch` | The rSkill's `embodiment_tags` or sensors do not match the robot |

## Next

- [Sim environments reference](../../reference/sim-environments.md): every shipped scene and the tier rules.
- [Sim backend installs](../../reference/sim-backend-installs.md): RoboCasa, GR1 and Isaac provisioning.
- [Run a benchmark](../benchmark/run-a-benchmark.md): turn a scene into a published number.
- [`scenes/README.md`](https://github.com/OpenRAL/openral/blob/master/scenes/README.md): the directory layout.
