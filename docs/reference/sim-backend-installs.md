# Sim backend installs

Most scene backends need an opt-in dependency group. A few (RoboCasa, GR1, Isaac Sim and the sidecar backends) also clone a fork, download assets or build a separate interpreter. This page lists what each one needs and what OpenRAL provisions for you. The code is [`_deps.py`](https://github.com/OpenRAL/openral/blob/master/python/sim/src/openral_sim/_deps.py) and the `provision_*` hooks in each backend module.

Always sync with `just sync --group <name>`, never a bare `uv sync`. Rules: [Managing the Python environment & dependency groups](../contributing/toolchain.md#managing-the-python-environment-dependency-groups).

## Backends and groups

| Scene backend | Group | Provisioned on first use | Notes |
| --- | --- | --- | --- |
| PushT, ALOHA, custom MuJoCo scenes | `sim` | nothing extra | CPU is enough for PushT |
| LIBERO | `libero` | nothing extra | Conflicts with `robocasa`; swap per task |
| MetaWorld | `metaworld` | nothing extra | Needs `uv run pip install metaworld==3.0.0 --no-deps` once (see the `sim-metaworld` recipe) |
| ManiSkill3, SimplerEnv | `maniskill3`, `simpler-env` | nothing extra | SAPIEN; scenes build sequentially (see below) |
| RoboCasa kitchen, GR1 tabletop | `robocasa` | fork clone, editable install, asset download | Details below |
| Isaac Sim | `isaacsim` | the sidecar interpreter (opt-in) | Details below |
| RoboTwin, RLBench, BEHAVIOR-1K | `robotwin`, `rlbench`, `behavior-groot` | a pinned sidecar interpreter you provision out of band | The group only adds the client wire (`pyzmq`, `msgpack`) |

Auto-install is on by default. Set `OPENRAL_AUTO_INSTALL_DEPS=0` to get a banner listing every install step and a confirmation prompt instead.

A failed install step streams its output to your terminal and quotes its tail in the `ROSConfigError`. Inside a ROS node the raw output is scattered through the launch log, so read the quoted tail first. A sidecar pinned to wheels that do not exist for your platform (usually aarch64) shows up as a `uv` "no source distribution or wheel" line; see [aarch64 CUDA hosts](aarch64-support.md).

## RoboCasa kitchen

`just sync --group robocasa` provides robosuite and its dependencies only. The first `openral sim run` on a `robocasa/<Task>` scene runs the rest (`_robocasa_kitchen_plan`):

- clones robosuite at a pinned commit and the kitchen fork (`git clone --depth=1`) into `~/.cache/openral/repos/`;
- adds the missing `__init__.py` files and empty `macros_private.py` stubs, and relaxes the fork's exact-version asserts on mujoco, numpy and robosuite at install time (there is no runtime version spoof);
- installs both editable with `uv pip install --no-deps -e`.

The CC-BY-4.0 kitchen asset bundle (about 11 GB) downloads to `~/.cache/openral/robocasa/` behind a license banner. `OPENRAL_ALLOW_ROBOCASA_ASSETS=1` skips the prompt.

```bash
OPENRAL_ALLOW_ROBOCASA_ASSETS=1 uv run openral sim run \
  --config scenes/sim/robocasa_pnp.yaml \
  --rskill rskills/rldx1-ft-rc365-nf4 --max-steps 200
```

The scene fixes the robot to `panda_mobile` (or `panda_mobile_vslam`); `--robot franka_panda` fails with `ROSConfigError`. `backend_options` is validated by `RoboCasaBackendOptions`.

**Procedural kitchens.** Set `scene.id: robocasa` (no slash) with `mode: procedural`. The keys are `kitchen_style` and `layout_id` (both 0 to 9), `fixtures`, `spawn_objects`, `task_verb` (`pnp`, `open`, `close`, `press`, `navigate`), `robots`, `controller` and `horizon`. A prebuilt scene uses `mode: prebuilt` instead; mixing the two is refused. "Procedural" varies the kitchen, not the robot.

LIBERO and RoboCasa pin conflicting robosuite versions, so they cannot share a venv. A fresh venv's older robosuite can also shadow the fork; provisioning raises `ROSConfigError` when it does.

## GR1 tabletop

The [RoboCasa GR1 tabletop fork](https://github.com/robocasa/robocasa-gr1-tabletop-tasks) adds 24 pick-and-place tasks on the Fourier GR-1 humanoid: 6 canonical `PnP*` tasks and 18 `Posttrain*` variants. Scene ids are `robocasa/gr1/<TaskName>`, fixed to the `gr1` robot.

The kitchen and GR1 forks share the Python package name `robocasa`, so a host holds one of them. The plan matching the requested scene id installs its fork (`robocasa_gr1`); the CC-BY-4.0 tabletop assets (about 1 GB) download on first use behind the same license prompt.

```bash
OPENRAL_ALLOW_ROBOCASA_ASSETS=1 OPENRAL_ALLOW_NONCOMMERCIAL=1 \
  uv run openral sim run \
    --config scenes/sim/robocasa_gr1_pnp_cup_to_drawer.yaml \
    --rskill rskills/rldx1-ft-gr1-nf4 --max-steps 30
```

`OPENRAL_ALLOW_NONCOMMERCIAL=1` is there because that rSkill's license is non-commercial.

To sweep the other tasks, change `scene.id`, `scene.backend_options.prebuilt_task` and `task.scene_id` together. The full list is `_GR1_TABLETOP_TASKS` in `openral_sim/backends/robocasa.py`. The suite `gr1_tabletop` (10 episodes) runs through `uv run openral benchmark run --suite gr1_tabletop --rskill rskills/<gr1-skill>`.

## Isaac Sim

Isaac Sim and Isaac Lab are RTX-only, license-gated and tens of GB. They are never vendored and run in a py3.11 sidecar. The client side needs `just sync --all-packages --group isaacsim --inexact`.

The sidecar interpreter resolves in this order:

1. `OPENRAL_ISAAC_SIDECAR_PYTHON`, an absolute path to a python.
2. A pip venv at `~/.cache/openral/isaac-sidecar/.venv`. Setting `OPENRAL_ISAAC_AUTO_PROVISION=1` builds it from NVIDIA's index (a multi-GB download); otherwise provisioning is manual and the error message prints the recipe.
3. A host-wide binary install at `/opt/isaac-sim` or `~/isaacsim` (its `python.sh`).

Other switches: `OPENRAL_ISAAC_AUTO_SPAWN=0` stops the client from launching the sidecar itself. Isaac Sim 6.1 is the verified release.

## Parallel versus sequential startup

`SimRunner.activate()` builds the env and the policy concurrently, so startup takes the slower of the two instead of their sum. Scenes registered with `sequential_init=True` build one after the other, because their env thread races the policy thread's `transformers` import or its bf16 default-dtype window. That covers `robocasa` and `robocasa/gr1`, `maniskill3`, `simpler_env`, `openarm_tabletop_pnp`, `tabletop_push` and the MakerMods Metal scenes. LIBERO and MetaWorld run in parallel. `OPENRAL_SIM_SEQUENTIAL_INIT=1` forces sequential for any scene.

Each run logs a `sim_init_parallel` record with `env_ms`, `policy_ms`, `total_ms` and `saved_ms`.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `ROSConfigError` quoting an install step's tail | Read the quoted lines; they are the failing command's output |
| `uv` "no source distribution or wheel for the current platform" | A sidecar pin has no wheel for your platform; see [aarch64 CUDA hosts](aarch64-support.md) |
| `Group ... conflicts` on sync | LIBERO and RoboCasa are mutually exclusive; sync the one you need |
| Isaac "sidecar venv not found" | Provision it, set `OPENRAL_ISAAC_SIDECAR_PYTHON`, or set `OPENRAL_ISAAC_AUTO_PROVISION=1` |
| RoboCasa asset prompt in CI | Set `OPENRAL_ALLOW_ROBOCASA_ASSETS=1` |
