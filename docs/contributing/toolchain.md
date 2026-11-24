# Toolchain

Use `just` and `uv`. Never `pip install` inside the workspace (use `uv add <pkg> --package <member>`), and never call `colcon build` outside `just` unless you know why. This page covers the Python environment rules, the ROS 2 build, the `openral` wrapper and the full `just` recipe index.

## Managing the Python environment & dependency groups

Read before touching the venv. In priority order:

**1. Always `just sync`, never bare `uv sync`.** The recipe runs [`scripts/repair_hf_libero_install.py`](https://github.com/OpenRAL/openral/blob/master/scripts/repair_hf_libero_install.py) before and after the sync and forces `--all-packages`. Bare `uv sync` skips the repair and fails with `Unable to uninstall hf-libero==0.1.3 ... distutils-installed distributions`, leaving the venv half-broken. Without `--all-packages`, `uv sync --group <x>` syncs only the workspace root and uninstalls every editable `openral-*` member, so the next command fails with `No module named 'openral_core'`. `just sync` is idempotent and safe on every host; scoping to one member with `--package` / `-p` is the only case where it does not add `--all-packages`. It also reinstalls `opencv-python-headless` if a group switch deleted the shared `cv2/` directory.

**2. aarch64 needs `download.pytorch.org`.** PyPI's aarch64 torch wheel is CPU-only, so the root `pyproject.toml` pins torch and torchvision to the `cu130` index under a `platform_machine == 'aarch64'` marker. Without it every in-process policy runs on CPU on a DGX Spark or Jetson. Upgrading an existing aarch64 venv needs a one-time `just sync --frozen --reinstall-package torchvision`. See [aarch64 CUDA hosts](../reference/aarch64-support.md).

**3. Default groups are `dev` + `vla`.** A bare `just sync` installs both. `vla` is the in-process lerobot policy runtime (π0.5, SmolVLA, xVLA, MolmoAct2, OpenVLA, Cosmos 3) with no simulator, so a freshly synced robot host can run those policies in `deploy run`. `deploy run` also checks the scene's `preload_rskill_id` before bringup and refuses to launch, printing the install command, when the policy cannot import. `deploy sim` installs the missing group instead (under `OPENRAL_AUTO_INSTALL_DEPS`). Set `HF_HUB_OFFLINE=1` on hosts whose Hub calls stall.

**4. Everything else is opt-in: `just sync --group <name>`.** A default sync removes optional groups, so simulator work needs at least `sim`.

| Group | Purpose |
|---|---|
| `dev` | Test and lint toolchain, docs, MuJoCo, ROS build helpers (`catkin-pkg`, `empy`). Default. |
| `vla` | In-process lerobot policy runtime, no simulator. Default; included by the simulator groups. |
| `sim` | gym-aloha, gym-pusht, gymnasium[mujoco], Diffusion Policy deps (`vla` included). |
| `libero` | `lerobot[libero]`: LIBERO suites (robosuite 1.4). |
| `robocasa` | robosuite 1.5+ and support deps for RoboCasa. |
| `metaworld` | MetaWorld scene adapter deps. The `metaworld==3.0.0` package itself: `uv run pip install metaworld==3.0.0 --no-deps`. |
| `maniskill3` | ManiSkill3 / SAPIEN backend. |
| `simpler-env` | SimplerEnv real-to-sim backend (the package is installed manually from git). |
| `gr00t` | In-process GR00T-N1.7 on lerobot (`lerobot[groot]`). The NF4 load is OpenRAL Pro. |
| `lowering` | Offline collision-lowering tool: yourdfpy, trimesh, mujoco, scipy. |
| `omdet` | OmDet-Turbo in-process open-vocabulary detector. |
| `sam2` | SAM 2.1 in-process promptable segmenter. |
| `onnx-export` | One-time RT-DETR ONNX export. |
| `clip` | OpenCLIP object matching for spatial memory. |
| `dataset` | LeRobotDataset v3 bridge (`openral-dataset[lerobot]`). |
| `opencv` | Optional OpenCV camera/detector tests. |
| `scene-usd` | `usd-core` to read authored USD catalog data without Isaac Sim. |
| `sidecar-wire` | `pyzmq` + `msgpack`: node-side transport for externally provisioned sidecars. |
| `rldx`, `behavior-groot`, `isaacsim`, `robotwin`, `rlbench`, `lingbot`, `locateanything`, `ovb`, `qwen-vlm` | `sidecar-wire` (plus `websockets` for `lingbot`). The model or simulator runs in its own externally provisioned venv; the group installs only the wire. |

Simulators and heavy deps live in groups, not in core dependencies, and some are provisioned outside the venv because of license or version conflicts (Isaac Sim, CoppeliaSim, RoboTwin assets).

**5. `libero` and `robocasa` are mutually exclusive.** LIBERO pins `robosuite==1.4`; RoboCasa needs `>=1.5`. `[tool.uv] conflicts` declares it, so `just sync --group libero --group robocasa` fails by design. Swap per task:

```bash
just sync --group robocasa      # before a RoboCasa run
just sync --group sim           # (or --group libero) to go back
```

**6. RoboCasa itself is installed editable at runtime, not by `just sync`.** The `robocasa` group provides only robosuite and support deps. The kitchen fork (a GitHub repo, no PyPI release) is cloned and `uv pip install -e --no-deps`'d by `openral_sim._deps.ensure_backend_deps('robocasa_kitchen')`, which the deploy-sim HAL calls from `on_configure`. Auto-install is on by default (`OPENRAL_AUTO_INSTALL_DEPS` unset or `=1`; `=0` prompts). To pre-build it and skip the install window:

```bash
OPENRAL_AUTO_INSTALL_DEPS=1 just sync --group robocasa
OPENRAL_AUTO_INSTALL_DEPS=1 uv run --no-sync python -c \
  "from openral_sim._deps import ensure_backend_deps; ensure_backend_deps('robocasa_kitchen')"
```

Never hand-install `robocasa` or `robosuite`, and never run bare `uv sync --all-packages` on a RoboCasa env: it uninstalls the runtime-editable install and the matching robosuite. Repair with `just sync --group robocasa`, then the `ensure_backend_deps` line above.

## ROS 2

```bash
source /opt/ros/jazzy/setup.bash
just ros2-build                 # colcon build of the packages the recipe lists
source install/setup.bash
just ros2-test                  # colcon test + test-result
```

`ros2-build` and `ros2-test` use the same `--packages-select` set; when a new ROS package lands under `packages/` or `cpp/`, add it to both.

**A new Python node must be committed mode 100755.** `ros2-build` uses `--symlink-install`, so the installed node keeps the source file's mode, `launch_ros` cannot find a 0644 file, and the entire launch description is abandoned. `chmod +x` alone leaves a 100644 blob in the commit; run `git update-index --chmod=+x <file>`. `tests/unit/test_ros_node_exec_bits.py` enforces it.

## GPU motion planning: cuMotion (optional)

NVIDIA Isaac ROS cuMotion is a CUDA MoveIt planning pipeline. The `rskill-moveit-*` family uses it per request (`MotionPlanRequest.pipeline_id`) when the host clears `RobotCapabilities.supports_cumotion()` (Ampere+, compute capability 8.0+, CUDA 13+, about 8 GB VRAM); otherwise MoveIt keeps OMPL. `just bootstrap` installs it on a capable jazzy host. To install by hand, add NVIDIA's [Isaac ROS apt repo](https://nvidia-isaac-ros.github.io/getting_started/isaac_apt_repository.html), then:

```bash
sudo apt install ros-$ROS_DISTRO-isaac-ros-cumotion-moveit \
                 ros-$ROS_DISTRO-isaac-ros-cumotion-robot-description
openral collision lower --robot robots/<robot>/robot.yaml \
        --emit-cumotion robots/<robot>/cumotion_spheres.yaml --write
```

Add the pipeline to your `moveit_config`'s `planning_pipelines`; `--emit-cumotion` generates the cuRobo collision config from the safety kernel's own geometry (the fitter is described in the [collision geometry review](../reference/collision-geometry-review.md); `--tight-link <link>` lowers a link as a box plus its exact convex hull). There is no Python cuRobo and no `uv` group. cuMotion never bypasses the safety kernel: planned trajectories replay through `/openral/candidate_action` and are validated waypoint by waypoint.

## The `openral` wrapper

`just quickstart` and `just install-cli` write `~/.local/bin/openral`, so `openral` runs from any terminal without `just`. The wrapper sources the ROS 2 and colcon overlays, then runs `.venv/bin/openral`. Bare `openral` opens a REPL; pass a subcommand for one-shot use. Run `openral --help` for the command list.

### Which checkout does `openral` run? (`OPENRAL_REPO_ROOT`)

The wrapper is baked to the checkout that generated it. With a second checkout or worktree it still runs the first one's venv, overlay and `robots/` manifests, so a run can be attributed to the wrong branch.

| Situation | What runs | stderr |
| --- | --- | --- |
| `OPENRAL_REPO_ROOT` unset, one checkout | the baked-in checkout | nothing |
| `OPENRAL_REPO_ROOT=/path/to/checkout` | that checkout | `openral: repo root ... (OPENRAL_REPO_ROOT override ...)` |
| `OPENRAL_REPO_ROOT` points at a tree with no executable `.venv/bin/openral` | nothing, exit 1 | an error naming the tree |
| unset, but cwd is in a different checkout with its own `.venv/bin/openral` | the baked-in checkout | `WARNING: cwd is inside the OpenRAL checkout ...` |

In a worktree, `export OPENRAL_REPO_ROOT=$(git rev-parse --show-toplevel)`, or re-run `just install-cli` from it.

## `just` recipe index

`just` with no arguments prints this list. Recipes starting with `_` are internal.

| Area | Recipes |
|---|---|
| Setup | `quickstart` (bootstrap if needed, sync, `ros2-build`, REPL), `bootstrap`, `install-hooks`, `install-cli`, `bootstrap-ollama` (opt-in Ollama + reasoner model pull), `sync`, `install-isaac-ros` (opt-in NVIDIA cuVSLAM/nvblox; runs `sudo`, closed-source EULA packages), `clean` |
| Quality | `lint` (ruff, ruff format check, `tools/run_mypy.sh` over every tracked `.py`, methods-index check, nav2-visual and ROS topic-graph drift, no-Pro-imports), `fmt`, `schema-export`, `gen-nav2-visual` |
| Unit and integration tests | `test`, `test-k <expr>`, `test-doctest`, `test-integration`, `test-ros-live`, `test-changed`, `test-changed-run`, `test-audit` |
| Sim tests | `test-sim`, `hal-twin-sweep`, `test-robocasa-sim` |
| ROS 2 | `ros2-env`, `ros2-build`, `ros2-test` |
| Safety kernel | `safety-kernel-build`, `safety-kernel-test`, `safety-kernel-lint` (clang-format, clang-tidy), `sim-safety <config>` |
| Sim evals | `sim-eval <config>` (canonical), `sim-libero`, `sim-xvla-libero`, `sim-pi05-libero`, `sim-act-libero`, `sim-metaworld`, `sim-act-aloha`, `sim-diffusion-pusht`, `sim-maniskill3`, `sim-simpler-widowx`, `sim-custom`, `sim-audit` (one episode per YAML under `scenes/`) |
| Validation | `validation-matrix`, `validation-matrix-verdicts`, `validation-matrix-diff`, `validation-matrix-import` ([guide](validation-matrix.md)), `profile-load <rskill>` (phase-by-phase load timing) |
| Hardware in the loop | `hil <robot>` (runs `tests/hil/test_<robot>.py`; skips cleanly without hardware), `hil-so101`, `hil-openarm-deploy` |
| Docker | `docker-build-x86`, `docker-build-l4t`, `docker-smoke-x86-deploy`, `docker-smoke-x86-reasoner`, `docker-smoke-x86-safety-kernel` |
| Docs and skills | `docs` (serve), `docs-build` (strict, CI parity), `skill-new <id>` (scaffold an rSkill from `rskills/template/`) |

Other tooling: `uv run ruff check . --fix` autofixes lint; `uv run pytest -k so100` filters by keyword.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `AMENT_TRACE_SETUP_FILES: unbound variable` from `openral` | The wrapper predates the nounset fix. Run `just install-cli`. |
| `openral` runs the wrong branch or manifests | See [`OPENRAL_REPO_ROOT`](#which-checkout-does-openral-run-openral_repo_root). |
| Unsure where a package imports from | `uv run python -c 'import openral_<pkg>; print(openral_<pkg>.__file__)'` |
| ROS topic missing | `ros2 topic list -t` in a shell that sourced `install/setup.bash`. |
| Schema diff in CI | `just schema-export`, then `git diff docs/reference/schemas/`. |
| `import cv2` fails after switching groups | `just sync` reinstalls `opencv-python-headless` automatically. |
| `No module named 'openral_core'` | A bare `uv sync --group` removed the editable members. Run `just sync --all-packages`. |
| CI flake | Re-run once; if it flakes again, label it `flake` and do not merge. |
| Hardware test fails | Check the runner's e-stop log first. Never push a change that makes it pass without knowing why it failed. |
| Out of GPU memory | Lower batch size, quantize or use a smaller skill variant. Never silently downcast or skip frames. |
