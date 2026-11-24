# Development setup

Set up a working OpenRAL environment (local Ubuntu, dev container or Codespace), then learn the day-to-day `just` recipes and the PR flow.

## Requirements

| Requirement | Minimum | Recommended |
|---|---|---|
| OS | Ubuntu 22.04 or 24.04 | Ubuntu 24.04 (ROS 2 Jazzy) |
| Python | 3.12 only (`>=3.12,<3.13`) | 3.12 |
| RAM | 8 GB | 16 GB |
| Disk | 20 GB free | 40 GB free |
| GPU | none (limited) | NVIDIA, 8 GB VRAM |
| Docker | 24+ (dev container) | 26+ |

macOS 14+ works for Python and tooling; ROS 2 on macOS runs in the dev container.

## Option A: local Ubuntu

```bash
git clone https://github.com/OpenRAL/openral
cd openral
just bootstrap                     # uv, ROS 2, system deps, git hooks (~5-10 min)
source /opt/ros/jazzy/setup.bash   # 'humble' on Ubuntu 22.04
just sync                          # Python workspace; never bare `uv sync`
```

Bootstrap picks ROS 2 Humble on Ubuntu 22.04 and Jazzy on 24.04. Verify:

```bash
just test                 # unit suite
just lint                 # see the recipe table below
uv run openral doctor     # host probe: each row is ok / info / absent / missing
```

## Option B: dev container

Install VS Code with the Dev Containers extension and Docker 24+, then run **Dev Containers: Reopen in Container**. The first build of `Dockerfile.dev` takes about 8 minutes and runs `uv sync --all-packages` for you. Without VS Code:

```bash
docker compose -f docker-compose.dev.yml up -d openral-dev
docker compose -f docker-compose.dev.yml exec openral-dev bash
just sync && just test      # inside the container
```

The compose file also starts Jaeger for OpenTelemetry traces at `http://localhost:16686`. The container runs `--privileged` with `/dev` mounted, which gives USB access on Linux hosts; on macOS and Windows hardware passthrough is limited, so use Option A for hardware work.

## Option C: GitHub Codespaces

**Code → Codespaces → Create codespace on master**, wait for the build (about 5 minutes), then `just test`. `/dev` and `/run/udev` are not available, so HIL tests cannot run there.

## Git hooks and DCO

`just install-hooks` (run by `just bootstrap`) does `git config core.hooksPath .githooks` and pre-builds the [pre-commit](https://pre-commit.com) environments. Use it instead of `pre-commit install`, which refuses to run while `core.hooksPath` is set.

The hooks give you:

- **DCO auto-sign-off** (`prepare-commit-msg`): appends `Signed-off-by` from your Git identity. A clone without the hooks silently produces unsigned commits, and the CI `Verify Signed-off-by` check then fails.
- **On commit** (`pre-commit` / `commit-msg`): trailing whitespace, EOF, YAML/TOML validity, large files, merge markers, private keys, ruff, ruff-format, mypy, the no-Pro-imports check, schema-export drift, repo-state-map drift, ROS topic-graph regeneration, clang-format, codespell and the Conventional Commits check.

The hooks never sync `.venv` (`UV_NO_SYNC=1`), so they run against the environment you last synced and a commit cannot revert an uncommitted `uv.lock` bump. Run `just sync` before your first commit on a fresh clone.

## Recipes

`just` with no arguments lists every recipe; the [full index](toolchain.md#just-recipe-index) is in the toolchain page.

| Recipe | What it runs |
|---|---|
| `just sync [--group <g>]` | `uv sync --all-packages` plus repairs (see [toolchain](toolchain.md#managing-the-python-environment-dependency-groups)) |
| `just test` | `pytest tests/unit/` (ROS pytest plugins disabled); GStreamer tests run only if `gi` imports |
| `just test-k <expr>` | the unit suite filtered with `-k` |
| `just test-changed` / `test-changed-run` | print / run only the pytest targets your diff needs ([selective testing](selective-testing.md)) |
| `just test-doctest` | `--doctest-modules` over a curated package list (`DOCTEST_TARGETS` in `tests/unit/test_doctest_runner.py`) |
| `just test-integration` | ROS-gated integration tests (ROS sourced) |
| `just test-ros-live` | live-ROS suite ([CI overview](ci.md#docker-build-and-the-live-ros-suite)) |
| `just test-sim`, `just hal-twin-sweep` | sim tier and the every-HAL digital-twin sweep (real weights / GPU as needed) |
| `just lint` | `ruff check`, `ruff format --check`, `tools/run_mypy.sh`, the [methods-index check](../METHODS.md) (`refresh_methods_linenos.py --check --coverage`), `gen_nav2_visual.py --check`, `gen_ros_topic_graph.py --check`, `check_no_pro_imports.py` |
| `just fmt` | `ruff format .` |
| `just schema-export` | regenerate `docs/reference/schemas/*.json`; commit them when you change a Pydantic model (`uv run python tools/schema_export.py --check` is the drift check) |
| `just ros2-build` / `ros2-test` | `colcon build` / `colcon test` of the ROS workspace; source ROS first, then `install/setup.bash` |
| `just sim-eval scenes/<name>.yaml` | canonical config-driven sim eval |
| `just sim-libero`, `sim-metaworld`, `sim-act-aloha`, ... | per-suite demo recipes (full list in the toolchain index) |
| `just test-audit` | regenerate the [test audit](test-audit.md) |
| `just validation-matrix` | [collision-stack validation matrix](validation-matrix.md) |
| `just docs` / `docs-build` | serve docs at `http://localhost:8000` / strict build (CI parity) |

`just ros2-build` first runs `scripts/check_ros_build_deps.sh`, which lists any missing ament packages with a ready `apt-get install` line instead of letting CMake fail 45 s in. It is advisory (`OPENRAL_ROS_DEPS_STRICT=1` makes it fatal) and does nothing when ROS 2 is not sourced.

The `pytest` and `launch_testing` ROS plugins break unit-test collection when ROS 2 is sourced; the recipes pass `-p no:launch_testing -p no:launch_ros`. Do the same when you call `uv run pytest` by hand in a ROS shell.

## Hardware-in-the-loop

HIL tests live in `tests/hil/` and run by hand on the cell; no CI lane exists ([why](ci.md#lab-hil-runners-none-registered)). Each file skips cleanly off-rig. `just hil <robot>` runs `tests/hil/test_<robot>.py`.

### Running the SO-101 HIL gate

The SO-101 bench arm gate is non-motion serial. On the host the arm is plugged into:

1. Give your user bus access: `sudo usermod -aG dialout <user>`, then re-login. Without it every test skips, green and useless.
2. Check the bench against the committed scene, then run the gate:

    ```bash
    ls -l /dev/v4l/by-id /dev/v4l/by-path
    uv run openral deploy validate --config scenes/deploy/so101_bench.yaml
    just hil-so101
    ```

    Device paths in `so101_bench.yaml` are host-specific; `deploy validate` flags a stale one as a warning. Fix the scene.

!!! warning "Keep a hand near the power switch"
    The lane sends no commands, but `connect()` cycles servo torque off and on, and re-enabling torque makes a servo hold its last `Goal_Position`. A stale one can twitch the arm. Whoever runs the gate stays within reach of the power switch. A motion test belongs behind an explicit attended-bench gate (see `tests/hil/test_openarm_slot_group_motion.py`, which needs two opt-in env vars), never unattended.

### OpenArm

`just hil-openarm-deploy`: non-motion, but it attaches to a graph that is not. `openral deploy run` on real OpenArm hardware moves both arms to zero at bringup, unramped. Start the graph yourself, with the cell clear and a hand on the hardware E-stop, then run the gate.

## Pull requests

1. Branch: `git switch -c feat/your-feature`.
2. Run `just lint && just test` before pushing, plus the relevant `just sim-*` recipe.
3. Changed a Pydantic schema? Run `just schema-export` and commit the JSON. Added or moved a public symbol? Update the matching file in `docs/methods/` ([CLAUDE.md §4.4](https://github.com/OpenRAL/openral/blob/master/CLAUDE.md)).
4. Open the PR with a [Conventional Commits](https://www.conventionalcommits.org/) title, e.g. `feat(core): add FooSchema`, using the PR template.
5. Every push runs `quality` and `select-and-test`; `heavy-lanes` starts by itself once the PR is ready. Merging needs all required checks green and every thread resolved.

### Review policy

The maintainer reviews and merges every PR; the ruleset requires four green checks and resolved threads. Full policy, check triggers and the heavy-lanes start conditions: [CI overview](ci.md#review-policy).

## Troubleshooting

| Symptom | Fix |
|---|---|
| `uv sync` fails with "package not found" or "Unable to uninstall hf-libero" | Use `just sync`, never bare `uv sync` ([why](toolchain.md#managing-the-python-environment-dependency-groups)). |
| mypy: "missing library stubs or py.typed" | Run `just sync` and make sure `which python` points into `.venv/`. |
| `openral doctor` shows ROS 2 "missing" | `source /opt/ros/jazzy/setup.bash` (or `humble`); add it to `~/.bashrc`. |
| Docker build cannot copy `python/core/pyproject.toml` | Build from the repo root (`docker-compose.dev.yml` sets `context: .`). |
| Commit is unsigned or hooks do not run | `just install-hooks`; amend with `git commit --amend -s`. |

Directory layout: [`repo-map.md`](../architecture/repo-map.md). Module status: [`repo-state-map.html`](../architecture/repo-state-map.html).
