# Development setup

This page walks you through getting a working OpenRAL development environment from scratch — whether on a local Ubuntu machine, inside a dev container, or in a GitHub Codespace. It also covers the day-to-day commands you'll use while contributing.

---

## System requirements

| Requirement | Minimum | Recommended |
|---|---|---|
| OS | Ubuntu 22.04 or 24.04 | Ubuntu 24.04 (ROS 2 Jazzy) |
| Python | 3.12 (only — `pyproject.toml` pins `>=3.12,<3.13`) | 3.12 |
| RAM | 8 GB | 16 GB |
| Disk | 20 GB free | 40 GB free |
| GPU | CPU-only (limited) | NVIDIA, 8 GB VRAM |
| Docker | 24+ (for dev container) | 26+ |

macOS 14+ is supported for Python/tooling work. ROS 2 on macOS runs inside the dev container.

---

## Option A — Local Ubuntu machine (fastest iteration)

### 1. Clone and bootstrap

```bash
git clone https://github.com/OpenRAL/openral
cd OpenRAL
just bootstrap          # installs uv, ROS 2, system deps (~5–10 min)
source /opt/ros/jazzy/setup.bash   # or 'humble' on Ubuntu 22.04
uv sync --all-packages  # install Python workspace deps
```

The bootstrap script auto-detects Ubuntu 22.04 (→ ROS 2 Humble) or 24.04 (→ ROS 2 Jazzy).

### 2. Verify the install

```bash
just test            # full unit suite, <30 s
just lint            # ruff + mypy --strict; expect no errors
uv run openral doctor     # diagnose host environment
```

`just test` should run the full unit suite and complete in well under
30 s. The current inventory (file + LOC counts, gaps, follow-ups) is in
[`tests/README.md`](https://github.com/OpenRAL/openral/blob/master/tests/README.md).

`uv run openral doctor` is the canonical environment probe — see the
[README's "Quick start" section](https://github.com/OpenRAL/openral/blob/master/README.md#quick-start)
for a sample output table; that one is the single source of truth.
Each row will read `ok` / `info` / `absent` / `missing` depending on
which deps you have installed.

---

## Option B — Dev container (VS Code / Docker Desktop)

### Prerequisites

- [VS Code](https://code.visualstudio.com/) with the [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)
- Docker Desktop 24+ running locally

### 1. Open in container

```
F1  →  "Dev Containers: Reopen in Container"
```

VS Code will build `Dockerfile.dev` (first build ~8 min; subsequent builds use the layer cache) and run `uv sync --all-packages` automatically.

### 2. Or use docker-compose directly

```bash
docker compose -f docker-compose.dev.yml up -d openral-dev
docker compose -f docker-compose.dev.yml exec openral-dev bash
# inside container:
uv sync --all-packages
just test
```

The compose file also starts a **Jaeger** container for OpenTelemetry traces at `http://localhost:16686`.

!!! note "Hardware access"
    The dev container is started with `--privileged` and `/dev` mounted. On Linux hosts this gives USB access to connected robots. On macOS/Windows Docker Desktop, hardware passthrough is limited — use Option A (native Ubuntu) for hardware-in-the-loop work.

---

## Option C — GitHub Codespaces (browser / no local install)

1. On the GitHub repo page: **Code → Codespaces → Create codespace on main**.
2. Wait for the container to build and `uv sync --all-packages` to complete (~5 min).
3. Open a terminal and run `just test`.

!!! note
    Hardware mounts (`/dev`, `/run/udev`) are not available in Codespaces. All unit and sim tests work; HIL tests require a self-hosted runner.

---

## Day-to-day commands

All commands use `just` as the task runner. Run `just` (no arguments) to list all targets.

### Python tests

```bash
just test                   # all tests in tests/unit/
just test-k so100           # filter by keyword
uv run pytest tests/unit/test_schemas_fuzz.py -v   # run a specific file
```

### Linting and formatting

```bash
just lint       # ruff check + ruff format --check + mypy --strict (CI parity)
just fmt        # auto-fix: ruff format + ruff check --fix
```

### Schema export

```bash
just schema-export          # regenerate docs/reference/schemas/*.json
uv run python tools/schema_export.py --check   # CI drift-check (exit 1 on drift)
```

If you change any Pydantic model in `python/core/`, run `just schema-export` and commit the updated JSON files.

### ROS 2 build and test

```bash
source /opt/ros/jazzy/setup.bash
just ros2-build             # colcon build --merge-install
source install/setup.bash
just ros2-test              # colcon test + colcon test-result --verbose
```

### Sim evals (closed-loop, opt-in — needs HF weights ± GPU)

```bash
just sim-eval scenes/<name>.yaml   # canonical entry point (ADR-0002)
just sim-libero                              # SmolVLA × LIBERO
just sim-xvla-libero                         # xVLA × LIBERO
just sim-pi05-libero                         # π0.5 × LIBERO (≥8 GB VRAM)
just sim-metaworld --task reach-v3           # SmolVLA × MetaWorld MT50
just sim-act-aloha                           # ACT × gym-aloha bimanual cube
just sim-diffusion-pusht                     # Diffusion Policy × gym-pusht (CPU)
```

### Docs

```bash
just docs                   # serve at http://localhost:8000 (live-reload)
just docs-build             # full build (CI parity, strict mode)
```

---

## Repository layout (quick reference)

```
python/core/         openral_core         — Pydantic v2 schemas (normative)
python/cli/          openral_cli          — `openral` CLI entry point
python/hal/          openral_hal          — HAL Protocol + per-robot adapters
                                                  (so100_sim, so100_follower, franka_panda,
                                                   ur5e/ur10e, ros_control)
python/sensors/      openral_sensors      — sensor catalog + vendor adapters
python/world_state/  openral_world_state  — WorldStateAggregator (30 Hz snapshot)
python/rskill/        openral_rskill        — Skill ABC, rSkill loader, runtimes,
                                                  SmolVLA adapter
python/sim/          openral_sim          — openral sim run registry/runner; LIBERO/MetaWorld
packages/msgs/                                — ROS 2 IDL (.msg, .action)
packages/world_state/                         — WorldState lifecycle node
packages/openral_hal_*/                   — per-robot lifecycle nodes (so100, ur5e,
                                                  ur10e, franka)
robots/                                       — canonical RobotDescription manifests
                                                  (so100_follower, franka_panda, sawyer,
                                                   aloha_bimanual, pusht_2d,
                                                   ur5e, ur10e); auto-discovered
                                                   by openral_sim at import.
skills/                                       — rSkill packages (manifest + eval/)
                                                  (smolvla-libero, smolvla-metaworld,
                                                   pi05-libero-nf4,
                                                   pi05-robocasa365-human300-nf4,
                                                   xvla-libero, act-aloha,
                                                   act-aloha-insertion,
                                                   diffusion-pusht)
examples/                                     — runnable end-to-end demos +
                                                  scenes/ SceneEnvironment YAMLs
tests/unit/                                   — pytest unit tests (<30 s total)
tests/integration/                            — launch_testing multi-node tests
tests/sim/                                    — closed-loop sim (CUDA + HF weights, opt-in)
tests/hil/                                    — hardware-in-the-loop (lab runners)
tools/schema_export.py                        — JSON Schema generator + drift check
tools/rskill_publisher.py                      — rSkill packaging / publish helper
docs/                                         — mkdocs-material site
```

Most directories carry a per-package `README.md` with usage examples and
links back to the canonical schemas — start with the package matching
the layer you're touching.

Full architecture is in [docs/architecture/overview.md](../architecture/overview.md). For a per-module status canvas (working / in-dev / planned / out-of-scope, with inputs, outputs, and schemas), open [docs/architecture/repo-state-map.html](../architecture/repo-state-map.html) — keep it in sync per CLAUDE.md §4.3.

---

## Making a pull request

1. Create a branch: `git switch -c feat/your-feature`.
2. Make your changes. Run `just lint && just test` before pushing.
3. If you changed a Pydantic schema, run `just schema-export` and commit the updated JSON files.
4. Open a PR. The title should follow [Conventional Commits](https://www.conventionalcommits.org/) — e.g. `feat(core): add FooSchema`.
5. All CI checks must be green before merge. See `.github/workflows/` for what runs.

The full PR checklist is in the repo-root `CLAUDE.md` (not linked from docs — open it directly in your editor).

---

## Troubleshooting

### `uv sync` fails with "package not found"

Run `uv sync --all-packages` instead of plain `uv sync`. The workspace has multiple member packages and the root manifest does not list them as direct dependencies.

### `mypy` reports "missing library stubs or py.typed"

Both `openral_core` and `openral_cli` ship `py.typed` markers. If mypy still complains, make sure you ran `uv sync --all-packages` and that your interpreter is the workspace venv (`which python` should point into `.venv/`).

### `openral doctor` shows ROS 2 as "missing" after bootstrap

Source the ROS 2 setup file:

```bash
source /opt/ros/jazzy/setup.bash   # Ubuntu 24.04
# or
source /opt/ros/humble/setup.bash  # Ubuntu 22.04
```

Add this line to your `~/.bashrc` for permanent effect.

### Docker build fails copying `python/core/pyproject.toml`

Make sure you're building from the **repo root** (the `docker-compose.dev.yml` sets `context: .`). Building directly with `docker build -f Dockerfile.dev .` from the repo root also works.

### Pre-commit hooks fail

Install the hooks once after cloning:

```bash
uv run pre-commit install
```

Then `git commit` will run ruff and mypy on changed files automatically.
