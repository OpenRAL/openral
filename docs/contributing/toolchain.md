# Toolchain cheatsheet

Always use `just` and `uv`. **Never** `pip install` inside the workspace — use `uv add <pkg> --package <member>`. **Never** call `colcon build` outside `just` unless you know why.

## Bootstrap (Ubuntu 22/24, macOS 14+)

```bash
just bootstrap                  # installs uv, ROS 2, system deps via scripts/
```

## Python workspace

```bash
just sync                       # resolve & install all workspace deps
                                # (wraps `uv sync` + scripts/repair_hf_libero_install.py)
just sync --group robocasa      # + an opt-in dep group (sim / libero / robocasa / rldx / …)
                                # → read "Managing the Python environment & dependency
                                #   groups" below before swapping groups or running RoboCasa
just test                       # run unit tests (<30 s)
just test-doctest               # run docstring examples on the curated set
uv run pytest -k so100          # filter by keyword
just lint                       # ruff check + ruff format --check + mypy --strict
                                # (mypy targets: openral_core, openral_cli, openral_sim, tools/)
uv run ruff check . --fix       # autofix
uv run ruff format .            # format
```

## Managing the Python environment & dependency groups

Read this before touching the venv — these are the rules that keep a working
tree, in priority order.

**1. Always `just sync`, never bare `uv sync`.** The Justfile `sync` recipe
wraps [`scripts/repair_hf_libero_install.py`](https://github.com/OpenRAL/openral/blob/master/scripts/repair_hf_libero_install.py)
**before and after** the sync and forces `--all-packages` for you. Bare
`uv sync` skips the repair and trips on:

```
error: Unable to uninstall hf-libero==0.1.3. distutils-installed distributions
do not include the metadata required to uninstall safely.
```

which aborts the resolve and leaves the venv half-broken (missing `h5py`,
`transformers`, `scipy`, …). `just sync` is idempotent and safe on every host.

```bash
just sync                       # the only correct full-workspace sync
```

**2. Opt-in dependency groups → `just sync --group <name>`.** The
`[dependency-groups]` in `pyproject.toml` define `sim`, `libero`, `robocasa`,
`metaworld`, `maniskill3`, `rldx`, … Heavy runtime deps — `transformers==5.3.0`,
`scipy`, `opencv`, robosuite — live in these groups, **not** in the core deps.
A default sync (no `--group`) deliberately *removes* them, so for any VLA / sim
work you need at least the `sim` group:

```bash
just sync --group sim           # minimum working VLA / sim baseline
just sync --group libero        # LIBERO suites
just sync --group robocasa      # RoboCasa robosuite + supporting deps
just sync --group metaworld     # MetaWorld
just sync --group maniskill3    # ManiSkill3 / SAPIEN
```

`just sync --group robocasa` is exactly `uv sync --all-packages --group robocasa`
— the wrapper supplies `--all-packages` so the editable `openral-*` members are
never silently uninstalled (a bare `uv sync --group <x>` drops them and the next
`openral …` can't `import openral_core`).

**3. The libero ↔ robocasa conflict — swap groups per task.** LIBERO pins
`robosuite==1.4` + a specific MuJoCo; RoboCasa needs `robosuite>=1.5` + a
different MuJoCo. They are declared mutually exclusive in `pyproject.toml`
(`[tool.uv] conflicts`) and **cannot** coexist in one resolution — a
`uv sync --group libero --group robocasa` fails by design. You **swap** the
active group per task:

```bash
just sync --group robocasa      # before a RoboCasa run
just sync --group sim           # (or --group libero) to go back to LIBERO/MuJoCo
```

**4. RoboCasa itself is installed editable AT RUNTIME — not by `just sync`.**
The `robocasa` group only provides robosuite + supporting deps. The RoboCasa
kitchen fork (a GitHub repo, no PyPI release) is git-cloned and
`uv pip install -e --no-deps`'d by
`openral_sim._deps.ensure_backend_deps('robocasa_kitchen')`, which the
deploy-sim HAL triggers from `on_configure`. Auto-install is **on by default**
(`OPENRAL_AUTO_INSTALL_DEPS` unset or `=1`; set `=0` to prompt instead). So the
correct way to run a RoboCasa scene is to let the HAL provision it:

```bash
OPENRAL_AUTO_INSTALL_DEPS=1 openral deploy sim \
  --config scenes/deploy/robocasa_navigate.yaml --rskill <rskill>
```

Do **not** hand-install `robocasa` / `robosuite` yourself — that pulls the wrong
robosuite and wrecks the managed env.

**5. Pre-build to skip the in-`on_configure` install window.** Provision the
RoboCasa clone once, ahead of time, so the lifecycle transition doesn't stall
on a first-run build:

```bash
OPENRAL_AUTO_INSTALL_DEPS=1 just sync --group robocasa
OPENRAL_AUTO_INSTALL_DEPS=1 python -c \
  "from openral_sim._deps import ensure_backend_deps; ensure_backend_deps('robocasa_kitchen')"
```

**6. Never `uv sync --all-packages` to "repair" the RoboCasa env.** Once the
runtime-editable RoboCasa install is in place, a plain `uv sync --all-packages`
*uninstalls* it (and the matching robosuite) and breaks the env. Repair with the
group re-applied — `just sync --group robocasa` — and, if RoboCasa itself was
evicted, re-run the `ensure_backend_deps('robocasa_kitchen')` line from point 5.

## ROS 2

```bash
just ros2-build                 # colcon build (msgs + hal_so100 + world_state + reasoner_ros
                                # + prompt_router + safety + safety_watchdog + safety_kernel
                                # + human_estop + skill_ros)
just ros2-test                  # colcon test + colcon test-result --verbose
source install/setup.bash       # after build
```

## Sim

```bash
just sim-eval <config>          # canonical config-driven entry point — ADR-0002
                                # (`openral sim run --config FILE`)
just sim-libero                 # SmolVLA × LIBERO (real lerobot[libero]; needs GPU + MUJOCO_GL)
just sim-xvla-libero            # xVLA × LIBERO   (Florence-2 backbone)
just sim-pi05-libero            # π0.5 × LIBERO   (≥8 GB VRAM)
just sim-metaworld --task TASK  # SmolVLA × MetaWorld (e.g. --task metaworld/reach-v3)
just sim-act-aloha              # ACT × gym-aloha bimanual cube transfer
just sim-diffusion-pusht        # Diffusion Policy × gym-pusht (CPU)
just sim-custom                 # custom example — ACT × gym-aloha insertion
```

## Hardware-in-loop (requires connected robot + USB perms)

```bash
just hil so100                  # SO-100 HIL tests
                                # (UR / Franka / G1 HIL are planned)
```

## Docs

```bash
just docs                       # mkdocs serve at :8000
just docs-build                 # mkdocs build --strict (CI parity)
just schema-export              # regenerate JSON Schema (CI compares)
```

## CLI (`openral`)

`just quickstart` automatically installs a `~/.local/bin/openral` wrapper so you can run `openral` (or `openral <cmd>`) from any terminal without `just`. To install or re-install it independently (e.g. after moving the repo):

```bash
just install-cli
```

The wrapper sources the ROS 2 distro overlay and the colcon workspace overlay before delegating to `.venv/bin/openral`, so ROS 2 node/topic/action commands work transparently. Pure-Python commands (`openral doctor`, `openral detect`, etc.) still work even if ROS 2 is not yet built.

Bare `openral` (no args) drops into an interactive REPL where subcommands run without the prefix (`sim run --config …`); pass a subcommand for one-shot mode in scripts/CI. See ADR-0021.

```bash
openral doctor                   # diagnose host: Python, OS, ROS 2 distro, GPU, USB
openral detect                   # auto-detect robot + sensors + GPU; write a full robot.yaml
openral connect --robot so100    # open a HAL connection (only so100 wired today)
openral calibrate camera --sensor S  # ros2 camera_calibration helper
openral install sim              # post-install opt-in dep groups (ADR-0021)
openral install ros              # re-run scripts/bootstrap_ubuntu.sh (sudo)
openral install list             # show every known dep group
openral skill install <hub-id>   # download an rSkill from HF Hub (license-gated)
openral skill list               # list installed rSkills
openral skill new <id>           # scaffold a new local rSkill from rskills/template/
openral sensor list              # browse the sensor catalog
openral sensor show <id>         # resolve a catalog entry to a SensorSpec/Bundle
openral benchmark report         # aggregate rskills/<id>/eval/*.json benchmark blocks
openral sim run --config FILE    # run a SimEnvironment YAML end-to-end (ADR-0009)
```

Planned (not yet wired): `openral run`, `openral benchmark run`.

## Tooling self-help

- **`openral` prints `AMENT_TRACE_SETUP_FILES: unbound variable` and exits?** Your `~/.local/bin/openral` predates the fix that sources the (not-`set -u`-safe) ROS 2 overlays with nounset disabled. Re-run `just install-cli` to regenerate it.
- **Python import unclear?** `uv run python -c 'import openral_<pkg>; print(openral_<pkg>.__file__)'`.
- **ROS 2 topic missing?** `ros2 topic list -t` in a `source install/setup.bash`-ed shell.
- **Schema diff?** `just schema-export` and check `git diff python/openral_core/schemas/`.
- **CI flake?** Re-run once. If it flakes again, triage with the `flake` label and don't merge.
- **Hardware test fails on a runner?** Check the runner's e-stop log first. Never push a "fix" that makes a hardware test pass without understanding why it failed.
- **Out of GPU memory?** Lower batch / quantize / use a smaller skill variant — never silently downcast or skip frames.
- **`uv sync` fails with "Unable to uninstall `hf-libero==0.1.3`. distutils-installed distributions..."?** The PyPI sdist for `hf-libero` drops a spurious top-level `*.egg-info` FILE next to the proper `.dist-info/` directory; uv then misclassifies the install as distutils-built. Run **`just sync <flags>`** instead of `uv sync <flags>` — the wrapper pre/post-runs `scripts/repair_hf_libero_install.py` to strip the bogus file + RECORD line. Idempotent and safe on every host.
- **`No module named 'openral_core'` after switching dependency groups?** You ran bare `uv sync --group <x>`. Without `--all-packages` that syncs only the workspace root and **uninstalls every editable `openral-*` member**, so the next REPL / `openral deploy sim` can't import the workspace. Always use **`just sync`**, which forces `--all-packages` for you (unless you scoped to a single `--package`/`-p`) — `just sync --group robocasa` is equivalent to `uv sync --all-packages --group robocasa`. Repair an already-broken venv with `just sync --all-packages`.
