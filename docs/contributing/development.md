# Development setup

Getting a working OpenRAL development environment from scratch — local Ubuntu, dev container, or GitHub Codespace — plus the day-to-day commands.

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
cd openral
just bootstrap          # installs uv, ROS 2, system deps (~5–10 min)
source /opt/ros/jazzy/setup.bash   # or 'humble' on Ubuntu 22.04
just sync               # install Python workspace deps (always `just sync`,
                        # never bare `uv sync` — see toolchain.md)
```

The bootstrap script auto-detects Ubuntu 22.04 (→ ROS 2 Humble) or 24.04 (→ ROS 2 Jazzy).

### 2. Verify the install

```bash
just test            # full unit suite, <30 s
just lint            # ruff + mypy --strict; expect no errors
uv run openral doctor     # diagnose host environment
```

Test inventory (file + LOC counts, gaps, follow-ups):
[`tests/README.md`](https://github.com/OpenRAL/openral/blob/master/tests/README.md).

`uv run openral doctor` is the canonical environment probe — sample output
table in the [README's "Quick start" section](https://github.com/OpenRAL/openral/blob/master/README.md#quick-start).
Each row reads `ok` / `info` / `absent` / `missing` depending on which deps
are installed.

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
# inside container (always `just sync`, never bare `uv sync`):
just sync
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
just test-ros-live          # the live-ROS pytest suite (see below)
```

`just ros2-build` first runs `scripts/check_ros_build_deps.sh`, which derives
the required ament packages from the in-tree `package.xml` files and reports
any that are not installed under `/opt/ros/$ROS_DISTRO/share`, with a ready
`apt-get install` line. Without it a single missing apt package (say
`ros-jazzy-octomap-msgs`) surfaces ~45 s in as a CMake `find_package` error
that aborts every remaining package, burying the cause. The check is
advisory — set `OPENRAL_ROS_DEPS_STRICT=1` to make missing deps fatal — and
no-ops when ROS 2 is not sourced.

#### The live-ROS suite (`OPENRAL_TEST_ROS_LIVE`)

A set of integration tests — the reasoner node's dispatch, VRAM and async-LLM
paths, the Tier-C critic producer, and the HAL sim-sensor bridge's
mobile-base `/tf` guard — needs more than an importable `rclpy`:
a real DDS graph and the colcon-built `openral_msgs` / `openral_reasoner_ros` /
`openral_prompt_router` overlay. They are gated behind `OPENRAL_TEST_ROS_LIVE=1`
so the ordinary `uv run pytest` run (where rclpy + DDS init can clash with a
glib pulled in transitively by torch/pyarrow) does not trip over them.

`scripts/ros_live_tests.sh` holds the target list and sets the variable. Both
callers exec it, so they cannot select different tests:

| Caller | Environment |
| --- | --- |
| `just test-ros-live` | your host, after `just ros2-build && source install/setup.bash` |
| `docker-build` workflow, "Live ROS tests" step | inside `openral:x86`, which bakes ROS 2 Jazzy + the colcon overlay |

No GPU is required — the free-VRAM readings are pinned at the `nvidia-smi`
process boundary. Pass extra pytest flags through the recipe, e.g.
`just test-ros-live -k dispatch_watchdog`.

Adding a live-ROS test means adding its file to `TARGETS` in that script — the
one place that decides what runs — and to the `paths:` filter in
`.github/workflows/docker-build.yml`, so a diff touching only that test still
triggers the build that runs it. `tests/unit/test_ros_live_targets.py` fails the
unit suite if a gated file under `tests/integration/` is missing from `TARGETS`.

The `docker-build` workflow is the only CI surface with a real ROS 2 install;
the `test-selective` runner has none, so anything not listed there silently
`importorskip`s in CI — and a live test parked under `tests/unit/` runs on no
lane at all, which is why they live in `tests/integration/`.

That workflow runs on **merge to `master`**, not on pull requests (a full image
build is ~15-20 min of runner time). Run it yourself with `just test-ros-live`
while developing, or dispatch the workflow against your branch
(`gh workflow run docker-build.yml --ref <branch>`) before merging a change to
these tests.

One live test — `tests/unit/test_gstreamer_perception_tee.py`'s end-to-end
publish — is deliberately excluded: it also needs PyGObject, which the open
deploy image does not ship (the GStreamer media stack is OpenRAL Pro). `just
test` runs it on a host that has the `gstreamer` extra installed.

#### The RoboCasa sim suite (manual — there is no CI lane, and there cannot be)

Seven tests under `tests/sim/` are gated on `importorskip("robocasa")` and need
three things at once: the colcon overlay (they spawn the real
`safety_kernel_node`), MuJoCo, and a provisioned RoboCasa kitchen backend. They
are the geometry-and-kernel evidence for the layout pins, the certified
distance instrument, the support-contact witness, the depth synth, and the
HAL's camera / body-twist paths on a real kitchen.

`scripts/robocasa_sim_tests.sh` holds the target list, and `just
test-robocasa-sim` is its only caller:

```bash
source /opt/ros/jazzy/setup.bash && just ros2-build \
  && source install/setup.bash && just test-robocasa-sim
```

A fresh venv may lack `robocasa` even after `just sync`. Provision it with:

```bash
OPENRAL_AUTO_INSTALL_DEPS=1 uv run --no-sync python -c \
  "from openral_sim._deps import ensure_backend_deps; ensure_backend_deps('robocasa_kitchen')"
```

Run the suite before merging anything that touches the kernel, the
`panda_mobile` manifest, the HAL sim bridge, or a `scenes/deploy/robocasa_*.yaml`
pin. Narrow it with `-k` on a busy machine — a kitchen compose is memory-hungry
and the whole suite in one pytest process has OOM-killed a 15 GB host. Extra
args are appended to the pytest invocation, so naming a path *adds* it to the
target list rather than selecting it: use `just test-robocasa-sim -k
geom_distance`.

**Why there is no CI lane.** Two independent reasons, both measured.

*Disk.* RoboCasa's assets are 23 GB — `objects/aigen_objs` 13 GB,
`objects/objaverse` 6.2 GB, `objects/lightwheel` 1.5 GB, `fixtures` 1.4 GB,
`generative_textures` 1.2 GB, `textures` 521 MB — downloaded per-bundle from
utexas.box.com with no sub-bundle granularity.

The only hosted CI surface with a colcon overlay is the `docker-build` image,
already 25.2 GB, building on a runner that has to `rm -rf` dotnet, android and
CodeQL to reclaim its ~14 GB. Even a trimmed set (fixtures + textures + one
object bundle, ~8 GB) is past that runner's whole disk. No GPU is needed, so
the constraint is disk alone.

*Security.* A self-hosted runner was built, registered, and then removed: it is
not a safe option for this repository. **A runner label is a routing request
made by a workflow, not an access control enforced by the runner.** A repository-scoped
self-hosted runner accepts jobs from *any* workflow in that repository naming
its labels, and for a `pull_request` event GitHub executes the workflow
definition from the **fork's** ref.

`OpenRAL/openral` is public and has three fork-reachable `pull_request`
workflows (`dco.yml`, `quality.yml`, `test-selective.yml`), so a fork PR can
edit one to `runs-on: [self-hosted, <label>]` with arbitrary `run:` steps and
execute code on the runner host — with that user's SSH keys, `gh` credentials
and LAN access to the lab robots. Restricting a runner to selected workflows
requires an organisation runner group, which this org's GitHub Free plan does
not offer, so no native control makes the label mean anything.

Note what does *not* help: giving the protected workflow safe triggers. The
exposed asset is the runner, not the workflow. Reasoning about the triggers of
the file you are adding is the mistake that makes this look safe.

**What holds the suite together instead.** `tests/unit/test_robocasa_sim_targets.py`
fails the unit suite if a RoboCasa-gated file is missing from `TARGETS`. With no
CI lane, that list is the only definition of what the suite is, and that guard
is the only thing that notices when a test drifts out of it.

The rest of `tests/sim/` is manual by declared policy
(`.github/workflows/test-selective.yml`).

### Sim evals (closed-loop, opt-in — needs HF weights ± GPU)

```bash
just sim-eval scenes/<name>.yaml   # canonical entry point
just sim-libero                              # SmolVLA × LIBERO
just sim-xvla-libero                         # xVLA × LIBERO
just sim-pi05-libero                         # π0.5 × LIBERO (≥8 GB VRAM)
just sim-metaworld --task metaworld/reach-v3 # SmolVLA × MetaWorld MT50
just sim-act-aloha                           # ACT × gym-aloha bimanual cube
just sim-diffusion-pusht                     # Diffusion Policy × gym-pusht (CPU)
```

### Docs

```bash
just docs                   # serve at http://localhost:8000 (live-reload)
just docs-build             # full build (CI parity, strict mode)
```

---

## Lab HIL runners (`[self-hosted, lab-<robot>]`)

**Status: documented, deliberately NOT registered.** No self-hosted runner is
registered for this repository and none should be until the blocker below
clears. The `[self-hosted, lab-<robot>]` labels in `tests/hil/` docstrings
describe where a gate *would* run, not a lane that exists. Today HIL runs by
hand on the cell, which is why every HIL file gates on an env var plus a live
hardware probe and skips — never fails — off-rig.

### The blocker, first

Read *Security* under "Why there is no CI lane" above before doing any of this.
The short version: **a runner label is a routing request made by a workflow,
not an access control enforced by the runner.** A repository-scoped runner
accepts jobs from any workflow naming its labels, and for `pull_request` GitHub
runs the workflow definition from the **fork's** ref. `OpenRAL/openral` is
public with three fork-reachable `pull_request` workflows (`dco.yml`,
`quality.yml`, `test-selective.yml`), so a fork PR can add
`runs-on: [self-hosted, lab-openarm]` with arbitrary `run:` steps and get code
execution on the lab host — with that user's SSH keys, `gh` credentials, LAN
access to the robots, and, for `lab-openarm`, the CAN buses of a powered
bimanual arm.

Registering a runner on a lab host is therefore gated on **one** of:

1. an **organisation runner group** restricted to selected repositories and
   selected workflows (the selected-workflows restriction needs GitHub
   Enterprise Cloud or Enterprise Server; Free and Team can create runner
   groups but cannot pin them to workflows), or
2. moving HIL to a **private** mirror repository with no fork-reachable
   `pull_request` triggers, dispatched only by `workflow_dispatch` /
   `repository_dispatch`.

Neither is in place. Until one is, the procedure below is reference material.

### Host prerequisites (OpenArm cell)

A `lab-openarm` host must already satisfy what the gates probe, or every job
is a green no-op that proves nothing:

| Requirement | Check |
| --- | --- |
| Both motor buses up, CAN FD 1 Mbit/5 Mbit | `ip -details link show openarm_left` (and `openarm_right`) — `state ERROR-ACTIVE`, `mtu 72` |
| udev naming pinned by `dev_id`, not `canN` order | `openral detect` reports `bh_robot_type: openarm` |
| `openarm_bringup` on the ament prefix path | `ros2 pkg prefix openarm_bringup` |
| ZED SDK + `zed_wrapper` for the world-map assertions (the scene's `drivers:` block launches it, so its overlay must be sourced before `deploy run`) | `source ~/zed_ws/install/setup.bash && ros2 pkg prefix zed_wrapper` |
| OpenRAL ROS overlay built | `just ros2-build` |

### Registering the runner

```bash
# On the lab host, as the account that owns the cell (NOT root).
mkdir -p ~/actions-runner && cd ~/actions-runner
curl -fsSLo runner.tar.gz \
    https://github.com/actions/runner/releases/download/v2.330.0/actions-runner-linux-arm64-2.330.0.tar.gz
tar xzf runner.tar.gz

# ORGANISATION registration, never repository: the URL is the org, the token
# comes from the org's Settings -> Actions -> Runners -> New runner, and
# --runnergroup names the restricted group from the gate above. A runner
# registered against the repository URL lands in the default group, which
# every fork-reachable workflow can request.
./config.sh --url https://github.com/OpenRAL \
    --token "$ORG_RUNNER_TOKEN" \
    --runnergroup lab-openarm \
    --labels self-hosted,lab-openarm \
    --name qorin1-openarm --work _work --unattended

# Before connecting the cell: confirm the runner sits in `lab-openarm` and that
# the group is restricted to selected repositories AND selected workflows
# (Organisation -> Settings -> Actions -> Runner groups). If either
# restriction is missing, remove the runner (`./config.sh remove`) first.
sudo ./svc.sh install "$USER" && sudo ./svc.sh start
```

Use `linux-arm64` on a Jetson; `linux-x64` elsewhere. Do **not** run the
service as root: a HIL job that can write CAN as root can also disable the
udev rules that keep left and right from swapping.

### Workflow shape

A HIL lane must be dispatch-only — never `pull_request`, never `push`:

```yaml
on:
  workflow_dispatch:
    inputs:
      robot: { description: "lab-<robot> label suffix", required: true }

jobs:
  hil:
    runs-on: [self-hosted, "lab-${{ inputs.robot }}"]
    timeout-minutes: 15
    steps:
      - uses: actions/checkout@v4
      - run: just sync
      - run: just hil-openarm-deploy
```

`timeout-minutes` is not optional: a wedged job holds the cell, and the HIL
tier's own budget is 10 minutes (CLAUDE.md §2).

### What a `lab-openarm` lane may NOT do unattended

`just hil-openarm-deploy` is non-motion and attaches to a graph someone else
started. It must stay that way. **Starting the OpenArm deploy graph is itself a
motion event** — `openarm_bringup`'s `OpenArmHW::on_activate` calls
`enable_all()` then `return_to_zero()`, an unramped MIT position command to 0.0
on all seven joints per side. A CI lane must never launch it, and
`tests/hil/test_openarm_slot_group_motion.py` (the one gate that commands the
arm) must keep its `OPENRAL_OPENARM_ATTENDED=1` gate unexported on any runner
host.

## Repository layout (quick reference)

For the maintained directory layout — every `python/*` / `packages/*`
package, all 21 `robots/`, all 48 `rskills/`, and the test tiers — see
[`docs/architecture/repo-map.md`](../architecture/repo-map.md).

Most directories carry a per-package `README.md` with usage examples and
links back to the canonical schemas — start with the package matching
the layer you're touching.

Full architecture is in [docs/architecture/overview.md](../architecture/overview.md). For a per-module status canvas (working / in-dev / planned / out-of-scope, with inputs, outputs, and schemas), open [docs/architecture/repo-state-map.html](../architecture/repo-state-map.html) — keep it in sync per CLAUDE.md §4.3.

---

## Registering a `lab-so101` HIL runner

The HIL tier (`tests/hil/`) is gated by `[self-hosted, lab-<robot>]` runner
labels. The SO-101 bench is the first of these with a physical rig behind it:
`.github/workflows/hil-so101.yml` is committed and runs
`openral deploy validate` plus the non-motion serial gate
(`tests/hil/test_so101_serial_live.py`) on manual dispatch (Actions tab →
"Run workflow") — there is no schedule trigger, since the bench arm is not
permanently attached to a runner.

It is **inert** until a runner carrying both labels exists; with no such
runner a dispatched run just queues and expires. Nothing else in CI depends
on it.

Registering the runner is an org/repo **settings** action and is deliberately
not automated from this repo. On the host with the arm attached:

1. **Settings → Actions → Runners → New self-hosted runner** (repo or org
   scope), follow the download/`config.sh` steps GitHub prints, and when it
   asks for labels enter `lab-so101` (the `self-hosted` label is added for
   you). The workflow matches on **both**.
2. Give the runner's user access to the bus: `sudo usermod -aG dialout <user>`,
   then re-login. Without it the arm's `/dev/ttyACM*` is unopenable and every
   test in the file skips, green and useless.
3. Install `just` and `uv` for that user — the workflow calls `uv` directly and
   the local recipes assume `just` is on `PATH`.
4. Confirm the bench matches the committed scene before trusting the lane:

    ```bash
    ls -l /dev/v4l/by-id /dev/v4l/by-path        # camera symlinks
    uv run openral deploy validate --config scenes/deploy/so101_bench.yaml
    just hil-so101
    ```

    Device paths in `scenes/deploy/so101_bench.yaml` are host-specific.
    `deploy validate` reports a stale one as a warning rather than discovering
    it mid-run; fixing the scene is the fix.
5. Since this lane is manual-dispatch only and the bench is not an always-on
   host, the runner only needs to be listening at dispatch time: run
   `./run.sh` in a terminal after connecting the arm, trigger the workflow
   from the Actions tab, then `Ctrl-C` it when done — no need to install it
   as a background service unless you want it always available.

!!! warning "Keep this lane non-motion — and still keep a hand near the power switch"
    Everything the SO-101 lane runs today is read-only on the servos — the
    port opens, the pre-flight ping and `read_state()` run, nothing is
    commanded via `send_action` or `reset_to_pose`. That is *not* the same as
    "nobody needs to be near the arm": `connect()` itself cycles servo torque
    off and back on, and re-enabling torque makes a servo hold its last
    `Goal_Position` — a stale one from a previous session can make the arm
    twitch toward it right there, before any command is sent. Whoever
    dispatches this workflow should be within reach of the power switch. A
    motion test belongs behind an explicit attended-bench gate (see
    `tests/hil/test_openarm_slot_group_motion.py`, which requires two separate
    opt-in env vars), never on unattended dispatch.

---

## Making a pull request

1. Create a branch: `git switch -c feat/your-feature`.
2. Make your changes. Run `just lint && just test` before pushing.
3. If you changed a Pydantic schema, run `just schema-export` and commit the updated JSON files.
4. Open a PR. The title should follow [Conventional Commits](https://www.conventionalcommits.org/) — e.g. `feat(core): add FooSchema`.
5. If your PR comes from a fork, the first push waits for a maintainer to
   click "Approve and run" on the workflow run before any check starts —
   that's a GitHub setting (`all_external_contributors`), not something you
   can skip.
6. Every push runs two fast required checks: `quality` (lint/type/schema/docs,
   <3 min) and `select-and-test` (the pytest targets your diff actually
   touches — see [Selective testing](selective-testing.md), usually a few
   minutes). A third required check, `heavy-lanes`, runs the opt-in
   dependency lanes (LIBERO, RoboCasa, GR00T, …) your diff selected — it
   starts by itself once the PR is ready, see [Review policy](#review-policy).
7. All required checks must be green, and the PR needs one approving review
   from a code owner, before it can merge. See `.github/workflows/` for what
   each workflow runs.

The full PR checklist is in the repo-root `CLAUDE.md` (not linked from docs — open it directly in your editor).

### Review policy

The branch ruleset on `master` requires one approving review from a
[`CODEOWNERS`](https://github.com/OpenRAL/openral/blob/master/.github/CODEOWNERS)
reviewer, with stale approvals dismissed on every push. GitHub does not allow
a PR author to approve their own PR, so with today's single code owner the
maintainer's own PRs merge via the ruleset's `OrganizationAdmin` bypass
instead of a satisfying review — expected, not a workaround; a second code
owner would remove the need for it.

`heavy-lanes` (the opt-in dependency lanes: LIBERO, RoboCasa, GR00T,
ManiSkill3, …) starts **automatically**, with no approval or label, once the
PR's head commit is:

- up to date with `master` (0 commits behind — rebase or merge `master` in);
- green on every other check;
- free of unresolved review threads;
- not a draft, and from a branch of this repo.

[`heavy-lanes-trigger`](https://github.com/OpenRAL/openral/blob/master/.github/workflows/heavy-lanes-trigger.yml)
re-checks each open PR whenever a PR workflow finishes and every 10 minutes
(GitHub has no event for "thread resolved" or "`master` moved", so those two
can take up to ~10 min to be noticed). It starts the lanes once per head
commit; a new push waits for the conditions again. Until then the check shows
"Expected — waiting" and blocks the merge. A failed lane is re-run from its
Actions run page.

**Fork PRs:** `workflow_dispatch` cannot target a fork branch, so nothing can
post `heavy-lanes` on a fork head and the check stays "Expected — waiting". A
maintainer runs the lanes by pushing the fork's head commit to a branch of this
repository (`git push origin <sha>:refs/heads/fork/<pr>`) and dispatching
**heavy-lanes** on that branch — the check lands on the same commit SHA, so the
PR picks it up — or merges through the ruleset bypass.

This keeps the expensive lanes — 19 parallel jobs — for PRs that are otherwise
mergeable, instead of paying for them on every push.
`select-and-test` — test-speed feedback on the code a diff actually
touches (`quality` is the lint/type/schema/docs check) — never waits on this.

---

## Troubleshooting

### `uv sync` fails with "package not found" (or "Unable to uninstall hf-libero")

Run **`just sync`** instead of any bare `uv sync`. The workspace has multiple
member packages (the root manifest does not list them as direct dependencies),
so the sync needs `--all-packages` — which `just sync` supplies — and the
`just sync` wrapper also runs `scripts/repair_hf_libero_install.py` before/after
to avoid the `hf-libero==0.1.3` distutils-uninstall abort. Add opt-in groups
with `just sync --group <name>`. See
[Managing the Python environment & dependency
groups](toolchain.md#managing-the-python-environment-dependency-groups).

### `mypy` reports "missing library stubs or py.typed"

Both `openral_core` and `openral_cli` ship `py.typed` markers. If mypy still complains, make sure you ran `just sync` and that your interpreter is the workspace venv (`which python` should point into `.venv/`).

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
just install-hooks
```

Then `git commit` runs ruff, ruff-format, mypy, codespell, and the
conventional-commit check on changed files automatically, plus DCO auto-sign-off.

Use `just install-hooks` rather than `pre-commit install`: the repo pins
`core.hooksPath=.githooks` (for DCO sign-off), and `pre-commit install` refuses
to run while `core.hooksPath` is set. `just install-hooks` instead wires the
committed `.githooks/` wrappers (which call `pre-commit run`) and pre-builds the
hook environments.
