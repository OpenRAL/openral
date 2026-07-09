# Run a deployment on a robot and open the dashboard

`openral deploy run` is the real-hardware sibling of `openral sim run`. It boots
the full production ROS graph — the HAL lifecycle node, the C++ safety kernel,
the reasoner, world state (plus SLAM/Nav2 when the robot declares a lidar) — and
ticks an rSkill against your **real** robot, driven by a `DeployScene`
YAML. This tutorial writes a deploy scene, dry-runs it
against a digital twin, then runs it on hardware with the live dashboard.

## Prerequisites

```bash
just bootstrap && just sync   # always `just sync`, never bare `uv sync` —
                              # see docs/contributing/toolchain.md
openral install ros           # the ROS 2 graph deploy run launches
openral doctor                # confirm ROS 2, GPU, and USB are visible
```

You need a `RobotDescription` for your robot under
[`robots/<robot_id>/robot.yaml`](https://github.com/OpenRAL/openral/blob/master/robots/)
(the in-tree manifests cover SO-100/101, Franka, UR5e/10e, ALOHA, OpenArm,
Rizon 4, H1, G1, panda_mobile), and an installed rSkill (see
[Write an rSkill](../rskill/write-and-publish-an-rskill.md)).

## 1. Write a `DeployScene` config

Deploy configs live in
[`scenes/deploy/`](https://github.com/OpenRAL/openral/blob/master/scenes/deploy/).
A `DeployScene` pins the workcell: scene id, `robot_id`, optional sim
composition, safety tightening, and additive allowed collision pairs. Robot
facts such as serial ports, IPs, sensors, poses, rates, and limits live in
`robots/<robot_id>/robot.yaml`.

### Option A — create/update the robot manifest with `openral detect`

If the robot is plugged in, let detection write or refresh the robot manifest:

```bash
# A bare detect resolves a plugged-in Feetech arm to so101_follower by default.
openral detect \
    --output robots/so101_follower/robot.yaml \
    --deployment scenes/deploy/so101_bench.yaml
```

Detection records robot-owned facts in `robot.yaml`; it does not create a
deploy scene unless `--deployment` is passed. The wizard always runs (there is
no `--interactive` flag) and opens the camera binding wizard so robot cameras
and workcell cameras land in that deploy scene.
Use `--include usb,gpu,cameras_v4l2,cameras_realsense` to limit probes, and
`--report detect.json --no-write` when you only want the raw detection report.
The rSkill that drives the robot is **not** set in deploy config — the reasoner
selects it at runtime from the installed `rskills/` registry.

> The SO-101 is electrically identical to the SO-100 over USB (same Feetech
> controller), so the bus alone can't distinguish them — the current SO-101 is
> the default. To target the older SO-100 instead, add `--robot so100` (the flag
> accepts a short slug like `so100` or a manifest directory name like
> `so100_follower`).

### Option B — write the workcell by hand

Create `scenes/deploy/so100_pick_cube.yaml`:

```yaml
scene:
  id: so100_pick_cube
  backend: mujoco
robot_id: so100_follower          # matches robots/so100_follower/robot.yaml
safety:
  workspace_box_min_xyz: [-0.3, -0.3, 0.0]
  workspace_box_max_xyz: [0.3, 0.3, 0.5]
```

The full schema is `openral_core.schemas.DeployScene`. `safety` is optional and
must tighten the robot manifest's `SafetyEnvelope`.

List what's available:

```bash
openral deploy list      # walks scenes/deploy/*.yaml
```

## 2. Dry-run against a digital twin first

Before touching hardware, validate the whole graph against a simulated HAL
with `openral deploy sim`. It boots the **same** graph (dashboard + safety
kernel + reasoner + prompt router + runtime + HAL) but against a digital-twin
HAL driven by the same `DeployScene` YAML — no robot required:

```bash
openral deploy sim \
  --config scenes/deploy/so100_pick_cube.yaml
```

`deploy sim` takes no `--rskill` — the reasoner picks the active rSkill from the
in-tree `rskills/` palette at `on_configure`, embodiment-filtered.

This is the safe place to shake out manifest, sensor, and rSkill-compatibility
errors.

### RoboCasa scenes — let the HAL provision the backend

RoboCasa kitchen scenes (e.g. `scenes/deploy/robocasa_navigate.yaml`) need the
RoboCasa fork, which is **not** installed by `just sync --group robocasa` — that
group only supplies robosuite + supporting deps. The fork is git-cloned and
installed editable **at runtime** by the deploy-sim HAL's `on_configure` via
`openral_sim._deps.ensure_backend_deps('robocasa_kitchen')`. Auto-install is on
by default; run it like so:

```bash
just sync --group robocasa    # robosuite + deps (swaps out the libero/sim group)
OPENRAL_AUTO_INSTALL_DEPS=1 openral deploy sim \
  --config scenes/deploy/robocasa_navigate.yaml
```

Do **not** hand-install `robocasa` / `robosuite` — that pulls the wrong
robosuite and wrecks the managed env. To avoid the first-run build stalling the
lifecycle transition, pre-build the clone once beforehand:

```bash
OPENRAL_AUTO_INSTALL_DEPS=1 python -c \
  "from openral_sim._deps import ensure_backend_deps; ensure_backend_deps('robocasa_kitchen')"
```

LIBERO and RoboCasa pin conflicting robosuite versions and cannot coexist, so
swap groups per task: `just sync --group robocasa` for kitchens, `just sync
--group sim` (or `--group libero`) to go back. Full details in
[Managing the Python environment & dependency
groups](../../contributing/toolchain.md#managing-the-python-environment-dependency-groups).

## 3. Run on hardware

With the robot powered, connected, and within a clear workspace:

```bash
openral deploy run --config scenes/deploy/so100_pick_cube.yaml
```

What happens:

- The robot is resolved from `--config`; `build_hal(mode="real")` constructs
  the real HAL. If no hardware is attached, `connect()` **fails loudly**; a
  simulation-only robot raises `ROSCapabilityMismatch` (use `deploy sim`).
- Robot HAL defaults (`port` / `robot_ip` / `fci_ip` / adapter params) come from
  `robots/<robot_id>/robot.yaml`. Override at the CLI with repeatable `--hal key=value`:

  ```bash
  openral deploy run --config scenes/deploy/so100_pick_cube.yaml --hal port=/dev/ttyUSB1
  ```

- The C++ safety kernel sits between the policy and the motors: Python
  proposes, C++ disposes, and `ROSSafetyViolation` is never silently caught.
  Keep your E-stop within reach.

### SO-101 SmolVLA TensorRT fast path (OpenRAL Pro)

For the public SO-101 pen-pick skill, the real deploy scene and rSkill are:

```bash
openral rskill install OpenRAL/rskill-smolvla-so101-pick-place-pen
openral deploy run \
  --config scenes/deploy/so101_bench.yaml
```

The `OPENRAL_SMOLVLA_TRT=1` split ONNX/TensorRT fast path (and the GStreamer
NVMM zero-copy camera leg it pairs with) is an **OpenRAL Pro plugin** — it ships in the
private `openral-pro-trt` package, not this repo. With `openral-pro-trt`
installed, `OPENRAL_SMOLVLA_TRT=1` before `openral deploy run` attaches the
same way it always did (the env var is read by the pro-side hook, looked up
by name via `openral_rskill.backend_registry.maybe_attach_pro_hooks`); see
`openral-pro`'s own docs for the engine pre-build recipe.

Without `openral-pro-trt` installed, the policy runs in eager PyTorch —
logged, not a silent skip. For the SO-101 NVMM camera path, either install
the OpenRAL Pro plugin or disable NVMM for the relevant cameras in the
deploy scene.

### Optional reward monitor

`deploy run` can bring up the same reward/progress monitor as `deploy sim`:

```bash
openral deploy run \
  --config scenes/deploy/so101_bench.yaml \
  --enable-reward-monitor
```

The monitor is advisory only; it serves `/openral/perception/query_task_progress`
for the reasoner and never gates motors.

## 4. Open the dashboard

`deploy run` spawns the live dashboard by default (`--dashboard/--no-dashboard`,
port `--dashboard-port`, default **4318**). It's a read-only pane over the OTel
stream — the most recent `rskill.execute`, `skill.chunk_inference`, and
`safety.check` spans, rolling metric histograms, per-camera thumbnails, and an
event log. Operator discovery / write endpoints still exist for explicit tooling
flows, but they are kept off the main dashboard surface.

```
http://localhost:4318
```

The connection indicator in the top right turns green within a few hundred
milliseconds. To run without it (e.g. headless CI), pass `--no-dashboard`; to
move the port, `--dashboard-port 4400`.

You can also launch the dashboard standalone and point other workloads at it:

```bash
openral dashboard            # binds 127.0.0.1:4318
```

then export `OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318` and
`OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf` for the workload (see the
[dashboard quickstart](../../quickstart/dashboard.md) for the in-process demo
mode).

## See also

- [`scenes/README.md`](https://github.com/OpenRAL/openral/blob/master/scenes/README.md) — DeployScene / SimScene / BenchmarkScene tiers.
- [`openral dashboard` quickstart](../../quickstart/dashboard.md).
- `openral detect` — auto-generate `robot.yaml` by probing USB devices and sensors.
