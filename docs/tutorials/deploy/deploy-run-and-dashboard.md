# Run a deployment on a robot and open the dashboard

`openral deploy run` is the real-hardware sibling of `openral sim run`. It boots
the full production ROS graph — the HAL lifecycle node, the C++ safety kernel,
the reasoner, world state (plus SLAM/Nav2 when the robot declares a lidar) — and
ticks an rSkill against your **real** robot, driven by a `RobotEnvironment`
YAML (ADR-0031/0032). This tutorial writes a deployment config, dry-runs it
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

## 1. Write a `RobotEnvironment` config

Deployment configs live in
[`deployments/`](https://github.com/OpenRAL/openral/blob/master/deployments/README.md).
A `RobotEnvironment` pins one deployment: `(robot × HAL × sensors × task ×
safety)` — the rSkill is reasoner-selected at runtime, not pinned. They are
intentionally **not** shipped in the open-core tree —
they encode a specific lab's robot IP / FCI port / camera serial — so you add
your own.

### Option A — scaffold it from `openral detect`

If the robot is plugged in, let detection write most of the config for you.
Detection knows the robot identity, the serial `port`, and every camera; it
*cannot* know the task or the policy, so those land as `TODO` placeholders:

```bash
# A bare detect resolves a plugged-in Feetech arm to so101_follower by default
# and emits the matching deploy config with --deployment:
openral detect \
    --output robots/so101_follower/robot.yaml \
    --deployment deployments/so101_pick_cube.yaml
```

This pre-fills `robot_id`, `hal.transport.port` (the detected `/dev/tty*`),
and one `sensors[]` reader per detected camera. `safety` is left `null` so the
robot's own envelope from `robots/so101_follower/robot.yaml` applies. Open the
file and replace the `TODO` `task` field (what to do), then continue to step 2.
The rSkill that drives the robot is **not** set here — the reasoner selects it
at runtime from the installed `rskills/` registry, filtered to the robot's
embodiment (the same path `openral deploy sim` uses).

> The SO-101 is electrically identical to the SO-100 over USB (same Feetech
> controller), so the bus alone can't distinguish them — the current SO-101 is
> the default. To target the older SO-100 instead, add `--robot so100` (the flag
> accepts a short slug like `so100` or a manifest directory name like
> `so100_follower`).

### Option B — write it by hand

Create `deployments/so100_pick_cube.yaml`:

```yaml
robot_id: so100_follower          # matches robots/so100_follower/robot.yaml
hal:
  adapter: so100_follower
  transport:
    port: /dev/ttyUSB0            # serial port (robot_ip / fci_ip for others)
sensors:
  - sensor_id: wrist_rgb
task:
  id: pick_cube/red
  scene_id: pick_cube/red
  instruction: "pick up the red cube"   # becomes the VLA's language prompt
rate_hz: 30.0
```

The driving rSkill is intentionally absent — at runtime the reasoner picks it
from the installed `rskills/` registry, filtered to the robot's embodiment, so
the config never pins a policy. The loader enforces that every
`sensors[].sensor_id` is unique. The full schema is
[`openral_core.schemas.RobotEnvironment`](https://github.com/OpenRAL/openral/blob/master/python/core/src/openral_core/schemas.py).
`safety` is optional and falls back to the robot manifest's `SafetyEnvelope`.

List what's available:

```bash
openral deploy list      # walks deployments/*.yaml; prints <none> if empty
```

## 2. Dry-run against a digital twin first

Before touching hardware, validate the whole graph against a simulated HAL
with `openral deploy sim`. It boots the **same** graph (dashboard + safety
kernel + reasoner + prompt router + runtime + HAL) but against a digital-twin
HAL driven by a `SceneEnvironment` YAML — no robot required:

```bash
openral deploy sim \
  --config scenes/benchmark/libero_spatial.yaml
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
openral deploy run --config deployments/so100_pick_cube.yaml
```

What happens:

- The robot is resolved from `--config`; `build_hal(mode="real")` constructs
  the real HAL. If no hardware is attached, `connect()` **fails loudly**; a
  simulation-only robot raises `ROSCapabilityMismatch` (use `deploy sim`).
- The robot's `hal.transport` (`port` / `robot_ip` / `fci_ip`) and `hal.params`
  are forwarded as HAL node params. Override at the CLI with repeatable
  `--hal key=value`:

  ```bash
  openral deploy run --config deployments/so100_pick_cube.yaml --hal port=/dev/ttyUSB1
  ```

- The C++ safety kernel sits between the policy and the motors: Python
  proposes, C++ disposes, and `ROSSafetyViolation` is never silently caught.
  Keep your E-stop within reach.

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

- [`deployments/README.md`](https://github.com/OpenRAL/openral/blob/master/deployments/README.md) — RobotEnvironment configs and the sim/real split.
- [`openral dashboard` quickstart](../../quickstart/dashboard.md).
- `openral detect` — auto-generate `robot.yaml` by probing USB devices and sensors.
- [ADR-0031 / ADR-0032](https://github.com/OpenRAL/openral/blob/master/docs/adr/) — the deploy graph design.
