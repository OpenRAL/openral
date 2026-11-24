# Run a deployment on a robot and open the dashboard

**Goal:** write a `DeployScene`, dry-run it against a digital twin, run it on
real hardware, give the robot a task and watch it in the dashboard.

`openral deploy run` is the real-hardware sibling of `openral sim run`. It boots
the production ROS graph (HAL lifecycle node, C++ safety kernel, reasoner, world
state, plus SLAM/Nav2 when the robot declares a lidar) and ticks an rSkill
against your robot, driven by a `DeployScene` YAML.

## Prerequisites

```bash
just bootstrap && just sync   # always `just sync`; see docs/contributing/toolchain.md
uv run openral install ros    # the ROS 2 graph deploy run launches
uv run openral doctor         # confirm ROS 2, GPU, USB and the reasoner
```

**A robot manifest.** Each robot has a `RobotDescription` at
[`robots/<robot_id>/robot.yaml`](https://github.com/OpenRAL/openral/blob/master/robots/).
In-tree manifests, by `embodiment_kind`:

| Class | `robot_id` |
| --- | --- |
| Single arm | `so100_follower`, `so101_follower`, `franka_panda`, `droid_franka_robotiq`, `ur5e`, `ur10e`, `rizon4`, `rizon_dual`, `sawyer`, `widowx`, `galaxea_a1`, `makermods_metal` |
| Bimanual | `aloha_bimanual`, `aloha_agilex`, `openarm`, `anvil_openarm_v2`, `makermods_metal_station` |
| Mobile manipulator | `panda_mobile`, `panda_mobile_vslam`, `google_robot`, `r1pro`, `openamrobot_v2` |
| Humanoid | `g1`, `h1`, `h2`, `gr1` |

`pusht_2d` is omitted: it is a sim-only scene pseudo-robot.

**Real hardware is a subset.** `deploy run` resolves `hal.real` from the
manifest, and 13 robots declare one: `so100_follower`, `so101_follower`,
`franka_panda`, `ur5e`, `ur10e`, `sawyer`, `galaxea_a1`, `aloha_bimanual`,
`openarm`, `g1`, `h2`, `openamrobot_v2` and `rizon_dual`. For the rest,
`build_hal` raises `ROSCapabilityMismatch` under `hal_mode:=real`; use
`deploy sim`. `openarm` (Enactic) has a real HAL; `anvil_openarm_v2` (Anvil) is
sim-only. Both MakerMods Metal manifests are sim-only because the arm has no
brakes and a real HAL waits on a reviewed stop behaviour.

**A reasoner model.** The S2 reasoner has no hidden default; the graph does not
plan without one:

```bash
export OPENRAL_REASONER_MODEL=<model-id>        # a curated id; see the reasoner reference
export OPENRAL_REASONER_API_KEY=<key>           # only where the endpoint needs one
```

The model list, endpoints and what `doctor`'s `Reasoner LLM` row checks are in
the [reasoner reference](../../reference/reasoner.md#reasoner-model-selection).
For an on-device model with no key, see the
[Cosmos 3 Edge reasoner](../../reference/cosmos3-edge-reasoner.md); it needs the
whole GPU, and `doctor` cannot tell whether the sidecar fits.

**An installed rSkill** ([Write an rSkill](../rskill/write-and-publish-an-rskill.md)).
By default `deploy run` also installs the rSkills on this robot's openral.com
list that are missing (permissive licences only). Opt out with
`--no-rskill-install` or `OPENRAL_RSKILL_AUTO_INSTALL=0`.

**The graph stays on this host.** `deploy run` confines the ROS graph to the
deploy host (`ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST`) on a per-robot
`ROS_DOMAIN_ID` and prints the result as `ros scope:`. A graph that must reach
another computer needs `OPENRAL_ALLOW_NETWORK_ROS=1`, SROS2 in Enforce mode
(`uv run openral security init <robot_id>`) and the same domain on every host;
otherwise the deploy refuses. The Foxglove bridge and dashboard are unaffected.
See [ROS graph security](../../reference/ros-graph-security.md).

## 1. Write a `DeployScene`

Deploy configs live in
[`scenes/deploy/`](https://github.com/OpenRAL/openral/blob/master/scenes/deploy/).
A `DeployScene` pins the workcell: scene id, `robot_id`, optional sim
composition, safety tightening and extra allowed collision pairs. Robot facts
(serial ports, IPs, sensors, poses, rates, limits) live in the robot manifest.
The rSkill is not set in the scene; the reasoner selects it at runtime from the
installed registry.

### Option A: detect the robot

With the robot plugged in, `detect` writes or refreshes its manifest:

```bash
uv run openral detect \
    --output robots/so101_follower/robot.yaml \
    --deployment scenes/deploy/so101_bench.yaml
```

Detection records robot facts in `robot.yaml` and creates a deploy scene only
with `--deployment`. The wizard always runs (there is no `--interactive` flag),
including camera binding: a robot camera's binding lands in `robot.yaml`, a
workcell camera in the scene. Use `--include usb,gpu,cameras_v4l2,cameras_realsense`
to limit probes, and `--report detect.json --no-write` for the raw report only.

The SO-101 and SO-100 are the same over USB, so a bare detect picks SO-101. For
an SO-100 add `--robot so100` (a slug, or a manifest directory name such as
`so100_follower`).

A robot type with several units or hosts keeps each host's camera bindings and
calibration in `robots/<id>/units/<unit>.yaml`; the scene's `robot_unit` or
`OPENRAL_ROBOT_UNIT` selects it, and a real deploy refuses until one is selected.

A real deploy also needs the unit's serial number so its telemetry and
compliance records name the machine: set `OPENRAL_UNIT_SERIAL` (or `serial:` in
the unit file), and optionally `OPENRAL_ORG_ID` / `OPENRAL_SITE_ID`. `deploy run`
refuses to start without one; `deploy sim` does not need it. Every record is also
kept in the [on-robot compliance log](../../reference/compliance-log.md)
(`~/.local/state/openral/compliance`, or `OPENRAL_COMPLIANCE_DIR`); check it with
`uv run openral compliance verify`. Set `OPENRAL_ACTOR_KEY` (a secret you keep)
and reset e-stops with `openral estop reset --operator <name>` so each reset
records who did it, pseudonymously. The robot stops if the log can no longer be
written.

### Option B: write it by hand

Create `scenes/deploy/so100_pick_cube.yaml` (the file the later commands use):

```yaml
scene:
  id: so100_pick_cube
  backend: mujoco
robot_id: so100_follower          # matches robots/so100_follower/robot.yaml
safety:
  workspace_box_min_xyz: [-0.3, -0.3, 0.0]
  workspace_box_max_xyz: [0.3, 0.3, 0.5]
```

The schema is `openral_core.schemas.DeployScene`. `safety` is optional and must
tighten the robot manifest's `SafetyEnvelope`. List the available scenes with
`uv run openral deploy list`.

Cameras that exist only as ROS topics (a ZED), world-map settings and the
Foxglove collision view are in
[Cameras, world map and the collision view](deploy-perception-and-world-map.md).

## 2. Dry-run against a digital twin

`openral deploy sim` boots the same graph (dashboard, safety kernel, reasoner,
prompt router, runtime, HAL) against a digital-twin HAL driven by the same
scene, with no robot. Use it to shake out manifest, sensor and rSkill errors:

```bash
uv run openral deploy sim --config scenes/deploy/so100_pick_cube.yaml
```

`deploy sim` takes no `--rskill`; the reasoner picks from the in-tree `rskills/`
palette, filtered by embodiment. To change a scene's runtime field without
editing it, pass `--runtime key=value` (repeatable), for example
`--runtime grounding_wait_s=8`.

Two cheaper checks first:

```bash
uv run openral deploy sim --config scenes/deploy/so100_pick_cube.yaml --dry-run
uv run openral deploy validate --config scenes/deploy/so100_pick_cube.yaml
```

`--dry-run` resolves the scene, HAL params and launch argv and prints them
without starting `ros2 launch`. `deploy validate` is the readiness check for
**real** hardware, with no launch and no robot. It resolves HAL params with the
same precedence as `deploy run` (`--hal` > scene `hal` > `robot.yaml`) and
separates **ERROR** (committed data is missing; exits non-zero) from **WARN**
(the device is not plugged in right now):

- **HAL transport**: a serial `port` is declared and the device exists now.
- **Calibration**: a serial HAL with `calibrate_on_connect: false` has an `id`
  and `calibration_dir`, and `<calibration_dir>/<id>.json` exists. Otherwise
  every `send_action` fails with "has no calibration registered".
- **Camera bindings**: each deploy sensor has a `deploy_binding` (without one it
  is never published and a camera VLA silently gets an empty observation), and
  any `/dev/*` path exists.

### RoboCasa scenes

RoboCasa kitchen scenes (for example `scenes/deploy/robocasa_navigate.yaml`) need
the RoboCasa fork, which `just sync --group robocasa` does not install (that
group supplies only robosuite and supporting deps). The deploy-sim HAL clones and
installs it editable at runtime through
`openral_sim._deps.ensure_backend_deps('robocasa_kitchen')`:

```bash
just sync --group robocasa    # swaps out the libero/sim group
OPENRAL_AUTO_INSTALL_DEPS=1 uv run openral deploy sim --config scenes/deploy/robocasa_navigate.yaml
```

Do not hand-install `robocasa` or `robosuite`; that pulls the wrong robosuite.
LIBERO and RoboCasa pin conflicting robosuite versions, so swap groups per task
(`just sync --group sim` or `--group libero` to go back). See
[Managing the Python environment](../../contributing/toolchain.md#managing-the-python-environment-dependency-groups).

## 3. Run on hardware

With the robot powered, connected and the workspace clear, keep your E-stop
within reach. The C++ safety kernel sits between the policy and the motors
(Python proposes, C++ disposes), and `ROSSafetyViolation` is never silently
caught.

```bash
uv run openral deploy run --config scenes/deploy/so100_pick_cube.yaml
```

You should see the `ros scope:` line, then the graph coming up and the dashboard
URL. If no hardware is attached, `connect()` fails loudly.

- HAL defaults (`port`, `robot_ip`, `fci_ip`, adapter params) come from
  `robots/<robot_id>/robot.yaml`. Override with repeatable `--hal key=value`, for
  example `--hal port=/dev/ttyUSB1`.
- `--rskill-install/--no-rskill-install` controls the auto-install described in
  the prerequisites.
- `--dry-run` prints the resolved launch argv without starting the graph.
- `--dataset-out <file.mcap>` records what the policy saw at each tick.

**Object tracking is sim-only.** `runtime.object_tracking` is `off` (default),
`tracking` or `on`. Any value other than `off` is refused on a real HAL, so a real
deploy runs it `off`; the former `shadow` and `bind` now load as `tracking` and
are refused too. `tracking` makes the object tracker ground the goal's queries
and own the held payload's `attachment_state`; `on` also needs a Pro safety
exemption provider. The tracker needs `grounding_wait_s > 0`. Try it on
`deploy sim`:

```bash
uv run openral deploy sim --config <scene> \
  --runtime object_tracking=tracking --runtime grounding_wait_s=8
```

An external USD [object catalog](../../reference/usd-object-catalog.md) needs the
tracker too, so it is sim-only for the same reason.

### SO-101 SmolVLA fast path (OpenRAL Pro)

For the public SO-101 SmolVLA skill:

```bash
uv run openral rskill install OpenRAL/rskill-smolvla-so101-eraser_place-bf16
uv run openral deploy run --config scenes/deploy/so101_bench.yaml
```

The `OPENRAL_SMOLVLA_TRT=1` TensorRT path and its NVMM zero-copy camera leg are
OpenRAL Pro. Without Pro the policy runs in eager PyTorch (logged, not a silent
skip); for the NVMM camera path, install Pro or disable NVMM for those cameras in
the scene.

## 4. Give the robot a task

The reasoner picks which rSkill to run, but idles without a goal. Three ways to
give it one.

**At launch:**

```bash
uv run openral deploy run --config scenes/deploy/so101_bench.yaml \
  --initial-task "pick the bowl and place it on the plate, then push the mug back"
```

A multi-step goal is decomposed into an ordered mission queue by the reasoner's
`decompose_mission` tool; the queue advances only when the active task passes the
reward gate.

**While it runs**, from another terminal (needs a sourced ROS 2 install):

```bash
uv run openral prompt --robot so101_follower "put the pen back in the cup"
```

This publishes one `PromptStamped` on `/openral/prompt_in/cli`; the prompt router
fans it out to `/openral/prompt`. `--robot` is the deploy's robot id: it derives
the same per-robot `ROS_DOMAIN_ID` as the deploy (printed first as `ros scope:`),
so nothing needs exporting. It acts only if that robot's HAL node is on the graph
it joined, and exits 3 otherwise, so a typo cannot reach another robot. An
exported `ROS_DOMAIN_ID` still wins over `--robot`.

A prompt arriving mid-mission is conversational context (an answer or a hint) and
does not rebuild the queue. To replace the mission, add `--new-goal`:

```bash
uv run openral prompt --robot so101_follower --new-goal "stop that, clear the table instead"
```

On a cold-booted graph the router may not have discovered the publisher yet;
`--discovery-wait-s 15` covers that (default 5 s).

**From the dashboard**, with its `prompt` action, which posts to the same path.

**After an e-stop**, clear it from the CLI:

```bash
uv run openral estop reset --robot so101_follower
```

Like the dashboard's reset button, it calls the kernel's `/openral/estop_reset`
first and broadcasts `/openral/estop_cleared` only if the kernel accepted, which
un-latches the HAL and the rSkill runner. A bare `ros2 service call
/openral/estop_reset ...` clears the kernel alone and the runner keeps rejecting
goals. Exit code 1 means the kernel refused the reset (often its cooldown).

**Optional reward monitor.** `--enable-reward-monitor` brings up the same
progress monitor as `deploy sim`. It is advisory: it serves
`/openral/perception/query_task_progress` for the reasoner and never gates motors.

## 5. Open the dashboard

`deploy run` starts the robot dashboard by default (`--dashboard/--no-dashboard`,
`--dashboard-port`, default **4318**) with a read-only Foxglove bridge behind its
3D view (`--no-foxglove` to skip). It serves the panels and actions this robot's
manifest declares: cameras, joints, safety state, metrics, events, E-STOP,
reset, prompt, and the installed rSkills.

```text
http://localhost:4318
```

On the deploy host you have full control; other LAN hosts (with `--host`) get a
read-only view plus E-STOP. The rSkill controls appear when
`OPENRAL_DASHBOARD_WRITE_CONTROLS=1`, or when the robot is linked to openral.com
(the account grants them).

To reach the robot from openral.com, add it under **Dashboard → Robots → Add
robot** and run the command it shows, `uv run openral robot link <code>`: no
install, no root, and the tunnel survives reboots. See the
[dashboard quickstart](../../quickstart/dashboard.md). `OPENRAL_DASHBOARD_UI_DIR`
points the dashboard at a local build of the robot UI instead of the released one.

For headless runs use `--no-dashboard`. To run the dashboard on its own, start
`uv run openral dashboard` (binds `127.0.0.1:4318`), then export
`OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318` and
`OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf` for the workload.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `ROSCapabilityMismatch` from `build_hal` | The robot has no `hal.real`; use `deploy sim`, or pick one of the 13 real robots. |
| `deploy run` refuses to start | A real deploy needs the unit serial (`OPENRAL_UNIT_SERIAL`) and, for multi-unit robots, a selected unit (`OPENRAL_ROBOT_UNIT`). Run `deploy validate` first. |
| Refused with a ROS scope message | The graph must stay on this host; see "The graph stays on this host". |
| `runtime.object_tracking` ... sim-only | Set it to `off` for `deploy run`. |
| Robot idles after launch | No goal yet; pass `--initial-task` or use `openral prompt`. |
| `openral prompt` exits 3 | That robot's HAL node is not on the graph the command joined; check `--robot` and the `ros scope:` line. |
| Goals rejected after an e-stop | Clear with `openral estop reset --robot <id>`, not a bare service call. |
| Reasoner does not plan | `OPENRAL_REASONER_MODEL` is unset; see the reasoner reference. |

## Next

- [Cameras, world map and the collision view](deploy-perception-and-world-map.md)
- [Attended real-world voxel check (OpenArm)](openarm-real-world-voxel-check.md)
- [Dashboard quickstart](../../quickstart/dashboard.md)
- [Reasoner (S2) reference](../../reference/reasoner.md)
- [Your first sim rollout](../sim/first-rollout.md), the no-hardware path
- [`scenes/README.md`](https://github.com/OpenRAL/openral/blob/master/scenes/README.md): DeployScene / SimScene / BenchmarkScene tiers
