# openral-detect

Hardware probing + `RobotDescription` assembly for `openral detect`.

Part of [**OpenRAL**](https://github.com/OpenRAL/openral) — the open Robot
Abstraction Layer for vision-language-action robotics. This package is one
member of the OpenRAL Python workspace; see the architecture overview and the
eight-layer model in the project docs.

- **Docs:** https://openral.github.io/openral/
- **Source:** https://github.com/OpenRAL/openral
- **License:** Apache-2.0

> All OpenRAL workspace packages move in lockstep at `0.1.x` until the first
> public release.

## Usage

`openral detect` is an always-interactive **custom `robot.yaml` builder**.
There is no `--interactive`/`-i` flag — probing a rig and writing a manifest
always walks the operator through naming the rig and binding its cameras.

```bash
# Build a custom robot.yaml (prompts for a rig name + per-camera sensor
# bindings; writes robots/<name>/robot.yaml by default).
openral detect

# Also scaffold a DeployScene with the workcell (non-robot) camera bindings.
openral detect \
  --output robots/my_so101_bench/robot.yaml \
  --deployment scenes/deploy/so101_bench.yaml

# Inspect probes without writing files or prompting (CI-safe).
openral detect --include usb,gpu,cameras_v4l2 --report detect.json --no-write
```

`openral detect` probes USB, DDS, GPU, V4L2, RealSense, Orbbec, and network
interfaces; resolves a canonical manifest (from USB/DDS inference or
`--robot`) as a **template**, never as the output; and records accelerator
capabilities used by `openral rskill check`. The interactive flow is:

1. Prompt for a custom robot name (default: the canonical rig's name). This
   becomes `RobotDescription.name`, and — unless `--output` is given — sets
   the default output path to `robots/<name>/robot.yaml`, so a `DeployScene`
   with `robot_id: <name>` resolves it.
2. Everything except the sensor list is inherited verbatim from the
   canonical manifest: joints, URDF/MJCF, safety envelope, capabilities,
   compute.
3. A sensor wizard walks every detected camera (V4L2 + RealSense + Orbbec)
   and asks which sensor it is. A thumbnail (opencv) is grabbed only for
   V4L2 (`/dev/video*`) devices; RealSense and Orbbec entries key on model +
   serial (no `device_path`) and get no thumbnail:
   - a canonical sensor name (e.g. `top`, `wrist`) reuses that manifest
     `SensorSpec` verbatim (frame, intrinsics, `vla_feature_key`) plus the
     real device binding, with a warning that the intrinsics are still the
     canonical rig's — best to supply your own calibrated fx/fy/cx/cy;
   - a new name creates a new robot sensor (prompts for `parent_frame`;
     generic intrinsics + the same calibration warning);
   - `w:<name>` records a **workcell/workspace** camera — written only into
     the `--deployment` `DeployScene`, never into the robot manifest;
   - Enter skips the device. Canonical sensors that are never bound to a
     device are dropped from the custom manifest.
4. An opt-in gate, "Customize joint limits & safety envelope? [y/N]" — `N`
   (default) inherits the canonical values verbatim; `y` walks each joint's
   position/velocity/effort limits and the safety scalars with
   Enter-to-keep-default prompts.
5. When the manifest is written outside the canonical rig's directory,
   `file:` asset refs (URDF/MJCF/SRDF) are rewritten repo-root-relative so
   they still resolve from the new location.
6. For a serial (lerobot Feetech) arm — SO-100/SO-101/… — the scaffolded
   `--deployment` scene needs a lerobot calibration (per-servo IDs + homing
   offsets) before `deploy run` can connect. When the scene's calibration
   directory is empty, `openral detect` prints a "Calibration required"
   notice pointing at the lerobot calibrate flow
   (<https://huggingface.co/docs/lerobot/v0.6.0/en/so101#calibrate>); link an
   existing `<id>.json` into that directory or generate one there.

Only `--no-write` short-circuits to probe-only inspection: no prompts,
cameras are auto-enriched from the sensor catalog (real intrinsics via
reverse lookup) instead of wizard-bound, and no `robot.yaml`/`DeployScene`
is written. This is the CI-safe path. `--report <path>` is an orthogonal
flag that always dumps the raw `DetectionReport` JSON to `path`; used
*alone* (without `--no-write`) it does **not** make the run non-interactive
— the full builder still prompts for a rig name, runs the camera wizard,
and writes `robot.yaml`. For a headless/CI probe, pass `--no-write`
(optionally together with `--report`).

Bare Feetech USB detection defaults to `so101_follower` because SO-100 and
SO-101 are electrically indistinguishable on the bus. Force the older arm with:

```bash
openral detect --robot so100
```

`--deployment` writes a `DeployScene` shell: `robot_id` set to the custom
name, the wizard's workcell camera bindings, and the HAL's `port` overridden
with the serial device actually detected on this host (e.g. `/dev/ttyACM0`).
The custom `robots/<name>/robot.yaml` this produces **is** read by
`openral deploy run` (via `robot_id`/`--robot`) — it is not a throwaway
inspection artifact. `--deployment` does not choose an rSkill; `deploy run`
lets the reasoner pick from installed, capability-matched rSkills.
