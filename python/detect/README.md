# openral-detect

Hardware probing + `RobotDescription` assembly for `openral detect`.

Part of [**OpenRAL**](https://github.com/OpenRAL/openral) — the open Robot
Abstraction Layer for vision-language-action robotics. This package is one
member of the OpenRAL Python workspace; see the architecture overview and the
seven-layer model in the project docs.

- **Docs:** https://openral.github.io/openral/
- **Source:** https://github.com/OpenRAL/openral
- **License:** Apache-2.0

> All OpenRAL workspace packages move in lockstep with the workspace SemVer
> version; the bump is computed from Conventional Commits by release-please.

## Usage

`openral detect` is an always-interactive **custom `robot.yaml` builder**
(there is no `--interactive`/`-i` flag): writing a manifest always walks the
operator through naming the rig and binding its cameras.

```bash
# Build a custom robot.yaml (prompts for a rig name + per-camera sensor
# bindings; writes robots/<name>/robot.yaml by default).
openral detect

# Also scaffold a DeployScene with the workcell (non-robot) camera bindings.
openral detect \
  --output robots/my_so101_bench/robot.yaml \
  --deployment scenes/deploy/so101_bench.yaml

# Inspect probes without writing files or prompting (CI-safe).
openral detect --include usb,can,gpu,cameras_v4l2 --report detect.json --no-write
```

`openral detect` probes USB, SocketCAN, DDS, GPU, V4L2, RealSense, and
network interfaces (the report has an Orbbec slot, but no probe fills it yet); resolves a canonical manifest (from DDS/CAN/USB inference
or `--robot`) as a **template**, never as the output; and records accelerator
capabilities used by `openral rskill check`.

### How a robot is identified

Three transports, consulted strongest-evidence-first:

| order | transport | signal | example |
| ----- | --------- | ------ | ------- |
| 1 | DDS | topic-name prefix | `/lowstate` → Unitree G1 |
| 2 | SocketCAN | udev-pinned interface name | `openarm_left` → OpenArm |
| 3 | USB serial | VID/PID of the controller chip | `1a86:7523` → SO-101 |

DDS outranks CAN because a robot publishing those topics is already up and
naming itself. CAN outranks USB because an interface name is a deliberate
provisioning choice, while one USB controller chip is often shared across arms
(every Feetech-bus arm looks alike).

CAN needs the udev rule because a CAN bus carries no vendor descriptor: the
adapter identifies itself, the motors behind it do not. A bare `can0` is left
unmatched on purpose; it is a kernel enumeration artefact, not a statement
about what is plugged in.

```bash
# What `openral detect` sees on a provisioned OpenArm cell:
#   can            6 interface(s), 2 up, 1 matched
#     openarm_left   openarm · PEAK PCAN-USB Pro FD · 1000k/5000k FD · ERROR-ACTIVE
#     openarm_right  openarm · PEAK PCAN-USB Pro FD · 1000k/5000k FD · ERROR-ACTIVE
openral detect --include can --no-write
```

`ERROR-PASSIVE` on an otherwise healthy bus means frames leave the adapter and
nothing acknowledges them, in practice unpowered motors. The probe warns about
it rather than leaving you to read the `state` field.

### Cameras without `v4l-utils`

The V4L2 probe reads `/sys/class/video4linux` first and only falls back to
`v4l2-ctl --list-devices`. sysfs is always present, needs no package and no
root, and is the only source of the camera's *own* USB descriptor, which
resolves a `usb_uvc` catalog signature so a recognised camera arrives with real
intrinsics instead of a generic placeholder. It also carries the serial, the
only way to tell two identical cameras apart on one rig.

### Interactive flow

1. Prompt for a custom robot name (default: the canonical rig's name). This
   becomes `RobotDescription.name`, and — unless `--output` is given — sets
   the default output path to `robots/<name>/robot.yaml`, so a `DeployScene`
   with `robot_id: <name>` resolves it.
2. Everything except the sensor list is inherited verbatim from the
   canonical manifest: joints, URDF/MJCF, safety envelope, capabilities,
   compute.
3. A sensor wizard walks every detected camera (V4L2 + RealSense)
   and asks which sensor it is. A thumbnail (opencv) is grabbed only for
   V4L2 (`/dev/video*`) devices; RealSense entries key on model +
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

Only `--no-write` short-circuits to probe-only inspection (the CI-safe path):
no prompts, cameras auto-enriched from the sensor catalog (real intrinsics via
reverse lookup) instead of wizard-bound, and no `robot.yaml`/`DeployScene`
written. `--report <path>` is orthogonal: it always dumps the raw
`DetectionReport` JSON to `path`, but alone it does **not** make the run
non-interactive. For a headless probe, pass `--no-write` (optionally with
`--report`).

Bare Feetech USB detection defaults to `so101_follower` because SO-100 and
SO-101 are electrically indistinguishable on the bus. Force the older arm with:

```bash
openral detect --robot so100
```

`--deployment` writes a `DeployScene` shell: `robot_id` set to the custom
name, the wizard's workcell camera bindings, and the HAL's `port` overridden
with the serial device actually detected on this host (e.g. `/dev/ttyACM0`).
`openral deploy run` reads the resulting `robots/<name>/robot.yaml` (via
`robot_id`/`--robot`); it is not a throwaway artifact. `--deployment` does not
choose an rSkill; `deploy run` lets the reasoner pick from installed,
capability-matched rSkills.
