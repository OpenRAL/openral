# Sensor Catalog & Schema Gaps

The first-party sensor catalog in `python/sensors/` (what ships, how to use it,
how to add an entry) and the modalities `SensorSpec` cannot yet express.
Numeric fields in catalog entries are *nominal* data-sheet values; for a real
deployment replace them with calibrated ones via `openral calibrate`.

Normative types: `SensorSpec`, `SensorBundle`, `SensorModality` and
`IntrinsicsPinhole` in `python/core/src/openral_core/schemas.py`. The reference
implementation pattern is `python/sensors/src/openral_sensors/realsense.py`.

**Shipping rule.** An entry ships only when a robot manifest, HAL adapter or
scene uses it, or a deployed rig carries it. Candidates not yet shipped
because nothing uses them: other RealSense SKUs, Orbbec depth cameras,
Logitech C922 / Brio, 2D and 3D LiDARs, standalone IMUs, ATI / Bota / OnRobot
force-torque sensors, GelSight DIGIT, XELA and AnySkin tactile skins.
Reintroduce one when a manifest or HAL factory needs it.

---

## Schema-compliance audit

The existing `SensorSpec` fields are sufficient unless flagged.

| Modality | `SensorModality` enum | Sufficient `SensorSpec` fields? | Notes |
|---|---|---|---|
| RGB | `RGB` | ✅ `intrinsics`, `encoding`, `fov_*` | — |
| Depth | `DEPTH` | ✅ `intrinsics`, `range_*`, `fov_*` | — |
| Stereo (passive) | `STEREO` | ⚠️ baseline missing | put `baseline_m` in `metadata` until the stereo-baseline gap below is resolved |
| RGBD bundle | RGB + DEPTH (+ IMU) | ✅ via `SensorBundle` | sync = `hardware` for RealSense / Orbbec |
| 2D LiDAR | `LIDAR_2D` | ✅ `range_*`, `fov_h_deg`, `rate_hz` | — |
| 3D LiDAR | `POINT_CLOUD` + `n_channels` | ✅ for rotating; metadata for non-repetitive solid-state | document Livox `non_repetitive=True` convention |
| IMU (6-DoF) | `IMU` | ✅ `accel_noise_density`, `gyro_noise_density` | — |
| IMU (9-DoF) | `IMU` | ⚠️ no magnetometer noise field | put `mag_noise_density` in `metadata` until the magnetometer gap below is resolved |
| GNSS-INS | `GPS` + `IMU` `SensorBundle` | ✅ | bundle the two |
| 6-axis F/T | `FORCE_TORQUE` + `n_axes=6` | ✅ | — |
| Vision tactile | `TACTILE_VISION` | ✅ | optionally fill `intrinsics` for the embedded camera |
| Array tactile | `TACTILE_ARRAY` + `tactile_grid` | ✅ | put magnetic / capacitive / barometric tech in `metadata` |
| Audio | `AUDIO` | ✅ | — |
| GPS | `GPS` | ✅ | — |
| Battery | `BATTERY` | ✅ | — |

---

## Schema gaps (proposals)

None of these blocks a shipped entry. Each is a **proposal** for a future ADR
(CLAUDE.md §1.6); today `SensorModality` has no `EVENT`, `THERMAL` or
`BAROMETER` member and `SensorSpec` has no `baseline_m`. Until then, sensors using these
technologies should encode the type via `metadata["physical_modality"]` and
choose the closest existing modality.

| Gap | Proposed enum addition | Why it matters | Closest current fit |
|---|---|---|---|
| **E. Event / neuromorphic vision** | `SensorModality.EVENT` | Prophesee EVK4, iniVation DAVIS — shipping in research drones and humanoids; very different data semantics (event streams, not frames) | misuse `RGB` today |
| **T. Thermal / LWIR** | `SensorModality.THERMAL` | Spot CAM+ thermal payload, FLIR Boson; common on inspection and SAR | misuse `IR` today |
| **U. Ultrasonic / radar / proximity** | `SensorModality.RANGE_1D` (covers ultrasonic + ToF range + radar) | Drone altitude, mobile-robot bumpers, humanoid foot proximity | misuse `DEPTH` today |
| **B. Barometer / altimeter** | `SensorModality.BAROMETER` | Drones, multi-floor mobile robots | metadata-only today |
| **M. Magnetometer (standalone)** | `SensorModality.MAGNETOMETER` (or extend IMU spec with magnetometer noise field) | 9-DoF IMUs that publish magnetometer separately, indoor magnetic localization | inside `IMU` metadata today |
| **C. Contact / binary touch** | `SensorModality.CONTACT` | Foot contact switches on humanoids/quadrupeds, gripper-close detection | misuse `FORCE_TORQUE` (n_axes=1) or `TACTILE_ARRAY` (1×1) today |
| **S. Stereo baseline field** | extend `SensorSpec` with `baseline_m: float \| None` | ZED, OAK-D, MultiSense — baseline is a first-class extrinsic, not metadata | inside `metadata` today |
| **W. Wheel encoder / odometry** | (no enum — keep as `nav_msgs/Odometry` topic) | Standard ROS 2 convention; bind via `WorldState.base_twist` | covered by `WorldState` |

Recommendation: bundle E + T + U + B + M + C + S into one additive ADR that
extends `SensorSpec` in place (manifests are at `schema_version: "0.2"`), with
a test loading a fixture manifest.

---

## Catalog status

Each entry is addressable by a stable id (`<vendor>/<model>`) through the
`SensorCatalog` registry and the `openral sensor list / show` CLI.

Robot-mounted physical sensors in `robots/<id>/robot.yaml` should set
`SensorSpec.catalog_id` when they match a catalog entry. The catalog id is
provenance, not lazy resolution: the manifest still owns calibrated intrinsics,
frames, serial/MXID, VLA feature keys, and sim placement. Scene-only cameras may
remain inline without catalog provenance because many are benchmark viewpoints
rather than deployable hardware.

### What shipped (9 entries, 6 modules)

| Module | Catalog ids | Modalities | Wired in |
|---|---|---|---|
| `realsense.py` | `intel/realsense_d435`, `intel/realsense_d435i`, `intel/realsense_d415` | RGB+DEPTH(+IMU) bundles | `ur10e_with_sensors`, `franka_panda_with_sensors`, `ur5e_with_sensors` |
| `usb_uvc.py` | `generic/usb_uvc_rgb`, `logitech/c920` | RGB | `robots/so100_follower`, `robots/so101_follower` (`catalog_id`), `so100_with_sensors` |
| `luxonis.py` | `luxonis/oak_d_pro` | RGB+DEPTH+IMU bundle | available catalog entry; SO-101's world-fixed front camera may use this hardware model but is not robot-mounted provenance |
| `stereolabs.py` | `stereolabs/zed_mini` | stereo RGB + DEPTH + IMU bundle | `robots/openarm`, `robots/openamrobot_v2` (`catalog_id`) |
| `arducam.py` | `arducam/b0495` | RGB (global shutter) | available catalog entry; no manifest references it |
| `force_torque.py` | `robotiq/ft_300s` | FORCE_TORQUE (n_axes=6) | `ur5e_with_sensors`, `ur10e_with_sensors` |

### How to use

```python
from openral_sensors import CATALOG

# List
print(CATALOG.list_ids())                       # 9 ids
print([e.id for e in CATALOG.filter(vendor="intel")])

# Build a SensorBundle / SensorSpec for a robot manifest
bundle = CATALOG.build("intel/realsense_d435i", name="head", parent_frame="head_link")
robot.sensor_bundles.append(bundle)
```

```bash
openral sensor list                                  # tabular catalog
openral sensor list --vendor intel
openral sensor list --modality rgb --json
openral sensor show intel/realsense_d435i --name head --parent-frame head_link
```

### Not yet shipped (gated on the schema ADR above)

| Module | Factories | Blocked-by |
|---|---|---|
| `stereolabs.py` | `zed_2i_bundle`, `zed_x_bundle` (`zed_mini_bundle` shipped) | gap S (stereo `baseline_m`) |
| `luxonis.py` | `oak_d_lite_bundle` (`oak_d_pro_bundle` shipped) | gap S |
| `event_camera.py` | `prophesee_evk4_spec`, `inivation_davis346_spec` | gap E |
| `thermal.py` | `flir_boson_spec`, `spot_thermal_payload_spec` | gap T |
| `proximity.py` | `maxbotix_ultrasonic_spec`, `teraranger_evo_spec` | gap U |
| `barometer.py` | `bmp388_spec` | gap B |

---

## Style for new modules

Every new module follows the conventions used by `realsense.py`,
`usb_uvc.py`, `force_torque.py`:

1. Module docstring with a runnable doctest example.
2. Nominal data-sheet constants at module top, behind comments citing the
   data-sheet revision (`realsense.py`).
3. One `<model>_bundle()` factory per multi-modal sensor returning a
   `SensorBundle(sync="hardware" | "approximate", sync_tolerance_ms=…)`.
4. One `<model>_spec()` factory per single-modality sensor returning a
   `SensorSpec`.
5. **`SensorCatalog` registration** at the bottom of the module — every
   public factory is registered with `replace=True` so re-imports during
   tests are idempotent.
6. Optional `bundle_to_node_params()` and `generate_launch_py()` for ROS 2
   wiring (mirrors `realsense2_camera`'s composable-node pattern, RealSense
   only today).
7. Optional `calibrate_*_cmd()` for calibration helpers (chessboard for RGB,
   data-sheet preset for IMU, etc.).
8. Unit tests added to `tests/unit/test_sensor_catalog.py` covering: factory
   shape, modality composition, parent-frame propagation, encoding, range,
   metadata, **catalog round-trip via `model_dump_json` / `model_validate_json`**.
9. Re-export from `python/sensors/src/openral_sensors/__init__.py` (the
   side-effect import is what populates the catalog).

---

## Next steps

1. Open one ADR for the schema additions above (event, thermal, range_1d,
   barometer, magnetometer, contact, stereo baseline) as a single future schema
   bump with one migration entry and regenerated `docs/reference/schemas/`.
2. Then ship the remaining factories (ZED 2i / ZED X, OAK-D Lite,
   `event_camera.py`, `thermal.py`, `proximity.py`, `barometer.py`) as small
   PRs with the test shape from the style section.
