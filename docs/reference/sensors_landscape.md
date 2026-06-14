# Robot Sensor Landscape & OpenRAL Catalog Roadmap

This document is the canonical reference for **which sensors are common on
production and research robot platforms in 2024–2026**, and **which of them
should ship as first-party `SensorBundle` factories under `python/sensors/`**.

It is the input for sequencing work in `python/sensors/` and for any future
ADR that proposes extending the `SensorModality` enum in
`openral_core.schemas`.

Method: a survey of vendor data sheets, model cards from major humanoid /
quadruped / manipulator vendors (Tesla, Figure, Unitree, Boston Dynamics,
Agility, ANYbotics, Franka, UR, Kuka, LeRobot SO-100/SO-101, Aloha), and the
LeRobot, ROS 2, and PX4 driver ecosystems. Numeric fields below come from
manufacturer data sheets and are *nominal* — for any deployment they must be
replaced with calibrated values via `openral calibrate`.

See also:
- `python/core/src/openral_core/schemas.py` — normative `SensorSpec`,
  `SensorBundle`, `SensorModality`, `IntrinsicsPinhole`.
- `python/sensors/src/openral_sensors/realsense.py` — reference
  implementation pattern (RealSense D435 / D455 today).
- `CLAUDE.md` §6.1 (layer discipline), §7.5 (working with sensors).

---

## 1. Snapshot — sensor stacks of representative 2024–2026 robots

| Robot | Class | RGB | Depth/RGBD | LiDAR | IMU | F/T | Tactile | Other |
|---|---|---|---|---|---|---|---|---|
| LeRobot SO-100 / SO-101 | manipulator | 1× wrist + 1× scene USB UVC (Logitech C920 / Arducam) | — | — | — | — | — | joint encoders (Feetech STS3215) |
| Aloha (bimanual) | manipulator | 4× USB (2 wrist + 2 scene) | — | — | — | — | — | leader-follower position |
| Franka FR3 / Panda | manipulator | wrist RGB(D) (often RealSense D405/D435i) | optional D435i / D405 | — | optional | wrist 6-axis (Bota SensONE / Robotiq FT-300) or joint-torque-derived | optional GelSight / DIGIT | 7× joint torque |
| UR5e / UR10e | manipulator | optional wrist cam | optional | — | optional | wrist 6-axis (Robotiq FT-300, OnRobot HEX-E) | optional | — |
| Stretch 3 (Hello Robot) | mobile manipulator | head pan-tilt RGBD (D435if) | D435if | RPLIDAR A1 | onboard | — | — | wheel odometry |
| Boston Dynamics Spot | quadruped | 5× mono fisheye | 5× depth | optional Velodyne / Ouster on payload | onboard | leg position/force | — | Spot CAM+ adds PTZ + thermal |
| Unitree Go2 | quadruped | HD wide-angle | optional Intel D435i | Unitree L1 (360°×90°) or Livox Mid-360 (Edu+) | onboard | foot-end force | — | — |
| Unitree G1 | humanoid | head depth (Mid-360 variant) + body cams | yes | optional 3D LiDAR | onboard | optional | optional fingertip tactile | joint torque sensors |
| Tesla Optimus Gen 2/3 | humanoid | 8× monocular RGB | — | — | onboard | — | fingertip tactile (gen 3) | — |
| Figure 02 / 03 | humanoid | 6× RGB (incl. palm cams on 03) | — | — | onboard | — | fingertip force on 03 (≈3 g sensitivity) | — |
| Agility Digit | humanoid | torso depth + perception array | yes | top-mounted lidar (variant-dependent) | onboard | foot/wrist F/T | — | — |
| ANYbotics ANYmal | quadruped | RGB + perception | depth | Velodyne / Robosense (variant) | onboard | foot F/T | — | — |
| Skydio 2+/X10 (drone) | drone | 6× RGB navigation | stereo-derived | — | onboard | — | — | GPS, barometer |

This is a snapshot for prioritization, not a normative table — entries vary by
hardware revision and customer-specific payload.

---

## 2. Catalog by modality

Each row maps cleanly to a `SensorSpec` unless flagged in the **Schema notes**
column. "Bundle" means the sensor naturally lives in a multi-sensor
`SensorBundle` (RGB + depth + IMU on a RealSense, etc.).

### 2.1 RGB cameras (`SensorModality.RGB`)

USB / UVC RGB cameras dominate the LeRobot ecosystem and most low-cost research
arms. Industrial vendors (FLIR, Basler) appear on factory-grade platforms.

| Vendor / model | Form factor | Native res @ fps | hFOV | Encoding | Used on | Schema notes |
|---|---|---|---|---|---|---|
| Logitech C920 / C920e / C922 | USB UVC | 1920×1080@30, 1280×720@60 | ~78° (C920) | `mjpeg`/`yuyv`/`rgb8` | LeRobot SO-100/SO-101 default, Aloha | clean fit |
| Logitech Brio 4K | USB UVC | 3840×2160@30 | 65°/78°/90° (selectable) | `mjpeg`/`rgb8` | research scene cam | clean fit |
| Arducam IMX179 / IMX291 USB | USB UVC | 1920×1080@30 | varies by lens | `mjpeg`/`yuyv` | community SO-100/SO-101 wrist | clean fit |
| Raspberry Pi Camera Module 3 | CSI-2 | 4608×2592@15, 1080p@50 | 66°/120° | `rgb8` | embedded research | clean fit |
| FLIR Blackfly S (Spinnaker) | GigE / USB3 | model-dependent | model-dependent | `bayer_rggb8`/`rgb8` | factory cells | clean fit |
| Basler ace 2 | GigE | model-dependent | model-dependent | `bayer_rggb8`/`rgb8` | factory cells | clean fit |
| GoPro Hero (HDMI capture) | action cam | 4K@60 / 1080p@240 | ~120° | `rgb8` | wrist/head research | clean fit |
| Leopard Imaging IMX477 | USB / MIPI | 4056×3040@30 | lens-dependent | `rgb8` | research stacks | clean fit |

### 2.2 Depth & RGBD (`SensorModality.DEPTH`, plus RGB+depth `SensorBundle`)

Active IR stereo (RealSense, Orbbec Gemini) and ToF (Femto, Azure Kinect)
dominate. Stereolabs ZED uses passive stereo (GPU required). Luxonis OAK-D adds
on-board AI. Status reflects vendor activity in 2025–2026.

| Vendor / model | Tech | Range (m) | Native depth res @ fps | hFOV | RGB included | Used on | Schema notes |
|---|---|---|---|---|---|---|---|
| Intel RealSense D435 | active IR stereo | 0.1 – 10 | 640×480@30 (up to 1280×720@90) | 87° | yes | Stretch, research arms — **already in `realsense.py`** | bundle |
| Intel RealSense D435i | D435 + BMI055 IMU | 0.1 – 10 | 640×480@30 | 87° | yes | Unitree Go2 Edu, Stretch — **add factory** | bundle (RGB+depth+IMU) |
| Intel RealSense D455 | wider-baseline IR stereo | 0.6 – 6 | 1280×720@30 | 87° | yes | research humanoids — **already in `realsense.py`** | bundle |
| Intel RealSense D405 | short-range IR stereo | 0.07 – 0.5 | 1280×720@90 | 87°×58° | yes (mono) | Franka wrist, dexterous manipulation — **high priority** | bundle |
| Intel RealSense D415 | rolling-shutter IR stereo | 0.16 – 10 | 1280×720@30 | 65°×40° | yes | legacy research | bundle |
| Intel RealSense L515 | solid-state lidar (EOL) | 0.25 – 9 | 1024×768@30 | 70° | yes | discontinued; legacy | bundle |
| Orbbec Gemini 335 | active IR stereo | 0.17 – 10 | 1280×800@30 | 87°×58° | yes | RealSense replacement — **high priority (active vendor support)** | bundle |
| Orbbec Gemini 336 / 336L | active IR stereo (long-range) | up to 20 | 1280×800@30 | varies | yes | mobile robots | bundle |
| Orbbec Femto Bolt | ToF (Sony chip; ex-Azure-Kinect) | 0.5 – 5.46 NFOV | 1024×1024 NFOV | 75°×65° NFOV / 120°×120° WFOV | yes (4K) | research humanoids; Azure Kinect successor | bundle |
| Orbbec Femto Mega | ToF | 0.5 – 5.46 | 1024×1024 | similar | yes | networked research | bundle |
| Microsoft Azure Kinect DK | ToF (EOL) | 0.5 – 3.86 NFOV | 1024×1024 | 75°×65° NFOV | yes (4K) | discontinued; legacy datasets | bundle |
| Stereolabs ZED 2 / 2i | passive stereo (GPU req.) | 0.3 – 20 | 2208×1242@15 | 110°×70° | yes (stereo) | outdoor research, drones | bundle; needs `STEREO` mod + `baseline_m` (see §4) |
| Stereolabs ZED Mini | passive stereo | 0.15 – 12 | 1280×720@60 | 90°×60° | yes | XR / wrist research | bundle |
| Stereolabs ZED X / X One | passive stereo (GMSL) | varies | 2560×1440@30 | varies | yes | NVIDIA Jetson + Orin platforms | bundle |
| Luxonis OAK-D / OAK-D Pro | active IR stereo + on-board AI | 0.2 – 4 typical | 1280×720@120 | 71°×56° | yes | research, drones | bundle; AI accelerator metadata |
| Carnegie Robotics MultiSense S21/S30 | passive stereo + spinning lidar | varies | 2048×1088@15 | varies | yes | Spot CAM+, Atlas legacy | bundle |

### 2.3 3D LiDAR (`SensorModality.POINT_CLOUD` with `n_channels`)

3D LiDARs are reported under `POINT_CLOUD` with `n_channels` set to the
vertical-channel / scan-line count (or omitted for non-repetitive solid-state
designs). The schema's `range_min_m` / `range_max_m` and `fov_h_deg` /
`fov_v_deg` cover the rest. No new modality is needed.

| Vendor / model | Tech | Range (m) | hFOV × vFOV | Channels / pts·s⁻¹ | Used on | Schema notes |
|---|---|---|---|---|---|---|
| Livox Mid-360 | hybrid solid-state, non-repetitive | 0.1 – 40 (@10% reflectivity) | 360° × 59° | n.r. / 200k | Unitree Go2 Edu+, AGVs, lawn mowers — **highest priority** | `n_channels = None`; note non-repetitive in metadata |
| Livox Mid-70 | hybrid solid-state | 0.05 – 90 | 70.4° × 4.5° | n.r. / 100k | research | metadata flag |
| Livox Avia | hybrid solid-state | up to 450 | 70° × 77° | n.r. / 240k | UAV mapping | metadata flag |
| Livox HAP | solid-state | up to 200 | 120° × 25° | n.r. | automotive | metadata flag |
| Ouster OS0-32 / OS0-64 / OS0-128 | rotating digital lidar | 0.3 – 50 | 360° × 90° | 32/64/128 / up to 5.2M | Spot payload, ANYmal, B1 | clean fit |
| Ouster OS1-32 / OS1-64 / OS1-128 | rotating digital lidar | 0.3 – 120 | 360° × 45° | 32/64/128 | mid-range mobile | clean fit |
| Ouster OS2-128 | long-range | 1 – 240 | 360° × 22.5° | 128 | autonomous platforms | clean fit |
| Velodyne VLP-16 (Puck) | rotating | 1 – 100 | 360° × 30° | 16 / 300k | legacy autonomous, research | clean fit |
| Velodyne VLP-32C / Alpha Prime | rotating | up to 200 / 245 | 360° × 40° / 40° | 32 / 128 | autonomous | clean fit |
| Hesai PandarXT-32 / XT16 | rotating | up to 120 | 360° × 31° | 32 / 16 | mobile | clean fit |
| Hesai JT16 (2025) | hemispherical + circular | up to 30 | 360° × 75° | 16 | small robotics | clean fit |
| Robosense Helios 16/32 | rotating | up to 150 | 360° × 31° | 16/32 | mobile | clean fit |
| Robosense Bpearl | hemispherical | up to 30 | 360° × 90° | 32 | quadruped, AMRs | clean fit |
| Robosense E1R (2025) | solid-state Flash | up to 30 | 120° × 90° | n/a | outdoor robotics | metadata flag |
| Unitree L1 | rotating-mirror solid-state | 0.05 – 40 | 360° × 90° | n/a | bundled with Go2 / G1 | metadata flag |

### 2.4 2D LiDAR (`SensorModality.LIDAR_2D`)

| Vendor / model | Range (m) | Angular range | Rate (Hz) | Used on | Schema notes |
|---|---|---|---|---|---|
| SLAMTEC RPLIDAR A1 | 12 | 360° | 5.5 | Stretch, hobby AMRs — **high priority (low-cost)** | clean fit |
| SLAMTEC RPLIDAR A2M12 | 16 | 360° | 10 | research AMRs — **high priority** | clean fit |
| SLAMTEC RPLIDAR S1 / S2 / S3 | 40 / 30 / 40 | 360° | 10 / 10 / 15 | indoor AMRs | clean fit |
| Hokuyo URG-04LX-UG01 | 4 | 240° | 10 | research, low-power | clean fit |
| Hokuyo UTM-30LX | 30 | 270° | 40 | outdoor mobile, legacy Atlas head | clean fit |
| Hokuyo UST-10LX / UST-20LX | 10 / 20 | 270° | 40 | industrial AMR — **high priority (industrial)** | clean fit |
| SICK TIM-561 / TIM-781 | 8 / 25 | 270° | 15 / 15 | factory AMR | clean fit |
| SICK LMS1xx / LMS5xx | 20 – 80 | 270° | 25 – 50 | factory AGV | clean fit |

### 2.5 IMU (`SensorModality.IMU`)

`accel_noise_density` and `gyro_noise_density` cover the noise model. A 9-axis
unit (with magnetometer) carries the magnetometer term in `metadata` until a
separate modality is added (see §4 gap M).

| Vendor / model | DoF | Rate | Application | Schema notes |
|---|---|---|---|---|
| Bosch BMI270 | 6 | up to 6.4 kHz | embedded boards (PX4, RPi HATs) — **medium priority** | clean fit |
| Bosch BMI088 | 6 | up to 1.6 kHz | PX4 drones, robotics | clean fit |
| InvenSense ICM-40609 | 6 | varies | embedded inside Livox Mid-360 | metadata only |
| InvenSense ICM-20689 / -20602 | 6 | up to 8 kHz | Pixhawk-class flight controllers | clean fit |
| Xsens MTi-3 / MTi-30 / MTi-100 | 9 | 100 – 400 Hz | research mobile / outdoor | clean fit (mag in metadata) |
| Xsens MTi-630 / MTi-680G | 9 + GNSS | 400 Hz | outdoor humanoids, AMRs | bundle (IMU + GPS) |
| VectorNav VN-100 / VN-110 | 9 | 800 Hz | drones, gimbals | clean fit |
| VectorNav VN-200 / VN-300 | 9 + GNSS | 400 Hz | UAV navigation | bundle (IMU + GPS) |
| Microstrain 3DM-CV5-25 / GX5-25 | 9 | up to 1 kHz | mobile robots | clean fit |
| Microstrain 3DM-GQ7 | 9 + dual-RTK GNSS | 1 kHz | outdoor autonomy | bundle |
| Analog Devices ADIS16470 / 16475 | 6/9 | up to 2 kHz | industrial / mil-spec | clean fit |
| STIM277 / STIM300 | 9 | 1 kHz | high-accuracy industrial | clean fit |

### 2.6 6-axis force/torque (`SensorModality.FORCE_TORQUE`, `n_axes = 6`)

| Vendor / model | Mounting | BW (Hz) | Range | Used on | Schema notes |
|---|---|---|---|---|---|
| ATI Mini40 / Mini45 | wrist (research) | 7000 | low | Franka, UR research | clean fit |
| ATI Gamma / Delta | wrist | 7000 | mid/high | industrial cells | clean fit |
| ATI Nano17 | fingertip | 7000 | very low | dexterous research | clean fit |
| Bota Systems SensONE | wrist (UR/Franka kits) | 800 | mid | UR / Franka — **medium priority** | clean fit |
| Bota Systems Rokubi | wrist | 1000 | mid | research | clean fit |
| Robotiq FT 300-S | wrist (UR-native) | 100 | mid | UR — **medium priority (UR ecosystem)** | clean fit |
| OnRobot HEX-E QC / HEX-H QC | gripper-arm interface | 500 | mid/high | UR, OnRobot ecosystem | clean fit |
| Wittenstein WT 24 / 26 | wrist | 1000 | mid | research | clean fit |
| (intrinsic) Franka Panda / FR3 joint torques | per-joint | 1000 | per-joint | Franka — covered by `JointSpec.has_torque_sensor` | covered |
| (intrinsic) Kuka iiwa joint torques | per-joint | 1000 | per-joint | Kuka — covered by `JointSpec.has_torque_sensor` | covered |

### 2.7 Tactile (`SensorModality.TACTILE_VISION`, `SensorModality.TACTILE_ARRAY`)

Vision-based fingertips fit `TACTILE_VISION`; magnetic / capacitive / resistive
arrays fit `TACTILE_ARRAY` with `tactile_grid = (rows, cols)`.

| Vendor / model | Tech | Resolution | Mount | Used on | Schema notes |
|---|---|---|---|---|---|
| GelSight Mini | vision-based (camera + gel) | ~320×240 image, ~30 µm spatial | fingertip | Franka, UR research, dexterous hands — **high priority** | `TACTILE_VISION` |
| GelSight DIGIT | vision-based (open-source design) | 640×480@60 | fingertip | open-source dexterous research — **high priority** | `TACTILE_VISION` |
| GelSlim 4 / GelSight Wedge | vision-based (flat finger) | varies | parallel-gripper finger | research | `TACTILE_VISION` |
| AnySkin (NYU/Pinto) | magnetic, replaceable | 5-axis force, ~32 magnets | hand skin | open-source dexterous — **high priority** | `TACTILE_ARRAY`; magnetic in metadata |
| ReSkin (CMU/Meta) | magnetic, replaceable | similar | hand skin | open-source | `TACTILE_ARRAY`; magnetic in metadata |
| XELA Robotics uSkin | magnetic | 4×4 taxels × 3-axis force | fingertip / palm | dexterous research | `TACTILE_ARRAY` 4×4 |
| Contactile PapillArray | barometric / capacitive | 9 taxels × 3-axis | fingertip | research | `TACTILE_ARRAY` 3×3 |
| PaXini DexH13 | capacitive | varies | fingertip | humanoid hands | `TACTILE_ARRAY` |
| TacTip family | vision-based | varies | fingertip | research | `TACTILE_VISION` |

### 2.8 Audio, GPS, battery — already covered

| Modality | Common hardware | Schema notes |
|---|---|---|
| `AUDIO` | ReSpeaker 4 / 6-Mic Array, USB lavaliers, robot built-ins | clean fit |
| `GPS` | u-blox ZED-F9P (RTK), u-blox NEO-M9N, Septentrio AsteRx-i / mosaic-X5 | clean fit |
| `BATTERY` | smart-battery telemetry over CAN/I²C; surfaced via `WorldState.battery_pct` | clean fit |
| `JOINT_STATE` | actuator encoders (Feetech STS3215 on SO-100/101, Dynamixel, Maxon, Harmonic Drive) | typically expressed via `JointSpec.has_position_sensor / has_velocity_sensor / has_torque_sensor`, not a standalone `SensorSpec` |

---

## 3. Schema-compliance audit

For each modality above, the existing `SensorSpec` fields are sufficient
(green) unless flagged. Summary:

| Modality | `SensorModality` enum | Sufficient `SensorSpec` fields? | Notes |
|---|---|---|---|
| RGB | `RGB` | ✅ `intrinsics`, `encoding`, `fov_*` | — |
| Depth | `DEPTH` | ✅ `intrinsics`, `range_*`, `fov_*` | — |
| Stereo (passive) | `STEREO` | ⚠️ baseline missing | put `baseline_m` in `metadata` until §4 gap S resolved |
| RGBD bundle | RGB + DEPTH (+ IMU) | ✅ via `SensorBundle` | sync = `hardware` for RealSense / Orbbec |
| 2D LiDAR | `LIDAR_2D` | ✅ `range_*`, `fov_h_deg`, `rate_hz` | — |
| 3D LiDAR | `POINT_CLOUD` + `n_channels` | ✅ for rotating; metadata for non-repetitive solid-state | document Livox `non_repetitive=True` convention |
| IMU (6-DoF) | `IMU` | ✅ `accel_noise_density`, `gyro_noise_density` | — |
| IMU (9-DoF) | `IMU` | ⚠️ no magnetometer noise field | put `mag_noise_density` in `metadata` until §4 gap M resolved |
| GNSS-INS | `GPS` + `IMU` `SensorBundle` | ✅ | bundle the two |
| 6-axis F/T | `FORCE_TORQUE` + `n_axes=6` | ✅ | — |
| Vision tactile | `TACTILE_VISION` | ✅ | optionally fill `intrinsics` for the embedded camera |
| Array tactile | `TACTILE_ARRAY` + `tactile_grid` | ✅ | put magnetic / capacitive / barometric tech in `metadata` |
| Audio | `AUDIO` | ✅ | — |
| GPS | `GPS` | ✅ | — |
| Battery | `BATTERY` | ✅ | — |

---

## 4. Identified schema gaps (proposals — **none of these block any factory work above**)

Each of the following is a **proposal** for a future ADR + minor SemVer bump
of `openral_core` (CLAUDE.md §1.6). Until then, sensors using these
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

Recommendation: bundle E + T + U + B + M + C + S into a single additive ADR
that extends `SensorSpec` in place against the pre-publish baseline
(`schema_version: "0.1"`), with a doctest loading a fixture manifest.

---

## 5. Catalog & factory implementation status

The catalog ships only the sensor entries that are actually wired into a HAL
adapter today.  Each entry is addressable by stable id (`<vendor>/<model>`)
through the `SensorCatalog` registry and surfaced via the `openral sensor list /
show` CLI.  Speculative entries (extra RealSense SKUs, Orbbec, Logitech C922 /
Brio, Arducam, every 2D / 3D LiDAR, standalone IMUs, ATI / Bota / OnRobot F/T,
GelSight DIGIT, XELA, AnySkin) were dropped in the cleanup pass; reintroduce
them only when a robot manifest or HAL factory needs them.

### What shipped (5 entries, 3 modules)

| Module | Catalog ids | Modalities | Wired in |
|---|---|---|---|
| `realsense.py` | `intel/realsense_d435`, `intel/realsense_d435i`, `intel/realsense_d415` | RGB+DEPTH(+IMU) bundles | `ur10e_with_sensors`, `franka_panda_with_sensors`, `ur5e_with_sensors` |
| `usb_uvc.py` | `logitech/c920` | RGB | `so100_with_sensors` |
| `force_torque.py` | `robotiq/ft_300s` | FORCE_TORQUE (n_axes=6) | `ur5e_with_sensors`, `ur10e_with_sensors` |

### How to use

```python
from openral_sensors import CATALOG

# List
print(CATALOG.list_ids())                       # 5 ids
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

### Not yet shipped — Tier 3 (gated on schema v0.3 ADR)

| Module | Factories | Blocked-by |
|---|---|---|
| `stereolabs.py` | `zed_2i_bundle`, `zed_x_bundle` | gap S (stereo `baseline_m`) |
| `luxonis.py` | `oak_d_pro_bundle`, `oak_d_lite_bundle` | gap S |
| `event_camera.py` | `prophesee_evk4_spec`, `inivation_davis346_spec` | gap E |
| `thermal.py` | `flir_boson_spec`, `spot_thermal_payload_spec` | gap T |
| `proximity.py` | `maxbotix_ultrasonic_spec`, `teraranger_evo_spec` | gap U |
| `barometer.py` | `bmp388_spec` | gap B |

---

## 6. Concrete style for new modules

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

## 7. Summary — status & next steps

1. ✅ **Catalog pruned to actually-wired sensors.**  `openral_sensors` now
   ships 5 catalog entries across 3 vendor modules (`realsense.py`,
   `usb_uvc.py`, `force_torque.py`), `SensorCatalog` registry, `openral sensor
   list / show` CLI, full unit-test coverage.  Any speculative entry (extra
   LiDAR, IMU, tactile, depth-camera SKUs, additional UVC cameras) was removed
   — reintroduce them on demand when a robot manifest or HAL adapter actually
   needs them.  No schema change required.
2. **Open one ADR** for the §4 schema additions (event, thermal, range_1d,
   barometer, magnetometer, contact, stereo baseline) bundled together as
   `schema v0.3`, with a single migration entry and a regenerated
   `docs/reference/schemas/`.
3. After the ADR lands, ship Tier 3 modules (`stereolabs.py`, `luxonis.py`,
   `event_camera.py`, `thermal.py`, `proximity.py`, `barometer.py`) as
   separate small PRs, each with the test shape mandated in §6.
4. Optional follow-up: add a `catalog:` reference field on
   `RobotDescription.sensors` / `RobotDescription.sensor_bundles` so a
   `robot.yaml` can say `- catalog: intel/realsense_d435i` instead of
   inlining the whole `SensorBundle`.  This is a tiny v0.2 → v0.3 schema
   migration; folding it into the same ADR keeps the migration count down.

---

*Last reviewed: 2026-05-06. Re-review whenever a new humanoid platform ships
or a major depth/lidar vendor changes status.*
