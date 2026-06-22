# Feasibility: NVIDIA Isaac ROS Visual SLAM (cuVSLAM) for lidar-less robots

> Investigation only. No code changes. Status: **feasible for localization, with caveats**; the
> requested framing ("add as an rSkill") is the wrong abstraction — it belongs in the
> Sensors / World-State layers, mirroring the existing `slam_toolbox` integration.
> Date: 2026-06-21.
>
> **Decision recorded in [ADR-0064](../adr/0064-vision-slam-lidarless-cuvslam-nvblox-monodepth.md)** —
> includes the nvblox occupancy path and a monocular metric-depth model survey (DA3-Small / Depth Pro)
> for feeding nvblox on camera-only robots.

## 1. The ask

Add [`isaac_ros_visual_slam`](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_visual_slam/index.html)
(NVIDIA-accelerated **cuVSLAM**) so robots **without a lidar** can localize/map from cameras and
feed ROS 2 SLAM, in both sim and real. The user proposed packaging it as an **rSkill**.

## 2. Headline findings

| Question | Answer |
|---|---|
| Is "rSkill" the right abstraction? | **No.** cuVSLAM is a continuous localization node → Layer 1 (Sensors) / Layer 2 (World State), like our existing `slam_toolbox`. None of the 7 rSkill kinds fit (see §3). |
| ROS distro match? | **Yes (now).** Current Isaac ROS supports **ROS 2 Jazzy**; OpenRAL is on Jazzy + Cyclone DDS. (Pre-2026 Isaac ROS was Humble-only — this would have been a hard blocker a year ago.) |
| Does it replace lidar SLAM 1:1? | **No.** cuVSLAM gives you the *localization* half (visual odometry + `map→odom` TF + loop closure). It does **not** emit a `nav_msgs/OccupancyGrid`. Nav2's global costmap needs that grid. (see §5). |
| Lidar-free Nav2 recipe exists? | **Yes** — NVIDIA's stack is **cuVSLAM + nvblox** (a.k.a. Isaac Perceptor). nvblox turns depth + VSLAM pose into the occupancy/ESDF grid Nav2 consumes. |
| Does the integration seam exist in OpenRAL? | **Yes, cleanly.** `openral_slam_bringup`, the `enable_slam` gate, `slam_bridge` OTel, Nav2 expecting `map→odom` + `/map`, and the camera/sensor publishing plumbing are all already there. |
| Effort | **Localization-only: Medium.** Full lidar-free Nav2 (localization + occupancy): **Larger** (adds nvblox + depth). Real-hardware: gated on a real mobile-base HAL that doesn't exist yet. |

## 3. Why it is NOT an rSkill

The rSkill manifest (`openral_core.schemas.RSkillManifest`) has exactly 7 `kind` values:
`vla, wam, ros_action, ros_service, detector, vlm, reward`. rSkills are **dispatch units** — the
Reasoner invokes them via `ExecuteSkill`, they emit action chunks (vla/wam/ros_action) or isolated
perception products (detector/vlm/reward). cuVSLAM is none of these:

- It is **continuous and ambient**, not goal-driven — there is no chunk cadence (`chunk_size`),
  and its latency is frame-rate-bound and spikes on loop closure (violates `latency_budget.per_chunk_ms`).
- `ros_action`/`ros_service` rSkills wrap a **request/response** ROS server. cuVSLAM is a
  free-running publisher of TF/odometry — it has no goal interface.
- Its output (`map→odom` TF, `/visual_slam/...` odometry) is **World State**, consumed by the
  `WorldStateAggregator` and Nav2 — not returned to the Reasoner.

**Correct home:** a new visual-SLAM backend inside `packages/openral_slam_bringup/` (a sibling of
the `slam_toolbox` launch/config), brought up as a Reasoner-managed lifecycle node exactly like
today's lidar SLAM (ADR-0025). The `RobotCapabilities`/gate logic chooses the backend.

This is a **layer-boundary / capability change → ADR required** (and a `RobotCapabilities` schema
addition → `schema_version` bump + migrator).

## 4. The integration seam already in place

From the repo investigation, almost all the plumbing a visual-SLAM backend needs exists:

- **SLAM service pattern**: `packages/openral_slam_bringup/` runs `slam_toolbox` as a lifecycle
  node, auto `UNCONFIGURED→INACTIVE`, Reasoner drives `→ACTIVE` (ADR-0025).
- **Gate**: `python/cli/src/openral_cli/deploy_sim.py` (~L441) — `enable_slam = bool(has_lidar)`,
  `enable_nav2` co-enables. A visual backend extends this to
  `has_lidar OR has_vision_slam`.
- **Nav2 contract**: `config/nav2_panda_mobile.yaml` expects `map` frame + `/map` OccupancyGrid
  (global_frame: map, static/obstacle layers).
- **Observability**: `python/runner/src/openral_runner/slam_bridge.py` throttles `/map`→1 Hz, emits
  `slam.occupancy_grid` OTel span — a visual backend reuses this.
- **Cameras**: `python/sensors/` publishes `Image` + `CameraInfo`; `hal/sim_sensor_bridge.py`
  renders cameras for **panda_mobile** and **OpenArm** in sim, with TF optical frames.
- **TF tree**: `MobileBaseBridge` publishes `odom→base_link`; today `slam_toolbox` fills
  `map→odom`. A visual backend fills the same `map→odom` edge.

So the "drop-in point" is unambiguous: **publish `map→odom` (and, for nav, `/map`) from cameras
instead of from `/scan`.**

## 5. The real catch: cuVSLAM ≠ occupancy map

cuVSLAM outputs **odometry + `map→odom` TF + a keypoint/landmark map for loop closure**. It does
**not** output a `nav_msgs/OccupancyGrid`. OpenRAL's Nav2 global costmap is built around an
OccupancyGrid `/map`. Therefore:

- **Localization for lidar-less robots** (fill the `map→odom` edge so poses are world-anchored):
  cuVSLAM alone is sufficient. **Medium effort.**
- **Full lidar-free navigation** (Nav2 planning through obstacles): you also need an occupancy
  source. NVIDIA's answer is **nvblox** (`isaac_mapping_ros` = cuVSLAM + nvblox builds the grid;
  this is the [Nav2 "lidar-free vision-based navigation"](https://docs.nav2.org/tutorials/docs/using_isaac_perceptor.html)
  tutorial / Isaac Perceptor). Alternative lighter path: `depthimage_to_laserscan` → feed Nav2's
  obstacle layer as a synthetic `/scan`. **Larger effort, second NVIDIA dependency.**

Be explicit about which goal we're funding before committing — they differ by an entire package.

## 6. Hard requirements & friction

**Platform (x86):** Ampere-or-newer NVIDIA GPU, CUDA 13.0+, Driver 580+, Ubuntu 24.04, 8 GB+ RAM,
32 GB+ disk. **Jetson:** currently **Thor only** (JetPack 7.1) on the latest release.
- Dev host is an 8 GB **Ada** GPU — architecture is fine (Ada > Ampere), but 8 GB VRAM is already
  tight for our NF4-quantized VLAs. cuVSLAM is comparatively light, but co-residency with a VLA
  on 8 GB needs measuring.

**Camera input:** primary mode is **stereo**; also mono+IMU, and **RGB-D** (added Feb 2026).
Min **30 Hz** framerate, **±2 ms** jitter.
- OpenRAL sim (panda_mobile) publishes a **single mono RGB** camera at **10 Hz**, no IMU, no stereo
  pair. To run cuVSLAM you must add **stereo, or depth (RGB-D), or an IMU stream**, and raise the
  rate. The 30 Hz/±2 ms jitter spec is genuinely hard in our **env-stepped deploy-sim clock**
  (no free-running sim clock) — a known constraint of our sim time model.

**Container / NITROS:** cuVSLAM ships as a NITROS-accelerated package expecting the **Isaac ROS
dev container**. OpenRAL has its own Jazzy/Cyclone `Dockerfile.dev`. Options: (a) pull Isaac ROS
apt packages into our image, or (b) run cuVSLAM in its own container and bridge over DDS. cuVSLAM
accepts standard `sensor_msgs/Image` topics, so bridging sim cameras in works — you just forgo
zero-copy NITROS.

**Simulator:** cuVSLAM's *supported* sim tutorial is **Isaac Sim only**. Our sim is
MuJoCo/robosuite/robocasa (the Isaac Sim seam is feasibility-blocked: py3.12 vs Omniverse). cuVSLAM
does not *require* Isaac Sim — it consumes ROS image topics — so MuJoCo-rendered stereo/depth
published as `sensor_msgs/Image` would work, but off the tested path, and subject to the
30 Hz/jitter friction above.

**License (compliance, per CLAUDE.md §9):** the `isaac_ros_visual_slam` ROS wrapper repo is open,
but the **cuVSLAM engine ships as a precompiled NVIDIA library** governed by an NVIDIA EULA — this
is *not* OpenRAL's Apache-2.0 code and is *not* an open weight. It must be treated like the GR00T
backend: **behind a license-posture flag + install-time/env guard, never bundled** (mirror
ADR-0046's pattern). Confirm the exact EULA terms before shipping. Do **not** describe it as open.

**Real hardware:** panda_mobile is **sim-only** — no real HAL exists. "Run on real" is aspirational
until there's a real mobile base + a real stereo/RGB-D camera HAL. The sim path is the only one
testable today.

## 7. Recommended path (if pursued)

1. **ADR** "Visual-SLAM backend for lidar-less localization" — adds `RobotCapabilities.has_vision_slam`
   (schema bump + migrator), defines backend selection in the SLAM gate, and records the
   cuVSLAM license posture + env guard.
2. **Phase 1 — localization only:** add an `isaac_ros_visual_slam` launch + config to
   `openral_slam_bringup`; publish a **stereo or RGB-D + IMU** camera stream for a target robot
   (start with panda_mobile in sim); wire `map→odom`; reuse `slam_bridge` for observability.
   Acceptance: world-anchored pose with no `/scan`.
3. **Phase 2 — navigation (optional):** add **nvblox** (or depth→laserscan) to produce the
   OccupancyGrid Nav2 needs; validate find→navigate in deploy-sim without a lidar.
4. **Real hardware:** deferred until a real mobile-base HAL + calibrated stereo/RGB-D camera exist.

## 8. Bottom line

- **"As an rSkill": no** — it's a Sensors/World-State ROS node, a second backend beside
  `slam_toolbox`. The seam for that is clean and already exists.
- **Distro/GPU: compatible** (Jazzy + Ada), unlike a year ago.
- **Localization for lidar-less robots: feasible, Medium effort.** The honest gap is that cuVSLAM
  gives pose, not an occupancy map — **full lidar-free Nav2 needs nvblox** and is a larger lift.
- **Top frictions:** stereo/depth+IMU camera at 30 Hz/±2 ms in an env-stepped sim; NITROS/Isaac
  container packaging; closed cuVSLAM binary license guard; and no real HAL yet.

### Sources
- Isaac ROS Visual SLAM: https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_visual_slam/index.html
- cuVSLAM concept: https://nvidia-isaac-ros.github.io/concepts/visual_slam/cuvslam/index.html
- Isaac Sim tutorial: https://nvidia-isaac-ros.github.io/concepts/visual_slam/cuvslam/tutorial_isaac_sim.html
- Lidar-free Nav2 (cuVSLAM + nvblox / Perceptor): https://docs.nav2.org/tutorials/docs/using_isaac_perceptor.html
- isaac_mapping_ros (occupancy from cuVSLAM + nvblox): https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_mapping_and_localization/isaac_mapping_ros/tutorial_map_creation.html
