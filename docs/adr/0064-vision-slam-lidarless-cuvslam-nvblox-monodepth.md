# ADR-0064 — Vision-based SLAM & mapping for lidar-less robots (cuVSLAM + nvblox + monocular metric depth)

- **Status:** Accepted — 2026-06-22. Phase 1 (cuVSLAM bring-up + `has_vision_slam` gate) and Phase 2 (nvblox bring-up + the DA3 monocular metric-depth sidecar/provider) implemented in-tree and **validated live end-to-end on an 8 GB Ada (RTX 4070 Laptop)** against the NVIDIA Isaac ROS stack (`ros-jazzy-isaac-ros-visual-slam` + `ros-jazzy-isaac-ros-nvblox` 4.4.0 + Jetson-x86_64 VPI/nvsci, installed via `just install-isaac-ros`):
  - **cuVSLAM trajectory:** `cuvslam.launch.py` ran cuVSLAM 15.0.0 on the Isaac ROS quickstart stereo bag and produced a real VO trajectory (17 poses, **1.285 m** net / 1.320 m path).
  - **Full lidar-free occupancy map:** the complete stack — bag stereo → cuVSLAM pose + bag RGB → the DA3 depth provider (`depth_provider`/`depth_convert` + DA3 sidecar) → 32FC1 metric depth → `nvblox.launch.py` — produced a coherent `nav_msgs/OccupancyGrid` (**80×64 cells @ 0.05 m = 4.0×3.2 m, 1566 occupied / 79 free cells**, 32 % observed), now remapped onto the backend-agnostic **`/map`** — **with no lidar**.
  - **Backend-agnostic Nav2 (Decision §5):** with nvblox's grid on `/map` and the **`nav2_visual.yaml`** costmap profile (static_layer on `/map`, no `/scan`), the full Nav2 stack activated ("Managed nodes are active") and **`ComputePathToPose` returned a 132-waypoint path (SUCCEEDED) consuming only `/map`** — proving Nav2 plans independently of how the 2D map is built. 3D-object-lift is already backend-agnostic (uses the `map` TF, not the grid topic).
  - Unit/gate/convert/launch tests green. Live-testing fixes landed: `nvblox.launch.py` remaps deploy-sim's manifest depth camera (`/openral/cameras/front_depth/depth/*`) through `depth_height_filter_node.py`, which derives a floor-excluded body-height band from `RobotDescription` footprint/collision/link measurements plus live TF before nvblox, then remaps nvblox's camera-namespaced `camera_0/depth/*` inputs **and** `~/static_occupancy_grid → /map`; `nvblox.yaml` deliberately keeps nvblox workspace bounds unbounded so the YAML does not bake in one scene's floor height; `cuvslam.yaml` documents `rectified_images` for raw (`rational_polynomial`) vs rectified cameras; `nav2_visual.yaml` static_layer uses `map_subscribe_transient_local: False` (nvblox `/map` is VOLATILE) and disables the collision_monitor scan source.
  - **Caveats (replay-only, vanish on a real robot):** a single bag replay can't satisfy both cuVSLAM's ~30 fps need and the DA3 ~2 fps path, so playback is slowed; and a consistent **sim-time** clock (`--clock` + `use_sim_time:=true`) is required so cuVSLAM's pose TF and the depth stamps align in nvblox's TF buffer. A live robot has native-rate stereo, continuous depth, and one shared clock.
- **Date:** 2026-06-21 (Phase 2 + measured results: 2026-06-22)
- **Related:** ADR-0025 (Reasoner-managed background services — the `slam_toolbox`
  lifecycle pattern this extends), ADR-0046 (NVIDIA out-of-process backend +
  version-keyed license posture — the precedent reused here), ADR-0006 (HF-Hub
  packaging + license guard), ADR-0045 (Isaac Sim backend feasibility), ADR-0048
  (deploy-sim `/clock` publisher), ADR-0049 (HAL executor / proprio snapshot),
  ADR-0050 (single-resident-skill VRAM eviction — the budget this competes with).

## Context

OpenRAL's only SLAM today is **`slam_toolbox`** (2D lidar; `packages/openral_slam_bringup/`,
ADR-0025), gated on `RobotCapabilities.has_lidar` in `deploy_sim.py` (`enable_slam =
bool(has_lidar)`; `enable_nav2` co-enables). A robot without a lidar gets **no `map` frame and
no Nav2** — it cannot navigate absolutely. Only `panda_mobile` declares a `lidar_2d` sensor; the
rest of the fleet (fixed-base arms, camera-only mobile manipulators) is shut out.

A prior investigation (`docs/feasibility/isaac-ros-visual-slam.md`) established:

1. **NVIDIA Isaac ROS Visual SLAM (cuVSLAM) now supports ROS 2 Jazzy** — matching OpenRAL's
   distro + Cyclone DDS. (Pre-2026 it was Humble-only; that mismatch is gone.)
2. **cuVSLAM is not an rSkill.** It is a continuous, ambient localization publisher (visual
   odometry + `map→odom` TF + loop closure), not a Reasoner-dispatched action/perception unit.
   None of the 7 `RSkillKind` values fit. Its home is **Layer 1/2 (Sensors / World State)**, as a
   second SLAM backend beside `slam_toolbox`.
3. **cuVSLAM gives pose, not an occupancy map.** Nav2's global costmap needs a
   `nav_msgs/OccupancyGrid`; cuVSLAM emits only a keypoint/landmark map for loop closure. The
   NVIDIA lidar-free recipe is **cuVSLAM + nvblox** (Isaac Perceptor): nvblox fuses **depth +
   pose** into a TSDF and emits a 2D costmap for Nav2 (`nvblox_nav2`).

The open problem for **camera-only** robots is the **depth input nvblox needs**. nvblox accepts a
`sensor_msgs/Image` depth + `camera_info` + a pose TF **from any source** — it does not require a
stereo or lidar sensor. That opens the door to **AI monocular metric-depth models**: synthesize
metric depth from a single RGB stream and feed nvblox, with cuVSLAM supplying pose.

### Monocular metric-depth model survey (the new investigation)

Goal: a model that runs **fast** and **fits alongside** a VLA on the lab's **8 GB Ada** host. nvblox
does not need high-rate depth (it integrates a TSDF at whatever rate arrives; 5–15 Hz is workable),
so the bar is metric accuracy + VRAM, not 30 Hz.

**Measured on the lab host** (RTX 4070 Laptop, 8 GB Ada, driver 580; fp16/native, single 640×480
frame, peak `torch.cuda.max_memory_allocated`, post-warmup latency) — these replace the earlier
web-sourced estimates:

| Model | Metric? | VRAM (measured) | Latency (measured) | Notes / license |
|---|---|---|---|---|
| **Depth Anything 3 — Small** (DA3, ByteDance, Nov 2025, [arXiv 2511.10647](https://arxiv.org/abs/2511.10647)) | yes — depth range 0.69–1.31 m **matched Depth Pro's 0.72–1.19 m** on the same frame (`is_metric` flag returned an empty dict — corroboration, not absolute-scale proof) | **0.27 GB** (reserved 0.35) | **36 ms ≈ 27 Hz** (fwd pass ~19 ms ≈ 50 Hz) at `process_res=504` | Plain-transformer; also predicts intrinsics. **Default depth provider** — ~30× faster and ~15× lighter than Depth Pro here, far under the earlier "≈2 GB" estimate. Ships as the `depth-anything-3` package (not transformers-native → isolated sidecar venv). License: verify per checkpoint. |
| DA3 — Base / Large / Giant | metric | not measured (≈4 / 8 / 16+ GB est.) | Large+ not real-time | Higher accuracy, batch/offline only. |
| **Depth Pro** ("**Sharp** Monocular Metric Depth in Less Than a Second", Apple, [arXiv 2410.02073](https://arxiv.org/abs/2410.02073), `apple/DepthPro-hf`) | metric, absolute scale | **3.95 GB** (reserved 6.04) | **~1040 ms ≈ 1 Hz** (fp16, 640×480) | transformers-native (`DepthProForDepthEstimation`). Best boundary sharpness, but on this 8 GB laptop Ada it is ~1 Hz — **slower/heavier than the web "~3 Hz" claim**. Opt-in "sharp-edges" provider, not the default. *This is the "SHARP" model in the request.* License: Apple ML-research terms — verify before bundling. |
| Metric3D v2 / UniDepth v2 | metric | not measured | moderate | Mid options; not pursued given DA3-Small's result. |

**Pick (validated by measurement):** **DA3-Small** is the default depth provider — at 27 Hz / 0.27 GB
it is effectively free on the budget and *exceeds* cuVSLAM's 30 Hz-class cadence (nvblox needs far
less). **Depth Pro** stays an opt-in provider for boundary fidelity when its ~1 Hz / ~4 GB cost is
acceptable. Caveat: absolute metric accuracy is corroborated (two independent models agree) but
**unverified against ground-truth depth** — confirm on a RealSense/sim depth pair before trusting
scale for tight-clearance navigation.

### The binding constraint: VRAM

**Correcting the platform-spec vs. actual-footprint confusion:** NVIDIA's "Ampere+, 8 GB+ VRAM" is
the *install/platform floor* for the Isaac ROS stack, **not** the runtime consumption of these nodes.
Measured ([cuVSLAM paper](https://arxiv.org/html/2506.04359v2), Jetson AGX Orin, RealSense 640×480):
cuVSLAM **stereo** uses **1.7 % GPU** (1.8 ms/call) and **stereo-inertial 2.2 %** — a few hundred MB
of working memory. **But mono-depth mode jumps to 55 % GPU / 15 ms** — the lidar-less, camera-only
case is the expensive cuVSLAM mode. nvblox is a *sparse* TSDF whose footprint scales with mapped
volume × voxel resolution (only observed 8³ blocks allocate); room-scale at 3–5 cm voxels is
~hundreds of MB to ~1–2 GB. **cuVSLAM + nvblox together ≈ 1–2 GB of real VRAM** for a room-scale map
— well under 8 GB.

So on the 8 GB Ada host the SLAM nodes are not the squeeze: **cuVSLAM + nvblox + DA3-Small co-fit
(~3–4 GB)**. The only thing that would overflow 8 GB is a resident VLA on top — **but that never
happens**, because navigation and manipulation are **sequential phases**, not concurrent: the robot
drives/maps (vision-nav stack resident), *then* loads a VLA to act after it has arrived. No VLA is
loaded while navigating. This is precisely the single-resident-skill eviction model of **ADR-0050** —
the vision-nav stack occupies the resident slot during the nav phase and is evicted before the VLA
loads for the manipulation phase. **No co-residency, no second GPU, no cloud dispatch required.** The
one footprint to watch is cuVSLAM's **mono-depth** mode (~55 % GPU) on truly camera-only robots.

### Other frictions (from the feasibility doc)

- **cuVSLAM input**: primary mode is **stereo**; also mono+IMU and RGB-D (Feb 2026). Sim publishes a
  single **mono RGB @ 10 Hz**, no IMU/stereo. Requires **30 Hz, ±2 ms jitter** — hard under the
  **env-stepped deploy-sim clock** (ADR-0048: no free-running sim clock).
- **Packaging/NITROS**: cuVSLAM/nvblox are NITROS packages expecting the Isaac ROS container; they
  accept standard image topics, so bridging works but forgoes zero-copy.
- **Closed binaries**: the cuVSLAM/nvblox ROS wrappers are open, but the **engines ship as
  precompiled NVIDIA libraries** under an NVIDIA EULA — third-party, not OpenRAL Apache-2.0.
- **Real hardware**: `panda_mobile` is sim-only (no real HAL). "Run on real" is deferred.

## Decision

Add a **vision-based SLAM + mapping backend for lidar-less robots**, as a Sensors/World-State
component — **not an rSkill**. Concretely:

1. **Backend in `openral_slam_bringup`.** Add a `visual` SLAM backend (launch + config) alongside
   `slam_toolbox`, brought up as a Reasoner-managed lifecycle node exactly like ADR-0025
   (`UNCONFIGURED→INACTIVE` auto; `→ACTIVE` Reasoner-driven). The stack is **cuVSLAM (pose,
   `map→odom`) + nvblox (depth+pose → `/map` OccupancyGrid for Nav2)**.

2. **Monocular metric-depth provider** for camera-only robots, as an **out-of-process model
   sidecar** (reuse the ZMQ pattern of ADR-0046/0010): RGB in → `sensor_msgs/Image` (32FC1 metric
   depth) + `camera_info` out, consumed by nvblox. Default **DA3-Small**; **Depth Pro** selectable.
   - **AI depth feeds nvblox only, never cuVSLAM odometry.** TSDF fusion tolerates per-frame depth
     noise; visual odometry does not — AI depth's scale drift / temporal inconsistency would
     corrupt pose. cuVSLAM keeps its **real geometry** path (stereo or mono+IMU).
   - Corollary (honest limit): a **truly mono, no-IMU** robot cannot get reliable **metric pose**
     from cuVSLAM. Such robots get **mapping in a drifting frame** at best; robust metric VO needs
     stereo or an IMU. This is called out, not papered over (CLAUDE.md §1.2).

3. **Capability gating.** Add `RobotCapabilities.has_vision_slam` (Pydantic schema change →
   `schema_version` bump + migrator, CLAUDE.md §1.6). Backend selection becomes:
   `has_lidar → slam_toolbox`; `elif has_vision_slam → cuVSLAM(+nvblox)`; `else → no SLAM`. Update
   the `deploy_sim.py` gate accordingly.

4. **License posture for the NVIDIA binaries.** Treat cuVSLAM + nvblox engines like the GR00T
   backend (ADR-0046): **behind a license-posture flag + install-time/env guard, never bundled**;
   confirm the NVIDIA EULA terms. The depth-model weights keep their **own upstream license**,
   verified per checkpoint (CLAUDE.md §9). Do not describe any of these as "open" until confirmed.

5. **One backend-agnostic `/map` interface.** Downstream consumers must not care whether the 2D map
   came from lidar `slam_toolbox` or vision `cuVSLAM+nvblox`:
   - **Occupancy grid:** `nvblox.launch.py` remaps nvblox's `~/static_occupancy_grid` → **`/map`**
     (`nav_msgs/OccupancyGrid`), the same topic slam_toolbox publishes. The launch inserts
     `depth_height_filter_node.py` before nvblox, zeroing depth pixels outside a floor-excluded
     robot-measurement-derived body-height band; otherwise forward/downward depth cameras project floor returns
     into `/map` as occupied cells. Isaac ROS nvblox 4.4 applies `static_mapper.workspace_bounds_*`
     to TSDF/ESDF view calculation but leaves the camera occupancy integrator unbounded, so the
     depth prefilter is the control that makes `/map` nav-quality. The filter loads the same
     `robot.yaml` as the deploy graph, derives a robot-relative band from the footprint,
     `collision_geometry`, and link transforms, then shifts that band by the live
     `map→base_frame` TF on every depth frame; camera pose is still used for per-pixel global-z
     projection. This removes the previous RoboCasa/panda_mobile hardcoded map-z band and keeps the
     filter independent of scene floor height. The dashboard `slam_bridge`, the
     reasoner's `occupancy_map_topic`, and Nav2's `static_layer` then consume `/map` unchanged.
     (Caveat: nvblox's `/map` is **RELIABLE + VOLATILE** — live-updating, not latched — so a
     consumer's `static_layer` must use `map_subscribe_transient_local: False`; verified live.)
   - **Nav2:** the base config navigates off `/scan` (lidar `obstacle_layer`), which a lidar-less
     robot lacks. The visual backend gets **`nav2_visual.yaml`** (derived from the base via
     `tools/gen_nav2_visual.py`): global+local costmaps consume `/map` via `static_layer` instead of
     `/scan`, and the collision_monitor's scan source is disabled. `nav2.launch.py slam_backend:=…`
     selects the profile (`visual` → `nav2_visual.yaml`); `sim_e2e.launch.py` forwards the resolved
     `slam_backend`. So Nav2 plans off `/map` **regardless of how the map was built**.
   - **3D-lifted detected objects (ADR-0035/0052):** already backend-agnostic — `object_lift.py`
     uses the **`base_link→map` TF2 transform** (which cuVSLAM publishes just like slam_toolbox),
     not the `/map` topic, and writes objects to the scene graph / `world_state_slow`, never into
     the occupancy grid. **No change needed**; it works identically on both backends.

## Scope

Live cuVSLAM/nvblox mapping needs the closed NVIDIA Isaac ROS stack on a GPU host and is
operator-run; the in-tree code + hermetic launch contracts + the GPU-validated depth path land now
(mirroring ADR-0046's PR split between landed code and operator-run live eval).

- **Phase 1 — localization (LANDED):** `has_vision_slam` capability + `_resolve_slam_backend`
  (`lidar|visual|none`) gate in `deploy_sim.py`; `cuvslam.launch.py` + `cuvslam.yaml` in
  `openral_slam_bringup`; `sim_e2e.launch.py` branches `enable_slam` on `slam_backend`. *Tested:*
  colcon build + launch-contract tests; real `ros2 launch cuvslam.launch.py` starts the container
  and stops exactly at loading `nvidia::isaac_ros::visual_slam::VisualSlamNode`. Operator step:
  install `isaac_ros_visual_slam` + a stereo/RGB-D/mono+IMU stream; acceptance is world-anchored
  pose with no `/scan`.
- **Phase 2 — mapping/nav (LANDED):** `nvblox.launch.py` + `nvblox.yaml` (composes
  `nvblox::NvbloxNode`, ESDF slice for the `nvblox_nav2` costmap); the monocular metric-depth
  provider — `openral_perception_ros/depth_convert.py` (32FC1 metres ↔ Image, unit-tested) +
  `depth_provider_node.py` + the DA3 sidecar (`tools/da3_depth_sidecar.py` /
  `tools/_da3_depth_server.py`); `sim_e2e.launch.py` composes nvblox in the visual branch when
  `enable_nav2`. *Tested:* launch contracts built + real-launched (reach `nvblox::NvbloxNode`);
  **DA3-Small validated live on an 8 GB Ada** — model + ZMQ sidecar roundtrip (0.27 GB, ~27 Hz,
  metric range matching Depth Pro). Operator step: install `nvblox_ros`, wire the `nvblox_nav2`
  costmap plugin, run the depth sidecar; acceptance is find→navigate in deploy-sim without a lidar.
- **Phase 3 — real hardware:** deferred until a real mobile-base HAL + calibrated camera exist.

## Consequences

- **Lidar-less robots gain absolute navigation** — the core goal — via the supported NVIDIA
  lidar-free stack, on a distro that now matches.
- **The clean seam is reused**: SLAM lifecycle, the gate, `slam_bridge`, Nav2's `map`/`/map`
  contract, and camera publishing already exist; the new work is a backend + a depth sidecar + a
  capability flag + an ADR-keyed license guard.
- **VRAM is a non-issue under sequential phasing.** The vision-nav stack (cuVSLAM + nvblox +
  DA3-Small ≈ 3–4 GB) and a VLA are never co-resident — navigation and manipulation run in separate
  phases, so the existing ADR-0050 single-resident eviction (nav stack → evict → VLA) covers it on
  8 GB with no second GPU or cloud offload. (Only watch item: cuVSLAM's **mono-depth** mode at
  ~55 % GPU on truly camera-only robots.)
- **New third-party closed binaries** enter the tree behind a guard, plus a depth-model weight whose
  license is verified per checkpoint — added compliance surface.
- **Sim fidelity caveat**: cuVSLAM's 30 Hz/±2 ms requirement is awkward under the env-stepped sim
  clock; the depth+nvblox path is more forgiving but still off NVIDIA's Isaac-Sim-only tested path.

## Alternatives considered

- **Package cuVSLAM as an rSkill** — rejected: wrong layer/abstraction (§Context #2); no `RSkillKind`
  fits a continuous localization publisher.
- **cuVSLAM RGB-D fed by AI depth for odometry** — rejected: AI depth's scale/temporal noise
  corrupts visual odometry; AI depth is confined to nvblox mapping.
- **`depthimage_to_laserscan` → Nav2 obstacle layer instead of nvblox** — lighter (no nvblox, no 3D
  TSDF), but loses 3D reconstruction and degrades to a single scan row; kept as a fallback for
  severe VRAM limits, not the default.
- **ORB-SLAM3 / RTAB-Map (open-source visual SLAM)** — avoids the NVIDIA closed binary and EULA, but
  forgoes NVIDIA acceleration/NITROS and the integrated nvblox costmap; worth a spike if the EULA
  guard proves too restrictive, but not the primary path given the existing Isaac ROS/Jazzy fit.
- **Stereo/RealSense-only (no AI depth)** — simplest and most accurate, but excludes the mono-only
  robots that motivate this ADR.
