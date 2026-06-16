# Feasibility: Porting the OpenRAL Dashboard to Foxglove

> Investigation only — no code changed. Branch: `investigate/foxglove-dashboard-port`.
> Date: 2026-06-16.

## TL;DR

**Verdict: ~60–70% is nearly free; ~30% is genuinely hard or doesn't fit Foxglove at all.**

The decisive fact is *where the data lives*. There are **two parallel data planes** in OpenRAL today:

1. **A live ROS 2 plane** — HAL, sensors, SLAM, and the runner publish real ROS topics
   (`sensor_msgs/Image`, `sensor_msgs/JointState`, `nav_msgs/OccupancyGrid`,
   `sensor_msgs/PointCloud2`, `geometry_msgs/Pose`, `/tf`). **Foxglove eats these natively.**
2. **An OTLP/OpenTelemetry plane** — the current dashboard is **not a ROS subscriber**. Bridge
   nodes (`slam_bridge.py`, `world_cloud_bridge.py`, `sim_sensor_bridge.py`, `producer.py`)
   *re-rasterize* the ROS data into **base64 PNG/JPEG strings inside OTel span attributes**, ship
   them over OTLP/HTTP to a FastAPI server, which fans them out to a vanilla-JS SPA over SSE.
   **Foxglove cannot ingest OTLP** — this entire plane (traces, metrics, system health, the
   reasoner decision card, the safety ledger) has no native home in Foxglove.

So "porting to Foxglove" is really two separate questions with opposite answers:

- **Visualization of robot/world state (images, nav, voxels, joints, TF):** Foxglove does this
  *better* than the current dashboard, and mostly with **zero code** — just point
  `foxglove_bridge` at the existing topics. The current dashboard pre-flattens 3D into 2D PNGs;
  Foxglove would render the *real* 3D.
- **Observability (traces, metrics, telemetry, OTel-only cards):** This is the dashboard's actual
  reason for existing (replay correlation per ADR-0018 F7, Jaeger linkage, OTLP semantic-convention
  system metrics). Foxglove is **not an observability tool** and there is no clean port — you'd be
  rebuilding, not porting.

## What the dashboard shows, and how hard each panel is in Foxglove

Legend: 🟢 native / near-free · 🟡 custom Foxglove extension or schema work · 🔴 doesn't fit Foxglove's model.

| Panel | Underlying live ROS source | Current dashboard path | Foxglove effort |
|---|---|---|---|
| **Camera images** | `sensor_msgs/Image` on `/openral/cameras/<n>/image` | JPEG thumbnail → OTel span | 🟢 **Native** Image panel. Zero code. |
| **2D nav / occupancy map** | `nav_msgs/OccupancyGrid` on `/map` | PNG → OTel span | 🟢 **Native** Map / 2D panel. Zero code. |
| **Robot footprint + pose on map** | `/tf` (`map→base_link`) | computed polygon in span attrs | 🟢 Native via TF + 3D panel (URDF) or 2D overlay. |
| **Nav path / goal** | `nav_msgs/Path`, `geometry_msgs/PoseStamped` | (not surfaced separately today) | 🟢 Native. |
| **Voxels — octomap cloud** | `sensor_msgs/PointCloud2` on `/octomap_point_cloud_centers` | oblique "chase-view" PNG → span | 🟢 **Native** 3D PointCloud — and *better* (real 3D vs fixed-angle raster). |
| **Voxels — `OccupancyVoxels`** | **custom** `openral_msgs/OccupancyVoxels` on `/openral/world_voxels` | (via the cloud above) | 🟡 Custom schema; needs a `.proto`/JSON-schema + a voxel extension, OR keep publishing the PointCloud2 instead. |
| **World collision capsules** | **custom** `openral_msgs/WorldCollision` on `/openral/world_collisions` | not in dashboard | 🟡 Map to `visualization_msgs/MarkerArray` (cylinders) → then native. |
| **Joint states** | `sensor_msgs/JointState` (inside `WorldStateStamped`, and HAL internals) | OTel `hal.read_state` span | 🟢 Native State Transitions / Plot / 3D URDF — *if* a clean `JointState` topic is published. Today it's embedded in a custom msg (see gaps). |
| **Commanded action / next chunk** | **custom** `openral_msgs/ActionChunk` | OTel `hal.send_action` span | 🟡 Custom schema; table via Raw-Message panel, trajectory needs an extension. |
| **World state / diagnostics / battery** | **custom** `openral_msgs/WorldStateStamped` (parallel arrays) | OTel `world_state.snapshot` span | 🟡 Parallel-array layout → needs schema + Diagnostics-style panel logic. |
| **Scene objects (spatial memory)** | reasoner OTel span (`world.scene_objects`) | OTel only | 🟡/🔴 No ROS topic today; would need a `MarkerArray`/`Detection3D` publisher first. |
| **Reasoner tick / tool decision** | reasoner OTel span (`reasoner.tick`) | OTel only | 🔴 OTel-only; no ROS equivalent. Rebuild or custom extension fed by a new topic. |
| **Safety check ledger** | safety-kernel OTel span (`safety.check`) | OTel only | 🔴 OTel-only. |
| **Traces (replay correlator, Jaeger link)** | OTLP spans | OTLP `/v1/traces` | 🔴 **No Foxglove equivalent.** This is OpenTelemetry, not ROS. |
| **Metrics + sparklines** | OTLP metrics | OTLP `/v1/metrics` | 🔴 Foxglove plots ROS topics, not OTLP metrics. |
| **System health (CPU/GPU/RAM)** | OTLP semantic-convention metrics | OTLP `/v1/metrics` | 🔴 Same — OTLP, not ROS. |
| **Event log / structlog** | OTLP logs | OTLP `/v1/logs` | 🔴 Foxglove has a Log panel but only for `rcl_interfaces/Log` ROS topics. |
| **Prompt input box → reasoner** | publishes `/openral/prompt_in/dashboard` | POST → ROS publish | 🟡 Foxglove has a Publish panel + Teleop; custom message means a Service/Publish setup. |
| **E-stop reset button** | `std_srvs/Trigger` service | POST → service call | 🟢/🟡 Native Service-Call panel. ⚠️ see safety note. |

## Effort buckets

**Bucket 1 — Free / hours (just run `foxglove_bridge`):**
Camera images, occupancy map, octomap point cloud, TF/robot pose, footprint, nav path,
joint states (once exposed as a plain topic). This is the visually impressive 3D half and it's
essentially **configuration, not code** — install `ros-${ROS_DISTRO}-foxglove-bridge`, launch it,
build a Foxglove layout, done. Foxglove renders the *real* 3D scene that the current dashboard
can only show as a fixed-angle PNG.

**Bucket 2 — Days (custom message schemas + light panels):**
The 8 custom `openral_msgs` types (`OccupancyVoxels`, `WorldStateStamped`, `ActionChunk`,
`FailureTrigger`, `WorldCollision`, `Tick`, `Episode`, `PromptStamped`). `foxglove_bridge`
auto-advertises ROS schemas, so these *appear* in Foxglove and render in the Raw-Message/table
panel for free — but the rich visuals (voxel grid, trajectory ribbon, diagnostics rollup) need
either (a) a small **Foxglove extension** in TypeScript per type, or (b) a converter node that
re-publishes into standard types (`MarkerArray`, `Detection3DArray`, `DiagnosticArray`). Option
(b) is usually cheaper and keeps Foxglove stock.

**Bucket 3 — Doesn't port (weeks, and arguably shouldn't):**
Everything on the **OTel/OTLP plane** — distributed traces, the replay correlator (ADR-0018 F7),
Jaeger linkage, OTLP metrics, system-health gauges, the event/log stream, and the OTel-only cards
(reasoner tick, safety ledger, scene objects). Foxglove is a robotics *visualization* tool, not an
*observability backend*. There is no "port" here — you would be reimplementing tracing UX that
Foxglove doesn't have a model for. The honest move is to **keep Jaeger/OTLP for traces and metrics**
and let Foxglove own the live 3D/2D scene.

## The structural mismatch that makes "port all of it" the wrong framing

The current dashboard deliberately chose OTLP-over-SSE *instead of* ROS subscription, so that one
loopback HTTP endpoint captures **everything** — sim and real, in-process and cross-node, with
trace IDs threaded through for offline replay. That design is the opposite of Foxglove's, which is
**ROS-topic-first** and stateless about traces.

Consequences:
- The data Foxglove wants (live topics) **already exists upstream** — the bridges prove it, because
  they subscribe to those very topics. So Foxglove doesn't need the OTel plane at all for the visual
  panels; it taps the source directly. **This is what makes Bucket 1 cheap.**
- The data Foxglove *can't* represent (traces/metrics/logs as first-class) is exactly the
  dashboard's differentiator. **This is what makes Bucket 3 a non-port.**

## Recommended shape (if pursued — needs an ADR)

A "replace the dashboard wholesale with Foxglove" goal is not advisable. A **hybrid** is:

1. **Foxglove (via `foxglove_bridge`) owns the live scene** — images, map, point cloud, TF, joints,
   markers. Near-zero effort, better 3D, MCAP recording for free, mobile/tablet, no bespoke JS to
   maintain.
2. **Add 1–2 small converter nodes** to re-publish the custom types worth seeing in 3D
   (`OccupancyVoxels`→`PointCloud2`/voxel markers, `WorldCollision`→`MarkerArray`,
   detections→`Detection3DArray`). Cheaper than per-type Foxglove extensions.
3. **Keep the OTel plane (Jaeger + OTLP) for traces/metrics/health.** Link to it from a Foxglove
   layout note, the same way the current dashboard links to Jaeger.

This crosses the Observability layer (Layer 7) boundary and changes a transport contract, so per
CLAUDE.md §3/§4 it **requires an ADR** before any code. It also must not regress the replay
correlator (ADR-0018 F7) or the loopback-only safety posture (issue #44), and the E-stop-reset
Service-Call panel touches the safety boundary — Foxglove exposing an E-stop *reset* button needs
safety-WG sign-off, not a config tweak.

## Bottom line for the asker

- **"Images, 2D nav, voxels, joint states"** → easy, mostly free, and *better* in Foxglove. The
  topics already exist; you've been flattening them to PNGs unnecessarily.
- **"Traces and telemetry"** → these are OpenTelemetry, not ROS. Foxglove has no native concept for
  them; this part isn't a port, it's a rewrite-or-keep-Jaeger decision.
- **Net:** A high-value, low-cost win is achievable for the *live-scene* half via `foxglove_bridge`
  with almost no code. "Port **all** functionality" is not realistic because a third of the
  dashboard is observability that Foxglove doesn't do. Right framing = **Foxglove for the scene,
  keep OTel/Jaeger for traces** — and gate it behind an ADR.
```
```

## Key file references

- Dashboard server (OTLP receiver + SSE): `python/observability/src/openral_observability/dashboard/{app.py,store.py}`
- Frontend SPA (vanilla JS, no build): `python/observability/src/openral_observability/dashboard/static/index.html`
- Rasterizing bridges (ROS→OTel PNG):
  - `python/runner/src/openral_runner/slam_bridge.py` (`/map` → PNG)
  - `python/runner/src/openral_runner/world_cloud_bridge.py` (`/octomap_point_cloud_centers` → PNG)
  - `python/hal/src/openral_hal/sim_sensor_bridge.py` (`/openral/cameras/<n>/image` → JPEG)
  - `python/observability/src/openral_observability/producer.py` (joint state, action)
- Custom messages (would need Foxglove schemas/converters): `packages/msgs/msg/*.msg`
- No existing Foxglove / foxglove_bridge / MCAP / rosbridge references in the repo.
