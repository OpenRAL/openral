# ADR-0010: Inference runner for hardware deployments

- Status: Accepted
- Date: 2026-05-24
- Amended: 2026-05-24 (see Amendments below)

## Context

`OpenRAL` today has every layer of the inference loop *individually*:
`Skill` ABC + `rSkill` loader + `ChunkedExecutor` (in `python/rskill/`),
`HAL` Protocol + working SO-100 adapter (in `python/hal/`),
`WorldStateAggregator` with 30 Hz staleness-latched snapshots (in
`python/world_state/`), and OTel span helpers (`rskill_span`,
`inference_span`, `safety_span`) in `python/observability/`. But
**nothing wires them together for hardware**. The only working
end-to-end loop is `python/sim/src/openral_sim/runner.py:run_episode`,
which drives LIBERO / MetaWorld / gym-aloha / gym-pusht sim adapters and
never touches HAL. The roadmap labels a "Skill executor lifecycle node"
as planned (CLAUDE.md §2 repo map; `docs/architecture/repo-state-map.html`).

The north-star use case — "given an `rSkill` and a task, run on any
robot, any hardware" — needs a runner that closes the loop:

```
SensorReader.read_latest → WorldStateAggregator.snapshot → Skill.step →
SafetyClient.check → HAL.send_action
```

At a configured cadence (default 30 Hz to match the WorldState contract),
in a single process, with OTel correlation across the whole tick.

The runner also has to address camera capture. Reading lerobot's source
(`src/lerobot/cameras/opencv/camera_opencv.py`, `src/lerobot/rollout/
inference/{sync,rtc}.py`, `src/lerobot/scripts/lerobot_record.py`)
confirms lerobot uses **only** per-camera OpenCV background threads —
no GStreamer, no NVDEC, no V4L2 zero-copy, no DMA-BUF. That ceiling is
fine for USB UVC at 30 fps but leaves several workloads OpenRAL
explicitly wants on the table:

- Hardware-accelerated camera decode on Jetson (`nvv4l2decoder`, NVMM
  memory, DMA-BUF zero-copy to CUDA tensors).
- RTSP cameras (`rtspsrc`).
- Synchronized multi-camera capture (PTS pairing, `nvstreammux`).

A pure-GStreamer approach would bypass ROS — but OpenRAL already
depends on ROS 2 for tf2, `/joint_states`, ros2_control, `rosbag2`, and
the `WorldStateAggregator` lifecycle node. A pure-ROS approach would
serialize every frame through `cv_bridge` / `sensor_msgs/Image` and
never reach NVMM. The right answer is hybrid: GStreamer for capture +
decode on the hot path, ROS for everything else, with an optional ROS
publisher branch off the GStreamer pipeline for observability.

## Decision

1. **New `InferenceRunner` Protocol + `InferenceRunnerBase` class in a
   new `python/runner/` (`openral_runner`) workspace member.**
   The base owns the rate-limited loop (`run()`), the OTel parent
   `rskill.tick` span, the `RunResult` / `TickResult` collection, and
   the deadline-overrun policy. Subclasses override `tick()`:

   - `SimRunner` (in `openral_sim`) is a thin shim around the
     existing `run_episode` — no behavior change to `openral sim run`.
   - `HardwareRunner` (in `openral_runner`) wires the
     `SensorReader → WorldStateAggregator → Skill → SafetyClient → HAL`
     chain.

2. **New `SensorReader` Protocol with three backends.** All live in
   `python/runner/src/openral_runner/backends/`:

   - `OpenCVThreadSensorReader` (default) — per-camera `Thread` posting
     to a `latest_frame` slot guarded by `Lock` + `Event`. Mirrors
     `lerobot/cameras/opencv/camera_opencv.py`. Exposes
     `read_latest(max_age_ms)` for non-blocking peek and
     `read_synced(deadline_ns)` for the WorldState aggregator to pull on
     its own clock.
   - `Ros2ImageSensorReader` — subscribes to a ROS 2 image topic
     published by a vendor driver (RealSense / Orbbec ROS nodes,
     ros2_control camera plugins).
   - `GStreamerSensorReader` — pipeline-string from config; appsink
     delivers frames. NVMM / CUDA tensor on Jetson when `nvv4l2decoder`
     is present; CPU `numpy.ndarray` elsewhere. Optional `tee` to a ROS
     publisher when `publish_to_ros=True` so `rosbag2` / `rqt_image_view`
     still see a downsampled stream.

3. **`openral deploy --config R.yaml` CLI as the sibling of `openral sim run`.**
   YAML schema validation up front, license gating via
   `rSkill.from_yaml`, then hand off to the `HardwareRunner`.
   `openral sim run` remains untouched (decided in the plan phase: keep
   `openral sim run`, add `openral deploy`).

4. **Schemas first (this PR).** Add the on-disk contracts to
   `openral_core` so subsequent PRs build against locked types:

   - `RobotEnvironment` — the YAML artefact `openral deploy` loads.
   - `HalConfig`, `SensorReaderConfig`, `SensorReaderBackend`,
     `DeadlineOverrunPolicy` — composed inside `RobotEnvironment`.
   - `SensorFrame`, `FrameEncoding` — runtime carrier passed from
     `SensorReader` into `WorldState.image_frames` and into traces.
     Binary payload JSON-serializes as base64 via a Pydantic field
     serializer so arbitrary bytes round-trip cleanly.
   - `TickResult`, `RunResult` — runner outputs.

5. **Extend `WorldState`** with an optional
   `image_frames: dict[str, SensorFrame] | None` field for no-ROS
   deployments (laptop + USB SO-100). Default `None` preserves the
   existing topic-ref path (`WorldState.images: dict[str, str]`) so all
   existing consumers, the ROS 2 lifecycle node, and the sim runner are
   unchanged.

6. **Promote `ChunkedExecutor`** from
   `python/rskill/src/openral_rskill/smolvla.py:114` to a shared
   `python/rskill/src/openral_rskill/executor.py` (next PR, M2 of the
   inference-runner roadmap). Re-export from `smolvla` for back-compat.
   Bypass it for scripted skills declaring `chunk_size=1`.

7. **Promote `precise_sleep`** to
   `python/runner/src/openral_runner/clock.py`. Mirrors lerobot's
   `precise_sleep` shape (`time.sleep(target - 1ms)` + busy-wait the
   final ~1 ms on `time.perf_counter()`). Used by both Sim and Hardware
   runners.

8. **Safety integration via a `SafetyClient` stub PR.** Per the user
   call in plan-phase Q2, the safety client lands in its own PR before
   M5 (the first end-to-end HardwareRunner). The runner calls
   `SafetyClient.check(action)` inside a `safety_span`; the stub
   returns OK + logs through the existing
   `openral_observability.safety_span` so the trace surface is
   fully wired by the time the real C++ safety kernel arrives.

9. **OTel correlation.** Each tick opens one `rskill.tick` parent span
   (new helper) enclosing child spans `sensors.read`,
   `world_state.snapshot`, `inference_span(name="skill.chunk_inference")`
   (existing), `safety_span` (existing), and `hal.send_action`. The
   `TickResult` is populated from the per-stage timings the spans
   already record.

10. **Hybrid GStreamer ↔ ROS, explicit per-sensor.** GStreamer is **not
    in core**; it is one of three optional `SensorReader` backends
    selected per sensor in `SensorReaderConfig.backend`. The Jetson /
    RTSP / multi-camera-sync pipelines are concrete examples in the
    backend doc; nothing in `openral_core` mentions `pygobject` or
    `gst-python`. The optional dependency lives in
    `python/runner/pyproject.toml` `[project.optional-dependencies]`
    `gstreamer = ["pygobject"]` plus a `just bootstrap-jetson` script
    for system-level Gst plugins.

11. **No cloud-dispatcher commentary in this ADR** (per plan-phase Q4).
    The runner takes a `Skill` instance; an `EdgeDispatcher` /
    `SplitDispatcher` decorator on `Skill` is a future, separate ADR.

12. **No multi-skill orchestration in this ADR.** A future S2 reasoner
    will wrap `InferenceRunner` and swap the inner `Skill` on
    plan-transition. The Protocol seam is preserved by accepting
    `Skill` as a constructor argument; the runner is single-skill
    today.

## Consequences

- **Pros**
  - One Protocol — `InferenceRunner` — covers sim and hardware. The
    same outer code can target LIBERO or an SO-100 by swapping the YAML.
  - `openral_runner` is plain Python; no ROS install required for
    laptop + USB deployments. ROS-native deployments get a thin
    `SkillExecutorNode` lifecycle wrapper in `packages/skill_executor/`
    (later PR) that subscribes to `/world_state` and ticks the runner.
  - The `SensorReader` Protocol gives a single seam for three very
    different capture stacks (OpenCV / ROS / GStreamer). Picking
    GStreamer for a Jetson + multi-cam workload is a per-sensor config
    flip, not a code path bifurcation.
  - The hybrid GStreamer + ROS-tee design preserves `rosbag2` /
    `rqt_image_view` / `image_transport_plugins` for off-line
    observability without putting them on the hot path.
  - One OTel parent span per tick gives end-to-end timing in `ral
    replay` without re-instrumenting.

- **Cons**
  - New top-level workspace member `python/runner/`. Cost: one
    `pyproject.toml`, root `pyproject.toml` workspace entry, mkdocs
    nav, METHODS.md section, repo-state-map blocks.
  - New responsibility composing six layers (S1, HAL, Sensors,
    WorldState, Safety, Observability). Per CLAUDE.md §6.1 that needs
    an ADR — which is this file.
  - `WorldState.image_frames` adds a second carry-mode for frames
    (topic ref vs inline `SensorFrame`). Mitigated by making it
    `Optional` with default `None`; ROS-only deployments see no change.
  - GStreamer brings a heavy system dep (`gstreamer1.0-plugins-*`, on
    Jetson `nvidia-l4t-gstreamer`). Mitigated by gating it as an
    optional backend with `bootstrap-jetson` helper.
  - `openral_core` gains six models + three enums; bumps
    `0.3.0 → 0.4.0` (additive minor under CLAUDE.md §1.6 pre-1.0
    rules).

## Migration

Phased — one PR per phase. This ADR is **PR A** and ships in the same
PR as the schemas.

- **PR A (this PR)**: ADR-0010 + the schema additions
  (`RobotEnvironment`, `HalConfig`, `SensorReaderConfig`,
  `SensorReaderBackend`, `DeadlineOverrunPolicy`, `SensorFrame`,
  `FrameEncoding`, `TickResult`, `RunResult`) + `WorldState.image_frames`
  extension. Hypothesis fuzz tests, JSON Schema export, repo-state-map
  blocks, METHODS.md entries. Bumps `openral-core 0.3.0 → 0.4.0`.
  No runtime code yet.
- **PR B**: Promote `ChunkedExecutor` to
  `openral_rskill.executor` and `precise_sleep` to
  `openral_runner.clock`. Pure refactor; existing SmolVLA tests
  continue to pass.
- **PR C**: `InferenceRunnerBase` + `InferenceRunner` Protocol;
  rate-limited `run()` skeleton + `rskill.tick` parent span. Dummy
  in-process HAL + `HelloSkill` 100-tick cadence test (±2 ms tolerance).
- **PR D**: `SensorReader` Protocol + `OpenCVThreadSensorReader`. v4l2
  loopback test pattern; assert frame age + FPS.
- **PR E**: `SafetyClient` stub. `safety_span`-wired Protocol that
  returns OK and records action metadata. Standalone PR per plan-phase
  Q2.
- **PR F**: `HardwareRunner` end-to-end against `SO100DigitalTwin` HAL
  + a real `hello-skill` rSkill. No ROS.
  `tests/integration/test_inference_runner_so100_digital_twin.py`.
- **PR G**: `openral deploy --config` CLI entry.
- **PR H**: `SkillExecutorNode` ROS 2 lifecycle wrapper in
  `packages/skill_executor/`. `launch_testing` integration test.
- **PR I**: `GStreamerSensorReader`. CPU `videotestsrc` test; Jetson
  `nvv4l2decoder` test gated by `[self-hosted, lab-jetson]`.
- **PR J**: First HIL — SO-100 + SmolVLA + RealSense via GStreamer
  reader. Pick-cube smoke test gated by `[self-hosted, lab-so100]`.

## Why not other options

- **ROS 2 lifecycle node first (single deployment shape).** Forces
  every dev to `source install/setup.bash` to iterate on the skill
  loop. Heavier dev loop; harder to demo `openral deploy` on a laptop with
  a USB arm. Mirrors `WorldStateAggregator` core/wrapper split.
- **Pure GStreamer for camera capture.** Frames don't appear on a ROS
  topic so `rosbag2` / `rqt_image_view` / `image_transport_plugins`
  silently break. We do not want to replace tf2 / `/joint_states` /
  ros2_control — the rest of the robot stays ROS-native; only the
  capture hot path goes through GStreamer. Hybrid wins.
- **Pure ROS image_transport for camera capture.** No NVMM → CUDA
  zero-copy on Jetson; `h264` plugin decodes back to CPU
  `sensor_msgs/Image` — wrong direction for a Jetson-deployed VLA.
  Works for slower / non-Jetson setups; covered by
  `Ros2ImageSensorReader` (one of the three backends) when applicable.
- **`InferenceRunner` extends an existing class instead of a new
  Protocol.** The sim runner is a free function (`run_episode`) and
  `Skill` is the policy ABC, not a runner. Reusing either would
  conflate "the thing that runs" with "the thing being run". A new
  Protocol keeps the contract narrow.
- **Carry pixel data inside `WorldState.images` directly (replace
  `dict[str, str]` with `dict[str, SensorFrame]`).** Breaking change
  for every ROS-topic-ref consumer (the WorldState ROS 2 lifecycle
  node, every skill adapter that reads frames off topics). Rejected;
  adding `WorldState.image_frames: dict[str, SensorFrame] | None` is
  additive.
- **Make `SensorFrame.data` use `pydantic.Base64Bytes`.** Forces every
  caller to base64-encode before construction (the Pydantic
  `Base64Bytes` type expects already-encoded input). A plain `bytes`
  field with `field_serializer(when_used="json")` for base64 on dump
  and a `field_validator(mode="before")` to decode on load is the
  ergonomic constructor + transparent round-trip pair. Chosen.
- **Bundle the `ChunkedExecutor` promotion and the
  `InferenceRunnerBase` extraction into one PR.** Violates CLAUDE.md
  §7.2 "smallest viable PR". The promotion is purely a refactor with
  zero behavior change; the base class introduces a new abstraction.
  Splitting keeps reverts cheap.

---

## Amendments

### 2026-05-12 — Sensor-ingest backend evaluation (M8 / PR I)

When milestone I (`GStreamerSensorReader`) opened we re-evaluated the
choice of ingest backbone against two NVIDIA alternatives that had
matured since the original ADR landed: **NVIDIA Holoscan SDK** (Apache-2.0
operator-graph framework, official forward-bet for Thor / Spark) and
**DeepStream SDK** (proprietary, GStreamer-based, ships nvinfer / nvtracker
as plugins). Reframing question: does the agentic harness's "swap rSkill
at runtime" requirement disqualify lean GStreamer?

| Criterion | Lean GStreamer | NVIDIA Holoscan | NVIDIA DeepStream |
|---|---|---|---|
| License | LGPL plugins + Apache-2.0 friendly | Apache-2.0 ✓ | Proprietary — disqualified by §1.9 / §12 |
| x86 + RTX local dev | ✅ today | ✅ (CUDA 12 / 13 wheels via `pip install holoscan-cu12`) | ⚠️ x86 license-restricted for production |
| Jetson Orin Nano | ✅ no known issues | ⚠️ V4L2 bug + reported CUDA kernel mismatch | ✅ |
| Sensor sources we need (V4L2 / CSI / RTSP / H.264 / NVMM) | ✅ all stock | ⚠️ V4L2 + H.264 ✓, **no RTSP, no Argus / MIPI CSI in core** ([HoloHub](https://nvidia-holoscan.github.io/holohub/operators/)) | ✅ all there |
| ROS 2 integration | ✅ `gscam2` (mature image_transport) | ⚠️ `holoscan_ros2` is HoloHub-grade, no CI signals | ✅ |
| NVMM/CUDA zero-copy to PyTorch | ctypes `NvBufSurface` (Apache-2.0 binding, port-once) | DLPack-native (cleaner) | NvBufSurface, same as GStreamer |
| Pipeline mutation at runtime | pad probes / tee branches / full rebuild | `set_dynamic_flows()` **only routes between pre-instantiated operators** ([docs](https://docs.nvidia.com/holoscan/sdk-user-guide/holoscan_dynamic_flow_control.html)) — restart cost matches GStreamer for topology change | same as GStreamer |
| "Avoid custom ops" feasibility | One ported ctypes module shared across architectures | Would write new operators for RTSP + Argus + ROS bridge | None — but locked into proprietary |
| Reuse of existing reference code | ~10–20% adaptation | ~50–70% rewrite (DLPack tensors / GXF operators) | partial |
| Container footprint | ~50–200 MB plugins + ~8 MB pygobject | ~1–2 GB SDK + CUDA | ~1–2 GB |

**Dynamic-graph reframe.** Holoscan's `set_dynamic_flows()` is more
limited than the marketing implies: it cannot add or remove operators
after `compose()`. To swap a sensor topology (e.g. activate a depth
camera a new skill demands) both backends require a sub-second cold
pipeline rebuild. The right pattern on either backbone is therefore
**static superset pipeline + Python-side skill gating**: instantiate
every sensor any installed rSkill could need at startup, let frames
flow to all latched slots, and the rSkill-of-the-moment reads the
subset its capabilities advertise. No runtime graph mutation; no
GPU memory churn; no GStreamer pad-probe footgun.

**Decision (amendment).** Stay with **lean GStreamer + custom
NvBufSurface glue**. Counter-intuitively this best serves the
"avoid custom ops" goal because Holoscan would force us to write the
RTSP, Argus, and ROS-bridge operators that GStreamer already ships
stock. DeepStream is left as a free downstream compatibility property:
a user can prepend `nvinfer config-file-path=...` to any of our
pipeline strings without us changing code (both produce / consume
NvBufSurface over `memory:NVMM` caps).

**Seam preserved.** `SensorReaderBackend.HOLOSCAN` is added to the
enum reserved-but-unimplemented (commit #5 of PR I). When Thor HIL
becomes routine and Holoscan's Argus / ROS-bridge operators land in
core (or HoloHub matures CI), we can register a `holoscan` backend
additively without bumping the schema.

**Local-dev gotcha worth noting.** PyGObject installed by apt
(`python3-gi`) shares its GLib link with torch's bundled GLib on
JetPack / on the Docker images we ship in this PR. The dev-host
shortcut of symlinking system gi into a venv that also imports torch
(via numpy / pyarrow / friends) segfaults; `just test` therefore runs
the GStreamer test files in a separate pytest invocation. Inside the
inference Docker images (`docker/inference/Dockerfile.{x86,l4t}`) the
conflict disappears because pygobject and torch link to a single GLib
managed by apt — a single `pytest` invocation works there.

### 2026-05-12 — Implementation milestones for PR I

* PR I/1 — Pipeline builder + platform detect (pure Python, no `gi`).
* PR I/2 — `GStreamerSensorReader` CPU appsink path + factory wiring + first end-to-end via `openral deploy`.
* PR I/3 — NvBufSurface ctypes + shared CUDA context glue (NVMM zero-copy on Tegra). See [ADR-0011](0011-nvmm-handoff.md) for the cross-layer design.
* PR I/4 — ROS 2 tee branch. Landed in commit `edfe8ba`: `ros_tee.py`
  (296 LOC) + 234 LOC of unit tests, wired through
  `factory.py` (`SensorReaderConfig.publish_to_ros` →
  `PipelineSpec.enable_ros_tee`). The live publish/subscribe
  round-trip is validated end-to-end inside the x86-ros Docker image
  (commit `d9a87f3`) — `just docker-smoke-x86-ros` opens a real
  `videotestsrc` pipeline, the reader publishes
  `sensor_msgs/Image`, a real `rclpy` subscriber receives a
  160×120 / bgr8 / 57600-byte frame, exit 0. Spark / lab HIL with
  rosbag2 capture still pending.
* PR I/5 — `SensorReaderBackend.HOLOSCAN` enum reserved.
* PR I/6 — Documentation: this amendment + ADR-0011 + repo-state-map flip.
* PR I/7 — Inference deploy Dockerfiles (`docker/inference/Dockerfile.{x86,l4t}`).

### 2026-05-13 — ROS-enabled Docker, "all on GPU" end-to-end, and a real glass MJCF

The original PR I package (`I/1`–`I/7`) shipped the GStreamer ingest
backend, the ROS-tee branch, and the deploy Dockerfiles, but several
load-bearing demonstrations were short of the end-to-end claim the
ADR makes (sensor → GPU → rSkill → ROS). The 2026-05-13 follow-ups
close those gaps:

* PR I/8 — **ROS-enabled inference Docker image + live ROS-tee
  validation.** New `docker/inference/Dockerfile.x86-ros` adds
  ros-jazzy + sensor_msgs + rclpy + cyclonedds on top of the x86
  image; `entrypoint-ros.sh` sources the ROS setup; `smoke_ros_tee.py`
  drives `openral deploy` with `publish_to_ros=true` and runs a real rclpy
  subscriber in the same container. Three platform-level surprises
  surfaced and were root-caused inside this commit:
  (a) `pydantic v2`'s Rust core + `gst-cuda` plugin scan +
  Fast DDS' SHM transport segfault `rclpy.Node()`; the image now pins
  `RMW_IMPLEMENTATION=rmw_cyclonedds_cpp`.
  (b) `cv2` import — pulled in eagerly from
  `openral_runner.backends.OpenCVThreadSensorReader` —
  initialises its own glib state that segfaults a later `rclpy.Node()`.
  The `openral_runner.backends.__init__` and
  `openral_runner.__init__` modules now lazy-load via PEP 562
  `__getattr__` so the gstreamer subpackage does not eagerly drag in
  cv2 / torch / hal.
  (c) `Gst.init()` must run **before** any `import rclpy`. The reader
  module now calls `Gst.init()` at module load (not in `open()`), and
  `open()` is reordered so the ROS publisher is created BEFORE the
  pipeline transitions to PLAYING.
  Validated: `just docker-smoke-x86-ros` exits 0, real
  `sensor_msgs/Image` received (160×120 bgr8 / 57600 bytes / `frame_id=cam0`).

* PR I/9 — **True no-CPU-decoded-frame H.264 GStreamer pipeline.**
  The webcam YAML's `cudaupload` seam already moved the frame onto the
  GPU at ingress, but the **decoded raw frame** still lived on the CPU
  briefly. New `deployments/so100_hello_gstreamer_h264_gpu_decode.yaml`
  drives `filesrc → qtdemux → h264parse → nvh264dec` so the decoded
  raw frame is born on the GPU in
  `video/x-raw(memory:CUDAMemory),format=NV12` — only the encoded
  bitstream bytes (~10 KB/frame) and the final BGR frame after the
  single `cudadownload` at the inference seam touch host memory. This
  is as close to "no CPU copy at all" as the current `SensorFrame.data: bytes`
  contract allows; closing the last `cudadownload` is the
  [ADR-0011](0011-nvmm-handoff.md) follow-up. A 59 KB test asset
  (`examples/assets/test_h264_ball_6s.mp4`) is checked in, generated
  itself end-to-end on the GPU via `nvh264enc`.

  The webcam YAML (`so100_hello_gstreamer_v4l2_camera.yaml`) is
  reverted to the `jpegdec → cudaupload` path with a prominent
  header note explaining why: the Lenovo Integrated Camera (and most
  laptop UVC webcams) emit MJPG with 4:2:2 chroma subsampling
  (`yuvj422p`); NVIDIA's `nvjpegdec` only handles 4:2:0 and the
  YUYV alternative is also 4:2:2 packed. This is a hardware limit of
  the specific webcam, not the pipeline.

* PR I/10 — **`GpuPassthroughSkill` — an rSkill that provably runs on
  GPU.** New `python/rskill/src/openral_rskill/gpu_passthrough.py`
  is a weight-less Skill whose `_step_impl` uploads each
  `SensorFrame` to `torch.cuda`, runs a per-channel mean reduction
  with explicit `torch.cuda.synchronize` (so the OTel latency span
  covers the actual GPU compute, not just dispatch), and emits the
  reduced means as the Action's `confidence`. `configure()` RAISES
  if `device='cuda'` is requested and `torch.cuda.is_available()` is
  False — no silent CPU fallback. Wired into `SKILL_REGISTRY` as
  `gpu_passthrough`. The `so100_gpu_passthrough_h264.yaml` config
  combines this skill with the PR I/9 H.264 pipeline for the literal
  "decoded raw frame never on CPU + rSkill all on GPU" demonstration:
  p99 inference 3.85 ms (vs HelloSkill's 0.171 ms — the delta IS the
  cudaMemcpy + kernel + synchronize cost).

* PR I/11 — **Custom MJCF for "robot picks up a glass" (M8 sim
  demo).** Adds the openral-owned asset
  `python/sim/src/openral_sim/{policies,backends}/_assets/pickup_glass/bimanual_viperx_pickup_glass.xml`
  — derived from gym-aloha's `bimanual_viperx_transfer_cube.xml` but
  with the 4 cm red cube replaced by a 4 cm × 8 cm translucent
  pale-blue cylinder, 30 g mass (vs 50 g), μ=0.4 (vs 1.0), and a
  recomputed diagonal inertia for a solid cylinder. The new scene
  adapter (`openral_sim.{policies,backends}.pickup_glass`) stages our XML
  alongside symlinks to gym-aloha's `scene.xml`,
  `vx300s_dependencies.xml`, the per-arm XMLs and STL meshes, loads
  the staged XML via `mujoco.Physics.from_xml_path`, and drives it
  with `dm_control.rl.control.Environment` + gym-aloha's
  `TransferCubeTask` (the geom name `red_box` and body name `box`
  are preserved so the upstream reward / contact-pair logic is
  reused verbatim — only physics and visuals are OpenRAL's).
  Registered as the new scene id `aloha_pickup_glass`. ACT
  (cube-trained) success drops from 1/1 (cube scene) to 0/1
  (glass scene) — out-of-distribution failure that confirms the new
  physics is being applied rather than the old cube path silently
  being reused.

* PR I/12 — **End-to-end demo tutorial.**
  `docs/tutorials/laptop_camera_glass_demo.md` walks through all of
  the above with the validated commands and outcomes. The closing
  "What this demo proves and what it doesn't" table is the canonical
  honest-claim ledger for the M8 demo surface (4:2:2 chroma
  constraint on laptop UVC; H.264 source for no-CPU-decoded-frame;
  cube-trained ACT failing the glass).

### 2026-05-13 — Local-dev gotcha update (cv2 / torch / glib ordering)

The earlier amendment noted that the dev-host pygobject ↔ torch GLib
conflict forced `just test` to split GStreamer tests into a separate
pytest invocation. PR I/8 surfaced a related but distinct ordering
constraint that applies even inside the x86-ros Docker image where
the GLib link is consistent: **`cv2` and `torch`, when imported,
initialise glib state that segfaults a later `rclpy.Node()` if
`Gst.init()` has not yet run.** The reader module now calls
`Gst.init()` at import; the `openral_runner.{__init__,
backends.__init__}` modules lazy-load symbols via PEP 562
`__getattr__` so importing the gstreamer subpackage does NOT eagerly
import cv2 or torch. Inside the ROS image, `RMW_IMPLEMENTATION` is
also pinned to `rmw_cyclonedds_cpp` for the same reason — Fast DDS'
SHM transport adds a third ABI to the conflict that cyclone sidesteps.

### 2026-05-14 — Custom-scene adapter consolidation

The **PR I/11 "robot picks up a glass" custom MJCF scene** (referenced
above as `aloha_pickup_glass`, registered by
`python/sim/src/openral_sim/{policies,backends}/pickup_glass.py`) has been
**removed** as part of consolidating the OpenRAL custom-scene
authoring story onto a single adapter.

The near-identical salad-dressing / bbq-sauce examples were also removed as
replications of the same target-swap concept.

What was removed:

* `python/sim/src/openral_sim/{policies,backends}/pickup_glass.py` and its
  `_assets/pickup_glass/` directory.
* `scenes/benchmarks/act_aloha_pickup_glass.yaml`.
* `docs/tutorials/laptop_camera_glass_demo.md` (the M8 PR I/8-9
  end-to-end demo whose Section 3 wrapped the glass-pickup scene).

What was kept (still works unchanged):

* The Docker images, `GStreamerSensorReader`, `nvh264dec` /
  `GpuPassthroughSkill` deploy path, and every other M8 PR I/8-10
  deliverable. Only the sim-side custom scene that wrapped them into
  one tutorial was retired.
* The upstream `gym_aloha` benchmark scenes (`aloha_transfer_cube`,
  `aloha_insertion`) and ACT policy adapter are untouched — users
  who need ACT + bimanual ViperX in sim can still run those
  unchanged.

The original Decision text above is preserved as the historical
record of why pickup_glass was added in the first place.

### 2026-05-14 — Cross-platform support contract (cross-reference to ADR-0016)

The `Platform` enum (`TEGRA / NVIDIA_DESKTOP / CPU_ONLY`) and the
`detect_platform()` probe introduced in PR I (the 2026-05-12
sensor-ingest backend evaluation amendment above) are the first
piece of explicit multi-platform support in the repo,
but they cover only the sensor-ingest layer. The broader question —
*"what does it take to guarantee that all code in the repo runs
correctly on x86 (CUDA + CPU) and L4T (Orin + older Jetson)?"* — is
addressed by [ADR-0016 (Multi-platform support)](0016-multi-platform-support.md),
which closes [issue #89](https://github.com/OpenRAL/openral/issues/89).

ADR-0016 pins the canonical image set to the **same two Dockerfiles**
this ADR introduced (`docker/inference/Dockerfile.x86`,
`docker/inference/Dockerfile.l4t`) and explicitly endorses the
`Platform` enum as **sufficient and final** for the supported targets
— a future contributor must not split `TEGRA` into per-board enum
values. SoC-level distinctions belong in
`JetsonInfo.cuda_compute_capability`, not in `Platform`.

ADR-0016 also makes the deferral of Holoscan permanent (carried
forward from this ADR's 2026-05-12 amendment) and routes new
detection / quantization / CI follow-ups (`RobotCapabilities.nvmm_available`,
explicit Xavier vs Nano branches in `_probe_jetson`,
`auto_select_quant` pin-tests, `[self-hosted, l4t]` runner pool,
`linux/arm64` Buildx matrix) to two follow-up PRs that cite ADR-0016
as their authority.

This amendment is purely a cross-reference; no decision in this ADR
is reversed.

### 2026-05-14 — DeepStream EULA findings + opt-in container path

The 2026-05-12 amendment above rejected NVIDIA DeepStream on
license grounds with a one-line summary in the criteria table
(*"Proprietary — disqualified by §1.9 / §12"*). During the
ADR-0016 PR 3/3 verification we discovered a latent regression
this prior decision missed: the M8 GStreamer pipeline builder
hardcoded `nvvideoconvert` (a DeepStream-only element) on the
`Platform.NVIDIA_DESKTOP` branch and claimed `memory:NVMM` caps
on the same branch, even though neither is available in the
open-source `gstreamer1.0-plugins-bad` `nvcodec` family.

The bug was latent because every committed smoke test uses an
explicit `pipeline:` YAML string instead of the auto-builder.
Reproduced inside `openral:x86-latest`:

```
$ python -c "from gi.repository import Gst; Gst.init(None);
  from openral_runner.backends.gstreamer import ...;
  Gst.parse_launch(build_pipeline_string(rtsp_spec, NVIDIA_DESKTOP))"
parse_launch FAILED: Error gst_parse_error: no element "nvvideoconvert"
```

The fix is straightforward — `_build_convert` returns
`videoconvert` on `Platform.NVIDIA_DESKTOP`, and `_build_caps`
restricts NVMM to `Platform.TEGRA`. H.264 / H.265 / JPEG / AV1
dec/enc still run on the GPU via `nvh264dec` / `nvh264enc` etc.;
only the colour-space convert step runs on the CPU. Net cost at
30fps 1080p is negligible because data is already in system
memory on x86 (no NVMM caps reachable without DeepStream).

A user who genuinely needs `nvvideoconvert` on x86 still has a
path: the new opt-in
`docker/inference/Dockerfile.x86-deepstream` image, modelled on
NVIDIA's own [deepstream_dockers](https://github.com/NVIDIA-AI-IOT/deepstream_dockers)
pattern. It expects the user to download the SDK tarball locally
(active EULA acceptance) and lives outside the GHCR push matrix
because §2.c forbids relicensing the resulting bundled image as
Apache-2.0. The user can then patch `_build_convert` /
`_build_caps` downstream to re-enable the NVMM path; that patch
must NOT be merged upstream.

**Granular EULA findings** (NVIDIA DeepStream License, v.
February 15, 2022, retrieved 2026-05-14 from
[developer.nvidia.com](https://developer.nvidia.com/downloads/deepstream-eula-ngc)):

| Clause | Plain-English impact |
|---|---|
| §1.c | Distribution of derived CONTAINERs bundling DeepStream + "other primary functionality" is permitted. OpenRAL counts as primary functionality. |
| §2.a | App must have material additional functionality beyond DeepStream. Met — OpenRAL is a robot agent harness, not a DeepStream wrapper. |
| §2.b | Modified source must carry `"This software contains source code provided by NVIDIA Corporation"`. |
| §2.c | **Downstream distribution must be under terms at least as protective as NVIDIA's.** The bundled image cannot be Apache-2.0; it's a mixed license. Public-registry push is therefore forbidden. |
| §4.c | No stand-alone DeepStream redistribution. |
| **§4.g** | **No benchmark or competitive-analysis publication without prior NVIDIA written permission.** Directly conflicts with OpenRAL's `openral benchmark run` / `eval/*.json` story (CLAUDE.md §6.4). Any `RSkillEvalResult` JSON measuring a DeepStream pipeline cannot ship. |
| **§4.h (second)** | **No use in life-critical applications** (avionics, medical, military, navigation) without a separate NVIDIA agreement. OpenRAL's safety stance (CLAUDE.md §1.1) and many of its target domains (surgical, industrial autonomy) overlap with NVIDIA's exclusion zone. |
| §8 | OSI-approved user-code licenses are explicitly allowed for the *user's app*; Apache-2.0 stays valid for OpenRAL sources. |
| §12 | Total NVIDIA cumulative liability capped at US$10.00. |
| §13 | NVIDIA may terminate the license at will. |

**Decision (sub-amendment).**

1. **Open-core fix**: `_build_convert` / `_build_caps` no longer
   reference `nvvideoconvert` / NVMM on `Platform.NVIDIA_DESKTOP`.
   The unit test that pinned the bad behaviour is inverted to
   assert absence, so a future regression to the broken code
   fails on the next CI run.
2. **Opt-in image**: `docker/inference/Dockerfile.x86-deepstream`
   + Justfile target that refuses to build without the tarball.
   Image is NEVER pushed by `docker-build.yml`.
3. **Documentation**: `docker/inference/README.md` carries the
   full clause-by-clause EULA breakdown for any user considering
   the opt-in path.

The 2026-05-12 amendment's broader "stay with lean GStreamer"
decision is unaffected; this amendment just reconciles the
implementation with that decision and adds an escape hatch.

### 2026-05-14 — Single-Dockerfile consolidation + CUDA-13 / DeepStream-9 alignment

ADR-0016 PR 3/3 (issue #89) shipped four deploy Dockerfiles and two
entrypoints under `docker/inference/`: `Dockerfile.x86` with
`BUILDER_BASE`/`RUNTIME_BASE` build args (CUDA ↔ Ubuntu CPU base),
`Dockerfile.x86-ros` (adds ROS Jazzy + gi splice + custom entrypoint),
`Dockerfile.x86-deepstream` (extends the prebuilt `:x86-latest` with
side-loaded CUDA-13 runtime libs + DeepStream SDK 9.0), and
`Dockerfile.l4t` (JetPack r36.4 / Ubuntu 22.04 / Py 3.10 / aarch64).
Plus `entrypoint-ros.sh` hardcoding `source /opt/ros/jazzy/setup.bash`.

The Ultraplan-reviewed consolidation collapses this into **one
Dockerfile + one entrypoint script**:

- **`docker/inference/Dockerfile.x86`** — Ubuntu 24.04 + Py 3.12 +
  **CUDA 13** + ROS 2 Jazzy + GStreamer 1.24, with a
  `WITH_DEEPSTREAM_STAGE=on|off` build arg gating the optional
  DeepStream SDK install via BuildKit named-stage indirection
  (`FROM ds-${WITH_DEEPSTREAM_STAGE} AS final`).
- **`docker/inference/entrypoint.sh`** — probes `/opt/ros/*/setup.bash`
  and sources the first match, then `exec "$@"`. Replaces the
  hardcoded `entrypoint-ros.sh`. Probe-style so a future arch (Humble
  on L4T, none on a minimal variant) reuses the same script without
  re-implementing the lookup.
- **`docker/inference/deepstream/`** — dedicated directory for the
  EULA-gated SDK tarball, loaded via
  `docker buildx build --build-context ds=...`. Gitignored.

**Dropped surface (out of scope, deliberate):**

- **L4T / Tegra / Jetson Orin** — `Dockerfile.l4t`, `hil-l4t.yml`,
  `docs/contributing/l4t-runner-onboarding.md`. Returns when the
  `[self-hosted, l4t]` runner pool is online and there is user demand.
- **CPU-only variant** — `Dockerfile.x86` no longer accepts a
  `WITH_CUDA=0` swap; the runtime always assumes an NVIDIA
  CUDA-13-capable host (driver ≥ 580).
- **No-ROS variant** — the consolidated image always carries ROS 2
  Jazzy. The `Dockerfile.x86-ros` flavour folds in unconditionally
  because the ROS-tee branch of `openral_runner` (ADR-0010 PR I/4)
  is part of the default deployment now.

**Why CUDA 13 (not 12.6) is the new base:**

PR #93's `Dockerfile.x86-deepstream` (commit `d8afc81`) discovered
that DeepStream 9 needs `libcudart.so.13`, `libnppig.so.13`,
`libnppidei.so.13` — and it solved that by `apt install
cuda-cudart-13-0 libnpp-13-0` *alongside* the CUDA-12.6 base, with
`LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:…` so DeepStream plugins
could dlopen the 13 set while torch kept dlopen-ing its bundled 12.
Bumping the base to `nvidia/cuda:13.0.0-{cudnn-,}runtime-ubuntu24.04`
eliminates the side-load entirely: one CUDA stack, one set of libs
at `/usr/local/cuda/lib64/libcudart.so.13`, ~250 MB saved.

torch 2.10 cu128 wheels (the workspace's current pin) continue to
work on the CUDA-13 base because they bundle their own
`libcudart.so.12` and the NVIDIA driver is forward-compatible
(a driver that can serve CUDA 13 also serves CUDA 12). Verified
locally: `python -c "import torch; print(torch.version.cuda)"`
inside the new image returns `12.8`.

The cost: the host driver minimum moves from "any modern NVIDIA
driver" (CUDA 12.6 needs ≥ 525) to "≥ 580.65" (CUDA 13 minimum).
The README's driver-requirement table documents the failure modes
on older drivers (575-class CUDA 12.9-class drivers: image still
runs non-CUDA pipelines, all `nvcodec` plugins fail to register,
`torch.cuda.is_available()` returns `False`).

**Why DeepStream stays gated by a build arg (not a separate file):**

The arg pattern keeps the cache hot — flipping
`WITH_DEEPSTREAM_STAGE=on|off` does not invalidate the apt layer
or the gi splice in the `runtime-base` stage. BuildKit's
`FROM stage-${ARG}` literal-expansion is the BuildKit-blessed
pattern for this (named-stage selection rather than shell `if`).
The tarball is supplied via a separate BuildKit named context
(`--build-context ds=docker/inference/deepstream/`) so it never
inflates non-DS builds — the 1.5 GB tarball stays out of the
default build context entirely when `WITH_DEEPSTREAM_STAGE=off`.

**Why not "one Dockerfile with `TARGET_ARCH` input":**

The literal user request was "one Dockerfile with an input
target arch". I evaluated this and rejected it: the L4T base
(JetPack r36.4 / Ubuntu 22.04 / Py 3.10) and the x86 base
(`nvidia/cuda:13.0.0` / Ubuntu 24.04 / Py 3.12) differ on the
*base image* — which Dockerfile must pick *before* any RUN step
executes. Parameterizing via `ARG BASE_IMAGE` is doable, but
every downstream RUN must then branch on a second
`TARGET_ARCH` arg (Ubuntu version, Python version, apt repo set,
gi-splice path, ROS distro: Jazzy on 24.04 vs Humble on 22.04
— different package names). The result is a shell ladder in
every RUN, which breaks BuildKit's per-layer cache hashing and
hides bugs. Industry pattern (NVIDIA `deepstream_dockers`,
PyTorch's `pytorch/pytorch` vs `dustynv/l4t-pytorch`, HF
Transformers' accelerator matrix) is one Dockerfile per arch
family.

The consolidation here is for the *single supported arch* — what
the user asked for, just scoped down. When the L4T variant returns,
it returns as a peer `Dockerfile.l4t` (same internal shape:
`runtime-base` → `ds-off`/`ds-on` → `final`) and the shared
`entrypoint.sh` survives unchanged.

**Tag continuity / deprecations:**

The default image tag `openral:x86-latest` continues to exist
and is logically equivalent to PR #93's `:x86-ros-latest` (ROS is
now unconditional). Tags removed:

- `openral:x86-cpu-latest` — gone; the CPU-only variant is out
  of scope.
- `openral:x86-ros-latest` — gone; folded into `:x86-latest`.
- `openral:l4t-latest` — gone; L4T variant returns in a future PR.

CI matrix collapses from three rows (`x86`, `x86-cpu`, `l4t`) to one
(`x86`). The DeepStream variant stays out of CI (EULA §2.c).

**Risks accepted:**

- Older-driver hosts (CUDA-12.x-class drivers) lose GPU-accelerated
  paths in this image. The smoke still passes on a 575-class driver
  (videotestsrc → videoconvert → appsink runs without CUDA), but
  `nvh264dec` etc. won't register and `torch.cuda` is unavailable.
  Documented prominently in `docker/inference/README.md` with a
  driver-by-driver compatibility table.
- The L4T deletion is irreversible without a future PR. The
  Dockerfile lives in git history (`git show
  worktree-issue-89-pr3-ci-docker:docker/inference/Dockerfile.l4t`)
  if a future contributor wants to lift it.
- PR #93's commits tagged "ADR-0016 PR 3/3" reference an ADR-0016
  that doesn't exist as a separate file (the cross-platform work
  landed as the 2026-05-14 amendment here). The tag is left dangling;
  the consolidation's amendment supersedes the DS-specific decisions
  without renaming the prior commits.

### 2026-05-16 — Full sim unification: SimRunner adopts per-step ticks

**Amendment 1.** The original Decision text proposed `SimRunner` as
"a thin shim around the existing `run_episode` — no behavior change
to `openral sim run`". In practice the shim only re-asserted the
Protocol's name; sim and hardware kept different tick semantics
(episode-vs-step) and `run_evaluation` stayed in the call graph.
This amendment closes the unification:

**What changed:**

- **`SimRunner` ticks at one inference step per `tick()`**, matching
  `HardwareRunner` exactly. One `runner.tick()` advances one
  `env.step` (a "step-tick") or, between episodes, one
  `env.reset` + `policy.reset` ("reset-tick").
- **`TickResult` v2** (additive, optional defaults): five new
  fields — `step_idx`, `episode_idx`, `reward`, `terminated`,
  `truncated` — that `SimRunner` populates and `HardwareRunner`
  leaves at `None`. Hardware ticks serialise byte-identically with
  v1 JSON under `model_dump(exclude_none=True)`. `openral_core`
  bumped 0.5.0 → 0.6.0 (minor, non-breaking).
- **`InferenceRunnerBase` gains a `_should_terminate(self) -> bool`
  hook** (default `False`). `SimRunner` overrides it to stop once
  `n_episodes` `EpisodeResult`s have been emitted, so callers pass
  a `max_ticks` ceiling and rely on the hook for the real stop.
  Hardware behaviour unchanged.
- **`run_episode`, `run_evaluation`, and `python/eval-shim/` are
  removed**. `openral sim run` and `openral benchmark run` both now drive
  `SimRunner.activate / run / deactivate`. Episodes are a derived
  view over the tick stream: `SimRunner.episode_results` is the
  list the CLI summary, video writer, and benchmark aggregator
  consume — same `EpisodeResult` shape as before.

**Why:**

The dual driver (Protocol-style `HardwareRunner` + free-function
`run_evaluation`) forced every consumer of either path to know the
difference. With one tick semantic and one driver, the next callers
(notebook UX, fleet shim, planned `ral fleet` multi-robot harness)
inherit the unified Protocol for free.

**Risks accepted:**

- The "thin shim" framing in the original Decision is overruled.
  Anyone reading the Decision text first must follow this
  Amendments section to see that `SimRunner` is now the only
  episode driver in tree.
- `TickResult` consumers that didn't use `exclude_none=True` (none
  in-tree at the time of this amendment) would see `None` values
  for the five new fields on hardware ticks. CLAUDE.md §1.6
  migrator entry is identity (no on-disk artefacts use `TickResult`).

**Why this is still additive enough to not be a new ADR:**

The Decision text (PR C — InferenceRunnerBase + Protocol) is
preserved; the sim adapter (PR ~G in the original PR ladder) is
the only piece this amendment changes, and ADR-0010 was still
Proposed when the amendment landed.

### 2026-05-17 — End-to-end OpenTelemetry telemetry across the tick

**What changed:**

The runner's OTel surface is now full-featured: traces + metrics +
structlog→OTLP log bridge, plus W3C TraceContext propagation helpers
in `openral_observability.propagation` so the ROS-side IDL
`trace_id` fields on `ActionChunk.msg`, `ExecuteRskill.action`, and
`FailureTrigger.msg` can carry a proper `traceparent` value to the
C++ safety kernel (Option B from the OTel design doc — no IDL
break).

What landed on this branch (`claude/add-otel-robot-tracing-Hxf2R`):

- `openral_observability.semconv` — single source of truth for every
  OpenRAL OTel attribute / span / event / metric / label name.
  Legacy prefixes (`rskill.*` / `skill.*` / `inference.*` /
  `safety.*`) are kept verbatim; new layers use `openral.<layer>.*`.
- `openral_observability.metrics` — cached meter instruments for
  every metric in the design doc: tick / inference / HAL /
  sensors / world-state histograms, safety-violation /
  deadline-miss / sensor-stale counters, world-state stale-component
  up-down counter, plus `record_histogram_ms` (drops negatives /
  NaN).
- `openral_observability._sdk` — `MeterProvider` + `OTLPMetricExporter`
  installed alongside the existing tracer + log providers. Reader
  interval is configurable via `OPENRAL_OTEL_METRIC_INTERVAL_MS`
  (default 5 s). No-op fallback when `OTEL_EXPORTER_OTLP_ENDPOINT`
  is unset is preserved.
- `openral_observability.cli_command_span` — root `cli.command` span
  wrapping every `ral` invocation. The top-level
  `openral_cli.main:_root` callback now opens it on the click
  `Context` so it spans the whole subcommand. The sim leaf's
  `configure_observability` + `shutdown_observability` pair was
  removed — the leaf calling shutdown before context teardown was
  silently dropping the `cli.command` export.
- `WorldStateAggregator.snapshot()` — emits a `world_state.snapshot`
  span; events `openral.event.staleness_latched` /
  `openral.event.error_latched` fire only on the first tick a
  component transitions; per-component `openral.world_state.staleness_ms`
  histogram + `openral.world_state.components_stale` up-down counter.
- `InferenceRunnerBase.tick` / `_on_deadline_overrun` — records
  `openral.tick.duration` / `openral.inference.duration` histograms;
  increments `openral.tick.budget_violations` /
  `openral.tick.deadline_misses` counters; fires
  `openral.event.deadline_missed` and `record_exception` +
  `set_status(ERROR)` on the parent span when the policy raises.
- `HardwareRunner._tick_impl` — wraps `HAL.read_state` /
  `HAL.send_action` in dedicated spans + duration histograms;
  catches `ROSSafetyViolation` at the supervisor boundary with
  `record_exception` + `openral.event.safety_violation` +
  `openral.safety.violations{check_name=<exception type>, severity}`
  counter; catches `ROSPerceptionStale` per sensor reader and emits
  `openral.event.sensor_stale` + `openral.sensors.stale_reads`
  counter.
- `TickResult.trace_context` — optional `str` field carrying the
  full W3C `traceparent` for the tick's `rskill.tick` span, for
  offline consumers that can't re-derive it from a closed span.
- `RSkillEvalResult.trace_id` — optional 32-hex pointer from the
  eval JSON back into the OTel trace tree.

**Why:**

The previous decision text already required OTel correlation across
the tick. What it didn't specify was the *signal types* (this
amendment adds metrics + cross-process propagation), the
*semantic-convention namespace* (this amendment locks in
`openral.<layer>.*`), or the *exception → telemetry mapping* (this
amendment lands `record_exception` + counter at every supervisor
boundary). Without those, the OTel surface couldn't be queried
beyond "did this run emit anything?".

**Open questions deferred to the next ADR / amendment:**

1. **Sampler shape.** The CLI today always exports. For hardware
   runs at 100 Hz × 24 h that is ~7 M tick spans/day. A future
   amendment should adopt `ParentBased(TraceIdRatioBased(0.1))` for
   `openral.run.mode == "hardware"` and keep `ALWAYS_ON` for sim /
   benchmark.
2. **IDL major-version rename.** The `string trace_id` field on
   `ActionChunk.msg` / `ExecuteRskill.action` / `FailureTrigger.msg`
   is interim Option B (whole `traceparent` in the same field).
   The Option-A rename (`traceparent` + `tracestate`) is a
   SemVer-major bump on `openral_msgs` and requires a migrator
   entry per CLAUDE.md §1.6 — deferred until the C++ safety kernel
   consumer is wired.
3. **LeRobotDataset linkage.** ~~No dataset writer exists in-tree;
   the `(trace_id, span_id)` columns proposal lives in the design
   doc and waits for the writer ADR.~~ **Resolved 2026-06-09 (issue
   #109).** The ADR-0019 writer lands the per-frame `(trace_id,
   span_id)` columns at write time and `openral replay --frame
   <repo>/<ep>/<frame>` pivots a row back into its trace. See
   [ADR-0019 → Amendments 2026-06-09](0019-rosbag2-lerobot-dataset-bridge.md).

**Why this is still additive enough to not be a new ADR:**

The Decision text (PR C — InferenceRunnerBase + Protocol) is
preserved; this amendment specifies *how* the OTel correlation
required by the original Decision is realised, and references the
on-disk artifacts (`semconv.py`, `metrics.py`, `propagation.py`)
that land alongside the runner.

### 2026-05-18 — Status flipped Proposed → Accepted

The Decision text's PR-by-PR ladder is fully landed:

- `python/runner/` exists as an `openral_runner` workspace member with
  `protocol.py` (`InferenceRunner` Protocol, `SensorReader` Protocol,
  `SafetyClient` Protocol), `base.py` (`InferenceRunnerBase`,
  rate-limited loop, `rskill.tick` OTel span, deadline policy),
  `hardware.py` (`HardwareRunner` wiring
  `SensorReader → WorldStateAggregator → Skill → SafetyClient → HAL`),
  `clock.py`, `factory.py`, `safety.py` (`NullSafetyClient`), and
  `sensor_reader.py` (`GStreamerSensorReader`).
- `SimRunner` lives at `python/sim/src/openral_sim/` and shares the
  same per-step `InferenceRunner` Protocol surface — the unification
  declared in the 2026-05-16 amendment is live.
- Root `pyproject.toml:51` declares `openral-runner = { workspace = true }`
  with the explicit comment "reason: ADR-0010 inference runner".
- CLAUDE.md §2 already marks the runner package with `✓` and references
  this ADR by number.

No behavioural change against the Decision text. Open questions from the
2026-05-17 amendment (sampler shape, IDL major-version rename, dataset
writer linkage) remain deferred to their own ADRs / amendments.

### 2026-06-08 — Three-tier scene paths (ADR-0041)

ADR-0041 split `scenes/` into deploy/sim/benchmark tiers and stripped
rSkill names from filenames. The "what was removed" list above keeps the
historical `scenes/benchmarks/...` paths as a factual record of the
pre-refactor state. Schema, runner contract, and decision text are unchanged.
See ADR-0041 and
[`scenes/README.md`](https://github.com/OpenRAL/openral/blob/master/scenes/README.md) for the tier hierarchy and
per-tier authoring guide.
