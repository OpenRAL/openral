# Inference deploy image

OpenRAL ships **one** open deploy image, and it is **GStreamer-free**. A
consolidation decision ("Single-Dockerfile consolidation + CUDA-13 alignment")
replaced the four-Dockerfile matrix that PR #93 introduced (`Dockerfile.x86`,
`Dockerfile.x86-ros`, `Dockerfile.x86-deepstream`, `Dockerfile.l4t`) with a
single source of truth. The OpenRAL Pro split then moved the entire proprietary
media stack — **GStreamer + the NVMM zero-copy fast path + DeepStream +
TensorRT** — out of this repo. It is now an OpenRAL Pro plugin
(`openral-pro`'s `docker/Dockerfile.pro`, which `FROM`s the image built here and
installs that stack itself).

| Image | Built by | Pushed to GHCR? | License | When to use |
|---|---|---|---|---|
| `openral:x86` | `just docker-build-x86` | ✅ On merge to `master` (`docker-build.yml`) | Apache-2.0 + NVIDIA CUDA runtime EULA | The default open deploy target. x86 with NVIDIA dGPU, host driver ≥ 580. CUDA 13, ROS 2 Jazzy. **No GStreamer / PyGObject / DeepStream / NVMM / TensorRT.** Cameras run through the open `opencv_thread` backend (cv2.VideoCapture over V4L2/USB); the object detector + Robometer reward monitor run as plain ROS-topic consumers; inference is pytorch / onnxruntime. |
| `openral:x86-deepstream-latest` | openral-pro's `Dockerfile.pro` | ❌ **No** | Apache-2.0 **+ NVIDIA DeepStream EULA** | OpenRAL Pro only. `FROM`s `openral:x86` and installs the whole media stack: GStreamer + `nvvideoconvert`, NVMM caps on x86, `nvinfer`, `nvstreammux`, the TensorRT engine runtime. Local / private-registry only — see `openral-pro` for the build flow and the EULA breakdown. |

Scenes that target `openral:x86` must bind their cameras with
`backend: opencv_thread` (not `gstreamer`) — see
[`scenes/deploy/so101_bench.yaml`](../../scenes/deploy/so101_bench.yaml) for the
reference SO-101 workcell. The GStreamer `SensorReader` + perception-tee remain
in-repo as the opt-in `gstreamer` extra (PyGObject), but are **not** in this
image. The L4T / Tegra / Jetson Orin variant and the no-ROS variant from PR #93
are deliberately out of scope here. See
[`docs/decisions.md`](../../docs/decisions.md) for the trade-off rationale.

## Host driver requirements

The image's base is `nvidia/cuda:13.0.0-runtime-ubuntu24.04`. CUDA 13 needs
**host NVIDIA driver ≥ 580.65**. On older drivers the image still imports and
runs (`openral deploy run` works — OpenCV camera capture and the ROS graph are
CPU-side), but `torch.cuda.is_available()` returns `False` and every
CUDA-touching op falls back to (or fails on) CPU.

| Host driver | What works | What fails |
|---|---|---|
| **≥ 580.65** | Everything: torch CUDA, onnxruntime CUDA EP, GPU-resident VLA + reward + detector models | — |
| **570 – 579** (e.g. 575.57 — CUDA 12.9-class) | The ROS graph, OpenCV camera capture, CPU inference. `openral deploy run` comes up. | `torch.cuda.is_available()` returns `False`; GPU-resident models (VLA / Robometer / omdet) OOM or refuse at the VRAM preflight |
| **< 570** | nothing — base image's CUDA 13 stack stops loading entirely | everything |

The `openral doctor` command surfaces the driver version so users see the
mismatch up front.

## What's in the image

- **Base**: `nvidia/cuda:13.0.0-runtime-ubuntu24.04` (Ubuntu 24.04 noble, Py 3.12).
- **NO GStreamer / PyGObject / cairo.** Cameras use the `opencv_thread`
  `SensorReader` (cv2.VideoCapture). `import gi` is absent by design. `libgl1` +
  `libglib2.0-0` are installed only so `import cv2` can dlopen `libGL`/`libgthread`
  in a headless container (plain graphics/threading libs, not gstreamer).
- **ROS 2 Jazzy** (`ros-jazzy-ros-base`, `ros-jazzy-sensor-msgs`,
  `ros-jazzy-rclpy`, `ros-jazzy-rmw-cyclonedds-cpp`) in both build and runtime
  stages, plus moveit / nav2 / slam-toolbox / octomap for the wrapped-ROS
  rSkills and lidar/mobile robots. The cyclonedds rmw is preferred over Fast DDS
  because Fast DDS' SHM transport interacts badly with pydantic v2's Rust core.
- **uv-managed workspace venv** at `/workspace/.venv` with the OpenRAL Python
  packages plus:
  - the `sim` group — the lerobot / transformers / accelerate / bitsandbytes
    stack the VLA policy adapters AND the in-process Robometer reward monitor
    import at load time (needed on real hardware too);
  - the `omdet` group — the open-vocabulary object detector (omdet-turbo,
    transformers-based) the `ros_image_detector_node` runs;
  - `opencv-python` + `onnxruntime`, installed explicitly post-sync (opencv is a
    per-member extra not pulled by `uv sync --all-packages`; onnxruntime lives
    only in the `dev` group and backs the detector's CPU-ONNX / RT-DETR tier and
    `runtime: onnx` rSkills);
  - `feetech-servo-sdk` + `deepdiff` — the SO-100 / SO-101 real servo bus,
    pulled automatically via openral-hal's `lerobot[feetech]` base dependency.

  `uv sync` runs **without** the `gstreamer`/PyGObject extra and without the
  `tensorrt` group (an OpenRAL Pro plugin, layered on by `Dockerfile.pro`).
- **colcon `install/` overlay** at `/workspace/install/` (baked by the builder
  stage), mirroring `just ros2-build`:
  - `openral_msgs` — the action + message IDL the graph consumes
  - `opentelemetry_cpp_vendor` — builds opentelemetry-cpp from source; the safety
    kernel links against it
  - `openral_safety_kernel` — C++ deny-by-default safety process (binary at
    `/workspace/install/lib/openral_safety_kernel/safety_kernel_node`)
  - `openral_hal_so100`, `openral_hal_openarm` — HAL lifecycle nodes
  - `openral_world_state` — 30 Hz world-state snapshot node
  - `openral_reasoner_ros` — LLM tool dispatch
  - `openral_prompt_router` — prompt fan-in
  - `openral_safety`, `openral_safety_watchdog` — safety envelope + deadman watchdog
  - `openral_human_estop` — human e-stop forwarder
  - `openral_foxglove_bringup` — read-only allowlists imported by the rskill launch
  - `openral_rskill_ros` — the `ExecuteRskill` action server
  - `openral_octomap_bridge` — octree → world-voxels bridge
  - `openral_perception_ros` — the **gstreamer-free** ROS-Image object detector +
    reward-monitor nodes (both subscribe to `/openral/cameras/<name>/image`)

  The non-ROS trees the launch resolves from its `_REPO_ROOT`
  (`/workspace/install`) — `tools/`, `rskills/`, `scenes/`, `.venv` — are COPY'd
  to `/workspace` and symlinked under `install/`, so
  `deploy run --config scenes/deploy/<workcell>.yaml` needs no host bind-mount.
  `git` is installed in the runtime stage for the Robometer sidecar's first-use
  venv provisioning. `Python3_EXECUTABLE=/workspace/.venv/bin/python` is baked
  into every ament-python package so the lifecycle nodes resolve `structlog`,
  `openral_*`, and the OTel SDK through the venv.
- **`/entrypoint.sh`** probes `/opt/ros/*/setup.bash` AND
  `/workspace/install/setup.bash`, sources both, and exec's the user command.
- ENV: `ROS_DISTRO=jazzy`, `RMW_IMPLEMENTATION=rmw_cyclonedds_cpp`,
  `ROS_DOMAIN_ID=0`, `PATH=/workspace/.venv/bin:$PATH`, `PYTHONUNBUFFERED=1`.
- Default `ENTRYPOINT`: `openral deploy run` (the CLI ships as `openral`).

## The GStreamer / DeepStream / TensorRT stack moved to OpenRAL Pro

GStreamer's NVMM zero-copy fast path, DeepStream, and TensorRT are the
proprietary, EULA-restricted media/inference stack. A 2026-05-12 decision
(refined 2026-05-14) rejected bundling DeepStream into the default image, and a
2026-07-08 decision moved the whole opt-in variant into the private `openral-pro`
repo. `openral-pro`'s `docker/Dockerfile.pro` `FROM`s the gstreamer-free
`openral:x86` and installs GStreamer + `nvvideoconvert` / NVMM caps / `nvinfer` /
the TensorRT-accelerated SmolVLA/ACT engines itself. The open-core image built
from this directory never bundles any of it and stays pure Apache-2.0 + NVIDIA
CUDA runtime EULA.

The GStreamer `SensorReader` and perception-tee Python code
(`python/runner/src/openral_runner/backends/gstreamer/`) still ships in this repo
as the opt-in `gstreamer` extra — it is import-safe without `gi` (the `gi`/`Gst`
imports are lazy) and is what OpenRAL Pro builds on. It is simply not installed
in this image.

## Build-time and runtime caches

Two settings in `Dockerfile.x86` exist purely to keep cold starts sane, and
both are easy to drop in a refactor without noticing:

- **`ENV UV_COMPILE_BYTECODE=1`** — uv skips `.pyc` generation by default, so
  without it the *first* import of the torch/lerobot/transformers tree inside a
  fresh container byte-compiles the whole thing at runtime. Measured live on an
  SO-101 deploy: the lazy SmolVLA import inside `runtime_node` crawled for 15+
  minutes, GIL-starved to ~12% of a core by the two 30 fps camera threads
  sharing its process. Compiling at build time costs a few minutes once.
- **`ENV HF_HOME=/opt/openral/hf-cache`** — pins the Hugging Face cache to a
  stable, mountable path instead of `$HOME/.cache/huggingface` in the
  container's writable layer. **Naming it does not persist it.** Mount it:

  ```bash
  docker run --rm --gpus all \
      -v ~/.cache/huggingface:/opt/openral/hf-cache \
      openral:x86 deploy run --config scenes/deploy/so101_bench.yaml
  ```

  Without the mount every `--rm` run re-downloads the rSkill's weights —
  gigabytes and minutes per launch (SmolVLA ~1.4 GB, Robometer-4B ~3.5 GB) —
  and the cost is invisible in a `phase_timer` breakdown because it lands
  inside `from_pretrained`.

## Driving the container's ROS graph from the host

The image sets `RMW_IMPLEMENTATION=rmw_cyclonedds_cpp`. A host shell with
`RMW_IMPLEMENTATION` unset gets Fast-DDS on Jazzy, so host-side `ros2` tooling
**cannot see the container's nodes even with `--network host`** — an
`ros2 action send_goal` just hangs until its timeout against a running action
server. Either dispatch from inside the container:

```bash
docker exec <name> bash -lc 'source /workspace/install/setup.bash && ros2 action send_goal …'
```

or match the middleware on the host:

```bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
```

## CI

`.github/workflows/docker-build.yml` builds this image **only when its inputs
change** (`docker/inference/**`, `pyproject.toml`, `uv.lock`, `packages/**`,
`cpp/**`, or the workflow itself). On a pull request it builds and runs the
`docker-smoke-x86-deploy` smoke (asserts `gi` absent, cv2 / feetech /
onnxruntime / omdet present, `deploy run --dry-run` resolves) but does **not**
push. On merge to `master` it builds and pushes `openral:x86` to GHCR. The image
is ~25 GB, so the workflow frees runner disk before building.

### Build caching

The image is large and its code is baked in, not mounted, so a naive rebuild is
~30 minutes. Four mechanisms keep an incremental PR build far below that. If you
change the Dockerfile's layer order, keep them in mind — they are easy to defeat
by accident.

1. **Containerd image store, `docker` buildx driver.** The build writes straight
   into the store dockerd reads from. Under the older `docker-container` driver,
   `--load` had to serialize the finished 7.9 GB image to a tarball and replay it
   into the daemon — 45% of total build time, paid on every run no matter how
   good the cache was. The workflow asserts the snapshotter is active and fails
   the job if it is not.

2. **Registry-backed layer cache on GHCR.** `master` pushes write
   `$IMAGE:buildcache` (`mode=max`, so the builder stage's layers are cached
   too, not just the final stage); every run reads it.

3. **Per-PR layer cache.** Same-repo PRs additionally read and write
   `$IMAGE:buildcache-pr-<n>`, so the second commit on a branch reuses what the
   first one built instead of starting again from the master cache. Fork PRs
   have a read-only token and are detected and skipped. The ref is deleted when
   the PR closes, by `docker-build-cache-cleanup.yml`.

4. **Layer ordering that isolates the two expensive steps.** Two rules:
   - **The dependency sync must not depend on Python source.** A `manifests`
     stage prunes `python/` to the workspace members' `pyproject.toml` files, and
     `uv sync --no-install-workspace` installs the 215-package third-party
     closure against *that*. The real `python/` is copied afterwards and a second
     sync adds the 14 local packages in ~2s. Copying `python/` before the sync —
     the obvious-looking arrangement — makes every one-line source edit
     re-download ~4.4 GiB of wheels. Note the BuildKit cache mount on
     `/root/.cache/uv` does **not** save you: cache mounts are not exported by
     the registry cache exporter, so on a fresh runner it is always cold.
   - **`opentelemetry_cpp_vendor` gets its own layer.** It builds
     opentelemetry-cpp from source and serially gates `openral_safety_kernel`,
     together ~210s of the colcon step. It is copied and built before
     `packages/` lands so that a `packages/` or `python/` diff cannot invalidate
     it.

To validate a change to the build driver or image store before it reaches
`master`, run the workflow via **workflow_dispatch**: dispatch runs exercise the
registry cache *export* path against a scratch `:buildcache-dispatch` ref that
`master` never reads.

## Image sizes

| Image | Size |
|---|---|
| `openral:x86` (gstreamer-free) | 25.2 GB |
| `openral:x86-deepstream-latest` (built by openral-pro) | larger (adds the media stack) |
