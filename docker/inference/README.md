# Inference deploy image

OpenRAL ships **one** open deploy image, built from `Dockerfile.x86`. It is
**GStreamer-free**: CUDA 13, ROS 2 Jazzy, cameras through the open
`opencv_thread` backend, inference through pytorch / onnxruntime. The
proprietary media stack (GStreamer, the NVMM zero-copy path, DeepStream,
TensorRT) is an OpenRAL Pro plugin that layers on top of this image.

| Image | Built by | Pushed to GHCR | License | Use it for |
|---|---|---|---|---|
| `openral:x86` | `just docker-build-x86` | Yes, on merge to `master` (`docker-build.yml`) | Apache-2.0 + NVIDIA CUDA runtime EULA | The default open deploy target: x86 with an NVIDIA dGPU, host driver 580.65 or newer. |
| `openral:l4t` | `just docker-build-l4t` | No | Same | Jetson (aarch64). The same Dockerfile under an aarch64 tag; the CUDA base is multi-arch. Verified on a Jetson Thor. |
| `openral:x86-deepstream-latest` | openral-pro's `Dockerfile.pro` | No | Apache-2.0 + NVIDIA DeepStream EULA | OpenRAL Pro only. `FROM`s `openral:x86` and adds the media stack. Local or private registry. |

Scenes that target the open image must bind cameras with `backend: opencv_thread`
(not `gstreamer`); see
[`scenes/deploy/so101_bench.yaml`](../../scenes/deploy/so101_bench.yaml).

## Quick start

```bash
just docker-build-x86                       # build openral:x86 (or pull it from GHCR)
docker run --rm --gpus all openral:x86 doctor
docker run --rm --gpus all \
    -v ~/.cache/huggingface:/opt/openral/hf-cache \
    openral:x86 deploy run --config scenes/deploy/so101_bench.yaml
```

The image's `ENTRYPOINT` is the bare `openral` CLI, so the arguments after the
image name are `openral` subcommands (`doctor`, `detect`, `deploy run --config ...`).
The default `CMD` is `deploy run --help`. A compose file is also provided:
`docker compose -f docker/inference/docker-compose.yaml up inference-x86`.

**Mount the Hugging Face cache.** The image sets `HF_HOME=/opt/openral/hf-cache`
but naming the path does not persist it. Without the `-v` mount, every `--rm` run
re-downloads the rSkill's weights, and the cost hides inside `from_pretrained`.

## Host driver requirements

The runtime base is `nvidia/cuda:13.0.0-runtime-ubuntu24.04` (the builder uses
the `-cudnn-runtime` variant). CUDA 13 needs **host NVIDIA driver 580.65 or newer**.

| Host driver | What works | What fails |
|---|---|---|
| **580.65 or newer** | Everything: torch CUDA, onnxruntime CUDA EP, GPU-resident VLA, reward and detector models | nothing |
| **570 to 579** | The ROS graph, OpenCV capture, CPU inference; `deploy run` comes up | `torch.cuda.is_available()` is `False`; GPU-resident models run out of memory or refuse at the VRAM preflight |
| **below 570** | nothing: the CUDA 13 stack stops loading | everything |

`openral doctor` reports the driver version, so the mismatch shows up front.

## What's in the image

- **Base:** Ubuntu 24.04 (noble), Python 3.12.
- **No GStreamer / PyGObject / cairo.** `import gi` is absent by design. `libgl1`
  and `libglib2.0-0` are installed only so `import cv2` loads in a headless
  container.
- **ROS 2 Jazzy** (`ros-base`, `sensor-msgs`, `rclpy`, `rmw-cyclonedds-cpp`) in
  both stages, plus moveit, nav2, slam-toolbox and octomap for the wrapped-ROS
  rSkills and mobile robots.
- **A uv-managed venv** at `/workspace/.venv` with the OpenRAL Python packages and
  these dependency groups:
    - `sim`: the lerobot / transformers / accelerate stack the VLA adapters and
      the in-process Robometer reward monitor import (needed on real hardware too);
    - `omdet`: the open-vocabulary detector (omdet-turbo) the
      `ros_image_detector_node` runs;
    - `lingbot`: the LingBot protocol dependencies (msgpack, websocket);
    - `opencv-python` and `onnxruntime`, installed after the sync (they back the
      detector's CPU-ONNX tier and `runtime: onnx` rSkills);
    - `feetech-servo-sdk` and `deepdiff` for the SO-100 / SO-101 servo bus, pulled
      in by `openral-hal`'s `lerobot[feetech]`.

  The sync runs without the `gstreamer` extra and without the `tensorrt` group
  (OpenRAL Pro).
- **A colcon `install/` overlay** at `/workspace/install/`, mirroring `just ros2-build`:
  `openral_msgs`, `opentelemetry_cpp_vendor`, the C++ `openral_safety_kernel`,
  `openral_hal_node` and `openral_hal_openarm`, `openral_world_state`,
  `openral_reasoner_ros`, `openral_prompt_router`, `openral_safety` and
  `openral_safety_watchdog`, `openral_human_estop`, `openral_compliance`,
  `openral_foxglove_bringup`, `openral_rskill_ros`, `openral_octomap_bridge`, and
  `openral_perception_ros` (the GStreamer-free detector and reward-monitor nodes).
  `tools/`, `rskills/`, `scenes/` and `.venv` are copied to `/workspace` and
  symlinked under `install/`, so `deploy run --config scenes/deploy/<workcell>.yaml`
  needs no host bind-mount. `git` is in the runtime stage for the Robometer
  sidecar's first-use venv.
- **`/entrypoint.sh`** sources `/opt/ros/*/setup.bash` and
  `/workspace/install/setup.bash`, then execs the command.
- **Environment:** `ROS_DISTRO=jazzy`, `RMW_IMPLEMENTATION=rmw_cyclonedds_cpp`,
  `PATH=/workspace/.venv/bin:$PATH`, `PYTHONUNBUFFERED=1`, `HF_HOME=/opt/openral/hf-cache`.
  There is **no `ROS_DOMAIN_ID`**: `deploy run` confines a real robot's graph to
  the host on a per-robot domain and refuses a networked scope without
  `OPENRAL_ALLOW_NETWORK_ROS=1` and SROS2
  ([ROS graph security](../../docs/reference/ros-graph-security.md)). Pass
  `-e ROS_DOMAIN_ID=N` only to pin a domain, and never run a real robot on 0.
  `UV_COMPILE_BYTECODE=1` is set at build time so the first import of the
  torch / lerobot tree does not byte-compile at runtime.

## Driving the container's ROS graph from the host

The image uses `rmw_cyclonedds_cpp`. A host shell with `RMW_IMPLEMENTATION` unset
gets Fast DDS on Jazzy, so host-side `ros2` tools cannot see the container's nodes
even with `--network host`, and `ros2 action send_goal` hangs. Either run the
command inside the container:

```bash
docker exec <name> bash -lc 'source /workspace/install/setup.bash && ros2 action send_goal ...'
```

or match the middleware on the host: `export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp`.

The `ros-jazzy-*` apt packages are not version-pinned. Each build records what it
resolved in `/opt/openral/ros-jazzy-versions.txt` (runtime) and
`/opt/openral/ros-jazzy-versions.builder.txt` (what colcon compiled against); diff
them between two images when a ROS-level regression appears without a code change.

## OpenRAL Pro variant

GStreamer's NVMM fast path, DeepStream and TensorRT are EULA-restricted and live
in the private `openral-pro` repo, whose `docker/Dockerfile.pro` layers them on
`openral:x86` (and `Dockerfile.l4t-pro` on `openral:l4t`). The image built here
never bundles them. The GStreamer `SensorReader` code
(`python/runner/src/openral_runner/backends/gstreamer/`) still ships in this repo
as the opt-in `gstreamer` extra; it is import-safe without `gi` and is not
installed in this image.

## For maintainers: CI and build caching

`.github/workflows/docker-build.yml` builds `openral:x86` **on merge to `master`
only**, and only when its inputs change (`docker/inference/**` except this README,
`.dockerignore`, `pyproject.toml`, `uv.lock`, `python/**`, `packages/**`,
`cpp/**`, the live-ROS tests it hosts, or the workflow itself). It runs the smoke
tests, then the live-ROS suite, then pushes to GHCR. It does not run on pull
requests; if your PR touches the image, dispatch it against your branch with
`gh workflow run docker-build.yml --ref <branch>` (a dispatch run builds and tests
the same way but never pushes).

The smoke tests are also Justfile recipes, so you can run them locally:

| Recipe | Checks |
|---|---|
| `just docker-smoke-x86-deploy` | `gi` is absent; cv2, feetech, onnxruntime, omdet and the LingBot protocol dependencies import; `deploy run --config so101_bench.yaml --dry-run` resolves |
| `just docker-smoke-x86-reasoner` | a live prompt-router and reasoner round-trip (no LLM key needed) |
| `just docker-smoke-x86-safety-kernel` | the safety kernel binary is present and `openral_msgs` imports |

The image is large and its code is baked in, not mounted, so layer order matters.
Three mechanisms keep incremental builds fast; they are easy to defeat by
accident:

1. **Containerd image store with the `docker` buildx driver.** The build writes
   straight into the store dockerd reads from. The older `docker-container`
   driver had to serialize every layer to a tarball and replay it on `--load`.
   The workflow asserts the snapshotter is active and fails otherwise.
2. **Registry-backed layer cache on GHCR.** Pushes to `master` write
   `$IMAGE:buildcache` (`mode=max`, so the builder stage is cached too); every
   run reads it.
3. **Layer ordering that isolates the expensive steps.**
    - The dependency sync must not depend on Python source. A `manifests` stage
      prunes `python/` to the workspace members' `pyproject.toml` files, and
      `uv sync --no-install-workspace` installs the third-party closure against
      that. The real `python/` is copied afterwards for a second, fast sync.
      Copying `python/` first makes every one-line edit re-download the wheels.
      A BuildKit cache mount on `/root/.cache/uv` does not help, because the
      registry cache exporter does not export cache mounts.
    - `opentelemetry_cpp_vendor` gets its own layer. It builds opentelemetry-cpp
      from source and gates `openral_safety_kernel`, so it is built before
      `packages/` is copied and a `packages/` or `python/` diff cannot
      invalidate it.

To validate a change to the build driver or image store before it reaches
`master`, run the workflow via `workflow_dispatch`: it exercises the cache export
against a scratch `:buildcache-dispatch` ref that `master` never reads.

**Keep the buildcache in the same GHCR repository as the image**
(`ghcr.io/openral/openral`). GHCR deduplicates blobs per repository: because
`--cache-to type=registry,mode=max` already writes every layer there, the final
push mostly reports `Layer already exists`. A separate cache repository breaks
the dedup, the push reverts to uploading the full image, and nothing fails or
warns; the publish step just gets minutes slower.
