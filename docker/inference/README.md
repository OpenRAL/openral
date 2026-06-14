# Inference deploy image

OpenRAL ships **one** deploy image. ADR-0010 amendment
"Single-Dockerfile consolidation + CUDA-13/DeepStream-9 alignment"
replaced the four-Dockerfile matrix that PR #93 introduced
(`Dockerfile.x86`, `Dockerfile.x86-ros`, `Dockerfile.x86-deepstream`,
`Dockerfile.l4t`) with a single source of truth + one optional flag.

| Image | Built by | Pushed to GHCR? | License | When to use |
|---|---|---|---|---|
| `openral:x86-latest` | `just docker-build-x86` | ✅ Yes (`docker-build.yml`) | Apache-2.0 + NVIDIA CUDA runtime EULA | x86 with NVIDIA dGPU, host driver ≥ 580. The default deploy target. Carries CUDA 13, ROS 2 Jazzy, GStreamer 1.24. |
| `openral:x86-deepstream-latest` | `just docker-build-x86-deepstream` | ❌ **No** | Apache-2.0 **+ NVIDIA DeepStream EULA** | Opt-in. Adds `nvvideoconvert`, NVMM caps on x86, `nvinfer`, `nvstreammux`. Local / private-registry only. |

The L4T / Tegra / Jetson Orin variant, the CPU-only variant, and the
no-ROS variant from PR #93 are deliberately out of scope here. See the
ADR amendment for the trade-off rationale.

## Host driver requirements

The image's base is `nvidia/cuda:13.0.0-runtime-ubuntu24.04`. CUDA 13
needs **host NVIDIA driver ≥ 580.65**. On older drivers the image
still imports cleanly and runs non-CUDA pipelines (videotestsrc →
videoconvert → appsink, the default smoke), but every CUDA-touching
plugin will warn `CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE: forward
compatibility was attempted on non supported HW` and skip the GPU
path. `torch.cuda.is_available()` returns `False`.

| Host driver | What works | What fails |
|---|---|---|
| **≥ 580.65** | Everything: `nvh264dec/enc`, `nvjpegdec`, torch CUDA, and (DeepStream variant) NVMM allocator + `nvvideoconvert` runtime use | — |
| **570 – 579** (e.g. 575.57 — CUDA 12.9-class) | Non-CUDA GStreamer paths (`videoconvert`, `avdec_h264`, `appsink`). 30-tick smoke against `videotestsrc` passes. `openral deploy` runs. | All `nvcodec` plugins fail to register; `torch.cuda.is_available()` returns `False`; DeepStream `nvvideoconvert` registers but `Cuda failure status=804` on first NVMM frame |
| **< 570** | nothing — base image's CUDA 13 stack stops loading entirely | everything |

The `openral doctor` command surfaces the driver version so users see the
mismatch up front.

## What's in the image

The same userspace stack regardless of flavour:

- **Base**: `nvidia/cuda:13.0.0-runtime-ubuntu24.04` (Ubuntu 24.04 noble, Py 3.12).
- **GStreamer 1.24** plugin tier (`-base`, `-good`, `-bad`, `-ugly`, `libav`, `rtsp`).
  The `nvcodec` plugin (`nvh264dec/enc`, `nvjpegdec`, `cudaupload`, etc.) is
  registered when the host driver can serve CUDA 13.
- **PyGObject** — apt-managed (`python3-gi`), spliced into the workspace venv
  at `/workspace/.venv/lib/python3.12/site-packages/gi` so it shares the same
  GLib link as ROS 2's rclpy. Mixing PyPI's PyGObject build with apt's rclpy
  segfaults at `rclpy.init()` on the gi-then-rclpy import order.
- **ROS 2 Jazzy** (`ros-jazzy-ros-base`, `ros-jazzy-sensor-msgs`,
  `ros-jazzy-rclpy`, `ros-jazzy-rmw-cyclonedds-cpp`) installed in both
  build and runtime stages. The cyclonedds rmw is preferred over Fast DDS
  because Fast DDS' SHM transport interacts badly with pydantic v2's Rust
  core + gst-cuda plugin scan.
- **uv-managed workspace venv** at `/workspace/.venv` with the OpenRAL
  Python packages. `uv sync` runs **without `--extra gstreamer`** — see
  the gi-splice note above.
- **colcon `install/` overlay** at `/workspace/install/` (baked by the
  builder stage). Carries every ROS / C++ package the deploy graph
  needs, mirroring `just ros2-build`:
  - `openral_msgs` — the action + message IDL the rest of the graph
    consumes (`python -c "import openral_msgs.msg"` works inside the
    image, no host `colcon build` required)
  - `opentelemetry_cpp_vendor` — builds opentelemetry-cpp from source;
    the safety kernel links against it
  - `openral_safety_kernel` — C++ deny-by-default safety process. The
    binary lands at
    `/workspace/install/openral_safety_kernel/bin/safety_kernel`
  - `openral_hal_so100`, `openral_hal_openarm` — HAL lifecycle nodes
  - `openral_world_state` — ADR-0018 F8 30 Hz snapshot node
  - `openral_reasoner_ros` — ADR-0018 F4 LLM tool dispatch
  - `openral_prompt_router` — ADR-0018 F10 prompt fan-in
  - `openral_safety`, `openral_safety_watchdog` — ADR-0018 F5 + deadman
  - `openral_human_estop` — ADR-0018 F5 forwarder
  - `openral_rskill_ros` — ADR-0018 F1 `ExecuteSkill` action server

  `Python3_EXECUTABLE=/workspace/.venv/bin/python` is baked into every
  ament-python package's `CTestTestfile.cmake` so the lifecycle nodes
  resolve `structlog`, `openral_*`, and the OTel SDK through the venv
  automatically — no parallel system-python install.

  Previously the perception-tee + reasoner smoke recipes (and any
  downstream consumer) bind-mounted the host's locally-built
  `install/` into the container; that requirement is gone now that
  the colcon tree ships in the image.
- **`/entrypoint.sh`** probes `/opt/ros/*/setup.bash` AND
  `/workspace/install/setup.bash`, sources both (system distro first,
  local overlay second), and exec's the user command. Probe-style
  rather than hardcoded so future arches reuse this script.
- ENV: `ROS_DISTRO=jazzy`, `RMW_IMPLEMENTATION=rmw_cyclonedds_cpp`,
  `ROS_DOMAIN_ID=0`, `PATH=/workspace/.venv/bin:$PATH`,
  `PYTHONUNBUFFERED=1`, `GST_DEBUG=2`.
- Default `ENTRYPOINT`: `openral deploy run` (post-ADR-0021 — the CLI
  ships as `openral`, no `ral` alias).

DeepStream-only additions on `:x86-deepstream-latest`:

- `bzip2`, `libyaml-cpp0.8`, `libssl3`, `libjansson4`, `libuuid1`,
  `gstreamer1.0-plugins-base-apps` — DeepStream installer runtime deps.
- DeepStream SDK 9.0 unpacked at `/opt/nvidia/deepstream/deepstream/`.
  Samples directory pruned to keep the image lean.
- `LD_LIBRARY_PATH=/opt/nvidia/deepstream/deepstream/lib:...`
- `GST_PLUGIN_PATH=/opt/nvidia/deepstream/deepstream/lib/gst-plugins`
- Plugins now registered: `nvvideoconvert`, `nvstreammux`, `nvtracker`,
  `nvjpegdec`, `nvv4l2decoder`, `nvmsgconv`, `nvmsgbroker`,
  `nvdsanalytics`.

There is **no separate CUDA-13 side-load step**. PR #93's
`Dockerfile.x86-deepstream` installed `cuda-cudart-13-0` + `libnpp-13-0`
alongside the CUDA-12.6 base to make DeepStream 9 work; that hack is
gone now that the base itself is CUDA 13.

## The DeepStream variant — read this before building

DeepStream is **proprietary, EULA-restricted, and NOT open source**.
[ADR-0010 Amendment 2026-05-12](../../docs/adr/0010-inference-runner.md)
(refined 2026-05-14) rejected bundling DeepStream into the default
image; this opt-in variant is the escape hatch for users who have
separately accepted NVIDIA's terms.

### EULA you are accepting

By building `openral:x86-deepstream-latest` you agree to the
[NVIDIA DeepStream EULA](https://developer.download.nvidia.com/assets/Deepstream/LicenseAgreement-NGC.pdf).
The clauses most relevant to OpenRAL usage:

| Clause | What it says | What it means for you |
|---|---|---|
| §1.c | You may distribute "derived CONTAINERs" that bundle DeepStream with other primary-functionality software | OpenRAL counts as "other primary-functionality software", so bundling into this image is permitted |
| §2.a | The application must have "material additional functionality" beyond DeepStream | OpenRAL is a robot agent harness; trivially satisfied |
| §2.b | Modified source must carry the notice *"This software contains source code provided by NVIDIA Corporation"* | Add this notice to any forked code derived from DeepStream samples |
| §2.c | Downstream distribution must be under terms at least as protective as NVIDIA's | The resulting image is a **mixed license**: Apache-2.0 for OpenRAL + NVIDIA EULA for the DeepStream binaries. **The image is NOT redistributable as Apache-2.0.** Do not push it to a public registry. |
| §4.c | No stand-alone DeepStream redistribution | n/a — we always bundle |
| **§4.g** | **No benchmark / competitive-analysis publication without prior NVIDIA written permission** | You may NOT publish `SkillEvalResult` JSONs that compare a DeepStream-bundled pipeline against the open-core path. This applies to the entire `skills/<name>/eval/` story for this image. |
| **§4.h (second)** | Not for life-critical applications (avionics, medical, military, navigation) without a separate NVIDIA agreement | If your robot agent runs in any of those domains, do not use this image. |
| §8 | OSI-licensed user-code is explicitly allowed | Your OpenRAL application code can remain Apache-2.0 |
| §12 | Total NVIDIA liability capped at US$10.00 | Standard but worth noting |
| §13 | NVIDIA may terminate the license at will | Plan a fallback path to the default `:x86-latest` image |

### Build flow

```bash
# 1. Visit https://developer.nvidia.com/deepstream-getting-started
# 2. Read the EULA, accept, download deepstream_sdk_v9.0.0_x86_64.tbz2 (~1.5 GB)
# 3. Move the tarball into the dedicated build-context directory:
mv ~/Downloads/deepstream_sdk_v9.0.0_x86_64.tbz2 docker/inference/deepstream/

# 4. Build (the tarball is in .gitignore, never committed; the
#    BuildKit `--build-context ds=...` flag makes it visible only to
#    the ds-on stage)
just docker-build-x86-deepstream

# 5. Verify the new elements are registered
docker run --rm --gpus all --entrypoint bash openral:x86-deepstream-latest \
    -c 'gst-inspect-1.0 nvvideoconvert | head -3'

# 6. Smoke a real pipeline through nvvideoconvert (system memory)
docker run --rm --gpus all --entrypoint bash openral:x86-deepstream-latest \
    -c 'gst-launch-1.0 -e videotestsrc num-buffers=30 ! nvvideoconvert \
        ! "video/x-raw,format=BGR,width=320,height=240" ! fakesink'
```

### Why the DeepStream variant is NOT in CI

`.github/workflows/docker-build.yml` pushes images to GHCR (public). Per
EULA §2.c, the resulting bundled image cannot be relicensed as
Apache-2.0; making it pullable from a public registry would equate to
public redistribution of DeepStream Deliverables under terms that are
not "at least as protective" as NVIDIA's. The opt-in image lives
outside CI by design.

### How the runtime sees DeepStream

The open-core pipeline builder
(`python/runner/src/openral_runner/backends/gstreamer/pipeline.py`)
returns `videoconvert` on `Platform.NVIDIA_DESKTOP` because
`nvvideoconvert` is not in the open-source `gst-plugins-bad` `nvcodec`
family. Users running the DeepStream image can patch `_build_convert`
and `_build_caps` downstream to:

- Return `nvvideoconvert` on `Platform.NVIDIA_DESKTOP`
- Emit `video/x-raw(memory:NVMM)` caps on `Platform.NVIDIA_DESKTOP`

That patch must NOT be merged upstream — it would force every default
user to accept the DeepStream EULA implicitly.

## File naming

The DeepStream tarball filename `deepstream_sdk_v9.0.0_x86_64.tbz2` is
referenced in:

- `Dockerfile.x86`'s `ds-on` stage `COPY --from=ds` line.
- `Justfile`'s `docker-build-x86-deepstream` preflight check.

If NVIDIA updates the version, change both call sites. The
`/opt/nvidia/deepstream/deepstream/` symlink is created by `install.sh`
and points at the version-specific install dir, so the rest of the
Dockerfile is version-agnostic.

## Image sizes

Measured 2026-05-14 on the consolidation worktree:

| Image | Size |
|---|---|
| `openral:x86-latest` | ~12.5 GB |
| `openral:x86-deepstream-latest` | ~15.1 GB |

The default image is larger than PR #93's `:x86-latest` (~11.9 GB)
because ROS Jazzy is now baked in unconditionally. The DeepStream
image is comparable to PR #93's `:x86-deepstream-latest` (~14.5 GB),
slightly larger because ROS is bundled but smaller per-DS-byte
because the side-load `cuda-cudart-13-0` step is gone.
