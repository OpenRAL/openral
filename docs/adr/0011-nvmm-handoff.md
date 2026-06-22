# ADR-0011: NVMM → CUDA tensor handoff across the Sensors / Skill boundary

* **Status**: Proposed (2026-05-24)
* **Deciders**: Adrian Llopart, OpenRAL Contributors
* **Supersedes / amends**: extends [ADR-0010](0010-inference-runner.md)
  §SensorReader and the §6.1 layer-boundary discipline.

## Context

ADR-0010 defined a `SensorReader` Protocol with three carry-modes on
`SensorFrame`: `data` (inline bytes), `topic` (ROS 2 ref), and `handle`
(opaque in-process integer — reserved for "NVMM pointer, DMA-BUF fd").
M8 / PR I implemented the `GStreamerSensorReader` backend. On Jetson
Orin / Spark / Thor the GStreamer pipeline produces frames in
`video/x-raw(memory:NVMM)` caps; the underlying buffer is an
`NvBufSurface` whose `surfaceList[i].dataPtr` is a **CUDA device
pointer**. Passing that pointer to a downstream Skill / TensorRT
engine is the zero-copy path the inference runner needs to hit its
latency budgets — at 30 Hz × 4 cameras × 1080p RGB, a `bytes(...)`
copy through Python is roughly 30 ms of wasted CPU per tick before
inference even starts.

The unresolved question ADR-0010 left open: **how does a Skill consume
a `handle`-flavoured `SensorFrame`?** That's a cross-layer concern —
the producer is Layer 1 (Sensors), the consumer is Layer 3 (Skill),
and the handle's lifetime is owned by GStreamer at Layer 1.

## Decision

A `handle`-flavoured `SensorFrame` carries **three pieces of
information** that together let any CUDA-aware consumer read pixels
without an intermediate copy:

1. `SensorFrame.handle: int` — the raw `dataPtr` from
   `NvBufSurfaceParams[0]`. Suitable as a kernel argument, a
   `torch.cuda.Tensor` view via `__cuda_array_interface__`, or input
   to `cudaMemcpyAsync`.
2. `SensorFrame.encoding == FrameEncoding.CUDA_NV12` — tells the
   consumer the layout is semi-planar Y (full-res) + UV (half-res
   interleaved). NV12 is what `nvarguscamerasrc` / `nvv4l2decoder` /
   `nvvidconv` natively produce; we do not force a colour conversion
   on the NVMM side because the cost of that conversion outweighs
   the convenience of BGR for most VLA inputs (which prefer NV12 or
   YUV420 anyway).
3. `SensorFrame.metadata["nvbufsurface"]` — a Pydantic-validated
   dump of `NvBufSurfaceHandle` (gpu_ptr, width, height, pitch,
   color_format, size, batch_size). Lets consumers assert the
   pitch and pick the right stride.

### Lifetime contract

The Sensors layer owns the GStreamer buffer ref. The
`GStreamerSensorReader` holds **both** the latest `Gst.Buffer` *and*
its `Gst.MapInfo` in lock-guarded slots for the lifetime of the
latest-frame slot. The pair is released (`buffer.unmap(map_info)`)
when a newer frame replaces the slot or `close()` is called.
Consumers that need the data to persist past one tick **must copy
it** — either to system memory (`cudaMemcpy` to a host buffer) or to
their own CUDA allocation (`cudaMallocAsync` + `cudaMemcpyAsync`).

This matches GStreamer's normal buffer-recycling semantics: the
upstream element is free to reuse the buffer slot as soon as the
appsink stops referencing it.

### Shared CUDA context

`openral_runner.backends.gstreamer.cuda_context.get_shared_cuda_context()`
returns a process-wide singleton context bound to
`OPENRAL_CUDA_DEVICE_INDEX` (default `0`). Consumers that go through this
helper share one context with GStreamer + TRT + any future Skill GPU
code — necessary to avoid the
`CUDA_ERROR_INVALID_CONTEXT` failure mode when multiple PyCUDA users
each call `cuda.init()` independently.

### Skill consumer pattern

Skills that opt into the NVMM path import the helper module and
construct a CUDA-aware view in their `step()`:

```python
import torch
from torch.utils import dlpack

def step(self, world_state):
    frame = world_state.image_frames["wrist_csi"]
    if frame.encoding is FrameEncoding.CUDA_NV12:
        meta = frame.metadata["nvbufsurface"]
        # Build a __cuda_array_interface__ dict directly from the handle.
        # Concrete adapter helpers will land alongside the first NVMM
        # skill (out of scope for this ADR).
        ...
    elif frame.data is not None:
        # CPU fallback; current behavior for skills that don't opt in yet.
        ...
```

Skills that have not opted in continue to receive CPU-bytes frames
because the reader's NVMM branch only fires when the pipeline
*itself* negotiates NVMM caps — i.e. the user wrote a Jetson YAML.
The CPU path is unchanged.

### Layer-boundary status

This decision crosses the Sensors → Skill boundary by introducing a
GPU-resident contract. Per CLAUDE.md §6.1 that requires an ADR; this
file is that ADR. The boundary is **not** widened — Skills still
consume a `SensorFrame`, the only change is that `handle` becomes a
load-bearing field for one of the four `SensorReaderBackend` values.
The HAL, Reasoning, WAM, Safety, and Observability layers see no
change: DeployRunner's tick loop passes the `SensorFrame` through
to `Skill.step` verbatim.

## Consequences

### Positive

* Latency: per-camera frame delivery drops from ~5–10 ms (`bytes(...)` copy + JSON-safe base64 serialization on CPU) to ~50 µs (NvBufSurface struct read) on Jetson AGX-class hardware.
* DeepStream compatibility: our pipelines emit and consume NvBufSurface, so a user can prepend `nvinfer` to any pipeline string without our code changing.
* Skill freedom: Skills that want a numpy array can ignore the handle and let the runner fall back to the CPU caps form by setting `enable_nvmm: false` in the YAML.

### Negative

* The handle is non-serializable. Trace capture for the NVMM path can only persist a hash / footprint, not the pixels themselves. The reproducibility-from-trace property (CLAUDE.md §1.8) is preserved by storing a downsampled CPU copy in the trace alongside the handle. This is a future-PR concern; flagged here for visibility.
* Skills opting into the NVMM path are bound to PyCUDA / torch.cuda. We accept this; the alternative (forcing a CPU copy at the boundary) defeats the purpose.

### Mitigations

* `openral_runner.backends.gstreamer.cuda_context.cuda_context_state()` is exposed for `openral doctor` so misconfigured devices surface during onboarding rather than during a hot rollout.
* `NvBufSurfaceHandle` is a frozen Pydantic model — consumers cannot mutate the handle in-place, which keeps the lifetime invariant trivially auditable.

## Alternatives considered

* **Copy NVMM → CPU at the reader boundary** (`cudaMemcpy` inside `_on_new_sample` to system memory, then deliver `data=bytes`). Loses the headline latency benefit; equivalent to running on the CPU path. Rejected.
* **Use DLPack tensors directly** at the `SensorFrame` level instead of an opaque `handle` int. Cleaner consumer API but pulls a hard dependency on a DLPack-capable framework (torch / cupy / numpy ≥ 1.22) into `openral_core`. Rejected for layering reasons; the helper adapter (future PR) can wrap the handle in a DLPack capsule on demand.
* **Use Holoscan tensors**. See [ADR-0010 Amendment 2026-05-12](0010-inference-runner.md#amendments) — Holoscan's tensor map is cleaner but the SDK adds ~1.5 GB and would require a 50–70% rewrite of the NVMM handoff. Deferred via the `SensorReaderBackend.HOLOSCAN` enum reservation.

## Provenance

The `nvbufsurface.py` ctypes binding and `cuda_context.py` shared
context manager derive from work originally authored by Adrian Llopart
(adrianllopart@gmail.com) — see the module docstrings for explicit
re-licensing. Struct layout follows NVIDIA's publicly distributed
`nvbufsurface.h` (L4T Multimedia API, JetPack-bundled).
