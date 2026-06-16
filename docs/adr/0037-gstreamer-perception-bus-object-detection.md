# ADR-0037: GStreamer perception bus — reasoner-activated tee consumers + object-detection rSkill

- Status: **Proposed**
- Date: 2026-06-03
- Related: [ADR-0010](0010-inference-runner.md) (inference runner — the GStreamer sensor pipeline / NVMM reader, PR I);
  [ADR-0018](0018-ros2-reasoner-supervisor.md) (reasoner tool palette + the F6 perception tee /
  `/openral/perception/<kind>`); [ADR-0024](0024-ros-wrapped-rskills.md) (the "result-only, no
  `Action` emitted" rSkill pattern this reuses); [ADR-0022](0022-rskill-action-vocabulary.md)
  (rSkill palette metadata); CLAUDE.md §3 (layer boundaries; dual-system), §1.5 (hot path is C++
  and bounded), §1.11 (real components), §9 (license lineage). Builds on the `TensorRTRuntime`
  ONNX→TRT backend (PR #223).

## Context

We want an **object-detection rSkill** the S2 reasoner can activate/deactivate, that runs on the
GPU with **zero GPU↔CPU copies**, teeing off the camera GStreamer pipeline after frames are
already on the GPU. Investigating the path surfaced three structural facts:

1. **rSkills do not consume the GStreamer pipeline at all today.** The skill observation path is
   CPU numpy delivered over ROS image topics: a camera/HAL node publishes `sensor_msgs/Image` →
   the `world_state` node calls `WorldStateAggregator.update_image_frame` → the skill reads it in
   `snapshot()`. The skill_runner's policy-adapter observation assembly drops any GPU handle
   (`if frame.data is None: continue`, then `np.frombuffer`), so only CPU pixels reach the
   policy. The NVMM zero-copy reader (ADR-0010) exists but its `bh_sink` policy leg is
   effectively unconsumed by skills.
2. **A CUDA device pointer / NVMM surface is process-local.** Zero-copy GPU inference can only
   happen in the process that owns the pipeline's CUDA context. The current `skill_runner`
   process is not that process (the pipeline, when run, lives in a separate sensor process that
   publishes BGR-over-ROS — a copy at that boundary).
3. **The detector is the missing producer the spatial-memory work needs.** PRs #217–#220 (the
   persistent 3D spatial-memory + reasoner-query stack) each close by noting that *no node fills
   `WorldState.detected_objects` from live perception*. This rSkill is that producer.

The product goal is broader than one skill: **all perception/policy consumers (VLAs, the object
detector, other ROS algorithms) should tee off one camera pipeline, and the reasoner should add
and remove those tees on demand.** That is the north star this ADR commits the architecture to,
while scoping the first increment to the object detector.

DeepStream is available opt-in and EULA-gated via `docker/inference/Dockerfile.x86` `ds-on`
(`WITH_DEEPSTREAM_STAGE=on`, DeepStream 9 / CUDA 13), which provides `nvinfer` + NVMM caps on
x86; it is **not** bundled in open-core. (The "no `nvvideoconvert` on x86" comments in the
pipeline builder predate the `ds-on` stage and are stale.)

## Decision

1. **A GStreamer "perception bus."** One pipeline per camera owns the GPU frames. The
   `GStreamerSensorReader` is co-located **in the `runtime_node` process** (alongside
   `world_state` + `skill_runner`, sharing one CUDA context), so in-process consumers can read
   NVMM zero-copy. Consumers attach as **tee branches**; results leave the pipeline only as typed
   ROS messages (`ObjectsMetadata`, `candidate_action`), never GPU buffers across process lines.
2. **A `TeeManager` performs dynamic pad add/remove on the live pipeline.** `attach(branch) →
   handle` requests a `tee` src pad, installs a blocking pad-probe, links
   `queue leaky=downstream ! <branch> ! appsink`, syncs state, unblocks; `detach(handle)` blocks,
   unlinks, releases the request pad, tears the branch down. Each branch gets its own
   `queue leaky=downstream max-size-buffers=2` so a stalled/crashing consumer never backpressures
   the policy leg (the ADR-0018 §3 isolation invariant). The policy and ROS-observability legs
   are never disturbed.
3. **The reasoner controls tees through the existing `ExecuteSkill` verb — no new
   `ReasonerToolCall` variant.** Activating a detector rSkill *is* "attach a tee"; its
   cancel/deadline/deactivate *is* "detach." The palette, license guard, registry refresh, and
   replanning ladder are reused unchanged. (The stubbed `ReloadGstPipelineTool` / GH-126 is
   untouched and out of scope.)
4. **A new rSkill `kind: "detector"`.** It declares `runtime: tensorrt` (ships ONNX; the engine
   is built on load by `TensorRTRuntime`, PR #223), `role: s1` (required for palette inclusion —
   `build_tool_palette` filters to `s1`; it carries **no actuation authority** regardless of
   slot), a class/output contract, and a latency budget; it forbids the VLA `model_family` /
   action contract. Its `step()` publishes `ObjectsMetadata` and returns **no `Action`** —
   modeled on the ADR-0024 "result-only" mode — so the safety kernel / HAL are never driven by
   it.
5. **Two-tier inference, selected by a capability probe** (mirroring `detect_platform`):
   - **Zero-copy NVMM tier.** When DeepStream is present, the official **`nvinfer`** element runs
     on the NVMM buffer. Without DeepStream but with NVMM (Tegra), a **clean-room, Apache-2.0**
     `GstBase.Aggregator` element consumes the `NvBufSurface` device pointer on the shared CUDA
     context. (Technique informed by, never copied from, the proprietary `videotech`
     reference — §9.)
   - **CPU fallback tier.** On hosts without NVMM (x86 open-core without DeepStream, and
     `deploy sim`), a system-memory BGR branch runs ONNXRuntime. Not zero-copy, but it is the
     testable path on GPU-less CI and gives `deploy sim` a working detector (§1.11 — real
     components, no mocks).
   Zero-copy applies only to **in-process** consumers; "other ROS algorithm" consumers in
   separate processes attach to a tee terminating in an appsink published over ROS — a
   deliberate, rate-limited copy at that boundary.
6. **Output contract: reuse the existing 2D schemas.** Detections publish as
   `ObjectsMetadata{ detections: list[ObjectDetection2D], model_id }` on
   `/openral/perception/objects` (already consumed by the reasoner). **2D only** in this epic; the
   2D→3D `DetectedObject` pose-lift (needs depth + camera intrinsics/extrinsics via TF2) →
   `WorldState.detected_objects` → spatial-memory ingest is a **separate, later decision**,
   coordinated with the spatial-memory PRs.
7. **The skill runner stays single-active for now.** No concurrent rSkills are required yet (a
   detector and a VLA do not run simultaneously). `TeeManager` is nonetheless built to hold
   multiple branches so a future concurrency increment needs no redesign.

## Alternatives considered

- **Full pipeline rebuild on activate** (the `ReloadGstPipelineTool` intent) — rejected as the
  attach mechanism: it tears down + rebuilds the reader, dropping frames and blacking out the
  policy leg for seconds on every activate/deactivate. Unacceptable mid-task.
- **Valve-gated dormant leg** (build the detection branch at startup behind a closed
  `valve drop=true`, toggle on activate) — simpler and deadlock-free, but keeps the engine
  resident in VRAM while dormant and does not literally "attach." Rejected in favor of dynamic
  pads, which match the "tee on demand, close when done" intent and keep idle cost at zero.
- **A "fat" rSkill consuming the NVMM handle in the `skill_runner` process across the ROS
  boundary** — rejected: CUDA pointers are process-local, so this would require CUDA-IPC handle
  passing (fragile) or a copy (defeats the goal). Co-locating the pipeline in `runtime_node`
  (Decision 1) is what makes in-process zero-copy possible.
- **Modeling the detector as `kind: vla` or `kind: ros_action`** — rejected: `vla` requires
  `model_family` + `weights_uri` and is an action-chunk policy (the detector ships ONNX weights
  but emits no actions), and `ros_action` forbids weights and is action-server shaped. A detector
  emits perception events and owns ONNX weights; neither existing kind fits.
- **A new `ReasonerToolCall` "attach detector tee" variant** — rejected as redundant:
  `ExecuteSkill` already expresses "run this rSkill," and lifecycle (cancel/deadline/deactivate)
  already expresses "stop it." Adding a parallel verb would fork the palette/replanning machinery.

## Consequences

- **Layer touch (ADR-gated, hence this ADR).** Layer 1 (the GStreamer pipeline gains a
  `TeeManager` and moves into the `runtime_node` process) ↔ Layer 3 (a new rSkill kind) ↔ the
  reasoner palette (Layer 4 surfaces the detector via `ExecuteSkill`). The detector (Layer 3)
  driving a tee on the pipeline (Layer 1) is the boundary crossing this ADR authorizes.
- **Schema change.** `openral_core.RSkillManifest` gains a `kind: "detector"` discriminated
  variant. On-disk `schema_version` stays `"0.1"` (no migrator pre-publish, CLAUDE.md §6); the
  repo state map and `docs/METHODS.md` update in the implementing PR.
- **Builds on PR #223.** `TensorRTRuntime` (ONNX→TRT build-on-load) is the detector's runtime;
  the NVMM device-pointer `infer` entry point is added by the detector-element PR.
- **License posture (§9).** `nvinfer` lives in EULA-gated DeepStream (opt-in image, not bundled);
  the open-core NVMM element is clean-room Apache-2.0; detector weights are RT-DETR / D-FINE
  (Apache-2.0); LibreYOLO is used build-time only to export ONNX, never in the hot path.
- **Delivered as a 6-PR epic** (smallest-viable PRs, §4.2): (1) refactor the pipeline builder to a
  named-tee structure; (2) **this ADR**; (3) `TensorRTRuntime` (PR #223, done); (4) co-locate the
  reader in `runtime_node` + `TeeManager`; (5) the two-tier detector element; (6) the
  `kind: detector` rSkill package.
- **Follow-ups (own PRs / later decisions):** the 2D→3D pose-lift + spatial-memory ingest;
  migrating VLAs to consume NVMM tees in-process (retiring the ROS-image CPU path for policies);
  `deploy sim` GStreamer parity via an `appsrc`-fed pipeline; concurrent rSkills; a configurable
  per-skill TRT optimization profile; INT8 calibration.

---

## Amendment — 2026-06-03 (PR C4: NVMM aggregator tier wired end-to-end; ADR filed as PR 2)

**What landed (PR5b / C1–C4).**
The clean-room NVMM zero-copy detector tier is fully wired:

- **`TrtNvmmExecutor`** (`openral_runner.backends.gstreamer.trt_nvmm`) — a clean-room CUDA
  RGBA→planar-NCHW kernel (`rgba_to_nchw_norm`) is compiled at load via **nvrtc** to a SASS
  cubin for the runtime-detected GPU arch, launched via the **cuda-python** (`cuda.bindings`)
  driver API directly on the NVMM `dataPtr`, writing into the TensorRT input device buffer; the
  engine then runs in the **CUDA primary context** (no GPU→CPU copy). **No pycuda / no nvcc / no
  g++** — the tier is deployable in the lean `ds-on` runtime image (see "Deployability finding").
- **`NvmmObjectsDetector`** (`openral_runner.backends.gstreamer.nvmm_detector`) — wraps
  `TrtNvmmExecutor`, applies `postprocess_rtdetr`, and emits `ObjectsMetadata`.
- **`DetectorRunner` NVMM_AGGREGATOR branch** (`openral_runner.backends.gstreamer.detector_runner`)
  — on `tier=DetectorTier.NVMM_AGGREGATOR`, `DetectorRunner` dynamically attaches a
  `<nvmm_convert_element> ! video/x-raw(memory:NVMM),format=RGBA,width=W,height=H ! appsink`
  branch to the named `openral_cam_tee`, and runs `NvmmObjectsDetector` on every NVMM buffer
  delivered to that appsink.
- **`nvmm_convert_element()`** (`pipeline.py`) resolves the NVMM-aware colour-convert element
  at runtime: `nvvideoconvert` when DeepStream (ds-on / DS9) is installed; `nvvidconv` on
  Tegra/L4T; `None` when neither is available (CPU-only hosts).

**Validation environment.** The aggregator tier is validated in the `ds-on` (DS9) container
(`docker/inference/Dockerfile.x86`, `WITH_DEEPSTREAM_STAGE=on`, DeepStream 9 / CUDA 13), where
`nvvideoconvert` + `libnvbufsurface.so` are available on x86. Both gated integration tests in
`tests/integration/test_nvmm_aggregator_ds.py` pass for real inside `ds-on`
(`test_nvmm_buffer_flow_to_wrap_buffer`: live NVMM buffer → `wrap_buffer` → valid GPU `dataPtr`;
`test_nvmm_aggregator_emits_objectsmetadata`: full `nvvideoconvert → NVMM → nvrtc kernel → TRT →
ObjectsMetadata`), and skip automatically on hosts without the DeepStream stack. The host GPU
tests (`test_trt_nvmm_executor`, `test_nvmm_detector`) cover the kernel + TRT-on-`dataPtr` path.

**Deployability finding (why nvrtc, not pycuda).** Container validation showed the lean `ds-on`
runtime image ships CUDA *runtime* libs + `libnvrtc.so` but **no `g++`, no `nvcc`, no CUDA dev
headers** — so `pycuda` cannot install (its C-extension build needs `g++`/`cuda.h`) and
`pycuda.SourceModule` (nvcc-based runtime compilation) cannot run there. The executor therefore
compiles the kernel via **nvrtc** (a runtime library already in the image) and uses **cuda-python**
(a pure wheel, already a `tensorrt`-group dependency) for memory/stream/module/launch, dropping
pycuda + the `get_shared_cuda_context()` dependency for this path. The DeepStream NVMM `dataPtr`
is accessible from the CUDA primary context (validated by a direct `cudaMemcpy`), so no explicit
shared context is needed. `reader.py`'s NVMM path still uses pycuda via `cuda_context` and is
unaffected (separate, currently-unconsumed path).

**Other validation fixes (PR5b).** (a) NVMM GstBuffers map **read-only**, so the surface struct
is read via `ctypes.from_buffer_copy` (copies only the small `NvBufSurface` metadata, not the GPU
frame — `DetectorRunner._on_sample_nvmm`); the original `from_buffer` raised "underlying buffer is
not writable". (b) `docker/inference/Dockerfile.x86` `COPY examples/` was stale (the dir was
reorganized away in `d450e55`), breaking the build — removed in a prior `fix(docker)` commit.

**`nvinfer` tier status — deferred.** The Decision §5 `nvinfer` tier (DeepStream-native inference
element) is **deferred**, not implemented: no `nvinfer` code is merged. The go/no-go spike was not
run because (i) the clean-room nvrtc `TrtNvmmExecutor` aggregator is the working, deployable tier,
and (ii) the `nvinfer` GStreamer plugin does not register in the built `ds-on` image
(`gst-inspect-1.0 nvinfer` fails while `nvvideoconvert` resolves), so a spike would first require
diagnosing that plugin-load failure. `nvinfer` remains a documented future option (an alternative
that would attach detection metadata natively / use NVIDIA-maintained inference); revisiting it is
a separate follow-up. The open-core NVMM aggregator tier operates entirely without `nvinfer`.

---

## Amendment — 2026-06-09 (PR: open-vocabulary VLM detector tier — LocateAnything-3B)

**What this adds.** A fourth detector tier, `DetectorTier.VLM_SIDECAR`, lets a `kind: detector`
rSkill be backed by an open-vocabulary visual-grounding VLM instead of a fixed-label ONNX model.
The first such skill is `rskills/locateanything-3b-nf4` (`nvidia/LocateAnything-3B`: MoonViT vision
tower + Qwen2.5-3B; emits `<ref>label</ref><box><x1><y1><x2><y2></box>` tokens normalized to
`[0,1000]`). It is selected for manifests with `runtime: pytorch`.

**Why a sidecar (not in-process).** The model's `trust_remote_code` modeling files require
`transformers==4.57.1`; the openral runtime is `transformers>=5` (CLAUDE.md §3), which removed/renamed
the APIs the custom code calls (`config.rope_theta`, GenerationMixin inheritance, the
`_check_and_adjust_attn_implementation` signature). It therefore runs out-of-process in an isolated
`transformers==4.57.1` venv — the same version-isolation pattern as the RLDX-1 sidecar
(`tools/rldx_sidecar.py`). The process boundary is the sole permitted test double (CLAUDE.md §1.11).

**What landed.**
- **`tools/_locateanything_server.py`** — thin ZMQ/msgpack inference server (runs inside the sidecar
  venv): loads the model once with bitsandbytes NF4 (`load_in_4bit`, `nf4`, bf16 compute,
  double-quant), and returns the model's *raw* grounding text. On an 8 GB RTX 4070: 2.9 GB resident,
  ~3.5 GB peak.
- **`tools/locateanything_sidecar.py`** — boot helper: provisions the venv (pinned to the
  nvidia/LocateAnything Space requirements + ZMQ deps) and `execvpe`s the server, dropping
  `PYTHONPATH`/`PYTHONHOME` so the runtime's transformers wheels can't shadow it.
- **`LocateAnythingDetector`** (`openral_runner.backends.gstreamer.locateanything_detector`) — the
  ZMQ client. Exposes the same `detect(frame_bgr, width, height, sensor_id) -> ObjectsMetadata`
  interface as `ObjectsDetector`, so it reuses the `CPU_ONNX` system-memory BGR appsink branch
  unchanged. Connection is lazy (first `detect()`), with an auto-spawn lifecycle mirroring the RLDX
  adapter (ping → `Popen` → poll → `close`). All `<ref>`/`<box>` parsing, degenerate-box filtering,
  and `ObjectsMetadata` construction are pure functions in the main env (`parse_grounding_answer`,
  `build_objects_metadata`) — unit-testable without a GPU.
- **`build_manifest_detector`** (`openral_runner.backends.gstreamer.detector_factory`) — a gi-free
  dispatch seam: `runtime: pytorch` → `LocateAnythingDetector` + `VLM_SIDECAR`; `onnx`/`tensorrt` →
  the existing ONNX detectors via `make_objects_detector`. `DetectorRunner` delegates to it and
  `onnx_path` is now optional (`None` for the VLM tier).

**Open-vocabulary query (static default + dynamic override).** Unlike the fixed-label ONNX
detectors, the query is free text. The static default is the manifest's `detector.labels` (joined
for the grounding prompt); `LocateAnythingDetector.set_query(text)` overrides it at runtime. The
dynamic override is a **perception-side channel**, NOT the actuation path: detectors are
deliberately excluded from the `ExecuteRskillTool` palette (`palette.py` — "perception producers,
not actuators"), so `goal_params_json` never reaches a detector. Instead `RosImageObjectDetectorNode`
subscribes a `std_msgs/String` topic (`/openral/perception/detector_query`) and calls `set_query` on
each message (and honours an initial `query` param). The S2 reasoner retargets the detector by
publishing to that topic; a typed `SetDetectorQueryTool` reasoner-palette variant is the remaining
follow-up.

**deploy-sim integration.** `RosImageObjectDetectorNode` is now manifest-driven: with a
`manifest_path` it builds its backend via `build_manifest_detector` (RT-DETR ONNX or the VLM
sidecar). `openral deploy sim --object-detector-manifest rskills/locateanything-3b-nf4/rskill.yaml
[--object-detector-query "red mug"]` brings the VLM detector up on the camera tee (auto-enables the
leg, no ONNX file needed; throttled to ~0.5 Hz since each VLM frame is ~1–2 s). The legacy RT-DETR
path (no `manifest_path`) is byte-for-byte unchanged.

**Confidence semantics.** LocateAnything is a grounding model: it emits boxes without per-box
scores, so every `ObjectDetection2D` is reported at `confidence=1.0` and `score_threshold` does not
apply. Recorded honestly rather than fabricating a score (CLAUDE.md §1.2).

**Schema.** No `RSkillManifest` change: `runtime: pytorch` was already a valid `RSkillRuntime` and
the `kind: detector` validator already accepts it. `DetectorContract.input_size` is unused by the
VLM tier (LocateAnything resizes dynamically); it is kept only because the contract requires it.

**Hardware note.** The `switzerchees/LocateAnything-3B-NVFP4` variant is Blackwell-only (NVFP4 has
no Ada/Lovelace support); NF4 bitsandbytes is the path for ≤8 GB Ada GPUs. The
`Reza2kn/...-ONNX-WebGPU-INT4` variant targets browser WebGPU and is out of scope for this tier.

**Validation.** `tests/unit/test_locateanything_detector.py` — pure parsing/conversion + manifest
validation + dispatch tests always run; a GPU+sidecar-gated e2e boots the real sidecar and grounds
"cat" on the real `coco_sample.jpg` fixture (≥2 detections), confirms "person" yields none, and
exercises the dynamic `set_query` override. Gated on `OPENRAL_LOCATEANYTHING_SIDECAR_VENV` + a local
GPU (the legitimate CI skip path, CLAUDE.md §12).

**Follow-ups.** (i) a typed `SetDetectorQueryTool` reasoner-palette variant that publishes to
`/openral/perception/detector_query` (the topic + node-side `set_query` already land here; this adds
the LLM-facing tool so the reasoner can retarget the detector mid-task); (ii) publish the pre-packed
NF4 weights to `OpenRAL/rskill-locateanything-3b-nf4` so the sidecar can load them directly instead
of quantizing upstream on each boot.

---

## Amendment — 2026-06-12 (PR: in-process OmDet-Turbo zero-shot detector — `engine: zeroshot_hf`)

**What this adds.** A fifth detector tier, `DetectorTier.ZEROSHOT_HF`, lets a `kind: detector`
rSkill be backed by a first-class Transformers open-vocabulary detector
(`AutoModelForZeroShotObjectDetection`) run **in process** over a **fixed** class vocabulary. The
first such skill is `rskills/omdet-turbo-indoor` (`omlab/omdet-turbo-swin-tiny-hf`: OmDet-Turbo,
Swin-tiny backbone; **Apache-2.0**, commercial-safe — CLAUDE.md §1.9), configured with a curated
~266-class indoor vocabulary (kitchenware, tableware, appliances, furniture, tools, containers).

**Why this tier, and why not the existing ones.** The two RT-DETR rSkills are fixed-label ONNX
detectors capped at the 80 COCO classes — most of which (zebra, surfboard, airplane) never appear
indoors, while indoor staples (mug, drawer, switch, jar, pan) are absent. `locateanything-3b-nf4`
covers open vocabulary but (a) its weights are NVIDIA **non-commercial**, and (b) it is **query
driven** — it grounds a prompt, not "whatever is in the scene". The goal here is to *populate the
world object list* with far more than 80 classes from an **unprompted background producer** under a
**permissive** license. OmDet-Turbo fits: a large fixed vocabulary, evaluated every frame, behaves
like a closed large-vocabulary detector while staying Apache-2.0 and real-time.

**Why in-process (not a sidecar).** Unlike LocateAnything (pinned to `transformers==4.57.1`, hence
the out-of-process venv), OmDet-Turbo is a core `transformers` architecture that loads under the
runtime's own `transformers>=5`. It therefore runs in the runner process — no sidecar venv, no ZMQ.
`torch`/`transformers` are lazy-imported (the `omdet` dependency group) so ONNX-only runners are
unaffected.

**Fixed vocabulary, unprompted (the contrast with LocateAnything).** This backend deliberately has
**no `set_query`** and the detector node does **not** subscribe it to
`/openral/perception/detector_query`. The vocabulary is the manifest `detector.labels`, fixed for
the life of the skill; the reasoner does not retarget it. That is the property that makes it a
"runs in the background, builds the object list" producer rather than a grounding tool.

**What landed.**
- **`DetectorEngine`** (`openral_core.schemas`) — a manifest-level backend discriminator on
  `DetectorContract.engine` (`rtdetr_onnx` / `vlm_sidecar` / `zeroshot_hf`). `None` (default)
  preserves the legacy `runtime`-keyed dispatch, so the existing rtdetr + locateanything manifests
  are untouched; the new rSkill sets `engine: zeroshot_hf` explicitly. This disambiguates two
  backends that share `runtime: pytorch` (the VLM sidecar and the in-process zero-shot detector).
- **`OmDetTurboDetector`** (`openral_runner.backends.gstreamer.omdet_turbo_detector`) — loads the
  processor + model on first `detect()` (CUDA when available, CPU fallback), runs the fixed
  vocabulary via `processor.post_process_grounded_object_detection`, and builds `ObjectsMetadata`.
  All score-thresholding / clipping / degenerate-box filtering is a pure function
  (`build_objects_metadata_from_results`), unit-testable without torch or a GPU. Exposes the same
  `detect(frame_bgr, width, height, sensor_id) -> ObjectsMetadata | None` interface as
  `ObjectsDetector`, so it reuses the `CPU_ONNX` system-memory BGR appsink branch unchanged.
- **`build_manifest_detector`** dispatch gains an `engine: zeroshot_hf` branch → `OmDetTurboDetector`
  + `DetectorTier.ZEROSHOT_HF` (checked before the legacy runtime branches). `DetectorRunner` and
  `RosImageObjectDetectorNode` route through it generically (uniform `detect()` interface), so no
  GStreamer-branch or node changes are needed beyond the new tier reusing the system-memory leg.

**Schema.** `DetectorContract` gains an optional `engine: DetectorEngine | None = None` field
(on-disk `schema_version` stays `"0.1"`, no migrator — CLAUDE.md §6). The `kind: detector` validator
is unchanged: `weights_uri` still required (here `hf://omlab/omdet-turbo-swin-tiny-hf`).

**Validation.** `tests/unit/test_omdet_turbo_detector.py` — pure conversion (threshold / sort / clip
/ degenerate-box / length-mismatch), manifest validation, in-process dispatch (lazy build, no torch),
the `omdet` dependency-group guard, and a regression check that the new engine branch does not shadow
the legacy ONNX dispatch — all run without a GPU. A GPU-gated e2e loads the real Apache-2.0 weights
and grounds indoor classes on `coco_sample.jpg` (skipped on GPU-less hosts, CLAUDE.md §12).

**Follow-ups.** (i) embedding-cache the fixed class text so the per-frame cost is the vision forward
only; (ii) mirror the same `engine: zeroshot_hf` tier for grounding-DINO / OWLv2 if a
higher-accuracy (non-real-time) variant is wanted; (iii) wire the `deploy sim` throttle defaults for
the larger vocabulary.

## Amendment (2026-06-15): omdet-turbo-indoor is the deploy-sim default detector

`openral deploy sim` previously defaulted to the fixed-label RT-DETR COCO ONNX (`rtdetr-coco-r18`)
and auto-enabled the detector leg only when those weights happened to exist on disk. COCO-80 has no
`bread`/`baguette` (or most household objects), so the default detector was structurally blind to the
objects deploy-sim scenes actually contain — a `deploy sim` run would silently bring up a detector
that could never ground the target object. This amendment changes the **default**, not the
contract:

- **Default backend = `omdet-turbo-indoor`** (open-vocab `engine: zeroshot_hf`, 266 indoor/kitchen
  classes incl. `bread`) when its runtime deps (`transformers` + `timm`, the `omdet` group) are
  importable, resolved by `openral_cli.deploy_sim._omdet_runtime_available()`. **Graceful fallback**
  to the in-tree RT-DETR COCO ONNX when they are not, so a base-group checkout still comes up.
- **Detector on by default.** The CLI flag is now `--object-detector/--no-object-detector` (was
  `--enable-object-detector/--no-enable-object-detector`, auto-by-weights). The leg auto-downgrades
  to off (with a console notice) only when neither backend is available — no hard-fail at node build.
  This is the enable-default half of ADR-0035's detector leg.
- **Engine-aware throttle** (resolves follow-up iii): `sim_e2e.launch.py` caps the publish rate by
  `DetectorContract.engine` — `vlm_sidecar` 0.5 Hz, `zeroshot_hf` 2 Hz, RT-DETR ONNX 5 Hz — instead
  of the prior `runtime == "pytorch" → 0.5 Hz` heuristic that throttled the in-process OmDet backend
  as slowly as the LocateAnything VLM sidecar.

Explicit `--object-detector-manifest` / `--object-detector-onnx` overrides are unchanged. No schema
change; on-disk `schema_version` stays `"0.1"`. Arbitrary-object grounding (e.g. `baguette`) still
needs the on-demand open-vocab locator (`locate_in_view`, ADR-0043/0051) — a separate work item.
