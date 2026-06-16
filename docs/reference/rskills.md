# rSkills Reference

rSkills are HuggingFace-Hub-shaped packages — manifest + weights + reproducible `eval/` — loaded via `rSkill.from_pretrained(...)` and gated by license, embodiment tags, and capabilities. See [CLAUDE.md §3](https://github.com/OpenRAL/openral/blob/master/CLAUDE.md) for packaging details.

## Install & manage

```bash
openral rskill install OpenRAL/rskill-smolvla-libero
openral rskill list                # list installed rSkills
openral rskill check               # which installed rSkills run on this host?
```

## VLA policy rSkills

All entries are published under `OpenRAL/rskill-*` on HuggingFace Hub and exercised end-to-end by a config in [`scenes/`](../../scenes/).

| rSkill | Backbone / family | Targets | License |
|---|---|---|---|
| [`smolvla-libero`](../../rskills/smolvla-libero/) | SmolVLA fine-tuned | `franka_panda` | Apache-2.0 |
| [`smolvla-metaworld`](../../rskills/smolvla-metaworld/) | SmolVLA fine-tuned | `sawyer` | Apache-2.0 |
| [`smolvla-maniskill-franka`](../../rskills/smolvla-maniskill-franka/) | SmolVLA × ManiSkill3 `PickCube-v1` | `franka_panda` | Apache-2.0 |
| [`xvla-libero`](../../rskills/xvla-libero/) | xVLA (Florence-2) | `franka_panda` | Apache-2.0 |
| [`act-libero`](../../rskills/act-libero/) | ACT | `franka_panda` | Apache-2.0 |
| [`molmoact2-libero-nf4`](../../rskills/molmoact2-libero-nf4/) | MolmoAct2 NF4 (Molmo2-ER VLM + flow-matching, ~5.5 B) | `franka_panda` | Apache-2.0 |
| [`pi05-libero-nf4`](../../rskills/pi05-libero-nf4/) | π0.5 NF4 | `franka_panda` | Permissive research (weights non-Apache) |
| [`pi05-openarm-vision-nf4`](../../rskills/pi05-openarm-vision-nf4/) | π0.5 NF4 (bimanual, absolute joint targets) | `openarm` | Apache-2.0 |
| [`pi05-robocasa365-human300-nf4`](../../rskills/pi05-robocasa365-human300-nf4/) | π0.5 NF4 (RoboCasa365 + 300 human eps fine-tune) | `panda_mobile` | Apache-2.0 (weights: research) |
| [`pi05-so101-pickplace-nf4`](../../rskills/pi05-so101-pickplace-nf4/) | π0.5 NF4 (SO-101 pick-place fine-tune, 3 RGB streams) | `so101_follower` | Apache-2.0 |
| [`act-aloha`](../../rskills/act-aloha/) | ACT (Action Chunking Transformer) | `aloha_bimanual` | MIT |
| [`act-aloha-insertion`](../../rskills/act-aloha-insertion/) | ACT insertion checkpoint — *custom example* | `aloha_bimanual` | MIT |
| [`diffusion-pusht`](../../rskills/diffusion-pusht/) | Diffusion Policy | `pusht_2d` | Apache-2.0 |
| [`rldx1-ft-libero-nf4`](../../rskills/rldx1-ft-libero-nf4/) | RLWRLD RLDX-1 (Qwen3-VL-8B + MSAT, ~6.9 B) | `franka_panda` | RLWRLD non-commercial — sidecar runtime |
| [`rldx1-ft-gr1-nf4`](../../rskills/rldx1-ft-gr1-nf4/) | RLDX-1 (GR1 bimanual) | `gr1` | RLWRLD non-commercial — sidecar runtime |
| [`rldx1-ft-rc365-nf4`](../../rskills/rldx1-ft-rc365-nf4/) | RLDX-1 (RoboCasa365 fine-tune) | `panda_mobile` | RLWRLD non-commercial — sidecar runtime |
| [`rldx1-ft-simpler-widowx-nf4`](../../rskills/rldx1-ft-simpler-widowx-nf4/) | RLDX-1 (SimplerEnv WidowX fine-tune) | `widowx` | RLWRLD non-commercial — sidecar runtime |
| [`rldx1-pt-nf4`](../../rskills/rldx1-pt-nf4/) | RLDX-1 (foundation pretrain) | `franka_panda` | RLWRLD non-commercial — sidecar runtime |
| [`gr00t-n17-libero`](../../rskills/gr00t-n17-libero/) | NVIDIA Isaac GR00T N1.7 (3B, Cosmos-Reason2-2B VLM backbone) | `franka_panda` | NVIDIA Open Model License (commercial OK) — out-of-process sidecar (ADR-0046) |

## Perception rSkills (`kind: detector`)

Object-detection rSkills emit `ObjectsMetadata` (2-D detections lifted to 3-D in the deploy graph) instead of an `Action`. See [ADR-0035](../adr/0035-perception-spatial-memory-object-lift.md) and [ADR-0037](../adr/0037-gstreamer-perception-bus-object-detection.md).

| rSkill | Backbone | Notes |
|---|---|---|
| [`rtdetr-coco-r18`](../../rskills/rtdetr-coco-r18/) | RT-DETR R18 (COCO) | lightweight ONNX export |
| [`rtdetr-v2-r50vd`](../../rskills/rtdetr-v2-r50vd/) | RT-DETR v2 R50vd | higher-accuracy variant |
| [`locateanything-3b-nf4`](../../rskills/locateanything-3b-nf4/) | NVIDIA LocateAnything-3B NF4 | open-vocabulary grounding; runs via the `VLM_SIDECAR` detector tier (out-of-process sidecar); dynamic reasoner-driven query via the read-only `locate_in_view` tool (ADR-0043) |
| [`omdet-turbo-indoor`](../../rskills/omdet-turbo-indoor/) | OmDet-Turbo Swin-tiny (`omlab/omdet-turbo-swin-tiny-hf`) | **Apache-2.0** open-vocabulary detector run **in-process** over a fixed ~266-class curated indoor vocabulary; `engine: zeroshot_hf` → `DetectorTier.ZEROSHOT_HF`; `mode: continuous` background producer, far more than the 80 COCO classes (ADR-0037 2026-06-12 amendment) |
| [`omdet-turbo-locator`](../../rskills/omdet-turbo-locator/) | OmDet-Turbo Swin-tiny (`omlab/omdet-turbo-swin-tiny-hf`) | **Apache-2.0** on-demand sibling — same weights/engine, `mode: on_demand`; the reasoner prompts it via `locate_in_view`. Lightweight, real-time, in-process alternative to the 3B LocateAnything VLM (ADR-0051) |

The RT-DETR rSkills are Apache-2.0 and runnable on any camera-equipped embodiment. They are consumed by `openral_perception_ros` (`RosImageObjectDetectorNode`) in the `openral deploy sim` / `deploy run` graph. LocateAnything is NVIDIA non-commercial and ships as an NF4 PyTorch/custom-code artifact. Because its custom code needs `transformers==4.57.1` (incompatible with the runtime's `transformers>=5`), it runs **out-of-process** in an isolated venv (`tools/locateanything_sidecar.py`) and is driven by the `LocateAnythingDetector` backend over ZMQ — the `DetectorTier.VLM_SIDECAR` path selected for `runtime: pytorch` detector manifests (ADR-0037 2026-06-09 amendment). The detector-node side of that ZMQ link needs the `pyzmq` + `msgpack` client, shipped in the `locateanything` dependency group (`uv sync --group locateanything`); without it the `deploy sim --object-detector-manifest` leg fails per-request with `No module named 'zmq'`. The backend parses its `<ref>`/`<box>` text into `ObjectsMetadata` and exposes `set_query()` for the open-vocabulary query (static default = manifest `labels`; dynamic override via the `/openral/perception/detector_query` topic for the continuous leg, and the read-only `locate_in_view` reasoner tool + service for a one-shot on-demand check, ADR-0043).

`omdet-turbo-indoor` is the **commercially-permissive** open-vocabulary alternative: OmDet-Turbo is a first-class `transformers` architecture (`AutoModelForZeroShotObjectDetection`), so it loads under the runtime's own `transformers>=5` and runs **in-process** (no sidecar, no ZMQ) via the `OmDetTurboDetector` backend — the `DetectorTier.ZEROSHOT_HF` path selected when `detector.engine` is `zeroshot_hf` (ADR-0037 2026-06-12 amendment). Unlike LocateAnything it is **not** query-driven: it has no `set_query` and the detector node does not subscribe it to `/openral/perception/detector_query`. Its fixed ~266-class indoor vocabulary (manifest `labels`) is evaluated on every frame, so it acts as an unprompted background producer that populates the world object list with far more than the 80 COCO classes. `torch` + `transformers` ship in the `omdet` dependency group (`uv sync --group omdet`); without them the in-process backend fails on first `detect()`.

### Invocation mode: continuous vs on-demand (ADR-0051)

Detectors carry a `detector.mode` (`DetectorMode`) that is **orthogonal** to `detector.engine` — where `engine` says *how* the model runs, `mode` says *when the reasoner invokes it*:

- **`continuous`** (default) — an always-on background producer (`rtdetr-coco-r18`, `rtdetr-v2-r50vd`, `omdet-turbo-indoor`). Runs on the camera tee every frame, streams `ObjectsMetadata` into `WorldState.detected_objects`; the reasoner reads it **passively** (world state / `recall_object`) and never prompts it. It is not an ExecuteSkill tool. `build_tool_palette` collects these into `ToolPalette.continuous_detectors` so the LLM is told *what is already tracked for free*.
- **`on_demand`** — a prompted open-vocab locator (`locateanything-3b-nf4`, `omdet-turbo-locator`). The reasoner invokes it via the read-only `locate_in_view` tool (ADR-0043) only when it needs a specific object **right now** that the continuous bank doesn't cover. `omdet-turbo-locator` wraps the **same** Apache-2.0 OmDet-Turbo weights as `omdet-turbo-indoor` but in on-demand mode — a lightweight (~115M, real-time, in-process) alternative to the 3B LocateAnything VLM for simple "find X" queries; LocateAnything stays the higher-quality option for complex referring expressions. The `OmDetTurboDetector` backend exposes `set_query` / `detect_with_query`, which the detector node binds by `hasattr`, so the same backend serves either mode (packaging two single-purpose rSkills, not one dual-mode rSkill, is what keeps the modes from straddling).

This cleanly separates open-vocabulary from prompting: the `locate_in_view` tool description is made coverage-aware (it lists the continuous detectors' class counts + keywords), so the LLM's rule is mechanical — *object within continuous coverage → read world state; object outside it (novel / attribute-qualified) → `locate_in_view`*.

## Scene-VLM rSkills (`kind: vlm`)

`kind: vlm` rSkills (ADR-0047) are vision/video-language models used as **S2 scene-understanding** components: they answer open-ended natural-language questions about the current camera view (task-progress / success verification — "has the robot grasped the mug?", "did we drop the object?", "is the task complete?") and emit **text**, never actions. They are not localizers — use a `kind: detector` rSkill (`locate_in_view`) to find *where* an object is. A scene VLM is reached through the read-only `query_scene` reasoner tool, never `ExecuteSkill` (so `role: s2`, excluded from the actuation palette by design).

| rSkill | Backbone | Notes |
|---|---|---|
| [`qwen35-4b-nf4`](../../rskills/qwen35-4b-nf4/) | Qwen3.5-4B NF4 (natively-multimodal, hybrid linear attention) | Apache-2.0; pre-quantized NF4 checkpoint (~3.3 GB, fits 8 GB); runs out-of-process via `tools/qwen_vlm_sidecar.py` + the `QwenSceneVlm` backend over ZMQ; served by `openral_perception_ros.scene_vlm_node` on `/openral/perception/query_scene`; drives the reasoner's read-only `query_scene` tool (ADR-0047) |

Like the LocateAnything detector, the Qwen scene VLM runs in an **isolated sidecar venv** (its bitsandbytes / `qwen-vl-utils` / Gated-DeltaNet stack would perturb the lerobot-pinned `transformers==5.3.0` runtime, and a 4B model + CUDA context should not share the `rclpy` process). The node-side ZMQ + msgpack client ships in the `qwen-vlm` dependency group (`uv sync --group qwen-vlm`). The rSkill's `weights_uri` is a **pre-quantized** NF4 checkpoint (transformers-native `save_pretrained` layout with an embedded `quantization_config`) built by `tools/build_qwen_vlm_nf4_checkpoint.py`; it loads directly as 4-bit with no bf16 load spike. `source_repo` records the SHA-pinned upstream Apache-2.0 Qwen model (provenance). The reasoner offers `query_scene` when launched with `scene_query_available:=true`.

## Reward-monitor rSkills (`kind: reward`)

`kind: reward` rSkills (ADR-0057) are robotic **reward / progress-monitor** models that run **in parallel with a VLA policy** and score the rollout: given the VLA's camera frames + the task instruction, they emit **per-frame normalized progress (0–1)** and **per-frame success probability**. Where a scene VLM (`query_scene`) returns free text, a reward monitor returns quantitative scalars + trends. It is reached through the read-only `query_task_progress` reasoner tool, never `ExecuteSkill` (so `role: s2`, excluded from the actuation palette); the signal is **advisory** — it feeds the replanning ladder, never the motors.

| rSkill | Backbone | Notes |
|---|---|---|
| [`robometer-4b`](../../rskills/robometer-4b/) | Robometer-4B (Qwen3-VL-4B reward foundation model, arXiv 2603.02115) | **Apache-2.0**; NF4-quantized on load (~3.3 GB resident, 3.56 GB peak, fits 8 GB alongside a small VLA); runs out-of-process via `tools/robometer_sidecar.py` + the `RobometerReward` backend over ZMQ; served by `openral_perception_ros.reward_monitor_node` on `/openral/perception/query_task_progress`; drives the reasoner's read-only `query_task_progress` tool (ADR-0057) |

Like the Qwen scene VLM, the Robometer monitor runs in an **isolated sidecar venv**: its `RBM` class cannot be loaded by vanilla `transformers.AutoModel` (its HF `config.json` advertises `architectures: ["RFM"]` with no `auto_map`), so the sidecar `uv pip install`s the pinned upstream `robometer` package and **forces `transformers==4.57.1`** (the resolver pulls 5.x, which drops `input_ids` from the processor). The node-side ZMQ + msgpack client ships in the `robometer` dependency group (`uv sync --group robometer`). The sidecar is a **stateless scorer**; the rolling frame buffer (`RollingFrameBuffer`, fed by the same `sensor_msgs/Image` topic the VLA uses — GStreamer tee on real hardware, sim HAL publisher in `deploy-sim`) lives node-side. `weights_uri` accepts the SHA-pinned Apache-2.0 upstream (NF4-quantized on load), a published pre-quantized OpenRAL repo, or `local:///abs/dir`. The pre-quantized path (built by `tools/build_robometer_nf4_checkpoint.py`) loads the packed NF4 weights DIRECTLY on the `meta` device — no bf16 materialization, no requantize (~25 s process→ready vs ~110 s + a 19 GB CPU spike), bit-identical to the bf16+quantize path with determinism pinned (math SDP + `use_deterministic_algorithms` + `CUBLAS_WORKSPACE_CONFIG`). The forward's activation memory scales with frame count × resolution, so the client evenly subsamples the window to `max_frames` (8) to stay co-resident with the sim (and a small NF4 VLA) on 8 GB. In `deploy-sim`, `openral deploy sim --enable-reward-monitor` brings the monitor up parallel to the VLA and sets `task_progress_available:=true` so the reasoner is offered `query_task_progress` (validated live on openarm). The upstream `robometer` code is not an OpenRAL-trusted org — it is pinned by commit and runs only in the isolated venv.

## Manifest format

```yaml
# rskills/smolvla-libero/rskill.yaml (excerpt)
name: "OpenRAL/rskill-smolvla-libero"
version: "0.1.0"
license: "apache-2.0"
role: "s1"
embodiment_tags: ["franka_panda"]
sensors_required:
  - modality: "rgb"
    vla_feature_key: "observation.images.camera1"
  # … second RGB stream + proprioception
```

## License notes

See [CLAUDE.md §3](https://github.com/OpenRAL/openral/blob/master/CLAUDE.md) for the full VLA license matrix. Key restrictions:
- **GR00T-based checkpoints** — license is version-specific (ADR-0046): N1/N1.5/N1.6 are NVIDIA OneWay Noncommercial (`nvidia_non_commercial`; requires `OPENRAL_ALLOW_NONCOMMERCIAL=1`); N1.7+ are NVIDIA Open Model License (`nvidia_open_model`, commercial OK).
- **LocateAnything-3B** — NVIDIA License, non-commercial research/evaluation; private HF rSkill, requires `OPENRAL_ALLOW_NONCOMMERCIAL=1` and remote-code acceptance.
- **π0 / π0.5 weights** — permissive research (not full Apache-2.0 for commercial deployment).
- **RLDX-1** — RLWRLD non-commercial; runs as an out-of-process sidecar.

Provenance signing via sigstore is planned but not yet implemented — the loader emits an `rskill.unverified_provenance` warning on every load. Set `OPENRAL_REQUIRE_SIGNED_SKILLS=1` to fail closed.
