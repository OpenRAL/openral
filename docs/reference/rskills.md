# rSkills Reference

rSkills are HuggingFace-Hub-shaped packages — manifest + weights + reproducible `eval/` — loaded via `rSkill.from_pretrained(...)` and gated by license, embodiment tags, and capabilities. See [CLAUDE.md §3](https://github.com/OpenRAL/openral/blob/master/CLAUDE.md) for packaging details.

## Install & manage

```bash
openral rskill search aloha                    # discover skills on the OpenRAL Hub org
openral rskill search --kind detector          # …filter by kind/role/embodiment/license
openral rskill install OpenRAL/rskill-smolvla-franka_panda-libero_spatial-bf16
openral rskill list                # list installed rSkills
openral rskill check               # which installed rSkills run on this host?
```

`rskill install` expects an `org/name` Hub id. A bare name (e.g. `rskill-smolvla-franka_panda-libero_spatial-bf16`)
fails fast with the canonical `OpenRAL/…` suggestion rather than a raw Hub 404 — use
`rskill search` if you don't know the id. `rskill search` queries the OpenRAL HF Hub org
(`HfApi.list_models`), validates each candidate's `rskill.yaml`, and prints a paste-able
`repo_id` table.

## Discovery views (`SKILL.md`)

Each `rskills/<id>/` also carries a generated `SKILL.md` — a standard agent-skill
view (YAML `name` + `description` frontmatter, rSkill fields under `metadata:`) so
tools that read the agent-skill format can **discover** OpenRAL rSkills. It is
**discovery-only**: an agent can use it to *select* a skill, but execution always
goes through `rSkill.from_pretrained` + the robot HAL — markdown cannot hold
weights or the typed capability/license gates. `rskill.yaml` remains the single
source of truth (CLAUDE.md §1.3); regenerate (never hand-edit) with:

```bash
python tools/generate_rskill_skillmd.py            # all rskills/<id>/
python tools/generate_rskill_skillmd.py --check    # CI: fail if any are stale
```

The same `SKILL.md` is mirrored to each `OpenRAL/rskill-*` Hub repo.

## VLA policy rSkills

All entries are published under `OpenRAL/rskill-*` on HuggingFace Hub and exercised end-to-end by a config in [`scenes/`](https://github.com/OpenRAL/openral/tree/master/scenes/).

| rSkill | Backbone / family | Targets | License |
|---|---|---|---|
| [`smolvla-libero`](https://github.com/OpenRAL/openral/tree/master/rskills/smolvla-libero/) | SmolVLA fine-tuned | `franka_panda` | Apache-2.0 |
| [`smolvla-metaworld`](https://github.com/OpenRAL/openral/tree/master/rskills/smolvla-metaworld/) | SmolVLA fine-tuned | `sawyer` | Apache-2.0 |
| [`smolvla-maniskill-franka`](https://github.com/OpenRAL/openral/tree/master/rskills/smolvla-maniskill-franka/) | SmolVLA × ManiSkill3 `PickCube-v1` | `franka_panda` | Apache-2.0 |
| [`xvla-libero`](https://github.com/OpenRAL/openral/tree/master/rskills/xvla-libero/) | xVLA (Florence-2) | `franka_panda` | Apache-2.0 |
| [`act-libero`](https://github.com/OpenRAL/openral/tree/master/rskills/act-libero/) | ACT | `franka_panda` | Apache-2.0 |
| [`act-so101-pen`](https://github.com/OpenRAL/openral/tree/master/rskills/act-so101-pen/) | ACT pen checkpoint | `so101_follower` | Apache-2.0 |
| [`molmoact2-libero-nf4`](https://github.com/OpenRAL/openral/tree/master/rskills/molmoact2-libero-nf4/) | MolmoAct2 NF4 (Molmo2-ER VLM + flow-matching, ~5.5 B) | `franka_panda` | Apache-2.0 |
| [`molmoact2-so101-nf4`](https://github.com/OpenRAL/openral/tree/master/rskills/molmoact2-so101-nf4/) | MolmoAct2 NF4 SO-101 checkpoint | `so101_follower` | Apache-2.0 |
| [`pi05-libero-int8`](https://github.com/OpenRAL/openral/tree/master/rskills/pi05-libero-int8/) | π0.5 NF4 | `franka_panda` | Permissive research (weights non-Apache) |
| [`act-aloha`](https://github.com/OpenRAL/openral/tree/master/rskills/act-aloha/) | ACT (Action Chunking Transformer) | `aloha_bimanual` | MIT |
| [`act-aloha-insertion`](https://github.com/OpenRAL/openral/tree/master/rskills/act-aloha-insertion/) | ACT insertion checkpoint — *custom example* | `aloha_bimanual` | MIT |
| [`diffusion-pusht`](https://github.com/OpenRAL/openral/tree/master/rskills/diffusion-pusht/) | Diffusion Policy | `pusht_2d` | Apache-2.0 |
| [`3d-diffuser-actor-rlbench`](https://github.com/OpenRAL/openral/tree/master/rskills/3d-diffuser-actor-rlbench/) | 3D Diffuser Actor keyframe policy for RLBench PerAct tasks | `franka_panda` | MIT — CoppeliaSim/PyRep sidecar runtime |
| [`rldx1-ft-libero-nf4`](https://github.com/OpenRAL/openral/tree/master/rskills/rldx1-ft-libero-nf4/) | RLWRLD RLDX-1 (Qwen3-VL-8B + MSAT, ~6.9 B) | `franka_panda` | RLWRLD non-commercial — sidecar runtime |
| [`rldx1-ft-gr1-nf4`](https://github.com/OpenRAL/openral/tree/master/rskills/rldx1-ft-gr1-nf4/) | RLDX-1 (GR1 bimanual) | `gr1` | RLWRLD non-commercial — sidecar runtime |
| [`rldx1-ft-rc365-nf4`](https://github.com/OpenRAL/openral/tree/master/rskills/rldx1-ft-rc365-nf4/) | RLDX-1 (RoboCasa365 fine-tune) | `panda_mobile` | RLWRLD non-commercial — sidecar runtime |
| [`rldx1-ft-simpler-widowx-nf4`](https://github.com/OpenRAL/openral/tree/master/rskills/rldx1-ft-simpler-widowx-nf4/) | RLDX-1 (SimplerEnv WidowX fine-tune) | `widowx` | RLWRLD non-commercial — sidecar runtime |
| [`openvla-oft-simpler-widowx-nf4`](https://github.com/OpenRAL/openral/tree/master/rskills/openvla-oft-simpler-widowx-nf4/) | OpenVLA-OFT (RLinf PPO ManiSkill3 PutOnPlateInScene25; NF4) | `widowx` | MIT — transformers custom-code; validated 2/5 on SimplerEnv carrot |
| [`gr00t-n17-libero`](https://github.com/OpenRAL/openral/tree/master/rskills/gr00t-n17-libero/) | NVIDIA Isaac GR00T N1.7 (3B, Cosmos-Reason2-2B VLM backbone) | `franka_panda` | NVIDIA Open Model License (commercial OK) — in-process lerobot 0.6.0 `GrootPolicy`, backbone-only NF4 |
| [`gr00t-n17-so101-fruit`](https://github.com/OpenRAL/openral/tree/master/rskills/gr00t-n17-so101-fruit/) | GR00T N1.7 (3B) SO-101 fruit pick-and-place (`new_embodiment`, 6-D) | `so101_follower` | NVIDIA Open Model License (commercial OK) — in-process `GrootPolicy` whole-model NF4 (`quantize_scope: model`); GPU-verified 5.8 GiB peak on 8 GB |
| [`smolvla-so101-pen`](https://github.com/OpenRAL/openral/tree/master/rskills/smolvla-so101-pen/) | SmolVLA SO-101 pen checkpoint | `so101_follower` | Apache-2.0 |
| [`smolvla-so101-pick-place-pen`](https://github.com/OpenRAL/openral/tree/master/rskills/smolvla-so101-pick-place-pen/) | SmolVLA SO-101 pick/place pen checkpoint; optional split ONNX/TensorRT fast path | `so101_follower` | Apache-2.0 |
| [`rskill-smolvla-so101-eraser_place-bf16`](https://github.com/OpenRAL/openral/tree/master/rskills/rskill-smolvla-so101-eraser_place-bf16/) | SmolVLA SO-101 "place the erase on the blue square" checkpoint (makermods, 25 real teleop episodes) | `so101_follower` | Apache-2.0 — weights stay upstream; chunk verified against the training data |
| [`smolvla-robotwin`](https://github.com/OpenRAL/openral/tree/master/rskills/smolvla-robotwin/) | SmolVLA finetuned on RoboTwin 2.0 (50 bimanual SAPIEN tasks) | `aloha_agilex` | Apache-2.0 — py3.10 SAPIEN sidecar |
| [`smolvla-vlabench`](https://github.com/OpenRAL/openral/tree/master/rskills/smolvla-vlabench/) | SmolVLA finetuned on VLABench (`lerobot/vlabench_unified`, 97 tasks) — integration baseline, 0% on current tasks | `franka_panda` | Apache-2.0 |

## Perception rSkills (`kind: detector`)

Object-detection rSkills emit `ObjectsMetadata` (2-D detections lifted to 3-D in the deploy graph) instead of an `Action`.

| rSkill | Backbone | Notes |
|---|---|---|
| [`rtdetr-coco-r18`](https://github.com/OpenRAL/openral/tree/master/rskills/rtdetr-coco-r18/) | RT-DETR R18 (COCO) | lightweight ONNX export |
| `rtdetr-v2-r50vd` | RT-DETR v2 R50vd | higher-accuracy variant; `runtime: tensorrt` — moved to the private `openral-pro` repo since it depends on the TensorRT engine runtime |
| [`locateanything-3b-nf4`](https://github.com/OpenRAL/openral/tree/master/rskills/locateanything-3b-nf4/) | NVIDIA LocateAnything-3B NF4 | open-vocabulary grounding; runs via the `VLM_SIDECAR` detector tier (out-of-process sidecar); dynamic reasoner-driven query via the read-only `locate_in_view` tool |
| [`omdet-turbo-indoor`](https://github.com/OpenRAL/openral/tree/master/rskills/omdet-turbo-indoor/) | OmDet-Turbo Swin-tiny (`omlab/omdet-turbo-swin-tiny-hf`) | **Apache-2.0** open-vocabulary detector run **in-process** over a fixed ~266-class curated indoor vocabulary; `engine: zeroshot_hf` → `DetectorTier.ZEROSHOT_HF`; `mode: continuous` background producer, far more than the 80 COCO classes (as of the 2026-06-12 amendment) |
| [`omdet-turbo-locator`](https://github.com/OpenRAL/openral/tree/master/rskills/omdet-turbo-locator/) | OmDet-Turbo Swin-tiny (`omlab/omdet-turbo-swin-tiny-hf`) | **Apache-2.0** on-demand sibling — same weights/engine, `mode: on_demand`; the reasoner prompts it via `locate_in_view`. Lightweight, real-time, in-process alternative to the 3B LocateAnything VLM |

The RT-DETR rSkills are Apache-2.0 and runnable on any camera-equipped embodiment. They are consumed by `openral_perception_ros` (`RosImageObjectDetectorNode`) in the `openral deploy sim` / `deploy run` graph. LocateAnything is NVIDIA non-commercial and ships as an NF4 PyTorch/custom-code artifact. Because its custom code needs `transformers==4.57.1` (incompatible with the runtime's `transformers>=5`), it runs **out-of-process** in an isolated venv (`tools/locateanything_sidecar.py`) and is driven by the `LocateAnythingDetector` backend over ZMQ — the `DetectorTier.VLM_SIDECAR` path selected for `runtime: pytorch` detector manifests (as of the 2026-06-09 amendment). The detector-node side of that ZMQ link needs the `pyzmq` + `msgpack` client, shipped in the `locateanything` dependency group (`uv sync --group locateanything`); without it the `deploy sim --object-detector-manifest` leg fails per-request with `No module named 'zmq'`. The backend parses its `<ref>`/`<box>` text into `ObjectsMetadata` and exposes `set_query()` for the open-vocabulary query (static default = manifest `labels`; dynamic override via the `/openral/perception/detector_query` topic for the continuous leg, and the read-only `locate_in_view` reasoner tool + service for a one-shot on-demand check).

`omdet-turbo-indoor` is the **commercially-permissive** open-vocabulary alternative: OmDet-Turbo is a first-class `transformers` architecture (`AutoModelForZeroShotObjectDetection`), so it loads under the runtime's own `transformers>=5` and runs **in-process** (no sidecar, no ZMQ) via the `OmDetTurboDetector` backend — the `DetectorTier.ZEROSHOT_HF` path selected when `detector.engine` is `zeroshot_hf` (as of the 2026-06-12 amendment). Unlike LocateAnything it is **not** query-driven: it has no `set_query` and the detector node does not subscribe it to `/openral/perception/detector_query`. Its fixed ~266-class indoor vocabulary (manifest `labels`) is evaluated on every frame, so it acts as an unprompted background producer that populates the world object list with far more than the 80 COCO classes. `torch` + `transformers` ship in the `omdet` dependency group (`uv sync --group omdet`); without them the in-process backend fails on first `detect()`.

### Invocation mode: continuous vs on-demand

Detectors carry a `detector.mode` (`DetectorMode`) that is **orthogonal** to `detector.engine` — where `engine` says *how* the model runs, `mode` says *when the reasoner invokes it*:

- **`continuous`** (default) — an always-on background producer (`rtdetr-coco-r18`, `rtdetr-v2-r50vd`, `omdet-turbo-indoor`). Runs on the camera tee every frame, streams `ObjectsMetadata` into `WorldState.detected_objects`; the reasoner reads it **passively** (world state / `recall_object`) and never prompts it. It is not an ExecuteRskill tool. `build_tool_palette` collects these into `ToolPalette.continuous_detectors` so the LLM is told *what is already tracked for free*.
- **`on_demand`** — a prompted open-vocab locator (`locateanything-3b-nf4`, `omdet-turbo-locator`). The reasoner invokes it via the read-only `locate_in_view` tool only when it needs a specific object **right now** that the continuous bank doesn't cover. `omdet-turbo-locator` wraps the **same** Apache-2.0 OmDet-Turbo weights as `omdet-turbo-indoor` but in on-demand mode — a lightweight (~115M, real-time, in-process) alternative to the 3B LocateAnything VLM for simple "find X" queries; LocateAnything stays the higher-quality option for complex referring expressions. The `OmDetTurboDetector` backend exposes `set_query` / `detect_with_query`, which the detector node binds by `hasattr`, so the same backend serves either mode (packaging two single-purpose rSkills, not one dual-mode rSkill, is what keeps the modes from straddling).

This cleanly separates open-vocabulary from prompting: the `locate_in_view` tool description is made coverage-aware (it lists the continuous detectors' class counts + keywords), so the LLM's rule is mechanical — *object within continuous coverage → read world state; object outside it (novel / attribute-qualified) → `locate_in_view`*.

## Scene-VLM rSkills (`kind: vlm`)

`kind: vlm` rSkills are vision/video-language models used as **S2 scene-understanding** components: they answer open-ended natural-language questions about the current camera view (task-progress / success verification — "has the robot grasped the mug?", "did we drop the object?", "is the task complete?") and emit **text**, never actions. They are not localizers — use a `kind: detector` rSkill (`locate_in_view`) to find *where* an object is. A scene VLM is reached through the read-only `query_scene` reasoner tool, never `ExecuteRskill` (so `role: s2`, excluded from the actuation palette by design).

| rSkill | Backbone | Notes |
|---|---|---|
| [`qwen35-4b-nf4`](https://github.com/OpenRAL/openral/tree/master/rskills/qwen35-4b-nf4/) | Qwen3.5-4B NF4 (natively-multimodal, hybrid linear attention) | Apache-2.0; pre-quantized NF4 checkpoint (~3.3 GB, fits 8 GB); runs out-of-process via `tools/qwen_vlm_sidecar.py` + the `QwenSceneVlm` backend over ZMQ; served by `openral_perception_ros.scene_vlm_node` on `/openral/perception/query_scene`; drives the reasoner's read-only `query_scene` tool |

Like the LocateAnything detector, the Qwen scene VLM runs in an **isolated sidecar venv** (its bitsandbytes / `qwen-vl-utils` / Gated-DeltaNet stack would perturb the lerobot-pinned `transformers>=5.4.0,<5.14.0` runtime, and a 4B model + CUDA context should not share the `rclpy` process). The node-side ZMQ + msgpack client ships in the `qwen-vlm` dependency group (`uv sync --group qwen-vlm`). The rSkill's `weights_uri` is a **pre-quantized** NF4 checkpoint (transformers-native `save_pretrained` layout with an embedded `quantization_config`) built by `tools/build_qwen_vlm_nf4_checkpoint.py`; it loads directly as 4-bit with no bf16 load spike. `source_repo` records the SHA-pinned upstream Apache-2.0 Qwen model (provenance). The reasoner offers `query_scene` when launched with `scene_query_available:=true`.

## Reward-monitor rSkills (`kind: reward`)

`kind: reward` rSkills are robotic **reward / progress-monitor** models that run **in parallel with a VLA policy** and score the rollout: given the VLA's camera frames + the task instruction, they emit **per-frame normalized progress (0–1)** and **per-frame success probability**. Where a scene VLM (`query_scene`) returns free text, a reward monitor returns quantitative scalars + trends. It is reached through the read-only `query_task_progress` reasoner tool, never `ExecuteRskill` (so `role: s2`, excluded from the actuation palette); the signal is **advisory** — it feeds the replanning ladder, never the motors.

| rSkill | Backbone | Notes |
|---|---|---|
| [`robometer-4b`](https://github.com/OpenRAL/openral/tree/master/rskills/robometer-4b/) | Robometer-4B (Qwen3-VL-4B reward foundation model, arXiv 2603.02115) | **Apache-2.0**; pre-quantized NF4 at `hf://OpenRAL/rskill-robometer_4b-any-general-nf4`, meta-loaded directly as 4-bit (~3.3 GB resident, fits 8 GB alongside a small VLA); runtime is in-process inside `openral_perception_ros.reward_monitor_node` via `RobometerInProcessReward`; served on `/openral/perception/query_task_progress`; drives the reasoner's read-only `query_task_progress` tool. |
| [`topreward-qwen3vl-4b-nf4`](https://github.com/OpenRAL/openral/tree/master/rskills/topreward-qwen3vl-4b-nf4/) | TOPReward (arXiv 2602.19313) — **zero-shot** reward on `Qwen/Qwen3-VL-4B-Instruct` via lerobot 0.6.0's first-party `lerobot.rewards.topreward` | **Apache-2.0** packaging (method MIT; Qwen3-VL weights keep the Qwen license). Zero-shot: no fine-tuned checkpoint — the model reads `log P("True" \| video, instruction)`; **per-frame progress (0–1)** comes from lerobot's native prefix sweep + per-episode min-max normalization. NF4 (bitsandbytes) on the base VLM to fit 8 GB — **measured 3.13 GB peak** on RTX 4070 Laptop, live-validated on LIBERO `libero_object` ep0 (first-20% 0.41 → last-20% 0.92). Runtime backend still to land as a follow-up |

Robometer runs **in-process inside the reward-monitor ROS node**. That still keeps it out of the VLA runner, reasoner, and HAL processes used by `deploy-sim` / `deploy-run`, but avoids an extra ZMQ process boundary now that lerobot 0.6.0 ships `lerobot.rewards.robometer.RobometerRewardModel` directly. There is **no** pinned `robometer` package, **no** `transformers==4.57.1` force-pin, and **no** dedicated venv requirement. The rolling frame buffer (`RollingFrameBuffer`, fed by the selected `sensor_msgs/Image` topic) lives node-side; `deploy-sim` selects the first RGB camera from `robot.yaml` and falls back to `agentview_left`. `weights_uri` accepts the published pre-quantized OpenRAL repo or `local:///abs/dir`. The pre-quantized path (built by `tools/build_robometer_nf4_checkpoint.py`) loads the packed NF4 weights DIRECTLY on the `meta` device — no bf16 materialization, no requantize (~25 s process→ready vs ~110 s + a 19 GB CPU spike), bit-identical to the bf16+quantize path with determinism pinned (math SDP + `use_deterministic_algorithms` + `CUBLAS_WORKSPACE_CONFIG`). The forward's activation memory scales with frame count × resolution, so the backend evenly subsamples the window to `max_frames` (8) to stay co-resident with the sim (and a small NF4 VLA) on 8 GB. In `deploy-sim`, `openral deploy sim --enable-reward-monitor` brings the monitor up parallel to the VLA and sets `task_progress_available:=true` so the reasoner is offered `query_task_progress` (validated live on LIBERO deploy-sim with SmolVLA). The reward model is lerobot's own first-party module (Apache-2.0), so no untrusted third-party package is executed.

## Playbook rSkills (`kind: playbook`)

`kind: playbook` rSkills are **symbolic, human-authored S2 decision procedures** — Markdown SOPs the reasoner *reads*, not neural policies it executes. A playbook ships a `PLAYBOOK.md` body plus a `PlaybookContract` (`trigger` natural-language retrieval key, `body_uri` path to the SOP, `composes_tools` advisory list of `ReasonerToolCall` variants the SOP uses, `done_predicate` acceptance test, `max_steps` hard tool-call bound). It carries **no weights, actuators, ROS server, or action/state contract** — the symbolic counterpart to a `vla` policy, `role: s2`, excluded from the actuation palette. At palette-seed time the reasoner gathers the installed, capability-matched playbooks and appends their bodies to the system prompt under `## PLAYBOOKS`, so the LLM follows the relevant procedure when its trigger matches the goal. Every motion still crosses `execute_rskill` + the C++ safety kernel.

| rSkill | Procedure |
|---|---|
| [`decompose-mission`](https://github.com/OpenRAL/openral/tree/master/rskills/decompose-mission/) | break a compound goal into ordered, individually-verifiable subtasks (drives the `decompose_mission` tool → `MissionState` queue) |
| [`verify-outcome`](https://github.com/OpenRAL/openral/tree/master/rskills/verify-outcome/) | Inner-Monologue: after a skill, confirm it actually succeeded (`query_scene` / `query_task_progress`) before advancing |
| [`clarify-ambiguity`](https://github.com/OpenRAL/openral/tree/master/rskills/clarify-ambiguity/) | ask-don't-guess: resolve an ambiguous goal from memory/scene, else ask the operator; never guess on an irreversible action |
| [`preflight-reach`](https://github.com/OpenRAL/openral/tree/master/rskills/preflight-reach/) | check the target is within the embodiment's reachable workspace (vs the `## ROBOT` self-model) before a grasp; stage or hand off |
| [`stage-for-manipulation`](https://github.com/OpenRAL/openral/tree/master/rskills/stage-for-manipulation/) | move to the skill's declared pre-grasp pose (MoveIt approach) and verify before the manipulation policy runs |
| [`find-object`](https://github.com/OpenRAL/openral/tree/master/rskills/find-object/) | locate a target via `recall_object` (memory) → `locate_in_view` (live) → bounded active search before manipulation |

The reasoner also maintains a self-written **`MEMORY.md`** — a persistent semantic memory it reads each tick and edits through the typed `memory_write` / `memory_search` tools, loaded at deploy time via `openral deploy sim/run --memory-dir`. See the [Reasoner reference](reasoner.md).

## Wrapped-ROS rSkills (`kind: ros_action`)

`kind: ros_action` rSkills wrap an existing ROS 2 action/service server (MoveIt, Nav2) behind the rSkill contract instead of shipping learned weights — the `RosIntegration` block (see manifest format below) names the package, interface type, and running server, and `ROSActionRskill` builds the client at configure time. They are `role: s1` (actuation) and dispatch through `execute_rskill` + the safety kernel like any policy, but carry **no `model.safetensors`**.

| rSkill | Wraps | Targets | License |
|---|---|---|---|
| [`rskill-moveit-eef-pose`](https://github.com/OpenRAL/openral/tree/master/rskills/rskill-moveit-eef-pose/) | MoveIt `MoveGroup` (Cartesian EEF-pose goal) | `franka_panda`, `ur5e`, `ur10e`, `so100_follower`, `openarm`, `rizon4`, `sawyer` | Apache-2.0 |
| [`rskill-moveit-joints`](https://github.com/OpenRAL/openral/tree/master/rskills/rskill-moveit-joints/) | MoveIt `MoveGroup` (joint-space goal) | same arm set as above | Apache-2.0 |
| [`rskill-moveit-look-at`](https://github.com/OpenRAL/openral/tree/master/rskills/rskill-moveit-look-at/) | MoveIt `MoveGroup` (aim wrist camera at a point) | same arm set as above | Apache-2.0 |
| [`rskill-nav2-navigate-to-pose`](https://github.com/OpenRAL/openral/tree/master/rskills/rskill-nav2-navigate-to-pose/) | Nav2 `NavigateToPose` (result-only; Nav2 publishes `/cmd_vel` itself) | `mobile_base` | Apache-2.0 |

## Manifest format

```yaml
# rskills/smolvla-libero/rskill.yaml (excerpt)
name: "OpenRAL/rskill-smolvla-franka_panda-libero_spatial-bf16"
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
- **GR00T-based checkpoints** — license is version-specific: N1/N1.5/N1.6 are NVIDIA OneWay Noncommercial (`nvidia_non_commercial`; requires `OPENRAL_ALLOW_NONCOMMERCIAL=1`); N1.7+ are NVIDIA Open Model License (`nvidia_open_model`, commercial OK).
- **LocateAnything-3B** — NVIDIA License, non-commercial research/evaluation; private HF rSkill, requires `OPENRAL_ALLOW_NONCOMMERCIAL=1` and remote-code acceptance.
- **π0 / π0.5 weights** — permissive research (not full Apache-2.0 for commercial deployment).
- **RLDX-1** — RLWRLD non-commercial; runs as an out-of-process sidecar.

Provenance signing via sigstore is planned but not yet implemented — the loader emits an `rskill.unverified_provenance` warning on every load. Set `OPENRAL_REQUIRE_SIGNED_SKILLS=1` to fail closed.
