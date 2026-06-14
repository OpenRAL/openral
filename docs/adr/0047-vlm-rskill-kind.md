# ADR-0047: `vlm` rSkill kind for video-language scene-understanding models

**Status:** Accepted  
**Date:** 2026-06-11  
**Author:** Adrian Llopart

---

## Context

OpenRAL's rSkill taxonomy (introduced in ADR-0024 and extended in ADR-0037)
covers:

- `"vla"` — learnable action-producing policies (S1)
- `"ros_action"` / `"ros_service"` — wrappers for existing ROS 2 servers (S1)
- `"detector"` — exported ONNX/TRT bounding-box detectors (S1)
- `"wam"` — reserved World Action Model slot (S2, not yet dispatched)

Modern video-language models (VLMs) such as Qwen3.5-4B serve a different
purpose in a robot stack: they are pure perception/reasoning components that
receive camera frames and a natural-language query and return a text answer.
They run at S2 speed (~0.2–1 Hz), emit no action chunks or bounding boxes, and
require no actuator contract.

Neither `"detector"` (requires a `DetectorContract` block and an ONNX engine,
enforces empty `actuators_required` for a different reason) nor `"vla"`
(requires `model_family`, `weights_uri`, ≥1 actuator) is the right fit.
Forcing a VLM into `"detector"` would misrepresent its output contract and
silently break the detector runner that expects `ObjectsMetadata`.

## Decision

Add `"vlm"` as a new `RSkillKind` value with the following invariants
enforced by `RSkillManifest._check_kind_consistency`:

| Field | Constraint |
|---|---|
| `weights_uri` | REQUIRED (HF model repo) |
| `actuators_required` | MUST be empty |
| `detector` | FORBIDDEN |
| `ros_integration` | FORBIDDEN |
| `action_contract` | FORBIDDEN |
| `state_contract` | FORBIDDEN |
| `processors` | FORBIDDEN (VLMs manage their own preprocessing) |
| `image_preprocessing` | FORBIDDEN |
| `n_action_steps` | FORBIDDEN |
| `starting_pose` | FORBIDDEN |
| `model_family` | OPTIONAL (metadata only) |
| `role` | SHOULD be `"s2"` (loader may warn if `"s1"`) |

Add `QUERY = "query"` to `RSkillAction` so `vlm` manifests can declare
their action verb against the closed vocabulary.

The first rSkill using this kind is `rskills/qwen35-4b-nf4` (Qwen3.5-4B NF4
bitsandbytes), which wraps `Qwen/Qwen3.5-4B` — a natively-multimodal 4B model
with hybrid linear attention that outperforms Qwen2.5-VL-7B on video benchmarks
at lower VRAM cost.

### Runtime dispatch (implemented)

A `kind: vlm` skill is **not** dispatched through the `ExecuteSkill` path — it
produces text, not actions, and is `role: "s2"`, so it is correctly excluded
from `build_tool_palette`'s `ExecuteSkill` palette (which only admits `role:
"s1"`). Instead the reasoner reaches it through a **read-only scene-query tool**,
exactly mirroring the `locate_in_view` detector tool (ADR-0043):

- **Sidecar** (`tools/qwen_vlm_sidecar.py` + `tools/_qwen_vlm_server.py`): boots
  the NF4 Qwen3.5-4B model in its own venv and serves a ZMQ REQ/REP + msgpack
  protocol (`{"op": "query", "image", "question"}` → `{"ok", "answer"}`). Same
  pattern as the LocateAnything sidecar. Out-of-process for dependency / VRAM
  isolation (the runtime venv hard-pins `transformers==5.3.0` for lerobot; the
  VLM wants bitsandbytes + `qwen-vl-utils` + Gated-DeltaNet kernels).
- **Backend** (`openral_runner.backends.gstreamer.qwen_scene_vlm.QwenSceneVlm`):
  the node-side ZMQ client — lazy connect, auto-spawn, teardown only the child
  it started. `build_scene_vlm(manifest)` builds it from a `kind: "vlm"`
  manifest. Returns *text*, not `ObjectsMetadata`.
- **Service node** (`openral_perception_ros.scene_vlm_node`): subscribes the
  cameras, caches frames, and serves `/openral/perception/query_scene`
  (`openral_msgs/srv/QueryScene`). Separate from the detector node because a
  scene VLM is a reasoning aid, not a continuous detector.
- **Reasoner tool**: `QuerySceneTool` (new `ReasonerToolCall` member,
  discriminator `"query_scene"`) is surfaced to the LLM only when
  `ToolPalette.scene_query_available` is set (the `scene_query_available`
  reasoner param, mirroring `detector_available`). The reasoner's
  `_dispatch_query_scene` calls the service async and feeds the free-text answer
  back as a `PromptStamped` (frame_id `"scene_vlm"`) — the prompt cascade.

`scene_query_available` and `detector_available` are independent flags:
localization (`locate_in_view`) and scene-state reasoning (`query_scene`) are
separately provisioned backends.

The rSkill runner resolver (`rskill_runner_node._resolve`) still does **not**
handle `kind: vlm` — that path is for actuating/ROS skills, and a `vlm` skill
must never be dispatched there. A `vlm` skill is only ever reached via the
read-only `query_scene` tool above.

The exact Qwen3.5 processor / generate entrypoints in `_qwen_vlm_server.py`
follow the canonical Qwen-VL transformers recipe and were **validated live** on
an RTX 4070 Laptop (8 GB) via the GPU-gated end-to-end test in
`tests/unit/test_qwen_scene_vlm.py` (`test_e2e_query_coco_sample`,
`OPENRAL_QWEN_VLM_SIDECAR_VENV`): NF4 loads to ~3.3 GB resident and real image
queries return correct answers, including the task-verification case
("Has a robot gripper grasped any object?" → "No"). Two load-time facts the
real run surfaced (both handled in the server): the model loads via
`AutoModelForImageTextToText` (registers as `Qwen3_5ForConditionalGeneration`),
and transformers 5.x's parallel loader must be forced **serial**
(`core_model_loading.GLOBAL_WORKERS = 1`) + `expandable_segments` so the bf16
load transient doesn't OOM an 8 GB card before bitsandbytes quantizes.

## Consequences

### Positive

- Manifests for scene-understanding VLMs are now first-class citizens with
  a correct per-kind contract that prevents misrouting to the detector runner.
- The `role: "s2"` + `kind: "vlm"` combination cleanly separates the
  S2 perception backbone from S1 action policies in the registry.
- Qwen3.5-4B NF4 (~2.5 GB VRAM) can coexist on the same 8 GB edge GPU with
  an S1 VLA skill stack.

### Negative / risks

- The scene VLM is a second on-device model competing for the 8 GB GPU. It runs
  on-demand (not continuously) and NF4 (~2.5 GB), but an operator must budget
  VRAM against the active S1 policy + any detector. The sidecar owns its own
  VRAM and can be torn down between queries.
- The Qwen3.5 processor/generate API in the sidecar server is validated only by
  the GPU-gated E2E test, not the always-on unit suite — API drift in a future
  transformers pin would surface there, not in CI without a GPU (CLAUDE.md §12).
- A new `RSkillKind` value is a schema surface change. All manifests in-tree
  are tested by `test_rskill_manifest.py::TestInTreeManifests`; the test suite
  must pass before merge.

## Alternatives Considered

1. **Reuse `kind: "detector"`** — rejected. The detector runner expects an ONNX
   engine and structured `ObjectsMetadata`; routing a Transformers VLM through
   it would be incorrect and would fail silently at inference time.
2. **Reuse `kind: "wam"`** — rejected. WAMs are planning-layer mental-simulation
   components (CLAUDE.md §3), not perception backbones. Overloading `wam` for
   both purposes muddles the layer semantics.
3. **Keep VLMs as external Reasoner LLMs only** — feasible but limits the
   rSkill packaging system from representing a useful class of perception
   components. The `vlm` kind lets the registry, capability matching,
   and VRAM budget tracking apply to VLMs as first-class rSkills.
