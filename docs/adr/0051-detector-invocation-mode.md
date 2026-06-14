# ADR-0051: Detector invocation mode — continuous background producers vs on-demand prompted locators

- Status: **Accepted**
- Date: 2026-06-12
- Related: [ADR-0037](0037-gstreamer-perception-bus-object-detection.md) (the `kind: detector` rSkill,
  detector tiers, and the 2026-06-12 `zeroshot_hf` amendment); [ADR-0043](0043-locate-in-view-reasoner-tool.md)
  (the read-only `locate_in_view` reasoner tool); [ADR-0035](0035-perception-spatial-memory-object-lift.md)
  / [ADR-0038](0038-persistent-semantic-spatial-memory.md) (world-state object lift + scene-graph memory);
  [ADR-0018](0018-ros2-reasoner-supervisor.md) §4 (tool palette); [ADR-0050](0050-single-resident-skill-vram-eviction.md)
  (single-resident-skill VRAM eviction); CLAUDE.md §3 (layer boundaries — this crosses the detector
  contract ↔ reasoner palette boundary, hence this ADR).

## Context

After [ADR-0037](0037-gstreamer-perception-bus-object-detection.md) (2026-06-12 amendment) there are
three perception detector rSkills, and they reach the reasoner through **three different mechanisms**
that are currently assigned **implicitly**:

| rSkill | Vocabulary | How it runs | How the reasoner sees it |
|---|---|---|---|
| `rtdetr-coco-r18` / `-v2-r50vd` | closed (80 COCO) | always-on tee producer | excluded from the ExecuteSkill palette; output → `WorldState.detected_objects` |
| `omdet-turbo-indoor` | frozen-open (266 fixed) | always-on tee producer | same — background producer |
| `locateanything-3b-nf4` | open (any text) | *both* a static-default continuous leg **and** an on-demand service | surfaces the `locate_in_view` tool (gated by the runtime service existing) |

Two **orthogonal axes** are conflated into the model identity:

1. **Vocabulary** — *closed/fixed* (RT-DETR, OmDet) vs *open* (LocateAnything).
2. **Invocation** — *continuous background producer* (streams into world state; the reasoner reads it
   passively and never prompts it) vs *on-demand prompted locator* (the reasoner asks "find X **now**").

[ADR-0037](0037-gstreamer-perception-bus-object-detection.md)'s `DetectorEngine` already names the
*how-it-runs* axis (`rtdetr_onnx` / `vlm_sidecar` / `zeroshot_hf`). The *when-the-reasoner-invokes*
axis has **no typed contract** — it is inferred from whether a `/openral/perception/locate_in_view`
service happens to be wired. Consequences:

- **LocateAnything straddles both modes** (continuous default-query leg *and* `locate_in_view`), which
  is the root of the confusion.
- **OmDet-Turbo is an open-vocabulary *model* deliberately frozen into the background role**, but
  nothing in its manifest *declares* that — the reasoner cannot tell it is not meant to be prompted.
- **The reasoner has no typed view of which detector covers which vocabulary**, so it cannot make the
  one decision that matters: is `mug` already tracked continuously (read world state) or do I need to
  prompt the open-vocab locator for `the red stapler with the company logo`?

## Decision

1. **A typed `DetectorMode` on `DetectorContract.mode`** (`openral_core.schemas`) — the invocation
   axis, orthogonal to `DetectorEngine`:
   - `continuous` (**default**) — an always-on background producer. Runs on the camera tee every
     frame, streams `ObjectsMetadata` into `WorldState.detected_objects`; the reasoner reads it
     **passively** (world state / `recall_object`) and never prompts it. Not ExecuteSkill-dispatchable;
     no actuation authority. May still be toggled via `LifecycleTransitionTool` to free VRAM (ADR-0050).
   - `on_demand` — a prompted locator the reasoner invokes only when it needs a specific object **now**,
     via the read-only `locate_in_view` tool (ADR-0043). Not run continuously.

   `engine` × `mode` together give the reasoner a complete, typed mental model (*how it runs* × *when
   it is invoked*) instead of inferring intent from a service's existence.

2. **Single responsibility per rSkill** — mode is per-rSkill, not per-model. The manifests declare the
   split:
   - `rtdetr-coco-r18`, `rtdetr-v2-r50vd`, `omdet-turbo-indoor` → `mode: continuous` (the **background
     bank**: closed + frozen-open vocabularies, optimised for coverage/throughput).
   - `locateanything-3b-nf4`, `omdet-turbo-locator` → `mode: on_demand` (the **open-vocab locators**,
     activated on demand and the natural candidates for the ADR-0050 load-on-prompt / evict-after VRAM
     lifecycle). LocateAnything (3B VLM) is the higher-quality option for complex referring
     expressions; `omdet-turbo-locator` is a lightweight (~115M, real-time, in-process, Apache-2.0)
     alternative for simple "find X" queries.

   The same OmDet-Turbo weights are packaged in **both** modes (`omdet-turbo-indoor` continuous /
   `omdet-turbo-locator` on-demand) rather than one dual-mode rSkill — packaging two single-purpose
   rSkills is what keeps the modes from straddling (the LocateAnything lesson). The
   `OmDetTurboDetector` backend exposes `set_query` / `detect_with_query` so the on-demand package can
   back the `locate_in_view` service; the detector node binds those by `hasattr`, so the same backend
   serves either mode.

3. **Surface continuous coverage to the LLM** (reasoner palette). `build_tool_palette` collects every
   `mode: continuous` detector for the active robot into `ToolPalette.continuous_detectors`
   (`ContinuousDetectorEntry`: id + description + objects + scenes + label count — a compact *coverage
   characterisation*, **not** the full label list, to keep the prompt bounded). When `locate_in_view`
   is advertised (`detector_available`), its tool description is augmented with that coverage so the
   decision rule reaches the LLM **at the point of choice**:

   > object within a continuous detector's coverage → read world state / `recall_object`;
   > object outside it (novel / specific / attribute-qualified) → `locate_in_view`.

   This cleanly separates "open-vocabulary" from "prompting": **prompting is for the long tail the
   always-on bank does not cover.**

4. **The perception node enforces `mode`.** `RosImageObjectDetectorNode` resolves the manifest's
   `mode` at `on_configure` via the pure `detector_node_wiring` policy
   (`openral_runner.backends.gstreamer.detector_factory`): a `continuous` detector runs the primary
   camera's detect+publish leg and does **not** expose `locate_in_view` / subscribe `detector_query`;
   an `on_demand` detector exposes the `locate_in_view` service + `detector_query` topic and does
   **not** publish continuously (frames are still cached so the service can answer). The legacy ONNX
   path (no manifest) is `continuous`. `detector_available` / `scene_query_available` remain runtime
   flags `reasoner_node` sets from the live services (defence in depth).

## Alternatives considered

- **Overload `DetectorEngine` to encode mode** (e.g. a `vlm_sidecar_on_demand` value) — rejected:
  conflates the two orthogonal axes the whole ADR exists to separate, and combinatorially explodes
  (any engine can in principle run in either mode).
- **Infer mode from the presence of the `locate_in_view` service** (status quo) — rejected: the intent
  is invisible to the typed contract, so the reasoner cannot reason about coverage, and a detector's
  role depends on deployment wiring rather than its manifest.
- **Dump the full label list into the palette** — rejected: 80 + 266 labels per tick is prompt bloat;
  a coverage characterisation plus world-state lookup for specifics is sufficient and bounded.
- **Make continuous detectors ExecuteSkill tools** — rejected (and explicitly guarded against in
  `build_tool_palette`): a detector emits perception, not actions; admitting it would let the LLM
  dispatch it as if it actuated the robot (ADR-0037 Decision 4).

## Consequences

- **Schema.** `DetectorContract` gains `mode: DetectorMode` (default `continuous`); on-disk
  `schema_version` stays `"0.1"` (no migrator — CLAUDE.md §6). Existing continuous detectors are
  unaffected by the default; `locateanything-3b-nf4` is set to `on_demand` explicitly.
- **Reasoner palette.** `ToolPalette` gains `continuous_detectors: tuple[ContinuousDetectorEntry, ...]`;
  the `locate_in_view` tool description is coverage-aware. No new tool, no new dispatch path.
- **Layer touch (ADR-gated).** Layer 3 (detector manifest contract) ↔ Layer 4 (reasoner palette /
  tool descriptions). No actuation path is touched; all surfaced tools remain read-only.
- **Behavioural change.** With node-side enforcement (Decision 4), `locateanything-3b-nf4` — now
  `mode: on_demand` — **no longer publishes continuously**; it answers `locate_in_view` only.
  Deployments that want a continuous world-state producer should run a `continuous` detector
  (`rtdetr-*` or `omdet-turbo-indoor`) and use LocateAnything / `omdet-turbo-locator` as the on-demand
  locator. (This supersedes the interim arrangement where LocateAnything's static-default continuous
  leg also published — e.g. PR #316's demo, which should migrate to `omdet-turbo-indoor` for the
  continuous leg.)
- **Follow-ups.** (i) ~~node-side `mode` provisioning~~ — **done** (Decision 4). (ii) tie
  `mode: on_demand` to the ADR-0050 load-on-prompt / evict-after VRAM lifecycle; (iii) feed the
  continuous coverage into the reasoner system prompt as well as the tool description.
