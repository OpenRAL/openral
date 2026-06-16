# ADR-0056: On-demand detectors as prompt-able read-only reasoner tools (multi-detector graph + model-selectable locate)

- Status: **Accepted**
- Date: 2026-06-15
- Related: [ADR-0037](0037-gstreamer-perception-bus-object-detection.md) (`kind: detector` rSkill,
  detector tiers, `DetectorEngine`); [ADR-0043](0043-locate-in-view-reasoner-tool.md) (the read-only
  `locate_in_view` reasoner tool + `LocateInView.srv`); [ADR-0051](0051-detector-invocation-mode.md)
  (continuous vs on-demand invocation mode — **this ADR extends, and partially revises, 0051**);
  [ADR-0050](0050-single-resident-skill-vram-eviction.md) (single-resident-skill VRAM eviction);
  [ADR-0035](0035-perception-spatial-memory-object-lift.md) / [ADR-0038](0038-persistent-semantic-spatial-memory.md)
  (object lift + scene-graph memory); [ADR-0018](0018-ros2-reasoner-supervisor.md) §4 (tool palette);
  [ADR-0055](0055-rskill-registry-model-and-discoverability.md) (rSkill registry); CLAUDE.md §3
  (layer boundaries — this crosses the detector contract ↔ reasoner palette boundary).

> Decided 2026-06-15. Node topology = node-per-detector (§Decision 1); tool surface = one
> `locate_in_view` with a `detector` field (§Decision 2); default on-demand locator =
> `omdet-turbo-locator`, LocateAnything opt-in (§Decision 2). Richer VLM scene-understanding
> (`kind: vlm` / VQA) is **explicitly out of scope** here — a separate ADR if pursued.

## Context

After ADR-0037/0043/0051 there are five detector rSkills across two invocation modes:

| rSkill | engine | mode | role today |
| --- | --- | --- | --- |
| `rtdetr-coco-r18`, `rtdetr-v2-r50vd` | closed (COCO-80) | continuous | background producer → `WorldState.detected_objects` |
| `omdet-turbo-indoor` | frozen-open (266) | continuous | background producer (deploy-sim default after ADR-0037 2026-06-15 amendment) |
| `omdet-turbo-locator` | open | on_demand | lightweight `locate_in_view` backend |
| `locateanything-3b-nf4` | open (any text) | on_demand | high-quality grounding VLM `locate_in_view` backend |

Three limitations block the desired operator/reasoner experience:

1. **Single-detector-node graph.** `sim_e2e.launch.py` brings up exactly one `openral_ros_image_detector`
   node with one manifest in one mode. So a deployment gets *either* a continuous bank *or* one
   on-demand locator — never both — and never two continuous detectors as independent toggle targets.
2. **No model selection in the locate path.** `LocateInView.srv` carries only `query` + `camera`
   (ADR-0043). The reasoner cannot choose LocateAnything vs omdet-turbo-locator; it hits whatever
   single backend is wired. `locate_in_view` is also only exposed when the *running* detector is
   on_demand (ADR-0051 Decision 4), so in a continuous-default deployment it is absent entirely.
3. **On-demand detectors are not surfaced as first-class, prompt-able tools.** ADR-0051 deliberately
   excludes all `kind: detector` from the ExecuteSkill palette because a detector emits perception,
   not actions. That invariant is correct — but it conflated "no actuation authority" with "not a
   tool the reasoner invokes." An on-demand, open-vocabulary detector *is* prompt-able (you pass it a
   free-text query) and gives the reasoner on-request scene understanding; the product intent is for
   the reasoner to *choose and invoke* one, read-only.

**Goal (operator-stated, 2026-06-15):** a scene comes up with a continuous detector by default; the
reasoner can (a) activate/deactivate continuous detectors at will (sparingly, for VRAM), and (b)
invoke an on-demand locator — choosing LocateAnything *or* omdet-turbo-locator — to ask "is object X
in view?" and get better scene understanding.

## Decision (proposed)

**Hold the safety invariant: on-demand detectors are prompt-able *read-only* tools, not actuating
`execute_rskill` dispatches.** We extend the read-only `locate_in_view` mechanism (ADR-0043) rather
than reverse ADR-0051's "detector ≠ actuation authority" guard. This satisfies the goal without
letting the LLM drive a detector as if it moved the robot (CLAUDE.md §1.1, §3).

### 1. Multi-detector graph (topology — DECIDED: node-per-detector)

Allow N detectors co-resident: one (or more) `mode: continuous` producer(s) **plus** the
`mode: on_demand` locators. Each detector is its own lifecycle node under a
namespaced service/topic prefix — `/openral/perception/<rskill_id>/locate_in_view`,
`/openral/perception/<rskill_id>/objects`, `/openral/perception/<rskill_id>/detector_query`. This:

- matches ADR-0050's **node-level** VRAM eviction (toggle each detector independently);
- keeps modes from straddling (the LocateAnything lesson, ADR-0051 Decision 2);
- makes "which models are available" a property of which nodes are in the graph.

Trade-off: more processes; on an 8 GB host the on-demand locators must be **load-on-prompt /
evict-after** (they cannot be co-resident with a VLA + a continuous detector). Alternative topology
in §Alternatives.

### 2. Prompt-able, model-selectable locate tool (read-only)

- `LocateInView.srv` gains a `detector` field (rSkill id / short alias; empty = deployment default).
  **IDL change → on-disk/contract `schema_version` review per CLAUDE.md §4.4** (additive, but it is a
  message-contract change; ship the bump + note even though existing callers default it empty).
- `LocateInViewTool` (the `ReasonerToolCall` variant) gains the matching `detector` field, and its
  tool description enumerates the on-demand locators actually in the graph (id + one-line capability),
  so the LLM can pick — e.g. omdet-turbo-locator for fast simple "find X", LocateAnything for complex
  referring expressions. The reasoner routes to `/openral/perception/<detector>/locate_in_view`.
- The palette continues to surface continuous detectors as `ContinuousDetectorEntry` *coverage*
  (read world state / `recall_object` for those) and reserves the locate tool for objects outside
  that bank (ADR-0051 Decision 3) — now across multiple on-demand backends.

### 3. Continuous detectors as independent lifecycle peers

Each continuous detector node is registered as its own `lifecycle_peer_node_ids` + `vram_lifecycle_peers`
entry so the reasoner can `LifecycleTransitionTool` it ACTIVE/INACTIVE individually (ADR-0050), with
the existing "don't thrash" guidance in the system prompt.

### 4. VRAM

On-demand locators are load-on-prompt / evict-after (ADR-0050). `locate_in_view` on an INACTIVE
locator triggers a managed activate → one-shot → (optional) deactivate, surfaced in traces.

## Alternatives considered

- **One node, multiple backends (single-process router).** A single detector node hosts the
  continuous backend + lazily-loaded on-demand backends; `locate_in_view` gains a `model` field the
  node routes internally. Fewer processes, but per-backend lifecycle/VRAM toggling is murky and it
  re-introduces the mode-straddling that ADR-0051 Decision 2 spent effort removing. Viable as a
  lighter first increment if process count is a concern.
- **Make on-demand detectors `execute_rskill` tools.** Rejected: violates ADR-0051's invariant
  (detector emits perception, not actions) and the CLAUDE.md §1.1 safety priority. The read-only
  `locate_in_view` surface already gives the reasoner "invoke a prompt-able detector" semantics
  without any actuation authority — the same UX without the risk.
- **Status quo (single on-demand detector, no model choice).** Rejected: cannot satisfy "continuous
  default + reasoner-chosen locator," which is the operator goal.

## Consequences

- **Layer touch (ADR-gated):** Layer 1/3 (detector node + manifest contract) ↔ Layer 4 (reasoner
  palette / tool descriptions). **No actuation path is touched; all surfaced detector tools remain
  read-only** (preserves ADR-0051's core property).
- IDL change to `LocateInView.srv` (additive `detector` field) — needs a `schema_version`/contract
  note + a real-fixture test (CLAUDE.md §4.4, §1.6).
- `sim_e2e.launch.py` grows multi-detector wiring + per-detector namespaces and lifecycle-peer lists;
  `deploy_sim.py` gains flags to choose which continuous detector(s) and which on-demand locator(s)
  to bring up (default: omdet-turbo-indoor continuous + omdet-turbo-locator on-demand; LocateAnything
  opt-in for VRAM/licence reasons — it is NVIDIA non-commercial, ADR-0046/licence matrix).
- Tests: palette unit tests for multi-detector admission + tool-description enumeration; an
  integration test that brings up a continuous + an on-demand detector and exercises a model-selected
  `locate_in_view`; the find→navigate→grab e2e gains a real on-demand-locate step.

## Resolved decisions (2026-06-15)

1. **Topology:** node-per-detector (each detector its own lifecycle node, namespaced services). Not
   the single-node router.
2. **Tool surface:** one `locate_in_view` tool with a `detector` field; its description enumerates the
   on-demand locators in the graph. Not one tool entry per locator.
3. **Default on-demand locator:** `omdet-turbo-locator` (light, ~115M, Apache-2.0, commercial-OK).
   `locateanything-3b-nf4` is opt-in (NVIDIA non-commercial, 5 GB) via an explicit flag.
4. **VLM/VQA scene-understanding is out of scope** for this ADR. LocateAnything is used here only as a
   box-returning open-vocab locator. Richer `kind: vlm` / VQA capability, if ever pursued, is a
   separate ADR — not added here.

## Amendment — 2026-06-15 (recall→locate escalation; GitHub #10)

The on-demand locator is now reachable **autonomously**, not only when the LLM explicitly
picks `locate_in_view`. In `ReasonerNode._dispatch_spatial_query`, a `recall_object` **miss**
(`SpatialQueryOutcome.found == False`) escalates to a live `locate_in_view` for the same query
term **before** the active-search budget reaches human-handoff — provided a detector is available
(`detector_available`) and the term hasn't already been escalated this search streak
(`_locate_escalated`, reset alongside the search bound). This is **policy**, encoded in the node,
so it does not depend on the LLM choosing the tool (local models often don't). It closes the
"object seen but stored under a different label" gap — e.g. a goal naming "baguette" when the
continuous detector ingested it as "bread" — by grounding the exact prompt verbatim through the
open-vocab locator. Read-only invariant unchanged (the locator never actuates). Verified live on
`scenes/deploy/robocasa_baguette.yaml`: recall miss → locate `found=True` → autonomous pick.
