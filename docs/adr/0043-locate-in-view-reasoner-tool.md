# ADR-0043 — `locate_in_view`: on-demand live-detector query for the reasoner

- **Status:** Accepted
- **Date:** 2026-06-10
- **Related:** ADR-0039 (LLM task planning + active search; `recall_object` / `resolve_place`), ADR-0037 (GStreamer perception bus + VLM detector tier), ADR-0035 (ROS-Image object detector + object-lift), ADR-0018 §4 (reasoner tool palette)

> **2026-06-10 amendment — `find_object` renamed to `recall_object`.** Introducing
> `locate_in_view` (look at the *current frame*) exposed that ADR-0039's
> `find_object` is a misnomer: it never touches a live camera — it only *recalls*
> objects from the ADR-0038 scene-graph **memory** (the accumulated dynamic object
> list). The verb "find" read as live detection and collided with `locate_in_view`.
> Renamed the whole memory-recall family for clarity — tool `find_object` →
> `recall_object` (`FindObjectTool` → `RecallObjectTool`) and the underlying ADR-0038
> query surface `FindObjectQuery`/`Match`/`Result` → `RecallObject*`,
> `SpatialMemory.find_object()` → `recall_object()`. `resolve_place` is unchanged (it
> resolves a reference to a *navigation goal*, not a pure recall). Runtime tool
> contract only — on-disk `schema_version` unaffected.

## Context

ADR-0039 gave the S2 reasoner two **read-only** query tools — `recall_object` and
`resolve_place` — that recall objects/places from the ADR-0038 scene-graph
**memory**. They answer "where did I *see* the mug?" from accumulated past
detections. They do **not** answer "is the mug in front of me *right now*?".

That live check is exactly what the ADR-0037 open-vocabulary VLM detector
(LocateAnything) can do: given a free-text query it grounds the object in the
current camera frame. The reasoner should be able to ask it on demand — to
verify a candidate before committing a motion, to confirm a grasp target, to
disambiguate during active search — without waiting for a navigate-and-observe
cycle.

The naive route ("dispatch the detector via `ExecuteRskillTool`") is wrong by
design: `ExecuteRskillTool` is the **actuation** path (F1 action server → F5
safety kernel). Detectors are perception producers with no actuation authority,
and are deliberately excluded from that palette (ADR-0037). A live perception
query needs its own read-only channel.

## Decision

Add a read-only `locate_in_view` reasoner tool that calls a live detector over a
ROS service, mirroring the `recall_object` pattern but against the **current
frame** instead of remembered memory.

1. **Contract** — `LocateInViewTool` variant on the `ReasonerToolCall` discriminated
   union (`openral_core.schemas`): `query: str` (the object to look for) +
   `camera: str = ""` (optional viewpoint selector; empty = the detector's primary
   camera). Frozen, `extra="forbid"`, no actuation authority.

2. **Palette gating** — `ToolPalette.detector_available` (default `False`). The
   reasoner sets it (param `detector_available`) only when an object detector is in
   the graph; the launch sets it from `enable_object_detector`. `tool_use` advertises
   `locate_in_view` to the LLM only under that flag — no hidden tool (CLAUDE.md §1.4).

3. **Service** — `openral_msgs/srv/LocateInView` (`query`, `camera` → `found`,
   `camera`, `metadata_json`). The detector node
   (`openral_perception_ros.ros_image_detector_node`) serves
   `/openral/perception/locate_in_view`: it caches the latest frame per camera and,
   on request, runs a **one-shot** detection on the requested camera's frame. For the
   VLM it uses `LocateAnythingDetector.detect_with_query` so the on-demand query does
   **not** disturb the continuous detection leg's persistent query.

4. **Dispatch** — `reasoner_node._dispatch_locate_in_view` calls the service with
   `call_async` + a done-callback (never blocking the reasoner executor on the
   ~1–2 s VLM inference), then republishes the rendered answer as a `PromptStamped`
   with frame_id `"detector"` — consumed by `_on_prompt`, feeding the next tick (the
   ADR-0039 prompt cascade). Read-only: no `FailureTrigger`, no actuation.

5. **Camera-agnostic detector** — the detector node no longer bakes in a camera name.
   A `cameras` param maps logical ids → image topics (falling back to a single
   `image_topic` under `primary_camera`); the **primary** drives the continuous leg
   and every camera's latest frame is cached. `LocateInViewTool.camera` / the service
   `camera` field select a viewpoint by id. The rSkill manifest stays camera-agnostic
   (`sensors_required: rgb`, no topic).

## Consequences

- **Layer touch (ADR-gated).** Layer 4 (reasoner) gains a read-only tool + dispatch
  that calls a Layer 1/3 perception service. No actuation boundary is crossed; the
  safety kernel never sees it. Mirrors ADR-0039's read-only query tools.
- **Schema change.** `openral_core` gains `LocateInViewTool` on the `ReasonerToolCall`
  union; on-disk `schema_version` is unaffected (runtime tool contract, not a manifest).
  `docs/METHODS.md` + the repo state map update in this PR.
- **New IDL.** `openral_msgs/srv/LocateInView.srv`. (Build note: rosidl must use the
  ROS python3.12, not a stray conda/miniforge python3.13, or the C typesupport links
  the wrong libpython.)
- **Blocking on the server, not the client.** The VLM detect (~1–2 s) runs in the
  detector node's single-threaded executor (which serialises with the continuous leg —
  desirable, since the ZMQ REQ socket to the sidecar is not concurrency-safe). The
  reasoner client is async. A `MultiThreadedExecutor` + per-detector lock is a future
  option if continuous-leg latency during a query becomes a problem.

## Follow-ups

- Tie repeated `locate_in_view` calls into the ADR-0039 `SearchBudget` so active-search
  loops are bounded (today each call resets the spatial-search counter).
- Let the reasoner enumerate available camera ids in the tool description so the LLM
  picks valid viewpoints.
- Active-search integration: `plan_active_search` calls `locate_in_view` to filter
  candidates before committing a NAVIGATE.
