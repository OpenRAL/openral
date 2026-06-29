# ADR-0076 — Detection-time object identity + a camera-space `in_view` enumeration for the reasoner

- **Status:** Accepted 2026-06-28. Amended 2026-06-29 (§4 — sticky `located` line).
- **Date:** 2026-06-28
- **ADR number:** `0076`. The integer is not load-bearing — cross-refs use
  filenames.
- **Related:**
  - [ADR-0035](0035-perception-spatial-memory-object-lift.md) — the 2D→3D
    object-center lift and the `ObjectMemory` IoU association that mints
    `track_id`. **This ADR amends it**: identity is moved *earlier* (to detection
    time) so it exists even when the 3D lift cannot run, and the lift propagates
    that id instead of dropping it (`track_id=None`).
  - [ADR-0038](0038-persistent-semantic-spatial-memory.md) — the spatial-memory
    `object` node (a superset of `DetectedObject`, `map`-frame). Unchanged; it
    still consumes 3D `DetectedObject`s, now carrying the detection-time id.
  - [ADR-0056](0056-on-demand-detectors-as-promptable-reasoner-tools.md) — the
    on-demand `locate_in_view` locator. Complementary: that answers "is X visible
    right now"; this surfaces the **continuous** detector's full enumeration so
    the reasoner can decompose a collective goal without polling.
  - [ADR-0075](0075-grounded-decomposition-contract.md) — grounded decomposition.
    **This ADR supplies its missing input in the field**: the grounded
    `decompose_mission` needs a per-object enumeration; the isolated eval had one
    pre-populated, but the live deploy did not (see Context), so the LLM looped on
    `locate_in_view` and never decomposed.

## Context

Live `libero_object` deploy-sim (2026-06-28, glm-5.2) stalled: faced with the
collective goal *"put all the objects on the table into the basket"* the reasoner
made **13 read-only `locate_in_view`/`recall_object` calls over ~15 min and never
decomposed or executed** — the arm never moved. The ADR-0075 grounding fix relies
on a `scene_objects` enumeration in the reasoner's context (the isolated eval
pre-populated it; glm decomposed 9/10). In the deploy that enumeration was empty.

Root cause (traced to code): `scene_objects` is rendered only from
`WorldState.detected_objects`, which is the **3D lift** output
(`object_lift.py`). The lift fuses each 2D detection with an **occupancy voxel
grid** to recover a world position; `lift()` returns nothing when there are zero
voxels (`if n == 0: return out`). LIBERO scenes ship **RGB-only** cameras (no
depth), so the octomap is disabled (`octomap: disabled (no depth SensorSpec)`),
`occupied_centers_base` is empty, and **no objects are ever lifted** — spatial
memory stays at "0 nodes" and `scene_objects` is empty. The open-vocab detector
*is* detecting objects (the on-demand locator returned `found=True`); the 2D
labels are simply discarded because the lift can't place them in 3D.

The reasoner does not need 3D poses to *decompose* — it needs the **set of object
labels**, and a per-object **id** to disambiguate duplicates and to refer to a
specific object across ticks. That enumeration is depth-free. Two facts make this
cheap: (1) the reasoner *already* subscribes to `/openral/perception/objects` and
receives the raw `ObjectsMetadata` (it just renders it as `"N objects"` text and
throws the detections away); (2) the only thing missing is identity —
`ObjectDetection2D` has no id, and `track_id` is minted at *3D lift time*.

## Decision

**Move object identity to detection time, and surface a camera-space `in_view`
enumeration into the reasoner context.** Identity exists with or without depth and
unifies into the 3D path when depth returns.

### 1. Detection-time identity (Layer 2 → schema)

`ObjectDetection2D` gains `det_id: int` (default `-1` = untracked). A continuous
detector stamps a **stable** id per object via a 2D-IoU tracker
(`DetectionTracker2D`, a 2D analog of ADR-0035's `ObjectMemory`): greedy
same-label IoU association across frames, a fresh monotonic id for an unmatched
box, a small miss-budget before an id is retired. The id is **camera-space and
per-detector** — it identifies "the object the *detector* is tracking in this
camera", not a world entity.

### 2. Camera-space `in_view` line (Layer 4 → reasoner context)

`ContextRenderer` renders a new line from the latest `ObjectsMetadata`:

```
in_view[top]: #0 milk @px(412,233), #1 ketchup @px(388,251), #2 alphabet soup @px(440,210), #3 basket @px(120,300)
```

`@px(cx,cy)` is the **pixel center** in the detector's frame — explicitly image
space, never dressed up as a 3D pose. It is rendered **separately** from the 3D
`scene_objects[map]: …@(x,y,z)` line so the two coordinate spaces never blur. The
`in_view` line is "what the camera sees right now (with ids)"; `scene_objects` is
"what has been lifted into 3D / spatial memory". The reasoner's grounding/decompose
guidance (ADR-0075) reads *either* — whichever is populated — to enumerate.

### 3. Propagate the id into 3D (Layer 3 → lift + memory), so it all works with octomap

When depth/voxels *are* available, the lift no longer discards identity:
`object_lift.lift()` sets `DetectedObject.track_id = det.det_id` (was `None`).
`ObjectMemory.tick()` keeps its 3D-IoU association for **persistence** (a matched
track keeps its stored id — 3D association is more stable than 2D), but a **new**
(unmatched) candidate **adopts** its incoming `det_id` instead of minting a fresh
`_next_id` when one is present (falls back to minting for legacy detectors with
`det_id < 0`). Net: a physical object carries **one id** whether it appears in the
2D `in_view` line (no depth) or the 3D `scene_objects` line (depth) — and the
existing 3D-only behavior is preserved byte-for-byte when no 2D tracker is wired.

### 4. Sticky `located` line from open-vocab locate hits (amendment 2026-06-29)

Live deploy on `libero_object` exposed a gap in §2: the `in_view` line is fed by
the **continuous** detector (`omdet-turbo-indoor`), whose **fixed ~230-class
indoor vocabulary** mislabels the task objects — on the real scene it emitted 29
detections of `cup`/`bottle`/`mug`/`stool`/`sink`/… with **no `basket`** and
`ketchup`/`milk` collapsed into `bottle`/`pitcher`. The reasoner therefore could
not find the goal nouns in `in_view`, fell back to `recall_object` (empty without
the 3D lift) → auto-escalated to the **open-vocab** `locate_in_view` (which *does*
find them when prompted) → but that returned only a transient re-prompt, so the
next tick clobbered `in_view` and the loop repeated — the exact stall this ADR set
out to kill, resurfacing through the fixed-vocab feed.

Fix: persist successful open-vocab `locate_in_view` hits. `ContextRenderer.note_located`
stores them keyed by lowercased label (latest-wins, capped at `_LOCATED_CAP=12`)
and `_render_in_view` emits a second, **sticky** line that survives the continuous
detector's per-frame `set_in_view` clobber:

```
in_view[top]:  #0 cup @px(...), #1 bottle @px(...), …      (fixed-vocab, clobbered each frame)
located[top]:  basket @px(150,350), teapot @px(...), …     (open-vocab hits, sticky)
```

`reasoner_node._on_locate_in_view_response` calls `note_located` on every `found`
hit. Net: a goal noun the reasoner confirmed once stays grounded, so it decomposes
and dispatches instead of re-locating. Verified live — glm-5.2 went from looping to
`decompose_mission` (6 grounded subtasks) → `execute_rskill` (VLA) → mission ladder
advancing on plateau.

### What stays the same

- The 3D lift still requires depth/voxels — this ADR does **not** invent a
  depth-free 3D pose. It only stops *identity* and *enumeration* from being
  hostage to depth.
- `scene_objects` (3D) is unchanged in shape; it now carries `det_id`-derived ids.
- Spatial memory (ADR-0038) is unchanged; it consumes the same `DetectedObject`s.

## Consequences

- **Positive.** The reasoner gets the per-object enumeration that ADR-0075's
  decomposition needs, in RGB-only deploys (LIBERO, most tabletop sims) where the
  3D lift can't run — directly addressing the observed `locate_in_view` stall.
  Duplicate objects are disambiguable by id. One identity across 2D/3D.
- **Negative / cost.** `ObjectDetection2D` gains a field (additive — wire
  contract on the perception bus; no on-disk migrator, it is an ephemeral
  `metadata_json` payload). New `DetectionTracker2D` to maintain. The 2D id is
  per-camera/per-detector (a left/right swap on heavy occlusion can re-id) — the
  3D association corrects this for lifted objects; for 2D-only it is best-effort
  and labelled as such.
- **Neutral.** No safety surface (advisory perception, ADR-0038 invariant). No
  new layer dependency — perception already feeds world-state and the reasoner.

## Alternatives considered

1. **Fake a 3D pose from the 2D box (assume a table plane).** Rejected: invents
   geometry, wrong for non-tabletop, and `scene_objects` would carry untrustworthy
   coords. Camera-space-as-camera-space is honest.
2. **Mint the id only in 2D, keep 3D `track_id` separate.** Rejected: a physical
   object would carry two ids (a 2D view-id and a 3D track-id), which the operator
   directive ("give them an id … make sure it all works if octomap is available")
   explicitly argues against. Propagating one id is the coherent choice.
3. **Anti-thrash nudge only** (force decompose after N read-only locate calls,
   no enumeration). Rejected as the primary fix: it papers over the symptom (the
   locate loop) without removing the cause (no enumeration). May still be added as
   defence-in-depth, but the enumeration is the real fix.

## Rollout

1. Schema (`det_id`) + `DetectionTracker2D` + `aabb_iou_2d`.
2. Detector node stamps `det_id` on the continuous leg.
3. `object_lift` propagates `det_id`; `ObjectMemory` adopts it for new tracks.
4. `ContextRenderer.set_in_view` + the reasoner `_on_perception` wiring.
5. Tests (schema fuzz, tracker stability, lift propagation, context render) +
   `docs/methods` + repo-state-map. Re-run the `libero_object` deploy to confirm
   the reasoner now decomposes from `in_view`.
