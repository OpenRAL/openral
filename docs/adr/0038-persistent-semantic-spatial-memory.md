# ADR-0038: Persistent spatial memory — object-centric recall with a hierarchical scene graph

- Status: **Proposed**
- Date: 2026-06-02
- Related: [ADR-0030](0030-geometric-safety-collision-checking.md) (the
  ephemeral OctoMap → `OccupancyVoxels` → kernel collision pipeline this ADR
  sits *beside*, not on top of); [ADR-0018](0018-ros2-reasoner-supervisor.md)
  (the S2 Reasoner and its `ReasonerToolCall` palette);
  [ADR-0039](0039-llm-task-planning-active-search.md) (LLM task planning + active
  object search that **consume** this world model and expose its query contracts
  to the reasoner — planning and search are *there*, not this ADR's);
  [ADR-0024](0024-ros-wrapped-rskills.md) (the manipulation skills — open an
  articulated door, grasp, pour — that *act on* objects this ADR only
  *remembers*); [ADR-0025](0025-reasoner-managed-background-services.md) (the
  perception-bridge pattern); [ADR-0012](0012-open-core-licensing.md) (license
  lineage); [ADR-0003](0003-pydantic-over-dataclasses.md) (Pydantic is the
  contract); CLAUDE.md §1.1 (safety beats helpfulness — memory is advisory,
  never a safety input), §1.3 (types are the contract), §1.8 (reproducibility),
  §1.9 (license lineage), §3 Layer 2 / Layer 4.

## Context

Two capabilities motivate this ADR, one simple and one rich:

1. **Object recall.** An operator names an object — *"find the mug"* — and the
   robot recalls its 3D position and drives to a standoff pose where its
   gripper-mounted camera faces it, ready to manipulate.
2. **Long-horizon household tasks.** *"Bring me a cup of wine."* The robot must
   know the wine lives **in the fridge**, go to **the kitchen**, reach the
   fridge, open it, grasp the wine; **find a glass** (searching likely places —
   cabinets, the kitchen table); pour; and **return to where I was standing** in
   the **living room**.

OpenRAL can answer *"is the space in front of the gripper occupied right now?"*
but neither of these. The geometric pipeline is built (ADR-0030):
`octomap_server` → `openral_octomap_bridge` → base-frame `OccupancyVoxels` →
kernel collision check. That pipeline is **ephemeral and local by design** — a
bounded box around the robot holding occupancy probability per voxel, with **no
object identity, no places, no rooms, and no memory**.

The only object information today is `WorldState.detected_objects: list[DetectedObject]`
(`schemas.py`): a flat, **per-snapshot** list of `DetectedObject(label,
confidence, pose: Pose6D, bbox_3d, track_id)`. It is momentary — nothing
accumulates, so the robot forgets the mug the instant it leaves frame, and it
has no notion that the wine is *inside* the fridge *in* the kitchen.

### What the deployed literature shows — and the boundary between the two tasks

- The **object-recall** task is solved on real robots by an **object/feature
  memory queried by language**, not a graph: **OK-Robot** (Liu/Paxton/
  Shafiullah/Pinto, 2024 — pick-and-drop in 10 homes, 58.5%/82% success, a CLIP
  "VoxelMap"; [arXiv 2401.12202](https://arxiv.org/abs/2401.12202)) and
  **DynaMem** (same group, 2024 — a dynamic point-cloud memory that updates as
  objects move; 70% on non-stationary objects;
  [arXiv 2411.04999](https://arxiv.org/abs/2411.04999)). **ReMEmbR** (NVIDIA)
  stores observations with coordinates+timestamps in a vector DB for navigation
  recall ([arXiv 2409.13682](https://arxiv.org/abs/2409.13682)).
- The **"bring me wine"** task is the canonical case for a **hierarchical 3D
  scene graph + LLM planner**: **SayPlan** plans multi-room household tasks over
  a 3D scene graph (collapse the hierarchy → semantic-search subgraphs →
  iterative replanning; validated on 3 floors / 36 rooms / 140 objects on a real
  mobile manipulator; [arXiv 2307.06135](https://arxiv.org/abs/2307.06135));
  **HOV-SG** adds floor→room→object hierarchy with a multistory navigation graph
  ([arXiv 2403.17846](https://arxiv.org/abs/2403.17846)); **Hydra** maintains
  objects/places/rooms **and an agents layer** in real time
  ([arXiv 2201.13360](https://arxiv.org/abs/2201.13360)).

The wine task needs *containment* (wine **in** fridge), *rooms/places* (kitchen,
living room, cabinets, table), and an *agent* (the requester's location) — none
of which a flat object list can express. So this ADR makes the object memory the
**foundation** and grows it into a **hierarchical scene graph**: the object layer
plus `place` / `room` / `agent` nodes and relational edges.

### Scope boundary (important)

This ADR specifies the **world-model representation and its query surface**
(Layer 2). It deliberately does **not** specify:

- **LLM task planning and active search** — decomposing "bring me wine" into
  steps, hypothesizing where a glass might be, and replanning on failure. That is
  the **S2 Reasoner's** job (ADR-0018); this ADR only gives it something to query.
- **Manipulation skills** — opening the articulated fridge door, grasping,
  pouring. Those are **rSkills** (Layer 3, ADR-0024) and still cross the ADR-0030
  safety gate.

The walkthrough in §4 shows the seam: what the representation answers, and what
it hands to planning and skills.

## Decision

### 1. A persistent hierarchical scene graph lives in Layer 2, beside (never on) the collision path

We add a durable **scene graph** that accumulates the momentary
`WorldState.detected_objects` and structures it into objects, places, rooms, and
agents. It is distinct from the ADR-0030 collision grid (fast bounded geometry
vs. slow accumulated semantics):

| | Geometric collision layer (ADR-0030) | Scene-graph memory layer (this ADR) |
|---|---|---|
| Stores | occupancy probability per voxel | typed nodes + relations + provenance |
| Scope | bounded local box, base frame | global, accumulated, `map` frame |
| Lifetime | ephemeral (rebuilt continuously) | persistent (survives the session) |
| Consumer | C++ safety kernel (hot path) | S2 Reasoner (event-driven query) |
| Safety role | **authoritative gate** | **advisory only — never a safety input** |

The memory layer **never touches the kernel hot path** and is **never a safety
input** (§1.1). The kernel keeps gating exclusively on the live, bounded
ADR-0030 world. A stale, wrong, or empty scene graph can at worst produce a bad
*plan* (which the kernel still vetoes geometrically) — it can never relax a
safety check.

### 2. Representation — object foundation + a typed hierarchy

**Nodes** (`SpatialNodeKind`):

- **`object`** — the foundation; a superset of `DetectedObject`: `label`,
  `confidence`, `bbox_3d`, `pose: Pose6D` **in the `map` frame**, provenance
  (`first_seen_ns`, `last_seen_ns`, `observation_count`), an optional
  `embedding_ref` for open-vocabulary matching (§5), and two container
  attributes — **`is_container`** and **`occludes_contents`** — so a fridge or
  cabinet can hold objects that are not observable until it is opened.
- **`place`** — a free-space navigation anchor (a standable waypoint, e.g. "in
  front of the fridge", "cabinet shelf region", "kitchen table"). The targets the
  robot actually drives to.
- **`room`** — a semantic area (kitchen, living room) grouping places/objects.
- **`agent`** — a person or robot with a pose; **the requester is an `agent`
  node** whose pose is captured when the command is issued, so "bring it back to
  me" resolves to a concrete return goal. (Mirrors Hydra's agents layer.)

**Edges** (`SpatialRelationKind`):

- **`contains`** — `room`→`object`/`place`, and **container `object`→`object`**
  (fridge `contains` wine; cabinet `contains` glass). The relation that makes
  "the wine is in the fridge" representable.
- **`at_place`** — `object`/`agent`→`place` (which waypoint to stand at to
  reach/observe it).
- **`traversable_to`** — `place`↔`place` / `room`↔`room`; the topological graph
  the planner walks for inter-room navigation.
- **`on` / `near`** — incidental object↔object spatial relations.

Every node pose is a **`Pose6D` in `map`** with timestamps; consumers resolve to
the live base frame at query time via TF2 — the pattern the octomap bridge
already uses. **No raw 4×4, no hardcoded `frame_id`** (§2 ROS rules).

The object layer alone (nodes, no edges) is exactly the simple object-recall
memory; the hierarchy is **additive** on top of it.

### 3. Substrate — Pydantic contract + in-memory graph + embeddable vector store

- **Pydantic v2 schemas** (`openral_core.schemas`) define every node/edge/query
  type — *the* contract (§1.3), owned on-disk schema, `schema_version` `"0.1"`.
- **`networkx`** (**BSD-3-Clause**) holds the graph in memory for
  containment/traversability/neighbor/shortest-path queries. Pure-Python, trivial
  at the hundreds–to–thousands-of-nodes scale of one robot; serializes via
  `node_link` JSON.
- **`sqlite-vec`** (**MIT OR Apache-2.0**, zero-dependency embeddable SQLite
  extension) persists nodes + optional embeddings + `map`-frame poses +
  timestamps and serves open-vocab nearest-neighbour retrieval — **one file, no
  server**. `faiss-cpu` (MIT) is the drop-in ANN alternative.
- **Migration path:** `spark_dsg` (**BSD-2-Clause**, prebuilt cp312 wheels, **no
  GTSAM/ROS** at install) provides the same objects→places→rooms→agents layered
  DSG natively with JSON serialization, if MIT-SPARK/Hydra interop is later
  wanted. Full Hydra (online graph construction from live SLAM) stays a separate
  future ADR. No graph **database** is needed (Neo4j is GPLv3, Memgraph is BSL —
  both fail §1.9; Kùzu (MIT) is the only clean DB option if ever required).

### 4. Worked example — "bring me a cup of wine" (what the representation answers vs. delegates)

Graph state (abbreviated): `room:living_room`, `room:kitchen`,
`traversable_to(living_room, kitchen)`; `object:fridge {is_container,
occludes_contents}` `at_place place:front_of_fridge`, `contains(fridge,
object:wine)`; `object:cabinet {is_container, occludes_contents}` `contains
glass`; `agent:requester pose@living_room` (captured at command time).

| Task step | **This ADR (representation) provides** | **Delegated to** |
|---|---|---|
| "wine is in the fridge" | `contains(fridge, wine)` edge if seen before; else the planner's commonsense seeds the hypothesis | planner (ADR-0018) supplies prior |
| go to the kitchen / fridge | `room:kitchen`, `place:front_of_fridge`, `traversable_to` path → nav goal `Pose6D` | nav skill (ADR-0024) |
| open the fridge door | `fridge.is_container & occludes_contents` ⇒ contents not observable until opened (a fact the planner reads) | articulated-open **skill** |
| find & grasp the wine | after opening, recall `wine` pose + camera-facing approach viewpoint (§6) | grasp skill |
| find a glass (search cabinets/table) | if `glass` in memory → its pose; **if not**, the graph supplies the *candidate places* (`cabinet`, `kitchen_table` regions) to search | **active search** = planner (ADR-0018) |
| pour wine into glass | `wine`, `glass` poses + `on`/`near` relations | pour **skill** |
| bring it to me (living room) | `agent:requester` pose → return nav goal | nav skill |

So the scene graph answers **"what exists, where it is, what contains what, how
to get there, and where the requester is."** It does **not** decide the *order*
of steps, *hypothesize* unseen glass locations, or *move* anything — those are
the planner and skills. The representation is what makes all of them queryable.

### 5. Open-vocabulary matching — optional, additive, compute-gated

`object` (and `place`/`room`) nodes carry an **optional CLIP/SigLIP embedding**
indexed in `sqlite-vec`, so *"the red wine"* / *"a wine glass"* match by cosine
similarity (the OK-Robot/DynaMem mechanism), not just exact labels. Additive:
with no embedder, matching falls back to `label` + relations + recency. The
embedder is selected explicitly (no hidden default, §1.4) and is the only
GPU-class cost — gated behind config, never required for the memory to function.

**GPU cost is small — because we embed per *object*, not per pixel/voxel.** This
is the key difference from the deferred dense feature fields (§Alternatives:
LERF/F3RM), which distill per-pixel CLIP across a whole scene (minutes + GBs).
Here the cost is *one image-encoder forward per detected object on observation*
plus *one text-encoder pass per query*. At ~5–20 objects/observation and a low
(~1–5 Hz, event-driven) memory-update rate, that is tens of encodes/sec —
trivial on any robot GPU (Jetson Orin included) and CPU-feasible with a small
model. Embeddings are ~512 floats = ~2 KB/object (1000 objects ≈ 2 MB), and
brute-force cosine over thousands of vectors is sub-millisecond (no ANN index
needed at robot scale). **Decision: the default embedder is OpenCLIP ViT-B/32
(MIT, dim 512)** — license-clean under §1.9 and single-digit-ms per object on the
reference GPU (verified host: RTX 4070, 8 GB); **SigLIP2-B/16 (Apache-2.0, dim
768)** is the higher-accuracy alternative. **MobileCLIP** has the best perf/CPU
story but its *weights* are Apple ML-Research TOU (non-OSI) — treat like GR00T
(install-time guard), code is MIT. The expensive path (dense feature fields) is
explicitly out of scope.

### 6. Query surface — read-only typed tools for the Reasoner

New **read-only** variants of the `ReasonerToolCall` union (ADR-0018), generated
into the tool palette — **no free-form JSON** (§3 Reasoner):

- **`RecallObjectTool`** — query by text (embedding) or `label`, with optional
  proximity/recency filters. Returns ranked matches, each with the object's
  **`map`-frame `Pose6D`** + provenance **and** a computed **approach
  viewpoint**: a base/EE goal `Pose6D` at a configurable standoff along the
  approach vector, oriented so the **gripper-mounted camera faces the object**
  (using the camera mount `SensorSpec.frame_id` via TF2). When the object is
  inside a closed container, the result flags the containing node so the planner
  knows it must open it first.
- **`ResolvePlaceTool`** — resolve a `room`/`place`/`agent` reference (e.g.
  "the kitchen", "where I was standing") to a navigation goal `Pose6D`, and
  return the `traversable_to` path to it.

This ADR defines `RecallObjectTool` / `ResolvePlaceTool` as **query/result
contracts** (read-only, side-effect-free, no actuation, no `ROSSafetyViolation`).
Their *exposure to the closed `ReasonerToolCall` palette*, the *planning* that
strings them together, and *active search* when a query returns nothing are the
Reasoner's concern and are specified in **[ADR-0039](0039-llm-task-planning-active-search.md)**
— out of scope here (§Context). A decision is replayable from the trace + the
stored nodes/stamps (§1.8).

### 7. Provenance and dynamics

Re-observing a node (matched by `track_id`, else `label` + proximity) updates its
pose and bumps `last_seen_ns` / `observation_count`. A node not re-observed where
expected is **flagged stale, never silently deleted** — the planner decides
whether to re-verify (a DynaMem-style "is it still there?" check). Full
spatio-temporal change detection (Khronos; BSD-3) is a deferred upgrade.

### 8. Contract surface added by this ADR (contracts-only first PR)

- **Pydantic (`openral_core.schemas`)**: `SpatialNodeKind` enum (`object | place
  | room | agent`); `SpatialNode` (kind, `Pose6D` in `map`, optional `label`,
  `bbox_3d`, `embedding_ref`, `is_container`, `occludes_contents`, provenance) —
  `object` nodes reuse `DetectedObject`/`Pose6D`, no duplication (§1.13);
  `SpatialRelationKind` enum (`contains | at_place | traversable_to | on |
  near`); `SpatialEdge` (src, dst, kind); `SceneGraph` container (nodes, edges,
  `schema_version`); `RecallObjectQuery`/`RecallObjectResult` (recall pose + approach
  viewpoint + containing-node flag); `ResolvePlaceQuery`/`ResolvePlaceResult`
  (goal pose + path).
- **Query contracts**: `RecallObjectQuery` / `RecallObjectResult` (with
  `RecallObjectMatch` + `ApproachViewpoint`) and `ResolvePlaceQuery` /
  `ResolvePlaceResult`. The `RecallObjectTool` / `ResolvePlaceTool` palette
  variants that expose these to the reasoner are **[ADR-0039](0039-llm-task-planning-active-search.md)**
  (the closed `ReasonerToolCall` palette extension lives there, with the ADR-0018
  dispatch + CLAUDE.md read-surface note).
- **Exception**: `ROSObjectNotInMemory(ROSPerceptionStale)` — a query matching
  nothing / only stale nodes degrades by returning "unknown", never by
  fabricating a pose (§1.2, §1.4).
- **Package**: builder/store + graph in a Layer-2 module (`openral_world_state`
  or an `openral_core` submodule — decided in the implementing PR).

`schema_version` stays `"0.1"` and all new fields default empty / `None`, so
every existing manifest and fixture loads unchanged (§1.6, no migrators). The
implementing PR updates `docs/METHODS.md` and the repo state map (§4.3).

### 9. License posture (verified against upstream)

| Use (Apache / MIT / BSD — clean) | Reject / TSC-review (copy-left or restricted) |
|---|---|
| `networkx` (BSD-3), `sqlite-vec` (MIT/Apache-2.0), `pydantic` (MIT) | **Neo4j** Community (**GPLv3**, server AGPLv3) |
| `spark_dsg` (BSD-2) — interop/migration path | **Memgraph** core (BSL — source-available) |
| `faiss-cpu` (MIT), Qdrant local (Apache-2.0), Chroma (Apache-2.0) — vector alternatives | **PostGIS** / `pgvector` server (pgvector permissive but **needs a server**, not embeddable) |
| CLIP / SigLIP open-vocab embedders (MIT/Apache) | **FoundationPose** (NVIDIA **non-commercial** — guard like GR00T if ever used for object pose) |

`sqlite-vec` supersedes `sqlite-vss`; **Kùzu** (MIT, embeddable) is the only
license-clean graph-*database* option, if one is ever wanted.

## Alternatives considered

- **Flat object memory only (no hierarchy).** Sufficient for "find the mug",
  but cannot express containment, rooms, or the requester's location, so it
  cannot serve "bring me a cup of wine". Retained as the **object layer** of this
  graph, not as the whole design.
- **Open-vocabulary voxel/point-feature map (OK-Robot, DynaMem).** A dense CLIP
  feature per voxel; proven for nav+pick and reuses the OctoMap leg, but needs a
  GPU embedder over every voxel and gives no object identity, containment, or
  rooms. **Retained as a fallback** for objects never explicitly detected; not
  the primary representation.
- **Adopt full Hydra now.** Real-time BSD-2 scene-graph SLAM with the exact
  objects/places/rooms/agents layers — but heavy (GTSAM, ROS, optional GPU
  segmentation) and it *builds* the graph from its own SLAM front-end,
  duplicating OpenRAL perception. We adopt the **Spark-DSG data structure** as an
  interop path, not the producer; Hydra is a future ADR.
- **Neural feature fields (LERF/F3RM/LangSplat).** Offline per-scene, not
  online-updatable, GPU-heavy. A future manipulation-memory R&D track.
- **Specify planning / active search / manipulation in this ADR.** Rejected to
  keep it a focused Layer-2 world-model ADR; planning+search is ADR-0018, skills
  are ADR-0024.
- **Make the memory a safety input.** Rejected outright (§1.1): remembered
  geometry is stale and uncertain; the kernel gates only on the live, bounded
  ADR-0030 world. Memory informs *planning*, never the *veto*.

## Consequences

- The Reasoner gains a durable, queryable world model: object recall *and* the
  rooms/places/containment/agent structure that long-horizon household tasks
  ("bring me wine") require. The wine task becomes expressible — its planning,
  active search, and skills are tracked in ADR-0018 / ADR-0024.
- A new persistent artifact (the scene graph + SQLite store) is lifecycle-managed
  and versioned; it is replayable from the trace (§1.8).
- The memory layer adds **no safety surface**: read-only to actuation, raises no
  `ROSSafetyViolation`, kernel guarantees unchanged. No safety-WG review for the
  memory layer; the implementing PR still ships unit + integration + sim tests
  against real fixtures (§1.7, §1.11).
- New deps (`networkx`, `sqlite-vec`, optional embedder) are Apache/MIT/BSD — no
  license review; rejected copy-left options are documented.

## Phasing

1. **Contracts + this ADR (no behaviour change).** Node/edge/graph + query
   schemas, `RecallObjectTool` / `ResolvePlaceTool`, `ROSObjectNotInMemory`, a real
   fixture (a small home scene graph — kitchen/living-room/fridge/wine/cabinet —
   under `examples/sim/`), `docs/METHODS.md` + repo-state-map updates, and
   round-trip / JSON-Schema fuzz tests.
2. **Object foundation.** Accumulate `WorldState.detected_objects` into `object`
   nodes (instance association by `track_id`, else `label` + proximity),
   `map`-anchored, with provenance; `sqlite-vec` persistence. `RecallObjectTool` +
   the camera-facing approach-viewpoint helper. Sim tests on a real MuJoCo scene
   (no mocks, §1.11).
3. **Hierarchy.** `place` / `room` / `agent` nodes and `contains` / `at_place` /
   `traversable_to` edges (places from clustering free space / nav waypoints;
   rooms from grouping; the requester `agent` captured at command time);
   `networkx` traversal; `ResolvePlaceTool`. Containment + container attributes.
4. **Open-vocabulary matching (done — `OpenClipEmbedder`).** Optional
   `TextEmbedder` (OpenCLIP `ViT-B-32-quickgelu` / `openai`, MIT;
   `uv sync --group clip`): object labels are embedded on creation and a
   free-text query matches by CLIP cosine ≥ `min_text_similarity` (default 0.85,
   calibrated) in addition to label substring — recalling synonyms/paraphrases
   ("red wine" → "bottle of wine") that label matching misses. Compute-gated and
   GPU-verified (RTX 4070). Implemented as in-memory brute-force cosine over the
   per-object label embeddings (sub-ms at robot scale; labels re-embedded on
   load, so no vector persistence needed yet). **Deferred:** image-crop
   embeddings (true visual open-vocab — needs an ingest contract that carries
   pixels) and a `sqlite-vec` store (only earns its place once embeddings can't
   be recomputed from labels).
5. **Dynamics.** DynaMem-style re-verification / staleness; (later) Khronos-style
   change detection or Spark-DSG interop.

**Observability (done — dashboard surfacing).** The remembered `object` nodes are
published as a `world.scene_objects` OTel span (`openral_world_state.emit_scene_objects_span`,
attrs under `openral.world_state.scene_objects.*`) and rendered on the ADR-0017
dashboard two ways: a **"scene objects" table card** (label · map-frame position ·
confidence · last-seen · obs-count) and **labelled markers overlaid on the SLAM 2D
map** (reusing the occupancy-grid `worldToPixel` transform — objects share the
robot's `map` frame). Advisory telemetry only, never a safety input. Emitted by
the Reasoner from its preloaded `spatial_memory_path` map today; once the
perception object-lift producer lands (ADR-0035 / PR #229) the World-State node
becomes the canonical emitter of the same span — no dashboard change needed.

Planning + active search (ADR-0018) and the manipulation skills (ADR-0024) that
complete the wine task are tracked under their own ADRs. Each runtime phase ships
sim tests against real fixtures and updates all affected docs in the same PR
(§1.14).

## Boundary with ADR-0035 (perception object-lift) — the live feed

ADR-0035 (PR #229) added the **short-horizon** producer: the world-state node
lifts 2D detections to 3D (`VoxelFrustumLifter`) and stabilises them frame-to-frame
in an `ObjectMemory` (IoU-gated association, FOV-guarded eviction, per-session
monotonic `track_id`), populating `WorldState.detected_objects` in the `map` frame.
That is **not** durable memory — it forgets an object shortly after it leaves view,
and its `track_id` resets whenever the node restarts.

This ADR's `SpatialMemory` is the **long-horizon** consumer. The hand-off is the
`DetectedObject` list on the snapshot — no new contract. The boundary:

| | ADR-0035 `ObjectMemory` | ADR-0038 `SpatialMemory` |
|---|---|---|
| Horizon | seconds (current view) | the whole task / session, persisted |
| Eviction | aggressive (in-FOV miss → drop) | never — "last seen at X" endures |
| Identity | per-session `track_id` | label + proximity (durable) |
| Home | world-state node | the reasoner's query backend |

The edge is wired in the reasoner (it already receives `/openral/world_state_slow`
and is the sole querier): with `spatial_memory_ingest`, each tick folds
`snapshot.detected_objects` into the durable graph. Because `ObjectMemory`'s
`track_id` is not stable across restarts, `SpatialMemory._associate` treats it as
a within-session hint **guarded by label match** and otherwise associates by
label + proximity — so a recycled id can't merge a cup into a mug's node, and an
object that returns under a fresh id re-associates with its existing node.

## End-to-end validation (deploy sim + panda_mobile)

The full chain — camera → detector → lift → durable memory → recall + dashboard —
runs under one command once the workspace is colcon-built:

```bash
openral deploy sim --robot panda_mobile \
  --config scenes/deploy/robocasa_pnp.yaml \
  --enable-octomap --enable-object-detector --spatial-memory-ingest
```

- `panda_mobile` HAL (SimAttachedHAL) publishes `/openral/cameras/<cam>/image`
  (ADR-0034 sensor bridge) and, with `--enable-octomap`, `/openral/world_voxels`.
- `ros_image_detector_node` (RT-DETR) publishes `ObjectsMetadata` on
  `/openral/perception/objects`.
- The world-state node lifts to 3D and fills `WorldState.detected_objects`
  (ADR-0035), serialised onto `/openral/world_state_slow`.
- The reasoner (`spatial_memory_ingest`, auto-on with the detector) accumulates
  the durable `SpatialMemory`; a `recall_object` query recalls a seen object with
  its 3D pose + camera-facing `ApproachViewpoint`, and the dashboard shows the
  `scene objects` card + SLAM-map markers.

Automated coverage: `tests/integration/test_object_lift_world_state.py` (producer,
ADR-0035) + `tests/integration/test_reasoner_node_end_to_end.py::test_spatial_memory_ingest_accumulates_from_world_state`
(consumer, this ADR, `OPENRAL_TEST_ROS_LIVE=1`). The headless full-graph smoke is
gated by `OPENRAL_DEPLOY_SIM_SMOKE=1` on a built workspace.

## Amendment 2026-06-08 — three-tier scene paths

ADR-0041 split `scenes/` into deploy/sim/benchmark tiers. The end-to-end
validation command above now points at `scenes/deploy/robocasa_pnp.yaml`
because there is no DeployScene sibling for the old
`scenes/benchmarks/panda_mobile_navigate_kitchen.yaml` and
`openral deploy sim` rejects non-DeployScene tiers strictly. The
substrate (`panda_mobile` HAL in a robocasa kitchen) is unchanged; the
robocasa scene id is `PickPlaceCounterToCabinet` rather than
`NavigateKitchen`, which does not affect the camera → detector → lift
→ durable memory pipeline exercised by `--enable-octomap
--enable-object-detector --spatial-memory-ingest`. See ADR-0041 and
[`scenes/README.md`](https://github.com/OpenRAL/openral/blob/master/scenes/README.md) for the per-tier strict-CLI
matrix.
