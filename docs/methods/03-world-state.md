# Layer 3 — World State

> Part of the OpenRAL [public-symbol inventory](../METHODS.md). Hand-curated; `(LNN)` markers are refreshed by `tools/refresh_methods_linenos.py`.

### `python/state_adapter/src/openral_state_adapter/`
_ADR-0027 — layout adapter registry. Assembles per-checkpoint state vectors from manifest-declared `StateContractBindings` + live `/tf` + live `/joint_states`. Pure-Python, rclpy-free; the skill_runner wraps `tf2_ros.Buffer.lookup_transform` into the `TfLookup` Protocol at call time._

- `@dataclass TransformView` — rclpy-free view of a `geometry_msgs/TransformStamped`. Fields: `position: tuple[float, float, float]`, `quaternion_xyzw: tuple[float, float, float, float]`.
- `Protocol TfLookup` — `__call__(target_frame: str, source_frame: str) -> TransformView`. Implementations MUST raise on missing transforms — assembler never silently substitutes identity.
- `Protocol Assembler` — `__call__(bindings: StateContractBindings, joint_positions: dict[str, float], tf_lookup: TfLookup) -> NDArray[float32]`. Pure-function signature every layout file implements.
- `register(layout: StateLayout, assembler: Assembler) -> None` — Bind `assembler` to `layout` in the package-global registry. Layout files call this at module load.
- `registered_layouts() -> frozenset[StateLayout]` — Snapshot of currently-registered layouts. Reasoner palette filter consults this to admit wrapped-task-space rSkills.
- `assemble_state(layout, bindings, joint_positions, tf_lookup) -> NDArray[float32]` — Look up the assembler for `layout` and run it. Raises `ROSConfigError` when no assembler is registered.
- `assemble_human300_16d(bindings, joint_positions, tf_lookup) -> NDArray[float32]` — RoboCasa365 / pi05_pretrain_human300 layout: `[base_to_eef.pos(3), base_to_eef.quat(4), world_to_base.pos(3), world_to_base.quat(4), gripper_qpos(2)] = 16`. Registered as `"human300_16d"` at import.
- `assemble_libero_eef8d(bindings, joint_positions, tf_lookup) -> NDArray[float32]` — LIBERO task-space layout: `[eef_pos(3), eef_axisangle(3), gripper_qpos(2)] = 8`, world-frame EE pose via `tf_lookup(bindings.world_frame, bindings.eef_frame)` + axis-angle (byte-matching the benchmark `openral_sim.backends.libero._quat_to_axisangle`, `[x,y,z,w]`→`axis·2·acos(w)`) + gripper (1 joint mirrored to `[v,-v]` or 2 per-finger). Registered as `"libero_eef8d"` at import. Lets `openral deploy sim` feed the lerobot/smolvla_libero (and pi05-/xvla-libero) checkpoints the same task-space proprio they get in the benchmark (else the runner falls back to raw joint-space state and the policy fails). `world_frame` defaults to `"map"` (SLAM root) on the binding — LIBERO manifests MUST set it to the HAL sim root.

### `python/world_state/src/openral_world_state/aggregator.py`
_WorldStateAggregator — tf2-aware, injectable snapshot producer._

- `class WorldStateAggregator` — Aggregates sensor data and produces `WorldState` snapshots. (L107)
  - `__init__(description, *, staleness_limit_s=DEFAULT_STALENESS_S, clock_fn=None)` (L160)
  - `update_joint_state(state) -> None` — Record a fresh joint reading. (L235)
  - `update_image(sensor_name, topic, stamp_ns) -> None` — Record image arrival. (L246)
  - `update_ee_pose(ee_name, pose) -> None` — Record EE pose from tf2. (L302)
  - `update_base_pose(pose, twist=None) -> None` — Record base pose (and optional twist). (L316)
  - `update_battery(pct) -> None` — Record battery %. (L333)
  - `set_error(component, status='error') -> None` — Latch a forced diagnostic. (L357)
  - `clear_error(component) -> None` — Remove a forced diagnostic. (L373)
  - `snapshot() -> WorldState` — Produce a typed snapshot (hot path, acquires lock). Emits a `world_state.snapshot` OTel span with `openral.world_state.components_stale` + `openral.world_state.has_latched_error` attributes, fires `openral.event.staleness_latched` / `openral.event.error_latched` events on first transition, records per-component `openral.world_state.staleness_ms` histogram + `openral.world_state.components_stale` up-down counter. (L384)
  - `update_detected_objects(objects: list[DetectedObject]) -> None` — Replace the remembered detected-object set (ADR-0035). Thread-safe; the next `snapshot()` call returns the new list. Called by the world-state lifecycle node's memory tick. (L342)
  - `_emit_snapshot_telemetry(span, diag, ages_ms) -> None` — Internal: lift the snapshot diagnostics onto the OTel span + meter instruments. (L501)

### `python/world_state/src/openral_world_state/spatial_memory.py`
_SpatialMemory — ADR-0038 Phase 2 persistent object-centric scene-graph memory (advisory; never a safety input)._

- `compute_approach_viewpoint(target, *, standoff_m=DEFAULT_STANDOFF_M, camera_frame_id=DEFAULT_CAMERA_FRAME, approach_from=None) -> ApproachViewpoint` — Standoff pose `standoff_m` from `target` (on the `approach_from` side, else −X), yawed so the gripper camera faces it. (ADR-0038)
- `class SpatialMemory` — Accumulates `WorldState.detected_objects` into a queryable `SceneGraph`; pure-Python (typed BFS for traversal, JSON persistence — no graph-engine dep). (ADR-0038)
  - `__init__(*, assoc_distance_m=DEFAULT_ASSOC_DISTANCE_M, default_standoff_m=DEFAULT_STANDOFF_M, camera_frame_id=DEFAULT_CAMERA_FRAME, map_frame=DEFAULT_MAP_FRAME, embedder=None, min_text_similarity=DEFAULT_MIN_TEXT_SIMILARITY)` — `embedder` (ADR-0038 §5, optional `TextEmbedder`) enables open-vocab matching: object labels are embedded on creation and free-text queries match by CLIP cosine ≥ `min_text_similarity`.
  - `upsert_node(node) -> None` / `add_edge(src, dst, kind) -> None` — Mutators (edge endpoints must exist).
  - `ingest_detected_objects(objects, *, now_ns) -> list[str]` — Fold detections into object nodes; instance association by `track_id` else label+proximity (`assoc_distance_m`); updates pose/last_seen/observation_count, keeps higher confidence; embeds the label when an embedder is set. Returns touched node ids.
  - `recall_object(query: RecallObjectQuery, *, now_ns) -> RecallObjectResult` — Rank object nodes by `max(label-match, CLIP-cosine)` × confidence (proximity/recency tiebreak); an embedding hit needs cosine ≥ `min_text_similarity`, a label substring always qualifies. Each match carries an `ApproachViewpoint` + `inside_container_id`. Empty result = unknown (caller may raise `ROSObjectNotInMemory`).
  - `resolve_place(query: ResolvePlaceQuery, *, from_node_id=None) -> ResolvePlaceResult` — Resolve a place/room/agent reference to a goal pose (an object resolves to its `at_place`) + a `traversable_to` BFS path. Raises `ROSObjectNotInMemory` when unresolved.
  - `to_scene_graph() -> SceneGraph` / `from_scene_graph(graph, *, embedder=None) -> SpatialMemory` (classmethod) — Snapshot / rebuild; rebuild re-embeds labels when an embedder is given (embeddings aren't serialized).
  - `save(path) -> None` / `load(path, *, embedder=None) -> SpatialMemory` (classmethod) — JSON persistence via the `SceneGraph` contract.
  - Module constants: `DEFAULT_ASSOC_DISTANCE_M`, `DEFAULT_STANDOFF_M`, `DEFAULT_CAMERA_FRAME`, `DEFAULT_MAP_FRAME`, `DEFAULT_MIN_TEXT_SIMILARITY`.

### `python/world_state/src/openral_world_state/geometry.py`
_ADR-0044 Phase 1 — shared gaze geometry. **Relocated to `openral_core.geometry` in ADR-0065**; this module is now a thin re-export shim so `from openral_world_state.geometry import …` keeps working. Canonical entries (`ViewAxis`, `look_at_quat_wxyz`, `compute_gaze_pose`, `rotation_to_quat_wxyz`) are documented under [00-core-schemas.md](00-core-schemas.md)._

### `python/world_state/src/openral_world_state/grid.py`
_ADR-0044 Phase 2 — occupancy-grid queries + approach-pose refinement (planning-layer proposal; the ADR-0030 Phase 6 kernel `check_nav_goal` gate stays the enforcement)._

- `FREE_MAX` (module constant, `25`) — Highest `nav_msgs/OccupancyGrid` value still treated as free; `-1` unknown and anything above block placement and sight (conservative).
- `class OccupancyGridIndex` — Queryable view over one grid snapshot; ROS-free (`from_msg` duck-types the message). `__init__(data (h,w) int8, *, resolution_m, origin_xy, origin_yaw=0.0)`; handles rotated origins. (ADR-0044)
  - `from_msg(msg) -> OccupancyGridIndex` (classmethod) — decode a (duck-typed) `nav_msgs/OccupancyGrid`.
  - `world_to_cell(x, y) -> tuple[row, col] | None` — `None` off-grid. `resolution_m` property.
  - `is_free(x, y, *, inflation_m=0.0) -> bool` — point + world-space inflation disc all free; off-grid (or disc off-grid) counts blocked.
  - `line_of_sight(a_xy, b_xy) -> bool` — Bresenham; every cell strictly before the endpoint free (the target's own footprint cell is exempt — a mug shares its cell with its counter).
- `refine_approach_pose(grid, viewpoint, target_xyz, *, inflation_m=0.25, max_radius_m=2.0, min_standoff_m=None, max_standoff_m=None) -> ApproachViewpoint | None` — Return the viewpoint unchanged when already free + sighted; else ring-search outward for the nearest admissible point (standoff within `[0.5x, 2.0x]` ideal by default), re-aim via `compute_approach_viewpoint`. `None` = no reachable viewpoint (caller reports honestly, never fabricates). (ADR-0044)

### `python/world_state/src/openral_world_state/scene_objects_span.py`
_ADR-0038 — publish the durable object nodes as a dashboard OTel span (advisory; never a safety input)._

- `scene_objects_payload(graph) -> list[dict[str, object]]` — Project a `SceneGraph`'s `object`-kind nodes to JSON-friendly dicts (`id,label,x,y,z,frame_id,confidence,last_seen_ns,observation_count,is_container`); non-object nodes skipped.
- `emit_scene_objects_span(graph, *, source_node) -> int` — Emit one `world.scene_objects` span carrying the object list (attrs under `openral.world_state.scene_objects.*`); returns the object count. Emitted by the Reasoner from its preloaded map today; the World-State node post-producer (ADR-0035 / PR #229).

### `python/world_state/src/openral_world_state/embedder.py`
_ADR-0038 §5 open-vocabulary text embedder (optional; `uv sync --group clip`)._

- `class TextEmbedder(Protocol)` — `dim: int` + `embed_text(texts) -> NDArray[float32]` (L2-normalized).
- `class OpenClipEmbedder` — OpenCLIP `ViT-B-32-quickgelu` / `openai` weights (MIT). `__init__(*, model_name=DEFAULT_CLIP_MODEL, pretrained=DEFAULT_CLIP_PRETRAINED, device=None)` raises `ROSConfigError` if open-clip-torch/torch missing or weights unfetchable; lazy-imports torch/open_clip so the base install stays light. `dim` (512); `embed_text(texts)`.
- Module constants: `DEFAULT_CLIP_MODEL` (`"ViT-B-32-quickgelu"`), `DEFAULT_CLIP_PRETRAINED` (`"openai"`).

### `python/world_state/src/openral_world_state/object_lift.py`
_2D→3D object-center lift helpers — pure, ROS-free, unit-testable (ADR-0035)._

- `class ObjectsLiftError(ValueError)` — Raised by geometry helpers on degenerate inputs (zero-norm quaternion; occupancy buffer size mismatch). (L37)
- `homogeneous_from_quat_xyz(translation, quat_xyzw) -> NDArray[np.float64]` — Build a 4×4 homogeneous transform from a translation + xyzw quaternion (normalised). Raises `ObjectsLiftError` if the quaternion norm is effectively zero. (L41)
- `decode_occupied_centers(*, origin, resolution, size_xyz, occupancy) -> NDArray[np.float64]` — Occupied voxel centres `(N, 3)` in the grid frame; row-major x-fastest. Center of cell `(ix, iy, iz)` = `origin + (index + 0.5) * resolution`. Returns `(0, 3)` when none are occupied. Raises `ObjectsLiftError` if `len(occupancy) != size_x * size_y * size_z`. (L80)
- `depth_cloud_to_centers_base(points_cloud, t_base_from_cloud, *, max_points=0) -> NDArray[np.float64]` — #11 octomap-free fallback depth source for `VoxelFrustumLifter`: drops non-finite returns, uniformly subsamples to `max_points` (0 = no cap), and maps the `(N, 3)` cloud from its optical frame into the base frame. Output is interchangeable with `decode_occupied_centers` as the lifter's `occupied_centers_base`. Returns `(0, 3)` when the cloud has no finite points. (L126)
- `aabb_iou_3d(a, b) -> float` — 3D axis-aligned bbox IoU in `[0, 1]`. Boxes are `(xmin, ymin, zmin, xmax, ymax, zmax)`. Returns 0.0 for disjoint or degenerate boxes. This is the single IoU helper in the repo — reuse it; do not add another. (L162)
- `class VoxelFrustumLifter(*, k_nearest=25, min_voxels=3)` — Lift 2D detections to 3D object centres via voxel-frustum K-nearest. Frustum-projects `/openral/world_voxels` occupied centres into the RGB camera image plane (using `SensorSpec.intrinsics` + TF2 transforms), selects the K voxels nearest each box centre, and returns one `DetectedObject` per detection with `pose.frame_id = "map"`. Skips a detection when fewer than `min_voxels` voxels survive the frustum filter. (L188)
  - `lift(*, detections, occupied_centers_base, intrinsics, frame_size, t_cam_from_base, t_map_from_base, map_frame="map") -> list[DetectedObject]` — Core lift call (vectorised over voxels; per-detection masking after). Returns `[]` when `occupied_centers_base` is empty (best-effort). `track_id` is `None`; assigned later by `ObjectMemory`. (L207)
- `build_in_fov_predicate(*, intrinsics, t_cam_from_map) -> Callable[[DetectedObject], bool]` — Build a predicate that returns `True` iff the object's `map`-frame centre is in front of the camera and projects within the image bounds (`(0, 0)` to `(intrinsics.width, intrinsics.height)`); behind-camera or out-of-bounds → `False`. Used by `ObjectMemory.tick` for FOV-guarded eviction. (L298)

### `python/world_state/src/openral_world_state/object_memory.py`
_IoU-gated spatial object memory — pure, ROS-free, unit-testable (ADR-0035)._

- `class ObjectMemory(*, iou_threshold=0.3, max_misses=1)` — IoU-gated spatial memory with freeze-on-match, in-FOV-guarded eviction, and out-of-FOV retention. Maintains a monotonic `track_id` counter. (L30)
  - `tick(candidates: list[DetectedObject], *, stamp_ns: int, in_fov: Callable[[DetectedObject], bool]) -> list[DetectedObject]` — Run one association+eviction step. Greedy highest-confidence-first matching: same-label + `aabb_iou_3d ≥ iou_threshold` → freeze (leave pose/bbox unchanged, bump confidence, reset miss count); no match → new track. Unmatched existing tracks: `in_fov` true → `miss_count += 1`; evict when `miss_count >= max_misses`; `in_fov` false → retain unchanged. Returns surviving tracks as `list[DetectedObject]`. (L64)
