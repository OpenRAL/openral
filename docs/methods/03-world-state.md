# Layer 3 — World State

> Part of the OpenRAL [public-symbol inventory](../METHODS.md). Hand-curated; `(LNN)` markers are refreshed by `tools/refresh_methods_linenos.py`.

### `python/state_adapter/src/openral_state_adapter/`
_Layout adapter registry that assembles per-checkpoint state vectors from manifest-declared `StateContractBindings` plus live `/tf` and `/joint_states`. Pure-Python and rclpy-free; the skill_runner supplies `TfLookup` via `tf2_ros.Buffer.lookup_transform` at call time._

- `@dataclass TransformView` (_protocol.py L21) — rclpy-free view of a `geometry_msgs/TransformStamped` (position + quaternion_xyzw).
- `Protocol TfLookup` (_protocol.py L36) — `__call__(target_frame: str, source_frame: str) -> TransformView`. Implementations MUST raise on missing transforms — assembler never silently substitutes identity.
- `Protocol Assembler` (_protocol.py L49) — `__call__(bindings: StateContractBindings, joint_positions: dict[str, float], tf_lookup: TfLookup) -> NDArray[float32]`. Pure-function signature every layout file implements.
- const `_LAYOUT_ASSEMBLERS: dict[StateLayout, Assembler]` (_registry.py L29) — Package-global `layout → assembler` map; layout files populate it at import via `register`.
- `register(layout: StateLayout, assembler: Assembler) -> None` (_registry.py L32) — Bind `assembler` to `layout` in the package-global registry. Layout files call this at module load.
- `registered_layouts() -> frozenset[StateLayout]` (_registry.py L43) — Snapshot of currently-registered layouts. Reasoner palette filter consults this to admit wrapped-task-space rSkills.
- `assemble_state(layout, bindings, joint_positions, tf_lookup) -> NDArray[float32]` (_registry.py L55) — Look up and run the assembler for `layout` after checking every bound joint is present. Raises `ROSConfigError` when no assembler is registered or a bound joint is missing; `ROSPerceptionStale` when no joint frame has arrived at all.
- const `_DIM = 16` (layouts/human300_16d.py L35) — Output vector width.
- const `_N_GRIPPER_JOINTS = 2` (layouts/human300_16d.py L36) — Per-finger gripper-joint count.
- `assemble_human300_16d(bindings, joint_positions, tf_lookup) -> NDArray[float32]` (layouts/human300_16d.py L86) — RoboCasa365 / pi05_pretrain_human300 16-D task-space layout (EE pose + base pose + gripper). Registered as `"human300_16d"` at import.
- const `_DIM = 8` (layouts/libero_eef8d.py L41) — Output vector width.
- const `_N_GRIPPER_JOINTS = 2` (layouts/libero_eef8d.py L42) — Per-finger gripper-joint count.
- const `_EPS = 1e-10` (layouts/libero_eef8d.py L43) — Near-identity threshold below which `_quat_xyzw_to_axisangle` returns the zero vector rather than dividing by a ~zero norm.
- `assemble_libero_eef8d(bindings, joint_positions, tf_lookup) -> NDArray[float32]` (layouts/libero_eef8d.py L67) — LIBERO 8-D task-space layout (EE pos + axis-angle + gripper), byte-matching the benchmark's `_quat_to_axisangle` so `deploy sim` feeds LIBERO checkpoints the same proprio as the benchmark. `world_frame` defaults to `"map"` — LIBERO manifests MUST override it to the HAL sim root. Registered as `"libero_eef8d"` at import.
- `rc365` layout (`layouts/rc365.py`) registers `assemble_human300_16d` above under the `"rc365"` key — same 16-D vector, re-sliced downstream by the RLDX-1 policy adapter; no separate assembler.

### `python/world_state/src/openral_world_state/aggregator.py`
_WorldStateAggregator — tf2-aware, injectable snapshot producer._

- const `DEFAULT_RATE_HZ: float = 30.0` — Advertised default snapshot rate. (L91)
- const `DEFAULT_STALENESS_S: float = 0.5` — General component staleness window, covering every heterogeneous-rate component with margin over the slowest expected stream; a freshness indicator, not a safety gate. (L98)
- const `DEFAULT_POLICY_STATE_STALENESS_S: float = 5.0` — Staleness window for `policy_state`, which is step-locked not rate-locked; wide enough for a heavy sidecar sim while still flagging a wedged one faster than its 120 s ZMQ timeout. (L105)
- const `DEFAULT_TRACKED_OBJECTS_STALE_S: float = 1.5` — Age past which the object tracker's last set is dropped from the snapshot (three missed beats of its ≥ 2 Hz heartbeat, issue #418). Calibration point. (L108)
- `class WorldStateAggregator` — Aggregates sensor data and produces `WorldState` snapshots. (L111)
  - `__init__(description, *, staleness_limit_s=DEFAULT_STALENESS_S, image_staleness_limit_s=None, policy_state_staleness_limit_s=DEFAULT_POLICY_STATE_STALENESS_S, clock_fn=None, tracked_objects_stale_s=DEFAULT_TRACKED_OBJECTS_STALE_S)` — Camera and policy-state streams get independent staleness windows: `image_staleness_limit_s` defaults to the general window but deploy sim passes 5.0 s for slow rendered frames vs 0.5 s for real deploys; `policy_state` defaults to 5.0 s since it's step-locked. (L174)
  - `update_joint_state(state) -> None` — Record a fresh joint reading. (L291)
  - `update_policy_state(values) -> None` — Store a defensive copy of simulator-native checkpoint proprioception for `WorldState.policy_state`. (L314)
  - `update_image_frame(sensor_name, frame: SensorFrame) -> None` — Record an inline pixel payload for a named sensor (unlike `update_image`, which stores only the topic ref). (L325)
  - `update_image(sensor_name, topic, stamp_ns) -> None` — Record image arrival. (L302)
  - `update_ee_pose(ee_name, pose) -> None` — Record EE pose from tf2. (L359)
  - `update_base_pose(pose, twist=None) -> None` — Record base pose (and optional twist). (L373)
  - `update_battery(pct) -> None` — Record battery %. (L390)
  - `update_attached_objects(objects: list[AttachedCollisionObject], *, revision=0, stamp_ns=None, place_declaration: PlaceDeclaration | None = None, grasp_declaration: GraspDeclaration | None = None) -> None` — Atomically replaces the complete attached-payload set; duplicate ids or backwards revisions raise `ValueError`. `place_declaration` is replaced in the same atomic step as its payload, and liveness is judged against the stream's own `stamp_ns`, never `clock_fn`, so a sim-clock declaration is never wrongly judged stale by the wall clock. `grasp_declaration` is stored under the same atomic step and the same liveness gate; a set that carries none clears it. (L414)
  - `update_tracked_objects(tracked: TrackedObjectSet) -> None` — Store the object tracker's newest set (issue #418), stamped with its arrival on `clock_fn`; `snapshot` relays it verbatim as `WorldState.tracked_objects` (the kernel judges frame, caps, revision, stamp) and drops it (`None`, i.e. `tracked_objects_valid` false) once older than `tracked_objects_stale_s`. (L494)
  - `set_error(component, status='error') -> None` — Latch a forced diagnostic. (L515)
  - `clear_error(component) -> None` — Remove a forced diagnostic. (L531)
  - `snapshot() -> WorldState` — Produce a typed snapshot (hot path, acquires lock), emitting OTel span/metrics for staleness and latched errors. `staleness_latched` fires only for a component that has had data before — a never-received one counts as stale but doesn't latch, so bringup stays quiet before the HAL's first publish. (L542)
  - `update_detected_objects(objects: list[DetectedObject]) -> None` — Replace the remembered detected-object set (thread-safe); the next `snapshot()` reflects it. Called by the world-state lifecycle node's memory tick. (L399)
  - `_emit_snapshot_telemetry(span, diag, ages_ms) -> None` — Internal: lift the snapshot diagnostics onto the OTel span + meter instruments. (L680)

### `python/world_state/src/openral_world_state/spatial_memory.py`
_SpatialMemory — persistent object-centric scene-graph memory (advisory; never a safety input)._

- `compute_approach_viewpoint(target, *, standoff_m=DEFAULT_STANDOFF_M, camera_frame_id=DEFAULT_CAMERA_FRAME, approach_from=None) -> ApproachViewpoint` — Standoff pose `standoff_m` from `target` (on the `approach_from` side, else −X), yawed so the gripper camera faces it. (L97)
- `class SpatialMemory` — Accumulates `WorldState.detected_objects` into a queryable `SceneGraph`; pure-Python (typed BFS for traversal, JSON persistence — no graph-engine dep). (L140)
  - `__init__(*, assoc_distance_m=DEFAULT_ASSOC_DISTANCE_M, default_standoff_m=DEFAULT_STANDOFF_M, camera_frame_id=DEFAULT_CAMERA_FRAME, map_frame=DEFAULT_MAP_FRAME, embedder=None, min_text_similarity=DEFAULT_MIN_TEXT_SIMILARITY)` — `embedder` (optional `TextEmbedder`) enables open-vocab matching: object labels are embedded on creation and free-text queries match by CLIP cosine ≥ `min_text_similarity`. (L151)
  - `upsert_node(node) -> None` — Mutator; add or replace a node. (L187)
  - `add_edge(src, dst, kind) -> None` — Mutator; both endpoints must already exist. (L191)
  - `ingest_detected_objects(objects, *, now_ns) -> list[str]` — Fold detections into object nodes, associating by `track_id` else label+proximity (`assoc_distance_m`); updates pose/last_seen/confidence (keeping the higher), embeds the label if set. Returns touched node ids. (L199)
  - `recall_object(query: RecallObjectQuery, *, now_ns) -> RecallObjectResult` — Rank object nodes by `max(label-match, CLIP-cosine)` × confidence, tiebreaking on proximity/recency; each match carries an `ApproachViewpoint` + `inside_container_id`. Empty result means unknown (caller may raise `ROSObjectNotInMemory`). (L290)
  - `resolve_place(query: ResolvePlaceQuery, *, from_node_id=None) -> ResolvePlaceResult` — Resolve a place/room/agent reference to a goal pose (an object resolves to its `at_place`) + a `traversable_to` BFS path. Raises `ROSObjectNotInMemory` when unresolved. (L341)
  - `to_scene_graph() -> SceneGraph` — Snapshot the current graph. (L439)
  - `from_scene_graph(graph, *, embedder=None) -> SpatialMemory` (classmethod) — Rebuild from a snapshot; re-embeds labels when an embedder is given (embeddings aren't serialized). (L444)
  - `save(path) -> None` — JSON persistence via the `SceneGraph` contract. (L461)
  - `load(path, *, embedder=None) -> SpatialMemory` (classmethod) — Inverse of `save`; re-embeds labels when an embedder is given. (L466)
- const `DEFAULT_ASSOC_DISTANCE_M` (L74) — Default `assoc_distance_m` (see field docstring in the source for calibration).
- const `DEFAULT_STANDOFF_M` (L77) — Default `default_standoff_m`.
- const `DEFAULT_CAMERA_FRAME` (L80) — Default `camera_frame_id`.
- const `DEFAULT_MAP_FRAME` (L83) — Default `map_frame`.
- const `DEFAULT_MIN_TEXT_SIMILARITY` (L86) — Default `min_text_similarity`; min CLIP cosine for an embedding-only match (calibrated for ViT-B/32 openai) — an exact/substring label match always qualifies regardless.
- const `_MIN_DIR_NORM = 1e-6` (L93) — Below this horizontal distance, `compute_approach_viewpoint`'s approach direction is treated as degenerate.

### `python/world_state/src/openral_world_state/geometry.py`
_Shared rotation geometry, relocated to `openral_core.geometry`; this module is a re-export shim. Canonical entries (`ViewAxis`, `look_at_quat_wxyz`, `compute_gaze_pose`, `rotation_to_quat_wxyz`, `yaw_to_quat_xyzw`, `yaw_to_quat_wxyz`, `quat_xyzw_to_yaw`) are documented under [00-core-schemas.md](00-core-schemas.md)._

### `python/world_state/src/openral_world_state/grid.py`
_Occupancy-grid queries + approach-pose refinement (planning-layer proposal; the kernel `check_nav_goal` gate stays the enforcement)._

- const `_GRID_NDIM = 2` (L33) — Required `data.ndim` for `OccupancyGridIndex.__init__`'s validation guard.
- `FREE_MAX` (module constant, `25`) — Highest `nav_msgs/OccupancyGrid` value still treated as free; `-1` unknown and anything above block placement and sight (conservative). (L34)
- `class OccupancyGridIndex` — Queryable view over one grid snapshot; ROS-free (`from_msg` duck-types the message). `__init__(data (h,w) int8, *, resolution_m, origin_xy, origin_yaw=0.0)`; handles rotated origins. (L39)
  - `from_msg(msg) -> OccupancyGridIndex` (classmethod) — decode a (duck-typed) `nav_msgs/OccupancyGrid`. (L79)
  - `resolution_m` (property) — Cell edge length in metres. (L98)
  - `world_to_cell(x, y) -> tuple[row, col] | None` — `None` off-grid. (L102)
  - `is_free(x, y, *, inflation_m=0.0) -> bool` — point + world-space inflation disc all free; off-grid (or disc off-grid) counts blocked. (L114)
  - `line_of_sight(a_xy, b_xy) -> bool` — Bresenham; every cell strictly before the endpoint free (the target's own footprint cell is exempt — a mug shares its cell with its counter). (L148)
- `refine_approach_pose(grid, viewpoint, target_xyz, *, inflation_m=0.25, max_radius_m=2.0, min_standoff_m=None, max_standoff_m=None) -> ApproachViewpoint | None` — Return the viewpoint unchanged when already free + sighted; else ring-search outward for the nearest admissible point (standoff within `[0.5x, 2.0x]` ideal by default), re-aim via `compute_approach_viewpoint`. `None` = no reachable viewpoint (caller reports honestly, never fabricates). (L183)

### `python/world_state/src/openral_world_state/scene_objects_span.py`
_Publish the durable object nodes as a dashboard OTel span (advisory; never a safety input)._

- const `_DEFAULT_FRAME = "map"` (L27) — Default frame name used by callers building `SceneGraph` nodes for this emitter (the SLAM 2D map frame the dashboard overlays markers on).
- `scene_objects_payload(graph) -> list[dict[str, object]]` — Project a `SceneGraph`'s `object`-kind nodes to JSON-friendly dicts; non-object nodes are skipped. (L30)
- `emit_scene_objects_span(graph, *, source_node) -> int` — Emit one `world.scene_objects` span carrying the object list; returns the object count. Currently emitted by the Reasoner from its preloaded map. (L66)

### `python/world_state/src/openral_world_state/embedder.py`
_Open-vocabulary text embedder (optional; `uv sync --group clip`)._

- const `DEFAULT_CLIP_MODEL = "ViT-B-32-quickgelu"` (L33)
- const `DEFAULT_CLIP_PRETRAINED = "openai"` (L37)
- `class TextEmbedder(Protocol)` — Text-embedding interface open-vocab matching consumes. (L42)
  - `dim` (property) — Embedding width. (L46)
  - `embed_text(texts) -> NDArray[float32]` — L2-normalized embeddings, one row per input text. (L50)
- `class OpenClipEmbedder` — OpenCLIP `ViT-B-32-quickgelu` / `openai` weights (MIT). `__init__(*, model_name=DEFAULT_CLIP_MODEL, pretrained=DEFAULT_CLIP_PRETRAINED, device=None)` raises `ROSConfigError` if open-clip-torch/torch missing or weights unfetchable; lazy-imports torch/open_clip so the base install stays light. (L55)
  - `dim` (property) — 512. (L102)
  - `embed_text(texts) -> NDArray[float32]` — L2-normalized CLIP text embeddings. (L106)

### `python/world_state/src/openral_world_state/object_lift.py`
_2D→3D object-center lift helpers — pure, ROS-free, unit-testable._

- const `_DEPTH_EPS = 1e-6` (L35) — Minimum +z (metres) for a voxel/point to count as "in front of" the camera.
- const `_QUAT_NORM_TOL = 1e-6` (L39) — Tolerance on `|q|^2` for a grid orientation quaternion; wide enough for one that has been through a wire round-trip, far too tight for the all-zero value an unset field carries.
- const `_DEPTH_EPS = 1e-6` (L35) — Minimum +z (metres) for a voxel/point to count as "in front of" the camera.
- const `_QUAT_NORM_TOL = 1e-6` (L39) — Tolerance on `|q|^2` for a grid orientation quaternion; wide enough for one that has been through a wire round-trip, far too tight for the all-zero value an unset field carries.
- `class ObjectsLiftError(ValueError)` — Raised by geometry helpers on degenerate inputs (zero-norm quaternion; occupancy buffer size mismatch). (L42)
- `homogeneous_from_quat_xyz(translation, quat_xyzw) -> NDArray[np.float64]` — Build a 4×4 homogeneous transform from translation + xyzw quaternion; delegates to `openral_core.geometry.homogeneous_from_quat_xyz` (layer 0 needs the same step) and re-raises its `ROSConfigError` as `ObjectsLiftError`. (L46)
- `decode_occupied_centers(*, origin, resolution, size_xyz, occupancy, orientation_xyzw=(0,0,0,1)) -> NDArray[np.float64]` — Occupied voxel centres `(N, 3)` in the grid's reference frame, honoring `orientation_xyzw` since `OccupancyVoxels` is an oriented lattice — dropping the rotation misplaces objects whenever the base isn't map-aligned. Raises `ObjectsLiftError` on a size mismatch or non-unit orientation quaternion (an unset field's all-zero value is never read as identity). (L76)
- `depth_cloud_to_centers_base(points_cloud, t_base_from_cloud, *, max_points=0) -> NDArray[np.float64]` — Octomap-free fallback depth source for `VoxelFrustumLifter`: drops non-finite returns, subsamples to `max_points`, and maps the cloud from its optical frame to base frame. Output is interchangeable with `decode_occupied_centers`. (L162)
- `aabb_iou_3d(a, b) -> float` — 3D axis-aligned bbox IoU in `[0, 1]`, boxes as `(xmin, ymin, zmin, xmax, ymax, zmax)`; 0.0 for disjoint/degenerate. The only IoU helper in the repo — reuse it, don't add another. (L198)
- `class VoxelFrustumLifter(*, k_nearest=25, min_voxels=3)` — Lift 2D detections to 3D object centres via voxel-frustum K-nearest: projects occupied voxels into the camera image plane and takes the K nearest each box centre, returning one `DetectedObject` (`pose.frame_id="map"`) per detection. Skips a detection with fewer than `min_voxels` surviving voxels. (L224)
  - `lift(*, detections, occupied_centers_base, intrinsics, frame_size, t_cam_from_base, t_map_from_base, map_frame="map") -> list[DetectedObject]` — Core lift call (vectorised over voxels). Returns `[]` when `occupied_centers_base` is empty. `track_id` carries the 2D detection's `det_id` into 3D so a physical object keeps one id across the 2D and 3D views; `ObjectMemory` adopts it for new tracks. (L243)
- `build_in_fov_predicate(*, intrinsics, t_cam_from_map) -> Callable[[DetectedObject], bool]` — Build a predicate that's `True` iff the object's `map`-frame centre is in front of the camera and within the image bounds, else `False`. Used by `ObjectMemory.tick` for FOV-guarded eviction. (L338)

### `python/world_state/src/openral_world_state/object_memory.py`
_IoU-gated spatial object memory — pure, ROS-free, unit-testable._

- `class ObjectMemory(*, iou_threshold=0.3, max_misses=1)` — IoU-gated spatial memory with freeze-on-match, in-FOV-guarded eviction, and out-of-FOV retention. Maintains a monotonic `track_id` counter. (L30)
  - `tick(candidates: list[DetectedObject], *, stamp_ns: int, in_fov: Callable[[DetectedObject], bool]) -> list[DetectedObject]` — Run one association+eviction step: greedy highest-confidence matching by same-label + `aabb_iou_3d ≥ iou_threshold` freezes a track (unchanged pose/bbox, bumped confidence); no match mints a new track. An unmatched track in FOV accrues `miss_count` and is evicted at `max_misses`; out of FOV it's retained unchanged. (L64)

### `python/world_state/src/openral_world_state/object_tracker/__init__.py`
_Object-centric tracker core (issue #418) — ROS-free; a lifecycle node wraps it. The module docstring is the **binding/release contract**: those modules mutate tracks only through `ObjectTracker.store` (`TrackStore.get` / `all` / `replace`); the tracker never fits, ages or confirms a BOUND track and hands BOUND fits back in `TrackerOutput.bound_fits`._

### `python/world_state/src/openral_world_state/object_tracker/fit.py`
_Containing fit (#418 step 3, the #266 rule): `C = cone(dilate(S)) ∩ {z ≥ support} ∩ {z_opt ≥ z_front − ε} ∪ prism(P, support)`, never a hull of sampled surface points._

- const `ROBOT_PIXEL = 255` — `robot_filter` mask value for robot pixels (0 is free). (L64)
- const `UNCERTAIN_PIXEL = 128` — `robot_filter` mask value for the uncertain band around the robot. (L65)
- `class FitParams(BaseModel)` — Frozen calibration: `k_px` (mask + extrinsic error), `w_occ_px` (occluder band), `eps_m` (A2/A3 tolerance), `occluder_tol_m`, `min_points`, `min_depth_valid`, depth range. Every value a calibration point. (L71)
- `@dataclass ContainingFit` — `provisional` (the robot touches the mask and reaches past the occluder band: may under-contain; replaces the old `a4_robot_occluder_wide` refusal); `box` (gravity-aligned, min-area yaw, +½ cell sides/top, bottom on the support), `tight` (one box primitive whose stage-2 hull is the 64-vertex polytope), `surface` (`P`, base frame), `support_z`, `assumptions` (A1-A4 evidence: `a1_border_clear`, `a2_gap_m`, `a3_depth_valid`, `a4_occluder_px`), `refusal`. (L102)
- `containing_fit(mask, depth_m, robot_mask, intrinsics, t_base_from_cam, *, support_z, resolution, frame_id, stamp_ns, params=None, detector_box=None) -> ContainingFit` — `detector_box` (the open-vocabulary detector box the mask was prompted with; never a track's own projection) joins the silhouette and its non-robot depths the front cut, so a mask that lost a rim the detector saw is still contained (A1); Robot pixels leave the mask, the uncertain band and an occluder band (where a nearer surface or the robot cuts the boundary) re-enter, `k_px` dilation; the surface `P` and the front plane come from the mask pixels the filter is sure are not the robot (a robot or uncertain-band pixel never lifts the prism or the front cut: logged `a5_robot_band_px`); the cone over the silhouette's 16-direction outer polygon between the front plane and the support, union the prism of `P`'s footprint: exactly 64 vertices, no hull solver, no inward step. Marks `provisional` (the robot touches the mask and reaches past the occluder band; a provisional refit never replaces a clear track's geometry and a clear refit clears the flag), refuses `too_few_points`, `a1_border_truncated` (mask touches the image border), `a2_below_support` (a surface point under the support), `a3_depth_invalid`, `ray_misses_support`, and the grasp-region caps. A4 (a hidden part wider than the occluder band) is logged, not provable here — the tracker's `θ_vis` gate covers it. (L184)
- `project_box(box, intrinsics, t_base_from_cam) -> tuple[tuple[int, int, int, int], float] | None` — A box's clipped image rectangle and nearest optical depth; `None` behind the camera or outside the image. (L387)
- `visible_fraction(box, depth_m, robot_mask, intrinsics, t_base_from_cam, *, tol_m=0.01) -> float | None` — Z-buffered fraction of the box's image rectangle not covered by the robot (robot or uncertain), a nearer surface, or missing depth; `None` out of view. Ceiling: an occluder between the box's nearest corner and the object's surface counts as visible. (L410)

### `python/world_state/src/openral_world_state/object_tracker/multi_view.py`
_Multi-view containing fit (#418 phase 4): `K = conv(C_primary) ∩ cone_j ∩ …` — only second views' silhouette cones intersect, never their front cut or prism; pure numpy, no scipy._

- `@dataclass MultiViewFit` — `fit` (the multi-view `ContainingFit`, or the primary single-view fit on any fallback), `fit_kind` (`MULTI_VIEW_CONE` when a cone cut the result, else `SILHOUETTE_CONE`), `refusals` (per view index: `primary` / `fallback:<why>`, `contributed`, or why the view did not cut: its own fit refusal such as `a1_border_truncated`, `cone_layout`, `silhouette_too_small`). (L56)
- `intersect_views(views, *, resolution, frame_id, stamp_ns) -> MultiViewFit` — `views` are `(containing_fit result, T_base_from_optical)` per depth camera. A second view cuts only when its own fit was not refused (A1 border clear) and its cone holds every measured surface point of every view (a too-small silhouette is refused whole). `K` is exact (plane-triple vertex enumeration), reduced outwards to the 64-vertex hull cap (two horizontal slabs' 16-direction outer polygons when `K` has more), boxed by `fit._boxed`; falls back (logged `object_tracker.multi_view_fallback`) on `no_second_view`, `empty_intersection`, `views_disagree` (a measured point outside the result) or `no_gain` (box not smaller). Not wired into `ObjectTracker.feed` yet: the tracker fits per depth frame, one camera at a time. (L157)

### `python/world_state/src/openral_world_state/object_tracker/cover.py`
_Size-aware map confirmation (#418 step 4)._

- `@dataclass CoverVerdict` — `visible` (`|E|`), `occupied` (`|E ∩ occupied|`), `kind` (`""` / `map_disagrees` / `map_split` / `map_split_own`), `why`. (L34)
- `surface_cells(grid, surface, *, exclude=()) -> frozenset[tuple[int, int, int]]` — `E`: lattice cells holding the fit's surface points, minus cells centred in `exclude` (the robot's own space). (L56)
- `cover_verdict(grid, cells, box, *, support_z, own=(), others=()) -> CoverVerdict` — `others` (the other CONFIRMED/OCCLUDED tracks' boxes): a second body wholly inside one is that object, not a split (design step 4, closure minus other tracks); `|E ∩ occupied| ≥ max(2, ⌈0.5·|E|⌉)` (cover on expected cells, not a padded footprint), then `object_primitives.map_confirms` for one body; a second body wholly inside a declared hand's `own` space is `map_split_own`. (L89)

### `python/world_state/src/openral_world_state/object_tracker/tracks.py`
_Track table, association by continuity, source-time ageing (#418 steps 5-6)._

- const `LIVE` — Statuses a fit refreshes by continuity (TENTATIVE, CONFIRMED, OCCLUDED, SPLIT, BOUND). (L40)
- `class TrackStore(frame_id)` — The `TrackedObject` rows; every change bumps `revision`. Contract for binding/release: `get`, `all`, `replace`. (L52)
  - `revision` (property) — Strictly increasing. (L96)
  - `get(object_id) -> TrackedObject | None` — (L100)
  - `all() -> list[TrackedObject]` — Mint order. (L104)
  - `replace(obj) -> None` — Swap the row for an updated copy; bumps `revision`; `ROSConfigError` on an unknown id, a frame mismatch or an epoch going backwards; logs status transitions. Never mints. (L108)
  - `mint(**fields) -> TrackedObject` — TENTATIVE `trk:<n>`, never reused; past 32 rows the oldest LOST is dropped, else `ROSConfigError`. (L137)
  - `drop(object_id) -> None` — Forget a row. (L160)
  - `snapshot(stamp_ns) -> TrackedObjectSet` — (L165)
- `associate(tracks, centroid, *, voxel_m) -> tuple[str | None, str]` — `continuity` (a live box holds the centroid or its centre is within a voxel), `tie` (two candidates within one voxel of each other: update nothing, HZ-0115-2), `inferred` (exactly one LOST track within a diameter, or RELEASED track within its fall column), else `new`. (L180) A SPLIT track takes no fit by continuity (its box holds two bodies: a neighbour inside it is a new object). The tracker ignores a refit of a CONFIRMED/OCCLUDED track whose new box splits on the map (`split_refit_ignored`).
- `age_tracks(store, now_ns, *, max_age_s, lost_s, keep_lost_s) -> None` — Source-clock ageing: CONFIRMED/OCCLUDED without a confirming grid → TENTATIVE (ineligible); TENTATIVE/SPLIT without evidence → LOST; LOST forgotten after `keep_lost_s`. BOUND/RELEASED are not aged here. (L259)

### `python/world_state/src/openral_world_state/object_tracker/grounding.py`
_Standing queries (#418 Addendum G): a query lives as long as its goal; never a first-frame snapshot._

- `@dataclass StandingQuery` — `(goal_id, query_id)`, the `ObjectQuery`, `declaration_key`, `ObjectQueryStatus`, search start (`since_ns`), fairness key (`last_served_ns`), `escalated`. (L46)
  - `key` (property) — `(goal_id, query_id)`. (L70)
  - `matches(track) -> bool` — Same label and role. (L74)
- `@dataclass QueryEscalation` — Active-search data for the reasoner, never executed by the tracker: `kind` `unresolved` (start `find-object`) / `exhausted` (budget spent → replanning ladder), `last_seen` (last LOST match's box, a search prior). (L80)
- `class QueryBook(*, search_start_s, search_max_s, continuous_s=2.0)` — Every live goal's queries. (L100)
  - `queries` (property) — (L129)
  - `get(goal_id, query_id) -> StandingQuery | None` — (L133)
  - `open(goal_id, query_id, query, declaration_key, *, stamp_ns) -> None` — Start a standing query at dispatch. (L137)
  - `close_goal(goal_id) -> None` — End a goal's queries. (L152)
  - `ready(goal_id) -> bool` — `grounding_ready`: every query RESOLVED. (L157)
  - `update(tracks, *, now_ns) -> list[QueryEscalation]` — UNRESOLVED → RESOLVING (TENTATIVE/RELEASED match) → RESOLVED (≥ `min_instances` CONFIRMED/OCCLUDED/BOUND); all lost again → UNRESOLVED with the timer restarted; one `unresolved` escalation after `search_start_s`, `EXHAUSTED` + `exhausted` at `search_max_s`, all in source time. (L162)
  - `due(*, now_ns) -> list[StandingQuery]` — Passive-search order: unresolved queries least-recently-served first, then resolved `count: "all"` queries (they keep detecting more instances) and every other resolved query whose last locate is `continuous_s` of source time old (#418, user 2026-10-17: detection runs for the whole goal, so an object that only becomes visible later, e.g. one the arm hid at goal start, is still picked up; `TrackerParams.continuous_locate_s`).

### `python/world_state/src/openral_world_state/object_tracker/scheduler.py`
_One segmenter/detector request in flight, rate-capped in wall time (#418 step 7)._

- const `PRIORITY_QUERY = 1` — Step-7 priority of unresolved queries. (L39)
- const `PRIORITY_RELEASED = 2` — RELEASED tracks, searched in their fall column. (L40)
- const `PRIORITY_DISAGREE = 3` — Tracks the map disagrees with. (L41)
- const `PRIORITY_BOUND = 4` — Bound-object protrusion checks. (L42)
- const `PRIORITY_REFRESH = 5` — Oldest visible CONFIRMED refresh and resolved `count: "all"` re-detection. (L43)
- `@dataclass TrackerRequest` — One `locate_in_view` / `segment_in_view` prompt on the frame `(camera, stamp_ns)`: `priority`, fairness `served_ns`, `label`, `query_key` or `object_id`, `box_xyxy` / `point_uv` (prompts), `within_xyxy` (a bound on the mask pick only: a map seed's image rectangle), `seed_cell` (the lattice cell a map-seeded prompt sits on: a served one is skipped until a locate is served), `request_id` (0 until issued). (L47) `roi_norm_xyxy` (issue #418): a locate's region of interest in `[0, 1]` of the frame, sent as `LocateInView.roi_norm_xyxy`. `prompt`: a locate's open-vocabulary text, `label` then the query's `synonyms`, comma-separated (the node sends it as `LocateInView.query`).
- `class Scheduler(*, segment_hz, budget_wall_s)` — Lowest priority number, then served longest ago; abandons (and logs) a request unanswered past `budget_wall_s`. (L93)
  - `in_flight` (property) — (L123)
  - `complete(request_id) -> TrackerRequest | None` — Free the slot; `None` (logged) for an unknown id. (L127)
  - `next(candidates, *, wall_s) -> TrackerRequest | None` — Issue the best candidate when the slot is free and the rate allows. (L136)

### `python/world_state/src/openral_world_state/object_tracker/tracker.py`
_`ObjectTracker` — steps 0-7 per depth frame; pure (no ROS, no clock read)._

- `VIEW_WINDOW_S = 0.5` (L146) — Source seconds within which one track's fits from different depth cameras are combined: a refit from camera B is passed with the other cameras' newest clear fits no older than this to `intersect_views` (this fit primary), and the track takes its `fit_kind` (`MULTI_VIEW_CONE` when a second view contributed, else `SILHOUETTE_CONE`); provisional fits neither contribute nor are intersected (`ObjectTracker._combine_views`).
- `ROBOT_BOX_FRACTION = 0.5` (L151) — A detector box more than this fraction ROBOT pixels is re-prompted (below).
- `ROBOT_BOX_NEGATIVES = 3` (L153) — Issue #418 (PickOrange orange_3 under the open jaw: the locator boxed the arm as "orange" and SAM segmented the arm): a detector box more than half ROBOT pixels is re-prompted (`ObjectTracker._robot_box_reprompt`) with a positive point on the free pixels in the box that stand above the support -- the connected region lying most beside the robot (`_most_hidden`), at its pixel nearest its centroid -- and negative points on the robot (`TrackerRequest.negative_uv`, sent as `SegmentInView.negative_points`), the box kept as the mask's bound. Logged `object_tracker.robot_box_reprompt`. The fit then goes through the robot (PROVISIONAL) and needs only `TrackerParams.theta_vis_provisional` (0.25) of the object visible, not `theta_vis` (0.5). A served re-prompt clears its pending box; a resolved count-all query that keeps detecting logs `object_tracker.count_all_research` (at most every 5 s).
- `LOCATE_ROI_PAD = 0.25` (L149) — A locate's crop: the projected search region grown by this fraction of its size on every side. The region is the query's seed, else `ObjectTracker.search_region` (the active grasp declaration's `workspace_box`, set by `ObjectTrackerSession.on_grasp_declaration`); none = the whole frame (`ObjectTracker._locate_roi`). #418 restock: the whole 1920×1080 head frame squashed to omdet's 640×640 left a carton ~40 px and `found` in ~24 % of locates.
- `REFUSAL_LOG_PERIOD_S = 5.0` (L143) — Source seconds between INFO reports of one `(key, reason)` refusal.
- `ObjectTracker.refusal_counts: Counter[tuple[str, str]]` — Every refusal by `(key, reason)`: `<label>@<camera>` for a fit (`too_few_points`, `no_support`, the fit's own refusal, `occluded`, `candidate_split`, `locate_found_nothing`, `no_mask`, `no_grid`), `camera:<name>` for a frame (`no_camera_pose`, `no_robot_mask`). Each is reported as an INFO `object_tracker.refused` event (key, why, count, the numbers) at most every `REFUSAL_LOG_PERIOD_S` per key and reason (`_refuse`), never DEBUG-only (#418 restock grounded nothing with no reason at INFO); the node publishes the counts on `/diagnostics` (`openral_object_tracker: refusals`, key `<key>|<why>`) at most every 5 s.
- `class TrackerParams(BaseModel)` — Frozen calibration: `theta_vis`, `max_age_s`, `lost_s`, `keep_lost_s`, `disagree_grids`, `segment_hz`, `segment_budget_wall_s`, `search_start_s`, `search_max_s`, support probe, `fit: FitParams`. (L73)
- `@dataclass CameraFrame` — Depth at its source stamp with intrinsics, `t_base_from_cam` and `robot_mask` (either `None` = ageing only). (L119)
- `@dataclass TrackerResult` — Answer to an issued request: `mask` (segmenter) or `boxes_xyxy` (detector); `alternatives` (smaller qualifying candidates, largest first: `feed` tries the next when a candidate cannot be fitted or its box splits on the map, and mints a SPLIT track only from the last). (L159)
- `@dataclass TrackerOutput` — `requests` (≤ 1), `tracked` (`TrackedObjectSet`), `escalations`, `bound_fits` (fits associated with BOUND tracks, for the in-hand update). (L178)
- `class ObjectTracker(frame_id, params=None)` — `store: TrackStore`, `queries: QueryBook`. (L194)
  - `open_query(goal_id, query_id, query, declaration_key, *, stamp_ns) -> None` — Start a standing query. (L241)
  - `close_goal(goal_id) -> None` — End a goal's queries; tracks persist. (L253)
  - `feed(frame, *, grid, grid_stamp_ns, wall_s, results=(), hand_spaces=()) -> TrackerOutput` — Results (answers on this same frame) → support (ring counted only below the mask's lowest point, truncated one cell under the measured top) → containing fit → `θ_vis` gate → association → mint/refit → immediate cover check; a new grid re-confirms every track (map_split → SPLIT, own-hand split → unchanged and not re-stamped, `disagree_grids` refutations → LOST, RELEASED → CONFIRMED epoch+1 after a fresh fit); visibility (CONFIRMED ↔ OCCLUDED, OCCLUDED never re-fits); source-time ageing; query update and escalations; at most one new request. (L258)

### `python/world_state/src/openral_world_state/object_tracker/pinch.py`
_Issue #418 WP C2. The hand's pinch volume `J_h` by mimic-aware joint-state FK (`openral_core.kinematics`), pure numpy over the manifest, no HAL import._

- `class PinchVolume(centre, axes, half, palm_m)` (L45) — An oriented box in the attach link, axes `(c, a, b)` = closing, approach, pad width; `palm_m` the palm plane `x·a ≥ palm_m` it never crosses.
  - `box() -> Box` (L60) — `(centre, rotation, half)` for `box_gap_lower_bound_m`.
  - `grown(margin_m) -> PinchVolume` (L64) — Every face out by `margin_m` (`J_h` takes one cell).
- `class HandModel(description, gripper_joint)` (L69) — One `role: gripper` joint: attach link = its parent link, `touch_links` = `grasp_contact_links([joint])`, pads = touch links no other finger hangs off (the Robotiq's distal/proximal phalanges, the SO-101's moving jaw), closed end = `closure_calibration.closed_position`. Axes fixed per hand at contact: `c` the principal direction of the pads' closing motion, `a` palm → pads square to `c`. The palm plane is the jaw joint's origin, moved forward to the wrist link's front face over the carrying joint's range where the wrist reaches past it (SO-101: 17 mm; g20's cube hit it). `ROSConfigError` on a non-gripper joint, no calibration, or padless fingers.
  - `pinch_in_link(jaw) -> PinchVolume` (L184) — `J_h` (unpadded): the box in `(c, a, b)` around the pads swept from the measured jaw to closed (16 samples), clipped at the palm plane.
  - `open_end` (property) (L199) — The end of the jaw's `position_limits` farther from the closed end: the manifest's full stroke.
  - `fallback_in_link(jaw, opening=None) -> PinchVolume` (L204) — The `PINCH_FALLBACK` payload, OT-10 bounded (#418 r4: the old pad-width box under-covered every held orange, banana and carton by 4–16 cm): along `c` the pad gap at `opening`, the jaw measured just before closure (the whole stroke when none was seen); along `a` and `b` the full stroke (the pad gap at `open_end`) from `J_h`'s centre at `jaw`; back face on the palm plane.
  - `hand_boxes(frame, positions) -> list[Box]` (L227) — The fingers', palm link's and wrist's collision primitives as boxes in `frame` (never evidence of an object).
  - `t_frame_from_attach(frame, positions) -> NDArray` (L235) — `T_frame<-attach_link` by FK.

### `python/world_state/src/openral_world_state/object_tracker/binding.py`
_Issue #418 WP C2, design "Grasp binding and in-hand update". ATTACH binds what meets `J_h`; the bound geometry then only grows (the kernel's K4 union). Logs `object_tracker.bound` / `untracked_grasp` / `ambiguous_grasp` / `bound_geometry_grew` / `bound_geometry_cap` / `protrusion_unexplained`._

- `MAX_PRIMITIVES = 16` (L67) — The kernel's per-object primitive cap (K4).
- `TRACKED_CONFIDENCE = 0.7` (L69) — A tracked binding's confidence (`region_attachment`'s).
- `FALLBACK_CONFIDENCE = 0.2` (L71) — A pinch fallback's (the HAL jaw-span fallback's).
- `class TrackStoreLike(Protocol)` (L86) — The C1 `TrackStore` interface.
  - `get(object_id) -> TrackedObject | None` (L89) — One track.
  - `all() -> list[TrackedObject]` (L92) — Every track.
  - `replace(obj) -> None` (L95) — Store under its id; bumps the set revision.
- `class BindOutcome(StrEnum)` (L99) — `TRACKED`, `AMBIGUOUS`, `PROVISIONAL`, `UNTRACKED`.
- `class Binding(outcome, attachment, candidates, reground)` (L109) — One hand's held payload as published on `/openral/attachment_state`; `reground` asks for a re-ground after lift.
- `class GraspBinder(hand, *, frame_id, cell_m)` (L247) — One hand. `frame_id` must be the manifest's `base_frame` (FK root), else `ROSConfigError`.
  - `pinch_in_frame(positions) -> Box` (L283) — `J_h` grown by one cell, in the grid frame.
  - `object_pose(positions) -> Pose6D` (L289) — The held object frame in the grid frame, `FK(q) · pose_in_link` (the release pose).
  - `observe_jaw(jaw) -> None` (L274) — Keep the most open jaw since the last ATTACH (fed every joint state by the session); it bounds the next fallback's width along `c`.
  - `on_attach(store, positions, stamp_ns) -> Binding` (L306) — Provisional candidates never bind an identity: with clear candidates they are ignored; with only provisional ones the payload is `PROVISIONAL`, `PINCH_FALLBACK` of the fallback box ∪ their boxes cut at the palm plane (`provisional_grasp`, `reground=True`): measured but maybe short, so unioned it only grows the payload. Candidates: free tracks (TENTATIVE/CONFIRMED/OCCLUDED/SPLIT) of any role whose measured geometry holds `J_h`'s centre within one cell (their tight k-DOP when they carry one, else the box; a padded box merely meeting the swept pinch volume bound a tracked neighbour of an ungrounded held orange, p2po_on2), and RELEASED tracks whose `fall_column` meets it. One: `TRACKED_OBJECT` under the track id (`pose_in_link = T_link⁻¹·T_obj`, the shape — `tight` hulls or the box — unioned with itself slid along `c` onto the pinch axis; a RELEASED one bound re-centred on `J_h` under `epoch + 1`), the track set BOUND. Two or more: `PINCH_FALLBACK` of the fallback box ∪ the candidates' boxes (cut at the palm plane) (one enclosing box past 15), object id `pinch:<joint>@<stamp>`, no track touched. None: the fallback box alone (`HandModel.fallback_in_link` at the jaw `observe_jaw` saw most open), `reground=True`. A tracked binding's primitives are cut at the palm plane (`_palm_clipped`: nothing held lies behind the palm).
  - `on_regrasp(positions, stamp_ns) -> Binding | None` (L508) — Unions the shape slid onto the new pinch axis, `geometry_revision + 1`; `None` when already on it or past the cap (union kept).
  - `grow_by_protrusion(store, grid, positions, stamp_ns) -> Binding | None` (L521) — Tracked bindings only: occupied cells within the payload's reach of the pinch axis that neither the bound geometry, another non-bound track nor the hand explain (½ cell), 26-connected to the payload; fits the rotation θ about the pinch axis (2° grid, residual < 1 cell, nearest the previous θ) and unions the arc in steps of ≤ one cell at the payload's reach. `None` when nothing protrudes, no θ fits, or the cap would be passed.
  - `grow_untracked(store, grid, positions, stamp_ns) -> Binding | None` (L610) — UNTRACKED / PROVISIONAL payloads only (#418 r4: their revision stayed 0 while 2–10 cm of the held object stuck out): occupied cells within `_GROW_RADIUS_M` (0.3 m) of the pinch centre that neither the hand nor another track explains, 26-connected to the payload and in front of the palm, keyed in the attach link; a key seen again after the hand moved ≥ `_RIGID_MIN_MOVE_CELLS` (2) cells whose first grid-frame place is free on this grid moved with the hand. The new cells' box (+½ cell, object frame) is unioned (`moves_with_hand`). Called by `ObjectTrackerSession.on_grid` on every newer grid.
  - `on_release() -> None` (L710) — Forget the held payload.

### `python/world_state/src/openral_world_state/object_tracker/release.py`
_Issue #418 WP C2, design "Release". Source-clock stamps only._

- `class JawReleaseMonitor(hold, closed, release_gap, release_dwell_ns)` (L39) — The jaw-open trigger (the kernel's K5, mirrored): further from closed than `hold + release_gap` continuously for the dwell; a source clock going back restarts it.
  - `for_hand(hand, *, hold, release_dwell_s) -> JawReleaseMonitor` (L62) — `release_gap` = half the manifest's `stall_gap` (its release hysteresis).
  - `observe(jaw, stamp_ns) -> bool` (L68) — Fold one sample in; `True` once open for the dwell.
- `release_track(store, object_id, *, object_pose, support_z, cell_m, stamp_ns) -> TrackedObject | None` (L98) — A BOUND track → RELEASED at the release pose (yaw-only box containing the tilted body, hulls dropped, binding fields cleared, support/floor from the support measured under the release pose); a box breaking the caps goes LOST (logged).
- `fall_column(track, cell_m) -> PlaceRegion` (L150) — The release box down to its support, horizontal gate one cell + the release height (45° cone; no bounce model).
- `reassociate_released(store, fit, *, fit_kind, support_z, support_ref, visible_cells, occupied_cells, cell_m, stamp_ns) -> TrackedObject | None` (L203) — A map-confirmed fit whose centre lies in exactly one RELEASED track's fall column returns it CONFIRMED, same id, `epoch + 1`, `identity_inferred`, on the new support; two columns is ambiguous (logged), none a new object.

### `python/world_state/src/openral_world_state/object_tracker/session.py`
_Issue #418 WP D1. `TrackerSession` — the ROS-free glue the object-tracker lifecycle node drives: goals from the declarations, the grounding wait, binding/release at grasp events and the jaw-open rule. All stamps source time. A RELEASED track's return (same id, `epoch + 1`) is the tracker core's own association + confirmation; `release.reassociate_released` is not called twice. Logs `object_tracker.goal_closed` / `grounding_ready` / `grounding_incomplete` (error level under `grounding_required`) / `reground_queued` / `jaw_open_release` / `ambiguous_pruned` / `grasp_event_unposed`._

- `class TrackingMode(StrEnum)` (L78) — `shadow`, `bind`, `on` (`DeployRuntime.object_tracking`; `off` never starts the node).
- `class GraspEventKind(StrEnum)` (L86) — `attach`, `regrasp`, `detach` (`openral_msgs/GraspEvent.kind`).
- `declaration_key(target_id, stamp_ns) -> str` (L109) — The kernel's `"<target_id>@<stamp_ns>"`.
- `class TrackerSession(description, *, cell_m, grounding_wait_s, grounding_required=False, release_dwell_s=0.3, released_clear_timeout_s=3.0, params=None, require_hands=False)` (L119) — One `ObjectTracker` in the manifest's `base_frame` and one `GraspBinder` per gripper joint (`require_hands`: a gripper the binder cannot model is a `ROSConfigError`, else logged and skipped).
  - `now_ns` (property) (L196) — The newest source stamp seen.
  - `on_grid(grid, source_stamp_ns) -> None` (L231) — Keep the newest grid in the tracks' frame (another frame: logged, ignored).
  - `on_joint_state(positions, stamp_ns) -> None` (L246) — Manifest-named joints; a hand's `JawReleaseMonitor` firing releases it.
  - `open_standing_queries(queries) -> None` — The deploy's prompt-time queries (`object_queries_json`) as goal `deploy:deploy@0`: they ground whatever declarations are in force (a scene with no place declaration still grounds its plate); a declaration's own goal re-keys the tracks they match. (L222)
  - `on_grasp_declaration(decl) -> None` (L259) — Opens one standing query per `target_queries` entry under goal `grasp:<key>` (key = `declaration_key`), re-keys the live tracks the queries match (label + role) to it; a retraction or a new key closes the old goal.
  - `on_place_declaration(decl) -> None` (L266) — The same for `destination_query` (goal `place:<key>`).
  - `on_grasp_event(kind, gripper_joint, *, jaw, hold, stamp_ns) -> Binding | None` (L272) — A non-finite `jaw` (the HAL had no measured jaw) falls back to the newest joint sample, else the event is logged and ignored; a non-finite `hold` takes the jaw. ATTACH binds at the measured jaw (a second ATTACH releases the first hold), arms the jaw-open monitor at `hold`, re-opens the live grasp goal's queries on an untracked grasp; REGRASP grows; DETACH releases (`release_track` at `FK(q)·pose_in_link`, support = the highest occupied cell top under the body, else the track's old support). `None` when the hand is unknown or the joints are unposed.
  - `on_frame(frame, *, wall_s, results=()) -> TrackerOutput` (L312) — `ObjectTracker.feed` with the newest grid and the hands' spaces; each bound fit triggers `grow_by_protrusion`; an ambiguous union drops every candidate re-fitted after the ATTACH whose box no longer meets the pinch volume (`geometry_revision + 1`); grounding logged once per goal.
  - `snapshot() -> TrackedObjectSet` (L334) — The table at the newest source stamp.
  - `attachments() -> list[AttachedCollisionObject]` — Every hand's held payload, then each released `TRACKED_OBJECT` binding still on the wire: kept (the kernel freezes it, the voxel bridge clears its box) until the track leaves RELEASED with that epoch (re-confirmed `epoch + 1`, or LOST) or `released_clear_timeout_s` passes (source time), whichever first — the update that drops it bumps `attachment_changes`. `released_clear_timeout_s` must equal the kernel's `tracked_release_timeout_s` and the voxel bridge's `released_clear_timeout_s`. (L338)
  - `workspace_region() -> PlaceRegion | None` (L343) — The live grasp declaration's `workspace_box` while the newest grid covers `WORKSPACE_MIN_COVER` of its footprint, re-stamped with the grid's `source_stamp` (the HAL producer's rule); the node relays it as `workspace_region` in bind/on, where the producer stands down.
  - `workspace_region() -> PlaceRegion | None` (L343) — The live grasp declaration's `workspace_box` while the newest grid covers `WORKSPACE_MIN_COVER` of its footprint, re-stamped with the grid's `source_stamp` (the HAL producer's rule); the node relays it as `workspace_region` in bind/on, where the producer stands down.
  - `hand_spaces() -> list[PlaceRegion]` (L361) — The hands' collision boxes at the measured configuration (empty while unposed).

### `packages/openral_octomap_bridge/include/openral_octomap_bridge/self_filter.hpp`
_C++ (Layer 2). The real-camera counterpart of the sim depth renderer's robot transparency: removes the robot's (and a held payload's) own returns from a depth cloud before `octomap_server` inserts it. Poses the kernel's own collision primitives via an FK that mirrors the kernel's operation for operation (a test links the kernel library to pin it; the runtime does not, since Layer 2 may not depend on Layer 5). The node is `src/robot_self_filter.hpp` (`robot_self_filter` executable), wired by `deploy_e2e.launch.py` on the real path only._

- `enum class SelfJointKind` — The kernel's joint codes: `kFixed` / `kRevolute` / `kPrismatic` / `kMimicRevolute` / `kMimicPrismatic` (a mimic kind reads another joint's column and moves by `mimic_multiplier * q + mimic_offset`, the OpenArm's follower finger; `SelfModelParams::mimic_multiplier` / `mimic_offset` are the per-link `collision_mimic_*` arrays, empty = identity).
- `kSelfDopAxes` / `kSelfDopAxis` / `kSelfMaxHullVertices` / `kSelfHullContainmentEpsilonM` / `kSelfGjkMaxIterations` / `kSelfGjkTolerance` — The kernel's `kDopAxes` / `kDopAxis` / `kMaxTightHullVertices` / `kTightContainmentEpsilonM` / `kGjkMaxIterations` / `kGjkTolerance`, pinned equal by `test_self_filter.cpp`.
- `struct SelfHull` — One box's tight geometry in the box frame: 13 `dop_lo` / `dop_hi` slabs plus the exact hull's `vertices` (empty = DOP only).
- `self_hull_distance(hull, local, stop_above, stop_at_or_below = -inf) -> double` — Lower bound on the point-to-hull distance (`<= 0` inside): the 26-DOP slab bound, then GJK on the vertices with an exhaustive support scan; exact to 1e-9 on convergence, never an over-report. Stops early once the answer is settled against either threshold. Pinned against the kernel's `dop_cell_lower_bound` + `hull_cell_distance` (`half_side = 0`) on panda_mobile's hulls. Allocation-free.
- `struct SelfModelParams` — The kernel's `collision_*` parameter arrays, exactly as the launch passes them, including the tight-geometry CSR arrays (`box_hull`, `hull_dop_lo`/`_hi`, `hull_vertex_first`/`_count`, `hull_vertices`).
- `struct SelfModel` — Flattened tree + every link-attached primitive (capsule, sphere as a zero-length capsule, box) in its link frame, `primitive_hull` (parallel, index into `hulls` or -1) and `hulls`; `link_index(name) -> int`.
- `self_transform_from_xyz_rpy(x, y, z, roll, pitch, yaw) -> tf2::Transform` — `Rz·Ry·Rx`, the kernel's `transform_from_xyz_rpy` element for element.
- `build_self_model(params, out, error) -> bool` — All-or-nothing, like the kernel's `load_collision_model`: any shape disagreement, out-of-order parent, unknown joint code, out-of-range primitive link or non-finite/negative dimension refuses the model. Tight geometry is held to the kernel's loader shape checks and `validate_tight_hull` chain (vertices ⊆ DOP ⊆ box, finite, non-inverted, positive extent, <= 320 vertices); where the kernel falls back to the box, the filter refuses the model.
- `self_forward_kinematics(model, q, n_dof, link_world)` — Link frames in the model root; pinned to 1e-12 against `openral_safety_kernel::forward_kinematics` by `test_self_filter.cpp`.
- `place_self_primitives(model, link_world, frame_from_root, out, out_hulls = nullptr)` — Every robot primitive placed in the cloud's frame; `out_hulls` receives each one's `SelfHull*` (or null) in parallel.
- `kMaxSelfFilterPaddingM = 0.10` / `valid_self_filter_padding(padding_m) -> bool` — Cap on the self-filter's `padding_m` (the blind shell around the arm, hazard log Entry 035) and its check: in [0, 0.10], finite. `RobotSelfFilter` logs an ERROR and forwards nothing otherwise. Mirrors `DeployRuntime.robot_self_filter_padding_m <= 0.10` (pinned by `tests/unit/test_perception_caps_mirror.py`).
- `class SelfMask` — Primitives in the cloud frame with inverse poses and broad-phase bounds precomputed once per cloud; `set(primitives, padding_m, hulls = {}, extra_padding_m = {})` (per-entry padding `padding_m + extra_padding_m[i]`, clamped to `kMaxSelfFilterPaddingM`: a fast link is under-padded, a false stop, never more world erased; entries past `extra_padding_m` — payloads — get `padding_m`), `contains(p) -> bool`, `padding(i) -> double`. A box with a hull: box shell first (cheap reject, hull ⊆ box), then `self_hull_distance`.
- `struct JointSample` / `enum class JointStateVerdict` (`kOk`, `kNoState`, `kGap`, `kExtrapolation`) / `joint_state_at(history, stamp_ns, period_s, max_skew_s, q, qdot) -> JointStateVerdict` — Joint positions and rates at a stamp, linearly interpolated between the bracketing samples of a stamp-sorted history; refused when they are > 2 periods apart, when reaching the stamp needs > 1 period of extrapolation (one sample is never extrapolated), or when the nearest sample is past `max_skew_s`. A stamp on a sample is that sample. Issue #418.
- `self_primitive_motion(model, link_a, link_b, out)` — Per robot primitive, `|Δorigin| + bounding radius × rotation angle` between two FK poses: an upper bound on any of its points' displacement. The node's velocity padding (FK at `q` vs `q + qdot·velocity_padding_dt_s`).
- `kRobotMaskFree/Uncertain/Robot` (0/128/255) / `kRobotMaskDepthTolM = 0.005` / `kRobotMaskHandMarginM = 0` / `kRobotMaskArmMarginM = 0.010` / `kRobotMaskSeeThroughM = 0.03` (#418: a return that far behind the rendered front and > 10 mm outside every primitive is seen past the robot's envelope, not ROBOT) / `struct PinholeCamera` / `struct RobotMaskScratch` / `render_robot_mask(primitives, hulls, backstop_margin_m, depth_m, camera, uncertain_px, scratch, mask)` — The robot mask of one depth image from UNPADDED primitives in the optical frame: per primitive, a ray per pixel of its projected bounding box only (capsule/sphere analytically, a box or a hulled box's 26-DOP by slabs; the DOP contains the hull, so the silhouette only grows) into a z-buffer `Z_r`; ROBOT iff `Z_r <= D + 5 mm` (an object in front of a finger stays FREE) or the back-projected return is within its primitive's backstop margin; UNCERTAIN within `uncertain_px` (Chebyshev) of both a ROBOT and a non-ROBOT z-test pixel, or where the robot covers a pixel with no return. Buffers reused across frames. Issue #418.
- `struct SelfFilterStats` — `points_in`, `non_finite`, `removed`, `kept` for one call.
- `filter_xyz_points(data, n_points, point_step, x_offset, y_offset, z_offset, mask, out) -> SelfFilterStats` — Copies every finite point the mask does not contain, whole records, in order.
- `xyz_records(data, point_step, x_offset, y_offset, z_offset) -> std::vector<std::uint8_t>` — The x/y/z of each record as packed 12-byte FLOAT32 records (offsets 0, 4, 8): a coloured cloud's points for octomap, which reads nothing else.
- `class RobotSelfFilter` (src/robot_self_filter.hpp) — The node: joint positions interpolated at the cloud's CAPTURE stamp (`joint_state_at`: `joint_state_period_s`, default 0.05 s, the launch passes `1 / control_freq_hz`; `max_joint_state_skew_s`; manifest names or their `collision_joint_aliases`; only complete samples are kept, sorted by stamp), camera pose from tf2 at that stamp, held payloads from `/openral/world_state_fast`. No pose at the stamp — including a refused gap or extrapolation (`totals().refused_joint_gap` / `refused_extrapolation`, WARN rate-limited) — drops the cloud (the map goes stale and the kernel fails closed); an unplaceable payload stays in the cloud. Robot primitives are padded `padding_m + motion within velocity_padding_dt_s` (δt_sync, default 0.02 s), capped at `kMaxSelfFilterPaddingM`. Optional robot mask (`robot_mask_enabled`, default off; `robot_mask_cameras` + parallel `robot_mask_depth_topics` / `robot_mask_camera_info_topics` / `robot_mask_topics`, `robot_mask_hand_links` get the 0 mm backstop, others 10 mm, `robot_mask_uncertain_px` = 3): per depth image (32FC1 or 16UC1 mm) with a matching `CameraInfo`, the unpadded model posed at the DEPTH stamp is rendered (`render_robot_mask`) and published mono8 with the depth header, BEST_EFFORT KEEP_LAST 5; a frame it cannot pose publishes nothing (`masks_dropped`). Forwards the capture stamp unchanged; logs `self-filter:` every 5 s. `cloud_out` is x/y/z only; an input with more fields (a sim cloud's `rgb`) also goes out whole, robot removed, on `cloud_out_rgb` (`/openral/world_cloud/self_filtered_rgb`) for viewers, only while that topic has a subscriber.

### `packages/openral_octomap_bridge/include/openral_octomap_bridge/payload_clearing.hpp`
_C++ (Layer 2). The attached payload's own occupancy leaves the published `OccupancyVoxels` grid here, partitioned against the safety kernel's support-contact witness (ADR-0092 D6); header-documented and gtested against a real `octomap::OcTree`._

- `struct PayloadPrimitive` — One attached-payload collision primitive placed in the grid frame, mirroring the kernel's `AttachedPrimitive` conventions (a capsule's segment on local +Z, a box's half-extents on its own axes).
- `struct SupportPatch` — One attested support contact placed in the grid frame (point + outward normal + patch radius + max penetration); the wire form is `openral_msgs/SupportContactWitness`, lifted here through the payload's live pose. `has_anchor` / `anchor_point` / `anchor_normal`: the attestation's plane on its first grid (`SupportPatchAnchors`); when set, a cell is withheld only if both planes claim it.
- `pose_from_msg(pose, out) -> bool` — A wire pose as a transform; `false` on a non-finite position or a zero quaternion (never silently the identity).
- `place_attached_object(object, grid_from_link, out, out_patches) -> bool` — Place one wire `AttachedCollisionObject`'s primitives (and support patch, when valid) using the kernel's own pose composition. Fail-closed on any of the kernel's `ingest_attached_objects` validation rules, returning `false` and appending nothing — so a rejected object clears nothing.
- `support_patch_withholds(patch, center, resolution) -> bool` — Does an attested patch withhold this cell from clearing? Mirrors the kernel's `support_contact_exempts` (with zero slack) so the withheld set is always a subset of what the kernel exempts — a withheld cell can never be the one that stops the robot. With an anchor it is the AND of the live and anchored planes (only ever narrower). Deliberate mirror of the kernel predicate; see [14-duplication-watch.md](14-duplication-watch.md) item 8.
- `clear_attached_payload_cells(grid, primitives, patches, padding_m) -> size_t` — Zero every occupied cell within `resolution·√3/2 + padding_m` of a payload primitive's surface, except cells a patch withholds; returns the count cleared. Only ever writes 0 (never marks); a grid with mismatched dimensions or non-positive resolution is left untouched.
- `struct AttachedObjectRevision` — One payload's attachment-record identity: the wire `object_id` plus `WorldStateStamped.attachment_revision`, the producer's counter bumped once per atomic change, never per frame.
- `struct AttachSweepObservation` — That identity plus the payload's `payload_position` on one grid, stated in the OctoMap frame — not `base_link`, since a payload motionless in the base frame on a driving base still leaves residue behind.
- `class AttachSweepLedger` — The bridge's attach-window memory (the other is `SupportPatchAnchors`): the open attach-transition window per live object (identity, the position it is anchored at, whether it has shut). Not a latch on cells — nothing about which cells were cleared is remembered, and no cell is held out of a later grid.
  - `sweep(present, window_m) -> vector<uint8_t>` — Which of `present` are still inside their attach-transition window, answering and recording in one call. A key seen for the first time opens its window; the window latches closed once the payload has moved more than `window_m` from its anchor, and an absent object is forgotten (so detach restarts the window).
  - `size() -> size_t` — How many windows the last swept grid carried.
- `struct SupportWitnessKey` — The kernel's support-witness re-arm key (object id, support id, producer `stamp_ns`), with `operator==`.
- `struct SupportPatchObservation` — One attestation on one grid: its key and its `SupportPatch` in the world-fixed OctoMap frame.
- `class SupportPatchAnchors` — The first-grid plane per live attestation, so the bridge's support band stays on the support instead of riding a lifted payload into its own stale interior (Isaac i64/i70 false stop). Same commit discipline as `AttachSweepLedger`.
  - `anchor(present) -> vector<SupportPatch>` — The first-seen patch for each of `present`, in order; an absent key is forgotten.
- `using AttachLinkTfFrames = std::unordered_map<std::string, std::string>` — Manifest attach link → the TF frame it is looked up as.
- `parse_attach_link_tf_frames(entries, out, error) -> bool` — Parse the bridge's `attach_link_tf_frames` param (`"link=frame"`, the HAL's `vision_attachment_tf_frames` strings; empty entries skipped). A malformed entry or a link mapped to two frames returns `false` with `error` set and `out` EMPTY, so every link falls back to its own name and a renamed hand's payload stays in the map (fail closed).
- `tf_frame_for(frames, attach_link) -> const std::string&` — The frame `clear_attached_payload` asks tf2 for: the mapping, else `attach_link` itself.
- `attach_transition_padding(steady_padding_m, attach_sweep_padding_m) -> double` — The clearing padding one object gets on one grid: steady padding plus the attach-sweep padding while its window is open, so widened sweep is never tighter than an ordinary frame. Non-finite or negative inputs contribute 0 — a parameter can never narrow the clearing below what the payload's volume explains.
- `struct SweptBox` — Axis-aligned box (`lo`, `hi`, `valid`, `add(point)`) in the octomap frame: one carried-payload primitive at one attached pose; the bridge clears these from the octree via `payload_sweep_clear_service` (octomap_server `~/clear_bbox`) every `payload_sweep_clear_period_s`, attached poses only (no pre-attach backfill). Off by default; the launch sets the service only for `DeployRuntime.payload_sweep_clear_enabled` on `hal_mode == "sim"`. User-approved 2026-10-16; Safety-WG review + hazard-log row owed (the payload's volume is cleared from the world map).
- `add_payload_to_swept_box(box, primitives, frame_from_base) -> void` — Grow the box by the exact axis-aligned bounds of the placed primitives (box corners; sphere/capsule caps grown by the radius); non-finite primitives add nothing.
- `payload_primitive_boxes(primitives, frame_from_base) -> vector<SweptBox>` — One box per primitive at its exact bounds (`add_payload_to_swept_box` on that primitive alone): the payload's own posed volume at one pose, never the AABB of its motion (PR #417 review: a diagonal approach's AABB took in the neighbour beside the path).
- `enum class SweepFloor { kSupport, kMeasuredPayload, kNone }` — Where `floor_swept_box` put a box's floor; `kNone` = do not clear it.
- `struct SweepSupports` — Every support known for a swept box, in its frame: `patches` (attested during the carry, as first attested), `static_points` (declared static geometry), `has_measured_floor` / `measured_floor_z` (a measured payload's lowest point at attach).
- `floor_swept_box(box, supports, resolution) -> SweepFloor` — The swept-volume `clear_bbox` never reaches below a support: one cell above every patch under the box and every static point under its footprint below its top, and never below the REGION payload's attach bottom when measured (the highest bound wins; a support under the box never switches it off) (the node sets `SweepSupports::measured_floor_z` only for `evidence_kind == "grasp_target_region"`; a segmented payload's bottom can include counter / depth-edge points, LeIsaac g17); with neither (a segmented or gripper-closure payload), `kNone` and the box is not cleared (DROID t7 fix). A non-vertical patch under the box is `kNone`; every `kNone` leaves `box.valid == false`.
- `floor_pose_boxes(boxes, supports, resolution, out) -> SweepFloor` — One clearing period's `clear_bbox` set: each posed box floored by `floor_swept_box` and never below the union's floor (the highest bound wins, so it clears a subset of what the one union box would); `kNone` boxes and boxes inside one already in `out` are dropped; the union's `kNone` clears nothing.

### `packages/openral_octomap_bridge/include/openral_octomap_bridge/identity_clearing.hpp`
_C++ (Layer 2). Octomap clearing by identity (issue #418): an unambiguous tracker binding's cells, its sweep since `t_attach` and its vacated box; `PINCH_FALLBACK` / ambiguous bindings get none (hazard OT-8). Wired in `OctomapVoxelBridge::clear_by_identity` (release window: `clear_released_box`)._

- `kTrackedObjectEvidence` / `kPinchFallbackEvidence` — The `AttachmentEvidenceKind` wire values; the legacy path skips both.
- `kIdentityBackfillMaxS = 5.0`, `kIdentityBackfillStepS = 0.05`, `kIdentityMaxInterpGapS = 0.25`, `kIdentityMaxClearBoxes = 64` — Backfill depth and step, the longest gap bridged by interpolation, and `clear_bbox` requests per clear.
- `kReleasedClearTimeoutS = 3.0`, `kReleasedClearTimeoutMaxS = 10.0` — Default and bound of `released_clear_timeout_s`; must equal the kernel's `tracked_release_timeout_s` (K5).
- `IdentityKeys`, `struct ReleaseClear`, `ReleaseClears` — `(object_id, epoch)` set; one open release window (epoch, opened stamp) and the windows by id.
- `advance_release_clears(windows, was_bound, state, grid_frame, now_ns, timeout_s) -> vector<const TrackedObject*>` — Release window (#418 F2): opens for a `released` track whose `(id, epoch)` the bridge cleared by identity last tick; kept while the track is unique, `released`, same epoch, the wire carries exactly one `tracked_object` attachment with that id and epoch, and within the timeout; else erased for good.
- `released_clear_pieces(set, track, frame_from_grid, pieces, keep) -> bool` — A RELEASED track's box as the clearing piece plus its keep-out (`floor_z`, other tracks' boxes).
- `tracked_set_usable(state, grid_frame) -> bool` — Valid set whose frame and every box's frame is the grid frame.
- `identity_clearing_track(object, state, grid_frame) -> const TrackedObject*` — The bound track licensing identity clearing, or null (wrong kind, unusable set, no or duplicated id, not `bound`, epoch / attach-link mismatch, non-finite `floor_z`, another track bound to the link, another payload on it).
- `track_box(track, out) -> bool` — A track's box as a `SHAPE_BOX` `PayloadPrimitive`; fail-closed on a bad extent or pose.
- `struct IdentityPiece` — One placed primitive, its inverse pose and optional tight hull (`SelfHull`).
- `place_identity_pieces(object, frame_from_link, out) -> bool` — `place_attached_object` plus each primitive's tight refinement; malformed tight geometry clears nothing.
- `identity_covers(pieces, c, half_cell) -> bool` — Cell centre within half a cell of a primitive AND its tight hull.
- `struct IdentityKeepOut` — Floor plane + boxes no cleared cell may meet.
- `identity_keep_out(set, self, frame_from_grid, keep_own_box, out) -> bool` — `self.floor_z` plane and every other track's box (own box too when `keep_own_box`), in the clearing frame.
- `identity_keeps(keep, c, resolution) -> bool` — Below the floor or within the circumradius of a keep-out box.
- `clear_identity_cells(grid, pieces, keep) -> size_t` — Grid current-pose clear; never marks.
- `append_swept_poses(out, a, b, reach_m, step_m) -> void` — Interpolated poses (lerp + slerp) so no point within `reach_m` moves more than `step_m`.
- `octree_clear_boxes(tree, window, clear, max_boxes, truncated) -> vector<SweptBox>` — Accepted occupied finest leaves as cell-exact `clear_bbox` boxes (x-runs merged along y), capped.

### `packages/openral_octomap_bridge/include/openral_octomap_bridge/octree_freshness.hpp`
_C++ (Layer 2), header-only. When the bridge may still republish the last octree it received (hazard log Entry 033)._

- `kMaxOctreeAgeS = 2.0` — Cap on `max_octree_age_s`: the kernel's own `world_voxel_deadline_ms` cap. Mirrors `DeployRuntime.world_voxel_deadline_s <= 2.0` (pinned by `tests/unit/test_perception_caps_mirror.py`).
- `valid_max_octree_age(max_age_s) -> bool` — Is `max_octree_age_s` usable: in (0, `kMaxOctreeAgeS`]; NaN and infinity are not. The node logs an ERROR and publishes nothing otherwise.
- `octree_is_fresh(age_s, max_age_s) -> bool` — May an octree received `age_s` ago (receipt time, the node's clock — the one the kernel times voxel freshness on) still be published? False for an unusable bound and for a negative or non-finite age, so a map of unknown age is never republished; past the bound the bridge goes silent and the kernel's `world_voxel_deadline_ms` drops with `DROP_VOXEL_UNAVAILABLE`.
- `OccupancyVoxels.source_stamp` (set by the bridge node in `src/octomap_voxel_bridge.hpp`) — The octree's own stamp, which `octomap_server` takes from the capture stamp of the cloud it inserted, carried into every grid built from it. `header.stamp` is production time (fresh on every republish) and cannot say how old the WORLD is; this can, and the kernel's `world_voxel_data_age_budget_ms` budgets it (an unset value reads as stale). The bridge logs `world_voxels data age N ms at publish` every 5 s.

### `packages/openral_octomap_bridge/include/openral_octomap_bridge/octree_to_grid.hpp`
_C++ (Layer 2). OctoMap → dense base-frame grid lowering, ROS-graph-free._

- `struct GridSpec` — The volume to cover: `center[3]` + `radius`, a ball (not a box) in the base frame, since only a ball keeps cell count heading-invariant as the octree's lattice turns with `base_frame`. `radius` must reach wherever the kernel's checked geometry can go — a manifest property, so the bridge node has no default that publishes.
- `rasterize_octree_to_grid(tree, base_to_octomap, spec, base_frame) -> openral_msgs::msg::OccupancyVoxels` — Lower a real `octomap::OcTree` onto the kernel grid on the octree's own lattice, so occupied volume carries over exactly. Returns an empty (unpublished) grid on an unusable radius/spec rather than throwing from the timer callback; still inherits octomap's own ≤1-resolution sensor-ward inflation.
- `mark_base_points(grid, base_points) -> std::size_t` — Marks every cell of a published grid that contains one of the base-frame points occupied, on the grid's own placement (`origin`/`orientation`/`resolution`); only adds occupancy; returns cells newly marked; a malformed grid is left untouched. The bridge applies it to the declared static geometry (`static_points_file` param) before payload clearing. (#349)
- `load_points_file(path, out, error) -> bool` — Reads base-frame `x y z` lines (`#`/blank skipped); false + `error` (`path:line:`) on an unreadable file or malformed line, `out` then empty. A configured file that does not load is fatal at bridge start.
