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

- const `DEFAULT_RATE_HZ: float = 30.0` — Advertised default snapshot rate. (L89)
- const `DEFAULT_STALENESS_S: float = 0.5` — General component staleness window, covering every heterogeneous-rate component with margin over the slowest expected stream; a freshness indicator, not a safety gate. (L96)
- const `DEFAULT_POLICY_STATE_STALENESS_S: float = 5.0` — Staleness window for `policy_state`, which is step-locked not rate-locked; wide enough for a heavy sidecar sim while still flagging a wedged one faster than its 120 s ZMQ timeout. (L103)
- `class WorldStateAggregator` — Aggregates sensor data and produces `WorldState` snapshots. (L106)
  - `__init__(description, *, staleness_limit_s=DEFAULT_STALENESS_S, image_staleness_limit_s=None, policy_state_staleness_limit_s=DEFAULT_POLICY_STATE_STALENESS_S, clock_fn=None)` — Camera and policy-state streams get independent staleness windows: `image_staleness_limit_s` defaults to the general window but deploy sim passes 5.0 s for slow rendered frames vs 0.5 s for real deploys; `policy_state` defaults to 5.0 s since it's step-locked. (L166)
  - `update_joint_state(state) -> None` — Record a fresh joint reading. (L274)
  - `update_policy_state(values) -> None` — Store a defensive copy of simulator-native checkpoint proprioception for `WorldState.policy_state`. (L297)
  - `update_image_frame(sensor_name, frame: SensorFrame) -> None` — Record an inline pixel payload for a named sensor (unlike `update_image`, which stores only the topic ref). (L308)
  - `update_image(sensor_name, topic, stamp_ns) -> None` — Record image arrival. (L285)
  - `update_ee_pose(ee_name, pose) -> None` — Record EE pose from tf2. (L342)
  - `update_base_pose(pose, twist=None) -> None` — Record base pose (and optional twist). (L356)
  - `update_battery(pct) -> None` — Record battery %. (L373)
  - `update_attached_objects(objects: list[AttachedCollisionObject], *, revision=0, stamp_ns=None, place_declaration: PlaceDeclaration | None = None) -> None` — Atomically replaces the complete attached-payload set; duplicate ids or backwards revisions raise `ValueError`. `place_declaration` is replaced in the same atomic step as its payload, and liveness is judged against the stream's own `stamp_ns`, never `clock_fn`, so a sim-clock declaration is never wrongly judged stale by the wall clock. (L397)
  - `set_error(component, status='error') -> None` — Latch a forced diagnostic. (L459)
  - `clear_error(component) -> None` — Remove a forced diagnostic. (L475)
  - `snapshot() -> WorldState` — Produce a typed snapshot (hot path, acquires lock), emitting OTel span/metrics for staleness and latched errors. `staleness_latched` fires only for a component that has had data before — a never-received one counts as stale but doesn't latch, so bringup stays quiet before the HAL's first publish. (L486)
  - `update_detected_objects(objects: list[DetectedObject]) -> None` — Replace the remembered detected-object set (thread-safe); the next `snapshot()` reflects it. Called by the world-state lifecycle node's memory tick. (L382)
  - `_emit_snapshot_telemetry(span, diag, ages_ms) -> None` — Internal: lift the snapshot diagnostics onto the OTel span + meter instruments. (L622)

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

### `packages/openral_octomap_bridge/include/openral_octomap_bridge/self_filter.hpp`
_C++ (Layer 2). The real-camera counterpart of the sim depth renderer's robot transparency: removes the robot's (and a held payload's) own returns from a depth cloud before `octomap_server` inserts it. Poses the kernel's own collision primitives via an FK that mirrors the kernel's operation for operation (a test links the kernel library to pin it; the runtime does not, since Layer 2 may not depend on Layer 5). The node is `src/robot_self_filter.hpp` (`robot_self_filter` executable), wired by `deploy_e2e.launch.py` on the real path only._

- `enum class SelfJointKind` — The kernel's joint codes: `kFixed` / `kRevolute` / `kPrismatic`.
- `struct SelfModelParams` — The kernel's `collision_*` parameter arrays, exactly as the launch passes them.
- `struct SelfModel` — Flattened tree + every link-attached primitive (capsule, sphere as a zero-length capsule, box) in its link frame; `link_index(name) -> int`.
- `self_transform_from_xyz_rpy(x, y, z, roll, pitch, yaw) -> tf2::Transform` — `Rz·Ry·Rx`, the kernel's `transform_from_xyz_rpy` element for element.
- `build_self_model(params, out, error) -> bool` — All-or-nothing, like the kernel's `load_collision_model`: any shape disagreement, out-of-order parent, unknown joint code, out-of-range primitive link or non-finite/negative dimension refuses the model.
- `self_forward_kinematics(model, q, n_dof, link_world)` — Link frames in the model root; pinned to 1e-12 against `openral_safety_kernel::forward_kinematics` by `test_self_filter.cpp`.
- `place_self_primitives(model, link_world, frame_from_root, out)` — Every robot primitive placed in the cloud's frame.
- `class SelfMask` — Primitives in the cloud frame with inverse poses and broad-phase bounds precomputed once per cloud; `set(primitives, padding_m)`, `contains(p) -> bool`.
- `struct SelfFilterStats` — `points_in`, `non_finite`, `removed`, `kept` for one call.
- `filter_xyz_points(data, n_points, point_step, x_offset, y_offset, z_offset, mask, out) -> SelfFilterStats` — Copies every finite point the mask does not contain, whole records, in order.
- `class RobotSelfFilter` (src/robot_self_filter.hpp) — The node: joint positions from a short history at the cloud's CAPTURE stamp (`max_joint_state_skew_s`, manifest names or their `collision_joint_aliases`), camera pose from tf2 at that stamp, held payloads from `/openral/world_state_fast`. No pose at the stamp drops the cloud (the map goes stale and the kernel fails closed); an unplaceable payload stays in the cloud. Forwards the capture stamp unchanged; logs `self-filter:` every 5 s.

### `packages/openral_octomap_bridge/include/openral_octomap_bridge/payload_clearing.hpp`
_C++ (Layer 2). The attached payload's own occupancy leaves the published `OccupancyVoxels` grid here, partitioned against the safety kernel's support-contact witness (ADR-0092 D6); header-documented and gtested against a real `octomap::OcTree`._

- `struct PayloadPrimitive` — One attached-payload collision primitive placed in the grid frame, mirroring the kernel's `AttachedPrimitive` conventions (a capsule's segment on local +Z, a box's half-extents on its own axes).
- `struct SupportPatch` — One attested support contact placed in the grid frame (point + outward normal + patch radius + max penetration); the wire form is `openral_msgs/SupportContactWitness`, lifted here through the payload's live pose.
- `place_attached_object(object, grid_from_link, out, out_patches) -> bool` — Place one wire `AttachedCollisionObject`'s primitives (and support patch, when valid) using the kernel's own pose composition. Fail-closed on any of the kernel's `ingest_attached_objects` validation rules, returning `false` and appending nothing — so a rejected object clears nothing.
- `support_patch_withholds(patch, center, resolution) -> bool` — Does an attested patch withhold this cell from clearing? Mirrors the kernel's `support_contact_exempts` (with zero slack) so the withheld set is always a subset of what the kernel exempts — a withheld cell can never be the one that stops the robot. Deliberate mirror of the kernel predicate; see [14-duplication-watch.md](14-duplication-watch.md) item 8.
- `clear_attached_payload_cells(grid, primitives, patches, padding_m) -> size_t` — Zero every occupied cell within `resolution·√3/2 + padding_m` of a payload primitive's surface, except cells a patch withholds; returns the count cleared. Only ever writes 0 (never marks); a grid with mismatched dimensions or non-positive resolution is left untouched.
- `struct AttachedObjectRevision` — One payload's attachment-record identity: the wire `object_id` plus `WorldStateStamped.attachment_revision`, the producer's counter bumped once per atomic change, never per frame.
- `struct AttachSweepObservation` — That identity plus the payload's `payload_position` on one grid, stated in the OctoMap frame — not `base_link`, since a payload motionless in the base frame on a driving base still leaves residue behind.
- `class AttachSweepLedger` — The bridge's only memory: the open attach-transition window per live object (identity, the position it is anchored at, whether it has shut). Not a latch on cells — nothing about which cells were cleared is remembered, and no cell is held out of a later grid.
  - `sweep(present, window_m) -> vector<uint8_t>` — Which of `present` are still inside their attach-transition window, answering and recording in one call. A key seen for the first time opens its window; the window latches closed once the payload has moved more than `window_m` from its anchor, and an absent object is forgotten (so detach restarts the window).
  - `size() -> size_t` — How many windows the last swept grid carried.
- `attach_transition_padding(steady_padding_m, attach_sweep_padding_m) -> double` — The clearing padding one object gets on one grid: steady padding plus the attach-sweep padding while its window is open, so widened sweep is never tighter than an ordinary frame. Non-finite or negative inputs contribute 0 — a parameter can never narrow the clearing below what the payload's volume explains.

### `packages/openral_octomap_bridge/include/openral_octomap_bridge/octree_freshness.hpp`
_C++ (Layer 2), header-only. When the bridge may still republish the last octree it received (hazard log Entry 033)._

- `valid_max_octree_age(max_age_s) -> bool` — Is `max_octree_age_s` usable: finite and strictly positive. The node logs an ERROR and publishes nothing otherwise.
- `octree_is_fresh(age_s, max_age_s) -> bool` — May an octree received `age_s` ago (receipt time, the node's clock — the one the kernel times voxel freshness on) still be published? False for an unusable bound and for a negative or non-finite age, so a map of unknown age is never republished; past the bound the bridge goes silent and the kernel's `world_voxel_deadline_ms` drops with `DROP_VOXEL_UNAVAILABLE`.
- `OccupancyVoxels.source_stamp` (set by the bridge node in `src/octomap_voxel_bridge.hpp`) — The octree's own stamp, which `octomap_server` takes from the capture stamp of the cloud it inserted, carried into every grid built from it. `header.stamp` is production time (fresh on every republish) and cannot say how old the WORLD is; this can, and the kernel's `world_voxel_data_age_budget_ms` budgets it (an unset value reads as stale). The bridge logs `world_voxels data age N ms at publish` every 5 s.

### `packages/openral_octomap_bridge/include/openral_octomap_bridge/octree_to_grid.hpp`
_C++ (Layer 2). OctoMap → dense base-frame grid lowering, ROS-graph-free._

- `struct GridSpec` — The volume to cover: `center[3]` + `radius`, a ball (not a box) in the base frame, since only a ball keeps cell count heading-invariant as the octree's lattice turns with `base_frame`. `radius` must reach wherever the kernel's checked geometry can go — a manifest property, so the bridge node has no default that publishes.
- `rasterize_octree_to_grid(tree, base_to_octomap, spec, base_frame) -> openral_msgs::msg::OccupancyVoxels` — Lower a real `octomap::OcTree` onto the kernel grid on the octree's own lattice, so occupied volume carries over exactly. Returns an empty (unpublished) grid on an unusable radius/spec rather than throwing from the timer callback; still inherits octomap's own ≤1-resolution sensor-ward inflation.
