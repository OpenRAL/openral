# openral-world-state

OpenRAL World State: the tf2-aware snapshot aggregator (Layer 2), the 2D-to-3D object memory and
the object-centric tracker core. ROS-free; the ROS nodes live in
[`packages/world_state`](../../packages/world_state/README.md).

Part of [**OpenRAL**](https://github.com/OpenRAL/openral) — the open Robot
Abstraction Layer for vision-language-action robotics. This package is one
member of the OpenRAL Python workspace; see the architecture overview and the
seven-layer model in the project docs.

- **Docs:** https://docs.openral.com/
- **Source:** https://github.com/OpenRAL/openral
- **License:** Apache-2.0

> All OpenRAL workspace packages move in lockstep with the workspace SemVer
> version; the bump is computed from Conventional Commits by release-please.

## Modules

| Module | Role |
|---|---|
| `aggregator` | `WorldStateAggregator`: the one place joint, sensor and attachment state fuse into the `WorldState` snapshot read by S1 and S2. |
| `object_lift`, `object_memory` | `VoxelFrustumLifter` lifts a 2D detection to a `map`-frame 3D centre; `ObjectMemory` keeps a temporal existence belief per object (`build_miss_classifier`, `MissEvidence`). |
| `spatial_memory` | `SpatialMemory` scene graph for `recall_object`, `compute_approach_viewpoint`, gaze and approach poses. |
| `grid`, `geometry`, `embedder` | `OccupancyGridIndex` and `refine_approach_pose`; pose helpers; `OpenClipEmbedder` (`TextEmbedder`). |
| `scene_objects_span` | The `world.scene_objects` span payload (`emit_scene_objects_span`). |
| `object_tracker` | The tracker core below. |

## Object-centric tracker core (`openral_world_state.object_tracker`)

ROS-free core of the Layer-2 object tracker; `object_tracker_node` wraps it through
`TrackerSession`. `ObjectTracker.feed(frame, grid=..., grid_stamp_ns=..., wall_s=...,
results=...)` takes one depth frame (with robot mask, intrinsics and `T_base<-cam` at the depth
stamp), the newest voxel grid and the answers to the tracker's own requests. It returns at most
one `LocateInView` / `SegmentInView` request, query escalations and the whole
`TrackedObjectSet`.

| Module | Role |
|---|---|
| `fit` | The containing fit: silhouette cone between the front plane and the support, unioned with the depth points' prism; a 64-vertex polytope that contains the object under the logged assumptions A1-A4. A mask wider than any capped fit is refused (`half_extent_cap`). |
| `multi_view` | Intersects fits from several cameras. |
| `cover` | Size-aware map confirmation (`\|E and occupied\| >= max(2, ceil(0.5\|E\|))`) and the one-body check. A second body needs `MIN_SPLIT_CELLS` cells to split a box; a split inside a declared hand is the hand's own. Cover is visibility-aware: cells hidden behind robot pixels (`cells_behind_robot`) leave the count, and a confirmation publishes the visible count as `visible_cells`, the kernel's cover bar. |
| `tracks` | `TrackStore` (`get` / `all` / `replace`, the only interface binding and release mutate), association by continuity, ageing on the source clock only. |
| `grounding` | Standing queries (`UNRESOLVED`, `RESOLVING`, `RESOLVED`, `EXHAUSTED` once the search budget is spent) with bounded find-object escalation emitted as data. |
| `scheduler` | One request in flight, at most `segment_hz` per wall second; one visible confirmed-track refresh turn after query work when both wait. A hand at a TARGET / DESTINATION track puts that track's refresh first. |
| `binding`, `release`, `pinch`, `commands`, `stall` | Hand binding at ATTACH, release at DETACH or jaw opening, pinch geometry, the stalled-jaw read from approved gripper commands (`JawStallDetector`). |
| `hidden_side`, `catalog`, `surface_model` | Hidden-side evidence, declared-object catalog geometry, the opt-in observed-surface pilot. |
| `heartbeat` | `StarvationFill`, the cached-set republish. |
| `session` | `TrackerSession`: every decision the node needs, with no ROS. |

Behaviour:

- **Contact refits.** At a TARGET or DESTINATION the hand touches (`hand_touches`), a body
  within one cell of the hand is the piece the fingers cut off. A disagreement left by a push
  inside the box re-anchors the expected cells on what the map holds in the box
  (`contact_reanchor`); any other split or disagreement is held without a re-stamp
  (`contact_hold`), so the kernel's age bound ends it. While touched, a refit neither shrinks
  nor grows the box: the old box and hulls stand and the refit only re-stamps it
  (`contact_refit_kept`).
- **Growth bound.** No refit grows a track past `refit_growth_factor` (1.5) times its first
  clear fit (`refit_growth_refused`); one with two thirds of its surface still inside the box
  re-dates it (`refit_growth_redated`). A re-orientation is no growth (`refit_reoriented`). A
  refit that drops surface cells the map still holds is ignored (`partial_refit_ignored`). A
  moved refit through the robot keeps the clear box's size, re-centred in x/y
  (`covered_move_size_kept`), and turns the track provisional; a shift within one cell is noise
  (`covered_move_within_cell`).
- **Binding.** A normal `TRACKED_OBJECT` ATTACH needs a non-future fit within
  `TrackerParams.max_age_s`; map confirmation cannot renew that age, though a refit that leaves
  the clear box in place re-dates it. An OCCLUDED target's map agreement counts as fresh (the
  hand hides it from the camera, not the map). The grasp pose is the hold's, or the newest
  within 1 s before it whose pinch was in a fresh box. A derived ATTACH while the hand already
  holds is the same hold, never a rebind. A DESTINATION track is never bound.
- **Release.** A released object is never followed: its track stays RELEASED for the release
  window only (the kernel freezes the payload there), takes no fit, prompt or bind, and is then
  retired. The object is plain world until the target query finds it afresh under a new id.
- **Bound tracks.** A BOUND track's `box` follows its in-hand camera refits, so the kernel's
  binding-meets-hand check sees a carried object on the hand and a dropped one off it. Two
  consecutive in-hand fits that meet no hand box are a drop: the binding is released as a jaw
  opening releases it.
- **Regions** (`TrackerSession`). In `tracking` / `on` the place declaration's region is the one
  CONFIRMED / OCCLUDED DESTINATION track labelled as its `destination_query`, stamped with its
  last map confirmation, while a payload is held (`place_region`). In `tracking` only, the
  grasp declaration's region is the one CONFIRMED / OCCLUDED TARGET track keyed to it
  (`grasp_region`): `None` with no unique target, past the grasp-region caps, or once a hold
  started. It is a pre-grasp allowance the attachment snapshot never latches.

Every threshold is a calibration point (`tracking_thresholds.yaml`). Containment is
property-tested on the SO-101 PickOrange, DROID and OpenArm cameras
(`tests/unit/test_object_tracker_*.py`).
