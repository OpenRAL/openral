# openral-world-state

OpenRAL World State aggregator — tf2-aware snapshot producer

Part of [**OpenRAL**](https://github.com/OpenRAL/openral) — the open Robot
Abstraction Layer for vision-language-action robotics. This package is one
member of the OpenRAL Python workspace; see the architecture overview and the
seven-layer model in the project docs.

- **Docs:** https://docs.openral.com/
- **Source:** https://github.com/OpenRAL/openral
- **License:** Apache-2.0

> All OpenRAL workspace packages move in lockstep with the workspace SemVer
> version; the bump is computed from Conventional Commits by release-please.

## Object-centric tracker core (`openral_world_state.object_tracker`, issue #418)

ROS-free core of the Layer-2 object tracker; a lifecycle node wraps it.
`ObjectTracker.feed(frame, grid=..., grid_stamp_ns=..., wall_s=..., results=...)` takes one
depth frame (with its robot mask, intrinsics and `T_base<-cam` at the depth stamp), the
newest voxel grid and the answers to the tracker's own requests, and returns at most one
`LocateInView` / `SegmentInView` request, query escalations and the whole
`TrackedObjectSet`.

- `fit` — the containing fit: silhouette cone between the front plane and the support,
  union the depth points' prism; a 64-vertex polytope that contains the object under the
  logged assumptions A1-A4.
- `cover` — size-aware map confirmation (`|E ∩ occupied| >= max(2, ceil(0.5|E|))`) plus
  the one-body check; a split inside a declared hand is the hand's own.
- `tracks` — `TrackStore` (`get` / `all` / `replace`, the binding/release interface),
  association by continuity (a tie within one voxel updates nothing), ageing on the
  source clock only.
- `grounding` — standing queries (`UNRESOLVED` → `RESOLVING` → `RESOLVED`, `EXHAUSTED`
  when the search budget is spent) with bounded `find-object` escalation emitted as data.
- `scheduler` — one request in flight, at most `segment_hz` per wall second, in the
  design's priority order.

Every threshold is a calibration point; containment is property-tested on the SO-101
PickOrange, DROID and OpenArm cameras (`tests/unit/test_object_tracker_*.py`).
