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
  the one-body check; a split inside a declared hand is the hand's own. A second body needs
  `MIN_SPLIT_CELLS` (3) cells to split a box. Cover is visibility-aware: expected cells hidden
  behind robot pixels (`cells_behind_robot`) leave the count, and a confirmation publishes the
  visible count as `visible_cells`, which is the kernel's cover bar. At a TARGET or
  DESTINATION the hand touches (`hand_touches`), a body within a cell of the hand is the piece the fingers cut off;
  a disagreement left by a push inside the box re-anchors the expected cells on what the map
  holds in the box, the support layer left out, when they number at least the old cover's need
  and at most twice the expected cells (`box_cells`, `contact_reanchor`); any other split or disagreement is held without a re-stamp
  (`contact_hold`), so the kernel's age bound ends it. While touched, a refit never shrinks
  the box: it becomes the box enclosing the old and new fits, both hulls kept
  (`contact_union`; past the grasp caps the refit is ignored, `contact_union_capped`). The
  re-anchor also runs on a hidden track under contact, against what the track filled in the
  map at its last plain pass. A mask wider than any capped fit is refused
  (`half_extent_cap`) before the support scan, which a whole-table mask made take ~2 s. A refit that drops surface cells the map still holds is ignored
  (`partial_refit_ignored`). A refit ignored for geometry (through the robot, or while
  occluded, the object still inside its box) still replaces the cells the cover test expects,
  so a target pushed inside its own box is confirmed where its surface is now. Issue #418,
  2026-10-21; Safety-WG review owed.
- `tracks` — `TrackStore` (`get` / `all` / `replace`, the binding/release interface),
  association by continuity (a tie within one voxel updates nothing), ageing on the
  source clock only.
- `grounding` — standing queries (`UNRESOLVED` → `RESOLVING` → `RESOLVED`, `EXHAUSTED`
  when the search budget is spent) with bounded `find-object` escalation emitted as data.
- `scheduler` — one request in flight, at most `segment_hz` per wall second, with one visible
  confirmed-track refresh turn after query work when both are waiting. A hand at a TARGET /
  DESTINATION track puts that track's refresh first among refreshes.
- Late ATTACH requires the existing freshness window to contain a geometry fit; map coverage
  confirmation does not make old fitted geometry fresh. A refit through the robot that leaves
  the clear box in place re-dates that fit (the box is unchanged). The grasp pose is the hold's,
  or the newest within 1 s before it whose pinch was in a fresh box (the policy lifts while
  the jaw still settles). The map is not asked whether the carried box is occupied (the
  self-filter hides a held object, old cells clear late); a wrong bind is released by the
  in-hand drop check below. A derived ATTACH while the hand already holds is the same hold
  (a chattering command re-reads it), never a rebind.
- In `bind` / `on` the place declaration's region is the tracker's: the one CONFIRMED /
  OCCLUDED DESTINATION track labelled as its `destination_query`, its box stamped with its
  last map confirmation, while a payload is held (`TrackerSession.place_region`); `None`
  otherwise. The HAL's flat-support producer never arms over a container.
- Normal `TRACKED_OBJECT` ATTACH also requires a nonfuture fit within `TrackerParams.max_age_s`;
  map confirmation cannot renew that geometry-age check.
- A DESTINATION track is never bound: the place target is where the payload goes, never the
  payload (2026-10-22, Spark R193 / R205: a jaw closing inside the bowl's box bound the bowl).
- A released object is never followed (2026-10-22, Spark R177-R185: the released box stayed
  where the hand let go while the object fell or rolled). Its track stays RELEASED for the
  release window only (the kernel freezes the payload there), takes no fit, prompt or bind,
  and is then retired; the object is plain world until the target query finds it afresh under
  a new id.
- A BOUND track's `box` follows its in-hand camera refits (where the object is seen, never the
  hand's kinematics), so the kernel's binding-meets-hand check reads a carried object on the
  hand and a dropped one off it; the payload primitives and `provisional` stay the bind's.
  Its in-hand refresh is prompted where the hand carries the payload now (the binder's
  kinematics: prompt only, the fit stays the camera's). Two consecutive in-hand fits that
  meet no hand box are a drop: the binding is released as a jaw opening releases it (a jaw
  that never opens cannot release a payload that slipped out).

Every threshold is a calibration point; containment is property-tested on the SO-101
PickOrange, DROID and OpenArm cameras (`tests/unit/test_object_tracker_*.py`).
