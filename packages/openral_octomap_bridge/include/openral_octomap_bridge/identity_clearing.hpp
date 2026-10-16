// SPDX-License-Identifier: Apache-2.0
// Octomap clearing by identity (issue #418, "Octomap clearing by identity").
//
// When the Layer-2 tracker binds a tracked object to a hand it publishes the
// attachment with `evidence_kind == "tracked_object"` and its `(object_id, epoch)`,
// and the object itself in `WorldStateStamped.tracked_objects`. Only such a binding,
// unambiguous (`identity_clearing_track`), clears by identity:
//
// * current pose: grid cells whose centre lies within half a cell of the bound
//   geometry, refined by its tight hull when the producer ships one;
// * sweep: the same test at every pose since `t_attach` (TF history backfill),
//   interpolated half a cell apart, cleared from the octree cell by cell;
// * vacated cells: once, when a tracked object has left its last confirmed box.
//
// Every clear excludes, always: cells centred below the object's `floor_z`, cells
// meeting any OTHER track's box, and cells holding declared static geometry.
// `PINCH_FALLBACK` and ambiguous bindings get none of it: their cells stay obstacles.
//
// Hazard OT-8 (issue #418): this erases occupancy. Every ambiguity resolves to
// clearing less — see the package README, "Clearing by identity".

#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include <octomap/OcTree.h>
#include <tf2/LinearMath/Transform.h>

#include <openral_msgs/msg/attached_collision_object.hpp>
#include <openral_msgs/msg/occupancy_voxels.hpp>
#include <openral_msgs/msg/tracked_object.hpp>
#include <openral_msgs/msg/tracked_object_set.hpp>
#include <openral_msgs/msg/world_state_stamped.hpp>

#include "openral_octomap_bridge/payload_clearing.hpp"
#include "openral_octomap_bridge/self_filter.hpp"

namespace openral_octomap_bridge {

/// `AttachmentEvidenceKind.TRACKED_OBJECT` / `.PINCH_FALLBACK` wire values.
inline const std::string kTrackedObjectEvidence = "tracked_object";
inline const std::string kPinchFallbackEvidence = "pinch_fallback";

/// Backfill reaches at most this far before the first grid that carries a binding.
inline constexpr double kIdentityBackfillMaxS = 5.0;
/// TF-history sampling step of the backfill.
inline constexpr double kIdentityBackfillStepS = 0.05;
/// Two pose samples further apart in time are not interpolated (the path between
/// them is unknown, and a straight chord could clear where the object never was).
inline constexpr double kIdentityMaxInterpGapS = 0.25;
/// `clear_bbox` requests per clear at most (each one makes `octomap_server`
/// republish the whole map); cells past it stay in the octree.
inline constexpr std::size_t kIdentityMaxClearBoxes = 64;

/// Is the tracked set on `state` one the bridge may act on? Valid, and the set and
/// every box in `grid_frame` (the kernel's own frame rule).
bool tracked_set_usable(const openral_msgs::msg::WorldStateStamped& state,
                        const std::string& grid_frame);

/// The bound track that licenses identity clearing of `object`, or null.
///
/// Null unless: `evidence_kind == "tracked_object"`; the set is usable; exactly one
/// track carries `object.object_id`, with status `bound`, the same `epoch` and
/// `attach_link`, and a finite `floor_z`; no other track is bound to that link and no
/// other attachment rides it (either is an ambiguous binding).
const openral_msgs::msg::TrackedObject*
identity_clearing_track(const openral_msgs::msg::AttachedCollisionObject& object,
                        const openral_msgs::msg::WorldStateStamped& state,
                        const std::string& grid_frame);

/// A track's box as a `SHAPE_BOX` primitive in the grid frame. `false` on a
/// non-positive or non-finite extent or a degenerate pose.
bool track_box(const openral_msgs::msg::TrackedObject& track, PayloadPrimitive& out);

/// One convex piece of a bound object's clearing geometry, placed in one frame.
struct IdentityPiece {
  PayloadPrimitive prim;            ///< placed in the clearing frame
  tf2::Transform local_from_frame;  ///< `prim.pose.inverse()`
  bool has_tight{false};
  SelfHull tight;  ///< the box's tight refinement, box frame
};

/// Place `object`'s primitives and their tight refinement (`tight_dop_*`,
/// `tight_hull_vertices`) through `frame_from_link`, replacing `out`. Fails closed
/// (`false`, `out` empty) wherever `place_attached_object` does, and on malformed
/// tight geometry: refinement on a non-box, a DOP not 13 + 13 finite non-inverted
/// slabs, vertices without a DOP, a ragged or non-finite vertex list, or more than
/// `kSelfMaxHullVertices` vertices.
bool place_identity_pieces(const openral_msgs::msg::AttachedCollisionObject& object,
                           const tf2::Transform& frame_from_link, std::vector<IdentityPiece>& out);

/// Does the cell centred at `c` lie in the pieces ⊕ `half_cell`? Each piece is
/// its primitive AND, when present, its tight hull: the hull only ever narrows.
bool identity_covers(const std::vector<IdentityPiece>& pieces, const tf2::Vector3& c,
                     double half_cell);

/// What an identity clear never takes, in the clearing frame.
struct IdentityKeepOut {
  tf2::Vector3 floor_point{0.0, 0.0, 0.0};   ///< a point of the `floor_z` plane
  tf2::Vector3 floor_normal{0.0, 0.0, 1.0};  ///< its up direction
  std::vector<PayloadPrimitive> boxes;       ///< boxes no cleared cell may meet
};

/// The keep-out for clearing `self`'s cells, carried into the clearing frame by
/// `frame_from_grid`: its `floor_z` plane and the box of every other track in
/// `set`, whatever its status, plus `self`'s own box when `keep_own_box`. `false`
/// (clear nothing) when any of those boxes cannot be placed.
bool identity_keep_out(const openral_msgs::msg::TrackedObjectSet& set,
                       const openral_msgs::msg::TrackedObject& self,
                       const tf2::Transform& frame_from_grid, bool keep_own_box,
                       IdentityKeepOut& out);

/// Must the cell centred at `c` stay? Centre below the floor plane, or within the
/// cell's circumradius of a keep-out box (a superset of the cells meeting it).
bool identity_keeps(const IdentityKeepOut& keep, const tf2::Vector3& c, double resolution);

/// Clear from `grid` (cell centres in its own frame, the base frame) every occupied
/// cell `identity_covers` and `identity_keeps` does not. Returns cells cleared.
/// Never marks; a grid that does not describe itself is left untouched.
std::size_t clear_identity_cells(openral_msgs::msg::OccupancyVoxels& grid,
                                 const std::vector<IdentityPiece>& pieces,
                                 const IdentityKeepOut& keep);

/// Append to `out` the poses from `a` (exclusive) to `b` (inclusive), interpolated
/// so no point within `reach_m` of the link origin moves more than `step_m` between
/// two of them: the hull sweep between consecutive samples.
void append_swept_poses(std::vector<tf2::Transform>& out, const tf2::Transform& a,
                        const tf2::Transform& b, double reach_m, double step_m);

/// The occupied leaves of `tree` inside `window` whose centre `clear` accepts, as
/// `clear_bbox` boxes that each cover only accepted cells (x-runs merged along y),
/// at most `max_boxes` (`truncated` set when more were needed; the rest stay).
///
/// Only finest-resolution leaves qualify: a pruned (coarser) occupied leaf is never
/// cleared, since `clear_bbox` frees a whole leaf it touches.
std::vector<SweptBox> octree_clear_boxes(const octomap::OcTree& tree, const SweptBox& window,
                                         const std::function<bool(const tf2::Vector3&)>& clear,
                                         std::size_t max_boxes, bool& truncated);

}  // namespace openral_octomap_bridge
