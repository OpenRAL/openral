// SPDX-License-Identifier: Apache-2.0
// Attach-time (and ongoing) clearing of a grasped payload's own occupancy.
//
// `openral_msgs/AttachedCollisionObject` states the contract: "The same object
// must be absent from world occupancy while attached." Before the grasp the
// object legitimately *is* world occupancy — its cells were marked by honest
// sensor returns. At the attach transition those cells stop describing the
// world and start describing the robot's own payload, which the kernel already
// re-checks as collision-active attached geometry. Nothing removed them, so
// they stayed in the map and stopped the arm against the thing it was holding.
//
// This is the removal, done where occupancy is owned (this Layer-2 bridge) and
// on the map itself, not as an exemption inside the kernel: every consumer of
// `/openral/world_voxels` sees the payload gone, and the kernel keeps its one
// unconditional rule — an occupied cell is an obstacle.
//
// The clearing is PARTITIONED against the kernel's support-contact witness
// (ADR-0092 D6). A payload resting on a counter shares its bottom cell layer
// with the counter's own top surface, so a clearing that only knows the
// payload's volume takes the counter with it: the surface the witness is
// attested against disappears from the map, the kernel's occupancy-based
// liveness test finds nothing left to keep the witness alive
// (`update_support_contact_witnesses`), and the very next frame in which a
// support cell falls back outside the clearing reach stops the robot against
// the surface it is resting on — unexempted, because the witness is dead and
// only World State can re-arm it. That is the 2026-08-14 failure, reproduced
// 2/2 on baguette+counter and cup+island, with the nearest surviving cell
// measured 21.77 mm from the payload (one circumradius out).
//
// So the two mechanisms divide the cells between them, and the division is
// exhaustive: within the payload's reach a cell is either CLEARED by this
// bridge or EXEMPTED by the kernel's witness, never neither, never both.
// `support_patch_withholds` is this side of that line — the same geometry the
// kernel's `support_contact_exempts` uses, with zero slack, so what this bridge
// keeps is a subset of what the kernel exempts and no withheld cell can stop
// the robot. Everything else around the payload clears exactly as before:
// withholding only ever puts occupancy BACK into the published map, which is
// also what Nav2 and SLAM need, because those cells are the counter.

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include <tf2/LinearMath/Transform.h>
#include <tf2/LinearMath/Vector3.h>

#include <openral_msgs/msg/attached_collision_object.hpp>
#include <openral_msgs/msg/occupancy_voxels.hpp>

namespace openral_octomap_bridge {

/// One attached-payload collision primitive, already placed in the grid frame.
///
/// The same convex shapes the kernel ingests from
/// `openral_msgs/AttachedCollisionPrimitive`, with the same conventions: a
/// capsule's central segment lies on the primitive frame's local +Z, and a
/// box's `half_extents` are along its own local axes.
struct PayloadPrimitive {
  /// `AttachedCollisionPrimitive::SHAPE_SPHERE` / `_CAPSULE` / `_BOX`.
  std::uint8_t shape_type{0};
  /// Primitive origin expressed in the occupancy grid's frame.
  tf2::Transform pose{tf2::Transform::getIdentity()};
  double radius{0.0};                        ///< sphere / capsule radius (m)
  double half_length{0.0};                   ///< capsule central-segment HALF length (m)
  tf2::Vector3 half_extents{0.0, 0.0, 0.0};  ///< box half extents (m)
};

/// One attested support-contact patch, already placed in the grid frame.
///
/// The wire form (`openral_msgs/SupportContactWitness`, carried on
/// `AttachedCollisionObject.support_contact`) states the geometry in the
/// attached **object's** frame; this is that geometry lifted into the grid
/// frame by the same live pose the payload primitives are placed with, which is
/// where the cells it protects live. `normal` points **from the supporting
/// solid toward the payload**, so the support occupies the half-space
/// `(x - point)·normal <= 0` — the same convention the kernel's
/// `AttachedObject::support_normal` uses.
struct SupportPatch {
  tf2::Vector3 point{0.0, 0.0, 0.0};   ///< attested contact point, grid frame
  tf2::Vector3 normal{0.0, 0.0, 1.0};  ///< unit outward support normal, grid frame
  double patch_radius{0.0};            ///< lateral radius of the supported patch (m, > 0)
  double max_penetration{0.0};         ///< attested PHYSICAL contact depth bound (m, >= 0)
};

/// Place one wire object's primitives — and its support-contact attestation, if
/// it carries one — into the grid frame, appending them to `out` / `out_patches`.
///
/// `grid_from_link` maps a point in the object's attach-link frame into the
/// grid frame (i.e. `lookupTransform(base_frame, object.attach_link)`); each
/// primitive lands at `grid_from_link · pose_in_link · pose_in_object`, the
/// same composition the kernel uses to place the payload it checks. The witness
/// is per-object, not per-primitive, so its point and normal land at
/// `grid_from_link · pose_in_link` (the object frame) and one object appends at
/// most one patch. `support_contact_valid == false` — the honest default for
/// any producer that cannot measure support contact — appends no patch, and the
/// object's cells then clear exactly as they did before the witness existed.
///
/// Fail-closed: an object carrying no primitives, an unrecognised shape tag,
/// too few shape dimensions for the tag, a non-finite or negative dimension, a
/// degenerate (zero) pose quaternion, or a malformed attestation (non-finite
/// contact point, degenerate normal, non-positive patch radius, negative
/// penetration — the kernel's own `ingest_attached_objects` rules) returns
/// `false` and appends **nothing** to either output — the caller must then
/// clear nothing at all. Refusing to clear only ever leaves occupancy in the
/// map, which is the conservative direction; a producer that cannot describe
/// its own support contact is one whose payload volume is not trusted either,
/// exactly as the kernel treats it.
bool place_attached_object(const openral_msgs::msg::AttachedCollisionObject& object,
                           const tf2::Transform& grid_from_link, std::vector<PayloadPrimitive>& out,
                           std::vector<SupportPatch>& out_patches);

/// Does an attested support patch claim the cell centred at `center`, and
/// therefore forbid the clearing from removing it?
///
/// The predicate is the kernel's `support_contact_exempts` with `slack = 0`,
/// evaluated on the same lifted plane: with `s = (center - point)·normal` the
/// cell centre's height above the attested support face, the cell is withheld
/// when it is inside the patch laterally (padded by the cell cube's
/// circumradius, `resolution·√3/2`) **and** `s <= w + max_penetration`, where
/// `w = half_resolution·(|n.x| + |n.y| + |n.z|)` is the exact half-width of the
/// cell cube projected on the support normal.
///
/// Dropping the kernel's slack term is what makes the partition safe rather
/// than merely symmetric: the kernel adds `attached_contact_tolerance` (1 mm of
/// physical FK/pose slack) to the same bound, so every cell this withholds is a
/// cell the kernel exempts **for the object that attested it**, and no cell kept
/// by this function can be the one that stops the robot against that object.
/// The reverse inclusion is deliberately not claimed — the kernel may exempt a
/// cell this bridge still clears, and a cleared cell stops nothing. Nor does the
/// containment survive two scope conditions the caller must know about: the
/// kernel's exemption is per-object while `clear_attached_payload_cells`
/// withholds per message (see its docs), and the kernel's lifecycle node caps
/// the attestation it will accept (`support_witness_max_patch_radius_m`,
/// `support_witness_max_penetration_m`) where this predicate takes the wire
/// values as given. Both are stated, with their tests, in the package README.
///
/// The withheld region is the support HALF-SPACE — the slab above the plane
/// plus everything below it — never the patch cylinder. At 25 mm cells the slab
/// reaches at most 21.65 mm (the pad, largest for a cube-diagonal normal) plus
/// the attested depth above the plane, which is what keeps the payload-side
/// residue above a support surface in the clearing's hands.
///
/// A cell BELOW the plane is withheld without a lower bound: that is the body
/// of the counter, real furniture that the map is supposed to carry and that
/// the payload's volume merely happens to pass within a circumradius of.
bool support_patch_withholds(const SupportPatch& patch, const tf2::Vector3& center,
                             double resolution) noexcept;

/// Clear from `grid` every occupied cell the attached payload's own volume
/// explains **except** those an attested support patch claims, in place.
/// Returns the number of cells cleared.
///
/// A cell is cleared when its centre lies no further from a primitive's surface
/// than the cell cube's circumradius (`resolution·√3/2`) plus `padding_m`, and
/// no patch in `patches` withholds it (`support_patch_withholds`) — every
/// object's attestation guards every object's clearing, which is one step wider
/// than the kernel's per-object exemption and is the one place a withheld cell
/// can still stop the robot (README, "Two scope conditions"). Passing an
/// empty `patches` is the un-partitioned behaviour and clears the support
/// surface out from under a resting payload — see the header comment. Because
/// withholding can only ever *skip* a clear, the cells this removes are always
/// a subset of the cells it would remove with no patches at all: the partition
/// puts occupancy back into the map and never takes any away.
///
/// The circumradius is the exact discretisation slop: every cell the payload's
/// volume actually intersects has its centre within it, so this clears all of
/// them, at the price of also clearing a cell the payload merely comes within
/// one circumradius of — bounded by a single cell layer and unavoidable on a
/// discretised map. It is the same bound `support_contact_exempts` uses in the
/// kernel. `padding_m` (default 0 at the call site) buys pose uncertainty on a
/// real robot and does remove cells the payload cannot explain: it is
/// protection given up, so keep it at 0 unless a measured pose error demands
/// otherwise.
///
/// Only cells inside each primitive's inflated AABB are visited, so the cost is
/// the payload's volume, not the grid's. Clearing never *marks*: the function
/// only ever writes 0. A grid whose `occupancy` length disagrees with its
/// dimensions, or whose resolution is not positive, is left untouched (0
/// returned).
std::size_t clear_attached_payload_cells(openral_msgs::msg::OccupancyVoxels& grid,
                                         const std::vector<PayloadPrimitive>& primitives,
                                         const std::vector<SupportPatch>& patches,
                                         double padding_m);

}  // namespace openral_octomap_bridge
