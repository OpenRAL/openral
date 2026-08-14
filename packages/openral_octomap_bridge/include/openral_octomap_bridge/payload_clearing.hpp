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

/// Place one wire object's primitives into the grid frame and append them to
/// `out`.
///
/// `grid_from_link` maps a point in the object's attach-link frame into the
/// grid frame (i.e. `lookupTransform(base_frame, object.attach_link)`); each
/// primitive lands at `grid_from_link · pose_in_link · pose_in_object`, the
/// same composition the kernel uses to place the payload it checks.
///
/// Fail-closed: an object carrying no primitives, an unrecognised shape tag,
/// too few shape dimensions for the tag, a non-finite or negative dimension, or
/// a degenerate (zero) pose quaternion returns `false` and appends **nothing**
/// — the caller must then clear nothing at all. Refusing to clear only ever
/// leaves occupancy in the map, which is the conservative direction.
bool place_attached_object(const openral_msgs::msg::AttachedCollisionObject& object,
                           const tf2::Transform& grid_from_link,
                           std::vector<PayloadPrimitive>& out);

/// Clear from `grid` every occupied cell the attached payload's own volume
/// explains, in place. Returns the number of cells cleared.
///
/// A cell is cleared when its centre lies no further from a primitive's surface
/// than the cell cube's circumradius (`resolution·√3/2`) plus `padding_m`. The
/// circumradius is the exact discretisation slop: every cell the payload's
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
                                         double padding_m);

}  // namespace openral_octomap_bridge
