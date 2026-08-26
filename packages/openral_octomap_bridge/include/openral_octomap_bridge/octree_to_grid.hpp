// SPDX-License-Identifier: Apache-2.0
// Testable core of the OctoMap → OccupancyVoxels bridge: lower an octree into a
// dense, lattice-aligned occupancy grid the safety kernel can ingest.

#pragma once

#include <cstdint>
#include <string>

#include <octomap/OcTree.h>
#include <tf2/LinearMath/Transform.h>

#include <openral_msgs/msg/occupancy_voxels.hpp>

namespace openral_octomap_bridge {

/// The local volume to cover, in the robot base frame.
///
/// A BALL, not a box, and that is the point: the published grid's lattice is
/// the octree's, so the grid's own axes turn with the robot relative to
/// `base_frame`. A box specified in base-frame axes would need its rotated
/// bounding box covered, which costs up to 2x the cells at 45 degrees and makes
/// the cell count depend on where the robot happens to be pointing. A ball is
/// invariant: `ceil(2*radius/resolution)^3` cells at every yaw.
///
/// `radius` must cover wherever the kernel's *checked geometry* can reach from
/// `center`, or the kernel is blind to obstacles out there. It is a property of
/// the robot, derived from its manifest, not of the grid.
struct GridSpec {
  double center[3]{0.0, 0.0, 0.0};  ///< centre of the covered ball, in base_frame
  double radius{0.0};               ///< covered ball radius (m)
};

/// Rasterize `tree` into a dense occupancy grid on the octree's OWN lattice.
///
/// One published cell is one octree cell — same size, same phase, same axes —
/// so the grid carries the octree's occupied volume EXACTLY. It neither loses
/// occupancy nor adds any.
///
/// **Why the grid is not aligned to `base_frame`.** It was until 2026-08-25,
/// and a base-aligned lattice cannot represent an octree lattice without
/// dilating it. The two share a resolution but not a phase, and on a mobile
/// base not a yaw either, so a base cell that merely *overlaps* an occupied
/// leaf might be the one holding the surface and must be marked. That closure
/// is the minimum SOUND cover — the previous rule was already optimal for its
/// output format — and it cost 29-35 mm of median extra reach on 25 mm cells,
/// 40 mm worst case, holding 48% of the live start-state E-stops (issue #173).
/// The dilation was a property of the format, so the format is what changed.
///
/// Publishing on the octree's lattice removes the phase and the yaw from the
/// problem instead of chasing them: there is no re-expression left to pay for.
/// It also removes the octree resolution / grid resolution distinction — the
/// grid's resolution IS `tree.getResolution()`, and the caller does not choose
/// it.
///
/// The remaining discretisation is the octree's own and is untouched here:
/// `insertPointCloud` marks the cell *containing* the ray endpoint, so a
/// surface is known only to within one tree resolution. That is the map's, not
/// this function's (see the package README).
///
/// A leaf coarser than the tree resolution — octomap prunes uniform siblings,
/// and a published `/octomap_binary` is pruned — covers exactly
/// `(leaf_size/resolution)^3` cells, marked by integer index arithmetic.
///
/// `base_to_octomap` is the transform that maps a base-frame point into the
/// octree frame (i.e. `lookupTransform(octomap_frame, base_frame)`).
///
/// The returned message's `origin` and `orientation` are the base-frame pose of
/// cell (0,0,0)'s minimum corner; `occupancy` is indexed along the GRID's axes,
/// which are the octree's.
openral_msgs::msg::OccupancyVoxels rasterize_octree_to_grid(const octomap::OcTree& tree,
                                                            const tf2::Transform& base_to_octomap,
                                                            const GridSpec& spec,
                                                            const std::string& base_frame);

}  // namespace openral_octomap_bridge
