// SPDX-License-Identifier: Apache-2.0
// Testable core of the OctoMap → OccupancyVoxels bridge: rasterize
// an octree into a dense, base-frame occupancy grid the safety kernel can
// ingest.

#pragma once

#include <cstdint>
#include <string>

#include <octomap/OcTree.h>
#include <tf2/LinearMath/Transform.h>

#include <openral_msgs/msg/occupancy_voxels.hpp>

namespace openral_octomap_bridge {

/// The local volume to rasterize, in the robot base frame.
struct GridSpec {
  double box_min[3]{0.0, 0.0, 0.0};  ///< min corner of voxel (0,0,0)
  double resolution{0.05};           ///< voxel edge length (m)
  std::uint32_t sx{0};               ///< grid dimensions (cells)
  std::uint32_t sy{0};
  std::uint32_t sz{0};
};

/// Rasterize `tree` into a dense base-frame occupancy grid.
///
/// A cell is marked occupied (1) when its **cube overlaps** the cube of an
/// occupied octree leaf — not when its centre point-queries occupied. The two
/// lattices are independent (the octree's lives in the map/odom frame, the
/// grid's in `base_frame`), so a centre sample quantizes every surface onto
/// whichever lattice the centre landed on: at a half-cell relative phase the
/// outermost occupied column lands a full voxel toward the robot and the column
/// the surface is actually in is left empty. Overlap has no phase: every cell
/// the occupied volume enters is marked, for every phase and every relative
/// yaw. Touching cubes share no volume and do not mark each other, so a
/// phase-aligned grid rasterizes one-for-one as before.
///
/// Occupancy can only grow relative to the centre-sampling rule: a centre
/// inside a leaf is a cube overlapping that leaf.
///
/// Overlap is against the octree's *leaves*, whose edge length is the tree
/// resolution or coarser wherever octomap pruned uniform siblings — a coarse
/// leaf marks every base cell under it.
///
/// `base_to_octomap` is the transform that maps a base-frame point into the
/// octree frame (i.e. `lookupTransform(octomap_frame, base_frame)`).
///
/// The grid inherits octomap's own ≤1-resolution inflation toward the sensor:
/// `insertPointCloud` marks the cell *containing* the ray endpoint, which
/// already reaches up to one tree resolution in front of the true surface. That
/// is the map's discretisation, not this function's, and it is not corrected
/// here (see the package README).
openral_msgs::msg::OccupancyVoxels rasterize_octree_to_grid(const octomap::OcTree& tree,
                                                            const tf2::Transform& base_to_octomap,
                                                            const GridSpec& spec,
                                                            const std::string& base_frame);

}  // namespace openral_octomap_bridge
