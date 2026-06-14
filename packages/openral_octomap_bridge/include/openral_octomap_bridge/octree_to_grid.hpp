// SPDX-License-Identifier: Apache-2.0
// ADR-0030 — testable core of the OctoMap → OccupancyVoxels bridge: rasterize
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
/// For each voxel centre (in the base frame) the point is transformed into the
/// octree frame via `base_to_octomap` and the octree is queried; a cell is
/// marked occupied (1) when the octree node there exists and is occupied.
/// `base_to_octomap` is the transform that maps a base-frame point into the
/// octree frame (i.e. `lookupTransform(octomap_frame, base_frame)`).
openral_msgs::msg::OccupancyVoxels rasterize_octree_to_grid(const octomap::OcTree& tree,
                                                            const tf2::Transform& base_to_octomap,
                                                            const GridSpec& spec,
                                                            const std::string& base_frame);

}  // namespace openral_octomap_bridge
