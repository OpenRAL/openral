// SPDX-License-Identifier: Apache-2.0
// ADR-0030 — OctoMap → dense base-frame occupancy grid (testable core).

#include "openral_octomap_bridge/octree_to_grid.hpp"

#include <cstddef>

#include <tf2/LinearMath/Vector3.h>

namespace openral_octomap_bridge {

openral_msgs::msg::OccupancyVoxels rasterize_octree_to_grid(const octomap::OcTree& tree,
                                                            const tf2::Transform& base_to_octomap,
                                                            const GridSpec& spec,
                                                            const std::string& base_frame) {
  openral_msgs::msg::OccupancyVoxels msg;
  msg.header.frame_id = base_frame;
  msg.origin.x = spec.box_min[0];
  msg.origin.y = spec.box_min[1];
  msg.origin.z = spec.box_min[2];
  msg.resolution = spec.resolution;
  msg.size_x = spec.sx;
  msg.size_y = spec.sy;
  msg.size_z = spec.sz;
  const std::size_t cells =
      static_cast<std::size_t>(spec.sx) * static_cast<std::size_t>(spec.sy) * spec.sz;
  msg.occupancy.assign(cells, 0);

  for (std::uint32_t iz = 0; iz < spec.sz; ++iz) {
    for (std::uint32_t iy = 0; iy < spec.sy; ++iy) {
      for (std::uint32_t ix = 0; ix < spec.sx; ++ix) {
        // Voxel centre in the base frame → octree frame.
        const double bx = spec.box_min[0] + (ix + 0.5) * spec.resolution;
        const double by = spec.box_min[1] + (iy + 0.5) * spec.resolution;
        const double bz = spec.box_min[2] + (iz + 0.5) * spec.resolution;
        const tf2::Vector3 p = base_to_octomap * tf2::Vector3(bx, by, bz);
        const octomap::OcTreeNode* node = tree.search(p.x(), p.y(), p.z());
        if (node != nullptr && tree.isNodeOccupied(node)) {
          const std::size_t idx =
              static_cast<std::size_t>(ix) +
              static_cast<std::size_t>(spec.sx) *
                  (static_cast<std::size_t>(iy) + static_cast<std::size_t>(spec.sy) * iz);
          msg.occupancy[idx] = 1;
        }
      }
    }
  }
  return msg;
}

}  // namespace openral_octomap_bridge
