// SPDX-License-Identifier: Apache-2.0
// OctoMap → OccupancyVoxels bridge node entry point; the node is in
// `octomap_voxel_bridge.hpp`.

#include <memory>

#include <rclcpp/rclcpp.hpp>

#include "octomap_voxel_bridge.hpp"

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<openral_octomap_bridge::OctomapVoxelBridge>());
  rclcpp::shutdown();
  return 0;
}
