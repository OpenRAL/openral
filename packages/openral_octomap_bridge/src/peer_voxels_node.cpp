// SPDX-License-Identifier: Apache-2.0
// Entry point for the peer producer; see peer_voxels.hpp.

#include <memory>

#include <rclcpp/rclcpp.hpp>

#include "peer_voxels.hpp"

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<openral_octomap_bridge::PeerVoxels>());
  rclcpp::shutdown();
  return 0;
}
