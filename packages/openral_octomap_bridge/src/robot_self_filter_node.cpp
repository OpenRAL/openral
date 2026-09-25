// SPDX-License-Identifier: Apache-2.0
// Entry point for the robot self-filter; see robot_self_filter.hpp.

#include <memory>

#include <rclcpp/rclcpp.hpp>

#include "robot_self_filter.hpp"

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<openral_octomap_bridge::RobotSelfFilter>());
  rclcpp::shutdown();
  return 0;
}
