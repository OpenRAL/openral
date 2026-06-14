// SPDX-License-Identifier: Apache-2.0
// ADR-0020 — safety_kernel_node main(). Single-threaded executor; the
// validator must complete inside the chunk-deadline so multi-threaded
// dispatch would only add latency.

#include <memory>

#include <rclcpp/rclcpp.hpp>

#include "openral_safety_kernel/lifecycle_kernel.hpp"

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::executors::SingleThreadedExecutor executor;
  auto node = std::make_shared<openral_safety_kernel::SafetyKernelLifecycleNode>();
  executor.add_node(node->get_node_base_interface());
  executor.spin();
  rclcpp::shutdown();
  return 0;
}
