// SPDX-License-Identifier: Apache-2.0
// safety_kernel_node main(). Single-threaded executor; the
// validator must complete inside the chunk-deadline so multi-threaded
// dispatch would only add latency.

#include "openral_safety_kernel/lifecycle_kernel.hpp"

#include <memory>

#include <rclcpp/rclcpp.hpp>

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::executors::SingleThreadedExecutor executor;
  auto node = std::make_shared<openral_safety_kernel::SafetyKernelLifecycleNode>();
  executor.add_node(node->get_node_base_interface());
  executor.spin();
  // SIGINT/SIGTERM returns here with the node still ACTIVE. Run its shutdown
  // before main() returns: on_cleanup drains the OTel BatchSpanProcessor
  // (shutdown_tracing). Left to exit-time static destruction, that drain
  // exports queued spans while the OTLP HTTP client's static objects are already
  // gone, and the kernel dies with SIGSEGV. Nothing is spinning any more, so
  // no callback can run against the members on_cleanup releases. Safety is
  // unchanged: a stopped kernel publishes nothing, which the HAL's
  // /openral/safety_status staleness abort and the deadman watchdog handle.
  executor.remove_node(node->get_node_base_interface());
  node->on_shutdown(node->get_current_state());
  node.reset();
  rclcpp::shutdown();
  return 0;
}
