// SPDX-License-Identifier: Apache-2.0
// NVIDIA's isaac_ros_deploy InferenceController with the stale-input watchdog its v4.6-0
// release lacks (issue #565).
//
// isaac_ros_deploy v4.6-0 (the newest revision that builds on ROS 2 Jazzy) has no
// `topic_input_timeout_ms` / `topic_input_sources`: they arrived in v5.0-0, which needs
// ROS 2 Lyrical. A G1 policy file that sets `topic_input_timeout_ms: 500.0` is silently
// ignored by v4.6, so a crashed or stalled OpenRAL side leaves the robot walking on its
// last velocity command forever.
//
// This subclass restores v5.0's behaviour on top of v4.6 and changes nothing else:
//   * declares `topic_input_timeout_ms` (double, ms, default 0 = no watchdog) and
//     `topic_input_sources` (string array), and refuses to configure on a negative,
//     non-finite or non-double timeout, or on a listed source that v4.6 does not read
//     from a topic (v5.0 would subscribe to it; v4.6 reads it from hardware or not at all);
//   * with a timeout, watches every topic input of the policy (as v5.0 does) through its
//     own subscription. A topic that has not delivered a message since activation, or not
//     within the timeout, invalidates every command interface of the policy (NaN) and skips
//     the policy step. GuardedSafetyController then holds the measured pose captured at the
//     first invalid tick, until the policy writes finite commands again.
// Freshness is measured on the controller manager's update clock (a message counter is
// sampled each tick), so a node clock that differs from it cannot make a stale input
// look fresh.

#ifndef OPENRAL_HAL_UNITREE__GUARDED_INFERENCE_CONTROLLER_HPP_
#define OPENRAL_HAL_UNITREE__GUARDED_INFERENCE_CONTROLLER_HPP_

#include <atomic>
#include <cstdint>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>

// Upstream forward-declares its adapters and leaves the destructor implicit, so every
// translation unit that destroys an InferenceController needs them complete.
#include "isaac_ros_deploy_ros2_control/adapters/command_interface_adapter.hpp"
#include "isaac_ros_deploy_ros2_control/adapters/state_interface_adapter.hpp"
#include "isaac_ros_deploy_ros2_control/controllers/inference_controller.hpp"

namespace openral_hal_unitree {

/// A non-zero `topic_input_timeout_ms` must lie in [kMin, kMax] ms: below, it rounds to
/// 0 ns (no watchdog); far above, the ns conversion overflows.
constexpr double kMinTopicInputTimeoutMs = 1.0;
constexpr double kMaxTopicInputTimeoutMs = 3.6e6;  // one hour

/// Why a `topic_input_timeout_ms` value is refused, or nullopt when it is allowed.
std::optional<std::string> topic_input_timeout_refusal(double timeout_ms);

/// Why `topic_input_sources` is refused, or nullopt when every listed source is one the
/// policy reads from a topic (`topic_sources`).
std::optional<std::string> topic_input_sources_refusal(const std::vector<std::string>& listed,
                                                       const std::set<std::string>& topic_sources);

class GuardedInferenceController
    : public isaac_ros_deploy_ros2_control::controllers::InferenceController {
public:
  controller_interface::CallbackReturn on_init() override;
  controller_interface::CallbackReturn
  on_configure(const rclcpp_lifecycle::State& previous_state) override;
  controller_interface::CallbackReturn
  on_activate(const rclcpp_lifecycle::State& previous_state) override;
  controller_interface::return_type update(const rclcpp::Time& time,
                                           const rclcpp::Duration& period) override;

private:
  struct WatchedTopic {
    std::string topic;
    rclcpp::GenericSubscription::SharedPtr subscription;
    // Bumped by the subscription (executor thread), which co-owns it.
    std::shared_ptr<std::atomic<uint64_t>> received = std::make_shared<std::atomic<uint64_t>>(0);
    uint64_t seen{0};                    // update thread only
    std::optional<int64_t> fresh_at_ns;  // update thread only: last tick a new message was seen
  };

  bool topic_inputs_are_fresh(int64_t now_ns);

  int64_t timeout_ns_{0};
  std::vector<std::unique_ptr<WatchedTopic>> watched_;
};

}  // namespace openral_hal_unitree

#endif  // OPENRAL_HAL_UNITREE__GUARDED_INFERENCE_CONTROLLER_HPP_
