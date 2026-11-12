// SPDX-License-Identifier: Apache-2.0
// NVIDIA's isaac_ros_deploy SafetyController with its blend parameters locked down
// for a real Unitree hg humanoid (issue #548, hazard-log Entry 078 / HZ-DDS-7).
//
// Upstream (isaac_ros_deploy f3aa6b89) declares `blend_ratio` and
// `max_blend_ratio_speed` writable and applies a runtime set on the next tick. Any
// participant on the ROS graph can therefore set the blend speed to 10 /s (the
// policy is fully blended in within 0.1 s) or to 0 /s (the blend freezes and the
// operator can no longer blend the policy out). Upstream's on_activate also snaps
// the applied blend to its target with no ramp, so re-activating the chain after an
// e-stop while `blend_ratio` is still 1.0 starts a walking policy fully blended in.
//
// This subclass changes nothing on the hot path. It only:
//   * refuses to configure unless 0 < max_blend_ratio_speed <= kMaxBlendRatioSpeed;
//   * refuses every runtime set of `max_blend_ratio_speed`;
//   * resets `blend_ratio` to 0.0 on activation, so every activation starts blended out
//     and the policy only ever blends in through the rate limit. (Not a refusal: a
//     failed activation inside a strict controller swap keeps the swap's deactivations,
//     which on the hg hardware leaves the joints unclaimed at zero torque.)
// `blend_ratio` stays writable: it is the operator's gantry blend-in path. Who may
// set it is a graph-access question (host confinement today, SROS2 enclaves next).

#ifndef OPENRAL_HAL_UNITREE__GUARDED_SAFETY_CONTROLLER_HPP_
#define OPENRAL_HAL_UNITREE__GUARDED_SAFETY_CONTROLLER_HPP_

#include <atomic>
#include <optional>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>

#include "isaac_ros_deploy_ros2_control/controllers/safety_controller.hpp"

namespace openral_hal_unitree {

/// Fastest blend ramp a configuration may ask for, in blend units per second: the
/// policy takes at least one second to go from blended out to fully blended in.
constexpr double kMaxBlendRatioSpeed = 1.0;

/// Why `max_blend_ratio_speed` is refused at configure, or nullopt when it is allowed.
std::optional<std::string> blend_speed_refusal(double speed);

/// Why a runtime parameter set is refused, or nullopt when every parameter may change.
/// While `activating`, a non-zero `blend_ratio` is refused too: upstream's on_activate
/// copies the target into the applied blend with no ramp, and the parameter service runs
/// on another executor thread.
std::optional<std::string> runtime_set_refusal(const std::vector<rclcpp::Parameter>& parameters,
                                               bool activating);

class GuardedSafetyController
    : public isaac_ros_deploy_ros2_control::controllers::SafetyController {
public:
  controller_interface::CallbackReturn
  on_configure(const rclcpp_lifecycle::State& previous_state) override;
  controller_interface::CallbackReturn
  on_activate(const rclcpp_lifecycle::State& previous_state) override;

private:
  rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr guard_handle_;
  std::atomic<bool> activating_{false};
};

}  // namespace openral_hal_unitree

#endif  // OPENRAL_HAL_UNITREE__GUARDED_SAFETY_CONTROLLER_HPP_
