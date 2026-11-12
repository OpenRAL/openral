// SPDX-License-Identifier: Apache-2.0
// See guarded_safety_controller.hpp for what is guarded and why.

#include "openral_hal_unitree/guarded_safety_controller.hpp"

#include <cmath>
#include <sstream>

#include <rcl_interfaces/msg/set_parameters_result.hpp>

#include <rclcpp/rclcpp.hpp>

#include "pluginlib/class_list_macros.hpp"

namespace openral_hal_unitree {

namespace {
constexpr char kBlendRatioParam[] = "blend_ratio";
constexpr char kMaxBlendRatioSpeedParam[] = "max_blend_ratio_speed";
}  // namespace

std::optional<std::string> blend_speed_refusal(double speed) {
  if (std::isfinite(speed) && speed > 0.0 && speed <= kMaxBlendRatioSpeed) {
    return std::nullopt;
  }
  std::ostringstream reason;
  reason << "max_blend_ratio_speed is " << speed << "; it must be in (0, " << kMaxBlendRatioSpeed
         << "] /s: faster blends the policy in too quickly, 0 freezes the blend "
            "so it can no longer be blended out";
  return reason.str();
}

std::optional<std::string> runtime_set_refusal(const std::vector<rclcpp::Parameter>& parameters) {
  for (const auto& param : parameters) {
    if (param.get_name() == kMaxBlendRatioSpeedParam) {
      return std::string(kMaxBlendRatioSpeedParam) +
             " is fixed at configure on a real robot; edit the controller config and "
             "relaunch";
    }
  }
  return std::nullopt;
}

controller_interface::CallbackReturn
GuardedSafetyController::on_configure(const rclcpp_lifecycle::State& previous_state) {
  const auto node = get_node();
  if (const auto refusal =
          blend_speed_refusal(node->get_parameter(kMaxBlendRatioSpeedParam).as_double())) {
    RCLCPP_ERROR(node->get_logger(), "Refusing to configure: %s", refusal->c_str());
    return controller_interface::CallbackReturn::ERROR;
  }
  // Drop our guard before upstream re-registers its own callback, then add it back
  // after. rclcpp runs the most recently added callback first and stops at the first
  // refusal; upstream stores the new value inside its callback, so ours must run
  // before it or a refused set would already be applied.
  guard_handle_.reset();
  const auto result = SafetyController::on_configure(previous_state);
  if (result != controller_interface::CallbackReturn::SUCCESS) {
    return result;
  }
  guard_handle_ =
      node->add_on_set_parameters_callback([](const std::vector<rclcpp::Parameter>& parameters) {
        rcl_interfaces::msg::SetParametersResult out;
        const auto refusal = runtime_set_refusal(parameters);
        out.successful = !refusal.has_value();
        out.reason = refusal.value_or("");
        return out;
      });
  return result;
}

controller_interface::CallbackReturn
GuardedSafetyController::on_activate(const rclcpp_lifecycle::State& previous_state) {
  // Upstream's on_activate applies the target blend at once, with no ramp. Reset the target
  // to 0.0 first, through the parameter service so upstream's own callback stores it and
  // `ros2 param get` shows it. Refusing to activate instead is NOT safe: a strict
  // switch_controller that fails an activation keeps its deactivations, so re-arming from
  // freeze_controller would leave every joint unclaimed — zero torque on the hg hardware.
  const auto node = get_node();
  const double blend_ratio = node->get_parameter(kBlendRatioParam).as_double();
  if (blend_ratio != 0.0) {
    const auto reset = node->set_parameter(rclcpp::Parameter(kBlendRatioParam, 0.0));
    if (!reset.successful) {
      // Unreachable with the in-tree callbacks (upstream and ours both accept 0.0).
      RCLCPP_ERROR(node->get_logger(), "Refusing to activate: could not reset %s to 0.0: %s",
                   kBlendRatioParam, reset.reason.c_str());
      return controller_interface::CallbackReturn::ERROR;
    }
    RCLCPP_WARN(node->get_logger(),
                "%s was %g at activation; reset to 0.0 so the policy starts blended out. "
                "Raise it again to blend the policy in at max_blend_ratio_speed",
                kBlendRatioParam, blend_ratio);
  }
  return SafetyController::on_activate(previous_state);
}

}  // namespace openral_hal_unitree

PLUGINLIB_EXPORT_CLASS(openral_hal_unitree::GuardedSafetyController,
                       controller_interface::ChainableControllerInterface)
