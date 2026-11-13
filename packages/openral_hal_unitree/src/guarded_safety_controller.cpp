// SPDX-License-Identifier: Apache-2.0
// See guarded_safety_controller.hpp for what is guarded and why.

#include "openral_hal_unitree/guarded_safety_controller.hpp"

#include <cmath>
#include <sstream>

#include <rcl_interfaces/msg/set_parameters_result.hpp>

#include <rclcpp/rclcpp.hpp>

#include "pluginlib/class_list_macros.hpp"

namespace isaac_ros_deploy_ros2_control::controllers {

controller_interface::return_type
SafetyControllerTestAccess::update_holding(SafetyController& c, const rclcpp::Time& time,
                                           const rclcpp::Duration& period,
                                           std::vector<double>& hold, bool& hold_active) {
  // Upstream v4.6-0 SafetyController::update_and_write_commands, line for line, plus the
  // two input overrides v5.0 applies while its stale-position hold is active.
  if (!c.safety_controller_) {
    return controller_interface::return_type::ERROR;
  }
  const int64_t timestamp_ns = time.nanoseconds();
  c.populate_inputs_from_interfaces(period, timestamp_ns);

  const size_t joints = c.joint_names_.size();
  if (!hold_active) {
    const auto measured = c.inputs_[1].tensor.accessor<float, 2>();
    for (size_t i = 0; i < joints; ++i) {
      hold[i] = static_cast<double>(measured[0][i]);
    }
    hold_active = true;
    RCLCPP_WARN(c.get_node()->get_logger(),
                "Position reference became invalid; holding measured positions captured at "
                "stale entry");
  }
  auto command = c.inputs_[0].tensor.accessor<float, 2>();
  for (size_t i = 0; i < joints; ++i) {
    command[0][i] = static_cast<float>(hold[i]);
  }
  c.inputs_[2].tensor.accessor<float, 2>()[0][0] = 1.0F;

  auto result = c.safety_controller_->advance(timestamp_ns, c.inputs_, c.outputs_);
  if (!result.has_value()) {
    const auto error_message = std::string(result.error().message);
    if (c.velocity_threshold_enabled_ &&
        result.error().code == isaac_deploy_core::Error::Code::kFailedPrecondition) {
      c.latch_emergency(error_message);
      return controller_interface::return_type::OK;
    }
    RCLCPP_ERROR(c.get_node()->get_logger(), "Safety controller advance failed: %s",
                 error_message.c_str());
    return controller_interface::return_type::ERROR;
  }
  c.write_outputs_to_interfaces();
  return controller_interface::return_type::OK;
}

}  // namespace isaac_ros_deploy_ros2_control::controllers

namespace openral_hal_unitree {

namespace {
constexpr char kBlendRatioParam[] = "blend_ratio";
constexpr char kMaxBlendRatioSpeedParam[] = "max_blend_ratio_speed";
// Upstream's reference layout per joint: position, velocity, effort, kp, kd.
constexpr size_t kReferencesPerJoint = 5;
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

std::optional<std::string> runtime_set_refusal(const std::vector<rclcpp::Parameter>& parameters,
                                               bool activating) {
  for (const auto& param : parameters) {
    if (param.get_name() == kMaxBlendRatioSpeedParam) {
      return std::string(kMaxBlendRatioSpeedParam) +
             " is fixed at configure on a real robot; edit the controller config and "
             "relaunch";
    }
    if (activating && param.get_name() == kBlendRatioParam &&
        param.get_type() == rclcpp::ParameterType::PARAMETER_DOUBLE && param.as_double() != 0.0) {
      return std::string(kBlendRatioParam) +
             " cannot be raised while the controller activates; set it again once active";
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
  guard_handle_ = node->add_on_set_parameters_callback(
      [this](const std::vector<rclcpp::Parameter>& parameters) {
        rcl_interfaces::msg::SetParametersResult out;
        const auto refusal = runtime_set_refusal(parameters, activating_.load());
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
  // From here until upstream has copied the target into the applied blend, the guard refuses
  // any non-zero blend_ratio, so a set racing in from the executor cannot be snapped in.
  activating_.store(true);
  const auto node = get_node();
  const double blend_ratio = node->get_parameter(kBlendRatioParam).as_double();
  if (blend_ratio != 0.0) {
    const auto reset = node->set_parameter(rclcpp::Parameter(kBlendRatioParam, 0.0));
    if (!reset.successful) {
      // Unreachable with the in-tree callbacks (upstream and ours both accept 0.0).
      RCLCPP_ERROR(node->get_logger(), "Refusing to activate: could not reset %s to 0.0: %s",
                   kBlendRatioParam, reset.reason.c_str());
      activating_.store(false);
      return controller_interface::CallbackReturn::ERROR;
    }
    RCLCPP_WARN(node->get_logger(),
                "%s was %g at activation; reset to 0.0 so the policy starts blended out. "
                "Raise it again to blend the policy in at max_blend_ratio_speed",
                kBlendRatioParam, blend_ratio);
  }
  const auto result = SafetyController::on_activate(previous_state);
  activating_.store(false);
  hold_.assign(reference_interfaces_.size() / kReferencesPerJoint, 0.0);
  hold_active_ = false;
  return result;
}

controller_interface::return_type
GuardedSafetyController::update_and_write_commands(const rclcpp::Time& time,
                                                   const rclcpp::Duration& period) {
  bool invalid = false;
  for (size_t i = 0; i < hold_.size(); ++i) {
    if (std::isnan(reference_interfaces_[i * kReferencesPerJoint])) {
      invalid = true;
      break;
    }
  }
  if (invalid) {
    return isaac_ros_deploy_ros2_control::controllers::SafetyControllerTestAccess::update_holding(
        *this, time, period, hold_, hold_active_);
  }
  if (hold_active_) {
    hold_active_ = false;
    RCLCPP_INFO(get_node()->get_logger(), "Position reference recovered; resuming normal blending");
  }
  return SafetyController::update_and_write_commands(time, period);
}

}  // namespace openral_hal_unitree

PLUGINLIB_EXPORT_CLASS(openral_hal_unitree::GuardedSafetyController,
                       controller_interface::ChainableControllerInterface)
