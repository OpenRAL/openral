// SPDX-License-Identifier: Apache-2.0
// GuardedSafetyController against NVIDIA's real SafetyController (isaac_ros_deploy,
// built from the deps.repos pin). Built only where that package is installed.

#include "openral_hal_unitree/guarded_safety_controller.hpp"

#include <atomic>
#include <cmath>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include <controller_interface/controller_interface_params.hpp>
#include <gtest/gtest.h>
#include <hardware_interface/handle.hpp>
#include <hardware_interface/loaned_command_interface.hpp>
#include <hardware_interface/loaned_state_interface.hpp>
#include <pluginlib/class_loader.hpp>

#include <rclcpp/rclcpp.hpp>

// Upstream befriends this name for its own tests; reading the applied values is the
// only way to prove a refused set never reached the controller.
namespace isaac_ros_deploy_ros2_control::controllers {
class SafetyControllerTestAccess {
public:
  static double max_blend_ratio_speed(const SafetyController& c) {
    return c.max_blend_ratio_speed_.load();
  }
  static double target_blend_ratio(const SafetyController& c) {
    return c.target_blend_ratio_.load();
  }
  static double applied_blend_ratio(const SafetyController& c) { return c.current_blend_ratio_; }
};
}  // namespace isaac_ros_deploy_ros2_control::controllers

namespace {

using isaac_ros_deploy_ros2_control::controllers::SafetyControllerTestAccess;
using openral_hal_unitree::GuardedSafetyController;
using CallbackReturn = controller_interface::CallbackReturn;

constexpr size_t kInterfacesPerJoint = 5;
constexpr const char* kCommandInterfaces[kInterfacesPerJoint] = {"position", "velocity", "effort",
                                                                 "kp", "kd"};
const std::vector<std::string> kJoints = {"left_shoulder_pitch_joint",
                                          "right_shoulder_pitch_joint"};

TEST(BlendSpeedRefusal, AllowsOnlyAPositiveRampAtMostOnePerSecond) {
  EXPECT_FALSE(openral_hal_unitree::blend_speed_refusal(1.0).has_value());
  EXPECT_FALSE(openral_hal_unitree::blend_speed_refusal(0.25).has_value());
  for (const double bad :
       {0.0, -1.0, 1.0001, 10.0, std::nan(""), std::numeric_limits<double>::infinity()}) {
    EXPECT_TRUE(openral_hal_unitree::blend_speed_refusal(bad).has_value()) << bad;
  }
}

TEST(RuntimeSetRefusal, RefusesTheBlendSpeedEvenInsideABatch) {
  EXPECT_FALSE(
      openral_hal_unitree::runtime_set_refusal({rclcpp::Parameter("blend_ratio", 1.0)}, false)
          .has_value());
  EXPECT_TRUE(
      openral_hal_unitree::runtime_set_refusal(
          {rclcpp::Parameter("blend_ratio", 1.0), rclcpp::Parameter("max_blend_ratio_speed", 0.5)},
          false)
          .has_value());
}

TEST(RuntimeSetRefusal, RefusesRaisingTheBlendWhileActivating) {
  // The window between the guard's reset and upstream's no-ramp copy of the target.
  EXPECT_TRUE(
      openral_hal_unitree::runtime_set_refusal({rclcpp::Parameter("blend_ratio", 1.0)}, true)
          .has_value());
  EXPECT_FALSE(
      openral_hal_unitree::runtime_set_refusal({rclcpp::Parameter("blend_ratio", 0.0)}, true)
          .has_value());
}

class GuardedSafetyControllerTest : public ::testing::Test {
protected:
  static void SetUpTestSuite() { rclcpp::init(0, nullptr); }
  static void TearDownTestSuite() { rclcpp::shutdown(); }

  CallbackReturn init_and_configure(double max_blend_ratio_speed) {
    controller_interface::ControllerInterfaceParams params;
    params.controller_name = "safety_controller";
    params.controller_manager_update_rate = 200;
    params.update_rate = 200;
    params.node_options.parameter_overrides(
        {rclcpp::Parameter("joints", kJoints),
         rclcpp::Parameter("max_blend_ratio_speed", max_blend_ratio_speed)});
    EXPECT_EQ(controller_.init(params), controller_interface::return_type::OK);
    return controller_.on_configure(rclcpp_lifecycle::State());
  }

  bool set(const std::string& name, double value) {
    return controller_.get_node()->set_parameter(rclcpp::Parameter(name, value)).successful;
  }

  void assign_interfaces() {
    state_values_.assign(kJoints.size(), 0.1);
    command_values_.assign(kJoints.size() * kInterfacesPerJoint, 0.0);
    std::vector<hardware_interface::LoanedStateInterface> loaned_state;
    std::vector<hardware_interface::LoanedCommandInterface> loaned_command;
    for (size_t i = 0; i < kJoints.size(); ++i) {
      state_interfaces_.push_back(std::make_shared<hardware_interface::StateInterface>(
          kJoints[i], "position", &state_values_[i]));
      loaned_state.emplace_back(std::const_pointer_cast<const hardware_interface::StateInterface>(
          state_interfaces_.back()));
      for (size_t j = 0; j < kInterfacesPerJoint; ++j) {
        command_interfaces_.push_back(std::make_shared<hardware_interface::CommandInterface>(
            kJoints[i], kCommandInterfaces[j], &command_values_[i * kInterfacesPerJoint + j]));
        loaned_command.emplace_back(command_interfaces_.back(), [] {});
      }
    }
    controller_.assign_interfaces(std::move(loaned_command), std::move(loaned_state));
  }

  GuardedSafetyController controller_;
  std::vector<double> state_values_;
  std::vector<double> command_values_;
  std::vector<hardware_interface::StateInterface::SharedPtr> state_interfaces_;
  std::vector<hardware_interface::CommandInterface::SharedPtr> command_interfaces_;
};

TEST(GuardedSafetyControllerPlugin, LoadsUnderTheNameTheG1ConfigUses) {
  // controller_manager.yaml names this type; a mismatch is a bringup that cannot start.
  pluginlib::ClassLoader<controller_interface::ChainableControllerInterface> loader(
      "controller_interface", "controller_interface::ChainableControllerInterface");
  const auto controller =
      loader.createSharedInstance("openral_hal_unitree/GuardedSafetyController");
  EXPECT_NE(std::dynamic_pointer_cast<GuardedSafetyController>(controller), nullptr);
}

TEST_F(GuardedSafetyControllerTest, RefusesToConfigureWithAFastBlend) {
  EXPECT_EQ(init_and_configure(10.0), CallbackReturn::ERROR);
}

TEST_F(GuardedSafetyControllerTest, RefusesToConfigureWithAFrozenBlend) {
  EXPECT_EQ(init_and_configure(0.0), CallbackReturn::ERROR);
}

TEST_F(GuardedSafetyControllerTest, ARuntimeBlendSpeedSetNeverReachesTheController) {
  ASSERT_EQ(init_and_configure(1.0), CallbackReturn::SUCCESS);
  for (const double speed : {10.0, 0.0, 0.5}) {
    EXPECT_FALSE(set("max_blend_ratio_speed", speed)) << speed;
    EXPECT_EQ(SafetyControllerTestAccess::max_blend_ratio_speed(controller_), 1.0) << speed;
  }
  EXPECT_EQ(controller_.get_node()->get_parameter("max_blend_ratio_speed").as_double(), 1.0);
}

TEST_F(GuardedSafetyControllerTest, TheGuardSurvivesAReconfigure) {
  ASSERT_EQ(init_and_configure(1.0), CallbackReturn::SUCCESS);
  // Upstream re-registers its own callback on every configure; ours must still run first.
  ASSERT_EQ(controller_.on_configure(rclcpp_lifecycle::State()), CallbackReturn::SUCCESS);
  EXPECT_FALSE(set("max_blend_ratio_speed", 10.0));
  EXPECT_EQ(SafetyControllerTestAccess::max_blend_ratio_speed(controller_), 1.0);
}

TEST_F(GuardedSafetyControllerTest, TheOperatorBlendPathStillWorks) {
  ASSERT_EQ(init_and_configure(1.0), CallbackReturn::SUCCESS);
  EXPECT_TRUE(set("blend_ratio", 0.5));
  EXPECT_EQ(SafetyControllerTestAccess::target_blend_ratio(controller_), 0.5);
  // Upstream's own range check still applies.
  EXPECT_FALSE(set("blend_ratio", 1.5));
  EXPECT_EQ(SafetyControllerTestAccess::target_blend_ratio(controller_), 0.5);
}

TEST_F(GuardedSafetyControllerTest, ActivationStartsBlendedOutEvenIfTheTargetWasRaised) {
  ASSERT_EQ(init_and_configure(1.0), CallbackReturn::SUCCESS);
  assign_interfaces();
  // E.g. re-arming after an e-stop with the policy still blended in. Upstream would
  // apply 1.0 on the first tick; the guard must succeed (a refusal drops the swap's
  // freeze) with the policy blended out.
  ASSERT_TRUE(set("blend_ratio", 1.0));
  EXPECT_EQ(controller_.on_activate(rclcpp_lifecycle::State()), CallbackReturn::SUCCESS);
  EXPECT_EQ(SafetyControllerTestAccess::applied_blend_ratio(controller_), 0.0);
  EXPECT_EQ(SafetyControllerTestAccess::target_blend_ratio(controller_), 0.0);
  EXPECT_EQ(controller_.get_node()->get_parameter("blend_ratio").as_double(), 0.0);
  // The operator can blend in again once active (the activation window is closed); it
  // goes through the ramp.
  EXPECT_TRUE(set("blend_ratio", 1.0));
  EXPECT_EQ(SafetyControllerTestAccess::target_blend_ratio(controller_), 1.0);
  EXPECT_EQ(SafetyControllerTestAccess::applied_blend_ratio(controller_), 0.0);
}

TEST_F(GuardedSafetyControllerTest, ActivatesBlendedOut) {
  ASSERT_EQ(init_and_configure(1.0), CallbackReturn::SUCCESS);
  assign_interfaces();
  EXPECT_EQ(controller_.on_activate(rclcpp_lifecycle::State()), CallbackReturn::SUCCESS);
  EXPECT_EQ(SafetyControllerTestAccess::applied_blend_ratio(controller_), 0.0);
}

}  // namespace
