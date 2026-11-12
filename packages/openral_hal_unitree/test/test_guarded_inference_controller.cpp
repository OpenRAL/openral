// SPDX-License-Identifier: Apache-2.0
// GuardedInferenceController (issue #565) against NVIDIA's real InferenceController
// (isaac_ros_deploy v4.6-0, built from the deps.repos pin) running the real G1 S0 policy:
// the GR00T Decoupled WBC LEAPP bundle `tools/build_gr00t_wbc_leapp.py` builds, through
// Triton's ONNX Runtime backend. Built only where isaac_ros_deploy is installed; the
// policy tests skip unless OPENRAL_G1_LEAPP_CONFIG names the bundle's YAML (and Triton
// finds its backends through TRITON_BACKEND_DIRECTORY).

#include "openral_hal_unitree/guarded_inference_controller.hpp"

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <limits>
#include <memory>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include <controller_interface/controller_interface_params.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <gtest/gtest.h>
#include <hardware_interface/handle.hpp>
#include <hardware_interface/loaned_command_interface.hpp>
#include <hardware_interface/loaned_state_interface.hpp>
#include <pluginlib/class_loader.hpp>

#include <rclcpp/rclcpp.hpp>

namespace {

using isaac_ros_deploy_ros2_control::controllers::InferenceController;
using openral_hal_unitree::GuardedInferenceController;
using CallbackReturn = controller_interface::CallbackReturn;
using Clock = std::chrono::steady_clock;

constexpr char kVelocitySource[] = "command/body/velocity";
constexpr char kTopic[] = "/openral_issue565/cmd_vel";
constexpr double kTimeoutMs = 200.0;

std::string bundle() {
  const char* env = std::getenv("OPENRAL_G1_LEAPP_CONFIG");
  return env != nullptr && std::filesystem::is_regular_file(env) ? env : "";
}

#define REQUIRE_BUNDLE()                                                                           \
  if (bundle().empty()) {                                                                          \
    GTEST_SKIP() << "OPENRAL_G1_LEAPP_CONFIG does not name a GR00T WBC LEAPP yaml "                \
                    "(build it with tools/build_gr00t_wbc_leapp.py)";                              \
  }

/// The policy file's inference_controller parameters (config/g1/policies/), decimation 1.
std::vector<rclcpp::Parameter> policy_params(const std::string& config_path) {
  return {rclcpp::Parameter("config_path", config_path),
          rclcpp::Parameter("decimation", 1),
          rclcpp::Parameter("command_prefix", "safety_controller"),
          rclcpp::Parameter("command_suffix", "_raw"),
          rclcpp::Parameter("topic_input_sources", std::vector<std::string>{kVelocitySource}),
          rclcpp::Parameter(std::string("source_to_topic.") + kVelocitySource, kTopic),
          rclcpp::Parameter(std::string("source_message_type.") + kVelocitySource,
                            "geometry_msgs/msg/Twist"),
          rclcpp::Parameter("topic_input_timeout_ms", kTimeoutMs)};
}

std::vector<rclcpp::Parameter> with(std::vector<rclcpp::Parameter> params,
                                    const rclcpp::Parameter& replacement) {
  for (auto& p : params) {
    if (p.get_name() == replacement.get_name()) {
      p = replacement;
      return params;
    }
  }
  params.push_back(replacement);
  return params;
}

/// One InferenceController (upstream's or the guarded one) on stand-in interfaces: the
/// G1's joint/IMU state and safety_controller's `<joint>/<interface>_raw` references.
template <class Controller>
class Rig {
public:
  /// init + configure; false when either refuses.
  bool configure(const std::vector<rclcpp::Parameter>& params) {
    controller_interface::ControllerInterfaceParams init;
    init.controller_name = "inference_controller";
    init.controller_manager_update_rate = 200;
    init.update_rate = 200;
    init.node_options.parameter_overrides(params);
    if (controller.init(init) != controller_interface::return_type::OK) {
      return false;
    }
    return controller.on_configure(rclcpp_lifecycle::State()) == CallbackReturn::SUCCESS;
  }

  void activate() {
    const auto states = controller.state_interface_configuration().names;
    const auto commands = controller.command_interface_configuration().names;
    ASSERT_FALSE(commands.empty());
    state_values_.assign(states.size(), 0.0);
    command_values_.assign(commands.size(), 0.0);
    std::vector<hardware_interface::LoanedStateInterface> loaned_state;
    std::vector<hardware_interface::LoanedCommandInterface> loaned_command;
    for (size_t i = 0; i < states.size(); ++i) {
      if (states[i] == "imu/orientation.w") {
        state_values_[i] = 1.0;  // upright
      }
      const auto cut = states[i].rfind('/');
      state_handles_.push_back(std::make_shared<hardware_interface::StateInterface>(
          states[i].substr(0, cut), states[i].substr(cut + 1), &state_values_[i]));
      loaned_state.emplace_back(
          std::const_pointer_cast<const hardware_interface::StateInterface>(state_handles_.back()));
    }
    for (size_t i = 0; i < commands.size(); ++i) {
      const auto cut = commands[i].rfind('/');
      command_handles_.push_back(std::make_shared<hardware_interface::CommandInterface>(
          commands[i].substr(0, cut), commands[i].substr(cut + 1), &command_values_[i]));
      loaned_command.emplace_back(command_handles_.back(), [] {});
    }
    controller.assign_interfaces(std::move(loaned_command), std::move(loaned_state));
    ASSERT_EQ(controller.on_activate(rclcpp_lifecycle::State()), CallbackReturn::SUCCESS);
    executor_.add_node(controller.get_node()->get_node_base_interface());
    executor_.add_node(publisher_node_);
  }

  /// Deliver pending messages, then one control tick on the node's clock.
  void tick() {
    executor_.spin_some();
    ASSERT_EQ(
        controller.update(controller.get_node()->now(), rclcpp::Duration::from_seconds(0.005)),
        controller_interface::return_type::OK);
  }

  void drain() {
    const auto start = Clock::now();
    while (Clock::now() - start < std::chrono::milliseconds(200)) {
      executor_.spin_some();
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
  }

  void publish(double vx) {
    geometry_msgs::msg::Twist msg;
    msg.linear.x = vx;
    publisher_->publish(msg);
  }

  bool all_finite() const {
    for (const double v : command_values_) {
      if (!std::isfinite(v)) {
        return false;
      }
    }
    return true;
  }
  bool all_nan() const {
    for (const double v : command_values_) {
      if (!std::isnan(v)) {
        return false;
      }
    }
    return true;
  }

  /// Publish and tick at 200 Hz until the policy writes finite commands; the time it took.
  std::chrono::milliseconds until_finite(std::chrono::milliseconds within) {
    const auto start = Clock::now();
    while (Clock::now() - start < within) {
      publish(0.3);
      tick();
      if (all_finite()) {
        return std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - start);
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    return within;
  }

  /// Stop publishing and tick at 200 Hz until every command is NaN; the time it took.
  std::chrono::milliseconds until_nan(std::chrono::milliseconds within) {
    const auto start = Clock::now();
    while (Clock::now() - start < within) {
      tick();
      if (all_nan()) {
        return std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - start);
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    return within;
  }

  Controller controller;

private:
  std::vector<double> state_values_;
  std::vector<double> command_values_;
  std::vector<hardware_interface::StateInterface::SharedPtr> state_handles_;
  std::vector<hardware_interface::CommandInterface::SharedPtr> command_handles_;
  rclcpp::Node::SharedPtr publisher_node_ = std::make_shared<rclcpp::Node>("issue565_cmd_vel");
  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr publisher_ =
      publisher_node_->create_publisher<geometry_msgs::msg::Twist>(kTopic, 10);
  rclcpp::executors::SingleThreadedExecutor executor_;
};

class InferenceWatchdogTest : public ::testing::Test {
protected:
  static void SetUpTestSuite() { rclcpp::init(0, nullptr); }
  static void TearDownTestSuite() { rclcpp::shutdown(); }
};

TEST(TopicInputTimeoutRefusal, AllowsZeroOrABoundedTimeout) {
  EXPECT_FALSE(openral_hal_unitree::topic_input_timeout_refusal(0.0).has_value());
  EXPECT_FALSE(openral_hal_unitree::topic_input_timeout_refusal(500.0).has_value());
  // 1e-7 ms rounds to 0 ns and 1e13 ms overflows int64 ns: both would disarm it silently.
  for (const double bad :
       {-1.0, 1e-7, 1e13, std::nan(""), std::numeric_limits<double>::infinity()}) {
    EXPECT_TRUE(openral_hal_unitree::topic_input_timeout_refusal(bad).has_value()) << bad;
  }
}

TEST(TopicInputSourcesRefusal, RefusesASourceThePolicyDoesNotReadFromATopic) {
  const std::set<std::string> topic_sources = {kVelocitySource};
  EXPECT_FALSE(openral_hal_unitree::topic_input_sources_refusal({}, topic_sources).has_value());
  EXPECT_FALSE(openral_hal_unitree::topic_input_sources_refusal({kVelocitySource}, topic_sources)
                   .has_value());
  EXPECT_TRUE(openral_hal_unitree::topic_input_sources_refusal(
                  {kVelocitySource, "state/joint/position"}, topic_sources)
                  .has_value());
}

TEST(GuardedInferenceControllerPlugin, LoadsUnderTheNameTheG1ConfigUses) {
  // controller_manager.yaml names this type; a mismatch is a bringup that cannot start.
  pluginlib::ClassLoader<controller_interface::ControllerInterface> loader(
      "controller_interface", "controller_interface::ControllerInterface");
  const auto controller =
      loader.createSharedInstance("openral_hal_unitree/GuardedInferenceController");
  EXPECT_NE(std::dynamic_pointer_cast<GuardedInferenceController>(controller), nullptr);
}

TEST_F(InferenceWatchdogTest, UpstreamV46IgnoresTheTimeoutAndKeepsWalkingOnTheLastCommand) {
  // The hazard (issue #565): v4.6 accepts the policy file's topic_input_timeout_ms without
  // a word and keeps running the policy on the last velocity command forever.
  REQUIRE_BUNDLE();
  Rig<InferenceController> rig;
  ASSERT_TRUE(rig.configure(policy_params(bundle())));
  rig.activate();
  ASSERT_LT(rig.until_finite(std::chrono::seconds(5)), std::chrono::seconds(5));
  const auto start = Clock::now();
  while (Clock::now() - start < std::chrono::milliseconds(static_cast<int>(5 * kTimeoutMs))) {
    rig.tick();
    ASSERT_TRUE(rig.all_finite());
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
  std::cout << "[ MEASURED ] upstream v4.6: policy commands still finite " << 5 * kTimeoutMs
            << " ms after the last velocity command (timeout " << kTimeoutMs << " ms)\n";
}

TEST_F(InferenceWatchdogTest, NeverReceivedFreshStaleAndFreshAgain) {
  REQUIRE_BUNDLE();
  Rig<GuardedInferenceController> rig;
  ASSERT_TRUE(rig.configure(policy_params(bundle())));
  rig.activate();

  // Startup: nothing published yet. Every command is invalid, so safety_controller holds.
  for (int i = 0; i < 20; ++i) {
    rig.tick();
    ASSERT_TRUE(rig.all_nan()) << "tick " << i;
  }

  // Fresh: the policy runs on the velocity command.
  const auto to_fresh = rig.until_finite(std::chrono::seconds(5));
  ASSERT_LT(to_fresh, std::chrono::seconds(5));

  // Stale: the publisher stops. Within the timeout plus a tick, every command is NaN.
  const auto to_stale = rig.until_nan(std::chrono::seconds(2));
  RecordProperty("stale_after_ms", static_cast<int>(to_stale.count()));
  std::cout << "[ MEASURED ] policy commands invalid " << to_stale.count()
            << " ms after the last velocity command (timeout " << kTimeoutMs << " ms)\n";
  EXPECT_GE(to_stale.count(), kTimeoutMs - 10.0);
  EXPECT_LE(to_stale.count(), kTimeoutMs + 100.0);
  // ...and stays NaN while nothing arrives.
  for (int i = 0; i < 20; ++i) {
    rig.tick();
    ASSERT_TRUE(rig.all_nan()) << "tick " << i;
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }

  // Fresh again: the policy resumes.
  EXPECT_LT(rig.until_finite(std::chrono::seconds(2)), std::chrono::seconds(2));
}

TEST_F(InferenceWatchdogTest, AMessageFromBeforeActivationDoesNotCount) {
  REQUIRE_BUNDLE();
  Rig<GuardedInferenceController> rig;
  ASSERT_TRUE(rig.configure(policy_params(bundle())));
  rig.activate();
  ASSERT_LT(rig.until_finite(std::chrono::seconds(5)), std::chrono::seconds(5));
  rig.drain();  // every message published so far is delivered before the re-activation
  ASSERT_EQ(rig.controller.on_deactivate(rclcpp_lifecycle::State()), CallbackReturn::SUCCESS);
  ASSERT_EQ(rig.controller.on_activate(rclcpp_lifecycle::State()), CallbackReturn::SUCCESS);
  rig.tick();
  EXPECT_TRUE(rig.all_nan());
}

TEST_F(InferenceWatchdogTest, RefusesToConfigureAWatchdogItCannotEnforce) {
  // With a readable policy, so only the guard can be what refuses (the base would accept).
  REQUIRE_BUNDLE();
  for (const auto& bad : {rclcpp::Parameter("topic_input_timeout_ms", -1.0),
                          rclcpp::Parameter("topic_input_timeout_ms", std::nan("")),
                          rclcpp::Parameter("topic_input_timeout_ms", 1e-7),  // rounds to 0 ns
                          rclcpp::Parameter("topic_input_timeout_ms", 500),   // int, not ms
                          rclcpp::Parameter("topic_input_timeout_ms", "500")}) {
    Rig<GuardedInferenceController> rig;
    EXPECT_FALSE(rig.configure(with(policy_params(bundle()), bad))) << bad.value_to_string();
  }
  for (const auto& sources : std::vector<std::vector<std::string>>{
           {kVelocitySource, "state/joint/position"},  // v4.6 reads it from hardware
           {"command/body/does_not_exist"}}) {
    Rig<GuardedInferenceController> rig;
    EXPECT_FALSE(rig.configure(
        with(policy_params(bundle()), rclcpp::Parameter("topic_input_sources", sources))))
        << sources.front();
  }
  // The policy file as shipped configures.
  Rig<GuardedInferenceController> rig;
  EXPECT_TRUE(rig.configure(policy_params(bundle())));
}

}  // namespace
