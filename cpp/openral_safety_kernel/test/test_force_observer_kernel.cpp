// SPDX-License-Identifier: Apache-2.0
// Issue #373 — the external-torque collision tier inside the real lifecycle
// kernel: configure refusals, a stop on an external torque over the ceiling,
// no stop on gravity-consistent effort, and no stop (or tolerance) from an
// invalid estimate. The robot is a 2-dof pendulum whose effort the test
// computes in closed form, so the only external torque is the one injected.

#include "openral_safety_kernel/lifecycle_kernel.hpp"

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <gtest/gtest.h>
#include <sensor_msgs/msg/joint_state.hpp>

#include <lifecycle_msgs/msg/state.hpp>
#include <openral_msgs/msg/failure_trigger.hpp>
#include <openral_msgs/msg/safety_status.hpp>
#include <rclcpp/rclcpp.hpp>

namespace osk = openral_safety_kernel;
using CallbackReturn = osk::SafetyKernelLifecycleNode::CallbackReturn;

namespace {

constexpr double kM1 = 1.3, kM2 = 0.7, kL1 = 0.4, kLc1 = 0.2, kLc2 = 0.25, kG = 9.81;

/// Static effort that holds the pendulum at (q1, q2) against gravity: both
/// joints about +y, links along +x, so a link's height is -x·sin(angle).
std::vector<double> holding_effort(double q1, double q2) {
  const double c12 = std::cos(q1 + q2);
  return {-kG * ((kM1 * kLc1 + kM2 * kL1) * std::cos(q1) + kM2 * kLc2 * c12),
          -kG * kM2 * kLc2 * c12};
}

class ForceObserverKernelTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Same isolation as test_lifecycle_kernel.cpp (#222): no foreign E-stop.
    setenv("ROS_AUTOMATIC_DISCOVERY_RANGE", "OFF", 1);
    rclcpp::init(0, nullptr);
  }
  void TearDown() override { rclcpp::shutdown(); }

  static std::vector<rclcpp::Parameter> params(std::vector<double> ceiling, bool observer) {
    std::vector<rclcpp::Parameter> p{
        {"n_dof", std::int64_t{2}},
        {"joint_position_min", std::vector<double>{-3.0, -3.0}},
        {"joint_position_max", std::vector<double>{3.0, 3.0}},
        {"joint_velocity_max", std::vector<double>{3.0, 3.0}},
        {"joint_torque_max", std::vector<double>{30.0, 30.0}},
        {"self_collision_enabled", true},
        {"collision_n_links", std::int64_t{3}},
        {"collision_parent", std::vector<std::int64_t>{-1, 0, 1}},
        {"collision_joint_kind", std::vector<std::int64_t>{0, 1, 1}},
        {"collision_dof_index", std::vector<std::int64_t>{-1, 0, 1}},
        {"collision_origin_xyzrpy",
         std::vector<double>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, kL1, 0, 0, 0, 0, 0}},
        {"collision_axis", std::vector<double>{0, 0, 1, 0, 1, 0, 0, 1, 0}},
        {"collision_capsule_link", std::vector<std::int64_t>{1, 2}},
        {"collision_capsule_radius", std::vector<double>{0.03, 0.03}},
        {"collision_capsule_half_length", std::vector<double>{0.1, 0.1}},
        {"collision_capsule_origin_xyzrpy",
         std::vector<double>{0.2, 0, 0, 0, 1.5708, 0, 0.2, 0, 0, 0, 1.5708, 0}},
        {"collision_allowed_pairs", std::vector<std::int64_t>{0, 1, 1, 2, 0, 2}},
        {"collision_link_names", std::vector<std::string>{"base", "upper", "lower"}},
        {"collision_joint_names", std::vector<std::string>{"j1", "j2"}},
        {"joint_external_torque_max", ceiling},
    };
    if (observer) {
      p.emplace_back("force_observer_enabled", true);
      p.emplace_back("force_observer_gain_per_s", 40.0);
      p.emplace_back("force_observer_link_mass", std::vector<double>{0.0, kM1, kM2});
      p.emplace_back("force_observer_link_com",
                     std::vector<double>{0, 0, 0, kLc1, 0, 0, kLc2, 0, 0});
      p.emplace_back("force_observer_link_inertia",
                     std::vector<double>{0, 0, 0, 0, 0, 0, 0.01, 0.011, 0.01, 0, 0, 0, 0.005, 0.006,
                                         0.005, 0, 0, 0});
      for (const char* name : {"force_observer_joint_armature", "force_observer_joint_viscous",
                               "force_observer_joint_coulomb", "force_observer_joint_offset"}) {
        p.emplace_back(name, std::vector<double>{0.0, 0.0, 0.0});
      }
    }
    return p;
  }

  static std::shared_ptr<osk::SafetyKernelLifecycleNode> make(const std::string& name,
                                                              std::vector<rclcpp::Parameter> p) {
    rclcpp::NodeOptions opts;
    opts.parameter_overrides(p);
    return std::make_shared<osk::SafetyKernelLifecycleNode>(name, opts);
  }

  static CallbackReturn configure(osk::SafetyKernelLifecycleNode& node) {
    rclcpp_lifecycle::State unconf(lifecycle_msgs::msg::State::PRIMARY_STATE_UNCONFIGURED, "uc");
    return node.on_configure(unconf);
  }

  struct Observed {
    std::atomic<int> force_triggers{0};
    std::atomic<std::uint8_t> status_reason{openral_msgs::msg::SafetyStatus::DROP_NONE};
    std::string evidence;
  };

  /// Activate `node`, then hold the pendulum at (q1, q2) for `seconds` with
  /// the holding effort plus `bias` (effort reads τ_m = g − τ_ext, so a bias
  /// of −b is an external torque of +b). `with_effort=false` publishes an
  /// empty effort array ("not measured").
  static void hold(osk::SafetyKernelLifecycleNode& node, Observed& seen, double q1, double q2,
                   std::vector<double> bias, double seconds, bool with_effort = true) {
    rclcpp_lifecycle::State inactive(lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE, "in");
    ASSERT_EQ(node.on_activate(inactive), CallbackReturn::SUCCESS);
    rclcpp::Node helper("force_helper");
    rclcpp::QoS js_qos(rclcpp::KeepLast(1));
    js_qos.best_effort();
    auto js_pub = helper.create_publisher<sensor_msgs::msg::JointState>("/joint_states", js_qos);
    rclcpp::QoS reliable(rclcpp::KeepLast(10));
    reliable.reliable();
    auto failure_sub = helper.create_subscription<openral_msgs::msg::FailureTrigger>(
        "/openral/failure/safety", reliable,
        [&seen](const openral_msgs::msg::FailureTrigger::SharedPtr m) {
          if (m->kind == openral_msgs::msg::FailureTrigger::KIND_FORCE) {
            seen.evidence = m->evidence_json;
            ++seen.force_triggers;
          }
        });
    rclcpp::QoS status_qos(rclcpp::KeepLast(1));
    status_qos.reliable().transient_local();
    auto status_sub = helper.create_subscription<openral_msgs::msg::SafetyStatus>(
        "/openral/safety_status", status_qos,
        [&seen](const openral_msgs::msg::SafetyStatus::SharedPtr m) {
          seen.status_reason = m->drop_reason;
        });
    rclcpp::executors::SingleThreadedExecutor exec;
    exec.add_node(node.get_node_base_interface());
    exec.add_node(helper.get_node_base_interface());

    const auto effort = holding_effort(q1, q2);
    sensor_msgs::msg::JointState js;
    js.name = {"j1", "j2"};
    js.position = {q1, q2};
    js.velocity = {0.0, 0.0};
    if (with_effort) {
      js.effort = {effort[0] + bias[0], effort[1] + bias[1]};
    }
    const auto end = std::chrono::steady_clock::now() + std::chrono::duration<double>(seconds);
    while (std::chrono::steady_clock::now() < end) {
      js.header.stamp = helper.now();
      js_pub->publish(js);
      exec.spin_some(std::chrono::milliseconds(2));
      std::this_thread::sleep_for(std::chrono::milliseconds(4));
    }
    exec.spin_some(std::chrono::milliseconds(20));
  }
};

}  // namespace

TEST_F(ForceObserverKernelTest, ACeilingWithNoObserverToEnforceItRefusesConfigure) {
  auto node = make("force_no_model", params({0.0, 2.0}, /*observer=*/false));
  EXPECT_EQ(configure(*node), CallbackReturn::FAILURE);
}

TEST_F(ForceObserverKernelTest, AMalformedModelRefusesConfigure) {
  auto p = params({0.0, 2.0}, true);
  for (auto& param : p) {
    if (param.get_name() == "force_observer_link_mass") {
      param = rclcpp::Parameter("force_observer_link_mass", std::vector<double>{0.0, -1.0, kM2});
    }
  }
  EXPECT_EQ(configure(*make("force_bad_mass", p)), CallbackReturn::FAILURE);
  EXPECT_EQ(configure(*make("force_bad_len", params({1.0}, true))), CallbackReturn::FAILURE);
  EXPECT_EQ(configure(*make("force_ok", params({0.0, 2.0}, true))), CallbackReturn::SUCCESS);
}

TEST_F(ForceObserverKernelTest, GravityConsistentEffortDoesNotStop) {
  auto node = make("force_quiet", params({2.0, 2.0}, true));
  ASSERT_EQ(configure(*node), CallbackReturn::SUCCESS);
  Observed seen;
  hold(*node, seen, 0.3, -0.5, {0.0, 0.0}, 0.5);
  EXPECT_FALSE(node->fault_latched());
  EXPECT_EQ(seen.force_triggers.load(), 0);
}

TEST_F(ForceObserverKernelTest, AnExternalTorqueOverTheCeilingStopsAndSaysWhich) {
  auto node = make("force_trip", params({0.0, 2.0}, true));
  ASSERT_EQ(configure(*node), CallbackReturn::SUCCESS);
  Observed seen;
  // A 3 N·m push on j2 (ceiling 2 N·m): the estimate passes 2 N·m about
  // ln(3) / K = 27 ms after the first sample, well inside the window.
  hold(*node, seen, 0.3, -0.5, {0.0, -3.0}, 0.5);
  EXPECT_TRUE(node->fault_latched());
  EXPECT_EQ(seen.force_triggers.load(), 1) << "one latch, one FailureTrigger";
  EXPECT_NE(seen.evidence.find(R"("joint_or_ee":"j2")"), std::string::npos) << seen.evidence;
  EXPECT_NE(seen.evidence.find(R"("unit":"Nm")"), std::string::npos) << seen.evidence;
  EXPECT_EQ(seen.status_reason.load(), openral_msgs::msg::SafetyStatus::KIND_FORCE);
}

TEST_F(ForceObserverKernelTest, AnUnmonitoredJointNeverStops) {
  auto node = make("force_unmonitored", params({0.0, 2.0}, true));
  ASSERT_EQ(configure(*node), CallbackReturn::SUCCESS);
  Observed seen;
  hold(*node, seen, 0.3, -0.5, {-3.0, 0.0}, 0.4);  // the push is on j1, ceiling 0
  EXPECT_FALSE(node->fault_latched());
}

TEST_F(ForceObserverKernelTest, AnInvalidEstimateStopsNothing) {
  // Unmeasured effort, and a joint inside its limit margin: either way the
  // estimate is invalid, so the same push that trips above is not judged.
  auto unmeasured = make("force_no_effort", params({0.0, 2.0}, true));
  ASSERT_EQ(configure(*unmeasured), CallbackReturn::SUCCESS);
  Observed seen_a;
  hold(*unmeasured, seen_a, 0.3, -0.5, {0.0, -3.0}, 0.4, /*with_effort=*/false);
  EXPECT_FALSE(unmeasured->fault_latched());
}

TEST_F(ForceObserverKernelTest, AJointAtItsLimitIsNotReadAsContact) {
  auto node = make("force_at_limit", params({0.0, 2.0}, true));
  ASSERT_EQ(configure(*node), CallbackReturn::SUCCESS);
  Observed seen;
  hold(*node, seen, 0.3, 2.995, {0.0, -3.0}, 0.4);  // j2 within 0.02 rad of +3.0
  EXPECT_FALSE(node->fault_latched());
}
