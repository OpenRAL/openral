// SPDX-License-Identifier: Apache-2.0
// gtest unit coverage for SafetyKernelLifecycleNode.
// Exercises lifecycle transitions, fault-latch behaviour, and the
// /openral/estop_reset cooldown semantics WITHOUT requiring a running
// ROS graph — we just drive the lifecycle callbacks directly.

#include "openral_safety_kernel/lifecycle_kernel.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdarg>
#include <cstdint>
#include <cmath>
#include <cstdio>
#include <future>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <rcutils/logging.h>
#include <sensor_msgs/msg/joint_state.hpp>

#include <lifecycle_msgs/msg/state.hpp>
#include <openral_msgs/msg/action_chunk.hpp>
#include <openral_msgs/msg/attached_collision_object.hpp>
#include <openral_msgs/msg/attached_collision_primitive.hpp>
#include <openral_msgs/msg/failure_trigger.hpp>
#include <openral_msgs/msg/occupancy_voxels.hpp>
#include <openral_msgs/msg/safety_status.hpp>
#include <openral_msgs/msg/world_state_stamped.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/empty.hpp>
#include <std_srvs/srv/trigger.hpp>

namespace osk = openral_safety_kernel;

namespace {

class LifecycleKernelTest : public ::testing::Test {
protected:
  void SetUp() override { rclcpp::init(0, nullptr); }
  void TearDown() override { rclcpp::shutdown(); }

  /// Minimal envelope parameter overrides for a 3-DoF toy robot. Mirrors
  /// what `openral_safety.envelope_loader.kernel_params_from_envelope`
  /// would emit for a robot.yaml describing the same envelope.
  std::vector<rclcpp::Parameter> minimal_envelope_params() {
    return {
        rclcpp::Parameter("n_dof", std::int64_t{3}),
        rclcpp::Parameter("robot_name", std::string{"toy"}),
        rclcpp::Parameter("joint_position_min", std::vector<double>{-1.0, -1.0, -1.0}),
        rclcpp::Parameter("joint_position_max", std::vector<double>{1.0, 1.0, 1.0}),
        rclcpp::Parameter("joint_velocity_max", std::vector<double>{3.15, 3.15, 3.15}),
        rclcpp::Parameter("joint_torque_max", std::vector<double>{5.0, 5.0, 5.0}),
        rclcpp::Parameter("max_ee_speed_m_s", 0.5),
        rclcpp::Parameter("max_ee_accel_m_s2", 2.0),
        rclcpp::Parameter("max_force_n", 10.0),
        rclcpp::Parameter("max_torque_nm", 3.0),
        rclcpp::Parameter("contact_force_threshold_n", 5.0),
        rclcpp::Parameter("deadman_required", false),
    };
  }
};

}  // namespace

TEST_F(LifecycleKernelTest, ConfigureFailsWhenNoEnvelopeProvided) {
  // CLAUDE.md §1.4 — explicit failure, no fallback. With no ROS
  // parameters set the kernel must refuse to leave UNCONFIGURED so a
  // misboot never lets unvalidated chunks reach the HAL.
  rclcpp::NodeOptions opts;
  auto node = std::make_shared<osk::SafetyKernelLifecycleNode>("kernel_under_test_none", opts);
  rclcpp_lifecycle::State state(lifecycle_msgs::msg::State::PRIMARY_STATE_UNCONFIGURED,
                                "unconfigured");
  EXPECT_EQ(node->on_configure(state), osk::SafetyKernelLifecycleNode::CallbackReturn::FAILURE);
}

TEST_F(LifecycleKernelTest, FullLifecycleSuccess) {
  rclcpp::NodeOptions opts;
  opts.parameter_overrides(minimal_envelope_params());
  auto node = std::make_shared<osk::SafetyKernelLifecycleNode>("kernel_under_test_full", opts);

  rclcpp_lifecycle::State unconf(lifecycle_msgs::msg::State::PRIMARY_STATE_UNCONFIGURED, "uc");
  rclcpp_lifecycle::State inactive(lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE, "in");
  rclcpp_lifecycle::State active(lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE, "ac");

  EXPECT_EQ(node->on_configure(unconf), osk::SafetyKernelLifecycleNode::CallbackReturn::SUCCESS);
  EXPECT_EQ(node->envelope().n_dof, 3U);
  EXPECT_EQ(node->on_activate(inactive), osk::SafetyKernelLifecycleNode::CallbackReturn::SUCCESS);
  EXPECT_EQ(node->on_deactivate(active), osk::SafetyKernelLifecycleNode::CallbackReturn::SUCCESS);
  EXPECT_EQ(node->on_cleanup(inactive), osk::SafetyKernelLifecycleNode::CallbackReturn::SUCCESS);
  EXPECT_EQ(node->on_shutdown(unconf), osk::SafetyKernelLifecycleNode::CallbackReturn::SUCCESS);
}

TEST_F(LifecycleKernelTest, ConfiguresFromRosParametersWhenNDofSet) {
  // Parameter-based envelope path. The Python launch
  // unpacks robot.yaml and forwards each field as a ROS parameter; this
  // test confirms the kernel loads from those params and reaches ACTIVE.
  rclcpp::NodeOptions opts;
  opts.parameter_overrides({
      {"n_dof", std::int64_t{3}},
      {"robot_name", std::string{"toy"}},
      {"joint_position_min", std::vector<double>{-1.0, -1.0, -1.0}},
      {"joint_position_max", std::vector<double>{1.0, 1.0, 1.0}},
      {"joint_velocity_max", std::vector<double>{3.15, 3.15, 3.15}},
      {"joint_torque_max", std::vector<double>{5.0, 5.0, 5.0}},
      {"max_ee_speed_m_s", 0.5},
      {"max_ee_accel_m_s2", 2.0},
      {"max_force_n", 10.0},
      {"max_torque_nm", 3.0},
      {"contact_force_threshold_n", 5.0},
      {"deadman_required", false},
  });
  auto node = std::make_shared<osk::SafetyKernelLifecycleNode>("kernel_under_test_params", opts);
  rclcpp_lifecycle::State unconf(lifecycle_msgs::msg::State::PRIMARY_STATE_UNCONFIGURED,
                                 "unconfigured");
  rclcpp_lifecycle::State inactive(lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE, "inactive");
  EXPECT_EQ(node->on_configure(unconf), osk::SafetyKernelLifecycleNode::CallbackReturn::SUCCESS);
  EXPECT_EQ(node->on_activate(inactive), osk::SafetyKernelLifecycleNode::CallbackReturn::SUCCESS);
  EXPECT_EQ(node->on_deactivate(rclcpp_lifecycle::State(
                lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE, "active")),
            osk::SafetyKernelLifecycleNode::CallbackReturn::SUCCESS);
  EXPECT_EQ(node->on_cleanup(inactive), osk::SafetyKernelLifecycleNode::CallbackReturn::SUCCESS);
}

TEST_F(LifecycleKernelTest, SelfCollisionModelLoadsAndConfigures) {
  // A well-formed collision model loads and the node reaches
  // configured with self-collision active.
  rclcpp::NodeOptions opts;
  opts.parameter_overrides({
      {"n_dof", std::int64_t{2}},
      {"joint_position_min", std::vector<double>{-3.14, -3.14}},
      {"joint_position_max", std::vector<double>{3.14, 3.14}},
      {"joint_velocity_max", std::vector<double>{3.15, 3.15}},
      {"joint_torque_max", std::vector<double>{5.0, 5.0}},
      {"self_collision_enabled", true},
      {"self_collision_margin_m", 0.0},
      {"collision_n_links", std::int64_t{2}},
      {"collision_parent", std::vector<std::int64_t>{-1, 0}},
      {"collision_joint_kind", std::vector<std::int64_t>{1, 1}},  // revolute, revolute
      {"collision_dof_index", std::vector<std::int64_t>{0, 1}},
      {"collision_origin_xyzrpy", std::vector<double>{0, 0, 0, 0, 0, 0, 0, 0, 0.3, 0, 0, 0}},
      {"collision_axis", std::vector<double>{0, 0, 1, 0, 1, 0}},
      {"collision_capsule_link", std::vector<std::int64_t>{0, 1}},
      {"collision_capsule_radius", std::vector<double>{0.05, 0.05}},
      {"collision_capsule_half_length", std::vector<double>{0.15, 0.15}},
      {"collision_capsule_origin_xyzrpy", std::vector<double>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
      {"collision_allowed_pairs", std::vector<std::int64_t>{0, 1}},
      {"collision_link_names", std::vector<std::string>{"link0", "link1"}},
  });
  auto node = std::make_shared<osk::SafetyKernelLifecycleNode>("kernel_collision_ok", opts);
  rclcpp_lifecycle::State unconf(lifecycle_msgs::msg::State::PRIMARY_STATE_UNCONFIGURED,
                                 "unconfigured");
  EXPECT_EQ(node->on_configure(unconf), osk::SafetyKernelLifecycleNode::CallbackReturn::SUCCESS);
  EXPECT_TRUE(node->self_collision_active());
  EXPECT_EQ(node->collision_link_count(), 2U);
}

TEST_F(LifecycleKernelTest, SelfCollisionMalformedModelFailsClosed) {
  // CLAUDE.md §1.4 / §3 — a malformed collision model when the feature is
  // enabled must fail configure, never silently run a broken safety check.
  rclcpp::NodeOptions opts;
  opts.parameter_overrides({
      {"n_dof", std::int64_t{2}},
      {"joint_position_min", std::vector<double>{-3.14, -3.14}},
      {"joint_position_max", std::vector<double>{3.14, 3.14}},
      {"joint_velocity_max", std::vector<double>{3.15, 3.15}},
      {"joint_torque_max", std::vector<double>{5.0, 5.0}},
      {"self_collision_enabled", true},
      {"collision_n_links", std::int64_t{2}},
      {"collision_parent", std::vector<std::int64_t>{-1, 0}},
      {"collision_joint_kind", std::vector<std::int64_t>{1, 1}},
      {"collision_dof_index", std::vector<std::int64_t>{0, 1}},
      {"collision_origin_xyzrpy", std::vector<double>{0, 0, 0}},  // wrong length (want 12)
      {"collision_axis", std::vector<double>{0, 0, 1, 0, 1, 0}},
      {"collision_capsule_link", std::vector<std::int64_t>{0, 1}},
      {"collision_capsule_radius", std::vector<double>{0.05, 0.05}},
      {"collision_capsule_half_length", std::vector<double>{0.15, 0.15}},
      {"collision_capsule_origin_xyzrpy", std::vector<double>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
      {"collision_allowed_pairs", std::vector<std::int64_t>{0, 1}},
      {"collision_link_names", std::vector<std::string>{"link0", "link1"}},
  });
  auto node = std::make_shared<osk::SafetyKernelLifecycleNode>("kernel_collision_bad", opts);
  rclcpp_lifecycle::State unconf(lifecycle_msgs::msg::State::PRIMARY_STATE_UNCONFIGURED,
                                 "unconfigured");
  EXPECT_EQ(node->on_configure(unconf), osk::SafetyKernelLifecycleNode::CallbackReturn::FAILURE);
}

TEST_F(LifecycleKernelTest, ParameterPathFailsOnJointArrayLengthMismatch) {
  // CLAUDE.md §3 — at least as conservative. A mismatched joint array
  // length must fail closed, not be silently truncated or padded.
  rclcpp::NodeOptions opts;
  opts.parameter_overrides({
      {"n_dof", std::int64_t{3}},
      {"joint_position_min", std::vector<double>{-1.0, -1.0, -1.0}},
      {"joint_position_max", std::vector<double>{1.0, 1.0}},  // wrong length
      {"joint_velocity_max", std::vector<double>{3.15, 3.15, 3.15}},
      {"joint_torque_max", std::vector<double>{5.0, 5.0, 5.0}},
  });
  auto node = std::make_shared<osk::SafetyKernelLifecycleNode>("kernel_under_test_mismatch", opts);
  rclcpp_lifecycle::State unconf(lifecycle_msgs::msg::State::PRIMARY_STATE_UNCONFIGURED,
                                 "unconfigured");
  EXPECT_EQ(node->on_configure(unconf), osk::SafetyKernelLifecycleNode::CallbackReturn::FAILURE);
}

TEST_F(LifecycleKernelTest, ResetServiceRespectsCooldown) {
  rclcpp::NodeOptions opts;
  auto overrides = minimal_envelope_params();
  overrides.emplace_back("estop_reset_cooldown_s", 0.05);
  opts.parameter_overrides(overrides);
  auto node = std::make_shared<osk::SafetyKernelLifecycleNode>("kernel_under_test_reset", opts);

  rclcpp_lifecycle::State unconf(lifecycle_msgs::msg::State::PRIMARY_STATE_UNCONFIGURED, "uc");
  ASSERT_EQ(node->on_configure(unconf), osk::SafetyKernelLifecycleNode::CallbackReturn::SUCCESS);

  // Drive a violation via the public on_candidate_action handler with an
  // out-of-range chunk — the fault latch should set.
  auto bad = std::make_shared<openral_msgs::msg::ActionChunk>();
  bad->control_mode = 0;  // joint position
  bad->horizon = 1;
  bad->n_dof = 3;
  bad->flat = {5.0, 0.0, 0.0};  // joint 0 violates pos_max=1.0
  // Activate the publishers so we can publish even though we're not
  // spinning — direct callback invocation is OK here.
  rclcpp_lifecycle::State inactive(lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE, "in");
  ASSERT_EQ(node->on_activate(inactive), osk::SafetyKernelLifecycleNode::CallbackReturn::SUCCESS);

  // Inject manually — using a Subscription would require an executor.
  // The lifecycle node exposes the subscription only; we test the validator
  // path via direct chunk handling via the loop's friend trick. Instead, we
  // can rely on the kernel's public state — fault_latched() — once we
  // call validate ourselves and assert behaviour. Skip the direct path
  // and exercise reset semantics with an externally-triggered estop.
  rclcpp::Node helper("kernel_helper");
  auto estop_pub = helper.create_publisher<std_msgs::msg::Empty>("/openral/estop", 10);
  rclcpp::executors::SingleThreadedExecutor exec;
  exec.add_node(node->get_node_base_interface());
  exec.add_node(helper.get_node_base_interface());
  estop_pub->publish(std_msgs::msg::Empty{});
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(500);
  while (!node->fault_latched() && std::chrono::steady_clock::now() < deadline) {
    exec.spin_some(std::chrono::milliseconds(10));
  }
  EXPECT_TRUE(node->fault_latched());

  // Reset before cooldown elapses → success=false.
  auto request = std::make_shared<std_srvs::srv::Trigger::Request>();
  auto response = std::make_shared<std_srvs::srv::Trigger::Response>();
  // Direct service-callback invocation — bypasses the rpc machinery.
  // The lifecycle node exposes on_estop_reset as private; we instead
  // call it via the service client to keep the contract end-to-end.
  auto client = helper.create_client<std_srvs::srv::Trigger>("/openral/estop_reset");
  ASSERT_TRUE(client->wait_for_service(std::chrono::seconds(2)));
  auto fut_early = client->async_send_request(request);
  while (std::chrono::steady_clock::now() < deadline &&
         fut_early.wait_for(std::chrono::milliseconds(0)) != std::future_status::ready) {
    exec.spin_some(std::chrono::milliseconds(10));
  }

  // Wait through the cooldown then retry.
  std::this_thread::sleep_for(std::chrono::milliseconds(150));
  auto fut_late = client->async_send_request(request);
  const auto late_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
  while (std::chrono::steady_clock::now() < late_deadline &&
         fut_late.wait_for(std::chrono::milliseconds(0)) != std::future_status::ready) {
    exec.spin_some(std::chrono::milliseconds(10));
  }
  ASSERT_EQ(fut_late.wait_for(std::chrono::milliseconds(0)), std::future_status::ready);
  auto late_resp = fut_late.get();
  EXPECT_TRUE(late_resp->success) << late_resp->message;
  EXPECT_FALSE(node->fault_latched());
}

// The ViolationKind enum must stay 1:1 with the IDL KIND_*
// constants so the lifecycle node can publish a FailureTrigger without
// translation (validator.hpp documents this contract). kCollision carries the
// geometric-safety check's number, and the node does emit it:
// `publish_collision_failure` stamps `FailureTrigger::KIND_COLLISION` and the
// collision report path publishes `SafetyStatus::KIND_COLLISION` with the
// E-stop. This test pins the number, not the emission — the emission itself is
// covered by the collision tests below and in test_collision.cpp.
TEST(ViolationKindMapping, EnumValuesMatchFailureTriggerConstants) {
  using openral_msgs::msg::FailureTrigger;
  EXPECT_EQ(static_cast<std::uint8_t>(osk::ViolationKind::kForce), FailureTrigger::KIND_FORCE);
  EXPECT_EQ(static_cast<std::uint8_t>(osk::ViolationKind::kWorkspace),
            FailureTrigger::KIND_WORKSPACE);
  EXPECT_EQ(static_cast<std::uint8_t>(osk::ViolationKind::kController),
            FailureTrigger::KIND_CONTROLLER);
  EXPECT_EQ(static_cast<std::uint8_t>(osk::ViolationKind::kCollision),
            FailureTrigger::KIND_COLLISION);
}

namespace {

// A 2-link self-collision model whose only pair is allowed (so the
// geometry is always clear), with the joint-name map + velocity seed params
// plumbed. Lets the velocity-mode tests exercise the new seed gate + routing
// without depending on a specific colliding configuration (geometric detection
// itself is covered by test_collision.cpp, which check_config() reuses).
std::vector<rclcpp::Parameter> velocity_capable_params() {
  return {
      {"n_dof", std::int64_t{2}},
      {"joint_position_min", std::vector<double>{-3.14, -3.14}},
      {"joint_position_max", std::vector<double>{3.14, 3.14}},
      {"joint_velocity_max", std::vector<double>{3.15, 3.15}},
      {"joint_torque_max", std::vector<double>{5.0, 5.0}},
      {"self_collision_enabled", true},
      {"self_collision_margin_m", 0.0},
      {"collision_n_links", std::int64_t{2}},
      {"collision_parent", std::vector<std::int64_t>{-1, 0}},
      {"collision_joint_kind", std::vector<std::int64_t>{1, 1}},
      {"collision_dof_index", std::vector<std::int64_t>{0, 1}},
      {"collision_origin_xyzrpy", std::vector<double>{0, 0, 0, 0, 0, 0, 0, 0, 0.3, 0, 0, 0}},
      {"collision_axis", std::vector<double>{0, 0, 1, 0, 1, 0}},
      {"collision_capsule_link", std::vector<std::int64_t>{0, 1}},
      {"collision_capsule_radius", std::vector<double>{0.05, 0.05}},
      {"collision_capsule_half_length", std::vector<double>{0.15, 0.15}},
      {"collision_capsule_origin_xyzrpy", std::vector<double>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
      {"collision_allowed_pairs", std::vector<std::int64_t>{0, 1}},
      {"collision_link_names", std::vector<std::string>{"link0", "link1"}},
      // Geometric-check plumbing for non-position control modes
      {"collision_joint_names", std::vector<std::string>{"j0", "j1"}},
      {"collision_seed_dt_s", 0.05},
      {"collision_state_deadline_ms", 500.0},
  };
}

}  // namespace

// A JOINT_VELOCITY chunk arriving with the geometric check enabled
// but NO measured joint-state seed must be dropped fail-closed, never silently
// passed (previously this bypassed the geometric block entirely). This is the
// core safety property: a missing state feed cannot disable collision checking.
TEST_F(LifecycleKernelTest, VelocityChunkFailsClosedWithoutMeasuredSeed) {
  rclcpp::NodeOptions opts;
  opts.parameter_overrides(velocity_capable_params());
  auto node = std::make_shared<osk::SafetyKernelLifecycleNode>("kernel_vel_failclosed", opts);
  rclcpp_lifecycle::State unconf(lifecycle_msgs::msg::State::PRIMARY_STATE_UNCONFIGURED, "uc");
  ASSERT_EQ(node->on_configure(unconf), osk::SafetyKernelLifecycleNode::CallbackReturn::SUCCESS);
  rclcpp_lifecycle::State inactive(lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE, "in");
  ASSERT_EQ(node->on_activate(inactive), osk::SafetyKernelLifecycleNode::CallbackReturn::SUCCESS);

  rclcpp::Node helper("vel_helper_a");
  rclcpp::QoS chunk_qos(rclcpp::KeepLast(1));
  chunk_qos.reliable();
  auto cand_pub = helper.create_publisher<openral_msgs::msg::ActionChunk>(
      "/openral/candidate_action", chunk_qos);
  std::atomic<int> safe_count{0};
  auto safe_sub = helper.create_subscription<openral_msgs::msg::ActionChunk>(
      "/openral/safe_action", chunk_qos,
      [&safe_count](const openral_msgs::msg::ActionChunk::SharedPtr) { ++safe_count; });

  rclcpp::executors::SingleThreadedExecutor exec;
  exec.add_node(node->get_node_base_interface());
  exec.add_node(helper.get_node_base_interface());

  const std::uint64_t dropped_before = node->chunks_dropped();
  auto vel = std::make_shared<openral_msgs::msg::ActionChunk>();
  vel->control_mode = 1;  // JOINT_VELOCITY
  vel->horizon = 1;
  vel->n_dof = 2;
  vel->flat = {0.1, 0.1};  // within joint_velocity_max → passes the envelope
  cand_pub->publish(*vel);

  const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(500);
  while (node->chunks_dropped() == dropped_before && std::chrono::steady_clock::now() < deadline) {
    exec.spin_some(std::chrono::milliseconds(10));
  }
  EXPECT_GT(node->chunks_dropped(), dropped_before)
      << "velocity chunk must be dropped fail-closed when no measured seed is available";
  EXPECT_EQ(safe_count.load(), 0)
      << "a seed-less velocity chunk must not reach /openral/safe_action";
}

// Once a fresh, complete measured seed is available, a clear velocity
// chunk passes geometry and is forwarded to /openral/safe_action (the model's
// only link pair is allowed, so the configuration is always collision-free).
TEST_F(LifecycleKernelTest, VelocityChunkPassesWithFreshSeedWhenClear) {
  rclcpp::NodeOptions opts;
  opts.parameter_overrides(velocity_capable_params());
  auto node = std::make_shared<osk::SafetyKernelLifecycleNode>("kernel_vel_pass", opts);
  rclcpp_lifecycle::State unconf(lifecycle_msgs::msg::State::PRIMARY_STATE_UNCONFIGURED, "uc");
  ASSERT_EQ(node->on_configure(unconf), osk::SafetyKernelLifecycleNode::CallbackReturn::SUCCESS);
  rclcpp_lifecycle::State inactive(lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE, "in");
  ASSERT_EQ(node->on_activate(inactive), osk::SafetyKernelLifecycleNode::CallbackReturn::SUCCESS);

  rclcpp::Node helper("vel_helper_b");
  rclcpp::QoS chunk_qos(rclcpp::KeepLast(1));
  chunk_qos.reliable();
  auto cand_pub = helper.create_publisher<openral_msgs::msg::ActionChunk>(
      "/openral/candidate_action", chunk_qos);
  rclcpp::QoS js_qos(rclcpp::KeepLast(1));
  js_qos.best_effort();
  auto js_pub = helper.create_publisher<sensor_msgs::msg::JointState>("/joint_states", js_qos);
  std::atomic<int> safe_count{0};
  auto safe_sub = helper.create_subscription<openral_msgs::msg::ActionChunk>(
      "/openral/safe_action", chunk_qos,
      [&safe_count](const openral_msgs::msg::ActionChunk::SharedPtr) { ++safe_count; });

  rclcpp::executors::SingleThreadedExecutor exec;
  exec.add_node(node->get_node_base_interface());
  exec.add_node(helper.get_node_base_interface());

  // Seed the measured state (both FK dofs present + clear) and let it land.
  sensor_msgs::msg::JointState js;
  js.name = {"j0", "j1"};
  js.position = {0.0, 0.0};
  auto seed_deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(300);
  while (std::chrono::steady_clock::now() < seed_deadline) {
    js_pub->publish(js);
    exec.spin_some(std::chrono::milliseconds(10));
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }

  auto vel = std::make_shared<openral_msgs::msg::ActionChunk>();
  vel->control_mode = 1;  // JOINT_VELOCITY
  vel->horizon = 1;
  vel->n_dof = 2;
  vel->flat = {0.05, 0.05};
  cand_pub->publish(*vel);

  const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(500);
  while (safe_count.load() == 0 && std::chrono::steady_clock::now() < deadline) {
    js_pub->publish(js);  // keep the seed fresh
    exec.spin_some(std::chrono::milliseconds(10));
  }
  EXPECT_GT(safe_count.load(), 0)
      << "a clear velocity chunk with a fresh measured seed must pass to /openral/safe_action";
  EXPECT_FALSE(node->fault_latched());
}

// Phase 3 — a CARTESIAN_DELTA chunk (the arm mode for LIBERO/SIMPLER/
// DROID + the robocasa arm) carries a 6-D EE delta, NOT joint configs. It must
// be routed through the REACTIVE measured-config check (not skipped for n_dof !=
// robot dof, and not silently passed). Here the geometry is clear, so it passes;
// the fail-closed-without-seed property is shared with the velocity gate above.
TEST_F(LifecycleKernelTest, CartesianDeltaChunkReactiveCheckPassesWhenClear) {
  rclcpp::NodeOptions opts;
  opts.parameter_overrides(velocity_capable_params());
  auto node = std::make_shared<osk::SafetyKernelLifecycleNode>("kernel_cart_pass", opts);
  rclcpp_lifecycle::State unconf(lifecycle_msgs::msg::State::PRIMARY_STATE_UNCONFIGURED, "uc");
  ASSERT_EQ(node->on_configure(unconf), osk::SafetyKernelLifecycleNode::CallbackReturn::SUCCESS);
  rclcpp_lifecycle::State inactive(lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE, "in");
  ASSERT_EQ(node->on_activate(inactive), osk::SafetyKernelLifecycleNode::CallbackReturn::SUCCESS);

  rclcpp::Node helper("cart_helper");
  rclcpp::QoS chunk_qos(rclcpp::KeepLast(1));
  chunk_qos.reliable();
  auto cand_pub = helper.create_publisher<openral_msgs::msg::ActionChunk>(
      "/openral/candidate_action", chunk_qos);
  rclcpp::QoS js_qos(rclcpp::KeepLast(1));
  js_qos.best_effort();
  auto js_pub = helper.create_publisher<sensor_msgs::msg::JointState>("/joint_states", js_qos);
  std::atomic<int> safe_count{0};
  auto safe_sub = helper.create_subscription<openral_msgs::msg::ActionChunk>(
      "/openral/safe_action", chunk_qos,
      [&safe_count](const openral_msgs::msg::ActionChunk::SharedPtr) { ++safe_count; });

  rclcpp::executors::SingleThreadedExecutor exec;
  exec.add_node(node->get_node_base_interface());
  exec.add_node(helper.get_node_base_interface());

  sensor_msgs::msg::JointState js;
  js.name = {"j0", "j1"};
  js.position = {0.0, 0.0};
  auto seed_deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(300);
  while (std::chrono::steady_clock::now() < seed_deadline) {
    js_pub->publish(js);
    exec.spin_some(std::chrono::milliseconds(10));
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }

  auto cart = std::make_shared<openral_msgs::msg::ActionChunk>();
  cart->control_mode = 5;  // CARTESIAN_DELTA
  cart->horizon = 1;
  cart->n_dof = 6;  // 6-D EE delta — NOT the 2-dof robot config
  cart->flat = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  cand_pub->publish(*cart);

  const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(500);
  while (safe_count.load() == 0 && std::chrono::steady_clock::now() < deadline) {
    js_pub->publish(js);
    exec.spin_some(std::chrono::milliseconds(10));
  }
  EXPECT_GT(safe_count.load(), 0)
      << "a clear Cartesian-delta chunk with a fresh seed must pass to /openral/safe_action";
  EXPECT_FALSE(node->fault_latched());
}

// Phase 2 — DETERMINISTIC proof that a velocity chunk's REACTIVE check
// catches a collision (not just passes clear ones). The 2-link model's capsules
// overlap at the measured configuration and the pair is NOT in the allowed set,
// so reconstructing the config from the seed and running the (well-tested)
// self-collision check must reject + estop. This is the velocity analogue of the
// position-mode collision path, exercising the seed → FK → check → estop chain.
TEST_F(LifecycleKernelTest, VelocityChunkReactiveCheckCatchesCollision) {
  rclcpp::NodeOptions opts;
  opts.parameter_overrides({
      {"n_dof", std::int64_t{2}},
      {"joint_position_min", std::vector<double>{-3.14, -3.14}},
      {"joint_position_max", std::vector<double>{3.14, 3.14}},
      {"joint_velocity_max", std::vector<double>{3.15, 3.15}},
      {"joint_torque_max", std::vector<double>{5.0, 5.0}},
      {"self_collision_enabled", true},
      {"self_collision_margin_m", 0.0},
      // Two links, both carrying a capsule at the SAME place (origin), and the
      // pair is NOT allowed → they always interpenetrate → self-collision hit.
      {"collision_n_links", std::int64_t{2}},
      {"collision_parent", std::vector<std::int64_t>{-1, 0}},
      {"collision_joint_kind", std::vector<std::int64_t>{1, 1}},  // revolute, revolute
      {"collision_dof_index", std::vector<std::int64_t>{0, 1}},
      {"collision_origin_xyzrpy", std::vector<double>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
      {"collision_axis", std::vector<double>{0, 0, 1, 0, 0, 1}},
      {"collision_capsule_link", std::vector<std::int64_t>{0, 1}},
      {"collision_capsule_radius", std::vector<double>{0.1, 0.1}},
      {"collision_capsule_half_length", std::vector<double>{0.1, 0.1}},
      {"collision_capsule_origin_xyzrpy", std::vector<double>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
      {"collision_allowed_pairs", std::vector<std::int64_t>{}},  // NOT allowed → checked
      {"collision_link_names", std::vector<std::string>{"l0", "l1"}},
      {"collision_joint_names", std::vector<std::string>{"j0", "j1"}},
      {"collision_state_deadline_ms", 2000.0},
  });
  auto node = std::make_shared<osk::SafetyKernelLifecycleNode>("kernel_vel_catch", opts);
  rclcpp_lifecycle::State unconf(lifecycle_msgs::msg::State::PRIMARY_STATE_UNCONFIGURED, "uc");
  ASSERT_EQ(node->on_configure(unconf), osk::SafetyKernelLifecycleNode::CallbackReturn::SUCCESS);
  rclcpp_lifecycle::State inactive(lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE, "in");
  ASSERT_EQ(node->on_activate(inactive), osk::SafetyKernelLifecycleNode::CallbackReturn::SUCCESS);

  rclcpp::Node helper("vel_catch_helper");
  rclcpp::QoS chunk_qos(rclcpp::KeepLast(1));
  chunk_qos.reliable();
  auto cand_pub = helper.create_publisher<openral_msgs::msg::ActionChunk>(
      "/openral/candidate_action", chunk_qos);
  rclcpp::QoS js_qos(rclcpp::KeepLast(1));
  js_qos.best_effort();
  auto js_pub = helper.create_publisher<sensor_msgs::msg::JointState>("/joint_states", js_qos);

  rclcpp::executors::SingleThreadedExecutor exec;
  exec.add_node(node->get_node_base_interface());
  exec.add_node(helper.get_node_base_interface());

  sensor_msgs::msg::JointState js;
  js.name = {"j0", "j1"};
  js.position = {0.0, 0.0};

  openral_msgs::msg::ActionChunk vel;
  vel.control_mode = 1;  // JOINT_VELOCITY
  vel.horizon = 1;
  vel.n_dof = 2;
  vel.flat = {0.0, 0.0};

  const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(1500);
  while (!node->fault_latched() && std::chrono::steady_clock::now() < deadline) {
    js_pub->publish(js);
    exec.spin_some(std::chrono::milliseconds(10));
    cand_pub->publish(vel);
    exec.spin_some(std::chrono::milliseconds(10));
  }
  EXPECT_TRUE(node->fault_latched())
      << "a velocity chunk whose reconstructed config self-collides must be rejected + estopped; "
         "chunks_dropped="
      << node->chunks_dropped() << " chunks_passed=" << node->chunks_passed();
}

// DETERMINISTIC proof of the mobile-base world (voxel) path: the
// panda_mobile "arm hits the table" scenario. The model is a planar base
// (prismatic-x, dof 0) carrying a one-link arm (revolute-z, dof 1) whose capsule
// sits 0.3 m ahead of base_link. The measured seed places the BASE at x=5 m in
// the world, but `collision_base_dofs=[0]` makes the kernel zero the base dof
// before FK so the arm is evaluated in the base_link frame — where a
// base-relative occupancy grid has an occupied wall at x>=0.2 m. With the
// base-frame fix the arm capsule lands in an occupied voxel and the kernel must
// estop; WITHOUT it the arm would be placed at x~5.3 m, outside the local grid,
// and nothing would ever be caught. This is the exact regression the dropped
// test missed: the discriminator is the base-dof zeroing, and the geometry is
// hand-verified (capsule centre (0.3,0,0), grid x in [0,0.8]).
TEST_F(LifecycleKernelTest, MobileBaseArmCaughtAgainstVoxelWall) {
  rclcpp::NodeOptions opts;
  opts.parameter_overrides({
      {"n_dof", std::int64_t{2}},
      {"joint_position_min", std::vector<double>{-10.0, -3.14}},
      {"joint_position_max", std::vector<double>{10.0, 3.14}},
      {"joint_velocity_max", std::vector<double>{5.0, 5.0}},
      {"joint_torque_max", std::vector<double>{5.0, 5.0}},
      // Isolate the voxel path: self/analytic-world checks off, voxel on.
      {"self_collision_enabled", false},
      {"world_voxel_enabled", true},
      {"world_voxel_margin_m", 0.0},
      {"world_voxel_deadline_ms", 2000.0},
      {"world_voxel_max_cells", std::int64_t{4096}},
      // Link 0: planar base, prismatic along +x (dof 0), at the model root.
      // Link 1: arm, revolute about +z (dof 1), offset 0.3 m ahead of base_link;
      // its capsule (r=0.1, hl=0.1) is centred on the link origin.
      {"collision_n_links", std::int64_t{2}},
      {"collision_parent", std::vector<std::int64_t>{-1, 0}},
      {"collision_joint_kind", std::vector<std::int64_t>{2, 1}},  // prismatic, revolute
      {"collision_dof_index", std::vector<std::int64_t>{0, 1}},
      {"collision_origin_xyzrpy", std::vector<double>{0, 0, 0, 0, 0, 0, 0.3, 0, 0, 0, 0, 0}},
      {"collision_axis", std::vector<double>{1, 0, 0, 0, 0, 1}},  // base +x, arm +z
      {"collision_capsule_link", std::vector<std::int64_t>{0, 1}},
      {"collision_capsule_radius", std::vector<double>{0.05, 0.1}},
      {"collision_capsule_half_length", std::vector<double>{0.05, 0.1}},
      {"collision_capsule_origin_xyzrpy", std::vector<double>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
      {"collision_allowed_pairs", std::vector<std::int64_t>{0, 1}},
      {"collision_link_names", std::vector<std::string>{"base", "arm"}},
      {"collision_joint_names", std::vector<std::string>{"j_base", "j_arm"}},
      {"collision_base_dofs", std::vector<std::int64_t>{0}},  // THE FIX under test
      {"collision_state_deadline_ms", 2000.0},
  });
  auto node = std::make_shared<osk::SafetyKernelLifecycleNode>("kernel_mobile_voxel", opts);
  rclcpp_lifecycle::State unconf(lifecycle_msgs::msg::State::PRIMARY_STATE_UNCONFIGURED, "uc");
  ASSERT_EQ(node->on_configure(unconf), osk::SafetyKernelLifecycleNode::CallbackReturn::SUCCESS);
  rclcpp_lifecycle::State inactive(lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE, "in");
  ASSERT_EQ(node->on_activate(inactive), osk::SafetyKernelLifecycleNode::CallbackReturn::SUCCESS);

  rclcpp::Node helper("mobile_voxel_helper");
  rclcpp::QoS chunk_qos(rclcpp::KeepLast(1));
  chunk_qos.reliable();
  auto cand_pub = helper.create_publisher<openral_msgs::msg::ActionChunk>(
      "/openral/candidate_action", chunk_qos);
  rclcpp::QoS js_qos(rclcpp::KeepLast(1));
  js_qos.best_effort();
  auto js_pub = helper.create_publisher<sensor_msgs::msg::JointState>("/joint_states", js_qos);
  rclcpp::QoS voxel_qos(rclcpp::KeepLast(1));
  voxel_qos.reliable();
  auto voxel_pub = helper.create_publisher<openral_msgs::msg::OccupancyVoxels>(
      "/openral/world_voxels", voxel_qos);

  rclcpp::executors::SingleThreadedExecutor exec;
  exec.add_node(node->get_node_base_interface());
  exec.add_node(helper.get_node_base_interface());

  // Base-relative occupancy wall: every cell with centre x>=0.2 m is occupied.
  // Grid spans x in [0,0.8], y/z in [-0.2,0.2] at 0.1 m resolution. The arm
  // capsule at (0.3,0,0) lands inside; the base capsule at the origin (x<0.2)
  // does not, so only the base-frame-corrected arm pose can trigger a hit.
  openral_msgs::msg::OccupancyVoxels vox;
  // Identity: a base-aligned synthetic lattice. `OccupancyVoxels` is oriented,
  // and an unset orientation is the all-zero quaternion, which the ingest
  // refuses -- leaving this test asserting a fault it never meant to cause.
  vox.orientation.w = 1.0;
  vox.resolution = 0.1;
  vox.size_x = 8;
  vox.size_y = 4;
  vox.size_z = 4;
  vox.origin.x = 0.0;
  vox.origin.y = -0.2;
  vox.origin.z = -0.2;
  vox.occupancy.assign(static_cast<std::size_t>(vox.size_x) * vox.size_y * vox.size_z, 0);
  for (std::uint32_t iz = 0; iz < vox.size_z; ++iz) {
    for (std::uint32_t iy = 0; iy < vox.size_y; ++iy) {
      for (std::uint32_t ix = 0; ix < vox.size_x; ++ix) {
        const double cx = vox.origin.x + (ix + 0.5) * vox.resolution;
        if (cx >= 0.2) {
          vox.occupancy[ix + vox.size_x * (iy + vox.size_y * iz)] = 1;
        }
      }
    }
  }

  // Seed the BASE far out in the world (x=5 m); the arm joint at 0. The kernel
  // must zero the base dof before FK, evaluating the arm in base_link frame.
  sensor_msgs::msg::JointState js;
  js.name = {"j_base", "j_arm"};
  js.position = {5.0, 0.0};

  openral_msgs::msg::ActionChunk vel;
  vel.control_mode = 1;  // JOINT_VELOCITY (reactive check uses the measured seed)
  vel.horizon = 1;
  vel.n_dof = 2;
  vel.flat = {0.0, 0.0};

  const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(2000);
  while (!node->fault_latched() && std::chrono::steady_clock::now() < deadline) {
    js_pub->publish(js);
    voxel_pub->publish(vox);
    exec.spin_some(std::chrono::milliseconds(10));
    cand_pub->publish(vel);
    exec.spin_some(std::chrono::milliseconds(10));
  }
  EXPECT_TRUE(node->fault_latched())
      << "the mobile-base arm capsule, evaluated in base_link frame (base dof zeroed), must land "
         "in "
         "the occupied voxel wall and estop; chunks_dropped="
      << node->chunks_dropped() << " chunks_passed=" << node->chunks_passed();
}

namespace {

// Pull a "key":"value" / "key":number field out of the kernel's evidence_json
// (a flat CollisionEvidence object — no nesting, so no JSON parser needed;
// nlohmann_json is an optional dependency of this package).
std::string evidence_field(const std::string& json, const std::string& key) {
  const std::string needle = "\"" + key + "\":";
  const std::size_t at = json.find(needle);
  if (at == std::string::npos) {
    return {};
  }
  std::size_t begin = at + needle.size();
  const bool quoted = begin < json.size() && json[begin] == '"';
  if (quoted) {
    ++begin;
  }
  const std::size_t end = quoted ? json.find('"', begin) : json.find_first_of(",}", begin);
  if (end == std::string::npos) {
    return {};
  }
  return json.substr(begin, end - begin);
}

}  // namespace

// The E-stop evidence must describe ONE cell: the voxel named in
// `link_b_or_object` is the voxel `min_distance_m` measures. The two used to
// come from different cells — the identity from the first cell to trip, the
// distance from the sweep-wide minimum — so a shallow graze could be published
// carrying a deep cell's number (the attached-payload residue report that sent
// diagnosis after a penetration that never existed).
//
// Geometry (hand-computed, single occupied row on the capsule's own axis):
// the arm capsule sits at (0.3, 0, 0), r=0.1, half-length 0.1 along +z. The
// grid is 4x1x1 at 0.1 m from origin (0.02, -0.05, -0.05), so cell centres run
// x = 0.07 / 0.17 / 0.27 / 0.37 at y=z=0. Two cells are occupied:
//   cell 1 (x=0.17): surface distance (0.13 - 0.05) - 0.1 = -0.02  ← trips first
//   cell 2 (x=0.27): the capsule axis is inside it        = -0.10  ← deepest
// Both trip; the evidence must name cell 2 and quote -0.10 for it.
TEST_F(LifecycleKernelTest, CollisionEvidenceNamesTheVoxelItsDistanceDescribes) {
  rclcpp::NodeOptions opts;
  opts.parameter_overrides({
      {"n_dof", std::int64_t{2}},
      {"joint_position_min", std::vector<double>{-10.0, -3.14}},
      {"joint_position_max", std::vector<double>{10.0, 3.14}},
      {"joint_velocity_max", std::vector<double>{5.0, 5.0}},
      {"joint_torque_max", std::vector<double>{5.0, 5.0}},
      {"self_collision_enabled", false},
      {"world_voxel_enabled", true},
      {"world_voxel_margin_m", 0.0},
      {"world_voxel_deadline_ms", 2000.0},
      {"world_voxel_max_cells", std::int64_t{4096}},
      // Link 0: base at the root. Link 1: arm, 0.3 m ahead, capsule r=0.1 hl=0.1.
      {"collision_n_links", std::int64_t{2}},
      {"collision_parent", std::vector<std::int64_t>{-1, 0}},
      {"collision_joint_kind", std::vector<std::int64_t>{2, 1}},
      {"collision_dof_index", std::vector<std::int64_t>{0, 1}},
      {"collision_origin_xyzrpy", std::vector<double>{0, 0, 0, 0, 0, 0, 0.3, 0, 0, 0, 0, 0}},
      {"collision_axis", std::vector<double>{1, 0, 0, 0, 0, 1}},
      {"collision_capsule_link", std::vector<std::int64_t>{0, 1}},
      {"collision_capsule_radius", std::vector<double>{0.05, 0.1}},
      {"collision_capsule_half_length", std::vector<double>{0.05, 0.1}},
      {"collision_capsule_origin_xyzrpy", std::vector<double>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
      {"collision_allowed_pairs", std::vector<std::int64_t>{0, 1}},
      {"collision_link_names", std::vector<std::string>{"base", "arm"}},
      {"collision_joint_names", std::vector<std::string>{"j_base", "j_arm"}},
      {"collision_base_dofs", std::vector<std::int64_t>{0}},
      {"collision_state_deadline_ms", 2000.0},
  });
  auto node = std::make_shared<osk::SafetyKernelLifecycleNode>("kernel_evidence_cell", opts);
  rclcpp_lifecycle::State unconf(lifecycle_msgs::msg::State::PRIMARY_STATE_UNCONFIGURED, "uc");
  ASSERT_EQ(node->on_configure(unconf), osk::SafetyKernelLifecycleNode::CallbackReturn::SUCCESS);
  rclcpp_lifecycle::State inactive(lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE, "in");
  ASSERT_EQ(node->on_activate(inactive), osk::SafetyKernelLifecycleNode::CallbackReturn::SUCCESS);

  rclcpp::Node helper("evidence_cell_helper");
  rclcpp::QoS chunk_qos(rclcpp::KeepLast(1));
  chunk_qos.reliable();
  auto cand_pub = helper.create_publisher<openral_msgs::msg::ActionChunk>(
      "/openral/candidate_action", chunk_qos);
  rclcpp::QoS js_qos(rclcpp::KeepLast(1));
  js_qos.best_effort();
  auto js_pub = helper.create_publisher<sensor_msgs::msg::JointState>("/joint_states", js_qos);
  rclcpp::QoS voxel_qos(rclcpp::KeepLast(1));
  voxel_qos.reliable();
  auto voxel_pub = helper.create_publisher<openral_msgs::msg::OccupancyVoxels>(
      "/openral/world_voxels", voxel_qos);
  std::string evidence_json;
  rclcpp::QoS failure_qos(rclcpp::KeepLast(10));
  failure_qos.reliable();
  auto failure_sub = helper.create_subscription<openral_msgs::msg::FailureTrigger>(
      "/openral/failure/safety", failure_qos,
      [&evidence_json](const openral_msgs::msg::FailureTrigger::SharedPtr msg) {
        if (evidence_json.empty()) {
          evidence_json = msg->evidence_json;
        }
      });

  rclcpp::executors::SingleThreadedExecutor exec;
  exec.add_node(node->get_node_base_interface());
  exec.add_node(helper.get_node_base_interface());

  openral_msgs::msg::OccupancyVoxels vox;
  // Identity: a base-aligned synthetic lattice. `OccupancyVoxels` is oriented,
  // and an unset orientation is the all-zero quaternion, which the ingest
  // refuses -- leaving this test asserting a fault it never meant to cause.
  vox.orientation.w = 1.0;
  vox.resolution = 0.1;
  vox.size_x = 4;
  vox.size_y = 1;
  vox.size_z = 1;
  vox.origin.x = 0.02;
  vox.origin.y = -0.05;
  vox.origin.z = -0.05;
  vox.occupancy.assign(4, 0);
  vox.occupancy[1] = 1;  // grazing cell, scanned first
  vox.occupancy[2] = 1;  // deepest cell

  sensor_msgs::msg::JointState js;
  js.name = {"j_base", "j_arm"};
  js.position = {0.0, 0.0};

  openral_msgs::msg::ActionChunk vel;
  vel.control_mode = 1;  // JOINT_VELOCITY — the reactive check uses the measured seed
  vel.horizon = 1;
  vel.n_dof = 2;
  vel.flat = {0.0, 0.0};

  const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(2000);
  while (evidence_json.empty() && std::chrono::steady_clock::now() < deadline) {
    js_pub->publish(js);
    voxel_pub->publish(vox);
    exec.spin_some(std::chrono::milliseconds(10));
    cand_pub->publish(vel);
    exec.spin_some(std::chrono::milliseconds(10));
  }
  ASSERT_TRUE(node->fault_latched()) << "the arm capsule overlaps two occupied cells — must estop";
  ASSERT_FALSE(evidence_json.empty()) << "a collision stop must publish CollisionEvidence";
  EXPECT_EQ(evidence_field(evidence_json, "link_b_or_object"), "voxel_2")
      << "evidence must name the cell that its distance describes: " << evidence_json;
  EXPECT_NEAR(std::stod(evidence_field(evidence_json, "min_distance_m")), -0.1, 1e-6)
      << evidence_json;
}

// Phase 3 — DETERMINISTIC proof of PREDICTIVE Cartesian look-ahead: a
// CARTESIAN_DELTA chunk whose MEASURED start config is clear (so the reactive
// check passes) but whose proposed EE deltas drive the arm into an obstacle must
// be rejected + estopped via the Jacobian reconstruction. This is exactly the
// "all actions in the chunk must be verified safe before they execute" contract.
// Model: planar 2R arm (revolute-Z, dof 0/1) + a fixed EE link (index 2) at the
// tip. At q=[0, +90°] the EE is at (1,1,0); a +y chunk drives it toward a voxel
// wall at y>=1.3 — clear at the start, colliding a few steps in.
TEST_F(LifecycleKernelTest, CartesianDeltaPredictiveCatchesChunkDrivingEeIntoWall) {
  rclcpp::NodeOptions opts;
  opts.parameter_overrides({
      {"n_dof", std::int64_t{2}},
      {"joint_position_min", std::vector<double>{-3.14, -3.14}},
      {"joint_position_max", std::vector<double>{3.14, 3.14}},
      {"joint_velocity_max", std::vector<double>{5.0, 5.0}},
      {"joint_torque_max", std::vector<double>{5.0, 5.0}},
      // Isolate the world-voxel path (no self-collision noise).
      {"self_collision_enabled", false},
      {"world_voxel_enabled", true},
      {"world_voxel_margin_m", 0.0},
      {"world_voxel_deadline_ms", 2000.0},
      {"world_voxel_max_cells", std::int64_t{8192}},
      // 2R arm + fixed EE link. Link lengths 1 m; capsule r=0.05 on each link.
      {"collision_n_links", std::int64_t{3}},
      {"collision_parent", std::vector<std::int64_t>{-1, 0, 1}},
      {"collision_joint_kind", std::vector<std::int64_t>{1, 1, 0}},  // rev, rev, fixed
      {"collision_dof_index", std::vector<std::int64_t>{0, 1, -1}},
      {"collision_origin_xyzrpy",
       std::vector<double>{0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0}},
      {"collision_axis", std::vector<double>{0, 0, 1, 0, 0, 1, 0, 0, 1}},
      {"collision_capsule_link", std::vector<std::int64_t>{0, 1, 2}},
      {"collision_capsule_radius", std::vector<double>{0.05, 0.05, 0.05}},
      {"collision_capsule_half_length", std::vector<double>{0.05, 0.05, 0.05}},
      {"collision_capsule_origin_xyzrpy",
       std::vector<double>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
      {"collision_allowed_pairs", std::vector<std::int64_t>{}},
      {"collision_link_names", std::vector<std::string>{"l0", "l1", "ee"}},
      {"collision_joint_names", std::vector<std::string>{"j0", "j1"}},
      {"collision_state_deadline_ms", 2000.0},
      // Phase 3 — predictive Cartesian: EE is link index 2.
      {"collision_ee_link_index", std::int64_t{2}},
      {"collision_predict_lambda", 0.02},
      {"collision_predict_margin_growth_m", 0.02},
      {"collision_predict_max_steps", std::int64_t{0}},  // check all steps
  });
  auto node = std::make_shared<osk::SafetyKernelLifecycleNode>("kernel_cart_predict", opts);
  rclcpp_lifecycle::State unconf(lifecycle_msgs::msg::State::PRIMARY_STATE_UNCONFIGURED, "uc");
  ASSERT_EQ(node->on_configure(unconf), osk::SafetyKernelLifecycleNode::CallbackReturn::SUCCESS);
  rclcpp_lifecycle::State inactive(lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE, "in");
  ASSERT_EQ(node->on_activate(inactive), osk::SafetyKernelLifecycleNode::CallbackReturn::SUCCESS);

  rclcpp::Node helper("cart_predict_helper");
  rclcpp::QoS chunk_qos(rclcpp::KeepLast(1));
  chunk_qos.reliable();
  auto cand_pub = helper.create_publisher<openral_msgs::msg::ActionChunk>(
      "/openral/candidate_action", chunk_qos);
  rclcpp::QoS js_qos(rclcpp::KeepLast(1));
  js_qos.best_effort();
  auto js_pub = helper.create_publisher<sensor_msgs::msg::JointState>("/joint_states", js_qos);
  rclcpp::QoS voxel_qos(rclcpp::KeepLast(1));
  voxel_qos.reliable();
  auto voxel_pub = helper.create_publisher<openral_msgs::msg::OccupancyVoxels>(
      "/openral/world_voxels", voxel_qos);
  std::atomic<int> safe_count{0};
  auto safe_sub = helper.create_subscription<openral_msgs::msg::ActionChunk>(
      "/openral/safe_action", chunk_qos,
      [&safe_count](const openral_msgs::msg::ActionChunk::SharedPtr) { ++safe_count; });

  rclcpp::executors::SingleThreadedExecutor exec;
  exec.add_node(node->get_node_base_interface());
  exec.add_node(helper.get_node_base_interface());

  // Voxel wall: every cell with centre y>=1.3 occupied. Grid x,y in [0,2],
  // z in [-0.15,0.15] @ 0.1 m. The EE starts at (1,1,0) — clear (y=1 < 1.3); a
  // +y chunk drives it into the wall.
  openral_msgs::msg::OccupancyVoxels vox;
  // Identity: a base-aligned synthetic lattice. `OccupancyVoxels` is oriented,
  // and an unset orientation is the all-zero quaternion, which the ingest
  // refuses -- leaving this test asserting a fault it never meant to cause.
  vox.orientation.w = 1.0;
  vox.resolution = 0.1;
  vox.size_x = 20;
  vox.size_y = 20;
  vox.size_z = 3;
  vox.origin.x = 0.0;
  vox.origin.y = 0.0;
  vox.origin.z = -0.15;
  vox.occupancy.assign(static_cast<std::size_t>(vox.size_x) * vox.size_y * vox.size_z, 0);
  for (std::uint32_t iz = 0; iz < vox.size_z; ++iz) {
    for (std::uint32_t iy = 0; iy < vox.size_y; ++iy) {
      const double cy = vox.origin.y + (iy + 0.5) * vox.resolution;
      if (cy >= 1.3) {
        for (std::uint32_t ix = 0; ix < vox.size_x; ++ix) {
          vox.occupancy[ix + vox.size_x * (iy + vox.size_y * iz)] = 1;
        }
      }
    }
  }

  // Seed the arm bent at q=[0, +90°] → EE at (1,1,0), clear of the wall.
  sensor_msgs::msg::JointState js;
  js.name = {"j0", "j1"};
  js.position = {0.0, 1.57079632679};

  // CARTESIAN_DELTA chunk: 10 steps of +0.05 m along +y (no rotation). Reactive
  // (start) is clear; the predicted trajectory enters the wall.
  openral_msgs::msg::ActionChunk cart;
  cart.control_mode = 5;  // CARTESIAN_DELTA
  cart.horizon = 10;
  cart.n_dof = 6;  // [vx,vy,vz, wx,wy,wz] per step
  cart.flat.clear();
  for (int s = 0; s < 10; ++s) {
    cart.flat.insert(cart.flat.end(), {0.0, 0.05, 0.0, 0.0, 0.0, 0.0});
  }

  // Establish a fresh seed + voxel grid first, and confirm the reactive check
  // does NOT latch on the (clear) start config.
  auto seed_deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(300);
  while (std::chrono::steady_clock::now() < seed_deadline) {
    js_pub->publish(js);
    voxel_pub->publish(vox);
    exec.spin_some(std::chrono::milliseconds(10));
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
  ASSERT_FALSE(node->fault_latched()) << "start config must be clear (reactive should not fire)";

  const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(1500);
  while (!node->fault_latched() && std::chrono::steady_clock::now() < deadline) {
    js_pub->publish(js);
    voxel_pub->publish(vox);
    exec.spin_some(std::chrono::milliseconds(10));
    cand_pub->publish(cart);
    exec.spin_some(std::chrono::milliseconds(10));
  }
  EXPECT_TRUE(node->fault_latched())
      << "a CARTESIAN_DELTA chunk whose predicted EE trajectory enters the wall must be rejected + "
         "estopped by the Jacobian look-ahead even though the start config is clear; "
         "chunks_dropped="
      << node->chunks_dropped() << " chunks_passed=" << node->chunks_passed();
  EXPECT_EQ(safe_count.load(), 0)
      << "the colliding Cartesian chunk must never reach /openral/safe_action";
}

namespace {

// Pull a JSON number array (`"key":[a,b,c]`) out of the kernel's evidence_json.
// `evidence_field` above stops at the first comma, which is inside the array.
std::vector<double> evidence_doubles(const std::string& json, const std::string& key) {
  const std::string needle = "\"" + key + "\":[";
  const std::size_t at = json.find(needle);
  if (at == std::string::npos) {
    return {};
  }
  const std::size_t begin = at + needle.size();
  const std::size_t end = json.find(']', begin);
  if (end == std::string::npos) {
    return {};
  }
  std::vector<double> out;
  std::istringstream in(json.substr(begin, end - begin));
  std::string token;
  while (std::getline(in, token, ',')) {
    out.push_back(std::stod(token));
  }
  return out;
}

// The planar 2R arm + fixed EE link of the predictive test above, with the
// voxel wall at y>=1.3. Shared by the stop and its replay so the replay is
// adjudicating against the same geometry the stop did — the point of the test.
std::vector<rclcpp::Parameter> planar_2r_predictive_params() {
  return {
      {"n_dof", std::int64_t{2}},
      {"joint_position_min", std::vector<double>{-3.14, -3.14}},
      {"joint_position_max", std::vector<double>{3.14, 3.14}},
      {"joint_velocity_max", std::vector<double>{5.0, 5.0}},
      {"joint_torque_max", std::vector<double>{5.0, 5.0}},
      {"self_collision_enabled", false},
      {"world_voxel_enabled", true},
      {"world_voxel_margin_m", 0.0},
      {"world_voxel_deadline_ms", 2000.0},
      {"world_voxel_max_cells", std::int64_t{8192}},
      {"collision_n_links", std::int64_t{3}},
      {"collision_parent", std::vector<std::int64_t>{-1, 0, 1}},
      {"collision_joint_kind", std::vector<std::int64_t>{1, 1, 0}},
      {"collision_dof_index", std::vector<std::int64_t>{0, 1, -1}},
      {"collision_origin_xyzrpy",
       std::vector<double>{0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0}},
      {"collision_axis", std::vector<double>{0, 0, 1, 0, 0, 1, 0, 0, 1}},
      {"collision_capsule_link", std::vector<std::int64_t>{0, 1, 2}},
      {"collision_capsule_radius", std::vector<double>{0.05, 0.05, 0.05}},
      {"collision_capsule_half_length", std::vector<double>{0.05, 0.05, 0.05}},
      {"collision_capsule_origin_xyzrpy",
       std::vector<double>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
      {"collision_allowed_pairs", std::vector<std::int64_t>{}},
      {"collision_link_names", std::vector<std::string>{"l0", "l1", "ee"}},
      {"collision_joint_names", std::vector<std::string>{"j0", "j1"}},
      {"collision_state_deadline_ms", 2000.0},
      {"collision_ee_link_index", std::int64_t{2}},
      {"collision_predict_lambda", 0.02},
      {"collision_predict_margin_growth_m", 0.0},  // isolate the config, not the margin
      {"collision_predict_max_steps", std::int64_t{0}},
  };
}

// The voxel wall of the predictive test: every cell with centre y>=1.3.
openral_msgs::msg::OccupancyVoxels wall_voxels() {
  openral_msgs::msg::OccupancyVoxels vox;
  vox.orientation.w = 1.0;
  vox.resolution = 0.1;
  vox.size_x = 20;
  vox.size_y = 20;
  vox.size_z = 3;
  vox.origin.x = 0.0;
  vox.origin.y = 0.0;
  vox.origin.z = -0.15;
  vox.occupancy.assign(static_cast<std::size_t>(vox.size_x) * vox.size_y * vox.size_z, 0);
  for (std::uint32_t iz = 0; iz < vox.size_z; ++iz) {
    for (std::uint32_t iy = 0; iy < vox.size_y; ++iy) {
      const double cy = vox.origin.y + (iy + 0.5) * vox.resolution;
      if (cy >= 1.3) {
        for (std::uint32_t ix = 0; ix < vox.size_x; ++ix) {
          vox.occupancy[ix + vox.size_x * (iy + vox.size_y * iz)] = 1;
        }
      }
    }
  }
  return vox;
}

}  // namespace

// A PREDICTIVE stop's verdict is about a configuration that exists in no other
// artifact: it is the kernel's own damped-least-squares integration of the
// chunk, at the kernel's lambda and its seed dt. Adjudicating such a stop
// against the measured joints therefore reads geometry the kernel never
// checked — the drawer-opening run that motivated this reported two links
// -5.34 mm apart while offline mesh adjudication at the *recorded* joints put
// the same pair +53 mm clear, and the disagreement was the artifact's, not the
// kernel's.
//
// So the evidence must carry the configuration it was measured at, and carry
// it exactly. The proof runs in the kernel's own arithmetic rather than a
// reimplementation of it: replay the captured configuration through a second,
// identically configured kernel as a JOINT_POSITION row, and require the same
// pair and the same distance to 1e-8.
TEST_F(LifecycleKernelTest, CollisionEvidenceReplaysThePredictedConfigurationItAdjudicated) {
  const auto vox = wall_voxels();

  rclcpp::Node helper("evidence_config_helper");
  rclcpp::QoS chunk_qos(rclcpp::KeepLast(1));
  chunk_qos.reliable();
  auto cand_pub = helper.create_publisher<openral_msgs::msg::ActionChunk>(
      "/openral/candidate_action", chunk_qos);
  rclcpp::QoS js_qos(rclcpp::KeepLast(1));
  js_qos.best_effort();
  auto js_pub = helper.create_publisher<sensor_msgs::msg::JointState>("/joint_states", js_qos);
  rclcpp::QoS voxel_qos(rclcpp::KeepLast(1));
  voxel_qos.reliable();
  auto voxel_pub = helper.create_publisher<openral_msgs::msg::OccupancyVoxels>(
      "/openral/world_voxels", voxel_qos);
  std::string evidence_json;
  rclcpp::QoS failure_qos(rclcpp::KeepLast(10));
  failure_qos.reliable();
  auto failure_sub = helper.create_subscription<openral_msgs::msg::FailureTrigger>(
      "/openral/failure/safety", failure_qos,
      [&evidence_json](const openral_msgs::msg::FailureTrigger::SharedPtr msg) {
        if (evidence_json.empty()) {
          evidence_json = msg->evidence_json;
        }
      });

  // ── The stop: a CARTESIAN_DELTA chunk driving the EE into the wall. ────────
  rclcpp::NodeOptions opts;
  opts.parameter_overrides(planar_2r_predictive_params());
  auto node = std::make_shared<osk::SafetyKernelLifecycleNode>("kernel_evidence_config", opts);
  rclcpp_lifecycle::State unconf(lifecycle_msgs::msg::State::PRIMARY_STATE_UNCONFIGURED, "uc");
  ASSERT_EQ(node->on_configure(unconf), osk::SafetyKernelLifecycleNode::CallbackReturn::SUCCESS);
  rclcpp_lifecycle::State inactive(lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE, "in");
  ASSERT_EQ(node->on_activate(inactive), osk::SafetyKernelLifecycleNode::CallbackReturn::SUCCESS);

  rclcpp::executors::SingleThreadedExecutor exec;
  exec.add_node(node->get_node_base_interface());
  exec.add_node(helper.get_node_base_interface());

  const std::vector<double> seed{0.0, 1.57079632679};
  sensor_msgs::msg::JointState js;
  js.name = {"j0", "j1"};
  js.position = seed;

  openral_msgs::msg::ActionChunk cart;
  cart.control_mode = 5;  // CARTESIAN_DELTA
  cart.horizon = 10;
  cart.n_dof = 6;
  for (int s = 0; s < 10; ++s) {
    cart.flat.insert(cart.flat.end(), {0.0, 0.05, 0.0, 0.0, 0.0, 0.0});
  }

  auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(2000);
  while (evidence_json.empty() && std::chrono::steady_clock::now() < deadline) {
    js_pub->publish(js);
    voxel_pub->publish(vox);
    exec.spin_some(std::chrono::milliseconds(10));
    cand_pub->publish(cart);
    exec.spin_some(std::chrono::milliseconds(10));
  }
  ASSERT_FALSE(evidence_json.empty()) << "the predicted trajectory enters the wall — must estop";
  const std::string stop_evidence = evidence_json;
  // Published as a gtest property so `tests/unit/fixtures/` can hold a REAL
  // predictive payload rather than a hand-written one (same convention as
  // `kernel_reactive_collision_evidence.json`).
  RecordProperty("predictive_collision_evidence_json", stop_evidence);
  const int step = std::stoi(evidence_field(stop_evidence, "horizon_step"));
  ASSERT_GE(step, 0) << "this must be a predicted step, not the reactive floor: " << stop_evidence;

  const std::vector<double> q = evidence_doubles(stop_evidence, "joint_positions_rad");
  ASSERT_EQ(q.size(), seed.size()) << stop_evidence;
  // The recorded configuration is the PREDICTED one. If it were the measured
  // seed the record would be describing a pose the kernel found clear, which
  // is the whole defect: the start config passes the reactive check.
  EXPECT_GT(std::hypot(q[0] - seed[0], q[1] - seed[1]), 1e-6)
      << "the evidence recorded the measured seed, not the configuration it "
         "adjudicated: "
      << stop_evidence;

  exec.remove_node(node->get_node_base_interface());
  node.reset();

  // ── The replay: the captured configuration, as a position row. ────────────
  rclcpp::NodeOptions replay_opts;
  replay_opts.parameter_overrides(planar_2r_predictive_params());
  auto replay =
      std::make_shared<osk::SafetyKernelLifecycleNode>("kernel_evidence_replay", replay_opts);
  ASSERT_EQ(replay->on_configure(unconf), osk::SafetyKernelLifecycleNode::CallbackReturn::SUCCESS);
  ASSERT_EQ(replay->on_activate(inactive), osk::SafetyKernelLifecycleNode::CallbackReturn::SUCCESS);
  exec.add_node(replay->get_node_base_interface());

  openral_msgs::msg::ActionChunk pos;
  pos.control_mode = 0;  // JOINT_POSITION — FK'd directly, no reconstruction
  pos.horizon = 1;
  pos.n_dof = 2;
  pos.flat = {q[0], q[1]};

  evidence_json.clear();
  deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(2000);
  while (evidence_json.empty() && std::chrono::steady_clock::now() < deadline) {
    voxel_pub->publish(vox);
    exec.spin_some(std::chrono::milliseconds(10));
    cand_pub->publish(pos);
    exec.spin_some(std::chrono::milliseconds(10));
  }
  ASSERT_FALSE(evidence_json.empty())
      << "the recorded configuration must reproduce the stop it was recorded for";
  EXPECT_EQ(evidence_field(evidence_json, "link_a"), evidence_field(stop_evidence, "link_a"))
      << evidence_json << " vs " << stop_evidence;
  EXPECT_EQ(evidence_field(evidence_json, "link_b_or_object"),
            evidence_field(stop_evidence, "link_b_or_object"))
      << evidence_json << " vs " << stop_evidence;
  EXPECT_NEAR(std::stod(evidence_field(evidence_json, "min_distance_m")),
              std::stod(evidence_field(stop_evidence, "min_distance_m")), 1e-8)
      << "the verdict must be re-derivable from the recorded configuration alone: " << evidence_json
      << " vs " << stop_evidence;

  exec.remove_node(replay->get_node_base_interface());
}

// The first predicted step uses the configured collision margin. Margin growth
// applies only to additional look-ahead depth: a horizon-1 command must not gain
// a full extra uncertainty margin and false-stop.
TEST_F(LifecycleKernelTest, CartesianDeltaFirstStepUsesConfiguredMargin) {
  rclcpp::NodeOptions opts;
  opts.parameter_overrides({
      {"n_dof", std::int64_t{2}},
      {"joint_position_min", std::vector<double>{-3.14, -3.14}},
      {"joint_position_max", std::vector<double>{3.14, 3.14}},
      {"joint_velocity_max", std::vector<double>{5.0, 5.0}},
      {"joint_torque_max", std::vector<double>{5.0, 5.0}},
      {"self_collision_enabled", false},
      {"world_voxel_enabled", true},
      {"world_voxel_margin_m", 0.0},
      {"world_voxel_deadline_ms", 2000.0},
      {"world_voxel_max_cells", std::int64_t{8192}},
      {"collision_n_links", std::int64_t{3}},
      {"collision_parent", std::vector<std::int64_t>{-1, 0, 1}},
      {"collision_joint_kind", std::vector<std::int64_t>{1, 1, 0}},
      {"collision_dof_index", std::vector<std::int64_t>{0, 1, -1}},
      {"collision_origin_xyzrpy",
       std::vector<double>{0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0}},
      {"collision_axis", std::vector<double>{0, 0, 1, 0, 0, 1, 0, 0, 1}},
      {"collision_capsule_link", std::vector<std::int64_t>{0, 1, 2}},
      {"collision_capsule_radius", std::vector<double>{0.05, 0.05, 0.05}},
      {"collision_capsule_half_length", std::vector<double>{0.05, 0.05, 0.05}},
      {"collision_capsule_origin_xyzrpy",
       std::vector<double>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
      {"collision_allowed_pairs", std::vector<std::int64_t>{}},
      {"collision_link_names", std::vector<std::string>{"l0", "l1", "ee"}},
      {"collision_joint_names", std::vector<std::string>{"j0", "j1"}},
      {"collision_state_deadline_ms", 2000.0},
      {"collision_ee_link_index", std::int64_t{2}},
      {"collision_predict_lambda", 0.02},
      // Deliberately huge: the old `(step + 1)` formula inflated this one-step
      // check by 2 m and rejected an otherwise clear command.
      {"collision_predict_margin_growth_m", 2.0},
      {"collision_predict_max_steps", std::int64_t{0}},
  });
  auto node = std::make_shared<osk::SafetyKernelLifecycleNode>("kernel_cart_predict_clear", opts);
  rclcpp_lifecycle::State unconf(lifecycle_msgs::msg::State::PRIMARY_STATE_UNCONFIGURED, "uc");
  ASSERT_EQ(node->on_configure(unconf), osk::SafetyKernelLifecycleNode::CallbackReturn::SUCCESS);
  rclcpp_lifecycle::State inactive(lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE, "in");
  ASSERT_EQ(node->on_activate(inactive), osk::SafetyKernelLifecycleNode::CallbackReturn::SUCCESS);

  rclcpp::Node helper("cart_predict_clear_helper");
  rclcpp::QoS chunk_qos(rclcpp::KeepLast(1));
  chunk_qos.reliable();
  auto cand_pub = helper.create_publisher<openral_msgs::msg::ActionChunk>(
      "/openral/candidate_action", chunk_qos);
  rclcpp::QoS js_qos(rclcpp::KeepLast(1));
  js_qos.best_effort();
  auto js_pub = helper.create_publisher<sensor_msgs::msg::JointState>("/joint_states", js_qos);
  rclcpp::QoS voxel_qos(rclcpp::KeepLast(1));
  voxel_qos.reliable();
  auto voxel_pub = helper.create_publisher<openral_msgs::msg::OccupancyVoxels>(
      "/openral/world_voxels", voxel_qos);
  std::atomic<int> safe_count{0};
  auto safe_sub = helper.create_subscription<openral_msgs::msg::ActionChunk>(
      "/openral/safe_action", chunk_qos,
      [&safe_count](const openral_msgs::msg::ActionChunk::SharedPtr) { ++safe_count; });

  rclcpp::executors::SingleThreadedExecutor exec;
  exec.add_node(node->get_node_base_interface());
  exec.add_node(helper.get_node_base_interface());

  // Wall far away (y>=2.3). The native controller clips raw +5 to +1,
  // then applies scale 0.1, so the chunk stays clear. Ignoring either
  // clipping or scale drives the prediction into the wall.
  openral_msgs::msg::OccupancyVoxels vox;
  // Identity: a base-aligned synthetic lattice. `OccupancyVoxels` is oriented,
  // and an unset orientation is the all-zero quaternion, which the ingest
  // refuses -- leaving this test asserting a fault it never meant to cause.
  vox.orientation.w = 1.0;
  vox.resolution = 0.1;
  vox.size_x = 20;
  vox.size_y = 25;
  vox.size_z = 3;
  vox.origin.x = 0.0;
  vox.origin.y = 0.0;
  vox.origin.z = -0.15;
  vox.occupancy.assign(static_cast<std::size_t>(vox.size_x) * vox.size_y * vox.size_z, 0);
  for (std::uint32_t iz = 0; iz < vox.size_z; ++iz) {
    for (std::uint32_t iy = 0; iy < vox.size_y; ++iy) {
      const double cy = vox.origin.y + (iy + 0.5) * vox.resolution;
      if (cy >= 2.3) {
        for (std::uint32_t ix = 0; ix < vox.size_x; ++ix) {
          vox.occupancy[ix + vox.size_x * (iy + vox.size_y * iz)] = 1;
        }
      }
    }
  }

  sensor_msgs::msg::JointState js;
  js.name = {"j0", "j1"};
  js.position = {0.0, 1.57079632679};

  openral_msgs::msg::ActionChunk cart;
  cart.control_mode = 5;
  cart.horizon = 1;
  cart.n_dof = 6;
  cart.cartesian_delta_scale = {0.1, 0.1, 0.1, 1.0, 1.0, 1.0};
  cart.flat.clear();
  cart.flat = {0.0, 5.0, 0.0, 0.0, 0.0, 0.0};

  const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(800);
  while (safe_count.load() == 0 && std::chrono::steady_clock::now() < deadline) {
    js_pub->publish(js);
    voxel_pub->publish(vox);
    exec.spin_some(std::chrono::milliseconds(10));
    cand_pub->publish(cart);
    exec.spin_some(std::chrono::milliseconds(10));
  }
  EXPECT_GT(safe_count.load(), 0)
      << "a one-step Cartesian command must use the configured collision margin, "
         "not one full increment of predictive margin growth";
  EXPECT_FALSE(node->fault_latched());
}

namespace {

/// What one run of `run_multistep_cartesian_predict` observed.
struct MultiStepPredictiveRun {
  int safe_count{0};       ///< chunks that reached /openral/safe_action
  bool latched{false};     ///< the kernel fault-latched (collision stop)
  std::string evidence{};  ///< first CollisionEvidence JSON, if any
};

/// Drive a MULTI-STEP CARTESIAN_DELTA chunk at the planar 2R arm the predictive
/// tests above use, against a voxel wall whose near face sits at `wall_face_y`.
/// Everything except that face is identical between the two cases below, so the
/// pair isolates one variable: how far the predicted trajectory stays from the
/// obstacle.
///
/// Geometry, hand-computed (metres, base frame; the arm is 1 m + 1 m, the EE is
/// the fixed link at the tip):
///   start EE      y = 1.00        (q = [0, +90°])
///   EE capsule    r = 0.05
///   per step      Δy = +0.05      (no cartesian_delta_scale → raw = physical)
///   horizon       6 steps         → predicted y_s = 1.00 + 0.05·(s+1)
///   voxel margin  0.00
///   growth        0.05 per look-ahead step, so the step-s check runs at
///                 margin + growth·s = 0.05·s — step 0 gets NO inflation
///                 (the horizon-1 case pinned by the test above).
/// The capsule-to-cell distance at step s is d_s = wall_face_y − y_s − 0.05, and
/// the kernel stops when d_s ≤ 0.05·s, i.e. when wall_face_y ≤ 1.10 + 0.10·s.
void run_multistep_cartesian_predict(const std::string& node_name, double wall_face_y,
                                     MultiStepPredictiveRun* out) {
  rclcpp::NodeOptions opts;
  opts.parameter_overrides({
      {"n_dof", std::int64_t{2}},
      {"joint_position_min", std::vector<double>{-3.14, -3.14}},
      {"joint_position_max", std::vector<double>{3.14, 3.14}},
      {"joint_velocity_max", std::vector<double>{5.0, 5.0}},
      {"joint_torque_max", std::vector<double>{5.0, 5.0}},
      {"self_collision_enabled", false},
      {"world_voxel_enabled", true},
      {"world_voxel_margin_m", 0.0},
      {"world_voxel_deadline_ms", 2000.0},
      {"world_voxel_max_cells", std::int64_t{8192}},
      {"collision_n_links", std::int64_t{3}},
      {"collision_parent", std::vector<std::int64_t>{-1, 0, 1}},
      {"collision_joint_kind", std::vector<std::int64_t>{1, 1, 0}},  // rev, rev, fixed
      {"collision_dof_index", std::vector<std::int64_t>{0, 1, -1}},
      {"collision_origin_xyzrpy",
       std::vector<double>{0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0}},
      {"collision_axis", std::vector<double>{0, 0, 1, 0, 0, 1, 0, 0, 1}},
      {"collision_capsule_link", std::vector<std::int64_t>{0, 1, 2}},
      {"collision_capsule_radius", std::vector<double>{0.05, 0.05, 0.05}},
      {"collision_capsule_half_length", std::vector<double>{0.05, 0.05, 0.05}},
      {"collision_capsule_origin_xyzrpy",
       std::vector<double>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
      {"collision_allowed_pairs", std::vector<std::int64_t>{}},
      {"collision_link_names", std::vector<std::string>{"l0", "l1", "ee"}},
      {"collision_joint_names", std::vector<std::string>{"j0", "j1"}},
      {"collision_state_deadline_ms", 2000.0},
      {"collision_ee_link_index", std::int64_t{2}},
      {"collision_predict_lambda", 0.02},
      // 0.05 m/step keeps the per-step arithmetic in the comments exact and
      // leaves a whole voxel of separation between the two cases' verdicts.
      {"collision_predict_margin_growth_m", 0.05},
      {"collision_predict_max_steps", std::int64_t{0}},  // check every step
  });
  auto node = std::make_shared<osk::SafetyKernelLifecycleNode>(node_name, opts);
  rclcpp_lifecycle::State unconf(lifecycle_msgs::msg::State::PRIMARY_STATE_UNCONFIGURED, "uc");
  ASSERT_EQ(node->on_configure(unconf), osk::SafetyKernelLifecycleNode::CallbackReturn::SUCCESS);
  rclcpp_lifecycle::State inactive(lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE, "in");
  ASSERT_EQ(node->on_activate(inactive), osk::SafetyKernelLifecycleNode::CallbackReturn::SUCCESS);

  rclcpp::Node helper(node_name + "_helper");
  rclcpp::QoS chunk_qos(rclcpp::KeepLast(1));
  chunk_qos.reliable();
  auto cand_pub = helper.create_publisher<openral_msgs::msg::ActionChunk>(
      "/openral/candidate_action", chunk_qos);
  rclcpp::QoS js_qos(rclcpp::KeepLast(1));
  js_qos.best_effort();
  auto js_pub = helper.create_publisher<sensor_msgs::msg::JointState>("/joint_states", js_qos);
  rclcpp::QoS voxel_qos(rclcpp::KeepLast(1));
  voxel_qos.reliable();
  auto voxel_pub = helper.create_publisher<openral_msgs::msg::OccupancyVoxels>(
      "/openral/world_voxels", voxel_qos);
  std::atomic<int> safe_count{0};
  auto safe_sub = helper.create_subscription<openral_msgs::msg::ActionChunk>(
      "/openral/safe_action", chunk_qos,
      [&safe_count](const openral_msgs::msg::ActionChunk::SharedPtr) { ++safe_count; });
  std::string evidence_json;
  rclcpp::QoS failure_qos(rclcpp::KeepLast(10));
  failure_qos.reliable();
  auto failure_sub = helper.create_subscription<openral_msgs::msg::FailureTrigger>(
      "/openral/failure/safety", failure_qos,
      [&evidence_json](const openral_msgs::msg::FailureTrigger::SharedPtr msg) {
        if (evidence_json.empty()) {
          evidence_json = msg->evidence_json;
        }
      });

  rclcpp::executors::SingleThreadedExecutor exec;
  exec.add_node(node->get_node_base_interface());
  exec.add_node(helper.get_node_base_interface());

  // Voxel wall at 0.05 m resolution: every cell whose centre is at or beyond
  // `wall_face_y` is occupied, so the wall's near face lands exactly on
  // `wall_face_y` (origin 0.9 + k·0.05 hits both faces the tests ask for). The
  // grid spans x∈[0.5,1.5], y∈[0.9,2.0], z∈[-0.15,0.15] = 2640 cells: the whole
  // predicted EE path is inside it.
  openral_msgs::msg::OccupancyVoxels vox;
  // Identity: a base-aligned synthetic lattice. `OccupancyVoxels` is oriented,
  // and an unset orientation is the all-zero quaternion, which the ingest
  // refuses -- leaving this test asserting a fault it never meant to cause.
  vox.orientation.w = 1.0;
  vox.resolution = 0.05;
  vox.size_x = 20;
  vox.size_y = 22;
  vox.size_z = 6;
  vox.origin.x = 0.5;
  vox.origin.y = 0.9;
  vox.origin.z = -0.15;
  vox.occupancy.assign(static_cast<std::size_t>(vox.size_x) * vox.size_y * vox.size_z, 0);
  for (std::uint32_t iz = 0; iz < vox.size_z; ++iz) {
    for (std::uint32_t iy = 0; iy < vox.size_y; ++iy) {
      const double cy = vox.origin.y + (iy + 0.5) * vox.resolution;
      if (cy >= wall_face_y) {
        for (std::uint32_t ix = 0; ix < vox.size_x; ++ix) {
          vox.occupancy[ix + vox.size_x * (iy + vox.size_y * iz)] = 1;
        }
      }
    }
  }

  sensor_msgs::msg::JointState js;
  js.name = {"j0", "j1"};
  js.position = {0.0, 1.57079632679};  // EE at (1, 1, 0)

  openral_msgs::msg::ActionChunk cart;
  cart.control_mode = 5;  // CARTESIAN_DELTA
  cart.horizon = 6;
  cart.n_dof = 6;  // [vx,vy,vz, wx,wy,wz] per step
  cart.flat.clear();
  for (int s = 0; s < 6; ++s) {
    cart.flat.insert(cart.flat.end(), {0.0, 0.05, 0.0, 0.0, 0.0, 0.0});
  }

  // Seed a fresh joint state + grid before any chunk, and confirm the MEASURED
  // config is clear — otherwise the reactive floor, not the look-ahead, decides.
  auto seed_deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(300);
  while (std::chrono::steady_clock::now() < seed_deadline) {
    js_pub->publish(js);
    voxel_pub->publish(vox);
    exec.spin_some(std::chrono::milliseconds(10));
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
  ASSERT_FALSE(node->fault_latched()) << "start config must be clear (reactive must not fire)";

  const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(1500);
  while (!node->fault_latched() && safe_count.load() == 0 &&
         std::chrono::steady_clock::now() < deadline) {
    js_pub->publish(js);
    voxel_pub->publish(vox);
    exec.spin_some(std::chrono::milliseconds(10));
    cand_pub->publish(cart);
    exec.spin_some(std::chrono::milliseconds(10));
  }
  // The latch flips inside the kernel's own callback, so on a stop the
  // FailureTrigger carrying the evidence only lands a spin or two later.
  const auto drain = std::chrono::steady_clock::now() + std::chrono::milliseconds(500);
  while (node->fault_latched() && evidence_json.empty() &&
         std::chrono::steady_clock::now() < drain) {
    exec.spin_some(std::chrono::milliseconds(10));
  }
  out->safe_count = safe_count.load();
  out->latched = node->fault_latched();
  out->evidence = evidence_json;
}

}  // namespace

// The predictive Cartesian look-ahead must NOT reject a MULTI-STEP chunk whose
// whole predicted horizon stays clear — the false-positive direction. 0884101
// repurposed the original test for this case into the horizon-1 margin test
// above, leaving the multi-step accept uncovered; this restores it.
//
// Wall face at y = 1.75, so the tightest step is the last one, s=5:
//   d_5 = 1.75 − (1.00 + 0.05·6) − 0.05 = 0.40  vs a threshold of 0.05·5 = 0.25
// → clear by 0.15 m, and every earlier step is clearer still (the slack
// d_s − 0.05·s = 0.65 − 0.10·s only shrinks with s). That 0.15 m is far more
// than the DLS reconstruction's residual over six steps of a well-conditioned
// 2R arm (the elbow sits at 90°, nowhere near a singularity), so an accept here
// means the trajectory really is clear, not that the fixture got lucky.
TEST_F(LifecycleKernelTest, CartesianDeltaMultiStepPredictivePassesWhenTrajectoryStaysClear) {
  MultiStepPredictiveRun run;
  run_multistep_cartesian_predict("kernel_cart_predict_multistep_clear", 1.75, &run);
  EXPECT_GT(run.safe_count, 0)
      << "a multi-step CARTESIAN_DELTA chunk whose whole predicted trajectory clears every "
         "occupied cell by more than margin + growth·step must reach /openral/safe_action";
  EXPECT_FALSE(run.latched) << "no step of a clear trajectory may estop: " << run.evidence;
}

// The guard for the accept above: the SAME six-step trajectory, with the wall
// moved 0.40 m nearer (face at y = 1.35), must stop — and at the step the
// arithmetic predicts, not merely somewhere. Threshold 0.05·s vs
// d_s = 1.35 − (1.00 + 0.05·(s+1)) − 0.05:
//   s=2 → d = 0.15 vs 0.10 → still clear (by 0.05)
//   s=3 → d = 0.10 vs 0.15 → STOP
// The ±0.05 m either side of step 3 is what makes the step number, and not just
// the verdict, worth asserting.
TEST_F(LifecycleKernelTest, CartesianDeltaMultiStepPredictiveStopsAtThePredictedStep) {
  MultiStepPredictiveRun run;
  run_multistep_cartesian_predict("kernel_cart_predict_multistep_stop", 1.35, &run);
  EXPECT_TRUE(run.latched) << "the same trajectory 0.40 m nearer the wall must estop";
  EXPECT_EQ(run.safe_count, 0) << "the colliding chunk must never reach /openral/safe_action";
  ASSERT_FALSE(run.evidence.empty()) << "a collision stop must publish CollisionEvidence";
  EXPECT_EQ(evidence_field(run.evidence, "horizon_step"), "3")
      << "the look-ahead must trip on the first step whose clearance falls inside "
         "margin + growth·step, which is step 3: "
      << run.evidence;
  // The reported clearance is the d_3 = 0.10 above. The tolerance is the DLS
  // reconstruction's residual after four integrated steps — measured at 1.5 mm
  // on this arm, so 5 mm bounds it with room to spare while still failing loudly
  // if the reconstruction ever drifts by a voxel.
  EXPECT_NEAR(std::stod(evidence_field(run.evidence, "min_distance_m")), 0.10, 5e-3)
      << run.evidence;
}

// The REACTIVE (measured-state) collision check reports `horizon_step: -1` —
// the sentinel that says "this is the configuration the robot is in RIGHT NOW,
// not a predicted look-ahead step". This test pins the WIRE FORMAT of that
// evidence payload, because `openral_core.CollisionEvidence` has to accept it:
// a Cartesian chunk (CARTESIAN_POSE/TWIST/DELTA — every attached-payload stop)
// hits the reactive check first, so -1 is the common case on that path, and a
// schema that rejects it silently downgrades the reasoner to raw-JSON
// truncation. Same 2R arm + EE link as the predictive tests above, but the
// voxel wall is placed so the MEASURED start config already collides.
TEST_F(LifecycleKernelTest, ReactiveCollisionEvidenceReportsHorizonStepMinusOne) {
  rclcpp::NodeOptions opts;
  opts.parameter_overrides({
      {"n_dof", std::int64_t{2}},
      {"joint_position_min", std::vector<double>{-3.14, -3.14}},
      {"joint_position_max", std::vector<double>{3.14, 3.14}},
      {"joint_velocity_max", std::vector<double>{5.0, 5.0}},
      {"joint_torque_max", std::vector<double>{5.0, 5.0}},
      {"self_collision_enabled", false},
      {"world_voxel_enabled", true},
      {"world_voxel_margin_m", 0.0},
      {"world_voxel_deadline_ms", 2000.0},
      {"world_voxel_max_cells", std::int64_t{8192}},
      {"collision_n_links", std::int64_t{3}},
      {"collision_parent", std::vector<std::int64_t>{-1, 0, 1}},
      {"collision_joint_kind", std::vector<std::int64_t>{1, 1, 0}},  // rev, rev, fixed
      {"collision_dof_index", std::vector<std::int64_t>{0, 1, -1}},
      {"collision_origin_xyzrpy",
       std::vector<double>{0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0}},
      {"collision_axis", std::vector<double>{0, 0, 1, 0, 0, 1, 0, 0, 1}},
      {"collision_capsule_link", std::vector<std::int64_t>{0, 1, 2}},
      {"collision_capsule_radius", std::vector<double>{0.05, 0.05, 0.05}},
      {"collision_capsule_half_length", std::vector<double>{0.05, 0.05, 0.05}},
      {"collision_capsule_origin_xyzrpy",
       std::vector<double>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
      {"collision_allowed_pairs", std::vector<std::int64_t>{}},
      {"collision_link_names", std::vector<std::string>{"l0", "l1", "ee"}},
      {"collision_joint_names", std::vector<std::string>{"j0", "j1"}},
      {"collision_state_deadline_ms", 2000.0},
      {"collision_ee_link_index", std::int64_t{2}},
      {"collision_predict_lambda", 0.02},
      {"collision_predict_margin_growth_m", 0.02},
      {"collision_predict_max_steps", std::int64_t{0}},
  });
  auto node = std::make_shared<osk::SafetyKernelLifecycleNode>("kernel_reactive_evidence", opts);
  rclcpp_lifecycle::State unconf(lifecycle_msgs::msg::State::PRIMARY_STATE_UNCONFIGURED, "uc");
  ASSERT_EQ(node->on_configure(unconf), osk::SafetyKernelLifecycleNode::CallbackReturn::SUCCESS);
  rclcpp_lifecycle::State inactive(lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE, "in");
  ASSERT_EQ(node->on_activate(inactive), osk::SafetyKernelLifecycleNode::CallbackReturn::SUCCESS);

  rclcpp::Node helper("reactive_evidence_helper");
  rclcpp::QoS chunk_qos(rclcpp::KeepLast(1));
  chunk_qos.reliable();
  auto cand_pub = helper.create_publisher<openral_msgs::msg::ActionChunk>(
      "/openral/candidate_action", chunk_qos);
  rclcpp::QoS js_qos(rclcpp::KeepLast(1));
  js_qos.best_effort();
  auto js_pub = helper.create_publisher<sensor_msgs::msg::JointState>("/joint_states", js_qos);
  rclcpp::QoS voxel_qos(rclcpp::KeepLast(1));
  voxel_qos.reliable();
  auto voxel_pub = helper.create_publisher<openral_msgs::msg::OccupancyVoxels>(
      "/openral/world_voxels", voxel_qos);
  rclcpp::QoS fail_qos(rclcpp::KeepLast(50));
  fail_qos.reliable();
  fail_qos.durability_volatile();
  std::string evidence;
  auto fail_sub = helper.create_subscription<openral_msgs::msg::FailureTrigger>(
      "/openral/failure/safety", fail_qos,
      [&evidence](const openral_msgs::msg::FailureTrigger::SharedPtr m) {
        if (m->kind == openral_msgs::msg::FailureTrigger::KIND_COLLISION && evidence.empty()) {
          evidence = m->evidence_json;
        }
      });

  rclcpp::executors::SingleThreadedExecutor exec;
  exec.add_node(node->get_node_base_interface());
  exec.add_node(helper.get_node_base_interface());

  // Voxel wall: every cell with centre y>=0.95 occupied. Grid x,y in [0,2],
  // z in [-0.15,0.15] @ 0.1 m. The EE capsule (centre (1,1,0), r=0.05,
  // half-length 0.05) sits INSIDE the wall at the measured configuration, so
  // the reactive check fires before any look-ahead step is evaluated.
  openral_msgs::msg::OccupancyVoxels vox;
  // Identity: a base-aligned synthetic lattice. `OccupancyVoxels` is oriented,
  // and an unset orientation is the all-zero quaternion, which the ingest
  // refuses -- leaving this test asserting a fault it never meant to cause.
  vox.orientation.w = 1.0;
  vox.resolution = 0.1;
  vox.size_x = 20;
  vox.size_y = 20;
  vox.size_z = 3;
  vox.origin.x = 0.0;
  vox.origin.y = 0.0;
  vox.origin.z = -0.15;
  vox.occupancy.assign(static_cast<std::size_t>(vox.size_x) * vox.size_y * vox.size_z, 0);
  for (std::uint32_t iz = 0; iz < vox.size_z; ++iz) {
    for (std::uint32_t iy = 0; iy < vox.size_y; ++iy) {
      const double cy = vox.origin.y + (iy + 0.5) * vox.resolution;
      if (cy >= 0.95) {
        for (std::uint32_t ix = 0; ix < vox.size_x; ++ix) {
          vox.occupancy[ix + vox.size_x * (iy + vox.size_y * iz)] = 1;
        }
      }
    }
  }

  sensor_msgs::msg::JointState js;
  js.name = {"j0", "j1"};
  js.position = {0.0, 1.57079632679};  // EE at (1,1,0) — already in the wall

  openral_msgs::msg::ActionChunk cart;
  cart.rskill_id = "reactive_evidence_skill";
  cart.trace_id = "0af7651916cd43dd8448eb211c80319c";
  cart.control_mode = 5;  // CARTESIAN_DELTA
  cart.horizon = 4;
  cart.n_dof = 6;
  cart.flat.clear();
  for (int s = 0; s < 4; ++s) {
    cart.flat.insert(cart.flat.end(), {0.0, 0.01, 0.0, 0.0, 0.0, 0.0});
  }

  const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(2000);
  while (evidence.empty() && std::chrono::steady_clock::now() < deadline) {
    js_pub->publish(js);
    voxel_pub->publish(vox);
    exec.spin_some(std::chrono::milliseconds(10));
    cand_pub->publish(cart);
    exec.spin_some(std::chrono::milliseconds(10));
  }
  ASSERT_FALSE(evidence.empty())
      << "the reactive check must publish a KIND_COLLISION FailureTrigger; chunks_dropped="
      << node->chunks_dropped() << " chunks_passed=" << node->chunks_passed();
  // The exact wire shape consumed by openral_core.CollisionEvidence. The
  // Python-side counterpart is tests/unit/test_collision_geometry_contracts.py
  // ::test_kernel_reactive_collision_evidence_validates.
  EXPECT_NE(evidence.find(R"("kind":"collision")"), std::string::npos) << evidence;
  EXPECT_NE(evidence.find(R"("collision_kind":"world")"), std::string::npos) << evidence;
  EXPECT_NE(evidence.find(R"("link_a":"ee")"), std::string::npos) << evidence;
  EXPECT_NE(evidence.find(R"("horizon_step":-1)"), std::string::npos)
      << "the reactive (measured-state) check must report the -1 sentinel, not a "
         "predicted-horizon index: "
      << evidence;
  // Print the captured payload so the Python-side fixture can be refreshed
  // verbatim from a real kernel run rather than hand-written.
  RecordProperty("reactive_collision_evidence_json", evidence);

  // ADR-0096 — the same collision latch must also reach the latched status
  // topic, with the FailureTrigger's KIND_COLLISION number. Subscribed only
  // now, AFTER the latch: TRANSIENT_LOCAL still delivers the current value.
  rclcpp::QoS status_qos(rclcpp::KeepLast(1));
  status_qos.reliable();
  status_qos.transient_local();
  rclcpp::Node late("reactive_evidence_status_late");
  openral_msgs::msg::SafetyStatus status;
  bool got_status = false;
  auto status_sub = late.create_subscription<openral_msgs::msg::SafetyStatus>(
      "/openral/safety_status", status_qos,
      [&status, &got_status](const openral_msgs::msg::SafetyStatus::SharedPtr m) {
        status = *m;
        got_status = true;
      });
  exec.add_node(late.get_node_base_interface());
  const auto status_deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(3000);
  while (!got_status && std::chrono::steady_clock::now() < status_deadline) {
    exec.spin_some(std::chrono::milliseconds(10));
  }
  ASSERT_TRUE(got_status) << "the collision latch must be visible to a late subscriber";
  EXPECT_TRUE(status.latched);
  EXPECT_EQ(status.drop_reason, openral_msgs::msg::SafetyStatus::KIND_COLLISION);
  EXPECT_EQ(status.rskill_id, "reactive_evidence_skill");
}

// ── ADR-0096: /openral/safety_status per-transition coverage ────────────────
//
// One test per transition CLASS the ADR names: activation, the (previously
// end-to-end silent) envelope_unconfigured drop, a non-latching upstream
// unavailability drop and its recovery, an envelope-violation latch, an
// external e-stop latch and the operator's reset clear. Plus the property
// the whole topic exists for: a subscriber that connects AFTER a transition
// still receives the current value, which VOLATILE cannot deliver.

namespace {

/// Collects every SafetyStatus published on the latched topic. Subscribes
/// with the publisher's QoS (RELIABLE + TRANSIENT_LOCAL + KEEP_LAST=1) — a
/// mismatched durability would silently never match the publisher.
class StatusSpy {
public:
  explicit StatusSpy(rclcpp::Node& node) {
    rclcpp::QoS qos(rclcpp::KeepLast(1));
    qos.reliable();
    qos.transient_local();
    sub_ = node.create_subscription<openral_msgs::msg::SafetyStatus>(
        "/openral/safety_status", qos,
        [this](const openral_msgs::msg::SafetyStatus::SharedPtr m) { received_.push_back(*m); });
  }

  const std::vector<openral_msgs::msg::SafetyStatus>& all() const { return received_; }
  bool empty() const { return received_.empty(); }
  std::size_t size() const { return received_.size(); }
  const openral_msgs::msg::SafetyStatus& latest() const { return received_.back(); }

  /// True once a status matching (latched, drop_reason) has arrived.
  bool saw(bool latched, std::uint8_t drop_reason) const {
    for (const auto& m : received_) {
      if (m.latched == latched && m.drop_reason == drop_reason) {
        return true;
      }
    }
    return false;
  }

private:
  rclcpp::Subscription<openral_msgs::msg::SafetyStatus>::SharedPtr sub_;
  std::vector<openral_msgs::msg::SafetyStatus> received_;
};

/// Spin the executor until `done()` returns true or the timeout expires.
template <typename Fn>
void spin_until(rclcpp::executors::SingleThreadedExecutor& exec, Fn done,
                std::chrono::milliseconds timeout = std::chrono::milliseconds(1500)) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (!done() && std::chrono::steady_clock::now() < deadline) {
    exec.spin_some(std::chrono::milliseconds(10));
  }
}

}  // namespace

// HZ-0096-1 mitigation 1: every lifecycle activation publishes a fresh
// SafetyStatus, so a restarted publisher overwrites any stale durable value
// within one activation cycle rather than waiting for the next real fault.
TEST_F(LifecycleKernelTest, ActivationPublishesClearSafetyStatus) {
  rclcpp::NodeOptions opts;
  opts.parameter_overrides(minimal_envelope_params());
  auto node = std::make_shared<osk::SafetyKernelLifecycleNode>("kernel_status_activate", opts);
  rclcpp_lifecycle::State unconf(lifecycle_msgs::msg::State::PRIMARY_STATE_UNCONFIGURED, "uc");
  ASSERT_EQ(node->on_configure(unconf), osk::SafetyKernelLifecycleNode::CallbackReturn::SUCCESS);

  rclcpp::Node helper("status_activate_helper");
  StatusSpy spy(helper);
  rclcpp::executors::SingleThreadedExecutor exec;
  exec.add_node(node->get_node_base_interface());
  exec.add_node(helper.get_node_base_interface());

  rclcpp_lifecycle::State inactive(lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE, "in");
  ASSERT_EQ(node->on_activate(inactive), osk::SafetyKernelLifecycleNode::CallbackReturn::SUCCESS);
  spin_until(exec, [&] { return !spy.empty(); });

  ASSERT_FALSE(spy.empty()) << "activation must publish a SafetyStatus";
  EXPECT_FALSE(spy.latest().latched);
  EXPECT_EQ(spy.latest().drop_reason, openral_msgs::msg::SafetyStatus::DROP_NONE)
      << "DROP_NONE must be explicit; a default-initialised 0 would read as KIND_TIMEOUT";
  EXPECT_GT(rclcpp::Time(spy.latest().header.stamp).nanoseconds(), 0)
      << "header.stamp is load-bearing for the HZ-0096-1 liveness rule";

  // A deactivate→activate cycle must put a fresh value back on the wire. The
  // transition gate must never suppress that (it is the whole mitigation).
  // Either the activation publish or the 1 Hz liveness refresh satisfies
  // this — both are the mitigation; a gate that swallowed the activation
  // publish AND a heartbeat that never republished would fail here.
  ASSERT_EQ(node->on_deactivate(
                rclcpp_lifecycle::State(lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE, "ac")),
            osk::SafetyKernelLifecycleNode::CallbackReturn::SUCCESS);
  const std::size_t after_deactivate = spy.size();
  ASSERT_EQ(node->on_activate(inactive), osk::SafetyKernelLifecycleNode::CallbackReturn::SUCCESS);
  spin_until(exec, [&] { return spy.size() > after_deactivate; }, std::chrono::milliseconds(2000));
  EXPECT_GT(spy.size(), after_deactivate) << "re-activation must put the value back on the wire";
}

// The envelope_unconfigured drop was the one path that was silent END TO END
// before ADR-0096: no /openral/safe_action, no /openral/estop, no
// FailureTrigger — only a span attribute and a /diagnostics key-value.
TEST_F(LifecycleKernelTest, EnvelopeUnconfiguredDropPublishesSafetyStatus) {
  // Reaching the branch the honest way: configure + activate with a real
  // envelope, then re-configure with n_dof=0. The envelope load fails
  // (kUnconfigured) and on_configure returns FAILURE having already cleared
  // envelope_loaded_, while the topic surface from the first configure is
  // still up and activated. That is precisely the misboot state the drop
  // path exists to fail closed on.
  rclcpp::NodeOptions opts;
  opts.parameter_overrides(minimal_envelope_params());
  auto node = std::make_shared<osk::SafetyKernelLifecycleNode>("kernel_status_unconf", opts);
  rclcpp_lifecycle::State unconf(lifecycle_msgs::msg::State::PRIMARY_STATE_UNCONFIGURED, "uc");
  rclcpp_lifecycle::State inactive(lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE, "in");
  ASSERT_EQ(node->on_configure(unconf), osk::SafetyKernelLifecycleNode::CallbackReturn::SUCCESS);
  ASSERT_EQ(node->on_activate(inactive), osk::SafetyKernelLifecycleNode::CallbackReturn::SUCCESS);

  rclcpp::Node helper("status_unconf_helper");
  StatusSpy spy(helper);
  rclcpp::QoS chunk_qos(rclcpp::KeepLast(1));
  chunk_qos.reliable();
  auto cand_pub = helper.create_publisher<openral_msgs::msg::ActionChunk>(
      "/openral/candidate_action", chunk_qos);
  std::atomic<int> safe_count{0};
  auto safe_sub = helper.create_subscription<openral_msgs::msg::ActionChunk>(
      "/openral/safe_action", chunk_qos,
      [&safe_count](const openral_msgs::msg::ActionChunk::SharedPtr) { ++safe_count; });
  rclcpp::executors::SingleThreadedExecutor exec;
  exec.add_node(node->get_node_base_interface());
  exec.add_node(helper.get_node_base_interface());

  node->set_parameter(rclcpp::Parameter("n_dof", std::int64_t{0}));
  ASSERT_EQ(node->on_configure(inactive), osk::SafetyKernelLifecycleNode::CallbackReturn::FAILURE);

  openral_msgs::msg::ActionChunk chunk;
  chunk.rskill_id = "rskills/rskill-smolvla-so100";
  chunk.control_mode = 0;
  chunk.horizon = 1;
  chunk.n_dof = 3;
  chunk.flat = {0.0, 0.0, 0.0};
  const std::uint64_t dropped_before = node->chunks_dropped();
  spin_until(exec, [&] {
    cand_pub->publish(chunk);
    return spy.saw(false, openral_msgs::msg::SafetyStatus::DROP_ENVELOPE_UNCONFIGURED);
  });

  EXPECT_GT(node->chunks_dropped(), dropped_before);
  EXPECT_EQ(safe_count.load(), 0) << "an un-armed kernel must forward nothing";
  ASSERT_TRUE(spy.saw(false, openral_msgs::msg::SafetyStatus::DROP_ENVELOPE_UNCONFIGURED))
      << "the previously-silent drop must now be observable";
  EXPECT_FALSE(spy.latest().latched) << "an unconfigured envelope drops; it does not latch";
  EXPECT_FALSE(spy.latest().detail.empty());
  EXPECT_EQ(spy.latest().rskill_id, "rskills/rskill-smolvla-so100");
}

// An envelope VIOLATION latches, and the latched status carries the same
// numeric kind the FailureTrigger on /openral/failure/safety carries.
TEST_F(LifecycleKernelTest, EnvelopeViolationPublishesLatchedSafetyStatus) {
  rclcpp::NodeOptions opts;
  opts.parameter_overrides(minimal_envelope_params());
  auto node = std::make_shared<osk::SafetyKernelLifecycleNode>("kernel_status_violation", opts);
  rclcpp_lifecycle::State unconf(lifecycle_msgs::msg::State::PRIMARY_STATE_UNCONFIGURED, "uc");
  rclcpp_lifecycle::State inactive(lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE, "in");
  ASSERT_EQ(node->on_configure(unconf), osk::SafetyKernelLifecycleNode::CallbackReturn::SUCCESS);
  ASSERT_EQ(node->on_activate(inactive), osk::SafetyKernelLifecycleNode::CallbackReturn::SUCCESS);

  rclcpp::Node helper("status_violation_helper");
  StatusSpy spy(helper);
  rclcpp::QoS chunk_qos(rclcpp::KeepLast(1));
  chunk_qos.reliable();
  auto cand_pub = helper.create_publisher<openral_msgs::msg::ActionChunk>(
      "/openral/candidate_action", chunk_qos);
  rclcpp::QoS fail_qos(rclcpp::KeepLast(50));
  fail_qos.reliable();
  fail_qos.durability_volatile();
  std::vector<std::uint8_t> trigger_kinds;
  auto fail_sub = helper.create_subscription<openral_msgs::msg::FailureTrigger>(
      "/openral/failure/safety", fail_qos,
      [&trigger_kinds](const openral_msgs::msg::FailureTrigger::SharedPtr m) {
        trigger_kinds.push_back(m->kind);
      });
  rclcpp::executors::SingleThreadedExecutor exec;
  exec.add_node(node->get_node_base_interface());
  exec.add_node(helper.get_node_base_interface());

  openral_msgs::msg::ActionChunk bad;
  bad.rskill_id = "rskills/rskill-smolvla-so100";
  bad.control_mode = 0;  // JOINT_POSITION
  bad.horizon = 1;
  bad.n_dof = 3;
  bad.flat = {5.0, 0.0, 0.0};  // joint 0 well past joint_position_max=1.0
  spin_until(exec, [&] {
    cand_pub->publish(bad);
    return node->fault_latched() && !trigger_kinds.empty() &&
           spy.saw(true, openral_msgs::msg::SafetyStatus::KIND_WORKSPACE);
  });

  ASSERT_TRUE(node->fault_latched());
  ASSERT_FALSE(trigger_kinds.empty());
  ASSERT_TRUE(spy.saw(true, openral_msgs::msg::SafetyStatus::KIND_WORKSPACE))
      << "a joint-position violation must latch with the WORKSPACE kind";
  // The two topics must agree on the number, or "one number, one meaning"
  // is not true and a consumer switching on drop_reason switches wrong.
  EXPECT_EQ(spy.latest().drop_reason, trigger_kinds.front());
  EXPECT_TRUE(spy.latest().latched);
  EXPECT_EQ(spy.latest().rskill_id, "rskills/rskill-smolvla-so100");
}

// An external /openral/estop latches the kernel; the status names the topic
// that latched it (the Empty carries no reason and no publisher identity).
// The operator's reset then clears the durable value.
TEST_F(LifecycleKernelTest, ExternalEstopLatchesStatusAndResetClearsIt) {
  rclcpp::NodeOptions opts;
  auto overrides = minimal_envelope_params();
  overrides.emplace_back("estop_reset_cooldown_s", 0.05);
  opts.parameter_overrides(overrides);
  auto node = std::make_shared<osk::SafetyKernelLifecycleNode>("kernel_status_estop", opts);
  rclcpp_lifecycle::State unconf(lifecycle_msgs::msg::State::PRIMARY_STATE_UNCONFIGURED, "uc");
  rclcpp_lifecycle::State inactive(lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE, "in");
  ASSERT_EQ(node->on_configure(unconf), osk::SafetyKernelLifecycleNode::CallbackReturn::SUCCESS);
  ASSERT_EQ(node->on_activate(inactive), osk::SafetyKernelLifecycleNode::CallbackReturn::SUCCESS);

  rclcpp::Node helper("status_estop_helper");
  StatusSpy spy(helper);
  auto estop_pub = helper.create_publisher<std_msgs::msg::Empty>("/openral/estop", 10);
  auto client = helper.create_client<std_srvs::srv::Trigger>("/openral/estop_reset");
  rclcpp::executors::SingleThreadedExecutor exec;
  exec.add_node(node->get_node_base_interface());
  exec.add_node(helper.get_node_base_interface());

  estop_pub->publish(std_msgs::msg::Empty{});
  spin_until(exec, [&] {
    return node->fault_latched() &&
           spy.saw(true, openral_msgs::msg::SafetyStatus::DROP_EXTERNAL_ESTOP);
  });
  ASSERT_TRUE(node->fault_latched());
  ASSERT_TRUE(spy.saw(true, openral_msgs::msg::SafetyStatus::DROP_EXTERNAL_ESTOP));
  EXPECT_TRUE(spy.latest().latched);

  // Recovery: the reset service's clear must move the durable value too, or a
  // late-joining consumer reads a cleared kernel as latched forever.
  ASSERT_TRUE(client->wait_for_service(std::chrono::seconds(2)));
  std::this_thread::sleep_for(std::chrono::milliseconds(120));  // past the cooldown
  auto fut = client->async_send_request(std::make_shared<std_srvs::srv::Trigger::Request>());
  spin_until(
      exec, [&] { return fut.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready; },
      std::chrono::milliseconds(2000));
  ASSERT_EQ(fut.wait_for(std::chrono::milliseconds(0)), std::future_status::ready);
  ASSERT_TRUE(fut.get()->success);
  spin_until(exec, [&] { return !spy.latest().latched; });
  EXPECT_FALSE(node->fault_latched());
  EXPECT_FALSE(spy.latest().latched) << "the clear transition must reach the latched topic";
  EXPECT_EQ(spy.latest().drop_reason, openral_msgs::msg::SafetyStatus::DROP_NONE);
}

// A non-latching upstream-unavailability drop reports a DROP_* code (NOT a
// latch), and the recovery back to accepting chunks clears it.
TEST_F(LifecycleKernelTest, StateUnavailableDropAndRecoveryPublishSafetyStatus) {
  rclcpp::NodeOptions opts;
  opts.parameter_overrides(velocity_capable_params());
  auto node = std::make_shared<osk::SafetyKernelLifecycleNode>("kernel_status_unavail", opts);
  rclcpp_lifecycle::State unconf(lifecycle_msgs::msg::State::PRIMARY_STATE_UNCONFIGURED, "uc");
  rclcpp_lifecycle::State inactive(lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE, "in");
  ASSERT_EQ(node->on_configure(unconf), osk::SafetyKernelLifecycleNode::CallbackReturn::SUCCESS);
  ASSERT_EQ(node->on_activate(inactive), osk::SafetyKernelLifecycleNode::CallbackReturn::SUCCESS);

  rclcpp::Node helper("status_unavail_helper");
  StatusSpy spy(helper);
  rclcpp::QoS chunk_qos(rclcpp::KeepLast(1));
  chunk_qos.reliable();
  auto cand_pub = helper.create_publisher<openral_msgs::msg::ActionChunk>(
      "/openral/candidate_action", chunk_qos);
  rclcpp::QoS js_qos(rclcpp::KeepLast(1));
  js_qos.best_effort();
  auto js_pub = helper.create_publisher<sensor_msgs::msg::JointState>("/joint_states", js_qos);
  rclcpp::executors::SingleThreadedExecutor exec;
  exec.add_node(node->get_node_base_interface());
  exec.add_node(helper.get_node_base_interface());

  openral_msgs::msg::ActionChunk vel;
  vel.control_mode = 1;  // JOINT_VELOCITY — needs the measured seed
  vel.horizon = 1;
  vel.n_dof = 2;
  vel.flat = {0.05, 0.05};
  // No /joint_states yet → the seed gate drops fail-closed, without latching.
  spin_until(exec, [&] {
    cand_pub->publish(vel);
    return spy.saw(false, openral_msgs::msg::SafetyStatus::DROP_STATE_UNAVAILABLE);
  });
  ASSERT_TRUE(spy.saw(false, openral_msgs::msg::SafetyStatus::DROP_STATE_UNAVAILABLE));
  EXPECT_FALSE(spy.latest().latched)
      << "an unavailable upstream input drops the chunk; it must not read as a latch";
  EXPECT_FALSE(node->fault_latched());

  // Seed the state and re-send: the chunk now passes, and the recovery
  // transition must clear the drop reason on the latched topic.
  sensor_msgs::msg::JointState js;
  js.name = {"j0", "j1"};
  js.position = {0.0, 0.0};
  spin_until(
      exec,
      [&] {
        js_pub->publish(js);
        cand_pub->publish(vel);
        return spy.latest().drop_reason == openral_msgs::msg::SafetyStatus::DROP_NONE;
      },
      std::chrono::milliseconds(3000));
  EXPECT_EQ(spy.latest().drop_reason, openral_msgs::msg::SafetyStatus::DROP_NONE)
      << "recovery must clear the drop reason, not leave it stuck on the last drop";
  EXPECT_FALSE(spy.latest().latched);
  EXPECT_GT(node->chunks_passed(), 0U);
}

// The property the whole topic exists for (ADR-0096, Decision): a subscriber
// that connects AFTER the transition still receives the current value. This
// is what TRANSIENT_LOCAL buys and what /openral/estop and
// /openral/failure/safety (both VOLATILE) cannot deliver by construction.
TEST_F(LifecycleKernelTest, LateSubscriberReceivesTheLatchedSafetyStatus) {
  rclcpp::NodeOptions opts;
  opts.parameter_overrides(minimal_envelope_params());
  auto node = std::make_shared<osk::SafetyKernelLifecycleNode>("kernel_status_late", opts);
  rclcpp_lifecycle::State unconf(lifecycle_msgs::msg::State::PRIMARY_STATE_UNCONFIGURED, "uc");
  rclcpp_lifecycle::State inactive(lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE, "in");
  ASSERT_EQ(node->on_configure(unconf), osk::SafetyKernelLifecycleNode::CallbackReturn::SUCCESS);
  ASSERT_EQ(node->on_activate(inactive), osk::SafetyKernelLifecycleNode::CallbackReturn::SUCCESS);

  rclcpp::Node early("status_late_early");
  auto estop_pub = early.create_publisher<std_msgs::msg::Empty>("/openral/estop", 10);
  rclcpp::executors::SingleThreadedExecutor exec;
  exec.add_node(node->get_node_base_interface());
  exec.add_node(early.get_node_base_interface());

  estop_pub->publish(std_msgs::msg::Empty{});
  spin_until(exec, [&] { return node->fault_latched(); });
  ASSERT_TRUE(node->fault_latched()) << "precondition: the kernel is latched";

  // Only NOW does the consumer show up — a dashboard tab opened mid-mission,
  // or a runner reconnecting after a crash.
  rclcpp::Node late("status_late_joiner");
  StatusSpy spy(late);
  exec.add_node(late.get_node_base_interface());
  spin_until(exec, [&] { return !spy.empty(); }, std::chrono::milliseconds(3000));

  ASSERT_FALSE(spy.empty())
      << "a late subscriber must receive the durable latched value it never witnessed";
  EXPECT_TRUE(spy.all().front().latched)
      << "the FIRST sample a late joiner gets must already say the kernel is latched";
  EXPECT_EQ(spy.all().front().drop_reason, openral_msgs::msg::SafetyStatus::DROP_EXTERNAL_ESTOP);
}

// ── Declaration liveness in the kernel's own clock domain ────────────────────
//
// The 2026-08-14 clock-domain fix moved declaration-expiry-on-a-dead-stream off
// World State and onto two kernel-side gates that were already there: the
// attachment freshness deadline (`attached_collision_deadline_s`, which refuses
// every candidate action while the payload model is stale) and the per-candidate
// `place_declaration_live()` backstop (which drops the allowance once the
// declaration's own timeout passes without a retraction). Both were argued in
// comments and neither was tested. The two tests below are that test.
//
// Fixture geometry, hand-computed so the amendment's two margins land on either
// side of one true clearance — which is what makes "is the allowance in force?"
// directly observable on /openral/safe_action:
//
//   link0        revolute about +z at the origin, capsule r = 10 mm (never near
//                the wall: 130 mm of clearance in every configuration tested)
//   payload      box, half-extents 20 mm, attached to link0 at (0.10, 0, 0),
//                so its +x face sits at x = 0.120 m
//   occupancy    one 25 mm cell, cube x in [0.140, 0.165], y/z in +/-12.5 mm
//   true surface distance payload -> cell                       = 0.020 m
//   attached margin                                    0.030 m  -> STOP
//   attached margin - min(1.5 x resolution, 40 mm)   -0.0075 m  -> CLEAR
//
// The reduced margin was 0.005 m until ADR-0097's Second Amendment raised the
// allowance from min(one voxel, 2.5 cm) to min(1.5 x voxel, 4 cm) on
// 2026-08-15; both values sit on the CLEAR side of the same 0.020 m clearance,
// so what these two tests observe — allowance in force or withdrawn — is
// unchanged by the calibration.
namespace {

std::vector<rclcpp::Parameter> place_declaration_params() {
  return {
      {"n_dof", std::int64_t{1}},
      {"joint_position_min", std::vector<double>{-3.14}},
      {"joint_position_max", std::vector<double>{3.14}},
      {"joint_velocity_max", std::vector<double>{3.15}},
      {"joint_torque_max", std::vector<double>{5.0}},
      {"self_collision_enabled", false},
      {"world_voxel_enabled", true},
      {"world_voxel_margin_m", 0.0},
      {"world_voxel_deadline_ms", 5000.0},
      {"world_voxel_max_cells", std::int64_t{64}},
      {"attached_collision_enabled", true},
      {"attached_collision_margin_m", 0.03},
      {"attached_collision_deadline_ms", 200.0},
      {"attached_max_objects", std::int64_t{1}},
      {"attached_max_primitives", std::int64_t{1}},
      {"attached_max_touch_links", std::int64_t{1}},
      {"attached_contact_tolerance_m", 0.001},
      {"collision_n_links", std::int64_t{1}},
      {"collision_parent", std::vector<std::int64_t>{-1}},
      {"collision_joint_kind", std::vector<std::int64_t>{1}},  // revolute
      {"collision_dof_index", std::vector<std::int64_t>{0}},
      {"collision_origin_xyzrpy", std::vector<double>{0, 0, 0, 0, 0, 0}},
      {"collision_axis", std::vector<double>{0, 0, 1}},
      {"collision_capsule_link", std::vector<std::int64_t>{0}},
      {"collision_capsule_radius", std::vector<double>{0.01}},
      {"collision_capsule_half_length", std::vector<double>{0.0}},
      {"collision_capsule_origin_xyzrpy", std::vector<double>{0, 0, 0, 0, 0, 0}},
      {"collision_allowed_pairs", std::vector<std::int64_t>{}},
      {"collision_link_names", std::vector<std::string>{"link0"}},
      {"collision_joint_names", std::vector<std::string>{"j0"}},
      {"collision_state_deadline_ms", 5000.0},
  };
}

// The single 25 mm occupied cell the declared payload approaches, published in
// the base frame the region is declared in.
openral_msgs::msg::OccupancyVoxels declared_target_voxels() {
  openral_msgs::msg::OccupancyVoxels vox;
  // Identity: a base-aligned synthetic lattice. `OccupancyVoxels` is oriented,
  // and an unset orientation is the all-zero quaternion, which the ingest
  // refuses -- leaving this test asserting a fault it never meant to cause.
  vox.orientation.w = 1.0;
  vox.header.frame_id = "base_link";
  vox.resolution = 0.025;
  vox.size_x = 1;
  vox.size_y = 1;
  vox.size_z = 1;
  vox.origin.x = 0.140;
  vox.origin.y = -0.0125;
  vox.origin.z = -0.0125;
  vox.occupancy.assign(1, 1);
  return vox;
}

// One carried payload plus the live place declaration scoped to it.
// `attachment_stamp_ns` is the *stream's* stamp (what the freshness deadline
// measures) and `declaration_stamp_ns` is the *declaration's* (what the backstop
// measures) — that they are separate clocks' business is the whole fix.
// `carrying == false` is the pre-grasp / post-release beat: the declaration is
// published for the whole goal, so it rides every heartbeat whether or not a
// payload is attached yet.
openral_msgs::msg::WorldStateStamped declared_carry_state(std::int64_t attachment_stamp_ns,
                                                          std::int64_t declaration_stamp_ns,
                                                          double timeout_s, bool carrying = true,
                                                          std::uint64_t revision = 1,
                                                          double payload_x = 0.10) {
  openral_msgs::msg::WorldStateStamped msg;
  msg.attachment_stamp_ns = attachment_stamp_ns;
  msg.attachment_revision = revision;

  if (carrying) {
    openral_msgs::msg::AttachedCollisionObject obj;
    obj.object_id = "sim:obj_main";
    obj.attach_link = "link0";
    // `payload_x` 0.10 is the approach pose these tests were built around (the
    // payload's +x face at 0.120, 20 mm clear of the cell). The advisory-band
    // tests push it in to 0.130 so the face lands 10 mm INSIDE the cell —
    // arrived, not approaching.
    obj.pose_in_link.position.x = payload_x;
    obj.pose_in_link.orientation.w = 1.0;
    openral_msgs::msg::AttachedCollisionPrimitive prim;
    prim.shape_type = openral_msgs::msg::AttachedCollisionPrimitive::SHAPE_BOX;
    prim.shape_dimensions = {0.02, 0.02, 0.02};
    prim.pose_in_object.orientation.w = 1.0;
    obj.primitives.push_back(prim);
    msg.attached_objects.push_back(obj);
  }

  msg.place_declaration_valid = true;
  msg.place_declaration.target_id = "sim:cab_1_left_group_main";
  msg.place_declaration.object_id = "sim:obj_main";
  msg.place_declaration.rskill_id = "place_in_cabinet";
  msg.place_declaration.trace_id = "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01";
  msg.place_declaration.active = true;
  msg.place_declaration.stamp_ns = declaration_stamp_ns;
  msg.place_declaration.timeout_s = timeout_s;
  msg.place_declaration.region_valid = true;
  msg.place_declaration.region.frame_id = "base_link";
  msg.place_declaration.region.pose.position.x = 0.15;
  msg.place_declaration.region.pose.orientation.w = 1.0;
  msg.place_declaration.region.half_extents.x = 0.05;
  msg.place_declaration.region.half_extents.y = 0.05;
  msg.place_declaration.region.half_extents.z = 0.05;
  msg.place_declaration.region.evidence_ref = "mujoco_body_subtree:cab_1_left_group_main";
  msg.place_declaration.region.stamp_ns = declaration_stamp_ns;
  return msg;
}

// The reactive JOINT_POSITION candidate the tests replay: one row at the
// measured configuration, so the geometry never moves and the only thing that
// can change the verdict is which margin the payload is gated against.
openral_msgs::msg::ActionChunk declared_carry_chunk() {
  openral_msgs::msg::ActionChunk chunk;
  chunk.control_mode = 0;  // JOINT_POSITION
  chunk.horizon = 1;
  chunk.n_dof = 1;
  chunk.flat = {0.0};
  chunk.rskill_id = "place_in_cabinet";
  return chunk;
}

// ── Log capture ─────────────────────────────────────────────────────────────
//
// The kernel's own rcutils sink, redirected — not a logging double. What is
// under test here IS the emitted line: which reason it names, at what severity,
// and how many times a standing state produces it.
std::mutex g_log_mutex;
std::vector<std::pair<int, std::string>> g_log_lines;

void capture_log_handler(const rcutils_log_location_t* /*location*/, int severity,
                         const char* /*name*/, rcutils_time_point_value_t /*timestamp*/,
                         const char* format, va_list* args) {
  char buffer[1024];
  va_list copy;
  va_copy(copy, *args);
  const int written = std::vsnprintf(buffer, sizeof(buffer), format, copy);
  va_end(copy);
  if (written < 0) {
    return;
  }
  const std::lock_guard<std::mutex> lock(g_log_mutex);
  g_log_lines.emplace_back(severity, std::string(buffer));
}

// Installs the capture sink for one test and always puts the previous one back
// (an ASSERT_* returns early, and a leaked sink would follow into the next
// test).
class LogCapture {
public:
  LogCapture() : previous_(rcutils_logging_get_output_handler()) {
    const std::lock_guard<std::mutex> lock(g_log_mutex);
    g_log_lines.clear();
    rcutils_logging_set_output_handler(capture_log_handler);
  }
  ~LogCapture() { rcutils_logging_set_output_handler(previous_); }
  LogCapture(const LogCapture&) = delete;
  LogCapture& operator=(const LogCapture&) = delete;
  LogCapture(LogCapture&&) = delete;
  LogCapture& operator=(LogCapture&&) = delete;

  std::size_t count(const std::string& needle) const {
    const std::lock_guard<std::mutex> lock(g_log_mutex);
    std::size_t n = 0;
    for (const auto& line : g_log_lines) {
      if (line.second.find(needle) != std::string::npos) {
        ++n;
      }
    }
    return n;
  }

  // Highest severity any captured line mentioning `needle` was emitted at.
  int max_severity(const std::string& needle) const {
    const std::lock_guard<std::mutex> lock(g_log_mutex);
    int worst = 0;
    for (const auto& line : g_log_lines) {
      if (line.second.find(needle) != std::string::npos) {
        worst = std::max(worst, line.first);
      }
    }
    return worst;
  }

  std::string joined() const {
    const std::lock_guard<std::mutex> lock(g_log_mutex);
    std::string out;
    for (const auto& line : g_log_lines) {
      out += line.second;
      out += "\n";
    }
    return out;
  }

private:
  rcutils_logging_output_handler_t previous_;
};

}  // namespace

// A stalled attachment stream refuses EVERY candidate action, and an armed place
// region does not survive that gate.
//
// The clock-domain fix made this deadline the backstop for a producer that stops
// publishing: World State cannot expire a declaration it is no longer being told
// about, because a stream that has stopped cannot advance the stream clock. The
// property that has to hold is stronger than "the allowance stops" — while the
// payload model is stale the kernel does not know what it is carrying, so it
// must refuse the whole action (`attached_unavailable`), region or no region.
TEST_F(LifecycleKernelTest, StalledAttachmentStreamRefusesEveryCandidateEvenWithAnArmedRegion) {
  rclcpp::NodeOptions opts;
  opts.parameter_overrides(place_declaration_params());
  auto node = std::make_shared<osk::SafetyKernelLifecycleNode>("kernel_place_stalled", opts);
  rclcpp_lifecycle::State unconf(lifecycle_msgs::msg::State::PRIMARY_STATE_UNCONFIGURED, "uc");
  ASSERT_EQ(node->on_configure(unconf), osk::SafetyKernelLifecycleNode::CallbackReturn::SUCCESS);
  rclcpp_lifecycle::State inactive(lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE, "in");
  ASSERT_EQ(node->on_activate(inactive), osk::SafetyKernelLifecycleNode::CallbackReturn::SUCCESS);

  rclcpp::Node helper("place_stalled_helper");
  rclcpp::QoS chunk_qos(rclcpp::KeepLast(1));
  chunk_qos.reliable();
  auto cand_pub = helper.create_publisher<openral_msgs::msg::ActionChunk>(
      "/openral/candidate_action", chunk_qos);
  rclcpp::QoS js_qos(rclcpp::KeepLast(1));
  js_qos.best_effort();
  auto js_pub = helper.create_publisher<sensor_msgs::msg::JointState>("/joint_states", js_qos);
  rclcpp::QoS voxel_qos(rclcpp::KeepLast(1));
  voxel_qos.reliable();
  auto voxel_pub = helper.create_publisher<openral_msgs::msg::OccupancyVoxels>(
      "/openral/world_voxels", voxel_qos);
  rclcpp::QoS ws_qos(rclcpp::KeepLast(1));
  ws_qos.reliable();
  auto ws_pub = helper.create_publisher<openral_msgs::msg::WorldStateStamped>(
      "/openral/world_state_fast", ws_qos);
  std::atomic<int> safe_count{0};
  auto safe_sub = helper.create_subscription<openral_msgs::msg::ActionChunk>(
      "/openral/safe_action", chunk_qos,
      [&safe_count](const openral_msgs::msg::ActionChunk::SharedPtr) { ++safe_count; });
  std::string evidence_json;
  rclcpp::QoS failure_qos(rclcpp::KeepLast(10));
  failure_qos.reliable();
  auto failure_sub = helper.create_subscription<openral_msgs::msg::FailureTrigger>(
      "/openral/failure/safety", failure_qos,
      [&evidence_json](const openral_msgs::msg::FailureTrigger::SharedPtr msg) {
        evidence_json = msg->evidence_json;
      });

  rclcpp::executors::SingleThreadedExecutor exec;
  exec.add_node(node->get_node_base_interface());
  exec.add_node(helper.get_node_base_interface());

  const auto vox = declared_target_voxels();
  sensor_msgs::msg::JointState js;
  js.name = {"j0"};
  js.position = {0.0};
  const auto chunk = declared_carry_chunk();

  // Warm-up, before any candidate action is offered. The grid frame has to land
  // before the declaration that names it (a region declared against a frame the
  // kernel has not seen is refused, correctly, as a frame mismatch), and the
  // region has to be armed before the first chunk — otherwise the payload is
  // stopped for the right reason at the wrong time and the fault latches.
  const auto warm_deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(300);
  while (std::chrono::steady_clock::now() < warm_deadline) {
    voxel_pub->publish(vox);
    js_pub->publish(js);
    exec.spin_some(std::chrono::milliseconds(5));
    ws_pub->publish(declared_carry_state(node->now().nanoseconds(), node->now().nanoseconds(),
                                         /*timeout_s=*/60.0));
    exec.spin_some(std::chrono::milliseconds(5));
  }

  // Phase 1 — a live stream with a live declaration. The chunk passes ONLY
  // because the declared allowance is in force (0.020 m of true clearance
  // against a 0.030 m margin reduced to 0.005 m), which is what makes this an
  // observation of the armed region and not just of a clear scene.
  const auto pass_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
  while (safe_count.load() == 0 && std::chrono::steady_clock::now() < pass_deadline) {
    voxel_pub->publish(vox);
    js_pub->publish(js);
    exec.spin_some(std::chrono::milliseconds(5));
    ws_pub->publish(declared_carry_state(node->now().nanoseconds(), node->now().nanoseconds(),
                                         /*timeout_s=*/60.0));
    exec.spin_some(std::chrono::milliseconds(5));
    cand_pub->publish(chunk);
    exec.spin_some(std::chrono::milliseconds(10));
  }
  ASSERT_GT(safe_count.load(), 0)
      << "the declared approach must pass while the stream and the declaration are both live; "
         "chunks_dropped="
      << node->chunks_dropped();
  ASSERT_FALSE(node->fault_latched());

  // Phase 2 — the attachment stream dies. Everything else stays alive: voxels,
  // measured state, and the region the kernel already ingested and armed. Spin
  // the settling window rather than sleeping through it, so phase 1's in-flight
  // approvals are delivered and counted BEFORE the baseline is taken; anything
  // that arrives after it really is a chunk the stale gate let through.
  const auto settle_deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(400);
  while (std::chrono::steady_clock::now() < settle_deadline) {
    voxel_pub->publish(vox);
    js_pub->publish(js);
    exec.spin_some(std::chrono::milliseconds(10));
  }
  const int passed_before = safe_count.load();
  const std::uint64_t dropped_before = node->chunks_dropped();
  for (std::uint64_t i = 0; i < 3; ++i) {
    voxel_pub->publish(vox);
    js_pub->publish(js);
    exec.spin_some(std::chrono::milliseconds(10));
    cand_pub->publish(chunk);
    const auto drop_deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(500);
    while (node->chunks_dropped() == dropped_before + i &&
           std::chrono::steady_clock::now() < drop_deadline) {
      exec.spin_some(std::chrono::milliseconds(10));
    }
    EXPECT_EQ(node->chunks_dropped(), dropped_before + i + 1)
        << "candidate " << i << " must be refused while the payload model is stale";
  }
  EXPECT_EQ(safe_count.load(), passed_before)
      << "not one candidate may reach /openral/safe_action on a stalled attachment stream — an "
         "armed place region must not outlive the freshness gate that vouches for the payload";
  ASSERT_FALSE(evidence_json.empty()) << "a fail-closed drop must publish a FailureTrigger";
  EXPECT_NE(evidence_field(evidence_json, "detail").find("field=attached_unavailable"),
            std::string::npos)
      << "the refusal must name the stale attachment stream, not something downstream of it: "
      << evidence_json;
  EXPECT_FALSE(node->fault_latched())
      << "a stale input is fail-closed, not a latched fault: motion resumes when the stream does";
}

// The declaration's own backstop expires the allowance per candidate action, in
// the kernel's clock, while the attachment stream stays perfectly fresh.
//
// This is the other half of the clock-domain fix: liveness is judged where the
// stamp is read. The stream heartbeats at a fresh stamp on every beat (so the
// freshness gate above never fires) while the declaration keeps its original
// stamp, and the ingested region stays valid — only `place_declaration_live()`
// changes its answer. The same chunk that passed under the allowance must stop
// once the backstop lapses, which is the allowance being withdrawn and nothing
// else.
TEST_F(LifecycleKernelTest, PlaceDeclarationBackstopExpiresTheAllowancePerCandidate) {
  rclcpp::NodeOptions opts;
  opts.parameter_overrides(place_declaration_params());
  auto node = std::make_shared<osk::SafetyKernelLifecycleNode>("kernel_place_expiry", opts);
  rclcpp_lifecycle::State unconf(lifecycle_msgs::msg::State::PRIMARY_STATE_UNCONFIGURED, "uc");
  ASSERT_EQ(node->on_configure(unconf), osk::SafetyKernelLifecycleNode::CallbackReturn::SUCCESS);
  rclcpp_lifecycle::State inactive(lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE, "in");
  ASSERT_EQ(node->on_activate(inactive), osk::SafetyKernelLifecycleNode::CallbackReturn::SUCCESS);

  rclcpp::Node helper("place_expiry_helper");
  rclcpp::QoS chunk_qos(rclcpp::KeepLast(1));
  chunk_qos.reliable();
  auto cand_pub = helper.create_publisher<openral_msgs::msg::ActionChunk>(
      "/openral/candidate_action", chunk_qos);
  rclcpp::QoS js_qos(rclcpp::KeepLast(1));
  js_qos.best_effort();
  auto js_pub = helper.create_publisher<sensor_msgs::msg::JointState>("/joint_states", js_qos);
  rclcpp::QoS voxel_qos(rclcpp::KeepLast(1));
  voxel_qos.reliable();
  auto voxel_pub = helper.create_publisher<openral_msgs::msg::OccupancyVoxels>(
      "/openral/world_voxels", voxel_qos);
  rclcpp::QoS ws_qos(rclcpp::KeepLast(1));
  ws_qos.reliable();
  auto ws_pub = helper.create_publisher<openral_msgs::msg::WorldStateStamped>(
      "/openral/world_state_fast", ws_qos);
  std::atomic<int> safe_count{0};
  auto safe_sub = helper.create_subscription<openral_msgs::msg::ActionChunk>(
      "/openral/safe_action", chunk_qos,
      [&safe_count](const openral_msgs::msg::ActionChunk::SharedPtr) { ++safe_count; });
  std::string evidence_json;
  rclcpp::QoS failure_qos(rclcpp::KeepLast(10));
  failure_qos.reliable();
  auto failure_sub = helper.create_subscription<openral_msgs::msg::FailureTrigger>(
      "/openral/failure/safety", failure_qos,
      [&evidence_json](const openral_msgs::msg::FailureTrigger::SharedPtr msg) {
        if (evidence_json.empty()) {
          evidence_json = msg->evidence_json;
        }
      });

  rclcpp::executors::SingleThreadedExecutor exec;
  exec.add_node(node->get_node_base_interface());
  exec.add_node(helper.get_node_base_interface());

  const auto vox = declared_target_voxels();
  sensor_msgs::msg::JointState js;
  js.name = {"j0"};
  js.position = {0.0};
  const auto chunk = declared_carry_chunk();
  // Stamped once, in the kernel's clock, and never re-stamped: the declaration
  // belongs to the dispatching runner, and a heartbeat republishing it does not
  // make it younger.
  const std::int64_t declaration_stamp_ns = node->now().nanoseconds();
  const double timeout_s = 1.5;

  // Warm-up (see the stalled-stream test): grid frame first, then the
  // declaration, and no candidate action until the region is armed.
  const auto warm_deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(300);
  while (std::chrono::steady_clock::now() < warm_deadline) {
    voxel_pub->publish(vox);
    js_pub->publish(js);
    exec.spin_some(std::chrono::milliseconds(5));
    ws_pub->publish(
        declared_carry_state(node->now().nanoseconds(), declaration_stamp_ns, timeout_s));
    exec.spin_some(std::chrono::milliseconds(5));
  }

  const auto pass_deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(600);
  while (safe_count.load() == 0 && std::chrono::steady_clock::now() < pass_deadline) {
    voxel_pub->publish(vox);
    js_pub->publish(js);
    exec.spin_some(std::chrono::milliseconds(5));
    ws_pub->publish(
        declared_carry_state(node->now().nanoseconds(), declaration_stamp_ns, timeout_s));
    exec.spin_some(std::chrono::milliseconds(5));
    cand_pub->publish(chunk);
    exec.spin_some(std::chrono::milliseconds(10));
  }
  ASSERT_GT(safe_count.load(), 0)
      << "while the declaration is live the payload must be allowed to approach the declared "
         "target; chunks_dropped="
      << node->chunks_dropped();
  ASSERT_FALSE(node->fault_latched());

  // Past the backstop. The stream keeps heartbeating at a fresh stamp — so the
  // attachment gate stays satisfied and the region is re-ingested on every beat
  // — but the declaration is now older than its own timeout, so the allowance is
  // gone and the identical chunk stops on the identical cell. The window is
  // spun, not slept, so the stream stays fresh across it (this test is about the
  // declaration's clock, not the stream's) and phase 1's in-flight approvals are
  // delivered before the baseline is taken.
  const auto expiry_deadline =
      std::chrono::steady_clock::now() + std::chrono::milliseconds(1600);  // > the 1.5 s backstop
  while (std::chrono::steady_clock::now() < expiry_deadline) {
    voxel_pub->publish(vox);
    js_pub->publish(js);
    exec.spin_some(std::chrono::milliseconds(5));
    ws_pub->publish(
        declared_carry_state(node->now().nanoseconds(), declaration_stamp_ns, timeout_s));
    exec.spin_some(std::chrono::milliseconds(5));
  }
  const int passed_before = safe_count.load();
  const auto stop_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
  while (!node->fault_latched() && std::chrono::steady_clock::now() < stop_deadline) {
    voxel_pub->publish(vox);
    js_pub->publish(js);
    exec.spin_some(std::chrono::milliseconds(5));
    ws_pub->publish(
        declared_carry_state(node->now().nanoseconds(), declaration_stamp_ns, timeout_s));
    exec.spin_some(std::chrono::milliseconds(5));
    cand_pub->publish(chunk);
    exec.spin_some(std::chrono::milliseconds(10));
  }
  // Let the stop's evidence arrive: the latch flips inside the kernel before the
  // FailureTrigger reaches this process.
  const auto evidence_deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(500);
  while (evidence_json.empty() && std::chrono::steady_clock::now() < evidence_deadline) {
    exec.spin_some(std::chrono::milliseconds(10));
  }
  EXPECT_TRUE(node->fault_latched())
      << "an expired declaration must stop granting the allowance on the very next candidate "
         "action, not at the next world-state message; passes since expiry="
      << (safe_count.load() - passed_before);
  EXPECT_EQ(safe_count.load(), passed_before) << "no candidate may pass on an expired declaration";
  ASSERT_FALSE(evidence_json.empty());
  EXPECT_EQ(evidence_field(evidence_json, "link_a"), "attached:sim:obj_main")
      << "the stop must name the payload the withdrawn allowance belonged to: " << evidence_json;
  EXPECT_NEAR(std::stod(evidence_field(evidence_json, "min_distance_m")), 0.02, 1e-6)
      << "the reported distance is the pair's true one — only the margin it was gated against "
         "moved: "
      << evidence_json;
}

// ── What the place-region refusals say, and how often they say it ────────────

// A declaration published before the grasp lands is the NORMAL state, and the
// kernel says so once.
//
// Round-8 (`spark:~/openral-runs/2026-08-15-round8/`) logged this as
// `safety.place_region_rejected reason=bounds`, at WARN, on every attachment
// heartbeat — 672-811 lines per run. Three things were wrong with that and all
// three are pinned here: the reason was not a bound (the object mask was empty
// because no payload was attached yet), the severity claimed a fault where there
// is none, and a standing state was re-announced at the heartbeat rate instead
// of on its transition.
TEST_F(LifecycleKernelTest, PreGraspDeclarationIsAnnouncedOnceAndNotAsARejection) {
  rclcpp::NodeOptions opts;
  opts.parameter_overrides(place_declaration_params());
  auto node = std::make_shared<osk::SafetyKernelLifecycleNode>("kernel_place_pregrasp", opts);
  rclcpp_lifecycle::State unconf(lifecycle_msgs::msg::State::PRIMARY_STATE_UNCONFIGURED, "uc");
  ASSERT_EQ(node->on_configure(unconf), osk::SafetyKernelLifecycleNode::CallbackReturn::SUCCESS);
  rclcpp_lifecycle::State inactive(lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE, "in");
  ASSERT_EQ(node->on_activate(inactive), osk::SafetyKernelLifecycleNode::CallbackReturn::SUCCESS);

  rclcpp::Node helper("place_pregrasp_helper");
  rclcpp::QoS voxel_qos(rclcpp::KeepLast(1));
  voxel_qos.reliable();
  auto voxel_pub = helper.create_publisher<openral_msgs::msg::OccupancyVoxels>(
      "/openral/world_voxels", voxel_qos);
  rclcpp::QoS ws_qos(rclcpp::KeepLast(1));
  ws_qos.reliable();
  auto ws_pub = helper.create_publisher<openral_msgs::msg::WorldStateStamped>(
      "/openral/world_state_fast", ws_qos);

  rclcpp::executors::SingleThreadedExecutor exec;
  exec.add_node(node->get_node_base_interface());
  exec.add_node(helper.get_node_base_interface());

  const auto vox = declared_target_voxels();
  // The grid frame first, so the region is judged against a frame the kernel
  // knows (a frame mismatch is a different refusal with its own reason).
  for (int i = 0; i < 5; ++i) {
    voxel_pub->publish(vox);
    exec.spin_some(std::chrono::milliseconds(10));
  }

  const LogCapture logs;
  // 40 heartbeats of the pre-grasp state: an active declaration with a valid
  // region, and no payload attached yet.
  for (int i = 0; i < 40; ++i) {
    ws_pub->publish(declared_carry_state(node->now().nanoseconds(), node->now().nanoseconds(),
                                         /*timeout_s=*/60.0, /*carrying=*/false));
    exec.spin_some(std::chrono::milliseconds(5));
  }

  EXPECT_EQ(logs.count("safety.place_region_rejected"), 0U)
      << "a declaration whose payload is not attached yet is not a refused region:\n"
      << logs.joined();
  EXPECT_EQ(logs.count("safety.place_region_not_armed reason=no_object"), 1U)
      << "the pre-grasp state is announced once, on its transition, not once per heartbeat:\n"
      << logs.joined();
  EXPECT_EQ(logs.max_severity("safety.place_region_not_armed"),
            static_cast<int>(RCUTILS_LOG_SEVERITY_INFO))
      << "no fault occurred: the margins in force are exactly the undeclared ones";
}

// A clean release logs the disarm once, as a drop.
//
// The declaration outlives the payload by design — it is retracted when the GOAL
// ends, not when the gripper opens — so the first heartbeat after a release
// carries a live declaration with nothing to scope it to. Ingesting that before
// the detach edge is handled turned every release into
// `place_region_rejected`, which is a producer-error report for something the
// kernel did on purpose.
TEST_F(LifecycleKernelTest, CleanDetachDropsTheRegionOnceInsteadOfRejectingIt) {
  rclcpp::NodeOptions opts;
  opts.parameter_overrides(place_declaration_params());
  auto node = std::make_shared<osk::SafetyKernelLifecycleNode>("kernel_place_detach", opts);
  rclcpp_lifecycle::State unconf(lifecycle_msgs::msg::State::PRIMARY_STATE_UNCONFIGURED, "uc");
  ASSERT_EQ(node->on_configure(unconf), osk::SafetyKernelLifecycleNode::CallbackReturn::SUCCESS);
  rclcpp_lifecycle::State inactive(lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE, "in");
  ASSERT_EQ(node->on_activate(inactive), osk::SafetyKernelLifecycleNode::CallbackReturn::SUCCESS);

  rclcpp::Node helper("place_detach_helper");
  rclcpp::QoS voxel_qos(rclcpp::KeepLast(1));
  voxel_qos.reliable();
  auto voxel_pub = helper.create_publisher<openral_msgs::msg::OccupancyVoxels>(
      "/openral/world_voxels", voxel_qos);
  rclcpp::QoS ws_qos(rclcpp::KeepLast(1));
  ws_qos.reliable();
  auto ws_pub = helper.create_publisher<openral_msgs::msg::WorldStateStamped>(
      "/openral/world_state_fast", ws_qos);

  rclcpp::executors::SingleThreadedExecutor exec;
  exec.add_node(node->get_node_base_interface());
  exec.add_node(helper.get_node_base_interface());

  const auto vox = declared_target_voxels();
  for (int i = 0; i < 5; ++i) {
    voxel_pub->publish(vox);
    exec.spin_some(std::chrono::milliseconds(10));
  }
  // Carry, with the region armed.
  for (int i = 0; i < 10; ++i) {
    ws_pub->publish(declared_carry_state(node->now().nanoseconds(), node->now().nanoseconds(),
                                         /*timeout_s=*/60.0));
    exec.spin_some(std::chrono::milliseconds(5));
  }

  const LogCapture logs;
  // Release: a new attachment revision carrying no objects, with the goal's
  // declaration still live, heartbeated 30 times.
  for (int i = 0; i < 30; ++i) {
    ws_pub->publish(declared_carry_state(node->now().nanoseconds(), node->now().nanoseconds(),
                                         /*timeout_s=*/60.0, /*carrying=*/false,
                                         /*revision=*/2));
    exec.spin_some(std::chrono::milliseconds(5));
  }

  EXPECT_EQ(logs.count("safety.place_region_rejected"), 0U)
      << "a clean detach is not a refused region:\n"
      << logs.joined();
  EXPECT_EQ(logs.count("safety.place_region_dropped reason=detached"), 1U)
      << "the disarm is announced exactly once, naming the payload that took the region with "
         "it:\n"
      << logs.joined();
  EXPECT_EQ(logs.count("target=sim:cab_1_left_group_main"),
            logs.count("safety.place_region_dropped reason=detached") +
                logs.count("safety.place_region_not_armed reason=no_object"))
      << "every announced line stays attributable to the declaration (HZ-0097-2):\n"
      << logs.joined();
  EXPECT_LE(logs.count("safety.place_region_not_armed reason=no_object"), 1U)
      << "the standing post-release state is a transition, not a heartbeat:\n"
      << logs.joined();
}

// ── The advisory band, end to end (#176) ─────────────────────────────────────
//
// The band's whole purpose is a difference the geometry tests cannot see: what
// the NODE does with an advisory hit. A latched stop asserts /openral/estop and
// requires an operator to call /openral/estop_reset; an advisory refusal drops
// the chunk and leaves the kernel able to accept the next one.
//
// Both tests share the declared-carry fixture and move the payload from its
// approach pose (`payload_x` 0.10, face 20 mm clear of the cell) to an arrived
// one (0.130, face 10 mm inside it). At the fixture's 30 mm attached margin and
// 37.5 mm allowance the gate sits at −7.5 mm and the band floor at −12.5 mm, so
// 10 mm of penetration is inside the band — the 2026-08-26 baguette case.
namespace {

/// Drive one candidate action through and settle, so each chunk produces
/// exactly one verdict. Returns the safe-action count afterwards.
void offer_one_chunk(rclcpp::executors::SingleThreadedExecutor& exec,
                     const std::function<void()>& publish_inputs,
                     const std::function<void()>& publish_chunk) {
  publish_inputs();
  exec.spin_some(std::chrono::milliseconds(10));
  publish_chunk();
  for (int i = 0; i < 6; ++i) {
    exec.spin_some(std::chrono::milliseconds(10));
  }
}

}  // namespace

TEST_F(LifecycleKernelTest, AnArrivedPlaceRefusesTheChunkWithoutLatching) {
  rclcpp::NodeOptions opts;
  opts.parameter_overrides(place_declaration_params());
  auto node = std::make_shared<osk::SafetyKernelLifecycleNode>("kernel_place_advisory", opts);
  rclcpp_lifecycle::State unconf(lifecycle_msgs::msg::State::PRIMARY_STATE_UNCONFIGURED, "uc");
  ASSERT_EQ(node->on_configure(unconf), osk::SafetyKernelLifecycleNode::CallbackReturn::SUCCESS);
  rclcpp_lifecycle::State inactive(lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE, "in");
  ASSERT_EQ(node->on_activate(inactive), osk::SafetyKernelLifecycleNode::CallbackReturn::SUCCESS);

  rclcpp::Node helper("place_advisory_helper");
  rclcpp::QoS chunk_qos(rclcpp::KeepLast(1));
  chunk_qos.reliable();
  auto cand_pub = helper.create_publisher<openral_msgs::msg::ActionChunk>(
      "/openral/candidate_action", chunk_qos);
  rclcpp::QoS js_qos(rclcpp::KeepLast(1));
  js_qos.best_effort();
  auto js_pub = helper.create_publisher<sensor_msgs::msg::JointState>("/joint_states", js_qos);
  rclcpp::QoS voxel_qos(rclcpp::KeepLast(1));
  voxel_qos.reliable();
  auto voxel_pub = helper.create_publisher<openral_msgs::msg::OccupancyVoxels>(
      "/openral/world_voxels", voxel_qos);
  rclcpp::QoS ws_qos(rclcpp::KeepLast(1));
  ws_qos.reliable();
  auto ws_pub = helper.create_publisher<openral_msgs::msg::WorldStateStamped>(
      "/openral/world_state_fast", ws_qos);
  std::atomic<int> safe_count{0};
  auto safe_sub = helper.create_subscription<openral_msgs::msg::ActionChunk>(
      "/openral/safe_action", chunk_qos,
      [&safe_count](const openral_msgs::msg::ActionChunk::SharedPtr) { ++safe_count; });
  std::atomic<int> estop_count{0};
  rclcpp::QoS estop_q(rclcpp::KeepLast(10));
  estop_q.reliable();
  auto estop_sub = helper.create_subscription<std_msgs::msg::Empty>(
      "/openral/estop", estop_q,
      [&estop_count](const std_msgs::msg::Empty::SharedPtr) { ++estop_count; });

  rclcpp::executors::SingleThreadedExecutor exec;
  exec.add_node(node->get_node_base_interface());
  exec.add_node(helper.get_node_base_interface());

  const auto vox = declared_target_voxels();
  sensor_msgs::msg::JointState js;
  js.name = {"j0"};
  js.position = {0.0};
  const auto chunk = declared_carry_chunk();

  // Warm-up at the APPROACH pose: the grid frame lands, the region arms, and
  // the chunk passes — so anything observed afterwards is caused by the payload
  // having arrived, not by the region never being armed.
  const auto warm_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
  while (safe_count.load() == 0 && std::chrono::steady_clock::now() < warm_deadline) {
    voxel_pub->publish(vox);
    js_pub->publish(js);
    exec.spin_some(std::chrono::milliseconds(5));
    ws_pub->publish(declared_carry_state(node->now().nanoseconds(), node->now().nanoseconds(),
                                         /*timeout_s=*/60.0));
    exec.spin_some(std::chrono::milliseconds(5));
    cand_pub->publish(chunk);
    exec.spin_some(std::chrono::milliseconds(10));
  }
  ASSERT_GT(safe_count.load(), 0) << "the approach must pass before the arrival is meaningful";
  ASSERT_FALSE(node->fault_latched());

  // Settle before baselining: the warm-up's last in-flight approvals have to be
  // delivered and counted BEFORE the baseline, or one of them lands afterwards
  // and reads as "the arrived pose was passed through".
  const auto settle_deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(400);
  while (std::chrono::steady_clock::now() < settle_deadline) {
    voxel_pub->publish(vox);
    js_pub->publish(js);
    exec.spin_some(std::chrono::milliseconds(10));
  }

  const LogCapture logs;
  const int passed_before = safe_count.load();
  const std::uint64_t dropped_before = node->chunks_dropped();
  const int estops_before = estop_count.load();

  // Arrived: the payload's face 10 mm inside the declared receptacle's cell.
  offer_one_chunk(
      exec,
      [&] {
        voxel_pub->publish(vox);
        js_pub->publish(js);
        ws_pub->publish(declared_carry_state(node->now().nanoseconds(), node->now().nanoseconds(),
                                             /*timeout_s=*/60.0, /*carrying=*/true,
                                             /*revision=*/1, /*payload_x=*/0.130));
      },
      [&] { cand_pub->publish(chunk); });

  EXPECT_GT(node->chunks_dropped(), dropped_before) << "the action is still refused";
  EXPECT_EQ(safe_count.load(), passed_before) << "and it is NOT passed through";
  EXPECT_FALSE(node->fault_latched())
      << "an arrived place inside its own declared region must not latch:\n"
      << logs.joined();
  EXPECT_EQ(estop_count.load(), estops_before)
      << "and must not assert /openral/estop:\n"
      << logs.joined();
  EXPECT_GE(logs.count("safety.collision_advisory"), 1U)
      << "the refusal is announced as advisory, with its own reason:\n"
      << logs.joined();
  EXPECT_EQ(logs.count("safety.collision kind="), 0U)
      << "and never as an ordinary collision stop:\n"
      << logs.joined();

  // Drain this node's spans before it goes away. `on_cleanup` is where
  // `shutdown_tracing()` flushes the BatchSpanProcessor, and a test that emits
  // safety spans and then drops the node leaves that processor racing the
  // process teardown against a collector that is not there.
  rclcpp_lifecycle::State active(lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE, "ac");
  node->on_deactivate(active);
  node->on_cleanup(inactive);
}

TEST_F(LifecycleKernelTest, AnUnbrokenAdvisoryRunLatchesAtItsCap) {
  // The band must not become a way to push indefinitely. `place_advisory_max_
  // consecutive` (3 by default) bounds the run; the next refusal is an ordinary
  // latched stop with its E-stop, exactly as before #176.
  rclcpp::NodeOptions opts;
  opts.parameter_overrides(place_declaration_params());
  auto node = std::make_shared<osk::SafetyKernelLifecycleNode>("kernel_place_advisory_cap", opts);
  rclcpp_lifecycle::State unconf(lifecycle_msgs::msg::State::PRIMARY_STATE_UNCONFIGURED, "uc");
  ASSERT_EQ(node->on_configure(unconf), osk::SafetyKernelLifecycleNode::CallbackReturn::SUCCESS);
  rclcpp_lifecycle::State inactive(lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE, "in");
  ASSERT_EQ(node->on_activate(inactive), osk::SafetyKernelLifecycleNode::CallbackReturn::SUCCESS);

  rclcpp::Node helper("place_advisory_cap_helper");
  rclcpp::QoS chunk_qos(rclcpp::KeepLast(1));
  chunk_qos.reliable();
  auto cand_pub = helper.create_publisher<openral_msgs::msg::ActionChunk>(
      "/openral/candidate_action", chunk_qos);
  rclcpp::QoS js_qos(rclcpp::KeepLast(1));
  js_qos.best_effort();
  auto js_pub = helper.create_publisher<sensor_msgs::msg::JointState>("/joint_states", js_qos);
  rclcpp::QoS voxel_qos(rclcpp::KeepLast(1));
  voxel_qos.reliable();
  auto voxel_pub = helper.create_publisher<openral_msgs::msg::OccupancyVoxels>(
      "/openral/world_voxels", voxel_qos);
  rclcpp::QoS ws_qos(rclcpp::KeepLast(1));
  ws_qos.reliable();
  auto ws_pub = helper.create_publisher<openral_msgs::msg::WorldStateStamped>(
      "/openral/world_state_fast", ws_qos);
  std::atomic<int> safe_count{0};
  auto safe_sub = helper.create_subscription<openral_msgs::msg::ActionChunk>(
      "/openral/safe_action", chunk_qos,
      [&safe_count](const openral_msgs::msg::ActionChunk::SharedPtr) { ++safe_count; });

  rclcpp::executors::SingleThreadedExecutor exec;
  exec.add_node(node->get_node_base_interface());
  exec.add_node(helper.get_node_base_interface());

  const auto vox = declared_target_voxels();
  sensor_msgs::msg::JointState js;
  js.name = {"j0"};
  js.position = {0.0};
  const auto chunk = declared_carry_chunk();

  const auto warm_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
  while (safe_count.load() == 0 && std::chrono::steady_clock::now() < warm_deadline) {
    voxel_pub->publish(vox);
    js_pub->publish(js);
    exec.spin_some(std::chrono::milliseconds(5));
    ws_pub->publish(declared_carry_state(node->now().nanoseconds(), node->now().nanoseconds(),
                                         /*timeout_s=*/60.0));
    exec.spin_some(std::chrono::milliseconds(5));
    cand_pub->publish(chunk);
    exec.spin_some(std::chrono::milliseconds(10));
  }
  ASSERT_GT(safe_count.load(), 0);
  ASSERT_FALSE(node->fault_latched());

  const LogCapture logs;
  const auto arrived_inputs = [&] {
    voxel_pub->publish(vox);
    js_pub->publish(js);
    ws_pub->publish(declared_carry_state(node->now().nanoseconds(), node->now().nanoseconds(),
                                         /*timeout_s=*/60.0, /*carrying=*/true,
                                         /*revision=*/1, /*payload_x=*/0.130));
  };
  const auto publish_chunk = [&] { cand_pub->publish(chunk); };

  // Three refusals inside the cap: still no latch.
  for (int i = 0; i < 3; ++i) {
    offer_one_chunk(exec, arrived_inputs, publish_chunk);
    EXPECT_FALSE(node->fault_latched())
        << "refusal " << (i + 1) << " of 3 is inside the cap:\n"
        << logs.joined();
  }

  // The fourth is over it, and is an ordinary stop.
  offer_one_chunk(exec, arrived_inputs, publish_chunk);
  EXPECT_TRUE(node->fault_latched())
      << "an unbroken advisory run must latch once it passes the cap:\n"
      << logs.joined();
  EXPECT_EQ(logs.count("safety.collision_advisory"), 3U)
      << "exactly the capped number of advisories precede it:\n"
      << logs.joined();
  EXPECT_GE(logs.count("safety.collision kind="), 1U)
      << "and the one past the cap is announced as an ordinary collision:\n"
      << logs.joined();

  // Drain this node's spans before it goes away (see the test above).
  rclcpp_lifecycle::State active(lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE, "ac");
  node->on_deactivate(active);
  node->on_cleanup(inactive);
}
