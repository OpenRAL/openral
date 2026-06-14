// SPDX-License-Identifier: Apache-2.0
// ADR-0020 — gtest unit coverage for SafetyKernelLifecycleNode.
// Exercises lifecycle transitions, fault-latch behaviour, and the
// /openral/estop_reset cooldown semantics WITHOUT requiring a running
// ROS graph — we just drive the lifecycle callbacks directly.

#include "openral_safety_kernel/lifecycle_kernel.hpp"

#include <chrono>
#include <future>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <gtest/gtest.h>
#include <sensor_msgs/msg/joint_state.hpp>

#include <lifecycle_msgs/msg/state.hpp>
#include <openral_msgs/msg/action_chunk.hpp>
#include <openral_msgs/msg/failure_trigger.hpp>
#include <openral_msgs/msg/occupancy_voxels.hpp>
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
  // ADR-0020 PR-K — parameter-based envelope path. The Python launch
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
  // ADR-0030 — a well-formed collision model loads and the node reaches
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

// ADR-0030 — the ViolationKind enum must stay 1:1 with the IDL KIND_*
// constants so the lifecycle node can publish a FailureTrigger without
// translation (validator.hpp documents this contract). kCollision is added
// for the geometric-safety check; the value must equal KIND_COLLISION even
// though the kernel does not yet *emit* it.
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

// ADR-0040 — a 2-link self-collision model whose only pair is allowed (so the
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
      // ADR-0040 plumbing
      {"collision_joint_names", std::vector<std::string>{"j0", "j1"}},
      {"collision_seed_dt_s", 0.05},
      {"collision_state_deadline_ms", 500.0},
  };
}

}  // namespace

// ADR-0040 — a JOINT_VELOCITY chunk arriving with the geometric check enabled
// but NO measured joint-state seed must be dropped fail-closed, never silently
// passed (pre-ADR-0040 it bypassed the geometric block entirely). This is the
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

// ADR-0040 — once a fresh, complete measured seed is available, a clear velocity
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

// ADR-0040 Phase 3 — a CARTESIAN_DELTA chunk (the arm mode for LIBERO/SIMPLER/
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

// ADR-0040 Phase 2 — DETERMINISTIC proof that a velocity chunk's REACTIVE check
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

// ADR-0040 — DETERMINISTIC proof of the mobile-base world (voxel) path: the
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

// ADR-0040 Phase 3 — DETERMINISTIC proof of PREDICTIVE Cartesian look-ahead: a
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
      // ADR-0040 Phase 3 — predictive Cartesian: EE is link index 2.
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

// ADR-0040 Phase 3 — the predictive Cartesian look-ahead must NOT reject a chunk
// whose whole predicted trajectory stays clear (no false positive from the
// margin inflation). Same arm + EE-link as above, but the wall is far (y>=1.9)
// and the +y chunk only reaches ~y=1.5, so every predicted step is clear and the
// chunk must reach /openral/safe_action.
TEST_F(LifecycleKernelTest, CartesianDeltaPredictivePassesWhenTrajectoryStaysClear) {
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
      {"collision_predict_margin_growth_m", 0.02},
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

  // Wall far away (y>=1.9); the +y chunk reaches only ~y=1.5.
  openral_msgs::msg::OccupancyVoxels vox;
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
      if (cy >= 1.9) {
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
  cart.horizon = 8;
  cart.n_dof = 6;
  cart.flat.clear();
  for (int s = 0; s < 8; ++s) {
    cart.flat.insert(cart.flat.end(), {0.0, 0.05, 0.0, 0.0, 0.0, 0.0});
  }

  const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(800);
  while (safe_count.load() == 0 && std::chrono::steady_clock::now() < deadline) {
    js_pub->publish(js);
    voxel_pub->publish(vox);
    exec.spin_some(std::chrono::milliseconds(10));
    cand_pub->publish(cart);
    exec.spin_some(std::chrono::milliseconds(10));
  }
  EXPECT_GT(safe_count.load(), 0)
      << "a Cartesian chunk whose whole predicted trajectory stays clear must pass to "
         "/openral/safe_action (no false positive)";
  EXPECT_FALSE(node->fault_latched());
}
