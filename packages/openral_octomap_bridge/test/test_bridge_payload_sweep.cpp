// SPDX-License-Identifier: Apache-2.0
// A carried payload's swept volume is cleared from the octree, pre-attach path included.
//
// Deploy trial g12 (SO-101, LeIsaac PickOrange, 15 mm octomap): the carried orange's own
// returns, captured 0.27 s before ATTACH reached the self-filter, stayed in the octree
// 85-120 mm behind the payload; the grid clearing follows only the payload's current pose,
// and every later ray through them ended on the robot. The kernel stopped the carry on
// them. DROID t6: one banana return mid-carry never cleared either.
//
// Runs the REAL node in-process against real TF (/tf history) and real DDS, with a real
// `BoundingBoxQuery` server standing in for `octomap_server`'s `~/clear_bbox` (a process
// boundary): every request the bridge sends is recorded.

#include <chrono>
#include <cmath>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include <geometry_msgs/msg/transform_stamped.hpp>
#include <gtest/gtest.h>
#include <octomap/OcTree.h>
#include <octomap_msgs/conversions.h>
#include <octomap_msgs/msg/octomap.hpp>
#include <octomap_msgs/srv/bounding_box_query.hpp>
#include <tf2_ros/transform_broadcaster.h>

#include <openral_msgs/msg/attached_collision_object.hpp>
#include <openral_msgs/msg/attached_collision_primitive.hpp>
#include <openral_msgs/msg/world_state_stamped.hpp>
#include <rclcpp/rclcpp.hpp>

#include "octomap_voxel_bridge.hpp"

namespace bridge = openral_octomap_bridge;
using namespace std::chrono_literals;
using BBox = octomap_msgs::srv::BoundingBoxQuery;

namespace {

constexpr double kRes = 0.015;
constexpr double kHalf[3] = {0.032, 0.033, 0.054};  // g12's accepted payload
constexpr double kSupportTop = 0.030;               // the counter

class BridgePayloadSweep : public ::testing::Test {
protected:
  static void SetUpTestSuite() { rclcpp::init(0, nullptr); }
  static void TearDownTestSuite() { rclcpp::shutdown(); }

  void SetUp() override {
    std::vector<rclcpp::Parameter> overrides{
        {"base_frame", "base_link"},
        {"octomap_topic", "/test_octomap_sweep"},
        {"output_topic", "/test_world_voxels_sweep"},
        {"world_state_topic", "/test_world_state_sweep"},
        {"resolution", kRes},
        {"coverage_radius_m", 0.4},
        {"publish_rate_hz", 20.0},
        {"payload_sweep_clear_service", "/test_clear_bbox"},
        {"payload_sweep_clear_period_s", 0.2},
        {"payload_sweep_backfill_s", 0.5},
    };
    rclcpp::NodeOptions options;
    options.parameter_overrides(overrides);
    bridge_ = std::make_shared<bridge::OctomapVoxelBridge>(options);
    peer_ = std::make_shared<rclcpp::Node>("bridge_payload_sweep_peer");
    tf_ = std::make_unique<tf2_ros::TransformBroadcaster>(peer_);
    octomap_pub_ = peer_->create_publisher<octomap_msgs::msg::Octomap>("/test_octomap_sweep",
                                                                       rclcpp::QoS(1).reliable());
    world_state_pub_ = peer_->create_publisher<openral_msgs::msg::WorldStateStamped>(
        "/test_world_state_sweep", rclcpp::QoS(1).reliable());
    server_ = peer_->create_service<BBox>(
        "/test_clear_bbox",
        [this](const std::shared_ptr<BBox::Request> req, std::shared_ptr<BBox::Response> /*resp*/) {
          const std::lock_guard<std::mutex> lock(mutex_);
          requests_.push_back(*req);
        });
    executor_.add_node(bridge_);
    executor_.add_node(peer_);
    spin_for(300ms);  // discovery
  }

  void TearDown() override {
    executor_.remove_node(peer_);
    executor_.remove_node(bridge_);
  }

  void spin_for(std::chrono::milliseconds d) {
    const auto end = std::chrono::steady_clock::now() + d;
    while (std::chrono::steady_clock::now() < end) {
      executor_.spin_some(10ms);
      std::this_thread::sleep_for(2ms);
    }
  }

  // The hand (= payload centre) at x along the carry, on /tf with the current stamp.
  void place_hand(double x, double z) {
    geometry_msgs::msg::TransformStamped base;
    base.header.stamp = peer_->now();
    base.header.frame_id = "map";
    base.child_frame_id = "base_link";
    base.transform.rotation.w = 1.0;
    geometry_msgs::msg::TransformStamped hand = base;
    hand.header.frame_id = "base_link";
    hand.child_frame_id = "hand";
    hand.transform.translation.x = x;
    hand.transform.translation.z = z;
    tf_->sendTransform({base, hand});
  }

  void send_octree() {
    octomap::OcTree tree(kRes);
    tree.updateNode(octomap::point3d(0.3F, 0.3F, 0.3F), true);  // anything: a live octree
    octomap_msgs::msg::Octomap msg;
    ASSERT_TRUE(octomap_msgs::binaryMapToMsg(tree, msg));
    msg.header.frame_id = "map";
    msg.header.stamp = peer_->now();
    octomap_pub_->publish(msg);
  }

  void send_world_state() {
    openral_msgs::msg::AttachedCollisionObject object;
    object.object_id = "grasped_payload:gripper";
    object.attach_link = "hand";
    object.pose_in_link.orientation.w = 1.0;
    openral_msgs::msg::AttachedCollisionPrimitive box;
    box.shape_type = openral_msgs::msg::AttachedCollisionPrimitive::SHAPE_BOX;
    box.shape_dimensions = {kHalf[0], kHalf[1], kHalf[2]};
    box.pose_in_object.orientation.w = 1.0;
    object.primitives.push_back(box);
    object.support_contact_valid = true;
    auto& w = object.support_contact;
    w.support_id = "counter";
    w.contact_point_in_object.z = -kHalf[2];
    w.contact_normal_in_object.z = 1.0;
    w.patch_radius_m = 0.03;
    w.max_penetration_m = 0.002;
    w.stamp_ns = 1;
    openral_msgs::msg::WorldStateStamped state;
    state.header.stamp = peer_->now();
    state.attached_objects.push_back(object);
    state.attachment_revision = 7;
    world_state_pub_->publish(state);
  }

  std::shared_ptr<bridge::OctomapVoxelBridge> bridge_;
  std::shared_ptr<rclcpp::Node> peer_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_;
  rclcpp::Publisher<octomap_msgs::msg::Octomap>::SharedPtr octomap_pub_;
  rclcpp::Publisher<openral_msgs::msg::WorldStateStamped>::SharedPtr world_state_pub_;
  rclcpp::Service<BBox>::SharedPtr server_;
  rclcpp::executors::SingleThreadedExecutor executor_;
  std::mutex mutex_;
  std::vector<BBox::Request> requests_;
};

TEST_F(BridgePayloadSweep, ThePreAttachPathAndTheCarryAreClearedAboveTheSupport) {
  // 0.4 s of carry with nothing attached yet (the returns the self-filter kept): x 0 -> 0.08.
  const double z = kSupportTop + kHalf[2] + 0.02;
  for (int i = 0; i <= 8; ++i) {
    place_hand(0.01 * i, z);
    send_octree();
    spin_for(50ms);
  }
  // ATTACH: the attachment appears; the carry continues to x = 0.16.
  for (int i = 9; i <= 16; ++i) {
    place_hand(0.01 * i, z);
    send_octree();
    send_world_state();
    spin_for(50ms);
  }
  spin_for(300ms);
  const std::lock_guard<std::mutex> lock(mutex_);
  ASSERT_FALSE(requests_.empty()) << "no clear_bbox request";
  double lo_x = 1e9;
  double hi_x = -1e9;
  double lo_z = 1e9;
  for (const auto& r : requests_) {
    lo_x = std::min(lo_x, r.min.x);
    hi_x = std::max(hi_x, r.max.x);
    lo_z = std::min(lo_z, r.min.z);
  }
  // The pre-attach pose 0.3 s back (x ~ 0.05) is cleared through the TF-history backfill.
  EXPECT_LE(lo_x, 0.05 - kHalf[0] + 1e-6) << "backfill missed the pre-attach path";
  EXPECT_GE(hi_x, 0.16 + kHalf[0] - 0.02) << "the carry was not cleared";
  // Never into the counter: one cell above the attested support (+ its depth bound).
  EXPECT_GE(lo_z, kSupportTop + 0.002 + kRes - 1e-6);
}

TEST_F(BridgePayloadSweep, NothingIsClearedWithoutAnAttachment) {
  for (int i = 0; i <= 10; ++i) {
    place_hand(0.01 * i, 0.1);
    send_octree();
    spin_for(50ms);
  }
  const std::lock_guard<std::mutex> lock(mutex_);
  EXPECT_TRUE(requests_.empty());
}

}  // namespace
