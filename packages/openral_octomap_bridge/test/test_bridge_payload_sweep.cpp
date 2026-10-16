// SPDX-License-Identifier: Apache-2.0
// A carried payload's own posed volume is cleared from the octree, attached poses only.
//
// Deploy trial g12 (SO-101, LeIsaac PickOrange, 15 mm octomap): the carried orange's own
// returns stayed in the octree behind the payload; the grid clearing follows only the
// payload's current pose, and every later ray through them ended on the robot. DROID t6:
// one banana return mid-carry never cleared either. The bridge clears each attached pose's
// primitive boxes; the hand's path BEFORE the attach is never cleared (PR #417 review: the
// payload posed rigidly in the hand along the approach is where the object never was).
//
// Runs the REAL node in-process against real TF (/tf history) and real DDS, with a real
// `BoundingBoxQuery` server standing in for `octomap_server`'s `~/clear_bbox` (a process
// boundary): every request the bridge sends is recorded.

#include <chrono>
#include <cmath>
#include <memory>
#include <mutex>
#include <string>
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

  void send_world_state(bool attest_support = true,
                        const std::string& evidence_kind = "vision_segmentation") {
    openral_msgs::msg::AttachedCollisionObject object;
    object.object_id = "grasped_payload:gripper";
    object.attach_link = "hand";
    object.pose_in_link.orientation.w = 1.0;
    openral_msgs::msg::AttachedCollisionPrimitive box;
    box.shape_type = openral_msgs::msg::AttachedCollisionPrimitive::SHAPE_BOX;
    box.shape_dimensions = {kHalf[0], kHalf[1], kHalf[2]};
    box.pose_in_object.orientation.w = 1.0;
    object.primitives.push_back(box);
    object.evidence_kind = evidence_kind;
    object.support_contact_valid = attest_support;
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

TEST_F(BridgePayloadSweep, TheCarryIsClearedAboveTheSupportAndThePreAttachPathIsNot) {
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
    // One posed payload box per request, never the AABB of the period's motion.
    EXPECT_LE(r.max.x - r.min.x, 2.0 * kHalf[0] + 1e-6) << "a motion AABB was cleared";
  }
  // Nothing before the first attached pose (x = 0.09): no pre-attach backfill.
  EXPECT_GE(lo_x, 0.09 - kHalf[0] - 1e-6) << "the pre-attach approach was cleared";
  EXPECT_GE(hi_x, 0.16 + kHalf[0] - 0.02) << "the carry was not cleared";
  // Never into the counter: one cell above the attested support (+ its depth bound).
  EXPECT_GE(lo_z, kSupportTop + 0.002 + kRes - 1e-6);
}

TEST_F(BridgePayloadSweep, WithoutASupportARegionPayloadIsFlooredAtItsAttachBottom) {
  // No support attested (the vision pick's attestation lapsed): the floor is the region
  // payload's lowest point at attach, never below it.
  const double z = kSupportTop + kHalf[2];
  for (int i = 0; i <= 8; ++i) {
    place_hand(0.01 * i, z - 0.002 * i);  // the hand dips 16 mm while carrying
    send_octree();
    send_world_state(false, "grasp_target_region");
    spin_for(50ms);
  }
  spin_for(300ms);
  const std::lock_guard<std::mutex> lock(mutex_);
  ASSERT_FALSE(requests_.empty()) << "a region payload is still cleared";
  for (const auto& r : requests_) {
    EXPECT_GE(r.min.z, kSupportTop - 1e-6) << "cleared below the payload's attach bottom";
  }
}

TEST_F(BridgePayloadSweep, ASegmentedPayloadWithoutASupportIsNotCleared) {
  // Deploy trial g17's second grasp: a segmented payload whose bottom took in counter and
  // depth-edge points floored three boxes at z 0.020-0.027 under a 0.030 counter top.
  // Its bottom floors nothing: with no support known, nothing is cleared.
  const double z = kSupportTop + kHalf[2];
  for (int i = 0; i <= 8; ++i) {
    place_hand(0.01 * i, z);
    send_octree();
    send_world_state(false, "vision_segmentation");
    spin_for(50ms);
  }
  spin_for(300ms);
  const std::lock_guard<std::mutex> lock(mutex_);
  EXPECT_TRUE(requests_.empty()) << "a segmented payload's bottom floored a clear";
}

TEST_F(BridgePayloadSweep, ASegmentedPayloadOverAnAttestedSupportIsClearedAboveIt) {
  const double z = kSupportTop + kHalf[2] + 0.02;
  for (int i = 0; i <= 8; ++i) {
    place_hand(0.01 * i, z);
    send_octree();
    send_world_state(true, "vision_segmentation");
    spin_for(50ms);
  }
  spin_for(300ms);
  const std::lock_guard<std::mutex> lock(mutex_);
  ASSERT_FALSE(requests_.empty()) << "a supported segmented payload is cleared";
  for (const auto& r : requests_) {
    EXPECT_GE(r.min.z, kSupportTop + 0.002 + kRes - 1e-6) << "cleared into the support";
  }
}

TEST_F(BridgePayloadSweep, AGripperClosurePayloadWithoutASupportIsNotCleared) {
  // DROID t7's second grasp: the jaw box (GRIPPER_CLOSURE fallback), no support known.
  const double z = kSupportTop + kHalf[2];
  for (int i = 0; i <= 8; ++i) {
    place_hand(0.01 * i, z);
    send_octree();
    send_world_state(false, "gripper_closure");
    spin_for(50ms);
  }
  spin_for(300ms);
  const std::lock_guard<std::mutex> lock(mutex_);
  EXPECT_TRUE(requests_.empty()) << "a box with no floor was cleared";
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
