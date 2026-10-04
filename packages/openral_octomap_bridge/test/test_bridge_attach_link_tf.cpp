// SPDX-License-Identifier: Apache-2.0
// A held payload is cleared on a cell whose TF tree renames the hand link.
//
// `AttachedCollisionObject.attach_link` carries the MANIFEST link name (the
// kernel's collision model uses manifest names). The real OpenArm cell's TF
// tree publishes the same hand body as `openarm_left_ee_base_link`, not
// `openarm_left_link7`, so the bridge's `lookupTransform(base, attach_link)`
// failed and the payload stayed an obstacle to its own gripper.
// `attach_link_tf_frames` maps the one to the other.
//
// These run the REAL node in-process against a real `octomap::OcTree`, real
// static TF that knows ONLY the vendor frame, and real DDS topics. The
// obstacle cell beside the payload must survive in every case: the mapping
// may only clear what the payload's own volume explains.

#include <atomic>
#include <chrono>
#include <cstddef>
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
#include <tf2_ros/static_transform_broadcaster.h>

#include <openral_msgs/msg/attached_collision_object.hpp>
#include <openral_msgs/msg/attached_collision_primitive.hpp>
#include <openral_msgs/msg/occupancy_voxels.hpp>
#include <openral_msgs/msg/world_state_stamped.hpp>
#include <rclcpp/rclcpp.hpp>

#include "octomap_voxel_bridge.hpp"

namespace bridge = openral_octomap_bridge;
using namespace std::chrono_literals;

namespace {

constexpr char kManifestHand[] = "openarm_left_link7";
constexpr char kVendorHand[] = "openarm_left_ee_base_link";
// The hand (and the payload it holds) 0.1 m in front of the base at z 0.5; a
// real obstacle 0.1 m behind it. 0.2 m apart, far beyond any clearing reach.
constexpr float kPayloadX = 0.1F;
constexpr float kObstacleX = -0.1F;
constexpr float kZ = 0.5F;

class BridgeAttachLinkTf : public ::testing::Test {
protected:
  static void SetUpTestSuite() { rclcpp::init(0, nullptr); }
  static void TearDownTestSuite() { rclcpp::shutdown(); }

  void start(const std::vector<std::string>& attach_link_tf_frames) {
    const std::string suffix = ::testing::UnitTest::GetInstance()->current_test_info()->name();
    octomap_topic_ = "/test_octomap_" + suffix;
    voxel_topic_ = "/test_world_voxels_" + suffix;
    world_state_topic_ = "/test_world_state_" + suffix;

    std::vector<rclcpp::Parameter> overrides{
        {"base_frame", "base_link"},    {"octomap_topic", octomap_topic_},
        {"output_topic", voxel_topic_}, {"world_state_topic", world_state_topic_},
        {"coverage_radius_m", 0.3},     {"publish_rate_hz", 20.0},
    };
    if (!attach_link_tf_frames.empty()) {
      overrides.emplace_back("attach_link_tf_frames", attach_link_tf_frames);
    }
    rclcpp::NodeOptions options;
    options.parameter_overrides(overrides);
    bridge_ = std::make_shared<bridge::OctomapVoxelBridge>(options);
    peer_ = std::make_shared<rclcpp::Node>("bridge_attach_link_tf_peer_" + suffix);

    // The cell's TF tree: the vendor hand frame exists, the manifest one does not.
    tf_ = std::make_unique<tf2_ros::StaticTransformBroadcaster>(peer_);
    geometry_msgs::msg::TransformStamped base;
    base.header.stamp = peer_->now();
    base.header.frame_id = "map";
    base.child_frame_id = "base_link";
    base.transform.rotation.w = 1.0;
    geometry_msgs::msg::TransformStamped hand = base;
    hand.header.frame_id = "base_link";
    hand.child_frame_id = kVendorHand;
    hand.transform.translation.x = kPayloadX;
    hand.transform.translation.z = kZ;
    tf_->sendTransform({base, hand});

    octomap_pub_ = peer_->create_publisher<octomap_msgs::msg::Octomap>(octomap_topic_,
                                                                       rclcpp::QoS(1).reliable());
    world_state_pub_ = peer_->create_publisher<openral_msgs::msg::WorldStateStamped>(
        world_state_topic_, rclcpp::QoS(1).reliable());
    voxel_sub_ = peer_->create_subscription<openral_msgs::msg::OccupancyVoxels>(
        voxel_topic_, rclcpp::QoS(1).reliable(),
        [this](openral_msgs::msg::OccupancyVoxels::SharedPtr msg) {
          const std::lock_guard<std::mutex> lock(mutex_);
          last_ = *msg;
          ++grids_;
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

  // Two occupied cells, as octomap_server sends them: the payload's own
  // silhouette at the hand, and an obstacle 0.2 m away.
  void send_octree() {
    octomap::OcTree tree(0.05);
    for (const float x : {kPayloadX, kObstacleX}) {
      tree.updateNode(octomap::point3d(x, 0.0F, kZ), true);
      tree.updateNode(octomap::point3d(x, 0.0F, kZ), true);
    }
    octomap_msgs::msg::Octomap msg;
    ASSERT_TRUE(octomap_msgs::binaryMapToMsg(tree, msg));
    msg.header.frame_id = "map";
    msg.header.stamp = peer_->now();
    octomap_pub_->publish(msg);
  }

  // A 2 cm sphere held at the hand, attached to the MANIFEST link name.
  void send_world_state() {
    openral_msgs::msg::AttachedCollisionObject object;
    object.object_id = "box";
    object.attach_link = kManifestHand;
    object.pose_in_link.orientation.w = 1.0;
    openral_msgs::msg::AttachedCollisionPrimitive sphere;
    sphere.shape_type = openral_msgs::msg::AttachedCollisionPrimitive::SHAPE_SPHERE;
    sphere.shape_dimensions = {0.02};
    sphere.pose_in_object.orientation.w = 1.0;
    object.primitives.push_back(sphere);

    openral_msgs::msg::WorldStateStamped state;
    state.header.stamp = peer_->now();
    state.attached_objects.push_back(object);
    state.attachment_revision = 1;
    world_state_pub_->publish(state);
  }

  // Occupied cells of the last grid after ~1 s of live octrees + attachments.
  std::size_t run_and_count_occupied() {
    for (int i = 0; i < 6; ++i) {
      send_octree();
      send_world_state();
      spin_for(170ms);
    }
    const std::lock_guard<std::mutex> lock(mutex_);
    EXPECT_GT(grids_.load(), 0) << "no grid published";
    std::size_t n = 0;
    for (const auto v : last_.occupancy) {
      n += (v != 0) ? 1U : 0U;
    }
    return n;
  }

  std::string octomap_topic_;
  std::string voxel_topic_;
  std::string world_state_topic_;
  std::shared_ptr<bridge::OctomapVoxelBridge> bridge_;
  std::shared_ptr<rclcpp::Node> peer_;
  std::unique_ptr<tf2_ros::StaticTransformBroadcaster> tf_;
  rclcpp::Publisher<octomap_msgs::msg::Octomap>::SharedPtr octomap_pub_;
  rclcpp::Publisher<openral_msgs::msg::WorldStateStamped>::SharedPtr world_state_pub_;
  rclcpp::Subscription<openral_msgs::msg::OccupancyVoxels>::SharedPtr voxel_sub_;
  rclcpp::executors::SingleThreadedExecutor executor_;
  std::mutex mutex_;
  openral_msgs::msg::OccupancyVoxels last_;
  std::atomic<int> grids_{0};
};

TEST_F(BridgeAttachLinkTf, WithoutTheMappingTheHeldPayloadStaysAnObstacle) {
  // The pre-fix behaviour on the real cell, kept as the unmapped default: the
  // manifest link is not a TF frame, the lookup fails, nothing is cleared.
  start({});
  EXPECT_EQ(run_and_count_occupied(), 2U);
}

TEST_F(BridgeAttachLinkTf, TheMappingClearsThePayloadAndOnlyThePayload) {
  start({std::string(kManifestHand) + "=" + kVendorHand});
  EXPECT_EQ(run_and_count_occupied(), 1U) << "the payload's cell must clear; the obstacle stays";
}

TEST_F(BridgeAttachLinkTf, AMalformedMappingClearsNothing) {
  // Refused whole: every link at identity, the lookup fails, fail closed.
  start({std::string(kManifestHand) + "=" + kVendorHand, "no_separator"});
  EXPECT_EQ(run_and_count_occupied(), 2U);
}

}  // namespace
