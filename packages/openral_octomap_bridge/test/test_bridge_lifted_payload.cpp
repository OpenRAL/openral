// SPDX-License-Identifier: Apache-2.0
// A lifted payload's own interior stays cleared while its support witness lives.
//
// Isaac i64/i70 (2026-10-05, OpenArm, a 7.5 x 5 x 4.5 cm brick on a table,
// 15 mm octomap): the attach grid cleared the brick's TOP cell layer and
// withheld the two below it (the support band reaches half-width + attested
// depth + one voxel = 32.5 mm above the plane). The band is stated in the
// object frame, so it rose with the lifted payload; past ~5 mm of rise it
// claimed the top layer too and re-published it, inside the payload. The
// kernel never baselined those cells as embedded residue (absent from the
// attach grid), so only the witness exempted them — and the producer retires
// the witness at 15 mm. The kernel drops it on ingest; the grid it held until
// the bridge's next publish still carried the top layer:
// `safety.collision kind=world a=attached:<payload> b=voxel_N step=0
// min_distance_m=-0.0145`, on the brick's own top layer, 0.19 s after the
// retirement. The published-grid trace of the same trial's right-hand pick
// (`spark:~/livesim/trials/i70/cells.jsonl`) shows the layer go 24 -> 0 cells at
// the attach grid and come back 0 -> 9 -> 24 at ~5 mm of rise.
//
// Runs the REAL node in-process against a real `octomap::OcTree`, real TF and
// real DDS topics, on the trial's geometry.

#include <atomic>
#include <chrono>
#include <cmath>
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

constexpr double kRes = 0.015;
constexpr double kSupportTop = 0.48;  // 32 cells: a cell face, as the trial's -0.48
constexpr double kHalfZ = 0.0487;     // the trial's attached half height
constexpr double kHalfXY = 0.04;
// Cell centres: table 0.4725; brick layers 0.4875 / 0.5025 / 0.5175 (the brick
// is three cells tall, its top face at 0.525 — the trial's -0.435).
constexpr double kTableZ = kSupportTop - 0.5 * kRes;
constexpr double kBrickZ[3] = {kSupportTop + 0.5 * kRes, kSupportTop + 1.5 * kRes,
                               kSupportTop + 2.5 * kRes};
constexpr double kBrickXY[2] = {-0.5 * kRes, 0.5 * kRes};
constexpr char kHand[] = "hand";

class BridgeLiftedPayload : public ::testing::Test {
protected:
  static void SetUpTestSuite() { rclcpp::init(0, nullptr); }
  static void TearDownTestSuite() { rclcpp::shutdown(); }

  void SetUp() override {
    std::vector<rclcpp::Parameter> overrides{
        {"base_frame", "base_link"},
        {"octomap_topic", "/test_octomap_lift"},
        {"output_topic", "/test_world_voxels_lift"},
        {"world_state_topic", "/test_world_state_lift"},
        {"resolution", kRes},
        {"coverage_radius_m", 0.15},
        {"publish_rate_hz", 20.0},
    };
    rclcpp::NodeOptions options;
    options.parameter_overrides(overrides);
    bridge_ = std::make_shared<bridge::OctomapVoxelBridge>(options);
    peer_ = std::make_shared<rclcpp::Node>("bridge_lifted_payload_peer");
    tf_ = std::make_unique<tf2_ros::StaticTransformBroadcaster>(peer_);
    octomap_pub_ = peer_->create_publisher<octomap_msgs::msg::Octomap>("/test_octomap_lift",
                                                                       rclcpp::QoS(1).reliable());
    world_state_pub_ = peer_->create_publisher<openral_msgs::msg::WorldStateStamped>(
        "/test_world_state_lift", rclcpp::QoS(1).reliable());
    voxel_sub_ = peer_->create_subscription<openral_msgs::msg::OccupancyVoxels>(
        "/test_world_voxels_lift", rclcpp::QoS(1).reliable(),
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

  // The hand holds the payload's centre; the payload's lower face sits `rise`
  // above the support top.
  void place_hand(double rise) {
    geometry_msgs::msg::TransformStamped base;
    base.header.stamp = peer_->now();
    base.header.frame_id = "map";
    base.child_frame_id = "base_link";
    base.transform.rotation.w = 1.0;
    geometry_msgs::msg::TransformStamped hand = base;
    hand.header.frame_id = "base_link";
    hand.child_frame_id = kHand;
    hand.transform.translation.z = kSupportTop + kHalfZ + rise;
    tf_->sendTransform({base, hand});
  }

  // The map the trial's octree held: the table and the brick's three layers,
  // which stay where they were measured while the brick is carried (occluded
  // by the hand and the brick itself; no clearing ray reaches them).
  void send_octree() {
    octomap::OcTree tree(kRes);
    const auto mark = [&tree](double x, double y, double z) {
      const octomap::point3d p(static_cast<float>(x), static_cast<float>(y), static_cast<float>(z));
      tree.updateNode(p, true);
      tree.updateNode(p, true);
    };
    // 6 x 6 cells, all under the payload's footprint (inside the witness patch).
    for (int i = -3; i < 3; ++i) {
      for (int j = -3; j < 3; ++j) {
        mark((i + 0.5) * kRes, (j + 0.5) * kRes, kTableZ);
      }
    }
    for (const double z : kBrickZ) {
      for (const double x : kBrickXY) {
        for (const double y : kBrickXY) {
          mark(x, y, z);
        }
      }
    }
    octomap_msgs::msg::Octomap msg;
    ASSERT_TRUE(octomap_msgs::binaryMapToMsg(tree, msg));
    msg.header.frame_id = "map";
    msg.header.stamp = peer_->now();
    octomap_pub_->publish(msg);
  }

  // The region payload resting on the support, with the HAL's plane witness:
  // contact under the centre on the lower face, normal +z, the footprint's
  // half-diagonal, depth = the extrinsic bound capped at the kernel's 10 mm.
  void send_world_state(bool witness) {
    openral_msgs::msg::AttachedCollisionObject object;
    object.object_id = "approach:brick";
    object.attach_link = kHand;
    object.pose_in_link.orientation.w = 1.0;
    openral_msgs::msg::AttachedCollisionPrimitive box;
    box.shape_type = openral_msgs::msg::AttachedCollisionPrimitive::SHAPE_BOX;
    box.shape_dimensions = {kHalfXY, kHalfXY, kHalfZ};
    box.pose_in_object.orientation.w = 1.0;
    object.primitives.push_back(box);
    if (witness) {
      object.support_contact_valid = true;
      auto& w = object.support_contact;
      w.support_id = "map_support_under:approach:brick";
      w.contact_point_in_object.z = -kHalfZ;
      w.contact_normal_in_object.z = 1.0;
      w.patch_radius_m = std::hypot(kHalfXY, kHalfXY);
      w.max_penetration_m = 0.01;
      w.stamp_ns = 1791210577359385376;
    }
    openral_msgs::msg::WorldStateStamped state;
    state.header.stamp = peer_->now();
    state.attached_objects.push_back(object);
    state.attachment_revision = 4;
    world_state_pub_->publish(state);
  }

  // Grids for ~0.5 s at the current hand pose; returns occupied cells per brick
  // layer and the table's, from the last grid.
  struct Counts {
    int table{0};
    int layer[3]{0, 0, 0};
  };
  Counts run(bool witness) {
    for (int i = 0; i < 3; ++i) {
      send_octree();
      send_world_state(witness);
      spin_for(170ms);
    }
    const std::lock_guard<std::mutex> lock(mutex_);
    Counts c;
    EXPECT_GT(grids_.load(), 0) << "no grid published";
    EXPECT_NEAR(std::fabs(last_.orientation.w), 1.0, 1e-9) << "test assumes an axis-aligned grid";
    const auto sx = static_cast<std::size_t>(last_.size_x);
    const auto sy = static_cast<std::size_t>(last_.size_y);
    for (std::size_t idx = 0; idx < last_.occupancy.size(); ++idx) {
      if (last_.occupancy[idx] == 0) {
        continue;
      }
      const double z = last_.origin.z + (static_cast<double>(idx / (sx * sy)) + 0.5) * kRes;
      const double y = last_.origin.y + (static_cast<double>((idx / sx) % sy) + 0.5) * kRes;
      const double x = last_.origin.x + (static_cast<double>(idx % sx) + 0.5) * kRes;
      if (std::fabs(z - kTableZ) < 1e-6) {
        ++c.table;
      }
      for (int k = 0; k < 3; ++k) {
        if (std::fabs(z - kBrickZ[k]) < 1e-6 && std::fabs(x) < kRes && std::fabs(y) < kRes) {
          ++c.layer[k];
        }
      }
    }
    return c;
  }

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

TEST_F(BridgeLiftedPayload, TheBandStaysOnTheSupportWhileThePayloadRises) {
  // ATTACH: the band (32.5 mm) withholds the table and the brick's lower two
  // layers (+7.5, +22.5 mm); the top layer (+37.5 mm) is the payload's and clears.
  place_hand(0.0);
  const Counts attach = run(true);
  EXPECT_EQ(attach.table, 36) << "the support the witness is measured against stays";
  EXPECT_EQ(attach.layer[0], 4);
  EXPECT_EQ(attach.layer[1], 4);
  EXPECT_EQ(attach.layer[2], 0) << "the brick's top layer is payload: cleared at attach";

  // 12 mm up: inside the producer's 15 mm rise tolerance, so the witness lives.
  // A band riding the object frame now reaches +44.5 mm over the table and
  // claims the top layer again — inside the payload, never baselined.
  place_hand(0.012);
  const Counts lifted = run(true);
  EXPECT_EQ(lifted.layer[2], 0)
      << "the payload's own top layer came back inside the lifted payload — the kernel "
         "has no residue baseline for it, so the witness's retirement stops the arm on it";
  EXPECT_EQ(lifted.layer[0], 4) << "cells withheld since the attach grid stay withheld";
  EXPECT_EQ(lifted.layer[1], 4);
  EXPECT_EQ(lifted.table, 36);

  // Retired at the rise tolerance: nothing withholds, every brick cell in reach clears.
  place_hand(0.016);
  const Counts retired = run(false);
  EXPECT_EQ(retired.layer[0] + retired.layer[1] + retired.layer[2], 0);
}

}  // namespace
