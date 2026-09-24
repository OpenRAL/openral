// SPDX-License-Identifier: Apache-2.0
// The bridge must not republish an octree that has stopped arriving.
//
// `octomap_server` publishes `/octomap_binary` only when it inserts a cloud, so
// a dead camera silences it. The bridge used to keep re-rasterizing the LAST
// octree on its 10 Hz timer and stamping each grid `now()`, and the safety
// kernel — which times voxel freshness from receipt — kept seeing a fresh
// world. Its fail-closed `DROP_VOXEL_UNAVAILABLE` never fired, and anything
// that entered the workspace afterwards was invisible (observed on Thor
// 2026-09-24: ZED driver stopped, `/octomap_binary` silent, `/openral/world_voxels`
// still 7.1 Hz). Hazard log Entry 033.
//
// These run the REAL node in-process against a real `octomap::OcTree`
// serialized by `octomap_msgs`, a real static TF and real DDS topics.

#include <atomic>
#include <chrono>
#include <cmath>
#include <limits>
#include <memory>
#include <string>
#include <thread>

#include <geometry_msgs/msg/transform_stamped.hpp>
#include <gtest/gtest.h>
#include <octomap/OcTree.h>
#include <octomap_msgs/conversions.h>
#include <octomap_msgs/msg/octomap.hpp>
#include <tf2_ros/static_transform_broadcaster.h>

#include <openral_msgs/msg/occupancy_voxels.hpp>
#include <rclcpp/rclcpp.hpp>

#include "octomap_voxel_bridge.hpp"
#include "openral_octomap_bridge/octree_freshness.hpp"

namespace bridge = openral_octomap_bridge;
using namespace std::chrono_literals;

namespace {

// ── The gate itself ──────────────────────────────────────────────────────────

TEST(OctreeFreshness, AnOctreeInsideTheBoundIsFreshAndOnePastItIsNot) {
  EXPECT_TRUE(bridge::octree_is_fresh(0.0, 0.5));
  EXPECT_TRUE(bridge::octree_is_fresh(0.5, 0.5));
  EXPECT_FALSE(bridge::octree_is_fresh(0.5 + 1e-9, 0.5));
  EXPECT_FALSE(bridge::octree_is_fresh(3600.0, 0.5));
}

TEST(OctreeFreshness, AnAgeTheClockCannotExplainIsNotFresh) {
  // A receipt time in the future (a sim clock reset, a clock stepped back)
  // would otherwise read as fresh until the clock caught up again.
  EXPECT_FALSE(bridge::octree_is_fresh(-0.001, 0.5));
  EXPECT_FALSE(bridge::octree_is_fresh(std::numeric_limits<double>::quiet_NaN(), 0.5));
  EXPECT_FALSE(bridge::octree_is_fresh(std::numeric_limits<double>::infinity(), 0.5));
}

TEST(OctreeFreshness, AnUnusableBoundMakesNothingFresh) {
  for (const double bad : {0.0, -0.5, std::numeric_limits<double>::quiet_NaN(),
                           std::numeric_limits<double>::infinity()}) {
    EXPECT_FALSE(bridge::valid_max_octree_age(bad)) << bad;
    EXPECT_FALSE(bridge::octree_is_fresh(0.0, bad)) << bad;
  }
  EXPECT_TRUE(bridge::valid_max_octree_age(0.5));
}

// ── The node ─────────────────────────────────────────────────────────────────

constexpr double kMaxOctreeAgeS = 0.4;

class BridgeStaleness : public ::testing::Test {
protected:
  static void SetUpTestSuite() { rclcpp::init(0, nullptr); }
  static void TearDownTestSuite() { rclcpp::shutdown(); }

  // Topics are per test so no two tests in this process hear each other.
  void start(double max_octree_age_s) {
    const auto* info = ::testing::UnitTest::GetInstance()->current_test_info();
    const std::string suffix = info->name();
    octomap_topic_ = "/test_octomap_" + suffix;
    voxel_topic_ = "/test_world_voxels_" + suffix;

    rclcpp::NodeOptions options;
    options.parameter_overrides({
        {"base_frame", "base_link"},
        {"octomap_topic", octomap_topic_},
        {"output_topic", voxel_topic_},
        {"coverage_radius_m", 0.3},
        {"publish_rate_hz", 20.0},
        {"max_octree_age_s", max_octree_age_s},
    });
    bridge_ = std::make_shared<bridge::OctomapVoxelBridge>(options);
    peer_ = std::make_shared<rclcpp::Node>("bridge_staleness_peer_" + suffix);

    tf_ = std::make_unique<tf2_ros::StaticTransformBroadcaster>(peer_);
    geometry_msgs::msg::TransformStamped tf;
    tf.header.stamp = peer_->now();
    tf.header.frame_id = "map";
    tf.child_frame_id = "base_link";
    tf.transform.rotation.w = 1.0;
    tf_->sendTransform(tf);

    octomap_pub_ = peer_->create_publisher<octomap_msgs::msg::Octomap>(octomap_topic_,
                                                                       rclcpp::QoS(1).reliable());
    voxel_sub_ = peer_->create_subscription<openral_msgs::msg::OccupancyVoxels>(
        voxel_topic_, rclcpp::QoS(1).reliable(),
        [this](openral_msgs::msg::OccupancyVoxels::SharedPtr) { ++grids_; });

    executor_.add_node(bridge_);
    executor_.add_node(peer_);
    // Let discovery settle before the first octree goes out.
    spin_for(300ms);
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

  // One occupied cell inside the coverage ball, as octomap_server sends it.
  void send_octree() {
    octomap::OcTree tree(0.05);
    tree.updateNode(octomap::point3d(0.1F, 0.0F, 0.5F), true);
    tree.updateNode(octomap::point3d(0.1F, 0.0F, 0.5F), true);
    octomap_msgs::msg::Octomap msg;
    ASSERT_TRUE(octomap_msgs::binaryMapToMsg(tree, msg));
    msg.header.frame_id = "map";
    msg.header.stamp = peer_->now();
    octomap_pub_->publish(msg);
  }

  std::string octomap_topic_;
  std::string voxel_topic_;
  std::shared_ptr<bridge::OctomapVoxelBridge> bridge_;
  std::shared_ptr<rclcpp::Node> peer_;
  std::unique_ptr<tf2_ros::StaticTransformBroadcaster> tf_;
  rclcpp::Publisher<octomap_msgs::msg::Octomap>::SharedPtr octomap_pub_;
  rclcpp::Subscription<openral_msgs::msg::OccupancyVoxels>::SharedPtr voxel_sub_;
  rclcpp::executors::SingleThreadedExecutor executor_;
  std::atomic<int> grids_{0};
};

TEST_F(BridgeStaleness, AFreshOctreeIsPublishedAndAStoppedOneGoesSilentUntilTheNext) {
  start(kMaxOctreeAgeS);

  send_octree();
  spin_for(250ms);
  const int while_fresh = grids_.load();
  EXPECT_GT(while_fresh, 0) << "a fresh octree must reach the kernel";

  // Past the bound (0.4 s) plus a timer period and delivery slack, the octree
  // has stopped arriving: the bridge must add nothing more, however long the
  // silence lasts. Pre-fix it kept publishing the frozen map at the timer rate.
  spin_for(300ms);
  const int at_bound = grids_.load();
  spin_for(800ms);
  EXPECT_EQ(grids_.load(), at_bound)
      << "the bridge republished an octree older than max_octree_age_s as a fresh grid";

  // The next octree resumes publication on its own.
  send_octree();
  spin_for(250ms);
  EXPECT_GT(grids_.load(), at_bound) << "a new octree must resume publication";
}

TEST_F(BridgeStaleness, AnOctreeKeptArrivingKeepsTheGridFlowing) {
  // The republish's legitimate job — a grid that follows the robot through TF
  // between octrees — is untouched while octrees arrive inside the bound. At
  // Thor's measured 3.2 Hz octomap cadence every gap is ~0.31 s < 0.4 s.
  start(kMaxOctreeAgeS);
  int last = 0;
  for (int i = 0; i < 6; ++i) {
    send_octree();
    spin_for(310ms);
    EXPECT_GT(grids_.load(), last) << "octree " << i << " produced no grid";
    last = grids_.load();
  }
}

TEST_F(BridgeStaleness, TheGridCarriesTheOctreesCaptureStampAsSourceStamp) {
  // `header.stamp` is when the grid was produced (fresh on every republish);
  // `source_stamp` is the octree's own stamp, which octomap_server takes from
  // the cloud it inserted. Only the latter says how old the world is, and it
  // is what the kernel budgets.
  start(kMaxOctreeAgeS);
  openral_msgs::msg::OccupancyVoxels last;
  std::atomic<bool> got{false};
  auto sub = peer_->create_subscription<openral_msgs::msg::OccupancyVoxels>(
      voxel_topic_, rclcpp::QoS(1).reliable(),
      [&](openral_msgs::msg::OccupancyVoxels::SharedPtr m) {
        last = *m;
        got = true;
      });

  const rclcpp::Time capture = peer_->now() - rclcpp::Duration::from_seconds(0.2);
  octomap::OcTree tree(0.05);
  tree.updateNode(octomap::point3d(0.1F, 0.0F, 0.5F), true);
  octomap_msgs::msg::Octomap msg;
  ASSERT_TRUE(octomap_msgs::binaryMapToMsg(tree, msg));
  msg.header.frame_id = "map";
  msg.header.stamp = capture;
  octomap_pub_->publish(msg);
  spin_for(250ms);
  ASSERT_TRUE(got.load()) << "no grid published";

  const rclcpp::Time source(last.source_stamp, peer_->get_clock()->get_clock_type());
  EXPECT_EQ(source.nanoseconds(), capture.nanoseconds())
      << "source_stamp must be the octree's stamp, unchanged";
  const rclcpp::Time produced(last.header.stamp, peer_->get_clock()->get_clock_type());
  EXPECT_GT(produced.nanoseconds(), capture.nanoseconds())
      << "header.stamp is the production time, after capture";
  EXPECT_LT((peer_->now() - produced).seconds(), 0.25) << "header.stamp must stay fresh";
}

TEST_F(BridgeStaleness, AnUnusableBoundPublishesNothing) {
  // Fail closed, as an unset coverage radius does: silence, which the kernel
  // turns into DROP_VOXEL_UNAVAILABLE, never a map of unknown age.
  start(0.0);
  send_octree();
  spin_for(500ms);
  EXPECT_EQ(grids_.load(), 0);
}

}  // namespace
