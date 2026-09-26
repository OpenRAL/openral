// SPDX-License-Identifier: Apache-2.0
// The bridge keeps its `publish_rate_hz` at the deploy grid size.
//
// On the Orin OpenArm cell (2026-09-25) a probe reported /openral/world_voxels
// at 4.4-6.1 Hz while /octomap_binary ran at 9.8 Hz, and the OpenArm scenes'
// world-voxel deadline was raised to 1.5 s to tolerate it. The bridge was not
// slow: its own throttled log landed on every 100 ms tick for the whole run,
// and a replay of that run's real octree through this node on the Orin CPU
// reached a RELIABLE subscriber at 10.17 Hz (max gap 0.112 s). The probe
// subscribed BEST_EFFORT, and a 1.19 MB grid (106^3 cells at 0.02 m, radius
// 1.05 m) is a fragmented sample a best-effort reader drops whole whenever it
// misses one fragment — 3.27 Hz with 3.6 s gaps on a dev laptop for the very
// same publisher. The safety kernel subscribes RELIABLE.
//
// These pin the two halves of that: the per-octree work is a small fraction of
// the 100 ms period on a ray-cast map (free space included, which the kitchen
// scene in test_octree_to_grid has none of), and the real node, at the deploy
// grid size, reaches a reliable KEEP_LAST(1) subscriber — the kernel's QoS — at
// its timer rate. Bounds are generous; they catch a regression of the kind
// suspected here (a per-message cost near the period), not jitter.

#include <atomic>
#include <chrono>
#include <cmath>
#include <iostream>
#include <memory>
#include <string>

#include <geometry_msgs/msg/transform_stamped.hpp>
#include <gtest/gtest.h>
#include <octomap/OcTree.h>
#include <octomap_msgs/conversions.h>
#include <octomap_msgs/msg/octomap.hpp>
#include <tf2_ros/static_transform_broadcaster.h>

#include <openral_msgs/msg/occupancy_voxels.hpp>
#include <rclcpp/rclcpp.hpp>

#include "octomap_voxel_bridge.hpp"
#include "openral_octomap_bridge/octree_to_grid.hpp"

namespace bridge = openral_octomap_bridge;
using namespace std::chrono_literals;

namespace {

/// Deploy geometry for a real arm: `_octomap_resolution("real")` and
/// `_octomap_coverage_radius()` in deploy_e2e.launch.py — 106^3 cells.
constexpr double kResolution = 0.02;
constexpr double kCoverageRadius = 1.05;

/// First surface of a tabletop scene (floor, table, a box on it, a wall 3 m
/// out) along `dir` from `origin`; false past the 4 m sensor range.
bool first_hit(const octomap::point3d& origin, const octomap::point3d& dir, octomap::point3d& hit) {
  double best = 4.0;
  // The plane `axis == at`, bounded on the other two axes (in cyclic order).
  const auto plane = [&](int axis, double at, double lo_a, double hi_a, double lo_b, double hi_b) {
    const double d = dir(axis);
    if (std::fabs(d) < 1e-9) {
      return;
    }
    const double t = (at - origin(axis)) / d;
    if (!(t > 0.0 && t < best)) {
      return;
    }
    const octomap::point3d p = origin + dir * static_cast<float>(t);
    const int a = (axis + 1) % 3;
    const int b = (axis + 2) % 3;
    if (p(a) >= lo_a && p(a) <= hi_a && p(b) >= lo_b && p(b) <= hi_b) {
      best = t;
    }
  };
  plane(2, 0.0, -10, 10, -10, 10);       // floor (x, y)
  plane(2, 0.75, 0.3, 1.1, -0.6, 0.6);   // table top
  plane(2, 0.95, 0.6, 0.75, -0.1, 0.1);  // box lid
  plane(0, 0.6, -0.1, 0.1, 0.75, 0.95);  // box face (y, z)
  plane(0, 3.0, -10, 10, 0, 3);          // wall
  if (best >= 4.0) {
    return false;
  }
  hit = origin + dir * static_cast<float>(best);
  return true;
}

/// A map built the way octomap_server builds one: ray-cast depth scans from a
/// head camera panning over the table, so free space is carved and stored.
///
/// ~79k leaves and ~5.1k occupied cells in the ball — the occupied count of the
/// Orin run, on a map 5-6x that run's real one (14k leaves, replayed from its
/// recorded self-filtered ZED clouds). Deserialization walks the WHOLE map, not
/// the ball: 0.7 ms at 14k leaves, ~9 ms here, ~60 ms at 437k leaves (this scene
/// at a 4 m range and a quarter the rays) on a dev laptop. A room-scale map
/// could therefore reach the period; this bench-scale one does not.
octomap::OcTree ray_cast_map() {
  octomap::OcTree tree(kResolution);
  tree.setOccupancyThres(0.6);  // deploy_e2e's real-map values
  tree.setProbMiss(0.4);
  tree.setClampingThresMax(0.97);
  const octomap::point3d camera(0.0F, 0.0F, 1.2F);
  for (int scan = 0; scan < 6; ++scan) {
    const float pan = 0.4F * std::sin(static_cast<float>(scan));
    octomap::Pointcloud cloud;
    for (int v = 0; v < 180; ++v) {
      for (int u = 0; u < 320; ++u) {
        const float az = pan + (static_cast<float>(u) / 319.0F - 0.5F) * 1.9F;
        const float el = -0.25F - (static_cast<float>(v) / 179.0F) * 0.95F;
        const octomap::point3d dir(std::cos(el) * std::cos(az), std::cos(el) * std::sin(az),
                                   std::sin(el));
        octomap::point3d hit;
        if (first_hit(camera, dir, hit)) {
          cloud.push_back(hit);
        }
      }
    }
    tree.insertPointCloud(cloud, camera, 2.0, false, true);
  }
  tree.prune();
  return tree;
}

bridge::GridSpec deploy_spec() {
  bridge::GridSpec spec;
  spec.center[2] = 0.5;
  spec.radius = kCoverageRadius;
  return spec;
}

}  // namespace

TEST(BridgeRateBudget, ARayCastMapCostsASmallFractionOfThePublishPeriod) {
  const octomap::OcTree map = ray_cast_map();
  octomap_msgs::msg::Octomap msg;
  ASSERT_TRUE(octomap_msgs::binaryMapToMsg(map, msg));
  msg.header.frame_id = "map";

  // Everything the node does per octree and per tick: deserialize, rasterize.
  constexpr int kIterations = 10;
  std::size_t leaves = 0;
  std::size_t occupied = 0;
  const auto started = std::chrono::steady_clock::now();
  for (int i = 0; i < kIterations; ++i) {
    std::unique_ptr<octomap::AbstractOcTree> tree(octomap_msgs::msgToMap(msg));
    auto* octree = dynamic_cast<octomap::OcTree*>(tree.get());
    ASSERT_NE(octree, nullptr);
    const auto grid = bridge::rasterize_octree_to_grid(*octree, tf2::Transform::getIdentity(),
                                                       deploy_spec(), "base_link");
    ASSERT_EQ(grid.size_x, 106U);
    leaves = octree->getNumLeafNodes();
    occupied = 0;
    for (const auto cell : grid.occupancy) {
      occupied += cell;
    }
  }
  const double per_octree_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started)
          .count() /
      kIterations;
  std::cout << "[ bench    ] " << leaves << " leaves, " << occupied
            << " occupied cells: " << per_octree_ms << " ms per octree\n";
  // A map of this kind, not a toy: free space dominates the leaf count, and
  // the table and box put thousands of occupied cells inside the ball.
  EXPECT_GT(leaves, 20000U);
  EXPECT_GT(occupied, 2000U);
  EXPECT_LT(per_octree_ms, 50.0) << "deserialize + rasterize took " << per_octree_ms
                                 << " ms of the 100 ms period (" << leaves << " leaves)";
}

TEST(BridgeRateBudget, TheDeployGridReachesAReliableSubscriberAtTheTimerRate) {
  rclcpp::init(0, nullptr);
  {
    const std::string octomap_topic = "/test_rate_octomap";
    const std::string voxel_topic = "/test_rate_world_voxels";
    rclcpp::NodeOptions options;
    options.parameter_overrides({
        {"base_frame", "base_link"},
        {"octomap_topic", octomap_topic},
        {"output_topic", voxel_topic},
        {"resolution", kResolution},
        {"coverage_radius_m", kCoverageRadius},
        {"publish_rate_hz", 10.0},
        {"max_octree_age_s", 1.0},
    });
    auto node = std::make_shared<bridge::OctomapVoxelBridge>(options);
    auto peer = std::make_shared<rclcpp::Node>("bridge_rate_peer");

    tf2_ros::StaticTransformBroadcaster tf(peer);
    geometry_msgs::msg::TransformStamped identity;
    identity.header.stamp = peer->now();
    identity.header.frame_id = "map";
    identity.child_frame_id = "base_link";
    identity.transform.rotation.w = 1.0;
    tf.sendTransform(identity);

    octomap_msgs::msg::Octomap octree_msg;
    ASSERT_TRUE(octomap_msgs::binaryMapToMsg(ray_cast_map(), octree_msg));
    octree_msg.header.frame_id = "map";
    auto octomap_pub = peer->create_publisher<octomap_msgs::msg::Octomap>(
        octomap_topic, rclcpp::QoS(1).reliable());
    // As octomap_server does: a new octree per inserted cloud, ~10 Hz.
    auto octomap_timer = peer->create_wall_timer(100ms, [&]() {
      octree_msg.header.stamp = peer->now();
      octomap_pub->publish(octree_msg);
    });

    // The safety kernel's subscription: RELIABLE, VOLATILE, KEEP_LAST(1).
    std::atomic<int> grids{0};
    std::atomic<std::size_t> cells{0};
    auto sub = peer->create_subscription<openral_msgs::msg::OccupancyVoxels>(
        voxel_topic, rclcpp::QoS(1).reliable(),
        [&](openral_msgs::msg::OccupancyVoxels::SharedPtr grid) {
          ++grids;
          cells = grid->occupancy.size();
        });

    rclcpp::executors::SingleThreadedExecutor executor;
    executor.add_node(node);
    executor.add_node(peer);
    const auto spin_for = [&](std::chrono::milliseconds span) {
      const auto until = std::chrono::steady_clock::now() + span;
      while (std::chrono::steady_clock::now() < until) {
        executor.spin_some(10ms);
      }
    };
    // Discovery, TF and the first octree.
    spin_for(1500ms);
    ASSERT_GT(grids.load(), 0) << "no grid within 1.5 s";
    EXPECT_EQ(cells.load(), 106U * 106U * 106U);

    grids = 0;
    constexpr auto kWindow = 4000ms;
    spin_for(kWindow);
    const double hz = grids.load() / std::chrono::duration<double>(kWindow).count();
    // A 10 Hz timer; the Orin probe read 4.4-6.1. Above 7 the node is not the
    // bottleneck it was suspected to be.
    EXPECT_GT(hz, 7.0) << "a reliable subscriber saw " << hz << " Hz of 1.19 MB grids";
    executor.remove_node(peer);
    executor.remove_node(node);
  }
  rclcpp::shutdown();
}
