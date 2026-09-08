// SPDX-License-Identifier: Apache-2.0
// A self-filtered depth ray must still reach OctoMap as a clearing ray.
//
// Deploy-sim tuning (`packages/openral_rskill_ros/launch/deploy_e2e.launch.py`):
// `occupancy_thres 0.8` (one hit never makes a safety voxel; needs a second
// frame to confirm), `sensor_model.max 0.85` (one clearing ray retires a
// confirmed hit). Pre-fix bug: the bridge built octomap's cloud from the depth
// raster, whose `0.0` means "no measurement", and the self-filter collapsed
// "ray hit the robot's own arm" onto that same `0.0` — no ray, so the arm's
// silhouette became a write-only map region. Matches the 2026-08-14
// `panda_link1` stop against `voxel_76001` (no physical geometry backed it).
//
// Pins both the transient-rejection threshold and the restored-ray clearing
// on a real `octomap::OcTree` with the deploy numbers.

#include <gtest/gtest.h>
#include <octomap/OcTree.h>
#include <octomap/Pointcloud.h>
#include <tf2/LinearMath/Transform.h>

#include "openral_octomap_bridge/octree_to_grid.hpp"

namespace bridge = openral_octomap_bridge;

namespace {

// The deploy-sim octomap_server parameters, in one place.
constexpr double kResolution = 0.025;
constexpr double kOccupancyThres = 0.8;
constexpr double kClampingMax = 0.85;
constexpr double kProbHit = 0.7;
constexpr double kProbMiss = 0.4;
constexpr double kMaxRange = 4.0;

octomap::OcTree deploy_sim_tree() {
  octomap::OcTree tree(kResolution);
  tree.setOccupancyThres(kOccupancyThres);
  tree.setClampingThresMax(kClampingMax);
  tree.setProbHit(kProbHit);
  tree.setProbMiss(kProbMiss);
  return tree;
}

// The cell the arm stands in front of: 0.30 m down the +x axis from the sensor.
const octomap::point3d kSensor(0.0F, 0.0F, 0.0F);
const octomap::point3d kCell(0.30F, 0.0F, 0.0F);
// A clearing ray along the same bearing, ending past the cell at max range.
const octomap::point3d kClearingEndpoint(static_cast<float>(kMaxRange), 0.0F, 0.0F);

void insert_return(octomap::OcTree& tree, const octomap::point3d& endpoint) {
  octomap::Pointcloud cloud;
  cloud.push_back(endpoint);
  tree.insertPointCloud(cloud, kSensor, kMaxRange);
}

bool occupied(const octomap::OcTree& tree, const octomap::point3d& p) {
  const octomap::OcTreeNode* node = tree.search(p);
  return node != nullptr && tree.isNodeOccupied(node);
}

}  // namespace

TEST(OccupancyPersistence, OneHitIsNotYetASafetyVoxel) {
  // The whole point of occupancy_thres 0.8: a single frame's return — a
  // transient, a speckle, a ray that grazed something — must not stop the arm.
  octomap::OcTree tree = deploy_sim_tree();
  insert_return(tree, kCell);
  EXPECT_FALSE(occupied(tree, kCell));
}

TEST(OccupancyPersistence, TwoHitsConfirmASafetyVoxel) {
  octomap::OcTree tree = deploy_sim_tree();
  insert_return(tree, kCell);
  insert_return(tree, kCell);
  EXPECT_TRUE(occupied(tree, kCell));
}

TEST(OccupancyPersistence, AConfirmedVoxelSurvivesWhenNoRayEverCrossesIt) {
  // Pre-fix bridge: no ray inserted for pixels covering the robot's own body.
  // Ten frames of "the sensor said nothing" leave the phantom in place.
  octomap::OcTree tree = deploy_sim_tree();
  insert_return(tree, kCell);
  insert_return(tree, kCell);
  ASSERT_TRUE(occupied(tree, kCell));

  for (int frame = 0; frame < 10; ++frame) {
    const octomap::Pointcloud empty;
    tree.insertPointCloud(empty, kSensor, kMaxRange);
  }
  EXPECT_TRUE(occupied(tree, kCell)) << "fixture assumption changed";

  const auto grid = bridge::rasterize_octree_to_grid(
      tree, tf2::Transform::getIdentity(),
      [] {
        // Covers the stretch of the ray's bearing this test cares about. The
        // covered volume is a ball (grid axes are the map's); only one
        // occupied cell exists anywhere near it, so the count is unaffected.
        bridge::GridSpec s;
        s.center[0] = 0.45;
        s.center[1] = 0.0;
        s.center[2] = 0.0;
        s.radius = 0.20;
        return s;
      }(),
      "base_link");
  int occupied_cells = 0;
  for (const auto v : grid.occupancy) {
    occupied_cells += v;
  }
  EXPECT_EQ(occupied_cells, 1) << "the phantom reaches the kernel's grid";
}

TEST(OccupancyPersistence, OneRestoredClearingRayRetiresAConfirmedVoxel) {
  // Self-filter's clearing ray (endpoint at max range, same bearing) restored:
  // a single frame retires it — what `sensor_model.max 0.85` was tuned for.
  octomap::OcTree tree = deploy_sim_tree();
  insert_return(tree, kCell);
  insert_return(tree, kCell);
  ASSERT_TRUE(occupied(tree, kCell));

  insert_return(tree, kClearingEndpoint);
  EXPECT_FALSE(occupied(tree, kCell));
}

TEST(OccupancyPersistence, ClearingRaysDoNotBecomeAWayToMark) {
  // Clearing must not smuggle occupancy in the other direction: the endpoint
  // of a clearing ray is a single hit at max range and stays below the
  // threshold on its own, exactly like any other one-frame return.
  octomap::OcTree tree = deploy_sim_tree();
  insert_return(tree, kClearingEndpoint);
  EXPECT_FALSE(occupied(tree, kClearingEndpoint));
}

TEST(OccupancyPersistence, AClearedCellStillComesBackWhenSomethingRealIsThere) {
  // Clearing is not a veto. A cell driven to the clamping floor by clearing
  // rays is re-confirmed by repeated genuine returns — it just costs the
  // log-odds hysteresis the tuning bought (four 10 Hz frames, ~0.4 s, from
  // the floor at clamping_min 0.12 to occupancy_thres 0.8 at prob_hit 0.7).
  // Stated here so the cost of the fix is a number, not an assumption.
  octomap::OcTree tree = deploy_sim_tree();
  for (int frame = 0; frame < 4; ++frame) {
    insert_return(tree, kClearingEndpoint);
  }
  ASSERT_FALSE(occupied(tree, kCell));

  for (int frame = 0; frame < 3; ++frame) {
    insert_return(tree, kCell);
    EXPECT_FALSE(occupied(tree, kCell)) << "re-confirmed too early at frame " << frame;
  }
  insert_return(tree, kCell);
  EXPECT_TRUE(occupied(tree, kCell));
}
