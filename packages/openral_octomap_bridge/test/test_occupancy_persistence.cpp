// SPDX-License-Identifier: Apache-2.0
// Why a self-filtered depth ray must still reach OctoMap.
//
// The deploy-sim launch (`packages/openral_rskill_ros/launch/sim_e2e.launch.py`)
// tunes octomap_server for exact simulated depth: `occupancy_thres 0.8` so one
// hit can never make a safety voxel (a transient needs a second frame to
// confirm it), and `sensor_model.max 0.85` so ONE clearing ray can retire a
// confirmed hit. Both halves of that bargain assume the ray exists.
//
// It did not. The bridge builds octomap's cloud from the depth RASTER, whose
// `0.0` means "no measurement" — and the self-filter used to collapse "the only
// thing this ray hit was the robot's own arm" onto that same `0.0`. So every
// pixel covering the robot contributed NO ray, the arm's silhouette became a
// write-only region of the map, and a cell marked inside it survived every
// later frame: exactly the shape of the 2026-08-14 `panda_link1` stop against
// `voxel_76001`, which no physical geometry backed.
//
// These tests pin both properties on a real `octomap::OcTree` with the deploy
// numbers: the transient rejection the threshold exists for, and the clearing
// that the restored ray makes possible.

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
  // The pre-fix bridge: the robot's own body fills these pixels, the raster
  // reads 0.0, no ray is inserted at all. Ten frames of "the sensor said
  // nothing" leave the phantom exactly where it was — this is the leak, stated
  // as a property rather than a story.
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
        // Covers the stretch of the ray's bearing this test cares about. It
        // used to be a 16x1x1 strip; the covered volume is a ball now (the
        // grid's axes are the map's, so only a ball is invariant to them), and
        // the count below is unaffected because this scene has exactly one
        // occupied cell anywhere near it.
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
  // …and with the self-filter's clearing ray back on the cloud (an endpoint at
  // max range along the same bearing), a single frame retires it. That is what
  // `sensor_model.max 0.85` was tuned for.
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
