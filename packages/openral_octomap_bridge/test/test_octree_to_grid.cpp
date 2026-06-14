// SPDX-License-Identifier: Apache-2.0
// ADR-0030 — unit coverage for the OctoMap → OccupancyVoxels rasterization
// core. Builds a real octree (no ROS graph / TF), queries it, and checks the
// grid.

#include <gtest/gtest.h>
#include <octomap/OcTree.h>
#include <tf2/LinearMath/Transform.h>
#include <tf2/LinearMath/Vector3.h>

#include "openral_octomap_bridge/octree_to_grid.hpp"

namespace bridge = openral_octomap_bridge;

namespace {

bridge::GridSpec two_cubed(double res) {
  bridge::GridSpec s;
  s.resolution = res;
  s.sx = 2;
  s.sy = 2;
  s.sz = 2;
  s.box_min[0] = 0.0;
  s.box_min[1] = 0.0;
  s.box_min[2] = 0.0;
  return s;
}

int occupied_count(const openral_msgs::msg::OccupancyVoxels& g) {
  int n = 0;
  for (const auto v : g.occupancy) {
    n += v;
  }
  return n;
}

}  // namespace

TEST(OctreeToGrid, OccupiedNodeMapsToTheCoveringVoxel) {
  octomap::OcTree tree(0.1);
  tree.updateNode(octomap::point3d(0.05F, 0.05F, 0.05F),
                  true);  // occupy one octree voxel

  const auto grid = bridge::rasterize_octree_to_grid(tree, tf2::Transform::getIdentity(),
                                                     two_cubed(0.1), "base_link");

  EXPECT_EQ(grid.size_x, 2U);
  EXPECT_EQ(grid.size_y, 2U);
  EXPECT_EQ(grid.size_z, 2U);
  EXPECT_EQ(grid.occupancy.size(), 8U);
  EXPECT_EQ(grid.header.frame_id, "base_link");
  EXPECT_EQ(grid.resolution, 0.1);
  // Grid voxel (0,0,0)'s centre (0.05,0.05,0.05) falls in the occupied octree
  // cell; nothing else does.
  EXPECT_EQ(grid.occupancy[0], 1);
  EXPECT_EQ(occupied_count(grid), 1);
}

TEST(OctreeToGrid, EmptyTreeGivesAllFree) {
  const octomap::OcTree tree(0.1);
  const auto grid = bridge::rasterize_octree_to_grid(tree, tf2::Transform::getIdentity(),
                                                     two_cubed(0.1), "base_link");
  EXPECT_EQ(occupied_count(grid), 0);
}

TEST(OctreeToGrid, TransformShiftsTheQuery) {
  // Occupy a cell at octree (1.05, 0.05, 0.05). With a base→octree transform
  // that translates +1 m in x, the base-frame voxel (0,0,0) (centre 0.05,…)
  // maps onto the occupied octree cell.
  octomap::OcTree tree(0.1);
  tree.updateNode(octomap::point3d(1.05F, 0.05F, 0.05F), true);

  tf2::Transform base_to_octomap;
  base_to_octomap.setIdentity();
  base_to_octomap.setOrigin(tf2::Vector3(1.0, 0.0, 0.0));

  const auto grid =
      bridge::rasterize_octree_to_grid(tree, base_to_octomap, two_cubed(0.1), "base_link");
  EXPECT_EQ(grid.occupancy[0], 1);
  EXPECT_EQ(occupied_count(grid), 1);
}
