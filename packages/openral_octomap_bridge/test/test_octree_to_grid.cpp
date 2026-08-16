// SPDX-License-Identifier: Apache-2.0
// Unit coverage for the OctoMap → OccupancyVoxels rasterization
// core. Builds a real octree (no ROS graph / TF), queries it, and checks the
// grid.
//
// The second half of this file is about the LATTICE PHASE. The octree's lattice
// is fixed in the map/odom frame and the grid's in `base_link`, so their
// relative phase is whatever the robot's pose makes it. Sampling each base
// cell's centre — the rule until 2026-08-16 — snaps every surface onto the
// lattice the centre happened to land on, and on the `robocasa_drawer_utensil`
// field run (25 mm cells, ~12.1 mm phase, half a cell) that put a cabinet door
// panel whose true front face is at base x = +0.0614 into the column
// x ∈ [0.025, 0.050) — a full voxel closer to the robot — with ZERO cells where
// the panel actually is. The best rigid shift between the recorded map and a
// ground-truth rasterization of the same camera at the same pose was exactly
// (−1, 0, 0) cells; z was benign only by luck of its own phase.
//
// The rule is now overlap: a base cell is occupied when its cube shares volume
// with an occupied leaf's cube. These tests pin that it never drops a cell the
// centre sample would have marked (at any phase), that it puts the panel back
// in its own column, and that it does not dilate a phase-aligned grid.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <gtest/gtest.h>
#include <octomap/OcTree.h>
#include <octomap/Pointcloud.h>
#include <tf2/LinearMath/Quaternion.h>
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

// ── The lattice phase ────────────────────────────────────────────────────────

namespace {

// The deploy-sim octomap_server parameters (sim_e2e.launch.py), as the other
// two suites in this package use them.
constexpr double kRes = 0.025;
constexpr double kOccupancyThres = 0.8;
constexpr double kClampingMax = 0.85;
constexpr double kProbHit = 0.7;
constexpr double kProbMiss = 0.4;
constexpr double kMaxRange = 6.0;

octomap::OcTree deploy_sim_tree() {
  octomap::OcTree tree(kRes);
  tree.setOccupancyThres(kOccupancyThres);
  tree.setClampingThresMax(kClampingMax);
  tree.setProbHit(kProbHit);
  tree.setProbMiss(kProbMiss);
  return tree;
}

// One frame of real returns; two frames confirm a cell at occupancy_thres 0.8.
void insert_frame(octomap::OcTree& tree, const octomap::point3d& sensor,
                  const std::vector<octomap::point3d>& endpoints) {
  octomap::Pointcloud cloud;
  for (const auto& p : endpoints) {
    cloud.push_back(p);
  }
  tree.insertPointCloud(cloud, sensor, kMaxRange);
}

void insert_confirmed(octomap::OcTree& tree, const octomap::point3d& sensor,
                      const std::vector<octomap::point3d>& endpoints) {
  insert_frame(tree, sensor, endpoints);
  insert_frame(tree, sensor, endpoints);
}

std::size_t index_of(const openral_msgs::msg::OccupancyVoxels& grid, std::uint32_t ix,
                     std::uint32_t iy, std::uint32_t iz) {
  return static_cast<std::size_t>(ix) +
         static_cast<std::size_t>(grid.size_x) *
             (static_cast<std::size_t>(iy) + static_cast<std::size_t>(grid.size_y) * iz);
}

// The rule this function used until 2026-08-16, transcribed: a cell is occupied
// iff its CENTRE, carried into the octree frame, point-queries an occupied
// node. Every conservativeness claim below is measured against this.
openral_msgs::msg::OccupancyVoxels centre_sampled(const octomap::OcTree& tree,
                                                  const tf2::Transform& base_to_octomap,
                                                  const bridge::GridSpec& spec) {
  openral_msgs::msg::OccupancyVoxels grid;
  grid.origin.x = spec.box_min[0];
  grid.origin.y = spec.box_min[1];
  grid.origin.z = spec.box_min[2];
  grid.resolution = spec.resolution;
  grid.size_x = spec.sx;
  grid.size_y = spec.sy;
  grid.size_z = spec.sz;
  grid.occupancy.assign(
      static_cast<std::size_t>(spec.sx) * static_cast<std::size_t>(spec.sy) * spec.sz, 0);
  for (std::uint32_t iz = 0; iz < spec.sz; ++iz) {
    for (std::uint32_t iy = 0; iy < spec.sy; ++iy) {
      for (std::uint32_t ix = 0; ix < spec.sx; ++ix) {
        const tf2::Vector3 p =
            base_to_octomap * tf2::Vector3(spec.box_min[0] + (ix + 0.5) * spec.resolution,
                                           spec.box_min[1] + (iy + 0.5) * spec.resolution,
                                           spec.box_min[2] + (iz + 0.5) * spec.resolution);
        const octomap::OcTreeNode* node = tree.search(p.x(), p.y(), p.z());
        if (node != nullptr && tree.isNodeOccupied(node)) {
          grid.occupancy[index_of(grid, ix, iy, iz)] = 1;
        }
      }
    }
  }
  return grid;
}

// ── The field fixture: `robocasa_drawer_utensil`'s cabinet door panel ────────

// The panel's true front face, in the base frame, and the relative phase
// measured on the run: the octree's cell boundaries land 12.1 mm in +x of the
// base grid's, i.e. just under half of a 25 mm cell.
constexpr double kPanelFaceX = 0.0614;
constexpr double kFieldPhase = 0.0121;

// The camera that saw it, at base height, looking down +x.
const octomap::point3d kPanelSensor(-0.30F, 0.0F, 0.35F);

// A base→octomap transform whose x translation puts the octree's boundaries
// `phase` in front of the base grid's (the grid's own box_min is a multiple of
// the resolution, so its boundaries sit on the base lattice). y and z are left
// aligned: the field defect was in x, and z was benign only by luck of phase.
tf2::Transform phased(double phase) {
  tf2::Transform xf;
  xf.setIdentity();
  xf.setOrigin(tf2::Vector3(kRes - phase, 0.0, 0.0));
  return xf;
}

// The grid the panel is rasterized into: 6 columns of 25 mm out from the base,
// 4x4 across the panel, its min corner on the base lattice.
bridge::GridSpec panel_spec() {
  bridge::GridSpec spec;
  spec.resolution = kRes;
  spec.sx = 6;
  spec.sy = 4;
  spec.sz = 4;
  spec.box_min[0] = 0.0;
  spec.box_min[1] = -0.05;
  spec.box_min[2] = 0.30;
  return spec;
}

// The returns off the panel's front face, one per (y, z) grid cell, at the
// exact face depth. Stated in the BASE frame and carried into the octree frame
// by the caller's transform, because the face is a fact about the world and the
// phase is a fact about where the octree's lattice happens to lie.
octomap::OcTree panel_tree(const tf2::Transform& base_to_octomap) {
  octomap::OcTree tree = deploy_sim_tree();
  std::vector<octomap::point3d> returns;
  for (const double y : {-0.0375, -0.0125, 0.0125, 0.0375}) {
    for (const double z : {0.3125, 0.3375, 0.3625, 0.3875}) {
      const tf2::Vector3 p = base_to_octomap * tf2::Vector3(kPanelFaceX, y, z);
      returns.emplace_back(static_cast<float>(p.x()), static_cast<float>(p.y()),
                           static_cast<float>(p.z()));
    }
  }
  const tf2::Vector3 s =
      base_to_octomap * tf2::Vector3(kPanelSensor.x(), kPanelSensor.y(), kPanelSensor.z());
  const octomap::point3d sensor(static_cast<float>(s.x()), static_cast<float>(s.y()),
                                static_cast<float>(s.z()));
  insert_confirmed(tree, sensor, returns);
  return tree;
}

// The x columns marked anywhere in the grid, in ascending order.
std::vector<std::uint32_t> marked_columns(const openral_msgs::msg::OccupancyVoxels& grid) {
  std::vector<std::uint32_t> columns;
  for (std::uint32_t ix = 0; ix < grid.size_x; ++ix) {
    bool marked = false;
    for (std::uint32_t iz = 0; iz < grid.size_z && !marked; ++iz) {
      for (std::uint32_t iy = 0; iy < grid.size_y && !marked; ++iy) {
        marked = grid.occupancy[index_of(grid, ix, iy, iz)] != 0;
      }
    }
    if (marked) {
      columns.push_back(ix);
    }
  }
  return columns;
}

// The column whose cube contains base-frame x.
std::uint32_t column_of(const openral_msgs::msg::OccupancyVoxels& grid, double x) {
  return static_cast<std::uint32_t>(std::floor((x - grid.origin.x) / grid.resolution));
}

double column_center(const openral_msgs::msg::OccupancyVoxels& grid, std::uint32_t ix) {
  return grid.origin.x + (static_cast<double>(ix) + 0.5) * grid.resolution;
}

// ── A kitchen the whole grid has to be rasterized out of ─────────────────────

// A back wall, a counter top and a side wall, seen from a depth camera on the
// robot and inserted through real rays, so the octree carries real free space
// and octomap prunes it where it can.
octomap::OcTree kitchen_scene() {
  octomap::OcTree tree = deploy_sim_tree();
  const octomap::point3d sensor(0.0F, 0.0F, 0.6F);
  std::vector<octomap::point3d> returns;
  for (double y = -1.2; y <= 1.2; y += kRes) {
    for (double z = 0.0; z <= 1.6; z += kRes) {
      returns.emplace_back(1.20F, static_cast<float>(y), static_cast<float>(z));
    }
  }
  for (double x = 0.4; x <= 1.2; x += kRes) {
    for (double y = -1.2; y <= 1.2; y += kRes) {
      returns.emplace_back(static_cast<float>(x), static_cast<float>(y), 0.90F);
    }
  }
  for (double x = 0.2; x <= 1.2; x += kRes) {
    for (double z = 0.0; z <= 1.6; z += kRes) {
      returns.emplace_back(static_cast<float>(x), 1.10F, static_cast<float>(z));
    }
  }
  insert_confirmed(tree, sensor, returns);
  tree.updateInnerOccupancy();
  return tree;
}

// The deploy-sim grid: a 2 m box around the robot at the bridge's default
// 50 mm, i.e. 64000 cells (the kernel's world_voxel_max_cells is 262144).
bridge::GridSpec kitchen_spec(double resolution) {
  bridge::GridSpec spec;
  spec.resolution = resolution;
  spec.sx = static_cast<std::uint32_t>(std::lround(2.0 / resolution));
  spec.sy = spec.sx;
  spec.sz = spec.sx;
  spec.box_min[0] = -1.0;
  spec.box_min[1] = -1.0;
  spec.box_min[2] = -0.5;
  return spec;
}

// Is every cell of `subset` also in `superset`? Returns the first counter-
// example index, or -1.
long first_dropped(const openral_msgs::msg::OccupancyVoxels& subset,
                   const openral_msgs::msg::OccupancyVoxels& superset) {
  for (std::size_t i = 0; i < subset.occupancy.size(); ++i) {
    if (subset.occupancy[i] != 0 && superset.occupancy[i] == 0) {
      return static_cast<long>(i);
    }
  }
  return -1;
}

}  // namespace

TEST(OctreeToGrid, TheFieldPanelKeepsTheColumnItActuallyOccupies) {
  // `robocasa_drawer_utensil`, reproduced: the panel's front face at base
  // x = +0.0614 with the run's 12.1 mm lattice phase. The occupied octree cell
  // holding the face spans base x ∈ [0.0371, 0.0621), so the base cell centre
  // at 0.0625 misses it by 0.4 mm and the one at 0.0375 catches it — which is
  // the whole defect: the panel is reported a voxel closer than it is, and
  // nothing at all is reported where it is.
  const tf2::Transform base_to_octomap = phased(kFieldPhase);
  const octomap::OcTree tree = panel_tree(base_to_octomap);
  const auto spec = panel_spec();

  const auto before = centre_sampled(tree, base_to_octomap, spec);
  const auto grid = bridge::rasterize_octree_to_grid(tree, base_to_octomap, spec, "base_link");
  const std::uint32_t face_column = column_of(grid, kPanelFaceX);
  ASSERT_EQ(face_column, 2U) << "fixture assumption: the face is in [0.050, 0.075)";

  // The defect itself, stated on the old rule so it cannot quietly come back.
  ASSERT_EQ(marked_columns(before), (std::vector<std::uint32_t>{1U}))
      << "the centre sample marked one column, a voxel in front of the panel";

  // The fix: the face's own column is marked, for every (y, z) the camera saw.
  for (std::uint32_t iz = 0; iz < spec.sz; ++iz) {
    for (std::uint32_t iy = 0; iy < spec.sy; ++iy) {
      EXPECT_EQ(grid.occupancy[index_of(grid, face_column, iy, iz)], 1)
          << "no cell where the panel is at (" << iy << ", " << iz << ")";
    }
  }
  // …and it is the OUTERMOST marked column: overlap adds the column the surface
  // is in, it does not smear the panel deeper into the cabinet.
  EXPECT_EQ(marked_columns(grid), (std::vector<std::uint32_t>{1U, 2U}));
  // Conservative: the column the old rule marked is still marked.
  EXPECT_EQ(first_dropped(before, grid), -1);
}

TEST(OctreeToGrid, ASlabIsNeverMissedAndNeverFurtherForwardThanTheLatticesExplain) {
  // The same panel swept through every phase of a 25 mm lattice, in 1 mm steps.
  // Two properties, both of them about the safety envelope:
  //
  //  * the column whose cube CONTAINS the true face is always marked — the
  //    property the field run lost;
  //  * no marked cell's centre is further forward (toward the robot) than
  //    `tree resolution + half a grid cell` from the true face. One tree
  //    resolution is octomap's own: `insertPointCloud` marks the cell
  //    CONTAINING the endpoint, which already reaches up to one resolution in
  //    front of the surface. The half cell is this function's, and is the whole
  //    price of the fix — the centre rule's bound was one tree resolution.
  const double kForwardBound = kRes + 0.5 * kRes;
  for (int step = 0; step < 25; ++step) {
    const double phase = 0.001 * step;
    const tf2::Transform base_to_octomap = phased(phase);
    const octomap::OcTree tree = panel_tree(base_to_octomap);
    const auto spec = panel_spec();
    const auto grid = bridge::rasterize_octree_to_grid(tree, base_to_octomap, spec, "base_link");
    const std::uint32_t face_column = column_of(grid, kPanelFaceX);

    const auto columns = marked_columns(grid);
    ASSERT_FALSE(columns.empty()) << "phase " << phase << ": the panel vanished";
    EXPECT_NE(std::find(columns.begin(), columns.end(), face_column), columns.end())
        << "phase " << phase << ": nothing marked in the panel's own column";
    EXPECT_GE(column_center(grid, columns.front()), kPanelFaceX - kForwardBound - 1e-6)
        << "phase " << phase << ": marked " << kPanelFaceX - column_center(grid, columns.front())
        << " m in front of the panel";
  }
}

TEST(OctreeToGrid, EveryPhaseAndYawKeepsEveryCellTheCentreSampleWouldHaveMarked) {
  // The conservativeness proof, swept: for every relative phase of the two
  // lattices — and for three relative yaws, since a mobile base supplies those
  // too — the marked set is a SUPERSET of what centre sampling marks. This
  // change can only ever add occupancy to the kernel's grid.
  const octomap::OcTree tree = kitchen_scene();
  const auto spec = kitchen_spec(kRes);
  std::size_t phases_with_more_cells = 0;
  for (const double yaw : {0.0, 0.37, 0.785398}) {
    for (int step = 0; step < 5; ++step) {
      const double phase = 0.005 * step;
      tf2::Transform base_to_octomap;
      tf2::Quaternion q;
      q.setRPY(0.0, 0.0, yaw);
      base_to_octomap.setRotation(q);
      base_to_octomap.setOrigin(tf2::Vector3(phase, 0.6 * phase, 0.3 * phase));

      const auto before = centre_sampled(tree, base_to_octomap, spec);
      const auto after = bridge::rasterize_octree_to_grid(tree, base_to_octomap, spec, "base_link");
      EXPECT_EQ(first_dropped(before, after), -1)
          << "yaw " << yaw << ", phase " << phase << ": a cell the centre sample marked was lost";
      if (occupied_count(after) > occupied_count(before)) {
        ++phases_with_more_cells;
      }
    }
  }
  // …and the superset is a strict one, or the sweep above would prove nothing.
  EXPECT_GT(phases_with_more_cells, 0U);
}

TEST(OctreeToGrid, APhaseAlignedGridRasterizesExactlyAsCentreSamplingDid) {
  // The other half of "conservative": overlap must not DILATE. With the two
  // lattices in phase and at the same resolution, each base cube is one octree
  // cube, the neighbours only touch it, and the grid is bit-for-bit the one the
  // centre sample produced — no 27-cell blob around every leaf.
  const octomap::OcTree tree = kitchen_scene();
  const auto spec = kitchen_spec(kRes);  // box_min is a multiple of the resolution
  const auto before = centre_sampled(tree, tf2::Transform::getIdentity(), spec);
  const auto after =
      bridge::rasterize_octree_to_grid(tree, tf2::Transform::getIdentity(), spec, "base_link");
  ASSERT_GT(occupied_count(before), 0);
  EXPECT_EQ(after.occupancy, before.occupancy);
}

TEST(OctreeToGrid, ACoarseLeafMarksEveryCellUnderIt) {
  // octomap prunes eight siblings that carry the same value into ONE leaf of
  // twice the edge length, and a published /octomap_binary is pruned. So the
  // overlap is against the leaf's own size, not the tree resolution: get that
  // wrong and 7/8 of a pruned obstacle leaves the kernel's grid.
  //
  // Seeded with updateNode rather than rays because pruning needs the eight
  // cells to hold exactly equal values, and every ray that reaches one of them
  // crosses (and so updates) its neighbours.
  octomap::OcTree tree(0.05);
  for (const double x : {0.025, 0.075}) {
    for (const double y : {0.025, 0.075}) {
      for (const double z : {0.025, 0.075}) {
        tree.updateNode(
            octomap::point3d(static_cast<float>(x), static_cast<float>(y), static_cast<float>(z)),
            true);
      }
    }
  }
  tree.prune();
  double coarse_leaf_size = 0.0;
  for (auto it = tree.begin_leafs(), end = tree.end_leafs(); it != end; ++it) {
    if (tree.isNodeOccupied(*it)) {
      coarse_leaf_size = std::max(coarse_leaf_size, it.getSize());
    }
  }
  ASSERT_DOUBLE_EQ(coarse_leaf_size, 0.1)
      << "fixture assumption: octomap pruned the eight 50 mm cells into one 100 mm leaf";

  // A 25 mm grid at a half-cell phase under the 100 mm leaf, which spans
  // [0, 0.1) on every axis: cells 0..4 span [-0.0125, 0.1125) and each enters
  // it; cell 5 starts at 0.1125 and misses. Against the tree's resolution
  // instead of the leaf's own size, cells 2..4 would be lost.
  bridge::GridSpec spec;
  spec.resolution = 0.025;
  spec.sx = 6;
  spec.sy = 6;
  spec.sz = 6;
  spec.box_min[0] = -0.0125;
  spec.box_min[1] = -0.0125;
  spec.box_min[2] = -0.0125;
  const auto grid =
      bridge::rasterize_octree_to_grid(tree, tf2::Transform::getIdentity(), spec, "base_link");
  EXPECT_EQ(occupied_count(grid), 125);
  for (std::uint32_t iz = 0; iz < 5; ++iz) {
    for (std::uint32_t iy = 0; iy < 5; ++iy) {
      for (std::uint32_t ix = 0; ix < 5; ++ix) {
        EXPECT_EQ(grid.occupancy[index_of(grid, ix, iy, iz)], 1)
            << "coarse leaf missed (" << ix << ", " << iy << ", " << iz << ")";
      }
    }
  }
}

TEST(OctreeToGrid, AnObliqueLeafMarksTheCellsItEntersAndNotTheirDiagonals) {
  // Exactness under rotation. A leaf yawed 45° about a shared cell centre pokes
  // its corners 20.7% of a cell into the four edge-neighbours but reaches only
  // 1.207 cells toward a diagonal neighbour, whose own support is 1.414 — so
  // the cross is marked and the corners are not. Bounding the rotated leaf by
  // its axis-aligned box instead (the cheap way to do this) would mark all
  // nine, which is a 41% inflation of every obstacle on a yawed base.
  octomap::OcTree tree(0.1);
  const octomap::point3d leaf_center(0.05F, 0.05F, 0.05F);  // spans [0, 0.1)^3
  tree.updateNode(leaf_center, true);

  tf2::Transform base_to_octomap;
  tf2::Quaternion q;
  q.setRPY(0.0, 0.0, 0.785398163397448);
  base_to_octomap.setRotation(q);
  base_to_octomap.setOrigin(tf2::Vector3(0.0, 0.0, 0.0));

  // A 3x3x1 grid of the same cell size, its middle cell centred on the leaf's
  // own centre and its single z layer exactly the leaf's (yaw leaves z alone).
  const tf2::Vector3 center =
      base_to_octomap.inverse() * tf2::Vector3(leaf_center.x(), leaf_center.y(), leaf_center.z());
  bridge::GridSpec spec;
  spec.resolution = 0.1;
  spec.sx = 3;
  spec.sy = 3;
  spec.sz = 1;
  spec.box_min[0] = center.x() - 1.5 * spec.resolution;
  spec.box_min[1] = center.y() - 1.5 * spec.resolution;
  spec.box_min[2] = center.z() - 0.5 * spec.resolution;
  const auto grid = bridge::rasterize_octree_to_grid(tree, base_to_octomap, spec, "base_link");
  EXPECT_EQ(occupied_count(grid), 5);
  EXPECT_EQ(grid.occupancy[index_of(grid, 1, 1, 0)], 1);
  EXPECT_EQ(grid.occupancy[index_of(grid, 0, 1, 0)], 1);
  EXPECT_EQ(grid.occupancy[index_of(grid, 2, 1, 0)], 1);
  EXPECT_EQ(grid.occupancy[index_of(grid, 1, 0, 0)], 1);
  EXPECT_EQ(grid.occupancy[index_of(grid, 1, 2, 0)], 1);
  EXPECT_EQ(grid.occupancy[index_of(grid, 0, 0, 0)], 0);
  EXPECT_EQ(grid.occupancy[index_of(grid, 2, 2, 0)], 0);
}

TEST(OctreeToGrid, AGridOutsideTheOctreesKeyRangeStillSeesEveryLeaf) {
  // The leaves are fetched with octomap's bbx iterator, whose key conversion
  // FAILS (silently, into an empty iteration) for coordinates outside the
  // tree's addressable ±32768·resolution. An empty iteration is an empty grid,
  // which is the one direction this node must never fail in, so an unusable
  // bbx falls back to walking the whole tree.
  octomap::OcTree tree(0.1);  // addressable to ±3276.8 m
  tree.updateNode(octomap::point3d(0.05F, 0.05F, 0.05F), true);

  bridge::GridSpec spec;
  spec.resolution = 5000.0;
  spec.sx = 4;
  spec.sy = 4;
  spec.sz = 4;
  spec.box_min[0] = -10000.0;
  spec.box_min[1] = -10000.0;
  spec.box_min[2] = -10000.0;
  const auto grid =
      bridge::rasterize_octree_to_grid(tree, tf2::Transform::getIdentity(), spec, "base_link");
  EXPECT_EQ(occupied_count(grid), 1);
  EXPECT_EQ(grid.occupancy[index_of(grid, 2, 2, 2)], 1);
}

TEST(OctreeToGrid, RasterizingTheKitchenStaysInsideThePublishBudget) {
  // The bridge re-derives the grid from the octree on EVERY published grid
  // (`publish_rate_hz` 10.0 → a 100 ms period), so the rasterization's cost is
  // a contract, not an implementation detail. Iterating the occupied leaves is
  // what keeps it there: the work is now proportional to the surfaces in the
  // box rather than to its cell count, which is why the finer grid below is no
  // more expensive than the coarse one. Measured on the 2026-08-16 reference
  // host: 0.29 ms at 50 mm and 0.29 ms at 25 mm, against 2.3 ms / 15.2 ms for
  // the centre sample it replaced. The ceiling is a quarter of the publish
  // period — two orders of magnitude of headroom, so it catches a return to
  // per-cell work without tracking a host's mood.
  const octomap::OcTree tree = kitchen_scene();
  for (const double resolution : {0.05, 0.025}) {
    const auto spec = kitchen_spec(resolution);
    auto grid = bridge::rasterize_octree_to_grid(tree, tf2::Transform::getIdentity(), spec,
                                                 "base_link");  // warm
    ASSERT_GT(occupied_count(grid), 0);
    const auto started = std::chrono::steady_clock::now();
    constexpr int kIterations = 10;
    for (int i = 0; i < kIterations; ++i) {
      grid =
          bridge::rasterize_octree_to_grid(tree, tf2::Transform::getIdentity(), spec, "base_link");
    }
    const double ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started)
            .count() /
        kIterations;
    EXPECT_LT(ms, 25.0) << "rasterizing a " << spec.sx << "^3 grid took " << ms << " ms";
  }
}
