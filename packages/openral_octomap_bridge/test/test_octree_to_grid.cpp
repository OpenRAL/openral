// SPDX-License-Identifier: Apache-2.0
// Unit coverage for the OctoMap → OccupancyVoxels rasterization core. Builds a
// real octree (no ROS graph / TF), queries it, and checks the grid.
//
// The second half pins EXACTNESS: the published grid carries the octree's
// occupied volume cell for cell at every relative phase/yaw — never less (an
// obstacle lost) and never more (reach given up for nothing).
//
// Rejected prior rules (see README "Why not a base-aligned grid"): centre
// point-query (until 2026-08-16, zero cells on a real panel on the
// `robocasa_drawer_utensil` run) and overlap (sound but costs 29-35mm median /
// 40mm worst-case dilation, held 48% of live start-state E-stops, issue #173).
// Current rule: grid lattice = octree lattice, rotation on the wire in
// `OccupancyVoxels.orientation` — no phase/yaw left to be exact about.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
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

bridge::GridSpec ball(double cx, double cy, double cz, double radius) {
  bridge::GridSpec s;
  s.center[0] = cx;
  s.center[1] = cy;
  s.center[2] = cz;
  s.radius = radius;
  return s;
}

int occupied_count(const openral_msgs::msg::OccupancyVoxels& g) {
  int n = 0;
  for (const auto v : g.occupancy) {
    n += v;
  }
  return n;
}

std::size_t index_of(const openral_msgs::msg::OccupancyVoxels& g, std::uint32_t ix,
                     std::uint32_t iy, std::uint32_t iz) {
  return static_cast<std::size_t>(ix) +
         static_cast<std::size_t>(g.size_x) * (iy + static_cast<std::size_t>(g.size_y) * iz);
}

/// The grid's pose in the base frame, from the wire fields.
tf2::Transform grid_pose(const openral_msgs::msg::OccupancyVoxels& g) {
  return tf2::Transform(
      tf2::Quaternion(g.orientation.x, g.orientation.y, g.orientation.z, g.orientation.w),
      tf2::Vector3(g.origin.x, g.origin.y, g.origin.z));
}

/// Base-frame centre of cell (ix, iy, iz) — built along the GRID's axes, then
/// carried into the base frame. The grid is not base-aligned.
tf2::Vector3 cell_center(const openral_msgs::msg::OccupancyVoxels& g, std::uint32_t ix,
                         std::uint32_t iy, std::uint32_t iz) {
  return grid_pose(g) * tf2::Vector3((ix + 0.5) * g.resolution, (iy + 0.5) * g.resolution,
                                     (iz + 0.5) * g.resolution);
}

}  // namespace

TEST(OctreeToGrid, OccupiedNodeMapsToTheCoveringVoxel) {
  octomap::OcTree tree(0.1);
  tree.updateNode(octomap::point3d(0.05F, 0.05F, 0.05F), true);
  const auto grid = bridge::rasterize_octree_to_grid(tree, tf2::Transform::getIdentity(),
                                                     ball(0.05, 0.05, 0.05, 0.05), "base_link");
  EXPECT_EQ(occupied_count(grid), 1);
  EXPECT_DOUBLE_EQ(grid.resolution, 0.1) << "the grid's resolution is the octree's";
  EXPECT_DOUBLE_EQ(grid.orientation.w, 1.0) << "identity transform → identity orientation";
}

TEST(OctreeToGrid, EmptyTreeGivesAllFree) {
  const octomap::OcTree tree(0.1);
  const auto grid = bridge::rasterize_octree_to_grid(tree, tf2::Transform::getIdentity(),
                                                     ball(0.1, 0.1, 0.1, 0.1), "base_link");
  EXPECT_EQ(occupied_count(grid), 0);
  EXPECT_GT(grid.occupancy.size(), 0U) << "an empty map is a grid of frees, not an absent grid";
}

TEST(OctreeToGrid, AnUnplaceableLatticePublishesNoCellsRatherThanWrongOnes) {
  // Radius zero (the node's unset default) names no volume; a guessed volume
  // would be a grid the kernel trusts and the robot is not inside.
  octomap::OcTree tree(0.1);
  tree.updateNode(octomap::point3d(0.05F, 0.05F, 0.05F), true);
  const auto grid = bridge::rasterize_octree_to_grid(tree, tf2::Transform::getIdentity(),
                                                     ball(0.05, 0.05, 0.05, 0.0), "base_link");
  EXPECT_EQ(grid.occupancy.size(), 0U);
  EXPECT_DOUBLE_EQ(grid.orientation.w, 1.0) << "still a valid quaternion, not the all-zero one";
}

TEST(OctreeToGrid, TransformShiftsTheQuery) {
  // Cell occupied at octree (1.05, 0.05, 0.05); a +1 m x translation maps
  // base-frame (0.05, 0.05, 0.05) onto it.
  octomap::OcTree tree(0.1);
  tree.updateNode(octomap::point3d(1.05F, 0.05F, 0.05F), true);

  tf2::Transform base_to_octomap;
  base_to_octomap.setIdentity();
  base_to_octomap.setOrigin(tf2::Vector3(1.0, 0.0, 0.0));

  const auto grid = bridge::rasterize_octree_to_grid(tree, base_to_octomap,
                                                     ball(0.05, 0.05, 0.05, 0.05), "base_link");
  EXPECT_EQ(occupied_count(grid), 1);
}

// ── The lattice ──────────────────────────────────────────────────────────────

namespace {

// The deploy-sim octomap_server parameters (deploy_e2e.launch.py), as the other
// two suites in this package use them.
constexpr double kRes = 0.025;
constexpr double kOccupancyThres = 0.8;
constexpr double kClampingMax = 0.85;
constexpr double kProbHit = 0.7;
constexpr double kProbMiss = 0.4;
constexpr double kMaxRange = 6.0;

octomap::OcTree deploy_sim_tree_at(double resolution) {
  octomap::OcTree tree(resolution);
  tree.setOccupancyThres(kOccupancyThres);
  tree.setClampingThresMax(kClampingMax);
  tree.setProbHit(kProbHit);
  tree.setProbMiss(kProbMiss);
  return tree;
}

octomap::OcTree deploy_sim_tree() { return deploy_sim_tree_at(kRes); }

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

// ── A kitchen the whole grid has to be rasterized out of ─────────────────────

// A back wall, a counter top and a side wall, seen from a depth camera on the
// robot and inserted through real rays, so the octree carries real free space
// and octomap prunes it where it can.
octomap::OcTree kitchen_scene_at(double resolution) {
  octomap::OcTree tree = deploy_sim_tree_at(resolution);
  const octomap::point3d sensor(0.0F, 0.0F, 0.6F);
  std::vector<octomap::point3d> returns;
  for (double y = -1.2; y <= 1.2; y += resolution) {
    for (double z = 0.0; z <= 1.6; z += resolution) {
      returns.emplace_back(1.20F, static_cast<float>(y), static_cast<float>(z));
    }
  }
  for (double x = 0.4; x <= 1.2; x += resolution) {
    for (double y = -1.2; y <= 1.2; y += resolution) {
      returns.emplace_back(static_cast<float>(x), static_cast<float>(y), 0.90F);
    }
  }
  for (double x = 0.2; x <= 1.2; x += resolution) {
    for (double z = 0.0; z <= 1.6; z += resolution) {
      returns.emplace_back(static_cast<float>(x), 1.10F, static_cast<float>(z));
    }
  }
  insert_confirmed(tree, sensor, returns);
  tree.updateInnerOccupancy();
  return tree;
}

octomap::OcTree kitchen_scene() { return kitchen_scene_at(kRes); }

// The deploy-sim coverage: the ball the panda_mobile arm's checked geometry
// lives in, at the bridge's sim resolution.
bridge::GridSpec kitchen_spec() { return ball(0.0, 0.0, 0.5, 1.0); }

/// How the published grid differs from the octree it came from, over the cells
/// the grid covers. Both counts must be zero.
struct LatticeDiff {
  long missing{0};  ///< occupied in the octree, free in the grid — an obstacle lost
  long extra{0};    ///< occupied in the grid, free in the octree — reach given up
};

/// Compare cell for cell against the tree, by asking the tree what is at each
/// published cell's own centre. That is the exactness claim stated directly:
/// a published cell is occupied if and only if the octree cell it names is.
LatticeDiff lattice_diff(const octomap::OcTree& tree, const tf2::Transform& base_to_octomap,
                         const openral_msgs::msg::OccupancyVoxels& grid) {
  LatticeDiff diff;
  for (std::uint32_t iz = 0; iz < grid.size_z; ++iz) {
    for (std::uint32_t iy = 0; iy < grid.size_y; ++iy) {
      for (std::uint32_t ix = 0; ix < grid.size_x; ++ix) {
        const tf2::Vector3 c = base_to_octomap * cell_center(grid, ix, iy, iz);
        const octomap::OcTreeNode* node = tree.search(octomap::point3d(
            static_cast<float>(c.x()), static_cast<float>(c.y()), static_cast<float>(c.z())));
        const bool tree_occupied = node != nullptr && tree.isNodeOccupied(node);
        const bool grid_occupied = grid.occupancy[index_of(grid, ix, iy, iz)] != 0;
        diff.missing += (tree_occupied && !grid_occupied) ? 1 : 0;
        diff.extra += (!tree_occupied && grid_occupied) ? 1 : 0;
      }
    }
  }
  return diff;
}

// ── The field panel (`robocasa_drawer_utensil`) ──────────────────────────────

constexpr double kPanelFaceX = 0.0614;
const tf2::Vector3 kPanelSensor(0.0, 0.0, 0.35);

// A relative lattice phase in x. y and z are left aligned: the field defect was
// in x, and z was benign only by luck of its own phase.
tf2::Transform phased(double phase) {
  tf2::Transform xf;
  xf.setIdentity();
  xf.setOrigin(tf2::Vector3(kRes - phase, 0.0, 0.0));
  return xf;
}

// The returns off the panel's front face, one per (y, z) cell, at the exact
// face depth. Stated in the BASE frame and carried into the octree frame by the
// caller's transform, because the face is a fact about the world and the phase
// is a fact about where the octree's lattice happens to lie.
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

/// How far the frontmost occupied cell's NEAR FACE sits in front of `face_x`,
/// in the base frame. Positive means the grid reports the surface closer to the
/// robot than it is — the quantity every one of these rules is judged on.
double forward_excess(const openral_msgs::msg::OccupancyVoxels& grid, double face_x) {
  double nearest = std::numeric_limits<double>::infinity();
  for (std::uint32_t iz = 0; iz < grid.size_z; ++iz) {
    for (std::uint32_t iy = 0; iy < grid.size_y; ++iy) {
      for (std::uint32_t ix = 0; ix < grid.size_x; ++ix) {
        if (grid.occupancy[index_of(grid, ix, iy, iz)] == 0) {
          continue;
        }
        nearest = std::min(nearest, cell_center(grid, ix, iy, iz).x() - 0.5 * grid.resolution);
      }
    }
  }
  return face_x - nearest;
}

}  // namespace

TEST(OctreeToGrid, TheFieldPanelIsNeverReportedNearerThanTheOctreeItselfSaysAtAnyPhase) {
  // `robocasa_drawer_utensil` regression, swept across the full relative
  // phase. On the octree's own lattice the forward error is the octree's
  // alone (the return-endpoint cell, ≤1 resolution in front of the true
  // face); nothing the bridge adds. That bound is what this asserts.
  for (int step = 0; step < 10; ++step) {
    const double phase = kRes * step / 10.0;
    const tf2::Transform base_to_octomap = phased(phase);
    const octomap::OcTree tree = panel_tree(base_to_octomap);
    const auto grid = bridge::rasterize_octree_to_grid(tree, base_to_octomap,
                                                       ball(0.06, 0.0, 0.35, 0.10), "base_link");
    ASSERT_GT(occupied_count(grid), 0) << "phase " << phase << ": the panel vanished";
    const double excess = forward_excess(grid, kPanelFaceX);
    EXPECT_LE(excess, kRes + 1e-9)
        << "phase " << phase << ": reported the panel " << excess * 1e3
        << " mm in front of its face — more than the octree's own cell explains";
    EXPECT_GE(excess, -1e-9) << "phase " << phase << ": the panel's own cell is missing";
  }
}

TEST(OctreeToGrid, TheGridIsTheOctreeCellForCellAtEveryPhaseAndYaw) {
  // Exactness proof, swept over relative phase and four relative yaws (a
  // mobile base supplies those too). `missing` = obstacle the kernel would
  // not see; `extra` = reach surrendered for nothing. Both zero, every pose.
  // The rule this replaced only promised `missing == 0`; `extra` was its
  // cost, and issue #173 measured that cost holding 48% of the live stops.
  const octomap::OcTree tree = kitchen_scene();
  for (const double yaw : {0.0, 0.37, 0.785398163397448, 1.9}) {
    for (int step = 0; step < 5; ++step) {
      const double phase = 0.005 * step;
      tf2::Transform base_to_octomap;
      tf2::Quaternion q;
      q.setRPY(0.0, 0.0, yaw);
      base_to_octomap.setRotation(q);
      base_to_octomap.setOrigin(tf2::Vector3(phase, 0.6 * phase, 0.3 * phase));

      const auto grid =
          bridge::rasterize_octree_to_grid(tree, base_to_octomap, kitchen_spec(), "base_link");
      ASSERT_GT(occupied_count(grid), 0) << "yaw " << yaw << ", phase " << phase;
      const LatticeDiff diff = lattice_diff(tree, base_to_octomap, grid);
      EXPECT_EQ(diff.missing, 0) << "yaw " << yaw << ", phase " << phase << ": obstacle(s) lost";
      EXPECT_EQ(diff.extra, 0) << "yaw " << yaw << ", phase " << phase << ": cell(s) the octree "
                               << "does not have — dilation is back";
    }
  }
}

TEST(OctreeToGrid, TheOrientationIsTheOctreeToBaseRotationAndIsAlwaysAUnitQuaternion) {
  // Wire contract: consumers refuse a non-unit quaternion rather than assume
  // identity; the rotation must carry grid axes (the octree's) into
  // `base_frame` or every cell lands somewhere the robot isn't.
  const octomap::OcTree tree = kitchen_scene();
  for (const double yaw : {0.0, 0.37, 2.6}) {
    tf2::Transform base_to_octomap;
    tf2::Quaternion q;
    q.setRPY(0.1, -0.2, yaw);
    base_to_octomap.setRotation(q);
    base_to_octomap.setOrigin(tf2::Vector3(0.3, -0.2, 0.15));

    const auto grid =
        bridge::rasterize_octree_to_grid(tree, base_to_octomap, kitchen_spec(), "base_link");
    const tf2::Quaternion published(grid.orientation.x, grid.orientation.y, grid.orientation.z,
                                    grid.orientation.w);
    EXPECT_NEAR(published.length(), 1.0, 1e-12) << "yaw " << yaw;
    // Same rotation as octomap→base, up to the quaternion double cover.
    const tf2::Quaternion expected = base_to_octomap.inverse().getRotation().normalized();
    const double dot = std::fabs(published.dot(expected));
    EXPECT_NEAR(dot, 1.0, 1e-12) << "yaw " << yaw << ": orientation is not octree→base";
  }
}

TEST(OctreeToGrid, ACoarseLeafMarksEveryCellUnderIt) {
  // octomap prunes eight equal-valued siblings into one leaf of twice the edge
  // length (a published /octomap_binary is pruned), so a leaf covers
  // (leaf_size/resolution)^3 cells, not one — get that wrong and 7/8 of a
  // pruned obstacle leaves the kernel's grid. Seeded with updateNode (not
  // rays): pruning needs the eight cells at exactly equal values.
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

  // The 100 mm leaf spans [0, 0.1) on every axis. At the tree's own 50 mm
  // resolution that is 2x2x2 = 8 cells, every one of them occupied.
  const auto grid = bridge::rasterize_octree_to_grid(tree, tf2::Transform::getIdentity(),
                                                     ball(0.05, 0.05, 0.05, 0.05), "base_link");
  EXPECT_EQ(occupied_count(grid), 8) << "a coarse leaf must fill every cell beneath it";
}

TEST(OctreeToGrid, AGridStraddlingTheOctreesKeyRangeStillSeesEveryLeaf) {
  // octomap's bbx iterator key conversion FAILS silently (empty iteration) for
  // coordinates outside the tree's addressable ±32768·resolution, so an
  // unusable bbx falls back to walking the whole tree. The grid must straddle
  // the boundary: entirely-beyond-range holds no leaves to lose.
  octomap::OcTree tree(0.1);  // addressable to ±3276.8 m
  tree.updateNode(octomap::point3d(3276.05F, 0.05F, 0.05F), true);

  const auto grid = bridge::rasterize_octree_to_grid(tree, tf2::Transform::getIdentity(),
                                                     ball(3276.5, 0.05, 0.05, 1.0), "base_link");
  ASSERT_FALSE(grid.occupancy.empty()) << "the lattice must still be placeable out here";
  EXPECT_EQ(occupied_count(grid), 1) << "the fallback walk must still find the leaf";
}

TEST(OctreeToGrid, AnAbsurdSpecIsRefusedRatherThanAllocated) {
  // A radius sizing an allocation no kernel could accept is a
  // misconfiguration; `std::bad_alloc` crashes the node, an empty grid does
  // not. The node logs and publishes nothing rather than a zero-cell grid.
  octomap::OcTree tree(0.1);
  tree.updateNode(octomap::point3d(0.05F, 0.05F, 0.05F), true);
  const auto grid = bridge::rasterize_octree_to_grid(tree, tf2::Transform::getIdentity(),
                                                     ball(0.05, 0.05, 0.05, 4000.0), "base_link");
  EXPECT_TRUE(grid.occupancy.empty());
}

TEST(OctreeToGrid, RasterizationCostAcrossTreeResolutions) {
  // The PRODUCER half of the 25 -> 15 mm lever. `PLAN.md` §5 un-struck that
  // lever after measuring the kernel CONSUMING a finer grid (p99 0.825 ms at
  // 15 mm, against a 26.7 ms estimate) and named this as the other half: the
  // bridge rebuilds the grid on every publish, and the grid's resolution IS the
  // octree's, so a finer kernel grid means a finer TREE.
  //
  // Two costs move oppositely with a finer tree, which is why this is measured:
  // the dense `occupancy` buffer is O(1/res^3) to allocate and zero, while the
  // marking loop iterates OCCUPIED LEAVES, which are a surface and grow as
  // O(1/res^2). The estimate-versus-measurement error that made the original
  // strike wrong by 32x is available here too, so this prints the curve instead
  // of asserting a shape.
  //
  // `kitchen_spec()`'s 1.0 m ball is slightly smaller than the shipped 1.05 m
  // coverage radius (`deploy_e2e.launch.py::_octomap_coverage_radius`), so these
  // cell counts are a mild under-estimate of the deployed ones.
  constexpr double kResolutions[] = {0.025, 0.020, 0.015, 0.0125};
  for (const double resolution : kResolutions) {
    const octomap::OcTree tree = kitchen_scene_at(resolution);
    const auto spec = kitchen_spec();
    (void)bridge::rasterize_octree_to_grid(tree, tf2::Transform::getIdentity(), spec, "base_link");

    const auto started = std::chrono::steady_clock::now();
    constexpr int kIterations = 10;
    std::size_t cells = 0;
    std::size_t occupied = 0;
    for (int i = 0; i < kIterations; ++i) {
      const auto grid =
          bridge::rasterize_octree_to_grid(tree, tf2::Transform::getIdentity(), spec, "base_link");
      cells = grid.occupancy.size();
      occupied = occupied_count(grid);
    }
    const double per_call_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started)
            .count() /
        kIterations;
    std::cout << "  resolution " << resolution * 1e3 << " mm: " << cells << " cells, " << occupied
              << " occupied, " << per_call_ms << " ms/call" << std::endl;

    if (cells == 0) {
      // Not a failure: `kMaxCells` (4e6) is a real allocation guard, and at the
      // shipped coverage radius 12.5 mm crosses it. Refusing is the correct
      // fail-closed behaviour — but it means 12.5 mm is unreachable without
      // raising that guard, which is a decision, not a manifest edit. Pinned
      // here so the boundary is a recorded fact rather than a surprise.
      EXPECT_LT(resolution, 0.015) << "an unexpected resolution was refused outright";
      continue;
    }
    ASSERT_GT(occupied, 0U) << "non-empty grid carried no occupancy at " << resolution << " m";
    // The publish period is the contract at every resolution, not only the
    // shipped one: a tree the bridge cannot rasterize in time is not a usable
    // lever however cheap the kernel finds it.
    EXPECT_LT(per_call_ms, 100.0) << "rasterization at " << resolution * 1e3 << " mm took "
                                  << per_call_ms << " ms, over the 10 Hz publish period";
  }
}

TEST(OctreeToGrid, RasterizingTheKitchenStaysInsideThePublishBudget) {
  // Bridge re-derives the grid every published tick (`publish_rate_hz` 10.0 →
  // 100 ms period). Cost is proportional to occupied leaves in the volume,
  // not cell count (integer index arithmetic vs. a separating-axis test).
  const octomap::OcTree tree = kitchen_scene();
  const auto spec = kitchen_spec();
  // Warm the tree's internal state so the timing is the rasterization's.
  (void)bridge::rasterize_octree_to_grid(tree, tf2::Transform::getIdentity(), spec, "base_link");

  const auto started = std::chrono::steady_clock::now();
  constexpr int kIterations = 10;
  for (int i = 0; i < kIterations; ++i) {
    const auto grid =
        bridge::rasterize_octree_to_grid(tree, tf2::Transform::getIdentity(), spec, "base_link");
    ASSERT_GT(occupied_count(grid), 0);
  }
  const auto elapsed = std::chrono::steady_clock::now() - started;
  const double per_call_ms =
      std::chrono::duration<double, std::milli>(elapsed).count() / kIterations;
  EXPECT_LT(per_call_ms, 25.0) << "rasterization took " << per_call_ms
                               << " ms, a quarter of the 10 Hz publish period";
}
