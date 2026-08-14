// SPDX-License-Identifier: Apache-2.0
// The grasped payload must leave world occupancy — and only the payload.
//
// `openral_msgs/AttachedCollisionObject` has always said "The same object must
// be absent from world occupancy while attached", and nothing enforced it.
// Before the grasp the object IS world occupancy: honest sensor returns marked
// its cells. At the attach transition those cells stop describing the world and
// start describing the robot's own payload — which the kernel re-checks as
// collision-active attached geometry — but they stayed in the map, and on the
// 2026-08-14 acceptance run the arm stopped 1.8 mm off the surface of the
// object it was holding, +32 mm above the attested support plane (so the
// support-contact witness correctly refused to exempt it).
//
// The depth self-filter's transparency (8149344 + f02fe7d) clears the payload's
// silhouette only where a ray still crosses it: a cell the camera cannot reach
// — occluded, outside the frustum, or simply between two rays — is never
// touched, and `OccupancyPersistence.AConfirmedVoxelSurvivesWhenNoRayEverCrosses
// It` (test_occupancy_persistence.cpp) pins what happens to it: nothing, for
// ever. So the transition needs an explicit removal, which is what these tests
// pin, on a real `octomap::OcTree` seeded through real rays with the deploy-sim
// parameters.

#include <cmath>
#include <cstddef>
#include <vector>

#include <gtest/gtest.h>
#include <octomap/OcTree.h>
#include <octomap/Pointcloud.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Transform.h>
#include <tf2/LinearMath/Vector3.h>

#include <openral_msgs/msg/attached_collision_object.hpp>
#include <openral_msgs/msg/attached_collision_primitive.hpp>

#include "openral_octomap_bridge/octree_to_grid.hpp"
#include "openral_octomap_bridge/payload_clearing.hpp"

namespace bridge = openral_octomap_bridge;
using Primitive = openral_msgs::msg::AttachedCollisionPrimitive;
using WireObject = openral_msgs::msg::AttachedCollisionObject;

namespace {

// The deploy-sim octomap_server parameters (sim_e2e.launch.py).
constexpr double kResolution = 0.025;
constexpr double kOccupancyThres = 0.8;
constexpr double kClampingMax = 0.85;
constexpr double kProbHit = 0.7;
constexpr double kProbMiss = 0.4;
constexpr double kMaxRange = 4.0;

// The grid's cell-cube circumradius: resolution·√3/2 ≈ 21.65 mm at 25 mm cells.
const double kCircumradius = 0.5 * kResolution * 1.7320508075688772;

// The payload: a 80x80x200 mm box primitive (a carried baguette and the
// fingers holding it) centred 0.40 m in front of the base at z = 0.35, inside
// which the physical object's own 60x60x180 mm surface returns fall.
const tf2::Vector3 kPayloadCenter(0.40, 0.0, 0.35);
const tf2::Vector3 kPayloadHalfExtents(0.04, 0.04, 0.10);

// The grid: 12³ cells of 25 mm around the payload, its min corner on the
// octree's own lattice so a grid cell centre and an octree voxel centre are the
// same point (both are 0.0125 + n·0.025).
bridge::GridSpec grid_spec() {
  bridge::GridSpec spec;
  spec.resolution = kResolution;
  spec.sx = 12;
  spec.sy = 12;
  spec.sz = 12;
  spec.box_min[0] = 0.30;
  spec.box_min[1] = -0.15;
  spec.box_min[2] = 0.25;
  return spec;
}

octomap::OcTree deploy_sim_tree() {
  octomap::OcTree tree(kResolution);
  tree.setOccupancyThres(kOccupancyThres);
  tree.setClampingThresMax(kClampingMax);
  tree.setProbHit(kProbHit);
  tree.setProbMiss(kProbMiss);
  return tree;
}

// The depth camera that saw the object before the grasp, at base height.
const octomap::point3d kSensor(0.0F, 0.0F, 0.35F);

// Insert one frame of returns; two frames confirm a cell at occupancy_thres 0.8.
void insert_frame(octomap::OcTree& tree, const std::vector<octomap::point3d>& endpoints) {
  octomap::Pointcloud cloud;
  for (const auto& p : endpoints) {
    cloud.push_back(p);
  }
  tree.insertPointCloud(cloud, kSensor, kMaxRange);
}

void insert_confirmed(octomap::OcTree& tree, const std::vector<octomap::point3d>& endpoints) {
  insert_frame(tree, endpoints);
  insert_frame(tree, endpoints);
}

// The pre-grasp returns off the physical object's near face: inside the
// payload primitive, since the primitive bounds the object plus its grasp.
std::vector<octomap::point3d> object_returns() {
  std::vector<octomap::point3d> pts;
  for (const double y : {-0.031, -0.011, 0.011, 0.031}) {
    for (const double z : {0.281, 0.321, 0.361, 0.406, 0.431}) {
      pts.emplace_back(0.371F, static_cast<float>(y), static_cast<float>(z));
    }
  }
  return pts;
}

// A real obstacle beside the payload — the counter edge the arm must still stop
// against. Its ray passes wide of the payload (|y| >= 0.087 for every x the
// payload spans), so inserting it never clears an object cell. The cell it
// lands in is named by its centre, the only unambiguous way to name a cell.
const octomap::point3d kObstacleReturn(0.463F, 0.113F, 0.363F);
const tf2::Vector3 kObstacleCell(0.4625, 0.1125, 0.3625);

std::size_t occupied_cells(const openral_msgs::msg::OccupancyVoxels& grid) {
  std::size_t n = 0;
  for (const auto v : grid.occupancy) {
    n += (v != 0) ? 1U : 0U;
  }
  return n;
}

// Index of the cell whose centre is nearest coordinate `p` along one axis
// (`llround` off the centre lattice, so a point sitting exactly on a cell
// boundary lands in the cell that boundary opens — octomap's own convention —
// without the floating-point knife edge a bare floor would have).
long cell_axis(double p, double min_corner, double resolution) {
  return std::llround((p - min_corner) / resolution - 0.5);
}

// Linear index of the cell containing base-frame point p (caller guarantees it
// is inside the grid).
std::size_t cell_index(const openral_msgs::msg::OccupancyVoxels& grid, double x, double y,
                       double z) {
  const auto ix = static_cast<std::size_t>(cell_axis(x, grid.origin.x, grid.resolution));
  const auto iy = static_cast<std::size_t>(cell_axis(y, grid.origin.y, grid.resolution));
  const auto iz = static_cast<std::size_t>(cell_axis(z, grid.origin.z, grid.resolution));
  return ix + grid.size_x * (iy + grid.size_y * iz);
}

// Centre of the cell containing base-frame point p — the point the clearing
// predicate (and the kernel's voxel check) actually measures against.
tf2::Vector3 cell_center(const openral_msgs::msg::OccupancyVoxels& grid, double x, double y,
                         double z) {
  return tf2::Vector3(
      grid.origin.x + (cell_axis(x, grid.origin.x, grid.resolution) + 0.5) * grid.resolution,
      grid.origin.y + (cell_axis(y, grid.origin.y, grid.resolution) + 0.5) * grid.resolution,
      grid.origin.z + (cell_axis(z, grid.origin.z, grid.resolution) + 0.5) * grid.resolution);
}

openral_msgs::msg::OccupancyVoxels lowered(const octomap::OcTree& tree) {
  return bridge::rasterize_octree_to_grid(tree, tf2::Transform::getIdentity(), grid_spec(),
                                          "base_link");
}

// The payload as the kernel sees it: one box primitive, posed in the attach
// link, with the attach link itself somewhere else in the base frame.
const tf2::Vector3 kAttachLinkOrigin(0.30, 0.0, 0.30);

tf2::Transform base_from_attach_link() {
  return tf2::Transform(tf2::Quaternion::getIdentity(), kAttachLinkOrigin);
}

WireObject box_payload() {
  WireObject obj;
  obj.object_id = "obj_baguette";
  obj.attach_link = "panda_hand";
  obj.touch_links = {"panda_leftfinger", "panda_rightfinger"};
  obj.pose_in_link.position.x = kPayloadCenter.x() - kAttachLinkOrigin.x();
  obj.pose_in_link.position.y = kPayloadCenter.y() - kAttachLinkOrigin.y();
  obj.pose_in_link.position.z = kPayloadCenter.z() - kAttachLinkOrigin.z();
  obj.pose_in_link.orientation.w = 1.0;
  Primitive prim;
  prim.shape_type = Primitive::SHAPE_BOX;
  prim.shape_dimensions = {kPayloadHalfExtents.x(), kPayloadHalfExtents.y(),
                           kPayloadHalfExtents.z()};
  prim.pose_in_object.orientation.w = 1.0;
  obj.primitives = {prim};
  return obj;
}

std::vector<bridge::PayloadPrimitive> placed(const WireObject& obj) {
  std::vector<bridge::PayloadPrimitive> out;
  EXPECT_TRUE(bridge::place_attached_object(obj, base_from_attach_link(), out));
  return out;
}

}  // namespace

// ── the defect ───────────────────────────────────────────────────────────────

TEST(PayloadClearing, TheGraspedObjectsOwnCellsReachTheKernelsGrid) {
  // The fixture, stated as a fact: 20 cells of the object the robot is holding,
  // marked by honest pre-grasp returns, plus one real obstacle beside it. The
  // lowering step has no idea any of them are the payload.
  octomap::OcTree tree = deploy_sim_tree();
  insert_confirmed(tree, object_returns());
  insert_confirmed(tree, {kObstacleReturn});

  const auto grid = lowered(tree);
  EXPECT_EQ(occupied_cells(grid), 21U);
  EXPECT_NE(grid.occupancy[cell_index(grid, 0.3625, 0.0125, 0.3625)], 0);
}

// ── (a) the attach transition ────────────────────────────────────────────────

TEST(PayloadClearing, AttachClearsThePayloadsOwnCellsAndNothingElse) {
  octomap::OcTree tree = deploy_sim_tree();
  insert_confirmed(tree, object_returns());
  insert_confirmed(tree, {kObstacleReturn});
  auto grid = lowered(tree);
  ASSERT_EQ(occupied_cells(grid), 21U);

  const std::size_t cleared =
      bridge::clear_attached_payload_cells(grid, placed(box_payload()), 0.0);

  EXPECT_EQ(cleared, 20U);
  EXPECT_EQ(occupied_cells(grid), 1U) << "the obstacle beside the payload must survive";
  EXPECT_NE(
      grid.occupancy[cell_index(grid, kObstacleCell.x(), kObstacleCell.y(), kObstacleCell.z())], 0);
}

TEST(PayloadClearing, TheStopCellJustOffThePayloadSurfaceIsCleared) {
  // The acceptance run's tripping cell sat 1.8 mm off the payload surface —
  // outside the primitive, so an "inside the primitive" rule would leave the
  // false stop exactly where it was. A cell whose centre is that close is a
  // cell the payload's own volume passes through, and it goes.
  auto grid = lowered(deploy_sim_tree());
  const std::size_t idx = cell_index(grid, 0.40, 0.0, 0.35);
  grid.occupancy[idx] = 1;

  bridge::PayloadPrimitive sphere;
  sphere.shape_type = Primitive::SHAPE_SPHERE;
  // Sphere centred one cell below, its surface 1.8 mm short of the cell centre.
  const tf2::Vector3 center = cell_center(grid, 0.40, 0.0, 0.35);
  sphere.pose =
      tf2::Transform(tf2::Quaternion::getIdentity(), center - tf2::Vector3(0.0, 0.0, kResolution));
  sphere.radius = kResolution - 0.0018;

  EXPECT_EQ(bridge::clear_attached_payload_cells(grid, {sphere}, 0.0), 1U);
  EXPECT_EQ(grid.occupancy[idx], 0);
}

TEST(PayloadClearing, ClearingStopsAtTheCellCircumradius) {
  // …and stops there. A cell a millimetre beyond the circumradius is a cell the
  // payload's volume cannot reach, so it keeps its occupancy: the protection
  // given up is bounded by the map's own discretisation, not by a fudge factor.
  auto grid = lowered(deploy_sim_tree());
  const std::size_t idx = cell_index(grid, 0.40, 0.0, 0.35);
  grid.occupancy[idx] = 1;

  bridge::PayloadPrimitive sphere;
  sphere.shape_type = Primitive::SHAPE_SPHERE;
  const tf2::Vector3 center = cell_center(grid, 0.40, 0.0, 0.35);
  sphere.pose =
      tf2::Transform(tf2::Quaternion::getIdentity(), center - tf2::Vector3(0.0, 0.0, kResolution));
  sphere.radius = kResolution - kCircumradius - 0.001;

  EXPECT_EQ(bridge::clear_attached_payload_cells(grid, {sphere}, 0.0), 0U);
  EXPECT_NE(grid.occupancy[idx], 0);
}

TEST(PayloadClearing, ClearingNeverMarks) {
  // The one direction this must never run. An all-free grid stays all-free:
  // the function only ever writes 0.
  auto grid = lowered(deploy_sim_tree());
  ASSERT_EQ(occupied_cells(grid), 0U);

  EXPECT_EQ(bridge::clear_attached_payload_cells(grid, placed(box_payload()), 0.0), 0U);
  EXPECT_EQ(occupied_cells(grid), 0U);
}

// ── (b) while attached, and at detach ────────────────────────────────────────

TEST(PayloadClearing, ClearingFollowsThePayloadPoseEveryFrame) {
  // Not a one-shot snapshot at the attach instant: the payload is placed from
  // the live attachment state and the live TF on every published grid, so the
  // cells cleared are the ones the payload occupies NOW. Cells where it merely
  // used to be are ordinary world occupancy again (the depth self-filter's
  // clearing rays retire them, per test_occupancy_persistence).
  octomap::OcTree tree = deploy_sim_tree();
  insert_confirmed(tree, object_returns());
  insert_confirmed(tree, {kObstacleReturn});

  auto at_old_pose = lowered(tree);
  EXPECT_EQ(bridge::clear_attached_payload_cells(at_old_pose, placed(box_payload()), 0.0), 20U);
  EXPECT_NE(at_old_pose.occupancy[cell_index(at_old_pose, kObstacleCell.x(), kObstacleCell.y(),
                                             kObstacleCell.z())],
            0);

  // The arm carries the payload over to where the obstacle cell is.
  WireObject moved = box_payload();
  moved.pose_in_link.position.x = kObstacleCell.x() - kAttachLinkOrigin.x();
  moved.pose_in_link.position.y = kObstacleCell.y() - kAttachLinkOrigin.y();
  moved.pose_in_link.position.z = kObstacleCell.z() - kAttachLinkOrigin.z();
  auto at_new_pose = lowered(tree);
  const std::size_t cleared = bridge::clear_attached_payload_cells(at_new_pose, placed(moved), 0.0);

  EXPECT_EQ(cleared, 1U);
  EXPECT_EQ(at_new_pose.occupancy[cell_index(at_new_pose, kObstacleCell.x(), kObstacleCell.y(),
                                             kObstacleCell.z())],
            0);
  EXPECT_EQ(occupied_cells(at_new_pose), 20U) << "the cells it left behind are the world's again";
}

TEST(PayloadClearing, DetachReturnsThePayloadToWorldOccupancyImmediately) {
  // Released, the object is a real obstacle again. The clearing carries no
  // state of its own — it is derived from the attachment set on the wire — so
  // the frame after the object leaves that set is already the frame that
  // publishes it. (Re-marking a cell the transparency rays cleared costs the
  // two-hit confirmation octomap's tuning buys; that latency is octomap's, and
  // is pinned by test_occupancy_persistence.)
  octomap::OcTree tree = deploy_sim_tree();
  insert_confirmed(tree, object_returns());
  auto grid = lowered(tree);
  ASSERT_EQ(occupied_cells(grid), 20U);

  const std::vector<bridge::PayloadPrimitive> nothing_attached;
  EXPECT_EQ(bridge::clear_attached_payload_cells(grid, nothing_attached, 0.0), 0U);
  EXPECT_EQ(occupied_cells(grid), 20U);
}

// ── placement: the kernel's composition, or nothing at all ───────────────────

TEST(PayloadClearing, PlacementComposesTheAttachLinkObjectAndPrimitivePoses) {
  // FK(attach_link) · pose_in_link · pose_in_object, the same chain the kernel
  // uses. A second primitive offset in the object frame must land where that
  // chain puts it — and clearing it proves the composition, not just the parse.
  auto grid = lowered(deploy_sim_tree());
  const std::size_t idx = cell_index(grid, 0.40, 0.0, 0.5375);
  grid.occupancy[idx] = 1;

  WireObject obj = box_payload();
  Primitive lid;
  lid.shape_type = Primitive::SHAPE_SPHERE;
  lid.shape_dimensions = {0.02};
  // 0.18 m up the object frame → 0.35 + 0.18 = 0.53 in the base frame.
  lid.pose_in_object.position.z = 0.18;
  lid.pose_in_object.orientation.w = 1.0;
  obj.primitives.push_back(lid);

  const auto prims = placed(obj);
  ASSERT_EQ(prims.size(), 2U);
  EXPECT_EQ(bridge::clear_attached_payload_cells(grid, prims, 0.0), 1U);
  EXPECT_EQ(grid.occupancy[idx], 0);
}

TEST(PayloadClearing, ARotatedPayloadIsPlacedByItsOrientation) {
  // A capsule lying along the base +x axis (its own +Z rotated by 90° about y)
  // must clear the cells beside the attach link, not above it.
  auto grid = lowered(deploy_sim_tree());
  const std::size_t along_x = cell_index(grid, 0.4875, 0.0, 0.35);
  const std::size_t along_z = cell_index(grid, 0.40, 0.0, 0.4375);
  grid.occupancy[along_x] = 1;
  grid.occupancy[along_z] = 1;

  WireObject obj = box_payload();
  Primitive rod;
  rod.shape_type = Primitive::SHAPE_CAPSULE;
  rod.shape_dimensions = {0.01, 0.20};  // radius, FULL central-segment length
  tf2::Quaternion q;
  q.setRPY(0.0, 1.5707963267948966, 0.0);
  rod.pose_in_object.orientation.x = q.x();
  rod.pose_in_object.orientation.y = q.y();
  rod.pose_in_object.orientation.z = q.z();
  rod.pose_in_object.orientation.w = q.w();
  obj.primitives = {rod};

  EXPECT_EQ(bridge::clear_attached_payload_cells(grid, placed(obj), 0.0), 1U);
  EXPECT_EQ(grid.occupancy[along_x], 0) << "capsule runs along base +x";
  EXPECT_NE(grid.occupancy[along_z], 0) << "…and not along base +z";
}

TEST(PayloadClearing, AnUnreadablePayloadClearsNothingAtAll) {
  // Fail-closed, in the only direction that is safe here: a payload the bridge
  // cannot place stays in the map. Every rejection is the kernel's own ingest
  // rule (unknown tag, too few dimensions), plus a degenerate pose quaternion.
  std::vector<bridge::PayloadPrimitive> out;

  WireObject unknown_shape = box_payload();
  unknown_shape.primitives[0].shape_type = 9;
  EXPECT_FALSE(bridge::place_attached_object(unknown_shape, base_from_attach_link(), out));
  EXPECT_TRUE(out.empty());

  WireObject short_box = box_payload();
  short_box.primitives[0].shape_dimensions = {0.04, 0.04};
  EXPECT_FALSE(bridge::place_attached_object(short_box, base_from_attach_link(), out));
  EXPECT_TRUE(out.empty());

  WireObject no_primitives = box_payload();
  no_primitives.primitives.clear();
  EXPECT_FALSE(bridge::place_attached_object(no_primitives, base_from_attach_link(), out));
  EXPECT_TRUE(out.empty());

  WireObject zero_quat = box_payload();
  zero_quat.pose_in_link.orientation.w = 0.0;
  EXPECT_FALSE(bridge::place_attached_object(zero_quat, base_from_attach_link(), out));
  EXPECT_TRUE(out.empty());

  WireObject negative_radius = box_payload();
  negative_radius.primitives[0].shape_type = Primitive::SHAPE_SPHERE;
  negative_radius.primitives[0].shape_dimensions = {-0.02};
  EXPECT_FALSE(bridge::place_attached_object(negative_radius, base_from_attach_link(), out));
  EXPECT_TRUE(out.empty());

  // A second, valid object in the same message is still placed on its own.
  EXPECT_TRUE(bridge::place_attached_object(box_payload(), base_from_attach_link(), out));
  EXPECT_EQ(out.size(), 1U);
}

TEST(PayloadClearing, AMalformedGridIsLeftAlone) {
  auto grid = lowered(deploy_sim_tree());
  grid.occupancy.assign(grid.occupancy.size(), 1);
  auto truncated = grid;
  truncated.occupancy.pop_back();
  const auto before = truncated.occupancy;

  EXPECT_EQ(bridge::clear_attached_payload_cells(truncated, placed(box_payload()), 0.0), 0U);
  EXPECT_EQ(truncated.occupancy, before);

  auto no_resolution = grid;
  no_resolution.resolution = 0.0;
  EXPECT_EQ(bridge::clear_attached_payload_cells(no_resolution, placed(box_payload()), 0.0), 0U);
}
