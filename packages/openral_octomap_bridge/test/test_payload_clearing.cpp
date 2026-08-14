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

#include <algorithm>
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

// Centre of the cell at linear index `idx` — the point both predicates measure.
tf2::Vector3 center_of_index(const openral_msgs::msg::OccupancyVoxels& grid, std::size_t idx) {
  const std::size_t ix = idx % grid.size_x;
  const std::size_t iy = (idx / grid.size_x) % grid.size_y;
  const std::size_t iz = idx / (static_cast<std::size_t>(grid.size_x) * grid.size_y);
  return tf2::Vector3(grid.origin.x + (static_cast<double>(ix) + 0.5) * grid.resolution,
                      grid.origin.y + (static_cast<double>(iy) + 0.5) * grid.resolution,
                      grid.origin.z + (static_cast<double>(iz) + 0.5) * grid.resolution);
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

// The bridge's placement step exactly as the node runs it: one wire object in,
// its payload primitives AND its support-contact attestation out, both in the
// grid frame. Keeping them together in the tests is the point — clearing with
// the primitives but without the patches is the defect this file pins.
struct Placed {
  std::vector<bridge::PayloadPrimitive> primitives;
  std::vector<bridge::SupportPatch> patches;
};

Placed placed(const WireObject& obj) {
  Placed p;
  EXPECT_TRUE(bridge::place_attached_object(obj, base_from_attach_link(), p.primitives, p.patches));
  return p;
}

std::size_t clear_with(openral_msgs::msg::OccupancyVoxels& grid, const Placed& p,
                       double padding_m = 0.0) {
  return bridge::clear_attached_payload_cells(grid, p.primitives, p.patches, padding_m);
}

// ── the support-contact partition fixture (ADR-0092 D6) ──────────────────────
//
// A payload RESTING on a counter, which is the situation the two mechanisms
// collide in. The counter's top face is at z = 0.3593 in the base frame and
// the lattice phases the surface cell's centre 3.2 mm ABOVE it — the exact
// phase of the 2026-08-13 baguette run, where a ~1 mm physical contact read as
// 15.7 mm of cube penetration. The payload's bottom face sits on that face, so
// its bottom cell layer IS the counter's top cell layer: a clearing that knows
// only the payload's volume takes the counter with it.

constexpr double kSupportFaceZ = 0.3593;
const tf2::Vector3 kRestingHalfExtents(0.04, 0.04, 0.05);
const tf2::Vector3 kRestingCenter(0.40, 0.0, kSupportFaceZ + kRestingHalfExtents.z());
// What World State attests: a 60 mm patch about the contact point and the
// deepest of the six real MuJoCo counter-payload contacts of that run.
constexpr double kAttestedPatchRadius = 0.06;
constexpr double kAttestedPenetration = 0.00137;
// The bound the attestation buys at 25 mm cells: the cube's half-width
// projected on the +z normal (12.5 mm) plus that physical depth.
const double kWithholdCeiling = 0.5 * kResolution + kAttestedPenetration;

// The four cell-centre coordinates the payload's 80 mm footprint spans on each
// lateral axis (the next centre out is 22.5 mm clear of the box face, past the
// circumradius, so it is never in reach).
const std::vector<double> kFootprintX{0.3625, 0.3875, 0.4125, 0.4375};
const std::vector<double> kFootprintY{-0.0375, -0.0125, 0.0125, 0.0375};
// The counter's top cell layer (centre 3.2 mm above the attested face) and the
// three payload-silhouette layers above it (+28.2, +53.2, +78.2 mm).
constexpr double kSupportLayerZ = 0.3625;
const std::vector<double> kResidueLayerZ{0.3875, 0.4125, 0.4375};

WireObject resting_payload() {
  WireObject obj = box_payload();
  obj.pose_in_link.position.x = kRestingCenter.x() - kAttachLinkOrigin.x();
  obj.pose_in_link.position.y = kRestingCenter.y() - kAttachLinkOrigin.y();
  obj.pose_in_link.position.z = kRestingCenter.z() - kAttachLinkOrigin.z();
  obj.primitives[0].shape_dimensions = {kRestingHalfExtents.x(), kRestingHalfExtents.y(),
                                        kRestingHalfExtents.z()};
  // The witness, in the ATTACHED OBJECT's own frame: the contact point is the
  // payload's own bottom face, the normal points from the counter up into the
  // payload.
  obj.support_contact_valid = true;
  obj.support_contact.support_id = "sim:counter_main";
  obj.support_contact.contact_point_in_object.z = -kRestingHalfExtents.z();
  obj.support_contact.contact_normal_in_object.z = 1.0;
  obj.support_contact.patch_radius_m = kAttestedPatchRadius;
  obj.support_contact.max_penetration_m = kAttestedPenetration;
  obj.support_contact.evidence_kind = "sim_geom_distance";
  return obj;
}

// The pre-grasp returns off the resting payload's near face: three rows in the
// x = 0.3625 column, each ray reaching its own cell without crossing another
// marked one.
std::vector<octomap::point3d> resting_object_returns() {
  std::vector<octomap::point3d> pts;
  for (const double y : kFootprintY) {
    for (const double z : kResidueLayerZ) {
      pts.emplace_back(0.371F, static_cast<float>(y), static_cast<float>(z));
    }
  }
  return pts;
}

// The counter's top surface under the payload. It is marked directly rather
// than through rays, because while the payload rests on it NO ray can reach it
// — that is the whole reason it is still in the map, and the reason clearing it
// away is a loss the map has no way to recover.
void mark_support_surface(openral_msgs::msg::OccupancyVoxels& grid) {
  for (const double x : kFootprintX) {
    for (const double y : kFootprintY) {
      grid.occupancy[cell_index(grid, x, y, kSupportLayerZ)] = 1;
    }
  }
}

// The map as the bridge receives it: the payload's own silhouette (12 cells),
// the counter surface beneath it (16), and one real obstacle beside it.
openral_msgs::msg::OccupancyVoxels resting_grid() {
  octomap::OcTree tree = deploy_sim_tree();
  insert_confirmed(tree, resting_object_returns());
  insert_confirmed(tree, {kObstacleReturn});
  auto grid = lowered(tree);
  mark_support_surface(grid);
  return grid;
}

// The linear indices whose occupancy went from set to zero.
std::vector<std::size_t> cleared_indices(const openral_msgs::msg::OccupancyVoxels& before,
                                         const openral_msgs::msg::OccupancyVoxels& after) {
  std::vector<std::size_t> out;
  for (std::size_t i = 0; i < before.occupancy.size(); ++i) {
    if (before.occupancy[i] != 0 && after.occupancy[i] == 0) {
      out.push_back(i);
    }
  }
  return out;
}

// ── the kernel's side of the mirror ──────────────────────────────────────────
//
// `support_contact_exempts`, transcribed TERM FOR TERM from
// `cpp/openral_safety_kernel/src/collision.cpp` (the safety kernel is a separate
// colcon package this one must not depend on — Layer 2 does not link Layer 6 —
// so the mirror is deliberate and tracked as item 8 of
// `docs/methods/14-duplication-watch.md`). It is the SPECIFICATION that
// `support_patch_withholds` claims to implement with `slack = 0`; the tests
// below evaluate the two on the same cells, so a drift on either side of the
// mirror fails here instead of in the field.
bool kernel_support_contact_exempts(const bridge::SupportPatch& patch, const tf2::Vector3& center,
                                    double resolution, double slack) {
  if (!(patch.patch_radius > 0.0) || resolution <= 0.0) {
    return false;
  }
  const tf2::Vector3 delta = center - patch.point;
  const double height = delta.dot(patch.normal);
  const double lateral_sq = std::max(0.0, delta.length2() - height * height);
  const double half = 0.5 * resolution;
  const double normal_half_width =
      half *
      (std::fabs(patch.normal.x()) + std::fabs(patch.normal.y()) + std::fabs(patch.normal.z()));
  const double lateral_pad = half * 1.7320508075688772;  // sqrt(3)
  const double reach = patch.patch_radius + lateral_pad;
  if (lateral_sq > reach * reach) {
    return false;
  }
  return height <= normal_half_width + patch.max_penetration + slack;
}

// The kernel's `attached_contact_tolerance` — 1 mm of physical FK/pose slack,
// the slack term the bridge drops (`lifecycle_kernel.cpp`,
// `attached_contact_tolerance_m`).
constexpr double kKernelContactTolerance = 0.001;

// What the safety kernel's lifecycle node will ACCEPT on the wire before it
// fails the whole attachment message closed: `support_witness_max_penetration_m`
// (10 mm) and `support_witness_max_patch_radius_m` (0.5 m), declared in
// `cpp/openral_safety_kernel/src/lifecycle_kernel.cpp`. The producer holds the
// same two numbers (`openral_hal._sim_attachment_evidence`).
constexpr double kKernelMaxPenetration = 0.01;
constexpr double kKernelMaxPatchRadius = 0.5;

// A patch stated directly in the grid frame, for probing the predicate itself.
bridge::SupportPatch grid_frame_patch(const tf2::Vector3& point, const tf2::Vector3& normal,
                                      double patch_radius, double penetration) {
  bridge::SupportPatch patch;
  patch.point = point;
  patch.normal = normal.normalized();
  patch.patch_radius = patch_radius;
  patch.max_penetration = penetration;
  return patch;
}

// ── the 2026-08-14 round-5 residue class ─────────────────────────────────────
//
// Field run spark:/home/allopart/openral-runs/2026-08-14-round5/baguette/: a
// cell +35.8 mm ALONG THE OUTWARD SUPPORT NORMAL, laterally well inside the
// attested patch — inside the patch CYLINDER — reported as withheld, while the
// kernel correctly refused to exempt it (its bound here is the cube's projected
// half-width, 12.5 mm, plus the 1.37 mm of attested physical depth). A withhold
// shaped like the cylinder rather than like the support HALF-SPACE SLAB does
// exactly that, and it resurrects the residue 21ecf82 eliminated. The offset is
// carried as a constant so the class is pinned by its field number.
constexpr double kFieldResidueOffset = 0.0358;

// The same resting payload, lowered so that its attested support plane sits
// `kFieldResidueOffset` BELOW an occupancy cell-centre layer: the field cell is
// then a real cell of a real grid, inside the payload's own volume and inside
// the patch cylinder, and the along-normal bound is the only thing that decides
// it. Plane at 0.3625 − 0.0358 = 0.3267 m.
const double kFieldPlaneZ = kSupportLayerZ - kFieldResidueOffset;
const tf2::Vector3 kFieldCenter(0.40, 0.0, kFieldPlaneZ + kRestingHalfExtents.z());

WireObject field_round5_payload() {
  WireObject obj = resting_payload();
  obj.pose_in_link.position.x = kFieldCenter.x() - kAttachLinkOrigin.x();
  obj.pose_in_link.position.y = kFieldCenter.y() - kAttachLinkOrigin.y();
  obj.pose_in_link.position.z = kFieldCenter.z() - kAttachLinkOrigin.z();
  return obj;
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

  const std::size_t cleared = clear_with(grid, placed(box_payload()));

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

  EXPECT_EQ(bridge::clear_attached_payload_cells(grid, {sphere}, {}, 0.0), 1U);
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

  EXPECT_EQ(bridge::clear_attached_payload_cells(grid, {sphere}, {}, 0.0), 0U);
  EXPECT_NE(grid.occupancy[idx], 0);
}

TEST(PayloadClearing, ClearingNeverMarks) {
  // The one direction this must never run. An all-free grid stays all-free:
  // the function only ever writes 0.
  auto grid = lowered(deploy_sim_tree());
  ASSERT_EQ(occupied_cells(grid), 0U);

  EXPECT_EQ(clear_with(grid, placed(box_payload())), 0U);
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
  EXPECT_EQ(clear_with(at_old_pose, placed(box_payload())), 20U);
  EXPECT_NE(at_old_pose.occupancy[cell_index(at_old_pose, kObstacleCell.x(), kObstacleCell.y(),
                                             kObstacleCell.z())],
            0);

  // The arm carries the payload over to where the obstacle cell is.
  WireObject moved = box_payload();
  moved.pose_in_link.position.x = kObstacleCell.x() - kAttachLinkOrigin.x();
  moved.pose_in_link.position.y = kObstacleCell.y() - kAttachLinkOrigin.y();
  moved.pose_in_link.position.z = kObstacleCell.z() - kAttachLinkOrigin.z();
  auto at_new_pose = lowered(tree);
  const std::size_t cleared = clear_with(at_new_pose, placed(moved));

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
  EXPECT_EQ(bridge::clear_attached_payload_cells(grid, nothing_attached, {}, 0.0), 0U);
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

  const Placed p = placed(obj);
  ASSERT_EQ(p.primitives.size(), 2U);
  EXPECT_EQ(clear_with(grid, p), 1U);
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

  EXPECT_EQ(clear_with(grid, placed(obj)), 1U);
  EXPECT_EQ(grid.occupancy[along_x], 0) << "capsule runs along base +x";
  EXPECT_NE(grid.occupancy[along_z], 0) << "…and not along base +z";
}

TEST(PayloadClearing, AnUnreadablePayloadClearsNothingAtAll) {
  // Fail-closed, in the only direction that is safe here: a payload the bridge
  // cannot place stays in the map. Every rejection is the kernel's own ingest
  // rule (unknown tag, too few dimensions), plus a degenerate pose quaternion.
  std::vector<bridge::PayloadPrimitive> out;
  std::vector<bridge::SupportPatch> out_patches;

  WireObject unknown_shape = box_payload();
  unknown_shape.primitives[0].shape_type = 9;
  EXPECT_FALSE(
      bridge::place_attached_object(unknown_shape, base_from_attach_link(), out, out_patches));
  EXPECT_TRUE(out.empty());

  WireObject short_box = box_payload();
  short_box.primitives[0].shape_dimensions = {0.04, 0.04};
  EXPECT_FALSE(bridge::place_attached_object(short_box, base_from_attach_link(), out, out_patches));
  EXPECT_TRUE(out.empty());

  WireObject no_primitives = box_payload();
  no_primitives.primitives.clear();
  EXPECT_FALSE(
      bridge::place_attached_object(no_primitives, base_from_attach_link(), out, out_patches));
  EXPECT_TRUE(out.empty());

  WireObject zero_quat = box_payload();
  zero_quat.pose_in_link.orientation.w = 0.0;
  EXPECT_FALSE(bridge::place_attached_object(zero_quat, base_from_attach_link(), out, out_patches));
  EXPECT_TRUE(out.empty());

  WireObject negative_radius = box_payload();
  negative_radius.primitives[0].shape_type = Primitive::SHAPE_SPHERE;
  negative_radius.primitives[0].shape_dimensions = {-0.02};
  EXPECT_FALSE(
      bridge::place_attached_object(negative_radius, base_from_attach_link(), out, out_patches));
  EXPECT_TRUE(out.empty());

  // A second, valid object in the same message is still placed on its own.
  EXPECT_TRUE(
      bridge::place_attached_object(box_payload(), base_from_attach_link(), out, out_patches));
  EXPECT_EQ(out.size(), 1U);
  EXPECT_TRUE(out_patches.empty()) << "box_payload() attests no support contact";
}

TEST(PayloadClearing, AMalformedGridIsLeftAlone) {
  auto grid = lowered(deploy_sim_tree());
  grid.occupancy.assign(grid.occupancy.size(), 1);
  auto truncated = grid;
  truncated.occupancy.pop_back();
  const auto before = truncated.occupancy;

  EXPECT_EQ(clear_with(truncated, placed(box_payload())), 0U);
  EXPECT_EQ(truncated.occupancy, before);

  auto no_resolution = grid;
  no_resolution.resolution = 0.0;
  EXPECT_EQ(clear_with(no_resolution, placed(box_payload())), 0U);
}

// ── the partition against the support-contact witness (ADR-0092 D6) ──────────
//
// The clearing above and the kernel's support-contact witness are each correct
// on their own and destroy each other when combined. The witness stays alive
// only while some OCCUPIED cell it would exempt is still touching the payload
// (`update_support_contact_witnesses`), and the cells that satisfy that are the
// counter's own top layer — which is inside the resting payload's volume and so
// was being cleared away. Observed 2/2 on 2026-08-14 (baguette+counter,
// cup+island): witness arms, the clearing removes the support cells,
// `support_witness_separated live=0x0 was=0x1` fires 2.7 s later with ground
// truth still touching at +0.000 mm, and the same contact then re-trips
// unexempted (`sweep_min == min_distance`: nothing was exempted at all).
//
// The fix is a partition. Within the payload's reach every cell is either
// cleared here or exempted by the kernel, never neither and never both: the
// cells inside the attested support patch are the counter, and they stay.

TEST(PayloadClearing, TheAttestedSupportSurfaceSurvivesTheClearing) {
  // The defect, and the fix, in one assertion set. The payload's own silhouette
  // goes; the counter it is resting on stays — with its cells still occupied,
  // still describing the counter, and still available to the kernel's witness
  // as the evidence that the payload has not left its support.
  auto grid = resting_grid();
  ASSERT_EQ(occupied_cells(grid), 29U) << "12 payload cells + 16 counter cells + 1 obstacle";

  const std::size_t cleared = clear_with(grid, placed(resting_payload()));

  EXPECT_EQ(cleared, 12U) << "the payload's own silhouette, and only that";
  for (const double x : kFootprintX) {
    for (const double y : kFootprintY) {
      EXPECT_NE(grid.occupancy[cell_index(grid, x, y, kSupportLayerZ)], 0)
          << "counter cell (" << x << ", " << y << ") must survive the clearing";
    }
  }
  for (const double y : kFootprintY) {
    for (const double z : kResidueLayerZ) {
      EXPECT_EQ(grid.occupancy[cell_index(grid, 0.3625, y, z)], 0)
          << "payload silhouette cell (" << y << ", " << z << ") must go";
    }
  }
  EXPECT_NE(
      grid.occupancy[cell_index(grid, kObstacleCell.x(), kObstacleCell.y(), kObstacleCell.z())], 0)
      << "the real obstacle beside the payload still stops the kernel";
  EXPECT_EQ(occupied_cells(grid), 17U);
}

TEST(PayloadClearing, WithoutTheAttestationTheSupportSurfaceIsClearedAway) {
  // The counterfactual, kept as a fact: the same payload with no support
  // attestation on the wire — the honest default of a producer that cannot
  // measure support contact — clears the counter out from under itself. That is
  // the pre-partition behaviour and the mechanism of the 2026-08-14 defect. It
  // is also the correct behaviour for a payload nobody has attested support
  // for: the kernel exempts nothing there, so leaving those cells would stop
  // the robot against them.
  auto grid = resting_grid();
  WireObject unattested = resting_payload();
  unattested.support_contact_valid = false;

  const Placed p = placed(unattested);
  EXPECT_TRUE(p.patches.empty());
  EXPECT_EQ(clear_with(grid, p), 28U) << "12 payload cells AND the 16 counter cells beneath";
  for (const double x : kFootprintX) {
    for (const double y : kFootprintY) {
      EXPECT_EQ(grid.occupancy[cell_index(grid, x, y, kSupportLayerZ)], 0);
    }
  }
}

TEST(PayloadClearing, APlacePhaseWitnessIsWithheldExactlyAsAPickPhaseOneIs) {
  // ADR-0097 rollout item 4, verified rather than assumed. The place-phase
  // witness rides the SAME `AttachedCollisionObject.support_contact` field as
  // the pick witness, and `place_attached_object` / `support_patch_withholds`
  // read only the geometry on that field — never `support_id`, never
  // `evidence_kind`, never anything that could tell the two phases apart.
  //
  // So the claim to prove is an identity, not an approximation: the same
  // payload, the same pose, the same map, attested against the cabinet shelf it
  // is being PLACED on instead of the counter it was PICKED from, must produce
  // a bit-identical partition. If it did not, the place witness would arrive at
  // the kernel with its supporting occupancy cleared out from under it — the
  // exact 2026-08-14 defect this partition exists to fix, re-opened for the
  // phase the fix was not written against.
  WireObject place = resting_payload();
  place.support_contact.support_id = "sim:cab_1_left_group_main";

  const auto before = resting_grid();
  auto pick_cleared = before;
  clear_with(pick_cleared, placed(resting_payload()));
  auto place_cleared = before;
  const std::size_t cleared = clear_with(place_cleared, placed(place));

  EXPECT_EQ(cleared, 12U) << "the payload's own silhouette, and only that";
  EXPECT_EQ(place_cleared.occupancy, pick_cleared.occupancy)
      << "the partition must not be able to tell a place witness from a pick witness";
  for (const double x : kFootprintX) {
    for (const double y : kFootprintY) {
      EXPECT_NE(place_cleared.occupancy[cell_index(place_cleared, x, y, kSupportLayerZ)], 0)
          << "the declared place surface must survive for the kernel to exempt it";
    }
  }
  // And the obstacle beside the payload — the cabinet wall the payload is NOT
  // attested against — is untouched by either phase, so it still stops.
  EXPECT_NE(place_cleared.occupancy[cell_index(place_cleared, kObstacleCell.x(), kObstacleCell.y(),
                                               kObstacleCell.z())],
            0);
}

TEST(PayloadClearing, WithholdingOnlyEverPutsOccupancyBack) {
  // The conservatism argument, mechanised. Withholding can only ever SKIP a
  // clear, so the cells the partitioned clearing removes are a strict subset of
  // the cells the un-partitioned one removed: this change hands occupancy back
  // to the map and never takes any away, from the kernel or from Nav2.
  const auto before = resting_grid();

  auto partitioned = before;
  clear_with(partitioned, placed(resting_payload()));
  WireObject unattested = resting_payload();
  unattested.support_contact_valid = false;
  auto unpartitioned = before;
  clear_with(unpartitioned, placed(unattested));

  const auto with_patch = cleared_indices(before, partitioned);
  const auto without_patch = cleared_indices(before, unpartitioned);
  EXPECT_LT(with_patch.size(), without_patch.size());
  for (const std::size_t idx : with_patch) {
    EXPECT_NE(std::find(without_patch.begin(), without_patch.end(), idx), without_patch.end())
        << "cell " << idx << " is cleared with the patch but not without it";
  }

  // …and the other half of the same claim, which is the safety-relevant one:
  // every cell the partition WITHHELD (cleared without the patch, kept with it)
  // is a cell the kernel's own exemption predicate accepts at ZERO slack. A
  // withheld cell outside that set is a cell nothing exempts and the payload
  // still reaches — the class that tripped the E-stop on the 2026-08-14 round-5
  // run — so the property is asserted against the kernel predicate itself
  // rather than against the bridge's paraphrase of it.
  const Placed p = placed(resting_payload());
  ASSERT_EQ(p.patches.size(), 1U);
  std::size_t withheld = 0;
  for (const std::size_t idx : without_patch) {
    if (std::find(with_patch.begin(), with_patch.end(), idx) != with_patch.end()) {
      continue;
    }
    ++withheld;
    const tf2::Vector3 center = center_of_index(before, idx);
    EXPECT_TRUE(kernel_support_contact_exempts(p.patches[0], center, before.resolution, 0.0))
        << "withheld cell " << idx << " at (" << center.x() << ", " << center.y() << ", "
        << center.z() << ") is not exempt in the kernel";
    EXPECT_TRUE(kernel_support_contact_exempts(p.patches[0], center, before.resolution,
                                               kKernelContactTolerance));
  }
  EXPECT_EQ(withheld, 16U) << "the counter's 16 cells under the payload, and only those";
}

TEST(PayloadClearing, WithholdingIsTheKernelsExemptionPredicateAtZeroSlack) {
  // The mirror, probed rather than asserted in prose. Across the along-normal
  // axis (the axis the round-5 field cell lives on), across the lateral radius,
  // and across attested geometries the kernel would accept, the bridge's
  // withhold predicate agrees CELL FOR CELL with the kernel's exemption
  // predicate at zero slack — and every cell it withholds is still exempt once
  // the kernel adds its 1 mm of physical tolerance, which is the containment
  // the partition rests on.
  const std::vector<tf2::Vector3> normals{
      tf2::Vector3(0.0, 0.0, 1.0),  tf2::Vector3(0.0, 0.0, -1.0), tf2::Vector3(0.0, -1.0, 0.0),
      tf2::Vector3(1.0, 1.0, 1.0),  tf2::Vector3(0.2, -0.3, 0.9), tf2::Vector3(-0.6, 0.1, 0.8),
      tf2::Vector3(1.0, -1.0, 0.05)};
  const std::vector<double> penetrations{0.0, 0.00137, 0.005, kKernelMaxPenetration};
  const std::vector<double> radii{0.01, kAttestedPatchRadius, 0.2, kKernelMaxPatchRadius};
  const tf2::Vector3 origin(0.40, 0.0, kSupportFaceZ);

  std::size_t agreed = 0;
  std::size_t withheld = 0;
  for (const auto& n : normals) {
    for (const double penetration : penetrations) {
      for (const double radius : radii) {
        const bridge::SupportPatch patch = grid_frame_patch(origin, n, radius, penetration);
        // Probe on the patch's own axes: `s` along the normal (the bound in
        // question), `t` across it.
        tf2::Vector3 across = patch.normal.cross(tf2::Vector3(0.0, 0.0, 1.0));
        if (across.length2() < 1e-9) {
          across = patch.normal.cross(tf2::Vector3(1.0, 0.0, 0.0));
        }
        across = across.normalized();
        for (int si = -12; si <= 12; ++si) {
          const double s = 0.005 * si;  // ±60 mm along the normal, 5 mm steps
          for (int ti = 0; ti <= 12; ++ti) {
            const double t = 0.05 * ti;  // 0 … 600 mm across it
            const tf2::Vector3 center = origin + patch.normal * s + across * t;
            const bool mine = bridge::support_patch_withholds(patch, center, kResolution);
            const bool theirs = kernel_support_contact_exempts(patch, center, kResolution, 0.0);
            EXPECT_EQ(mine, theirs) << "s=" << s << " t=" << t << " radius=" << radius
                                    << " penetration=" << penetration;
            agreed += (mine == theirs) ? 1U : 0U;
            if (mine) {
              ++withheld;
              EXPECT_TRUE(kernel_support_contact_exempts(patch, center, kResolution,
                                                         kKernelContactTolerance))
                  << "withheld but not exempt with the kernel's own slack: s=" << s;
            }
          }
        }
      }
    }
  }
  EXPECT_EQ(agreed, normals.size() * penetrations.size() * radii.size() * 25U * 13U);
  EXPECT_GT(withheld, 0U) << "a probe that withholds nothing proves nothing";
}

TEST(PayloadClearing, NoAttestationTheKernelAcceptsWithholdsTheRoundFiveFieldCell) {
  // The round-5 cell, as a bound rather than as one example: at 25 mm cells the
  // withhold ceiling is at most half·(|n.x|+|n.y|+|n.z|) + attested depth =
  // 21.65 mm (a cube diagonal normal) + 10 mm (the kernel's
  // `support_witness_max_penetration_m`) = 31.65 mm above the attested plane.
  // +35.8 mm is outside it for EVERY attestation the kernel would accept, so no
  // wire message can make the bridge withhold that cell — which is what makes
  // "withheld ⊂ exempt" a property of the predicate rather than of the fixture.
  const tf2::Vector3 origin(0.40, 0.0, kSupportFaceZ);
  const double ceiling = 0.5 * kResolution * 1.7320508075688772 + kKernelMaxPenetration;
  ASSERT_LT(ceiling, kFieldResidueOffset) << "the field offset must be outside the widest slab";

  for (const auto& n : {tf2::Vector3(0.0, 0.0, 1.0), tf2::Vector3(1.0, 1.0, 1.0),
                        tf2::Vector3(0.3, -0.9, 0.4), tf2::Vector3(-1.0, 0.2, 0.7)}) {
    for (const double penetration : {0.0, 0.00137, kKernelMaxPenetration}) {
      const bridge::SupportPatch patch =
          grid_frame_patch(origin, n, kKernelMaxPatchRadius, penetration);
      const tf2::Vector3 field_cell = origin + patch.normal * kFieldResidueOffset;
      EXPECT_FALSE(bridge::support_patch_withholds(patch, field_cell, kResolution))
          << "+35.8 mm along the outward normal is payload-side solid, not support";
      EXPECT_FALSE(
          kernel_support_contact_exempts(patch, field_cell, kResolution, kKernelContactTolerance))
          << "…and the kernel does not exempt it either: the partition holds";
    }
  }
}

TEST(PayloadClearing, TheRoundFiveResidueCellClearsThoughItIsInsideThePatchCylinder) {
  // The same thing on a real grid, because a predicate that is right in
  // isolation and a clearing that never asks it are indistinguishable in the
  // field. The payload rests on its attested plane 35.8 mm below an occupancy
  // cell-centre layer, so three cell layers of its own silhouette sit above the
  // plane, all of them deep inside the attested patch cylinder (53 mm of lateral
  // offset against a 60 + 21.65 mm reach). The support layer stays; everything
  // above the slab goes.
  auto grid = lowered(deploy_sim_tree());
  const std::size_t support_cell = cell_index(grid, 0.40, 0.0, kFieldPlaneZ + 0.0108);
  const std::size_t field_cell = cell_index(grid, 0.40, 0.0, kSupportLayerZ);
  const std::size_t corner_cell = cell_index(grid, 0.4375, 0.0375, kSupportLayerZ);
  grid.occupancy[support_cell] = 1;
  grid.occupancy[field_cell] = 1;
  grid.occupancy[corner_cell] = 1;

  const Placed p = placed(field_round5_payload());
  ASSERT_EQ(p.patches.size(), 1U);
  EXPECT_NEAR(p.patches[0].point.z(), kFieldPlaneZ, 1e-12);
  const tf2::Vector3 field_center = center_of_index(grid, field_cell);
  EXPECT_NEAR(field_center.z() - kFieldPlaneZ, kFieldResidueOffset, 1e-12);
  // Inside the cylinder laterally — the only bound left is the along-normal one.
  EXPECT_LT((center_of_index(grid, corner_cell) - p.patches[0].point).length(),
            kAttestedPatchRadius + kCircumradius + kFieldResidueOffset);

  EXPECT_EQ(clear_with(grid, p), 2U);
  EXPECT_EQ(grid.occupancy[field_cell], 0)
      << "+35.8 mm above the attested plane: payload, and the kernel exempts nothing there";
  EXPECT_EQ(grid.occupancy[corner_cell], 0) << "…including at the edge of the patch cylinder";
  EXPECT_NE(grid.occupancy[support_cell], 0) << "+10.8 mm: inside the slab, the counter's own cell";
}

TEST(PayloadClearing, WithholdingStopsAtTheProjectedCubeHalfWidth) {
  // The depth bound, to the millimetre, and it is the kernel's: the cube's
  // half-width projected on the support normal plus the ATTESTED PHYSICAL
  // depth. A cell centre a millimetre below that ceiling is the support face
  // seen through the lattice and is withheld; one a millimetre above it is
  // solid genuinely higher than the attested face — the +32 mm class the
  // acceptance round tripped on — and clears.
  const Placed p = placed(resting_payload());
  ASSERT_EQ(p.patches.size(), 1U);
  const auto& patch = p.patches[0];
  EXPECT_NEAR(patch.point.z(), kSupportFaceZ, 1e-12);
  EXPECT_NEAR(patch.normal.z(), 1.0, 1e-12);

  const tf2::Vector3 below(0.40, 0.0, kSupportFaceZ + kWithholdCeiling - 0.001);
  const tf2::Vector3 above(0.40, 0.0, kSupportFaceZ + kWithholdCeiling + 0.001);
  EXPECT_TRUE(bridge::support_patch_withholds(patch, below, kResolution));
  EXPECT_FALSE(bridge::support_patch_withholds(patch, above, kResolution));
  // And a cell deep inside the counter body is withheld without a lower bound:
  // that is furniture the map is supposed to carry.
  EXPECT_TRUE(bridge::support_patch_withholds(patch, tf2::Vector3(0.40, 0.0, kSupportFaceZ - 0.20),
                                              kResolution));
}

TEST(PayloadClearing, WithholdingIsBoundedLaterallyByTheAttestedPatch) {
  // A witness licenses one contact, not a region. Past the attested patch
  // radius (padded by the cell circumradius, the map's own slop) the surface is
  // one nobody attested, and it clears exactly as any other payload cell does.
  const Placed p = placed(resting_payload());
  ASSERT_EQ(p.patches.size(), 1U);
  const double edge = kAttestedPatchRadius + kCircumradius;

  EXPECT_TRUE(bridge::support_patch_withholds(
      p.patches[0], tf2::Vector3(0.40 + edge - 0.001, 0.0, kSupportFaceZ), kResolution));
  EXPECT_FALSE(bridge::support_patch_withholds(
      p.patches[0], tf2::Vector3(0.40 + edge + 0.001, 0.0, kSupportFaceZ), kResolution));
}

TEST(PayloadClearing, TheLiftLeavesTheCounterAloneAndWithholdsNothing) {
  // Genuine separation. The attestation is in the object frame, so the patch
  // rides up with the payload — and the counter cells it used to protect are
  // now out of the payload's reach anyway. Nothing is cleared, nothing is
  // withheld, and the counter stands in the map exactly as it did: the kernel
  // then finds no exempt cell still touching the payload and lets the witness
  // die, which is the separation it is supposed to detect.
  auto grid = resting_grid();
  WireObject lifted = resting_payload();
  lifted.pose_in_link.position.z += 0.10;

  const Placed p = placed(lifted);
  EXPECT_EQ(clear_with(grid, p), 0U);
  EXPECT_EQ(occupied_cells(grid), 29U);

  // …and the patch had nothing to do with that: without it the result is the
  // same, so the lift is detected by geometry, not by the withholding.
  WireObject lifted_unattested = lifted;
  lifted_unattested.support_contact_valid = false;
  auto twin = resting_grid();
  EXPECT_EQ(clear_with(twin, placed(lifted_unattested)), 0U);
  EXPECT_EQ(twin.occupancy, grid.occupancy);
}

TEST(PayloadClearing, TheAttestedPlaneIsCarriedInTheObjectFrame) {
  // The witness geometry is stated in the attached object's own frame and must
  // be TRANSFORMED, not assumed to point at base +z. Rolled 90° about base x,
  // the payload hangs off a vertical wall: the attested normal becomes base −y,
  // the support half-space becomes y >= the contact plane, and the withheld
  // cells move with it.
  WireObject rolled = resting_payload();
  rolled.pose_in_link.position.x = 0.40 - kAttachLinkOrigin.x();
  rolled.pose_in_link.position.y = 0.0;
  rolled.pose_in_link.position.z = 0.40 - kAttachLinkOrigin.z();
  tf2::Quaternion q;
  q.setRPY(1.5707963267948966, 0.0, 0.0);
  rolled.pose_in_link.orientation.x = q.x();
  rolled.pose_in_link.orientation.y = q.y();
  rolled.pose_in_link.orientation.z = q.z();
  rolled.pose_in_link.orientation.w = q.w();

  const Placed p = placed(rolled);
  ASSERT_EQ(p.patches.size(), 1U);
  // Object +z -> base −y, so the contact point moves to +y of the object origin
  // and the normal points down −y.
  EXPECT_NEAR(p.patches[0].point.y(), kRestingHalfExtents.z(), 1e-12);
  EXPECT_NEAR(p.patches[0].point.z(), 0.40, 1e-12);
  EXPECT_NEAR(p.patches[0].normal.y(), -1.0, 1e-12);

  auto grid = lowered(deploy_sim_tree());
  const std::size_t wall_side = cell_index(grid, 0.4125, 0.0375, 0.4125);
  const std::size_t payload_side = cell_index(grid, 0.4125, 0.0125, 0.4125);
  grid.occupancy[wall_side] = 1;
  grid.occupancy[payload_side] = 1;

  EXPECT_EQ(clear_with(grid, p), 1U);
  EXPECT_NE(grid.occupancy[wall_side], 0) << "inside the attested support half-space";
  EXPECT_EQ(grid.occupancy[payload_side], 0) << "37.5 mm clear of it: the payload's own cell";
}

TEST(PayloadClearing, OneObjectsAttestationGuardsEveryObjectsClearing) {
  // The one place the withheld set is NOT a subset of what the kernel exempts,
  // stated as a fact so it cannot be lost: withholding here is per-MESSAGE
  // (every patch guards every object's clearing) while the kernel's exemption is
  // per-OBJECT — `check_attached_voxel_collision` only ever tests object i's
  // cells against object i's own witness. So with two payloads attached, a cell
  // inside the FIRST object's attested slab but reached only by the SECOND
  // object's volume stays in the map and is not exempt for the object that
  // reaches it. It is deliberate (a second payload's volume must not erase the
  // first one's support evidence) and it is conservative in the map — occupancy
  // stays — but it is a stop the kernel can still take, so it is not covered by
  // the "no withheld cell can be the cell that stops the robot" claim.
  auto grid = lowered(deploy_sim_tree());
  // 62.5 mm off the attested patch axis: inside the patch cylinder (60 + 21.65
  // mm) and 3.2 mm above the plane, but 22.5 mm clear of the resting payload's
  // own face, so the resting payload's clearing never reaches it.
  const std::size_t shared = cell_index(grid, 0.4625, 0.0125, kSupportLayerZ);
  grid.occupancy[shared] = 1;

  WireObject second;
  second.object_id = "obj_cup";
  second.attach_link = "panda_hand";
  second.pose_in_link.position.x = 0.4625 - kAttachLinkOrigin.x();
  second.pose_in_link.position.y = 0.0125 - kAttachLinkOrigin.y();
  second.pose_in_link.position.z = kSupportLayerZ - kAttachLinkOrigin.z();
  second.pose_in_link.orientation.w = 1.0;
  Primitive cup;
  cup.shape_type = Primitive::SHAPE_SPHERE;
  cup.shape_dimensions = {0.02};
  cup.pose_in_object.orientation.w = 1.0;
  second.primitives = {cup};

  // The second payload alone clears the cell: its volume explains it.
  auto alone = grid;
  const Placed only_second = placed(second);
  ASSERT_TRUE(only_second.patches.empty());
  EXPECT_EQ(clear_with(alone, only_second), 1U);
  EXPECT_EQ(alone.occupancy[shared], 0);

  // With the resting payload's attestation on the same message, it does not.
  std::vector<bridge::PayloadPrimitive> primitives;
  std::vector<bridge::SupportPatch> patches;
  ASSERT_TRUE(bridge::place_attached_object(resting_payload(), base_from_attach_link(), primitives,
                                            patches));
  ASSERT_TRUE(bridge::place_attached_object(second, base_from_attach_link(), primitives, patches));
  ASSERT_EQ(patches.size(), 1U);
  EXPECT_EQ(bridge::clear_attached_payload_cells(grid, primitives, patches, 0.0), 0U);
  EXPECT_NE(grid.occupancy[shared], 0);
  EXPECT_TRUE(kernel_support_contact_exempts(patches[0], center_of_index(grid, shared),
                                             grid.resolution, 0.0))
      << "exempt for the object that attested it — and for that object only";
}

TEST(PayloadClearing, AMalformedAttestationClearsNothingAtAll) {
  // Fail-closed, and closed on the WHOLE object — the kernel's own
  // `ingest_attached_objects` rule. Downgrading a malformed witness to "no
  // witness" would be the worst of both: the bridge would clear the support
  // surface away while the kernel refused the attachment set outright.
  std::vector<bridge::PayloadPrimitive> out;
  std::vector<bridge::SupportPatch> patches;

  WireObject degenerate_normal = resting_payload();
  degenerate_normal.support_contact.contact_normal_in_object.z = 0.0;
  EXPECT_FALSE(
      bridge::place_attached_object(degenerate_normal, base_from_attach_link(), out, patches));
  EXPECT_TRUE(out.empty());
  EXPECT_TRUE(patches.empty());

  WireObject no_patch = resting_payload();
  no_patch.support_contact.patch_radius_m = 0.0;
  EXPECT_FALSE(bridge::place_attached_object(no_patch, base_from_attach_link(), out, patches));
  EXPECT_TRUE(out.empty());

  WireObject negative_depth = resting_payload();
  negative_depth.support_contact.max_penetration_m = -0.001;
  EXPECT_FALSE(
      bridge::place_attached_object(negative_depth, base_from_attach_link(), out, patches));
  EXPECT_TRUE(out.empty());

  WireObject nan_point = resting_payload();
  nan_point.support_contact.contact_point_in_object.x = std::nan("");
  EXPECT_FALSE(bridge::place_attached_object(nan_point, base_from_attach_link(), out, patches));
  EXPECT_TRUE(out.empty());

  // An un-normalised normal is fine — it is a direction, and it gets normalised.
  WireObject long_normal = resting_payload();
  long_normal.support_contact.contact_normal_in_object.z = 4.0;
  EXPECT_TRUE(bridge::place_attached_object(long_normal, base_from_attach_link(), out, patches));
  ASSERT_EQ(patches.size(), 1U);
  EXPECT_NEAR(patches[0].normal.z(), 1.0, 1e-12);
}
