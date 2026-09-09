// SPDX-License-Identifier: Apache-2.0
// The grasped payload must leave world occupancy — and only the payload.
//
// `openral_msgs/AttachedCollisionObject` contract: "the same object must be
// absent from world occupancy while attached." Unenforced before this: on the
// 2026-08-14 acceptance run the arm stopped 1.8 mm off the held object's
// surface, +32 mm above the attested support plane (witness correctly
// refused to exempt it).
//
// The depth self-filter's transparency (8149344 + f02fe7d) only clears cells
// a ray still crosses; an occluded/out-of-frustum/between-rays cell is never
// touched (`OccupancyPersistence.AConfirmedVoxelSurvivesWhenNoRayEverCrossesIt`,
// test_occupancy_persistence.cpp). So the transition needs explicit removal —
// pinned here on a real `octomap::OcTree` seeded through real rays with the
// deploy-sim parameters.

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

// The deploy-sim octomap_server parameters (deploy_e2e.launch.py).
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

// The covered volume around the payload. The grid lands on the octree's own
// lattice by construction, so this only has to say WHERE to cover, not how
// to phase it. Radius covers the box this used to name explicitly
// (min (0.30, -0.15, 0.25), 12 cells of 25 mm on a side).
//
// These tests drive `clear_attached_payload_cells`, not the rasterizer, and
// every one uses an identity `base_to_octomap`.
bridge::GridSpec grid_spec() {
  bridge::GridSpec spec;
  spec.center[0] = 0.45;
  spec.center[1] = 0.00;
  spec.center[2] = 0.40;
  spec.radius = 0.15 * std::sqrt(3.0);
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
// A payload RESTING on a counter — the situation the two mechanisms collide
// in. Counter top face z = 0.3593 (base frame), lattice phases the surface
// cell's centre 3.2 mm above it (2026-08-13 baguette run's exact phase, ~1 mm
// physical contact reading as 15.7 mm cube penetration). Payload's bottom
// cell layer IS the counter's top cell layer.

constexpr double kSupportFaceZ = 0.3593;
const tf2::Vector3 kRestingHalfExtents(0.04, 0.04, 0.05);
const tf2::Vector3 kRestingCenter(0.40, 0.0, kSupportFaceZ + kRestingHalfExtents.z());
// What World State attests: a 60 mm patch about the contact point and the
// deepest of the six real MuJoCo counter-payload contacts of that run.
constexpr double kAttestedPatchRadius = 0.06;
constexpr double kAttestedPenetration = 0.00137;
// The bound the attestation buys at 25 mm cells: the cube's half-width
// projected on the +z normal (12.5 mm), plus that physical depth, plus the one
// voxel of co-planar headroom the kernel's `support_contact_exempts` gained on
// 2026-08-15 (hazard log Entry 012, "Calibration 2026-08-15") and this
// predicate mirrors in lockstep — 38.87 mm.
const double kWithholdCeiling = 0.5 * kResolution + kAttestedPenetration + kResolution;

// The four cell-centre coordinates the payload's 80 mm footprint spans on each
// lateral axis (the next centre out is 22.5 mm clear of the box face, past the
// circumradius, so it is never in reach).
const std::vector<double> kFootprintX{0.3625, 0.3875, 0.4125, 0.4375};
const std::vector<double> kFootprintY{-0.0375, -0.0125, 0.0125, 0.0375};
// The counter's top cell layer (centre 3.2 mm above the attested face) and the
// three payload-silhouette layers above it (+28.2, +53.2, +78.2 mm). Since the
// 2026-08-15 height calibration the +28.2 mm layer is INSIDE the withhold slab
// (ceiling 38.87 mm) — it is the co-planar band the calibration exists for — so
// only the top two layers are the clearing's.
constexpr double kSupportLayerZ = 0.3625;
const std::vector<double> kResidueLayerZ{0.3875, 0.4125, 0.4375};
// The silhouette layers that still clear, and the one now withheld with the
// counter.
const std::vector<double> kClearedResidueLayerZ{0.4125, 0.4375};
constexpr double kCoplanarBandLayerZ = 0.3875;

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
// `support_contact_exempts`, transcribed term for term from
// `cpp/openral_safety_kernel/src/collision.cpp` (Layer 2 must not link
// Layer 6, so the mirror is deliberate — `docs/methods/14-duplication-watch.md`
// item 8). Specification that `support_patch_withholds` implements at
// `slack = 0`; tests below evaluate both on the same cells.
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
  // Fourth term is the kernel's one voxel of co-planar headroom (hazard log
  // Entry 012, "Calibration 2026-08-15") — must move when the kernel moves.
  return height <= normal_half_width + patch.max_penetration + slack + resolution;
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
// Field run (2026-08-14-round5/baguette/): a cell +35.8 mm along the outward
// support normal, laterally inside the attested patch cylinder, reported as
// withheld while the kernel refused to exempt it (bound: cube projected
// half-width 12.5 mm + 1.37 mm attested physical depth). A cylinder-shaped
// withhold (vs. the support half-space slab) resurrects the residue 21ecf82
// eliminated.
constexpr double kFieldResidueOffset = 0.0358;

// A cell above the widened slab (21.65 + 10 + 25 = 56.65 mm for the widest
// attestation the kernel would accept). +60 mm plays the role +35.8 mm played
// before the 2026-08-15 co-planar-headroom calibration.
constexpr double kAboveWidenedSlabOffset = 0.060;

// Resting payload lowered so its attested support plane sits
// `kFieldResidueOffset` below an occupancy cell-centre layer: plane at
// 0.3625 − 0.0358 = 0.3267 m.
const double kFieldPlaneZ = kSupportLayerZ - kFieldResidueOffset;
const tf2::Vector3 kFieldCenter(0.40, 0.0, kFieldPlaneZ + kRestingHalfExtents.z());

// ── the 2026-08-14 round-6 attach-transition residue ─────────────────────────
//
// Round-5 forensics: baguette r1's E-stop cell `voxel_91633` sat 22.13 mm from
// the payload's own primitive surface at resolution 0.025 — 0.48 mm outside
// the clearing's reach (0.5·res·√3 = 21.65 mm). Stale pre-attach silhouette:
// fit error + lattice quantization leaves residue past one circumradius. Fix
// widens the one sweep that runs at the attach transition; steady-state reach
// unchanged.
constexpr double kFieldRound6Distance = 0.02213;
// One voxel, the `attach_sweep_padding_m` default.
constexpr double kAttachSweepPadding = kResolution;

// A payload whose surface is exactly `distance` from the centre of the cell at
// (0.40, 0.0, 0.35): a sphere directly below it, offset by `distance` plus its
// own radius, so the field number is reproduced to the micron on a real cell of
// a real grid. Same construction the 1.8 mm stop-cell test above uses.
bridge::PayloadPrimitive payload_at_distance(const openral_msgs::msg::OccupancyVoxels& grid,
                                             double distance) {
  bridge::PayloadPrimitive sphere;
  sphere.shape_type = Primitive::SHAPE_SPHERE;
  sphere.radius = 0.02;
  const tf2::Vector3 center = cell_center(grid, 0.40, 0.0, 0.35);
  sphere.pose = tf2::Transform(tf2::Quaternion::getIdentity(),
                               center - tf2::Vector3(0.0, 0.0, distance + sphere.radius));
  return sphere;
}

// The field run's own timeline: the E-stop fired at 1786710431.16, 2.78 s after
// the attach sweep at 1786710428.38 — ~28 published grids at the bridge's 10 Hz
// — with the payload still sitting on its stale silhouette. A one-shot sweep
// clears the residue from ONE grid and the octree hands it straight back on the
// next, so the window has to outlive that.
constexpr int kFieldGridsBeforeEstop = 28;

// The single object the window tests carry, and the node's own per-object
// decision in the one call the node spends on it: the ledger answers and
// records in one step, and `attach_transition_padding` turns the answer into a
// reach. `position` is in the OctoMap frame, which is where the node measures.
const bridge::AttachedObjectRevision kCarried{"sim:obj_baguette", 7};

double padding_for(bridge::AttachSweepLedger& ledger, const bridge::AttachedObjectRevision& key,
                   const tf2::Vector3& position) {
  const std::vector<std::uint8_t> open =
      ledger.sweep({bridge::AttachSweepObservation{key, position}}, kAttachSweepPadding);
  return open.at(0) != 0 ? bridge::attach_transition_padding(0.0, kAttachSweepPadding) : 0.0;
}

// Where the payload sits in the OctoMap frame while it has not moved.
const tf2::Vector3 kAttachPose(1.20, 0.35, 0.90);

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
  // Clearing carries no state of its own — derived from the attachment set on
  // the wire — so the frame after the object leaves that set already
  // publishes it as an obstacle. Re-marking costs octomap's two-hit
  // confirmation latency (pinned by test_occupancy_persistence).
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
// Clearing and the kernel's support-contact witness each correct alone,
// destroy each other combined: the witness stays alive only while some
// occupied cell it would exempt still touches the payload
// (`update_support_contact_witnesses`), and that cell is the counter's own
// top layer, inside the resting payload's volume. Observed 2/2 on 2026-08-14
// (baguette+counter, cup+island): `support_witness_separated live=0x0
// was=0x1` fires 2.7 s after the clearing removes the support cells, ground
// truth still touching at +0.000 mm, contact re-trips unexempted.
//
// Fix: a partition. Within the payload's reach every cell is either cleared
// here or exempted by the kernel — never neither, never both.

TEST(PayloadClearing, TheAttestedSupportSurfaceSurvivesTheClearing) {
  // Payload's own silhouette goes; the counter it rests on stays occupied,
  // available to the kernel's witness.
  auto grid = resting_grid();
  ASSERT_EQ(occupied_cells(grid), 29U) << "12 payload cells + 16 counter cells + 1 obstacle";

  const std::size_t cleared = clear_with(grid, placed(resting_payload()));

  EXPECT_EQ(cleared, 8U) << "the payload's silhouette above the widened slab, and only that";
  for (const double x : kFootprintX) {
    for (const double y : kFootprintY) {
      EXPECT_NE(grid.occupancy[cell_index(grid, x, y, kSupportLayerZ)], 0)
          << "counter cell (" << x << ", " << y << ") must survive the clearing";
    }
  }
  for (const double y : kFootprintY) {
    for (const double z : kClearedResidueLayerZ) {
      EXPECT_EQ(grid.occupancy[cell_index(grid, 0.3625, y, z)], 0)
          << "payload silhouette cell (" << y << ", " << z << ") must go";
    }
    // +28.2 mm layer is the co-planar band the 2026-08-15 calibration widened
    // the kernel's envelope to cover: withheld here, stays in the map,
    // exempted by the kernel for the attesting object.
    EXPECT_NE(grid.occupancy[cell_index(grid, 0.3625, y, kCoplanarBandLayerZ)], 0)
        << "co-planar band cell (" << y << ") is withheld, not cleared";
  }
  EXPECT_NE(
      grid.occupancy[cell_index(grid, kObstacleCell.x(), kObstacleCell.y(), kObstacleCell.z())], 0)
      << "the real obstacle beside the payload still stops the kernel";
  EXPECT_EQ(occupied_cells(grid), 21U);
}

TEST(PayloadClearing, WithoutTheAttestationTheSupportSurfaceIsClearedAway) {
  // Same payload, no support attestation on the wire (honest default of a
  // producer that can't measure support contact): clears the counter out from
  // under itself — pre-partition behaviour, the 2026-08-14 defect mechanism,
  // and correct for an un-attested payload (kernel exempts nothing there).
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
  // ADR-0097 rollout item 4. Place-phase witness rides the same
  // `AttachedCollisionObject.support_contact` field as the pick witness;
  // `place_attached_object` / `support_patch_withholds` read only that
  // field's geometry, never `support_id`/`evidence_kind`. Claim: same
  // payload/pose/map attested against a place target must produce a
  // bit-identical partition to the pick case — else the 2026-08-14 defect
  // reopens for the place phase.
  WireObject place = resting_payload();
  place.support_contact.support_id = "sim:cab_1_left_group_main";

  const auto before = resting_grid();
  auto pick_cleared = before;
  clear_with(pick_cleared, placed(resting_payload()));
  auto place_cleared = before;
  const std::size_t cleared = clear_with(place_cleared, placed(place));

  EXPECT_EQ(cleared, 8U) << "the payload's silhouette above the widened slab, and only that";
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
  // Withholding can only ever skip a clear: cells the partitioned clearing
  // removes are a strict subset of what the un-partitioned one removed.
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

  // Safety-relevant half: every cell the partition WITHHELD (cleared without
  // the patch, kept with it) is exempt under the kernel's own predicate at
  // zero slack — asserted against the kernel predicate itself, not the
  // bridge's paraphrase.
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
  EXPECT_EQ(withheld, 20U) << "the counter's 16 cells, plus the 4 co-planar-band cells the "
                              "2026-08-15 calibration moved from cleared to withheld";
}

TEST(PayloadClearing, WithholdingIsTheKernelsExemptionPredicateAtZeroSlack) {
  // Bridge's withhold predicate agrees cell for cell with the kernel's
  // exemption predicate at zero slack, over the along-normal axis, the
  // lateral radius, and attested geometries the kernel would accept; every
  // withheld cell is still exempt once the kernel adds its 1 mm tolerance.
  //
  // Along-normal probe reaches ±90 mm (not ±60) since the 2026-08-15 height
  // calibration (hazard log Entry 012): widest slab the kernel accepts now
  // reaches 21.65 + 10 + 25 = 56.65 mm above the plane.
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
        for (int si = -18; si <= 18; ++si) {
          const double s = 0.005 * si;  // ±90 mm along the normal, 5 mm steps
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
  EXPECT_EQ(agreed, normals.size() * penetrations.size() * radii.size() * 37U * 13U);
  EXPECT_GT(withheld, 0U) << "a probe that withholds nothing proves nothing";
}

TEST(PayloadClearing, NoAttestationTheKernelAcceptsWithholdsACellAboveTheWidenedSlab) {
  // Ceiling as a bound, not one example: at 25 mm cells, withhold ceiling ≤
  // half·(|n.x|+|n.y|+|n.z|) + attested depth + one voxel = 21.65 mm (cube
  // diagonal normal) + 10 mm (`support_witness_max_penetration_m`) + 25 mm
  // (2026-08-15 co-planar headroom) = 56.65 mm above the plane.
  // `kAboveWidenedSlabOffset` is outside it for every attestation the kernel
  // accepts. Before the calibration the ceiling was 31.65 mm and round-5's
  // +35.8 mm cell was outside it; now inside — see
  // `TheRoundFiveResidueCellIsWithheldInTheCoplanarBand`.
  const tf2::Vector3 origin(0.40, 0.0, kSupportFaceZ);
  const double ceiling =
      0.5 * kResolution * 1.7320508075688772 + kKernelMaxPenetration + kResolution;
  ASSERT_NEAR(ceiling, 0.05665, 1e-5);
  ASSERT_LT(ceiling, kAboveWidenedSlabOffset) << "the probe must be outside the widest slab";
  ASSERT_GT(ceiling, kFieldResidueOffset) << "…and the round-5 cell must be inside it now";

  for (const auto& n : {tf2::Vector3(0.0, 0.0, 1.0), tf2::Vector3(1.0, 1.0, 1.0),
                        tf2::Vector3(0.3, -0.9, 0.4), tf2::Vector3(-1.0, 0.2, 0.7)}) {
    for (const double penetration : {0.0, 0.00137, kKernelMaxPenetration}) {
      const bridge::SupportPatch patch =
          grid_frame_patch(origin, n, kKernelMaxPatchRadius, penetration);
      const tf2::Vector3 above = origin + patch.normal * kAboveWidenedSlabOffset;
      EXPECT_FALSE(bridge::support_patch_withholds(patch, above, kResolution))
          << "+60 mm along the outward normal is payload-side solid, not support";
      EXPECT_FALSE(
          kernel_support_contact_exempts(patch, above, kResolution, kKernelContactTolerance))
          << "…and the kernel does not exempt it either: the partition holds";
    }
  }
}

TEST(PayloadClearing, TheRoundFiveResidueCellIsWithheldInTheCoplanarBand) {
  // Same predicate on a real grid. Payload rests on its attested plane
  // 35.8 mm below an occupancy cell-centre layer; three silhouette layers
  // above the plane, all inside the attested patch cylinder (53 mm lateral
  // offset vs. 60 + 21.65 mm reach).
  //
  // Since the 2026-08-15 calibration the slab ceiling for this attestation is
  // 12.5 + 1.37 + 25 = 38.87 mm, so the +35.8 mm layer round-5 reported is
  // now inside the band and withheld, not cleared. The layer above it
  // (+60.8 mm) still clears.
  auto grid = lowered(deploy_sim_tree());
  const std::size_t support_cell = cell_index(grid, 0.40, 0.0, kFieldPlaneZ + 0.0108);
  const std::size_t field_cell = cell_index(grid, 0.40, 0.0, kSupportLayerZ);
  const std::size_t corner_cell = cell_index(grid, 0.4375, 0.0375, kSupportLayerZ);
  const std::size_t above_cell = cell_index(grid, 0.40, 0.0, kSupportLayerZ + kResolution);
  grid.occupancy[support_cell] = 1;
  grid.occupancy[field_cell] = 1;
  grid.occupancy[corner_cell] = 1;
  grid.occupancy[above_cell] = 1;

  const Placed p = placed(field_round5_payload());
  ASSERT_EQ(p.patches.size(), 1U);
  EXPECT_NEAR(p.patches[0].point.z(), kFieldPlaneZ, 1e-12);
  const tf2::Vector3 field_center = center_of_index(grid, field_cell);
  EXPECT_NEAR(field_center.z() - kFieldPlaneZ, kFieldResidueOffset, 1e-12);
  // Inside the cylinder laterally — the only bound left is the along-normal one.
  EXPECT_LT((center_of_index(grid, corner_cell) - p.patches[0].point).length(),
            kAttestedPatchRadius + kCircumradius + kFieldResidueOffset);
  ASSERT_LT(kFieldResidueOffset, kWithholdCeiling) << "+35.8 mm is inside the calibrated slab";
  ASSERT_GT(kFieldResidueOffset + kResolution, kWithholdCeiling) << "+60.8 mm is not";

  EXPECT_EQ(clear_with(grid, p), 1U);
  EXPECT_NE(grid.occupancy[field_cell], 0)
      << "+35.8 mm above the attested plane: the co-planar band, withheld and kernel-exempt";
  EXPECT_NE(grid.occupancy[corner_cell], 0) << "…including at the edge of the patch cylinder";
  EXPECT_EQ(grid.occupancy[above_cell], 0)
      << "+60.8 mm: past the calibrated slab, payload-side, and the clearing's";
  EXPECT_NE(grid.occupancy[support_cell], 0) << "+10.8 mm: inside the slab, the counter's own cell";

  // The containment the mirror exists for, on the two cells that moved.
  for (const std::size_t idx : {field_cell, corner_cell}) {
    const tf2::Vector3 center = center_of_index(grid, idx);
    EXPECT_TRUE(bridge::support_patch_withholds(p.patches[0], center, kResolution));
    EXPECT_TRUE(
        kernel_support_contact_exempts(p.patches[0], center, kResolution, kKernelContactTolerance));
  }
}

TEST(PayloadClearing, WithholdingStopsAtTheProjectedCubeHalfWidth) {
  // Height bound (the kernel's): cube half-width projected on the support
  // normal, plus attested physical depth, plus the one voxel of co-planar
  // headroom gained 2026-08-15. A millimetre below the ceiling is withheld;
  // a millimetre above clears.
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
  // Genuine separation: attestation is in the object frame, so the patch
  // rides up with the payload, and the counter cells are now out of reach.
  // Nothing cleared, nothing withheld; kernel finds no exempt cell touching
  // the payload and lets the witness die.
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
  // Witness geometry is in the object's own frame and must be transformed.
  // Rolled 90° about base x: attested normal becomes base −y, support
  // half-space becomes y >= the contact plane, withheld cells move with it.
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
  // 62.5 mm off the plane along the rolled normal, not 37.5: since the
  // 2026-08-15 calibration the slab reaches 38.87 mm, so the next cell out is
  // the first one that is unambiguously the payload's.
  const std::size_t payload_side = cell_index(grid, 0.4125, -0.0125, 0.4125);
  grid.occupancy[wall_side] = 1;
  grid.occupancy[payload_side] = 1;

  EXPECT_EQ(clear_with(grid, p), 1U);
  EXPECT_NE(grid.occupancy[wall_side], 0) << "inside the attested support half-space";
  EXPECT_EQ(grid.occupancy[payload_side], 0) << "62.5 mm clear of it: the payload's own cell";
}

TEST(PayloadClearing, OneObjectsAttestationGuardsEveryObjectsClearing) {
  // The one place the withheld set is NOT a subset of what the kernel
  // exempts: withholding is per-message (every patch guards every object's
  // clearing) while the kernel's exemption is per-object
  // (`check_attached_voxel_collision`). With two payloads attached, a cell
  // inside the first object's attested slab but reached only by the second
  // object's volume stays in the map and is not exempt for that object —
  // deliberate and conservative (occupancy stays), but still a kernel stop.
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
  // Fail-closed on the whole object (the kernel's own
  // `ingest_attached_objects` rule) — downgrading to "no witness" would clear
  // the support surface while the kernel refused the attachment set.
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

// ── the attach-transition sweep (2026-08-14 round 6) ─────────────────────────
//
// The partition above decides which cells within the reach are the payload's;
// this decides how far the reach goes, and it differs at the attach
// transition vs. while carrying. Pre-grasp cells were marked by a sensor
// seeing the real object; clearing measures against a fitted convex
// primitive — fit error + lattice quantization leaves a residue just outside
// one circumradius (`voxel_91633`, 22.13 mm out at 25 mm cells), never
// removed since no ray reaches an occluded cell
// (`OccupancyPersistence.AConfirmedVoxelSurvivesWhenNoRayEverCrossesIt`).
//
// A payload sweeps one voxel wider while still on that silhouette (a
// position, not a frame count — the grid re-rasterizes every tick, so a
// one-shot sweep loses the residue back next tick). Field E-stop fired 28
// grids after the attach.

TEST(PayloadClearing, TheRoundSixFieldCellSurvivesTheSteadyReachAndGoesOnTheAttachSweep) {
  // 22.13 mm is 0.48 mm past the 21.65 mm circumradius — steady-state
  // clearing never considered it (payload's own stale silhouette).
  EXPECT_GT(kFieldRound6Distance, kCircumradius);
  EXPECT_NEAR(kFieldRound6Distance - kCircumradius, 0.00048, 1e-5);

  auto grid = lowered(deploy_sim_tree());
  const std::size_t idx = cell_index(grid, 0.40, 0.0, 0.35);
  grid.occupancy[idx] = 1;
  const bridge::PayloadPrimitive payload = payload_at_distance(grid, kFieldRound6Distance);

  auto steady = grid;
  EXPECT_EQ(bridge::clear_attached_payload_cells(steady, {payload}, {}, 0.0), 0U);
  EXPECT_NE(steady.occupancy[idx], 0) << "0.48 mm past the circumradius: never a candidate";

  auto attach = grid;
  EXPECT_EQ(bridge::clear_attached_payload_cells(
                attach, {payload}, {}, bridge::attach_transition_padding(0.0, kAttachSweepPadding)),
            1U);
  EXPECT_EQ(attach.occupancy[idx], 0) << "the attach transition reaches one voxel further";
}

TEST(PayloadClearing, TheAttachWindowOutlivesTheFieldRunsTwentyEightGrids) {
  // E-stop fired 2.78 s after the attach sweep (~28 published grids at
  // 10 Hz), payload still on its stale silhouette. Grid re-rasterizes every
  // tick and nothing retires an occluded cell, so the residue re-supplies
  // every grid. Loop is inclusive of both ends: 29 grids (attach frame + 28
  // that elapsed before the stop).
  bridge::AttachSweepLedger ledger;
  const bridge::PayloadPrimitive payload =
      payload_at_distance(lowered(deploy_sim_tree()), kFieldRound6Distance);

  for (int frame = 0; frame <= kFieldGridsBeforeEstop; ++frame) {
    auto grid = lowered(deploy_sim_tree());
    const std::size_t idx = cell_index(grid, 0.40, 0.0, 0.35);
    grid.occupancy[idx] = 1;  // the octree hands the same cell back every tick

    // The payload has not moved: a 1 mm tremor, well inside the 25 mm window.
    const tf2::Vector3 pose = kAttachPose + tf2::Vector3(0.001 * (frame % 2), 0.0, 0.0);
    EXPECT_EQ(bridge::clear_attached_payload_cells(grid, {payload}, {},
                                                   padding_for(ledger, kCarried, pose)),
              1U)
        << "grid " << frame << " after the attach still publishes the residue";
    EXPECT_EQ(grid.occupancy[idx], 0);
  }
  EXPECT_EQ(ledger.size(), 1U) << "the memory is the live attachment set, not a growing log";
}

TEST(PayloadClearing, TheAttachWindowClosesOnceThePayloadHasMovedAVoxel) {
  // Once translated further than the sweep padding, every cell the padding
  // covered is either inside the steady reach (moved toward it) or outside
  // the padded one (moved away) — a cell 22.13 mm from the old pose is the
  // world's again.
  bridge::AttachSweepLedger ledger;
  const bridge::PayloadPrimitive at_attach_pose =
      payload_at_distance(lowered(deploy_sim_tree()), kFieldRound6Distance);

  auto opening = lowered(deploy_sim_tree());
  opening.occupancy[cell_index(opening, 0.40, 0.0, 0.35)] = 1;
  ASSERT_EQ(bridge::clear_attached_payload_cells(opening, {at_attach_pose}, {},
                                                 padding_for(ledger, kCarried, kAttachPose)),
            1U);

  // A millimetre inside the window: still open.
  auto still_open = lowered(deploy_sim_tree());
  still_open.occupancy[cell_index(still_open, 0.40, 0.0, 0.35)] = 1;
  const tf2::Vector3 nearly = kAttachPose + tf2::Vector3(0.0, kAttachSweepPadding - 0.001, 0.0);
  EXPECT_EQ(bridge::clear_attached_payload_cells(still_open, {at_attach_pose}, {},
                                                 padding_for(ledger, kCarried, nearly)),
            1U);

  // A millimetre past it: shut, and the steady reach cannot explain the cell.
  auto closed = lowered(deploy_sim_tree());
  const std::size_t idx = cell_index(closed, 0.40, 0.0, 0.35);
  closed.occupancy[idx] = 1;
  const tf2::Vector3 gone = kAttachPose + tf2::Vector3(0.0, kAttachSweepPadding + 0.001, 0.0);
  EXPECT_EQ(bridge::clear_attached_payload_cells(closed, {at_attach_pose}, {},
                                                 padding_for(ledger, kCarried, gone)),
            0U);
  EXPECT_NE(closed.occupancy[idx], 0) << "22.13 mm from where the payload WAS: the world's cell";
}

TEST(PayloadClearing, AClosedAttachWindowDoesNotReopenWhenThePayloadComesBack) {
  // Closing latches: a payload carried away and back does not get the
  // widened reach a second time — the map around its old pose is by then
  // evidence gathered while the payload was elsewhere.
  bridge::AttachSweepLedger ledger;
  const double open_reach = bridge::attach_transition_padding(0.0, kAttachSweepPadding);

  EXPECT_EQ(padding_for(ledger, kCarried, kAttachPose), open_reach);
  EXPECT_EQ(padding_for(ledger, kCarried, kAttachPose + tf2::Vector3(0.30, 0.0, 0.0)), 0.0);
  EXPECT_EQ(padding_for(ledger, kCarried, kAttachPose), 0.0) << "back home, and still shut";
}

TEST(PayloadClearing, ANewObjectRevisionOpensANewAttachWindow) {
  // `attachment_revision`: producer's own counter, bumped once per atomic
  // attachment-set change, never per frame. Re-grasp / second payload /
  // release-then-grasp each open a window; carrying one payload does not.
  bridge::AttachSweepLedger ledger;
  const double open_reach = bridge::attach_transition_padding(0.0, kAttachSweepPadding);
  const tf2::Vector3 moved = kAttachPose + tf2::Vector3(0.30, 0.0, 0.0);

  ASSERT_EQ(padding_for(ledger, kCarried, kAttachPose), open_reach);
  ASSERT_EQ(padding_for(ledger, kCarried, moved), 0.0) << "carried away: the window shut";

  // The producer publishes a new attachment record for the same payload.
  EXPECT_EQ(padding_for(ledger, {kCarried.object_id, 8}, moved), open_reach);
  // A second payload under the same revision has its own stale silhouette.
  EXPECT_EQ(padding_for(ledger, {"sim:obj_cup", 8}, moved), open_reach);

  // Release: the node sweeps an empty set, so the next grasp opens a window
  // even under a revision that has been seen before.
  EXPECT_TRUE(ledger.sweep({}, kAttachSweepPadding).empty());
  EXPECT_EQ(ledger.size(), 0U);
  EXPECT_EQ(padding_for(ledger, kCarried, moved), open_reach);
}

TEST(PayloadClearing, AFrameThatClearsNothingLeavesTheWindowExactlyAsItWas) {
  // Ledger is swept only on a frame that actually cleared: stale attachment
  // state, a missing attach-link TF, or an unplaceable payload clear nothing
  // and never advance the window (the node never calls `sweep`).
  bridge::AttachSweepLedger ledger;
  const double open_reach = bridge::attach_transition_padding(0.0, kAttachSweepPadding);

  // 30 grids' worth of refusals: the ledger never learns the payload exists.
  EXPECT_EQ(ledger.size(), 0U);

  // The first frame that does clear is still the attach transition; drift
  // during the outage does not close a window that was never anchored — it
  // anchors here.
  const tf2::Vector3 drifted = kAttachPose + tf2::Vector3(0.30, 0.0, 0.0);
  EXPECT_EQ(padding_for(ledger, kCarried, drifted), open_reach);
  EXPECT_EQ(padding_for(ledger, kCarried, drifted), open_reach) << "still at its anchor";
}

TEST(PayloadClearing, ThePaddedAttachSweepStillWithholdsTheAttestedSupportPatch) {
  // Partition is not weakened by the wider reach: cells the attach sweep
  // newly reaches inside the attested patch are withheld exactly as the
  // footprint is, so the round-6 fix does not reopen the 2026-08-14 defect.
  // The padding only buys the payload-side residue.
  const double padding = bridge::attach_transition_padding(0.0, kAttachSweepPadding);

  auto grid = resting_grid();
  // One counter cell a voxel outside the payload's footprint: 22.5 mm from the
  // payload's face (out of the steady reach, inside the padded one) and inside
  // the attested patch, laterally and in the slab.
  const std::size_t ring = cell_index(grid, 0.4625, 0.0125, kSupportLayerZ);
  // …and one payload-side cell at a comparable distance but 128 mm above the
  // attested plane: nothing exempts it, and it is what the padding is for.
  const std::size_t residue = cell_index(grid, 0.4625, 0.0125, 0.4875);
  grid.occupancy[ring] = 1;
  grid.occupancy[residue] = 1;
  ASSERT_EQ(occupied_cells(grid), 31U);

  const Placed p = placed(resting_payload());
  ASSERT_EQ(p.patches.size(), 1U);

  auto steady = grid;
  EXPECT_EQ(clear_with(steady, p, 0.0), 8U) << "the steady reach explains neither extra cell";
  EXPECT_NE(steady.occupancy[ring], 0);
  EXPECT_NE(steady.occupancy[residue], 0);

  auto attach = grid;
  EXPECT_EQ(clear_with(attach, p, padding), 9U);
  EXPECT_NE(attach.occupancy[ring], 0) << "attested support, withheld from the padded sweep too";
  EXPECT_EQ(attach.occupancy[residue], 0) << "payload-side residue, which is what the padding buys";
  for (const double x : kFootprintX) {
    for (const double y : kFootprintY) {
      EXPECT_NE(attach.occupancy[cell_index(attach, x, y, kSupportLayerZ)], 0)
          << "counter cell (" << x << ", " << y << ") must survive the padded sweep";
    }
  }
  EXPECT_NE(
      attach.occupancy[cell_index(attach, kObstacleCell.x(), kObstacleCell.y(), kObstacleCell.z())],
      0)
      << "the real obstacle beside the payload is still 75.9 mm out, padded reach or not";

  // The counterfactual that makes the withholding load-bearing rather than
  // incidental: with no attestation on the wire, the padded sweep eats the
  // counter, the ring, and the residue alike.
  WireObject unattested = resting_payload();
  unattested.support_contact_valid = false;
  auto unpartitioned = grid;
  EXPECT_EQ(clear_with(unpartitioned, placed(unattested), padding), 30U);
  EXPECT_EQ(unpartitioned.occupancy[ring], 0);
}

TEST(PayloadClearing, TheAttachSweepStopsOneVoxelPastTheSteadyReach) {
  // The widened reach is bounded, and bounded by the lattice rather than by a
  // fudge factor: circumradius + one voxel, to the millimetre either side.
  const double padding = bridge::attach_transition_padding(0.0, kAttachSweepPadding);
  const double reach = kCircumradius + kAttachSweepPadding;
  auto grid = lowered(deploy_sim_tree());
  const std::size_t idx = cell_index(grid, 0.40, 0.0, 0.35);
  grid.occupancy[idx] = 1;

  auto inside = grid;
  EXPECT_EQ(bridge::clear_attached_payload_cells(
                inside, {payload_at_distance(inside, reach - 0.001)}, {}, padding),
            1U);

  auto outside = grid;
  EXPECT_EQ(bridge::clear_attached_payload_cells(
                outside, {payload_at_distance(outside, reach + 0.001)}, {}, padding),
            0U);
  EXPECT_NE(outside.occupancy[idx], 0);
}

TEST(PayloadClearing, TheAttachPaddingNeverShrinksTheSteadyReach) {
  // The attach transition is the steady reach PLUS the sweep padding, so no
  // value of the new parameter can make the transition clear less than an
  // ordinary frame would — a parameter must not be a way to narrow the clearing
  // below what the payload's own volume explains.
  EXPECT_NEAR(bridge::attach_transition_padding(0.0, kAttachSweepPadding), kAttachSweepPadding,
              1e-12);
  EXPECT_NEAR(bridge::attach_transition_padding(0.004, kAttachSweepPadding),
              0.004 + kAttachSweepPadding, 1e-12);
  EXPECT_NEAR(bridge::attach_transition_padding(0.004, 0.0), 0.004, 1e-12);
  EXPECT_NEAR(bridge::attach_transition_padding(0.004, -0.01), 0.004, 1e-12);
  EXPECT_NEAR(bridge::attach_transition_padding(0.004, std::nan("")), 0.004, 1e-12);
  EXPECT_NEAR(bridge::attach_transition_padding(-1.0, kAttachSweepPadding), kAttachSweepPadding,
              1e-12);
}
