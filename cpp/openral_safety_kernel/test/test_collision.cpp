// SPDX-License-Identifier: Apache-2.0
// Unit coverage for the allocation-free self-collision
// core: hand-rolled forward kinematics + closed-form capsule-capsule
// distance + non-adjacent-pair self-collision over a joint configuration.
//
// Ground truth is hand-computed analytically (no MuJoCo dependency here);
// the MuJoCo-oracle cross-check lives in the sim-tier integration test that
// wires this core into the live kernel. test_no_alloc-style counting proves
// the hot path never allocates.

// GCC 13 emits a bogus `-Wnonnull` from inside <vector> when this file is
// built at -O3 (the Release build the x86 image uses): the fixtures below
// assign brace-init-lists to `std::vector<Transform>`, and in
// `_M_assign_aux`'s `else if (size() >= __len)` arm (bits/vector.tcc) a null
// `_M_start` implies `size() == 0` and therefore `__len == 0`, so the flagged
// `__builtin_memmove` sits in an unreachable block. `-Wnonnull` runs early in
// the middle-end, before the pass that deletes that block, so it warns anyway
// — upstream GCC PR 116851, still open, under root-cause PR 87489. Scoped to
// the standard headers so `-Wnonnull` still covers this file's own code.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnonnull"
#include "openral_safety_kernel/collision.hpp"

#include <atomic>
#include <cmath>
#include <cstdlib>
#include <iterator>
#include <new>
#include <set>
#include <string>
#include <vector>
#pragma GCC diagnostic pop

#include <gtest/gtest.h>

namespace osk = openral_safety_kernel;

namespace {

constexpr double kPi = 3.14159265358979323846;

// Identity rigid transform.
osk::Transform identity() { return osk::Transform{}; }

// Pure translation.
osk::Transform translate(double x, double y, double z) {
  osk::Transform t;
  t.t = {x, y, z};
  return t;
}

}  // namespace

// ── transform_from_xyz_rpy ────────────────────────────────────────────────────

TEST(TransformFromXyzRpy, YawRotatesAboutZWithTranslation) {
  const osk::Transform t = osk::transform_from_xyz_rpy(1.0, 2.0, 3.0, 0.0, 0.0, kPi / 2.0);
  EXPECT_NEAR(t.t.x, 1.0, 1e-9);
  EXPECT_NEAR(t.t.y, 2.0, 1e-9);
  EXPECT_NEAR(t.t.z, 3.0, 1e-9);
  // Rz(90) row-major.
  EXPECT_NEAR(t.r[0], 0.0, 1e-9);
  EXPECT_NEAR(t.r[1], -1.0, 1e-9);
  EXPECT_NEAR(t.r[3], 1.0, 1e-9);
  EXPECT_NEAR(t.r[4], 0.0, 1e-9);
  EXPECT_NEAR(t.r[8], 1.0, 1e-9);
}

TEST(TransformFromXyzRpy, RollRotatesAboutX) {
  const osk::Transform t = osk::transform_from_xyz_rpy(0.0, 0.0, 0.0, kPi / 2.0, 0.0, 0.0);
  // Rx(90) row-major: [[1,0,0],[0,0,-1],[0,1,0]].
  EXPECT_NEAR(t.r[0], 1.0, 1e-9);
  EXPECT_NEAR(t.r[4], 0.0, 1e-9);
  EXPECT_NEAR(t.r[5], -1.0, 1e-9);
  EXPECT_NEAR(t.r[7], 1.0, 1e-9);
  EXPECT_NEAR(t.r[8], 0.0, 1e-9);
}

// ── capsule_distance ──────────────────────────────────────────────────────────

TEST(CapsuleDistance, ParallelCapsulesReportSurfaceGap) {
  // Two vertical (local +Z) capsules 0.5 m apart on the x-axis; radii 0.1.
  // Centerline distance 0.5; surface gap 0.5 - 0.1 - 0.1 = 0.3.
  const double d = osk::capsule_distance(identity(), 0.1, 0.2, translate(0.5, 0.0, 0.0), 0.1, 0.2);
  EXPECT_NEAR(d, 0.3, 1e-9);
}

TEST(CapsuleDistance, OverlappingCapsulesReportNegativeDistance) {
  // Centerline 0.15, radii sum 0.2 → penetration of -0.05.
  const double d = osk::capsule_distance(identity(), 0.1, 0.2, translate(0.15, 0.0, 0.0), 0.1, 0.2);
  EXPECT_LT(d, 0.0);
  EXPECT_NEAR(d, -0.05, 1e-9);
}

TEST(CapsuleDistance, EndCapToEndCapAlongAxis) {
  // Same axis (z); A spans [-0.2, 0.2], B centered at z=1.0 spans [0.8, 1.2].
  // Closest endcaps: A top (z=0.2) to B bottom (z=0.8) = 0.6 centerline,
  // minus radii 0.1 + 0.1 → 0.4 surface gap.
  const double d = osk::capsule_distance(identity(), 0.1, 0.2, translate(0.0, 0.0, 1.0), 0.1, 0.2);
  EXPECT_NEAR(d, 0.4, 1e-9);
}

// ── forward_kinematics ────────────────────────────────────────────────────────

TEST(ForwardKinematics, RevoluteAboutZRotatesChildFrame) {
  // link0: fixed root at the origin. link1: revolute about local +Z, sitting
  // 1 m along +x of the root; rotate it by +90°.
  osk::CollisionModel m;
  m.n_links = 2;
  m.parent = {-1, 0};
  m.joint_kind = {osk::JointKind::kFixed, osk::JointKind::kRevolute};
  m.dof_index = {-1, 0};
  m.origin = {identity(), translate(1.0, 0.0, 0.0)};
  m.axis = {{0.0, 0.0, 1.0}, {0.0, 0.0, 1.0}};

  osk::CollisionScratch scratch;
  scratch.link_world.resize(2);

  const std::vector<double> qpos = {kPi / 2.0};
  osk::forward_kinematics(m, qpos.data(), qpos.size(), scratch);

  // The joint origin point does not move (rotation is about its own axis).
  EXPECT_NEAR(scratch.link_world[1].t.x, 1.0, 1e-9);
  EXPECT_NEAR(scratch.link_world[1].t.y, 0.0, 1e-9);
  EXPECT_NEAR(scratch.link_world[1].t.z, 0.0, 1e-9);
  // Rotation is Rz(90): local +x maps to world +y.
  // Row-major R: r[0]=cos=0, r[1]=-sin=-1, r[3]=sin=1, r[4]=cos=0, r[8]=1.
  EXPECT_NEAR(scratch.link_world[1].r[0], 0.0, 1e-9);
  EXPECT_NEAR(scratch.link_world[1].r[1], -1.0, 1e-9);
  EXPECT_NEAR(scratch.link_world[1].r[3], 1.0, 1e-9);
  EXPECT_NEAR(scratch.link_world[1].r[4], 0.0, 1e-9);
  EXPECT_NEAR(scratch.link_world[1].r[8], 1.0, 1e-9);
}

// ── check_self_collision ──────────────────────────────────────────────────────

namespace {

// 3-link model with a capsule on each link; adjacent pairs (0,1) and (1,2)
// are in the allowed-collision matrix (they touch by design).
osk::CollisionModel three_capsule_model() {
  osk::CollisionModel m;
  m.n_links = 3;
  m.parent = {-1, 0, 1};
  m.joint_kind = {osk::JointKind::kFixed, osk::JointKind::kFixed, osk::JointKind::kFixed};
  m.dof_index = {-1, -1, -1};
  m.origin = {identity(), identity(), identity()};
  m.axis = {{0, 0, 1}, {0, 0, 1}, {0, 0, 1}};
  osk::Capsule c;
  c.radius = 0.1;
  c.half_length = 0.2;
  c.origin = identity();
  m.capsule_link = {0, 1, 2};
  m.capsules = {c, c, c};
  m.allowed_pairs = {{0, 1}, {1, 2}};
  return m;
}

}  // namespace

TEST(SelfCollision, DetectsOverlapBetweenNonAdjacentLinks) {
  const auto m = three_capsule_model();
  osk::CollisionScratch s;
  s.link_world = {identity(), translate(5.0, 0.0, 0.0), translate(0.05, 0.0, 0.0)};
  // cap0 (x=0) vs cap2 (x=0.05): centerline 0.05, radii 0.2 → -0.15 overlap.
  // pair (0,2) is NOT in the allowed matrix → must fire.
  const auto hit = osk::check_self_collision(m, s, 0.0);
  EXPECT_TRUE(hit.hit);
  EXPECT_EQ(hit.link_a, 0);
  EXPECT_EQ(hit.link_b, 2);
  EXPECT_NEAR(hit.min_distance, -0.15, 1e-9);
}

TEST(SelfCollision, IgnoresAllowedPair) {
  auto m = three_capsule_model();
  m.allowed_pairs = {{0, 1}, {1, 2}, {0, 2}};  // overlap of (0,2) is now allowed
  osk::CollisionScratch s;
  s.link_world = {identity(), translate(5.0, 0.0, 0.0), translate(0.05, 0.0, 0.0)};
  const auto hit = osk::check_self_collision(m, s, 0.0);
  EXPECT_FALSE(hit.hit);
}

TEST(SelfCollision, NoHitWhenSeparatedAndReportsMinDistance) {
  const auto m = three_capsule_model();
  osk::CollisionScratch s;
  s.link_world = {identity(), translate(5.0, 0.0, 0.0), translate(1.0, 0.0, 0.0)};
  // Closest non-allowed pair (0,2): centerline 1.0, radii 0.2 → 0.8 gap.
  const auto hit = osk::check_self_collision(m, s, 0.0);
  EXPECT_FALSE(hit.hit);
  EXPECT_NEAR(hit.min_distance, 0.8, 1e-9);
}

TEST(SelfCollision, MarginTreatsNearMissAsCollision) {
  const auto m = three_capsule_model();
  osk::CollisionScratch s;
  s.link_world = {identity(), translate(5.0, 0.0, 0.0), translate(1.0, 0.0, 0.0)};
  // 0.8 surface gap < 1.0 margin → fire.
  const auto hit = osk::check_self_collision(m, s, 1.0);
  EXPECT_TRUE(hit.hit);
  EXPECT_EQ(hit.link_a, 0);
  EXPECT_EQ(hit.link_b, 2);
}

TEST(SelfCollision, MultipleCapsulesPerLinkAreCheckedIndependently) {
  // link0 carries TWO capsules (one far, one near link1); link1 carries one.
  // Only link0's near capsule overlaps link1 — and the two link0 capsules,
  // sharing a link, must never self-collide.
  osk::CollisionModel m;
  m.n_links = 2;
  m.parent = {-1, 0};
  m.joint_kind = {osk::JointKind::kFixed, osk::JointKind::kFixed};
  m.dof_index = {-1, -1};
  m.origin = {identity(), identity()};
  m.axis = {{0, 0, 1}, {0, 0, 1}};
  osk::Capsule far_c;
  far_c.radius = 0.1;
  far_c.half_length = 0.2;
  far_c.origin = translate(5.0, 0.0, 0.0);
  osk::Capsule near_c;
  near_c.radius = 0.1;
  near_c.half_length = 0.2;
  near_c.origin = translate(0.05, 0.0, 0.0);
  osk::Capsule l1;
  l1.radius = 0.1;
  l1.half_length = 0.2;
  l1.origin = identity();
  m.capsule_link = {0, 0, 1};  // two capsules on link0, one on link1
  m.capsules = {far_c, near_c, l1};

  osk::CollisionScratch s;
  s.link_world = {identity(), identity()};
  const auto hit = osk::check_self_collision(m, s, 0.0);
  EXPECT_TRUE(hit.hit);
  EXPECT_EQ(hit.link_a, 0);
  EXPECT_EQ(hit.link_b, 1);
  EXPECT_NEAR(hit.min_distance, -0.15, 1e-9);  // near capsule (x=0.05) vs link1 (x=0)
}

// ── check_world_collision ─────────────────────────────────────────────────────

namespace {

osk::CollisionModel one_capsule_model() {
  osk::CollisionModel m;
  m.n_links = 1;
  m.parent = {-1};
  m.joint_kind = {osk::JointKind::kFixed};
  m.dof_index = {-1};
  m.origin = {identity()};
  m.axis = {{0, 0, 1}};
  osk::Capsule c;
  c.radius = 0.1;
  c.half_length = 0.2;
  c.origin = identity();
  m.capsule_link = {0};
  m.capsules = {c};
  return m;
}

osk::WorldModel world_obstacle_at(double x) {
  osk::WorldModel w;
  osk::Capsule obs;
  obs.radius = 0.1;
  obs.half_length = 0.2;
  obs.origin = translate(x, 0.0, 0.0);
  w.capsules = {obs};
  return w;
}

}  // namespace

TEST(WorldCollision, DetectsRobotCapsuleVsWorldObstacle) {
  const auto m = one_capsule_model();
  osk::CollisionScratch s;
  s.link_world = {identity()};  // robot capsule centered at the origin
  const auto w = world_obstacle_at(0.15);
  const auto hit = osk::check_world_collision(m, s, w, 0.0);
  EXPECT_TRUE(hit.hit);
  EXPECT_EQ(hit.link_a, 0);                    // robot link 0
  EXPECT_EQ(hit.link_b, 0);                    // world obstacle 0
  EXPECT_NEAR(hit.min_distance, -0.05, 1e-9);  // centerline 0.15, radii 0.2
}

TEST(WorldCollision, NoHitWhenSeparatedReportsMinDistance) {
  const auto m = one_capsule_model();
  osk::CollisionScratch s;
  s.link_world = {identity()};
  const auto hit = osk::check_world_collision(m, s, world_obstacle_at(1.0), 0.0);
  EXPECT_FALSE(hit.hit);
  EXPECT_NEAR(hit.min_distance, 0.8, 1e-9);
}

TEST(WorldCollision, MarginTreatsNearMissAsCollision) {
  const auto m = one_capsule_model();
  osk::CollisionScratch s;
  s.link_world = {identity()};
  const auto hit = osk::check_world_collision(m, s, world_obstacle_at(1.0), 1.0);
  EXPECT_TRUE(hit.hit);
}

TEST(WorldCollision, EmptyWorldNeverHits) {
  const auto m = one_capsule_model();
  osk::CollisionScratch s;
  s.link_world = {identity()};
  const osk::WorldModel empty;
  const auto hit = osk::check_world_collision(m, s, empty, 0.0);
  EXPECT_FALSE(hit.hit);
}

// ── check_voxel_collision ─────────────────────────────────────────────────────

namespace {

// 5x5x5 grid, 0.1 m voxels, origin (-0.25,-0.25,-0.25) so voxel (2,2,2)'s
// centre sits at the base-frame origin.
osk::VoxelGrid make_grid(const std::vector<std::uint8_t>& occ) {
  osk::VoxelGrid g;
  g.origin = {-0.25, -0.25, -0.25};
  g.resolution = 0.1;
  g.sx = 5;
  g.sy = 5;
  g.sz = 5;
  g.occupancy = occ.data();
  return g;
}

int voxel_index(int x, int y, int z) { return x + 5 * (y + 5 * z); }

}  // namespace

TEST(VoxelCollision, DetectsCapsuleOverlappingOccupiedVoxel) {
  const auto m = one_capsule_model();  // capsule centred at origin, radius 0.1
  osk::CollisionScratch s;
  s.link_world = {identity()};
  std::vector<std::uint8_t> occ(125, 0);
  occ[static_cast<std::size_t>(voxel_index(2, 2, 2))] = 1;  // centre voxel at the origin
  const auto hit = osk::check_voxel_collision(m, s, make_grid(occ), 0.0);
  EXPECT_TRUE(hit.hit);
  EXPECT_EQ(hit.link_a, 0);
  EXPECT_EQ(hit.link_b, voxel_index(2, 2, 2));
  EXPECT_LT(hit.min_distance, 0.0);
}

TEST(VoxelCollision, NoHitWhenAllFree) {
  const auto m = one_capsule_model();
  osk::CollisionScratch s;
  s.link_world = {identity()};
  const std::vector<std::uint8_t> occ(125, 0);
  const auto hit = osk::check_voxel_collision(m, s, make_grid(occ), 0.0);
  EXPECT_FALSE(hit.hit);
}

TEST(VoxelCollision, NoHitWhenOccupiedVoxelIsFar) {
  const auto m = one_capsule_model();
  osk::CollisionScratch s;
  s.link_world = {identity()};
  std::vector<std::uint8_t> occ(125, 0);
  occ[static_cast<std::size_t>(voxel_index(4, 4, 4))] = 1;  // corner voxel, well clear
  const auto hit = osk::check_voxel_collision(m, s, make_grid(occ), 0.0);
  EXPECT_FALSE(hit.hit);
}

TEST(VoxelCollision, MarginTreatsNearMissAsCollision) {
  const auto m = one_capsule_model();
  osk::CollisionScratch s;
  s.link_world = {identity()};
  std::vector<std::uint8_t> occ(125, 0);
  occ[static_cast<std::size_t>(voxel_index(4, 4, 4))] = 1;
  const auto hit = osk::check_voxel_collision(m, s, make_grid(occ), 1.0);
  EXPECT_TRUE(hit.hit);
}

TEST(VoxelCollision, CubeCornerDoesNotUseCircumscribedSphere) {
  const auto m = one_capsule_model();
  osk::CollisionScratch s;
  s.link_world = {identity()};
  const std::vector<std::uint8_t> occ = {1};
  osk::VoxelGrid grid;
  grid.origin = {0.09, 0.09, 0.20};  // centre=(0.14,0.14,0.25), side=0.1
  grid.resolution = 0.1;
  grid.sx = 1;
  grid.sy = 1;
  grid.sz = 1;
  grid.occupancy = occ.data();
  const auto hit = osk::check_voxel_collision(m, s, grid, 0.02);
  EXPECT_FALSE(hit.hit)
      << "the real voxel cube is >2 cm clear; a circumscribed sphere would false-hit";
  EXPECT_GT(hit.min_distance, 0.02);
}

TEST(VoxelCollision, NullGridNeverHits) {
  const auto m = one_capsule_model();
  osk::CollisionScratch s;
  s.link_world = {identity()};
  const osk::VoxelGrid empty;  // occupancy == nullptr
  const auto hit = osk::check_voxel_collision(m, s, empty, 0.0);
  EXPECT_FALSE(hit.hit);
}

// ── ADR-0095: base-frame alignment leaves the envelope untouched ──────────────

namespace {

// A PandaMobile-shaped arm root: a fixed base link plus one link carrying the
// arm's first capsule, mounted `root_z` above `base_link` — the manifest's
// `panda_joint1` origin. RoboCasa's pedestal makes that either 1.033 (the
// pre-ADR-0095 root, which absorbed the 0.700 m pedestal inside the kernel) or
// 0.333 (the Franka URDF value, once `base_link` itself is the arm mount).
osk::CollisionModel arm_root_model(double root_z) {
  osk::CollisionModel m;
  m.n_links = 2;
  m.parent = {-1, 0};
  m.joint_kind = {osk::JointKind::kFixed, osk::JointKind::kFixed};
  m.dof_index = {-1, -1};
  m.origin = {identity(), translate(0.0, 0.0, root_z)};
  m.axis = {{0, 0, 1}, {0, 0, 1}};
  osk::Capsule c;
  c.radius = 0.05;
  c.half_length = 0.1;
  c.origin = identity();
  m.capsule_link = {1};
  m.capsules = {c};
  return m;
}

// One occupied row of counter cells, `origin_z` metres up the base frame.
osk::VoxelGrid counter_grid(const std::vector<std::uint8_t>& occ, double origin_z) {
  osk::VoxelGrid g;
  g.origin = {-0.1, -0.1, origin_z};
  g.resolution = 0.025;
  g.sx = 8;
  g.sy = 8;
  g.sz = 8;
  g.occupancy = occ.data();
  return g;
}

}  // namespace

TEST(VoxelCollision, BaseFrameAlignmentPreservesTheProtectiveEnvelope) {
  // ADR-0095 lands two equal-and-opposite 0.700 m shifts in ONE commit: the
  // grid's *content* drops by the RoboCasa pedestal height (it stops being
  // world-referenced and becomes genuinely `base_link`-referenced), and the
  // collision FK root drops by the same 0.700 m (1.033 -> 0.333). A
  // capsule-vs-voxel distance depends only on their difference, so the
  // kernel's decision — hit, evidence cell, and penetration depth — is
  // bit-identical across the pair. The fix is a change of *origin*, never a
  // relaxation: nothing here shrinks the envelope. Landing only one half
  // would put the kernel 0.700 m out (hazard HZ-0095-1), which is exactly
  // what the two configurations below would show if they disagreed.
  constexpr double kPedestal = 0.700;
  std::vector<std::uint8_t> occ(8 * 8 * 8, 0);
  for (int iy = 0; iy < 8; ++iy) {
    for (int ix = 0; ix < 8; ++ix) {
      occ[static_cast<std::size_t>(ix + 8 * (iy + 8 * 2))] = 1;  // one counter row
    }
  }

  osk::CollisionScratch before_scratch;
  before_scratch.link_world.resize(2);
  const auto before_model = arm_root_model(1.033);
  const std::vector<double> qpos;
  osk::forward_kinematics(before_model, qpos.data(), qpos.size(), before_scratch);
  const auto before = osk::check_voxel_collision(before_model, before_scratch,
                                                 counter_grid(occ, 0.15 + kPedestal), 0.01);

  osk::CollisionScratch after_scratch;
  after_scratch.link_world.resize(2);
  const auto after_model = arm_root_model(0.333);
  osk::forward_kinematics(after_model, qpos.data(), qpos.size(), after_scratch);
  const auto after =
      osk::check_voxel_collision(after_model, after_scratch, counter_grid(occ, 0.15), 0.01);

  EXPECT_TRUE(before.hit) << "fixture must actually exercise a hit";
  EXPECT_EQ(before.hit, after.hit);
  EXPECT_EQ(before.link_a, after.link_a);
  EXPECT_EQ(before.link_b, after.link_b) << "same evidence cell index";
  EXPECT_NEAR(before.min_distance, after.min_distance, 1e-12);
  EXPECT_NEAR(before.sweep_min_distance, after.sweep_min_distance, 1e-12);
}

TEST(VoxelCollision, HalfAppliedFrameAlignmentMovesTheKernelByThePedestal) {
  // The atomicity requirement, stated as a test: keeping the pre-ADR-0095 FK
  // root while the producer has already been fixed (or vice versa) leaves the
  // arm a full pedestal height away from the obstacles it is checked against,
  // so the counter row it should be 0.01 m from is not seen at all.
  std::vector<std::uint8_t> occ(8 * 8 * 8, 0);
  for (int iy = 0; iy < 8; ++iy) {
    for (int ix = 0; ix < 8; ++ix) {
      occ[static_cast<std::size_t>(ix + 8 * (iy + 8 * 2))] = 1;
    }
  }
  osk::CollisionScratch scratch;
  scratch.link_world.resize(2);
  const auto stale_model = arm_root_model(1.033);  // producer fixed, kernel not
  const std::vector<double> qpos;
  osk::forward_kinematics(stale_model, qpos.data(), qpos.size(), scratch);
  const auto hit = osk::check_voxel_collision(stale_model, scratch, counter_grid(occ, 0.15), 0.01);
  EXPECT_FALSE(hit.hit) << "a half-applied ADR-0095 blinds the kernel to the counter";
}

// ── jacobian_dls_step (predictive Cartesian) ─────────────────────────

namespace {

// Planar 2R arm: two revolute-Z joints (link lengths 1 m) + a fixed EE link at
// the tip. At q=[0,0] the EE sits at (2,0,0). EE link index = 2.
osk::CollisionModel two_r_planar_arm() {
  osk::CollisionModel m;
  m.n_links = 3;
  m.parent = {-1, 0, 1};
  m.joint_kind = {osk::JointKind::kRevolute, osk::JointKind::kRevolute, osk::JointKind::kFixed};
  m.dof_index = {0, 1, -1};
  m.origin = {identity(), translate(1.0, 0.0, 0.0), translate(1.0, 0.0, 0.0)};
  m.axis = {{0.0, 0.0, 1.0}, {0.0, 0.0, 1.0}, {0.0, 0.0, 0.0}};
  return m;
}

}  // namespace

TEST(JacobianDls, RealizesSmallTranslationTwistOn2RArm) {
  const auto m = two_r_planar_arm();
  osk::CollisionScratch s;
  s.link_world.resize(3);
  std::vector<double> q = {0.0, 0.0};
  osk::forward_kinematics(m, q.data(), q.size(), s);
  // EE starts at (2,0,0).
  ASSERT_NEAR(s.link_world[2].t.x, 2.0, 1e-9);
  ASSERT_NEAR(s.link_world[2].t.y, 0.0, 1e-9);

  // Ask for a +1 cm EE translation along +y (no rotation).
  const double twist[6] = {0.0, 0.01, 0.0, 0.0, 0.0, 0.0};
  std::vector<double> dq(2, 0.0);
  ASSERT_TRUE(osk::jacobian_dls_step(m, s, /*ee_link=*/2, twist, /*lambda=*/1e-4, dq.data(), 2));

  q[0] += dq[0];
  q[1] += dq[1];
  osk::forward_kinematics(m, q.data(), q.size(), s);
  // The reconstructed step must move the EE to ~(2, 0.01, 0) (small-angle exact
  // for this non-singular config; loose tol absorbs DLS damping + linearization).
  EXPECT_NEAR(s.link_world[2].t.x, 2.0, 1e-3);
  EXPECT_NEAR(s.link_world[2].t.y, 0.01, 1e-3);
}

TEST(JacobianDls, FailsSafeOnInvalidEeLinkAndZeroesDq) {
  const auto m = two_r_planar_arm();
  osk::CollisionScratch s;
  s.link_world.resize(3);
  std::vector<double> q = {0.3, -0.2};
  osk::forward_kinematics(m, q.data(), q.size(), s);
  const double twist[6] = {0.01, 0.0, 0.0, 0.0, 0.0, 0.0};
  std::vector<double> dq = {7.0, 7.0};  // sentinel — must be zeroed on failure
  EXPECT_FALSE(osk::jacobian_dls_step(m, s, /*ee_link=*/99, twist, 1e-4, dq.data(), 2));
  EXPECT_EQ(dq[0], 0.0);
  EXPECT_EQ(dq[1], 0.0);
}

TEST(JacobianDls, BlockedDofIsExcludedFromTheSolution) {
  // Blocking dof 0 must force the whole EE twist onto dof 1 (dq[0] stays 0).
  const auto m = two_r_planar_arm();
  osk::CollisionScratch s;
  s.link_world.resize(3);
  std::vector<double> q = {0.0, 0.3};
  osk::forward_kinematics(m, q.data(), q.size(), s);
  const double twist[6] = {0.0, 0.01, 0.0, 0.0, 0.0, 0.0};
  std::vector<double> dq(2, 0.0);
  const std::uint8_t blocked[2] = {1, 0};  // block dof 0
  ASSERT_TRUE(osk::jacobian_dls_step(m, s, 2, twist, 1e-4, dq.data(), 2, blocked));
  EXPECT_EQ(dq[0], 0.0);
  EXPECT_NE(dq[1], 0.0);
}

TEST(JacobianDls, DampingKeepsStepBoundedNearSingularStretch) {
  // Fully stretched (q=[0,0]) the 2R arm is singular for +x motion (cannot move
  // the EE further along x). With damping the requested +x twist must yield a
  // FINITE, bounded dq rather than blowing up.
  const auto m = two_r_planar_arm();
  osk::CollisionScratch s;
  s.link_world.resize(3);
  std::vector<double> q = {0.0, 0.0};
  osk::forward_kinematics(m, q.data(), q.size(), s);
  const double twist[6] = {0.05, 0.0, 0.0, 0.0, 0.0, 0.0};  // along the singular x
  std::vector<double> dq(2, 0.0);
  ASSERT_TRUE(osk::jacobian_dls_step(m, s, 2, twist, /*lambda=*/0.05, dq.data(), 2));
  EXPECT_TRUE(std::isfinite(dq[0]));
  EXPECT_TRUE(std::isfinite(dq[1]));
  EXPECT_LT(std::fabs(dq[0]), 1.0);
  EXPECT_LT(std::fabs(dq[1]), 1.0);
}

// ── allocation-free guarantee ─────────────────────────────────────────────────

namespace {
std::atomic<std::uint64_t> g_alloc_count{0};
std::atomic<bool> g_count_enabled{false};
}  // namespace

void* operator new(std::size_t size) {
  if (g_count_enabled.load(std::memory_order_relaxed)) {
    g_alloc_count.fetch_add(1, std::memory_order_relaxed);
  }
  void* p = std::malloc(size);
  if (p == nullptr) {
    throw std::bad_alloc{};
  }
  return p;
}

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmismatched-new-delete"
#endif
void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif

TEST(NoAlloc, ForwardKinematicsAndSelfCollisionAreAllocationFree) {
  // Build the model + pre-size scratch OUTSIDE the counted window.
  osk::CollisionModel m;
  m.n_links = 3;
  m.parent = {-1, 0, 1};
  m.joint_kind = {osk::JointKind::kFixed, osk::JointKind::kRevolute, osk::JointKind::kRevolute};
  m.dof_index = {-1, 0, 1};
  m.origin = {identity(), translate(0.0, 0.0, 0.3), translate(0.0, 0.0, 0.3)};
  m.axis = {{0, 0, 1}, {0, 1, 0}, {0, 1, 0}};
  osk::Capsule c;
  c.radius = 0.05;
  c.half_length = 0.15;
  c.origin = identity();
  m.capsule_link = {0, 1, 2};
  m.capsules = {c, c, c};
  // A box on link 0 exercises the box↔capsule / box↔box paths under the counter.
  osk::Obb obb;
  obb.half_extents = {0.05, 0.05, 0.05};
  obb.origin = identity();
  m.box_link = {0};
  m.boxes = {obb};
  m.allowed_pairs = {{0, 1}, {1, 2}};

  osk::CollisionScratch scratch;
  scratch.link_world.resize(m.n_links);

  // Pre-built world obstacles + a voxel grid (allocated outside the counted window).
  osk::WorldModel world;
  osk::Capsule obs;
  obs.radius = 0.05;
  obs.half_length = 0.1;
  obs.origin = translate(0.3, 0.0, 0.3);
  world.capsules = {obs, obs};

  std::vector<std::uint8_t> occ(8 * 8 * 8, 0);
  occ[42] = 1;
  osk::VoxelGrid grid;
  grid.origin = {-0.4, -0.4, -0.4};
  grid.resolution = 0.1;
  grid.sx = 8;
  grid.sy = 8;
  grid.sz = 8;
  grid.occupancy = occ.data();

  const std::vector<double> qpos = {0.3, -0.4};

  g_alloc_count.store(0, std::memory_order_relaxed);
  g_count_enabled.store(true, std::memory_order_relaxed);
  for (int i = 0; i < 10000; ++i) {
    osk::forward_kinematics(m, qpos.data(), qpos.size(), scratch);
    const auto self_hit = osk::check_self_collision(m, scratch, 0.0);
    const auto world_hit = osk::check_world_collision(m, scratch, world, 0.0);
    const auto voxel_hit = osk::check_voxel_collision(m, scratch, grid, 0.0);
    (void)self_hit;
    (void)world_hit;
    (void)voxel_hit;
  }
  g_count_enabled.store(false, std::memory_order_relaxed);
  EXPECT_EQ(g_alloc_count.load(std::memory_order_relaxed), 0U)
      << "collision hot path allocated; the self-collision core must stay allocation-free.";
}

TEST(NoAlloc, AttachedCollisionChecksAreAllocationFree) {
  // Robot: base + one movable link carrying a capsule.
  osk::CollisionModel m;
  m.n_links = 2;
  m.parent = {-1, 0};
  m.joint_kind = {osk::JointKind::kFixed, osk::JointKind::kRevolute};
  m.dof_index = {-1, 0};
  m.origin = {identity(), translate(0.0, 0.0, 0.2)};
  m.axis = {{0, 0, 1}, {0, 1, 0}};
  osk::Capsule c;
  c.radius = 0.05;
  c.half_length = 0.1;
  c.origin = identity();
  m.capsule_link = {1};
  m.capsules = {c};

  osk::CollisionScratch scratch;
  scratch.link_world.resize(m.n_links);

  // Pre-sized attached model (fixed capacity), a world obstacle and a voxel
  // grid — all built OUTSIDE the counted window. One object owns one sphere
  // primitive (flattened buffers presized to their caps).
  osk::AttachedModel att;
  att.objects.assign(4, osk::AttachedObject{});
  att.primitives.assign(8, osk::AttachedPrimitive{});
  att.touch_links.assign(8, 0);
  att.objects[0].pose_in_link = translate(0.0, 0.0, 0.15);
  att.objects[0].attach_link = 1;
  att.objects[0].prim_first = 0;
  att.objects[0].prim_count = 1;
  att.objects[0].touch_first = 0;
  att.objects[0].touch_count = 0;
  att.primitives[0].kind = osk::AttachedShapeKind::kSphere;
  att.primitives[0].radius = 0.08;
  att.primitives[0].pose_in_object = identity();
  // Attested support contact, so the witness latch actually walks its cell
  // scan inside the counted window instead of short-circuiting.
  att.objects[0].has_support_witness = true;
  att.objects[0].support_point = {0.0, 0.0, 0.0};
  att.objects[0].support_normal = {0.0, 0.0, 1.0};
  att.objects[0].support_patch_radius = 0.2;
  att.objects[0].support_max_penetration = 0.001;
  att.n_objects = 1;
  att.n_primitives = 1;

  osk::WorldModel world;
  osk::Capsule obs;
  obs.radius = 0.05;
  obs.half_length = 0.1;
  obs.origin = translate(0.3, 0.0, 0.3);
  world.capsules = {obs, obs};

  std::vector<std::uint8_t> occ(8 * 8 * 8, 0);
  occ[42] = 1;
  osk::VoxelGrid grid;
  grid.origin = {-0.4, -0.4, -0.4};
  grid.resolution = 0.1;
  grid.sx = 8;
  grid.sy = 8;
  grid.sz = 8;
  grid.occupancy = occ.data();
  std::vector<std::uint8_t> contact_mask(occ.size(), 0);
  std::vector<double> contact_distance(occ.size(), std::numeric_limits<double>::infinity());
  grid.attached_contact_mask = contact_mask.data();
  grid.attached_contact_distance = contact_distance.data();
  grid.attached_contact_stride = occ.size();
  grid.support_witness_live = 0x1;

  const std::vector<double> qpos = {0.3};

  g_alloc_count.store(0, std::memory_order_relaxed);
  g_count_enabled.store(true, std::memory_order_relaxed);
  for (int i = 0; i < 10000; ++i) {
    osk::forward_kinematics(m, qpos.data(), qpos.size(), scratch);
    const auto self_hit = osk::check_attached_self_collision(m, att, scratch, 0.0);
    const auto world_hit = osk::check_attached_world_collision(m, att, scratch, world, 0.0);
    const auto voxel_hit = osk::check_attached_voxel_collision(m, att, scratch, grid, 0.0);
    const bool contacts_active = osk::update_attached_voxel_contacts(
        att, scratch, grid, contact_mask.data(), contact_distance.data(), contact_mask.size(),
        contact_distance.size(), false);
    const std::uint8_t live = osk::update_support_contact_witnesses(att, scratch, grid, 0x1, 0.0);
    (void)self_hit;
    (void)world_hit;
    (void)voxel_hit;
    (void)contacts_active;
    (void)live;
  }
  g_count_enabled.store(false, std::memory_order_relaxed);
  EXPECT_EQ(g_alloc_count.load(std::memory_order_relaxed), 0U)
      << "attached-object collision checks allocated; the hot path must stay allocation-free.";
}

TEST(NoAlloc, JacobianDlsStepIsAllocationFree) {
  // 2R arm + pre-sized scratch + caller-owned dq, all OUTSIDE the counted window.
  osk::CollisionModel m;
  m.n_links = 3;
  m.parent = {-1, 0, 1};
  m.joint_kind = {osk::JointKind::kRevolute, osk::JointKind::kRevolute, osk::JointKind::kFixed};
  m.dof_index = {0, 1, -1};
  m.origin = {identity(), translate(1.0, 0.0, 0.0), translate(1.0, 0.0, 0.0)};
  m.axis = {{0.0, 0.0, 1.0}, {0.0, 0.0, 1.0}, {0.0, 0.0, 0.0}};
  osk::CollisionScratch scratch;
  scratch.link_world.resize(3);
  std::vector<double> q = {0.2, -0.1};
  std::vector<double> dq(2, 0.0);
  const double twist[6] = {0.0, 0.01, 0.0, 0.0, 0.0, 0.0};
  osk::forward_kinematics(m, q.data(), q.size(), scratch);

  g_alloc_count.store(0, std::memory_order_relaxed);
  g_count_enabled.store(true, std::memory_order_relaxed);
  for (int i = 0; i < 10000; ++i) {
    const bool ok = osk::jacobian_dls_step(m, scratch, 2, twist, 1e-4, dq.data(), 2);
    (void)ok;
  }
  g_count_enabled.store(false, std::memory_order_relaxed);
  EXPECT_EQ(g_alloc_count.load(std::memory_order_relaxed), 0U)
      << "jacobian_dls_step allocated; the predictive Cartesian path must be allocation-free.";
}

// ── Box/OBB primitive (issue #84) ──────────────────────────────────
//
// A blocky link (SO-ARM100/101 `base`) carries an OBB instead of a capsule so
// its flat faces don't over-report clearance. Ground truth here is hand-computed
// analytically; the real-geometry reproduction (fat capsule false-E-stops the
// SO-101 home pose, box clears) lives in the Python MuJoCo-oracle test.

TEST(BoxCapsuleDistance, DisjointAlongFaceNormalIsExact) {
  // Box [-0.05,0.05]^3 at the origin; capsule central segment at x=0.2 (well
  // clear), running vertically. Nearest box face is x=+0.05 → segment↔box gap
  // 0.2-0.05=0.15; minus the 0.02 capsule radius → 0.13.
  const osk::Vec3 h{0.05, 0.05, 0.05};
  const double d = osk::box_capsule_distance(identity(), h, translate(0.2, 0.0, 0.0), 0.02, 0.1);
  EXPECT_NEAR(d, 0.13, 1e-6);
}

TEST(BoxCapsuleDistance, SegmentOnSurfaceGivesNegativeRadius) {
  // Capsule segment lies on the box's +x face (x=0.05) → segment↔box gap 0;
  // minus the radius → -0.02 (interpenetration by the radius).
  const osk::Vec3 h{0.05, 0.05, 0.05};
  const double d = osk::box_capsule_distance(identity(), h, translate(0.05, 0.0, 0.0), 0.02, 0.1);
  EXPECT_NEAR(d, -0.02, 1e-6);
}

TEST(BoxCapsuleDistance, ClosestFeatureIsTopFace) {
  // Capsule above the top face; nearest endpoint at z=0.1 → gap 0.1-0.05=0.05,
  // minus radius 0.01 → 0.04.
  const osk::Vec3 h{0.05, 0.05, 0.05};
  const double d = osk::box_capsule_distance(identity(), h, translate(0.0, 0.0, 0.2), 0.01, 0.1);
  EXPECT_NEAR(d, 0.04, 1e-6);
}

TEST(BoxBoxDistance, SeparatedAlongAxis) {
  const osk::Vec3 h{0.05, 0.05, 0.05};
  const double d = osk::box_box_distance(identity(), h, translate(0.2, 0.0, 0.0), h);
  EXPECT_NEAR(d, 0.1, 1e-9);
}

TEST(BoxBoxDistance, OverlapIsNegative) {
  const osk::Vec3 h{0.05, 0.05, 0.05};
  const double d = osk::box_box_distance(identity(), h, translate(0.08, 0.0, 0.0), h);
  EXPECT_LT(d, 0.0);
  EXPECT_NEAR(d, -0.02, 1e-9);
}

TEST(BoxBoxDistance, RotatedBoxIsPositiveAndConservative) {
  // B rotated 45° about z, centre 0.2 away. SAT max-gap stays a positive lower
  // bound on the true (corner-to-face) distance — never under-reports.
  const osk::Vec3 h{0.05, 0.05, 0.05};
  const osk::Transform b = osk::transform_from_xyz_rpy(0.2, 0.0, 0.0, 0.0, 0.0, kPi / 4.0);
  const double d = osk::box_box_distance(identity(), h, b, h);
  EXPECT_GT(d, 0.0);
  EXPECT_NEAR(d, 0.2 - 0.05 - 0.05 * std::sqrt(2.0), 1e-6);
}

namespace {
// One box on link 0, one capsule on link 1 (non-adjacent, not allowed). Frames
// are set directly on the scratch by each test.
osk::CollisionModel box_plus_capsule_model() {
  osk::CollisionModel m;
  m.n_links = 2;
  m.parent = {-1, 0};
  m.joint_kind = {osk::JointKind::kFixed, osk::JointKind::kFixed};
  m.dof_index = {-1, -1};
  m.origin = {identity(), identity()};
  m.axis = {{0, 0, 1}, {0, 0, 1}};
  m.box_link = {0};
  m.boxes = {osk::Obb{{0.055, 0.048, 0.036}, translate(0.0, 0.0, 0.05)}};
  m.capsule_link = {1};
  m.capsules = {osk::Capsule{0.05, 0.05, identity()}};
  m.allowed_pairs = {};
  return m;
}
}  // namespace

TEST(SelfCollisionBox, BoxClearsDistantCapsule) {
  // Capsule 0.13 m out (like the SO-101 home lower_arm): the tight box clears it
  // where the old fat r=0.075 base capsule would have false-fired.
  const auto m = box_plus_capsule_model();
  osk::CollisionScratch s;
  s.link_world = {identity(), translate(0.13, 0.0, 0.10)};
  const auto hit = osk::check_self_collision(m, s, 0.0);
  EXPECT_FALSE(hit.hit);
  EXPECT_GT(hit.min_distance, 0.0);
}

TEST(SelfCollisionBox, BoxFiresOnRealPenetration) {
  // Capsule driven into the block → box↔capsule distance negative → hit.
  const auto m = box_plus_capsule_model();
  osk::CollisionScratch s;
  s.link_world = {identity(), translate(0.07, 0.0, 0.05)};
  const auto hit = osk::check_self_collision(m, s, 0.0);
  EXPECT_TRUE(hit.hit);
  EXPECT_EQ(hit.link_a, 0);
  EXPECT_EQ(hit.link_b, 1);
  EXPECT_LT(hit.min_distance, 0.0);
}

TEST(SelfCollisionBox, AllowedPairSkipsBoxCapsule) {
  auto m = box_plus_capsule_model();
  m.allowed_pairs = {{0, 1}};  // sanction the pair → never fires
  osk::CollisionScratch s;
  s.link_world = {identity(), translate(0.07, 0.0, 0.05)};
  const auto hit = osk::check_self_collision(m, s, 0.0);
  EXPECT_FALSE(hit.hit);
}

TEST(SelfCollisionBox, BoxBoxPairFiresAndClears) {
  // Two blocky links: box↔box path exercised through check_self_collision.
  osk::CollisionModel m;
  m.n_links = 2;
  m.parent = {-1, 0};
  m.joint_kind = {osk::JointKind::kFixed, osk::JointKind::kFixed};
  m.dof_index = {-1, -1};
  m.origin = {identity(), identity()};
  m.axis = {{0, 0, 1}, {0, 0, 1}};
  m.box_link = {0, 1};
  m.boxes = {osk::Obb{{0.05, 0.05, 0.05}, identity()}, osk::Obb{{0.05, 0.05, 0.05}, identity()}};
  m.allowed_pairs = {};

  osk::CollisionScratch s;
  s.link_world = {identity(), translate(0.2, 0.0, 0.0)};
  EXPECT_FALSE(osk::check_self_collision(m, s, 0.0).hit);  // 0.1 m apart
  s.link_world = {identity(), translate(0.08, 0.0, 0.0)};
  const auto hit = osk::check_self_collision(m, s, 0.0);
  EXPECT_TRUE(hit.hit);  // overlapping
}

// ── Box link vs world / voxel (blocky links stay world-visible) ────

namespace {
osk::CollisionModel one_box_model() {
  osk::CollisionModel m;
  m.n_links = 1;
  m.parent = {-1};
  m.joint_kind = {osk::JointKind::kFixed};
  m.dof_index = {-1};
  m.origin = {identity()};
  m.axis = {{0, 0, 1}};
  m.box_link = {0};
  m.boxes = {osk::Obb{{0.05, 0.05, 0.05}, identity()}};
  return m;
}
}  // namespace

TEST(WorldCollisionBox, BoxLinkIsCheckedAgainstWorldObstacle) {
  const auto m = one_box_model();
  osk::CollisionScratch s;
  s.link_world = {identity()};
  // Obstacle capsule (radius 0.1) at x=0.10: box +x face at 0.05 → gap 0.05,
  // minus the obstacle radius → -0.05 (a hit the capsule-only loop would miss).
  const auto hit = osk::check_world_collision(m, s, world_obstacle_at(0.10), 0.0);
  EXPECT_TRUE(hit.hit);
  EXPECT_EQ(hit.link_a, 0);
  EXPECT_EQ(hit.link_b, 0);
  EXPECT_NEAR(hit.min_distance, -0.05, 1e-6);
  EXPECT_FALSE(osk::check_world_collision(m, s, world_obstacle_at(1.0), 0.0).hit);
}

TEST(VoxelCollisionBox, BoxLinkIsCheckedAgainstOccupiedVoxel) {
  const auto m = one_box_model();
  osk::CollisionScratch s;
  s.link_world = {identity()};
  std::vector<std::uint8_t> occ(8 * 8 * 8, 0);
  osk::VoxelGrid grid;
  grid.origin = {-0.4, -0.4, -0.4};
  grid.resolution = 0.1;
  grid.sx = 8;
  grid.sy = 8;
  grid.sz = 8;
  grid.occupancy = occ.data();
  // Empty grid: no hit.
  EXPECT_FALSE(osk::check_voxel_collision(m, s, grid, 0.0).hit);
  // Occupy the voxel centred at (0.05,0.05,0.05) — inside the box → hit.
  const int i = 4;  // floor((0.05-(-0.4))/0.1)=4 ; centre (-0.4+4.5*0.1)=0.05
  occ[static_cast<std::size_t>(i + 8 * (i + 8 * i))] = 1;
  const auto hit = osk::check_voxel_collision(m, s, grid, 0.0);
  EXPECT_TRUE(hit.hit);
  EXPECT_EQ(hit.link_a, 0);
  EXPECT_LT(hit.min_distance, 0.0);
}

// ── Attached collision objects (grasped payload; ADR-0092) ─────────────
//
// A grasped payload is removed from world occupancy and re-checked as
// collision-active robot geometry: against world obstacles, occupancy voxels,
// and the robot's own links EXCEPT the attach link and explicit touch links
// (the fingers legitimately holding it). Ground truth is hand-computed
// analytically; the full FK + wire-ingest path is exercised in the sim tier.

namespace {

// One sphere primitive (posed in its object's frame).
osk::AttachedPrimitive sphere_prim(double radius,
                                   const osk::Transform& pose_in_object = osk::Transform{}) {
  osk::AttachedPrimitive p;
  p.kind = osk::AttachedShapeKind::kSphere;
  p.radius = radius;
  p.half_length = 0.0;
  p.pose_in_object = pose_in_object;
  return p;
}

// Append an object owning `prims` to `att`, keeping the flattened primitive /
// touch-link buffers and the n_* counts consistent. `touch` names robot link
// indices explicitly allowed to contact the object.
void append_object(osk::AttachedModel& att, int attach_link, const osk::Transform& pose_in_link,
                   const std::vector<osk::AttachedPrimitive>& prims,
                   const std::vector<int>& touch = {}) {
  osk::AttachedObject o;
  o.pose_in_link = pose_in_link;
  o.attach_link = attach_link;
  o.prim_first = static_cast<int>(att.primitives.size());
  o.prim_count = static_cast<int>(prims.size());
  o.touch_first = static_cast<int>(att.touch_links.size());
  o.touch_count = static_cast<int>(touch.size());
  for (const auto& p : prims) {
    att.primitives.push_back(p);
  }
  for (int t : touch) {
    att.touch_links.push_back(t);
  }
  att.objects.push_back(o);
  att.n_objects = att.objects.size();
  att.n_primitives = att.primitives.size();
}

// Robot capsule `c` attached to link index `link`.
void add_capsule(osk::CollisionModel& m, int link, double radius, double half_length,
                 const osk::Transform& origin) {
  osk::Capsule c;
  c.radius = radius;
  c.half_length = half_length;
  c.origin = origin;
  m.capsule_link.push_back(link);
  m.capsules.push_back(c);
}

// A 4-link fixed-chain hand: 0=base, 1=hand (attach), 2=finger, 3=arm.
osk::CollisionModel hand_model() {
  osk::CollisionModel m;
  m.n_links = 4;
  m.parent = {-1, 0, 1, 0};
  m.joint_kind = {osk::JointKind::kFixed, osk::JointKind::kFixed, osk::JointKind::kFixed,
                  osk::JointKind::kFixed};
  m.dof_index = {-1, -1, -1, -1};
  m.origin = {identity(), identity(), identity(), identity()};
  m.axis = {{0, 0, 1}, {0, 0, 1}, {0, 0, 1}, {0, 0, 1}};
  return m;
}

}  // namespace

TEST(AttachedSelfCollision, AllowedFingerContactIsNotAHit) {
  osk::CollisionModel m = hand_model();
  add_capsule(m, 2, 0.05, 0.0, identity());  // finger capsule at origin (overlaps payload)
  osk::CollisionScratch s;
  s.link_world = {identity(), identity(), identity(), identity()};

  osk::AttachedModel att;
  append_object(att, 1, identity(), {sphere_prim(0.1)}, {2});  // finger (link 2) may touch

  const auto hit = osk::check_attached_self_collision(m, att, s, 0.0);
  EXPECT_FALSE(hit.hit) << "a grasped payload legitimately overlaps its touch-link finger";
}

TEST(AttachedSelfCollision, AttachLinkIsAlwaysAllowed) {
  osk::CollisionModel m = hand_model();
  add_capsule(m, 1, 0.05, 0.0, identity());  // capsule on the ATTACH link itself
  osk::CollisionScratch s;
  s.link_world = {identity(), identity(), identity(), identity()};

  osk::AttachedModel att;
  append_object(att, 1, identity(), {sphere_prim(0.1)});  // no touch links

  const auto hit = osk::check_attached_self_collision(m, att, s, 0.0);
  EXPECT_FALSE(hit.hit) << "the attach link is implicitly allowed even without a touch entry";
}

TEST(AttachedSelfCollision, PayloadHittingNonTouchLinkIsRejected) {
  osk::CollisionModel m = hand_model();
  add_capsule(m, 2, 0.05, 0.0, identity());                // finger (touch) at origin
  add_capsule(m, 3, 0.1, 0.0, translate(0.15, 0.0, 0.0));  // arm (non-touch) near payload
  osk::CollisionScratch s;
  s.link_world = {identity(), identity(), identity(), identity()};

  osk::AttachedModel att;
  append_object(att, 1, identity(), {sphere_prim(0.1)}, {2});  // only the finger is a touch link

  const auto hit = osk::check_attached_self_collision(m, att, s, 0.0);
  EXPECT_TRUE(hit.hit);
  EXPECT_EQ(hit.link_a, 0);                    // attached object 0
  EXPECT_EQ(hit.link_b, 3);                    // the non-touch arm link
  EXPECT_NEAR(hit.min_distance, -0.05, 1e-9);  // centerline 0.15, radii 0.2
}

TEST(AttachedWorldCollision, PayloadVsWorldObstacleIsRejected) {
  osk::CollisionModel m = hand_model();
  osk::CollisionScratch s;
  s.link_world = {identity(), identity(), identity(), identity()};
  osk::AttachedModel att;
  append_object(att, 1, identity(), {sphere_prim(0.1)});

  osk::WorldModel w;
  osk::Capsule obs;
  obs.radius = 0.1;
  obs.half_length = 0.2;
  obs.origin = translate(0.15, 0.0, 0.0);
  w.capsules = {obs};

  const auto hit = osk::check_attached_world_collision(m, att, s, w, 0.0);
  EXPECT_TRUE(hit.hit);
  EXPECT_EQ(hit.link_a, 0);
  EXPECT_EQ(hit.link_b, 0);
  EXPECT_NEAR(hit.min_distance, -0.05, 1e-9);
}

TEST(AttachedWorldCollision, ClearPayloadReportsMinDistanceNoHit) {
  osk::CollisionModel m = hand_model();
  osk::CollisionScratch s;
  s.link_world = {identity(), identity(), identity(), identity()};
  osk::AttachedModel att;
  append_object(att, 1, identity(), {sphere_prim(0.1)});

  osk::WorldModel w;
  osk::Capsule obs;
  obs.radius = 0.1;
  obs.half_length = 0.2;
  obs.origin = translate(1.0, 0.0, 0.0);
  w.capsules = {obs};

  const auto hit = osk::check_attached_world_collision(m, att, s, w, 0.0);
  EXPECT_FALSE(hit.hit);
  EXPECT_NEAR(hit.min_distance, 0.8, 1e-9);  // centerline 1.0 - 0.1 - 0.1
}

TEST(AttachedVoxelCollision, PayloadVsOccupiedVoxelIsRejected) {
  osk::CollisionModel m = hand_model();
  osk::CollisionScratch s;
  s.link_world = {identity(), identity(), identity(), identity()};
  osk::AttachedModel att;
  append_object(att, 1, identity(), {sphere_prim(0.1)});  // sphere at origin

  // 5x5x5 grid, 0.1 m voxels, centre voxel (2,2,2) at the base-frame origin.
  std::vector<std::uint8_t> occ(125, 0);
  occ[static_cast<std::size_t>(voxel_index(2, 2, 2))] = 1;
  const auto hit = osk::check_attached_voxel_collision(m, att, s, make_grid(occ), 0.0);
  EXPECT_TRUE(hit.hit);
  EXPECT_EQ(hit.link_a, 0);
  EXPECT_EQ(hit.link_b, voxel_index(2, 2, 2));
  EXPECT_LT(hit.min_distance, 0.0);
}

TEST(AttachedVoxelCollision, FreeGridNeverHits) {
  osk::CollisionModel m = hand_model();
  osk::CollisionScratch s;
  s.link_world = {identity(), identity(), identity(), identity()};
  osk::AttachedModel att;
  append_object(att, 1, identity(), {sphere_prim(0.1)});
  const std::vector<std::uint8_t> occ(125, 0);
  const auto hit = osk::check_attached_voxel_collision(m, att, s, make_grid(occ), 0.0);
  EXPECT_FALSE(hit.hit);
}

TEST(AttachedVoxelCollision, ClearingThePayloadsOwnCellsKeepsThePayloadVsWorldCheck) {
  // What the octomap bridge's payload clearing (`openral_octomap_bridge`,
  // `clear_attached_payload_cells`) takes out of the grid, and what it must
  // leave. The kernel is unchanged by that fix — this pins the property the fix
  // rests on, from the kernel's side.
  //
  // A payload sphere of radius 0.11 m at the origin. Three occupied cells reach
  // the kernel: (2,2,2) and (3,2,2), which lie inside the payload itself (its
  // own pre-grasp occupancy, still in the map), and (4,2,2) at 0.09 m from the
  // payload surface — a real obstacle, far enough out that the bridge's
  // circumradius bound (0.0866 m at 0.1 m cells) does not reach it.
  osk::CollisionModel m = hand_model();
  osk::CollisionScratch s;
  s.link_world = {identity(), identity(), identity(), identity()};
  osk::AttachedModel att;
  append_object(att, 1, identity(), {sphere_prim(0.11)});
  constexpr double kMargin = 0.05;

  std::vector<std::uint8_t> as_mapped(125, 0);
  as_mapped[static_cast<std::size_t>(voxel_index(2, 2, 2))] = 1;
  as_mapped[static_cast<std::size_t>(voxel_index(3, 2, 2))] = 1;
  as_mapped[static_cast<std::size_t>(voxel_index(4, 2, 2))] = 1;
  const auto uncleared =
      osk::check_attached_voxel_collision(m, att, s, make_grid(as_mapped), kMargin);
  ASSERT_TRUE(uncleared.hit);
  EXPECT_EQ(uncleared.link_b, voxel_index(2, 2, 2))
      << "uncleared, the payload stops the robot against itself";

  // The grid as the bridge now publishes it: the payload's own cells gone.
  std::vector<std::uint8_t> cleared = as_mapped;
  cleared[static_cast<std::size_t>(voxel_index(2, 2, 2))] = 0;
  cleared[static_cast<std::size_t>(voxel_index(3, 2, 2))] = 0;
  const auto after = osk::check_attached_voxel_collision(m, att, s, make_grid(cleared), kMargin);
  EXPECT_TRUE(after.hit) << "payload-vs-world is still checked, cell by cell";
  EXPECT_EQ(after.link_b, voxel_index(4, 2, 2));
  EXPECT_LE(after.min_distance, kMargin);
  EXPECT_NEAR(after.min_distance, 0.04, 1e-9) << "and at the true clearance to that cell";

  // And with nothing real beside it, the stop the clearing removes is the whole
  // stop: the payload alone no longer trips the check it is the subject of.
  std::vector<std::uint8_t> only_payload(125, 0);
  only_payload[static_cast<std::size_t>(voxel_index(2, 2, 2))] = 1;
  only_payload[static_cast<std::size_t>(voxel_index(3, 2, 2))] = 1;
  EXPECT_TRUE(osk::check_attached_voxel_collision(m, att, s, make_grid(only_payload), kMargin).hit);
  const std::vector<std::uint8_t> free_grid(125, 0);
  EXPECT_FALSE(osk::check_attached_voxel_collision(m, att, s, make_grid(free_grid), kMargin).hit);
}

TEST(AttachedVoxelCollision, AttachTimeContactAloneNeverExemptsAShallowCell) {
  // The index-keyed non-deepening baseline is gone: a cell merely in contact
  // at attach time (1 cm deep, well short of the embedded-residue threshold)
  // buys the payload nothing. Only a support-contact witness can exempt this,
  // and there is none here — fail-closed.
  osk::CollisionModel m = hand_model();
  osk::CollisionScratch s;
  s.link_world = {identity(), identity(), identity(), identity()};
  osk::AttachedModel att;
  append_object(att, 1, identity(), {sphere_prim(0.06)});

  std::vector<std::uint8_t> occ(125, 0);
  std::vector<std::uint8_t> contact_mask(125, 0);
  std::vector<double> contact_distance(125, std::numeric_limits<double>::infinity());
  occ[static_cast<std::size_t>(voxel_index(3, 2, 2))] = 1;
  auto grid = make_grid(occ);
  grid.attached_contact_mask = contact_mask.data();
  grid.attached_contact_distance = contact_distance.data();
  grid.attached_contact_stride = occ.size();
  grid.attached_contact_tolerance = 0.001;

  EXPECT_TRUE(osk::update_attached_voxel_contacts(att, s, grid, contact_mask.data(),
                                                  contact_distance.data(), contact_mask.size(),
                                                  contact_distance.size(), true));
  const auto hit = osk::check_attached_voxel_collision(m, att, s, grid, 0.0);
  EXPECT_TRUE(hit.hit);
  EXPECT_NEAR(hit.min_distance, -0.01, 1e-9);
}

TEST(AttachedVoxelCollision, EmbeddedResidueIsAllowedOnlyUntilSeparation) {
  osk::CollisionModel m = hand_model();
  osk::CollisionScratch s;
  s.link_world = {identity(), identity(), identity(), identity()};
  osk::AttachedModel att;
  append_object(att, 1, identity(), {sphere_prim(0.1)});

  std::vector<std::uint8_t> occ(125, 0);
  std::vector<std::uint8_t> contact_mask(125, 0);
  std::vector<double> contact_distance(125, std::numeric_limits<double>::infinity());
  occ[static_cast<std::size_t>(voxel_index(2, 2, 2))] = 1;
  auto grid = make_grid(occ);
  grid.attached_contact_mask = contact_mask.data();
  grid.attached_contact_distance = contact_distance.data();
  grid.attached_contact_stride = occ.size();
  grid.attached_contact_tolerance = 0.001;

  EXPECT_TRUE(osk::update_attached_voxel_contacts(att, s, grid, contact_mask.data(),
                                                  contact_distance.data(), contact_mask.size(),
                                                  contact_distance.size(), true));
  att.objects[0].pose_in_link = translate(0.02, 0.0, 0.0);
  EXPECT_FALSE(osk::check_attached_voxel_collision(m, att, s, grid, 0.0).hit);

  att.objects[0].pose_in_link = translate(0.3, 0.0, 0.0);
  EXPECT_FALSE(osk::update_attached_voxel_contacts(att, s, grid, contact_mask.data(),
                                                   contact_distance.data(), contact_mask.size(),
                                                   contact_distance.size(), false));
  att.objects[0].pose_in_link = identity();
  EXPECT_TRUE(osk::check_attached_voxel_collision(m, att, s, grid, 0.0).hit);
}

TEST(AttachedVoxelCollision, MultiObjectBaselinesUseDeclaredGridStride) {
  osk::CollisionModel m = hand_model();
  osk::CollisionScratch s;
  s.link_world = {identity(), identity(), identity(), identity()};
  osk::AttachedModel att;
  append_object(att, 1, identity(), {sphere_prim(0.06)});
  append_object(att, 1, translate(0.2, 0.0, 0.0), {sphere_prim(0.06)});

  std::vector<std::uint8_t> occ(125, 0);
  std::vector<std::uint8_t> contact_mask(125, 0);
  std::vector<double> contact_distance(250, std::numeric_limits<double>::infinity());
  const auto first = static_cast<std::size_t>(voxel_index(2, 2, 2));
  const auto second = static_cast<std::size_t>(voxel_index(4, 2, 2));
  occ[first] = 1;
  occ[second] = 1;
  auto grid = make_grid(occ);
  grid.attached_contact_mask = contact_mask.data();
  grid.attached_contact_distance = contact_distance.data();
  grid.attached_contact_stride = occ.size();

  EXPECT_TRUE(osk::update_attached_voxel_contacts(att, s, grid, contact_mask.data(),
                                                  contact_distance.data(), contact_mask.size(),
                                                  contact_distance.size(), true));
  EXPECT_NE(contact_mask[first] & 0x1U, 0U);
  EXPECT_NE(contact_mask[second] & 0x2U, 0U);
  EXPECT_TRUE(std::isfinite(contact_distance[first]));
  EXPECT_TRUE(std::isfinite(contact_distance[occ.size() + second]));
}

TEST(AttachedObjects, MultipleObjectsAreAllChecked) {
  osk::CollisionModel m = hand_model();
  osk::CollisionScratch s;
  // Attach link 1 at origin; attach link 3 shifted to x=1.0.
  s.link_world = {identity(), identity(), identity(), translate(1.0, 0.0, 0.0)};

  osk::AttachedModel att;
  append_object(att, 1, identity(), {sphere_prim(0.1)});  // object 0: at origin, clear
  append_object(att, 3, identity(), {sphere_prim(0.1)});  // object 1: at x=1.0, overlaps obstacle

  osk::WorldModel w;
  osk::Capsule obs;
  obs.radius = 0.1;
  obs.half_length = 0.2;
  obs.origin = translate(1.15, 0.0, 0.0);  // near object 1 only
  w.capsules = {obs};

  const auto hit = osk::check_attached_world_collision(m, att, s, w, 0.0);
  EXPECT_TRUE(hit.hit);
  EXPECT_EQ(hit.link_a, 1) << "the second attached object must be reached and reported";
  EXPECT_EQ(hit.link_b, 0);
}

// ── Multiple primitives per attached object (Phase 1) ──────────────────

TEST(AttachedMultiPrimitive, BothPrimitivesOnOneObjectAreChecked) {
  osk::CollisionModel m = hand_model();
  osk::CollisionScratch s;
  s.link_world = {identity(), identity(), identity(), identity()};

  // One object (attach link 1, identity object pose) owning TWO sphere
  // primitives: A at the object origin (clear), B offset +x by 0.3 (near the
  // obstacle). Only the second primitive collides; the object must still fire.
  osk::AttachedModel att;
  append_object(att, 1, identity(),
                {sphere_prim(0.05, identity()), sphere_prim(0.1, translate(0.3, 0.0, 0.0))});
  ASSERT_EQ(att.n_objects, 1U);
  ASSERT_EQ(att.n_primitives, 2U);

  osk::WorldModel w;
  osk::Capsule obs;
  obs.radius = 0.05;
  obs.half_length = 0.0;
  obs.origin = translate(0.35, 0.0, 0.0);  // 0.35 from origin; near primitive B only
  w.capsules = {obs};

  const auto hit = osk::check_attached_world_collision(m, att, s, w, 0.0);
  EXPECT_TRUE(hit.hit) << "the second primitive of the object must be checked and reported";
  EXPECT_EQ(hit.link_a, 0) << "evidence stays object-level (single object index 0)";
  EXPECT_EQ(hit.link_b, 0);
  // B centre at x=0.3 r=0.1; obstacle centre x=0.35 r=0.05 -> centreline 0.05, radii 0.15.
  EXPECT_NEAR(hit.min_distance, -0.1, 1e-9);
}

TEST(AttachedMultiPrimitive, FirstPrimitiveCollisionIsReported) {
  osk::CollisionModel m = hand_model();
  osk::CollisionScratch s;
  s.link_world = {identity(), identity(), identity(), identity()};

  // Now the FIRST primitive is the colliding one; the object still fires and
  // reports the (object-level) index with the minimum distance across prims.
  osk::AttachedModel att;
  append_object(att, 1, identity(),
                {sphere_prim(0.1, identity()), sphere_prim(0.05, translate(1.0, 0.0, 0.0))});

  osk::WorldModel w;
  osk::Capsule obs;
  obs.radius = 0.1;
  obs.half_length = 0.2;
  obs.origin = translate(0.15, 0.0, 0.0);  // overlaps primitive A at origin
  w.capsules = {obs};

  const auto hit = osk::check_attached_world_collision(m, att, s, w, 0.0);
  EXPECT_TRUE(hit.hit);
  EXPECT_EQ(hit.link_a, 0);
  EXPECT_NEAR(hit.min_distance, -0.05, 1e-9);
}

TEST(AttachedMultiPrimitive, MultipleObjectsWithMultiplePrimitives) {
  osk::CollisionModel m = hand_model();
  osk::CollisionScratch s;
  // Object 0 attach link 1 at origin; object 1 attach link 3 at x=2.0.
  s.link_world = {identity(), identity(), identity(), translate(2.0, 0.0, 0.0)};

  osk::AttachedModel att;
  // Object 0: two prims, both clear of the obstacle.
  append_object(att, 1, identity(),
                {sphere_prim(0.05, identity()), sphere_prim(0.05, translate(0.2, 0.0, 0.0))});
  // Object 1: two prims; the SECOND (offset +x 0.3 → world x=2.3) hits.
  append_object(att, 3, identity(),
                {sphere_prim(0.05, identity()), sphere_prim(0.1, translate(0.3, 0.0, 0.0))});
  ASSERT_EQ(att.n_objects, 2U);
  ASSERT_EQ(att.n_primitives, 4U);

  osk::WorldModel w;
  osk::Capsule obs;
  obs.radius = 0.05;
  obs.half_length = 0.0;
  obs.origin = translate(2.35, 0.0, 0.0);  // near object 1's second primitive only
  w.capsules = {obs};

  const auto hit = osk::check_attached_world_collision(m, att, s, w, 0.0);
  EXPECT_TRUE(hit.hit);
  EXPECT_EQ(hit.link_a, 1) << "object 1 (second object, second primitive) must be reported";
  EXPECT_EQ(hit.link_b, 0);
}

TEST(AttachedMultiPrimitive, PoseInLinkComposesWithPoseInObject) {
  // Quaternion composition: object pose_in_link is +90° about Z at the origin;
  // the primitive is offset +x 0.2 in the object frame. The composed world
  // centre is Rz(90)·(0.2,0,0) = (0, 0.2, 0), NOT (0.2, 0, 0).
  osk::CollisionModel m = hand_model();
  osk::CollisionScratch s;
  s.link_world = {identity(), identity(), identity(), identity()};

  const double q = std::sqrt(0.5);
  const osk::Transform obj_pose =
      osk::transform_from_translation_quat(0.0, 0.0, 0.0, 0.0, 0.0, q, q);  // +90° about Z

  osk::AttachedModel att;
  append_object(att, 1, obj_pose, {sphere_prim(0.05, translate(0.2, 0.0, 0.0))});

  // Obstacle at the composed (rotated) location → hit.
  osk::WorldModel hit_world;
  osk::Capsule near_obs;
  near_obs.radius = 0.05;
  near_obs.half_length = 0.0;
  near_obs.origin = translate(0.0, 0.2, 0.0);
  hit_world.capsules = {near_obs};
  const auto hit = osk::check_attached_world_collision(m, att, s, hit_world, 0.0);
  EXPECT_TRUE(hit.hit) << "primitive must land at the composed (rotated) pose (0, 0.2, 0)";
  EXPECT_LT(hit.min_distance, 0.0);

  // Obstacle at the UN-rotated location (0.2, 0, 0) → clear (would only hit if
  // the composition order were wrong).
  osk::WorldModel clear_world;
  osk::Capsule far_obs;
  far_obs.radius = 0.05;
  far_obs.half_length = 0.0;
  far_obs.origin = translate(0.2, 0.0, 0.0);
  clear_world.capsules = {far_obs};
  const auto clear = osk::check_attached_world_collision(m, att, s, clear_world, 0.0);
  EXPECT_FALSE(clear.hit) << "the pre-rotation position must be clear (composition order matters)";
}

// ── transform_from_translation_quat ────────────────────────────────────

TEST(TransformFromQuat, IdentityQuaternionKeepsAxes) {
  const osk::Transform t = osk::transform_from_translation_quat(1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0);
  EXPECT_NEAR(t.r[0], 1.0, 1e-12);
  EXPECT_NEAR(t.r[4], 1.0, 1e-12);
  EXPECT_NEAR(t.r[8], 1.0, 1e-12);
  EXPECT_NEAR(t.t.x, 1.0, 1e-12);
  EXPECT_NEAR(t.t.y, 2.0, 1e-12);
  EXPECT_NEAR(t.t.z, 3.0, 1e-12);
}

TEST(TransformFromQuat, NinetyAboutZMatchesRpy) {
  // Quaternion for +90 deg about Z: (0,0,sin45,cos45).
  const double s = std::sqrt(0.5);
  const osk::Transform q = osk::transform_from_translation_quat(0.0, 0.0, 0.0, 0.0, 0.0, s, s);
  const osk::Transform r = osk::transform_from_xyz_rpy(0.0, 0.0, 0.0, 0.0, 0.0, kPi / 2.0);
  for (int i = 0; i < 9; ++i) {
    EXPECT_NEAR(q.r[i], r.r[i], 1e-9) << "rotation element " << i;
  }
}

TEST(TransformFromQuat, NonUnitQuaternionIsNormalised) {
  // Same rotation as a unit +90-about-Z quaternion, scaled by 5.
  const double s = std::sqrt(0.5) * 5.0;
  const osk::Transform q = osk::transform_from_translation_quat(0.0, 0.0, 0.0, 0.0, 0.0, s, s);
  EXPECT_NEAR(q.r[0], 0.0, 1e-9);
  EXPECT_NEAR(q.r[1], -1.0, 1e-9);
  EXPECT_NEAR(q.r[3], 1.0, 1e-9);
}

// ── ingest_attached_objects (fixed-capacity, fail-closed) ──────────────

namespace {

osk::AttachedModel presized_model(std::size_t max_objects, std::size_t max_prims,
                                  std::size_t max_touch) {
  osk::AttachedModel m;
  m.objects.assign(max_objects, osk::AttachedObject{});
  m.primitives.assign(max_prims, osk::AttachedPrimitive{});
  m.touch_links.assign(max_touch, 0);
  m.n_objects = 0;
  m.n_primitives = 0;
  return m;
}

osk::AttachedPrimitiveInput sphere_prim_input(double radius) {
  osk::AttachedPrimitiveInput p;
  p.kind = osk::AttachedShapeKind::kSphere;
  p.radius = radius;
  p.half_length = 0.0;
  return p;
}

osk::AttachedObjectInput sphere_input(const std::string& attach,
                                      const std::vector<std::string>& touch) {
  osk::AttachedObjectInput in;
  in.attach_link = attach;
  in.touch_links = touch;
  in.primitives = {sphere_prim_input(0.05)};
  return in;
}

const std::vector<std::string> kLinkNames = {"base", "hand", "finger_left", "finger_right", "arm"};

}  // namespace

TEST(IngestAttached, ResolvesLinkAndTouchNames) {
  auto model = presized_model(4, 4, 8);
  const std::vector<osk::AttachedObjectInput> in = {
      sphere_input("hand", {"finger_left", "finger_right"})};
  const auto st = osk::ingest_attached_objects(in, kLinkNames, 4, 4, 8, model);
  EXPECT_EQ(st, osk::AttachIngestStatus::kOk);
  ASSERT_EQ(model.n_objects, 1U);
  ASSERT_EQ(model.n_primitives, 1U);
  EXPECT_EQ(model.objects[0].attach_link, 1);  // "hand"
  ASSERT_EQ(model.objects[0].prim_count, 1);
  EXPECT_EQ(model.objects[0].prim_first, 0);
  ASSERT_EQ(model.objects[0].touch_count, 2);
  EXPECT_EQ(model.touch_links[0], 2);  // finger_left
  EXPECT_EQ(model.touch_links[1], 3);  // finger_right
}

TEST(IngestAttached, UnknownAttachLinkFailsClosed) {
  auto model = presized_model(4, 4, 8);
  const std::vector<osk::AttachedObjectInput> in = {sphere_input("gripper_typo", {"finger_left"})};
  const auto st = osk::ingest_attached_objects(in, kLinkNames, 4, 4, 8, model);
  EXPECT_EQ(st, osk::AttachIngestStatus::kUnknownLink);
  EXPECT_EQ(model.n_objects, 0U) << "an unknown attach link must leave nothing checkable";
}

TEST(IngestAttached, UnknownTouchLinkFailsClosed) {
  auto model = presized_model(4, 4, 8);
  const std::vector<osk::AttachedObjectInput> in = {sphere_input("hand", {"third_finger"})};
  const auto st = osk::ingest_attached_objects(in, kLinkNames, 4, 4, 8, model);
  EXPECT_EQ(st, osk::AttachIngestStatus::kUnknownLink);
  EXPECT_EQ(model.n_objects, 0U);
}

TEST(IngestAttached, ObjectOverflowFailsClosed) {
  auto model = presized_model(1, 4, 8);
  const std::vector<osk::AttachedObjectInput> in = {sphere_input("hand", {"finger_left"}),
                                                    sphere_input("arm", {"finger_right"})};
  const auto st = osk::ingest_attached_objects(in, kLinkNames, 1, 4, 8, model);
  EXPECT_EQ(st, osk::AttachIngestStatus::kOverflow);
  EXPECT_EQ(model.n_objects, 0U);
}

TEST(IngestAttached, PrimitiveOverflowFailsClosed) {
  // One object carrying three primitives against a total-primitive cap of 2.
  auto model = presized_model(4, 2, 8);
  osk::AttachedObjectInput multi;
  multi.attach_link = "hand";
  multi.touch_links = {};
  multi.primitives = {sphere_prim_input(0.05), sphere_prim_input(0.04), sphere_prim_input(0.03)};
  const auto st = osk::ingest_attached_objects({multi}, kLinkNames, 4, 2, 8, model);
  EXPECT_EQ(st, osk::AttachIngestStatus::kOverflow);
  EXPECT_EQ(model.n_objects, 0U);
  EXPECT_EQ(model.n_primitives, 0U);
}

TEST(IngestAttached, TotalPrimitiveOverflowAcrossObjectsFailsClosed) {
  // Two objects with two primitives each (4 total) against a cap of 3.
  auto model = presized_model(4, 3, 8);
  osk::AttachedObjectInput a;
  a.attach_link = "hand";
  a.primitives = {sphere_prim_input(0.05), sphere_prim_input(0.04)};
  osk::AttachedObjectInput b;
  b.attach_link = "arm";
  b.primitives = {sphere_prim_input(0.05), sphere_prim_input(0.04)};
  const auto st = osk::ingest_attached_objects({a, b}, kLinkNames, 4, 3, 8, model);
  EXPECT_EQ(st, osk::AttachIngestStatus::kOverflow);
  EXPECT_EQ(model.n_objects, 0U);
}

TEST(IngestAttached, TouchLinkOverflowFailsClosed) {
  auto model = presized_model(4, 4, 1);
  const std::vector<osk::AttachedObjectInput> in = {
      sphere_input("hand", {"finger_left", "finger_right"})};
  const auto st = osk::ingest_attached_objects(in, kLinkNames, 4, 4, 1, model);
  EXPECT_EQ(st, osk::AttachIngestStatus::kOverflow);
  EXPECT_EQ(model.n_objects, 0U);
}

TEST(IngestAttached, MalformedRadiusFailsClosed) {
  auto model = presized_model(4, 4, 8);
  osk::AttachedObjectInput bad = sphere_input("hand", {"finger_left"});
  bad.primitives[0].radius = 0.0;  // non-positive radius
  const auto st = osk::ingest_attached_objects({bad}, kLinkNames, 4, 4, 8, model);
  EXPECT_EQ(st, osk::AttachIngestStatus::kMalformed);
  EXPECT_EQ(model.n_objects, 0U);
}

TEST(IngestAttached, EmptyPrimitivesObjectFailsClosed) {
  auto model = presized_model(4, 4, 8);
  osk::AttachedObjectInput no_geom;
  no_geom.attach_link = "hand";
  no_geom.touch_links = {"finger_left"};
  no_geom.primitives = {};  // an object with no geometry is not safe to ignore
  const auto st = osk::ingest_attached_objects({no_geom}, kLinkNames, 4, 4, 8, model);
  EXPECT_EQ(st, osk::AttachIngestStatus::kMalformed);
  EXPECT_EQ(model.n_objects, 0U);
}

TEST(IngestAttached, MultiplePrimitivesPerObjectResolve) {
  auto model = presized_model(4, 8, 8);
  osk::AttachedObjectInput cap;
  cap.attach_link = "hand";
  cap.touch_links = {"finger_left"};
  osk::AttachedPrimitiveInput c;
  c.kind = osk::AttachedShapeKind::kCapsule;
  c.radius = 0.03;
  c.half_length = 0.06;
  osk::AttachedPrimitiveInput box;
  box.kind = osk::AttachedShapeKind::kBox;
  box.half_extents = {0.02, 0.03, 0.04};
  cap.primitives = {c, box};  // one object, two primitives

  const auto st = osk::ingest_attached_objects({cap}, kLinkNames, 4, 8, 8, model);
  EXPECT_EQ(st, osk::AttachIngestStatus::kOk);
  ASSERT_EQ(model.n_objects, 1U);
  ASSERT_EQ(model.n_primitives, 2U);
  ASSERT_EQ(model.objects[0].prim_count, 2);
  EXPECT_EQ(model.objects[0].prim_first, 0);
  EXPECT_EQ(model.primitives[0].kind, osk::AttachedShapeKind::kCapsule);
  EXPECT_NEAR(model.primitives[0].half_length, 0.06, 1e-12);
  EXPECT_EQ(model.primitives[1].kind, osk::AttachedShapeKind::kBox);
  EXPECT_NEAR(model.primitives[1].half_extents.z, 0.04, 1e-12);
  EXPECT_EQ(model.objects[0].touch_count, 1);
}

TEST(IngestAttached, TwoObjectsFlattenPrimitiveSlicesInOrder) {
  auto model = presized_model(4, 8, 8);
  osk::AttachedObjectInput a;
  a.attach_link = "hand";
  a.primitives = {sphere_prim_input(0.05), sphere_prim_input(0.04)};
  osk::AttachedObjectInput b;
  b.attach_link = "arm";
  b.primitives = {sphere_prim_input(0.03)};
  const auto st = osk::ingest_attached_objects({a, b}, kLinkNames, 4, 8, 8, model);
  EXPECT_EQ(st, osk::AttachIngestStatus::kOk);
  ASSERT_EQ(model.n_objects, 2U);
  ASSERT_EQ(model.n_primitives, 3U);
  EXPECT_EQ(model.objects[0].prim_first, 0);
  EXPECT_EQ(model.objects[0].prim_count, 2);
  EXPECT_EQ(model.objects[1].prim_first, 2);
  EXPECT_EQ(model.objects[1].prim_count, 1);
  EXPECT_EQ(model.objects[1].attach_link, 4);  // "arm"
}

TEST(IngestAttached, EmptyInputIsOkAndEmpty) {
  auto model = presized_model(4, 4, 8);
  const auto st = osk::ingest_attached_objects({}, kLinkNames, 4, 4, 8, model);
  EXPECT_EQ(st, osk::AttachIngestStatus::kOk);
  EXPECT_EQ(model.n_objects, 0U);
  EXPECT_EQ(model.n_primitives, 0U);
}

// ── Evidence coherence: the reported pair and the reported distance ────
//
// A `CollisionHit` is the kernel's E-stop evidence: `link_a` / `link_b` name
// the geometry that tripped the check and `min_distance` is the number the
// FailureTrigger and the operator log quote for it. Those three fields must
// describe ONE pair. They used to be sampled from two different places — the
// identity from the first pair to trip the gate, the distance from the
// sweep-wide minimum over every pair the check touched, including pairs the
// gate deliberately exempted (an attached payload's attach-time contact
// baseline). A payload resting on a shelf then reported its own uncleared
// occupancy residue's ~-40 mm as if it were the support contact's ~-0.4 mm,
// and downstream diagnosis chased a penetration that never existed.
//
// The contract pinned below: on a hit the evidence describes the DEEPEST
// pair that actually tripped the gate; the sweep-wide minimum is carried
// separately in `sweep_min_distance`. Reporting only — the tests at the end
// of this section pin that the stop/no-stop decision itself is untouched.

namespace {

// Base-frame transform of an occupied cell, recovered from the linear voxel
// index the evidence reports (`idx = x + sx*(y + sy*z)`) — lets a test
// recompute the distance of the cell the kernel named and compare it with the
// distance the kernel reported.
osk::Transform voxel_transform_at(const osk::VoxelGrid& grid, int linear_index) {
  const int ix = linear_index % grid.sx;
  const int iy = (linear_index / grid.sx) % grid.sy;
  const int iz = linear_index / (grid.sx * grid.sy);
  osk::Transform t;
  t.t = {grid.origin.x + (ix + 0.5) * grid.resolution, grid.origin.y + (iy + 0.5) * grid.resolution,
         grid.origin.z + (iz + 0.5) * grid.resolution};
  return t;
}

// One box primitive (posed in its object's frame). A box payload gives the
// voxel check an exact SAT overlap depth, so a test can place a deep
// "payload residue" cell and a shallow support cell with hand-computed
// distances.
osk::AttachedPrimitive box_prim(double hx, double hy, double hz,
                                const osk::Transform& pose_in_object = osk::Transform{}) {
  osk::AttachedPrimitive p;
  p.kind = osk::AttachedShapeKind::kBox;
  p.half_extents = {hx, hy, hz};
  p.pose_in_object = pose_in_object;
  return p;
}

const osk::Vec3 kVoxelHalf{0.05, 0.05, 0.05};  // make_grid()'s 0.1 m cells

// ── Support-contact witness fixture (ADR-0092 D6) ───────────────────────────
//
// The 2026-08-13 baguette run, to the tenth of a millimetre. A 300x50x50 mm
// payload rests on a counter with its bottom face at z = 0. The counter is
// mapped at 25 mm resolution and the lattice phase puts the surface cell's
// centre 3.2 mm ABOVE that face, so the cell cube's top face is 15.7 mm above
// it: the check reads -15.70 mm of penetration for a contact MuJoCo measured
// at -0.87..-1.37 mm. That inflation is pure discretisation, and the witness
// exists to tolerate exactly it — and nothing else.

constexpr int kSupportN = 24;
constexpr double kSupportRes = 0.025;
// Attested physical depth: the deepest of the six real MuJoCo counter-payload
// contacts in that run.
constexpr double kAttestedPenetration = 0.00137;

int support_index(int x, int y, int z) { return x + kSupportN * (y + kSupportN * z); }

osk::VoxelGrid support_grid(const std::vector<std::uint8_t>& occ) {
  osk::VoxelGrid g;
  // Phased so cell (12, 12, 12) is centred at (0, 0, +0.0032).
  g.origin = {-0.3125, -0.3125, -0.3093};
  g.resolution = kSupportRes;
  g.sx = kSupportN;
  g.sy = kSupportN;
  g.sz = kSupportN;
  g.occupancy = occ.data();
  g.attached_contact_tolerance = 0.001;  // physical slack, not a quantisation fudge
  return g;
}

// The carried baguette: one box spanning z in [0, 0.05], attached at link 1.
void append_baguette(osk::AttachedModel& att, double patch_radius_m) {
  append_object(att, 1, identity(), {box_prim(0.15, 0.025, 0.025, translate(0.0, 0.0, 0.025))});
  osk::AttachedObject& o = att.objects.back();
  o.has_support_witness = patch_radius_m > 0.0;
  o.support_point = {0.0, 0.0, 0.0};   // the payload's own support face
  o.support_normal = {0.0, 0.0, 1.0};  // counter -> payload
  o.support_patch_radius = patch_radius_m;
  o.support_max_penetration = kAttestedPenetration;
}

}  // namespace

TEST(SupportContactWitness, QuantisationInflatedSupportContactDoesNotStopTheRobot) {
  // The exact false stop this witness exists to remove.
  osk::CollisionModel m = hand_model();
  osk::CollisionScratch s;
  s.link_world = {identity(), identity(), identity(), identity()};
  osk::AttachedModel att;
  append_baguette(att, 0.10);

  std::vector<std::uint8_t> occ(kSupportN * kSupportN * kSupportN, 0);
  occ[static_cast<std::size_t>(support_index(12, 12, 12))] = 1;
  auto grid = support_grid(occ);

  // First: what the kernel actually sees without any attestation — a 15.70 mm
  // penetration, which is exactly what E-stopped the baguette run.
  const auto unattested = osk::check_attached_voxel_collision(m, att, s, grid, 0.0);
  ASSERT_TRUE(unattested.hit) << "no witness is live yet, so this must still stop";
  EXPECT_NEAR(unattested.min_distance, -0.0157, 1e-9);

  // Now arm the witness. Same geometry, same cell, same -15.70 mm reading —
  // but the depth that decides is measured against the attested support plane,
  // where it is 3.2 mm, inside the 12.5 mm cube half-width the lattice can
  // legitimately produce.
  grid.support_witness_live = 0x1;
  EXPECT_EQ(osk::update_support_contact_witnesses(att, s, grid, 0x1, 0.0), 0x1)
      << "the payload is still on its attested support, so the witness stays live";
  EXPECT_FALSE(osk::check_attached_voxel_collision(m, att, s, grid, 0.0).hit);
}

TEST(SupportContactWitness, NoAttestationMeansNoExemption) {
  // Fail-closed: a payload with no witness behaves exactly as it does today.
  osk::CollisionModel m = hand_model();
  osk::CollisionScratch s;
  s.link_world = {identity(), identity(), identity(), identity()};
  osk::AttachedModel att;
  append_baguette(att, 0.0);  // patch 0 => has_support_witness == false
  ASSERT_FALSE(att.objects[0].has_support_witness);

  std::vector<std::uint8_t> occ(kSupportN * kSupportN * kSupportN, 0);
  occ[static_cast<std::size_t>(support_index(12, 12, 12))] = 1;
  auto grid = support_grid(occ);
  // Even with the live bit wrongly set, an object that attests nothing is
  // exempt from nothing, and the latch drops the bit on the next refresh.
  grid.support_witness_live = 0x1;
  EXPECT_TRUE(osk::check_attached_voxel_collision(m, att, s, grid, 0.0).hit);
  EXPECT_EQ(osk::update_support_contact_witnesses(att, s, grid, 0x1, 0.0), 0x0);
}

TEST(SupportContactWitness, DeepeningIntoTheSupportStillStops) {
  // The payload drives 40 mm further into the counter. The cell's height above
  // the attested support face rises millimetre for millimetre with the sink,
  // and past the lattice envelope it stops the robot.
  //
  // 40 mm rather than the 20 mm this test used before ADR-0097's Second
  // Amendment (hazard log Entry 012, "Calibration 2026-08-15"): the envelope
  // gained one voxel, so it is now 12.5 + 1.37 + 1 + 25 = 39.87 mm and the sink
  // has to put the cell past THAT to prove deepening still stops. At 40 mm of
  // sink the cell sits 43.2 mm above the attested plane.
  osk::CollisionModel m = hand_model();
  osk::CollisionScratch s;
  s.link_world = {identity(), identity(), identity(), identity()};
  osk::AttachedModel att;
  append_baguette(att, 0.10);

  std::vector<std::uint8_t> occ(kSupportN * kSupportN * kSupportN, 0);
  occ[static_cast<std::size_t>(support_index(12, 12, 12))] = 1;
  auto grid = support_grid(occ);
  grid.support_witness_live = 0x1;
  ASSERT_FALSE(osk::check_attached_voxel_collision(m, att, s, grid, 0.0).hit);

  // 20 mm of sink is 23.2 mm above the plane — inside the widened envelope, and
  // that is the calibration doing exactly what it was approved to do.
  att.objects[0].pose_in_link = translate(0.0, 0.0, -0.02);
  EXPECT_FALSE(osk::check_attached_voxel_collision(m, att, s, grid, 0.0).hit);

  att.objects[0].pose_in_link = translate(0.0, 0.0, -0.04);
  const auto hit = osk::check_attached_voxel_collision(m, att, s, grid, 0.0);
  EXPECT_TRUE(hit.hit);
  EXPECT_EQ(hit.link_b, support_index(12, 12, 12));
  EXPECT_LT(hit.min_distance, 0.0);
}

TEST(SupportContactWitness, SeparationKillsTheExemptionAndRecontactIsANewViolation) {
  // Hysteresis. Lift the payload clear, and the attestation dies with the
  // contact it described; lowering back onto the same cell is a fresh contact
  // that only a fresh attestation could ever exempt.
  osk::CollisionModel m = hand_model();
  osk::CollisionScratch s;
  s.link_world = {identity(), identity(), identity(), identity()};
  osk::AttachedModel att;
  append_baguette(att, 0.10);

  std::vector<std::uint8_t> occ(kSupportN * kSupportN * kSupportN, 0);
  occ[static_cast<std::size_t>(support_index(12, 12, 12))] = 1;
  auto grid = support_grid(occ);

  std::uint8_t live = 0x1;
  grid.support_witness_live = live;
  ASSERT_FALSE(osk::check_attached_voxel_collision(m, att, s, grid, 0.0).hit);

  // Lift 10 cm: nothing the witness would exempt is in contact any more.
  att.objects[0].pose_in_link = translate(0.0, 0.0, 0.10);
  live = osk::update_support_contact_witnesses(att, s, grid, live, 0.0);
  EXPECT_EQ(live, 0x0);

  // Come back down onto the very same cell. The latch never re-arms itself.
  att.objects[0].pose_in_link = identity();
  live = osk::update_support_contact_witnesses(att, s, grid, live, 0.0);
  EXPECT_EQ(live, 0x0) << "only a fresh World State attestation may re-arm the witness";
  grid.support_witness_live = live;
  EXPECT_TRUE(osk::check_attached_voxel_collision(m, att, s, grid, 0.0).hit);
}

TEST(SupportContactWitness, NewContactOutsideThePatchStillStopsMidCarry) {
  // A live support witness must not license a contact against anything else.
  // The wall cell sits 150 mm out — past the 100 mm attested patch plus the
  // cube's circumradius — while the support cell underneath stays exempt.
  osk::CollisionModel m = hand_model();
  osk::CollisionScratch s;
  s.link_world = {identity(), identity(), identity(), identity()};
  osk::AttachedModel att;
  append_baguette(att, 0.10);

  std::vector<std::uint8_t> occ(kSupportN * kSupportN * kSupportN, 0);
  occ[static_cast<std::size_t>(support_index(12, 12, 12))] = 1;
  occ[static_cast<std::size_t>(support_index(18, 12, 12))] = 1;  // 150 mm along +x
  auto grid = support_grid(occ);
  grid.support_witness_live = 0x1;

  const auto hit = osk::check_attached_voxel_collision(m, att, s, grid, 0.0);
  ASSERT_TRUE(hit.hit);
  EXPECT_EQ(hit.link_b, support_index(18, 12, 12))
      << "the stop must name the unattested cell, never the exempt support cell";
}

TEST(SupportContactWitness, ProtrusionInsideThePatchStillStops) {
  // Inside the patch, but genuinely higher than the attested support face: an
  // object standing on the counter under the payload. 50 mm of solid above the
  // support plane is not discretisation, and it stops the robot.
  //
  // Two cells up rather than one, since ADR-0097's Second Amendment (hazard log
  // Entry 012, "Calibration 2026-08-15"): the height envelope gained one voxel
  // precisely so that ADJACENT CO-PLANAR structure one voxel up — a raised edge
  // or a neighbouring stack on the same surface — stops reading as a protrusion.
  // That one-voxel band is now exempt by decision (see
  // `…TheRound8CoplanarCellIsInsideTheWidenedEnvelope`), so what proves the
  // bound still bites is solid a further voxel above it.
  osk::CollisionModel m = hand_model();
  osk::CollisionScratch s;
  s.link_world = {identity(), identity(), identity(), identity()};
  osk::AttachedModel att;
  append_baguette(att, 0.10);

  std::vector<std::uint8_t> occ(kSupportN * kSupportN * kSupportN, 0);
  occ[static_cast<std::size_t>(support_index(12, 12, 12))] = 1;
  occ[static_cast<std::size_t>(support_index(12, 12, 14))] = 1;  // stacked two cells up
  auto grid = support_grid(occ);
  grid.support_witness_live = 0x1;

  const auto hit = osk::check_attached_voxel_collision(m, att, s, grid, 0.0);
  ASSERT_TRUE(hit.hit);
  EXPECT_EQ(hit.link_b, support_index(12, 12, 14));
}

// ── the 2026-08-15 battery's height-envelope calibration ────────────────────
//
// `spark:~/openral-runs/2026-08-15-baguette-battery`, recorded as
// "Calibration 2026-08-15" on hazard-log Entry 012 (HZ-0092-1) and approved by
// the maintainer alongside ADR-0097's Second Amendment. The battery refuted the
// suspected "sliding" class outright — run 4 moved the payload 10.2 mm during
// the stop and the lateral patch gate (135.7 mm) never came near the 84-86 mm
// actual offset — and named the real one: cells of ADJACENT CO-PLANAR STRUCTURE
// (a raised edge, a neighbouring stack on the same support surface) sit
// approximately ONE VOXEL above the attested plane while the payload is still
// in genuine, continuing support contact. Round-8 r2 measured +42.9 mm against
// a ~15-19 mm envelope: an excess of +24.4 mm, one 25 mm voxel.
//
// The bound therefore gains a fourth additive term, one full voxel, INSIDE the
// existing lateral patch (the patch radius is untouched — this is height, not
// reach):
//
//   half_resolution·(|n.x|+|n.y|+|n.z|) + attested depth + slack + resolution
//
// Fixture: the same 24³ / 25 mm lattice, re-phased so a chosen cell layer lands
// at a chosen height above the attested plane, and the attested depth set to
// the run's own 5 mm. The envelope is then
//   12.5 + 5.0 + 1.0 + 25.0 = 43.5 mm exactly,
// so r2's +42.9 mm clears it by 0.6 mm — and nothing rounder would.

namespace {

constexpr double kBatteryPenetration = 0.005;  // r2's attested physical depth
constexpr double kBatteryPatchRadius = 0.114;  // the run's patch: 135.65 mm gate
constexpr double kBatteryEnvelope = 0.0435;    // 12.5 + 5 + 1 + 25 mm
constexpr double kR2CoplanarHeight = 0.0429;   // round-8 r2's measured excess
constexpr double kR2LateralOffset = 0.085;     // inside the run's 84-86 mm band

// The battery lattice, phased so cell layer `iz = 14` centres exactly
// `top_height` above the attested plane (z = 0, the payload's own support face)
// and column `ix = 15` centres exactly `kR2LateralOffset` out along +x. The
// support cell the witness measures against is then layer 12, two voxels (50 mm)
// below the probed layer, in column 12.
osk::VoxelGrid battery_grid(const std::vector<std::uint8_t>& occ, double top_height) {
  osk::VoxelGrid g;
  g.origin = {kR2LateralOffset - 15.5 * kSupportRes, -0.3125, top_height - 14.5 * kSupportRes};
  g.resolution = kSupportRes;
  g.sx = kSupportN;
  g.sy = kSupportN;
  g.sz = kSupportN;
  g.occupancy = occ.data();
  g.attached_contact_tolerance = 0.001;
  return g;
}

void append_battery_baguette(osk::AttachedModel& att) {
  append_baguette(att, kBatteryPatchRadius);
  att.objects.back().support_max_penetration = kBatteryPenetration;
}

// Base-frame centre of cell (ix, iy, iz) on a battery lattice.
osk::Vec3 battery_center(const osk::VoxelGrid& g, int ix, int iy, int iz) {
  return osk::Vec3{g.origin.x + (ix + 0.5) * g.resolution, g.origin.y + (iy + 0.5) * g.resolution,
                   g.origin.z + (iz + 0.5) * g.resolution};
}

}  // namespace

TEST(SupportContactWitness, TheRound8CoplanarCellIsInsideTheWidenedEnvelope) {
  // Round-8 r2, to the tenth of a millimetre. The payload is in continuing
  // support contact (the cell two layers down is occupied, exempt, and keeps the
  // latch alive) and the co-planar cell 85 mm out sits +42.9 mm above the
  // attested plane — one voxel over the pre-calibration envelope, 0.6 mm inside
  // the calibrated one. Before the calibration this stopped the flagship run.
  osk::CollisionModel m = hand_model();
  osk::CollisionScratch s;
  s.link_world = {identity(), identity(), identity(), identity()};
  osk::AttachedModel att;
  append_battery_baguette(att);

  std::vector<std::uint8_t> occ(kSupportN * kSupportN * kSupportN, 0);
  occ[static_cast<std::size_t>(support_index(12, 12, 12))] = 1;  // the support surface
  occ[static_cast<std::size_t>(support_index(15, 12, 14))] = 1;  // the co-planar cell
  auto grid = battery_grid(occ, kR2CoplanarHeight);
  grid.support_witness_live = 0x1;

  // The fixture is the arithmetic, so state it before leaning on it: the cell is
  // where the run said it was, laterally and vertically.
  const osk::Vec3 coplanar = battery_center(grid, 15, 12, 14);
  ASSERT_NEAR(coplanar.z, kR2CoplanarHeight, 1e-12);
  ASSERT_NEAR(coplanar.x, kR2LateralOffset, 1e-12);
  ASSERT_LT(kR2CoplanarHeight, kBatteryEnvelope) << "0.6 mm of headroom, and not a millimetre more";
  ASSERT_LT(kBatteryEnvelope - kR2CoplanarHeight, 0.001);

  EXPECT_EQ(osk::update_support_contact_witnesses(att, s, grid, 0x1, 0.0), 0x1)
      << "support contact persists; the witness is live throughout";
  EXPECT_FALSE(osk::check_attached_voxel_collision(m, att, s, grid, 0.0).hit)
      << "the co-planar cell one voxel up is inside the calibrated envelope";

  // The bound itself, probed directly on either side of 43.5 mm — the fixture
  // passes because the arithmetic says so, not because it was rounded into
  // passing.
  const osk::AttachedObject& obj = att.objects[0];
  const double slack = grid.attached_contact_tolerance;
  EXPECT_TRUE(osk::support_contact_exempts(
      obj, identity(), osk::Vec3{kR2LateralOffset, 0.0, kBatteryEnvelope}, kSupportRes, slack));
  EXPECT_FALSE(osk::support_contact_exempts(
      obj, identity(), osk::Vec3{kR2LateralOffset, 0.0, kBatteryEnvelope + 1e-9}, kSupportRes,
      slack));
}

TEST(SupportContactWitness, SolidAboveTheCoplanarBandStillStops) {
  // The other side of the same calibration. +55 mm is 11.5 mm past the
  // calibrated envelope — a genuine protrusion standing on the support surface,
  // not the one-voxel co-planar band the battery measured — and it stops the
  // robot with the witness live and support contact unbroken.
  osk::CollisionModel m = hand_model();
  osk::CollisionScratch s;
  s.link_world = {identity(), identity(), identity(), identity()};
  osk::AttachedModel att;
  append_battery_baguette(att);

  std::vector<std::uint8_t> occ(kSupportN * kSupportN * kSupportN, 0);
  occ[static_cast<std::size_t>(support_index(12, 12, 12))] = 1;
  occ[static_cast<std::size_t>(support_index(15, 12, 14))] = 1;
  auto grid = battery_grid(occ, 0.055);
  grid.support_witness_live = 0x1;
  ASSERT_NEAR(battery_center(grid, 15, 12, 14).z, 0.055, 1e-12);
  ASSERT_GT(0.055, kBatteryEnvelope);

  EXPECT_EQ(osk::update_support_contact_witnesses(att, s, grid, 0x1, 0.0), 0x1);
  const auto hit = osk::check_attached_voxel_collision(m, att, s, grid, 0.0);
  ASSERT_TRUE(hit.hit) << "the calibration widened the envelope; it did not remove it";
  EXPECT_EQ(hit.link_b, support_index(15, 12, 14))
      << "the stop must name the protrusion, never the exempt support cell";
  EXPECT_FALSE(osk::support_contact_exempts(
      att.objects[0], identity(), osk::Vec3{kR2LateralOffset, 0.0, 0.055}, kSupportRes, 0.001));
}

TEST(SupportContactWitness, TheWidenedEnvelopeIsHeightOnlyNotReach) {
  // The half of the calibration that did NOT move. The extra voxel applies only
  // within the already-bounded lateral patch: a cell at the same +42.9 mm but
  // past the attested patch (padded by the cube circumradius) is a contact
  // nobody attested and still stops, exactly as
  // `NewContactOutsideThePatchStillStopsMidCarry` requires mid-carry.
  osk::AttachedModel att;
  append_battery_baguette(att);
  const osk::AttachedObject& obj = att.objects[0];
  const double reach = kBatteryPatchRadius + 0.5 * kSupportRes * 1.7320508075688772;
  ASSERT_NEAR(reach, 0.13565, 1e-5) << "the run's own 135.7 mm lateral gate";

  EXPECT_TRUE(osk::support_contact_exempts(
      obj, identity(), osk::Vec3{reach - 0.001, 0.0, kR2CoplanarHeight}, kSupportRes, 0.001));
  EXPECT_FALSE(osk::support_contact_exempts(
      obj, identity(), osk::Vec3{reach + 0.001, 0.0, kR2CoplanarHeight}, kSupportRes, 0.001));
}

TEST(SupportContactWitness, ExemptCellsStayOutOfTheReportedDistance) {
  // Evidence coherence, unchanged by the new exemption: the exempt support
  // cell reaches sweep_min_distance only, never min_distance.
  osk::CollisionModel m = hand_model();
  osk::CollisionScratch s;
  s.link_world = {identity(), identity(), identity(), identity()};
  osk::AttachedModel att;
  append_baguette(att, 0.10);

  std::vector<std::uint8_t> occ(kSupportN * kSupportN * kSupportN, 0);
  occ[static_cast<std::size_t>(support_index(12, 12, 12))] = 1;
  occ[static_cast<std::size_t>(support_index(18, 12, 12))] = 1;
  auto grid = support_grid(occ);
  grid.support_witness_live = 0x1;

  const auto hit = osk::check_attached_voxel_collision(m, att, s, grid, 0.0);
  ASSERT_TRUE(hit.hit);
  EXPECT_NEAR(hit.min_distance, -0.0125, 1e-9) << "the wall cell's own 12.5 mm overlap";
  EXPECT_NEAR(hit.sweep_min_distance, -0.0157, 1e-9)
      << "the exempt support cell's 15.7 mm stays available as a diagnostic only";
}

TEST(SupportContactWitness, LostOccupancyMapDropsEveryWitness) {
  // Fail-closed: with no map there is no way to confirm the payload is still on
  // its support, so no exemption may survive the refresh.
  osk::CollisionModel m = hand_model();
  osk::CollisionScratch s;
  s.link_world = {identity(), identity(), identity(), identity()};
  osk::AttachedModel att;
  append_baguette(att, 0.10);
  (void)m;

  std::vector<std::uint8_t> occ(kSupportN * kSupportN * kSupportN, 0);
  auto grid = support_grid(occ);
  grid.occupancy = nullptr;
  EXPECT_EQ(osk::update_support_contact_witnesses(att, s, grid, 0xFF, 0.0), 0x0);
}

// ── the partition against the Layer-2 payload clearing ──────────────────────
//
// The bridge that owns occupancy (`openral_octomap_bridge`) removes the
// payload's own cells from the published grid, and the witness above proves
// the payload is still on its support by finding an OCCUPIED cell it would
// exempt still touching the payload. A resting payload's bottom cell layer IS
// the counter's top cell layer, so the two mechanisms fight over the same
// cells: whichever wins decides whether the robot can carry anything.
//
// The division of the cells is stated from this side here, in the kernel's own
// terms and on the kernel's own fixture, so a change to either half has to
// come past it: the payload's silhouette ABOVE the attested plane belongs to
// the clearing, the cells inside the attested patch belong to the witness.

TEST(SupportContactWitness, ThePartitionedClearingLeavesTheWitnessItsEvidence) {
  // What the bridge must publish for a resting payload: the payload's own cell
  // one layer up is gone, the attested support cell underneath is not. The
  // witness then still measures against a real cell, stays live, and exempts
  // exactly it.
  osk::CollisionModel m = hand_model();
  osk::CollisionScratch s;
  s.link_world = {identity(), identity(), identity(), identity()};
  osk::AttachedModel att;
  append_baguette(att, 0.10);

  // The pre-clearing map, for contrast: the payload's own uncleared silhouette
  // at +53.2 mm above the attested plane is not support contact, and stops the
  // robot — the 2026-08-14 acceptance-run failure this clearing exists to fix.
  // Two layers up rather than one since the 2026-08-15 height calibration
  // (hazard log Entry 012): the +28.2 mm layer is now inside the widened
  // envelope, so it is the layer above it that still makes the contrast.
  std::vector<std::uint8_t> before(kSupportN * kSupportN * kSupportN, 0);
  before[static_cast<std::size_t>(support_index(12, 12, 12))] = 1;
  before[static_cast<std::size_t>(support_index(12, 12, 14))] = 1;
  auto pre = support_grid(before);
  pre.support_witness_live = 0x1;
  const auto pre_hit = osk::check_attached_voxel_collision(m, att, s, pre, 0.0);
  ASSERT_TRUE(pre_hit.hit);
  EXPECT_EQ(pre_hit.link_b, support_index(12, 12, 14));

  // After the partitioned clearing: the silhouette cell cleared, the support
  // cell withheld.
  std::vector<std::uint8_t> after(kSupportN * kSupportN * kSupportN, 0);
  after[static_cast<std::size_t>(support_index(12, 12, 12))] = 1;
  auto post = support_grid(after);
  post.support_witness_live = 0x1;
  EXPECT_EQ(osk::update_support_contact_witnesses(att, s, post, 0x1, 0.0), 0x1)
      << "the withheld support cell is what keeps the witness alive";
  EXPECT_FALSE(osk::check_attached_voxel_collision(m, att, s, post, 0.0).hit);
}

TEST(SupportContactWitness, ClearingTheAttestedPatchKillsTheWitnessAndTheReturningCellStops) {
  // The 2026-08-14 defect, from the kernel's side, and why the bridge must not
  // clear the attested patch. Reproduced 2/2 on baguette+counter and
  // cup+island: the clearing takes the counter with the payload, the witness
  // loses the only evidence it is allowed to consult and declares separation
  // (`support_witness_separated live=0x0 was=0x1`) while the payload has not
  // moved a micrometre — and the moment a support cell falls back outside the
  // clearing reach and returns to the map, the unchanged physical contact
  // E-stops, unexempted.
  osk::CollisionModel m = hand_model();
  osk::CollisionScratch s;
  s.link_world = {identity(), identity(), identity(), identity()};
  osk::AttachedModel att;
  append_baguette(att, 0.10);

  std::vector<std::uint8_t> occ(kSupportN * kSupportN * kSupportN, 0);
  auto cleared = support_grid(occ);  // the support cell cleared away with the payload
  cleared.support_witness_live = 0x1;
  const std::uint8_t live = osk::update_support_contact_witnesses(att, s, cleared, 0x1, 0.0);
  EXPECT_EQ(live, 0x0) << "no occupied cell left to attest to a contact that never ended";

  // One frame later the same cell is in the map again. The latch never re-arms
  // itself, so this is a fresh, unattested contact.
  occ[static_cast<std::size_t>(support_index(12, 12, 12))] = 1;
  auto returned = support_grid(occ);
  returned.support_witness_live = live;
  const auto hit = osk::check_attached_voxel_collision(m, att, s, returned, 0.0);
  EXPECT_TRUE(hit.hit);
  EXPECT_EQ(hit.link_b, support_index(12, 12, 12));
  EXPECT_NEAR(hit.min_distance, -0.0157, 1e-9);
  EXPECT_EQ(hit.sweep_min_distance, hit.min_distance)
      << "sweep == min is the field signature of a stop with no exemption active";
}

TEST(SupportContactWitness, MalformedAttestationFailsTheWholeIngestClosed) {
  const std::vector<std::string> links = {"base", "hand"};
  osk::AttachedModel out;
  out.objects.assign(2, osk::AttachedObject{});
  out.primitives.assign(4, osk::AttachedPrimitive{});
  out.touch_links.assign(4, 0);

  osk::AttachedObjectInput in;
  in.attach_link = "hand";
  osk::AttachedPrimitiveInput prim;
  prim.kind = osk::AttachedShapeKind::kSphere;
  prim.radius = 0.05;
  in.primitives = {prim};
  in.has_support_witness = true;
  in.support_point = {0.0, 0.0, 0.0};
  in.support_normal = {0.0, 0.0, 0.0};  // degenerate
  in.support_patch_radius = 0.1;
  in.support_max_penetration = 0.001;
  EXPECT_EQ(osk::ingest_attached_objects({in}, links, 2, 4, 4, out),
            osk::AttachIngestStatus::kMalformed);
  EXPECT_EQ(out.n_objects, 0U);

  in.support_normal = {0.0, 0.0, 4.0};  // un-normalised is fine; it gets normalised
  in.support_patch_radius = -1.0;       // but a non-positive patch is not
  EXPECT_EQ(osk::ingest_attached_objects({in}, links, 2, 4, 4, out),
            osk::AttachIngestStatus::kMalformed);

  in.support_patch_radius = 0.1;
  in.support_max_penetration = -0.001;  // nor is a negative depth
  EXPECT_EQ(osk::ingest_attached_objects({in}, links, 2, 4, 4, out),
            osk::AttachIngestStatus::kMalformed);

  in.support_max_penetration = 0.001;
  EXPECT_EQ(osk::ingest_attached_objects({in}, links, 2, 4, 4, out), osk::AttachIngestStatus::kOk);
  ASSERT_EQ(out.n_objects, 1U);
  EXPECT_TRUE(out.objects[0].has_support_witness);
  EXPECT_NEAR(out.objects[0].support_normal.z, 1.0, 1e-12) << "the normal is normalised at ingest";
  EXPECT_NEAR(out.objects[0].support_patch_radius, 0.1, 1e-12);
}

// ── Place-phase witness (ADR-0097) ──────────────────────────────────────────
//
// The kernel gains NOTHING for the place phase, and these tests exist to prove
// that rather than to assume it. `support_contact_exempts` is producer-blind:
// it does not branch on `evidence_kind`, it does not know what a declaration
// is, and it cannot — the declaration gates the PRODUCER, which is what keeps
// the kernel's own predicate evidence-based. A place witness is therefore
// exactly what a pick witness is: an attestation on
// `AttachedCollisionObject.support_contact`, exempting exactly the patch it
// describes.
//
// The scenario is round-5's
// (`spark:~/openral-runs/2026-08-14-round5/baguette/seed1_run2`), reproduced as
// a geometry class: a payload inserted into a mapped container region, resting
// on the shelf inside it. What made that run stop at -1.78 mm is that World
// State attested nothing for the shelf contact — there was no place phase to
// attest for — so the shelf read as an ordinary obstacle. In kernel terms the
// declared and undeclared cases differ in one bit, `has_support_witness`, and
// both directions are pinned below.

namespace {

// The cabinet's own occupancy, on the same 24^3 / 25 mm lattice: three shelf
// cells directly under the inserted payload, and one back-panel cell 150 mm
// along +x — inside the container, on the payload, and NOT on the attested
// support plane.
constexpr int kShelfCellsX[] = {11, 12, 13};
constexpr int kCabinetBackX = 18;
constexpr int kCabinetBackZ = 13;

void fill_cabinet(std::vector<std::uint8_t>& occ, bool with_back_panel) {
  for (const int ix : kShelfCellsX) {
    occ[static_cast<std::size_t>(support_index(ix, 12, 12))] = 1;
  }
  if (with_back_panel) {
    occ[static_cast<std::size_t>(support_index(kCabinetBackX, 12, kCabinetBackZ))] = 1;
  }
}

}  // namespace

TEST(SupportContactWitness, DeclaredPlaceContactRidesTheSameExemptionAsThePick) {
  // The declared insertion. The payload is down on the cabinet's shelf, World
  // State has attested that contact, and the kernel does not stop on it.
  osk::CollisionModel m = hand_model();
  osk::CollisionScratch s;
  s.link_world = {identity(), identity(), identity(), identity()};
  osk::AttachedModel att;
  append_baguette(att, 0.10);  // the place attestation: same shape, same caps

  std::vector<std::uint8_t> occ(kSupportN * kSupportN * kSupportN, 0);
  fill_cabinet(occ, /*with_back_panel=*/false);
  auto grid = support_grid(occ);
  grid.support_witness_live = 0x1;

  EXPECT_EQ(osk::update_support_contact_witnesses(att, s, grid, 0x1, 0.0), 0x1)
      << "the payload is resting on the surface it attested; the witness stays live";
  EXPECT_FALSE(osk::check_attached_voxel_collision(m, att, s, grid, 0.0).hit)
      << "the flagship counter->cabinet insertion completes";
}

TEST(SupportContactWitness, TheSameInsertionWithNoDeclarationStillStops) {
  // Round-5 itself. Identical geometry, identical cells, identical payload
  // pose — the ONLY difference is that nothing was attested, because no place
  // phase was declared. The kernel sees an ordinary obstacle and stops on it,
  // at the quantisation-inflated depth the run reported.
  osk::CollisionModel m = hand_model();
  osk::CollisionScratch s;
  s.link_world = {identity(), identity(), identity(), identity()};
  osk::AttachedModel att;
  append_baguette(att, 0.0);  // no attestation => has_support_witness == false
  ASSERT_FALSE(att.objects[0].has_support_witness);

  std::vector<std::uint8_t> occ(kSupportN * kSupportN * kSupportN, 0);
  fill_cabinet(occ, /*with_back_panel=*/false);
  auto grid = support_grid(occ);
  grid.support_witness_live = 0x1;  // even wrongly set, it exempts nothing

  const auto hit = osk::check_attached_voxel_collision(m, att, s, grid, 0.0);
  ASSERT_TRUE(hit.hit) << "an undeclared insertion must keep stopping the robot";
  EXPECT_NEAR(hit.min_distance, -0.0157, 1e-9);
  EXPECT_EQ(osk::update_support_contact_witnesses(att, s, grid, 0x1, 0.0), 0x0);
}

TEST(SupportContactWitness, TheDeclaredShelfIsExemptButTheCabinetAroundItIsNot) {
  // The anti-scope, inside the declared target itself. The shelf the payload
  // was attested against is exempt; the cabinet's back panel — same container,
  // same goal, 150 mm out and therefore outside the attested patch — is not.
  // A declaration buys the attested patch and nothing else.
  osk::CollisionModel m = hand_model();
  osk::CollisionScratch s;
  s.link_world = {identity(), identity(), identity(), identity()};
  osk::AttachedModel att;
  append_baguette(att, 0.10);

  std::vector<std::uint8_t> occ(kSupportN * kSupportN * kSupportN, 0);
  fill_cabinet(occ, /*with_back_panel=*/true);
  auto grid = support_grid(occ);
  grid.support_witness_live = 0x1;

  const auto hit = osk::check_attached_voxel_collision(m, att, s, grid, 0.0);
  ASSERT_TRUE(hit.hit);
  EXPECT_EQ(hit.link_b, support_index(kCabinetBackX, 12, kCabinetBackZ))
      << "the stop must name the back panel, never the attested shelf";
}

TEST(SupportContactWitness, DeepeningIntoTheDeclaredPlaceSurfaceStillStops) {
  // Symmetric with DeepeningIntoTheSupportStillStops. A declaration does not
  // license driving the payload THROUGH the shelf it declared.
  osk::CollisionModel m = hand_model();
  osk::CollisionScratch s;
  s.link_world = {identity(), identity(), identity(), identity()};
  osk::AttachedModel att;
  append_baguette(att, 0.10);

  std::vector<std::uint8_t> occ(kSupportN * kSupportN * kSupportN, 0);
  fill_cabinet(occ, /*with_back_panel=*/false);
  auto grid = support_grid(occ);
  grid.support_witness_live = 0x1;
  ASSERT_FALSE(osk::check_attached_voxel_collision(m, att, s, grid, 0.0).hit);

  // 40 mm of sink, for the same reason `DeepeningIntoTheSupportStillStops`
  // moved from 20: the envelope gained one voxel on 2026-08-15 and the pick and
  // place witnesses share the one predicate, so they share the new bound too.
  att.objects[0].pose_in_link = translate(0.0, 0.0, -0.04);
  EXPECT_TRUE(osk::check_attached_voxel_collision(m, att, s, grid, 0.0).hit);
}

TEST(SupportContactWitness, PickAndPlaceShareOneSlotSequentially) {
  // The concurrency question ADR-0097 left to implementation time, answered:
  // one witness per payload, reused. The pick witness dies on separation from
  // the counter; the place witness is a NEW attestation written into the same
  // slot, and it exempts its own plane only.
  osk::CollisionModel m = hand_model();
  osk::CollisionScratch s;
  s.link_world = {identity(), identity(), identity(), identity()};
  osk::AttachedModel att;
  append_baguette(att, 0.10);

  std::vector<std::uint8_t> occ(kSupportN * kSupportN * kSupportN, 0);
  occ[static_cast<std::size_t>(support_index(12, 12, 12))] = 1;  // the counter
  auto grid = support_grid(occ);

  std::uint8_t live = 0x1;
  grid.support_witness_live = live;
  ASSERT_FALSE(osk::check_attached_voxel_collision(m, att, s, grid, 0.0).hit);

  // Carry away. The pick witness dies and the kernel never revives it.
  att.objects[0].pose_in_link = translate(0.0, 0.0, 0.30);
  live = osk::update_support_contact_witnesses(att, s, grid, live, 0.0);
  ASSERT_EQ(live, 0x0);

  // The place attestation lands: the payload is now over the cabinet shelf,
  // and World State has attested THAT contact. The kernel re-arms only because
  // a new attestation arrived (the lifecycle node's (support_id, stamp_ns) key
  // rule), and the exemption follows the new plane.
  std::fill(occ.begin(), occ.end(), static_cast<std::uint8_t>(0));
  fill_cabinet(occ, /*with_back_panel=*/true);
  att.objects[0].pose_in_link = identity();
  live = 0x1;  // stands in for the fresh-attestation re-arm
  grid.support_witness_live = live;

  EXPECT_EQ(osk::update_support_contact_witnesses(att, s, grid, live, 0.0), 0x1);
  const auto hit = osk::check_attached_voxel_collision(m, att, s, grid, 0.0);
  ASSERT_TRUE(hit.hit);
  EXPECT_EQ(hit.link_b, support_index(kCabinetBackX, 12, kCabinetBackZ))
      << "the place witness exempts the shelf it describes and nothing else";
}

TEST(SupportContactWitness, AWrongDeclaredTargetChangesWhichContactNeverHowMuch) {
  // HZ-0097-2 mitigation 3. Give the attestation BOTH kernel caps at once —
  // the largest patch (0.5 m) and the deepest penetration (10 mm) a witness
  // may claim — i.e. the most any declaration, right or wrong, could ever buy.
  // The exemption is still bounded: the total a cell may sit above the attested
  // plane is 12.5 mm of cube half-width + 10 mm of attested depth + 1 mm of
  // pose-noise slack + 25 mm of co-planar headroom (ADR-0097's Second
  // Amendment, hazard log Entry 012 "Calibration 2026-08-15") = 48.5 mm, and
  // that arithmetic does not know or care what was declared.
  osk::CollisionModel m = hand_model();
  osk::CollisionScratch s;
  s.link_world = {identity(), identity(), identity(), identity()};
  osk::AttachedModel att;
  append_baguette(att, 0.50);                     // the maximum patch
  att.objects[0].support_max_penetration = 0.01;  // the maximum depth

  std::vector<std::uint8_t> occ(kSupportN * kSupportN * kSupportN, 0);
  occ[static_cast<std::size_t>(support_index(12, 12, 12))] = 1;
  auto grid = support_grid(occ);
  grid.support_witness_live = 0x1;

  // The plane rides in the payload's own frame, so sinking the payload sinks
  // it too. At 45 mm of sink the cell is 48.2 mm above the plane — inside the
  // 48.5 mm bound, and still exempt.
  att.objects[0].pose_in_link = translate(0.0, 0.0, -0.045);
  EXPECT_FALSE(osk::check_attached_voxel_collision(m, att, s, grid, 0.0).hit);

  // At 50 mm it is 53.2 mm above, and the caps do not stretch for it.
  att.objects[0].pose_in_link = translate(0.0, 0.0, -0.05);
  const auto hit = osk::check_attached_voxel_collision(m, att, s, grid, 0.0);
  EXPECT_TRUE(hit.hit) << "the caps bound a wrong declaration; they do not stretch for it";
  EXPECT_EQ(hit.link_b, support_index(12, 12, 12));
}

TEST(CollisionEvidence, AttachedVoxelReportsTheTriggeringCellNotTheExemptResidue) {
  // The observed field failure, to five decimals: a box payload carried into
  // an occupancy map that still holds the payload's own cell. That residue
  // cell is 100.4 mm "inside" the payload and is exempt (it was already
  // embedded at attach time); the cell that actually stops the robot is the
  // support cell it newly touches by 0.4 mm.
  osk::CollisionModel m = hand_model();
  osk::CollisionScratch s;
  s.link_world = {identity(), identity(), identity(), identity()};
  osk::AttachedModel att;
  append_object(att, 1, translate(-0.001, 0.0, 0.0), {box_prim(0.0504, 0.0504, 0.0504)});

  std::vector<std::uint8_t> occ(125, 0);
  const int residue = voxel_index(2, 2, 2);  // the payload's own occupancy residue
  const int support = voxel_index(3, 2, 2);  // the real, newly touched support cell
  occ[static_cast<std::size_t>(residue)] = 1;
  occ[static_cast<std::size_t>(support)] = 1;
  std::vector<std::uint8_t> contact_mask(125, 0);
  std::vector<double> contact_distance(125, std::numeric_limits<double>::infinity());
  auto grid = make_grid(occ);
  grid.attached_contact_mask = contact_mask.data();
  grid.attached_contact_distance = contact_distance.data();
  grid.attached_contact_stride = occ.size();
  grid.attached_contact_tolerance = 0.001;

  // Attach-time snapshot: only the residue cell is in contact (the support
  // cell is still 0.6 mm clear), so only the residue cell is exempt.
  ASSERT_TRUE(osk::update_attached_voxel_contacts(att, s, grid, contact_mask.data(),
                                                  contact_distance.data(), contact_mask.size(),
                                                  contact_distance.size(), true));
  ASSERT_FALSE(osk::check_attached_voxel_collision(m, att, s, grid, 0.0).hit);

  // 1 mm of approach: the payload now overlaps the support cell by 0.4 mm.
  att.objects[0].pose_in_link = identity();
  const auto hit = osk::check_attached_voxel_collision(m, att, s, grid, 0.0);
  ASSERT_TRUE(hit.hit);
  EXPECT_EQ(hit.link_a, 0);
  EXPECT_EQ(hit.link_b, support) << "the stop was caused by the support cell";
  EXPECT_NEAR(hit.min_distance, -0.0004, 1e-9)
      << "min_distance must describe the cell the evidence names, not the exempt residue";
  EXPECT_NEAR(hit.sweep_min_distance, -0.1004, 1e-9)
      << "the sweep-wide minimum (the exempt residue) stays available, separately";
}

TEST(CollisionEvidence, AttachedVoxelReportsTheDeepestTriggeringCell) {
  // Three occupied cells trip at once. The evidence must name one of them and
  // quote that one's distance; the deepest is the useful choice and the one
  // pinned here.
  osk::CollisionModel m = hand_model();
  osk::CollisionScratch s;
  s.link_world = {identity(), identity(), identity(), identity()};
  osk::AttachedModel att;
  append_object(att, 1, translate(0.02, 0.0, 0.0), {box_prim(0.09, 0.05, 0.05)});

  std::vector<std::uint8_t> occ(125, 0);
  occ[static_cast<std::size_t>(voxel_index(1, 2, 2))] = 1;  // scanned first, -0.02 m
  occ[static_cast<std::size_t>(voxel_index(2, 2, 2))] = 1;  // deepest, -0.10 m
  occ[static_cast<std::size_t>(voxel_index(3, 2, 2))] = 1;  // -0.06 m
  const auto grid = make_grid(occ);

  const auto hit = osk::check_attached_voxel_collision(m, att, s, grid, 0.0);
  ASSERT_TRUE(hit.hit);
  EXPECT_EQ(hit.link_b, voxel_index(2, 2, 2));
  // Recompute the named cell's distance: evidence must be self-consistent.
  const osk::Transform named = voxel_transform_at(grid, hit.link_b);
  const double named_distance = osk::box_box_distance(
      translate(0.02, 0.0, 0.0), osk::Vec3{0.09, 0.05, 0.05}, named, kVoxelHalf);
  EXPECT_NEAR(hit.min_distance, named_distance, 1e-9);
  EXPECT_NEAR(hit.sweep_min_distance, named_distance, 1e-9)
      << "no cell is exempt here, so the sweep minimum is the triggering cell";
}

TEST(CollisionEvidence, WorldVoxelDistanceDescribesTheReportedCell) {
  // Same coherence rule on the robot-link voxel path: the shallow cell is
  // scanned first, the deep cell second, and the evidence must not mix them.
  const auto m = one_capsule_model();  // capsule r=0.1, half-length 0.2 at the origin
  osk::CollisionScratch s;
  s.link_world = {identity()};
  std::vector<std::uint8_t> occ(125, 0);
  occ[static_cast<std::size_t>(voxel_index(1, 2, 2))] = 1;  // -0.05 m, scanned first
  occ[static_cast<std::size_t>(voxel_index(2, 2, 2))] = 1;  // -0.10 m, deepest
  const auto grid = make_grid(occ);

  const auto hit = osk::check_voxel_collision(m, s, grid, 0.0);
  ASSERT_TRUE(hit.hit);
  EXPECT_EQ(hit.link_b, voxel_index(2, 2, 2));
  const osk::Transform named = voxel_transform_at(grid, hit.link_b);
  const double named_distance = osk::box_capsule_distance(named, kVoxelHalf, identity(), 0.1, 0.2);
  EXPECT_NEAR(hit.min_distance, named_distance, 1e-9);
  EXPECT_NEAR(hit.sweep_min_distance, named_distance, 1e-9);
}

TEST(CollisionEvidence, SelfCollisionDistanceDescribesTheReportedLinkPair) {
  // Link 1 grazes link 2 by 1 mm (checked first) while it overlaps link 3 by
  // 50 mm. The evidence must not quote the 50 mm against the (1,2) pair.
  osk::CollisionModel m = hand_model();
  add_capsule(m, 1, 0.05, 0.0, identity());
  add_capsule(m, 2, 0.05, 0.0, identity());
  add_capsule(m, 3, 0.05, 0.0, identity());
  osk::CollisionScratch s;
  s.link_world = {identity(), identity(), translate(0.099, 0.0, 0.0), translate(0.0, 0.05, 0.0)};

  const auto hit = osk::check_self_collision(m, s, 0.0);
  ASSERT_TRUE(hit.hit);
  EXPECT_EQ(hit.link_a, 1);
  EXPECT_EQ(hit.link_b, 3) << "the deepest tripping pair is (link 1, link 3)";
  EXPECT_NEAR(hit.min_distance, -0.05, 1e-9);
  EXPECT_NEAR(hit.sweep_min_distance, -0.05, 1e-9);
}

TEST(CollisionEvidence, AttachedWorldDistanceDescribesTheReportedObstacle) {
  osk::CollisionModel m = hand_model();
  osk::CollisionScratch s;
  s.link_world = {identity(), identity(), identity(), identity()};
  osk::AttachedModel att;
  append_object(att, 1, identity(), {sphere_prim(0.1)});

  osk::WorldModel w;
  osk::Capsule grazing;
  grazing.radius = 0.1;
  grazing.half_length = 0.0;
  grazing.origin = translate(0.199, 0.0, 0.0);  // -1 mm, checked first
  osk::Capsule deep;
  deep.radius = 0.1;
  deep.half_length = 0.0;
  deep.origin = translate(0.15, 0.0, 0.0);  // -50 mm
  w.capsules = {grazing, deep};

  const auto hit = osk::check_attached_world_collision(m, att, s, w, 0.0);
  ASSERT_TRUE(hit.hit);
  EXPECT_EQ(hit.link_a, 0);
  EXPECT_EQ(hit.link_b, 1) << "the deepest tripping obstacle is the second one";
  EXPECT_NEAR(hit.min_distance, -0.05, 1e-9);
}

TEST(CollisionEvidence, SweepMinimumEqualsTheEvidenceDistanceWhenNothingTrips) {
  // With no hit there is no pair to describe, so `min_distance` keeps its
  // clearance meaning and both fields agree.
  const auto m = one_capsule_model();
  osk::CollisionScratch s;
  s.link_world = {identity()};
  const auto clear = osk::check_world_collision(m, s, world_obstacle_at(1.0), 0.0);
  EXPECT_FALSE(clear.hit);
  EXPECT_NEAR(clear.min_distance, 0.8, 1e-9);
  EXPECT_NEAR(clear.sweep_min_distance, clear.min_distance, 1e-12);

  // An empty check reports no clearance at all (infinite), on both fields.
  const osk::WorldModel empty;
  const auto nothing = osk::check_world_collision(m, s, empty, 0.0);
  EXPECT_FALSE(nothing.hit);
  EXPECT_TRUE(std::isinf(nothing.min_distance));
  EXPECT_TRUE(std::isinf(nothing.sweep_min_distance));
}

TEST(CollisionEvidence, GatingIsUnchangedAcrossTheAttachedContactLadder) {
  // Reporting-only guard rail. The same inputs must produce the same
  // stop/no-stop decision as before the evidence fix: the expectations below
  // are the pre-fix kernel's answers, recorded here so any future change to
  // the evidence path that moves a gate fails loudly.
  //
  // Fixture: a box payload attached at link 1, an occupancy residue cell it is
  // embedded in at attach time, and a support cell 0.6 mm ahead of it.
  const osk::CollisionModel m = hand_model();
  osk::CollisionScratch s;
  s.link_world = {identity(), identity(), identity(), identity()};

  struct Case {
    double approach_m;  // payload offset along +x from the snapshot pose
    bool expect_hit;
  };
  // 0.000: snapshot pose — residue exempt, support 0.6 mm clear.
  // 0.001: support newly touched by 0.4 mm — a new cell, so it stops.
  // 0.005: the new cell is 4.4 mm deep — stops.
  //
  // The `allow_new_shallow` column this ladder used to carry is gone with the
  // mechanism: a blanket "any new cell shallower than the tolerance is fine"
  // is not something the support-contact witness needs, and it exempted cells
  // no one had attested. Its removal moves every one of those rows from
  // no-stop to stop, i.e. strictly toward the conservative side.
  const std::vector<Case> cases = {{0.000, false}, {0.001, true}, {0.005, true}};

  for (const Case& c : cases) {
    osk::AttachedModel att;
    append_object(att, 1, translate(-0.001, 0.0, 0.0), {box_prim(0.0504, 0.0504, 0.0504)});
    std::vector<std::uint8_t> occ(125, 0);
    occ[static_cast<std::size_t>(voxel_index(2, 2, 2))] = 1;
    occ[static_cast<std::size_t>(voxel_index(3, 2, 2))] = 1;
    std::vector<std::uint8_t> contact_mask(125, 0);
    std::vector<double> contact_distance(125, std::numeric_limits<double>::infinity());
    auto grid = make_grid(occ);
    grid.attached_contact_mask = contact_mask.data();
    grid.attached_contact_distance = contact_distance.data();
    grid.attached_contact_stride = occ.size();
    grid.attached_contact_tolerance = 0.001;
    ASSERT_TRUE(osk::update_attached_voxel_contacts(att, s, grid, contact_mask.data(),
                                                    contact_distance.data(), contact_mask.size(),
                                                    contact_distance.size(), true));

    att.objects[0].pose_in_link = translate(-0.001 + c.approach_m, 0.0, 0.0);
    const auto hit = osk::check_attached_voxel_collision(m, att, s, grid, 0.0);
    EXPECT_EQ(hit.hit, c.expect_hit) << "gating changed at approach=" << c.approach_m;
    if (hit.hit) {
      EXPECT_LE(hit.min_distance, 0.0) << "a reported stop is never a clearance";
    }
  }
}

// ── Declaration-scoped approach allowance (ADR-0097's 2026-08-14 amendment) ──
//
// Round-6 (`spark:~/openral-runs/2026-08-14-round6/baguette/seed1_carry_*`) is
// the failure this section pins closed. With a correct place declaration and a
// payload genuinely headed for the declared cabinet, the kernel E-stopped at
// `horizon_step 0` on `attached:sim:obj_main` vs `voxel_178099` at a
// `min_distance` of `-4.9 mm`, while the payload was still `22-30 mm` from real
// shelf contact. Nothing was wrong with the declaration or the trajectory: the
// place witness is EARNED by measured contact, and the predictive check —
// unchanged and unrelated to the witness — stops the payload before it can make
// that contact, because 25 mm cells inflate the cabinet's thin opening geometry
// by up to one whole voxel. A witness earned by touching can never arm if the
// payload is stopped before it can touch.
//
// The amendment's answer is a margin allowance of `min(1.5 × voxel, 40 mm)`,
// applied ONLY to the declared payload, ONLY to cells whose centre is inside the
// producer-supplied region of the declared target, and ONLY while that
// declaration is live. It is not an exemption: the reported distance is the
// cell's true distance, and the hard stop behind the reduced margin is
// untouched.
//
// The cap was `min(one voxel, 25 mm)` until ADR-0097's SECOND AMENDMENT
// (2026-08-15, hazard log HZ-0097-4's "Calibration 2026-08-15"). The 5-run
// battery's run 1 (`spark:~/openral-runs/2026-08-15-baguette-battery/run1`) was
// the allowance's first in-vivo firing: 26.48 mm of predictive-check
// penetration read at a ground-truth −2.43 mm contact, i.e. 1.48 mm short. Since
// the map-vs-truth error at a placement pose is itself about one voxel, a
// one-voxel allowance is sized to absorb exactly the discretization and leaves
// nothing for the contact the witness is earned by making — structurally
// marginal, not occasionally short. At 25 mm cells the cap is now 37.5 mm; at a
// real 50 mm grid it is 40 mm, not 75 mm, because the absolute ceiling still
// binds and only its value moved (2.5 cm → 4 cm).
//
// Fixture: the same 24^3 / 25 mm lattice as the witness tests, with the cabinet
// modelled as what it physically is — a shelf below and a lip above, separated
// by an opening the payload must pass through. The lattice phase puts the shelf
// cells' cube top face at z = -9.3 mm and the lip cells' cube bottom face at
// z = +40.7 mm: a QUANTISED opening of exactly 50 mm for a 50 mm-tall baguette,
// i.e. zero predicted clearance where the real cabinet has centimetres.

namespace {

constexpr int kApproachShelfZ = 11;    // cube z in [-0.0343, -0.0093]
constexpr int kApproachLipZ = 14;      // cube z in [ 0.0407,  0.0657]
constexpr int kApproachOutsideX = 18;  // a shelf cell OUTSIDE the declared region
constexpr int kApproachCabinetX[] = {11, 12, 13};

void fill_cabinet_opening(std::vector<std::uint8_t>& occ, bool with_outside_obstacle) {
  for (const int ix : kApproachCabinetX) {
    occ[static_cast<std::size_t>(support_index(ix, 12, kApproachShelfZ))] = 1;
    occ[static_cast<std::size_t>(support_index(ix, 12, kApproachLipZ))] = 1;
  }
  if (with_outside_obstacle) {
    occ[static_cast<std::size_t>(support_index(kApproachOutsideX, 12, kApproachShelfZ))] = 1;
  }
}

// The region the sim producer computes from the declared body's model subtree:
// the cabinet's interior, 200 x 120 x 120 mm about the opening. It contains the
// shelf and lip cell centres (z = -21.8 mm and +53.2 mm) and the three cabinet
// columns, and NOT the ix=18 obstacle 150 mm away.
osk::PlaceApproachRegion declared_cabinet_region(std::uint8_t mask = 0x1) {
  osk::PlaceApproachRegion region;
  EXPECT_EQ(osk::ingest_place_region(translate(0.0, 0.0, 0.0157), osk::Vec3{0.10, 0.06, 0.06}, mask,
                                     region),
            osk::PlaceRegionStatus::kOk);
  return region;
}

// ── A coarse 50 mm lattice, for the absolute-cap tests ──────────────────────
//
// HZ-0095-2's episode in a new form: at real hardware's 5 cm resolution a purely
// resolution-relative allowance would be 50 mm, double sim's. The cap makes that
// impossible, and these helpers are what proves it.
constexpr int kCoarseN = 13;
constexpr double kCoarseRes = 0.05;

int coarse_index(int x, int y, int z) { return x + kCoarseN * (y + kCoarseN * z); }

osk::VoxelGrid coarse_grid(const std::vector<std::uint8_t>& occ) {
  osk::VoxelGrid g;
  // Phased so cell (6, 6, 6) is centred on the origin; its cube spans +/-25 mm.
  g.origin = {-0.325, -0.325, -0.325};
  g.resolution = kCoarseRes;
  g.sx = kCoarseN;
  g.sy = kCoarseN;
  g.sz = kCoarseN;
  g.occupancy = occ.data();
  g.attached_contact_tolerance = 0.001;
  return g;
}

}  // namespace

TEST(PlaceApproachAllowance, TheRound6ApproachReachesTheDeclaredShelf) {
  // The exact field stop, and its removal. The payload is inserted through the
  // quantised opening: 4.9 mm into the shelf cells' inflated cube, 4.9 mm clear
  // of the lip's. Undeclared that is round 6, to the tenth of a millimetre.
  osk::CollisionModel m = hand_model();
  osk::CollisionScratch s;
  s.link_world = {identity(), identity(), identity(), identity()};
  osk::AttachedModel att;
  append_baguette(att, 0.0);  // no witness yet — it is EARNED at contact
  ASSERT_FALSE(att.objects[0].has_support_witness);

  std::vector<std::uint8_t> occ(kSupportN * kSupportN * kSupportN, 0);
  fill_cabinet_opening(occ, /*with_outside_obstacle=*/false);
  auto grid = support_grid(occ);
  att.objects[0].pose_in_link = translate(0.0, 0.0, -0.0142);

  const auto round6 = osk::check_attached_voxel_collision(m, att, s, grid, 0.0);
  ASSERT_TRUE(round6.hit) << "with no declaration this is exactly what stopped the run";
  EXPECT_NEAR(round6.min_distance, -0.0049, 1e-9);
  EXPECT_FALSE(round6.place_allowance_active);

  // The declaration's region goes live. Same geometry, same cell, same true
  // -4.9 mm reading — the margin it is gated against is what moved.
  grid.place_region = declared_cabinet_region();
  EXPECT_FALSE(osk::check_attached_voxel_collision(m, att, s, grid, 0.0).hit)
      << "the declared approach must be able to continue through the opening";

  // And it continues all the way down to real shelf contact 19.1 mm lower, where
  // the true distance is -24.0 mm — inside the 37.5 mm cap, with the 13.5 mm of
  // headroom past the maximum inflation a 25 mm surface cell can produce that
  // the Second Amendment bought. At that pose the place witness arms and its
  // bounded, non-deepening exemption takes over.
  att.objects[0].pose_in_link = translate(0.0, 0.0, -0.0333);
  const auto at_contact = osk::check_attached_voxel_collision(m, att, s, grid, 0.0);
  EXPECT_FALSE(at_contact.hit);
  EXPECT_NEAR(at_contact.sweep_min_distance, -0.024, 1e-9)
      << "the sweep still reports the true distance; only the gate moved";
}

TEST(PlaceApproachAllowance, TheRun1FieldPenetrationNowReachesContact) {
  // ADR-0097's Second Amendment, on its own field evidence. Run 1 of the
  // 2026-08-15 battery tripped at 26.48 mm of predictive-check penetration on a
  // cell inside the declared region, 1.48 mm past the 25 mm allowance then in
  // force, at a ground-truth contact of -2.43 mm. The calibrated cap is 37.5 mm
  // at sim's 25 mm cells, so that reading is now 11.02 mm inside the allowance
  // and the insertion continues to the contact the witness is earned by.
  osk::CollisionModel m = hand_model();
  osk::CollisionScratch s;
  s.link_world = {identity(), identity(), identity(), identity()};
  osk::AttachedModel att;
  append_baguette(att, 0.0);  // no witness: the allowance is what is under test

  std::vector<std::uint8_t> occ(kSupportN * kSupportN * kSupportN, 0);
  fill_cabinet_opening(occ, /*with_outside_obstacle=*/false);
  auto grid = support_grid(occ);
  grid.place_region = declared_cabinet_region();
  EXPECT_NEAR(osk::place_approach_allowance(grid, 0, osk::Vec3{0.0, 0.0, -0.0218}), 0.0375, 1e-12)
      << "min(1.5 x 25 mm voxel, 40 mm absolute) is 37.5 mm";

  // The run-1 reading itself, reproduced to the hundredth of a millimetre.
  att.objects[0].pose_in_link = translate(0.0, 0.0, -0.03578);
  const auto run1 = osk::check_attached_voxel_collision(m, att, s, grid, 0.0);
  EXPECT_NEAR(run1.sweep_min_distance, -0.02648, 1e-9) << "the field penetration, unchanged";
  EXPECT_FALSE(run1.hit) << "1.48 mm short under the old cap; 11.02 mm inside the calibrated one";

  // The cap still bites past that, at 38.2 mm and at 40 mm — but not against
  // THIS payload: a 50 mm-tall baguette passing a 25 mm cell saturates at
  // exactly 37.5 mm of overlap on every axis (25 mm of half-extent + 12.5 mm of
  // cell half-width), so the calibrated cap is precisely the deepest reading the
  // flagship geometry can produce. `…DeepeningPastTheAllowanceInsideTheRegion
  // StillStops` carries the deeper payload that can, and pins both stops.

  // Undeclared, run 1 is still a stop at the same true distance: the calibration
  // moved a declaration-scoped margin, nothing else.
  grid.place_region = osk::PlaceApproachRegion{};
  att.objects[0].pose_in_link = translate(0.0, 0.0, -0.03578);
  const auto undeclared = osk::check_attached_voxel_collision(m, att, s, grid, 0.0);
  ASSERT_TRUE(undeclared.hit);
  EXPECT_NEAR(undeclared.min_distance, -0.02648, 1e-9);
  EXPECT_FALSE(undeclared.place_allowance_active);
}

TEST(PlaceApproachAllowance, TheSameApproachUndeclaredStopsExactlyAsRound6Did) {
  // The regression that keeps the relaxation declaration-scoped. Byte-identical
  // inputs, no live region: the pre-amendment kernel's answer, unchanged.
  osk::CollisionModel m = hand_model();
  osk::CollisionScratch s;
  s.link_world = {identity(), identity(), identity(), identity()};
  osk::AttachedModel att;
  append_baguette(att, 0.0);

  std::vector<std::uint8_t> occ(kSupportN * kSupportN * kSupportN, 0);
  fill_cabinet_opening(occ, /*with_outside_obstacle=*/false);
  auto grid = support_grid(occ);
  ASSERT_FALSE(grid.place_region.valid) << "a default VoxelGrid carries no allowance";

  for (const double dz : {-0.0142, -0.0333, -0.0500}) {
    att.objects[0].pose_in_link = translate(0.0, 0.0, dz);
    const auto hit = osk::check_attached_voxel_collision(m, att, s, grid, 0.0);
    EXPECT_TRUE(hit.hit) << "undeclared insertion must keep stopping at dz=" << dz;
    EXPECT_FALSE(hit.place_allowance_active);
  }
}

TEST(PlaceApproachAllowance, TheAllowanceAppliesToThePredictedHorizonToo) {
  // Round 6 stopped at horizon_step 0, not on the measured configuration. The
  // predictive path is the same function called with an inflated margin, so the
  // allowance has to hold there — otherwise the fix would not touch the failure
  // it exists for.
  osk::CollisionModel m = hand_model();
  osk::CollisionScratch s;
  s.link_world = {identity(), identity(), identity(), identity()};
  osk::AttachedModel att;
  append_baguette(att, 0.0);

  std::vector<std::uint8_t> occ(kSupportN * kSupportN * kSupportN, 0);
  fill_cabinet_opening(occ, /*with_outside_obstacle=*/false);
  auto grid = support_grid(occ);
  att.objects[0].pose_in_link = translate(0.0, 0.0, -0.0142);

  // collision_predict_margin_growth_m's default look-ahead inflation.
  constexpr double kPredictiveMargin = 0.01;
  ASSERT_TRUE(osk::check_attached_voxel_collision(m, att, s, grid, kPredictiveMargin).hit);
  grid.place_region = declared_cabinet_region();
  EXPECT_FALSE(osk::check_attached_voxel_collision(m, att, s, grid, kPredictiveMargin).hit);
}

TEST(PlaceApproachAllowance, ACoarserGridNeverWidensTheAllowance) {
  // HZ-0097-4 mitigation 1 as calibrated 2026-08-15, and the HZ-0095-2 lesson it
  // descends from. At 50 mm resolution the voxel-relative half of the formula
  // would be 1.5 x 50 = 75 mm — nearly twice sim's 37.5 mm, a permissiveness
  // increase arriving as a side effect of map coarseness and reviewed by nobody.
  // The absolute ceiling makes it 40 mm (the amendment's own worked example), so
  // 45 mm of penetration inside the declared region still stops.
  osk::CollisionModel m = hand_model();
  osk::CollisionScratch s;
  s.link_world = {identity(), identity(), identity(), identity()};
  osk::AttachedModel att;
  append_baguette(att, 0.0);

  std::vector<std::uint8_t> occ(kCoarseN * kCoarseN * kCoarseN, 0);
  occ[static_cast<std::size_t>(coarse_index(6, 6, 6))] = 1;
  auto grid = coarse_grid(occ);
  osk::PlaceApproachRegion region;
  ASSERT_EQ(osk::ingest_place_region(identity(), osk::Vec3{0.20, 0.10, 0.10}, 0x1, region),
            osk::PlaceRegionStatus::kOk);
  grid.place_region = region;

  EXPECT_NEAR(osk::place_approach_allowance(grid, 0, osk::Vec3{0.0, 0.0, 0.0}), 0.040, 1e-12)
      << "min(1.5 x 50 mm voxel, 40 mm absolute) is 40 mm, never 75";

  // 30 mm of penetration: inside the cap, forgiven.
  att.objects[0].pose_in_link = translate(0.0, 0.0, -0.005);
  EXPECT_FALSE(osk::check_attached_voxel_collision(m, att, s, grid, 0.0).hit);

  // 45 mm: past the cap, and it stops — which a resolution-relative allowance
  // would not have done.
  att.objects[0].pose_in_link = translate(0.0, 0.0, -0.020);
  const auto hit = osk::check_attached_voxel_collision(m, att, s, grid, 0.0);
  ASSERT_TRUE(hit.hit);
  EXPECT_NEAR(hit.min_distance, -0.045, 1e-9) << "the reported distance is the true one";
  EXPECT_TRUE(hit.place_allowance_active) << "a stop at a reduced margin must say so";
}

TEST(PlaceApproachAllowance, AnObjectAlreadyInTheRegionIsForgivenOnlyToTheCap) {
  // HZ-0097-4's accepted cost, and its bound. A jar already sitting on the
  // declared shelf when the declaration goes live receives the same allowance as
  // the shelf, for the declared window, whether or not the payload ever touches
  // it. That is a real reduction in the predictive check's conservatism —
  // accepted, bounded, and pinned here: on this 50 mm lattice the forgiveness
  // stops dead at the 40 mm absolute ceiling (ADR-0097's Second Amendment; the
  // accepted cost is unchanged in shape and re-accepted at the same severity).
  osk::CollisionModel m = hand_model();
  osk::CollisionScratch s;
  s.link_world = {identity(), identity(), identity(), identity()};
  osk::AttachedModel att;
  append_baguette(att, 0.0);

  std::vector<std::uint8_t> occ(kCoarseN * kCoarseN * kCoarseN, 0);
  occ[static_cast<std::size_t>(coarse_index(6, 6, 6))] = 1;  // the jar
  auto grid = coarse_grid(occ);
  osk::PlaceApproachRegion region;
  ASSERT_EQ(osk::ingest_place_region(identity(), osk::Vec3{0.20, 0.10, 0.10}, 0x1, region),
            osk::PlaceRegionStatus::kOk);
  grid.place_region = region;

  struct Case {
    double dz;
    double distance;
    bool expect_hit;
  };
  const std::vector<Case> cases = {{0.020, -0.005, false},
                                   {0.005, -0.020, false},
                                   {-0.005, -0.030, false},
                                   {-0.015, -0.040, true},  // exactly the ceiling: it trips
                                   {-0.020, -0.045, true}};
  for (const Case& c : cases) {
    att.objects[0].pose_in_link = translate(0.0, 0.0, c.dz);
    const auto hit = osk::check_attached_voxel_collision(m, att, s, grid, 0.0);
    EXPECT_EQ(hit.hit, c.expect_hit) << "jar penetration " << c.distance;
    EXPECT_NEAR(hit.hit ? hit.min_distance : hit.sweep_min_distance, c.distance, 1e-9);
  }
}

TEST(PlaceApproachAllowance, DeepeningPastTheAllowanceInsideTheRegionStillStops) {
  // HZ-0097-4 mitigation 3, at sim's own resolution, on the cap as calibrated
  // 2026-08-15. The allowance shrinks the predictive margin; it does not remove
  // the hard stop behind it. A payload driven THROUGH the declared shelf rather
  // than onto it still E-stops, and the declaration cannot stretch the bound —
  // 0.7 mm past the 37.5 mm cap is still a stop, and so is the 40 mm the
  // amendment's own headroom arithmetic quotes.
  //
  // The payload here is a 300x100x100 mm tote rather than the 50 mm-tall
  // baguette, and deliberately: against a 25 mm cell a 50 mm-tall box saturates
  // at exactly 37.5 mm of SAT overlap (25 mm of half-extent + 12.5 mm of cell
  // half-width) on every axis, so the flagship payload's own geometry cannot
  // reach past the calibrated cap at all. Proving the hard stop behind the
  // reduced margin therefore needs a payload deep enough to try.
  osk::CollisionModel m = hand_model();
  osk::CollisionScratch s;
  s.link_world = {identity(), identity(), identity(), identity()};
  osk::AttachedModel att;
  append_object(att, 1, identity(), {box_prim(0.15, 0.05, 0.05, translate(0.0, 0.0, 0.05))});

  std::vector<std::uint8_t> occ(kSupportN * kSupportN * kSupportN, 0);
  fill_cabinet_opening(occ, /*with_outside_obstacle=*/false);
  auto grid = support_grid(occ);
  grid.place_region = declared_cabinet_region();

  // At real shelf contact: -24.0 mm, inside the cap, allowed.
  att.objects[0].pose_in_link = translate(0.0, 0.0, -0.0333);
  ASSERT_FALSE(osk::check_attached_voxel_collision(m, att, s, grid, 0.0).hit);

  // 14.2 mm further, through the shelf: -38.2 mm, past the cap, stopped.
  att.objects[0].pose_in_link = translate(0.0, 0.0, -0.0475);
  const auto hit = osk::check_attached_voxel_collision(m, att, s, grid, 0.0);
  ASSERT_TRUE(hit.hit) << "a declaration never licenses driving THROUGH the declared shelf";
  EXPECT_EQ(hit.link_a, 0);
  EXPECT_NEAR(hit.min_distance, -0.0382, 1e-9) << "the reported distance is the true one";
  EXPECT_TRUE(hit.place_allowance_active) << "a stop at a reduced margin must say so";

  // And 40 mm — the reading run 1 would have had to produce for the OLD cap to
  // be the right one — stops too.
  att.objects[0].pose_in_link = translate(0.0, 0.0, -0.0493);
  const auto forty = osk::check_attached_voxel_collision(m, att, s, grid, 0.0);
  ASSERT_TRUE(forty.hit);
  EXPECT_NEAR(forty.min_distance, -0.0400, 1e-9);
}

TEST(PlaceApproachAllowance, OutsideTheDeclaredRegionTheMarginIsUnchanged) {
  // The scope that makes this a place allowance rather than a global one. A shelf
  // cell 150 mm out — same height, same map, same payload, outside the declared
  // receptacle — stops the robot at the full margin, and it is the cell the
  // evidence names.
  osk::CollisionModel m = hand_model();
  osk::CollisionScratch s;
  s.link_world = {identity(), identity(), identity(), identity()};
  osk::AttachedModel att;
  append_baguette(att, 0.0);

  std::vector<std::uint8_t> occ(kSupportN * kSupportN * kSupportN, 0);
  fill_cabinet_opening(occ, /*with_outside_obstacle=*/true);
  auto grid = support_grid(occ);
  grid.place_region = declared_cabinet_region();
  att.objects[0].pose_in_link = translate(0.0, 0.0, -0.0142);

  const auto hit = osk::check_attached_voxel_collision(m, att, s, grid, 0.0);
  ASSERT_TRUE(hit.hit);
  EXPECT_EQ(hit.link_b, support_index(kApproachOutsideX, 12, kApproachShelfZ))
      << "the stop must name the cell outside the declared region";
  EXPECT_FALSE(hit.place_allowance_active)
      << "that cell's margin was never reduced, and the evidence must not claim it was";
}

TEST(PlaceApproachAllowance, TheAllowanceFollowsTheDeclaredPayloadOnly) {
  // The declaration names one payload. A second object the robot happens to be
  // carrying is not in the mask and gets nothing, even inside the same region.
  osk::CollisionModel m = hand_model();
  osk::CollisionScratch s;
  s.link_world = {identity(), identity(), identity(), identity()};
  osk::AttachedModel att;
  append_baguette(att, 0.0);  // object 0 — the declared payload
  append_object(att, 1, translate(0.0, 0.0, -0.0142),
                {box_prim(0.15, 0.025, 0.025, translate(0.0, 0.0, 0.025))});

  std::vector<std::uint8_t> occ(kSupportN * kSupportN * kSupportN, 0);
  fill_cabinet_opening(occ, /*with_outside_obstacle=*/false);
  auto grid = support_grid(occ);
  grid.place_region = declared_cabinet_region(0x1);  // object 0 only
  att.objects[0].pose_in_link = translate(0.0, 0.0, -0.0142);

  const auto hit = osk::check_attached_voxel_collision(m, att, s, grid, 0.0);
  ASSERT_TRUE(hit.hit) << "the undeclared second payload still stops on the same cells";
  EXPECT_EQ(hit.link_a, 1);
  EXPECT_NEAR(osk::place_approach_allowance(grid, 1, osk::Vec3{0.0, 0.0, -0.0218}), 0.0, 1e-12);
  EXPECT_NEAR(osk::place_approach_allowance(grid, 0, osk::Vec3{0.0, 0.0, -0.0218}), 0.0375, 1e-12);
}

TEST(PlaceApproachAllowance, ArmVsWorldIsUntouchedByALiveRegion) {
  // The amendment's own words: "Arm-vs-world collision checks are unchanged."
  // The robot's own links may not enter the declared receptacle any more freely
  // than before, and the region is not an input to that check at all.
  osk::CollisionModel m = hand_model();
  add_capsule(m, 1, 0.02, 0.0, translate(0.0, 0.0, -0.0218));
  osk::CollisionScratch s;
  s.link_world = {identity(), identity(), identity(), identity()};

  std::vector<std::uint8_t> occ(kSupportN * kSupportN * kSupportN, 0);
  fill_cabinet_opening(occ, /*with_outside_obstacle=*/false);
  auto grid = support_grid(occ);

  const auto before = osk::check_voxel_collision(m, s, grid, 0.0);
  ASSERT_TRUE(before.hit) << "the arm capsule is inside the shelf cells";
  grid.place_region = declared_cabinet_region();
  const auto after = osk::check_voxel_collision(m, s, grid, 0.0);
  EXPECT_EQ(after.hit, before.hit);
  EXPECT_EQ(after.link_b, before.link_b);
  EXPECT_NEAR(after.min_distance, before.min_distance, 1e-15);
  EXPECT_FALSE(after.place_allowance_active);
}

TEST(PlaceApproachAllowance, TheRegionIsAnOrientedBoxNotItsAxisAlignedHull) {
  // The producer measures a receptacle, and a receptacle has an orientation. The
  // axis-aligned hull of a rotated one is strictly larger, i.e. strictly more
  // permissive, so the predicate tests the box itself.
  std::vector<std::uint8_t> occ(kSupportN * kSupportN * kSupportN, 0);
  auto grid = support_grid(occ);
  osk::PlaceApproachRegion region;
  ASSERT_EQ(
      osk::ingest_place_region(osk::transform_from_xyz_rpy(0.0, 0.0, 0.0, 0.0, 0.0, kPi / 4.0),
                               osk::Vec3{0.20, 0.02, 0.10}, 0x1, region),
      osk::PlaceRegionStatus::kOk);
  grid.place_region = region;

  // On the rotated long axis: inside.
  const double d = 0.15 / std::sqrt(2.0);
  EXPECT_NEAR(osk::place_approach_allowance(grid, 0, osk::Vec3{d, d, 0.0}), 0.0375, 1e-12);
  // The same distance along +x — inside the AABB of the rotated box, outside the
  // box.
  EXPECT_NEAR(osk::place_approach_allowance(grid, 0, osk::Vec3{0.15, 0.0, 0.0}), 0.0, 1e-12);
}

TEST(PlaceApproachAllowance, DegenerateAndOversizedRegionsGrantNothing) {
  // Fail-closed means "no allowance", not "a dropped message": unlike a malformed
  // support attestation, a bad region can only ever make the kernel more
  // permissive, so refusing it restores exactly the pre-amendment margin.
  osk::PlaceApproachRegion region;
  const osk::Vec3 sane{0.10, 0.06, 0.06};
  const double nan_value = std::numeric_limits<double>::quiet_NaN();

  // Each refusal reports WHICH way the region failed, and they are not
  // interchangeable: `kNoObject` is the ordinary pre-grasp state the node logs at
  // INFO, the rest are producer errors it warns about. Round-8 labelled all of
  // them `bounds`, which made the noisiest line in the run the one that described
  // its cause least.
  EXPECT_EQ(osk::ingest_place_region(identity(), sane, 0x0, region),
            osk::PlaceRegionStatus::kNoObject)
      << "a declaration whose payload is not carried arms nothing";
  EXPECT_EQ(osk::ingest_place_region(identity(), osk::Vec3{0.0, 0.06, 0.06}, 0x1, region),
            osk::PlaceRegionStatus::kDegenerate)
      << "a region with no interior";
  EXPECT_EQ(osk::ingest_place_region(identity(), osk::Vec3{-0.1, 0.06, 0.06}, 0x1, region),
            osk::PlaceRegionStatus::kDegenerate);
  EXPECT_EQ(osk::ingest_place_region(identity(), osk::Vec3{nan_value, 0.06, 0.06}, 0x1, region),
            osk::PlaceRegionStatus::kBadExtents);
  EXPECT_EQ(osk::ingest_place_region(identity(),
                                     osk::Vec3{std::numeric_limits<double>::infinity(), 0.06, 0.06},
                                     0x1, region),
            osk::PlaceRegionStatus::kBadExtents);
  EXPECT_EQ(osk::ingest_place_region(identity(), osk::Vec3{2.0, 0.06, 0.06}, 0x1, region),
            osk::PlaceRegionStatus::kOversize)
      << "a half-extent past 1.5 m is a producer error, not a receptacle";
  EXPECT_EQ(osk::ingest_place_region(identity(), osk::Vec3{1.4, 1.4, 1.4}, 0x1, region),
            osk::PlaceRegionStatus::kOversizeVolume)
      << "22 m^3 is a room, and a declaration names ONE receptacle";
  EXPECT_EQ(osk::ingest_place_region(translate(nan_value, 0.0, 0.0), sane, 0x1, region),
            osk::PlaceRegionStatus::kBadPose);

  // Every rejection leaves the struct inert, so a caller that ignores the return
  // value still gets no allowance.
  EXPECT_FALSE(region.valid);
  std::vector<std::uint8_t> occ(kSupportN * kSupportN * kSupportN, 0);
  auto grid = support_grid(occ);
  grid.place_region = region;
  EXPECT_NEAR(osk::place_approach_allowance(grid, 0, osk::Vec3{0.0, 0.0, 0.0}), 0.0, 1e-12);

  // The largest region the caps do admit is still admitted.
  EXPECT_EQ(osk::ingest_place_region(identity(), osk::Vec3{1.0, 1.0, 1.0}, 0x1, region),
            osk::PlaceRegionStatus::kOk);
  EXPECT_TRUE(region.valid);
}

TEST(PlaceApproachAllowance, EveryRefusalHasItsOwnReasonToken) {
  // The tokens are the `reason=` key of the kernel's place-region log lines, so
  // they are a contract with whoever reads a run: distinct per failure, stable,
  // and never null. Round-8's single `bounds` label for all of them is what made
  // the pre-grasp no-payload state indistinguishable from a malformed region.
  EXPECT_STREQ(osk::place_region_status_reason(osk::PlaceRegionStatus::kOk), "ok");
  EXPECT_STREQ(osk::place_region_status_reason(osk::PlaceRegionStatus::kNoObject), "no_object");
  EXPECT_STREQ(osk::place_region_status_reason(osk::PlaceRegionStatus::kBadPose), "bad_pose");
  EXPECT_STREQ(osk::place_region_status_reason(osk::PlaceRegionStatus::kBadExtents), "bad_extents");
  EXPECT_STREQ(osk::place_region_status_reason(osk::PlaceRegionStatus::kDegenerate), "degenerate");
  EXPECT_STREQ(osk::place_region_status_reason(osk::PlaceRegionStatus::kOversize), "oversize");
  EXPECT_STREQ(osk::place_region_status_reason(osk::PlaceRegionStatus::kOversizeVolume),
               "oversize_volume");

  const osk::PlaceRegionStatus all[] = {osk::PlaceRegionStatus::kOk,
                                        osk::PlaceRegionStatus::kNoObject,
                                        osk::PlaceRegionStatus::kBadPose,
                                        osk::PlaceRegionStatus::kBadExtents,
                                        osk::PlaceRegionStatus::kDegenerate,
                                        osk::PlaceRegionStatus::kOversize,
                                        osk::PlaceRegionStatus::kOversizeVolume};
  std::set<std::string> tokens;
  for (const auto status : all) {
    const char* reason = osk::place_region_status_reason(status);
    ASSERT_NE(reason, nullptr);
    tokens.insert(reason);
  }
  EXPECT_EQ(tokens.size(), std::size(all)) << "two outcomes sharing one token is the round-8 bug";
}

TEST(PlaceApproachAllowance, AMapWithNoResolutionGrantsNothing) {
  // Same fail-closed direction as update_support_contact_witnesses: no usable map
  // means no allowance, not an unbounded one.
  std::vector<std::uint8_t> occ(kSupportN * kSupportN * kSupportN, 0);
  auto grid = support_grid(occ);
  grid.place_region = declared_cabinet_region();
  grid.resolution = 0.0;
  EXPECT_NEAR(osk::place_approach_allowance(grid, 0, osk::Vec3{0.0, 0.0, 0.0}), 0.0, 1e-12);
  grid.resolution = kSupportRes;
  EXPECT_NEAR(osk::place_approach_allowance(grid, 8, osk::Vec3{0.0, 0.0, 0.0}), 0.0, 1e-12)
      << "past the eight-object schema cap there is no mask bit to consult";
}

TEST(PlaceApproachAllowance, TheWitnessTakesOverWhereTheAllowanceStops) {
  // The two mechanisms in sequence, which is the whole point of the amendment:
  // the allowance buys the approach, the witness buys the contact. With the
  // witness armed at the shelf the payload is exempt on that plane whether or not
  // the region is live, and the latch is still measured against the map.
  osk::CollisionModel m = hand_model();
  osk::CollisionScratch s;
  s.link_world = {identity(), identity(), identity(), identity()};
  osk::AttachedModel att;
  append_baguette(att, 0.10);  // the place attestation, earned at contact
  att.objects[0].support_point = {0.0, 0.0, 0.0};

  std::vector<std::uint8_t> occ(kSupportN * kSupportN * kSupportN, 0);
  fill_cabinet_opening(occ, /*with_outside_obstacle=*/false);
  auto grid = support_grid(occ);
  grid.support_witness_live = 0x1;
  att.objects[0].pose_in_link = translate(0.0, 0.0, -0.0333);
  EXPECT_FALSE(osk::check_attached_voxel_collision(m, att, s, grid, 0.0).hit);

  grid.place_region = declared_cabinet_region();
  EXPECT_FALSE(osk::check_attached_voxel_collision(m, att, s, grid, 0.0).hit);
  EXPECT_EQ(osk::update_support_contact_witnesses(att, s, grid, 0x1, 0.0), 0x1)
      << "the witness latch is measured against the map, not against the allowance";
}
