// SPDX-License-Identifier: Apache-2.0
// ADR-0030 phase 2 — unit coverage for the allocation-free self-collision
// core: hand-rolled forward kinematics + closed-form capsule-capsule
// distance + non-adjacent-pair self-collision over a joint configuration.
//
// Ground truth is hand-computed analytically (no MuJoCo dependency here);
// the MuJoCo-oracle cross-check lives in the sim-tier integration test that
// wires this core into the live kernel. test_no_alloc-style counting proves
// the hot path never allocates.

#include "openral_safety_kernel/collision.hpp"

#include <atomic>
#include <cmath>
#include <cstdlib>
#include <new>
#include <vector>

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

TEST(VoxelCollision, NullGridNeverHits) {
  const auto m = one_capsule_model();
  osk::CollisionScratch s;
  s.link_world = {identity()};
  const osk::VoxelGrid empty;  // occupancy == nullptr
  const auto hit = osk::check_voxel_collision(m, s, empty, 0.0);
  EXPECT_FALSE(hit.hit);
}

// ── jacobian_dls_step (ADR-0040 predictive Cartesian) ─────────────────────────

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
      << "collision hot path allocated; ADR-0030 requires it to be allocation-free.";
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
