// SPDX-License-Identifier: Apache-2.0
// Proves the validator does NOT allocate on the hot path.
//
// Strategy: globally-overridden ``operator new`` increments a counter
// when called. The validator is run 10,000 times under a pre-built
// EnvelopeIntersection + ChunkView; the counter must stay at 0 across
// all iterations.
//
// CLAUDE.md §5.2 — "no allocations inside hot loops". This test makes
// that guarantee enforceable in CI.

#include "openral_safety_kernel/collision.hpp"
#include "openral_safety_kernel/validator.hpp"

#include <atomic>
#include <bitset>
#include <cstdint>
#include <cstdlib>
#include <new>
#include <vector>

#include <gtest/gtest.h>

namespace {

// Atomic so the test stays sane under parallel gtest workers.
std::atomic<std::uint64_t> g_alloc_count{0};
std::atomic<bool> g_count_enabled{false};

}  // namespace

// Replace global ``operator new`` so we can detect any allocation. The
// gtest runtime itself allocates (for test fixtures, ASSERT messages,
// etc.) so we only count while `g_count_enabled` is true — the test
// arms it for the validator-only window.
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

void* operator new[](std::size_t size) {
  if (g_count_enabled.load(std::memory_order_relaxed)) {
    g_alloc_count.fetch_add(1, std::memory_order_relaxed);
  }
  void* p = std::malloc(size);
  if (p == nullptr) {
    throw std::bad_alloc{};
  }
  return p;
}

// The matching ``operator new`` overrides above route through std::malloc,
// so pairing these deletes with std::free is correct. GCC can't see the
// override at every gtest ``new TestClass`` call site, so silence the
// resulting -Wmismatched-new-delete here.
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmismatched-new-delete"
#endif
void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif

namespace osk = openral_safety_kernel;

TEST(NoAlloc, ValidatorIsAllocationFreeOverThousandsOfChunks) {
  // Set up envelope OUTSIDE the counted window (vectors allocate).
  osk::EnvelopeIntersection env;
  env.robot_name = "toy";
  env.n_dof = 6;
  env.joint_position_min.assign(6, -1.0);
  env.joint_position_max.assign(6, 1.0);
  env.joint_velocity_max.assign(6, 3.15);
  env.joint_torque_max.assign(6, 5.0);
  env.workspace_box.set = true;
  env.workspace_box.min_xyz = {-0.4, -0.4, 0.0};
  env.workspace_box.max_xyz = {0.4, 0.4, 0.6};
  env.max_ee_speed_m_s = 0.5;
  env.max_force_n = 10.0;
  env.max_torque_nm = 3.0;

  // Pre-allocate the chunk buffer too.
  std::vector<double> flat(16 * 6, 0.1);  // horizon=16
  osk::ChunkView view{};
  view.control_mode = static_cast<std::uint8_t>(osk::ControlMode::kJointPosition);
  view.horizon = 16;
  view.n_dof = 6;
  view.flat_data = flat.data();
  view.flat_size = flat.size();

  // ARM the allocation counter and run the validator 10,000 times.
  g_alloc_count.store(0, std::memory_order_relaxed);
  g_count_enabled.store(true, std::memory_order_relaxed);
  for (int i = 0; i < 10000; ++i) {
    const auto rc = osk::validate(view, env);
    ASSERT_TRUE(rc) << "iteration " << i << " unexpectedly failed";
  }
  g_count_enabled.store(false, std::memory_order_relaxed);
  EXPECT_EQ(g_alloc_count.load(std::memory_order_relaxed), 0U)
      << "validator allocated on the hot path; CLAUDE.md §5.2 forbids this.";
}

TEST(NoAlloc, ValidatorViolationPathIsAlsoAllocationFree) {
  osk::EnvelopeIntersection env;
  env.n_dof = 3;
  env.joint_position_min.assign(3, -1.0);
  env.joint_position_max.assign(3, 1.0);
  env.joint_velocity_max.assign(3, 3.15);
  env.joint_torque_max.assign(3, 5.0);

  std::vector<double> flat = {5.0, 0.0, 0.0};  // violates joint 0
  osk::ChunkView view{};
  view.control_mode = static_cast<std::uint8_t>(osk::ControlMode::kJointPosition);
  view.horizon = 1;
  view.n_dof = 3;
  view.flat_data = flat.data();
  view.flat_size = flat.size();

  g_alloc_count.store(0, std::memory_order_relaxed);
  g_count_enabled.store(true, std::memory_order_relaxed);
  for (int i = 0; i < 5000; ++i) {
    const auto rc = osk::validate(view, env);
    ASSERT_FALSE(rc);
  }
  g_count_enabled.store(false, std::memory_order_relaxed);
  EXPECT_EQ(g_alloc_count.load(std::memory_order_relaxed), 0U);
}

TEST(NoAlloc, VoxelCheckWithALiveGraspRegionIsAllocationFree) {
  // ADR-0115: the grasp-target exemption sits inside the world-voxel cell
  // loop, so the check with a live region must stay allocation-free. Model,
  // grid and region are built OUTSIDE the counted window.
  osk::CollisionModel m;
  m.n_links = 3;
  m.parent = {-1, 0, 0};
  m.joint_kind = {osk::JointKind::kFixed, osk::JointKind::kFixed, osk::JointKind::kFixed};
  m.dof_index = {-1, -1, -1};
  m.origin = {osk::Transform{}, osk::Transform{}, osk::Transform{}};
  m.axis = {{0, 0, 1}, {0, 0, 1}, {0, 0, 1}};
  osk::Obb finger;  // openarm_left_finger_pair's box, robots/openarm/robot.yaml
  finger.half_extents = {0.0294, 0.0711, 0.0811};
  finger.origin = osk::transform_from_xyz_rpy(0.0004, 0.0484, -0.0298, -2.8925, 0.0066, 3.1375);
  m.box_link = {1};
  m.boxes = {finger};
  m.capsule_link = {2};
  m.capsules = {osk::Capsule{0.01, 0.02, osk::Transform{}}};
  osk::CollisionScratch s;
  s.link_world = {osk::Transform{}, osk::Transform{}, osk::Transform{}};

  constexpr int kN = 16;
  std::vector<std::uint8_t> occ(kN * kN * kN, 0);
  for (std::size_t i = 0; i < occ.size(); i += 7) {
    occ[i] = 1;
  }
  osk::VoxelGrid grid;
  grid.pose.t = {-0.16, -0.16, -0.16};
  grid.resolution = 0.02;
  grid.sx = kN;
  grid.sy = kN;
  grid.sz = kN;
  grid.occupancy = occ.data();
  std::bitset<osk::kMaxGraspMaskLinks> mask;
  mask.set(1);
  mask.set(2);
  osk::Transform region_pose;
  region_pose.t = {0.0, 0.03, -0.02};
  ASSERT_EQ(
      osk::ingest_grasp_region(region_pose, osk::Vec3{0.04, 0.04, 0.04}, mask, grid.grasp_region),
      osk::GraspRegionStatus::kOk);

  g_alloc_count.store(0, std::memory_order_relaxed);
  g_count_enabled.store(true, std::memory_order_relaxed);
  double sink = 0.0;
  for (int i = 0; i < 10000; ++i) {
    const auto hit = osk::check_voxel_collision(m, s, grid, 0.02);
    sink += hit.sweep_min_distance;
  }
  g_count_enabled.store(false, std::memory_order_relaxed);
  EXPECT_EQ(g_alloc_count.load(std::memory_order_relaxed), 0U)
      << "the voxel check allocated with a live grasp region; CLAUDE.md §2 forbids this.";
  EXPECT_LT(sink, 0.0) << "the loop really exercised penetrating cells";
}

TEST(NoAlloc, RetiringAndCheckingGraspIdentitiesIsAllocationFree) {
  // The kernel retires a grasp declaration on the candidate path
  // (`handover_exit`): the retired set must not allocate, overflow included.
  openral_safety_kernel::RetiredGraspSet retired;
  g_alloc_count.store(0, std::memory_order_relaxed);
  g_count_enabled.store(true, std::memory_order_relaxed);
  bool seen = false;
  for (std::size_t n = 0; n < 1000; ++n) {
    retired.insert(n, 42);
    seen = retired.contains(n / 2, 42) || seen;
  }
  g_count_enabled.store(false, std::memory_order_relaxed);
  EXPECT_TRUE(seen);
  EXPECT_EQ(g_alloc_count.load(std::memory_order_relaxed), 0U);
}
