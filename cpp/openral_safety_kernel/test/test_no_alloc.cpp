// SPDX-License-Identifier: Apache-2.0
// Proves the validator does NOT allocate on the hot path.
//
// Strategy: globally-overridden ``operator new`` increments a counter
// when called. The validator is run 10,000 times under a pre-built
// EnvelopeIntersection + ChunkView; the counter must stay at 0 across
// all iterations.
//
// CLAUDE.md §2 — "no allocations inside hot loops". This test makes
// that guarantee enforceable in CI.

#include "openral_safety_kernel/collision.hpp"
#include "openral_safety_kernel/force_observer.hpp"
#include "openral_safety_kernel/validator.hpp"

#include <array>
#include <atomic>
#include <bitset>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <new>
#include <string>
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
      << "validator allocated on the hot path; CLAUDE.md §2 forbids this.";
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

  // #349: the workspace exemption and the hand-speed bound sit in the same hot
  // path (per candidate row); with a live, slow workspace the check stays
  // allocation-free and really tolerates pairs.
  osk::Transform ws_pose;
  ws_pose.t = {0.0, 0.0, 0.0};
  ASSERT_EQ(osk::ingest_workspace_region(ws_pose, osk::Vec3{0.15, 0.15, 0.15}, mask, 0.1, 0.05,
                                         grid.workspace_region),
            osk::GraspRegionStatus::kOk);
  grid.workspace_region.hand_speed_mps = 0.05;
  grid.workspace_region.hand_lead_m = 0.01;
  const osk::Transform moved = osk::transform_from_xyz_rpy(0.001, 0.0, 0.0, 0.0, 0.0, 0.01);
  g_alloc_count.store(0, std::memory_order_relaxed);
  g_count_enabled.store(true, std::memory_order_relaxed);
  int tolerated = 0;
  for (int i = 0; i < 10000; ++i) {
    sink += osk::link_displacement_bound(s.link_world[1], moved, 0.1);
    tolerated += osk::check_voxel_collision(m, s, grid, 0.02).workspace_pairs;
  }
  g_count_enabled.store(false, std::memory_order_relaxed);
  EXPECT_EQ(g_alloc_count.load(std::memory_order_relaxed), 0U)
      << "the voxel check allocated with a live workspace; CLAUDE.md §2 forbids this.";
  EXPECT_GT(tolerated, 0) << "the loop really exercised tolerated workspace pairs";

  // #349 review F: a REFUSED chunk — the hand over its speed bound, so the same pairs
  // trip — is the brake's dry-run path, and it stays allocation-free too.
  grid.workspace_region.hand_speed_mps = 0.5;
  g_alloc_count.store(0, std::memory_order_relaxed);
  g_count_enabled.store(true, std::memory_order_relaxed);
  int refused = 0;
  for (int i = 0; i < 10000; ++i) {
    const auto hit = osk::check_voxel_collision(m, s, grid, 0.02);
    refused += hit.hit && hit.workspace_denied == 2 ? 1 : 0;
  }
  g_count_enabled.store(false, std::memory_order_relaxed);
  EXPECT_EQ(g_alloc_count.load(std::memory_order_relaxed), 0U)
      << "the voxel check allocated on a refused workspace row; CLAUDE.md §2 forbids this.";
  EXPECT_EQ(refused, 10000) << "the loop really exercised the speed refusal";
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

TEST(NoAlloc, GripperEndEffectorLookupIsAllocationFree) {
  // The gripper case resolves ee_name against the envelope's names by
  // string_view comparison; pass and violation paths must both stay at 0.
  osk::EnvelopeIntersection env;
  env.n_dof = 16;
  env.joint_position_min.assign(16, -1.0);
  env.joint_position_max.assign(16, 1.0);
  env.joint_velocity_max.assign(16, 3.15);
  env.joint_torque_max.assign(16, 5.0);
  env.gripper_ee_names = {"left_gripper_with_a_name_longer_than_sso", "right_gripper"};
  env.gripper_command_min = {0.0, -0.7854};
  env.gripper_command_max = {0.7854, 0.0};

  const std::string ee = "right_gripper";
  std::vector<double> ok(8, -0.3);
  std::vector<double> bad(8, 0.3);
  osk::ChunkView view{};
  view.control_mode = static_cast<std::uint8_t>(osk::ControlMode::kGripperPosition);
  view.horizon = 8;
  view.n_dof = 1;
  view.ee_name = ee.data();
  view.ee_name_size = ee.size();

  g_alloc_count.store(0, std::memory_order_relaxed);
  g_count_enabled.store(true, std::memory_order_relaxed);
  for (int i = 0; i < 5000; ++i) {
    view.flat_data = ok.data();
    view.flat_size = ok.size();
    ASSERT_TRUE(osk::validate(view, env));
    view.flat_data = bad.data();
    view.flat_size = bad.size();
    ASSERT_FALSE(osk::validate(view, env));
  }
  g_count_enabled.store(false, std::memory_order_relaxed);
  EXPECT_EQ(g_alloc_count.load(std::memory_order_relaxed), 0U);
}

TEST(NoAlloc, TheTrackedBindingCheckIsAllocationFree) {
  // Issue #418 K3: the binding check runs on every world-state message against
  // the measured configuration; like the checks it gates, it must not allocate.
  osk::CollisionModel m;
  m.n_links = 2;
  m.parent = {-1, 0};
  m.joint_kind = {osk::JointKind::kFixed, osk::JointKind::kFixed};
  m.dof_index = {-1, -1};
  m.origin = {osk::Transform{}, osk::Transform{}};
  m.axis = {{0, 0, 1}, {0, 0, 1}};
  m.capsule_link = {1};
  m.capsules = {osk::Capsule{0.01, 0.02, osk::Transform{}}};
  osk::Obb finger;
  finger.half_extents = {0.01, 0.01, 0.03};
  m.box_link = {1};
  m.boxes = {finger};
  osk::CollisionScratch s;
  s.link_world = {osk::Transform{}, osk::Transform{}};
  osk::AttachedModel att;
  att.objects.assign(1, osk::AttachedObject{});
  att.touch_links.assign(1, 1);
  att.objects[0].attach_link = 0;
  att.objects[0].touch_count = 1;
  att.n_objects = 1;
  osk::Transform box;
  g_alloc_count.store(0, std::memory_order_relaxed);
  g_count_enabled.store(true, std::memory_order_relaxed);
  int met = 0;
  for (int i = 0; i < 10000; ++i) {
    box.t.x = (i % 2 == 0) ? 0.0 : 0.5;
    met += osk::tracked_box_meets_hand(m, s, att, att.objects[0], box, osk::Vec3{0.02, 0.02, 0.02},
                                       0.025)
               ? 1
               : 0;
  }
  g_count_enabled.store(false, std::memory_order_relaxed);
  EXPECT_EQ(g_alloc_count.load(std::memory_order_relaxed), 0U);
  EXPECT_EQ(met, 5000) << "the loop really exercised both verdicts";
}

TEST(NoAlloc, AFrozenPayloadsChecksAreAllocationFree) {
  // Issue #418 K5: the release window's separation test runs per world-state
  // message and the frozen payload is checked on every candidate.
  osk::CollisionModel m;
  m.n_links = 1;
  m.parent = {-1};
  m.joint_kind = {osk::JointKind::kFixed};
  m.dof_index = {-1};
  m.origin = {osk::Transform{}};
  m.axis = {{0, 0, 1}};
  m.capsule_link = {0};
  m.capsules = {osk::Capsule{0.01, 0.02, osk::Transform{}}};
  osk::CollisionScratch s;
  s.link_world = {osk::Transform{}};
  osk::AttachedModel att;
  att.objects.assign(1, osk::AttachedObject{});
  att.primitives.assign(1, osk::AttachedPrimitive{});
  att.primitives[0].radius = 0.01;
  att.objects[0].attach_link = -1;
  att.objects[0].release_hand_link = 0;
  att.objects[0].prim_count = 1;
  att.n_objects = 1;
  att.n_primitives = 1;
  g_alloc_count.store(0, std::memory_order_relaxed);
  g_count_enabled.store(true, std::memory_order_relaxed);
  double sum = 0.0;
  for (int i = 0; i < 10000; ++i) {
    att.objects[0].pose_in_link.t.x = 0.001 * (i % 100);
    sum += osk::released_payload_clearance(m, s, att, att.objects[0]);
    sum += osk::check_attached_self_collision(m, att, s, 0.01).hit ? 1.0 : 0.0;
  }
  g_count_enabled.store(false, std::memory_order_relaxed);
  EXPECT_EQ(g_alloc_count.load(std::memory_order_relaxed), 0U);
  EXPECT_TRUE(std::isfinite(sum));
}

TEST(NoAlloc, TheTrackedGraspExemptionIsAllocationFree) {
  // Issue #418 K2: the per-hand selection runs once per checked row inside the
  // voxel check, and the cover count once per grid; neither may allocate.
  osk::CollisionModel m;
  m.n_links = 3;
  m.parent = {-1, 0, 1};
  m.joint_kind = {osk::JointKind::kFixed, osk::JointKind::kFixed, osk::JointKind::kFixed};
  m.dof_index = {-1, -1, -1};
  m.origin = {osk::Transform{}, osk::Transform{}, osk::Transform{}};
  m.axis = {{0, 0, 1}, {0, 0, 1}, {0, 0, 1}};
  osk::Obb finger;
  finger.half_extents = {0.01, 0.01, 0.02};
  m.box_link = {2};
  m.boxes = {finger};
  m.capsule_link = {1};
  m.capsules = {osk::Capsule{0.015, 0.0, osk::Transform{}}};
  std::bitset<osk::kMaxGraspMaskLinks> allow;
  allow.set(2);
  osk::TrackedGraspTable table;
  ASSERT_TRUE(osk::build_grasp_hands(m, allow, table));
  table.valid = true;
  table.n_objects = 2;
  table.box[0].t = {0.0, 0.0, 0.0};
  table.half_extents[0] = {0.03, 0.03, 0.03};
  table.box[1].t = {0.1, 0.0, 0.0};
  table.half_extents[1] = {0.03, 0.03, 0.03};
  osk::CollisionScratch s;
  s.link_world = {osk::Transform{}, osk::Transform{}, osk::Transform{}};
  s.link_world[1].t = {-0.1, 0.0, 0.0};
  constexpr int kN = 16;
  std::vector<std::uint8_t> occ(kN * kN * kN, 0);
  for (std::size_t i = 0; i < occ.size(); i += 5) {
    occ[i] = 1;
  }
  osk::VoxelGrid grid;
  grid.pose.t = {-0.16, -0.16, -0.16};
  grid.resolution = 0.02;
  grid.sx = kN;
  grid.sy = kN;
  grid.sz = kN;
  grid.occupancy = occ.data();
  grid.tracked_grasp = &table;
  std::array<int, osk::kMaxGraspHands> sel{};
  g_alloc_count.store(0, std::memory_order_relaxed);
  g_count_enabled.store(true, std::memory_order_relaxed);
  double sink = 0.0;
  std::size_t cover = 0;
  for (int i = 0; i < 2000; ++i) {
    s.link_world[2].t.x = (i % 2 == 0) ? 0.0 : 0.1;
    osk::select_tracked_grasp(table, s, grid.resolution, sel);
    sink += sel[0];
    sink += osk::check_voxel_collision(m, s, grid, 0.02).sweep_min_distance;
    cover += osk::tracked_box_cover(grid, table.box[0], table.half_extents[0]);
  }
  g_count_enabled.store(false, std::memory_order_relaxed);
  EXPECT_EQ(g_alloc_count.load(std::memory_order_relaxed), 0U)
      << "the tracked grasp exemption allocated; CLAUDE.md §2 forbids this.";
  EXPECT_TRUE(std::isfinite(sink));
  EXPECT_GT(cover, 0U) << "the loop really counted occupied cells";
}

TEST(NoAlloc, TheExplainedCellsCheckIsAllocationFree) {
  // Issue #418 K7: every pair the live workspace would tolerate asks whether its
  // cell is explained (boxes, static cells, support), inside the voxel check.
  osk::CollisionModel m;
  m.n_links = 2;
  m.parent = {-1, 0};
  m.joint_kind = {osk::JointKind::kFixed, osk::JointKind::kFixed};
  m.dof_index = {-1, -1};
  m.origin = {osk::Transform{}, osk::Transform{}};
  m.axis = {{0, 0, 1}, {0, 0, 1}};
  osk::Obb hand;
  hand.half_extents = {0.03, 0.03, 0.03};
  m.box_link = {1};
  m.boxes = {hand};
  constexpr int kN = 16;
  std::vector<std::uint8_t> occ(kN * kN * kN, 0);
  for (std::size_t i = 0; i < occ.size(); i += 3) {
    occ[i] = 1;
  }
  osk::VoxelGrid grid;
  grid.pose.t = {-0.16, -0.16, -0.16};
  grid.resolution = 0.02;
  grid.sx = kN;
  grid.sy = kN;
  grid.sz = kN;
  grid.occupancy = occ.data();
  std::bitset<osk::kMaxGraspMaskLinks> hands;
  hands.set(1);
  ASSERT_EQ(osk::ingest_workspace_region(osk::Transform{}, {0.15, 0.15, 0.15}, hands, 0.5, 0.05,
                                         grid.workspace_region),
            osk::GraspRegionStatus::kOk);
  grid.workspace_region.hand_speed_mps = 0.0;
  grid.workspace_region.hand_lead_m = 0.0;
  std::vector<std::uint8_t> static_mask(occ.size(), 0);
  const std::vector<osk::Vec3> points = {{0.01, 0.01, 0.01}, {-0.03, 0.01, 0.01}};
  ASSERT_EQ(osk::mark_static_cells(grid, points, static_mask.data(), static_mask.size()), 2U);
  osk::WorkspaceExplanation e;
  e.n_objects = 2;
  e.half_extents[0] = {0.02, 0.02, 0.02};
  e.box[1].t = {0.05, 0.0, 0.0};
  e.half_extents[1] = {0.02, 0.02, 0.02};
  e.support_z[0] = -0.02;
  e.support_z[1] = std::numeric_limits<double>::quiet_NaN();
  e.static_cells = static_mask.data();
  e.static_cells_n = static_mask.size();
  grid.workspace_explained = &e;
  osk::CollisionScratch s;
  s.link_world = {osk::Transform{}, osk::Transform{}};
  g_alloc_count.store(0, std::memory_order_relaxed);
  g_count_enabled.store(true, std::memory_order_relaxed);
  double sink = 0.0;
  int explained = 0;
  for (int i = 0; i < 2000; ++i) {
    s.link_world[1].t.x = (i % 2 == 0) ? 0.0 : 0.05;
    const auto hit = osk::check_voxel_collision(m, s, grid, 0.02);
    sink += hit.sweep_min_distance + hit.workspace_denied;
    explained += osk::workspace_cell_explained(grid, {0.01, 0.01, 0.01}) ? 1 : 0;
  }
  g_count_enabled.store(false, std::memory_order_relaxed);
  EXPECT_EQ(g_alloc_count.load(std::memory_order_relaxed), 0U)
      << "the explained-cells check allocated; CLAUDE.md §2 forbids this.";
  EXPECT_TRUE(std::isfinite(sink));
  EXPECT_EQ(explained, 2000);
}

TEST(NoAlloc, ThePayloadOwnCellExemptionIsAllocationFree) {
  // Issue #418 OT-10 amendment: the exemption runs inside the payload-vs-voxel sweep on every
  // candidate; like the sweep it must not allocate.
  osk::CollisionModel m;
  osk::CollisionScratch s;
  s.link_world = {osk::Transform{}, osk::Transform{}};
  osk::AttachedModel att;
  att.objects.assign(1, osk::AttachedObject{});
  att.primitives.assign(2, osk::AttachedPrimitive{});
  att.touch_links.assign(1, 0);
  att.objects[0].attach_link = 1;
  att.objects[0].prim_count = 2;
  for (int p = 0; p < 2; ++p) {
    att.primitives[static_cast<std::size_t>(p)].kind = osk::AttachedShapeKind::kBox;
    att.primitives[static_cast<std::size_t>(p)].half_extents = {0.03, 0.03, 0.03};
  }
  att.primitives[1].source = osk::PrimitiveSource::kMeasured;
  att.n_objects = 1;
  att.n_primitives = 2;
  std::vector<std::uint8_t> occ(20 * 20 * 20, 0);
  for (std::size_t i = 0; i < occ.size(); i += 37) {
    occ[i] = 1;
  }
  osk::PayloadOwnCells own;
  own.valid = true;
  own.armed = 1;
  own.floor_z[0] = -0.3;
  own.n_others = 2;
  own.other_box[0].t = {0.1, 0.0, 0.0};
  own.other_half[0] = {0.02, 0.02, 0.02};
  own.other_box[1].t = {-0.1, 0.0, 0.0};
  own.other_half[1] = {0.02, 0.02, 0.02};
  osk::VoxelGrid grid;
  grid.pose.t = {-0.2, -0.2, -0.2};
  grid.resolution = 0.02;
  grid.sx = grid.sy = grid.sz = 20;
  grid.occupancy = occ.data();
  grid.payload_own = &own;
  g_alloc_count.store(0, std::memory_order_relaxed);
  g_count_enabled.store(true, std::memory_order_relaxed);
  int pairs = 0;
  for (int i = 0; i < 2000; ++i) {
    pairs += osk::check_attached_voxel_collision(m, att, s, grid, 0.01).payload_own_pairs;
  }
  g_count_enabled.store(false, std::memory_order_relaxed);
  EXPECT_EQ(g_alloc_count.load(std::memory_order_relaxed), 0U);
  EXPECT_GT(pairs, 0) << "the loop really exercised the exemption";
}

TEST(NoAlloc, TheForceObserverUpdateIsAllocationFree) {
  // Two revolute links off a fixed root, plus a mimic follower: every RNEA
  // branch (revolute, mimic projection, armature) runs inside the window.
  osk::CollisionModel m;
  m.n_links = 4;
  m.parent = {-1, 0, 1, 1};
  m.joint_kind = {osk::JointKind::kFixed, osk::JointKind::kRevolute, osk::JointKind::kRevolute,
                  osk::JointKind::kMimicRevolute};
  m.dof_index = {-1, 0, 1, 1};
  m.origin.assign(4, osk::Transform{});
  m.origin[2].t = {0.3, 0.0, 0.0};
  m.axis.assign(4, osk::Vec3{0.0, 1.0, 0.0});
  m.mimic_multiplier = {1.0, 1.0, 1.0, -1.0};
  m.mimic_offset = {0.0, 0.0, 0.0, 0.0};
  osk::ForceObserverModel dyn;
  dyn.links.assign(4, osk::LinkInertial{});
  dyn.joints.assign(4, osk::JointFrictionModel{});
  for (std::size_t i = 1; i < 4; ++i) {
    dyn.links[i].mass = 0.5;
    dyn.links[i].com = {0.1, 0.0, 0.0};
    dyn.links[i].inertia[0] = dyn.links[i].inertia[4] = dyn.links[i].inertia[8] = 0.01;
    dyn.joints[i].armature = 0.01;
    dyn.joints[i].coulomb = 0.2;
  }
  osk::ForceObserverState state;
  osk::init_force_observer(m, 2, state);
  double q[2] = {0.1, -0.2};
  const double qd[2] = {0.3, -0.5};
  const double effort[2] = {1.0, 0.5};
  g_alloc_count.store(0, std::memory_order_relaxed);
  g_count_enabled.store(true, std::memory_order_relaxed);
  int ok = 0;
  for (int i = 0; i < 5000; ++i) {
    q[0] += 1e-4;
    ok += osk::force_observer_update(m, dyn, q, qd, effort, 0.002, state) ? 1 : 0;
  }
  g_count_enabled.store(false, std::memory_order_relaxed);
  EXPECT_EQ(g_alloc_count.load(std::memory_order_relaxed), 0U);
  EXPECT_EQ(ok, 5000);
}
