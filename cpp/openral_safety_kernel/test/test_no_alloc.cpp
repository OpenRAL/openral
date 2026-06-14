// SPDX-License-Identifier: Apache-2.0
// ADR-0020 — proves the validator does NOT allocate on the hot path.
//
// Strategy: globally-overridden ``operator new`` increments a counter
// when called. The validator is run 10,000 times under a pre-built
// EnvelopeIntersection + ChunkView; the counter must stay at 0 across
// all iterations.
//
// CLAUDE.md §5.2 — "no allocations inside hot loops". This test makes
// that guarantee enforceable in CI.

#include "openral_safety_kernel/validator.hpp"

#include <atomic>
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
