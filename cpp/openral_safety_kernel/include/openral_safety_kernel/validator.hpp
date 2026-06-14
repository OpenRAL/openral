// SPDX-License-Identifier: Apache-2.0
// ADR-0020 — the allocation-free validator. Called from the hot path on
// every /openral/candidate_action; the test_no_alloc.cpp gtest pins this
// guarantee with a counting allocator.

#pragma once

#include <cstdint>
#include <cstring>
#include <string_view>

#include "openral_safety_kernel/envelope.hpp"
#include "openral_safety_kernel/result.hpp"

namespace openral_safety_kernel {

/// Violation kinds the validator can report.
///
/// Numeric values map 1:1 onto ``openral_msgs/FailureTrigger.KIND_*``
/// constants so the lifecycle node can publish the violation without
/// translation. CLAUDE.md §1.3 — schemas are the contract.
enum class ViolationKind : std::uint8_t {
  kForce = 1,        ///< openral_msgs::FailureTrigger::KIND_FORCE
  kWorkspace = 2,    ///< openral_msgs::FailureTrigger::KIND_WORKSPACE
  kController = 5,   ///< openral_msgs::FailureTrigger::KIND_CONTROLLER
  // ADR-0030 — geometric self/world collision. The value must exist so the
  // enum stays 1:1 with the IDL; the geometric check that *emits* it lands
  // in a follow-up PR (this kernel does not yet produce kCollision).
  kCollision = 10,   ///< openral_msgs::FailureTrigger::KIND_COLLISION
};

/// Sub-classification carried inside ControllerEvidence.state.
enum class ControllerSubKind : std::uint8_t {
  kNone = 0,
  kDimMismatch = 1,
  kNanInAction = 2,
  kNdofMismatch = 3,
  kEnvelopeUnconfigured = 4,
};

/// Fixed-size violation record — NO heap allocation. Returned by value
/// from `validate()`; the caller serialises it into `evidence_json` on
/// the publish path using a fixed-size buffer (no `std::string`).
struct Violation {
  ViolationKind kind{ViolationKind::kController};
  ControllerSubKind sub{ControllerSubKind::kNone};
  /// Joint index when the violation is per-joint; 0xFFFF otherwise.
  std::uint16_t joint_index{0xFFFF};
  /// Chunk step (horizon index) where the violation was first detected.
  std::uint16_t horizon_step{0xFFFF};
  /// Measured / proposed value.
  double offending_value{0.0};
  /// Bound the value crossed.
  double limit_value{0.0};

  // Fixed-size scratch for downstream evidence_json construction. Keeps
  // the validator allocation-free; the publisher copies bytes out and
  // never holds onto the buffer.
  static constexpr std::size_t kFieldLen = 32;
  char field[kFieldLen]{};

  void set_field(std::string_view sv) noexcept {
    const std::size_t n = sv.size() < (kFieldLen - 1) ? sv.size() : kFieldLen - 1;
    std::memcpy(field, sv.data(), n);
    field[n] = '\0';
  }
};

/// View onto the incoming `openral_msgs::ActionChunk` decoded enough for
/// the validator. Avoids depending on the IDL header in this file so
/// `validator.hpp` stays small and easy to unit-test.
struct ChunkView {
  std::uint8_t control_mode{0};
  std::uint16_t horizon{0};
  std::uint8_t n_dof{0};
  const double* flat_data{nullptr};
  std::size_t flat_size{0};
};

/// Control-mode constants mirroring ``openral_core.ControlMode``
/// (the canonical Python source-of-truth in
/// ``python/core/src/openral_core/schemas.py:CONTROL_MODE_TO_UINT8``).
/// MUST stay in lock-step with that mapping — both sides decode the
/// ``openral_msgs/ActionChunk.control_mode`` uint8 field.
enum class ControlMode : std::uint8_t {
  kJointPosition = 0,
  kJointVelocity = 1,
  kJointTorque = 2,
  kJointTrajectory = 3,
  kCartesianPose = 4,
  kCartesianDelta = 5,
  kCartesianTwist = 6,
  kBodyTwist = 7,
  kFootPlacement = 8,
  kGripperBinary = 9,
  kGripperPosition = 10,
  kDexHandJoint = 11,
  // ADR-0028d — sim-only robosuite-composite multiplexer flag (e.g.
  // HybridMobileBase reads ``action[-1]`` to switch between arm-active
  // and base-active modes). 1-D, value range ``[-1, +1]``. Real-HW
  // adapters ignore this mode (independent controllers run concurrently).
  kCompositeMode = 12,
};

/// Validate one chunk against the envelope intersection.
///
/// Allocation-free; returns `Result<void, Violation>` by value. The
/// fault latch is the caller's job — `validate()` is pure.
///
/// CLAUDE.md §1.4 — reject, do not clamp. Returning Violation means the
/// kernel drops the chunk and publishes FailureTrigger + estop.
Result<void, Violation> validate(const ChunkView& chunk,
                                 const EnvelopeIntersection& envelope) noexcept;

}  // namespace openral_safety_kernel
