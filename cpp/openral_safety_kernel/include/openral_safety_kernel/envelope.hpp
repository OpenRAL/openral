// SPDX-License-Identifier: Apache-2.0
// ADR-0020 — robot ceiling ∩ skill envelope. Loaded ONCE at
// on_configure() into an `EnvelopeIntersection` via ROS parameters
// populated by the Python launch from `robots/<id>/robot.yaml`
// (ADR-0020 PR-K). The hot path then consults the struct without
// further parsing. There is exactly one transport: ROS parameters.
// The flat-YAML envelope-file path the kernel had pre-PR-K is gone —
// `openral_safety.envelope_loader.kernel_params_from_envelope` is the
// canonical Python → ROS-params converter.

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

// Forward-declared so headers don't pull rclcpp transitively.
namespace rclcpp {
class Node;
}  // namespace rclcpp
namespace rclcpp_lifecycle {
class LifecycleNode;
}  // namespace rclcpp_lifecycle

namespace openral_safety_kernel {

/// IEEE-754 +infinity sentinel used for "no enforcement on this joint".
inline constexpr double kPosInfinity = std::numeric_limits<double>::infinity();
/// IEEE-754 -infinity sentinel used for "no enforcement on this joint".
inline constexpr double kNegInfinity = -std::numeric_limits<double>::infinity();

/// Per-axis Cartesian workspace AABB.
struct WorkspaceBox {
  std::array<double, 3> min_xyz{kNegInfinity, kNegInfinity, kNegInfinity};
  std::array<double, 3> max_xyz{kPosInfinity, kPosInfinity, kPosInfinity};
  /// `true` if both corners came from the manifest; `false` if the
  /// kernel should treat the workspace as unbounded (no Cartesian check).
  bool set{false};
};

/// The full envelope the validator consults per chunk.
///
/// Mirrors :class:`openral_safety.envelope_loader.EnvelopeIntersection`
/// in Python (which writes the YAML this struct slurps). Joint arrays
/// are sized to `n_dof`; the validator uses `n_dof` to bound-check the
/// incoming `ActionChunk.flat[]` shape.
struct EnvelopeIntersection {
  std::string robot_name;
  std::string rskill_id;
  std::string skill_revision;
  std::size_t n_dof{0};

  // Per-joint limits. Each vector has length `n_dof`.
  std::vector<double> joint_position_min;
  std::vector<double> joint_position_max;
  std::vector<double> joint_velocity_max;  ///< Already pre-multiplied by speed_factor.
  std::vector<double> joint_torque_max;

  // Cartesian envelope.
  WorkspaceBox workspace_box;
  double max_ee_speed_m_s{kPosInfinity};
  double max_ee_accel_m_s2{kPosInfinity};

  // Force / torque caps applied across all joints + cartesian.
  double max_force_n{kPosInfinity};
  double max_torque_nm{kPosInfinity};
  double contact_force_threshold_n{kPosInfinity};

  bool deadman_required{false};
};

/// Outcome of `load_envelope_from_ros_parameters`.
enum class EnvelopeLoadStatus : std::uint8_t {
  kOk = 0,
  kInvalidShape = 1,    ///< joint arrays disagree with `n_dof`
  kUnconfigured = 2,    ///< `n_dof` is 0 — no envelope supplied
};

/// Build the envelope from this node's ROS parameters.
///
/// The Python `sim_e2e.launch.py` unpacks `robots/<id>/robot.yaml` via
/// Pydantic at launch-time, calls
/// `openral_safety.envelope_loader.kernel_params_from_envelope`, and
/// forwards each canonical field as a ROS parameter on this kernel
/// node. The loader reads them back here, validates the joint-array
/// shapes against `n_dof`, and populates `out`.
///
/// Returns `kOk` on success. `kInvalidShape` if joint arrays disagree
/// with `n_dof`. `kUnconfigured` if `n_dof` is 0 (no envelope was
/// supplied — caller MUST refuse to activate the node). `out` is reset
/// on entry so a stale partial value can never leak into the
/// validator. CLAUDE.md §1.4 — explicit failure, no fallback.
EnvelopeLoadStatus load_envelope_from_ros_parameters(
    rclcpp_lifecycle::LifecycleNode& node,
    EnvelopeIntersection& out,
    std::string& error_message);

}  // namespace openral_safety_kernel
