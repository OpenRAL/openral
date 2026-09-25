// SPDX-License-Identifier: Apache-2.0
// Robot (and held-payload) self-filter for a real depth cloud, ahead of OctoMap.
//
// In simulation the depth renderer makes the robot's own bodies — and a grasped
// payload — transparent, so the world map never contains the robot. A real
// camera hands over a finished cloud in which the arm is just more points; they
// become occupancy around every link in view, and the kernel's world-voxel
// check then stops each link against its own surface. This is the real-camera
// counterpart of that exclusion, at the same place in the pipeline: points that
// lie within `padding` of the robot's collision primitives (or an attached
// payload's) are removed from the cloud before `octomap_server` inserts it.
//
// The geometry is the kernel's own. `SelfModel` is built from the same
// `collision_*` parameters the deploy launch hands the safety kernel and posed
// by a forward kinematics that mirrors the kernel's operation for operation
// (pinned by `test_self_filter.cpp` against the kernel library itself), so the
// filter removes exactly the returns that would otherwise become self-hits.
// The FK is a deliberate mirror rather than a link against the kernel: this is
// Layer 2 and the kernel is Layer 5, and a non-adjacent layer dependency is not
// allowed (CLAUDE.md §3). The kernel link exists only in the test.
//
// The cost is a blind shell: anything within `padding` of the robot's surface
// is removed too, so the kernel cannot see an obstacle touching the arm. That
// trade is recorded in the safety hazard log; `padding` is the knob.

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <tf2/LinearMath/Transform.h>
#include <tf2/LinearMath/Vector3.h>

#include "openral_octomap_bridge/payload_clearing.hpp"

namespace openral_octomap_bridge {

/// Joint connecting a link to its parent — the kernel's `JointKind` codes.
enum class SelfJointKind : std::uint8_t { kFixed = 0, kRevolute = 1, kPrismatic = 2 };

/// The kernel's `collision_*` parameter arrays, exactly as the launch passes them.
struct SelfModelParams {
  std::int64_t n_links{0};
  std::vector<std::int64_t> parent;
  std::vector<std::int64_t> joint_kind;
  std::vector<std::int64_t> dof_index;
  std::vector<double> origin_xyzrpy;
  std::vector<double> axis;
  std::vector<std::string> link_names;
  std::vector<std::int64_t> capsule_link;
  std::vector<double> capsule_radius;
  std::vector<double> capsule_half_length;
  std::vector<double> capsule_origin_xyzrpy;
  std::vector<std::int64_t> box_link;
  std::vector<double> box_half_extents;
  std::vector<double> box_origin_xyzrpy;
};

/// Flattened kinematic tree plus every link-attached primitive, link frame.
struct SelfModel {
  std::vector<std::string> link_names;
  std::vector<int> parent;
  std::vector<SelfJointKind> joint_kind;
  std::vector<int> dof_index;
  std::vector<tf2::Transform> origin;
  std::vector<tf2::Vector3> axis;
  std::vector<int> primitive_link;           ///< link index each primitive rides
  std::vector<PayloadPrimitive> primitives;  ///< pose = primitive in its link frame
  std::size_t required_dof{0};               ///< highest dof index used, plus one

  std::size_t n_links() const noexcept { return parent.size(); }
  /// Index of `name` in `link_names`, or -1.
  int link_index(const std::string& name) const noexcept;
};

/// `R = Rz(yaw)·Ry(pitch)·Rx(roll)` then translate — the kernel's
/// `transform_from_xyz_rpy` convention.
tf2::Transform self_transform_from_xyz_rpy(double x, double y, double z, double roll, double pitch,
                                           double yaw) noexcept;

/// Build the model from the launch's parameters. Fails (returns `false`, sets
/// `error`, leaves `out` empty) on any shape disagreement, an out-of-range
/// parent or primitive link, a parent that is not topologically earlier, an
/// unknown joint kind, or a non-finite or negative dimension — the same
/// all-or-nothing posture as the kernel's `load_collision_model`.
bool build_self_model(const SelfModelParams& params, SelfModel& out, std::string& error);

/// Forward kinematics for one joint-position vector (`q`, length `n_dof`):
/// fills `link_world[i]` with link `i`'s frame in the model root's frame. A dof
/// index at or past `n_dof` leaves that joint at zero, as the kernel does.
void self_forward_kinematics(const SelfModel& model, const double* q, std::size_t n_dof,
                             std::vector<tf2::Transform>& link_world);

/// Place every robot primitive in the frame `frame_from_root` maps the model
/// root into (typically the cloud's optical frame), appending to `out`.
void place_self_primitives(const SelfModel& model, const std::vector<tf2::Transform>& link_world,
                           const tf2::Transform& frame_from_root,
                           std::vector<PayloadPrimitive>& out);

/// A set of primitives, all in the cloud's frame, with each one's inverse pose
/// and broad-phase bound precomputed once per cloud.
class SelfMask {
public:
  /// Replace the mask's primitives. `padding_m` is clamped to be non-negative.
  void set(const std::vector<PayloadPrimitive>& primitives, double padding_m);
  /// Is `p` (cloud frame) within the padding of any primitive?
  bool contains(const tf2::Vector3& p) const noexcept;
  std::size_t size() const noexcept { return entries_.size(); }

private:
  struct Entry {
    PayloadPrimitive primitive;
    tf2::Transform local_from_frame;
    tf2::Vector3 center;
    double reach_sq{0.0};  ///< (bounding radius + padding)^2
  };
  std::vector<Entry> entries_;
  double padding_m_{0.0};
};

/// Counts from one `filter_xyz_points` call.
struct SelfFilterStats {
  std::size_t points_in{0};   ///< points in the input buffer
  std::size_t non_finite{0};  ///< dropped: a NaN/inf coordinate (no return)
  std::size_t removed{0};     ///< dropped: inside the mask
  std::size_t kept{0};        ///< copied to the output
};

/// Copy every finite point of a packed point buffer that the mask does NOT
/// contain into `out` (appended, `point_step` bytes each, fields untouched).
/// `x_offset` / `y_offset` / `z_offset` locate the three FLOAT32 coordinates in
/// each point. The input layout is `n_points` records of `point_step` bytes.
SelfFilterStats filter_xyz_points(const std::uint8_t* data, std::size_t n_points,
                                  std::size_t point_step, std::size_t x_offset,
                                  std::size_t y_offset, std::size_t z_offset, const SelfMask& mask,
                                  std::vector<std::uint8_t>& out);

}  // namespace openral_octomap_bridge
