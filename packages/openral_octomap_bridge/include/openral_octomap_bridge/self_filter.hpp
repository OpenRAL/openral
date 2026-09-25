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
// A box that carries the kernel's tight geometry (`collision_box_hull` /
// `collision_hull_*`: a 26-DOP and, when it fits the kernel's vertex budget,
// the link's exact convex hull) is filtered against that geometry, not the
// box. The box is the hull's broad-phase bound and its slack (tens of mm at the
// corners) would otherwise widen the blind shell below; see
// `self_hull_distance` for the distance and why it is never an over-report.
//
// The cost is a blind shell: anything within `padding` of the robot's surface
// is removed too, so the kernel cannot see an obstacle touching the arm. That
// trade is recorded in the safety hazard log; `padding` is the knob.

#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include <tf2/LinearMath/Transform.h>
#include <tf2/LinearMath/Vector3.h>

#include "openral_octomap_bridge/payload_clearing.hpp"

namespace openral_octomap_bridge {

/// Joint connecting a link to its parent — the kernel's `JointKind` codes.
enum class SelfJointKind : std::uint8_t { kFixed = 0, kRevolute = 1, kPrismatic = 2 };

/// Number of 26-DOP axis directions (3 face, 4 corner, 6 edge) — the kernel's
/// `kDopAxes`.
inline constexpr int kSelfDopAxes = 13;

/// The 26-DOP's unit axes in the owning box's own frame, row for row the
/// kernel's `kDopAxis` (pinned by `test_self_filter.cpp`). The first three ARE
/// the box's axes.
inline constexpr double kSelfDopAxis[kSelfDopAxes][3] = {
    {1.0, 0.0, 0.0},
    {0.0, 1.0, 0.0},
    {0.0, 0.0, 1.0},
    {0.5773502691896258, 0.5773502691896258, 0.5773502691896258},
    {0.5773502691896258, 0.5773502691896258, -0.5773502691896258},
    {0.5773502691896258, -0.5773502691896258, 0.5773502691896258},
    {0.5773502691896258, -0.5773502691896258, -0.5773502691896258},
    {0.7071067811865476, 0.7071067811865476, 0.0},
    {0.7071067811865476, -0.7071067811865476, 0.0},
    {0.7071067811865476, 0.0, 0.7071067811865476},
    {0.7071067811865476, 0.0, -0.7071067811865476},
    {0.0, 0.7071067811865476, 0.7071067811865476},
    {0.0, 0.7071067811865476, -0.7071067811865476},
};

/// The kernel's `kMaxTightHullVertices`: a hull over it is refused at
/// configure time. It bounds the per-point cost (the support scan is linear in
/// the vertex count), exactly as it bounds the kernel's per-cell cost.
inline constexpr int kSelfMaxHullVertices = 320;

/// The kernel's `kTightContainmentEpsilonM`: numerical slack on "a hull vertex
/// lies inside its own DOP" (the slabs are tangent to the same vertex set).
inline constexpr double kSelfHullContainmentEpsilonM = 1e-9;

/// The kernel's `kGjkMaxIterations` / `kGjkTolerance`. Stopping at the cap is
/// not a failure: every iterate is a lower bound on the distance.
inline constexpr int kSelfGjkMaxIterations = 24;
inline constexpr double kSelfGjkTolerance = 1e-9;

/// Tight geometry refining one box, in that box's own frame — the kernel's
/// `LinkHull` plus its CSR slice of `hull_vertices`.
struct SelfHull {
  double dop_lo[kSelfDopAxes]{};       ///< per-axis lower slab
  double dop_hi[kSelfDopAxes]{};       ///< per-axis upper slab
  std::vector<tf2::Vector3> vertices;  ///< exact convex hull; empty = DOP only
};

/// Lower bound on the Euclidean distance from `local` (a point in the box's
/// frame) to the link's tight geometry; `<= 0` inside it.
///
/// Mirrors the kernel's staged narrow phase with the voxel cube shrunk to a
/// point (`dop_cell_lower_bound` then `hull_cell_distance`, `half_side = 0`;
/// pinned against both by `test_self_filter.cpp`):
///
/// * Stage 1, the 26-DOP: the largest violation of any of the 26 slab
///   halfspaces. Each halfspace contains the DOP, the DOP contains the hull,
///   so the distance to one halfspace can never exceed the distance to the
///   hull: a lower bound, exact on a DOP face.
/// * Stage 2, when the hull's vertices are present: GJK on the vertex list.
///   Every iterate yields the supporting-hyperplane bound `n·p - h(n)` with
///   `h` the hull's TRUE support — an exhaustive vertex scan, never a hill
///   climb — so every value it can return is a lower bound too, and it equals
///   the distance to `kSelfGjkTolerance` on convergence. It stops early once
///   the bound exceeds `stop_above` (certainly farther than that), or once a
///   hull point is found within `stop_at_or_below` (certainly nearer: the
///   bound it returns is then <= that too); the iteration cap or an enclosing
///   simplex return the stage-1 figure. None of these can make the answer an
///   over-report. Pass `stop_above = +inf` for the converged distance.
///
/// The result is the larger of the two bounds, so it is never an over-report:
/// a return the hull explains is always inside the padding. Allocation-free;
/// cost is bounded by `kSelfGjkMaxIterations * kSelfMaxHullVertices` dot
/// products.
double
self_hull_distance(const SelfHull& hull, const tf2::Vector3& local, double stop_above,
                   double stop_at_or_below = -std::numeric_limits<double>::infinity()) noexcept;

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
  /// Tight geometry, the kernel's CSR layout: `box_hull[b]` indexes the hull
  /// refining box `b` (-1 = none); hull `h` owns 13 `dop_lo` / `dop_hi`
  /// entries and `vertex_count[h]` xyz triples of `vertices` starting at vertex
  /// `vertex_first[h]`. All empty = no tight geometry.
  std::vector<std::int64_t> box_hull;
  std::vector<double> hull_dop_lo;
  std::vector<double> hull_dop_hi;
  std::vector<std::int64_t> hull_vertex_first;
  std::vector<std::int64_t> hull_vertex_count;
  std::vector<double> hull_vertices;
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
  std::vector<int> primitive_hull;           ///< parallel: index into `hulls`, or -1
  std::vector<SelfHull> hulls;               ///< tight geometry, box frame
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
///
/// Tight geometry is held to the kernel's `validate_tight_hull` chain (hull
/// vertices inside their 26-DOP, the DOP inside its box, finite, non-inverted,
/// positive extent, at most `kSelfMaxHullVertices` vertices) plus the kernel
/// loader's shape checks. Where the kernel falls back to the plain box on a
/// hull it cannot prove, the filter refuses the whole model: its fallback is
/// forwarding nothing, which the kernel then sees as a stale map.
bool build_self_model(const SelfModelParams& params, SelfModel& out, std::string& error);

/// Forward kinematics for one joint-position vector (`q`, length `n_dof`):
/// fills `link_world[i]` with link `i`'s frame in the model root's frame. A dof
/// index at or past `n_dof` leaves that joint at zero, as the kernel does.
void self_forward_kinematics(const SelfModel& model, const double* q, std::size_t n_dof,
                             std::vector<tf2::Transform>& link_world);

/// Place every robot primitive in the frame `frame_from_root` maps the model
/// root into (typically the cloud's optical frame), appending to `out`. When
/// `out_hulls` is given, each primitive's tight geometry (or null) is appended
/// to it in parallel; the pointers are into `model` and live as long as it.
void place_self_primitives(const SelfModel& model, const std::vector<tf2::Transform>& link_world,
                           const tf2::Transform& frame_from_root,
                           std::vector<PayloadPrimitive>& out,
                           std::vector<const SelfHull*>* out_hulls = nullptr);

/// A set of primitives, all in the cloud's frame, with each one's inverse pose
/// and broad-phase bound precomputed once per cloud.
class SelfMask {
public:
  /// Replace the mask's primitives. `padding_m` is clamped to be non-negative.
  /// `hulls[i]`, where present and non-null, is primitive `i`'s tight geometry
  /// (a box's); primitives past the end of `hulls` (a payload's) have none.
  void set(const std::vector<PayloadPrimitive>& primitives, double padding_m,
           const std::vector<const SelfHull*>& hulls = {});
  /// Is `p` (cloud frame) within the padding of any primitive?
  bool contains(const tf2::Vector3& p) const noexcept;
  std::size_t size() const noexcept { return entries_.size(); }

private:
  struct Entry {
    PayloadPrimitive primitive;
    const SelfHull* hull{nullptr};  ///< refines a box primitive; null = none
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
