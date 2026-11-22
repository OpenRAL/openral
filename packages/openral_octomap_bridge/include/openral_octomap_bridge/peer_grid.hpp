// SPDX-License-Identifier: Apache-2.0
// The other robots of a workcell as occupied cells, for the safety kernel's peer
// grid (`/openral/peer_voxels`, ADR-0137, hazard log Entry 103).
//
// A peer is posed from its own collision model (the `collision_*` parameters its
// own kernel gets), its measured joint positions and velocities, and its base pose
// in the scene's shared `world` frame, then expressed in this robot's base frame
// and rasterized into the axis-aligned grid around this robot's coverage ball.
// Over the sweep horizon the peer is posed again along its measured joint velocity
// and base motion, so its cells still hold where it can be by the time the kernel
// refuses the grid as stale.
//
// Rasterizing is conservative: a cell is occupied when its centre lies within
// half a cell diagonal plus `pad_m` of a primitive. It adds occupancy; it never
// clears any.

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <tf2/LinearMath/Transform.h>
#include <tf2/LinearMath/Vector3.h>

#include "openral_octomap_bridge/payload_clearing.hpp"
#include "openral_octomap_bridge/self_filter.hpp"

namespace openral_octomap_bridge {

/// The base-frame grid around a coverage ball: `n` cells per axis, cell (0,0,0)'s
/// minimum corner at `origin`, x fastest (`OccupancyVoxels`' layout).
struct PeerGrid {
  tf2::Vector3 origin{0.0, 0.0, 0.0};
  double resolution{0.0};
  int n{0};
  std::vector<std::uint8_t> occupancy;
};

/// Cells per axis of the grid around a ball of `radius_m` at `resolution_m`: the
/// deploy launch's `_world_voxel_max_cells` is this cubed, so the grid always fits
/// the kernel's cell cap.
int peer_cells_per_axis(double radius_m, double resolution_m) noexcept;

/// An empty grid around `centre` (base frame).
PeerGrid empty_peer_grid(const tf2::Vector3& centre, double radius_m, double resolution_m);

/// Mark every cell whose centre is within half a cell diagonal plus `pad_m` of a
/// primitive (all in the grid's frame). Cells outside the grid are dropped: beyond
/// this robot's reach. Returns false, marking nothing more, on a primitive with a
/// non-finite pose or size: a peer that cannot be placed is never left out quietly.
bool rasterize_primitives(const std::vector<PayloadPrimitive>& primitives, double pad_m,
                          PeerGrid& grid);

/// One peer's measured joints, per dof: position, velocity and the stamp each was last
/// measured at. A `/joint_states` message refreshes only the dofs it names, so a stream
/// that stops is seen as stale even while another publisher on the topic carries on.
class PeerJoints {
public:
  explicit PeerJoints(std::size_t n_dof) : q_(n_dof, 0.0), dq_(n_dof, 0.0), stamp_ns_(n_dof, 0) {}

  /// One message: `dof[i]` is the dof of its i-th entry (-1 = not this robot's).
  /// `velocity` is empty or parallel to `position`. A dof without a finite measured
  /// velocity gets the change since its own previous sample, over at least 1 ms.
  void update(const std::vector<int>& dof, const std::vector<double>& position,
              const std::vector<double>& velocity, std::int64_t stamp_ns);

  /// Every one of `required` has been measured.
  bool complete(const std::vector<std::size_t>& required) const;
  /// The oldest and newest stamp among `required` (call only when complete).
  std::int64_t oldest_ns(const std::vector<std::size_t>& required) const;
  std::int64_t newest_ns(const std::vector<std::size_t>& required) const;

  const std::vector<double>& q() const { return q_; }
  const std::vector<double>& dq() const { return dq_; }

private:
  std::vector<double> q_;
  std::vector<double> dq_;
  std::vector<std::int64_t> stamp_ns_;  ///< 0 = never measured
};

/// Is a stamp `age_s` old usable within `max_age_s`? Both ways: a stamp further in the
/// future than the bound comes from a clock that disagrees with this one, and would
/// otherwise never age.
inline bool within_age(double age_s, double max_age_s) noexcept {
  return age_s >= -max_age_s && age_s <= max_age_s;
}

/// A peer's planar base motion in `world`, per second.
struct PlanarTwist {
  double vx{0.0};
  double vy{0.0};
  double wz{0.0};
};

/// `world_from_base` advanced by `twist` over `t` seconds (translation and yaw
/// about world z; the peer's base stays level).
tf2::Transform advance_planar(const tf2::Transform& world_from_base, const PlanarTwist& twist,
                              double t) noexcept;

/// The twist that takes `before` to `after` over `dt` seconds (0 when dt <= 0).
PlanarTwist planar_twist(const tf2::Transform& before, const tf2::Transform& after,
                         double dt) noexcept;

/// The peer's primitives in this robot's base frame over `horizon_s`: at each step
/// `t` the peer is posed at `q + dq t` on its base advanced by `peer_twist`, seen
/// from this robot's base advanced by `own_twist` (both move; what the kernel needs
/// is where the peer is relative to this robot's base at the same instant).
/// `base_dofs` are held at zero (the model is posed in the peer's base frame, as its
/// own kernel poses it). Steps are fine enough that no joint moves more than
/// `kPeerJointStepRad` and neither base more than `kPeerBaseStepM` /
/// `kPeerJointStepRad` between two poses. Returns false, with `out` empty, when that
/// takes more than `kPeerMaxSweepPoses` poses (an implausible or garbage velocity):
/// the caller withholds the grid rather than under-sweep it.
bool peer_swept_primitives(const SelfModel& model, const std::vector<double>& q,
                           const std::vector<double>& dq, const std::vector<std::size_t>& base_dofs,
                           const tf2::Transform& world_from_own_base, const PlanarTwist& own_twist,
                           const tf2::Transform& world_from_peer_base,
                           const PlanarTwist& peer_twist, double horizon_s,
                           std::vector<PayloadPrimitive>& out);

inline constexpr double kPeerJointStepRad = 0.02;
inline constexpr double kPeerBaseStepM = 0.01;
/// 2 rad/s, or 1 m/s of base travel, over 0.3 s is 30 poses.
inline constexpr int kPeerMaxSweepPoses = 100;

}  // namespace openral_octomap_bridge
