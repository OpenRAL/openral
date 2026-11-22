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

#include <cstdint>
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
/// this robot's reach.
void rasterize_primitives(const std::vector<PayloadPrimitive>& primitives, double pad_m,
                          PeerGrid& grid);

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

/// The peer's primitives in this robot's base frame, posed at its measured
/// joints `q` and over `horizon_s` along `dq` and `twist`. `base_dofs` are held at
/// zero (the model is posed in the peer's base frame, as its own kernel poses it).
/// Steps are fine enough that no joint moves more than `kPeerJointStepRad` and the
/// base no more than `kPeerBaseStepM` / `kPeerJointStepRad` between two poses.
void peer_swept_primitives(const SelfModel& model, const std::vector<double>& q,
                           const std::vector<double>& dq, const std::vector<std::size_t>& base_dofs,
                           const tf2::Transform& own_base_from_world,
                           const tf2::Transform& world_from_peer_base, const PlanarTwist& twist,
                           double horizon_s, std::vector<PayloadPrimitive>& out);

inline constexpr double kPeerJointStepRad = 0.02;
inline constexpr double kPeerBaseStepM = 0.01;

}  // namespace openral_octomap_bridge
