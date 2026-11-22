// SPDX-License-Identifier: Apache-2.0
// See peer_grid.hpp.

#include "openral_octomap_bridge/peer_grid.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>

namespace openral_octomap_bridge {

int peer_cells_per_axis(double radius_m, double resolution_m) noexcept {
  return static_cast<int>(std::floor(2.0 * radius_m / resolution_m + 0.5 + 1e-9)) + 1;
}

PeerGrid empty_peer_grid(const tf2::Vector3& centre, double radius_m, double resolution_m) {
  PeerGrid grid;
  grid.resolution = resolution_m;
  grid.n = peer_cells_per_axis(radius_m, resolution_m);
  const double half = grid.n * resolution_m / 2.0;
  grid.origin = centre - tf2::Vector3(half, half, half);
  grid.occupancy.assign(static_cast<std::size_t>(grid.n) * grid.n * grid.n, 0);
  return grid;
}

namespace {

bool finite(const tf2::Vector3& v) {
  return std::isfinite(v.x()) && std::isfinite(v.y()) && std::isfinite(v.z());
}

bool primitive_finite(const PayloadPrimitive& prim) {
  const tf2::Quaternion r = prim.pose.getRotation();
  return finite(prim.pose.getOrigin()) && std::isfinite(r.x()) && std::isfinite(r.y()) &&
         std::isfinite(r.z()) && std::isfinite(r.w()) && std::isfinite(prim.radius) &&
         std::isfinite(prim.half_length) && finite(prim.half_extents);
}

}  // namespace

bool rasterize_primitives(const std::vector<PayloadPrimitive>& primitives, double pad_m,
                          PeerGrid& grid) {
  const double res = grid.resolution;
  const double reach_pad = res * std::sqrt(3.0) / 2.0 + pad_m;
  const std::size_t n = static_cast<std::size_t>(grid.n);
  // Clamped as a double before the cast: a coordinate far outside the grid (or a
  // huge one) is a cell index past the edge, never an int overflow.
  const auto index = [&](double v, double o) {
    return static_cast<int>(std::clamp(std::floor((v - o) / res), 0.0, static_cast<double>(n)));
  };
  for (const auto& prim : primitives) {
    if (!primitive_finite(prim)) {
      return false;
    }
    const tf2::Vector3 c = prim.pose.getOrigin();
    const double r = primitive_bounding_radius(prim) + reach_pad;
    int lo[3];
    int hi[3];
    for (int a = 0; a < 3; ++a) {
      lo[a] = index(c[a] - r, grid.origin[a]);
      hi[a] = std::min(index(c[a] + r, grid.origin[a]) + 1, grid.n);
    }
    const tf2::Transform inv = prim.pose.inverse();
    for (int iz = lo[2]; iz < hi[2]; ++iz) {
      for (int iy = lo[1]; iy < hi[1]; ++iy) {
        for (int ix = lo[0]; ix < hi[0]; ++ix) {
          std::uint8_t& cell =
              grid.occupancy[static_cast<std::size_t>(ix) +
                             n * (static_cast<std::size_t>(iy) + n * static_cast<std::size_t>(iz))];
          if (cell != 0) {
            continue;  // an earlier primitive or swept pose already holds it
          }
          const tf2::Vector3 p =
              grid.origin + tf2::Vector3((ix + 0.5) * res, (iy + 0.5) * res, (iz + 0.5) * res);
          if (primitive_surface_distance(prim, inv * p) <= reach_pad) {
            cell = 1;
          }
        }
      }
    }
  }
  return true;
}

void PeerJoints::update(const std::vector<int>& dof, const std::vector<double>& position,
                        const std::vector<double>& velocity, std::int64_t stamp_ns) {
  constexpr std::int64_t kMinDifferenceNs = 1'000'000;  // 1 ms
  for (std::size_t i = 0; i < dof.size() && i < position.size(); ++i) {
    const int d = dof[i];
    if (d < 0 || static_cast<std::size_t>(d) >= q_.size() || !std::isfinite(position[i])) {
      continue;
    }
    const std::size_t k = static_cast<std::size_t>(d);
    if (stamp_ns <= stamp_ns_[k]) {
      continue;  // best-effort delivery: an older sample never replaces a newer one
    }
    if (i < velocity.size() && std::isfinite(velocity[i])) {
      dq_[k] = velocity[i];
    } else if (stamp_ns_[k] != 0 && stamp_ns - stamp_ns_[k] >= kMinDifferenceNs) {
      dq_[k] = (position[i] - q_[k]) / (static_cast<double>(stamp_ns - stamp_ns_[k]) * 1e-9);
    }
    q_[k] = position[i];
    stamp_ns_[k] = stamp_ns;
  }
}

std::vector<double> PeerJoints::q_at(std::int64_t t_ns) const {
  std::vector<double> out(q_);
  for (std::size_t d = 0; d < out.size(); ++d) {
    if (stamp_ns_[d] != 0) {
      out[d] += dq_[d] * static_cast<double>(t_ns - stamp_ns_[d]) * 1e-9;
    }
  }
  return out;
}

bool PeerJoints::complete(const std::vector<std::size_t>& required) const {
  return std::all_of(required.begin(), required.end(),
                     [&](std::size_t d) { return d < stamp_ns_.size() && stamp_ns_[d] != 0; });
}

std::int64_t PeerJoints::oldest_ns(const std::vector<std::size_t>& required) const {
  std::int64_t out = std::numeric_limits<std::int64_t>::max();
  for (const auto d : required) {
    out = std::min(out, stamp_ns_[d]);
  }
  return out;
}

std::int64_t PeerJoints::newest_ns(const std::vector<std::size_t>& required) const {
  std::int64_t out = std::numeric_limits<std::int64_t>::min();
  for (const auto d : required) {
    out = std::max(out, stamp_ns_[d]);
  }
  return out;
}

namespace {

double yaw_of(const tf2::Transform& x) {
  double roll = 0.0;
  double pitch = 0.0;
  double yaw = 0.0;
  tf2::Matrix3x3(x.getRotation()).getRPY(roll, pitch, yaw);
  return yaw;
}

}  // namespace

tf2::Transform advance_planar(const tf2::Transform& world_from_base, const PlanarTwist& twist,
                              double t) noexcept {
  tf2::Quaternion turn;
  turn.setRPY(0.0, 0.0, twist.wz * t);
  tf2::Transform out = world_from_base;
  out.setRotation(turn * world_from_base.getRotation());
  out.setOrigin(world_from_base.getOrigin() + tf2::Vector3(twist.vx * t, twist.vy * t, 0.0));
  return out;
}

PlanarTwist planar_twist(const tf2::Transform& before, const tf2::Transform& after,
                         double dt) noexcept {
  if (!(dt > 0.0)) {
    return {};
  }
  const tf2::Vector3 d = after.getOrigin() - before.getOrigin();
  return {d.x() / dt, d.y() / dt, std::remainder(yaw_of(after) - yaw_of(before), 2.0 * M_PI) / dt};
}

bool peer_swept_primitives(const SelfModel& model, const std::vector<double>& q,
                           const std::vector<double>& dq, const std::vector<std::size_t>& base_dofs,
                           const tf2::Transform& world_from_own_base, const PlanarTwist& own_twist,
                           const tf2::Transform& world_from_peer_base,
                           const PlanarTwist& peer_twist, double horizon_s,
                           std::vector<PayloadPrimitive>& out) {
  out.clear();
  const double h = std::max(horizon_s, 0.0);
  // The poses the sweep needs. A NaN (`std::max` would quietly skip it) or a step count
  // past the cap withholds the grid.
  double steps = 0.0;
  bool usable = true;
  const auto need = [&](double v) {
    usable = usable && std::isfinite(v);
    steps = std::max(steps, v);
  };
  // Only joints that move a collision link count: a fast auxiliary joint the peer's
  // geometry never uses must not withhold its grid.
  for (const int d : model.dof_index) {
    const auto i = static_cast<std::size_t>(d);
    if (d >= 0 && i < dq.size() &&
        std::find(base_dofs.begin(), base_dofs.end(), i) == base_dofs.end()) {
      need(std::abs(dq[i]) * h / kPeerJointStepRad);
    }
  }
  for (const PlanarTwist* tw : {&own_twist, &peer_twist}) {
    need(std::hypot(tw->vx, tw->vy) * h / kPeerBaseStepM);
    need(std::abs(tw->wz) * h / kPeerJointStepRad);
  }
  if (!usable || steps > static_cast<double>(kPeerMaxSweepPoses - 1)) {
    return false;
  }
  const int n = std::max(1, static_cast<int>(std::ceil(steps)));
  std::vector<double> qk(q.size());
  std::vector<tf2::Transform> link_world;
  for (int k = 0; k <= n; ++k) {
    const double t = h * k / n;
    for (std::size_t i = 0; i < q.size(); ++i) {
      const double v = i < dq.size() ? dq[i] : 0.0;
      qk[i] = q[i] + v * t;
    }
    for (const auto d : base_dofs) {
      if (d < qk.size()) {
        qk[d] = 0.0;
      }
    }
    self_forward_kinematics(model, qk.data(), qk.size(), link_world);
    place_self_primitives(model, link_world,
                          advance_planar(world_from_own_base, own_twist, t).inverse() *
                              advance_planar(world_from_peer_base, peer_twist, t),
                          out);
  }
  return true;
}

}  // namespace openral_octomap_bridge
