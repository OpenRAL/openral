// SPDX-License-Identifier: Apache-2.0
// See peer_grid.hpp.

#include "openral_octomap_bridge/peer_grid.hpp"

#include <algorithm>
#include <cmath>

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

void rasterize_primitives(const std::vector<PayloadPrimitive>& primitives, double pad_m,
                          PeerGrid& grid) {
  const double res = grid.resolution;
  const double reach_pad = res * std::sqrt(3.0) / 2.0 + pad_m;
  const auto index = [&](double v, double o) {
    return std::clamp(static_cast<int>(std::floor((v - o) / res)), 0, grid.n);
  };
  for (const auto& prim : primitives) {
    const tf2::Vector3 c = prim.pose.getOrigin();
    const double r = primitive_bounding_radius(prim) + reach_pad;
    int lo[3];
    int hi[3];
    for (int a = 0; a < 3; ++a) {
      lo[a] = index(c[a] - r, grid.origin[a]);
      hi[a] = index(c[a] + r, grid.origin[a]) + 1;
      hi[a] = std::min(hi[a], grid.n);
    }
    const tf2::Transform inv = prim.pose.inverse();
    for (int iz = lo[2]; iz < hi[2]; ++iz) {
      for (int iy = lo[1]; iy < hi[1]; ++iy) {
        for (int ix = lo[0]; ix < hi[0]; ++ix) {
          const tf2::Vector3 p =
              grid.origin + tf2::Vector3((ix + 0.5) * res, (iy + 0.5) * res, (iz + 0.5) * res);
          if (primitive_surface_distance(prim, inv * p) <= reach_pad) {
            grid.occupancy[static_cast<std::size_t>(ix) +
                           static_cast<std::size_t>(grid.n) *
                               (iy + static_cast<std::size_t>(grid.n) * iz)] = 1;
          }
        }
      }
    }
  }
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

void peer_swept_primitives(const SelfModel& model, const std::vector<double>& q,
                           const std::vector<double>& dq, const std::vector<std::size_t>& base_dofs,
                           const tf2::Transform& own_base_from_world,
                           const tf2::Transform& world_from_peer_base, const PlanarTwist& twist,
                           double horizon_s, std::vector<PayloadPrimitive>& out) {
  out.clear();
  const double h = std::max(horizon_s, 0.0);
  double steps = 0.0;
  for (std::size_t i = 0; i < q.size() && i < dq.size(); ++i) {
    if (std::find(base_dofs.begin(), base_dofs.end(), i) == base_dofs.end()) {
      steps = std::max(steps, std::abs(dq[i]) * h / kPeerJointStepRad);
    }
  }
  steps = std::max(steps, std::hypot(twist.vx, twist.vy) * h / kPeerBaseStepM);
  steps = std::max(steps, std::abs(twist.wz) * h / kPeerJointStepRad);
  const int n = std::max(1, static_cast<int>(std::ceil(steps)));
  std::vector<double> qk(q.size());
  std::vector<tf2::Transform> link_world;
  for (int k = 0; k <= n; ++k) {
    const double t = h * k / n;
    for (std::size_t i = 0; i < q.size(); ++i) {
      const double v = i < dq.size() && std::isfinite(dq[i]) ? dq[i] : 0.0;
      qk[i] = q[i] + v * t;
    }
    for (const auto d : base_dofs) {
      if (d < qk.size()) {
        qk[d] = 0.0;
      }
    }
    self_forward_kinematics(model, qk.data(), qk.size(), link_world);
    place_self_primitives(model, link_world,
                          own_base_from_world * advance_planar(world_from_peer_base, twist, t),
                          out);
  }
}

}  // namespace openral_octomap_bridge
