// SPDX-License-Identifier: Apache-2.0
// Robot (and held-payload) self-filter for a real depth cloud, ahead of OctoMap.

#include "openral_octomap_bridge/self_filter.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>

#include <openral_msgs/msg/attached_collision_primitive.hpp>

namespace openral_octomap_bridge {

namespace {

using Wire = openral_msgs::msg::AttachedCollisionPrimitive;

bool finite_non_negative(double v) { return std::isfinite(v) && v >= 0.0; }

bool all_finite(const std::vector<double>& v) {
  return std::all_of(v.begin(), v.end(), [](double x) { return std::isfinite(x); });
}

tf2::Transform xyzrpy_at(const std::vector<double>& flat, std::size_t i) {
  return self_transform_from_xyz_rpy(flat[6 * i + 0], flat[6 * i + 1], flat[6 * i + 2],
                                     flat[6 * i + 3], flat[6 * i + 4], flat[6 * i + 5]);
}

}  // namespace

int SelfModel::link_index(const std::string& name) const noexcept {
  for (std::size_t i = 0; i < link_names.size(); ++i) {
    if (link_names[i] == name) {
      return static_cast<int>(i);
    }
  }
  return -1;
}

tf2::Transform self_transform_from_xyz_rpy(double x, double y, double z, double roll, double pitch,
                                           double yaw) noexcept {
  const double cr = std::cos(roll);
  const double sr = std::sin(roll);
  const double cp = std::cos(pitch);
  const double sp = std::sin(pitch);
  const double cy = std::cos(yaw);
  const double sy = std::sin(yaw);
  // R = Rz(yaw) * Ry(pitch) * Rx(roll), row-major — the kernel's
  // `transform_from_xyz_rpy`, element for element.
  const tf2::Matrix3x3 r(cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr,  //
                         sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr,  //
                         -sp, cp * sr, cp * cr);
  return tf2::Transform(r, tf2::Vector3(x, y, z));
}

bool build_self_model(const SelfModelParams& p, SelfModel& out, std::string& error) {
  out = SelfModel{};
  if (p.n_links <= 0) {
    error = "collision_n_links must be > 0";
    return false;
  }
  const auto n = static_cast<std::size_t>(p.n_links);
  const std::size_t n_caps = p.capsule_radius.size();
  const std::size_t n_boxes = p.box_link.size();
  if (p.parent.size() != n || p.joint_kind.size() != n || p.dof_index.size() != n ||
      p.origin_xyzrpy.size() != 6 * n || p.axis.size() != 3 * n || p.link_names.size() != n ||
      p.capsule_link.size() != n_caps || p.capsule_half_length.size() != n_caps ||
      p.capsule_origin_xyzrpy.size() != 6 * n_caps || p.box_half_extents.size() != 3 * n_boxes ||
      p.box_origin_xyzrpy.size() != 6 * n_boxes) {
    error = "collision_* array shapes disagree with collision_n_links / primitive count";
    return false;
  }
  if (!all_finite(p.origin_xyzrpy) || !all_finite(p.axis) || !all_finite(p.capsule_origin_xyzrpy) ||
      !all_finite(p.box_origin_xyzrpy)) {
    error = "collision_* carries a non-finite origin or axis";
    return false;
  }
  SelfModel m;
  m.link_names = p.link_names;
  m.parent.resize(n);
  m.joint_kind.resize(n);
  m.dof_index.resize(n);
  m.origin.resize(n);
  m.axis.resize(n);
  for (std::size_t i = 0; i < n; ++i) {
    const auto parent = p.parent[i];
    if (parent >= static_cast<std::int64_t>(i) || parent < -1) {
      error = "collision_parent is not topologically ordered (parent must precede child)";
      return false;
    }
    if (p.joint_kind[i] < 0 || p.joint_kind[i] > 2) {
      error = "collision_joint_kind carries an unknown joint code";
      return false;
    }
    m.parent[i] = static_cast<int>(parent);
    m.joint_kind[i] = static_cast<SelfJointKind>(static_cast<std::uint8_t>(p.joint_kind[i]));
    m.dof_index[i] = static_cast<int>(p.dof_index[i]);
    m.origin[i] = xyzrpy_at(p.origin_xyzrpy, i);
    m.axis[i] = tf2::Vector3(p.axis[3 * i + 0], p.axis[3 * i + 1], p.axis[3 * i + 2]);
    if (m.dof_index[i] >= 0) {
      m.required_dof = std::max(m.required_dof, static_cast<std::size_t>(m.dof_index[i]) + 1);
    }
  }
  for (std::size_t c = 0; c < n_caps; ++c) {
    const auto link = p.capsule_link[c];
    if (link < 0 || static_cast<std::size_t>(link) >= n) {
      error = "collision_capsule_link out of range";
      return false;
    }
    if (!finite_non_negative(p.capsule_radius[c]) ||
        !finite_non_negative(p.capsule_half_length[c])) {
      error = "collision capsule with a non-finite or negative dimension";
      return false;
    }
    PayloadPrimitive prim;
    // A sphere is the kernel's zero-length capsule; the capsule segment runs
    // along the primitive's local Z in both conventions.
    prim.shape_type = Wire::SHAPE_CAPSULE;
    prim.radius = p.capsule_radius[c];
    prim.half_length = p.capsule_half_length[c];
    prim.pose = xyzrpy_at(p.capsule_origin_xyzrpy, c);
    m.primitive_link.push_back(static_cast<int>(link));
    m.primitives.push_back(prim);
  }
  for (std::size_t b = 0; b < n_boxes; ++b) {
    const auto link = p.box_link[b];
    if (link < 0 || static_cast<std::size_t>(link) >= n) {
      error = "collision_box_link out of range";
      return false;
    }
    const tf2::Vector3 he(p.box_half_extents[3 * b + 0], p.box_half_extents[3 * b + 1],
                          p.box_half_extents[3 * b + 2]);
    if (!finite_non_negative(he.x()) || !finite_non_negative(he.y()) ||
        !finite_non_negative(he.z())) {
      error = "collision box with a non-finite or negative half extent";
      return false;
    }
    PayloadPrimitive prim;
    prim.shape_type = Wire::SHAPE_BOX;
    prim.half_extents = he;
    prim.pose = xyzrpy_at(p.box_origin_xyzrpy, b);
    m.primitive_link.push_back(static_cast<int>(link));
    m.primitives.push_back(prim);
  }
  out = std::move(m);
  return true;
}

void self_forward_kinematics(const SelfModel& model, const double* q, std::size_t n_dof,
                             std::vector<tf2::Transform>& link_world) {
  const std::size_t n = model.n_links();
  link_world.resize(n);
  for (std::size_t i = 0; i < n; ++i) {
    tf2::Transform motion = tf2::Transform::getIdentity();
    const int dof = model.dof_index[i];
    if (dof >= 0 && static_cast<std::size_t>(dof) < n_dof) {
      const double angle = q[static_cast<std::size_t>(dof)];
      if (model.joint_kind[i] == SelfJointKind::kRevolute) {
        motion.setRotation(tf2::Quaternion(model.axis[i], angle));
      } else if (model.joint_kind[i] == SelfJointKind::kPrismatic) {
        motion.setOrigin(model.axis[i] * angle);
      }
    }
    const tf2::Transform local = model.origin[i] * motion;
    const int parent = model.parent[i];
    link_world[i] = parent < 0 ? local : link_world[static_cast<std::size_t>(parent)] * local;
  }
}

void place_self_primitives(const SelfModel& model, const std::vector<tf2::Transform>& link_world,
                           const tf2::Transform& frame_from_root,
                           std::vector<PayloadPrimitive>& out) {
  for (std::size_t k = 0; k < model.primitives.size(); ++k) {
    PayloadPrimitive placed = model.primitives[k];
    const auto link = static_cast<std::size_t>(model.primitive_link[k]);
    placed.pose = frame_from_root * link_world[link] * model.primitives[k].pose;
    out.push_back(placed);
  }
}

void SelfMask::set(const std::vector<PayloadPrimitive>& primitives, double padding_m) {
  padding_m_ = std::isfinite(padding_m) && padding_m > 0.0 ? padding_m : 0.0;
  entries_.clear();
  entries_.reserve(primitives.size());
  for (const auto& prim : primitives) {
    Entry e;
    e.primitive = prim;
    e.local_from_frame = prim.pose.inverse();
    e.center = prim.pose.getOrigin();
    const double reach = primitive_bounding_radius(prim) + padding_m_;
    e.reach_sq = reach * reach;
    entries_.push_back(e);
  }
}

bool SelfMask::contains(const tf2::Vector3& p) const noexcept {
  for (const auto& e : entries_) {
    if ((p - e.center).length2() > e.reach_sq) {
      continue;
    }
    if (primitive_surface_distance(e.primitive, e.local_from_frame * p) <= padding_m_) {
      return true;
    }
  }
  return false;
}

SelfFilterStats filter_xyz_points(const std::uint8_t* data, std::size_t n_points,
                                  std::size_t point_step, std::size_t x_offset,
                                  std::size_t y_offset, std::size_t z_offset, const SelfMask& mask,
                                  std::vector<std::uint8_t>& out) {
  SelfFilterStats stats;
  stats.points_in = n_points;
  out.reserve(out.size() + n_points * point_step);
  for (std::size_t i = 0; i < n_points; ++i) {
    const std::uint8_t* rec = data + i * point_step;
    float x = 0.0F;
    float y = 0.0F;
    float z = 0.0F;
    std::memcpy(&x, rec + x_offset, sizeof(float));
    std::memcpy(&y, rec + y_offset, sizeof(float));
    std::memcpy(&z, rec + z_offset, sizeof(float));
    if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) {
      ++stats.non_finite;
      continue;
    }
    if (mask.contains(tf2::Vector3(x, y, z))) {
      ++stats.removed;
      continue;
    }
    out.insert(out.end(), rec, rec + point_step);
    ++stats.kept;
  }
  return stats;
}

}  // namespace openral_octomap_bridge
