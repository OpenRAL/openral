// SPDX-License-Identifier: Apache-2.0
// Robot (and held-payload) self-filter for a real depth cloud, ahead of OctoMap.

#include "openral_octomap_bridge/self_filter.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iterator>
#include <limits>
#include <utility>

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

// ---------------------------------------------------------------------------
// Point-vs-hull GJK. The simplex routines are the kernel's (collision.cpp,
// `origin_closest_on_segment` .. `simplex_closest`) with the query point at
// the origin of the difference set {v - p}; only the vertex bookkeeping the
// kernel keeps for its per-cell witness cache is gone.
// ---------------------------------------------------------------------------

struct Simplex {
  tf2::Vector3 p[4];
  int n{0};
};

tf2::Vector3 origin_closest_on_segment(const tf2::Vector3& a, const tf2::Vector3& b,
                                       double& u) noexcept {
  const tf2::Vector3 ab = b - a;
  const double denom = ab.dot(ab);
  if (!(denom > 0.0)) {
    u = 0.0;
    return a;
  }
  const double t = std::clamp(-a.dot(ab) / denom, 0.0, 1.0);
  u = t;
  return a + ab * t;
}

// Closest point to the origin on triangle ABC (Voronoi-region decomposition).
// `mask` receives the supporting vertices' bits (1=A, 2=B, 4=C).
tf2::Vector3 origin_closest_on_triangle(const tf2::Vector3& a, const tf2::Vector3& b,
                                        const tf2::Vector3& c, int& mask) noexcept {
  const tf2::Vector3 ab = b - a;
  const tf2::Vector3 ac = c - a;
  const double d1 = -ab.dot(a);
  const double d2 = -ac.dot(a);
  if (d1 <= 0.0 && d2 <= 0.0) {
    mask = 1;
    return a;
  }
  const double d3 = -ab.dot(b);
  const double d4 = -ac.dot(b);
  if (d3 >= 0.0 && d4 <= d3) {
    mask = 2;
    return b;
  }
  const double vc = d1 * d4 - d3 * d2;
  if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) {
    mask = 3;
    return a + ab * (d1 / (d1 - d3));
  }
  const double d5 = -ab.dot(c);
  const double d6 = -ac.dot(c);
  if (d6 >= 0.0 && d5 <= d6) {
    mask = 4;
    return c;
  }
  const double vb = d5 * d2 - d1 * d6;
  if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) {
    mask = 5;
    return a + ac * (d2 / (d2 - d6));
  }
  const double va = d3 * d6 - d5 * d4;
  if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0) {
    mask = 6;
    return b + (c - b) * ((d4 - d3) / ((d4 - d3) + (d5 - d6)));
  }
  const double denom = va + vb + vc;
  if (!(denom > 0.0)) {
    double u = 0.0;  // degenerate (collinear) triangle
    mask = 3;
    return origin_closest_on_segment(a, b, u);
  }
  mask = 7;
  return a + ab * (vb / denom) + ac * (vc / denom);
}

bool origin_outside_face(const tf2::Vector3& a, const tf2::Vector3& b, const tf2::Vector3& c,
                         const tf2::Vector3& d) noexcept {
  const tf2::Vector3 n = (b - a).cross(c - a);
  return (-a.dot(n)) * (d - a).dot(n) < 0.0;
}

void simplex_keep(Simplex& s, const int* which, int k) noexcept {
  tf2::Vector3 p[4];
  for (int i = 0; i < k; ++i) {
    p[i] = s.p[which[i]];
  }
  for (int i = 0; i < k; ++i) {
    s.p[i] = p[i];
  }
  s.n = k;
}

tf2::Vector3 simplex_closest(Simplex& s, bool& contains_origin) noexcept {
  contains_origin = false;
  if (s.n <= 1) {
    return s.p[0];
  }
  if (s.n == 2) {
    double u = 0.0;
    const tf2::Vector3 q = origin_closest_on_segment(s.p[0], s.p[1], u);
    if (u <= 0.0) {
      const int w[1] = {0};
      simplex_keep(s, w, 1);
    } else if (u >= 1.0) {
      const int w[1] = {1};
      simplex_keep(s, w, 1);
    }
    return q;
  }
  if (s.n == 3) {
    int mask = 0;
    const tf2::Vector3 q = origin_closest_on_triangle(s.p[0], s.p[1], s.p[2], mask);
    int w[3];
    int k = 0;
    for (int i = 0; i < 3; ++i) {
      if ((mask & (1 << i)) != 0) {
        w[k++] = i;
      }
    }
    simplex_keep(s, w, k);
    return q;
  }
  static constexpr int kFace[4][3] = {{0, 1, 2}, {0, 2, 3}, {0, 3, 1}, {1, 3, 2}};
  static constexpr int kOpposite[4] = {3, 1, 2, 0};
  double best2 = std::numeric_limits<double>::infinity();
  tf2::Vector3 best(0.0, 0.0, 0.0);
  int keep_idx[3] = {0, 0, 0};
  int keep_n = 0;
  bool any_outside = false;
  for (int f = 0; f < 4; ++f) {
    const tf2::Vector3& fa = s.p[kFace[f][0]];
    const tf2::Vector3& fb = s.p[kFace[f][1]];
    const tf2::Vector3& fc = s.p[kFace[f][2]];
    if (!origin_outside_face(fa, fb, fc, s.p[kOpposite[f]])) {
      continue;
    }
    any_outside = true;
    int mask = 0;
    const tf2::Vector3 q = origin_closest_on_triangle(fa, fb, fc, mask);
    const double q2 = q.dot(q);
    if (q2 < best2) {
      best2 = q2;
      best = q;
      keep_n = 0;
      for (int i = 0; i < 3; ++i) {
        if ((mask & (1 << i)) != 0) {
          keep_idx[keep_n++] = kFace[f][i];
        }
      }
    }
  }
  if (!any_outside) {
    contains_origin = true;
    return tf2::Vector3(0.0, 0.0, 0.0);
  }
  simplex_keep(s, keep_idx, keep_n);
  return best;
}

// The hull's TRUE support along `dir`: an exhaustive scan (see
// `self_hull_distance` for why it may not be a hill climb).
const tf2::Vector3& hull_support(const SelfHull& hull, const tf2::Vector3& dir) noexcept {
  std::size_t best = 0;
  double best_dot = hull.vertices[0].dot(dir);
  for (std::size_t i = 1; i < hull.vertices.size(); ++i) {
    const double d = hull.vertices[i].dot(dir);
    if (d > best_dot) {
      best_dot = d;
      best = i;
    }
  }
  return hull.vertices[best];
}

// The kernel's `validate_tight_hull` against box half extents `he`, returning
// a reason or nullptr.
const char* tight_hull_error(const SelfHull& hull, const tf2::Vector3& he) {
  for (int i = 0; i < kSelfDopAxes; ++i) {
    if (!std::isfinite(hull.dop_lo[i]) || !std::isfinite(hull.dop_hi[i]) ||
        hull.dop_lo[i] > hull.dop_hi[i]) {
      return "collision_hull_dop_* carries a non-finite or inverted slab";
    }
  }
  // The first three DOP axes are the box's axes and a 26-DOP lies inside its
  // own first three slabs, so this proves DOP ⊆ box for the whole polytope.
  for (int k = 0; k < 3; ++k) {
    if (hull.dop_lo[k] < -he[k] || hull.dop_hi[k] > he[k]) {
      return "a collision hull's 26-DOP reaches outside the box it refines";
    }
    if (!(hull.dop_hi[k] > hull.dop_lo[k])) {
      return "a collision hull's 26-DOP has no extent on one of its box's axes";
    }
  }
  for (const auto& v : hull.vertices) {
    if (!std::isfinite(v.x()) || !std::isfinite(v.y()) || !std::isfinite(v.z())) {
      return "collision_hull_vertices carries a non-finite vertex";
    }
    for (int i = 0; i < kSelfDopAxes; ++i) {
      const double s =
          kSelfDopAxis[i][0] * v.x() + kSelfDopAxis[i][1] * v.y() + kSelfDopAxis[i][2] * v.z();
      if (s > hull.dop_hi[i] + kSelfHullContainmentEpsilonM ||
          s < hull.dop_lo[i] - kSelfHullContainmentEpsilonM) {
        return "a collision hull vertex lies outside its own 26-DOP";
      }
    }
  }
  return nullptr;
}

// Parse and check the CSR tight-geometry arrays; fills `hulls` and one entry of
// `box_hull_index` per box. Returns a reason or nullptr.
const char* load_hulls(const SelfModelParams& p, std::size_t n_boxes, std::vector<SelfHull>& hulls,
                       std::vector<int>& box_hull_index) {
  box_hull_index.assign(n_boxes, -1);
  const std::size_t n_hulls = p.hull_vertex_count.size();
  const auto n_axes = static_cast<std::size_t>(kSelfDopAxes);
  if (p.box_hull.empty()) {
    if (n_hulls != 0 || !p.hull_vertex_first.empty() || !p.hull_dop_lo.empty() ||
        !p.hull_dop_hi.empty() || !p.hull_vertices.empty()) {
      return "collision_hull_* supplied without collision_box_hull";
    }
    return nullptr;
  }
  if (p.box_hull.size() != n_boxes || p.hull_vertex_first.size() != n_hulls ||
      p.hull_dop_lo.size() != n_axes * n_hulls || p.hull_dop_hi.size() != n_axes * n_hulls ||
      p.hull_vertices.size() % 3 != 0) {
    return "collision_hull_* array shapes disagree with the box / hull count";
  }
  const std::size_t n_vertices = p.hull_vertices.size() / 3;
  hulls.resize(n_hulls);
  for (std::size_t h = 0; h < n_hulls; ++h) {
    const std::int64_t first = p.hull_vertex_first[h];
    const std::int64_t count = p.hull_vertex_count[h];
    if (count < 0 || count > kSelfMaxHullVertices) {
      return "a collision hull has a negative vertex count or more than the kernel's 320";
    }
    if (first < 0 ||
        static_cast<std::size_t>(first) + static_cast<std::size_t>(count) > n_vertices) {
      return "collision_hull_vertex_first / _count run past collision_hull_vertices";
    }
    SelfHull& hull = hulls[h];
    for (std::size_t i = 0; i < n_axes; ++i) {
      hull.dop_lo[i] = p.hull_dop_lo[h * n_axes + i];
      hull.dop_hi[i] = p.hull_dop_hi[h * n_axes + i];
    }
    hull.vertices.reserve(static_cast<std::size_t>(count));
    for (std::int64_t v = first; v < first + count; ++v) {
      const auto k = static_cast<std::size_t>(v);
      hull.vertices.emplace_back(p.hull_vertices[3 * k], p.hull_vertices[3 * k + 1],
                                 p.hull_vertices[3 * k + 2]);
    }
  }
  for (std::size_t b = 0; b < n_boxes; ++b) {
    const std::int64_t h = p.box_hull[b];
    if (h < -1 || h >= static_cast<std::int64_t>(n_hulls)) {
      return "collision_box_hull names a hull that does not exist";
    }
    if (h < 0) {
      continue;
    }
    const tf2::Vector3 he(p.box_half_extents[3 * b + 0], p.box_half_extents[3 * b + 1],
                          p.box_half_extents[3 * b + 2]);
    if (const char* why = tight_hull_error(hulls[static_cast<std::size_t>(h)], he)) {
      return why;
    }
    box_hull_index[b] = static_cast<int>(h);
  }
  return nullptr;
}

}  // namespace

double self_hull_distance(const SelfHull& hull, const tf2::Vector3& local, double stop_above,
                          double stop_at_or_below) noexcept {
  // Stage 1: the largest slab violation, and the slab's outward normal as
  // stage 2's seed direction.
  double best = -std::numeric_limits<double>::infinity();
  tf2::Vector3 axis(1.0, 0.0, 0.0);
  for (int i = 0; i < kSelfDopAxes; ++i) {
    const tf2::Vector3 u(kSelfDopAxis[i][0], kSelfDopAxis[i][1], kSelfDopAxis[i][2]);
    const double s = u.dot(local);
    const double above = s - hull.dop_hi[i];
    const double below = hull.dop_lo[i] - s;
    const double gap = above > below ? above : below;
    if (gap > best) {
      best = gap;
      axis = above > below ? u : -u;
    }
  }
  if (hull.vertices.empty() || best > stop_above) {
    return best;
  }
  // Stage 2: GJK on the difference set {v - local}; the origin's distance to
  // it is the point's distance to the hull.
  Simplex s;
  s.p[0] = hull_support(hull, axis) - local;
  s.n = 1;
  double lower = -std::numeric_limits<double>::infinity();
  for (int it = 0; it < kSelfGjkMaxIterations; ++it) {
    bool contains = false;
    const tf2::Vector3 v = simplex_closest(s, contains);
    const double vlen2 = v.dot(v);
    if (contains || vlen2 <= 1e-24) {
      return best;  // inside the hull: stage 1's bound (<= 0 here) stands
    }
    const double vlen = std::sqrt(vlen2);
    if (vlen <= stop_at_or_below) {
      // A hull point lies within `vlen` of the query, so the true distance is
      // at most this and the bound returned below is too: settled as "near".
      break;
    }
    const tf2::Vector3 w = hull_support(hull, -v / vlen) - local;
    // Supporting-hyperplane bound: no point of the difference set is closer to
    // the origin than this.
    lower = std::max(lower, v.dot(w) / vlen);
    if (lower > stop_above) {
      break;
    }
    if (vlen - lower <= kSelfGjkTolerance) {
      lower = vlen;  // converged on the exact distance
      break;
    }
    if (s.n == 4) {
      break;
    }
    s.p[s.n++] = w;
  }
  return std::max(lower, best);
}

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
  std::vector<SelfHull> hulls;
  std::vector<int> box_hull_index;
  if (const char* why = load_hulls(p, n_boxes, hulls, box_hull_index)) {
    error = why;
    return false;
  }
  if (!all_finite(p.origin_xyzrpy) || !all_finite(p.axis) || !all_finite(p.capsule_origin_xyzrpy) ||
      !all_finite(p.box_origin_xyzrpy)) {
    error = "collision_* carries a non-finite origin or axis";
    return false;
  }
  if (p.mimic_multiplier.size() != p.mimic_offset.size() ||
      (!p.mimic_multiplier.empty() && p.mimic_multiplier.size() != n) ||
      !all_finite(p.mimic_multiplier) || !all_finite(p.mimic_offset)) {
    error = "collision_mimic_* arrays disagree with collision_n_links or are non-finite";
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
    if (p.joint_kind[i] < 0 ||
        p.joint_kind[i] > static_cast<std::int64_t>(SelfJointKind::kMimicPrismatic)) {
      error = "collision_joint_kind carries an unknown joint code";
      return false;
    }
    m.parent[i] = static_cast<int>(parent);
    m.joint_kind[i] = static_cast<SelfJointKind>(static_cast<std::uint8_t>(p.joint_kind[i]));
    m.dof_index[i] = static_cast<int>(p.dof_index[i]);
    const bool mimic = m.joint_kind[i] == SelfJointKind::kMimicRevolute ||
                       m.joint_kind[i] == SelfJointKind::kMimicPrismatic;
    if (mimic &&
        (m.dof_index[i] < 0 || p.mimic_multiplier.empty() || p.mimic_multiplier[i] == 0.0)) {
      error = "a mimic link needs a dof column and a non-zero collision_mimic_multiplier";
      return false;
    }
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
    m.primitive_hull.push_back(-1);
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
    m.primitive_hull.push_back(box_hull_index[b]);
  }
  m.hulls = std::move(hulls);
  m.mimic_multiplier = p.mimic_multiplier;
  m.mimic_offset = p.mimic_offset;
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
      const SelfJointKind kind = model.joint_kind[i];
      double angle = q[static_cast<std::size_t>(dof)];
      const bool mimic =
          kind == SelfJointKind::kMimicRevolute || kind == SelfJointKind::kMimicPrismatic;
      if (mimic && i < model.mimic_multiplier.size() && i < model.mimic_offset.size()) {
        angle = model.mimic_multiplier[i] * angle + model.mimic_offset[i];  // the kernel's FK
      }
      if (kind == SelfJointKind::kRevolute || kind == SelfJointKind::kMimicRevolute) {
        motion.setRotation(tf2::Quaternion(model.axis[i], angle));
      } else if (kind == SelfJointKind::kPrismatic || kind == SelfJointKind::kMimicPrismatic) {
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
                           std::vector<PayloadPrimitive>& out,
                           std::vector<const SelfHull*>* out_hulls) {
  for (std::size_t k = 0; k < model.primitives.size(); ++k) {
    PayloadPrimitive placed = model.primitives[k];
    const auto link = static_cast<std::size_t>(model.primitive_link[k]);
    placed.pose = frame_from_root * link_world[link] * model.primitives[k].pose;
    out.push_back(placed);
    if (out_hulls != nullptr) {
      const int h = model.primitive_hull[k];
      out_hulls->push_back(h < 0 ? nullptr : &model.hulls[static_cast<std::size_t>(h)]);
    }
  }
}

void SelfMask::set(const std::vector<PayloadPrimitive>& primitives, double padding_m,
                   const std::vector<const SelfHull*>& hulls,
                   const std::vector<double>& extra_padding_m) {
  const double base = std::isfinite(padding_m) && padding_m > 0.0 ? padding_m : 0.0;
  entries_.clear();
  entries_.reserve(primitives.size());
  for (std::size_t i = 0; i < primitives.size(); ++i) {
    const PayloadPrimitive& prim = primitives[i];
    Entry e;
    e.primitive = prim;
    // A hull refines a box only; anything else keeps its own shape.
    e.hull = i < hulls.size() && prim.shape_type == Wire::SHAPE_BOX ? hulls[i] : nullptr;
    e.local_from_frame = prim.pose.inverse();
    e.center = prim.pose.getOrigin();
    const double extra =
        i < extra_padding_m.size() && std::isfinite(extra_padding_m[i]) && extra_padding_m[i] > 0.0
            ? extra_padding_m[i]
            : 0.0;
    e.padding = std::min(base + extra, std::max(base, kMaxSelfFilterPaddingM));
    const double reach = primitive_bounding_radius(prim) + e.padding;
    e.reach_sq = reach * reach;
    entries_.push_back(e);
  }
}

bool SelfMask::contains(const tf2::Vector3& p) const noexcept {
  for (const auto& e : entries_) {
    if ((p - e.center).length2() > e.reach_sq) {
      continue;
    }
    const tf2::Vector3 local = e.local_from_frame * p;
    if (primitive_surface_distance(e.primitive, local) > e.padding) {
      continue;  // outside the box's shell, so outside the hull's (hull ⊆ box)
    }
    if (e.hull == nullptr ||
        self_hull_distance(*e.hull, local, e.padding, e.padding) <= e.padding) {
      return true;
    }
  }
  return false;
}

JointStateVerdict joint_state_at(const std::deque<JointSample>& history, std::int64_t stamp_ns,
                                 double period_s, double max_skew_s, std::vector<double>& q,
                                 std::vector<double>& qdot) {
  if (history.empty()) {
    return JointStateVerdict::kNoState;
  }
  const auto period_ns = static_cast<std::int64_t>(period_s * 1e9);
  const auto skew_ns = static_cast<std::int64_t>(max_skew_s * 1e9);
  const auto hi =
      std::lower_bound(history.begin(), history.end(), stamp_ns,
                       [](const JointSample& s, std::int64_t t) { return s.stamp_ns < t; });
  const JointSample* a = nullptr;
  const JointSample* b = nullptr;
  std::int64_t nearest_ns = 0;
  if (hi != history.end() && hi->stamp_ns == stamp_ns) {
    // Exactly a sample: its pose; its rate from a neighbour if one is close enough.
    q = hi->q;
    const JointSample* n = hi != history.begin()            ? &*std::prev(hi)
                           : std::next(hi) != history.end() ? &*std::next(hi)
                                                            : nullptr;
    qdot.assign(q.size(), 0.0);
    if (n != nullptr && std::llabs(n->stamp_ns - stamp_ns) <= 2 * period_ns) {
      const double dt = 1e-9 * static_cast<double>(stamp_ns - n->stamp_ns);
      for (std::size_t d = 0; d < q.size() && d < n->q.size(); ++d) {
        qdot[d] = (q[d] - n->q[d]) / dt;
      }
    }
    return JointStateVerdict::kOk;
  }
  if (hi != history.end() && hi != history.begin()) {
    a = &*std::prev(hi);
    b = &*hi;
    nearest_ns = std::min(stamp_ns - a->stamp_ns, b->stamp_ns - stamp_ns);
  } else {
    if (history.size() < 2) {
      return JointStateVerdict::kExtrapolation;
    }
    const bool after = hi == history.end();
    a = after ? &history[history.size() - 2] : &history[0];
    b = after ? &history[history.size() - 1] : &history[1];
    nearest_ns = after ? stamp_ns - b->stamp_ns : a->stamp_ns - stamp_ns;
    if (nearest_ns > period_ns) {
      return JointStateVerdict::kExtrapolation;
    }
  }
  if (b->stamp_ns - a->stamp_ns > 2 * period_ns || b->stamp_ns <= a->stamp_ns) {
    return JointStateVerdict::kGap;
  }
  if (nearest_ns > skew_ns) {
    return JointStateVerdict::kNoState;
  }
  const double dt = 1e-9 * static_cast<double>(b->stamp_ns - a->stamp_ns);
  const double alpha = 1e-9 * static_cast<double>(stamp_ns - a->stamp_ns) / dt;
  const std::size_t n = std::min(a->q.size(), b->q.size());
  q.resize(n);
  qdot.resize(n);
  for (std::size_t d = 0; d < n; ++d) {
    qdot[d] = (b->q[d] - a->q[d]) / dt;
    q[d] = a->q[d] + alpha * (b->q[d] - a->q[d]);
  }
  return JointStateVerdict::kOk;
}

void self_primitive_motion(const SelfModel& model, const std::vector<tf2::Transform>& link_a,
                           const std::vector<tf2::Transform>& link_b, std::vector<double>& out) {
  out.resize(model.primitives.size());
  for (std::size_t k = 0; k < model.primitives.size(); ++k) {
    const auto link = static_cast<std::size_t>(model.primitive_link[k]);
    const tf2::Transform pa = link_a[link] * model.primitives[k].pose;
    const tf2::Transform pb = link_b[link] * model.primitives[k].pose;
    const double angle = pa.getRotation().angleShortestPath(pb.getRotation());
    out[k] = (pb.getOrigin() - pa.getOrigin()).length() +
             primitive_bounding_radius(model.primitives[k]) * angle;
  }
}

namespace {

constexpr double kInfinityM = std::numeric_limits<double>::infinity();

// Entry parameter `t >= 0` of the ray `o + t*d` into the convex polytope
// {x : lo[i] <= n_i·x <= hi[i]}; 0 when `o` is inside, +inf on a miss.
double slab_entry(const tf2::Vector3& o, const tf2::Vector3& d, const double (*axes)[3],
                  const double* lo, const double* hi, int n_axes) noexcept {
  double t_near = 0.0;
  double t_far = kInfinityM;
  for (int i = 0; i < n_axes; ++i) {
    const tf2::Vector3 n(axes[i][0], axes[i][1], axes[i][2]);
    const double s = n.dot(o);
    const double ds = n.dot(d);
    if (std::fabs(ds) < 1e-12) {
      if (s < lo[i] || s > hi[i]) {
        return kInfinityM;
      }
      continue;
    }
    double t1 = (lo[i] - s) / ds;
    double t2 = (hi[i] - s) / ds;
    if (t1 > t2) {
      std::swap(t1, t2);
    }
    t_near = std::max(t_near, t1);
    t_far = std::min(t_far, t2);
    if (t_near > t_far) {
      return kInfinityM;
    }
  }
  return t_near;
}

// Entry of `o + t*d` (t >= 0) into the ball of radius `r` about `c`.
double sphere_entry(const tf2::Vector3& o, const tf2::Vector3& d, const tf2::Vector3& c,
                    double r) noexcept {
  const tf2::Vector3 oc = o - c;
  const double a = d.dot(d);
  const double b = oc.dot(d);
  const double disc = b * b - a * (oc.dot(oc) - r * r);
  if (disc < 0.0) {
    return kInfinityM;
  }
  const double sq = std::sqrt(disc);
  if ((-b + sq) / a < 0.0) {
    return kInfinityM;
  }
  return std::max(0.0, (-b - sq) / a);
}

// Entry into the capsule `{x : dist(x, z-segment [-h, h]) <= r}`, local frame:
// the union of the two end balls and the finite cylinder between them.
double capsule_entry(const tf2::Vector3& o, const tf2::Vector3& d, double r, double h) noexcept {
  double t = std::min(sphere_entry(o, d, tf2::Vector3(0, 0, -h), r),
                      sphere_entry(o, d, tf2::Vector3(0, 0, h), r));
  const double a = d.x() * d.x() + d.y() * d.y();
  if (h > 0.0 && a > 1e-18) {
    const double b = o.x() * d.x() + o.y() * d.y();
    const double disc = b * b - a * (o.x() * o.x() + o.y() * o.y() - r * r);
    if (disc >= 0.0) {
      const double sq = std::sqrt(disc);
      const double t_in = (-b - sq) / a;
      const double t_out = (-b + sq) / a;
      // The z-interval of the slab |z| <= h along the ray, intersected with the
      // radial interval [t_in, t_out].
      double z_lo = 0.0;
      double z_hi = kInfinityM;
      if (std::fabs(d.z()) < 1e-12) {
        if (std::fabs(o.z()) > h) {
          z_lo = kInfinityM;
        }
      } else {
        z_lo = (-h - o.z()) / d.z();
        z_hi = (h - o.z()) / d.z();
        if (z_lo > z_hi) {
          std::swap(z_lo, z_hi);
        }
      }
      const double lo = std::max({t_in, z_lo, 0.0});
      const double hi = std::min(t_out, z_hi);
      if (lo <= hi) {
        t = std::min(t, lo);
      }
    }
  }
  return t;
}

constexpr double kBoxAxes[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};

// Entry of the camera ray `t*d` (camera frame) into one primitive.
double primitive_entry(const PayloadPrimitive& prim, const SelfHull* hull,
                       const tf2::Transform& local_from_cam, const tf2::Vector3& d) noexcept {
  const tf2::Vector3 o = local_from_cam.getOrigin();
  const tf2::Vector3 dl = local_from_cam.getBasis() * d;
  switch (prim.shape_type) {
  case Wire::SHAPE_SPHERE:
    return sphere_entry(o, dl, tf2::Vector3(0, 0, 0), prim.radius);
  case Wire::SHAPE_CAPSULE:
    return capsule_entry(o, dl, prim.radius, prim.half_length);
  default: {
    if (hull != nullptr) {
      // ponytail: the 26-DOP, not the exact hull (we hold no hull faces); the
      // DOP contains the hull, so the silhouette can only grow.
      return slab_entry(o, dl, kSelfDopAxis, hull->dop_lo, hull->dop_hi, kSelfDopAxes);
    }
    const double lo[3] = {-prim.half_extents.x(), -prim.half_extents.y(), -prim.half_extents.z()};
    const double hi[3] = {prim.half_extents.x(), prim.half_extents.y(), prim.half_extents.z()};
    return slab_entry(o, dl, kBoxAxes, lo, hi, 3);
  }
  }
}

// `dst[p] = 1` iff some `src` pixel within Chebyshev distance `k` of `p` is set.
void dilate(const std::vector<std::uint8_t>& src, std::vector<std::uint8_t>& dst,
            std::vector<std::uint8_t>& tmp, std::size_t w, std::size_t h, int k) {
  const auto kk = static_cast<std::size_t>(std::max(k, 0));
  tmp.assign(w * h, 0);
  dst.assign(w * h, 0);
  for (std::size_t y = 0; y < h; ++y) {
    for (std::size_t x = 0; x < w; ++x) {
      if (src[y * w + x] == 0) {
        continue;
      }
      const std::size_t x0 = x >= kk ? x - kk : 0;
      const std::size_t x1 = std::min(w - 1, x + kk);
      std::fill(tmp.begin() + static_cast<std::ptrdiff_t>(y * w + x0),
                tmp.begin() + static_cast<std::ptrdiff_t>(y * w + x1 + 1), 1);
    }
  }
  for (std::size_t y = 0; y < h; ++y) {
    const std::size_t y0 = y >= kk ? y - kk : 0;
    const std::size_t y1 = std::min(h - 1, y + kk);
    for (std::size_t x = 0; x < w; ++x) {
      if (tmp[y * w + x] == 0) {
        continue;
      }
      for (std::size_t yy = y0; yy <= y1; ++yy) {
        dst[yy * w + x] = 1;
      }
    }
  }
}

}  // namespace

void render_robot_mask(const std::vector<PayloadPrimitive>& primitives,
                       const std::vector<const SelfHull*>& hulls,
                       const std::vector<double>& backstop_margin_m, const float* depth_m,
                       const PinholeCamera& cam, int uncertain_px, RobotMaskScratch& s,
                       std::vector<std::uint8_t>& mask) {
  const std::size_t w = cam.width;
  const std::size_t h = cam.height;
  const std::size_t n = w * h;
  s.z_robot.assign(n, std::numeric_limits<float>::infinity());
  s.backstop.assign(n, 0);
  mask.assign(n, kRobotMaskFree);
  if (n == 0 || !(cam.fx > 0.0) || !(cam.fy > 0.0)) {
    return;
  }
  constexpr double kNearM = 1e-3;
  for (std::size_t k = 0; k < primitives.size(); ++k) {
    const PayloadPrimitive& prim = primitives[k];
    const SelfHull* hull =
        k < hulls.size() && prim.shape_type == Wire::SHAPE_BOX ? hulls[k] : nullptr;
    const double margin = k < backstop_margin_m.size() ? backstop_margin_m[k] : 0.0;
    const tf2::Vector3 c = prim.pose.getOrigin();
    const double r = primitive_bounding_radius(prim) + margin;
    if (c.z() + r <= kNearM) {
      continue;  // wholly behind the camera
    }
    // The footprint: the projection of the bounding sphere's cube, or the whole
    // image when that cube reaches behind the image plane.
    std::size_t u0 = 0;
    std::size_t u1 = w - 1;
    std::size_t v0 = 0;
    std::size_t v1 = h - 1;
    if (c.z() - r > kNearM) {
      double umin = kInfinityM;
      double umax = -kInfinityM;
      double vmin = kInfinityM;
      double vmax = -kInfinityM;
      for (int corner = 0; corner < 8; ++corner) {
        const double x = c.x() + ((corner & 1) != 0 ? r : -r);
        const double y = c.y() + ((corner & 2) != 0 ? r : -r);
        const double z = c.z() + ((corner & 4) != 0 ? r : -r);
        umin = std::min(umin, cam.fx * x / z + cam.cx);
        umax = std::max(umax, cam.fx * x / z + cam.cx);
        vmin = std::min(vmin, cam.fy * y / z + cam.cy);
        vmax = std::max(vmax, cam.fy * y / z + cam.cy);
      }
      if (umax < 0.0 || vmax < 0.0 || umin > static_cast<double>(w - 1) ||
          vmin > static_cast<double>(h - 1)) {
        continue;
      }
      u0 = static_cast<std::size_t>(std::max(0.0, std::floor(umin)));
      v0 = static_cast<std::size_t>(std::max(0.0, std::floor(vmin)));
      u1 = static_cast<std::size_t>(std::min(static_cast<double>(w - 1), std::ceil(umax)));
      v1 = static_cast<std::size_t>(std::min(static_cast<double>(h - 1), std::ceil(vmax)));
    }
    const tf2::Transform local_from_cam = prim.pose.inverse();
    for (std::size_t v = v0; v <= v1; ++v) {
      for (std::size_t u = u0; u <= u1; ++u) {
        const std::size_t p = v * w + u;
        const tf2::Vector3 d((static_cast<double>(u) - cam.cx) / cam.fx,
                             (static_cast<double>(v) - cam.cy) / cam.fy, 1.0);
        const double t = primitive_entry(prim, hull, local_from_cam, d);
        if (t < static_cast<double>(s.z_robot[p])) {
          s.z_robot[p] = static_cast<float>(t);  // d.z = 1: t IS the depth
        }
        const float dm = depth_m[p];
        if (s.backstop[p] == 0 && std::isfinite(dm) && dm > 0.0F) {
          const tf2::Vector3 local = local_from_cam * (d * static_cast<double>(dm));
          if (primitive_surface_distance(prim, local) <= margin &&
              (hull == nullptr || self_hull_distance(*hull, local, margin, margin) <= margin)) {
            s.backstop[p] = 1;
          }
        }
      }
    }
  }
  s.robot.assign(n, 0);
  s.not_robot.assign(n, 1);
  for (std::size_t p = 0; p < n; ++p) {
    const float dm = depth_m[p];
    if (std::isfinite(dm) && dm > 0.0F &&
        static_cast<double>(s.z_robot[p]) <= static_cast<double>(dm) + kRobotMaskDepthTolM) {
      s.robot[p] = 1;
      s.not_robot[p] = 0;
    }
  }
  dilate(s.robot, s.near_robot, s.tmp, w, h, uncertain_px);
  dilate(s.not_robot, s.near_free, s.tmp, w, h, uncertain_px);
  for (std::size_t p = 0; p < n; ++p) {
    const float dm = depth_m[p];
    const bool no_return = !(std::isfinite(dm) && dm > 0.0F);
    if (s.backstop[p] != 0) {
      mask[p] = kRobotMaskRobot;
    } else if (uncertain_px > 0 && s.near_robot[p] != 0 && s.near_free[p] != 0) {
      mask[p] = kRobotMaskUncertain;
    } else if (s.robot[p] != 0) {
      mask[p] = kRobotMaskRobot;
    } else if (no_return && std::isfinite(s.z_robot[p])) {
      mask[p] = kRobotMaskUncertain;
    }
  }
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

std::vector<std::uint8_t> xyz_records(const std::vector<std::uint8_t>& data, std::size_t point_step,
                                      std::size_t x_offset, std::size_t y_offset,
                                      std::size_t z_offset) {
  const std::size_t n = point_step == 0 ? 0 : data.size() / point_step;
  std::vector<std::uint8_t> out(n * 12);
  for (std::size_t i = 0; i < n; ++i) {
    const std::uint8_t* rec = data.data() + i * point_step;
    std::uint8_t* xyz = out.data() + i * 12;
    std::memcpy(xyz, rec + x_offset, sizeof(float));
    std::memcpy(xyz + 4, rec + y_offset, sizeof(float));
    std::memcpy(xyz + 8, rec + z_offset, sizeof(float));
  }
  return out;
}

}  // namespace openral_octomap_bridge
