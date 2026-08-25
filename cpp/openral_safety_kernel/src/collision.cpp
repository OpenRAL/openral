// SPDX-License-Identifier: Apache-2.0
// Allocation-free geometric self-collision core.
//
// Hand-rolled, dependency-free (no Eigen/KDL/Pinocchio) so the safety kernel
// stays small and auditable. All hot-path work is on the stack or in
// caller-owned pre-sized buffers; nothing here allocates (pinned by
// test_collision.cpp's counting-allocator test).

#include "openral_safety_kernel/collision.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <string>
#include <utility>

namespace openral_safety_kernel {

namespace {

// ── small rigid-transform algebra (row-major 3x3 + translation) ──────────────

Vec3 rotate(const double r[9], const Vec3& v) noexcept {
  return Vec3{
      r[0] * v.x + r[1] * v.y + r[2] * v.z,
      r[3] * v.x + r[4] * v.y + r[5] * v.z,
      r[6] * v.x + r[7] * v.y + r[8] * v.z,
  };
}

Vec3 add(const Vec3& a, const Vec3& b) noexcept { return Vec3{a.x + b.x, a.y + b.y, a.z + b.z}; }
Vec3 sub(const Vec3& a, const Vec3& b) noexcept { return Vec3{a.x - b.x, a.y - b.y, a.z - b.z}; }
Vec3 scale(const Vec3& a, double s) noexcept { return Vec3{a.x * s, a.y * s, a.z * s}; }
double dot(const Vec3& a, const Vec3& b) noexcept { return a.x * b.x + a.y * b.y + a.z * b.z; }
Vec3 cross(const Vec3& a, const Vec3& b) noexcept {
  return Vec3{a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

// Solve the 6x6 system A·x = b in place via Gaussian elimination with partial
// pivoting. `A` is row-major 6x6 (clobbered). Returns false if singular (no
// usable pivot); allocation-free. Used only for the DLS normal equations, which
// are SPD when lambda>0, so a near-singular pivot here means damping was too low
// — the caller treats failure as "cannot predict" and falls back to reactive.
bool solve6(double A[36], double b[6], double x[6]) noexcept {
  int idx[6] = {0, 1, 2, 3, 4, 5};
  for (int col = 0; col < 6; ++col) {
    // Partial pivot: largest |A[row][col]| among remaining rows.
    int piv = col;
    double best = std::fabs(A[idx[col] * 6 + col]);
    for (int r = col + 1; r < 6; ++r) {
      const double v = std::fabs(A[idx[r] * 6 + col]);
      if (v > best) {
        best = v;
        piv = r;
      }
    }
    if (best < 1e-12) {
      return false;
    }
    std::swap(idx[col], idx[piv]);
    const int pr = idx[col];
    const double pivval = A[pr * 6 + col];
    for (int r = col + 1; r < 6; ++r) {
      const int rr = idx[r];
      const double factor = A[rr * 6 + col] / pivval;
      if (factor == 0.0) {
        continue;
      }
      for (int c = col; c < 6; ++c) {
        A[rr * 6 + c] -= factor * A[pr * 6 + c];
      }
      b[rr] -= factor * b[pr];
    }
  }
  // Back-substitution.
  for (int row = 5; row >= 0; --row) {
    const int rr = idx[row];
    double s = b[rr];
    for (int c = row + 1; c < 6; ++c) {
      s -= A[rr * 6 + c] * x[c];
    }
    x[row] = s / A[rr * 6 + row];
  }
  return true;
}

// out = lhs * rhs (compose: apply rhs first, then lhs).
Transform compose(const Transform& lhs, const Transform& rhs) noexcept {
  Transform out;
  for (int row = 0; row < 3; ++row) {
    for (int col = 0; col < 3; ++col) {
      double acc = 0.0;
      for (int k = 0; k < 3; ++k) {
        acc += lhs.r[row * 3 + k] * rhs.r[k * 3 + col];
      }
      out.r[row * 3 + col] = acc;
    }
  }
  out.t = add(lhs.t, rotate(lhs.r, rhs.t));
  return out;
}

// Rodrigues rotation about a unit axis by `angle`, written row-major into `r`.
void axis_angle(const Vec3& axis, double angle, double r[9]) noexcept {
  const double c = std::cos(angle);
  const double s = std::sin(angle);
  const double t = 1.0 - c;
  const double x = axis.x;
  const double y = axis.y;
  const double z = axis.z;
  r[0] = t * x * x + c;
  r[1] = t * x * y - s * z;
  r[2] = t * x * z + s * y;
  r[3] = t * x * y + s * z;
  r[4] = t * y * y + c;
  r[5] = t * y * z - s * x;
  r[6] = t * x * z - s * y;
  r[7] = t * y * z + s * x;
  r[8] = t * z * z + c;
}

double clamp01(double v) noexcept { return v < 0.0 ? 0.0 : (v > 1.0 ? 1.0 : v); }

// Squared distance between segments [p1,q1] and [p2,q2]
// (Ericson, *Real-Time Collision Detection*, §5.1.9).
double segment_segment_dist2(const Vec3& p1, const Vec3& q1, const Vec3& p2,
                             const Vec3& q2) noexcept {
  constexpr double kEps = 1e-12;
  const Vec3 d1 = sub(q1, p1);
  const Vec3 d2 = sub(q2, p2);
  const Vec3 r = sub(p1, p2);
  const double a = dot(d1, d1);
  const double e = dot(d2, d2);
  const double f = dot(d2, r);

  double s = 0.0;
  double t = 0.0;
  if (a <= kEps && e <= kEps) {
    // Both segments are points.
    const Vec3 diff = sub(p1, p2);
    return dot(diff, diff);
  }
  if (a <= kEps) {
    // First segment is a point.
    t = clamp01(f / e);
  } else {
    const double c = dot(d1, r);
    if (e <= kEps) {
      // Second segment is a point.
      s = clamp01(-c / a);
    } else {
      const double b = dot(d1, d2);
      const double denom = a * e - b * b;
      if (denom > kEps || denom < -kEps) {
        s = clamp01((b * f - c * e) / denom);
      } else {
        s = 0.0;  // parallel: pick an arbitrary point on segment 1
      }
      t = (b * s + f) / e;
      if (t < 0.0) {
        t = 0.0;
        s = clamp01(-c / a);
      } else if (t > 1.0) {
        t = 1.0;
        s = clamp01((b - c) / a);
      }
    }
  }
  const Vec3 c1 = add(p1, scale(d1, s));
  const Vec3 c2 = add(p2, scale(d2, t));
  const Vec3 diff = sub(c1, c2);
  return dot(diff, diff);
}

int clamp_index(int v, int lo, int hi) noexcept { return v < lo ? lo : (v > hi ? hi : v); }

// The two endpoints of a capsule's central segment, given its frame.
void capsule_endpoints(const Transform& frame, double half_length, Vec3& p0, Vec3& p1) noexcept {
  // Local +Z axis mapped to the common frame = third column of the rotation.
  const Vec3 z_axis{frame.r[2], frame.r[5], frame.r[8]};
  p0 = sub(frame.t, scale(z_axis, half_length));
  p1 = add(frame.t, scale(z_axis, half_length));
}

bool is_allowed(const CollisionModel& model, int a, int b) noexcept {
  const int lo = a < b ? a : b;
  const int hi = a < b ? b : a;
  for (const auto& pair : model.allowed_pairs) {
    const int plo = pair.first < pair.second ? pair.first : pair.second;
    const int phi = pair.first < pair.second ? pair.second : pair.first;
    if (plo == lo && phi == hi) {
      return true;
    }
  }
  return false;
}

}  // namespace

Transform transform_from_xyz_rpy(double x, double y, double z, double roll, double pitch,
                                 double yaw) noexcept {
  const double cr = std::cos(roll);
  const double sr = std::sin(roll);
  const double cp = std::cos(pitch);
  const double sp = std::sin(pitch);
  const double cy = std::cos(yaw);
  const double sy = std::sin(yaw);
  Transform out;
  // R = Rz(yaw) * Ry(pitch) * Rx(roll), row-major.
  out.r[0] = cy * cp;
  out.r[1] = cy * sp * sr - sy * cr;
  out.r[2] = cy * sp * cr + sy * sr;
  out.r[3] = sy * cp;
  out.r[4] = sy * sp * sr + cy * cr;
  out.r[5] = sy * sp * cr - cy * sr;
  out.r[6] = -sp;
  out.r[7] = cp * sr;
  out.r[8] = cp * cr;
  out.t = Vec3{x, y, z};
  return out;
}

Transform transform_from_translation_quat(double x, double y, double z, double qx, double qy,
                                          double qz, double qw) noexcept {
  Transform out;
  const double norm = std::sqrt(qx * qx + qy * qy + qz * qz + qw * qw);
  if (norm < 1e-12) {
    // Degenerate (zero) quaternion — keep the identity rotation.
    out.t = Vec3{x, y, z};
    return out;
  }
  const double s = 1.0 / norm;
  const double xn = qx * s;
  const double yn = qy * s;
  const double zn = qz * s;
  const double wn = qw * s;
  // Standard unit-quaternion → row-major rotation matrix.
  out.r[0] = 1.0 - 2.0 * (yn * yn + zn * zn);
  out.r[1] = 2.0 * (xn * yn - zn * wn);
  out.r[2] = 2.0 * (xn * zn + yn * wn);
  out.r[3] = 2.0 * (xn * yn + zn * wn);
  out.r[4] = 1.0 - 2.0 * (xn * xn + zn * zn);
  out.r[5] = 2.0 * (yn * zn - xn * wn);
  out.r[6] = 2.0 * (xn * zn - yn * wn);
  out.r[7] = 2.0 * (yn * zn + xn * wn);
  out.r[8] = 1.0 - 2.0 * (xn * xn + yn * yn);
  out.t = Vec3{x, y, z};
  return out;
}

double capsule_distance(const Transform& a, double a_radius, double a_half_length,
                        const Transform& b, double b_radius, double b_half_length) noexcept {
  Vec3 a0;
  Vec3 a1;
  Vec3 b0;
  Vec3 b1;
  capsule_endpoints(a, a_half_length, a0, a1);
  capsule_endpoints(b, b_half_length, b0, b1);
  const double centerline = std::sqrt(segment_segment_dist2(a0, a1, b0, b1));
  return centerline - a_radius - b_radius;
}

namespace {

// Distance from a point (already in the box's local frame) to the axis-aligned
// box [-h, +h]; 0 when the point is inside. Standard closed form.
double point_aabb_distance(const Vec3& p, const Vec3& h) noexcept {
  const double ox = std::fabs(p.x) - h.x;
  const double oy = std::fabs(p.y) - h.y;
  const double oz = std::fabs(p.z) - h.z;
  const double cx = ox > 0.0 ? ox : 0.0;
  const double cy = oy > 0.0 ? oy : 0.0;
  const double cz = oz > 0.0 ? oz : 0.0;
  return std::sqrt(cx * cx + cy * cy + cz * cz);
}

// World point -> box-local coordinates: Rᵀ (p - t), for a row-major R.
Vec3 to_box_local(const Transform& frame, const Vec3& p) noexcept {
  const Vec3 d = sub(p, frame.t);
  return Vec3{frame.r[0] * d.x + frame.r[3] * d.y + frame.r[6] * d.z,
              frame.r[1] * d.x + frame.r[4] * d.y + frame.r[7] * d.z,
              frame.r[2] * d.x + frame.r[5] * d.y + frame.r[8] * d.z};
}

// Local axis k (0,1,2) of a row-major transform, expressed in the common frame.
Vec3 box_axis(const Transform& frame, int k) noexcept {
  return Vec3{frame.r[k], frame.r[k + 3], frame.r[k + 6]};
}

}  // namespace

double box_capsule_distance(const Transform& box, const Vec3& half_extents, const Transform& cap,
                            double cap_radius, double cap_half_length) noexcept {
  // The capsule's central segment, brought into the box's local frame where the
  // box is the axis-aligned [-h, +h]. The point→AABB distance is convex along
  // the segment, so a ternary search on t ∈ [0, 1] converges to the global
  // minimum surface gap; subtracting the capsule radius gives the capsule↔box
  // distance (exact when disjoint). Allocation-free (fixed iteration count).
  Vec3 c0;
  Vec3 c1;
  capsule_endpoints(cap, cap_half_length, c0, c1);
  const Vec3 a = to_box_local(box, c0);
  const Vec3 b = to_box_local(box, c1);
  const Vec3 ab = sub(b, a);
  double lo = 0.0;
  double hi = 1.0;
  for (int it = 0; it < 48; ++it) {
    const double m1 = lo + (hi - lo) / 3.0;
    const double m2 = hi - (hi - lo) / 3.0;
    const double f1 =
        point_aabb_distance(Vec3{a.x + m1 * ab.x, a.y + m1 * ab.y, a.z + m1 * ab.z}, half_extents);
    const double f2 =
        point_aabb_distance(Vec3{a.x + m2 * ab.x, a.y + m2 * ab.y, a.z + m2 * ab.z}, half_extents);
    if (f1 < f2) {
      hi = m2;
    } else {
      lo = m1;
    }
  }
  const double tm = 0.5 * (lo + hi);
  const double seg_box =
      point_aabb_distance(Vec3{a.x + tm * ab.x, a.y + tm * ab.y, a.z + tm * ab.z}, half_extents);
  return seg_box - cap_radius;
}

double box_box_distance(const Transform& a, const Vec3& a_half, const Transform& b,
                        const Vec3& b_half) noexcept {
  // Separating-axis theorem for two OBBs: project both onto each candidate axis
  // and take the largest surface gap. For disjoint boxes that maximum equals a
  // separating plane's clearance, which lower-bounds the true Euclidean distance
  // — so the kernel never under-reports a collision (safety §3). Negative on all
  // axes ⇒ overlap. Allocation-free (15 axes, no heap).
  const Vec3 ax[3] = {box_axis(a, 0), box_axis(a, 1), box_axis(a, 2)};
  const Vec3 bx[3] = {box_axis(b, 0), box_axis(b, 1), box_axis(b, 2)};
  const double ahv[3] = {a_half.x, a_half.y, a_half.z};
  const double bhv[3] = {b_half.x, b_half.y, b_half.z};
  const Vec3 dc = sub(b.t, a.t);
  double best = -std::numeric_limits<double>::infinity();
  for (int idx = 0; idx < 15; ++idx) {
    Vec3 axis_l;
    if (idx < 3) {
      axis_l = ax[idx];
    } else if (idx < 6) {
      axis_l = bx[idx - 3];
    } else {
      axis_l = cross(ax[(idx - 6) / 3], bx[(idx - 6) % 3]);
    }
    const double len = std::sqrt(dot(axis_l, axis_l));
    if (len < 1e-9) {
      continue;  // parallel edges — degenerate axis, already covered by faces
    }
    const Vec3 n = scale(axis_l, 1.0 / len);
    double ra = 0.0;
    double rb = 0.0;
    for (int k = 0; k < 3; ++k) {
      ra += ahv[k] * std::fabs(dot(ax[k], n));
      rb += bhv[k] * std::fabs(dot(bx[k], n));
    }
    const double gap = std::fabs(dot(dc, n)) - ra - rb;
    if (gap > best) {
      best = gap;
    }
  }
  return best;
}

// ---------------------------------------------------------------------------
// Staged tight narrow phase: 26-DOP separating axis, then GJK on the exact
// convex hull. Both stages are subsets of the shipped OBB, so the broad-phase
// window in `check_voxel_collision` is untouched.
// ---------------------------------------------------------------------------

TightGeometryStatus validate_tight_geometry(const CollisionModel& model,
                                            std::size_t& offending_box) noexcept {
  offending_box = 0;
  if (model.box_hull.empty()) {
    return TightGeometryStatus::kOk;  // nothing declared: the shipped path
  }
  if (model.box_hull.size() != model.boxes.size()) {
    return TightGeometryStatus::kBadArity;
  }
  for (std::size_t b = 0; b < model.box_hull.size(); ++b) {
    offending_box = b;
    const int h = model.box_hull[b];
    if (h < 0) {
      continue;
    }
    if (static_cast<std::size_t>(h) >= model.hulls.size()) {
      return TightGeometryStatus::kBadArity;
    }
    const LinkHull& hull = model.hulls[static_cast<std::size_t>(h)];
    const Vec3& he = model.boxes[b].half_extents;
    const double hev[3] = {he.x, he.y, he.z};
    for (int i = 0; i < kDopAxes; ++i) {
      if (!std::isfinite(hull.dop_lo[i]) || !std::isfinite(hull.dop_hi[i]) ||
          hull.dop_lo[i] > hull.dop_hi[i]) {
        return TightGeometryStatus::kDegenerate;
      }
    }
    // A 26-DOP lies inside its own first three slabs, and those three axes ARE
    // the box's axes, so this proves DOP ⊆ shipped OBB for the whole polytope.
    for (int k = 0; k < 3; ++k) {
      if (hull.dop_lo[k] < -hev[k] || hull.dop_hi[k] > hev[k]) {
        return TightGeometryStatus::kEscapesBox;
      }
    }
    if (hull.vertex_count == 0) {
      continue;  // stage 1 only
    }
    if (hull.vertex_count < 0 || hull.vertex_count > kMaxTightHullVertices) {
      return TightGeometryStatus::kTooManyVertices;
    }
    if (hull.vertex_first < 0 ||
        static_cast<std::size_t>(hull.vertex_first) + static_cast<std::size_t>(hull.vertex_count) >
            model.hull_vertices.size()) {
      return TightGeometryStatus::kBadArity;
    }
    for (int v = 0; v < hull.vertex_count; ++v) {
      const Vec3& p = model.hull_vertices[static_cast<std::size_t>(hull.vertex_first + v)];
      if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) {
        return TightGeometryStatus::kDegenerate;
      }
      for (int i = 0; i < kDopAxes; ++i) {
        const double s = kDopAxis[i][0] * p.x + kDopAxis[i][1] * p.y + kDopAxis[i][2] * p.z;
        if (s > hull.dop_hi[i] + kTightContainmentEpsilonM ||
            s < hull.dop_lo[i] - kTightContainmentEpsilonM) {
          return TightGeometryStatus::kHullEscapesDop;
        }
      }
    }
  }
  offending_box = 0;
  return TightGeometryStatus::kOk;
}

const char* tight_geometry_status_reason(TightGeometryStatus status) noexcept {
  switch (status) {
  case TightGeometryStatus::kOk:
    return "ok";
  case TightGeometryStatus::kBadArity:
    return "bad_arity";
  case TightGeometryStatus::kEscapesBox:
    return "escapes_box";
  case TightGeometryStatus::kHullEscapesDop:
    return "hull_escapes_dop";
  case TightGeometryStatus::kTooManyVertices:
    return "too_many_vertices";
  case TightGeometryStatus::kDegenerate:
    return "degenerate";
  }
  return "unknown";
}

void tight_pose_init(const LinkHull& hull, const Vec3* hull_vertices, const Transform& box_xf,
                     double half_side, TightPose& out) noexcept {
  out.box = box_xf;
  out.n_vertices = hull.vertex_count;
  out.vertices = hull.vertex_count > 0 ? hull_vertices + hull.vertex_first : nullptr;
  for (int i = 0; i < kDopAxes; ++i) {
    const double ux = kDopAxis[i][0];
    const double uy = kDopAxis[i][1];
    const double uz = kDopAxis[i][2];
    const double nx = box_xf.r[0] * ux + box_xf.r[1] * uy + box_xf.r[2] * uz;
    const double ny = box_xf.r[3] * ux + box_xf.r[4] * uy + box_xf.r[5] * uz;
    const double nz = box_xf.r[6] * ux + box_xf.r[7] * uy + box_xf.r[8] * uz;
    out.axis[i][0] = nx;
    out.axis[i][1] = ny;
    out.axis[i][2] = nz;
    out.axis_dot_t[i] = nx * box_xf.t.x + ny * box_xf.t.y + nz * box_xf.t.z;
    // Support radius of an axis-aligned cube of half-side `half_side` on `n`.
    out.cell_radius[i] = half_side * (std::fabs(nx) + std::fabs(ny) + std::fabs(nz));
    out.lo[i] = hull.dop_lo[i];
    out.hi[i] = hull.dop_hi[i];
  }
  const double mid[3] = {0.5 * (hull.dop_lo[0] + hull.dop_hi[0]),
                         0.5 * (hull.dop_lo[1] + hull.dop_hi[1]),
                         0.5 * (hull.dop_lo[2] + hull.dop_hi[2])};
  const double halfw[3] = {0.5 * (hull.dop_hi[0] - hull.dop_lo[0]),
                           0.5 * (hull.dop_hi[1] - hull.dop_lo[1]),
                           0.5 * (hull.dop_hi[2] - hull.dop_lo[2])};
  for (int k = 0; k < 3; ++k) {
    const double* row = box_xf.r + 3 * k;
    const double t = k == 0 ? box_xf.t.x : (k == 1 ? box_xf.t.y : box_xf.t.z);
    out.slab_center[k] = t + row[0] * mid[0] + row[1] * mid[1] + row[2] * mid[2];
    out.slab_extent[k] =
        std::fabs(row[0]) * halfw[0] + std::fabs(row[1]) * halfw[1] + std::fabs(row[2]) * halfw[2];
  }
}

double dop_cell_lower_bound(const TightPose& pose, const Vec3& center, double half_side,
                            Vec3* best_axis) noexcept {
  const double cv[3] = {center.x, center.y, center.z};
  double best = std::fabs(cv[0] - pose.slab_center[0]) - pose.slab_extent[0] - half_side;
  Vec3 axis{cv[0] >= pose.slab_center[0] ? 1.0 : -1.0, 0.0, 0.0};
  for (int k = 1; k < 3; ++k) {
    const double gap = std::fabs(cv[k] - pose.slab_center[k]) - pose.slab_extent[k] - half_side;
    if (gap > best) {
      best = gap;
      const double s = cv[k] >= pose.slab_center[k] ? 1.0 : -1.0;
      axis = Vec3{k == 0 ? s : 0.0, k == 1 ? s : 0.0, k == 2 ? s : 0.0};
    }
  }
  for (int i = 0; i < kDopAxes; ++i) {
    const double s = pose.axis[i][0] * center.x + pose.axis[i][1] * center.y +
                     pose.axis[i][2] * center.z - pose.axis_dot_t[i];
    const double above = s - pose.hi[i];
    const double below = pose.lo[i] - s;
    const double gap = (above > below ? above : below) - pose.cell_radius[i];
    if (gap > best) {
      best = gap;
      const double sgn = above > below ? 1.0 : -1.0;
      axis = Vec3{sgn * pose.axis[i][0], sgn * pose.axis[i][1], sgn * pose.axis[i][2]};
    }
  }
  if (best_axis != nullptr) {
    *best_axis = axis;
  }
  return best;
}

namespace {

// GJK working simplex: up to four Minkowski-difference points, each remembered
// by the hull vertex and cube corner that produced it so the witness survives
// into the next cell (see `GjkWitness`).
struct GjkSimplex {
  Vec3 p[4]{};
  int vertex[4]{};
  std::uint8_t corner[4]{};
  int n{0};
};

// Closest point to the origin on segment AB; `u` receives the parameter.
Vec3 origin_closest_on_segment(const Vec3& a, const Vec3& b, double& u) noexcept {
  const Vec3 ab = sub(b, a);
  const double denom = dot(ab, ab);
  if (!(denom > 0.0)) {
    u = 0.0;
    return a;
  }
  double t = -dot(a, ab) / denom;
  t = t < 0.0 ? 0.0 : (t > 1.0 ? 1.0 : t);
  u = t;
  return Vec3{a.x + t * ab.x, a.y + t * ab.y, a.z + t * ab.z};
}

// Closest point to the origin on triangle ABC (the standard Voronoi-region
// decomposition with the query point at the origin). `mask` receives the bits
// of the vertices that support the answer (1=A, 2=B, 4=C).
Vec3 origin_closest_on_triangle(const Vec3& a, const Vec3& b, const Vec3& c, int& mask) noexcept {
  const Vec3 ab = sub(b, a);
  const Vec3 ac = sub(c, a);
  const double d1 = -dot(ab, a);
  const double d2 = -dot(ac, a);
  if (d1 <= 0.0 && d2 <= 0.0) {
    mask = 1;
    return a;
  }
  const double d3 = -dot(ab, b);
  const double d4 = -dot(ac, b);
  if (d3 >= 0.0 && d4 <= d3) {
    mask = 2;
    return b;
  }
  const double vc = d1 * d4 - d3 * d2;
  if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) {
    const double v = d1 / (d1 - d3);
    mask = 3;
    return Vec3{a.x + v * ab.x, a.y + v * ab.y, a.z + v * ab.z};
  }
  const double d5 = -dot(ab, c);
  const double d6 = -dot(ac, c);
  if (d6 >= 0.0 && d5 <= d6) {
    mask = 4;
    return c;
  }
  const double vb = d5 * d2 - d1 * d6;
  if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) {
    const double w = d2 / (d2 - d6);
    mask = 5;
    return Vec3{a.x + w * ac.x, a.y + w * ac.y, a.z + w * ac.z};
  }
  const double va = d3 * d6 - d5 * d4;
  if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0) {
    const double w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
    mask = 6;
    return Vec3{b.x + w * (c.x - b.x), b.y + w * (c.y - b.y), b.z + w * (c.z - b.z)};
  }
  const double denom = va + vb + vc;
  if (!(denom > 0.0)) {
    // Degenerate (collinear) triangle: its longest edge carries the answer.
    double u = 0.0;
    mask = 3;
    return origin_closest_on_segment(a, b, u);
  }
  const double v = vb / denom;
  const double w = vc / denom;
  mask = 7;
  return Vec3{a.x + ab.x * v + ac.x * w, a.y + ab.y * v + ac.y * w, a.z + ab.z * v + ac.z * w};
}

// Is the origin on the far side of plane ABC from D?
bool origin_outside_face(const Vec3& a, const Vec3& b, const Vec3& c, const Vec3& d) noexcept {
  const Vec3 n = cross(sub(b, a), sub(c, a));
  return (-dot(a, n)) * dot(sub(d, a), n) < 0.0;
}

void simplex_keep(GjkSimplex& s, const int* which, int k) noexcept {
  Vec3 p[4];
  int vi[4];
  std::uint8_t co[4];
  for (int i = 0; i < k; ++i) {
    p[i] = s.p[which[i]];
    vi[i] = s.vertex[which[i]];
    co[i] = s.corner[which[i]];
  }
  for (int i = 0; i < k; ++i) {
    s.p[i] = p[i];
    s.vertex[i] = vi[i];
    s.corner[i] = co[i];
  }
  s.n = k;
}

// Closest point to the origin on the simplex, reducing it to the supporting
// sub-face. `contains_origin` is set when a tetrahedron encloses the origin.
Vec3 simplex_closest(GjkSimplex& s, bool& contains_origin) noexcept {
  contains_origin = false;
  if (s.n <= 1) {
    return s.p[0];
  }
  if (s.n == 2) {
    double u = 0.0;
    const Vec3 q = origin_closest_on_segment(s.p[0], s.p[1], u);
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
    const Vec3 q = origin_closest_on_triangle(s.p[0], s.p[1], s.p[2], mask);
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
  Vec3 best{};
  int keep_idx[3] = {0, 0, 0};
  int keep_n = 0;
  bool any_outside = false;
  for (int f = 0; f < 4; ++f) {
    const Vec3& fa = s.p[kFace[f][0]];
    const Vec3& fb = s.p[kFace[f][1]];
    const Vec3& fc = s.p[kFace[f][2]];
    if (!origin_outside_face(fa, fb, fc, s.p[kOpposite[f]])) {
      continue;
    }
    any_outside = true;
    int mask = 0;
    const Vec3 q = origin_closest_on_triangle(fa, fb, fc, mask);
    const double q2 = dot(q, q);
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
    return Vec3{};
  }
  simplex_keep(s, keep_idx, keep_n);
  return best;
}

// Base-frame position of hull vertex `i`.
Vec3 hull_vertex_base(const TightPose& pose, int i) noexcept {
  const Vec3& p = pose.vertices[i];
  const Transform& x = pose.box;
  return Vec3{x.t.x + x.r[0] * p.x + x.r[1] * p.y + x.r[2] * p.z,
              x.t.y + x.r[3] * p.x + x.r[4] * p.y + x.r[5] * p.z,
              x.t.z + x.r[6] * p.x + x.r[7] * p.y + x.r[8] * p.z};
}

// The TRUE support of the hull along base-frame direction `dir`: an exhaustive
// scan, which is what makes the supporting-hyperplane bound in
// `hull_cell_distance` sound. Returns the winning vertex index.
int hull_support(const TightPose& pose, const Vec3& dir, Vec3& out) noexcept {
  const Transform& x = pose.box;
  // The direction in the box's local frame (Rᵀ dir), so the scan is a plain dot
  // product against the stored local vertices.
  const double lx = x.r[0] * dir.x + x.r[3] * dir.y + x.r[6] * dir.z;
  const double ly = x.r[1] * dir.x + x.r[4] * dir.y + x.r[7] * dir.z;
  const double lz = x.r[2] * dir.x + x.r[5] * dir.y + x.r[8] * dir.z;
  const Vec3* v = pose.vertices;
  int best = 0;
  double best_dot = v[0].x * lx + v[0].y * ly + v[0].z * lz;
  for (int i = 1; i < pose.n_vertices; ++i) {
    const double d = v[i].x * lx + v[i].y * ly + v[i].z * lz;
    if (d > best_dot) {
      best_dot = d;
      best = i;
    }
  }
  out = hull_vertex_base(pose, best);
  return best;
}

std::uint8_t cell_corner_code(const Vec3& dir) noexcept {
  return static_cast<std::uint8_t>((dir.x >= 0.0 ? 1 : 0) | (dir.y >= 0.0 ? 2 : 0) |
                                   (dir.z >= 0.0 ? 4 : 0));
}

Vec3 cell_corner(const Vec3& center, double half_side, std::uint8_t code) noexcept {
  return Vec3{center.x + ((code & 1U) != 0U ? half_side : -half_side),
              center.y + ((code & 2U) != 0U ? half_side : -half_side),
              center.z + ((code & 4U) != 0U ? half_side : -half_side)};
}

}  // namespace

double hull_cell_distance(const TightPose& pose, const Vec3& center, double half_side,
                          const Vec3& seed_dir, double margin, double fallback,
                          GjkWitness& witness) noexcept {
  if (pose.vertices == nullptr || pose.n_vertices <= 0) {
    return fallback;
  }
  GjkSimplex s;
  for (int i = 0; i < witness.n && i < 4; ++i) {
    s.vertex[i] = witness.vertex[i];
    s.corner[i] = witness.corner[i];
    s.p[i] = sub(hull_vertex_base(pose, witness.vertex[i]),
                 cell_corner(center, half_side, witness.corner[i]));
    s.n = i + 1;
  }
  if (s.n == 0) {
    // The Minkowski point `a - b` runs from the cell toward the link, so the
    // hull's own support is along +seed_dir and the cell's corner along -it.
    Vec3 hp;
    s.vertex[0] = hull_support(pose, seed_dir, hp);
    s.corner[0] = cell_corner_code(Vec3{-seed_dir.x, -seed_dir.y, -seed_dir.z});
    s.p[0] = sub(hp, cell_corner(center, half_side, s.corner[0]));
    s.n = 1;
  }
  double lower = -std::numeric_limits<double>::infinity();
  for (int it = 0; it < kGjkMaxIterations; ++it) {
    bool contains = false;
    const Vec3 v = simplex_closest(s, contains);
    const double vlen2 = dot(v, v);
    if (contains || vlen2 <= 1e-24) {
      witness.n = 0;  // overlapping: stage 1's bound (<= 0 here) stands
      return fallback;
    }
    const double vlen = std::sqrt(vlen2);
    const Vec3 search{-v.x / vlen, -v.y / vlen, -v.z / vlen};
    Vec3 hp;
    const int vi = hull_support(pose, search, hp);
    const std::uint8_t co = cell_corner_code(v);
    const Vec3 w = sub(hp, cell_corner(center, half_side, co));
    // Supporting-hyperplane bound: no point of the difference set lies closer
    // to the origin than this, so it lower-bounds the surface distance.
    const double bound = dot(v, w) / vlen;
    if (bound > lower) {
      lower = bound;
    }
    if (lower > margin) {
      break;  // certainly clear; sharpening a diagnostic is not worth the scan
    }
    if (vlen - lower <= kGjkTolerance) {
      lower = vlen;  // converged on the exact surface distance
      break;
    }
    if (s.n == 4) {
      break;  // no room to grow; the bound already stands
    }
    s.p[s.n] = w;
    s.vertex[s.n] = vi;
    s.corner[s.n] = co;
    ++s.n;
  }
  witness.n = s.n;
  for (int i = 0; i < s.n; ++i) {
    witness.vertex[i] = s.vertex[i];
    witness.corner[i] = s.corner[i];
  }
  return lower > fallback ? lower : fallback;
}

void forward_kinematics(const CollisionModel& model, const double* qpos, std::size_t n_dof,
                        CollisionScratch& scratch) noexcept {
  for (std::size_t i = 0; i < model.n_links; ++i) {
    // Joint motion (parent's joint frame -> this link frame), driven by qpos.
    Transform motion;  // identity for fixed joints
    const int dof = model.dof_index[i];
    if (dof >= 0 && static_cast<std::size_t>(dof) < n_dof) {
      const double q = qpos[static_cast<std::size_t>(dof)];
      if (model.joint_kind[i] == JointKind::kRevolute) {
        axis_angle(model.axis[i], q, motion.r);
      } else if (model.joint_kind[i] == JointKind::kPrismatic) {
        motion.t = scale(model.axis[i], q);
      }
    }
    const Transform local = compose(model.origin[i], motion);
    const int parent = model.parent[i];
    scratch.link_world[i] =
        parent < 0 ? local : compose(scratch.link_world[static_cast<std::size_t>(parent)], local);
  }
}

bool jacobian_dls_step(const CollisionModel& model, const CollisionScratch& scratch, int ee_link,
                       const double ee_twist[6], double lambda, double* dq, std::size_t n_dof,
                       const std::uint8_t* dof_blocked) noexcept {
  if (dq == nullptr) {
    return false;
  }
  for (std::size_t i = 0; i < n_dof; ++i) {
    dq[i] = 0.0;
  }
  if (ee_link < 0 || static_cast<std::size_t>(ee_link) >= model.n_links ||
      n_dof > kMaxJacobianDof) {
    return false;
  }
  // Build the geometric Jacobian columns for every movable joint on the chain
  // root -> ee_link (other joints do not move the EE; their columns are zero so
  // we simply skip them). `Jv`/`Jw` are the linear/angular rows of each column;
  // `col_dof` remembers which qpos index the column drives.
  const Vec3 p_ee = scratch.link_world[static_cast<std::size_t>(ee_link)].t;
  double Jv[3][kMaxJacobianDof];
  double Jw[3][kMaxJacobianDof];
  std::size_t col_dof[kMaxJacobianDof];
  std::size_t m = 0;
  for (int link = ee_link; link >= 0; link = model.parent[static_cast<std::size_t>(link)]) {
    const std::size_t li = static_cast<std::size_t>(link);
    const int dof = model.dof_index[li];
    if (dof < 0 || static_cast<std::size_t>(dof) >= n_dof) {
      continue;  // fixed joint (or out-of-range) — no column
    }
    if (dof_blocked != nullptr && dof_blocked[static_cast<std::size_t>(dof)] != 0) {
      continue;  // excluded (e.g. mobile-base dof) — not driven by the arm twist
    }
    if (m >= kMaxJacobianDof) {
      return false;
    }
    // Joint axis in the base frame: R(link_world[li]) * axis (the FK rotation
    // about the joint axis leaves that axis invariant, so this equals the world
    // axis through the joint origin).
    const Vec3 z = rotate(scratch.link_world[li].r, model.axis[li]);
    const Vec3 p = scratch.link_world[li].t;
    if (model.joint_kind[li] == JointKind::kPrismatic) {
      Jv[0][m] = z.x;
      Jv[1][m] = z.y;
      Jv[2][m] = z.z;
      Jw[0][m] = 0.0;
      Jw[1][m] = 0.0;
      Jw[2][m] = 0.0;
    } else {  // revolute (fixed already skipped)
      const Vec3 jv = cross(z, sub(p_ee, p));
      Jv[0][m] = jv.x;
      Jv[1][m] = jv.y;
      Jv[2][m] = jv.z;
      Jw[0][m] = z.x;
      Jw[1][m] = z.y;
      Jw[2][m] = z.z;
    }
    col_dof[m] = static_cast<std::size_t>(dof);
    ++m;
  }
  if (m == 0) {
    return false;  // no movable joint feeds the EE — cannot predict
  }
  // Normal-equations matrix A = J·Jᵀ + λ²·I  (6x6, row-major), and solve A·y = b
  // with b = ee_twist; then dq = Jᵀ·y.
  double A[36] = {0.0};
  for (int r = 0; r < 6; ++r) {
    const double* Jr = (r < 3) ? Jv[r] : Jw[r - 3];
    for (int c = 0; c < 6; ++c) {
      const double* Jc = (c < 3) ? Jv[c] : Jw[c - 3];
      double s = 0.0;
      for (std::size_t k = 0; k < m; ++k) {
        s += Jr[k] * Jc[k];
      }
      A[r * 6 + c] = s;
    }
    A[r * 6 + r] += lambda * lambda;
  }
  double b[6] = {ee_twist[0], ee_twist[1], ee_twist[2], ee_twist[3], ee_twist[4], ee_twist[5]};
  double y[6] = {0.0};
  if (!solve6(A, b, y)) {
    return false;
  }
  for (std::size_t k = 0; k < m; ++k) {
    double s = 0.0;
    for (int r = 0; r < 3; ++r) {
      s += Jv[r][k] * y[r] + Jw[r][k] * y[r + 3];
    }
    dq[col_dof[k]] = s;
  }
  return true;
}

namespace {

// Fold one checked geometry pair into a sweep's evidence.
//
// `sweep_min` accumulates the minimum surface distance over EVERY pair a check
// touches — pairs that never reach the margin and pairs the caller's gate
// deliberately exempts included. `tripped` says whether this pair actually
// tripped the check; only a tripped pair may become the reported evidence, and
// it does so only while it is deeper than the pair already recorded. That keeps
// `link_a` / `link_b` / `min_distance` describing one and the same (deepest)
// tripping pair instead of pairing one cell's identity with another cell's
// distance.
//
// Reporting only: `hit.hit` flips for exactly the same set of pairs as a plain
// `if (tripped) hit.hit = true;` would.
// `allowance_active` is carried alongside the pair it belongs to for the same
// reason the distance is: the flag must describe the pair that is reported, not
// whichever pair happened to be checked last.
void fold_pair(CollisionHit& hit, double& sweep_min, double d, bool tripped, int link_a, int link_b,
               bool allowance_active = false) noexcept {
  if (d < sweep_min) {
    sweep_min = d;
  }
  if (!tripped) {
    return;
  }
  if (!hit.hit || d < hit.min_distance) {
    hit.hit = true;
    hit.link_a = link_a;
    hit.link_b = link_b;
    hit.min_distance = d;
    hit.place_allowance_active = allowance_active;
  }
}

// Close a sweep: publish the sweep-wide minimum, and — when nothing tripped —
// let `min_distance` keep its clearance meaning (there is no pair to describe).
CollisionHit finish_sweep(CollisionHit hit, double sweep_min) noexcept {
  hit.sweep_min_distance = sweep_min;
  if (!hit.hit) {
    hit.min_distance = sweep_min;
  }
  return hit;
}

}  // namespace

CollisionHit check_self_collision(const CollisionModel& model, const CollisionScratch& scratch,
                                  double margin) noexcept {
  CollisionHit result;
  double sweep_min = std::numeric_limits<double>::infinity();
  const std::size_t n_caps = model.capsules.size();
  for (std::size_t i = 0; i < n_caps; ++i) {
    const int li = model.capsule_link[i];
    const Transform cap_i =
        compose(scratch.link_world[static_cast<std::size_t>(li)], model.capsules[i].origin);
    for (std::size_t j = i + 1; j < n_caps; ++j) {
      const int lj = model.capsule_link[j];
      if (li == lj) {
        continue;  // capsules on the same link never self-collide
      }
      if (is_allowed(model, li, lj)) {
        continue;
      }
      const Transform cap_j =
          compose(scratch.link_world[static_cast<std::size_t>(lj)], model.capsules[j].origin);
      const double d =
          capsule_distance(cap_i, model.capsules[i].radius, model.capsules[i].half_length, cap_j,
                           model.capsules[j].radius, model.capsules[j].half_length);
      fold_pair(result, sweep_min, d, d <= margin, li, lj);
    }
  }
  // Blocky links carry an OBB instead of a capsule (issue #84):
  // every box↔capsule and box↔box pair is checked with the same allowed-pair /
  // same-link skips. A box↔capsule distance is exact; a box↔box distance is a
  // conservative lower bound (SAT). Boxes are few (one per blocky link), so
  // this stays cheap and allocation-free.
  const std::size_t n_boxes = model.boxes.size();
  for (std::size_t bi = 0; bi < n_boxes; ++bi) {
    const int lb = model.box_link[bi];
    const Transform box_i =
        compose(scratch.link_world[static_cast<std::size_t>(lb)], model.boxes[bi].origin);
    for (std::size_t cj = 0; cj < n_caps; ++cj) {
      const int lc = model.capsule_link[cj];
      if (lb == lc || is_allowed(model, lb, lc)) {
        continue;
      }
      const Transform cap_j =
          compose(scratch.link_world[static_cast<std::size_t>(lc)], model.capsules[cj].origin);
      const double d =
          box_capsule_distance(box_i, model.boxes[bi].half_extents, cap_j,
                               model.capsules[cj].radius, model.capsules[cj].half_length);
      fold_pair(result, sweep_min, d, d <= margin, lb, lc);
    }
    for (std::size_t bj = bi + 1; bj < n_boxes; ++bj) {
      const int lb2 = model.box_link[bj];
      if (lb == lb2 || is_allowed(model, lb, lb2)) {
        continue;
      }
      const Transform box_j =
          compose(scratch.link_world[static_cast<std::size_t>(lb2)], model.boxes[bj].origin);
      const double d = box_box_distance(box_i, model.boxes[bi].half_extents, box_j,
                                        model.boxes[bj].half_extents);
      fold_pair(result, sweep_min, d, d <= margin, lb, lb2);
    }
  }
  return finish_sweep(result, sweep_min);
}

CollisionHit check_world_collision(const CollisionModel& model, const CollisionScratch& scratch,
                                   const WorldModel& world, double margin) noexcept {
  CollisionHit result;
  double sweep_min = std::numeric_limits<double>::infinity();
  const std::size_t n_caps = model.capsules.size();
  const std::size_t n_world = world.capsules.size();
  for (std::size_t i = 0; i < n_caps; ++i) {
    const int li = model.capsule_link[i];
    const Transform cap_i =
        compose(scratch.link_world[static_cast<std::size_t>(li)], model.capsules[i].origin);
    for (std::size_t w = 0; w < n_world; ++w) {
      // World capsule origins are already absolute in the base frame.
      const double d = capsule_distance(cap_i, model.capsules[i].radius,
                                        model.capsules[i].half_length, world.capsules[w].origin,
                                        world.capsules[w].radius, world.capsules[w].half_length);
      // link_a: robot link; link_b: world obstacle index.
      fold_pair(result, sweep_min, d, d <= margin, li, static_cast<int>(w));
    }
  }
  // Blocky links (OBB) are checked against every world obstacle too,
  // so a boxed link is never invisible to the world check.
  const std::size_t n_boxes = model.boxes.size();
  for (std::size_t b = 0; b < n_boxes; ++b) {
    const int lb = model.box_link[b];
    const Transform box_w =
        compose(scratch.link_world[static_cast<std::size_t>(lb)], model.boxes[b].origin);
    for (std::size_t w = 0; w < n_world; ++w) {
      const double d =
          box_capsule_distance(box_w, model.boxes[b].half_extents, world.capsules[w].origin,
                               world.capsules[w].radius, world.capsules[w].half_length);
      fold_pair(result, sweep_min, d, d <= margin, lb, static_cast<int>(w));
    }
  }
  return finish_sweep(result, sweep_min);
}

CollisionHit check_voxel_collision(const CollisionModel& model, const CollisionScratch& scratch,
                                   const VoxelGrid& grid, double margin) noexcept {
  CollisionHit result;
  double sweep_min = std::numeric_limits<double>::infinity();
  if (grid.occupancy == nullptr || grid.sx <= 0 || grid.sy <= 0 || grid.sz <= 0 ||
      grid.resolution <= 0.0) {
    return finish_sweep(result, sweep_min);
  }
  // Occupied cells are axis-aligned cubes. Use the existing exact
  // capsule-vs-box / conservative box-vs-box routines rather than replacing
  // each cube by a circumscribed sphere: the sphere bulges 18.3 mm beyond a
  // 5 cm voxel face and false-stops close manipulation near voxel corners.
  const double half_side = grid.resolution * 0.5;
  const Vec3 voxel_half{half_side, half_side, half_side};
  const double inv_res = 1.0 / grid.resolution;
  // Voxel-index span of a base-frame interval, clamped to the grid (shared by
  // the capsule and box passes).
  const auto rng = [&](double lo, double hi, double org, int dim) {
    const int i0 = static_cast<int>(std::floor((lo - org) * inv_res));
    const int i1 = static_cast<int>(std::floor((hi - org) * inv_res));
    return std::pair<int, int>{clamp_index(i0, 0, dim - 1), clamp_index(i1, 0, dim - 1)};
  };
  const std::size_t n_caps = model.capsules.size();
  for (std::size_t c = 0; c < n_caps; ++c) {
    const int li = model.capsule_link[c];
    const Transform cap =
        compose(scratch.link_world[static_cast<std::size_t>(li)], model.capsules[c].origin);
    Vec3 p0;
    Vec3 p1;
    capsule_endpoints(cap, model.capsules[c].half_length, p0, p1);
    const double r = model.capsules[c].radius;
    const double reach = r + margin + half_side;
    const auto [ix0, ix1] =
        rng(std::min(p0.x, p1.x) - reach, std::max(p0.x, p1.x) + reach, grid.origin.x, grid.sx);
    const auto [iy0, iy1] =
        rng(std::min(p0.y, p1.y) - reach, std::max(p0.y, p1.y) + reach, grid.origin.y, grid.sy);
    const auto [iz0, iz1] =
        rng(std::min(p0.z, p1.z) - reach, std::max(p0.z, p1.z) + reach, grid.origin.z, grid.sz);
    for (int iz = iz0; iz <= iz1; ++iz) {
      for (int iy = iy0; iy <= iy1; ++iy) {
        for (int ix = ix0; ix <= ix1; ++ix) {
          const std::size_t idx = static_cast<std::size_t>(ix + grid.sx * (iy + grid.sy * iz));
          if (grid.occupancy[idx] == 0) {
            continue;
          }
          const Vec3 center{grid.origin.x + (ix + 0.5) * grid.resolution,
                            grid.origin.y + (iy + 0.5) * grid.resolution,
                            grid.origin.z + (iz + 0.5) * grid.resolution};
          Transform voxel;
          voxel.t = center;
          const double d =
              box_capsule_distance(voxel, voxel_half, cap, r, model.capsules[c].half_length);
          fold_pair(result, sweep_min, d, d <= margin, li, static_cast<int>(idx));
        }
      }
    }
  }
  // Blocky links (OBB) are voxel-checked too, against the same conservative
  // voxel cube. A link that declares tight geometry runs the staged 26-DOP →
  // exact-hull narrow phase instead of `box_box_distance`; the window below is
  // identical either way, because both stages are proved subsets of this same
  // OBB at configure time (`validate_tight_geometry`).
  const std::size_t n_boxes = model.boxes.size();
  const bool has_tight = model.box_hull.size() == n_boxes;
  // Bounded across every link in this call: stage 2's cost is set by how many
  // cells stage 1 cannot clear, which is a property of the map. Exhausting it
  // drops back to stage 1 + the shipped bound, i.e. to today's answer or better.
  int stage2_budget = kMaxStage2PerCheck;
  for (std::size_t b = 0; b < n_boxes; ++b) {
    const int lb = model.box_link[b];
    const Transform box_w =
        compose(scratch.link_world[static_cast<std::size_t>(lb)], model.boxes[b].origin);
    const Vec3 he = model.boxes[b].half_extents;
    const int hull_index = has_tight ? model.box_hull[b] : -1;
    TightPose tight;
    GjkWitness witness;
    if (hull_index >= 0) {
      tight_pose_init(model.hulls[static_cast<std::size_t>(hull_index)],
                      model.hull_vertices.empty() ? nullptr : model.hull_vertices.data(), box_w,
                      half_side, tight);
    }
    // World-AABB half-size of the oriented box: |R| * half_extents (row-major R).
    const double ex =
        std::fabs(box_w.r[0]) * he.x + std::fabs(box_w.r[1]) * he.y + std::fabs(box_w.r[2]) * he.z;
    const double ey =
        std::fabs(box_w.r[3]) * he.x + std::fabs(box_w.r[4]) * he.y + std::fabs(box_w.r[5]) * he.z;
    const double ez =
        std::fabs(box_w.r[6]) * he.x + std::fabs(box_w.r[7]) * he.y + std::fabs(box_w.r[8]) * he.z;
    const double reach = margin + half_side;
    const auto [ix0, ix1] =
        rng(box_w.t.x - ex - reach, box_w.t.x + ex + reach, grid.origin.x, grid.sx);
    const auto [iy0, iy1] =
        rng(box_w.t.y - ey - reach, box_w.t.y + ey + reach, grid.origin.y, grid.sy);
    const auto [iz0, iz1] =
        rng(box_w.t.z - ez - reach, box_w.t.z + ez + reach, grid.origin.z, grid.sz);
    for (int iz = iz0; iz <= iz1; ++iz) {
      for (int iy = iy0; iy <= iy1; ++iy) {
        for (int ix = ix0; ix <= ix1; ++ix) {
          const std::size_t idx = static_cast<std::size_t>(ix + grid.sx * (iy + grid.sy * iz));
          if (grid.occupancy[idx] == 0) {
            continue;
          }
          const Vec3 center{grid.origin.x + (ix + 0.5) * grid.resolution,
                            grid.origin.y + (iy + 0.5) * grid.resolution,
                            grid.origin.z + (iz + 0.5) * grid.resolution};
          double d;
          if (hull_index < 0) {
            Transform voxel;
            voxel.t = center;
            d = box_box_distance(box_w, he, voxel, voxel_half);
          } else {
            Vec3 seed;
            d = dop_cell_lower_bound(tight, center, half_side, &seed);
            if (d <= margin) {
              // Stage 2, on the cells stage 1 could not clear: the exact
              // hull-to-cell distance, or stage 1's bound again for a link that
              // ships no hull or a call that has spent its refinement budget.
              if (tight.n_vertices > 0 && stage2_budget > 0) {
                --stage2_budget;
                d = hull_cell_distance(tight, center, half_side, seed, margin, d, witness);
              }
              if (d <= margin) {
                // Still stopping. Fold in the shipped bound before committing to
                // that, because the DOP's 16 separating axes are NOT a superset
                // of the OBB SAT's 15 -- the box's edge-cross axes beat every DOP
                // axis at some cells, so the tighter *solid* can still yield the
                // looser *bound*. The maximum of two lower bounds is a lower
                // bound, so this costs no soundness and buys a strong property:
                // the staged path can never stop where `box_box_distance` alone
                // would not have. This change removes false stops; it must not be
                // able to add one.
                Transform voxel;
                voxel.t = center;
                const double shipped = box_box_distance(box_w, he, voxel, voxel_half);
                if (shipped > d) {
                  d = shipped;
                }
              }
            }
          }
          fold_pair(result, sweep_min, d, d <= margin, lb, static_cast<int>(idx));
        }
      }
    }
  }
  return finish_sweep(result, sweep_min);
}

namespace {

// World-frame transform of an attached object: compose the attach-link's FK'd
// frame with the object's link-relative pose. FK must already be run. Each
// owned primitive then composes its own `pose_in_object` on top of this.
Transform attached_object_transform(const AttachedObject& obj,
                                    const CollisionScratch& scratch) noexcept {
  return compose(scratch.link_world[static_cast<std::size_t>(obj.attach_link)], obj.pose_in_link);
}

// True when robot link `link` is the object's attach link or one of its
// explicit touch links (grasped payloads legitimately contact those).
bool attached_allows_link(const AttachedModel& attached, const AttachedObject& obj,
                          int link) noexcept {
  if (link == obj.attach_link) {
    return true;
  }
  for (int k = 0; k < obj.touch_count; ++k) {
    if (attached.touch_links[static_cast<std::size_t>(obj.touch_first + k)] == link) {
      return true;
    }
  }
  return false;
}

// Surface distance between one attached primitive (world frame `prim_xf`) and a
// robot/world capsule (world frame `cap_xf`). A sphere is a capsule with
// half_length 0; a box uses the exact box↔capsule routine.
double attached_primitive_capsule_distance(const AttachedPrimitive& prim, const Transform& prim_xf,
                                           const Transform& cap_xf, double cap_radius,
                                           double cap_half_length) noexcept {
  if (prim.kind == AttachedShapeKind::kBox) {
    return box_capsule_distance(prim_xf, prim.half_extents, cap_xf, cap_radius, cap_half_length);
  }
  return capsule_distance(prim_xf, prim.radius, prim.half_length, cap_xf, cap_radius,
                          cap_half_length);
}

double attached_object_voxel_distance(const AttachedModel& attached, const AttachedObject& object,
                                      const Transform& object_xf, const Transform& voxel,
                                      const Vec3& voxel_half) noexcept {
  double minimum = std::numeric_limits<double>::infinity();
  for (int p = 0; p < object.prim_count; ++p) {
    const AttachedPrimitive& primitive =
        attached.primitives[static_cast<std::size_t>(object.prim_first + p)];
    const Transform primitive_xf = compose(object_xf, primitive.pose_in_object);
    const double distance =
        primitive.kind == AttachedShapeKind::kBox
            ? box_box_distance(primitive_xf, primitive.half_extents, voxel, voxel_half)
            : box_capsule_distance(voxel, voxel_half, primitive_xf, primitive.radius,
                                   primitive.half_length);
    minimum = std::min(minimum, distance);
  }
  return minimum;
}

// Surface distance between one attached primitive (world frame `prim_xf`) and
// an occupied voxel treated conservatively as its own cube.
double attached_primitive_voxel_distance(const AttachedPrimitive& prim, const Transform& prim_xf,
                                         const Transform& voxel, const Vec3& voxel_half) noexcept {
  if (prim.kind == AttachedShapeKind::kBox) {
    return box_box_distance(prim_xf, prim.half_extents, voxel, voxel_half);
  }
  return box_capsule_distance(voxel, voxel_half, prim_xf, prim.radius, prim.half_length);
}

// Inclusive voxel-index box covering one attached primitive's world AABB grown
// by `reach`. Shared by the attached-voxel check and the witness latch so the
// two can never scan different cell sets for the same payload pose.
struct CellBox {
  int x0{0};
  int x1{-1};
  int y0{0};
  int y1{-1};
  int z0{0};
  int z1{-1};
};

CellBox primitive_cell_box(const AttachedPrimitive& prim, const Transform& prim_xf,
                           const VoxelGrid& grid, double reach) noexcept {
  // World-AABB half-size of the primitive (|R| * extents). For a sphere/capsule
  // the swept radius plus the endpoint span is the conservative extent.
  double ex;
  double ey;
  double ez;
  if (prim.kind == AttachedShapeKind::kBox) {
    const Vec3 he = prim.half_extents;
    ex = std::fabs(prim_xf.r[0]) * he.x + std::fabs(prim_xf.r[1]) * he.y +
         std::fabs(prim_xf.r[2]) * he.z;
    ey = std::fabs(prim_xf.r[3]) * he.x + std::fabs(prim_xf.r[4]) * he.y +
         std::fabs(prim_xf.r[5]) * he.z;
    ez = std::fabs(prim_xf.r[6]) * he.x + std::fabs(prim_xf.r[7]) * he.y +
         std::fabs(prim_xf.r[8]) * he.z;
  } else {
    // Capsule/sphere: axis is local +Z; half-length projects onto each axis.
    ex = std::fabs(prim_xf.r[2]) * prim.half_length + prim.radius;
    ey = std::fabs(prim_xf.r[5]) * prim.half_length + prim.radius;
    ez = std::fabs(prim_xf.r[8]) * prim.half_length + prim.radius;
  }
  const double inv_res = 1.0 / grid.resolution;
  const auto rng = [inv_res](double lo, double hi, double org, int dim) {
    const int i0 = static_cast<int>(std::floor((lo - org) * inv_res));
    const int i1 = static_cast<int>(std::floor((hi - org) * inv_res));
    return std::pair<int, int>{clamp_index(i0, 0, dim - 1), clamp_index(i1, 0, dim - 1)};
  };
  const auto [x0, x1] =
      rng(prim_xf.t.x - ex - reach, prim_xf.t.x + ex + reach, grid.origin.x, grid.sx);
  const auto [y0, y1] =
      rng(prim_xf.t.y - ey - reach, prim_xf.t.y + ey + reach, grid.origin.y, grid.sy);
  const auto [z0, z1] =
      rng(prim_xf.t.z - ez - reach, prim_xf.t.z + ez + reach, grid.origin.z, grid.sz);
  return CellBox{x0, x1, y0, y1, z0, z1};
}

// Base-frame centre of cell (ix, iy, iz).
Vec3 voxel_center(const VoxelGrid& grid, int ix, int iy, int iz) noexcept {
  return Vec3{grid.origin.x + (ix + 0.5) * grid.resolution,
              grid.origin.y + (iy + 0.5) * grid.resolution,
              grid.origin.z + (iz + 0.5) * grid.resolution};
}

// Is object `obj`'s attested support contact still doing work — i.e. is some
// occupied cell it would exempt still within `margin` of the payload? This is
// the liveness predicate behind the witness latch: when it goes false the
// payload has separated from its attested support and the exemption dies.
// Bounded to the payload's own inflated AABB; allocation-free.
bool support_witness_still_in_contact(const AttachedModel& attached, const AttachedObject& obj,
                                      const Transform& obj_xf, const VoxelGrid& grid,
                                      double margin) noexcept {
  const double half_side = grid.resolution * 0.5;
  const Vec3 voxel_half{half_side, half_side, half_side};
  for (int p = 0; p < obj.prim_count; ++p) {
    const AttachedPrimitive& prim =
        attached.primitives[static_cast<std::size_t>(obj.prim_first + p)];
    const Transform prim_xf = compose(obj_xf, prim.pose_in_object);
    const CellBox box = primitive_cell_box(prim, prim_xf, grid, margin + half_side);
    for (int iz = box.z0; iz <= box.z1; ++iz) {
      for (int iy = box.y0; iy <= box.y1; ++iy) {
        for (int ix = box.x0; ix <= box.x1; ++ix) {
          const std::size_t idx = static_cast<std::size_t>(ix + grid.sx * (iy + grid.sy * iz));
          if (grid.occupancy[idx] == 0) {
            continue;
          }
          const Vec3 center = voxel_center(grid, ix, iy, iz);
          if (!support_contact_exempts(obj, obj_xf, center, grid.resolution,
                                       grid.attached_contact_tolerance)) {
            continue;
          }
          Transform voxel;
          voxel.t = center;
          if (attached_primitive_voxel_distance(prim, prim_xf, voxel, voxel_half) <= margin) {
            return true;
          }
        }
      }
    }
  }
  return false;
}

}  // namespace

AttachIngestStatus ingest_attached_objects(const std::vector<AttachedObjectInput>& inputs,
                                           const std::vector<std::string>& link_names,
                                           std::size_t max_objects, std::size_t max_primitives,
                                           std::size_t max_touch_links,
                                           AttachedModel& out) noexcept {
  out.n_objects = 0;
  out.n_primitives = 0;
  // Object-count cap first (fail-closed on overflow). The total primitive and
  // touch-link caps are enforced incrementally below as their flattened
  // cursors advance, so an object set whose *sum* of primitives/touch-links
  // exceeds the budget also fails closed.
  const std::size_t n = inputs.size();
  if (n > max_objects || n > out.objects.size()) {
    return AttachIngestStatus::kOverflow;
  }
  const auto resolve = [&link_names](const std::string& name) -> int {
    for (std::size_t i = 0; i < link_names.size(); ++i) {
      if (link_names[i] == name) {
        return static_cast<int>(i);
      }
    }
    return -1;
  };
  std::size_t prim_cursor = 0;
  std::size_t touch_cursor = 0;
  for (std::size_t i = 0; i < n; ++i) {
    const AttachedObjectInput& in = inputs[i];
    const int attach = resolve(in.attach_link);
    if (attach < 0) {
      out.n_objects = 0;
      out.n_primitives = 0;
      return AttachIngestStatus::kUnknownLink;
    }
    // An object with no geometry is not safe to ignore — fail closed.
    const std::size_t n_prims = in.primitives.size();
    if (n_prims == 0) {
      out.n_objects = 0;
      out.n_primitives = 0;
      return AttachIngestStatus::kMalformed;
    }
    // Total-primitive cap (across all objects): incremental fail-closed.
    if (prim_cursor + n_prims > max_primitives || prim_cursor + n_prims > out.primitives.size()) {
      out.n_objects = 0;
      out.n_primitives = 0;
      return AttachIngestStatus::kOverflow;
    }
    const std::size_t n_touch = in.touch_links.size();
    if (touch_cursor + n_touch > max_touch_links ||
        touch_cursor + n_touch > out.touch_links.size()) {
      out.n_objects = 0;
      out.n_primitives = 0;
      return AttachIngestStatus::kOverflow;
    }
    // Support-contact attestation (ADR-0092 D6). A malformed witness is not
    // downgraded to "no witness" — it fails the whole ingest closed, because a
    // producer that cannot describe its own support contact is a producer whose
    // attachment set we do not trust either.
    Vec3 support_normal{0.0, 0.0, 1.0};
    if (in.has_support_witness) {
      const Vec3& normal_in = in.support_normal;
      const double norm_sq =
          normal_in.x * normal_in.x + normal_in.y * normal_in.y + normal_in.z * normal_in.z;
      if (!std::isfinite(norm_sq) || norm_sq <= 1e-12 || !std::isfinite(in.support_point.x) ||
          !std::isfinite(in.support_point.y) || !std::isfinite(in.support_point.z) ||
          !std::isfinite(in.support_patch_radius) || in.support_patch_radius <= 0.0 ||
          !std::isfinite(in.support_max_penetration) || in.support_max_penetration < 0.0) {
        out.n_objects = 0;
        out.n_primitives = 0;
        return AttachIngestStatus::kMalformed;
      }
      const double inv_norm = 1.0 / std::sqrt(norm_sq);
      support_normal = Vec3{normal_in.x * inv_norm, normal_in.y * inv_norm, normal_in.z * inv_norm};
    }
    AttachedObject& o = out.objects[i];
    o.pose_in_link = in.pose_in_link;
    o.attach_link = attach;
    o.has_support_witness = in.has_support_witness;
    o.support_point = in.support_point;
    o.support_normal = support_normal;
    o.support_patch_radius = in.has_support_witness ? in.support_patch_radius : 0.0;
    o.support_max_penetration = in.has_support_witness ? in.support_max_penetration : 0.0;
    o.prim_first = static_cast<int>(prim_cursor);
    o.prim_count = static_cast<int>(n_prims);
    o.touch_first = static_cast<int>(touch_cursor);
    o.touch_count = static_cast<int>(n_touch);
    for (std::size_t p = 0; p < n_prims; ++p) {
      const AttachedPrimitiveInput& pin = in.primitives[p];
      // Shape sanity: dimensions must be finite and positive.
      if (pin.kind == AttachedShapeKind::kBox) {
        if (!(std::isfinite(pin.half_extents.x) && std::isfinite(pin.half_extents.y) &&
              std::isfinite(pin.half_extents.z)) ||
            pin.half_extents.x <= 0.0 || pin.half_extents.y <= 0.0 || pin.half_extents.z <= 0.0) {
          out.n_objects = 0;
          out.n_primitives = 0;
          return AttachIngestStatus::kMalformed;
        }
      } else {
        if (!std::isfinite(pin.radius) || pin.radius <= 0.0 || !std::isfinite(pin.half_length) ||
            pin.half_length < 0.0) {
          out.n_objects = 0;
          out.n_primitives = 0;
          return AttachIngestStatus::kMalformed;
        }
      }
      AttachedPrimitive& op = out.primitives[prim_cursor++];
      op.kind = pin.kind;
      op.radius = pin.radius;
      op.half_length = pin.half_length;
      op.half_extents = pin.half_extents;
      op.pose_in_object = pin.pose_in_object;
    }
    for (std::size_t t = 0; t < n_touch; ++t) {
      const int tl = resolve(in.touch_links[t]);
      if (tl < 0) {
        out.n_objects = 0;
        out.n_primitives = 0;
        return AttachIngestStatus::kUnknownLink;
      }
      out.touch_links[touch_cursor++] = tl;
    }
  }
  out.n_objects = n;
  out.n_primitives = prim_cursor;
  return AttachIngestStatus::kOk;
}

CollisionHit check_attached_world_collision(const CollisionModel& /*model*/,
                                            const AttachedModel& attached,
                                            const CollisionScratch& scratch,
                                            const WorldModel& world, double margin) noexcept {
  CollisionHit result;
  double sweep_min = std::numeric_limits<double>::infinity();
  const std::size_t n_world = world.capsules.size();
  for (std::size_t i = 0; i < attached.n_objects; ++i) {
    const AttachedObject& obj = attached.objects[i];
    const Transform obj_xf = attached_object_transform(obj, scratch);
    for (int p = 0; p < obj.prim_count; ++p) {
      const AttachedPrimitive& prim =
          attached.primitives[static_cast<std::size_t>(obj.prim_first + p)];
      const Transform prim_xf = compose(obj_xf, prim.pose_in_object);
      for (std::size_t w = 0; w < n_world; ++w) {
        const double d = attached_primitive_capsule_distance(
            prim, prim_xf, world.capsules[w].origin, world.capsules[w].radius,
            world.capsules[w].half_length);
        // link_a: attached object index (evidence is object-level, not
        // per-primitive); link_b: world obstacle index.
        fold_pair(result, sweep_min, d, d <= margin, static_cast<int>(i), static_cast<int>(w));
      }
    }
  }
  return finish_sweep(result, sweep_min);
}

CollisionHit check_attached_voxel_collision(const CollisionModel& /*model*/,
                                            const AttachedModel& attached,
                                            const CollisionScratch& scratch, const VoxelGrid& grid,
                                            double margin) noexcept {
  CollisionHit result;
  double sweep_min = std::numeric_limits<double>::infinity();
  if (grid.occupancy == nullptr || grid.sx <= 0 || grid.sy <= 0 || grid.sz <= 0 ||
      grid.resolution <= 0.0) {
    return finish_sweep(result, sweep_min);
  }
  const double half_side = grid.resolution * 0.5;
  const Vec3 voxel_half{half_side, half_side, half_side};
  for (std::size_t i = 0; i < attached.n_objects; ++i) {
    const AttachedObject& obj = attached.objects[i];
    const Transform obj_xf = attached_object_transform(obj, scratch);
    for (int p = 0; p < obj.prim_count; ++p) {
      const AttachedPrimitive& prim =
          attached.primitives[static_cast<std::size_t>(obj.prim_first + p)];
      const Transform prim_xf = compose(obj_xf, prim.pose_in_object);
      const CellBox box = primitive_cell_box(prim, prim_xf, grid, margin + half_side);
      for (int iz = box.z0; iz <= box.z1; ++iz) {
        for (int iy = box.y0; iy <= box.y1; ++iy) {
          for (int ix = box.x0; ix <= box.x1; ++ix) {
            const std::size_t idx = static_cast<std::size_t>(ix + grid.sx * (iy + grid.sy * iz));
            if (grid.occupancy[idx] == 0) {
              continue;
            }
            const Vec3 center = voxel_center(grid, ix, iy, iz);
            Transform voxel;
            voxel.t = center;
            const double d = attached_primitive_voxel_distance(prim, prim_xf, voxel, voxel_half);
            // Declaration-scoped approach allowance (ADR-0097's 2026-08-14
            // amendment): inside the declared target's region this payload's
            // margin — and only this payload's, and only for cells in that
            // region — is reduced by min(1.5 x voxel, 40 mm), enough to pass the
            // voxel inflation of a thin opening and actually reach the support
            // contact the place witness is earned by. The hard stop behind the
            // reduced margin is untouched.
            const double allowance = place_approach_allowance(grid, i, center);
            const double cell_margin = margin - allowance;
            // Gate: a cell within the (possibly reduced) margin trips unless one
            // of the two bounded exemptions explains it — a live support-contact
            // witness (physical, index-free, quantisation-aware) or the payload's
            // own embedded attach-time occupancy residue. An exempted cell
            // contributes to `sweep_min` only — it is not the reason for any
            // stop and must never supply the reported distance.
            bool tripped = false;
            if (d <= cell_margin) {
              tripped = true;
              const bool witness_live =
                  i < 8 && obj.has_support_witness && (grid.support_witness_live & (1U << i)) != 0;
              if (witness_live && support_contact_exempts(obj, obj_xf, center, grid.resolution,
                                                          grid.attached_contact_tolerance)) {
                tripped = false;
              } else if (i < 8 && grid.attached_contact_mask != nullptr &&
                         grid.attached_contact_distance != nullptr &&
                         grid.attached_contact_stride > idx &&
                         (grid.attached_contact_mask[idx] & (1U << i)) != 0 &&
                         grid.attached_contact_distance[i * grid.attached_contact_stride + idx] <=
                             -0.5 * grid.resolution) {
                // Embedded residue: this cell was already at least half a voxel
                // inside the payload when the baseline was snapshotted, so it
                // is the payload's own uncleared occupancy, not an obstacle.
                tripped = false;
              }
            }
            fold_pair(result, sweep_min, d, tripped, static_cast<int>(i), static_cast<int>(idx),
                      allowance > 0.0);
          }
        }
      }
    }
  }
  return finish_sweep(result, sweep_min);
}

double place_approach_allowance(const VoxelGrid& grid, std::size_t object_index,
                                const Vec3& center) noexcept {
  const PlaceApproachRegion& region = grid.place_region;
  if (!region.valid || grid.resolution <= 0.0 || object_index >= 8) {
    return 0.0;
  }
  if ((region.object_mask & (1U << object_index)) == 0) {
    return 0.0;
  }
  const Vec3& half = region.half_extents;
  if (!(half.x > 0.0) || !(half.y > 0.0) || !(half.z > 0.0) || !std::isfinite(half.x) ||
      !std::isfinite(half.y) || !std::isfinite(half.z)) {
    return 0.0;
  }
  // Exact point-in-OBB: express the cell centre in the region box's own frame
  // (Rᵀ·(centre - box origin)) and compare against the half-extents. An oriented
  // box is what the producer can actually measure — the axis-aligned hull of a
  // rotated receptacle would be strictly larger, i.e. strictly more permissive.
  const Vec3 d = sub(center, region.pose.t);
  const double* r = region.pose.r;
  const double lx = r[0] * d.x + r[3] * d.y + r[6] * d.z;
  const double ly = r[1] * d.x + r[4] * d.y + r[7] * d.z;
  const double lz = r[2] * d.x + r[5] * d.y + r[8] * d.z;
  if (std::fabs(lx) > half.x || std::fabs(ly) > half.y || std::fabs(lz) > half.z) {
    return 0.0;
  }
  // Condition 1 of the amendment (as calibrated by its Second Amendment,
  // 2026-08-15: min(1.5 x voxel, 4 cm)), evaluated against the LIVE resolution
  // so a map that coarsens mid-run can only shrink the allowance toward the
  // ceiling.
  return place_approach_allowance_cap(grid.resolution);
}

PlaceRegionStatus ingest_place_region(const Transform& pose, const Vec3& half_extents,
                                      std::uint8_t object_mask, PlaceApproachRegion& out) noexcept {
  out = PlaceApproachRegion{};
  if (object_mask == 0) {
    return PlaceRegionStatus::kNoObject;  // the declared payload is not carried
  }
  if (!std::isfinite(pose.t.x) || !std::isfinite(pose.t.y) || !std::isfinite(pose.t.z)) {
    return PlaceRegionStatus::kBadPose;
  }
  for (std::size_t k = 0; k < 9; ++k) {
    if (!std::isfinite(pose.r[k])) {
      return PlaceRegionStatus::kBadPose;
    }
  }
  const double hx = half_extents.x;
  const double hy = half_extents.y;
  const double hz = half_extents.z;
  if (!std::isfinite(hx) || !std::isfinite(hy) || !std::isfinite(hz)) {
    return PlaceRegionStatus::kBadExtents;
  }
  if (hx <= 0.0 || hy <= 0.0 || hz <= 0.0) {
    return PlaceRegionStatus::kDegenerate;  // no interior licenses nothing
  }
  if (hx > kMaxPlaceRegionHalfExtentM || hy > kMaxPlaceRegionHalfExtentM ||
      hz > kMaxPlaceRegionHalfExtentM) {
    return PlaceRegionStatus::kOversize;
  }
  if (8.0 * hx * hy * hz > kMaxPlaceRegionVolumeM3) {
    return PlaceRegionStatus::kOversizeVolume;
  }
  out.valid = true;
  out.object_mask = object_mask;
  out.pose = pose;
  out.half_extents = half_extents;
  return PlaceRegionStatus::kOk;
}

const char* place_region_status_reason(PlaceRegionStatus status) noexcept {
  switch (status) {
  case PlaceRegionStatus::kOk:
    return "ok";
  case PlaceRegionStatus::kNoObject:
    return "no_object";
  case PlaceRegionStatus::kBadPose:
    return "bad_pose";
  case PlaceRegionStatus::kBadExtents:
    return "bad_extents";
  case PlaceRegionStatus::kDegenerate:
    return "degenerate";
  case PlaceRegionStatus::kOversize:
    return "oversize";
  case PlaceRegionStatus::kOversizeVolume:
    return "oversize_volume";
  }
  return "unknown";
}

bool support_contact_exempts(const AttachedObject& object, const Transform& object_xf,
                             const Vec3& center, double resolution, double slack) noexcept {
  if (!object.has_support_witness || resolution <= 0.0) {
    return false;
  }
  // Lift the attested plane out of the object frame into the base frame. Doing
  // it every call is what keeps the witness index-free: the payload moves, the
  // base drives, the lattice re-phases, and the plane still describes the same
  // physical support face.
  const Vec3 plane_point = add(object_xf.t, rotate(object_xf.r, object.support_point));
  const Vec3 normal = rotate(object_xf.r, object.support_normal);
  const Vec3 delta = sub(center, plane_point);
  const double height = dot(delta, normal);
  const double lateral_sq = std::max(0.0, dot(delta, delta) - height * height);
  const double half = 0.5 * resolution;
  // Both pads are exact discretisation geometry, not tuned tolerances: the
  // cube's half-width projected on the normal, and the cube's circumradius
  // laterally.
  const double normal_half_width =
      half * (std::fabs(normal.x) + std::fabs(normal.y) + std::fabs(normal.z));
  const double lateral_pad = half * 1.7320508075688772;  // sqrt(3)
  const double patch = object.support_patch_radius + lateral_pad;
  if (lateral_sq > patch * patch) {
    return false;
  }
  // Fourth term: ONE VOXEL of co-planar headroom (hazard log Entry 012,
  // "Calibration 2026-08-15", the 5-run baguette battery; approved by the
  // maintainer alongside ADR-0097's Second Amendment). Cells of adjacent
  // co-planar structure — a raised edge, a neighbouring stack on the same
  // support surface — sit about one voxel above the attested plane while the
  // payload is in genuine, continuing contact; round-8 r2 measured +42.9 mm
  // against a ~15-19 mm envelope, an excess of one 25 mm voxel. This widens
  // HEIGHT only, and only inside the lateral patch bound above, which is
  // unchanged. Deepening past the widened bound still stops.
  return height <= normal_half_width + object.support_max_penetration + slack + resolution;
}

std::uint8_t update_support_contact_witnesses(const AttachedModel& attached,
                                              const CollisionScratch& scratch,
                                              const VoxelGrid& grid, std::uint8_t live_mask,
                                              double margin) noexcept {
  // No usable map means no way to confirm the payload is still on its support:
  // fail closed and drop every witness rather than carry a stale exemption.
  if (grid.occupancy == nullptr || grid.sx <= 0 || grid.sy <= 0 || grid.sz <= 0 ||
      grid.resolution <= 0.0) {
    return 0;
  }
  const std::size_t objects = std::min<std::size_t>(attached.n_objects, 8);
  std::uint8_t live = live_mask;
  for (std::size_t i = 0; i < 8; ++i) {
    const auto bit = static_cast<std::uint8_t>(1U << i);
    if ((live & bit) == 0) {
      continue;
    }
    // The witness survives only while it is still doing work. The moment the
    // payload lifts clear of its attested support, the exemption dies — and
    // only a fresh attestation can bring it back.
    const AttachedObject& obj = attached.objects[i];
    const bool keep = i < objects && obj.has_support_witness &&
                      support_witness_still_in_contact(
                          attached, obj, attached_object_transform(obj, scratch), grid, margin);
    if (!keep) {
      live = static_cast<std::uint8_t>(live & ~bit);
    }
  }
  return live;
}

bool update_attached_voxel_contacts(const AttachedModel& attached, const CollisionScratch& scratch,
                                    const VoxelGrid& grid, std::uint8_t* contact_mask,
                                    double* contact_distance, std::size_t mask_capacity,
                                    std::size_t distance_capacity, bool snapshot) noexcept {
  const std::size_t cells = static_cast<std::size_t>(grid.sx) * static_cast<std::size_t>(grid.sy) *
                            static_cast<std::size_t>(grid.sz);
  const std::size_t objects = std::min<std::size_t>(attached.n_objects, 8);
  if (contact_mask == nullptr || contact_distance == nullptr || grid.occupancy == nullptr ||
      cells == 0 || mask_capacity < cells || grid.attached_contact_stride < cells ||
      distance_capacity < grid.attached_contact_stride * objects || grid.resolution <= 0.0) {
    return false;
  }
  if (snapshot) {
    std::fill(contact_mask, contact_mask + cells, static_cast<std::uint8_t>(0));
    std::fill(contact_distance, contact_distance + distance_capacity,
              std::numeric_limits<double>::infinity());
  }
  const double half_side = grid.resolution * 0.5;
  const Vec3 voxel_half{half_side, half_side, half_side};
  bool active = false;
  for (std::size_t idx = 0; idx < cells; ++idx) {
    if (grid.occupancy[idx] == 0) {
      contact_mask[idx] = 0;
      continue;
    }
    const int ix = static_cast<int>(idx % static_cast<std::size_t>(grid.sx));
    const std::size_t yz = idx / static_cast<std::size_t>(grid.sx);
    const int iy = static_cast<int>(yz % static_cast<std::size_t>(grid.sy));
    const int iz = static_cast<int>(yz / static_cast<std::size_t>(grid.sy));
    Transform voxel;
    voxel.t = Vec3{grid.origin.x + (ix + 0.5) * grid.resolution,
                   grid.origin.y + (iy + 0.5) * grid.resolution,
                   grid.origin.z + (iz + 0.5) * grid.resolution};
    for (std::size_t object_index = 0; object_index < objects; ++object_index) {
      const std::uint8_t bit = static_cast<std::uint8_t>(1U << object_index);
      if (!snapshot && (contact_mask[idx] & bit) == 0) {
        continue;
      }
      const AttachedObject& object = attached.objects[object_index];
      const Transform object_xf = attached_object_transform(object, scratch);
      const double distance =
          attached_object_voxel_distance(attached, object, object_xf, voxel, voxel_half);
      const std::size_t distance_index = object_index * grid.attached_contact_stride + idx;
      if (snapshot) {
        if (distance <= 0.0) {
          contact_mask[idx] |= bit;
          contact_distance[distance_index] = distance;
          active = true;
        }
      } else if (distance > 0.0) {
        contact_mask[idx] &= static_cast<std::uint8_t>(~bit);
        contact_distance[distance_index] = std::numeric_limits<double>::infinity();
      } else {
        active = true;
      }
    }
  }
  return active;
}

CollisionHit check_attached_self_collision(const CollisionModel& model,
                                           const AttachedModel& attached,
                                           const CollisionScratch& scratch,
                                           double margin) noexcept {
  CollisionHit result;
  double sweep_min = std::numeric_limits<double>::infinity();
  const std::size_t n_caps = model.capsules.size();
  const std::size_t n_boxes = model.boxes.size();
  for (std::size_t i = 0; i < attached.n_objects; ++i) {
    const AttachedObject& obj = attached.objects[i];
    const Transform obj_xf = attached_object_transform(obj, scratch);
    for (int p = 0; p < obj.prim_count; ++p) {
      const AttachedPrimitive& prim =
          attached.primitives[static_cast<std::size_t>(obj.prim_first + p)];
      const Transform prim_xf = compose(obj_xf, prim.pose_in_object);
      for (std::size_t c = 0; c < n_caps; ++c) {
        const int lc = model.capsule_link[c];
        if (attached_allows_link(attached, obj, lc)) {
          continue;
        }
        const Transform cap_j =
            compose(scratch.link_world[static_cast<std::size_t>(lc)], model.capsules[c].origin);
        const double d = attached_primitive_capsule_distance(
            prim, prim_xf, cap_j, model.capsules[c].radius, model.capsules[c].half_length);
        fold_pair(result, sweep_min, d, d <= margin, static_cast<int>(i), lc);
      }
      for (std::size_t b = 0; b < n_boxes; ++b) {
        const int lb = model.box_link[b];
        if (attached_allows_link(attached, obj, lb)) {
          continue;
        }
        const Transform box_j =
            compose(scratch.link_world[static_cast<std::size_t>(lb)], model.boxes[b].origin);
        double d;
        if (prim.kind == AttachedShapeKind::kBox) {
          d = box_box_distance(prim_xf, prim.half_extents, box_j, model.boxes[b].half_extents);
        } else {
          d = box_capsule_distance(box_j, model.boxes[b].half_extents, prim_xf, prim.radius,
                                   prim.half_length);
        }
        fold_pair(result, sweep_min, d, d <= margin, static_cast<int>(i), lb);
      }
    }
  }
  return finish_sweep(result, sweep_min);
}

}  // namespace openral_safety_kernel
