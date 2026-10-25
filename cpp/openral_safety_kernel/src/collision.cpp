// SPDX-License-Identifier: Apache-2.0
// Allocation-free geometric self-collision core.
//
// Hand-rolled, dependency-free (no Eigen/KDL/Pinocchio) so the safety kernel
// stays small and auditable. All hot-path work is on the stack or in
// caller-owned pre-sized buffers; nothing here allocates (pinned by
// test_collision.cpp's counting-allocator test).

#include "openral_safety_kernel/collision.hpp"

#include "openral_safety_kernel/exemption_provider.hpp"

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

}  // namespace

// out = lhs * rhs (compose: apply rhs first, then lhs). Exported (ADR-0121 providers).
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

namespace {

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

TightGeometryStatus validate_tight_hull(const LinkHull& hull, const Vec3& half_extents,
                                        const std::vector<Vec3>& vertices) noexcept {
  const double hev[3] = {half_extents.x, half_extents.y, half_extents.z};
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
    // A solid has positive extent on its own three axes. Without this a wire
    // refinement of 13 zero lo and 13 zero hi passes every check above —
    // finite, not inverted, inside any box — and collapses the payload to a
    // POINT, after which `dop_cell_lower_bound` reports clearance the payload
    // does not have for every cell. That is reachable from an ordinary
    // producer bug (resize the arrays to 13, forget to fill them), not only
    // from a deliberate lie, and it disables the payload-vs-world check in
    // silence. Refusing it falls back to the box.
    if (!(hull.dop_hi[k] > hull.dop_lo[k])) {
      return TightGeometryStatus::kDegenerate;
    }
  }
  if (hull.vertex_count == 0) {
    return TightGeometryStatus::kOk;  // stage 1 only
  }
  if (hull.vertex_count < 0 || hull.vertex_count > kMaxTightHullVertices) {
    return TightGeometryStatus::kTooManyVertices;
  }
  if (hull.vertex_first < 0 ||
      static_cast<std::size_t>(hull.vertex_first) + static_cast<std::size_t>(hull.vertex_count) >
          vertices.size()) {
    return TightGeometryStatus::kBadArity;
  }
  for (int v = 0; v < hull.vertex_count; ++v) {
    const Vec3& p = vertices[static_cast<std::size_t>(hull.vertex_first + v)];
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
  return TightGeometryStatus::kOk;
}

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
    const TightGeometryStatus status = validate_tight_hull(
        model.hulls[static_cast<std::size_t>(h)], model.boxes[b].half_extents, model.hull_vertices);
    if (status != TightGeometryStatus::kOk) {
      return status;
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

double hull_hull_distance(const TightPose& a, const TightPose& b, double /*margin*/,
                          double fallback, bool* depth_is_box_bound) noexcept {
  if (depth_is_box_bound != nullptr) {
    *depth_is_box_bound = false;
  }
  if (a.vertices == nullptr || a.n_vertices <= 0 || b.vertices == nullptr || b.n_vertices <= 0) {
    return fallback;  // one side stops at stage 1; the OBB bound stands
  }
  // Seed pointing from a's box origin toward b's, so the first support pair
  // already straddles the gap. Any unit direction is admissible — the bound
  // below is sound whatever the search starts from — so a degenerate
  // (co-located) pair just picks an axis.
  Vec3 seed = sub(b.box.t, a.box.t);
  const double seed2 = dot(seed, seed);
  seed = seed2 > 1e-24 ? scale(seed, 1.0 / std::sqrt(seed2)) : Vec3{1.0, 0.0, 0.0};

  // The simplex lives in the Minkowski difference A (-) B; its support along
  // `d` is `support_A(d) - support_B(-d)`. `GjkSimplex`'s vertex/corner slots
  // exist for `hull_cell_distance`'s witness cache and are unused here.
  GjkSimplex s;
  Vec3 pa;
  Vec3 pb;
  hull_support(a, seed, pa);
  hull_support(b, Vec3{-seed.x, -seed.y, -seed.z}, pb);
  s.p[0] = sub(pa, pb);
  s.n = 1;

  double lower = -std::numeric_limits<double>::infinity();
  for (int it = 0; it < kGjkMaxIterations; ++it) {
    bool contains = false;
    const Vec3 v = simplex_closest(s, contains);
    const double vlen2 = dot(v, v);
    if (contains || vlen2 <= 1e-24) {
      // Hulls OVERLAP. GJK proves it but doesn't measure depth (needs an
      // expanding-polytope step this kernel doesn't run), so the OBB bound
      // (<= 0 here) stands as the reported number — sound (hull is contained
      // in its box, so the box can only report MORE penetration, never
      // less) but can be far looser: -31.97 mm for a ~1.5 mm hull
      // interpenetration on panda_link5<->panda_link7. Say so rather than
      // let the caller read a box bound as a hull measurement.
      if (depth_is_box_bound != nullptr) {
        *depth_is_box_bound = true;
      }
      return fallback;
    }
    const double vlen = std::sqrt(vlen2);
    const Vec3 search{-v.x / vlen, -v.y / vlen, -v.z / vlen};
    hull_support(a, search, pa);
    hull_support(b, Vec3{-search.x, -search.y, -search.z}, pb);
    const Vec3 w = sub(pa, pb);
    // Supporting-hyperplane bound: no point of the difference set lies closer
    // to the origin than this, so it lower-bounds the surface distance.
    const double bound = dot(v, w) / vlen;
    if (bound > lower) {
      lower = bound;
    }
    // No hull_cell_distance-style `lower > margin` early exit here: there
    // it's right (hundreds of cells visited per step, only the minimum
    // reported, a clear cell needs no sharper number); here the answer IS
    // the reported one (sweep_min_distance feeds distance-graded velocity
    // scaling), and stopping at the first positive bound was measured
    // returning 4.2 mm for a pair genuinely 60.0 mm apart — sound but would
    // crawl the arm past a clear pose. A self-pair is checked at most once
    // per configuration, only after the boxes already failed to clear it,
    // so converging costs a handful of scans.
    if (vlen - lower <= kGjkTolerance) {
      lower = vlen;  // converged on the exact surface distance
      break;
    }
    if (s.n == 4) {
      break;  // no room to grow; the bound already stands
    }
    s.p[s.n] = w;
    ++s.n;
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
      const JointKind kind = model.joint_kind[i];
      double q = qpos[static_cast<std::size_t>(dof)];
      if (is_mimic(kind) && i < model.mimic_multiplier.size() && i < model.mimic_offset.size()) {
        // A mimic link reads its leader's column: q = multiplier * q_leader + offset.
        q = model.mimic_multiplier[i] * q + model.mimic_offset[i];
      }
      if (kind == JointKind::kRevolute || kind == JointKind::kMimicRevolute) {
        axis_angle(model.axis[i], q, motion.r);
      } else if (kind == JointKind::kPrismatic || kind == JointKind::kMimicPrismatic) {
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
    if (is_mimic(model.joint_kind[li])) {
      // No column of its own: a mimic link rides its leader's column, which
      // this walk adds when it reaches the leader. A mimic on the EE chain
      // (no gripper finger is: both jaws are leaves) would need the chain
      // rule folded into that column; refused at configure instead.
      continue;
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
// sweep_min accumulates the minimum surface distance over EVERY pair
// checked, including pairs that never reached the margin and pairs the
// caller's gate exempted. tripped says whether this pair tripped the
// check; only a tripped pair can become the reported evidence, and only
// while deeper than the pair already recorded — keeping link_a/link_b/
// min_distance describing one and the same (deepest) tripping pair,
// never one cell's identity paired with another's distance.
//
// Reporting only: hit.hit flips for the same set of pairs as a plain
// `if (tripped) hit.hit = true;` would. allowance_active travels with the
// pair it belongs to, for the same reason: it must describe the reported
// pair, not whichever was checked last.
//
// Pairs are ranked SEVERITY FIRST, then depth. Depth alone is wrong once
// advisory exists (#176): the gate threshold differs per pair in a sweep
// (undeclared trips at the plain margin, declared trips only past the
// approach allowance), so the deepest tripping pair isn't always the most
// serious. A -5 mm hard trip on an undeclared payload and a -38 mm
// advisory reading inside a declared receptacle can occur in the same
// sweep; reporting the deeper one would hand the caller an advisory hit
// and lose the latch the -5 mm pair earned. A hard trip outranks an
// advisory one at any depth; within one severity, deepest still wins.
void fold_pair(CollisionHit& hit, double& sweep_min, double d, bool tripped, int link_a, int link_b,
               bool allowance_active = false, bool advisory = false,
               bool place_target_adjudicated = false, bool depth_is_box_bound = false,
               int workspace_denied = 0) noexcept {
  if (d < sweep_min) {
    sweep_min = d;
  }
  if (!tripped) {
    return;
  }
  const bool outranks_by_severity = hit.advisory && !advisory;
  const bool same_severity = hit.advisory == advisory;
  if (!hit.hit || outranks_by_severity || (same_severity && d < hit.min_distance)) {
    hit.hit = true;
    hit.link_a = link_a;
    hit.link_b = link_b;
    hit.min_distance = d;
    hit.place_allowance_active = allowance_active;
    hit.advisory = advisory;
    hit.place_target_adjudicated = place_target_adjudicated;
    hit.depth_is_box_bound = depth_is_box_bound;
    hit.workspace_denied = workspace_denied;
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

// Re-ask a self-collision box pair of the two links' exact convex hulls.
// Returns `fallback` (the caller's `box_box_distance`) untouched unless both
// links ship a stage-2 hull, so this is inert for every manifest that declares
// no `tight_geometry` and for every pair with a stage-1-only link.
//
// Called only once the OBBs are already within `margin`, so the GJK never runs
// for a pair the cheap bound clears.
double refine_self_pair(const CollisionModel& model, std::size_t bi, const Transform& xf_i,
                        std::size_t bj, const Transform& xf_j, double margin, double fallback,
                        bool* depth_is_box_bound) noexcept {
  if (depth_is_box_bound != nullptr) {
    *depth_is_box_bound = false;
  }
  if (model.box_hull.size() != model.boxes.size()) {
    return fallback;
  }
  const int hi = model.box_hull[bi];
  const int hj = model.box_hull[bj];
  if (hi < 0 || hj < 0 || static_cast<std::size_t>(hi) >= model.hulls.size() ||
      static_cast<std::size_t>(hj) >= model.hulls.size()) {
    return fallback;
  }
  const Vec3* verts = model.hull_vertices.empty() ? nullptr : model.hull_vertices.data();
  // `half_side` is the voxel-grid term; there is no cell here, so it is 0 and
  // the `cell_radius` entries it fills go unread by `hull_hull_distance`.
  TightPose pose_i;
  TightPose pose_j;
  tight_pose_init(model.hulls[static_cast<std::size_t>(hi)], verts, xf_i, 0.0, pose_i);
  tight_pose_init(model.hulls[static_cast<std::size_t>(hj)], verts, xf_j, 0.0, pose_j);
  return hull_hull_distance(pose_i, pose_j, margin, fallback, depth_is_box_bound);
}

}  // namespace

namespace {
bool workspace_self_exempt(const WorkspaceContactRegion* ws, const CollisionScratch& scratch, int a,
                           int b) noexcept;
// Fold one self pair, exempting a slow hand-hand pair inside the workspace (#349).
void fold_self_pair(CollisionHit& result, double& sweep_min, double d, double margin, int a, int b,
                    const WorkspaceContactRegion* ws, const CollisionScratch& scratch,
                    const ProviderGate* gate, bool depth_is_box_bound = false) noexcept;
}  // namespace

CollisionHit check_self_collision(const CollisionModel& model, const CollisionScratch& scratch,
                                  double margin, const WorkspaceContactRegion* workspace,
                                  const ProviderGate* gate) noexcept {
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
      fold_self_pair(result, sweep_min, d, margin, li, lj, workspace, scratch, gate);
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
      fold_self_pair(result, sweep_min, d, margin, lb, lc, workspace, scratch, gate);
    }
    for (std::size_t bj = bi + 1; bj < n_boxes; ++bj) {
      const int lb2 = model.box_link[bj];
      if (lb == lb2 || is_allowed(model, lb, lb2)) {
        continue;
      }
      const Transform box_j =
          compose(scratch.link_world[static_cast<std::size_t>(lb2)], model.boxes[bj].origin);
      double d = box_box_distance(box_i, model.boxes[bi].half_extents, box_j,
                                  model.boxes[bj].half_extents);
      bool depth_is_box_bound = false;
      if (d <= margin) {
        // The OBBs cannot separate every pair that interleaves (see
        // `hull_hull_distance`). Refine with the exact hulls when both links
        // ship one; otherwise `d` comes back unchanged.
        d = refine_self_pair(model, bi, box_i, bj, box_j, margin, d, &depth_is_box_bound);
      }
      fold_self_pair(result, sweep_min, d, margin, lb, lb2, workspace, scratch, gate,
                     depth_is_box_bound);
    }
  }
  return finish_sweep(result, sweep_min);
}

// ---------------------------------------------------------------------------
// Working in GRID coordinates
//
// The grid's lattice is the source map's, not base_frame's (see
// VoxelGrid::pose). Every routine that walks a grid carries its query
// geometry into the grid's axes ONCE; from there cells are axis-aligned
// cubes at voxel_center_local exactly as before.
//
// Not merely tidier: the staged tight-geometry path REQUIRES it.
// dop_cell_lower_bound measures a cell against three world-axis slabs using
// half_side directly, only valid if the cell is axis-aligned in the frame
// tight_pose_init was given.
//
// Distances are rigid-motion invariant, so the numbers are unchanged; with
// an identity grid.pose.r every routine reduces exactly to what it computed
// before this became an oriented grid.
// ---------------------------------------------------------------------------

/// Inverse of a rigid transform: (R^T, -R^T t).
Transform inverse_rigid(const Transform& x) noexcept {
  Transform out;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      out.r[i * 3 + j] = x.r[j * 3 + i];
    }
  }
  out.t = Vec3{-(out.r[0] * x.t.x + out.r[1] * x.t.y + out.r[2] * x.t.z),
               -(out.r[3] * x.t.x + out.r[4] * x.t.y + out.r[5] * x.t.z),
               -(out.r[6] * x.t.x + out.r[7] * x.t.y + out.r[8] * x.t.z)};
  return out;
}

/// Apply a transform to a point.
Vec3 apply(const Transform& x, const Vec3& p) noexcept {
  return Vec3{x.t.x + x.r[0] * p.x + x.r[1] * p.y + x.r[2] * p.z,
              x.t.y + x.r[3] * p.x + x.r[4] * p.y + x.r[5] * p.z,
              x.t.z + x.r[6] * p.x + x.r[7] * p.y + x.r[8] * p.z};
}

/// Centre of cell (ix, iy, iz) in the GRID's own axes, where the lattice is
/// axis-aligned and its origin is 0 by construction.
Vec3 voxel_center_local(const VoxelGrid& grid, int ix, int iy, int iz) noexcept {
  return Vec3{(ix + 0.5) * grid.resolution, (iy + 0.5) * grid.resolution,
              (iz + 0.5) * grid.resolution};
}

/// Base-frame centre of cell (ix, iy, iz), for consumers comparing a cell
/// against something measured in `base_frame`.
Vec3 voxel_center(const VoxelGrid& grid, int ix, int iy, int iz) noexcept {
  return apply(grid.pose, voxel_center_local(grid, ix, iy, iz));
}

/// Half-extents, along the grid's axes, of an oriented box already expressed in
/// those axes: `|R| * he`, the standard OBB-AABB bound, for sizing a window.
Vec3 obb_extent(const Transform& box, const Vec3& he) noexcept {
  return Vec3{std::fabs(box.r[0]) * he.x + std::fabs(box.r[1]) * he.y + std::fabs(box.r[2]) * he.z,
              std::fabs(box.r[3]) * he.x + std::fabs(box.r[4]) * he.y + std::fabs(box.r[5]) * he.z,
              std::fabs(box.r[6]) * he.x + std::fabs(box.r[7]) * he.y + std::fabs(box.r[8]) * he.z};
}

// Exact point-in-OBB: express `p` in the box's own frame (Rᵀ·(p - origin))
// and compare against the half-extents. A degenerate or non-finite box
// contains nothing. Shared by the place allowance and the grasp exemption so
// the two declaration-scoped regions can never disagree on "inside".
bool point_in_obb(const Transform& box, const Vec3& half, const Vec3& p) noexcept {
  if (!(half.x > 0.0) || !(half.y > 0.0) || !(half.z > 0.0) || !std::isfinite(half.x) ||
      !std::isfinite(half.y) || !std::isfinite(half.z)) {
    return false;
  }
  const Vec3 d = sub(p, box.t);
  const double lx = box.r[0] * d.x + box.r[3] * d.y + box.r[6] * d.z;
  const double ly = box.r[1] * d.x + box.r[4] * d.y + box.r[7] * d.z;
  const double lz = box.r[2] * d.x + box.r[5] * d.y + box.r[8] * d.z;
  // `<=`, so a non-finite point is never inside (fails closed).
  return std::fabs(lx) <= half.x && std::fabs(ly) <= half.y && std::fabs(lz) <= half.z;
}

namespace {

// Is `link` one the live grasp region may exempt at all? Evaluated once per
// capsule/box, outside the cell loop, so a link outside the mask pays nothing.
bool grasp_mask_has(const GraspTargetRegion& region, int link) noexcept {
  return region.valid && link >= 0 && static_cast<std::size_t>(link) < kMaxGraspMaskLinks &&
         region.link_mask[static_cast<std::size_t>(link)];
}

}  // namespace

bool grasp_region_contains(const GraspTargetRegion& region, const Vec3& p) noexcept {
  return region.valid && point_in_obb(region.pose, region.half_extents, p);
}

bool grasp_target_exempts(const VoxelGrid& grid, int link_index, const Vec3& center) noexcept {
  return grasp_mask_has(grid.grasp_region, link_index) &&
         grasp_region_contains(grid.grasp_region, center);
}

namespace {

// The grasp exemption box a link may be exempt against in this check, or
// nullptr (issue #418 K2). With a valid tracked table it is the ONLY source:
// the selected box of the hand the link is a finger of. Otherwise the legacy
// region, for a link in its mask — exactly `grasp_target_exempts`.
struct GraspBox {
  const Transform* pose{nullptr};
  const Vec3* half{nullptr};
};

GraspBox grasp_box_for(const VoxelGrid& grid, const std::array<int, kMaxGraspHands>& selected,
                       int link) noexcept {
  const TrackedGraspTable* t = grid.tracked_grasp;
  if (t != nullptr && t->valid) {
    if (link < 0 || static_cast<std::size_t>(link) >= kMaxGraspMaskLinks) {
      return {};
    }
    for (std::size_t h = 0; h < t->n_hands && h < kMaxGraspHands; ++h) {
      if (t->hand_links[h][static_cast<std::size_t>(link)] && selected[h] >= 0) {
        const auto i = static_cast<std::size_t>(selected[h]);
        return {&t->box[i], &t->half_extents[i]};
      }
    }
    return {};
  }
  if (grasp_mask_has(grid.grasp_region, link)) {
    return {&grid.grasp_region.pose, &grid.grasp_region.half_extents};
  }
  return {};
}

// #418 K2 (hazard OT-6 amendment 2026-10-17; Safety-WG review pending): is the cell exempt
// against the selected hand's box? A voxel is occupied when ANY part of it touches the object,
// so with a tracked table a cell is the target's own when its VOLUME intersects the box (exact
// OBB-vs-cube separating-axis test), not only when its centre lies inside: widening K2 by at
// most half a cell (more on the diagonal), the same quantisation argument as the K7 static
// pad. The legacy region keeps the centre test. A non-finite box or cell never exempts.
bool grasp_cell_exempt(const VoxelGrid& grid, const GraspBox& gbox, const Transform& grid_from_base,
                       const Vec3& local_center, const Vec3& voxel_half,
                       const Vec3& base_center) noexcept {
  if (gbox.pose == nullptr) {
    return false;
  }
  const TrackedGraspTable* t = grid.tracked_grasp;
  if (t == nullptr || !t->valid) {
    return point_in_obb(*gbox.pose, *gbox.half, base_center);
  }
  const Vec3& h = *gbox.half;
  if (!(h.x > 0.0) || !(h.y > 0.0) || !(h.z > 0.0) || !std::isfinite(h.x) || !std::isfinite(h.y) ||
      !std::isfinite(h.z) || !std::isfinite(local_center.x) || !std::isfinite(local_center.y) ||
      !std::isfinite(local_center.z)) {
    return false;
  }
  Transform cell;
  cell.t = local_center;
  const double gap = box_box_distance(compose(grid_from_base, *gbox.pose), h, cell, voxel_half);
  return std::isfinite(gap) && gap <= 0.0;
}
}  // namespace

namespace {

// Is `link` a hand link of a live workspace whose hand is slow and close to its
// measured pose (speed AND lead within their bounds)? Evaluated once per
// capsule/box, outside the cell loop. A NaN/inf speed or lead (the lifecycle's
// "unknown") or an unsound bound never passes.
bool workspace_link_slow(const WorkspaceContactRegion& ws, int link) noexcept {
  return ws.valid && link >= 0 && static_cast<std::size_t>(link) < kMaxGraspMaskLinks &&
         ws.link_mask[static_cast<std::size_t>(link)] && std::isfinite(ws.max_speed_mps) &&
         ws.max_speed_mps > 0.0 && std::isfinite(ws.hand_speed_mps) &&
         ws.hand_speed_mps <= ws.max_speed_mps && std::isfinite(ws.max_lead_m) &&
         ws.max_lead_m > 0.0 && std::isfinite(ws.hand_lead_m) && ws.hand_lead_m <= ws.max_lead_m;
}

// ADR-0121: may the loaded provider be asked about this link at all? The kernel's own mask, never
// the provider's word.
bool provider_relaxable(const ProviderGate& gate, int link) noexcept {
  return gate.provider != nullptr && gate.links != nullptr && link >= 0 &&
         static_cast<std::size_t>(link) < gate.links_n &&
         gate.links[static_cast<std::size_t>(link)] != 0;
}

// ADR-0121: disclose one pair the provider tolerated (`other` is a cell index or, for a self or
// payload-self pair, the other link).
void note_provider_pair(CollisionHit& hit, double d, int link, int other) noexcept {
  ++hit.provider_pairs;
  if (d < hit.provider_min_distance) {
    hit.provider_min_distance = d;
    hit.provider_link = link;
    hit.provider_cell = other;
  }
}

// ADR-0121: ask the provider about one robot-link-vs-cell pair that would trip.
bool provider_tolerates(CollisionHit& hit, const VoxelGrid& grid, const CollisionScratch& scratch,
                        int link, std::size_t cell, const Vec3& cell_local, double d) noexcept {
  const ProviderGate& gate = grid.provider_gate;
  VoxelPairQuery q;
  q.link = link;
  q.cell = cell;
  q.cell_center_base = apply(grid.pose, cell_local);
  q.distance = d;
  q.row = gate.row;
  q.row_fk = &scratch;
  q.hand_speed_mps = gate.hand_speed_mps;
  q.hand_lead_m = gate.hand_lead_m;
  q.hand_contact = gate.hand_contact;
  if (gate.provider->voxel_pair(q) != ProviderVerdict::kTolerate) {
    return false;
  }
  note_provider_pair(hit, d, link, static_cast<int>(cell));
  return true;
}

// Disclose one tolerated workspace pair (within the margin, exempted).
void note_workspace_pair(CollisionHit& hit, double d, int link, int cell) noexcept {
  ++hit.workspace_pairs;
  if (d < hit.workspace_min_distance) {
    hit.workspace_min_distance = d;
    hit.workspace_link = link;
    hit.workspace_cell = cell;
  }
}

bool workspace_self_exempt(const WorkspaceContactRegion* ws, const CollisionScratch& scratch, int a,
                           int b) noexcept {
  if (ws == nullptr || !workspace_link_slow(*ws, a) || !workspace_link_slow(*ws, b)) {
    return false;
  }
  const auto inside = [&](int l) {
    return static_cast<std::size_t>(l) < scratch.link_world.size() &&
           point_in_obb(ws->pose, ws->half_extents,
                        scratch.link_world[static_cast<std::size_t>(l)].t);
  };
  return inside(a) && inside(b);
}

void fold_self_pair(CollisionHit& result, double& sweep_min, double d, double margin, int a, int b,
                    const WorkspaceContactRegion* ws, const CollisionScratch& scratch,
                    const ProviderGate* gate, bool depth_is_box_bound) noexcept {
  const bool exempt = d <= margin && workspace_self_exempt(ws, scratch, a, b);
  if (exempt) {
    note_workspace_pair(result, d, a, b);
  }
  // ADR-0121: last, and only when BOTH links are in the kernel's mask.
  bool provider_exempt = false;
  if (!exempt && d <= margin && gate != nullptr && provider_relaxable(*gate, a) &&
      provider_relaxable(*gate, b)) {
    SelfPairQuery q;
    q.link_a = a;
    q.link_b = b;
    q.distance = d;
    q.row = gate->row;
    q.row_fk = &scratch;
    q.hand_speed_mps = gate->hand_speed_mps;
    q.hand_lead_m = gate->hand_lead_m;
    q.hand_contact = gate->hand_contact;
    provider_exempt = gate->provider->self_pair(q) == ProviderVerdict::kTolerate;
    if (provider_exempt) {
      note_provider_pair(result, d, a, b);
    }
  }
  fold_pair(result, sweep_min, d, d <= margin && !exempt && !provider_exempt, a, b,
            /*allowance_active=*/false, /*advisory=*/false, /*place_target_adjudicated=*/false,
            depth_is_box_bound);
}

}  // namespace

bool workspace_contact_exempts(const VoxelGrid& grid, int link_index, const Vec3& center) noexcept {
  return workspace_link_slow(grid.workspace_region, link_index) &&
         point_in_obb(grid.workspace_region.pose, grid.workspace_region.half_extents, center) &&
         // #418 K7: a new conjunct only — an unexplained cell trips as outside.
         (grid.workspace_explained == nullptr || workspace_cell_explained(grid, center));
}

namespace {

// Why a live workspace does not tolerate `link` against a cell centred at `center`
// (base frame): see `CollisionHit::workspace_denied`. 0 when it would (or no live
// workspace).
int workspace_denial(const VoxelGrid& grid, int link, const Vec3& center) noexcept {
  const WorkspaceContactRegion& ws = grid.workspace_region;
  if (!ws.valid) {
    return 0;
  }
  if (link < 0 || static_cast<std::size_t>(link) >= kMaxGraspMaskLinks ||
      !ws.link_mask[static_cast<std::size_t>(link)]) {
    return 1;
  }
  if (!std::isfinite(ws.hand_speed_mps) || ws.hand_speed_mps > ws.max_speed_mps) {
    return 2;
  }
  if (!std::isfinite(ws.hand_lead_m) || ws.hand_lead_m > ws.max_lead_m) {
    return 3;
  }
  if (!point_in_obb(ws.pose, ws.half_extents, center)) {
    return 4;
  }
  if (grid.workspace_explained != nullptr && !workspace_cell_explained(grid, center)) {
    return 5;
  }
  return 0;
}

}  // namespace

double link_geometry_reach(const CollisionModel& model, int link) noexcept {
  double reach = 0.0;
  const auto norm = [](const Vec3& v) { return std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z); };
  for (std::size_t c = 0; c < model.capsules.size() && c < model.capsule_link.size(); ++c) {
    if (model.capsule_link[c] == link) {
      const Capsule& cap = model.capsules[c];
      reach = std::max(reach, norm(cap.origin.t) + cap.half_length + cap.radius);
    }
  }
  for (std::size_t b = 0; b < model.boxes.size() && b < model.box_link.size(); ++b) {
    if (model.box_link[b] == link) {
      const Obb& box = model.boxes[b];
      reach = std::max(reach, norm(box.origin.t) + norm(box.half_extents));
    }
  }
  return reach;
}

double attached_object_reach(const AttachedModel& attached, const AttachedObject& object) noexcept {
  double reach = 0.0;
  for (int p = 0; p < object.prim_count; ++p) {
    const auto k = static_cast<std::size_t>(object.prim_first + p);
    if (k >= attached.primitives.size()) {
      return std::numeric_limits<double>::infinity();  // a malformed model reads unbounded
    }
    const AttachedPrimitive& prim = attached.primitives[k];
    const Vec3 o = compose(object.pose_in_link, prim.pose_in_object).t;
    const Vec3& h = prim.half_extents;
    const double bound = prim.kind == AttachedShapeKind::kBox
                             ? std::sqrt(h.x * h.x + h.y * h.y + h.z * h.z)
                             : prim.half_length + prim.radius;
    reach = std::max(reach, std::sqrt(o.x * o.x + o.y * o.y + o.z * o.z) + bound);
  }
  return std::isfinite(reach) ? reach : std::numeric_limits<double>::infinity();
}

double link_displacement_bound(const Transform& from, const Transform& to, double reach) noexcept {
  const double dx = to.t.x - from.t.x;
  const double dy = to.t.y - from.t.y;
  const double dz = to.t.z - from.t.z;
  // trace(R0^T R1) = sum_ij R0_ij R1_ij (row-major 3x3).
  double trace = 0.0;
  for (std::size_t k = 0; k < 9; ++k) {
    trace += from.r[k] * to.r[k];
  }
  const double c = std::clamp((trace - 1.0) * 0.5, -1.0, 1.0);
  const double bound = std::sqrt(dx * dx + dy * dy + dz * dz) + std::acos(c) * reach;
  return std::isfinite(bound) && std::isfinite(trace) && std::isfinite(reach)
             ? bound
             : std::numeric_limits<double>::infinity();
}

CollisionHit check_voxel_collision(const CollisionModel& model, const CollisionScratch& scratch,
                                   const VoxelGrid& grid, double margin, double band_m) noexcept {
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
  // Voxel-index span of an interval ALONG THE GRID'S OWN AXES, clamped to the
  // grid (shared by the capsule and box passes). Callers hand it grid-axis
  // coordinates via `to_grid_axes`; the grid's origin is 0 there by
  // construction.
  const auto rng = [&](double lo, double hi, int dim) {
    const int i0 = static_cast<int>(std::floor(lo * inv_res));
    const int i1 = static_cast<int>(std::floor(hi * inv_res));
    return std::pair<int, int>{clamp_index(i0, 0, dim - 1), clamp_index(i1, 0, dim - 1)};
  };
  // Every link is carried into the grid's axes through this.
  const Transform grid_from_base = inverse_rigid(grid.pose);
  // #418 K2: each hand's selected tracked object at THIS configuration.
  std::array<int, kMaxGraspHands> grasp_selected{};
  grasp_selected.fill(-1);
  if (grid.tracked_grasp != nullptr && grid.tracked_grasp->valid) {
    select_tracked_grasp(*grid.tracked_grasp, scratch, grid.resolution, grasp_selected);
  }
  const std::size_t n_caps = model.capsules.size();
  for (std::size_t c = 0; c < n_caps; ++c) {
    const int li = model.capsule_link[c];
    const Transform cap =
        compose(grid_from_base, compose(scratch.link_world[static_cast<std::size_t>(li)],
                                        model.capsules[c].origin));
    Vec3 p0;
    Vec3 p1;
    capsule_endpoints(cap, model.capsules[c].half_length, p0, p1);
    const double r = model.capsules[c].radius;
    const GraspBox gbox = grasp_box_for(grid, grasp_selected, li);
    const bool grasp_link = gbox.pose != nullptr;
    const bool workspace_link = workspace_link_slow(grid.workspace_region, li);
    // `band_m` widens the WINDOW only; `d <= margin` below is still the trip.
    const double reach = r + margin + band_m + half_side;
    const auto [ix0, ix1] =
        rng(std::min(p0.x, p1.x) - reach, std::max(p0.x, p1.x) + reach, grid.sx);
    const auto [iy0, iy1] =
        rng(std::min(p0.y, p1.y) - reach, std::max(p0.y, p1.y) + reach, grid.sy);
    const auto [iz0, iz1] =
        rng(std::min(p0.z, p1.z) - reach, std::max(p0.z, p1.z) + reach, grid.sz);
    for (int iz = iz0; iz <= iz1; ++iz) {
      for (int iy = iy0; iy <= iy1; ++iy) {
        for (int ix = ix0; ix <= ix1; ++ix) {
          const std::size_t idx = static_cast<std::size_t>(ix + grid.sx * (iy + grid.sy * iz));
          if (grid.occupancy[idx] == 0) {
            continue;
          }
          Transform voxel;
          voxel.t = voxel_center_local(grid, ix, iy, iz);
          const double d =
              box_capsule_distance(voxel, voxel_half, cap, r, model.capsules[c].half_length);
          // ADR-0115: a declared contact link vs a cell centred in the grasp
          // region never trips and never supplies the reported evidence.
          const Vec3 base_center =
              grasp_link || workspace_link ? apply(grid.pose, voxel.t) : Vec3{};
          const bool exempt = grasp_link && grasp_cell_exempt(grid, gbox, grid_from_base, voxel.t,
                                                              voxel_half, base_center);
          // #349: a slow hand link vs a cell centred in the live workspace.
          // Never silent: a tolerated pair within the margin is counted.
          const bool ws_exempt =
              !exempt && workspace_link && workspace_contact_exempts(grid, li, base_center);
          if (ws_exempt && d <= margin) {
            note_workspace_pair(result, d, li, static_cast<int>(idx));
          }
          // ADR-0121: last, and only for a pair still about to trip on a relaxable link.
          const bool provider_exempt =
              !exempt && !ws_exempt && d <= margin && provider_relaxable(grid.provider_gate, li) &&
              provider_tolerates(result, grid, scratch, li, idx, voxel.t, d);
          const bool tripped = !exempt && !ws_exempt && !provider_exempt && d <= margin;
          fold_pair(result, sweep_min, d, tripped, li, static_cast<int>(idx), false, false, false,
                    false, tripped ? workspace_denial(grid, li, base_center) : 0);
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
    // In the GRID's axes, so the cells stay axis-aligned cubes — which the
    // staged tight path below requires, not merely prefers.
    const Transform box_w =
        compose(grid_from_base,
                compose(scratch.link_world[static_cast<std::size_t>(lb)], model.boxes[b].origin));
    const Vec3 he = model.boxes[b].half_extents;
    const int hull_index = has_tight ? model.box_hull[b] : -1;
    const GraspBox gbox = grasp_box_for(grid, grasp_selected, lb);
    const bool grasp_link = gbox.pose != nullptr;
    const bool workspace_link = workspace_link_slow(grid.workspace_region, lb);
    TightPose tight;
    GjkWitness witness;
    if (hull_index >= 0) {
      tight_pose_init(model.hulls[static_cast<std::size_t>(hull_index)],
                      model.hull_vertices.empty() ? nullptr : model.hull_vertices.data(), box_w,
                      half_side, tight);
    }
    // Bounding half-size of the oriented box on the grid's axes (`box_w` is
    // already expressed in them), which is what sizes an index window.
    const Vec3 ext = obb_extent(box_w, he);
    const double reach = margin + band_m + half_side;
    const auto [ix0, ix1] = rng(box_w.t.x - ext.x - reach, box_w.t.x + ext.x + reach, grid.sx);
    const auto [iy0, iy1] = rng(box_w.t.y - ext.y - reach, box_w.t.y + ext.y + reach, grid.sy);
    const auto [iz0, iz1] = rng(box_w.t.z - ext.z - reach, box_w.t.z + ext.z + reach, grid.sz);
    for (int iz = iz0; iz <= iz1; ++iz) {
      for (int iy = iy0; iy <= iy1; ++iy) {
        for (int ix = ix0; ix <= ix1; ++ix) {
          const std::size_t idx = static_cast<std::size_t>(ix + grid.sx * (iy + grid.sy * iz));
          if (grid.occupancy[idx] == 0) {
            continue;
          }
          const Vec3 center = voxel_center_local(grid, ix, iy, iz);
          // ADR-0115, as in the capsule pass. Decided BEFORE the narrow phase:
          // an exempt pair cannot trip whatever its exact distance, so it must
          // not spend the stage-2 budget the non-exempt pairs of this call
          // share — a finger buried in its target puts dozens of in-region
          // cells inside stage 1's margin, enough to exhaust the budget and
          // leave a non-exempt link on the looser fallback bound (a false
          // stop). Stage 1's lower bound is all its clamped slack needs.
          // #349's workspace exemption joins it here, under the same accounting.
          const Vec3 base_center = grasp_link || workspace_link ? apply(grid.pose, center) : Vec3{};
          const bool grasp_exempt =
              grasp_link &&
              grasp_cell_exempt(grid, gbox, grid_from_base, center, voxel_half, base_center);
          const bool ws_exempt =
              !grasp_exempt && workspace_link && workspace_contact_exempts(grid, lb, base_center);
          const bool exempt = grasp_exempt || ws_exempt;
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
              if (!exempt && tight.n_vertices > 0 && stage2_budget > 0) {
                --stage2_budget;
                d = hull_cell_distance(tight, center, half_side, seed, margin, d, witness);
              }
              if (d <= margin) {
                // Still stopping. Fold in the shipped bound first: the DOP's
                // 16 separating axes are NOT a superset of the OBB SAT's 15 —
                // the box's edge-cross axes beat every DOP axis at some
                // cells, so the tighter SOLID can still yield the looser
                // BOUND. Max of two lower bounds is a lower bound, so this
                // costs no soundness and guarantees the staged path can
                // never stop where box_box_distance alone would not have —
                // this change removes false stops, it must not add one.
                Transform voxel;
                voxel.t = center;
                const double shipped = box_box_distance(box_w, he, voxel, voxel_half);
                if (shipped > d) {
                  d = shipped;
                }
              }
            }
          }
          if (ws_exempt && d <= margin) {
            note_workspace_pair(result, d, lb, static_cast<int>(idx));
          }
          // ADR-0121: after the narrow phase, so the provider sees the distance that would trip.
          const bool provider_exempt =
              !exempt && d <= margin && provider_relaxable(grid.provider_gate, lb) &&
              provider_tolerates(result, grid, scratch, lb, idx, center, d);
          const bool tripped = !exempt && !provider_exempt && d <= margin;
          fold_pair(result, sweep_min, d, tripped, lb, static_cast<int>(idx), false, false, false,
                    false, tripped ? workspace_denial(grid, lb, base_center) : 0);
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
  if (obj.attach_link < 0) {
    return obj.pose_in_link;  // #418 K5: frozen in the base frame
  }
  return compose(scratch.link_world[static_cast<std::size_t>(obj.attach_link)], obj.pose_in_link);
}

// True when robot link `link` is the object's attach link or one of its
// explicit touch links (grasped payloads legitimately contact those).
bool attached_allows_link(const AttachedModel& attached, const AttachedObject& obj,
                          int link) noexcept {
  if (link >= 0 && (link == obj.attach_link || link == obj.release_hand_link)) {
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

/// Index window for a primitive. `prim_xf` MUST already be expressed in the
/// grid's axes (`compose(inverse_rigid(grid.pose), base_frame_pose)`); the
/// lattice is axis-aligned and origin-0 there.
CellBox primitive_cell_box(const AttachedPrimitive& prim, const Transform& prim_xf,
                           const VoxelGrid& grid, double reach) noexcept {
  // Bounding half-size of the primitive on the grid's axes (|R| * extents). For
  // a sphere/capsule the swept radius plus the endpoint span is conservative.
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
  const auto rng = [inv_res](double lo, double hi, int dim) {
    const int i0 = static_cast<int>(std::floor(lo * inv_res));
    const int i1 = static_cast<int>(std::floor(hi * inv_res));
    return std::pair<int, int>{clamp_index(i0, 0, dim - 1), clamp_index(i1, 0, dim - 1)};
  };
  const auto [x0, x1] = rng(prim_xf.t.x - ex - reach, prim_xf.t.x + ex + reach, grid.sx);
  const auto [y0, y1] = rng(prim_xf.t.y - ey - reach, prim_xf.t.y + ey + reach, grid.sy);
  const auto [z0, z1] = rng(prim_xf.t.z - ez - reach, prim_xf.t.z + ez + reach, grid.sz);
  return CellBox{x0, x1, y0, y1, z0, z1};
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
  const Transform grid_from_base = inverse_rigid(grid.pose);
  for (int p = 0; p < obj.prim_count; ++p) {
    const AttachedPrimitive& prim =
        attached.primitives[static_cast<std::size_t>(obj.prim_first + p)];
    const Transform prim_xf = compose(obj_xf, prim.pose_in_object);
    // Windowing and the cell distance run in the grid's axes; the support
    // attestation is measured in `base_frame` and keeps its own centre.
    const Transform prim_g = compose(grid_from_base, prim_xf);
    const CellBox box = primitive_cell_box(prim, prim_g, grid, margin + half_side);
    for (int iz = box.z0; iz <= box.z1; ++iz) {
      for (int iy = box.y0; iy <= box.y1; ++iy) {
        for (int ix = box.x0; ix <= box.x1; ++ix) {
          const std::size_t idx = static_cast<std::size_t>(ix + grid.sx * (iy + grid.sy * iz));
          if (grid.occupancy[idx] == 0) {
            continue;
          }
          if (!support_contact_exempts(obj, obj_xf, voxel_center(grid, ix, iy, iz), grid.resolution,
                                       grid.attached_contact_tolerance)) {
            continue;
          }
          Transform voxel;
          voxel.t = voxel_center_local(grid, ix, iy, iz);
          if (attached_primitive_voxel_distance(prim, prim_g, voxel, voxel_half) <= margin) {
            return true;
          }
        }
      }
    }
  }
  return false;
}

}  // namespace

namespace {

// Copy one wire refinement into `out`'s hull store IF it proves contained in
// the primitive's own box, returning the hull index or -1.
//
// Fail-DOWN rather than fail-closed, and deliberately: unlike a robot's link
// hulls — fixed, shipped, and validated once at configure time — a payload's
// arrives on every world-state message from a live producer. Refusing the
// whole attachment over an unprovable refinement would drop the payload's
// geometry entirely, which is the UNSAFE direction; dropping only the
// refinement leaves the primitive checked as the plain box, i.e. exactly
// today's behaviour.
//
// What this proof does and does NOT buy, stated precisely because the
// tempting summary is wrong. It proves the refinement is a subset of the box,
// so the broad-phase window — sized from `half_extents` alone — still visits
// every cell the narrow phase might need. It does NOT prove the refinement
// bounds the payload's real geometry, and it cannot: the kernel never sees a
// mesh. A producer that publishes 13 in-box-but-fictional slabs shrinks its
// own payload and loses stops it should have had.
//
// That is not a trust boundary this adds. `half_extents` is producer-supplied
// too, checked only for finite-and-positive, so the same producer could
// already under-report the payload by publishing a 1 um box. The refinement
// travels under exactly the trust its own box already has, and the mesh-side
// half of the containment chain is discharged where the mesh exists — in
// `_tight_geometry_from_points`, against the payload's own vertices.
//
// `slot` is the primitive's own index, so a hull's storage is a pure function
// of the slot it refines: the two buffers are pre-sized to the primitive cap
// at configure time and no cursor can outrun them.
int accept_tight(AttachedModel& out, const AttachedPrimitiveInput& pin, std::size_t slot) noexcept {
  if (!pin.has_tight || pin.kind != AttachedShapeKind::kBox || slot >= out.hulls.size()) {
    return -1;
  }
  const std::size_t budget = static_cast<std::size_t>(kMaxTightHullVertices);
  const std::size_t first = slot * budget;
  if (first + budget > out.hull_vertices.size()) {
    return -1;  // no hull store configured; the box stays the model
  }
  // Over the stage-2 cost budget: keep the SLABS and drop the vertices, which
  // is stage 1 only — the same representation the producer emits for the same
  // condition (`_tight_geometry_from_points`), and the same one `panda_link1`
  // ships. Refusing the whole refinement here would throw away a DOP that
  // validates, and would leave the two sides of the wire disagreeing about
  // what "over budget" means. The vertices are never TRUNCATED to the first
  // `budget` of them: a truncated hull no longer contains its mesh.
  const std::size_t count = pin.hull_vertices.size() > budget ? 0 : pin.hull_vertices.size();
  LinkHull hull;
  hull.vertex_first = static_cast<int>(first);
  hull.vertex_count = static_cast<int>(count);
  for (int i = 0; i < kDopAxes; ++i) {
    hull.dop_lo[i] = pin.dop_lo[i];
    hull.dop_hi[i] = pin.dop_hi[i];
  }
  for (std::size_t v = 0; v < count; ++v) {
    out.hull_vertices[first + v] = pin.hull_vertices[v];
  }
  if (validate_tight_hull(hull, pin.half_extents, out.hull_vertices) != TightGeometryStatus::kOk) {
    return -1;
  }
  out.hulls[slot] = hull;
  return static_cast<int>(slot);
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
    // ADR-0100 contact-force witness. Each derived flag is masked by
    // `has_contact_force_witness` for the same reason the support fields above
    // are: a stale magnitude left behind by a retracted witness must not be
    // readable, and zeroing here means the gate's own checks cannot be reached
    // with a value nobody attested.
    o.has_contact_force_witness = in.has_contact_force_witness;
    o.contact_force_calibrated = in.has_contact_force_witness && in.contact_force_calibrated;
    o.contact_force_target_matches =
        in.has_contact_force_witness && in.contact_force_target_matches;
    o.contact_force_stamp_ns = in.has_contact_force_witness ? in.contact_force_stamp_ns : 0;
    o.contact_force_magnitude = in.has_contact_force_witness ? in.contact_force_magnitude : 0.0;
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
      const std::size_t slot = prim_cursor++;
      AttachedPrimitive& op = out.primitives[slot];
      op.kind = pin.kind;
      op.radius = pin.radius;
      op.half_length = pin.half_length;
      op.half_extents = pin.half_extents;
      op.pose_in_object = pin.pose_in_object;
      op.source = pin.source;
      op.hull_index = accept_tight(out, pin, slot);
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

bool payload_own_cell_exempts(const VoxelGrid& grid, const AttachedModel& attached,
                              std::size_t object_index, const Transform& object_xf,
                              std::size_t cell_idx, const Vec3& center,
                              const Vec3& local) noexcept {
  const PayloadOwnCells* t = grid.payload_own;
  if (t == nullptr || !t->valid || object_index >= 8 || (t->armed & (1U << object_index)) == 0 ||
      object_index >= attached.n_objects || !(grid.resolution > 0.0) ||
      !std::isfinite(grid.resolution) || !std::isfinite(center.x) || !std::isfinite(center.y) ||
      !std::isfinite(center.z) || !std::isfinite(local.x) || !std::isfinite(local.y) ||
      !std::isfinite(local.z)) {
    return false;
  }
  // The support, never the payload: the centre must stand strictly above the floor.
  const double floor = t->floor_z[object_index];
  if (!std::isfinite(floor) || !(center.z > floor)) {
    return false;
  }
  // A declared static collider's cell (padded) is the world, never the payload.
  const std::size_t cells = static_cast<std::size_t>(grid.sx) * static_cast<std::size_t>(grid.sy) *
                            static_cast<std::size_t>(grid.sz);
  if (t->static_cells != nullptr) {
    // A static table of another layout cannot say which cells are static: nothing is exempt.
    if (t->static_cells_n != cells || cell_idx >= cells || t->static_cells[cell_idx] != 0) {
      return false;
    }
  }
  const double half_side = grid.resolution * 0.5;
  const Vec3 voxel_half{half_side, half_side, half_side};
  Transform cell;
  cell.t = local;
  const Transform grid_from_base = inverse_rigid(grid.pose);
  const AttachedObject& obj = attached.objects[object_index];
  // 1. A measured / grown box primitive of THIS payload whose box meets the cell's volume.
  bool own = false;
  for (int p = 0; p < obj.prim_count && !own; ++p) {
    const AttachedPrimitive& prim =
        attached.primitives[static_cast<std::size_t>(obj.prim_first + p)];
    const bool fallback_ok = (t->fallback_ok & (1U << object_index)) != 0;
    if ((prim.source == PrimitiveSource::kFallback && !fallback_ok) ||
        prim.kind != AttachedShapeKind::kBox || !(prim.half_extents.x > 0.0) ||
        !(prim.half_extents.y > 0.0) || !(prim.half_extents.z > 0.0) ||
        !std::isfinite(prim.half_extents.x) || !std::isfinite(prim.half_extents.y) ||
        !std::isfinite(prim.half_extents.z)) {
      continue;
    }
    const Transform prim_g = compose(grid_from_base, compose(object_xf, prim.pose_in_object));
    const double gap = box_box_distance(prim_g, prim.half_extents, cell, voxel_half);
    own = std::isfinite(gap) && gap <= 0.0;
  }
  if (!own) {
    return false;
  }
  // 2. A cell that also meets ANOTHER settled track's box is that body's, not the payload's.
  const std::size_t n_others = std::min(t->n_others, t->other_box.size());
  for (std::size_t k = 0; k < n_others; ++k) {
    if (k < 32 && (t->own[object_index] & (1U << k)) != 0) {
      continue;
    }
    const double gap = box_box_distance(compose(grid_from_base, t->other_box[k]), t->other_half[k],
                                        cell, voxel_half);
    if (!std::isfinite(gap) || gap <= 0.0) {
      return false;
    }
  }
  return true;
}

CollisionHit check_attached_voxel_collision(const CollisionModel& /*model*/,
                                            const AttachedModel& attached,
                                            const CollisionScratch& scratch, const VoxelGrid& grid,
                                            double margin, double band_m) noexcept {
  CollisionHit result;
  double sweep_min = std::numeric_limits<double>::infinity();
  if (grid.occupancy == nullptr || grid.sx <= 0 || grid.sy <= 0 || grid.sz <= 0 ||
      grid.resolution <= 0.0) {
    return finish_sweep(result, sweep_min);
  }
  const double half_side = grid.resolution * 0.5;
  const Vec3 voxel_half{half_side, half_side, half_side};
  const Transform grid_from_base = inverse_rigid(grid.pose);
  // Bounded across every payload primitive in this call, exactly as the
  // arm-link sweep bounds its own: stage 2's cost is set by how many cells
  // stage 1 cannot clear, a property of the map rather than of the payload.
  // Exhausting it drops back to stage 1 and then to the shipped box bound —
  // today's answer or better, never worse.
  int stage2_budget = kMaxStage2PerCheck;
  for (std::size_t i = 0; i < attached.n_objects; ++i) {
    const AttachedObject& obj = attached.objects[i];
    if (obj.attach_link < 0) {
      // #418 K5: a released payload, frozen in the base frame, is no longer carried: no
      // robot motion moves it, so its overlap with the map is a static box on static cells
      // (Spark R168-R176, 2026-10-22: the banana's frozen box on the bowl it was dropped
      // into latched a stop ~60 ms after release). It stays an obstacle to every robot link
      // (`check_attached_self_collision`); only the world check is skipped. Safety-WG review
      // owed.
      continue;
    }
    const Transform obj_xf = attached_object_transform(obj, scratch);
    for (int p = 0; p < obj.prim_count; ++p) {
      const AttachedPrimitive& prim =
          attached.primitives[static_cast<std::size_t>(obj.prim_first + p)];
      const Transform prim_xf = compose(obj_xf, prim.pose_in_object);
      // The cell distance runs in the grid's axes; the place region and the
      // support attestation are declared in `base_frame` and keep that centre.
      const Transform prim_g = compose(grid_from_base, prim_xf);
      // Stage-2 pose for a primitive that carries a proved-contained
      // refinement (#266). Built in the GRID's axes, which the staged path
      // requires rather than merely prefers: its cells must be axis-aligned
      // cubes. -1 (no refinement) leaves every line below on the shipped path.
      // Range-checked ONCE, here, so the init below and every use inside the
      // cell loop read the same predicate. Two different conditions is how a
      // later edit ends up refining against a default-constructed TightPose —
      // whose zeroed slabs report clearance the geometry does not have, i.e.
      // it fails OPEN. `accept_tight` already bounds the index; this makes
      // that a local property rather than one held at a distance.
      const int hull_index = (prim.hull_index >= 0 &&
                              static_cast<std::size_t>(prim.hull_index) < attached.hulls.size())
                                 ? prim.hull_index
                                 : -1;
      TightPose tight;
      GjkWitness witness;
      if (hull_index >= 0) {
        tight_pose_init(attached.hulls[static_cast<std::size_t>(hull_index)],
                        attached.hull_vertices.empty() ? nullptr : attached.hull_vertices.data(),
                        prim_g, half_side, tight);
      }
      // ADR-0098 (survey Path B): the declared target's own geometry, measured
      // once per payload primitive rather than once per cell — the receptacle
      // does not move between cells of the same sweep. `+infinity` whenever the
      // declaration ships no geometry, which is what collapses every use below
      // to the pre-ADR-0098 path without a second branch.
      const double target_distance = place_target_distance(grid.place_region, prim, prim_xf);
      const CellBox box = primitive_cell_box(prim, prim_g, grid, margin + band_m + half_side);
      for (int iz = box.z0; iz <= box.z1; ++iz) {
        for (int iy = box.y0; iy <= box.y1; ++iy) {
          for (int ix = box.x0; ix <= box.x1; ++ix) {
            const std::size_t idx = static_cast<std::size_t>(ix + grid.sx * (iy + grid.sy * iz));
            if (grid.occupancy[idx] == 0) {
              continue;
            }
            const Vec3 center = voxel_center(grid, ix, iy, iz);
            Transform voxel;
            voxel.t = voxel_center_local(grid, ix, iy, iz);
            const double d_box = attached_primitive_voxel_distance(prim, prim_g, voxel, voxel_half);
            // Tighten the payload's own side of the pair (#266). Every stage is
            // a LOWER bound on the true mesh-to-cell clearance
            // (mesh ⊆ hull ⊆ DOP ⊆ box, proved at ingest), so the max of them
            // is still a lower bound: this can only ever RAISE the reported
            // clearance, and therefore only ever remove a stop, never add one.
            // Skipped outright on a cell the box already clears, which is
            // almost all of them — refining a cell nothing was going to stop on
            // buys nothing and costs a GJK.
            double d_cell = d_box;
            if (hull_index >= 0 && d_box <= margin) {
              Vec3 seed;
              double staged = dop_cell_lower_bound(tight, voxel.t, half_side, &seed);
              if (staged <= margin && tight.n_vertices > 0 && stage2_budget > 0) {
                --stage2_budget;
                staged =
                    hull_cell_distance(tight, voxel.t, half_side, seed, margin, staged, witness);
              }
              if (staged > d_cell) {
                d_cell = staged;
              }
            }
            // Declaration-scoped approach allowance (ADR-0097's 2026-08-14
            // amendment): inside the declared target's region this payload's
            // margin — and only this payload's, and only for cells in that
            // region — is reduced by min(1.5 x voxel, 40 mm), enough to pass the
            // voxel inflation of a thin opening and actually reach the support
            // contact the place witness is earned by. The hard stop behind the
            // reduced margin is untouched.
            const double allowance = place_approach_allowance(grid, i, center);
            // `kGateTieEpsilonM` because the ceiling is a documented TRIP
            // boundary ("exactly the ceiling: it trips", ADR-0097's Second
            // Amendment) and `d` is the end of a chain of rigid transforms.
            // Which side of an exact tie the last ULP lands on is not a safety
            // decision anyone made; erring onto the stop is the safe reading,
            // and a picometre is thirteen orders below the lattice it is
            // measured on.
            double d = d_cell;
            double cell_margin = margin - allowance + kGateTieEpsilonM;
            bool adjudicated = false;
            // ADR-0098 (survey Path B). The blanket allowance hands the SAME
            // relief to every cell inside the declared box, whether or not
            // the declared body put the cell there — a bound on
            // quantisation error, applied blind, since the box says WHERE
            // the receptacle is, never WHAT it is. When the producer ships
            // the target's geometry, the kernel isn't blind: it measures
            // the payload against the receptacle itself.
            //
            // The gate then moves to the surface. A margin is a standoff for
            // geometry the robot must not touch; the declared target is the
            // one body it was dispatched to touch, so it's gated as an
            // INTENDED-CONTACT pair at zero clearance — the contract
            // Tesseract's negatable per-pair margins and MoveIt's
            // touch_links both express (survey §17.2, §3.2). The advisory
            // band below still covers arrival overshoot; past it the
            // latched stop is unchanged.
            //
            // Two bounds carry the safety argument:
            //  1. Against TRUTH, strictly MORE conservative than the
            //     blanket: the blanket lets the payload sit `allowance`
            //     inside a cube whose near face may be a whole voxel in
            //     front of the real surface (unknowable permission); this
            //     permits zero penetration of the real body.
            //  2. Against the CUBE, looser by exactly the amount the cube
            //     over-stated the surface, never more than the blanket
            //     allowance in force (the <= guard enforces this). A model
            //     claiming more clearance than the map's quantisation could
            //     explain (fitted-too-small primitive, stale articulated
            //     door) is refused outright and falls back to the blanket
            //     path bit for bit.
            //
            // Outside the branch (no geometry, or a model past that bound)
            // nothing changes.
            // Deliberately `d_box`, not the #266-tightened `d_cell`: this
            // bound is ADR-0098's own, calibrated against the box model the
            // declaration's geometry is adjudicated on. Letting a tighter
            // payload widen it would hand the receptacle branch relief that
            // nothing in ADR-0098 measured. #266 removes stops outside this
            // branch; inside it, behaviour is unchanged bit for bit.
            if (allowance > 0.0 && target_distance <= d_box + allowance) {
              d = target_distance;
              cell_margin = kGateTieEpsilonM;
              adjudicated = true;
            }
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
              } else if (payload_own_cell_exempts(grid, attached, i, obj_xf, idx, center,
                                                  voxel.t)) {
                // Issue #418 (OT-10 amendment): the payload's own measured / grown cell.
                tripped = false;
                ++result.payload_own_pairs;
                if (d < result.payload_own_min_distance) {
                  result.payload_own_min_distance = d;
                }
              } else if (workspace_contact_exempts(grid, obj.attach_link, center)) {
                // Manipulation workspace (#349, draft): a payload carried by a
                // hand link is part of that hand — inside the confirmed
                // workspace, while the hand (its speed and lead bounded with
                // the payload's reach, `attached_object_reach`) is slow, it may
                // touch. Same predicate as the hand links', on its attach link.
                tripped = false;
                note_workspace_pair(result, d, obj.attach_link, static_cast<int>(idx));
              } else if (provider_relaxable(grid.provider_gate, obj.attach_link)) {
                // ADR-0121: last, on a carried (never frozen) payload's masked attach link.
                PayloadPairQuery q;
                q.object = i;
                q.attached_model = &attached;
                q.attached = &obj;
                q.object_xf = obj_xf;
                q.attach_link = obj.attach_link;
                q.cell = idx;
                q.cell_center_base = center;
                q.cell_center_grid = voxel.t;
                q.distance = d;
                q.row = grid.provider_gate.row;
                q.row_fk = &scratch;
                q.hand_speed_mps = grid.provider_gate.hand_speed_mps;
                q.hand_lead_m = grid.provider_gate.hand_lead_m;
                q.hand_contact = grid.provider_gate.hand_contact;
                if (grid.provider_gate.provider->payload_voxel_pair(q) ==
                    ProviderVerdict::kTolerate) {
                  tripped = false;
                  note_provider_pair(result, d, obj.attach_link, static_cast<int>(idx));
                }
              }
            }
            // Advisory band (#176). A payload ARRIVED in its own declared
            // receptacle reads deeper than the approach allowance from map
            // discretisation, not force: octomap marks the cell CONTAINING
            // the ray endpoint (surface starts up to one resolution early),
            // and the payload is a fitted convex primitive that bulges past
            // the real object. 2026-08-26 baguette place read -38.22 mm at
            // a -1.4 mm physical contact.
            //
            // Inside the band the pair still TRIPS (a refusal, action
            // dropped); the band only spares the caller a latched fault +
            // E-stop over it. Every condition is required, each alone
            // restores today's latched stop: a live declaration covering
            // THIS object (allowance > 0, which place_approach_allowance
            // only returns for a cell inside the measured region), and a
            // depth still within one voxel of the allowance. A robot link
            // never reaches here — this is the attached-payload sweep.
            const double advisory_floor =
                cell_margin - place_advisory_depth(grid.resolution) - kGateTieEpsilonM;
            const bool advisory = tripped && allowance > 0.0 && d > advisory_floor;
            fold_pair(result, sweep_min, d, tripped, static_cast<int>(i), static_cast<int>(idx),
                      allowance > 0.0, advisory, adjudicated, false,
                      tripped ? workspace_denial(grid, obj.attach_link, center) : 0);
          }
        }
      }
    }
  }
  return finish_sweep(result, sweep_min);
}

ContactForceGateResult check_contact_force_gate(const AttachedModel& attached, std::int64_t now_ns,
                                                double robot_threshold_n) noexcept {
  ContactForceGateResult out{};
  const PlaceForceGate& gate = attached.force_gate;
  // Every early return here leaves `out.tripped` false,
  // which is the shipped geometry-only behaviour — the gate adds refusals and
  // failing to arm can therefore never be unsafe relative to today (ADR-0100
  // §1). Re-checking the range the ingest already validated is deliberate: a
  // permissive default is never the safe one on a safety path.
  if (!gate.valid || gate.object_mask == 0) {
    return out;
  }
  if (!(gate.threshold_n > 0.0) || !std::isfinite(gate.threshold_n) ||
      gate.threshold_n > kMaxContactForceThresholdN) {
    return out;
  }
  if (std::isnan(robot_threshold_n) || robot_threshold_n < 0.0) {
    return out;
  }
  const double threshold_n = std::min(gate.threshold_n, robot_threshold_n);
  // A malformed or expired declaration has no authority to arm the gate. Use
  // long double subtraction so hostile int64 stamps cannot overflow first.
  const long double declaration_age_ns =
      static_cast<long double>(now_ns) - static_cast<long double>(gate.declaration_stamp_ns);
  if (!(gate.declaration_timeout_s > 0.0) || !std::isfinite(gate.declaration_timeout_s) ||
      gate.declaration_timeout_s > kMaxPlaceForceGateTimeoutS || gate.declaration_stamp_ns <= 0 ||
      now_ns <= 0 || declaration_age_ns < 0.0L ||
      declaration_age_ns > static_cast<long double>(gate.declaration_timeout_s) * 1.0e9L) {
    return out;
  }
  if (!(gate.witness_max_age_s > 0.0) || !std::isfinite(gate.witness_max_age_s)) {
    return out;
  }
  const long double declaration_remaining_ns =
      static_cast<long double>(gate.declaration_timeout_s) * 1.0e9L - declaration_age_ns;
  const long double witness_age_limit_ns =
      std::min(declaration_remaining_ns, static_cast<long double>(gate.witness_max_age_s) * 1.0e9L);
  const std::size_t count =
      attached.n_objects < attached.objects.size() ? attached.n_objects : attached.objects.size();
  for (std::size_t i = 0; i < count && i < 8; ++i) {
    // Condition 3: the gate follows the DECLARED payload. A second object the
    // robot happens to be carrying is outside the declaration and outside this.
    if ((gate.object_mask & (1U << i)) == 0) {
      continue;
    }
    const AttachedObject& obj = attached.objects[i];
    // Condition 4: a witness exists and names the declared target.
    if (!obj.has_contact_force_witness || !obj.contact_force_target_matches) {
      continue;
    }
    // The witness must still be a measurement in the same ROS clock domain.
    // A stale/future sample is absent, so geometry retains its existing result.
    const long double witness_age_ns =
        static_cast<long double>(now_ns) - static_cast<long double>(obj.contact_force_stamp_ns);
    if (obj.contact_force_stamp_ns <= 0 || obj.contact_force_stamp_ns > now_ns ||
        witness_age_ns > witness_age_limit_ns) {
      continue;
    }
    // Condition 5: the magnitude is calibrated. Without this the value is a
    // producer quantity and NOT newtons (survey §21.7), so it is not read at
    // all rather than read optimistically.
    if (!obj.contact_force_calibrated) {
      continue;
    }
    const double magnitude = obj.contact_force_magnitude;
    if (!std::isfinite(magnitude) || magnitude <= threshold_n) {
      continue;
    }
    // Report the deepest overrun, so the evidence names the worst pair rather
    // than the first one scanned — the same single-pair-coherence rule
    // `CollisionHit` is under.
    if (!out.tripped || magnitude > out.magnitude_n) {
      out.tripped = true;
      out.object_index = static_cast<int>(i);
      out.magnitude_n = magnitude;
      out.threshold_n = threshold_n;
    }
  }
  return out;
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
  // Exact point-in-OBB (a degenerate or non-finite box contains nothing). An
  // oriented box is what the producer can actually measure — the axis-aligned
  // hull of a rotated receptacle would be strictly larger, i.e. strictly more
  // permissive.
  if (!point_in_obb(region.pose, region.half_extents, center)) {
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

PlaceRegionStatus ingest_place_target_geometry(const std::vector<AttachedPrimitiveInput>& geometry,
                                               std::vector<AttachedPrimitive>& store,
                                               PlaceApproachRegion& region) noexcept {
  region.geometry = nullptr;
  region.n_geometry = 0;
  if (geometry.empty()) {
    return PlaceRegionStatus::kOk;  // a box with no geometry: the ADR-0097 path
  }
  if (geometry.size() > kMaxPlaceTargetPrimitives || store.size() < geometry.size()) {
    return PlaceRegionStatus::kGeometryOverflow;
  }
  for (std::size_t g = 0; g < geometry.size(); ++g) {
    const AttachedPrimitiveInput& in = geometry[g];
    // Identical shape sanity to `ingest_attached_objects`, and for the identical
    // reason: a primitive the kernel cannot measure against is one it would
    // silently treat as absent, and an absent half of a receptacle is exactly
    // the geometry a payload would then be cleared to drive through.
    if (in.kind == AttachedShapeKind::kBox) {
      if (!(std::isfinite(in.half_extents.x) && std::isfinite(in.half_extents.y) &&
            std::isfinite(in.half_extents.z)) ||
          in.half_extents.x <= 0.0 || in.half_extents.y <= 0.0 || in.half_extents.z <= 0.0) {
        return PlaceRegionStatus::kBadGeometry;
      }
    } else if (!std::isfinite(in.radius) || in.radius <= 0.0 || !std::isfinite(in.half_length) ||
               in.half_length < 0.0) {
      return PlaceRegionStatus::kBadGeometry;
    }
    if (!std::isfinite(in.pose_in_object.t.x) || !std::isfinite(in.pose_in_object.t.y) ||
        !std::isfinite(in.pose_in_object.t.z)) {
      return PlaceRegionStatus::kBadGeometry;
    }
    for (std::size_t k = 0; k < 9; ++k) {
      if (!std::isfinite(in.pose_in_object.r[k])) {
        return PlaceRegionStatus::kBadGeometry;
      }
    }
    AttachedPrimitive& out = store[g];
    out.kind = in.kind;
    out.radius = in.radius;
    out.half_length = in.half_length;
    out.half_extents = in.half_extents;
    // The declared target's primitives are posed in the region's frame — the
    // robot base frame — not relative to the region box and not in an object
    // frame that would need composing. `pose_in_object` is the wire field's
    // name; here it already IS the base-frame pose.
    out.pose_in_object = in.pose_in_object;
  }
  region.geometry = store.data();
  region.n_geometry = geometry.size();
  return PlaceRegionStatus::kOk;
}

double place_target_distance(const PlaceApproachRegion& region, const AttachedPrimitive& prim,
                             const Transform& prim_xf) noexcept {
  if (!region.valid || region.geometry == nullptr || region.n_geometry == 0) {
    return std::numeric_limits<double>::infinity();
  }
  double minimum = std::numeric_limits<double>::infinity();
  for (std::size_t g = 0; g < region.n_geometry; ++g) {
    const AttachedPrimitive& target = region.geometry[g];
    // Both operands are already base-frame, so this is the same certified
    // primitive-pair arithmetic every other check in this file runs — no coal,
    // no EPA, no hot-path allocation (survey §4.1).
    const double d = target.kind == AttachedShapeKind::kBox
                         ? attached_primitive_voxel_distance(prim, prim_xf, target.pose_in_object,
                                                             target.half_extents)
                         : attached_primitive_capsule_distance(prim, prim_xf, target.pose_in_object,
                                                               target.radius, target.half_length);
    minimum = std::min(minimum, d);
  }
  return minimum;
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
  case PlaceRegionStatus::kBadGeometry:
    return "bad_geometry";
  case PlaceRegionStatus::kGeometryOverflow:
    return "geometry_overflow";
  }
  return "unknown";
}

namespace {

// The geometry half of a region ingest, shared by the grasp target and the
// workspace: a non-empty mask, a finite pose, finite positive half-extents
// within `max_half`, a volume within `max_volume`.
GraspRegionStatus check_region_box(const Transform& pose, const Vec3& half_extents,
                                   const std::bitset<kMaxGraspMaskLinks>& link_mask,
                                   double max_half, double max_volume) noexcept {
  if (link_mask.none()) {
    return GraspRegionStatus::kNoLinks;
  }
  if (!std::isfinite(pose.t.x) || !std::isfinite(pose.t.y) || !std::isfinite(pose.t.z)) {
    return GraspRegionStatus::kBadPose;
  }
  for (std::size_t k = 0; k < 9; ++k) {
    if (!std::isfinite(pose.r[k])) {
      return GraspRegionStatus::kBadPose;
    }
  }
  const double hx = half_extents.x;
  const double hy = half_extents.y;
  const double hz = half_extents.z;
  if (!std::isfinite(hx) || !std::isfinite(hy) || !std::isfinite(hz)) {
    return GraspRegionStatus::kBadExtents;
  }
  if (hx <= 0.0 || hy <= 0.0 || hz <= 0.0) {
    return GraspRegionStatus::kDegenerate;
  }
  if (hx > max_half || hy > max_half || hz > max_half) {
    return GraspRegionStatus::kOversize;
  }
  if (8.0 * hx * hy * hz > max_volume) {
    return GraspRegionStatus::kOversizeVolume;
  }
  return GraspRegionStatus::kOk;
}

}  // namespace

GraspRegionStatus ingest_grasp_region(const Transform& pose, const Vec3& half_extents,
                                      const std::bitset<kMaxGraspMaskLinks>& link_mask,
                                      GraspTargetRegion& out) noexcept {
  out = GraspTargetRegion{};
  const GraspRegionStatus status = check_region_box(
      pose, half_extents, link_mask, kMaxGraspRegionHalfExtentM, kMaxGraspRegionVolumeM3);
  if (status != GraspRegionStatus::kOk) {
    return status;
  }
  out.valid = true;
  out.link_mask = link_mask;
  out.pose = pose;
  out.half_extents = half_extents;
  return GraspRegionStatus::kOk;
}

GraspRegionStatus ingest_workspace_region(const Transform& pose, const Vec3& half_extents,
                                          const std::bitset<kMaxGraspMaskLinks>& link_mask,
                                          double max_speed_mps, double max_lead_m,
                                          WorkspaceContactRegion& out) noexcept {
  out = WorkspaceContactRegion{};
  const GraspRegionStatus status = check_region_box(
      pose, half_extents, link_mask, kMaxWorkspaceRegionHalfExtentM, kMaxWorkspaceRegionVolumeM3);
  if (status != GraspRegionStatus::kOk) {
    return status;
  }
  out.valid = true;
  out.link_mask = link_mask;
  out.pose = pose;
  out.half_extents = half_extents;
  out.max_speed_mps = max_speed_mps;
  out.max_lead_m = max_lead_m;
  return GraspRegionStatus::kOk;
}

bool grasp_links_one_hand(const std::vector<int>& parent,
                          const std::bitset<kMaxGraspMaskLinks>& allowlist,
                          const std::bitset<kMaxGraspMaskLinks>& mask) noexcept {
  const std::size_t n = std::min(parent.size(), kMaxGraspMaskLinks);
  if (mask.none()) {
    return false;
  }
  bool seen = false;
  int hand = -1;
  for (std::size_t c = 0; c < kMaxGraspMaskLinks; ++c) {
    if (!mask[c]) {
      continue;
    }
    if (c >= n) {
      return false;  // a mask bit naming no link of the model
    }
    int mount = parent[c];
    std::size_t steps = 0;
    while (mount >= 0 && static_cast<std::size_t>(mount) < n &&
           allowlist[static_cast<std::size_t>(mount)]) {
      if (++steps > n) {
        return false;  // a cyclic parent chain: fail closed
      }
      mount = parent[static_cast<std::size_t>(mount)];
    }
    if (seen && mount != hand) {
      return false;
    }
    seen = true;
    hand = mount;
  }
  return seen;
}

const char* grasp_region_status_reason(GraspRegionStatus status) noexcept {
  switch (status) {
  case GraspRegionStatus::kOk:
    return "ok";
  case GraspRegionStatus::kNoLinks:
    return "no_links";
  case GraspRegionStatus::kBadPose:
    return "bad_pose";
  case GraspRegionStatus::kBadExtents:
    return "bad_extents";
  case GraspRegionStatus::kDegenerate:
    return "degenerate";
  case GraspRegionStatus::kOversize:
    return "oversize";
  case GraspRegionStatus::kOversizeVolume:
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
  // "Calibration 2026-08-15", 5-run baguette battery; ADR-0097 Second
  // Amendment). Cells of adjacent co-planar structure (a raised edge, a
  // neighbouring stack on the same support surface) sit ~1 voxel above the
  // attested plane during genuine contact; round-8 r2 measured +42.9 mm
  // against a ~15-19 mm envelope, excess of one 25 mm voxel. Widens HEIGHT
  // only, inside the unchanged lateral patch bound above. Deepening past
  // the widened bound still stops.
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
  const Transform grid_from_base = inverse_rigid(grid.pose);
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
    voxel.t = voxel_center_local(grid, ix, iy, iz);
    for (std::size_t object_index = 0; object_index < objects; ++object_index) {
      const std::uint8_t bit = static_cast<std::uint8_t>(1U << object_index);
      if (!snapshot && (contact_mask[idx] & bit) == 0) {
        continue;
      }
      const AttachedObject& object = attached.objects[object_index];
      // In the grid's axes, to match the axis-aligned local cell above.
      const Transform object_xf =
          compose(grid_from_base, attached_object_transform(object, scratch));
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
                                           const CollisionScratch& scratch, double margin,
                                           const WorkspaceContactRegion* workspace,
                                           const ProviderGate* gate,
                                           const WorkspaceContactRegion* narrow_under) noexcept {
  const WorkspaceContactRegion* narrow = narrow_under != nullptr ? narrow_under : workspace;
  CollisionHit result;
  double sweep_min = std::numeric_limits<double>::infinity();
  const std::size_t n_caps = model.capsules.size();
  const std::size_t n_boxes = model.boxes.size();
  // #349: a carried payload is part of its hand — against another hand link, the
  // two-slow-hands-inside-the-workspace rule on its attach link.
  const auto fold = [&](double d, int object, int attach_link, int link) {
    const bool exempt = d <= margin && workspace_self_exempt(workspace, scratch, attach_link, link);
    if (exempt) {
      note_workspace_pair(result, d, attach_link, link);
    }
    // ADR-0121: last. A carried payload on its attach link, or a frozen one in its release
    // window on the hand that let it go (`release_hand_link`, the K5 window); that hand and the
    // other link both in the kernel's mask. A frozen payload outside a window is never offered.
    bool provider_exempt = false;
    const AttachedObject& obj = attached.objects[static_cast<std::size_t>(object)];
    const int hand_link = obj.attach_link >= 0 ? obj.attach_link : obj.release_hand_link;
    if (!exempt && d <= margin && gate != nullptr && hand_link >= 0 &&
        provider_relaxable(*gate, hand_link) && provider_relaxable(*gate, link)) {
      PayloadPairQuery q;
      q.object = static_cast<std::size_t>(object);
      q.attached_model = &attached;
      q.attached = &obj;
      q.object_xf = attached_object_transform(obj, scratch);
      q.attach_link = obj.attach_link;
      q.release_hand_link = obj.attach_link >= 0 ? -1 : obj.release_hand_link;
      q.other_link = link;
      q.distance = d;
      q.row = gate->row;
      q.row_fk = &scratch;
      q.hand_speed_mps = gate->hand_speed_mps;
      q.hand_lead_m = gate->hand_lead_m;
      q.hand_contact = gate->hand_contact;
      provider_exempt = gate->provider->payload_self_pair(q) == ProviderVerdict::kTolerate;
      if (provider_exempt) {
        note_provider_pair(result, d, hand_link, link);
      }
    }
    fold_pair(result, sweep_min, d, d <= margin && !exempt && !provider_exempt, object, link);
  };
  for (std::size_t i = 0; i < attached.n_objects; ++i) {
    const AttachedObject& obj = attached.objects[i];
    const Transform obj_xf = attached_object_transform(obj, scratch);
    // #418 K5: in a release window the payload is frozen (attach_link -1) but the
    // hand that let it go keeps the slow-hands rule it had while carrying it (Spark
    // R243/R250, 2026-10-22: link7, exempt against the carried banana, tripped on the
    // same box frozen 0.2 s later). Closed windows clear release_hand_link: no rule.
    // Safety-WG review owed.
    const int hand = obj.attach_link >= 0 ? obj.attach_link : obj.release_hand_link;
    for (int p = 0; p < obj.prim_count; ++p) {
      const AttachedPrimitive& prim0 =
          attached.primitives[static_cast<std::size_t>(obj.prim_first + p)];
      // #349: against the robot's own exact geometry, a box payload with a
      // stage-2 refinement is checked as the box its 26-DOP's first three slabs
      // span — the producer's MEASURED object (the refinement is proved inside
      // the shipped box at ingest); the shipped box's padding is for the map's
      // quantisation, not for the robot's own links (r24-r29: a padded carton
      // 1 cm from the torso read -15 to -40 mm against it). Only under a live
      // workspace (the sim-only #349 rule): without one this is today's check on
      // the shipped, padded box (#349 review).
      AttachedPrimitive inner = prim0;
      if (narrow != nullptr && narrow->valid && prim0.kind == AttachedShapeKind::kBox &&
          prim0.hull_index >= 0 &&
          static_cast<std::size_t>(prim0.hull_index) < attached.hulls.size()) {
        const LinkHull& hull = attached.hulls[static_cast<std::size_t>(prim0.hull_index)];
        const Vec3 lo{hull.dop_lo[0], hull.dop_lo[1], hull.dop_lo[2]};
        const Vec3 hi{hull.dop_hi[0], hull.dop_hi[1], hull.dop_hi[2]};
        Transform shift;
        shift.t = Vec3{0.5 * (lo.x + hi.x), 0.5 * (lo.y + hi.y), 0.5 * (lo.z + hi.z)};
        inner.pose_in_object = compose(prim0.pose_in_object, shift);
        inner.half_extents = Vec3{0.5 * (hi.x - lo.x), 0.5 * (hi.y - lo.y), 0.5 * (hi.z - lo.z)};
      }
      const AttachedPrimitive& prim = inner;
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
        fold(d, static_cast<int>(i), hand, lc);
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
        fold(d, static_cast<int>(i), hand, lb);
      }
    }
  }
  return finish_sweep(result, sweep_min);
}

namespace {

// Does some capsule or OBB of a link `allows` accepts lie within `slack` of the box?
template <typename Allows>
bool box_meets_links(const CollisionModel& model, const CollisionScratch& scratch,
                     const Transform& box, const Vec3& half_extents, double slack,
                     const Allows& allows) noexcept {
  if (!std::isfinite(slack) || !(half_extents.x > 0.0) || !(half_extents.y > 0.0) ||
      !(half_extents.z > 0.0) || !std::isfinite(half_extents.x) || !std::isfinite(half_extents.y) ||
      !std::isfinite(half_extents.z) || !std::isfinite(box.t.x) || !std::isfinite(box.t.y) ||
      !std::isfinite(box.t.z)) {
    return false;
  }
  const auto pose = [&](int link, const Transform& origin) {
    return compose(scratch.link_world[static_cast<std::size_t>(link)], origin);
  };
  for (std::size_t c = 0; c < model.capsules.size() && c < model.capsule_link.size(); ++c) {
    const int link = model.capsule_link[c];
    if (!allows(link)) {
      continue;
    }
    const Capsule& cap = model.capsules[c];
    if (box_capsule_distance(box, half_extents, pose(link, cap.origin), cap.radius,
                             cap.half_length) <= slack) {
      return true;
    }
  }
  for (std::size_t b = 0; b < model.boxes.size() && b < model.box_link.size(); ++b) {
    const int link = model.box_link[b];
    if (!allows(link)) {
      continue;
    }
    if (box_box_distance(box, half_extents, pose(link, model.boxes[b].origin),
                         model.boxes[b].half_extents) <= slack) {
      return true;
    }
  }
  return false;
}

}  // namespace

bool tracked_box_meets_hand(const CollisionModel& model, const CollisionScratch& scratch,
                            const AttachedModel& attached, const AttachedObject& object,
                            const Transform& box, const Vec3& half_extents, double slack) noexcept {
  return box_meets_links(model, scratch, box, half_extents, slack,
                         [&](int link) { return attached_allows_link(attached, object, link); });
}

bool box_meets_hand_links(const CollisionModel& model, const CollisionScratch& scratch,
                          const std::bitset<kMaxGraspMaskLinks>& hand_mask, const Transform& box,
                          const Vec3& half_extents, double slack) noexcept {
  return box_meets_links(model, scratch, box, half_extents, slack, [&](int link) {
    return link >= 0 && static_cast<std::size_t>(link) < kMaxGraspMaskLinks &&
           static_cast<std::size_t>(link) < scratch.link_world.size() &&
           hand_mask[static_cast<std::size_t>(link)];
  });
}

double released_payload_clearance(const CollisionModel& model, const CollisionScratch& scratch,
                                  const AttachedModel& attached,
                                  const AttachedObject& object) noexcept {
  double gap = std::numeric_limits<double>::infinity();
  const Transform obj_xf = attached_object_transform(object, scratch);
  for (int p = 0; p < object.prim_count; ++p) {
    const AttachedPrimitive& prim =
        attached.primitives[static_cast<std::size_t>(object.prim_first + p)];
    const Transform prim_xf = compose(obj_xf, prim.pose_in_object);
    for (std::size_t c = 0; c < model.capsules.size(); ++c) {
      const int link = model.capsule_link[c];
      if (!attached_allows_link(attached, object, link)) {
        continue;
      }
      const Capsule& cap = model.capsules[c];
      gap =
          std::min(gap, attached_primitive_capsule_distance(
                            prim, prim_xf,
                            compose(scratch.link_world[static_cast<std::size_t>(link)], cap.origin),
                            cap.radius, cap.half_length));
    }
    for (std::size_t b = 0; b < model.boxes.size(); ++b) {
      const int link = model.box_link[b];
      if (!attached_allows_link(attached, object, link)) {
        continue;
      }
      const Transform box_j =
          compose(scratch.link_world[static_cast<std::size_t>(link)], model.boxes[b].origin);
      gap = std::min(gap, prim.kind == AttachedShapeKind::kBox
                              ? box_box_distance(prim_xf, prim.half_extents, box_j,
                                                 model.boxes[b].half_extents)
                              : box_capsule_distance(box_j, model.boxes[b].half_extents, prim_xf,
                                                     prim.radius, prim.half_length));
    }
  }
  return gap;
}

// ── Issue #418 K2: the grasp exemption by tracked-object id ──────────────────

bool build_grasp_hands(const CollisionModel& model, const std::bitset<kMaxGraspMaskLinks>& fingers,
                       TrackedGraspTable& table,
                       const std::bitset<kMaxGraspMaskLinks>& extra) noexcept {
  table.n_hands = 0;
  table.hand_links = {};
  table.link_point = {};
  table.extra_links = extra & ~fingers;
  const std::size_t n = std::min(model.parent.size(), kMaxGraspMaskLinks);
  if (fingers.none()) {
    return false;
  }
  const std::bitset<kMaxGraspMaskLinks> allowlist = fingers | extra;
  std::array<int, kMaxGraspHands> mounts{};
  for (std::size_t c = 0; c < kMaxGraspMaskLinks; ++c) {
    if (!allowlist[c]) {
      continue;
    }
    if (c >= n) {
      table.n_hands = 0;
      return false;  // a bit naming no link of the model
    }
    // The mount: the first ancestor not itself allowlisted (grasp_links_one_hand).
    int mount = model.parent[c];
    std::size_t steps = 0;
    while (mount >= 0 && static_cast<std::size_t>(mount) < n &&
           allowlist[static_cast<std::size_t>(mount)]) {
      if (++steps > n) {
        table.n_hands = 0;
        return false;  // a cyclic parent chain: fail closed
      }
      mount = model.parent[static_cast<std::size_t>(mount)];
    }
    std::size_t h = 0;
    while (h < table.n_hands && mounts[h] != mount) {
      ++h;
    }
    if (h == table.n_hands) {
      if (h == kMaxGraspHands) {
        table.n_hands = 0;
        return false;
      }
      mounts[h] = mount;
      ++table.n_hands;
    }
    table.hand_links[h].set(c);
    // The finger's TCP sample: the mean of its primitives' origins (link frame).
    Vec3 sum{};
    int count = 0;
    for (std::size_t k = 0; k < model.capsules.size() && k < model.capsule_link.size(); ++k) {
      if (model.capsule_link[k] == static_cast<int>(c)) {
        sum = add(sum, model.capsules[k].origin.t);
        ++count;
      }
    }
    for (std::size_t k = 0; k < model.boxes.size() && k < model.box_link.size(); ++k) {
      if (model.box_link[k] == static_cast<int>(c)) {
        sum = add(sum, model.boxes[k].origin.t);
        ++count;
      }
    }
    table.link_point[c] = count > 0 ? scale(sum, 1.0 / count) : Vec3{};
  }
  for (std::size_t h = 0; h < table.n_hands; ++h) {
    if ((table.hand_links[h] & fingers).none()) {
      table.n_hands = 0;
      return false;  // a hand of extra links alone: no jaw to grasp with
    }
  }
  return table.n_hands > 0;
}

Vec3 grasp_hand_tcp(const TrackedGraspTable& table, const CollisionScratch& scratch,
                    std::size_t hand) noexcept {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  if (hand >= table.n_hands || hand >= kMaxGraspHands) {
    return Vec3{nan, nan, nan};
  }
  Vec3 sum{};
  int count = 0;
  for (std::size_t l = 0; l < kMaxGraspMaskLinks; ++l) {
    if (!table.hand_links[hand][l] || table.extra_links[l]) {
      continue;
    }
    if (l >= scratch.link_world.size()) {
      return Vec3{nan, nan, nan};
    }
    sum = add(sum, apply(scratch.link_world[l], table.link_point[l]));
    ++count;
  }
  return count > 0 ? scale(sum, 1.0 / count) : Vec3{nan, nan, nan};
}

namespace {

// Distance from `p` to the oriented box (0 inside); NaN for a non-finite input.
double point_box_gap(const Transform& box, const Vec3& half, const Vec3& p) noexcept {
  const Vec3 d = sub(p, box.t);
  const double l[3] = {box.r[0] * d.x + box.r[3] * d.y + box.r[6] * d.z,
                       box.r[1] * d.x + box.r[4] * d.y + box.r[7] * d.z,
                       box.r[2] * d.x + box.r[5] * d.y + box.r[8] * d.z};
  const double h[3] = {half.x, half.y, half.z};
  double sq = 0.0;
  for (int k = 0; k < 3; ++k) {
    const double out = std::fabs(l[k]) - h[k];
    if (out > 0.0) {
      sq += out * out;
    } else if (!(out <= 0.0)) {
      return std::numeric_limits<double>::quiet_NaN();
    }
  }
  return std::sqrt(sq);
}

}  // namespace

void select_tracked_grasp(const TrackedGraspTable& table, const CollisionScratch& scratch,
                          double tie_m, std::array<int, kMaxGraspHands>& out) noexcept {
  out.fill(-1);
  if (!table.valid || !std::isfinite(tie_m) || tie_m < 0.0) {
    return;
  }
  const std::size_t n_hands = std::min(table.n_hands, kMaxGraspHands);
  const std::size_t n_obj = std::min(table.n_objects, kMaxTrackedObjects);
  std::array<Vec3, kMaxGraspHands> tcp{};
  for (std::size_t h = 0; h < n_hands; ++h) {
    tcp[h] = grasp_hand_tcp(table, scratch, h);
  }
  for (std::size_t h = 0; h < n_hands; ++h) {
    int best = -1;
    double g1 = std::numeric_limits<double>::infinity();
    double g2 = std::numeric_limits<double>::infinity();
    bool finite = true;
    for (std::size_t i = 0; i < n_obj; ++i) {
      const double g = point_box_gap(table.box[i], table.half_extents[i], tcp[h]);
      if (!std::isfinite(g)) {
        finite = false;  // a NaN TCP or box: no selection for this hand
        break;
      }
      if (g < g1) {
        g2 = g1;
        g1 = g;
        best = static_cast<int>(i);
      } else if (g < g2) {
        g2 = g;
      }
    }
    if (!finite || best < 0 || g2 - g1 <= tie_m) {
      continue;  // none, or a tie within one voxel (HZ-0115-2)
    }
    // Never an object at least as near another hand (HZ-0115-12).
    const auto b = static_cast<std::size_t>(best);
    bool other_nearer = false;
    for (std::size_t k = 0; k < n_hands; ++k) {
      if (k != h && !(point_box_gap(table.box[b], table.half_extents[b], tcp[k]) > g1)) {
        other_nearer = true;  // `!(>)`: a NaN other-hand gap refuses too
      }
    }
    if (!other_nearer) {
      out[h] = best;
    }
  }
}

std::size_t tracked_box_cover(const VoxelGrid& grid, const Transform& box,
                              const Vec3& half_extents) noexcept {
  if (grid.occupancy == nullptr || grid.sx <= 0 || grid.sy <= 0 || grid.sz <= 0 ||
      !(grid.resolution > 0.0) || !std::isfinite(grid.resolution) || !(half_extents.x > 0.0) ||
      !(half_extents.y > 0.0) || !(half_extents.z > 0.0) || !std::isfinite(half_extents.x) ||
      !std::isfinite(half_extents.y) || !std::isfinite(half_extents.z)) {
    return 0;
  }
  // The box's window on the grid's own axes (its AABB there), clamped.
  const Transform box_g = compose(inverse_rigid(grid.pose), box);
  const Vec3 ext = obb_extent(box_g, half_extents);
  const double inv = 1.0 / grid.resolution;
  const auto span = [&](double c, double e, int dim, int& i0, int& i1) {
    const double lo = std::floor((c - e) * inv);
    const double hi = std::floor((c + e) * inv);
    if (!std::isfinite(lo) || !std::isfinite(hi) || hi < 0.0 || lo > dim - 1) {
      return false;
    }
    i0 = static_cast<int>(std::max(lo, 0.0));
    i1 = static_cast<int>(std::min(hi, static_cast<double>(dim - 1)));
    return true;
  };
  int x0 = 0;
  int x1 = 0;
  int y0 = 0;
  int y1 = 0;
  int z0 = 0;
  int z1 = 0;
  if (!span(box_g.t.x, ext.x, grid.sx, x0, x1) || !span(box_g.t.y, ext.y, grid.sy, y0, y1) ||
      !span(box_g.t.z, ext.z, grid.sz, z0, z1)) {
    return 0;
  }
  std::size_t count = 0;
  for (int iz = z0; iz <= z1; ++iz) {
    for (int iy = y0; iy <= y1; ++iy) {
      for (int ix = x0; ix <= x1; ++ix) {
        const std::size_t idx = static_cast<std::size_t>(ix + grid.sx * (iy + grid.sy * iz));
        if (grid.occupancy[idx] != 0 &&
            point_in_obb(box, half_extents, voxel_center(grid, ix, iy, iz))) {
          ++count;
        }
      }
    }
  }
  return count;
}

bool tracked_cover_sufficient(std::size_t occupied, std::uint32_t visible) noexcept {
  // max(2, ceil(0.5 * visible)), in integers.
  const std::size_t half_up = (static_cast<std::size_t>(visible) + 1) / 2;
  return occupied >= std::max<std::size_t>(2, half_up);
}

// ── Issue #418 K7: explained cells for workspace contact ─────────────────────

// Horizontal gap (m) from `p` to the box's footprint, in the box's own x / y
// (the tracker's boxes are gravity-aligned, so those are horizontal). NaN for a
// non-finite input, which never compares as within reach.
double footprint_gap(const Transform& box, const Vec3& half, const Vec3& p) noexcept {
  const Vec3 d = sub(p, box.t);
  const double lx = box.r[0] * d.x + box.r[3] * d.y + box.r[6] * d.z;
  const double ly = box.r[1] * d.x + box.r[4] * d.y + box.r[7] * d.z;
  const double ex = std::max(std::fabs(lx) - half.x, 0.0);
  const double ey = std::max(std::fabs(ly) - half.y, 0.0);
  if (!std::isfinite(lx) || !std::isfinite(ly) || !std::isfinite(half.x) ||
      !std::isfinite(half.y) || !(half.x > 0.0) || !(half.y > 0.0)) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  return std::sqrt(ex * ex + ey * ey);
}

// The cell of `grid` whose (base-frame) point `p` falls in, or -1 outside /
// degenerate — the octomap bridge's `mark_base_points` indexing.
long grid_cell_of(const VoxelGrid& grid, const Transform& grid_from_base, const Vec3& p) noexcept {
  const Vec3 l = apply(grid_from_base, p);
  const double fx = std::floor(l.x / grid.resolution);
  const double fy = std::floor(l.y / grid.resolution);
  const double fz = std::floor(l.z / grid.resolution);
  // `!(>=)` so a NaN coordinate is outside too.
  if (!(fx >= 0.0) || !(fy >= 0.0) || !(fz >= 0.0) || fx >= grid.sx || fy >= grid.sy ||
      fz >= grid.sz) {
    return -1;
  }
  return static_cast<long>(fx) +
         static_cast<long>(grid.sx) *
             (static_cast<long>(fy) + static_cast<long>(grid.sy) * static_cast<long>(fz));
}

namespace {

bool grid_lattice_sound(const VoxelGrid& grid) noexcept {
  return grid.sx > 0 && grid.sy > 0 && grid.sz > 0 && grid.resolution > 0.0 &&
         std::isfinite(grid.resolution);
}

}  // namespace

bool workspace_cell_explained(const VoxelGrid& grid, const Vec3& center) noexcept {
  const WorkspaceExplanation* e = grid.workspace_explained;
  if (e == nullptr || !std::isfinite(center.x) || !std::isfinite(center.y) ||
      !std::isfinite(center.z) || !grid_lattice_sound(grid)) {
    return false;
  }
  const double half_cell = 0.5 * grid.resolution;
  constexpr double kTolM = 1e-9;  // float slack on the layer bounds only
  const std::size_t n = std::min(e->n_objects, kMaxTrackedObjects);
  for (std::size_t i = 0; i < n; ++i) {
    // A tracked object's containing box.
    if (point_in_obb(e->box[i], e->half_extents[i], center)) {
      return true;
    }
    // The measured support's top layer (or the one under it), near the object.
    const double s = e->support_z[i];
    if (std::isfinite(s) && center.z >= s - half_cell - kTolM &&
        center.z <= s + half_cell + kTolM &&
        footprint_gap(e->box[i], e->half_extents[i], center) <= kExplainedSupportReachM) {
      return true;
    }
  }
  // A declared static collider's cell, or a cell of the connected support
  // layer, on this grid's own lattice.
  const auto cells = static_cast<std::size_t>(grid.sx) * static_cast<std::size_t>(grid.sy) *
                     static_cast<std::size_t>(grid.sz);
  const bool static_on = e->static_cells != nullptr && e->static_cells_n == cells;
  const bool support_on = e->support_cells != nullptr && e->support_cells_n == cells;
  if (!static_on && !support_on) {
    return false;
  }
  const long idx = grid_cell_of(grid, inverse_rigid(grid.pose), center);
  if (idx < 0) {
    return false;
  }
  const auto i = static_cast<std::size_t>(idx);
  return (static_on && e->static_cells[i] != 0) || (support_on && (e->support_cells[i] & 1U) != 0);
}

std::size_t mark_support_cells(const VoxelGrid& grid, const WorkspaceExplanation& e,
                               std::uint8_t* mask, std::size_t mask_n, std::uint32_t* queue,
                               std::size_t queue_n) noexcept {
  if (mask == nullptr || !grid_lattice_sound(grid)) {
    return 0;
  }
  const auto cells = static_cast<std::size_t>(grid.sx) * static_cast<std::size_t>(grid.sy) *
                     static_cast<std::size_t>(grid.sz);
  std::fill(mask, mask + std::min(mask_n, cells), std::uint8_t{0});
  const WorkspaceContactRegion& ws = grid.workspace_region;
  if (mask_n < cells || queue == nullptr || queue_n < cells || grid.occupancy == nullptr ||
      !ws.valid || cells > std::numeric_limits<std::uint32_t>::max()) {
    return 0;
  }
  constexpr std::uint8_t kMarked = 1U;
  constexpr std::uint8_t kVisited = 2U;  // this object's pass only
  constexpr double kTolM = 1e-9;         // the predicate's layer slack
  const double half_cell = 0.5 * grid.resolution;
  const auto sx = static_cast<std::size_t>(grid.sx);
  const auto sxy = sx * static_cast<std::size_t>(grid.sy);
  const std::size_t n = std::min(e.n_objects, kMaxTrackedObjects);
  std::size_t marked = 0;
  for (std::size_t o = 0; o < n; ++o) {
    const double s = e.support_z[o];
    if (!std::isfinite(s)) {
      continue;
    }
    const auto in_layer = [&](const Vec3& c) {
      return c.z >= s - half_cell - kTolM && c.z <= s + half_cell + kTolM &&
             point_in_obb(ws.pose, ws.half_extents, c);
    };
    // Seeds: the occupied cells the reach rule already explains as this support.
    std::size_t head = 0;
    std::size_t tail = 0;
    for (int iz = 0; iz < grid.sz; ++iz) {
      for (int iy = 0; iy < grid.sy; ++iy) {
        for (int ix = 0; ix < grid.sx; ++ix) {
          const std::size_t idx = static_cast<std::size_t>(ix) + sx * static_cast<std::size_t>(iy) +
                                  sxy * static_cast<std::size_t>(iz);
          if (grid.occupancy[idx] == 0) {
            continue;
          }
          const Vec3 c = voxel_center(grid, ix, iy, iz);
          if (in_layer(c) &&
              footprint_gap(e.box[o], e.half_extents[o], c) <= kExplainedSupportReachM) {
            mask[idx] |= kVisited;
            queue[tail++] = static_cast<std::uint32_t>(idx);
          }
        }
      }
    }
    // Grow through occupied in-layer cells inside the workspace, stepping over at most
    // one empty cell sideways (`kSupportBridgeCells`) and one layer up or down.
    constexpr int kB = static_cast<int>(kSupportBridgeCells);
    while (head < tail) {
      const std::size_t idx = queue[head++];
      const int ix = static_cast<int>(idx % sx);
      const int iy = static_cast<int>((idx % sxy) / sx);
      const int iz = static_cast<int>(idx / sxy);
      for (int dz = -1; dz <= 1; ++dz) {
        for (int dy = -kB; dy <= kB; ++dy) {
          for (int dx = -kB; dx <= kB; ++dx) {
            const int jx = ix + dx;
            const int jy = iy + dy;
            const int jz = iz + dz;
            if (jx < 0 || jy < 0 || jz < 0 || jx >= grid.sx || jy >= grid.sy || jz >= grid.sz) {
              continue;
            }
            const std::size_t j = static_cast<std::size_t>(jx) + sx * static_cast<std::size_t>(jy) +
                                  sxy * static_cast<std::size_t>(jz);
            if (grid.occupancy[j] == 0 || (mask[j] & kVisited) != 0 ||
                !in_layer(voxel_center(grid, jx, jy, jz))) {
              continue;
            }
            mask[j] |= kVisited;
            queue[tail++] = static_cast<std::uint32_t>(j);
          }
        }
      }
    }
    // Every cell this pass reached is marked; its visited bit is cleared.
    for (std::size_t q = 0; q < tail; ++q) {
      std::uint8_t& m = mask[queue[q]];
      marked += (m & kMarked) == 0 ? 1 : 0;
      m = kMarked;
    }
  }
  return marked;
}

std::size_t mark_static_cells(const VoxelGrid& grid, const std::vector<Vec3>& base_points,
                              std::uint8_t* mask, std::size_t mask_n,
                              std::size_t pad_cells) noexcept {
  if (mask == nullptr || !grid_lattice_sound(grid)) {
    return 0;
  }
  const auto cells = static_cast<std::size_t>(grid.sx) * static_cast<std::size_t>(grid.sy) *
                     static_cast<std::size_t>(grid.sz);
  if (mask_n < cells) {
    return 0;
  }
  std::fill(mask, mask + cells, std::uint8_t{0});
  const Transform grid_from_base = inverse_rigid(grid.pose);
  std::size_t marked = 0;
  for (const Vec3& p : base_points) {
    const long idx = grid_cell_of(grid, grid_from_base, p);
    if (idx >= 0 && mask[static_cast<std::size_t>(idx)] == 0) {
      mask[static_cast<std::size_t>(idx)] = 1;
      ++marked;
    }
  }
  if (pad_cells > 0 && marked > 0) {
    // Grow the declared cells (value 1) into 2s first, so the growth never chains: a cell
    // padded in this pass is not itself a seed of the same pass.
    const long sx = static_cast<long>(grid.sx);
    const long sy = static_cast<long>(grid.sy);
    const long sz = static_cast<long>(grid.sz);
    const long r = static_cast<long>(pad_cells);
    for (long k = 0; k < sz; ++k) {
      for (long j = 0; j < sy; ++j) {
        for (long i = 0; i < sx; ++i) {
          if (mask[static_cast<std::size_t>(i + sx * (j + sy * k))] != 1) {
            continue;
          }
          for (long dk = -r; dk <= r; ++dk) {
            for (long dj = -r; dj <= r; ++dj) {
              for (long di = -r; di <= r; ++di) {
                const long ni = i + di;
                const long nj = j + dj;
                const long nk = k + dk;
                if (ni < 0 || nj < 0 || nk < 0 || ni >= sx || nj >= sy || nk >= sz) {
                  continue;
                }
                std::uint8_t& cell = mask[static_cast<std::size_t>(ni + sx * (nj + sy * nk))];
                if (cell == 0) {
                  cell = 2;
                }
              }
            }
          }
        }
      }
    }
    for (std::size_t c = 0; c < cells; ++c) {
      if (mask[c] == 2) {
        mask[c] = 1;
      }
    }
  }
  return marked;
}

}  // namespace openral_safety_kernel
