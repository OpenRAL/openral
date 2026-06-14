// SPDX-License-Identifier: Apache-2.0
// ADR-0030 phase 2 — allocation-free geometric self-collision core.
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

// Distance from a point to the segment [a, b].
double point_segment_distance(const Vec3& p, const Vec3& a, const Vec3& b) noexcept {
  const Vec3 ab = sub(b, a);
  const double denom = dot(ab, ab);
  const double t = denom > 1e-12 ? clamp01(dot(sub(p, a), ab) / denom) : 0.0;
  const Vec3 diff = sub(p, add(a, scale(ab, t)));
  return std::sqrt(dot(diff, diff));
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

CollisionHit check_self_collision(const CollisionModel& model, const CollisionScratch& scratch,
                                  double margin) noexcept {
  CollisionHit result;
  result.min_distance = std::numeric_limits<double>::infinity();
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
      if (d < result.min_distance) {
        result.min_distance = d;
      }
      if (d <= margin && !result.hit) {
        result.hit = true;
        result.link_a = li;
        result.link_b = lj;
      }
    }
  }
  return result;
}

CollisionHit check_world_collision(const CollisionModel& model, const CollisionScratch& scratch,
                                   const WorldModel& world, double margin) noexcept {
  CollisionHit result;
  result.min_distance = std::numeric_limits<double>::infinity();
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
      if (d < result.min_distance) {
        result.min_distance = d;
      }
      if (d <= margin && !result.hit) {
        result.hit = true;
        result.link_a = li;                   // robot link
        result.link_b = static_cast<int>(w);  // world obstacle index
      }
    }
  }
  return result;
}

CollisionHit check_voxel_collision(const CollisionModel& model, const CollisionScratch& scratch,
                                   const VoxelGrid& grid, double margin) noexcept {
  CollisionHit result;
  result.min_distance = std::numeric_limits<double>::infinity();
  if (grid.occupancy == nullptr || grid.sx <= 0 || grid.sy <= 0 || grid.sz <= 0 ||
      grid.resolution <= 0.0) {
    return result;
  }
  // Occupied voxels are treated conservatively as spheres of the voxel
  // half-diagonal at the cell centre (so a capsule grazing any part of the box
  // counts), and only voxels inside each capsule's inflated AABB are tested.
  const double half_diag = grid.resolution * 0.86602540378443864676;  // sqrt(3)/2
  const double inv_res = 1.0 / grid.resolution;
  const std::size_t n_caps = model.capsules.size();
  for (std::size_t c = 0; c < n_caps; ++c) {
    const int li = model.capsule_link[c];
    const Transform cap =
        compose(scratch.link_world[static_cast<std::size_t>(li)], model.capsules[c].origin);
    Vec3 p0;
    Vec3 p1;
    capsule_endpoints(cap, model.capsules[c].half_length, p0, p1);
    const double r = model.capsules[c].radius;
    const double reach = r + margin + half_diag;
    const auto rng = [&](double lo, double hi, double org, int dim) {
      const int i0 = static_cast<int>(std::floor((lo - org) * inv_res));
      const int i1 = static_cast<int>(std::floor((hi - org) * inv_res));
      return std::pair<int, int>{clamp_index(i0, 0, dim - 1), clamp_index(i1, 0, dim - 1)};
    };
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
          const double d = point_segment_distance(center, p0, p1) - r - half_diag;
          if (d < result.min_distance) {
            result.min_distance = d;
          }
          if (d <= margin && !result.hit) {
            result.hit = true;
            result.link_a = li;
            result.link_b = static_cast<int>(idx);
          }
        }
      }
    }
  }
  return result;
}

}  // namespace openral_safety_kernel
