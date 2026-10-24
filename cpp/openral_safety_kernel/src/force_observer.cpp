// SPDX-License-Identifier: Apache-2.0
#include "openral_safety_kernel/force_observer.hpp"

#include <algorithm>
#include <cmath>

namespace openral_safety_kernel {

namespace {

Vec3 add(const Vec3& a, const Vec3& b) noexcept { return Vec3{a.x + b.x, a.y + b.y, a.z + b.z}; }
Vec3 sub(const Vec3& a, const Vec3& b) noexcept { return Vec3{a.x - b.x, a.y - b.y, a.z - b.z}; }
Vec3 scale(const Vec3& a, double s) noexcept { return Vec3{a.x * s, a.y * s, a.z * s}; }
double dot(const Vec3& a, const Vec3& b) noexcept { return a.x * b.x + a.y * b.y + a.z * b.z; }
Vec3 cross(const Vec3& a, const Vec3& b) noexcept {
  return Vec3{a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
Vec3 mul(const double r[9], const Vec3& v) noexcept {
  return Vec3{r[0] * v.x + r[1] * v.y + r[2] * v.z, r[3] * v.x + r[4] * v.y + r[5] * v.z,
              r[6] * v.x + r[7] * v.y + r[8] * v.z};
}
Vec3 mul_t(const double r[9], const Vec3& v) noexcept {
  return Vec3{r[0] * v.x + r[3] * v.y + r[6] * v.z, r[1] * v.x + r[4] * v.y + r[7] * v.z,
              r[2] * v.x + r[5] * v.y + r[8] * v.z};
}

bool revolute(JointKind kind) noexcept {
  return kind == JointKind::kRevolute || kind == JointKind::kMimicRevolute;
}
bool prismatic(JointKind kind) noexcept {
  return kind == JointKind::kPrismatic || kind == JointKind::kMimicPrismatic;
}

/// d(link joint value)/d(its dof column): the mimic multiplier, else 1.
double column_gain(const CollisionModel& model, std::size_t i) noexcept {
  if (is_mimic(model.joint_kind[i]) && i < model.mimic_multiplier.size()) {
    return model.mimic_multiplier[i];
  }
  return 1.0;
}

}  // namespace

void init_force_observer(const CollisionModel& model, std::size_t n_dof,
                         ForceObserverState& state) {
  const std::size_t n = model.n_links;
  state.n_dof = n_dof;
  state.initialized = false;
  for (auto* v :
       {&state.omega, &state.alpha, &state.acc, &state.force, &state.moment, &state.joint_axis}) {
    v->assign(n, Vec3{});
  }
  state.fk.link_world.assign(n, Transform{});
  state.dof_link.assign(n_dof, -1);
  for (std::size_t i = 0; i < n; ++i) {
    const int d = model.dof_index[i];
    if (d >= 0 && static_cast<std::size_t>(d) < n_dof && !is_mimic(model.joint_kind[i]) &&
        model.joint_kind[i] != JointKind::kFixed) {
      state.dof_link[static_cast<std::size_t>(d)] = static_cast<int>(i);
    }
  }
  state.mass_matrix.assign(n_dof * n_dof, 0.0);
  state.mass_matrix_prev.assign(n_dof * n_dof, 0.0);
  for (auto* v : {&state.momentum_prev, &state.drive_prev, &state.residual, &state.scratch_a,
                  &state.scratch_b, &state.scratch_c}) {
    v->assign(n_dof, 0.0);
  }
}

void rnea(const CollisionModel& model, const ForceObserverModel& dyn, const double* q,
          const double* qd, const double* qdd, bool with_gravity, bool run_fk,
          ForceObserverState& state, double* tau) noexcept {
  const std::size_t n_dof = state.n_dof;
  if (run_fk) {
    forward_kinematics(model, q, n_dof, state.fk);
  }
  for (std::size_t d = 0; d < n_dof; ++d) {
    tau[d] = 0.0;
  }
  // Forward pass: link origin velocities/accelerations, base-frame coordinates.
  // Gravity enters as the root's fictitious upward acceleration.
  const Vec3 root_acc = with_gravity ? scale(dyn.gravity, -1.0) : Vec3{};
  for (std::size_t i = 0; i < model.n_links; ++i) {
    const Transform& xf = state.fk.link_world[i];
    const int parent = model.parent[i];
    Vec3 omega_p{}, alpha_p{}, acc_p = root_acc, origin_p = xf.t;
    if (parent >= 0) {
      const auto pi = static_cast<std::size_t>(parent);
      omega_p = state.omega[pi];
      alpha_p = state.alpha[pi];
      acc_p = state.acc[pi];
      origin_p = state.fk.link_world[pi].t;
    }
    const Vec3 r = sub(xf.t, origin_p);
    Vec3 acc = add(acc_p, add(cross(alpha_p, r), cross(omega_p, cross(omega_p, r))));
    Vec3 omega = omega_p;
    Vec3 alpha = alpha_p;
    const int dof = model.dof_index[i];
    const JointKind kind = model.joint_kind[i];
    const Vec3 z = mul(xf.r, model.axis[i]);  // the motion leaves the axis fixed
    if (dof >= 0 && static_cast<std::size_t>(dof) < n_dof && kind != JointKind::kFixed) {
      const double g = column_gain(model, i);
      const double v = g * qd[dof];
      const double a = g * qdd[dof];
      if (revolute(kind)) {
        omega = add(omega_p, scale(z, v));
        alpha = add(alpha_p, add(scale(z, a), cross(omega_p, scale(z, v))));
      } else if (prismatic(kind)) {
        acc = add(acc, add(scale(z, a), scale(cross(omega_p, z), 2.0 * v)));
      }
    }
    state.omega[i] = omega;
    state.alpha[i] = alpha;
    state.acc[i] = acc;
    state.joint_axis[i] = z;

    const LinkInertial& body = dyn.links[i];
    const Vec3 c = mul(xf.r, body.com);  // origin -> COM, base frame
    const Vec3 acc_c = add(acc, add(cross(alpha, c), cross(omega, cross(omega, c))));
    const Vec3 f = scale(acc_c, body.mass);
    // I_base · v = R · I_local · Rᵀ · v.
    const Vec3 i_alpha = mul(xf.r, mul(body.inertia, mul_t(xf.r, alpha)));
    const Vec3 i_omega = mul(xf.r, mul(body.inertia, mul_t(xf.r, omega)));
    state.force[i] = f;
    state.moment[i] = add(add(i_alpha, cross(omega, i_omega)), cross(c, f));
  }
  // Backward pass: accumulate each subtree's wrench at its joint and project.
  for (std::size_t k = model.n_links; k-- > 0;) {
    const Vec3& f = state.force[k];
    const Vec3& m = state.moment[k];
    const int dof = model.dof_index[k];
    const JointKind kind = model.joint_kind[k];
    if (dof >= 0 && static_cast<std::size_t>(dof) < n_dof && kind != JointKind::kFixed) {
      const double g = column_gain(model, k);
      const Vec3& z = state.joint_axis[k];
      tau[dof] += g * (revolute(kind) ? dot(z, m) : dot(z, f));
      if (!is_mimic(kind)) {
        tau[dof] += dyn.joints[k].armature * qdd[dof];
      }
    }
    const int parent = model.parent[k];
    if (parent >= 0) {
      const auto pi = static_cast<std::size_t>(parent);
      const Vec3 lever = sub(state.fk.link_world[k].t, state.fk.link_world[pi].t);
      state.force[pi] = add(state.force[pi], f);
      state.moment[pi] = add(state.moment[pi], add(m, cross(lever, f)));
    }
  }
}

void mass_matrix(const CollisionModel& model, const ForceObserverModel& dyn, const double* q,
                 ForceObserverState& state) noexcept {
  const std::size_t n = state.n_dof;
  double* zero = state.scratch_a.data();
  double* unit = state.scratch_b.data();
  double* col = state.scratch_c.data();
  std::fill(zero, zero + n, 0.0);
  std::fill(unit, unit + n, 0.0);
  for (std::size_t j = 0; j < n; ++j) {
    if (state.dof_link[j] < 0) {
      for (std::size_t i = 0; i < n; ++i) {
        state.mass_matrix[i * n + j] = 0.0;
      }
      continue;
    }
    unit[j] = 1.0;
    rnea(model, dyn, q, zero, unit, /*with_gravity=*/false, /*run_fk=*/false, state, col);
    unit[j] = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
      state.mass_matrix[i * n + j] = col[i];
    }
  }
}

double friction_torque(const ForceObserverModel& dyn, const JointFrictionModel& joint,
                       double qd) noexcept {
  const double coulomb =
      dyn.coulomb_velocity_rad_s > 0.0 ? std::tanh(qd / dyn.coulomb_velocity_rad_s) : 0.0;
  return joint.offset + joint.coulomb * coulomb + joint.viscous * qd;
}

void reset_force_observer(ForceObserverState& state) noexcept {
  state.initialized = false;
  std::fill(state.residual.begin(), state.residual.end(), 0.0);
}

bool force_observer_update(const CollisionModel& model, const ForceObserverModel& dyn,
                           const double* q, const double* qd, const double* effort, double dt_s,
                           ForceObserverState& state) noexcept {
  const std::size_t n = state.n_dof;
  bool finite = std::isfinite(dt_s) && dt_s > 0.0;
  for (std::size_t d = 0; d < n && finite; ++d) {
    finite = std::isfinite(q[d]) && std::isfinite(qd[d]) && std::isfinite(effort[d]);
  }
  if (!finite) {
    reset_force_observer(state);
    return false;
  }
  forward_kinematics(model, q, n, state.fk);
  mass_matrix(model, dyn, q, state);  // clobbers scratch_a..c
  // h = C q̇ + g into scratch_a (scratch_c holds the zero q̈).
  double* h = state.scratch_a.data();
  double* p = state.scratch_b.data();
  double* zero = state.scratch_c.data();
  std::fill(zero, zero + n, 0.0);
  rnea(model, dyn, q, qd, zero, /*with_gravity=*/true, /*run_fk=*/false, state, h);
  for (std::size_t i = 0; i < n; ++i) {
    double acc = 0.0;
    for (std::size_t j = 0; j < n; ++j) {
      acc += state.mass_matrix[i * n + j] * qd[j];
    }
    p[i] = acc;
  }
  const double blend = 1.0 - std::exp(-dyn.gain_per_s * dt_s);
  for (std::size_t i = 0; i < n; ++i) {
    // β = g − Cᵀq̇ = h − Ṁ q̇ (Ṁ = C + Cᵀ); Ṁ by finite difference, 0 on the seed sample.
    double mdot_qd = 0.0;
    if (state.initialized) {
      for (std::size_t j = 0; j < n; ++j) {
        mdot_qd += (state.mass_matrix[i * n + j] - state.mass_matrix_prev[i * n + j]) * qd[j];
      }
      mdot_qd /= dt_s;
    }
    const int link = state.dof_link[i];
    const double friction =
        link >= 0 ? friction_torque(dyn, dyn.joints[static_cast<std::size_t>(link)], qd[i]) : 0.0;
    const double drive = (effort[i] - friction) - (h[i] - mdot_qd);
    if (state.initialized && link >= 0) {
      const double w = (p[i] - state.momentum_prev[i]) / dt_s - 0.5 * (drive + state.drive_prev[i]);
      state.residual[i] += blend * (w - state.residual[i]);
    } else {
      state.residual[i] = 0.0;
    }
    state.drive_prev[i] = drive;
    state.momentum_prev[i] = p[i];
  }
  state.mass_matrix_prev.swap(state.mass_matrix);
  state.initialized = true;
  return true;
}

}  // namespace openral_safety_kernel
