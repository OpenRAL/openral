// SPDX-License-Identifier: Apache-2.0
// External joint-torque observer (#373): inverse dynamics against closed-form
// pendulum dynamics, and the observer against a simulated plant with a known
// external torque, at the real OpenArm feedback rate and at a deploy-sim
// control rate.

#include "openral_safety_kernel/force_observer.hpp"

#include <array>
#include <cmath>
#include <vector>

#include <gtest/gtest.h>

namespace osk = openral_safety_kernel;

namespace {

constexpr double kG = 9.81;

// Planar double pendulum hanging off a fixed root, both joints about +y, links
// along +x. Link 1 COM at lc1 on its x axis, link 2 at lc2; link 2's joint sits
// at l1 along link 1's x axis. Per-link inertia about the COM is iyy only.
struct Pendulum {
  double m1{1.3}, m2{0.7}, l1{0.4}, lc1{0.2}, lc2{0.25}, i1{0.011}, i2{0.006};

  osk::CollisionModel model() const {
    osk::CollisionModel m;
    m.n_links = 3;
    m.parent = {-1, 0, 1};
    m.joint_kind = {osk::JointKind::kFixed, osk::JointKind::kRevolute, osk::JointKind::kRevolute};
    m.dof_index = {-1, 0, 1};
    m.origin = {osk::Transform{}, osk::Transform{}, osk::Transform{}};
    m.origin[2].t = osk::Vec3{l1, 0.0, 0.0};
    m.axis = {osk::Vec3{0, 0, 1}, osk::Vec3{0, 1, 0}, osk::Vec3{0, 1, 0}};
    return m;
  }

  osk::ForceObserverModel dynamics() const {
    osk::ForceObserverModel d;
    d.links.resize(3);
    d.joints.resize(3);
    d.links[1].mass = m1;
    d.links[1].com = osk::Vec3{lc1, 0.0, 0.0};
    d.links[1].inertia[4] = i1;
    d.links[2].mass = m2;
    d.links[2].com = osk::Vec3{lc2, 0.0, 0.0};
    d.links[2].inertia[4] = i2;
    d.gravity = osk::Vec3{0.0, 0.0, -kG};
    return d;
  }

  // Closed form. Rotating about +y by q takes x to (cos q, 0, -sin q), so the
  // planar angle in the x-z plane (z up) is θ = -q; M and the velocity terms
  // are even in that sign flip, gravity is derived from the heights directly.
  std::array<double, 4> mass(double q2) const {
    const double c2 = std::cos(q2);
    const double m11 = m1 * lc1 * lc1 + i1 + m2 * (l1 * l1 + lc2 * lc2 + 2 * l1 * lc2 * c2) + i2;
    const double m12 = m2 * (lc2 * lc2 + l1 * lc2 * c2) + i2;
    const double m22 = m2 * lc2 * lc2 + i2;
    return {m11, m12, m12, m22};
  }
  std::array<double, 2> bias(double q1, double q2, double v1, double v2) const {
    const double h = m2 * l1 * lc2 * std::sin(q2);
    // Heights: z1 = -lc1 sin q1, z2 = -l1 sin q1 - lc2 sin(q1 + q2); τ_g = dV/dq.
    const double g1 = -kG * (m1 * lc1 + m2 * l1) * std::cos(q1) - kG * m2 * lc2 * std::cos(q1 + q2);
    const double g2 = -kG * m2 * lc2 * std::cos(q1 + q2);
    return {-h * (2 * v1 * v2 + v2 * v2) + g1, h * v1 * v1 + g2};
  }
};

TEST(ForceObserverDynamics, RneaMatchesClosedFormDoublePendulum) {
  const Pendulum p;
  const auto model = p.model();
  const auto dyn = p.dynamics();
  osk::ForceObserverState state;
  osk::init_force_observer(model, 2, state);
  for (const auto& s : std::vector<std::array<double, 6>>{{0.3, -0.7, 1.1, -0.4, 2.0, -3.0},
                                                          {-1.2, 1.9, -0.6, 2.2, 0.0, 1.0}}) {
    const double q[2] = {s[0], s[1]};
    const double qd[2] = {s[2], s[3]};
    const double qdd[2] = {s[4], s[5]};
    double tau[2];
    osk::rnea(model, dyn, q, qd, qdd, true, true, state, tau);
    const auto m = p.mass(q[1]);
    const auto b = p.bias(q[0], q[1], qd[0], qd[1]);
    EXPECT_NEAR(tau[0], m[0] * qdd[0] + m[1] * qdd[1] + b[0], 1e-9);
    EXPECT_NEAR(tau[1], m[2] * qdd[0] + m[3] * qdd[1] + b[1], 1e-9);

    osk::mass_matrix(model, dyn, q, state);
    for (int k = 0; k < 4; ++k) {
      EXPECT_NEAR(state.mass_matrix[static_cast<std::size_t>(k)], m[static_cast<std::size_t>(k)],
                  1e-12);
    }
  }
}

TEST(ForceObserverDynamics, MimicLinkProjectsThroughItsMultiplier) {
  // A follower link mimicking joint 0 at multiplier -2: its torque lands on
  // column 0 scaled by -2, and it adds -2 * (-2) = 4x its own inertia to M.
  osk::CollisionModel m;
  m.n_links = 3;
  m.parent = {-1, 0, 0};
  m.joint_kind = {osk::JointKind::kFixed, osk::JointKind::kRevolute,
                  osk::JointKind::kMimicRevolute};
  m.dof_index = {-1, 0, 0};
  m.origin = {osk::Transform{}, osk::Transform{}, osk::Transform{}};
  m.axis = {osk::Vec3{0, 0, 1}, osk::Vec3{0, 0, 1}, osk::Vec3{0, 0, 1}};
  m.mimic_multiplier = {1.0, 1.0, -2.0};
  m.mimic_offset = {0.0, 0.0, 0.0};
  osk::ForceObserverModel d;
  d.links.resize(3);
  d.joints.resize(3);
  d.links[1].inertia[8] = 0.5;
  d.links[2].inertia[8] = 0.25;
  osk::ForceObserverState state;
  osk::init_force_observer(m, 1, state);
  const double q[1] = {0.4};
  osk::forward_kinematics(m, q, 1, state.fk);
  osk::mass_matrix(m, d, q, state);
  EXPECT_NEAR(state.mass_matrix[0], 0.5 + 4.0 * 0.25, 1e-12);
}

// Plant: the closed-form pendulum plus joint friction, under PD tracking of a
// sinusoid, stepped at 1 kHz with semi-implicit Euler. An external torque
// `ext` acts from `t_on`. The observer sees (q, q̇, motor torque) every
// `stride` steps.
struct Run {
  std::vector<double> t, r0, r1;
};

Run simulate(const Pendulum& p, const osk::ForceObserverModel& dyn, int stride,
             const std::array<double, 2>& ext, double t_on, double t_end,
             const std::array<double, 2>& plant_coulomb) {
  const auto model = p.model();
  osk::ForceObserverState state;
  osk::init_force_observer(model, 2, state);
  const double h = 1e-3;
  double q[2] = {0.2, -0.4}, v[2] = {0.0, 0.0};
  Run out;
  for (int k = 0; k * h < t_end; ++k) {
    const double t = k * h;
    const double target[2] = {0.2 + 0.6 * std::sin(1.5 * t), -0.4 + 0.5 * std::sin(2.3 * t)};
    double tau_m[2];
    for (int j = 0; j < 2; ++j) {
      tau_m[j] = 60.0 * (target[j] - q[j]) - 4.0 * v[j];
    }
    if (k % stride == 0) {
      osk::force_observer_update(model, dyn, q, v, tau_m, h * stride, state);
      out.t.push_back(t);
      out.r0.push_back(state.residual[0]);
      out.r1.push_back(state.residual[1]);
    }
    const auto m = p.mass(q[1]);
    const auto b = p.bias(q[0], q[1], v[0], v[1]);
    double rhs[2];
    for (std::size_t j = 0; j < 2; ++j) {
      const double friction = plant_coulomb[j] * std::tanh(v[j] / dyn.coulomb_velocity_rad_s);
      rhs[j] = tau_m[j] - friction - b[j] + (t >= t_on ? ext[j] : 0.0);
    }
    const double det = m[0] * m[3] - m[1] * m[2];
    const double a0 = (m[3] * rhs[0] - m[1] * rhs[1]) / det;
    const double a1 = (m[0] * rhs[1] - m[2] * rhs[0]) / det;
    v[0] += a0 * h;
    v[1] += a1 * h;
    q[0] += v[0] * h;
    q[1] += v[1] * h;
  }
  return out;
}

double max_abs_between(const Run& run, const std::vector<double>& r, double t0, double t1,
                       double ref) {
  double worst = 0.0;
  for (std::size_t i = 0; i < run.t.size(); ++i) {
    if (run.t[i] >= t0 && run.t[i] < t1) {
      worst = std::max(worst, std::abs(r[i] - ref));
    }
  }
  return worst;
}

TEST(ForceObserver, TracksAnExternalTorqueStepWhileMoving) {
  const Pendulum p;
  auto dyn = p.dynamics();
  dyn.gain_per_s = 40.0;
  const auto run = simulate(p, dyn, 1, {0.0, 1.5}, 1.0, 1.6, {0.0, 0.0});
  // Before the push: only discretisation error of the 1 kHz plant.
  EXPECT_LT(max_abs_between(run, run.r0, 0.1, 1.0, 0.0), 0.05);
  EXPECT_LT(max_abs_between(run, run.r1, 0.1, 1.0, 0.0), 0.05);
  // Five time constants after the step it reads the applied torque.
  EXPECT_LT(max_abs_between(run, run.r1, 1.0 + 5.0 / 40.0, 1.6, 1.5), 0.05);
  EXPECT_LT(max_abs_between(run, run.r0, 1.0 + 5.0 / 40.0, 1.6, 0.0), 0.05);
}

TEST(ForceObserver, StableAndConvergentAtADeploySimControlRate) {
  // 30 Hz samples with K = 50/s: K·dt = 1.7, which a forward-Euler observer
  // would ring on. The exact discretisation must still converge.
  const Pendulum p;
  auto dyn = p.dynamics();
  dyn.gain_per_s = 50.0;
  const auto run = simulate(p, dyn, 33, {-2.0, 0.0}, 1.0, 2.5, {0.0, 0.0});
  EXPECT_LT(max_abs_between(run, run.r0, 0.2, 1.0, 0.0), 0.35);
  EXPECT_LT(max_abs_between(run, run.r0, 1.3, 2.5, -2.0), 0.35);
}

TEST(ForceObserver, ModelledFrictionIsNotReadAsContact) {
  const Pendulum p;
  auto dyn = p.dynamics();
  dyn.gain_per_s = 40.0;
  dyn.joints[1].coulomb = 0.6;  // dof 0's joint drives link 1
  dyn.joints[2].coulomb = 0.3;
  const auto modelled = simulate(p, dyn, 1, {0.0, 0.0}, 9.0, 1.5, {0.6, 0.3});
  auto blind = dyn;
  blind.joints[1].coulomb = 0.0;
  blind.joints[2].coulomb = 0.0;
  const auto unmodelled = simulate(p, blind, 1, {0.0, 0.0}, 9.0, 1.5, {0.6, 0.3});
  EXPECT_LT(max_abs_between(modelled, modelled.r0, 0.2, 1.5, 0.0), 0.05);
  EXPECT_GT(max_abs_between(unmodelled, unmodelled.r0, 0.2, 1.5, 0.0), 0.5);
}

TEST(ForceObserver, NonFiniteInputResetsInsteadOfPoisoning) {
  const Pendulum p;
  const auto model = p.model();
  const auto dyn = p.dynamics();
  osk::ForceObserverState state;
  osk::init_force_observer(model, 2, state);
  const double q[2] = {0.1, 0.2}, v[2] = {0.0, 0.0}, tau[2] = {1.0, 1.0};
  ASSERT_TRUE(osk::force_observer_update(model, dyn, q, v, tau, 0.002, state));
  ASSERT_TRUE(osk::force_observer_update(model, dyn, q, v, tau, 0.002, state));
  const double bad[2] = {std::nan(""), 0.0};
  EXPECT_FALSE(osk::force_observer_update(model, dyn, q, v, bad, 0.002, state));
  EXPECT_FALSE(state.initialized);
  EXPECT_EQ(state.residual[0], 0.0);
  EXPECT_FALSE(osk::force_observer_update(model, dyn, q, v, tau, 0.0, state));
}

}  // namespace
