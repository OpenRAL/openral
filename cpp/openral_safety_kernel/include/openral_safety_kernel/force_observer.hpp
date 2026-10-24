// SPDX-License-Identifier: Apache-2.0
// Allocation-free external joint-torque observer (issue #373).
//
// A generalized-momentum observer over the kernel's own kinematic tree
// (`CollisionModel`, the same one the self-collision FK walks) plus per-link
// rigid-body inertials and a per-joint friction model. Inputs are the measured
// joint positions, velocities and efforts; output is the per-joint external
// torque estimate r ≈ τ_ext, which the kernel compares against per-joint
// ceilings (the collision tier).
//
// Dynamics: M(q) q̈ + C(q, q̇) q̇ + g(q) + τ_f(q̇) = τ_m + τ_ext.
// With p = M q̇ and Ṁ = C + Cᵀ, ṗ = τ_m − τ_f + τ_ext − β, β = g − Cᵀq̇. The
// observer ṙ = K (τ_ext − r) is discretised exactly, so it is stable for every
// sample period (the deploy-sim HAL publishes at the control rate, the real
// OpenArm at ~733 Hz):
//
//   ŵ_k = (p_k − p_{k−1}) / dt − ½(τ_k + τ_{k−1}) + ½(β_k + β_{k−1})
//   r_k = r_{k−1} + (1 − e^{−K dt}) (ŵ_k − r_{k−1})
//
// M comes from n RNEA passes with unit accelerations, g + C q̇ from one pass
// with the measured velocity, and Cᵀq̇ = Ṁq̇ − C q̇ with Ṁ by finite difference.
// No external dependency (no Eigen / KDL / Pinocchio), matching collision.hpp.
// Every buffer is sized once in `init_force_observer`; `force_observer_update`
// never allocates.

#pragma once

#include "openral_safety_kernel/collision.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace openral_safety_kernel {

/// Rigid-body inertial of one link, in that link's frame.
struct LinkInertial {
  double mass{0.0};  ///< kg
  Vec3 com{};        ///< centre of mass in the link frame (m)
  /// Inertia tensor about the COM, link-frame axes, row-major 3x3 (kg·m²).
  double inertia[9]{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
};

/// Friction + rotor model of the joint that connects a link to its parent.
/// τ_f(q̇) = offset + coulomb · tanh(q̇ / coulomb_velocity) + viscous · q̇, and
/// `armature` adds to M's diagonal (rotor inertia reflected through the gear).
struct JointFrictionModel {
  double armature{0.0};  ///< kg·m² (revolute) or kg (prismatic)
  double viscous{0.0};   ///< N·m·s/rad
  double coulomb{0.0};   ///< N·m
  double offset{0.0};    ///< N·m, constant effort-reading bias
};

/// Everything the observer needs beyond the kinematic tree. Built once at
/// configure time; read-only afterwards.
struct ForceObserverModel {
  std::vector<LinkInertial> links;         ///< parallel to CollisionModel links
  std::vector<JointFrictionModel> joints;  ///< parallel to links (joint into link i)
  Vec3 gravity{0.0, 0.0, -9.81};           ///< m/s², in the kinematic root frame
  double gain_per_s{50.0};                 ///< observer bandwidth K (1/s)
  double coulomb_velocity_rad_s{0.05};     ///< tanh smoothing of the Coulomb term
};

/// Observer state + scratch. `n_dof` is the action's dof count (the kernel's
/// column space); dofs no link drives stay zero.
struct ForceObserverState {
  std::size_t n_dof{0};
  bool initialized{false};
  // Per-link RNEA scratch, base-frame coordinates.
  std::vector<Vec3> omega, alpha, acc, force, moment, joint_axis;
  CollisionScratch fk;        ///< FK output for the sample being processed
  std::vector<int> dof_link;  ///< per dof: the non-mimic link its joint drives, or -1
  // Per-dof.
  std::vector<double> mass_matrix;                      ///< n_dof × n_dof, row-major
  std::vector<double> mass_matrix_prev;                 ///< previous sample's M
  std::vector<double> momentum_prev;                    ///< p_{k−1}
  std::vector<double> drive_prev;                       ///< τ_{k−1} − β_{k−1}
  std::vector<double> residual;                         ///< r_k, the τ_ext estimate (N·m)
  std::vector<double> scratch_a, scratch_b, scratch_c;  ///< RNEA in/out columns
};

/// Size every buffer for `model` × `n_dof`. Not on the hot path.
void init_force_observer(const CollisionModel& model, std::size_t n_dof, ForceObserverState& state);

/// Inverse dynamics τ = M(q) q̈ + C(q, q̇) q̇ + [g(q) when `with_gravity`]
/// plus armature · q̈ (no friction). `q`, `qd`, `qdd`, `tau` are n_dof long.
/// Runs FK into `state.fk` first when `run_fk`. Allocation-free.
void rnea(const CollisionModel& model, const ForceObserverModel& dyn, const double* q,
          const double* qd, const double* qdd, bool with_gravity, bool run_fk,
          ForceObserverState& state, double* tau) noexcept;

/// Joint-space inertia matrix into `state.mass_matrix` (n RNEA passes; FK must
/// already be in `state.fk` for `q`). Allocation-free.
void mass_matrix(const CollisionModel& model, const ForceObserverModel& dyn, const double* q,
                 ForceObserverState& state) noexcept;

/// Friction torque the model predicts for joint velocity `qd` of `joint`.
double friction_torque(const ForceObserverModel& dyn, const JointFrictionModel& joint,
                       double qd) noexcept;

/// One observer sample. `effort[d]` is the measured joint effort, already in
/// the action's dof order. `dt_s` is the time since the previous sample
/// (source clock). The first call after `reset_force_observer` only seeds
/// the state and leaves `residual` at zero. Returns false (and resets) on a
/// non-finite input or a non-positive `dt_s`. Allocation-free.
bool force_observer_update(const CollisionModel& model, const ForceObserverModel& dyn,
                           const double* q, const double* qd, const double* effort, double dt_s,
                           ForceObserverState& state) noexcept;

/// Forget the history: the next sample re-seeds and the residual reads zero.
void reset_force_observer(ForceObserverState& state) noexcept;

}  // namespace openral_safety_kernel
