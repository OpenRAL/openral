// SPDX-License-Identifier: Apache-2.0
// ADR-0020 — gtest unit coverage for the allocation-free validator.
// Real EnvelopeIntersection structs (no mocks); real chunk views.

#include "openral_safety_kernel/validator.hpp"

#include <array>
#include <cmath>
#include <limits>
#include <vector>

#include <gtest/gtest.h>

namespace osk = openral_safety_kernel;

namespace {

osk::EnvelopeIntersection make_env(std::size_t n_dof = 3,
                                   double pos_lo = -1.0,
                                   double pos_hi = 1.0,
                                   double vel_max = 3.15,
                                   double tau_max = 5.0) {
  osk::EnvelopeIntersection env;
  env.robot_name = "toy";
  env.n_dof = n_dof;
  env.joint_position_min.assign(n_dof, pos_lo);
  env.joint_position_max.assign(n_dof, pos_hi);
  env.joint_velocity_max.assign(n_dof, vel_max);
  env.joint_torque_max.assign(n_dof, tau_max);
  env.workspace_box.set = true;
  env.workspace_box.min_xyz = {-0.4, -0.4, 0.0};
  env.workspace_box.max_xyz = {0.4, 0.4, 0.6};
  env.max_ee_speed_m_s = 0.5;
  env.max_ee_accel_m_s2 = 2.0;
  env.max_force_n = 10.0;
  env.max_torque_nm = 3.0;
  env.contact_force_threshold_n = 5.0;
  return env;
}

osk::ChunkView make_chunk_view(const std::vector<double>& flat, std::uint16_t horizon,
                               std::uint8_t n_dof,
                               osk::ControlMode mode = osk::ControlMode::kJointPosition) {
  osk::ChunkView view{};
  view.control_mode = static_cast<std::uint8_t>(mode);
  view.horizon = horizon;
  view.n_dof = n_dof;
  view.flat_data = flat.empty() ? nullptr : flat.data();
  view.flat_size = flat.size();
  return view;
}

}  // namespace

TEST(Validator, RejectsWhenEnvelopeUnconfigured) {
  osk::EnvelopeIntersection env;  // n_dof = 0
  std::vector<double> flat(3, 0.1);
  const auto view = make_chunk_view(flat, 1, 3);
  const auto rc = osk::validate(view, env);
  ASSERT_FALSE(rc);
  EXPECT_EQ(rc.error().kind, osk::ViolationKind::kController);
  EXPECT_EQ(rc.error().sub, osk::ControllerSubKind::kEnvelopeUnconfigured);
}

TEST(Validator, AcceptsValidJointPositionChunk) {
  const auto env = make_env();
  const std::vector<double> flat = {0.0, 0.1, -0.1, 0.2, 0.0, 0.0};  // 2 horizon × 3 dof
  const auto view = make_chunk_view(flat, 2, 3);
  const auto rc = osk::validate(view, env);
  EXPECT_TRUE(rc) << "field=" << rc.error().field;
}

TEST(Validator, RejectsJointPositionAboveCeiling) {
  const auto env = make_env();
  // Step 1 joint 0 exceeds pos_hi=1.0.
  const std::vector<double> flat = {0.0, 0.1, -0.1, 5.0, 0.0, 0.0};
  const auto view = make_chunk_view(flat, 2, 3);
  const auto rc = osk::validate(view, env);
  ASSERT_FALSE(rc);
  EXPECT_EQ(rc.error().kind, osk::ViolationKind::kWorkspace);
  EXPECT_EQ(rc.error().joint_index, 0U);
  EXPECT_EQ(rc.error().horizon_step, 1U);
  EXPECT_EQ(rc.error().offending_value, 5.0);
  EXPECT_EQ(rc.error().limit_value, 1.0);
}

TEST(Validator, RejectsNanInAction) {
  const auto env = make_env();
  std::vector<double> flat = {0.0, 0.1, -0.1};
  flat[1] = std::numeric_limits<double>::quiet_NaN();
  const auto view = make_chunk_view(flat, 1, 3);
  const auto rc = osk::validate(view, env);
  ASSERT_FALSE(rc);
  EXPECT_EQ(rc.error().kind, osk::ViolationKind::kController);
  EXPECT_EQ(rc.error().sub, osk::ControllerSubKind::kNanInAction);
  EXPECT_EQ(rc.error().joint_index, 1U);
}

TEST(Validator, RejectsNdofMismatch) {
  const auto env = make_env(3);
  const std::vector<double> flat = {0.0, 0.1};  // dof=2
  const auto view = make_chunk_view(flat, 1, 2);
  const auto rc = osk::validate(view, env);
  ASSERT_FALSE(rc);
  EXPECT_EQ(rc.error().sub, osk::ControllerSubKind::kNdofMismatch);
  EXPECT_EQ(rc.error().offending_value, 2.0);
  EXPECT_EQ(rc.error().limit_value, 3.0);
}

TEST(Validator, RejectsFlatSizeMismatch) {
  const auto env = make_env(3);
  // n_dof=3, horizon=2 → expect 6 elements; supply 5.
  const std::vector<double> flat = {0.0, 0.1, -0.1, 0.2, 0.0};
  const auto view = make_chunk_view(flat, 2, 3);
  const auto rc = osk::validate(view, env);
  ASSERT_FALSE(rc);
  EXPECT_EQ(rc.error().sub, osk::ControllerSubKind::kDimMismatch);
}

TEST(Validator, JointVelocityCapEnforced) {
  const auto env = make_env(3, /*pos_lo=*/-100.0, /*pos_hi=*/100.0,
                            /*vel_max=*/1.0, /*tau_max=*/100.0);
  const std::vector<double> flat = {0.5, -0.9, 0.99,  // step 0 OK
                                    0.5, -0.5, 1.5};  // step 1 joint 2 violates
  const auto view = make_chunk_view(flat, 2, 3, osk::ControlMode::kJointVelocity);
  const auto rc = osk::validate(view, env);
  ASSERT_FALSE(rc);
  EXPECT_EQ(rc.error().kind, osk::ViolationKind::kWorkspace);
  EXPECT_EQ(rc.error().joint_index, 2U);
  EXPECT_EQ(rc.error().horizon_step, 1U);
}

TEST(Validator, JointTorqueCapEnforced) {
  const auto env = make_env(3, /*pos_lo=*/-100.0, /*pos_hi=*/100.0,
                            /*vel_max=*/100.0, /*tau_max=*/2.0);
  // step 0 joint 1 commands torque 3.0 > tau_max=2.0.
  const std::vector<double> flat = {0.5, 3.0, 0.5};
  const auto view = make_chunk_view(flat, 1, 3, osk::ControlMode::kJointTorque);
  const auto rc = osk::validate(view, env);
  ASSERT_FALSE(rc);
  EXPECT_EQ(rc.error().kind, osk::ViolationKind::kForce);
  EXPECT_EQ(rc.error().joint_index, 1U);
}

TEST(Validator, CartesianTwistSpeedCap) {
  auto env = make_env(6);
  env.max_ee_speed_m_s = 0.5;
  // (vx, vy, vz, wx, wy, wz) — |v|=sqrt(1+0+0)=1 > 0.5.
  const std::vector<double> flat = {1.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  const auto view = make_chunk_view(flat, 1, 6, osk::ControlMode::kCartesianTwist);
  const auto rc = osk::validate(view, env);
  ASSERT_FALSE(rc);
  EXPECT_EQ(rc.error().kind, osk::ViolationKind::kForce);
  EXPECT_NEAR(rc.error().offending_value, 1.0, 1e-9);
}

TEST(Validator, CartesianPoseWorkspaceAabb) {
  auto env = make_env(7);
  env.workspace_box.set = true;
  env.workspace_box.min_xyz = {-0.1, -0.1, 0.0};
  env.workspace_box.max_xyz = {0.1, 0.1, 0.5};
  // (xyzw, xyz_position) — position at (0.0, 0.0, 1.0) violates z-max.
  const std::vector<double> flat = {0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0};
  const auto view = make_chunk_view(flat, 1, 7, osk::ControlMode::kCartesianPose);
  const auto rc = osk::validate(view, env);
  ASSERT_FALSE(rc);
  EXPECT_EQ(rc.error().kind, osk::ViolationKind::kWorkspace);
}

// ADR-0028b — per-mode chunks pass the kernel structural check and
// delegate per-axis bounds to the Python openral_safety supervisor. The
// kernel only enforces ``chunk.n_dof == envelope.n_dof`` for JOINT
// modes; cartesian / twist / gripper chunks have a per-mode width
// (6, 6, 1, …) that has nothing to do with the joint count.

TEST(Validator, CartesianDeltaPassesWithSixDofWidthAndJointEnvelope) {
  auto env = make_env(11);  // panda_mobile 11-DoF envelope
  const std::vector<double> flat = {0.01, 0.02, 0.0, 0.0, 0.0, 0.0};
  const auto view = make_chunk_view(flat, 1, 6, osk::ControlMode::kCartesianDelta);
  const auto rc = osk::validate(view, env);
  EXPECT_TRUE(rc);
}

TEST(Validator, GripperPositionPassesWithUnaryWidthAndJointEnvelope) {
  auto env = make_env(11);
  const std::vector<double> flat = {0.04};
  const auto view = make_chunk_view(flat, 1, 1, osk::ControlMode::kGripperPosition);
  const auto rc = osk::validate(view, env);
  EXPECT_TRUE(rc);
}

TEST(Validator, BodyTwistPassesWithSixDofWidthAndJointEnvelope) {
  auto env = make_env(11);
  const std::vector<double> flat = {0.1, 0.0, 0.0, 0.0, 0.0, 0.0};
  const auto view = make_chunk_view(flat, 1, 6, osk::ControlMode::kBodyTwist);
  const auto rc = osk::validate(view, env);
  EXPECT_TRUE(rc);
}

TEST(Validator, NonJointModeStillRejectsNan) {
  // NaN scan runs BEFORE the per-mode dispatch, so per-mode chunks
  // still get the structural soundness guarantee.
  auto env = make_env(11);
  const std::vector<double> flat = {0.01,
                                    std::numeric_limits<double>::quiet_NaN(),
                                    0.0, 0.0, 0.0, 0.0};
  const auto view = make_chunk_view(flat, 1, 6, osk::ControlMode::kCartesianDelta);
  const auto rc = osk::validate(view, env);
  ASSERT_FALSE(rc);
  EXPECT_EQ(rc.error().kind, osk::ViolationKind::kController);
}

TEST(Validator, JointModeStillEnforcesNdofMismatch) {
  // Regression: the legacy joint-mode contract is unchanged. A
  // joint_position chunk with n_dof != envelope.n_dof must still fail
  // fast — the per-axis joint_position_min/max[] arrays are indexed by
  // joint id and a width mismatch would read out of bounds.
  auto env = make_env(11);
  const std::vector<double> flat(7, 0.0);
  const auto view = make_chunk_view(flat, 1, 7, osk::ControlMode::kJointPosition);
  const auto rc = osk::validate(view, env);
  ASSERT_FALSE(rc);
  EXPECT_EQ(rc.error().kind, osk::ViolationKind::kController);
}

TEST(Validator, FootPlacementStillRejectedAsUnsupported) {
  // The newly added enum values for unwired modes (FOOT_PLACEMENT,
  // DEX_HAND_JOINT) MUST keep failing; otherwise an unimplemented
  // mode would silently pass.
  auto env = make_env(11);
  const std::vector<double> flat(3, 0.0);
  const auto view = make_chunk_view(flat, 1, 3, osk::ControlMode::kFootPlacement);
  const auto rc = osk::validate(view, env);
  ASSERT_FALSE(rc);
  EXPECT_EQ(rc.error().kind, osk::ViolationKind::kController);
}
