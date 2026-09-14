// SPDX-License-Identifier: Apache-2.0
// gtest unit coverage for the allocation-free validator.
// Real EnvelopeIntersection structs (no mocks); real chunk views.

#include "openral_safety_kernel/validator.hpp"

#include <array>
#include <cmath>
#include <limits>
#include <vector>

#include <gtest/gtest.h>

namespace osk = openral_safety_kernel;

namespace {

osk::EnvelopeIntersection make_env(std::size_t n_dof = 3, double pos_lo = -1.0, double pos_hi = 1.0,
                                   double vel_max = 3.15, double tau_max = 5.0) {
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
                               osk::ControlMode mode = osk::ControlMode::kJointPosition,
                               const std::vector<double>* cartesian_delta_scale = nullptr) {
  osk::ChunkView view{};
  view.control_mode = static_cast<std::uint8_t>(mode);
  view.horizon = horizon;
  view.n_dof = n_dof;
  view.flat_data = flat.empty() ? nullptr : flat.data();
  view.flat_size = flat.size();
  if (cartesian_delta_scale != nullptr) {
    view.cartesian_delta_scale =
        cartesian_delta_scale->empty() ? nullptr : cartesian_delta_scale->data();
    view.cartesian_delta_scale_size = cartesian_delta_scale->size();
  }
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

TEST(Validator, AcceptsCartesianDeltaScaleAndRejectsMalformedScale) {
  const auto env = make_env();
  const std::vector<double> flat = {0.5, 0.0, -0.5, 0.0, 0.25, 0.0};
  const std::vector<double> valid_scale = {0.05, 0.05, 0.05, 0.5, 0.5, 0.5};
  auto view = make_chunk_view(flat, 1, 6, osk::ControlMode::kCartesianDelta, &valid_scale);
  EXPECT_TRUE(osk::validate(view, env));

  const std::vector<double> wrong_width = {0.05, 0.05, 0.05};
  view = make_chunk_view(flat, 1, 6, osk::ControlMode::kCartesianDelta, &wrong_width);
  auto result = osk::validate(view, env);
  ASSERT_FALSE(result);
  EXPECT_EQ(result.error().sub, osk::ControllerSubKind::kInvalidScale);

  const std::vector<double> nonpositive = {0.05, 0.05, 0.0, 0.5, 0.5, 0.5};
  view = make_chunk_view(flat, 1, 6, osk::ControlMode::kCartesianDelta, &nonpositive);
  result = osk::validate(view, env);
  ASSERT_FALSE(result);
  EXPECT_EQ(result.error().sub, osk::ControllerSubKind::kInvalidScale);
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

TEST(Validator, CartesianTwistAngularSpeedCapEnforced) {
  auto env = make_env(6);
  env.max_ee_speed_m_s = 0.5;
  env.max_ee_angular_speed_rad_s = 0.4;
  // Linear component within bound; angular |w|=1.0 > 0.4.
  const std::vector<double> flat = {0.1, 0.0, 0.0, 1.0, 0.0, 0.0};
  const auto view = make_chunk_view(flat, 1, 6, osk::ControlMode::kCartesianTwist);
  const auto rc = osk::validate(view, env);
  ASSERT_FALSE(rc);
  EXPECT_EQ(rc.error().kind, osk::ViolationKind::kForce);
  EXPECT_STREQ(rc.error().field, "ee_angular_speed");
  EXPECT_NEAR(rc.error().offending_value, 1.0, 1e-9);
  EXPECT_NEAR(rc.error().limit_value, 0.4, 1e-9);
}

TEST(Validator, CartesianTwistAngularSpeedDefaultsToUnboundedWhenUnset) {
  // env.max_ee_angular_speed_rad_s defaults to kPosInfinity in make_env() --
  // a robot that never declares this bound (every robot except the ones
  // fixed as part of this change) must see unchanged behaviour.
  auto env = make_env(6);
  env.max_ee_speed_m_s = 0.5;
  const std::vector<double> flat = {0.1, 0.0, 0.0, 100.0, 0.0, 0.0};
  const auto view = make_chunk_view(flat, 1, 6, osk::ControlMode::kCartesianTwist);
  const auto rc = osk::validate(view, env);
  EXPECT_TRUE(rc);
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

// Per-mode chunks pass the kernel structural check and
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

TEST(Validator, BodyTwistLinearSpeedCapEnforced) {
  auto env = make_env(11);
  env.max_base_linear_speed_m_s = 0.3;
  env.max_base_angular_speed_rad_s = 0.5;
  // (vx, vy, vz, wx, wy, wz) — |v|=sqrt(36)=6.0 > 0.3 (the F12 6.0-vs-0.3
  // m/s live-test scenario that motivated this check).
  const std::vector<double> flat = {6.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  const auto view = make_chunk_view(flat, 1, 6, osk::ControlMode::kBodyTwist);
  const auto rc = osk::validate(view, env);
  ASSERT_FALSE(rc);
  EXPECT_EQ(rc.error().kind, osk::ViolationKind::kForce);
  EXPECT_STREQ(rc.error().field, "base_linear_speed");
  EXPECT_NEAR(rc.error().offending_value, 6.0, 1e-9);
  EXPECT_NEAR(rc.error().limit_value, 0.3, 1e-9);
}

TEST(Validator, BodyTwistAngularSpeedCapEnforced) {
  auto env = make_env(11);
  env.max_base_linear_speed_m_s = 0.3;
  env.max_base_angular_speed_rad_s = 0.5;
  // Linear component within bound; angular |w|=1.0 > 0.5.
  const std::vector<double> flat = {0.1, 0.0, 0.0, 1.0, 0.0, 0.0};
  const auto view = make_chunk_view(flat, 1, 6, osk::ControlMode::kBodyTwist);
  const auto rc = osk::validate(view, env);
  ASSERT_FALSE(rc);
  EXPECT_EQ(rc.error().kind, osk::ViolationKind::kForce);
  EXPECT_STREQ(rc.error().field, "base_angular_speed");
  EXPECT_NEAR(rc.error().offending_value, 1.0, 1e-9);
  EXPECT_NEAR(rc.error().limit_value, 0.5, 1e-9);
}

TEST(Validator, BodyTwistDimMismatchRejected) {
  auto env = make_env(11);
  env.max_base_linear_speed_m_s = 0.3;
  // n_dof (row width) < 6: not a well-formed body-twist chunk.
  auto narrow_env = env;
  narrow_env.n_dof = 3;
  narrow_env.joint_position_min.assign(3, -1.0);
  narrow_env.joint_position_max.assign(3, 1.0);
  narrow_env.joint_velocity_max.assign(3, 1.0);
  narrow_env.joint_torque_max.assign(3, 1.0);
  const std::vector<double> flat = {0.1, 0.0, 0.0};
  const auto view = make_chunk_view(flat, 1, 3, osk::ControlMode::kBodyTwist);
  const auto rc = osk::validate(view, narrow_env);
  ASSERT_FALSE(rc);
  EXPECT_EQ(rc.error().sub, osk::ControllerSubKind::kDimMismatch);
}

// ── item 9 (execution_plan.md §8.3): kJointTrajectory / kCartesianDelta /
// kGripperPosition / kCompositeMode magnitude checks ────────────────────

TEST(Validator, JointTrajectoryPositionCapEnforced) {
  // JOINT_TRAJECTORY shares JOINT_POSITION's per-step position envelope
  // (ROSPublishingHAL._flatten_action_payload treats the two modes
  // identically). Previously delegated to the unlaunched Python stub.
  const auto env = make_env();  // pos in [-1, 1]
  const std::vector<double> flat = {0.0, 0.0, 1.5};  // joint 2 exceeds 1.0
  const auto view = make_chunk_view(flat, 1, 3, osk::ControlMode::kJointTrajectory);
  const auto rc = osk::validate(view, env);
  ASSERT_FALSE(rc);
  EXPECT_EQ(rc.error().kind, osk::ViolationKind::kWorkspace);
  EXPECT_STREQ(rc.error().field, "joint_position");
  EXPECT_EQ(rc.error().joint_index, 2);
}

TEST(Validator, CartesianDeltaStepCapEnforced) {
  auto env = make_env(11);
  env.max_cartesian_step_m = 0.05;
  // |dxyz| = sqrt(0.06^2*3) ≈ 0.1039 > 0.05.
  const std::vector<double> flat = {0.06, 0.06, 0.06, 0.0, 0.0, 0.0};
  const auto view = make_chunk_view(flat, 1, 6, osk::ControlMode::kCartesianDelta);
  const auto rc = osk::validate(view, env);
  ASSERT_FALSE(rc);
  EXPECT_EQ(rc.error().kind, osk::ViolationKind::kForce);
  EXPECT_STREQ(rc.error().field, "cartesian_step");
  EXPECT_NEAR(rc.error().limit_value, 0.05, 1e-9);
}

TEST(Validator, CartesianDeltaRotationStepCapEnforced) {
  auto env = make_env(11);
  env.max_cartesian_step_rad = 0.1;
  // xyz within (default unbounded); |drotvec| = 0.3 > 0.1.
  const std::vector<double> flat = {0.0, 0.0, 0.0, 0.3, 0.0, 0.0};
  const auto view = make_chunk_view(flat, 1, 6, osk::ControlMode::kCartesianDelta);
  const auto rc = osk::validate(view, env);
  ASSERT_FALSE(rc);
  EXPECT_EQ(rc.error().kind, osk::ViolationKind::kForce);
  EXPECT_STREQ(rc.error().field, "cartesian_step_rot");
  EXPECT_NEAR(rc.error().limit_value, 0.1, 1e-9);
}

TEST(Validator, CartesianDeltaStepCapAppliesScale) {
  // Raw [-1, 1] controller-native values * cartesian_delta_scale must be
  // converted to physical units BEFORE the magnitude check -- mirrors
  // lifecycle_kernel.cpp's apply-path formula exactly. Raw 1.0 (clamped)
  // * scale 0.2 = 0.2 physical metres per axis; |d|=sqrt(0.2^2*3)≈0.346.
  auto env = make_env(11);
  env.max_cartesian_step_m = 0.1;  // physical bound well under 0.346
  const std::vector<double> flat = {1.0, 1.0, 1.0, 0.0, 0.0, 0.0};
  const std::vector<double> scale = {0.2, 0.2, 0.2, 1.0, 1.0, 1.0};
  const auto view = make_chunk_view(flat, 1, 6, osk::ControlMode::kCartesianDelta, &scale);
  const auto rc = osk::validate(view, env);
  ASSERT_FALSE(rc);
  EXPECT_STREQ(rc.error().field, "cartesian_step");
  // Physical, not raw: offending_value should read ~0.346, not sqrt(3)≈1.73.
  EXPECT_NEAR(rc.error().offending_value, std::sqrt(3.0) * 0.2, 1e-6);
}

TEST(Validator, CartesianDeltaStepCapDefaultsToUnboundedWhenUnset) {
  // A robot that never declares max_cartesian_step_m/_rad keeps today's
  // behaviour: any finite delta passes (regression for the "no bound
  // declared" sentinel).
  auto env = make_env(11);
  const std::vector<double> flat = {5.0, 5.0, 5.0, 5.0, 5.0, 5.0};
  const auto view = make_chunk_view(flat, 1, 6, osk::ControlMode::kCartesianDelta);
  const auto rc = osk::validate(view, env);
  EXPECT_TRUE(rc);
}

TEST(Validator, CartesianDeltaUsesChunkWidthNotEnvelopeWidthAcrossHorizon) {
  // Regression for a real bug this item's implementation avoided
  // (research repo F35): a non-joint-mode chunk's per-step stride must be
  // chunk.n_dof (6 here), NOT envelope.n_dof (11 here) -- using the
  // latter would misindex every step past the first for horizon > 1.
  // Step 0 = (0.01,0,0,0,0,0) [in bound]; step 1 = (0.2,0,0,0,0,0) [out
  // of bound] -- correct 6-wide striding must reach step 1's real values,
  // not read six zeros/garbage from an 11-wide stride.
  auto env = make_env(11);
  env.max_cartesian_step_m = 0.05;
  const std::vector<double> flat = {0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0};
  const auto view = make_chunk_view(flat, 2, 6, osk::ControlMode::kCartesianDelta);
  const auto rc = osk::validate(view, env);
  ASSERT_FALSE(rc);
  EXPECT_STREQ(rc.error().field, "cartesian_step");
  EXPECT_EQ(rc.error().horizon_step, 1);
  EXPECT_NEAR(rc.error().offending_value, 0.2, 1e-9);
}

TEST(Validator, GripperPositionRangeCapEnforced) {
  auto env = make_env(11);
  env.gripper_min = 0.0;
  env.gripper_max = 0.8;
  const std::vector<double> flat = {0.95};  // > 0.8
  const auto view = make_chunk_view(flat, 1, 1, osk::ControlMode::kGripperPosition);
  const auto rc = osk::validate(view, env);
  ASSERT_FALSE(rc);
  EXPECT_EQ(rc.error().kind, osk::ViolationKind::kWorkspace);
  EXPECT_STREQ(rc.error().field, "gripper_range");
  EXPECT_NEAR(rc.error().offending_value, 0.95, 1e-9);
  EXPECT_NEAR(rc.error().limit_value, 0.8, 1e-9);
}

TEST(Validator, GripperPositionDefaultsToUnboundedWhenUnset) {
  auto env = make_env(11);  // no gripper-role joint -> gripper_min/max unset
  const std::vector<double> flat = {5.0};  // well outside [0,1] but no bound declared
  const auto view = make_chunk_view(flat, 1, 1, osk::ControlMode::kGripperPosition);
  const auto rc = osk::validate(view, env);
  EXPECT_TRUE(rc);
}

TEST(Validator, CompositeModeRangeCapEnforced) {
  auto env = make_env(11);
  const std::vector<double> flat = {1.5};  // > 1.0, fixed protocol contract
  const auto view = make_chunk_view(flat, 1, 1, osk::ControlMode::kCompositeMode);
  const auto rc = osk::validate(view, env);
  ASSERT_FALSE(rc);
  EXPECT_EQ(rc.error().kind, osk::ViolationKind::kWorkspace);
  EXPECT_STREQ(rc.error().field, "composite_mode_range");
  EXPECT_NEAR(rc.error().limit_value, 1.0, 1e-9);
}

TEST(Validator, CompositeModeWithinRangePasses) {
  auto env = make_env(11);
  const std::vector<double> flat = {-1.0};  // boundary value, must NOT reject
  const auto view = make_chunk_view(flat, 1, 1, osk::ControlMode::kCompositeMode);
  const auto rc = osk::validate(view, env);
  EXPECT_TRUE(rc);
}

TEST(Validator, GripperBinaryStillDelegatesUnchanged) {
  // GRIPPER_BINARY is explicitly out of this item's scope (see the
  // kGripperBinary case's comment) -- confirm it still passes through
  // untouched (same as every kind: procedural pre-existing behaviour),
  // not accidentally caught by the new kGripperPosition case.
  auto env = make_env(11);
  const std::vector<double> flat = {5.0};  // would violate [0,1] if checked
  const auto view = make_chunk_view(flat, 1, 1, osk::ControlMode::kGripperBinary);
  const auto rc = osk::validate(view, env);
  EXPECT_TRUE(rc);
}

TEST(Validator, NonJointModeStillRejectsNan) {
  // NaN scan runs BEFORE the per-mode dispatch, so per-mode chunks
  // still get the structural soundness guarantee.
  auto env = make_env(11);
  const std::vector<double> flat = {0.01, std::numeric_limits<double>::quiet_NaN(), 0.0, 0.0, 0.0,
                                    0.0};
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
