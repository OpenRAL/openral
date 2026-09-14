// SPDX-License-Identifier: Apache-2.0
// validator.cpp: the allocation-free hot path. Pinned by
// test_no_alloc.cpp via a counting allocator + mlockall assertion.

#include "openral_safety_kernel/validator.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace openral_safety_kernel {

namespace {

inline bool is_finite(double v) noexcept { return std::isfinite(v); }

Violation make_controller_violation(ControllerSubKind sub, std::string_view field) noexcept {
  Violation v{};
  v.kind = ViolationKind::kController;
  v.sub = sub;
  v.set_field(field);
  return v;
}

}  // namespace

Result<void, Violation> validate(const ChunkView& chunk,
                                 const EnvelopeIntersection& envelope) noexcept {
  // 1. Envelope must have been loaded; without n_dof we cannot bound-check.
  if (envelope.n_dof == 0) {
    return Result<void, Violation>::err(
        make_controller_violation(ControllerSubKind::kEnvelopeUnconfigured, "n_dof"));
  }

  // 2. Decide whether chunk.n_dof is a JOINT-COUNT (must match the
  // envelope) or a per-mode width (cartesian = 6, gripper = 1, etc.,
  // enforced per-mode by the openral_safety Python supervisor). Without
  // this split, every slot-dispatched per-mode chunk fails the n_dof
  // equality check and trips an estop before the supervisor's per-mode
  // bounds run — leaving openral unable to dispatch any RoboCasa pi0.5/rldx
  // rSkill in deploy_sim. (sim_run is unaffected: that path bypasses both
  // safety nodes and drives env.step directly.)
  const auto mode = static_cast<ControlMode>(chunk.control_mode);
  const bool is_joint_mode =
      (mode == ControlMode::kJointPosition || mode == ControlMode::kJointVelocity ||
       mode == ControlMode::kJointTorque || mode == ControlMode::kJointTrajectory);

  if (is_joint_mode) {
    // Joint chunks: n_dof must equal envelope.n_dof — every element is
    // a per-joint value, and the per-axis bound check below indexes
    // into envelope.joint_*_min/max[] by joint id.
    if (static_cast<std::size_t>(chunk.n_dof) != envelope.n_dof) {
      Violation v = make_controller_violation(ControllerSubKind::kNdofMismatch, "n_dof");
      v.offending_value = static_cast<double>(chunk.n_dof);
      v.limit_value = static_cast<double>(envelope.n_dof);
      return Result<void, Violation>::err(v);
    }
  }

  // 3. flat[] must have length horizon * <per-row width>. For joint
  // modes the width is envelope.n_dof; for per-mode chunks (cartesian,
  // twist, gripper) the chunk's own n_dof field declares the row width.
  const std::size_t per_row =
      is_joint_mode ? envelope.n_dof : static_cast<std::size_t>(chunk.n_dof);
  const std::size_t expected = static_cast<std::size_t>(chunk.horizon) * per_row;
  if (chunk.flat_size != expected || chunk.flat_data == nullptr) {
    Violation v = make_controller_violation(ControllerSubKind::kDimMismatch, "flat");
    v.offending_value = static_cast<double>(chunk.flat_size);
    v.limit_value = static_cast<double>(expected);
    return Result<void, Violation>::err(v);
  }

  // 4. Optional controller-native -> physical scale for CARTESIAN_DELTA.
  // Empty means identity (physical-unit policies and old publishers).
  if (chunk.cartesian_delta_scale_size > 0) {
    if (mode != ControlMode::kCartesianDelta || chunk.cartesian_delta_scale_size != per_row ||
        chunk.cartesian_delta_scale == nullptr) {
      Violation v =
          make_controller_violation(ControllerSubKind::kInvalidScale, "cartesian_delta_scale");
      v.offending_value = static_cast<double>(chunk.cartesian_delta_scale_size);
      v.limit_value = static_cast<double>(per_row);
      return Result<void, Violation>::err(v);
    }
    for (std::size_t i = 0; i < chunk.cartesian_delta_scale_size; ++i) {
      const double scale = chunk.cartesian_delta_scale[i];
      if (!is_finite(scale) || scale <= 0.0) {
        Violation v =
            make_controller_violation(ControllerSubKind::kInvalidScale, "cartesian_delta_scale");
        v.offending_value = scale;
        v.limit_value = 0.0;
        v.joint_index = static_cast<std::uint16_t>(i);
        return Result<void, Violation>::err(v);
      }
    }
  }

  // 5. NaN / Inf scan on the whole flat[].
  for (std::size_t i = 0; i < chunk.flat_size; ++i) {
    if (!is_finite(chunk.flat_data[i])) {
      Violation v = make_controller_violation(ControllerSubKind::kNanInAction, "flat");
      v.offending_value = chunk.flat_data[i];
      v.joint_index = per_row > 0 ? static_cast<std::uint16_t>(i % per_row) : 0;
      v.horizon_step = per_row > 0 ? static_cast<std::uint16_t>(i / per_row) : 0;
      return Result<void, Violation>::err(v);
    }
  }

  // 6. Per-step / per-joint enforcement keyed off control_mode.

  switch (mode) {
  case ControlMode::kJointPosition:
  case ControlMode::kJointTrajectory: {
    // JOINT_TRAJECTORY shares JOINT_POSITION's wire shape exactly (a
    // [horizon][n_dof] matrix of per-joint position targets --
    // ROSPublishingHAL._flatten_action_payload and
    // ManifestHALLifecycleNode._on_safe_action both already treat the two
    // modes identically) and the same per-joint position envelope applies
    // per waypoint. Previously delegated to the Python supervisor stub
    // (item 9, execution_plan.md §8.3) alongside CARTESIAN_DELTA/
    // GRIPPER_POSITION/COMPOSITE_MODE below.
    for (std::uint16_t s = 0; s < chunk.horizon; ++s) {
      for (std::size_t j = 0; j < envelope.n_dof; ++j) {
        const double v = chunk.flat_data[s * envelope.n_dof + j];
        if (v < envelope.joint_position_min[j] || v > envelope.joint_position_max[j]) {
          Violation viol{};
          viol.kind = ViolationKind::kWorkspace;
          viol.joint_index = static_cast<std::uint16_t>(j);
          viol.horizon_step = s;
          viol.offending_value = v;
          viol.limit_value = (v < envelope.joint_position_min[j]) ? envelope.joint_position_min[j]
                                                                  : envelope.joint_position_max[j];
          viol.set_field("joint_position");
          return Result<void, Violation>::err(viol);
        }
      }
    }
    break;
  }
  case ControlMode::kJointVelocity: {
    for (std::uint16_t s = 0; s < chunk.horizon; ++s) {
      for (std::size_t j = 0; j < envelope.n_dof; ++j) {
        const double v = chunk.flat_data[s * envelope.n_dof + j];
        const double cap = envelope.joint_velocity_max[j];
        if (std::abs(v) > cap) {
          Violation viol{};
          viol.kind = ViolationKind::kWorkspace;
          viol.joint_index = static_cast<std::uint16_t>(j);
          viol.horizon_step = s;
          viol.offending_value = v;
          viol.limit_value = cap;
          viol.set_field("joint_velocity");
          return Result<void, Violation>::err(viol);
        }
      }
    }
    break;
  }
  case ControlMode::kJointTorque: {
    for (std::uint16_t s = 0; s < chunk.horizon; ++s) {
      for (std::size_t j = 0; j < envelope.n_dof; ++j) {
        const double v = chunk.flat_data[s * envelope.n_dof + j];
        const double cap = std::min(envelope.joint_torque_max[j], envelope.max_torque_nm);
        if (std::abs(v) > cap) {
          Violation viol{};
          viol.kind = ViolationKind::kForce;
          viol.joint_index = static_cast<std::uint16_t>(j);
          viol.horizon_step = s;
          viol.offending_value = v;
          viol.limit_value = cap;
          viol.set_field("joint_torque");
          return Result<void, Violation>::err(viol);
        }
      }
    }
    break;
  }
  case ControlMode::kCartesianPose: {
    // Each step encodes (xyzw quaternion, xyz position) — 7 floats.
    // For v1 we only check the position against the workspace AABB
    // when it's set; quaternion bounds are out of scope.
    if (!envelope.workspace_box.set) {
      break;
    }
    // The cartesian flat layout is one 7-vec per step; chunk.n_dof
    // is expected to be 7 in that mode (the message ferries it). We
    // validate axis-by-axis.
    const std::size_t per_step = envelope.n_dof;
    if (per_step < 7) {
      // Not a well-formed Cartesian chunk — fall through to dim-mismatch.
      Violation v = make_controller_violation(ControllerSubKind::kDimMismatch, "cartesian_pose");
      return Result<void, Violation>::err(v);
    }
    for (std::uint16_t s = 0; s < chunk.horizon; ++s) {
      const double* p = chunk.flat_data + s * per_step + 4;  // skip xyzw
      for (std::size_t axis = 0; axis < 3; ++axis) {
        if (p[axis] < envelope.workspace_box.min_xyz[axis] ||
            p[axis] > envelope.workspace_box.max_xyz[axis]) {
          Violation viol{};
          viol.kind = ViolationKind::kWorkspace;
          viol.joint_index = static_cast<std::uint16_t>(axis);
          viol.horizon_step = s;
          viol.offending_value = p[axis];
          viol.limit_value = (p[axis] < envelope.workspace_box.min_xyz[axis])
                                 ? envelope.workspace_box.min_xyz[axis]
                                 : envelope.workspace_box.max_xyz[axis];
          viol.set_field("workspace_xyz");
          return Result<void, Violation>::err(viol);
        }
      }
    }
    break;
  }
  case ControlMode::kCartesianTwist: {
    // Each step encodes (vx, vy, vz, wx, wy, wz). Bound linear speed
    // against max_ee_speed_m_s and angular speed against
    // max_ee_angular_speed_rad_s.
    const std::size_t per_step = envelope.n_dof;
    if (per_step < 6) {
      Violation v = make_controller_violation(ControllerSubKind::kDimMismatch, "cartesian_twist");
      return Result<void, Violation>::err(v);
    }
    for (std::uint16_t s = 0; s < chunk.horizon; ++s) {
      const double* p = chunk.flat_data + s * per_step;
      const double linear_speed = std::sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]);
      if (linear_speed > envelope.max_ee_speed_m_s) {
        Violation viol{};
        viol.kind = ViolationKind::kForce;  // speed-induced
        viol.joint_index = 0xFFFF;
        viol.horizon_step = s;
        viol.offending_value = linear_speed;
        viol.limit_value = envelope.max_ee_speed_m_s;
        viol.set_field("ee_speed");
        return Result<void, Violation>::err(viol);
      }
      const double angular_speed = std::sqrt(p[3] * p[3] + p[4] * p[4] + p[5] * p[5]);
      if (angular_speed > envelope.max_ee_angular_speed_rad_s) {
        Violation viol{};
        viol.kind = ViolationKind::kForce;
        viol.joint_index = 0xFFFF;
        viol.horizon_step = s;
        viol.offending_value = angular_speed;
        viol.limit_value = envelope.max_ee_angular_speed_rad_s;
        viol.set_field("ee_angular_speed");
        return Result<void, Violation>::err(viol);
      }
    }
    break;
  }
  case ControlMode::kBodyTwist: {
    // Each step encodes (vx, vy, vz, wx, wy, wz) for the mobile base.
    // Bound linear speed against max_base_linear_speed_m_s and angular
    // speed against max_base_angular_speed_rad_s. Both default to
    // kPosInfinity (no bound declared) so a robot.yaml that never sets
    // them behaves exactly as before this check existed.
    const std::size_t per_step = envelope.n_dof;
    if (per_step < 6) {
      Violation v = make_controller_violation(ControllerSubKind::kDimMismatch, "body_twist");
      return Result<void, Violation>::err(v);
    }
    for (std::uint16_t s = 0; s < chunk.horizon; ++s) {
      const double* p = chunk.flat_data + s * per_step;
      const double linear_speed = std::sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]);
      if (linear_speed > envelope.max_base_linear_speed_m_s) {
        Violation viol{};
        viol.kind = ViolationKind::kForce;  // speed-induced
        viol.joint_index = 0xFFFF;
        viol.horizon_step = s;
        viol.offending_value = linear_speed;
        viol.limit_value = envelope.max_base_linear_speed_m_s;
        viol.set_field("base_linear_speed");
        return Result<void, Violation>::err(viol);
      }
      const double angular_speed = std::sqrt(p[3] * p[3] + p[4] * p[4] + p[5] * p[5]);
      if (angular_speed > envelope.max_base_angular_speed_rad_s) {
        Violation viol{};
        viol.kind = ViolationKind::kForce;
        viol.joint_index = 0xFFFF;
        viol.horizon_step = s;
        viol.offending_value = angular_speed;
        viol.limit_value = envelope.max_base_angular_speed_rad_s;
        viol.set_field("base_angular_speed");
        return Result<void, Violation>::err(viol);
      }
    }
    break;
  }
  case ControlMode::kCartesianDelta: {
    // Each step encodes (dx, dy, dz, rx, ry, rz): a per-step Cartesian
    // translation delta + axis-angle rotation delta. Bound the Euclidean
    // magnitude of each triplet against max_cartesian_step_m /
    // max_cartesian_step_rad (item 9) -- mirrors
    // openral_safety.supervisor_node._envelope_violation_cartesian_delta
    // (the Python stub deploy_e2e.launch.py never launches) exactly,
    // including its clip(raw, -1, 1) * scale physical-unit derivation when
    // cartesian_delta_scale is set (step 4 above already validated its
    // shape/positivity) -- the same formula lifecycle_kernel.cpp's apply
    // path already uses, so the magnitude checked here matches what
    // actually reaches the HAL.
    //
    // NOTE: uses per_row (== chunk.n_dof for a non-joint mode, already
    // computed above) as the per-step stride, NOT envelope.n_dof (the
    // robot's total joint count) -- see the research repo's F35 for why
    // envelope.n_dof would misindex a horizon > 1 chunk here.
    const std::size_t per_step = per_row;
    if (per_step < 6) {
      Violation v = make_controller_violation(ControllerSubKind::kDimMismatch, "cartesian_delta");
      return Result<void, Violation>::err(v);
    }
    const bool scaled = chunk.cartesian_delta_scale_size > 0;
    for (std::uint16_t s = 0; s < chunk.horizon; ++s) {
      const double* p = chunk.flat_data + s * per_step;
      double d[6];
      for (std::size_t i = 0; i < 6; ++i) {
        const double raw = p[i];
        d[i] = scaled ? std::clamp(raw, -1.0, 1.0) * chunk.cartesian_delta_scale[i] : raw;
      }
      const double mag_xyz = std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
      if (mag_xyz > envelope.max_cartesian_step_m) {
        Violation viol{};
        viol.kind = ViolationKind::kForce;
        viol.joint_index = 0xFFFF;
        viol.horizon_step = s;
        viol.offending_value = mag_xyz;
        viol.limit_value = envelope.max_cartesian_step_m;
        viol.set_field("cartesian_step");
        return Result<void, Violation>::err(viol);
      }
      const double mag_rot = std::sqrt(d[3] * d[3] + d[4] * d[4] + d[5] * d[5]);
      if (mag_rot > envelope.max_cartesian_step_rad) {
        Violation viol{};
        viol.kind = ViolationKind::kForce;
        viol.joint_index = 0xFFFF;
        viol.horizon_step = s;
        viol.offending_value = mag_rot;
        viol.limit_value = envelope.max_cartesian_step_rad;
        viol.set_field("cartesian_step_rot");
        return Result<void, Violation>::err(viol);
      }
    }
    break;
  }
  case ControlMode::kGripperPosition: {
    // Single scalar per step: the normalised jaw-fraction width. Bound
    // against gripper_min/gripper_max (item 9), sourced from the robot's
    // gripper-role joint's position_limits -- mirrors
    // openral_safety.supervisor_node._envelope_violation_gripper. NOT
    // applied to kGripperBinary: a strict 0.0/1.0 binary command has no
    // meaningful "out of range" case the same way a continuous width
    // does, and it was out of this item's scope (still delegates below).
    const std::size_t per_step = per_row;  // == chunk.n_dof (a non-joint mode); see F35.
    if (per_step < 1) {
      Violation v = make_controller_violation(ControllerSubKind::kDimMismatch, "gripper_position");
      return Result<void, Violation>::err(v);
    }
    for (std::uint16_t s = 0; s < chunk.horizon; ++s) {
      const double w = chunk.flat_data[s * per_step];
      if (w < envelope.gripper_min || w > envelope.gripper_max) {
        Violation viol{};
        viol.kind = ViolationKind::kWorkspace;
        viol.joint_index = 0xFFFF;
        viol.horizon_step = s;
        viol.offending_value = w;
        viol.limit_value = (w < envelope.gripper_min) ? envelope.gripper_min : envelope.gripper_max;
        viol.set_field("gripper_range");
        return Result<void, Violation>::err(viol);
      }
    }
    break;
  }
  case ControlMode::kCompositeMode: {
    // Single scalar per step: a robosuite-specific multiplexer flag,
    // fixed protocol contract [-1, +1] (sim-only) -- not a manifest-
    // derived bound like the others in this switch, so there is no
    // envelope field to read; an out-of-range flag would silently
    // confuse the downstream robosuite controller instead of being
    // caught here (item 9).
    const std::size_t per_step = per_row;  // == chunk.n_dof (a non-joint mode); see F35.
    if (per_step < 1) {
      Violation v = make_controller_violation(ControllerSubKind::kDimMismatch, "composite_mode");
      return Result<void, Violation>::err(v);
    }
    for (std::uint16_t s = 0; s < chunk.horizon; ++s) {
      const double flag = chunk.flat_data[s * per_step];
      if (flag < -1.0 || flag > 1.0) {
        Violation viol{};
        viol.kind = ViolationKind::kWorkspace;
        viol.joint_index = 0xFFFF;
        viol.horizon_step = s;
        viol.offending_value = flag;
        viol.limit_value = (flag < -1.0) ? -1.0 : 1.0;
        viol.set_field("composite_mode_range");
        return Result<void, Violation>::err(viol);
      }
    }
    break;
  }
  case ControlMode::kGripperBinary: {
    // Per-mode chunk. The C++ kernel intentionally delegates this one
    // mode's bound enforcement to the Python openral_safety/
    // supervisor_node.py, out of item 9's scope (a strict binary open/
    // close command has no continuous-width "out of range" case the way
    // GRIPPER_POSITION does; see that case's comment above). The kernel
    // already ran shape + NaN checks above, so routing unrejected lets
    // the supervisor do its job before the HAL applies -- without this
    // case the chunk would hit the default branch and estop before the
    // supervisor sees it. Net safety strictly improves vs pre-change:
    // "reject every per-mode chunk" -> "structural-validate then delegate".
    break;
  }
  case ControlMode::kFootPlacement:
  case ControlMode::kDexHandJoint: {
    // No openral_safety validator wired for these yet; reject until
    // either the C++ kernel or the Python supervisor implements
    // bounds. Surfaces as kDimMismatch so the operator sees a clear
    // "unimplemented" rather than a silent pass-through.
    Violation v = make_controller_violation(ControllerSubKind::kDimMismatch, "control_mode");
    v.offending_value = static_cast<double>(chunk.control_mode);
    return Result<void, Violation>::err(v);
  }
  default: {
    // Unknown control_mode — refuse rather than silently passing.
    Violation v = make_controller_violation(ControllerSubKind::kDimMismatch, "control_mode");
    v.offending_value = static_cast<double>(chunk.control_mode);
    return Result<void, Violation>::err(v);
  }
  }

  return Result<void, Violation>::ok();
}

}  // namespace openral_safety_kernel
