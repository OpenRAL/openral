// SPDX-License-Identifier: Apache-2.0
// ADR-0020 — validator.cpp: the allocation-free hot path. Pinned by
// test_no_alloc.cpp via a counting allocator + mlockall assertion.

#include "openral_safety_kernel/validator.hpp"

#include <cmath>
#include <cstring>

namespace openral_safety_kernel {

namespace {

inline bool is_finite(double v) noexcept {
  return std::isfinite(v);
}

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

  // 2. Decide whether ``chunk.n_dof`` is a JOINT-COUNT (must match the
  // envelope) or a per-mode width (cartesian = 6, gripper = 1, etc.,
  // which the openral_safety Python supervisor enforces per-mode under
  // ADR-0028b). Without this split, every slot-dispatched per-mode
  // chunk fails the n_dof equality check and trips an estop before
  // the Python supervisor's per-mode bounds get to run — leaving the
  // openral abstraction unable to dispatch any RoboCasa pi0.5 / rldx
  // rSkill in deploy_sim. (Sim_run is unaffected; that path bypasses
  // both safety nodes and drives env.step directly.)
  const auto mode = static_cast<ControlMode>(chunk.control_mode);
  const bool is_joint_mode = (mode == ControlMode::kJointPosition
                              || mode == ControlMode::kJointVelocity
                              || mode == ControlMode::kJointTorque
                              || mode == ControlMode::kJointTrajectory);

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
  const std::size_t per_row = is_joint_mode
                                  ? envelope.n_dof
                                  : static_cast<std::size_t>(chunk.n_dof);
  const std::size_t expected = static_cast<std::size_t>(chunk.horizon) * per_row;
  if (chunk.flat_size != expected || chunk.flat_data == nullptr) {
    Violation v = make_controller_violation(ControllerSubKind::kDimMismatch, "flat");
    v.offending_value = static_cast<double>(chunk.flat_size);
    v.limit_value = static_cast<double>(expected);
    return Result<void, Violation>::err(v);
  }

  // 4. NaN / Inf scan on the whole flat[].
  for (std::size_t i = 0; i < chunk.flat_size; ++i) {
    if (!is_finite(chunk.flat_data[i])) {
      Violation v = make_controller_violation(ControllerSubKind::kNanInAction, "flat");
      v.offending_value = chunk.flat_data[i];
      v.joint_index = per_row > 0 ? static_cast<std::uint16_t>(i % per_row) : 0;
      v.horizon_step = per_row > 0 ? static_cast<std::uint16_t>(i / per_row) : 0;
      return Result<void, Violation>::err(v);
    }
  }

  // 5. Per-step / per-joint enforcement keyed off control_mode.

  switch (mode) {
    case ControlMode::kJointPosition: {
      for (std::uint16_t s = 0; s < chunk.horizon; ++s) {
        for (std::size_t j = 0; j < envelope.n_dof; ++j) {
          const double v = chunk.flat_data[s * envelope.n_dof + j];
          if (v < envelope.joint_position_min[j] || v > envelope.joint_position_max[j]) {
            Violation viol{};
            viol.kind = ViolationKind::kWorkspace;
            viol.joint_index = static_cast<std::uint16_t>(j);
            viol.horizon_step = s;
            viol.offending_value = v;
            viol.limit_value = (v < envelope.joint_position_min[j])
                                   ? envelope.joint_position_min[j]
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
          if (p[axis] < envelope.workspace_box.min_xyz[axis]
              || p[axis] > envelope.workspace_box.max_xyz[axis]) {
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
      // against max_ee_speed_m_s.
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
      }
      break;
    }
    case ControlMode::kJointTrajectory:
    case ControlMode::kCartesianDelta:
    case ControlMode::kBodyTwist:
    case ControlMode::kGripperBinary:
    case ControlMode::kGripperPosition:
    case ControlMode::kCompositeMode: {
      // ADR-0028b — per-mode chunks. The C++ kernel intentionally
      // delegates per-axis bound enforcement to the Python
      // ``openral_safety/supervisor_node.py`` which knows the per-mode
      // bounds declared on the robot manifest (``max_cartesian_step_m``,
      // ``max_base_linear_speed_m_s``, ``max_base_angular_speed_rad_s``,
      // ``gripper_min`` / ``gripper_max``, …). The kernel still ran
      // shape + NaN checks above so the chunk is structurally sound;
      // routing it through unrejected lets the supervisor do its job
      // before the HAL applies. Without this case the per-mode chunks
      // hit the default branch and trip an estop before the supervisor
      // sees them. (Conservatism: net safety is strictly improved vs
      // pre-change — we go from "reject every per-mode chunk" to
      // "structural-validate then delegate to Python's per-mode bounds
      // checker".)
      //
      // ADR-0028d — ``kCompositeMode`` carries a single robosuite-
      // specific multiplexer flag value in [-1, +1] (sim-only). No
      // per-joint or workspace bound applies; the kernel validates
      // shape + NaN/Inf above and passes through.
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
