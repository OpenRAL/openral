// SPDX-License-Identifier: Apache-2.0
// ADR-0020 — envelope.cpp: build the EnvelopeIntersection from per-field
// ROS parameters the Python launch populates from
// `robots/<id>/robot.yaml`. Single-shot (configure-time) so allocations
// are fine here; the hot validator path never touches this code.

#include "openral_safety_kernel/envelope.hpp"

#include <sstream>
#include <vector>

#include <rclcpp_lifecycle/lifecycle_node.hpp>

namespace openral_safety_kernel {

namespace {

std::vector<double> read_double_array(rclcpp_lifecycle::LifecycleNode& node,
                                      const std::string& name) {
  return node.get_parameter(name).as_double_array();
}

bool read_xyz_array(rclcpp_lifecycle::LifecycleNode& node,
                    const std::string& name,
                    std::array<double, 3>& out) {
  const auto values = read_double_array(node, name);
  if (values.size() != 3) {
    return false;
  }
  out[0] = values[0];
  out[1] = values[1];
  out[2] = values[2];
  return true;
}

}  // namespace

EnvelopeLoadStatus load_envelope_from_ros_parameters(
    rclcpp_lifecycle::LifecycleNode& node,
    EnvelopeIntersection& out,
    std::string& error_message) {
  error_message.clear();
  out = EnvelopeIntersection{};

  const std::int64_t n_dof_signed = node.get_parameter("n_dof").as_int();
  if (n_dof_signed <= 0) {
    error_message =
        "n_dof is 0; no envelope supplied. Boot through "
        "`openral deploy sim` so the launch forwards robots/<id>/robot.yaml "
        "as ROS parameters on this node.";
    return EnvelopeLoadStatus::kUnconfigured;
  }
  out.n_dof = static_cast<std::size_t>(n_dof_signed);

  out.robot_name = node.get_parameter("robot_name").as_string();
  out.rskill_id = node.get_parameter("rskill_id").as_string();
  out.skill_revision = node.get_parameter("skill_revision").as_string();

  out.joint_position_min = read_double_array(node, "joint_position_min");
  out.joint_position_max = read_double_array(node, "joint_position_max");
  out.joint_velocity_max = read_double_array(node, "joint_velocity_max");
  out.joint_torque_max = read_double_array(node, "joint_torque_max");

  if (out.joint_position_min.size() != out.n_dof
      || out.joint_position_max.size() != out.n_dof
      || out.joint_velocity_max.size() != out.n_dof
      || out.joint_torque_max.size() != out.n_dof) {
    std::ostringstream oss;
    oss << "joint_* parameter arrays disagree with n_dof=" << out.n_dof
        << " (min=" << out.joint_position_min.size()
        << " max=" << out.joint_position_max.size()
        << " vel=" << out.joint_velocity_max.size()
        << " tau=" << out.joint_torque_max.size() << ")";
    error_message = oss.str();
    return EnvelopeLoadStatus::kInvalidShape;
  }

  const bool have_min = read_xyz_array(node, "workspace_box_min_xyz", out.workspace_box.min_xyz);
  const bool have_max = read_xyz_array(node, "workspace_box_max_xyz", out.workspace_box.max_xyz);
  out.workspace_box.set = have_min && have_max;

  out.max_ee_speed_m_s = node.get_parameter("max_ee_speed_m_s").as_double();
  out.max_ee_accel_m_s2 = node.get_parameter("max_ee_accel_m_s2").as_double();
  out.max_force_n = node.get_parameter("max_force_n").as_double();
  out.max_torque_nm = node.get_parameter("max_torque_nm").as_double();
  out.contact_force_threshold_n =
      node.get_parameter("contact_force_threshold_n").as_double();
  out.deadman_required = node.get_parameter("deadman_required").as_bool();

  return EnvelopeLoadStatus::kOk;
}

}  // namespace openral_safety_kernel
