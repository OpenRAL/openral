#!/usr/bin/env bash
# Dev smoke runner for the C++ safety kernel. Probes SCHED_FIFO
# availability, builds a tiny envelope file, and launches the kernel for
# manual interactive testing.
#
# Usage: cpp/openral_safety_kernel/scripts/dev_run.sh
#
# Pre-req: `colcon build --packages-select openral_safety_kernel` and a
# sourced `install/setup.bash` from the workspace root.

set -euo pipefail

if ! command -v ros2 >/dev/null 2>&1; then
  echo "[dev_run] ROS 2 not sourced; run 'source /opt/ros/<distro>/setup.bash' first." >&2
  exit 1
fi
if ! ros2 pkg executables openral_safety_kernel | grep -q safety_kernel_node; then
  echo "[dev_run] openral_safety_kernel not built; run 'colcon build --packages-select openral_safety_kernel'" >&2
  exit 1
fi

# RT-scheduling probe — best-effort; we never block on it.
if [[ -r /proc/sys/kernel/sched_rt_runtime_us ]]; then
  echo "[dev_run] kernel.sched_rt_runtime_us=$(cat /proc/sys/kernel/sched_rt_runtime_us)"
fi

ENV_FILE="${ENV_FILE:-/tmp/openral_safety_dev_envelope.yaml}"
cat >"${ENV_FILE}" <<'EOF'
schema_version: 1
robot_name: dev_so100
rskill_id: ""
skill_revision: ""
trace_id_at_load: ""
intersection:
  n_dof: 6
  joint_position_min: [-2.0944, -1.7453, -1.7453, -1.7453, -2.7925, 0.0]
  joint_position_max: [2.0944, 1.7453, 1.7453, 1.7453, 2.7925, 1.0]
  joint_velocity_max: [3.15, 3.15, 3.15, 3.15, 3.15, 3.15]
  joint_torque_max: [5.0, 5.0, 5.0, 5.0, 5.0, 5.0]
  workspace_box_min_xyz: [-0.4, -0.4, 0.0]
  workspace_box_max_xyz: [0.4, 0.4, 0.6]
  max_ee_speed_m_s: 0.5
  max_ee_accel_m_s2: 2.0
  max_force_n: 10.0
  max_torque_nm: 3.0
  contact_force_threshold_n: 5.0
  deadman_required: true
EOF
echo "[dev_run] wrote envelope to ${ENV_FILE}"

exec ros2 run openral_safety_kernel safety_kernel_node \
  --ros-args -p envelope_file:="${ENV_FILE}" \
             -p estop_reset_cooldown_s:=0.25
