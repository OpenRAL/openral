#!/usr/bin/env bash
# Launch the REAL OpenArm cell with the kernel's world-voxel check ON — for a human at the cell.
#
# Runbook: docs/tutorials/deploy/openarm-real-world-voxel-check.md (step 4).
#
# `openral deploy run` checks none of the OpenArm motion gates, and bringing the graph up
# MOVES the arms (on_activate steps all 16 motors to zero, unramped). This wrapper is the
# only sanctioned way to launch scenes/deploy/openarm_real_world_voxels.yaml, and it refuses
# unless, in this order:
#   1. OPENRAL_OPENARM_ALLOW_MOTION=1 and OPENRAL_OPENARM_ATTENDED=1 (the HIL tier's gates);
#   2. the head_zed extrinsic report verifies against the pose in robots/openarm/robot.yaml
#      (the camera is bolted to the robot, so its pose is robot geometry, not the scene's);
#   3. it runs in an interactive terminal and the operator types the confirmation.
# Extra arguments pass through to `openral deploy run` only from an allow-list of
# observability flags (--foxglove, --dataset-out <dir>, ...). Anything else — a second
# --config, a --no-enable-octomap-kernel-check — could swap or weaken the graph the gates
# just verified, so it is refused.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
scene="${root}/scenes/deploy/openarm_real_world_voxels.yaml"
robot="${root}/robots/openarm/robot.yaml"
report="${root}/robots/openarm/calibration/head_zed_extrinsic.json"

refuse() {
  echo "REFUSED: $*" >&2
  exit 2
}

# Allow-listed pass-through only. Flags taking a value consume the next argument.
passthrough=()
while (($#)); do
  case "$1" in
    --foxglove | --no-dashboard)
      passthrough+=("$1")
      shift
      ;;
    --dataset-out | --dataset-repo-id | --dataset-license | --dashboard-port | --foxglove-port)
      (($# >= 2)) || refuse "$1 needs a value."
      passthrough+=("$1" "$2")
      shift 2
      ;;
    *)
      refuse "argument '$1' is not allowed here: only observability flags pass through; the scene, robot and safety posture are fixed by this wrapper."
      ;;
  esac
done

[[ "${OPENRAL_OPENARM_ALLOW_MOTION:-}" == "1" ]] ||
  refuse "OPENRAL_OPENARM_ALLOW_MOTION is not 1 — this graph moves the arms at bringup."
[[ "${OPENRAL_OPENARM_ATTENDED:-}" == "1" ]] ||
  refuse "OPENRAL_OPENARM_ATTENDED is not 1 — export it only while you are at the E-stop."
[[ -n "${ROS_DISTRO:-}" ]] || refuse "ROS 2 is not sourced (and source the ZED overlay too)."
command -v openral >/dev/null || refuse "openral is not on PATH (activate the venv)."

python "${root}/tools/zed_extrinsic_check.py" verify --robot "${robot}" --report "${report}" ||
  refuse "head_zed extrinsic not verified for the manifest's pose (runbook step 2)."
[[ -t 0 ]] || refuse "not an interactive terminal; a person at the cell launches this."

echo "Bringup steps all 16 motors to zero UNRAMPED. Arms parked near zero, cell clear,"
echo "hardware E-stop in your hand, second person on the stop?"
read -r -p "Type ESTOP IN HAND to launch: " answer
[[ "${answer}" == "ESTOP IN HAND" ]] || refuse "not confirmed."

exec openral deploy run --config "${scene}" "${passthrough[@]}"
