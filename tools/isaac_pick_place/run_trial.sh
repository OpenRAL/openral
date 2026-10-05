#!/usr/bin/env bash
# One Isaac deploy-sim pick/place validation run with the scripted OpenArm driver.
#
# Usage: tools/isaac_pick_place/run_trial.sh <out_dir> [deadline_s]
#
# Brings up `openral deploy sim` on tools/isaac_pick_place/scene.yaml (octomap world
# check, grasp-target exemption and vision attachment leg on), waits for the HAL and the
# voxel map, dispatches the scripted driver as an rSkill goal, records run.mp4 and tears
# the graph down. <out_dir> gets graph.log, driver.jsonl, goal_result.json and run.mp4.
#
# Needs a sourced ROS 2 workspace built from this checkout (install/setup.bash), its
# .venv, ffmpeg, and Isaac Sim (OPENRAL_ISAAC_SIDECAR_PYTHON, default
# /opt/isaac-sim/python.sh). Env: ROS_DOMAIN_ID (default 91), PICK_PLACE_FIRST_SIDE
# (left/right), PICK_PLACE_PICKS, PICK_PLACE_VIZ=1 (dashboard on 4318, Foxglove on
# PICK_PLACE_FOXGLOVE_PORT, default 8766).
set -euo pipefail

[[ $# -ge 1 ]] || { echo "usage: $0 <out_dir> [deadline_s]" >&2; exit 2; }
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd "${here}/../.." && pwd)"
out="$(mkdir -p "$1" && cd "$1" && pwd)"
deadline="${2:-300}"
py="${root}/.venv/bin/python"
link="${root}/rskills/_isaac_pick_place_driver"

set +u  # the ROS setup scripts read unset variables
source /opt/ros/jazzy/setup.bash
source "${root}/install/setup.bash"
set -u
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-91}"
export OPENRAL_ISAAC_SIDECAR_PYTHON="${OPENRAL_ISAAC_SIDECAR_PYTHON:-/opt/isaac-sim/python.sh}"
export OMNI_KIT_ACCEPT_EULA=YES HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export PYTHONPATH="${here}${PYTHONPATH:+:${PYTHONPATH}}" PICK_PLACE_DRIVER=1
export PICK_PLACE_LOG="${out}/driver.jsonl"

# The driver's rSkill is visible to the runner only for this run (rskills/ is the
# runner's search path; the link is gitignored).
ln -sfn "${here}/rskill" "${link}"
pids=()
cleanup() {
  for p in "${pids[@]}"; do kill -INT "$p" 2>/dev/null || true; done
  [[ -n "${graph:-}" ]] && kill -INT -- "-${graph}" 2>/dev/null || true
  for _ in $(seq 1 60); do
    [[ -n "${graph:-}" ]] && kill -0 -- "-${graph}" 2>/dev/null || break
    sleep 1
  done
  [[ -n "${graph:-}" ]] && kill -KILL -- "-${graph}" 2>/dev/null || true
  # This checkout's Isaac sidecar and ROS nodes only: they can outlive the launch's
  # process group, and a leaked octomap_server poisons the next run on the domain.
  pkill -u "$(id -u)" -f "${root}/tools/isaac_sidecar.py" || true
  pkill -u "$(id -u)" -f "${root}/install/lib/" || true
  rm -f "${link}"
}
trap cleanup EXIT

viz=(--no-dashboard --no-foxglove)
if [[ "${PICK_PLACE_VIZ:-0}" == 1 ]]; then
  viz=(--dashboard --dashboard-port 4318 --foxglove --foxglove-port "${PICK_PLACE_FOXGLOVE_PORT:-8766}")
fi
# The Isaac HAL steps at ~11 Hz wall, so the grasp trigger's gap window is widened from
# the 30 Hz joint-state default (0.133 s) or one late sample clears its history.
cd "${root}"
setsid "${root}/.venv/bin/openral" deploy sim --config "${here}/scene.yaml" "${viz[@]}" \
  --hal viewer_enabled=false --hal depth_publish_rate_hz=30.0 \
  --hal vision_attachment_trigger_max_gap_s=0.5 > "${out}/graph.log" 2>&1 &
graph=$!

timeout 1500 bash -c "until grep -qE 'segmenter warmed|Failed to make transition' '${out}/graph.log' \
  && grep -q 'rskill_runner session id' '${out}/graph.log'; do sleep 3; done"
timeout 1500 bash -c "until timeout 5 ros2 topic echo --once /joint_states --field name >/dev/null 2>&1; do sleep 5; done"
"${py}" "${here}/wait_voxels.py" 30
"${py}" "${here}/recorder.py" "${out}/run.mp4" > "${out}/recorder.log" 2>&1 &
pids+=($!)
sleep 8
"${py}" "${here}/send_goal.py" "${out}/goal_result.json" "${deadline}" | tee "${out}/goal.log"
sleep 8
