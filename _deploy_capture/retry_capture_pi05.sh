#!/usr/bin/env bash
# Retry fresh launches until pi05 gets a healthy run (>=MINSTEP steps without an
# early near-miss estop). On success, leave the stack up, record the dashboard
# video during the run, and write the cinecam motion-start frame to a marker.
# The cinecam records continuously (offscreen) the whole time.
REPO=/home/allopart/workspace/openral
WT=$REPO/.claude/worktrees/deploy-scene-videos/_deploy_capture
VID=/home/allopart/workspace/_deploy_videos
CINE=$VID/cine_pnp_v2
DASHDIR=$VID/dashvid_pi05_v2
MINSTEP=80
cd "$WT"
mkdir -p "$DASHDIR"

teardown() {
  pkill -f "sim_e2e.launch" 2>/dev/null; pkill -f "lifecycle_node.py" 2>/dev/null
  pkill -f "reasoner_node" 2>/dev/null; pkill -f "runtime_node" 2>/dev/null
  pkill -f "safety_kernel_node" 2>/dev/null; pkill -f "prompt_router" 2>/dev/null
  pkill -f "openral_world_state" 2>/dev/null; pkill -f "openral dashboard" 2>/dev/null
  pkill -f "wait_for_action" 2>/dev/null; pkill -f "lifecycle_autostart" 2>/dev/null
  sleep 5
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do kill -9 $p 2>/dev/null; done
  sleep 2
}

for a in 1 2 3 4 5 6; do
  echo "===== ATTEMPT $a ====="
  teardown
  LOG=$VID/_pnp_v2_a$a.log
  nohup env OPENRAL_CINECAM_DIR="$CINE" OPENRAL_CINECAM_SIZE=1280x960 OPENRAL_CINECAM_FPS=15 \
    OPENRAL_CINECAM_AZ_OFFSET_DEG=30 OPENRAL_CINECAM_EL_OFFSET_DEG=30 OPENRAL_CINECAM_DIST_DELTA_M=-1 \
    bash launch_deploy.sh scenes/deploy/robocasa_pnp.yaml 4318 > "$LOG" 2>&1 &
  # wait for cinecam up (HAL active)
  for i in $(seq 1 30); do grep -q "cinecam recording" "$LOG" && break; sleep 3; done
  sleep 4
  MARK=$(ls "$CINE"/*.jpg 2>/dev/null | wc -l)
  nohup bash dispatch.sh "OpenRAL/rskill-pi05-robocasa365-human300-nf4" "Pick the object from the counter and place it in the cabinet." 200.0 > "$VID/_disp_v2_a$a.log" 2>&1 &
  for i in $(seq 1 30); do grep -qE "policy_step step=1 |refused observation" "$LOG" 2>/dev/null && break; sleep 2; done
  sleep 22
  STEP=$(grep -oE 'policy_step step=[0-9]+' "$LOG" | tail -1 | grep -oE '[0-9]+')
  echo "attempt $a reached step=${STEP:-0} (cine_mark=$MARK)"
  if [ "${STEP:-0}" -ge "$MINSTEP" ]; then
    echo "SUCCESS attempt=$a step=$STEP cine_mark=$MARK"
    echo "$MARK" > /tmp/v2_cine_mark.txt
    # record dashboard during the (still-running) pi05 run
    timeout 70 node record_dashboard.js "http://127.0.0.1:4318/" "$DASHDIR" 22000 2>&1 | tail -1
    pkill -9 -f "ms-playwright|headless_shell" 2>/dev/null
    FINAL=$(grep -oE 'policy_step step=[0-9]+' "$LOG" | tail -1 | grep -oE '[0-9]+')
    echo "CAPTURED cine_mark=$MARK final_step=$FINAL cine_frames=$(ls "$CINE"/*.jpg | wc -l)"
    exit 0
  fi
done
echo "ALL ATTEMPTS FAILED (pi05 near-miss estopped early every time)"
exit 1
