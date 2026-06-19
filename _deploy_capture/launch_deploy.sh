#!/usr/bin/env bash
# Launch a robocasa deploy-sim stack (panda_mobile) with viewer + dashboard.
# Runs from the MAIN repo (already built overlay + robocasa .venv) to stay
# within the 18GB disk budget. Usage: launch_deploy.sh <scene_yaml> <dash_port>
# NB: no `set -u` — ROS setup.bash references unbound AMENT_TRACE_SETUP_FILES.
REPO=/home/allopart/workspace/openral
SCENE="${1:-scenes/deploy/robocasa_pnp.yaml}"
PORT="${2:-4318}"
cd "$REPO"

source /opt/ros/jazzy/setup.bash
source install/setup.bash
# Worktree HAL source shadows the main editable install so the deploy-viewer
# camera tweak (depth_cloud.py lookat-lift + pullback) takes effect.
WT=/home/allopart/workspace/openral/.claude/worktrees/deploy-scene-videos
export PYTHONPATH="$WT/python/hal/src:$REPO/.venv/lib/python3.12/site-packages:${PYTHONPATH:-}"

# --- runtime env (deploy-sim robocasa recipe) ---
export MUJOCO_GL=egl                                   # offscreen camera render on GPU
export OPENRAL_AUTO_INSTALL_DEPS=1                      # no interactive robocasa prompt
export OPENRAL_ALLOW_NONCOMMERCIAL=1                   # rldx/pi05 robocasa weight posture
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True # 8GB headroom
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-0}

echo "[launch] scene=$SCENE dash=$PORT DISPLAY=$DISPLAY ROS_DOMAIN_ID=$ROS_DOMAIN_ID"
# SIM_CLOCK_FLAG: robocasa has a sim clock (use --enable-sim-clock); other
# MuJoCo twins (openarm/so101/libero franka) don't → must run wall-clock.
exec .venv/bin/openral deploy sim \
  --config "$SCENE" \
  --dashboard --dashboard-port "$PORT" \
  ${SIM_CLOCK_FLAG:---enable-sim-clock} \
  --no-enable-octomap-kernel-check \
  --no-object-detector \
  --hal viewer_enabled=false
