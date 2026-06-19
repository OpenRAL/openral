#!/usr/bin/env bash
# Dispatch a VLA rSkill directly to the deploy-sim skill runner (bypasses the
# flaky reasoner LLM). Blocks until the action completes or the deadline.
# Usage: dispatch.sh <rskill_id> <prompt> <deadline_s>
REPO=/home/allopart/workspace/openral
RSKILL="${1:-OpenRAL/rskill-pi05-robocasa365-human300-nf4}"
PROMPT="${2:-Pick the object from the counter and place it in the cabinet.}"
DEADLINE="${3:-180.0}"
cd "$REPO"
source /opt/ros/jazzy/setup.bash
source install/setup.bash
export PYTHONPATH="$REPO/.venv/lib/python3.12/site-packages:${PYTHONPATH:-}"
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-0}
echo "[dispatch] $RSKILL :: '$PROMPT' deadline=${DEADLINE}s"
ros2 action send_goal /openral/execute_rskill openral_msgs/action/ExecuteRskill \
  "{rskill_id: '$RSKILL', revision: '', prompt: '$PROMPT', prompt_metadata_json: '', goal_params_json: '', deadline_s: $DEADLINE}"
