#!/usr/bin/env bash
# Record a Nav2-driven "turn 90 left + drive 2m forward" on NavigateKitchen:
# emit the operator prompt, run /spin then /drive_on_heading, while the cinecam
# (offscreen) and the dashboard recorder capture it.
REPO=/home/allopart/workspace/openral
VID=/home/allopart/workspace/_deploy_videos
CINE=$VID/cine_nav_close
DASHDIR=$VID/dashvid_nav_cmd
WT=$REPO/.claude/worktrees/deploy-scene-videos/_deploy_capture
cd "$REPO"
source /opt/ros/jazzy/setup.bash; source install/setup.bash
export ROS_DOMAIN_ID=0
mkdir -p "$DASHDIR"

echo "MARK_IDLE=$(ls "$CINE"/*.jpg 2>/dev/null | wc -l)"
# start dashboard video recorder (background) for the whole motion
nohup node "$WT/record_dashboard.js" "http://127.0.0.1:4318/" "$DASHDIR" 30000 > "$VID/_navrec_dash.log" 2>&1 &
DASHPID=$!
sleep 3
echo "MARK_CMD=$(ls "$CINE"/*.jpg 2>/dev/null | wc -l)"
# turn 90 degrees left (CCW = +yaw)
echo "[spin 90 left]"
timeout 30 ros2 action send_goal /spin nav2_msgs/action/Spin "{target_yaw: 1.5708, time_allowance: {sec: 20, nanosec: 0}}" 2>&1 | grep -E "status:" | tail -1
# drive forward 2 meters
echo "[drive 2m forward]"
timeout 35 ros2 action send_goal /drive_on_heading nav2_msgs/action/DriveOnHeading "{target: {x: 2.0, y: 0.0, z: 0.0}, speed: 0.4, time_allowance: {sec: 30, nanosec: 0}}" 2>&1 | grep -E "status:" | tail -1
echo "MARK_END=$(ls "$CINE"/*.jpg 2>/dev/null | wc -l)"
wait $DASHPID 2>/dev/null
echo "DONE cine_frames=$(ls "$CINE"/*.jpg | wc -l)"
