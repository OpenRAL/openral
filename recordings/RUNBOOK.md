# Sim + Dashboard Demo Recording — Runbook

All commands run from the repo root: `cd /home/allopart/workspace/openral`

## 0. One-time per shell — env
```bash
source tools/_demo_env.sh        # strips conda, sources ROS overlay + venv, ROS_DOMAIN_ID=42, DISPLAY=:1
```
Dashboard (every scene) serves at **http://127.0.0.1:4318/**.
Open it in a window next to the viewer:
```bash
chromium --new-window --app=http://127.0.0.1:4318/ &
```

## Recording
The MuJoCo/Isaac viewer opens on the **external monitor DP-4-3** (region `2560x1440+2560+0`),
which is what `record_demo.sh` grabs by default. Arrange viewer + dashboard to fill that
monitor, then in a SECOND terminal:
```bash
tools/record_demo.sh <name> 300          # records ~5 min -> recordings/<name>.mp4
# stop early:  touch recordings/.<name>.stop
# other monitor instead: REC_REGION=2560x1600+0+940 tools/record_demo.sh <name> 300
```

---

## Clip 1 — MuJoCo · RoboCasa kitchen · panda_mobile · navigate→find→grab (pi05)

### 1a. Direct dispatch (simplest, deterministic)   [LAUNCH VERIFIED]
```bash
# terminal A — graph (viewer + dashboard + SLAM + Nav2 + detector)
openral deploy sim --config scenes/deploy/robocasa_baguette.yaml --no-enable-octomap

# terminal B — record, then dispatch the mobile-manip policy (drives base AND arm).
# Do NOT type in the dashboard prompt box (that hands control to the autonomous reasoner).
tools/record_demo.sh clip1_robocasa_rldx 300
ros2 action send_goal /openral/execute_rskill openral_msgs/action/ExecuteRskill \
  "{rskill_id: 'OpenRAL/rskill-rldx1_ft-panda_mobile-robocasa-nf4', \
    prompt: 'Pick the baguette from the counter and place it in the cabinet.', \
    deadline_s: 180.0}"
```
Expect: real base navigation + arm reach under safety-gated control.

### 1b. Autonomous + open-vocab perception (LocateAnything-3B)   [VRAM eviction — code-complete, live-verify pending]
Real open-vocab perception of the baguette, then the autonomous grab — fits 8 GB via the
single-resident-skill eviction (the detector's VRAM is freed before pi05 loads).
```bash
source tools/_demo_env.sh
openral deploy sim --config scenes/deploy/robocasa_baguette.yaml --no-enable-octomap \
  --object-detector-manifest rskills/locateanything-3b-nf4/rskill.yaml \
  --object-detector-query "baguette"
# Drive it: type a goal in the dashboard prompt (autonomous reasoner), e.g. "pick up the baguette".
# Flow: detector loads (~5.3 GB) → reasoner locate_in_view('baguette') → navigate
#       → reasoner DEACTIVATEs the detector (frees VRAM) → pi05 loads + grabs.
# If the LLM doesn't free the detector itself before the grab, do it manually:
ros2 lifecycle set /openral_ros_image_detector deactivate   # frees the detector's VRAM
```
**8 GB co-residency:** the detector (~5.3 GB) and pi05 (~4.3 GB) do not co-reside — the deactivate
above is what makes the autonomous loop fit. Watch `nvidia-smi`. Known live-verify risks: the
detector's `image_topic` default (`agentview_left`) may not match the robocasa camera names
(no `--object-detector-image-topic` override exists yet); the LLM may need a nudge to deactivate
the detector before grabbing.

---

## Clip 2 — MuJoCo · OpenArm tabletop · scene only   [no shipped VLA — perception/MoveIt only]
```bash
openral deploy sim --config scenes/deploy/openarm_tabletop.yaml
chromium --new-window --app=http://127.0.0.1:4318/ &
tools/record_demo.sh clip2_openarm 300
```
> **No in-tree VLA drives OpenArm.** The `pi05-openarm-vision-nf4` policy was
> removed (real-trained checkpoint, sim-to-real gap, no verified grasp). The
> OpenArm scene is retained for perception (detector) and MoveIt (`rskill-moveit-*`)
> demos; dispatch one of those, or skip this clip until a task-matched OpenArm
> policy lands.

---

## Clip 3 — Isaac Sim · OpenArm · warehouse pallet (pickable YCB props)
Isaac is resolved automatically (sidecar venv, or a binary install such as `/opt/isaac-sim`).
**One Kit app at a time** — make sure no other Isaac process of yours is running.

```bash
openral deploy sim --config scenes/deploy/isaac_openarm_warehouse.yaml --no-enable-octomap
chromium --new-window --app=http://127.0.0.1:4318/ &
tools/record_demo.sh clip3_isaac_openarm 300
```
The former Franka bowl→plate clip used the `bowl_plate` layout, removed because it never
booted on Isaac Sim 6.1.

---

## Clip 4 — Isaac Sim · panda_mobile · autonomous nav   [deploy e2e verified during the Isaac backend integration]
```bash
openral deploy sim --config scenes/deploy/isaac_panda_mobile_urdf.yaml
chromium --new-window --app=http://127.0.0.1:4318/ &
tools/record_demo.sh clip4_isaac_panda_mobile_nav 300
# send a Nav2 goal (RViz "2D Goal Pose", or):
ros2 action send_goal /navigate_to_pose nav2_msgs/action/NavigateToPose \
  "{pose: {header: {frame_id: 'map'}, pose: {position: {x: 2.0, y: 0.0}, \
    orientation: {w: 1.0}}}}"
```

---

## Teardown between scenes
```bash
pkill -f deploy_e2e.launch.py            # stop the deploy-sim graph
pkill -f isaac_sidecar.py             # stop Isaac sidecar (clips 3/4)
nvidia-smi                            # confirm VRAM freed before the next scene
```
