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
tools/record_demo.sh clip1_robocasa_pi05 300
ros2 action send_goal /openral/execute_rskill openral_msgs/action/ExecuteRskill \
  "{rskill_id: 'OpenRAL/rskill-pi05-robocasa365-human300-nf4', \
    prompt: 'Pick the baguette from the counter and place it in the cabinet.', \
    deadline_s: 180.0}"
```
Expect: real base navigation + arm reach + safety-gated grasp **attempt** (pi05 self-collides
non-deterministically → kernel E-stop; NOT a guaranteed completed grasp).

### 1b. Autonomous + open-vocab perception (LocateAnything-3B)   [ADR-0050 — code-complete, live-verify pending]
Real open-vocab perception of the baguette, then the autonomous grab — fits 8 GB via the
single-resident-skill eviction (the detector's VRAM is freed before pi05 loads).
```bash
source tools/_demo_env.sh
openral deploy sim --config scenes/deploy/robocasa_baguette.yaml --no-enable-octomap \
  --object-detector-manifest rskills/locateanything-3b-nf4/rskill.yaml \
  --object-detector-query "baguette"
# Drive it: type a goal in the dashboard prompt (autonomous reasoner), e.g. "pick up the baguette".
# Flow: detector loads (~5.3 GB) → reasoner locate_in_view('baguette') → navigate
#       → reasoner DEACTIVATEs the detector (frees VRAM, ADR-0050) → pi05 loads + grabs.
# If the LLM doesn't free the detector itself before the grab, do it manually:
ros2 lifecycle set /openral_ros_image_detector deactivate   # frees the detector's VRAM
```
**8 GB co-residency:** the detector (~5.3 GB) and pi05 (~4.3 GB) do not co-reside — the deactivate
above is what makes the autonomous loop fit. Watch `nvidia-smi`. Known live-verify risks: the
detector's `image_topic` default (`agentview_left`) may not match the robocasa camera names
(no `--object-detector-image-topic` override exists yet); the LLM may need a nudge to deactivate
the detector before grabbing.

---

## Clip 2 — MuJoCo · OpenArm tabletop · pick   [CONFIG REAL, e2e NOT verified this session]
```bash
openral deploy sim --config scenes/deploy/openarm_tabletop.yaml
chromium --new-window --app=http://127.0.0.1:4318/ &
tools/record_demo.sh clip2_openarm_pick 300
ros2 action send_goal /openral/execute_rskill openral_msgs/action/ExecuteRskill \
  "{rskill_id: 'OpenRAL/rskill-pi05-openarm-vision-nf4', \
    prompt: 'Pick up the object on the table.', deadline_s: 120.0}"
```
The reasoner picks the rSkill at runtime, or dispatch directly as above. If the dashboard cards
stay empty, fall back to the `openral sim run ... --video` 3-panel MP4 for this one.

---

## Clip 3 — Isaac Sim · Franka Panda · pick (bowl→plate)
Isaac runs via the py3.11 sidecar venv (`~/.cache/openral/isaac-sidecar/.venv`, auto-resolved).
**One Kit app at a time** — make sure no other Isaac process is running.

### 3a. deploy-sim path (WITH dashboard)   [CONFIG REAL, deploy e2e NOT verified this session]
```bash
openral deploy sim --config scenes/deploy/isaac_franka_bowl.yaml
chromium --new-window --app=http://127.0.0.1:4318/ &
tools/record_demo.sh clip3_isaac_franka 300
ros2 action send_goal /openral/execute_rskill openral_msgs/action/ExecuteRskill \
  "{rskill_id: 'OpenRAL/rskill-act-libero', prompt: 'put the bowl on the plate', deadline_s: 120.0}"
```

### 3b. sim-run path (NO dashboard, but VERIFIED in ADR-0045) — fallback
```bash
openral sim run --config scenes/sim/isaac_franka_bowl_plate.yaml \
  --rskill rskill://rskills/act-libero --video recordings/clip3_isaac_franka_simrun.mp4
```
Expect: real RTX render + ACT drives the arm via Lula IK. Task success is OOD → `success=False`
is expected; the point is perception + actuation, not a completed place.

---

## Clip 4 — Isaac Sim · panda_mobile · autonomous nav   [deploy e2e verified in ADR-0045]
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
pkill -f sim_e2e.launch.py            # stop the deploy-sim graph
pkill -f isaac_sidecar.py             # stop Isaac sidecar (clips 3/4)
nvidia-smi                            # confirm VRAM freed before the next scene
```
