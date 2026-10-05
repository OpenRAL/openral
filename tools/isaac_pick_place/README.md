# Isaac pick/place validation run (OpenArm)

A scripted, perception-driven OpenArm driver that picks objects and places them in the
Isaac deploy sim, through the same graph a VLA runs in: the runner, the C++ safety
kernel's octomap world-collision check, the grasp-target exemption (ADR-0115) and the
vision attachment leg. It validates that stack end to end without a trained policy. It
is not a policy and never runs on a real robot.

```bash
tools/isaac_pick_place/run_trial.sh outputs/isaac_pick_place/run1 300
```

Requirements: a ROS 2 workspace built from this checkout (`install/setup.bash`) and its
`.venv`, an RTX GPU with Isaac Sim 6.1 (`OPENRAL_ISAAC_SIDECAR_PYTHON`, default
`/opt/isaac-sim/python.sh`), access to NVIDIA's asset server, and `ffmpeg`.

| File | Role |
| --- | --- |
| `run_trial.sh` | Brings up `openral deploy sim` on `scene.yaml`, waits for the HAL and the voxel map, dispatches the driver as an rSkill goal, records `run.mp4`, tears the graph down. |
| `scene.yaml` | Two YCB foam bricks on the warehouse pallet; octomap check, grasp-target exemption and vision leg on, no reasoner. |
| `driver.py` | The driver: scan the voxel map, choose a target and hand, hover until the grasp region is measured, descend, close, lift, carry, place, retreat. |
| `kin.py` | IK on the OpenArm's MJCF (robot model only). |
| `rskill/rskill.yaml` | The driver's rSkill, linked into `rskills/` only during a run. |
| `sitecustomize.py` | Registers the driver's policy family in the runner process. |
| `send_goal.py`, `wait_voxels.py`, `recorder.py` | Goal dispatch, map readiness, video. |

The driver's world inputs are `/openral/world_voxels` and `/openral/attachment_state`
only; it reads no simulator ground truth. It never vetoes contact itself: without a
measured grasp region the kernel decides whether the fingers may enter the target.

`<out_dir>/driver.jsonl` logs every phase and decision (`target`, `grasp_centre`,
`region_yaw_kept`, `close_pose`, `attach_pose`, `place_spot`, ...). The goal ends
`deadline_exceeded`: the driver holds after its last place (`phase: done`) instead of
ending the episode.
