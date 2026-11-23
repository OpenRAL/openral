# Galaxea R1 Pro (BEHAVIOR-1K simulation)

This manifest describes the simulated R1 Pro used by the 2026 BEHAVIOR-1K
challenge. Joint position, velocity, and effort limits come from the upstream
`behavior-1k/omnigibson-robot-assets` `r1pro.urdf`.

`assets.urdf` is `behavior:r1pro`: that same `r1pro.urdf` at the pinned dataset
revision, downloaded on first use, with absolute mesh paths and without the
collision meshes BEHAVIOR does not publish — visual geometry only
(`openral_hal._behavior_r1pro_assets`).

`assets.mjcf` is the same `behavior:r1pro` ref as an MJCF: the `deploy sim`
MuJoCo twin built from that URDF (`ensure_r1pro_mjcf`):

- **Collision.** BEHAVIOR publishes no collision meshes, so each visual mesh is
  also a collision geom, which MuJoCo collides as its convex hull: a
  conservative envelope per link, never smaller than the part. Hulls collide
  with the world, not with each other (self-collision is the safety kernel's).
- **Base.** The swerve steer/wheel joints are fixed; planar `base_x` / `base_y` /
  `base_yaw` joints (qpos 0..2, not manifest joints) carry the base, placed
  kinematically by `BODY_TWIST` (holonomic, one control period per command).
- **Actuators.** 20 position actuators: torso, both arms, and each hand's
  `finger_joint1` (forcerange = the URDF effort limit; gains are a stiff loop,
  not motor data). `finger_joint2` follows `finger_joint1` through an MJCF
  equality, as OmniGibson drives both fingers from one command, so a
  `finger_joint2` target is ignored.
- **Grippers.** The end effectors declare `normalized_open_unit` (0 closed, 1
  open), the twin's encoding. The BEHAVIOR scene declares its own
  `normalized_open_symmetric`, which the runners pass as the scene convention,
  so BEHAVIOR runs are unchanged.
- **Rate.** `action_spec.control_freq_hz: 30`, BEHAVIOR's challenge
  `action_frequency`.
- **Cameras.** `head` on `zed_link`, the wrist cameras on `*_realsense_link`,
  each looking down the frame's +z (the URDF frames are REP-103 optical).

```bash
openral deploy sim --config scenes/deploy/r1pro_twin.yaml
```

The manifest has no `base_joints`, so `deploy sim` attaches no
`MobileBaseBridge`: the twin's base moves by `BODY_TWIST` from the HAL, not from
`/cmd_vel`, and there is no `/odom`.

Every manifest joint names its MJCF joint (`sim_joint_name`). Automatic sim grasp
evidence (`SimAttachmentEvidenceTracker`) stays off on this robot: it handles one
gripper parent link, and the R1 Pro has two, as the OpenArm and ALOHA do. Deploy
logs "automatic sim attachment evidence disabled".

The robot is **simulation-only** in OpenRAL. `hal.real` remains null; no physical
R1 Pro safety or transport claim is made.

```bash
openral deploy sim --config scenes/deploy/behavior_r1pro.yaml \
  --initial-task "turn on the radio"
```

The scene sidecar requires the official BEHAVIOR environment via
`OPENRAL_BEHAVIOR_SIDECAR_PYTHON`. The GR00T policy sidecar requires the pinned
Isaac-GR00T environment and organizer checkpoint documented in
`rskills/gr00t-n17-b1k-turning-on-radio/README.md`.
