# `droid_franka_robotiq` — Franka Panda + Robotiq 2F-85 (DROID platform arm)

The arm of the [DROID](https://droid-dataset.github.io/) platform as Isaac Lab-Arena
simulates it (`droid_abs_joint_pos`, `isaaclab_arena/embodiments/droid/droid.py` at
`738d20a2`): a Franka Panda with a Robotiq 2F-85 jaw. Simulation only (Isaac Sim 6.1);
there is no real HAL in this repo. Used by
`scenes/{sim,deploy}/isaac_arena_droid_banana_in_bowl.yaml` with
`rskills/pi05-droid-jointpos-arena`.

## Why not `franka_panda`

`robots/franka_panda` declares the Franka hand (`panda_hand`, two prismatic fingers).
DROID's jaw is a Robotiq 2F-85 on a different mount, and DROID's wrist camera rides
the Robotiq base, so kinematics, gripper and camera all differ.

## URDF (`droid_franka_robotiq.urdf`)

Composed once from two pinned `robot_descriptions` upstreams; the provenance is in
the file's header:

| Part | Source | License |
| --- | --- | --- |
| Arm `panda_link0` … `panda_link8` | [Gepetto/example-robot-data](https://github.com/Gepetto/example-robot-data) `d0d9098d`, `robots/panda_description/urdf/panda.urdf` (Franka hand and fingers removed) | BSD-3-Clause |
| Robotiq 2F-85 | [nickswalker/robotiq-2f-85](https://github.com/nickswalker/robotiq-2f-85) `319f7b94` (ros-industrial `robotiq`), `robotiq_2f_85_gripper_visualization/urdf/robotiq_arg2f_85.xacro`, xacro-expanded | BSD-2-Clause |

- **Mount** (`robotiq_mount_joint`): Arena's robot USD
  (`Arena/assets/robot_library/droid/franka_robotiq_2f_85_flattened.usd`, 14 157 865 B,
  sha256 `190810a1…0e98`) joins `panda_link8` to the Robotiq `base_link` with
  `localRot0` = Rz(−45°) and `localPos1` = (−0.018174, 0, 0), `localRot1` =
  (0.2706, 0.6533, 0.2706, 0.6533) wxyz. Composed, that is a pure +0.018174 m offset
  along `panda_link8` z once Arena's base frame (x = approach) is re-expressed as this
  URDF's `robotiq_arg2f_base_link` (z = approach, `(x, y, z)_urdf = (z, −y, x)_arena`).
  Both models put the outer-knuckle pivot 0.0547 / 0.0549 m from the base origin.
- **Gripper**: the leader `finger_joint` (0 = open … 0.8 rad = closed) with five
  `<mimic>` followers. Its `<limit>` effort 16.5 / velocity 5.0 are Arena's Robotiq
  drive (USD `maxForce`, `droid.py` gripper `velocity_limit`). The manifest's
  `finger_joint` carries OpenRAL's normalised `[0 = closed, 1 = open]` jaw; the
  Isaac scenes map it onto Arena's 0 … π/4 travel (`gripper_joint_curve`).
- **Meshes** resolve as `package://example-robot-data/` and
  `package://robotiq_2f_85_gripper_visualization/` through the pinned clones
  (`openral_hal.ros_package_overlay.PUBLIC_ROS_PACKAGES`), fetched on first use.

Arena's Robotiq is a different 2F-85 model (Isaac's, 0 … 45° leader travel, PhysX
mimic gearing); the jaw geometry here is the ros-industrial one.

## Cameras

`exterior` (Arena `external_camera`, 104.0° hFOV) and `wrist` (`wrist_camera`, 87.66°),
both 1280 × 720 like Arena's `PinholeCameraCfg`; their mounts are scene
`camera_mounts`.

## Not yet declared

No `collision_geometry` / SRDF: the safety kernel's self-collision model is not
generated yet (`openral collision lower --robot robots/droid_franka_robotiq/robot.yaml`
once the Robotiq meshes lower cleanly). Safety limits are `franka_panda`'s,
provisional.
