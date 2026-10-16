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
- **Meshes** are `rd:<robot_descriptions module>:<relpath>` refs into the pinned
  `panda_description` (example-robot-data) and `robotiq_2f85_v4_description` clones,
  fetched on first use — the form every vendored URDF here uses, so the collision
  lowering reads them. The Isaac backend imports a copy with them made absolute and
  `deploy_e2e` publishes them as `file://` URIs for Foxglove
  (`openral_core.assets.expand_rd_mesh_refs`).

Arena's Robotiq is a different 2F-85 model (Isaac's, 0 … 45° leader travel, PhysX
mimic gearing); the jaw geometry here is the ros-industrial one.

## Cameras

`exterior` (Arena `external_camera`, 104.0° hFOV) and `wrist` (`wrist_camera`, 87.66°),
both 1280 × 720 like Arena's `PinholeCameraCfg`; their mounts are scene
`camera_mounts`.

## Self-collision

The arm and the Robotiq base are the `openral collision lower` fit (`--tight-link` on
`panda_link0` … `panda_link7` and `robotiq_arg2f_base_link`): a box plus the exact hull
per link, like `franka_panda`. The `fixed_attachments` rows (`panda_joint8`,
`robotiq_mount_joint`) put the Robotiq base in the model.

**The Robotiq 2F-85 linkage** is in the model as moving links, posed from the jaw
fraction `f`. The eight links are, per side:

- `outer_knuckle`;
- `bar` (the outer finger);
- `proximal_phalanx` (the inner knuckle);
- `distal_phalanx` (the inner finger and its pad).

Each is a box plus the exact hull of its URDF meshes, in its own URDF frame, from the
same fitter (`urdf_lowering.lower_link_geometry`). The outer and inner knuckles and
the inner fingers hang on `mimic_joints` rows. The bars are `fixed_attachments` rows
on the outer knuckles. Origins and axes are the URDF joints'.

The rows compose the Isaac scenes' `gripper_joint_curve` `[[0, π/4], [1, 0]]` with
each URDF `<mimic>` multiplier:

| Links | URDF multiplier | Angle at fraction `f` |
| --- | --- | --- |
| outer knuckle, inner knuckle | +1 | `π/4 − π/4 · f` |
| inner finger | −1 | `π/4 · f − π/4` |

So `f = 1` (open) is 0 rad and `f = 0` (closed) is π/4, Arena's close target, not
the URDF's 0.8 rad limit. The kernel checks the linkage at the commanded fraction, as
for every link. The map is per backend:

| Backend | Kernel's linkage |
| --- | --- |
| Isaac (`tools/isaac_manifest_scene.py` drives the leader through the curve and each follower by its URDF multiplier) | exact: matches the URDF to < 1e-6 m over the stroke |
| anything else (another sim backend, a real Robotiq driver; no real HAL exists) | **unverified**: it must map `f` the same way |

`grasp_contact_links()` / `hand_links()` carry the six mimic links (both knuckles and
both inner fingers per side). The bars are fixed children of a mimic link, which
`grasp_contact_links()` does not follow, so the grasp and workspace-contact
exemptions do not cover them: they stay checked.

The URDF lowering emits only `joints` + `fixed_attachments` links, so the linkage
entries are spliced in by hand. The fleet drift guard therefore checks this robot's
ACM only. `tests/unit/test_droid_robotiq_finger_mimic.py` holds:

- every geometry entry to the fitter's output, byte for byte;
- the kernel's FK to the URDF over the stroke;
- the start pose free of trips.

The ACM is `openral collision lower --acm-only`, with `assets.srdf` from
`droid_franka_robotiq.srdf`:

- the adjacent pairs, which includes each linkage link with its parent;
- **two hand-reviewed rows**, `left_/right_proximal_phalanx ↔ left_/right_distal_phalanx`.

Those two links are pinned together on the real gripper: the closed four-bar pin
joint the URDF tree omits. Their collision meshes interpenetrate 3.95 / 5.89 /
5.97 mm at `f` = 0 / 0.5 / 1, and the kernel's hulls overlap 16.1 / 16.9 / 16.2 mm,
so the checked pair would trip at every opening.

At the scenes' start pose (0, −π/5, 0, −4π/5, 0, 3π/5, 0), no checked pair trips.
The nearest checked pair is:

| Fraction | Pair | Gap |
| --- | --- | --- |
| `f = 0` | the two inner knuckles | +2.1 mm |
| `f = 0.5` | Robotiq base ↔ inner finger | +2.5 mm |
| `f = 1` | outer knuckle ↔ inner knuckle | +4.2 mm |

**The linkage model and the two SRDF rows are user-approved 2026-10-16, needs
Safety-WG review + hazard-log row (CLAUDE.md §3).**

Safety limits are `franka_panda`'s, provisional.
