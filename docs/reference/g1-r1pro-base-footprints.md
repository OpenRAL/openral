# Base footprints for the Unitree G1 and the Galaxea R1 Pro

*Researched 2026-10-15.* Planar outlines, in metres, in each manifest's
`base_frame` (x forward, y left), for the Nav2 fields `footprint_polygon` and
`footprint_radius` on [`robots/g1/robot.yaml`](https://github.com/OpenRAL/openral/blob/master/robots/g1/robot.yaml) and
[`robots/r1pro/robot.yaml`](https://github.com/OpenRAL/openral/blob/master/robots/r1pro/robot.yaml). Neither manifest
declares either field today.

Every number below was either quoted from a manufacturer page or computed from
the official model geometry: mesh vertices posed by forward kinematics and
projected onto the floor. The scripts and the pinned sources are listed under
[Reproducing](#reproducing).

**The recommended values:**

| Robot | `base_frame` | `footprint_polygon` (CCW) | `footprint_radius` |
|---|---|---|---|
| G1 | `pelvis` | `[[0.15, 0.15], [0.06, 0.27], [-0.05, 0.27], [-0.08, 0.18], [-0.08, -0.18], [-0.05, -0.27], [0.06, -0.27], [0.15, -0.15]]` | `0.28` |
| R1 Pro | `base_link` | `[[0.24, 0.31], [0.20, 0.34], [-0.11, 0.34], [-0.36, 0.15], [-0.39, 0.11], [-0.40, 0.03], [-0.40, -0.03], [-0.39, -0.11], [-0.36, -0.15], [-0.11, -0.34], [0.20, -0.34], [0.24, -0.31]]` | `0.41` |

Read the [caveats](#confidence-and-caveats) before you commit either value.
The G1 outline covers the arms at the `stand` keyframe only. The R1 Pro outline
is the bare chassis. That follows the repo's base-only Nav2 policy (ADR-0099),
but it does **not** cover the torso, which leans 8.6 cm past the front bumper
in BEHAVIOR's evaluation reset pose.

## How Nav2 uses these fields here

`RobotDescription.footprint_radius` feeds Nav2's `robot_radius`.
`footprint_polygon` is a list of base-frame `(x, y)` vertices, counter-clockwise
by convention, with at least 3 vertices
([`python/core/src/openral_core/schemas.py`](https://github.com/OpenRAL/openral/blob/master/python/core/src/openral_core/schemas.py),
fields near L2768 and L2797; the validator is `_validate_footprint_polygon`).

Two of the repo's decisions constrain what a "correct" outline is
([`repo-state-map.html`](../architecture/repo-state-map.html), Nav2 bringup
block; ADR-0099):

- **Nav2 is base-only.** The costmap footprint is the **bare chassis**. The 3-D
  safety kernel owns the arms and anything the robot carries.
- **The scan self-filter uses the same polygon.** `payload_scan_filter_node`
  deletes every lidar return whose endpoint lies inside `footprint_polygon`, so
  a loose over-approximation deletes **real** obstacles near the robot. The
  outline should therefore be tight, not padded. Put any safety margin in Nav2's
  inflation instead.

The existing precedent is `robots/panda_mobile/robot.yaml`: a 0.70 × 0.50 m
rectangle taken from the robosuite collision box, with `footprint_radius: 0.35`.
Strictly, 0.35 is the inscribed half-length, not the circumscribed radius,
which is 0.43 m. This page reports the circumscribed radius, as the task asked.

---

## 1. Unitree G1 (`base_frame: pelvis`)

### Manufacturer specification

From [unitree.com/g1](https://www.unitree.com/g1), G1 column, fetched
2026-10-15:

| Spec | Value (verbatim) |
|---|---|
| Standing dimensions (H × W × thickness) | "1320x450x200mm" |
| Folded dimensions | "690x450x300mm" |
| Calf + thigh length | "0.6M" |
| Arm span | "About 0.45M" |
| Weight with battery | "About 35kg" |

The page does not say which arm pose the 450 mm width refers to.

### Geometry used

- **Model:** `unitree_g1/g1.xml` from
  [google-deepmind/mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie)
  at commit `feadf76d42f8a2162426f7d226a3b539556b3bf5` (2026-03-18). This is the
  local `~/.cache/robot_descriptions` checkout that `rd:g1_mj_description`
  resolves to, and it is the MJCF named in the manifest's `sim:` comment.
- **Pose:** the MJCF's own `stand` keyframe. The pelvis sits at z = 0.79 m, the
  legs are at zero, and each arm is set to `shoulder_pitch 0.2`,
  `shoulder_roll ±0.2`, `shoulder_yaw 0`, `elbow 1.28`, with the wrists at 0.
- **Method:** every visual mesh vertex (geom group 2) is posed by `mj_forward`
  and expressed in the `pelvis` body frame. The collision meshes (group 3) give
  the same y extent, ±0.258 m, and a shorter x extent, because the feet collide
  through four spheres rather than the sole mesh.

The repo has no G1 home pose of its own. The walking path resets to a
`knees_bent` keyframe from a separate walking MJCF
([`python/hal/src/openral_hal/g1.py`](https://github.com/OpenRAL/openral/blob/master/python/hal/src/openral_hal/g1.py)),
which this page did not measure.

### Results in the pelvis frame, `stand` keyframe

| Quantity | Value | Derivation |
|---|---|---|
| Foot length (sole mesh) | 0.208 m, x ∈ [−0.066, 0.142] | `*_ankle_roll_link` mesh |
| Foot contact span | x ∈ [−0.055, 0.125] | four `foot` spheres, centres at x −0.05 and 0.12, r = 0.005 |
| Foot width (mesh) | 0.075 m, \|y\| ∈ [0.081, 0.156] | `*_ankle_roll_link` mesh |
| Foot centre-line | y = ±0.1185 | midpoint of the contact-sphere rows (y 0.0885 to 0.1485) |
| Stance width, outer edge to outer edge | 0.313 m (y ±0.156) | sole mesh |
| Body without arms (legs + pelvis + torso + head) | x ∈ [−0.073, 0.142], y ±0.165, r_max 0.195 | |
| **Whole body with arms** | **x ∈ [−0.073, 0.142], y ±0.260, r_max 0.260** | widest points are the wrist and rubber-hand meshes (`*_wrist_yaw_link`, z −0.25 to −0.05) |
| Outline centre relative to the pelvis origin | (+0.035, 0) | the bounding box is front-heavy because of the toes |

In the stand keyframe the arms do not reach forward of the toes. The front
limit, x = 0.142, is the toe tip. The arms widen the outline from ±0.165 to
±0.260.

**For comparison, the all-zero pose:** the arms hang with the elbows at 0,
which on the G1 puts the forearms horizontal and pointing forward. The visual
outline is then x ∈ [−0.073, 0.373], y ±0.182, r_max 0.393. A pose that raises
the arms forward therefore more than doubles the forward extent.

### Spec sheet versus geometry

| | Spec | Geometry (`stand`) | Note |
|---|---|---|---|
| Thickness (x) | 200 mm | 215 mm | agrees within the toe overhang |
| Width (y) | 450 mm | 520 mm with arms; 330 mm without | the spec is 70 mm narrower than the arms-out keyframe. All-zero arms give 363 mm. The pose behind the spec's 450 mm is **unresolved**. |
| Height | 1320 mm | not measured | not needed for a 2-D footprint |

### Recommendation for the G1

- **`footprint_polygon`:** an octagon that contains every visual vertex of the
  whole standing body at `stand`. It was verified with 0 vertices outside.
  ```yaml
  footprint_polygon: [[0.15, 0.15], [0.06, 0.27], [-0.05, 0.27], [-0.08, 0.18],
                      [-0.08, -0.18], [-0.05, -0.27], [0.06, -0.27], [0.15, -0.15]]
  ```
- **`footprint_radius`:** 0.28, the circumscribed radius of that octagon
  (0.277), rounded up. The body itself reaches 0.260.
- **Arms-excluded alternative** (closer to "base only"):
  `[[0.15, 0.17], [-0.08, 0.17], [-0.08, -0.17], [0.15, -0.17]]` with
  `footprint_radius: 0.23`. It contains every non-arm vertex (0 outside).

---

## 2. Galaxea R1 Pro (`base_frame: base_link`)

### Manufacturer specification

From Galaxea's documentation centre, "About R1 Pro 2026"
([docs.galaxea-dynamics.com/R1Pro/en/docs/2026/](https://docs.galaxea-dynamics.com/R1Pro/en/docs/2026/)),
fetched 2026-10-15. Values are verbatim:

| Spec | Value |
|---|---|
| Mechanical Parameters → Width | "Chest width: 348mm, Shoulder width: 615mm, Chassis width: 675mm" |
| Chassis Details → Dimensions (L x W x H) | "675mm x 636mm x 346mm" |
| Chassis DOF | "6DOF, 360° omnidirectional movement" |
| Full-body height | "1700mm" |
| Operating range | "Horizontal radius: 716mm (861mm with gripper)" |

The 2025 page ([.../docs/2025/](https://docs.galaxea-dynamics.com/R1Pro/en/docs/2025/))
gives "Chassis: 675 mm" under Width and describes the chassis as
"Ackermann/translation/rotation".

**The page contradicts itself.** One table calls 675 mm the chassis *width*;
the other lists it as the *length* (675 L × 636 W). The geometry below sides
with "width 675".

### Geometry used

- **Model:** `models/r1pro/urdf/r1pro.urdf` from the Hugging Face dataset
  [`behavior-1k/omnigibson-robot-assets`](https://huggingface.co/datasets/behavior-1k/omnigibson-robot-assets),
  MIT licence, at revision `c325bd3b6d1560d3145f3c9c0e885e79e47f180c`. This is
  the URDF the manifest cites as its joint-limit source. It is BEHAVIOR-1K's
  asset, not a URDF published by Galaxea; no Galaxea-published URDF was located.
  - Only visual meshes were measured. The URDF's collision meshes
    (`meshes/collision/*-col-0.obj`) are not in the dataset.
  - The URDF does record their scale factors. `base_link-col-0` is scaled by
    `0.6278 0.6752 0.3649`, consistent with the visual extent below.
- **Poses:**
  - **`zero`:** every joint at 0. This is OmniGibson's "tucked" arm pose with
    the torso straight.
  - **`eval_reset`:** the BEHAVIOR-1K evaluator's `reset_joint_pos`. The torso
    is at `[1.025, -1.45, -0.47, 0.0]` and the arms are at 0, from
    [`OmniGibson/omnigibson/eval/r1pro.yaml`](https://github.com/StanfordVL/BEHAVIOR-1K/blob/7d6718cb8ac5f249bb3c610c23713627d97da1ef/OmniGibson/omnigibson/eval/r1pro.yaml).
    That file explains the values as "JoyLo's default trunk pose".
  - **`untucked`:** the `untucked_default_joint_pos` in the dataset's
    `r1pro.yaml`, which raises the arms out to the sides. This is included only
    to bound the worst case.
- **Method:** every visual mesh vertex is posed with `yourdfpy` forward
  kinematics, in the `base_link` frame.

### Results in the `base_link` frame

| Quantity | Value |
|---|---|
| `base_link` chassis mesh | x ∈ [−0.392, 0.235] (0.627 m), y ±0.337 (0.674 m), z ∈ [0.042, 0.405] |
| Chassis including wheels and steer modules | x ∈ [−0.397, 0.239] (**0.636 m**), y ±0.337 (**0.674 m**); the wheels lie within y ±0.320 |
| Wheel layout | 3 steered drive modules: `steer_motor_joint1` at (0.169, +0.28), `steer_motor_joint2` at (0.169, −0.28), `steer_motor_joint3` at (−0.327, 0) |
| Shape | Neither rectangular nor round. The front is flat at x = 0.239, the sides are flat at y ±0.337 from x ≈ −0.09 to 0.17, and the rear tapers to a rounded point at x = −0.397 over the single rear wheel. |
| `base_link` origin | at floor level (wheel bottoms at z = 0). The origin is **0.079 m forward** of the chassis bounding-box centre (x_c = −0.079, y_c = 0). `torso_joint1` sits at x = −0.079, the box centre. |
| Circumscribed radius about `base_link` | 0.399 (chassis), driven by the rear point |

**Spec sheet versus geometry.** The mesh measures 0.636 m along x including the
wheels, and 0.674 m along y. That matches "675 × 636" with **x as the 636 side**.
So the chassis is wider than it is long, the "Chassis width: 675mm" row is
right, and the "L x W" row has its labels swapped relative to `base_link`.

### Do the arms overhang the chassis?

| Pose | Whole-body outline | Overhang beyond the chassis |
|---|---|---|
| `zero` (tucked) | x ∈ [−0.397, 0.239], y ±0.337, r 0.399 | **None.** The arms and grippers stay within y ±0.326 and x ≤ 0.04. |
| `eval_reset` | x ∈ [−0.397, **0.325**], y ±0.337, r 0.399 | **Torso, +8.6 cm forward** over \|y\| ≤ 0.084 m, at z ≈ 0.41–0.89 m: `torso_link1` and `torso_link2` lean forward. The arms and grippers stay within x ≤ 0.185 and \|y\| ≤ 0.326. |
| `untucked` | x ∈ [−0.397, 0.418], y [−0.637, 0.616], r 0.717 | The arms are out sideways at z ≈ 1.4 m, about 0.3 m past the chassis on each side. |

### Recommendation for the R1 Pro

- **`footprint_polygon`, the bare chassis:** a 12-gon that contains every
  chassis, wheel and steer vertex (0 outside) and every vertex of the `zero`
  pose.
  ```yaml
  footprint_polygon: [[0.24, 0.31], [0.20, 0.34], [-0.11, 0.34], [-0.36, 0.15],
                      [-0.39, 0.11], [-0.40, 0.03], [-0.40, -0.03], [-0.39, -0.11],
                      [-0.36, -0.15], [-0.11, -0.34], [0.20, -0.34], [0.24, -0.31]]
  ```
  This follows the base-only policy.
- **`footprint_radius`:** 0.41, the polygon's circumscribed radius (0.405),
  rounded up. The chassis itself reaches 0.399. The same circle also contains
  the `eval_reset` torso lean, whose radius is 0.375.
- **Alternative that includes the evaluation torso lean:** add `[0.33, 0.09]`
  and `[0.33, -0.09]` at the front. The result contains every `eval_reset`
  vertex (0 outside), with the same 0.41 radius:
  ```yaml
  footprint_polygon: [[0.33, 0.09], [0.24, 0.31], [0.20, 0.34], [-0.11, 0.34],
                      [-0.36, 0.15], [-0.39, 0.11], [-0.40, 0.03], [-0.40, -0.03],
                      [-0.39, -0.11], [-0.36, -0.15], [-0.11, -0.34], [0.20, -0.34],
                      [0.24, -0.31], [0.33, -0.09]]
  ```
- **Simplest rectangle:** `[[0.24, 0.34], [-0.40, 0.34], [-0.40, -0.34], [0.24, -0.34]]`
  also contains the whole chassis. It over-covers the tapered rear corners by
  up to about 0.2 m, which the scan self-filter would treat as robot. Its
  circumscribed radius is 0.53.

---

## Confidence and caveats

- **G1 confidence: high for the stand keyframe.** The values are computed from
  the pinned menagerie MJCF the HAL loads. They hold **only** for that arm pose.
  Any pose that raises the arms forward breaks them: zero elbows reach
  x = 0.373. Under ADR-0099 the arms are the kernel's job, so the
  arms-excluded rectangle is the base-only-consistent choice. The whole-body
  octagon is what the task asked for.
- **The G1 is a walking base.** The pelvis frame sways and the stance changes
  while it walks. These outlines describe a static stand. No upstream source
  gives a dynamic footprint; that is **unresolved**.
- **G1 width:** the spec's 450 mm matches no measured pose (330, 363 or 520 mm).
  It is reported, not reconciled.
- **R1 Pro confidence: medium-high for the chassis.** The geometry is
  self-consistent and reproduces both of Galaxea's numbers, 636 and 675 mm.
  Three limits remain:
  - The URDF is BEHAVIOR-1K's sim asset, not a Galaxea-published description.
  - The collision meshes were unavailable.
  - Physical bumpers or covers that differ from the sim mesh are **unverified**.
- **R1 Pro arm pose:** the "home" pose depends on the stack. BEHAVIOR's
  evaluator resets the torso into a forward lean; OmniGibson's tucked default is
  upright. Which pose OpenRAL's deploy uses was not checked: the manifest has
  `hal.sim: null`, and the sidecar owns the reset. If the deploy holds
  `eval_reset` near tables at 0.4–0.9 m, prefer the torso-lean polygon or rely
  on the kernel.
- **Galaxea spec:** the page labels 675 mm as both width and length. The
  geometry resolves it as width, perpendicular to `base_link` x.
- **Not consulted:** the unitreerobotics `unitree_ros` G1 URDF (the menagerie
  MJCF derives from it) and any Galaxea GitHub description repository.

## Reproducing

All of this is in the session scratchpad; nothing is committed.

- **Scripts:** `g1.py` and `g1b.py` (MuJoCo), and `r1pro.py` and `r1pro2.py`
  (`yourdfpy` plus `trimesh`, with polygon containment through
  `matplotlib.path`). They run under `/home/allopart/workspace/openral/.venv/bin/python -I`.
- **R1 Pro assets:** downloaded at the pinned Hugging Face revision above, from
  the `models/r1pro/` tree (URDF, OBJ/STL meshes, curobo YAMLs).

## Sources

- Unitree G1 product page, specification table: <https://www.unitree.com/g1> (fetched 2026-10-15).
- MuJoCo Menagerie `unitree_g1/g1.xml`, commit `feadf76d42f8a2162426f7d226a3b539556b3bf5`: <https://github.com/google-deepmind/mujoco_menagerie/tree/feadf76d42f8a2162426f7d226a3b539556b3bf5/unitree_g1>
- Galaxea, "About R1 Pro 2026": <https://docs.galaxea-dynamics.com/R1Pro/en/docs/2026/>, and the 2025 edition: <https://docs.galaxea-dynamics.com/R1Pro/en/docs/2025/> (fetched 2026-10-15).
- BEHAVIOR-1K robot assets, `models/r1pro/urdf/r1pro.urdf` and `models/r1pro/r1pro.yaml`, revision `c325bd3b6d1560d3145f3c9c0e885e79e47f180c`: <https://huggingface.co/datasets/behavior-1k/omnigibson-robot-assets/tree/c325bd3b6d1560d3145f3c9c0e885e79e47f180c/models/r1pro>
- BEHAVIOR-1K evaluator reset pose, `OmniGibson/omnigibson/eval/r1pro.yaml`, commit `7d6718cb8ac5f249bb3c610c23713627d97da1ef`: <https://github.com/StanfordVL/BEHAVIOR-1K/blob/7d6718cb8ac5f249bb3c610c23713627d97da1ef/OmniGibson/omnigibson/eval/r1pro.yaml>
- Repo: `python/core/src/openral_core/schemas.py` (footprint fields), `robots/panda_mobile/robot.yaml` (precedent), `docs/architecture/repo-state-map.html` (Nav2 base-only policy and the scan self-filter, ADR-0099).
