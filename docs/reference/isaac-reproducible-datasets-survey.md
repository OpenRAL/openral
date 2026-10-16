# Isaac-reproducible datasets survey — which public checkpoints can we re-run in our Isaac Sim 6.1 backend?

> **Status: research survey (2026-10-15).** Rank 1 is packaged (2026-10-16), not yet run:
> `robots/droid_franka_robotiq`, `scenes/{sim,deploy}/isaac_arena_droid_banana_in_bowl.yaml`,
> `rskills/pi05-droid-jointpos-arena`. Corrections found while packaging it: Isaac Lab 3
> camera quaternions are **xyzw** (the §5.1 sketch reads them as wxyz); Arena places the
> objects per reset while RoboLab fixes them (the shipped scene uses RoboLab's layout);
> DROID runs at 15 Hz (Arena `sim.dt` 1/120 s × decimation 8).
> Companions: [sim-environments](sim-environments.md) (Isaac `placement_success`),
> [`scenes/README.md`](../../scenes/README.md) §"Isaac Sim: any stage, any robot",
> [VLA compatibility matrix](vla_compatibility.md).
>
> **Conventions.** "card says" = a Hugging Face model/dataset card; "source shows" = the file was
> read at the pinned commit linked; anything not checked against a primary source is
> **unverified** and supports no ranking. Commit pins: IsaacLab-Arena
> [`738d20a`](https://github.com/isaac-sim/IsaacLab-Arena/tree/738d20a28c12cc6ec137581588e60f76160585ae),
> LeIsaac [`24d3bcd`](https://github.com/LightwheelAI/leisaac/tree/24d3bcd3f1e4585740fc79921782c41617237812),
> RoboLab [`ad45d4f`](https://github.com/NVlabs/RoboLab/tree/ad45d4f974725d020f82c2b0d77d78533aeba2b3),
> openpi [`215abfb`](https://github.com/Physical-Intelligence/openpi/tree/215abfb217dbac7d5f1273282331b9b1866c0479).

## 1. The question

We want config-only Isaac scenes — `robot_id` + `base_pose` + `scene.assets_uri` +
`backend_options.objects` + a `placement_success` block — run under `openral sim run` and
`openral deploy sim` with an **already-trained public checkpoint of a family OpenRAL already runs**
(SmolVLA, π0/π0.5, GR00T, ACT, XR-1, …; `POLICIES.register` ids in
`python/sim/src/openral_sim/policies/`). The best case is a dataset that was *generated in Isaac
Sim*, ships or links its USD, and has a checkpoint whose own authors scored it in that scene.

**Bottom line.** One candidate clears every bar today with no new criterion type: **Isaac
Lab-Arena's DROID / RoboLab tabletop pick-and-place with OpenPI `pi05_droid_jointpos`** — Arena
`main` targets Isaac Sim **6.1.0**, the USDs are public over HTTPS under the Isaac 6.1 asset root,
the success test is "object resting on/in a destination", and a lerobot PyTorch port of the
checkpoint exists. Its cost is a new robot manifest (Franka + Robotiq 2F-85) and a DROID I/O
adapter. LeIsaac (SO-101) is second: we already have the robot, but the scene is Isaac 4.5–5.1
and its success test needs two features we lack. Everything humanoid (Arena GR1/G1, GR00T
X-Embodiment) is far behind on action-space mismatch.

## 2. Summary table

| # | Candidate | Robot (OpenRAL manifest?) | Policy family → supported? | Weights licence | USD availability | Isaac target | Success form → maps to `placement_success`? | Effort |
|---|---|---|---|---|---|---|---|---|
| 1 | Arena DROID / RoboLab pnp | Franka + Robotiq 2F-85 (**no**: `franka_panda` has the Franka hand) | π0.5 `pi05_droid_jointpos` (+ π0, π0-FAST) → **yes** (`pi05`) | openpi repo Apache-2.0; OpenRAL treats π0.5 weights as permissive research ([CLAUDE.md §3](../../CLAUDE.md)) | **public HTTPS** (Isaac 6.1 S3 root) | **Isaac Sim 6.1.0** / Lab 3.0 EA | contact-supported + settled on destination → **yes**, approximately | M |
| 2 | LeIsaac PickOrange | SO-101 (**yes**, URDF) | SmolVLA (community) → yes; GR00T N1.5/N1.6 → yes but **noncommercial** | SmolVLA ckpt Apache-2.0 (card); GR00T N1.5 base NC | GitHub release zip (Apache-2.0 repo) | Isaac Sim 4.5 / 5.0 / 5.1 | 3 oranges near plate **and** arm at rest pose → **no** (needs 2 additions) | M–L |
| 3 | Arena GR1 open-microwave | GR1**T2** + hands (**no**: ours is GR1T1 no-hand, no `control_freq_hz`) | SmolVLA, π0.5 (Apache-2.0 cards), GR00T N1.6 (**NC**) | see row | public HTTPS (Arena assets) | 6.1 (main) | door openness fraction → **no** (articulation criterion) | L |
| 4 | Arena G1 static / loco pnp | Unitree G1 43-DoF (`g1` exists, 29-DoF) | GR00T N1.7 → yes | card says NVIDIA Open Model Agreement **and** "non-commercial use only" (contradictory) | public HTTPS | 6.1 | pnp → partly | L |
| 5 | Lightwheel LW-BenchHub | Double-Piper, X7S, G1 (**no**) | SmolVLA, π0.5, GR00T N1.5 | Apache-2.0 data cards | Lightwheel SDK download, login-based; asset terms **unverified** | Isaac Lab "5.0.0" badge | per-task Python | XL |
| 6 | GR00T X-Embodiment-Sim | GR1, bimanual Panda, Panda | GR00T | data CC-BY-4.0 | none (RoboCasa/robosuite tasks) | not Isaac | — | out of scope |
| 7 | MobileManiBench / GenManip / IROS-25 / GenieSim 3.0 | unnamed / AgiBot (**no**) | various | IROS-25 & GenieSim **CC-BY-NC-SA-4.0** | some USD on HF | Isaac (version unverified) | — | out of scope now |

## 3. What OpenRAL's Isaac backend can grade today

Source shows (`python/sim/src/openral_sim/backends/isaac_sim.py`, sidecar
`tools/isaac_manifest_scene.py`, documented in [sim-environments.md L117–160](sim-environments.md)):

- **One criterion type: placement of a rigid object into a world-frame AABB.**
  `IsaacPlacementRegion` (`isaac_sim.py:916`) is an axis-aligned box `min_xyz/max_xyz` in the
  world frame; `IsaacPlacement` (`:938`) names an object, a `target` region and `adjacent`
  regions; `IsaacPlacementSuccess` (`:982`) maps the **verbatim** `task.instruction` to its
  placements (`tasks`), with rest thresholds `settle_s` 1.0 s, `rest_speed_m_s` 0.02,
  `rest_turn_deg_s` 10, `upright_tol_deg` 20 (`:1021–1027`).
- **Graded rubric.** `IsaacPlacementGrade` (`:952`) = name, score, `region: target|adjacent`,
  `requires_upright`; default `complete` 1.0 / `partial` 0.5 / `semi` 0.25 (`:975–978`); the
  first grade must score 1.0 and only it may (`:1030–1050`). Episode grade = worst placement;
  success terminates the episode (`:995`).
- **Grading loop.** Sidecar `placement_grade` (`tools/isaac_manifest_scene.py:284`) tests the
  object's centre in the box and its up axis (`object_up_axis`, `:241`, from the object's
  roll/pitch); `worst_grade` (`:339`); `rest_after` (`:350`); per-tick `_grade_placements`
  (`:2689`); `set_task` (`:2673`) switches rubric per `ExecuteRskill` prompt under `deploy sim`.
- **Graded objects must be scene objects.** `IsaacSimOptions._placements_name_scene_objects`
  (`isaac_sim.py:1197–1212`) requires each placed object to be a **dynamic**
  `backend_options.objects` entry (`IsaacSceneObject`, `:795`: `usd`, `name`, `xyz`,
  `roll/pitch/yaw`, `dynamic`, `pose_noise`). Prims that live *inside* the environment USD
  cannot be graded.
- **Actions.** The scene accepts only `JOINT_POSITION`, `GRIPPER_POSITION`, `BODY_TWIST`
  (`isaac_sim.py:317–337`) — no Cartesian/IK action path.
- **Robots** come from the manifest's **URDF** (Isaac URDF importer; `scenes/README.md`
  L237–263). No USD-robot import path.

**Not gradable today** (each needs a new criterion type): articulation joint position/openness
(microwave, drawer, door); robot-state terms (return-to-rest pose); regions relative to a
*moving* object (plate/bowl that is itself dynamic); "lifted above height"; contact-force
support (Arena's test). Note also the auto-provisioned sidecar pins `isaacsim==5.1.0.0`
(`isaac_sim.py:117–122`); a 6.1 binary install works via `OPENRAL_ISAAC_SIDECAR_PYTHON`.

## 4. Candidates

### 4.1 Isaac Lab-Arena DROID / RoboLab tabletop pick-and-place — rank 1

- **What it is.** Arena is "an open-source framework for scalable benchmark authoring and robot
  policy evaluation", Apache-2.0, `main` = Isaac Lab 3.0.0 EA / **Isaac Sim 6.1.0**
  ([README L8–13, L221–232](https://github.com/isaac-sim/IsaacLab-Arena/blob/738d20a28c12cc6ec137581588e60f76160585ae/README.md)).
  It ports 120 NVlabs RoboLab tasks; the catalog lists Arena counterparts
  ([`robolab/task_catalog.md`](https://github.com/isaac-sim/IsaacLab-Arena/blob/738d20a28c12cc6ec137581588e60f76160585ae/isaaclab_arena_environments/robolab/task_catalog.md)),
  ~45 task YAMLs exist under `isaaclab_arena_environments/robolab/tasks/`, e.g.
  [`banana_in_bowl.yaml`](https://github.com/isaac-sim/IsaacLab-Arena/blob/738d20a28c12cc6ec137581588e60f76160585ae/isaaclab_arena_environments/robolab/tasks/banana_in_bowl.yaml)
  ("pick up the banana and place it in the bowl", 70 s episode).
- **Robot.** `droid_abs_joint_pos`: Franka + Robotiq 2F-85 from
  `Arena/assets/robot_library/droid/franka_robotiq_2f_85_flattened.usd` on a stand
  ([`droid.py` L61, L67](https://github.com/isaac-sim/IsaacLab-Arena/blob/738d20a28c12cc6ec137581588e60f76160585ae/isaaclab_arena/embodiments/droid/droid.py#L61)),
  spawned at the origin with joints (0, −π/5, 0, −4π/5, 0, 3π/5, 0) (L383–397). Cameras: two
  1280×720 externals parented to `panda_link0` and a wrist camera on the Robotiq `base_link`,
  OpenGL convention (L567–610). **OpenRAL has no Franka+Robotiq manifest**; `franka_panda`
  declares `panda_hand` (`robots/franka_panda/robot.yaml:141`).
- **Policy.** Arena's OpenPI adapter supports `pi05_droid_jointpos_polaris`
  (`gs://openpi-assets-simeval/pi05_droid_jointpos`, open-loop horizon 15) and π0 (horizon 10)
  ([openpi.rst L214–231](https://github.com/isaac-sim/IsaacLab-Arena/blob/738d20a28c12cc6ec137581588e60f76160585ae/docs/pages/quickstart/running_a_real_policy/openpi.rst)).
  openpi defines it as `Pi0Config(action_horizon=15, pi05=True)` with
  `DroidActionSpace.JOINT_POSITION`
  ([`polaris_config.py` L30–55](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/training/misc/polaris_config.py#L30)).
  A lerobot `PI05Policy` port exists: **[`DAVIAN-Robotics/pi05_droid_jointpos`](https://huggingface.co/DAVIAN-Robotics/pi05_droid_jointpos)**
  — card says `max_state_dim=8`, `chunk_size=15`, DROID quantile norm, state = 7 joints +
  gripper, two images (exterior, wrist), the model predicts **joint deltas** that must be added
  to the current joints, and "an external adapter is required". Other ports:
  [`Fisher-Wang/pi05-droid-jointpos-polaris-pytorch`](https://huggingface.co/Fisher-Wang/pi05-droid-jointpos-polaris-pytorch),
  [`RLinf/RLinf-Pi05-Polaris-droid_jointpos`](https://huggingface.co/RLinf/RLinf-Pi05-Polaris-droid_jointpos) (cards not read).
- **Licences.** Arena and RoboLab code Apache-2.0 (RoboLab
  [README L177–179](https://github.com/NVlabs/RoboLab/blob/ad45d4f974725d020f82c2b0d77d78533aeba2b3/README.md)).
  openpi repo Apache-2.0 (GitHub licence field); its README states no separate checkpoint licence
  (**unverified** beyond that). The DAVIAN port's card declares **no licence** — ask or convert
  ourselves with openpi's `convert_jax_model_to_pytorch.py`. Asset licences: NVIDIA-hosted Isaac
  assets are used in place, never vendored ([`scenes/README.md` L261–263](../../scenes/README.md));
  RoboLab objects come from hope, handal, hot3d, ycb, vomp, fruits_veggies
  ([objects README L3](https://github.com/NVlabs/RoboLab/blob/ad45d4f974725d020f82c2b0d77d78533aeba2b3/assets/objects/README.md));
  per-source licences **unverified**.
- **USD availability.** Arena assets resolve under `ISAACLAB_NUCLEUS_DIR`
  ([`nucleus.py` L8–16](https://github.com/isaac-sim/IsaacLab-Arena/blob/738d20a28c12cc6ec137581588e60f76160585ae/isaaclab_arena/assets/nucleus.py#L8));
  maple table = `Arena/assets/object_library/srl_robolab_assets/scenes/maple_table.usda`,
  banana/bowl = `.../objects/ycb/{banana,bowl}.usd`
  ([`background_library.py` L208–210](https://github.com/isaac-sim/IsaacLab-Arena/blob/738d20a28c12cc6ec137581588e60f76160585ae/isaaclab_arena/assets/background_library.py#L208),
  [`object_library.py` L1541–1550](https://github.com/isaac-sim/IsaacLab-Arena/blob/738d20a28c12cc6ec137581588e60f76160585ae/isaaclab_arena/assets/object_library.py#L1541)).
  Probed 2026-10-15: `https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.1/Isaac/IsaacLab/Arena/...`
  returns HTTP 200 for `maple_table.usda`, `banana.usd`, `bowl.usd` and the DROID robot USD — so
  `isaac:Isaac/IsaacLab/Arena/...` should resolve under the 6.1 asset root. `maple_table.usda`
  payloads `../fixtures/table_maple.usd` at (0.199, 0.022, 0.003) and carries its own
  `CollisionPlane` at z = −0.697, satisfying the "own floor collider" rule.
- **Success.** `PickAndPlaceTask`: object settled → lifted above its settled height → resting on
  the destination by filtered contact force ≥ 0.1 N within a support cone, speed < 0.003 m/s, for
  N consecutive steps
  ([`pick_and_place_task.py` L65–75, L160–190](https://github.com/isaac-sim/IsaacLab-Arena/blob/738d20a28c12cc6ec137581588e60f76160585ae/isaaclab_arena/tasks/pick_and_place_task.py#L160)).
  **Maps onto `placement_success`** as "banana centre in a box over the bowl, at rest"; we lose
  the contact-force and "was lifted" terms (a push into the bowl would pass ours).
- **Reported numbers.** RoboLab paper Table I: π0.5 **28.0 %** overall on RoboLab-120 (π0-FAST
  15.5 %, GR00T N1.6 7.2 %, π0 5.0 %) ([arXiv 2604.09860](https://arxiv.org/abs/2604.09860)); per-task
  numbers are in the RoboLab repo (not read). DAVIAN card: BananaInBowlTask **7/8** (JAX baseline
  6/8). Arena docs: π0.5 hit 1.0 on 8 of 9 three-episode runs (`sugar_box` 0/3) —
  [openpi.rst L121–181](https://github.com/isaac-sim/IsaacLab-Arena/blob/738d20a28c12cc6ec137581588e60f76160585ae/docs/pages/quickstart/running_a_real_policy/openpi.rst).
  Arena DROID camera-sensitivity sample: 7/10
  ([`nvidia/Arena-DROID-Camera-Sensitivity-Workflow-Sample`](https://huggingface.co/datasets/nvidia/Arena-DROID-Camera-Sensitivity-Workflow-Sample), CC-BY-4.0, results only).
- **Risks.** (a) Gripper: we import from URDF, Arena uses a flattened USD with a six-joint
  Robotiq; a Franka+Robotiq URDF must be sourced and its mimic joints driven like OpenArm's.
  (b) Action: absolute joint positions (fits `JOINT_POSITION`), but the port's deltas need
  state-anchoring (OpenRAL has `relative_rtc_prefix_fn` for lerobot relative-action checkpoints,
  `python/rskill/src/openral_rskill/_vla_core.py:1080`; the DAVIAN port does not ship that
  processor step). (c) Camera extrinsics must be transcribed (Arena quaternion convention vs our
  `quat_wxyz` + `axes: usd` to be checked). (d) Control rate: **unverified** — DROID data is
  commonly cited at 15 Hz; the manifest's `control_freq_hz` sets our physics substeps. (e)
  RoboLab warns contact dynamics differ even between Isaac 5.0 and 5.1
  ([README L169–171](https://github.com/NVlabs/RoboLab/blob/ad45d4f974725d020f82c2b0d77d78533aeba2b3/README.md));
  RoboLab paper numbers were measured on 5.x, Arena's on 6.x. (f) Object poses come from Arena's
  relation solver, so our fixed `xyz` replaces a randomized layout.

### 4.2 Isaac Lab-Arena humanoids (GR1 microwave, G1 pick-and-place) — rank 3

- **Data.** [`nvidia/Arena-GR1-Manipulation-Task`](https://huggingface.co/datasets/nvidia/Arena-GR1-Manipulation-Task):
  CC-BY-4.0, 50 Mimic-generated demos (10 human), 50 Hz, action "joint desired positions for all
  body joints (36 DoF)", state 54-DoF, one 512×512 POV camera, "ready for commercial use".
  [`nvidia/Arena-G1-Static-PickNPlace-Task`](https://huggingface.co/datasets/nvidia/Arena-G1-Static-PickNPlace-Task):
  CC-BY-4.0, 200 XR-teleop demos at 50 Hz, 43-DoF. Also `-v3`, `PlaceItemCloseDoor`,
  `G1-Loco-Manipulation` variants (cards not read).
- **Models.** [`nvidia/smolvla-arena-gr1-microwave`](https://huggingface.co/nvidia/smolvla-arena-gr1-microwave)
  and [`nvidia/pi05-arena-gr1-microwave`](https://huggingface.co/nvidia/pi05-arena-gr1-microwave):
  Apache-2.0 cards; SmolVLA `config.json` = 54-D state + 22 hand + EEF pos/quat + root inputs,
  **36-D action**, chunk 50. [`GN1x-Tuned-Arena-GR1-Manipulation`](https://huggingface.co/nvidia/GN1x-Tuned-Arena-GR1-Manipulation):
  GR00T N1.6, **OneWay Noncommercial**. [`GN1x-Tuned-Arena-G1-Static-PickNPlace`](https://huggingface.co/nvidia/GN1x-Tuned-Arena-G1-Static-PickNPlace):
  GR00T **N1.7**, card cites the NVIDIA Open Model Agreement yet says "ready for non-commercial use
  only" — resolve before relying on N1.7's commercial posture.
- **Env.** `gr1_open_microwave` asserts a **GR1T2** embodiment `gr1_pink | gr1_joint`, robot at
  (−0.4, 0, 0) ([env L24, L44–46](https://github.com/isaac-sim/IsaacLab-Arena/blob/738d20a28c12cc6ec137581588e60f76160585ae/isaaclab_arena_environments/gr1_open_microwave_environment.py#L44));
  the lerobot eval uses `--env.embodiment=gr1_pink` ([`isaaclab-arena-envs` README](https://huggingface.co/nvidia/isaaclab-arena-envs)),
  i.e. an IK embodiment (whether the 36-D action is joints or IK targets: **unverified**).
  Success = door moved from rest past an openness fraction
  ([`open_door_task.py` L23–60](https://github.com/isaac-sim/IsaacLab-Arena/blob/738d20a28c12cc6ec137581588e60f76160585ae/isaaclab_arena/tasks/open_door_task.py#L23)) —
  **needs an articulation criterion**.
- **OpenRAL fit.** `robots/gr1/robot.yaml` is GR1T1 *no-hand* URDF (`rd:gr1_description`), 3+7+7
  joints, hands only as 6-DoF EE abstractions, and no `action_spec.control_freq_hz` (the Isaac
  scene refuses a manifest without it). G1 needs whole-body control for loco-manip. Reported
  success rates: none on the cards read (**unverified**).

### 4.3 LeIsaac (LightwheelAI, SO-101) — rank 2

- **Repo** [LightwheelAI/leisaac](https://github.com/LightwheelAI/leisaac) Apache-2.0; supports
  Isaac Sim 4.5 / 5.0 / 5.1 with Isaac Lab 2.1.1 / 2.2.1 / 2.3.0
  ([installation.md L70–75](https://github.com/LightwheelAI/leisaac/blob/24d3bcd3f1e4585740fc79921782c41617237812/docs/docs/docs/getting_started/installation.md)).
  Tasks: PickOrange, LiftCube, CleanToyTable (single/bi-arm), FoldCloth (bi-arm), LeKiwi
  CleanupTrash ([available_env.md](https://github.com/LightwheelAI/leisaac/blob/24d3bcd3f1e4585740fc79921782c41617237812/docs/docs/resources/available_env.md)).
- **Data.** [`LightwheelAI/leisaac-pick-orange`](https://huggingface.co/datasets/LightwheelAI/leisaac-pick-orange):
  Apache-2.0, `so101_follower`, 60 episodes, 30 fps, 6-D joint action/state (`*.pos`), front +
  wrist 640×480. Mimic variant `leisaac-pick-orange-mimic-v0`.
- **Models.** [`LightwheelAI/leisaac-pick-orange-v0`](https://huggingface.co/LightwheelAI/leisaac-pick-orange-v0):
  `config.json` architecture `GR00T_N1_5` (no card → licence unstated; GR00T N1.5 base is
  noncommercial per CLAUDE.md §3). [`edge-inference/smolvla-so101-pick-orange`](https://huggingface.co/edge-inference/smolvla-so101-pick-orange):
  SmolVLA, Apache-2.0, trained on the dataset above, "6D joint positions", evaluated in
  `LeIsaac-SO101-PickOrange-v0` (success rate **unverified**; already listed in
  [vla_compatibility.md §3.5](vla_compatibility.md)). GR00T N1.6 community fine-tunes exist
  (`tshiamor/groot-n1.6-leisaac-pick-block`, `12e21/gr00t_n1d6_leisaac_pick_orange`; cards not read).
- **USD.** Kitchen scene + `so101_follower.usd` in the
  [v0.1.0 release](https://github.com/LightwheelAI/leisaac/releases/tag/v0.1.0)
  (`kitchen_with_orange.zip` 72.9 MB), mirrored in the Apache-2.0
  [`LightwheelAI/leisaac_env`](https://huggingface.co/LightwheelAI/leisaac_env) repo. The
  oranges and plate are prims **inside** `scene.usd`, split out at load by
  `parse_usd_and_create_subassets` ([`pick_orange_env_cfg.py` L93](https://github.com/LightwheelAI/leisaac/blob/24d3bcd3f1e4585740fc79921782c41617237812/source/leisaac/leisaac/tasks/pick_orange/pick_orange_env_cfg.py#L93)).
  Robot spawn `pos=(2.2, −0.61, 0.89)`, `rot=(0, 0, 0, 1)` (Isaac Lab 2.x wxyz → 180° yaw),
  arm PD stiffness 17.8 / damping 0.60
  ([`lerobot.py` L25–49](https://github.com/LightwheelAI/leisaac/blob/24d3bcd3f1e4585740fc79921782c41617237812/source/leisaac/leisaac/assets/robots/lerobot.py#L25)).
- **Success.** `task_done`: every orange within ±0.10 m x/y and ±0.07 m z of the **plate's
  current** root pose **and** the arm inside `SO101_FOLLOWER_REST_POSE_RANGE`
  ([`terminations.py` L10–56](https://github.com/LightwheelAI/leisaac/blob/24d3bcd3f1e4585740fc79921782c41617237812/source/leisaac/leisaac/tasks/pick_orange/mdp/terminations.py#L10)).
  Ours can approximate the plate box with a fixed region (plate jitter ±3 cm at reset, L108),
  but **cannot** grade embedded prims or the rest pose.
- **Risks.** Isaac 5.1 → 6.1 USD/PhysX drift; the robot is imported from our URDF, not their
  USD (gains, joint zero, gripper geometry differ; `joint_drive_gains` can copy 17.8/0.60);
  joint units (LeRobot degrees vs radians); camera offsets transcribed from
  [`single_arm_env_cfg.py` L45–75](https://github.com/LightwheelAI/leisaac/blob/24d3bcd3f1e4585740fc79921782c41617237812/source/leisaac/leisaac/tasks/template/single_arm_env_cfg.py#L45).

### 4.4 Lightwheel LW-BenchHub — not recommended now

[`LightwheelAI/lightwheel_tasks`](https://huggingface.co/datasets/LightwheelAI/lightwheel_tasks)
(Apache-2.0): LIBERO tasks on G1-Controller (62), Double-Piper (130), X7S (117); RoboCasa tasks on
G1-WBC (32), X7S (89). [`Lightwheel-Tasks-Double-Piper`](https://huggingface.co/datasets/LightwheelAI/Lightwheel-Tasks-Double-Piper):
16-D state, 12-D joint-position action. Models: `smolvla-double-piper-pnp`, `pi05_pnp_orange`,
`gr00t15_LiftCube` (no cards). The [LW-BenchHub README](https://github.com/LightwheelAI/LW-BenchHub)
pulls kitchens "via the Lightwheel SDK" whose docs list login/token auth and defer asset terms to
"the platform's license terms" ([docs.lightwheel.net/sdk](https://docs.lightwheel.net/sdk/)) —
asset licence **unverified**, likely not redistributable. None of its robots has an OpenRAL
manifest. Re-check if Lightwheel publishes the kitchens openly.

### 4.5 NVIDIA GR00T X-Embodiment-Sim / DreamGen — out of scope

[`nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim`](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim)
(CC-BY-4.0): bimanual Panda (6×1k), GR1 tabletop (24×10k + 24×1k), single Panda kitchen
(RoboCasa task names, 3k each). The card does not name a simulator; the task names are RoboCasa /
robocasa-gr1-tabletop-tasks, which OpenRAL already runs in **MuJoCo** (`robots/gr1/robot.yaml`
header). No Isaac USD → not an Isaac target.

### 4.6 Isaac Lab Mimic / Arena RL — no supported family

Arena's Franka lift checkpoint [`nvidia/Arena-Franka-Lift-Object-RL-Task`](https://huggingface.co/nvidia/Arena-Franka-Lift-Object-RL-Task)
is a 105,871-parameter RSL-RL PPO MLP emitting EE target poses, OneWay **Noncommercial** — not a
family we run and not a joint-space action. Isaac Lab Mimic's Franka stack demos are trained with
robomimic BC in Isaac Lab's own workflow; no public VLA checkpoint scored in that scene was found
(**unverified** beyond the HF searches above).

### 4.7 MobileManiBench, GenManip / IROS-25, GenieSim 3.0 — out of scope now

- **MobileManiBench** ([arXiv 2602.05233](https://arxiv.org/abs/2602.05233)): Isaac Sim, >100
  tasks, 5 skills, two unnamed mobile platforms; abstract claims code/data/models released but
  the abstract page gives no links — **unverified**.
- **InternRobotics IROS-2025-Challenge-Manip** ([card](https://huggingface.co/datasets/InternRobotics/IROS-2025-Challenge-Manip)):
  ships validation scenes + objects as USD, but **CC-BY-NC-SA-4.0**; robot and checkpoints
  **unverified**. `InternData-A1` is gated (access denied to our token).
- **AgiBot GenieSim 3.0** ([`GenieSimAssets`](https://huggingface.co/datasets/agibot-world/GenieSimAssets)):
  gated, "CC BY-NC-SA 4.0"; AgiBot robots, no OpenRAL manifest.

### 4.8 Contrast: MuJoCo-origin LIBERO / RoboCasa

We already run these natively in MuJoCo. No USD port with matching checkpoints was found except
LW-BenchHub's re-collected LIBERO/RoboCasa data on *different robots* (§4.4) and an
undocumented `NathanWu7/Isaaclab_Libero` dataset (no card). A MuJoCo-trained checkpoint dropped
into a USD re-build faces a renderer + contact-model gap; expect it to fail without re-training.

## 5. Ranked recommendation

1. **Arena DROID / RoboLab `banana_in_bowl` with π0.5 `pi05_droid_jointpos`.** Isaac 6.1-native,
   public assets, existing success criterion, a published in-scene score to reproduce (7/8;
   28.0 % suite-wide). Work: `robots/droid_franka_robotiq/` manifest + URDF; a `pi05-droid-jointpos`
   rSkill with the DROID adapter (8-D state, 2 cameras, delta→absolute); camera mounts; object
   poses. Then the other ~45 RoboLab task YAMLs are config-only.
2. **LeIsaac PickOrange with `edge-inference/smolvla-so101-pick-orange`.** Robot and family
   already supported; needs `objects[].prim_path` (grade a prim inside the stage) and a robot
   rest-pose criterion, or a softened "oranges on plate" task.
3. **Arena GR1 open-microwave with `nvidia/smolvla-arena-gr1-microwave` / `pi05-arena-gr1-microwave`.**
   Apache-2.0 weights, Isaac 6.1, but needs a GR1T2+hands manifest, an articulation criterion,
   and the gr1_pink action semantics resolved.

### 5.1 First scene YAML for #1 (proposed `scenes/sim/isaac_droid_banana_in_bowl.yaml`)

```yaml
robot_id: "droid_franka_robotiq"   # NEW manifest: Franka + Robotiq 2F-85 URDF,
                                   # action_spec.control_freq_hz (DROID rate, to verify)
base_pose:                         # Arena spawns DROID at the origin (droid.py L383)
  xyz: [0.0, 0.0, 0.0]
  quat_xyzw: [0.0, 0.0, 0.0, 1.0]
  frame_id: world

task:
  id: isaac_sim/droid_banana_in_bowl
  scene_id: isaac_sim
  instruction: "pick up the banana and place it in the bowl"   # robolab/tasks/banana_in_bowl.yaml
  max_steps: 1050                  # 70 s episode_length_s at 15 Hz (rate unverified)
  success_key: is_success

scene:
  id: isaac_sim
  backend: isaacsim
  assets_uri: "isaac:Isaac/IsaacLab/Arena/assets/object_library/srl_robolab_assets/scenes/maple_table.usda"
  cameras: ["exterior", "wrist"]   # manifest sensor names in the new manifest
  backend_options:
    boot_timeout_s: 1200
    camera_mounts:                 # from DroidCameraCfg (droid.py L567-610); convention TO CHECK
      exterior: {link: panda_link0, xyz: [0.05, 0.57, 0.66],
                 quat_wxyz: [-0.195, 0.399, 0.805, -0.393], axes: usd}
      wrist:    {link: robotiq_base_link, xyz: [0.011, -0.031, -0.074],   # link name per new URDF
                 quat_wxyz: [0.570, 0.576, -0.409, -0.420], axes: usd}
    objects:
      - name: banana
        usd: "isaac:Isaac/IsaacLab/Arena/assets/object_library/srl_robolab_assets/objects/ycb/banana.usd"
        xyz: [0.45, -0.15, 0.05]   # TBD: table-top height and reachable layout to be measured
      - name: bowl
        usd: "isaac:Isaac/IsaacLab/Arena/assets/object_library/srl_robolab_assets/objects/ycb/bowl.usd"
        xyz: [0.45, 0.15, 0.05]    # TBD
        dynamic: false             # static target keeps the fixed region valid (Arena's is dynamic)
    placement_success:             # existing backend support (isaac_sim.py:982)
      regions:
        in_bowl: {min_xyz: [0.37, 0.07, 0.0], max_xyz: [0.53, 0.23, 0.12]}   # TBD from bowl bbox
      grades:
        - {name: complete, score: 1.0, region: target}   # no upright requirement for a banana
      settle_s: 1.0
      tasks:
        "pick up the banana and place it in the bowl":
          - {object: banana, target: in_bowl}
```

Needs new support / work: the `droid_franka_robotiq` manifest and URDF; the rSkill's DROID adapter;
confirming the camera quaternion convention; the table height and object poses (read from the
loaded stage). Nothing in `placement_success` is new. Optional backend additions that would
tighten fidelity: a "lifted before placed" term and a region that follows a dynamic object.

## 6. Backend gaps this survey surfaced

| Gap | Needed by | Shape |
|---|---|---|
| Grade a prim that lives inside the environment USD | LeIsaac | `objects[].prim_path` instead of `usd` |
| Robot-state term (joint pose in range) | LeIsaac (rest pose) | new grade condition on manifest joints |
| Articulation openness | Arena GR1 microwave, RoboLab open/close tasks | joint fraction of range on a stage prim |
| Region relative to a dynamic object | LeIsaac plate, Arena bowl | region `frame: <object>` |
| Lifted-then-placed / contact support | Arena pnp fidelity | optional predicate sequence |
| USD robot import | Arena DROID / GR1T2 USD robots | alternative to URDF import (layer decision) |
