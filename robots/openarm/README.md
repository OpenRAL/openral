# `openarm_v2` — Robot description

Canonical `RobotDescription` manifest for the **Enactic OpenArm v2** —
an open-hardware bimanual humanoid arm platform (each side: 7 revolute
arm + 1 hinge-jaw gripper). Originally designed by Enactic with
LeRobot upstream integration; fully open-source CAD + firmware +
control software (`enactic/openarm` on GitHub, project page at
[openarm.dev](https://openarm.dev/)).

The same manifest covers the real OpenArm (`OpenArmRealHAL` over
`openarm_bringup`'s `ros2_control` stack and the Damiao CAN FD buses —
not lerobot's upstream driver) and the real-physics MuJoCo digital twin
(`OpenArmMujocoHAL` on the `enactic/openarm_mujoco` **v2** bimanual
MJCF — PR #19 on master).

## At a glance

| Field | Value |
| --- | --- |
| `name` | `openarm_v2` |
| `embodiment_kind` | `bimanual` |
| Joints | 16 actuated (2 × (7 revolute arm + 1 hinge gripper)) |
| End-effectors | 2 × parallel-jaw grippers (hinge-driven, one per side) |
| Per-side payload | ~2 kg |
| Per-side reach | ~0.7 m |
| Embodiment tags | `openarm`, `openarm_v2`, `enactic`, `bimanual` |
| Supported VLA embodiments | `openarm_v2`, `openarm` |
| Supported control modes | `joint_position` |
| `sdk_kind` | `open` (Enactic OpenArm + upstream MJCF) |
| `hal.sim` | `openral_hal.openarm:OpenArmMujocoHAL` (`deploy sim`). The tabletop arena (table + cubes + drawer + overview camera) is **not** in this manifest — it lives on the scene (`scenes/deploy/openarm_tabletop.yaml` `composition:`; `scenes/sim/openarm_tabletop.yaml` `backend_options.top_camera_*`), per the scene-composition / robot-manifest separation convention. The robot / scene / rSkill are separate. |
| `hal.real` | `openral_hal.openarm_real:OpenArmRealHAL` (`deploy run`) |

## What v2 fixes vs the v1 era

The upstream `enactic/openarm_mujoco` **v2** MJCF replaces v1's
draft-quality actuator setup with a production-ready one:

- **Native `<position>` actuators** with per-class PD gains baked
  into the MJCF (DM8009: kp=230 kv=2.7; DM4340: kp=190 kv=2.2;
  DM4310: kp=30 kv=1.5; fingers: kp=30 kv=0.2). The OpenRAL HAL
  drops the v1-era software PD loop entirely — `send_action` just
  writes target → ctrl and steps.
- **Proper `ctrlrange` and `forcerange`** on every actuator. No more
  `ctrlrange=[0, 0]` workaround.
- **Symmetric LEFT / RIGHT finger gains** (both kp=30, kv=0.2). The
  v1 era's asymmetric-gain bug (LEFT gain=1, RIGHT gain=100) doesn't
  exist in v2.
- **Single driven finger per side** (not two finger actuators). The
  follower finger tracks via an `<equality>` constraint inside the
  MJCF — kinematically symmetric, but only one actuator slot per
  gripper from the HAL's perspective. Total: 16 actuators.
- **Hinge grippers** instead of v1's prismatic ones — the upstream
  mechanism rotates jaws rather than translating fingers.

## v2 fetch path

`robot_descriptions` still pins `enactic/openarm_mujoco` to a pre-v2
commit, so `openral_hal._openarm_v2_assets.ensure_openarm_v2_mjcf`
maintains a parallel clone under `$OPENRAL_CACHE_DIR/openarm_v2/`
pinned to a known-good v2 SHA. The helper goes away once
`robot_descriptions` bumps its pin past PR #19 (then
`OpenArmMujocoHAL` switches to a regular
`from robot_descriptions import openarm_v2_mj_description`).

## Pair with

| Component | Path |
| --- | --- |
| Python HAL adapter | `openral_hal.openarm.OpenArmMujocoHAL` (MuJoCo digital twin) |
| Python description | `openral_hal.OPENARM_DESCRIPTION` |
| Sim test | `tests/sim/test_openarm_hal_mujoco.py` |
| v2 fetch helper | `openral_hal._openarm_v2_assets.ensure_openarm_v2_mjcf` |
| Real-HW HAL | `openral_hal.openarm_real.OpenArmRealHAL` |
| Real-HW bench scene | `scenes/deploy/openarm_bench.yaml` (`openral deploy run`) |
| Real-HW bringup | `ros2 launch openral_hal_openarm real_bringup.launch.py` |
| Full-graph HIL gate | `tests/hil/test_openarm_deploy.py` (`just hil-openarm-deploy`) |
| Upstream URDF | [enactic/openarm](https://github.com/enactic/openarm) |
| Upstream MJCF | [enactic/openarm_mujoco](https://github.com/enactic/openarm_mujoco) (v2 on master) |

## Real hardware

`hal.real` is `OpenArmRealHAL`, which publishes to the four `ros2_control`
controllers `openarm_bringup` spawns (per-side arm + gripper) and refuses to
`connect()` unless both udev-pinned SocketCAN links (`openarm_left`,
`openarm_right`) are up. It never starts `controller_manager` itself — that
graph is C++ at 400 Hz and belongs under a vendor bringup (CLAUDE.md §1.5).

Real deploys use `scenes/deploy/openarm_bench.yaml`, which binds the cell's
real cameras and — the part that is easy to get silently wrong — pins
`runtime.octomap_cloud_topic` to the topic the ZED SDK actually publishes.
`head_zed` in `robot.yaml` auto-enables the octomap leg, but the topic keeps a
sim-only launch default unless the scene sets it, and the result is an empty
octree behind a graph where every node reports healthy.

> **Bringup moves both arms.** `OpenArmHW::on_activate` calls `enable_all()`
> and then `return_to_zero()`: an unramped MIT position command to 0.0 on all
> seven joints per side, issued before the current pose is sampled, then a
> 200 x 10 ms ramp to zero. There is no non-moving real bringup for this robot.
> Clear the cell and keep a hand on the hardware E-stop. The deploy graph's
> three software E-stop sources (deadman watchdog, `hardware_estop` bridge,
> `human_estop` forwarder) do not cover bringup: the watchdog arms only once a
> skill goal is accepted, and the other two have no producer on this cell, so
> during `return_to_zero()` the hardware E-stop is the only independent stop.

The manifest's hand-authored capsules are what the C++ kernel checks every
chunk against, and `tests/unit/test_collision_geometry_zero_pose.py` now
asserts that no non-allowed pair interpenetrates at the zero configuration.
It exists because the first policy dispatch on this cell was refused for a
right link 3 / link 5 self-collision of exactly −0.04575 m — at the real zero
pose and at the twin's bent-elbow pose alike. A distance that ignores the
elbow angle is a modelling error, not a hazard: link 3's capsule was 0.22 m
long from a joint 0.154 m above the elbow, reaching 6.6 cm past joint 4 into
link 5's capsule. It now ends at the elbow, which link 4's capsule covers.

### Running the restock policy

The cell's policy is `OpenRAL/rskill-pi05-openarm-restock_shelf-bf16`, a
**private** OpenRAL Hub repo (lerobot-format π0.5, 8.3 GB BF16, three RGB
views in, 35-step chunks of 16-D actions out), so it is installed per host
rather than shipped in `rskills/`. The weights are π0.5 derivatives under PI's
permissive-research terms, hence `--non-commercial` here and
`OPENRAL_ALLOW_NONCOMMERCIAL=1` at load:

```bash
HF_TOKEN=<token with OpenRAL org access> \
    openral rskill install OpenRAL/rskill-pi05-openarm-restock_shelf-bf16 --non-commercial --yes
openral rskill check OpenRAL/rskill-pi05-openarm-restock_shelf-bf16 --robot robots/openarm/robot.yaml
```

The bench scene binds every stream the manifest requires: `top` is the ZED's
rectified left image as `observation.images.context`, the two Arducams are
`observation.images.wrist_left` / `wrist_right`. `openral rskill check`
against that merged sensor set reports compatible. The PaliGemma tokenizer
(`google/paligemma-3b-pt-224`) must also be in the default Hugging Face
cache for an offline load; `openral rskill install` does not fetch it.

**Preload, not dispatch-time load.** The bench scene pins
`runtime.preload_rskill_id` and `preload_prompt`, so the skill runner
resolves and loads the policy right after it activates, in a worker thread,
and rejects goals until `rskill_runner.preload_done` is logged. This is not
an optimisation: the deadman watchdog opens its 120 s first-chunk window
the moment a goal is accepted, and even the fast bf16 load (meta-device
build, weights streamed onto the GPU, one warm-up forward) takes ~33 s on
the Orin under the live graph (measured 2026-09-23) — before it, ~350 s idle
and ~1230 s live. A cold load inside a goal is E-stopped, correctly. The
preload prompt must equal the goal's prompt character for character: the
resident key is `(id, revision, prompt)` and a mismatch evicts the warm
skill and pays the cold load inside the watchdog window. `preload_done`
names the time.

**Not yet usable end to end (2026-09-23).** A dispatched goal executes its
first 35-step chunk on the real arms, then the deadman watchdog E-stops
the cell: one π0.5 forward takes 6–7 s inside the live graph (1.6 s in an
idle process), the runner proposes nothing while it infers the next chunk,
and a `safe_action` gap past 5 s is a stall by the watchdog's contract.

Once it is resident, the operator dispatches the single goal directly, with
the training instruction **verbatim** — a drifted prompt is an
out-of-distribution instruction to real arms:

```bash
ros2 action send_goal /openral/execute_rskill openral_msgs/action/ExecuteRskill \
    "{rskill_id: OpenRAL/rskill-pi05-openarm-restock_shelf-bf16, prompt: restock-shelf-from-front-box, deadline_s: 60.0}"
```

Arm joints come out of the checkpoint as per-step deltas and the grippers as
absolutes; the integration to absolute targets happens inside lerobot's π0.5
postprocessor (`use_relative_actions` with the gripper dims excluded by
`action_feature_names`), not in the runner. The runner's slot dispatch then
clamps every joint target strictly inside the manifest's joint limits before
proposing it — the C++ kernel validates open intervals, and the policy's very
first tick put one joint 0.019 rad past its limit — and the kernel remains the
authority on what actually reaches the arm.

**What has run, and where each layer was proven (2026-09-22).** Until this
branch no VLA had ever produced a chunk through the deploy graph on any
robot: the Hub resolver returned a packaging handle instead of a runtime
skill, and the goal aborted with an empty `failure_reason`. Getting the
first chunk to the kernel took seven stacked fixes, each masked by the one
before it (resolver binding, the rSkill cache split, the preload above, the
expandable-segments allocator on Tegra, the warm-up's device/dtype and
autocast, the slot-dispatch clamp, and for the MuJoCo twin its slot-group
reassembly plus the attachment heartbeat). On the digital twin the chain
now runs goal → preloaded policy → 35×16 chunk → slot dispatch → kernel,
where the kernel's self-collision check refuses the first chunk: the
tabletop twin starts with both elbows at −π/2 and the policy sees a MuJoCo
table instead of the real shelf, so its actions are out of distribution
there. The real cell is the only in-distribution proving ground; its first
dispatch aborted in the observation decoder on the ZED depth frame, which
is fixed (`decode_inline_frame`). On 2026-09-23 the real cell executed its
first chunk on the arms; the per-forward latency that then trips the
deadman is the open item above.

## Action layout (16 DoF)

| Slot | Joint | Unit | Range |
| ---: | --- | --- | --- |
| 0 | `left_joint1` | rad | -3.491, 1.396 |
| 1 | `left_joint2` | rad | -3.316, 0.175 |
| 2 | `left_joint3` | rad | ±1.571 |
| 3 | `left_joint4` | rad | 0.0, 2.443 |
| 4 | `left_joint5` | rad | ±1.571 |
| 5 | `left_joint6` | rad | ±0.785 |
| 6 | `left_joint7` | rad | ±1.571 |
| 7 | `left_gripper` | rad | 0.0, 0.7854 (closed → open) |
| 8 | `right_joint1` | rad | -1.396, 3.491 (mirrored) |
| 9 | `right_joint2` | rad | -0.175, 3.316 (mirrored) |
| 10 | `right_joint3` | rad | ±1.571 |
| 11 | `right_joint4` | rad | 0.0, 2.443 |
| 12 | `right_joint5` | rad | ±1.571 |
| 13 | `right_joint6` | rad | ±0.785 |
| 14 | `right_joint7` | rad | ±1.571 |
| 15 | `right_gripper` | rad | -0.7854, 0 (mirrored: closed → open) |

The asymmetric arm ranges (LEFT joint1/2 negative-leaning, RIGHT
mirrored positive-leaning) and the asymmetric gripper ranges (LEFT
positive jaw, RIGHT negative jaw) come from the v2 MJCF — the
physical mechanism mirrors across the centreline, and the actuator
ctrlranges reflect that.

## Tests

`tests/sim/test_openarm_hal_mujoco.py` exercises `OpenArmMujocoHAL`
end-to-end against real MuJoCo physics on the **v2** MJCF: full
lifecycle, upstream schema-drift guard (verifies all 16 actuators
remain `<position>` mode with symmetric L/R finger gains), exact
closed-loop convergence on every arm and gripper slot, multi-joint
simultaneous targets, the `<equality>` follower-finger tracking
invariant, and a per-slot identity sweep with alternating signs to
catch wiring slips.

HIL for the real arm lives in `tests/hil/`: `test_openarm_can_live.py`
(CAN transport gates plus a read-only motor round-trip — it never
energises) and `test_openarm_bringup_agreement.py` (the HAL's
controller/joint table against `openarm_bringup`'s own YAML, no hardware
needed) both pass on the wired cell. `test_openarm_slot_group_motion.py`
is the command→motion gate and is double-gated on
`OPENRAL_OPENARM_ALLOW_MOTION=1` + `OPENRAL_OPENARM_ATTENDED=1`; it has no
recorded run. Nothing in CI runs `tests/hil/`.

`test_openarm_deploy.py` is the full-graph gate and **passed green on the
wired cell for the first time on 2026-09-22** (qorin1, attended, hand on the
hardware E-stop): 7 passed in 28.5 s against a live `deploy run` of
`scenes/deploy/openarm_bench.yaml`. What that run proved, on real hardware —
all four `JointTrajectoryController`s plus `joint_state_broadcaster` active;
`/joint_states` carrying all 16 ros2_control joints at 684 Hz; the TF tree
complete to both end effectors; the C++ kernel ACTIVE at 16 DoF with
self-collision armed over 19 links and emitting no `safe_action` unbidden;
and a non-empty `/openral/world_voxels` fed by the real ZED cloud, which is
the silent `octomap_cloud_topic` failure the bench scene exists to pin. The
scene's `drivers:` block brought the ZED up as part of the graph, so there
was no second terminal. The gate's teardown E-stop latched the HAL
(`hal.estop`) as designed.

## Asymmetric joint conventions

OpenArm v2 mirrors arm joint ranges across L/R (e.g.
`left_joint1` ∈ [-3.49, 1.40] vs `right_joint1` ∈ [-1.40, 3.49]) and
gripper rotation direction (LEFT jaw closes via positive rotation,
RIGHT jaw via negative). Commanding the same numeric value to LEFT
and RIGHT homologous slots is therefore inherently a kinematically
mirrored pose — not a HAL bug. Tests validate each side
independently or use sign-aware sentinels.

## E-stop and recovery

`/openral/estop` reaches `OpenArmRealHAL.estop()` through the lifecycle node
(issue #295). The stop deactivates all four controllers —
`left_joint_trajectory_controller`, `left_gripper_controller`,
`right_joint_trajectory_controller`, `right_gripper_controller` — in one
STRICT `controller_manager/switch_controller`, read back via
`list_controllers`; a deactivated `JointTrajectoryController` holds
position, drops any trajectory it is sent, and writes nothing until
re-activated. Any half-staged ADR-0102 slot group is dropped first.

**Recovery is `RESETTABLE`.** The controllers hold and the CAN buses stay
configured, so `/openral/estop_cleared` (published by the reset authority
after the kernel's cooldown-gated `estop_reset` succeeds) makes the node call
`reset_estop()`, which re-activates the four controllers through the same
seam and reconnects the HAL **only** once the manager confirms every one
`active`. A refused re-activation keeps the latch.

Evidence: `tests/unit/test_openarm_real_hal.py::TestSafety`,
`tests/integration/test_real_hal_estop_ros2_control_live.py` (the RESETTABLE
branch), and — for the bus itself — `tests/hil/test_openarm_can_live.py`.

## See also

- [openarm.dev](https://openarm.dev/) — project landing page.
- [`python/hal/README.md`](../../python/hal/README.md) — `OpenArmMujocoHAL`, supported robots.
- [`packages/openral_hal_openarm/README.md`](../../packages/openral_hal_openarm/README.md) — real-hardware bringup (`real_bringup.launch.py`) and why no real-HW scene ships in-tree.
- [LeRobot OpenArm docs](https://huggingface.co/docs/lerobot/openarm) — upstream driver (not the path OpenRAL takes).
- [enactic/openarm_mujoco PR #19](https://github.com/enactic/openarm_mujoco/pull/19) — the v2 introduction.
- [`robots/anvil_openarm_v2/README.md`](../anvil_openarm_v2/README.md) — the Anvil OpenARM 2.0 (this same v2 design with Anvil's J1/J6 range deltas and the wrist support bracket).
- [`robots/aloha_bimanual/README.md`](../aloha_bimanual/README.md) — sibling bimanual twin (different gripper convention).
