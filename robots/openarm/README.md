# `openarm_v2` — Robot description

Canonical `RobotDescription` manifest for the **Enactic OpenArm v2** —
an open-hardware bimanual humanoid arm platform (each side: 7 revolute
arm + 1 hinge-jaw gripper). Originally designed by Enactic with
LeRobot upstream integration; fully open-source CAD + firmware +
control software (`enactic/openarm` on GitHub, project page at
[openarm.dev](https://openarm.dev/)).

The same manifest covers the real OpenArm (via lerobot's upstream
driver, planned follow-up) and the real-physics MuJoCo digital twin
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
| `hal.sim` | `openral_hal.openarm:OpenArmMujocoHAL` (`deploy sim`). The tabletop arena (table + cubes + drawer + overview camera) is **not** in this manifest — it lives on the scene (`scenes/deploy/openarm_tabletop.yaml` `composition:`; `scenes/sim/openarm_tabletop.yaml` `backend_options.top_camera_*`), ADR-0066. The robot / scene / rSkill are separate. |
| `hal.real` | _null_ — sim-only until a lerobot OpenArm HAL lands (`deploy run` raises `ROSCapabilityMismatch`) |

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
| Future real-HW HAL | wrapper around [LeRobot's OpenArm driver](https://huggingface.co/docs/lerobot/openarm) |
| Upstream URDF | [enactic/openarm](https://github.com/enactic/openarm) |
| Upstream MJCF | [enactic/openarm_mujoco](https://github.com/enactic/openarm_mujoco) (v2 on master) |

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

HIL is planned alongside the real-HW HAL (wrapping lerobot's
upstream OpenArm driver).

## Asymmetric joint conventions

OpenArm v2 mirrors arm joint ranges across L/R (e.g.
`left_joint1` ∈ [-3.49, 1.40] vs `right_joint1` ∈ [-1.40, 3.49]) and
gripper rotation direction (LEFT jaw closes via positive rotation,
RIGHT jaw via negative). Commanding the same numeric value to LEFT
and RIGHT homologous slots is therefore inherently a kinematically
mirrored pose — not a HAL bug. Tests validate each side
independently or use sign-aware sentinels.

## See also

- [openarm.dev](https://openarm.dev/) — project landing page.
- [`python/hal/README.md`](../../python/hal/README.md) — `OpenArmMujocoHAL`, supported robots.
- [LeRobot OpenArm docs](https://huggingface.co/docs/lerobot/openarm) — upstream driver, future real-HW path.
- [enactic/openarm_mujoco PR #19](https://github.com/enactic/openarm_mujoco/pull/19) — the v2 introduction.
- [`robots/aloha_bimanual/README.md`](../aloha_bimanual/README.md) — sibling bimanual twin (different gripper convention).
