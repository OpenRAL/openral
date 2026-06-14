# `aloha_bimanual` — Robot description

Canonical `RobotDescription` manifest for the **Trossen ALOHA** bimanual
teleop platform — two 7-DoF arms with parallel grippers (14-DoF action
space), one top-down RGB camera. Two execution paths share the
manifest: `AlohaHAL` for the real Interbotix XS hardware (4
`ros2_control` controllers) and `AlohaMujocoHAL` for the real-physics
MuJoCo digital twin built on the
[`gym-aloha`](https://github.com/huggingface/gym-aloha) bimanual MJCF.

## At a glance

| Field | Value |
| --- | --- |
| `name` | `aloha_bimanual` |
| `embodiment_kind` | `bimanual` |
| Joints | 14 (2 × 6 revolute + 2 grippers) |
| End-effector | parallel grippers (one per arm) |
| Sensors | 1× top-down RGB camera (480 × 640 @ 30 Hz, key `observation.images.top`) |
| Embodiment tags | `aloha`, `lerobot` |
| Supported control modes | `joint_position` |
| `sdk_kind` | `closed` (Trossen / `gym_aloha` MuJoCo bindings) |

> **Note.** Joint kinematic detail is approximate — `gym-aloha` does not
> expose the URDF through its gym API. The values are sufficient for
> capability matching and `SafetyEnvelope` authoring; if a HAL adapter
> ever drives a real ALOHA the URDF wins.

## Pair with

| Component | Path |
| --- | --- |
| Python HAL adapters | `openral_hal.aloha.AlohaHAL` (real HW; Interbotix XS), `openral_hal.aloha.AlohaMujocoHAL` (MuJoCo digital twin; gym-aloha) |
| Compatible rSkill | [`skills/act-aloha/`](../../skills/act-aloha/README.md) — ACT cube transfer |
| BenchmarkScene config | [`scenes/benchmark/aloha_transfer_cube.yaml`](../../scenes/benchmark/aloha_transfer_cube.yaml) |
| Eval adapter | `openral_sim.adapters.aloha` |
| Sim test | `tests/sim/test_aloha_bimanual_act_aloha.py` (VLA rollout), `tests/sim/test_aloha_bimanual_hal_mujoco.py` (`AlohaMujocoHAL` end-to-end) |

## Tests

- Unit: covered indirectly via `tests/unit/test_eval_adapters_helpers.py`
  and the `RobotDescription` schema fuzz suite.
- Sim: `tests/sim/test_aloha_bimanual_act_aloha.py` (real `gym-aloha` MuJoCo with
  contact dynamics, routed through `run_evaluation`);
  `tests/sim/test_aloha_bimanual_hal_mujoco.py` exercises `AlohaMujocoHAL`
  end-to-end against the gym-aloha MJCF (lifecycle, schema-drift guard,
  closed-loop per-arm convergence, independent per-side gripper drive,
  identity check against the `AlohaHAL` 6/1/6/1 action split).

## Reproduction

```bash
just sim-act-aloha
# which runs:
#     openral sim run --config scenes/benchmark/aloha_transfer_cube.yaml --rskill rskills/act-aloha --save-video
```

## See also

- [`skills/act-aloha/README.md`](../../skills/act-aloha/README.md) — ACT rSkill.
- [`docs/reference/vla_compatibility.md`](../../docs/reference/vla_compatibility.md) — VLA × Robot × Sim matrix.
