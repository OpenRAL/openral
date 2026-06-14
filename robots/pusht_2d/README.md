# `pusht_2d` — Robot description

Canonical `RobotDescription` manifest for the **PushT 2-D pseudo-robot**:
a single free-floating 2-D end-effector that pushes a T-shaped block on
a `pymunk` rigid-body plane (the embodiment baked into
[`gym_pusht`](https://github.com/huggingface/gym-pusht)). There is no
real kinematic chain — the "robot" is the 2-D tip that the policy
commands. Sim-only.

## At a glance

| Field | Value |
| --- | --- |
| `name` | `pusht_2d` |
| `embodiment_kind` | `manipulator` |
| Joints | 2 synthetic prismatic "joints" representing the (x, y) tip position |
| Workspace | 512 × 512 px canvas (`gym_pusht/PushT-v0` default) |
| Sensors | 1× top-down RGB (96 × 96, key `observation.image`) |
| Supported control modes | 2-D end-effector position |
| `sdk_kind` | `closed` (`gym_pusht`, `pymunk`) |

> **Why prismatic joints for a 2-D tip?** There is no kinematic chain to
> model, but the schema requires at least one joint. The `tip_x` /
> `tip_y` prismatics let the manifest validate while documenting the
> action space. Position limits match the canvas (0 — 512 px); velocity /
> effort limits are nominal.

PushT predates the multi-cam `observation.images.cameraN` convention and
exposes the raw key `observation.image`; the rSkill manifest pins this
explicitly so capability matching gates correctly.

## Pair with

| Component | Path |
| --- | --- |
| Compatible rSkill | [`skills/diffusion-pusht/`](../../skills/diffusion-pusht/README.md) — Diffusion Policy |
| BenchmarkScene config | [`scenes/benchmark/pusht.yaml`](../../scenes/benchmark/pusht.yaml) |
| Eval adapter | `openral_sim.adapters.pusht` |
| Sim test | `tests/sim/test_pusht_2d_diffusion_pusht.py` |

## Tests

- Unit: covered via `tests/unit/test_eval_adapters_helpers.py` and the
  `RobotDescription` schema fuzz suite.
- Sim: `tests/sim/test_pusht_2d_diffusion_pusht.py` (`gym_pusht` + `pymunk`,
  routed through `run_evaluation`).

## Reproduction

```bash
just sim-diffusion-pusht
# which runs:
#     openral sim run --config scenes/benchmark/pusht.yaml --rskill rskills/diffusion-pusht --save-video
```

CPU-only; no GPU required.

## See also

- [`skills/diffusion-pusht/README.md`](../../skills/diffusion-pusht/README.md) — Diffusion Policy rSkill.
- [`docs/reference/vla_compatibility.md`](../../docs/reference/vla_compatibility.md) — VLA × Robot × Sim matrix.
