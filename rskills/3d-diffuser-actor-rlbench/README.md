<!--
  rSkill README — 3D Diffuser Actor (RLBench PerAct setup).
  Discovery + provenance card; mirrors rskill.yaml. ADR-0061.
-->

# rskill-3d-diffuser-actor-rlbench

3D Diffuser Actor — a diffusion policy over end-effector **keyposes** for RLBench,
running on the CoppeliaSim/PyRep RLBench benchmark backend (ADR-0061).

## What this skill does

Predicts the next end-effector keypose (position + orientation + gripper) from
multi-view RGB-D, conditioned on a language instruction. Used to benchmark
3D/keyframe manipulation on the RLBench **PerAct 18-task** suite. Ships the three
live-verified starter tasks: `open_drawer`, `meat_off_grill`, `close_jar`.

| Field | Value |
|---|---|
| Actions | open, close, pick, place (generalist keyframe policy) |
| Objects | drawer, grill/meat, jar — (PerAct task objects) |
| Scenes  | tabletop (RLBench / CoppeliaSim) |
| Embodiment | franka_panda |

## How it works

3D Diffuser Actor lifts the four RLBench camera RGB-D streams into a 3D point-cloud
scene token field, attends over it with a relative-position transformer, and runs a
DDPM diffusion head (100 denoising steps) to denoise an end-effector keypose
trajectory. Each predicted keypose is executed in RLBench by its sampling-based
motion planner (`EndEffectorPoseViaPlanning`), then the policy re-observes and
predicts the next keypose. The policy and the CoppeliaSim/PyRep scene run in an
out-of-process **py3.10 sidecar** (ZMQ + msgpack); the openral adapter
(`openral_sim.policies.rlbench_3dda`) forks it transparently.

### Observation → action contract

| dir | key | shape | notes |
|---|---|---|---|
| in | `observation.images.{left_shoulder,right_shoulder,wrist,front}` | `(H, W, 3) uint8` | RLBench PerAct cameras, 256×256 |
| in | `observation.point_clouds.{…}` | `(H, W, 3) float32` | per-camera world-frame point clouds |
| in | `observation.gripper_pose` | `(7,)` float32 | `[x y z qx qy qz qw]` |
| out | keyframe action | `(8,)` float32 | `[x y z qx qy qz qw gripper_open]` (world frame) |

## Upstream model / training

Weights are the authors' published RLBench PerAct multi-task checkpoint
(`diffuser_actor_peract.pth`); loaded verbatim, not retrained. Trained by the
authors on the PerAct 18-task RLBench demonstrations (multi-view RGB-D + keypose
supervision).

| Field | Value |
|---|---|
| Source repo | [`nickgkan/3d_diffuser_actor`](https://github.com/nickgkan/3d_diffuser_actor) |
| Weights | [`katefgroup/3d_diffuser_actor`](https://huggingface.co/katefgroup/3d_diffuser_actor) — `diffuser_actor_peract.pth` (168 MB) |
| Paper | [arxiv:2402.10885](https://arxiv.org/abs/2402.10885) — *3D Diffuser Actor: Policy Diffusion with 3D Scene Representations* |
| License | mit (code + checkpoints) — commercially permissive |
| Parameters | ~55 M |
| Training data | RLBench PerAct 18-task demonstrations |

## Supported robots

| Robot | Scene | Status | Notes |
|---|---|---|---|
| franka_panda | RLBench (CoppeliaSim) | ✓ validated | open_drawer 4/4, meat_off_grill 3/3, close_jar solved (8 GB Ada host, 2026-06-19) |

## Sensors required

| key | modality | resolution | dtype |
|---|---|---|---|
| `observation.images.left_shoulder` | RGB | 256 × 256 | `uint8` |
| `observation.images.right_shoulder` | RGB | 256 × 256 | `uint8` |
| `observation.images.wrist` | RGB | 256 × 256 | `uint8` |
| `observation.images.front` | RGB | 256 × 256 | `uint8` |

## Manifest summary

| Field | Value |
|---|---|
| `name` | `OpenRAL/rskill-3d-diffuser-actor-rlbench` |
| `version` | `0.1.0` |
| `license` | `mit` |
| `role` | `s1` |
| `model_family` | `diffuser_actor` |
| `embodiment_tags` | `franka_panda` |
| `runtime` | `pytorch` |
| `weights_uri` | `hf://katefgroup/3d_diffuser_actor` |
| `action_contract.dim` | `8` |
| `latency_budget.per_chunk_ms` | `3000.0` |

## Reproduction

```bash
# One-time: provision CoppeliaSim 4.1.0 + PyRep + RLBench@peract + the checkpoint
# in the py3.10 sidecar venv (see docs/adr/0061-rlbench-benchmark-backend.md).
openral benchmark scene \
  --config scenes/benchmark/rlbench_open_drawer.yaml \
  --rskill rskills/3d-diffuser-actor-rlbench
```

Inference VRAM peaks ~0.43 GB; runs comfortably on an 8 GB GPU. CoppeliaSim is
proprietary (free EDU license) and is **never** vendored — it is an
externally-provisioned dependency (CLAUDE.md §1.9 / ADR-0061).

## Evaluation

No full 25-episode `eval/rlbench.json` artifact is shipped in this starter PR.
The package is live-qualified on the three declared starter tasks and records the
proof in ADR-0061: `open_drawer` 4/4, `meat_off_grill` 3/3, and `close_jar`
solved on an 8 GB Ada host. Run the reproduction command above with
`--no-write-eval` for a non-mutating smoke run, or run the full suite to produce
`rskills/3d-diffuser-actor-rlbench/eval/rlbench.json`.

## License

OpenRAL wrapper files in this repository follow the project Apache-2.0 license.
The wrapped upstream 3D Diffuser Actor code and released
`diffuser_actor_peract.pth` checkpoint are MIT-licensed; the manifest therefore
uses `license: mit` for the consumer-visible weight/runtime posture.

## See also

- `scenes/benchmark/rlbench_open_drawer.yaml`
- `scenes/benchmark/rlbench_meat_off_grill.yaml`
- `scenes/benchmark/rlbench_close_jar.yaml`
- `benchmarks/rlbench.yaml`
- `docs/adr/0061-rlbench-benchmark-backend.md`
