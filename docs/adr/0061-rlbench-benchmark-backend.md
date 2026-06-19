# ADR-0061: RLBench (CoppeliaSim/PyRep) benchmark backend + 3D Diffuser Actor

- Status: **Accepted**
- Date: 2026-06-19
- Related: [ADR-0002](0002-eval-and-sim-environments.md) (eval & sim environments);
  [ADR-0045](0045-isaac-sim-backend-integration.md) (Isaac Sim sidecar — the direct
  precedent for a heavy, externally-provisioned, out-of-process scene backend);
  [ADR-0046](0046-nvidia-gr00t-backend.md) (out-of-process policy sidecar via ZMQ);
  [ADR-0060](0060-benchmark-task-data-compatibility-gate.md) (`evaluated_tasks` gate);
  [ADR-0012](0012-open-core-licensing.md) (open-core / weight-license posture).
- Implements: issue #53 (follow-up to PR #48, branch `feat/website-benchmark-videos`).

## Context

RLBench (James et al., 2020, [1909.12271](https://arxiv.org/abs/1909.12271)) is the
standard benchmark for **3D / keyframe** manipulation policies: 100 Franka tasks on
**CoppeliaSim** via **PyRep**. The strong released policies (PerAct, RVT/RVT-2,
3D Diffuser Actor, Act3D) are evaluated on the canonical **PerAct 18-task** multi-task
subset (Shridhar et al., 2209.05451).

OpenRAL drives every simulator through one minimal `SimRollout` seam registered in
`openral_sim.registry.SCENES`. RLBench does not fit in-process for three reasons:

1. **CoppeliaSim is the heaviest sim dependency in the tree.** It is a proprietary
   simulator (free EDU license), not redistributable (CLAUDE.md §1.9). PyRep builds a
   Cython extension that links a *specific* CoppeliaSim 4.1.0 install via
   `COPPELIASIM_ROOT`.
2. **The released 3D policies pin the `MohitShridhar/RLBench@peract` fork**, not upstream
   `stepjam/RLBench` — different camera placements, the 18 PerAct task models, a corrected
   `close_jar` success condition, and a 9-D action layout
   (`[pose(7), gripper(1), ignore_collisions(1)]`, vs upstream's 8-D).
3. **The 3D policies pin an older torch/CUDA stack** incompatible with the openral py3.12
   workspace (`numpy>=2` / `torch>=2.10`). They must be bumped to an Ada-compatible build
   (cu121) in their own venv.

This is exactly the situation ADR-0045 (Isaac Sim) settled: a heavy, proprietary,
externally-provisioned simulator that cannot live in the py3.12 workspace.

## Decision

**1. Run RLBench out-of-process behind a ZMQ + msgpack sidecar, mirroring Isaac Sim.**

- `openral_sim.backends.rlbench` registers a `SimRollout` factory under scene id
  `rlbench`, `fixed_robot="franka_panda"` (RLBench tasks are baked onto the Panda; the
  CLI rejects `--robot`). `PhysicsBackend.COPPELIASIM = "coppeliasim"` is added.
- `tools/rlbench_sidecar.py` (standalone, runs in the externally-provisioned py3.10 venv)
  owns CoppeliaSim + the RLBench task + the keyframe **mover** (plan-and-retry until the
  end-effector reaches the target pose, the same closed-loop execution the upstream
  evaluators use). The openral side only needs the `rlbench` dependency group
  (`pyzmq` + `msgpack`) — `_deps._rlbench_client_plan`.
- The `step` action is an 8-D keyframe `[x y z qx qy qz qw gripper_open]`; the sidecar
  appends the peract-fork `ignore_collisions` channel and executes it.

**2. Package one 3D-policy rSkill: 3D Diffuser Actor (MIT).**

- `openral_sim.policies.rlbench_3dda` registers a `PolicyAdapter` under
  `model_family="diffuser_actor"`, proxying `tools/rlbench_3dda_sidecar.py` (same py3.10
  venv) which holds the `DiffuserActor` model + the 3-step observation history.
- **License: MIT** (code + the `diffuser_actor_peract.pth` checkpoint) — commercially
  permissive, so no install-time license guard (unlike RVT/RVT-2, which are NVIDIA
  non-commercial). This is a *weight-lineage* note for a third-party model; it does not
  touch OpenRAL's own Apache-2.0 code (CLAUDE.md §1.9).
- No flash-attn / dgl / pytorch3d needed for inference: flash-attn is only imported by the
  training-only `converter.py`; the single `dgl.geometry.farthest_point_sampler` call is
  replaced by a pure-torch shim in the sidecar venv.

**3. Ship a starter subset of 3 tasks, expand later.** `scenes/benchmark/rlbench_*.yaml`
(open_drawer, meat_off_grill, close_jar) + `benchmarks/rlbench.yaml` suite, at the official
PerAct protocol (25 evaluation episodes per task, max 25 macro-keyposes). The remaining 15
PerAct tasks are a follow-up. `evaluated_tasks` (ADR-0060 gate) lists the three task ids —
**not** `rlbench` — so the gate never claims coverage of the other 97 tasks.

## Live verification (this host)

Provisioned + verified live on an 8 GB RTX 4070 Laptop (Ada, sm_89), Ubuntu 24.04,
2026-06-19. CoppeliaSim renders on CPU (software GL) → ~14 MiB GPU; the policy peaks
**~0.43 GB** VRAM. The 3D Diffuser Actor PerAct checkpoint **solves tasks live**:

| Task | Result (`task.reset()` protocol) |
|---|---|
| open_drawer | 4/4 |
| meat_off_grill | 3/3 |
| close_jar | solved |

Provisioning recipe (externally provisioned — CoppeliaSim is never vendored):

```bash
# 1) CoppeliaSim 4.1.0 (Ubuntu20_04 build) → COPPELIASIM_ROOT
# 2) uv venv --python 3.10 ~/.cache/openral/rlbench-policy/.venv
# 3) torch 2.3.1+cu121 (Ada), diffusers, openai-CLIP, einops, scipy, open3d
# 4) PyRep (stepjam) + RLBench (MohitShridhar@peract, editable) + gymnasium==1.0.0a2
# 5) diffuser_actor_peract.pth + instructions.pkl (HF katefgroup/3d_diffuser_actor)
```

## Consequences

- **Positive.** OpenRAL gains the standard 3D/keyframe benchmark and its first 3D keyframe
  policy, on the same `SimRollout` + rSkill seams, with a clean MIT license. The
  sidecar pattern keeps the heavy/proprietary stack fully isolated from the py3.12 workspace.
- **Cost.** CoppeliaSim + the policy stack are a multi-GB, externally-provisioned, py3.10
  install with no auto-provision plan (a proprietary simulator). Tests `pytest.skip` when
  the venv is absent (CI runners). Two sidecars (scene + policy) share the host GPU.
- **Deferred.** The remaining 15 PerAct tasks; the other released policies (RVT-2 is
  non-commercial → would need a license guard; PerAct pins pytorch3d 0.3.0 / torch 1.7.1,
  far more fragile on Ada). On-the-fly CLIP instruction encoding (the sidecar currently uses
  the authors' precomputed `instructions.pkl`, which is what the proof used).
