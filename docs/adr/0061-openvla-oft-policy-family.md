# ADR-0061 — OpenVLA / OpenVLA-OFT policy family

- **Status:** Accepted 2026-06-19. First adapter + a verified WidowX rSkill land on
  `feat/maniskill3-openvla-oft` (draft PR onto PR #48); a Panda gap-closer rSkill is attempted
  on the same adapter.
- **Date:** 2026-06-19
- **ADR number:** `0061`. `0058` (standardized-description-assets), `0059`
  (foxglove-live-scene-visualization), `0060` (benchmark task-data gate) precede it; the integer
  is not load-bearing — cross-refs use filenames.
- **Related:**
  - **ADR-0060** — the benchmark task-data gate (`evaluated_tasks`). This ADR is the first to add
    a policy *family* specifically to satisfy that gate on a ManiSkill3-registered env.
  - **ADR-0046** — NVIDIA GR00T backend. Precedent for adding a `ModelFamily` member; contrast:
    GR00T runs **out-of-process** (ZMQ sidecar, Py3.10). OpenVLA loads **in-process** via
    transformers like MolmoAct2, so no sidecar is introduced.
  - **ADR-0014 / ADR-0010** — the SimplerEnv backend + WidowX (Bridge V2) suite the verified rSkill
    targets.

## Context

Issue #55: no rSkill genuinely solves a ManiSkill3 benchmark task, so the MS3 scenes
(`maniskill_pick_cube.yaml`, `maniskill3_panda.yaml`) and the wider MS3-registered eval surface
can't be validated end-to-end. The viable trained MS3 policies online are all **OpenVLA /
OpenVLA-OFT** (`RLinf/RLinf-OpenVLAOFT-PPO-ManiSkill3-25ood`, `Juelg/openvla-7b-finetuned-maniskill`,
`FengQiuxuan/maniskill_*_stack_cube`), a family OpenRAL had no adapter for: `ModelFamily` ∈
`smolvla, pi05, xvla, act, diffusion, rldx, molmoact2, gr00t`.

A research pass (2026-06-19) established the strongest candidate's real contract, which **overturns
the issue's stated PickCube/Panda example**:

- `RLinf/RLinf-OpenVLAOFT-PPO-ManiSkill3-25ood` (7.5 B, MIT) is an OpenVLA-OFT bridge policy
  RL-tuned (PPO) on `PutOnPlateInScene25Main-v3` with a **WidowX 250 S** — `unnorm_key=bridge_orig`,
  proprio disabled, single `3rd_view_camera` at 224×224, 256-bin discrete action tokens, chunk
  size 8 × 7-D (3 EE pos Δ + 3 rot Δ + 1 gripper), de-norm via `BOUNDS_Q99`. It is **not** a
  Panda/PickCube policy. Verified from its `config.json` (embedded `bridge_orig` q01/q99 + mask)
  and the RLinf eval config.
- Running it on `maniskill3/PickCube-v1` (Panda) is exactly the task-mismatched, plausible-but-
  unsolvable rollout **ADR-0060 blocks** — it would score ~0 and the gate would (correctly) refuse
  it. Its honest home is the **SimplerEnv WidowX bridge** tasks OpenRAL already wires.
- The public SimplerEnv `PutCarrotOnPlateInScene-v1` path verified non-zero success only when the
  adapter mirrors RLinf's PPO eval path: `generate_action_verl`, right-padded prompt length 30,
  temperature 0.6, torch seed 0 reapplied on each policy reset, first-six-dim action scale 2.0,
  and binary gripper threshold 0.5. Seeded local validation on the 8 GB RTX 4070 Laptop host
  scored **2/5 success (40%)** at the canonical 60-step horizon. The exact upstream
  `PutOnPlateInScene25Main-v3` task registers from
  RLinf source, but its `assets/carrot/more_carrot/...` files were not present in the public source
  checkout or the normal ManiSkill/SimplerEnv asset cache, so those 25-object numbers are not
  claimed.

OpenVLA / OpenVLA-OFT ship as transformers *custom-code* models (`trust_remote_code`,
`AutoModelForVision2Seq` → `OpenVLAForActionPrediction`). The 7.5 B backbone needs NF4 to fit an
8 GB host (OpenVLA paper Table 2: int4 ≈ 7.0 GB and matches bf16 accuracy) — the same NF4 +
CUDA `expandable_segments` recipe MolmoAct2 / π0.5 already use in-tree.

## Decision

Add an `"openvla"` `ModelFamily` (layer-3) + an in-process `openral_sim.policies.openvla` adapter.

1. **Schema:** append `"openvla"` to the `ModelFamily` Literal. Additive + backward-compatible — no
   `schema_version` bump (CLAUDE.md §1.6). Not added to `_MODERN_PROCESSOR_FAMILIES`: OpenVLA uses
   its own `PrismaticProcessor` (custom-code) and embeds norm stats in `config.json` (`unnorm_key`),
   so no `RSkillProcessors` block is required.
2. **Adapter** (`policies/openvla.py`, modelled on `molmoact2.py`): lazily imports
   torch/transformers; loads via `AutoModelForVision2Seq.from_pretrained(repo,
   trust_remote_code=True, …)` gated behind `OPENRAL_ALLOW_REMOTE_CODE=1` (custom-code, rSkill
   provenance §3); NF4 + `expandable_segments` for 8 GB; builds the prompt
   `In: What action should the robot take to {instruction.lower()}?\nOut: `, runs the discrete-token
   head, de-normalizes with the embedded `unnorm_key` stats (`0.5*(x+1)*(q99-q01)+q01`, masked-False
   gripper passthrough), and replays the action chunk closed-loop. The adapter also accepts
   manifest `policy_extras` for OpenVLA-family evaluation variants: `generate_action_verl`,
   prompt padding length, sampling temperature, torch sampling seed, action scale, and gripper
   binarization.
3. **Verified rSkill (WidowX):** `rskills/openvla-oft-simpler-widowx-nf4` wraps the RLinf checkpoint
   on the SimplerEnv WidowX carrot-on-plate task it locally solved; `evaluated_tasks` declares only
   `simpler_env/widowx_carrot_on_plate` (ADR-0060 gate passes). It verified **success > 0** live on
   the 8 GB host (2/5 seeded local episodes).
4. **Panda gap-closer (attempt):** a second rSkill wraps a documented OpenVLA Panda checkpoint
   (candidate `FengQiuxuan/maniskill_three_robots_stack_cube_*`) on a core `maniskill3/*` Panda env
   in `maniskill3_panda.yaml`. If it verifies success > 0, issue #55's literal acceptance closes; if
   not, it ships **documented-pending** — never a faked number (CLAUDE.md §1.2).

## Consequences

- OpenRAL can run the entire OpenVLA / OpenVLA-OFT family for sim eval, in-process, on an 8 GB host.
- The WidowX SimplerEnv carrot task gains a second, documented bridge policy alongside
  `rldx1-ft-simpler-widowx-nf4` — a genuine, verifiable MS3-registered-env win that exercises the
  new adapter end-to-end.
- The `maniskill3_panda` (Panda) gap closes only if the Panda checkpoint verifies; either way the
  honest state is recorded (verified number or documented-pending), and ADR-0060 keeps a
  task-mismatched pairing from ever scoring.
- `trust_remote_code` stays gated behind `OPENRAL_ALLOW_REMOTE_CODE=1`; these rSkills are never
  described as "signed/verified" (no sigstore yet, §3).

## Alternatives considered

- **Reproduce the exact RLinf env `PutOnPlateInScene25Main-v3`** to match the checkpoint's reported
  numbers. Deferred: the RLinf task registrations import with a normal source checkout, but the
  exact 25-object scene needs carrot asset files missing from that checkout and from the standard
  ManiSkill/SimplerEnv asset cache. The public SimplerEnv WidowX carrot task is the same
  embodiment/control/`bridge_orig` distribution and is already wired.
- **Out-of-process ZMQ sidecar** (like GR00T/rldx). Rejected: OpenVLA loads fine in-process via
  transformers; a sidecar adds latency and a Py3.10 venv for no benefit here.
- **Force the RLinf checkpoint onto PickCube/Panda anyway.** Rejected: it cannot solve it
  (off-distribution, `bridge_orig` stats); ADR-0060 blocks it; reporting its 0 would be dishonest.
- **Continuous-L1-head OpenVLA-OFT (LIBERO) path.** Deferred: this checkpoint uses the discrete-token
  head and ships no separate `action_head.pt` / `proprio_projector.pt`. The adapter can grow an
  L1-head + proprio path later when a LIBERO OFT rSkill is added.
