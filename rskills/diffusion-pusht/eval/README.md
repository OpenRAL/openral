# `rskills/diffusion-pusht/eval/` — benchmark results

`pusht.json` is the PushT mean-coverage-IoU benchmark result block for this
rSkill. Validated against
[`openral_core.RSkillEvalResult`](../../../docs/reference/schemas/RSkillEvalResult.json)
at load time by the `rSkill` loader and surfaced by `openral benchmark report`.

| Field | Value |
| --- | --- |
| Source | Chi et al., 2023 — *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion* (arxiv:2303.04137) |
| Benchmark | PushT (`gym_pusht/PushT-v0`, pymunk 2-D rigid-body) |
| Robot | PushT 2-D pseudo-robot (single 2-D end-effector tip) |
| Reproduced locally? | ✗ — paper-only. `tests/sim/test_pusht_2d_diffusion_pusht.py` runs a single episode for IO + latency + VRAM verification. |
| Reproduce | `just sim-diffusion-pusht` (single episode); raise `--n-episodes 50` for the full paper protocol. |
