# `rskills/diffusion-pusht/eval/` — benchmark results

`pusht.json` is the PushT benchmark result block for this rSkill (50
episodes, max 300 steps: `avg_success_rate` 0.60, `mean_coverage_iou` 0.95).
`scene_pusht.json` is the single-episode `openral benchmark scene` demo run,
not the protocol result. Validated against
[`openral_core.RSkillEvalResult`](../../../docs/reference/schemas/RSkillEvalResult.json)
at load time by the `rSkill` loader and surfaced by `openral benchmark report`.

| Field | Value |
| --- | --- |
| Source | Chi et al., 2023 — *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion* (arxiv:2303.04137) |
| Benchmark | PushT (`gym_pusht/PushT-v0`, pymunk 2-D rigid-body) |
| Robot | PushT 2-D pseudo-robot (single 2-D end-effector tip) |
| Reproduced locally? | ✓ — `openral benchmark run`, 50-episode protocol. |
| Reproduce | `openral benchmark run --suite pusht --rskill diffusion-pusht`; `just sim-diffusion-pusht` runs a single-episode demo. |
