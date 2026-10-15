# `rskills/act-aloha/eval/` — benchmark results

`aloha_transfer_cube.json` is the ALOHA bimanual cube-transfer benchmark
result block for this rSkill (50 episodes, seeds 0–49, `avg_success_rate`
0.82). `scene_aloha_transfer_cube.json` is the single-episode
`openral benchmark scene` demo run, not the protocol result. Validated against
[`openral_core.RSkillEvalResult`](../../../docs/reference/schemas/RSkillEvalResult.json)
at load time by the `rSkill` loader and surfaced by `openral benchmark report`.

| Field | Value |
| --- | --- |
| Source | Zhao et al., 2023 — *Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware* (arxiv:2304.13705) |
| Benchmark | ALOHA bimanual cube transfer (`gym_aloha/AlohaTransferCube-v0`) |
| Robot | Trossen ALOHA (2 × 7-DoF + parallel grippers, 14-DoF action) |
| Reproduced locally? | ✓ — `openral benchmark run`, 50-episode paper protocol (max 400 steps). |
| Reproduce | `openral benchmark run --suite aloha --rskill act-aloha`; `just sim-act-aloha` runs a single-episode demo. |
