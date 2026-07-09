# `rskills/act-aloha/eval/` — benchmark results

`aloha_transfer_cube.json` is the ALOHA bimanual cube-transfer benchmark
result block for this rSkill. Validated against
[`openral_core.RSkillEvalResult`](../../../docs/reference/schemas/RSkillEvalResult.json)
at load time by the `rSkill` loader and surfaced by `openral benchmark report`.

| Field | Value |
| --- | --- |
| Source | Zhao et al., 2023 — *Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware* (arxiv:2304.13705) |
| Benchmark | ALOHA bimanual cube transfer (`gym_aloha/AlohaTransferCube-v0`) |
| Robot | Trossen ALOHA (2 × 7-DoF + parallel grippers, 14-DoF action) |
| Reproduced locally? | ✗ — paper-only. `tests/sim/test_aloha_bimanual_act_aloha.py` runs a single episode for IO + latency verification but does not aggregate the 50-trial protocol. |
| Reproduce | `just sim-act-aloha` (single episode); raise `--n-episodes 50` for the full paper protocol. |
