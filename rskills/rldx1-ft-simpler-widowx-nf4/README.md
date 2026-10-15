---
language:
- en
license: other
license_name: rlwrld-non-commercial
pipeline_tag: robotics
tags:
- OpenRAL
- rskill
- rldx
- vision-language-action
- nf4
- 4-bit
- widowx
- openral
- vla
- simpler
- manipulation
- non-commercial
inference: false
license_link: https://huggingface.co/RLWRLD/RLDX-1-PT
---

# rskill-rldx1_ft-widowx-simpler_env-nf4

> RLDX-1 finetuned on the [SIMPLER](https://github.com/simpler-env/SimplerEnv)
> WidowX benchmark (real-to-sim correlator), packaged for OpenRAL.

<!-- openral:rskill-readme-delegates-to: ../rldx1-ft-libero-nf4 -->

Sibling of [`rldx1-ft-libero-nf4`](https://huggingface.co/OpenRAL/rskill-rldx1_ft-franka_panda-libero_spatial-nf4),
whose README is canonical for the RLDX-1 family's architecture, license,
auto-managed sidecar lifecycle, and NF4 quantization. Read it first.

Differences from the canonical sibling:

- Upstream weights: `hf://RLWRLD/RLDX-1-FT-SIMPLER-WIDOWX`.
- Benchmark target: SIMPLER WidowX (real-to-sim correlation for the
  Bridge / WidowX evaluation protocol).
- Same sidecar NF4 quantization recipe (backbone and MSAT head; see the
  canonical README); same sidecar boot path; same non-commercial license posture
  (`OPENRAL_ALLOW_NONCOMMERCIAL=1` required for any activation, plus a
  vendor agreement with RLWRLD for commercial deployment).

Upstream: <https://huggingface.co/RLWRLD/RLDX-1-FT-SIMPLER-WIDOWX>
