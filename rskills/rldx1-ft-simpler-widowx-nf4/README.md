---
tags:
  - openral
  - rskill
  - rldx
  - vla
  - widowx
  - simpler
  - manipulation
  - non-commercial
license: other
license_name: rlwrld-model-license-v1.0
license_link: https://huggingface.co/RLWRLD/RLDX-1-PT
language:
  - en
---

# rskill-rldx1-ft-simpler-widowx-nf4

> RLDX-1 finetuned on the [SIMPLER](https://github.com/simpler-env/SimplerEnv)
> WidowX benchmark (real-to-sim correlator), packaged for OpenRAL.

<!-- openral:rskill-readme-delegates-to: ../rldx1-ft-libero-nf4 -->

This is a sibling of [`rldx1-ft-libero-nf4`](https://huggingface.co/OpenRAL/rskill-rldx1-ft-libero-nf4);
that README owns the canonical architecture, license, auto-managed
sidecar lifecycle, and NF4 quantization documentation for every member
of the RLDX-1 family. Read it first.

The skill-specific differences from the canonical sibling:

- Upstream weights: `hf://RLWRLD/RLDX-1-FT-SIMPLER-WIDOWX`.
- Benchmark target: SIMPLER WidowX (real-to-sim correlation for the
  Bridge / WidowX evaluation protocol).
- Same NF4-on-backbone, bf16-on-head quantization recipe; same sidecar
  boot path; same non-commercial license posture
  (`OPENRAL_ALLOW_NONCOMMERCIAL=1` required for any activation, plus a
  vendor agreement with RLWRLD for commercial deployment).

Upstream: <https://huggingface.co/RLWRLD/RLDX-1-FT-SIMPLER-WIDOWX>
