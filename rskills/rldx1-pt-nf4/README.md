---
tags:
  - openral
  - rskill
  - rldx
  - vla
  - franka
  - pretrained
  - foundation-model
  - non-commercial
license: other
license_name: rlwrld-model-license-v1.0
license_link: https://huggingface.co/RLWRLD/RLDX-1-PT
language:
  - en
---

# rskill-rldx1-pt-nf4

> Pretrain (foundation) checkpoint for the
> [RLDX-1 collection](https://huggingface.co/collections/RLWRLD/rldx-1).
> Listed in-tree so contributors who want to finetune RLDX-1 on a new
> embodiment can pin the upstream revision.

<!-- openral:rskill-readme-delegates-to: ../rldx1-ft-libero-nf4 -->

This is a sibling of [`rldx1-ft-libero-nf4`](https://huggingface.co/OpenRAL/rskill-rldx1-ft-libero-nf4);
that README owns the canonical architecture, license, auto-managed
sidecar lifecycle, and NF4 quantization documentation for every member
of the RLDX-1 family. Read it first.

**This is not a deployable s1 policy on its own** — RLDX-1-PT has no
per-suite finetune and is not benchmarked. Use it via:

```bash
# Finetune from the PT checkpoint for your embodiment (upstream workflow):
python -m rldx.training.finetune \
    --model-path RLWRLD/RLDX-1-PT \
    --dataset <your-LeRobotDataset> \
    --embodiment-tag <your-embodiment>

# Then re-package the resulting checkpoint as a new rskill mirroring
# the rldx1-ft-libero-nf4 layout.
```

Upstream: <https://huggingface.co/RLWRLD/RLDX-1-PT>
