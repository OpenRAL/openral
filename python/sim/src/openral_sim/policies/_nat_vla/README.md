# `_nat_vla` — vendored lerobot `nat_vla`

Unmodified copy of `src/lerobot/policies/nat_vla/` from
[huggingface/lerobot](https://github.com/huggingface/lerobot) at commit
`e5be0c16be1df4ae004abb4167842a7456c4cd54` (huggingface/lerobot#5416, Apache-2.0,
Copyright 2026 Ahad Jawaid and The HuggingFace Inc. team).

lerobot 0.7.0 predates it. `openral_sim.policies.nat_vla` imports lerobot's own
`lerobot.policies.nat_vla` when the installed lerobot ships it and only falls back to
this copy, so delete this directory once the workspace pins a lerobot release with it.

Kept byte-identical (excluded from ruff and mypy in `ruff.toml` / `mypy.ini`) so it
diffs cleanly against upstream. Refresh it with:

```bash
C=<lerobot commit>
for f in __init__.py configuration_nat_vla.py modeling_nat_vla.py nat_tokenizer.py processor_nat_vla.py; do
  curl -sf -o python/sim/src/openral_sim/policies/_nat_vla/$f \
    https://raw.githubusercontent.com/huggingface/lerobot/$C/src/lerobot/policies/nat_vla/$f
done
```
