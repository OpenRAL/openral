"""Diagnose meta-skeleton ↔ checkpoint key divergence (ADR-0057 load fix)."""

from __future__ import annotations

import pathlib
from dataclasses import fields

import torch
import yaml

CKPT = pathlib.Path("/tmp/robometer-nf4-ckpt/model.safetensors")


def main() -> int:
    from huggingface_hub import hf_hub_download
    from robometer.configs.experiment_configs import ExperimentConfig
    from robometer.models.rbm import RBM
    from safetensors import safe_open
    from transformers import AutoConfig

    base_id = "Qwen/Qwen3-VL-4B-Instruct"
    raw = yaml.safe_load(open(hf_hub_download("robometer/Robometer-4B", "config.yaml")))
    valid = {f.name for f in fields(ExperimentConfig)}
    exp_config = ExperimentConfig(**{k: v for k, v in raw.items() if k in valid})

    for cfg_src in ("robometer/Robometer-4B", base_id):
        config = AutoConfig.from_pretrained(cfg_src)
        with torch.device("meta"):
            model = RBM(
                config,
                None,
                None,
                base_model=None,
                base_model_id=base_id,
                model_config=exp_config.model,
            )
        mkeys = set(model.state_dict().keys())
        vis_fc1 = [k for k in mkeys if "visual.blocks.0.mlp" in k]
        vt = type(dict(model.named_modules()).get("model.visual.blocks.0.mlp.linear_fc1"))
        lin = dict(model.named_modules()).get("model.visual.blocks.0.mlp.linear_fc1")
        numel = lin.weight.numel() if lin is not None else None
        print(f"\n=== config from {cfg_src} ===")
        print(f"  model.visual.blocks.0.mlp keys: {vis_fc1}")
        print(f"  linear_fc1 type={vt} numel={numel}")

    with safe_open(str(CKPT), framework="pt") as f:
        skeys = set(f.keys())
    print("\n=== checkpoint file ===")
    print(f"  total keys: {len(skeys)}")
    print(f"  visual.blocks.0.mlp keys: {sorted(k for k in skeys if 'visual.blocks.0.mlp' in k)}")
    # model keys (last config = base) vs file
    only_model = sorted(mkeys - skeys)[:8]
    only_file = sorted(skeys - mkeys)[:8]
    print(f"\n  in model not file (sample): {only_model}")
    print(f"  in file not model (sample): {only_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
