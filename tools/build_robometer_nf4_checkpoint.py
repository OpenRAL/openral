"""Build the publishable pre-quantized Robometer-4B NF4 checkpoint (ADR-0057).

Loads the upstream Apache-2.0 ``robometer/Robometer-4B`` bf16 via the pinned
robometer loader (vanilla, ``use_unsloth=False``), NF4-quantizes in place, and
saves a self-contained directory the sidecar can meta-load directly as 4-bit:

  model.safetensors          packed NF4 weights + folded non-persistent rotary
                             buffers (~3.32 GB, ~1914 tensors)
  config.json                model config (resized vocab 151674)
  config.yaml                robometer ExperimentConfig (needed to rebuild RBM)
  tokenizer*/vocab/merges    tokenizer WITH robometer's added progress token
  *preprocessor_config.json  image/video processor

Run in the sidecar venv:
  ~/.cache/openral/robometer-sidecar/.venv/bin/python \
      tools/build_robometer_nf4_checkpoint.py --out /tmp/robometer-nf4-ckpt

Then upload the directory to ``OpenRAL/rskill-robometer-4b-nf4`` (see README).
"""

from __future__ import annotations

import argparse
import pathlib
import resource
import time
from dataclasses import fields

import _robometer_quant as q

q.set_cublas_workspace_env()

import torch
import yaml

q.apply_determinism()

_UPSTREAM = "robometer/Robometer-4B@beef63bc914c5c189329d49c6d712d96d632aa34"


def _rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def main() -> int:
    from huggingface_hub import hf_hub_download
    from robometer.configs.experiment_configs import ExperimentConfig
    from robometer.utils.save import resolve_checkpoint_path
    from robometer.utils.setup_utils import setup_model_and_processor
    from safetensors.torch import save_file

    ap = argparse.ArgumentParser()
    ap.add_argument("--upstream", default=_UPSTREAM, help="HF repo[@rev] to quantize")
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("/tmp/robometer-nf4-ckpt"))
    args = ap.parse_args()

    repo = args.upstream.split("@", 1)[0]
    t0 = time.monotonic()
    print(f"[build] loading {args.upstream} bf16 on CPU (vanilla, use_unsloth=False) ...",
          flush=True)
    resolved = resolve_checkpoint_path(args.upstream)
    raw = yaml.safe_load(open(hf_hub_download(repo, "config.yaml")))
    valid = {f.name for f in fields(ExperimentConfig)}
    exp = ExperimentConfig(**{k: v for k, v in raw.items() if k in valid})
    exp.model.use_unsloth = False
    tokenizer, processor, model = setup_model_and_processor(
        exp.model, str(resolved), peft_config=None)
    model = model.to("cpu").eval()
    print(f"[build] load took {time.monotonic()-t0:.1f}s; peak RSS {_rss_gb():.1f} GB",
          flush=True)

    t1 = time.monotonic()
    n = q.quantize_nf4_in_place(model, compute_dtype=torch.bfloat16)
    model.to("cuda")
    torch.cuda.synchronize()
    print(f"[build] NF4 {n} modules + to(cuda) in {time.monotonic()-t1:.1f}s; "
          f"{torch.cuda.memory_allocated()/1e9:.2f} GB VRAM", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    sd = {k: (v.detach().contiguous() if hasattr(v, "detach") else v)
          for k, v in model.state_dict().items()}
    # Fold non-persistent rotary inv_freq buffers in (load_state_dict skips them).
    folded = 0
    for name, buf in model.named_buffers():
        if name not in sd and buf is not None:
            sd[name] = buf.detach().contiguous()
            folded += 1
    save_file(sd, str(args.out / "model.safetensors"))
    processor.save_pretrained(str(args.out))
    tokenizer.save_pretrained(str(args.out))
    model.config.save_pretrained(str(args.out))
    # Ship the robometer ExperimentConfig so the sidecar can rebuild RBM offline.
    with open(args.out / "config.yaml", "w") as f:
        yaml.safe_dump(raw, f, sort_keys=False)

    size_gb = (args.out / "model.safetensors").stat().st_size / 1e9
    print(f"[build] saved {len(sd)} tensors ({folded} folded buffers), {size_gb:.2f} GB "
          f"-> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
