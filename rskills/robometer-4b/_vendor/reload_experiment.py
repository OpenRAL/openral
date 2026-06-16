"""Experiment: reload the pre-quantized 3.32 GB checkpoint WITHOUT the bf16 base
load — build the RBM skeleton from config (random init), install Linear4bit, then
load_state_dict the packed NF4 weights. Verify forward reproduces the ramp.

Prereq: run build_experiment.py first (writes /tmp/robometer-nf4-ckpt/model.safetensors).
Run: /tmp/robometer-env/bin/python rskills/robometer-4b/_vendor/reload_experiment.py
"""

from __future__ import annotations

import os
import pathlib
import resource
import time
from dataclasses import fields

import numpy as np
import yaml

# Match build_experiment: deterministic cuBLAS for byte-stable cross-process output.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

torch.backends.cudnn.allow_tf32 = False
torch.use_deterministic_algorithms(True, warn_only=True)
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

CKPT = pathlib.Path("/tmp/robometer-nf4-ckpt")
MIN_PARAMS = 4_000_000


def _rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


_BNB_META_SUFFIXES = (
    ".absmax", ".quant_map", ".nested_absmax", ".nested_quant_map",
    ".quant_state.bitsandbytes__nf4", ".quant_state.bitsandbytes__fp4",
)


def _install_prequantized(policy, state, device):
    """Inlined openral_sim._quantization.install_prequantized_linears."""
    import bitsandbytes as bnb

    consumed: set[str] = set()
    count = 0
    for prefix, module in policy.named_modules():
        if not isinstance(module, bnb.nn.Linear4bit):
            continue
        wkey = f"{prefix}.weight"
        if wkey not in state:
            continue
        stats = {}
        for suf in _BNB_META_SUFFIXES:
            full = f"{wkey}{suf}"
            if full in state:
                stats[suf.lstrip(".")] = state[full]
                consumed.add(full)
        consumed.add(wkey)
        module.weight = bnb.nn.Params4bit.from_prequantized(
            data=state[wkey], quantized_stats=stats, requires_grad=False, device=device)
        bkey = f"{prefix}.bias"
        if module.bias is not None and bkey in state:
            module.bias = torch.nn.Parameter(state[bkey].to(device), requires_grad=False)
            consumed.add(bkey)
        count += 1
    return count, consumed


def _quantize_structure(root, compute_dtype):
    """Replace large Linears with Linear4bit (empty packed params) — structure only."""
    import bitsandbytes as bnb

    n = 0

    def _replace(m):
        nonlocal n
        for name, child in list(m.named_children()):
            if isinstance(child, torch.nn.Linear) and child.weight.numel() >= MIN_PARAMS:
                new = bnb.nn.Linear4bit(child.in_features, child.out_features,
                                        bias=child.bias is not None,
                                        compute_dtype=compute_dtype, quant_type="nf4")
                setattr(m, name, new)
                n += 1
            else:
                _replace(child)

    _replace(root)
    return n


def main() -> int:
    from huggingface_hub import hf_hub_download
    from robometer.configs.experiment_configs import ExperimentConfig
    from robometer.models.rbm import RBM
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoProcessor, AutoTokenizer

    base_id = "Qwen/Qwen3-VL-4B-Instruct"
    # cheap pieces — no big model weights
    cfg_yaml = hf_hub_download("robometer/Robometer-4B", "config.yaml")
    raw = yaml.safe_load(open(cfg_yaml))
    valid = {f.name for f in fields(ExperimentConfig)}
    exp_config = ExperimentConfig(**{k: v for k, v in raw.items() if k in valid})

    # Load config + processor + tokenizer from the SELF-CONTAINED checkpoint dir
    # (resized vocab 151674 + robometer's added progress token), NOT the base.
    config = AutoConfig.from_pretrained(str(CKPT))
    processor = AutoProcessor.from_pretrained(str(CKPT))
    tokenizer = AutoTokenizer.from_pretrained(str(CKPT))
    # Direct construction (not from_pretrained) skips transformers' attn auto-select
    # and defaults to "eager"; production's setup_model_and_processor uses "sdpa"
    # (flash-attn absent). Force sdpa on every (sub)config so the meta path is
    # numerically identical to the bf16+quantize reference.
    for c in (config, getattr(config, "text_config", None),
              getattr(config, "vision_config", None)):
        if c is not None:
            c._attn_implementation = "sdpa"

    print(f"[reload] tf32 matmul={torch.backends.cuda.matmul.allow_tf32} "
          f"cudnn.tf32={torch.backends.cudnn.allow_tf32} "
          f"fp32_precision(matmul)={torch.get_float32_matmul_precision()}", flush=True)

    t0 = time.monotonic()
    print("[reload] building RBM skeleton on META (instant, no weights) ...", flush=True)
    with torch.device("meta"):
        model = RBM(config, processor, tokenizer, base_model=None,
                    base_model_id=base_id, model_config=exp_config.model)
    n = _quantize_structure(model, compute_dtype=torch.bfloat16)
    print(f"[reload] meta skeleton + {n} Linear4bit shells in "
          f"{time.monotonic()-t0:.1f}s; peak RSS {_rss_gb():.1f} GB", flush=True)

    t1 = time.monotonic()
    state = load_file(str(CKPT / "model.safetensors"), device="cuda")
    # 4-bit modules: rebuild packed weights directly on CUDA (no bf16 alloc).
    n_q, consumed = _install_prequantized(model, state, device="cuda")
    # everything else (embeddings, norms, heads): assign cuda tensors to meta params.
    leftover = {k: v for k, v in state.items() if k not in consumed}
    missing, unexpected = model.load_state_dict(leftover, strict=False, assign=True)
    # load_state_dict SKIPS non-persistent buffers (rotary inv_freq) — it reports
    # them as `unexpected` and never assigns them. So assign them by hand from the
    # checkpoint, by dotted name, bit-identically (no recompute). Only truly-absent
    # buffers fall back to recompute.
    loaded_bufs, recomputed_meta = 0, 0
    for bname, buf in list(model.named_buffers()):
        if not buf.is_meta:
            continue
        parent = model.get_submodule(bname.rsplit(".", 1)[0]) if "." in bname else model
        leaf = bname.rsplit(".", 1)[-1]
        if bname in state:  # persisted in the checkpoint -> exact restore
            parent.register_buffer(leaf, state[bname].to("cuda"), persistent=False)
            loaded_bufs += 1
        elif hasattr(parent, "rope_init_fn") and hasattr(parent, "config"):
            inv_freq, scaling = parent.rope_init_fn(parent.config, "cuda")
            parent.register_buffer(leaf, inv_freq, persistent=False)
            if hasattr(parent, "attention_scaling"):
                parent.attention_scaling = scaling
            recomputed_meta += 1
        else:  # Qwen3VLVisionRotaryEmbedding closed form
            dim = 2 * buf.shape[0]
            inv_freq = 1.0 / (10000.0 ** (torch.arange(0, dim, 2, dtype=torch.float,
                                                       device="cuda") / dim))
            parent.register_buffer(leaf, inv_freq, persistent=False)
            recomputed_meta += 1
    # Rebind the dangling non-buffer rope attributes off the (now-real) inv_freq.
    for _nm, mod in model.named_modules():
        ifb = getattr(mod, "inv_freq", None)
        if ifb is None:
            continue
        if hasattr(mod, "original_inv_freq"):
            mod.original_inv_freq = ifb
        if hasattr(mod, "rope_init_fn") and hasattr(mod, "config") \
                and getattr(mod, "attention_scaling", None) is None:
            _, scaling = mod.rope_init_fn(mod.config, "cuda")
            mod.attention_scaling = scaling
    print(f"[reload] rotary buffers loaded-from-ckpt={loaded_bufs} "
          f"recomputed-from-meta(expect 0)={recomputed_meta}")

    # any params/buffers still on meta (not in the checkpoint)?
    still_meta = [n for n, p in model.named_parameters() if p.is_meta]
    still_meta_buf = [n for n, b in model.named_buffers() if b.is_meta]
    print(f"[reload] meta params left: {still_meta[:6]}")
    print(f"[reload] meta buffers left after rotary fix: {still_meta_buf}")
    # sanity: is a vision weight real (install handled it)?
    vw = dict(model.named_parameters()).get("model.visual.blocks.0.mlp.linear_fc1.weight")
    print(f"[reload] vision fc1 weight is_meta={vw.is_meta if vw is not None else 'absent'} "
          f"dtype={vw.dtype if vw is not None else '-'}")
    torch.cuda.synchronize()
    print(f"[reload] install_prequantized({n_q}) + load_state_dict in {time.monotonic()-t1:.1f}s; "
          f"{torch.cuda.memory_allocated()/1e9:.2f} GB VRAM; "
          f"missing={len(missing)} unexpected={len(unexpected)} "
          f"meta_params_left={len(still_meta)} meta_bufs_left={len(still_meta_buf)}", flush=True)
    if missing:
        print(f"[reload] sample missing keys: {missing[:5]}")
    if unexpected:
        print(f"[reload] sample unexpected keys: {unexpected[:5]}")

    # forward on the real video → expect the ramp
    model.eval()
    from robometer.data.dataset_types import ProgressSample, Trajectory
    from robometer.evals.eval_server import compute_batch_outputs
    from robometer.utils.setup_utils import setup_batch_collator

    import decord

    vr = decord.VideoReader("/tmp/robometer_example.mp4")
    step = max(1, int(round(vr.get_avg_fps() / 3.0)))
    idx = list(range(0, len(vr), step))[:10]
    frames = vr.get_batch(idx).asnumpy().astype(np.uint8)
    collator = setup_batch_collator(processor, tokenizer, exp_config, is_eval=True)
    traj = Trajectory(frames=frames, frames_shape=tuple(frames.shape),
                      task="Pick up the object and place it in the container", id="0",
                      metadata={"subsequence_length": int(frames.shape[0])}, video_embeddings=None)
    batch = collator([ProgressSample(trajectory=traj, sample_type="progress")])
    inp = batch["progress_inputs"]
    for k, v in inp.items():
        if hasattr(v, "to"):
            inp[k] = v.to("cuda")
    with torch.no_grad():
        res = compute_batch_outputs(model, tokenizer, inp, sample_type="progress",
                                    is_discrete_mode=True, num_bins=100)
    prog = np.asarray(res["progress_pred"][0], dtype=np.float32)
    print(f"[reload] progress series (meta+prequantized path): "
          f"{[round(float(x),4) for x in prog]}")
    print(f"[reload] COMPARE to the build_experiment REFERENCE series — they must "
          f"match element-wise to ~1e-3 for the prequantized load to be trusted.",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
