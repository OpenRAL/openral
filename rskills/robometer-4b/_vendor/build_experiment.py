"""Experiment: can we produce a PRE-quantized Robometer checkpoint that reloads
directly as 4-bit (no bf16 base+ckpt read, no requantize)? (ADR-0057 load fix)

Measures the current load path's cost, then tries save_pretrained + reload.
Run: /tmp/robometer-env/bin/python rskills/robometer-4b/_vendor/build_experiment.py
"""

from __future__ import annotations

import json
import os
import pathlib
import resource
import time

# Deterministic cuBLAS so the reward ramp is byte-stable across process launches:
# without it, cuBLAS heuristic algo selection depends on process warmup history
# (a warmed process and a cold one differ ~0.006). Must precede CUDA init.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

torch.backends.cudnn.allow_tf32 = False
torch.use_deterministic_algorithms(True, warn_only=True)
# Force the math SDP kernel: flash/mem-efficient selection is process-state
# dependent (a warmed vs cold process picks different kernels -> ~0.006 drift).
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

OUT = pathlib.Path("/tmp/robometer-nf4-ckpt")
MIN_PARAMS = 4_000_000


def _rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6  # KB->GB on linux


def _quantize_nf4_in_place(root: torch.nn.Module, compute_dtype: torch.dtype) -> int:
    import bitsandbytes as bnb

    n = 0

    def _replace(m: torch.nn.Module) -> None:
        nonlocal n
        for name, child in list(m.named_children()):
            if isinstance(child, torch.nn.Linear) and child.weight.numel() >= MIN_PARAMS:
                new = bnb.nn.Linear4bit(child.in_features, child.out_features,
                                        bias=child.bias is not None,
                                        compute_dtype=compute_dtype, quant_type="nf4")
                new.weight = bnb.nn.Params4bit(child.weight.data.clone(), requires_grad=False,
                                               quant_type="nf4")
                if child.bias is not None:
                    new.bias = torch.nn.Parameter(child.bias.data.clone().to(compute_dtype),
                                                  requires_grad=False)
                setattr(m, name, new)
                n += 1
            else:
                _replace(child)

    _replace(root)
    return n


def main() -> int:
    from dataclasses import fields

    import yaml
    from huggingface_hub import hf_hub_download
    from robometer.configs.experiment_configs import ExperimentConfig
    from robometer.utils.save import resolve_checkpoint_path
    from robometer.utils.setup_utils import setup_model_and_processor

    t0 = time.monotonic()
    print("[exp] load bf16 on CPU via VANILLA path (use_unsloth=False) ...", flush=True)
    # Replicate load_model_from_hf's config assembly but force use_unsloth=False so
    # the model is built with vanilla Qwen3VLModel naming (matches the meta reload).
    resolved = resolve_checkpoint_path("robometer/Robometer-4B")
    cfg_yaml = hf_hub_download("robometer/Robometer-4B", "config.yaml")
    raw = yaml.safe_load(open(cfg_yaml))
    valid = {f.name for f in fields(ExperimentConfig)}
    exp_config = ExperimentConfig(**{k: v for k, v in raw.items() if k in valid})
    exp_config.model.use_unsloth = False
    tokenizer, processor, model = setup_model_and_processor(
        exp_config.model, str(resolved), peft_config=None)
    model = model.to("cpu").eval()
    t_load = time.monotonic() - t0
    print(f"[exp] LOAD took {t_load:.1f}s; peak RSS {_rss_gb():.1f} GB", flush=True)

    t1 = time.monotonic()
    n = _quantize_nf4_in_place(model, compute_dtype=torch.bfloat16)
    model.to("cuda")
    torch.cuda.synchronize()
    t_quant = time.monotonic() - t1
    print(f"[exp] QUANTIZE+to(cuda) took {t_quant:.1f}s; {n} modules; "
          f"{torch.cuda.memory_allocated()/1e9:.2f} GB VRAM", flush=True)

    # Save the packed state_dict explicitly (keys match a vanilla meta reload).
    from safetensors.torch import save_file

    OUT.mkdir(parents=True, exist_ok=True)
    sd = {k: (v.detach().contiguous() if hasattr(v, "detach") else v)
          for k, v in model.state_dict().items()}
    # Fold in the NON-persistent buffers (rotary inv_freq etc.) that state_dict()
    # omits, so the meta reload loads them bit-identically instead of recomputing
    # them (recompute drifts ~0.004 on the progress series). assign=True on reload
    # restores them exactly.
    n_extra = 0
    for name, buf in model.named_buffers():
        if name not in sd and buf is not None:
            sd[name] = buf.detach().contiguous()
            n_extra += 1
    print(f"[exp] folded {n_extra} non-persistent buffers into the checkpoint")
    save_file(sd, str(OUT / "model.safetensors"))
    # Save the SELF-CONTAINED processor/tokenizer/config (resized vocab + added
    # progress token) so the meta reload never touches the base model.
    processor.save_pretrained(str(OUT))
    tokenizer.save_pretrained(str(OUT))
    model.config.save_pretrained(str(OUT))
    size_gb = (OUT / "model.safetensors").stat().st_size / 1e9
    vis_keys = [k for k in sd if "visual.blocks.0.mlp.linear_fc1" in k]
    print(f"[exp] SAVE OK: {size_gb:.2f} GB, {len(sd)} tensors")
    print(f"[exp] vision mlp keys present (sample): {vis_keys}")

    # Reference forward on the SAME 10 frames the reload test uses → series_A.
    series = _forward_series(model, tokenizer, processor, exp_config)
    print(f"[exp] REFERENCE progress series (bf16+quantize, LIVE processor): "
          f"{[round(float(x), 4) for x in series]}", flush=True)

    # Same weights, but recompute with the processor/tokenizer RELOADED from the
    # checkpoint dir (what the meta reload uses). Isolates any processor
    # save_pretrained round-trip drift from any weight/construction drift.
    from transformers import AutoProcessor, AutoTokenizer

    ck_proc = AutoProcessor.from_pretrained(str(OUT))
    ck_tok = AutoTokenizer.from_pretrained(str(OUT))
    series_ck = _forward_series(model, ck_tok, ck_proc, exp_config)
    print(f"[exp] REFERENCE progress series (bf16+quantize, CKPT processor): "
          f"{[round(float(x), 4) for x in series_ck]}", flush=True)

    # --- In-process round-trip diagnostic: does from_prequantized reconstruct the
    # SAME 4-bit weights as the in-place quantization the build forward used? ---
    import bitsandbytes as bnb
    from safetensors.torch import load_file as _load_file

    probe = "model.language_model.layers.0.mlp.gate_proj"
    live_mod = dict(model.named_modules())[probe]
    saved = _load_file(str(OUT / "model.safetensors"), device="cuda")
    stats = {s.lstrip("."): saved[f"{probe}.weight{s}"]
             for s in (".absmax", ".quant_map", ".nested_absmax", ".nested_quant_map",
                       ".quant_state.bitsandbytes__nf4")
             if f"{probe}.weight{s}" in saved}
    reloaded_w = bnb.nn.Params4bit.from_prequantized(
        data=saved[f"{probe}.weight"], quantized_stats=stats, requires_grad=False,
        device="cuda")
    dq_live = bnb.functional.dequantize_4bit(
        live_mod.weight.data, live_mod.weight.quant_state).float()
    dq_reload = bnb.functional.dequantize_4bit(
        reloaded_w.data, reloaded_w.quant_state).float()
    max_abs = (dq_live - dq_reload).abs().max().item()
    print(f"[exp] DEQUANT max|live - from_prequantized| for {probe}: {max_abs:.3e} "
          f"(0.0 => bit-identical 4-bit round-trip)", flush=True)

    # Report the global numerics flags this (build) process is running under, so we
    # can compare them against the meta-reload process.
    print(f"[exp] tf32 matmul={torch.backends.cuda.matmul.allow_tf32} "
          f"cudnn.tf32={torch.backends.cudnn.allow_tf32} "
          f"fp32_precision(matmul)={torch.get_float32_matmul_precision()}", flush=True)

    # --- DEFINITIVE same-process A/B: construct the model the meta-reload way from
    # the file we just saved, run the SAME forward, compare to the build forward.
    # Identical process => identical env => isolates construction-path drift. ---
    del live_mod, dq_live, dq_reload, reloaded_w, saved
    torch.cuda.empty_cache()
    series_meta = _meta_reload_series(exp_config, base_id="Qwen/Qwen3-VL-4B-Instruct")
    print(f"[exp] META-RELOAD progress series (same process): "
          f"{[round(float(x), 4) for x in series_meta]}", flush=True)
    import numpy as np
    dmax = float(np.abs(np.asarray(series) - np.asarray(series_meta)).max())
    print(f"[exp] max|build - meta_reload| (same process) = {dmax:.3e}", flush=True)
    return 0


def _meta_reload_series(exp_config, base_id):
    """Build via meta + install_prequantized from OUT, return the progress series."""
    import bitsandbytes as bnb
    import numpy as np
    from robometer.models.rbm import RBM
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoProcessor, AutoTokenizer

    config = AutoConfig.from_pretrained(str(OUT))
    processor = AutoProcessor.from_pretrained(str(OUT))
    tokenizer = AutoTokenizer.from_pretrained(str(OUT))
    for c in (config, getattr(config, "text_config", None),
              getattr(config, "vision_config", None)):
        if c is not None:
            c._attn_implementation = "sdpa"
    with torch.device("meta"):
        model = RBM(config, processor, tokenizer, base_model=None,
                    base_model_id=base_id, model_config=exp_config.model)
    # Linear4bit shells for numel>=MIN_PARAMS (same rule as the build quantizer).
    def _shells(m):
        for name, child in list(m.named_children()):
            if isinstance(child, torch.nn.Linear) and child.weight.numel() >= MIN_PARAMS:
                setattr(m, name, bnb.nn.Linear4bit(
                    child.in_features, child.out_features, bias=child.bias is not None,
                    compute_dtype=torch.bfloat16, quant_type="nf4"))
            else:
                _shells(child)
    _shells(model)

    state = load_file(str(OUT / "model.safetensors"), device="cuda")
    sufs = (".absmax", ".quant_map", ".nested_absmax", ".nested_quant_map",
            ".quant_state.bitsandbytes__nf4", ".quant_state.bitsandbytes__fp4")
    consumed: set[str] = set()
    for prefix, module in model.named_modules():
        if not isinstance(module, bnb.nn.Linear4bit):
            continue
        wkey = f"{prefix}.weight"
        if wkey not in state:
            continue
        stats = {s.lstrip("."): state[f"{wkey}{s}"] for s in sufs if f"{wkey}{s}" in state}
        for s in sufs:
            consumed.add(f"{wkey}{s}")
        consumed.add(wkey)
        module.weight = bnb.nn.Params4bit.from_prequantized(
            data=state[wkey], quantized_stats=stats, requires_grad=False, device="cuda")
        bkey = f"{prefix}.bias"
        if module.bias is not None and bkey in state:
            module.bias = torch.nn.Parameter(state[bkey].to("cuda"), requires_grad=False)
            consumed.add(bkey)
    leftover = {k: v for k, v in state.items() if k not in consumed}
    model.load_state_dict(leftover, strict=False, assign=True)
    for bname, buf in list(model.named_buffers()):
        if not buf.is_meta:
            continue
        parent = model.get_submodule(bname.rsplit(".", 1)[0]) if "." in bname else model
        if bname in state:
            parent.register_buffer(bname.rsplit(".", 1)[-1], state[bname].to("cuda"),
                                   persistent=False)
    for _nm, mod in model.named_modules():
        ifb = getattr(mod, "inv_freq", None)
        if ifb is not None and hasattr(mod, "original_inv_freq"):
            mod.original_inv_freq = ifb
    return _forward_series(model, tokenizer, processor, exp_config)


def _forward_series(model, tokenizer, processor, exp_config):
    """Run the discrete-mode progress forward on a fixed 10-frame clip."""
    import decord
    import numpy as np
    from robometer.data.dataset_types import ProgressSample, Trajectory
    from robometer.evals.eval_server import compute_batch_outputs
    from robometer.utils.setup_utils import setup_batch_collator

    model.eval()
    vr = decord.VideoReader("/tmp/robometer_example.mp4")
    step = max(1, int(round(vr.get_avg_fps() / 3.0)))
    idx = list(range(0, len(vr), step))[:10]
    frames = vr.get_batch(idx).asnumpy().astype(np.uint8)
    collator = setup_batch_collator(processor, tokenizer, exp_config, is_eval=True)
    traj = Trajectory(frames=frames, frames_shape=tuple(frames.shape),
                      task="Pick up the object and place it in the container", id="0",
                      metadata={"subsequence_length": int(frames.shape[0])},
                      video_embeddings=None)
    batch = collator([ProgressSample(trajectory=traj, sample_type="progress")])
    inp = batch["progress_inputs"]
    for k, v in inp.items():
        if hasattr(v, "to"):
            inp[k] = v.to("cuda")
    with torch.no_grad():
        res = compute_batch_outputs(model, tokenizer, inp, sample_type="progress",
                                    is_discrete_mode=True, num_bins=100)
    return np.asarray(res["progress_pred"][0], dtype=np.float32)


if __name__ == "__main__":
    raise SystemExit(main())
