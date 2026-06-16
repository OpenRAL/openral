"""Phase 2 NF4 quantization probe for Robometer-4B (ADR-0057).

Replicates the repo's NF4 rewrite rule from openral_sim._quantization
(quantize_nf4_in_place: nn.Linear with weight.numel() >= 4M -> bnb.nn.Linear4bit,
quant_type="nf4", compute_dtype=bf16; pack happens on .to(cuda)). The rule is
inlined here because the isolated robometer venv cannot import openral_sim.

Loads RBM bf16 on CPU, quantizes, moves to CUDA, runs one forward, and reports
peak VRAM — the empirical answer to "how easy to quantize" + "does it leave 8 GB
headroom for a parallel VLA?".

    /tmp/robometer-env/bin/python rskills/robometer-4b/_vendor/quant_probe.py
"""

from __future__ import annotations

import numpy as np
import torch

MIN_PARAMS = 4_000_000  # DEFAULT_MIN_PARAMS_TO_QUANTIZE (openral_sim._quantization)


def quantize_nf4_in_place(root: torch.nn.Module, compute_dtype: torch.dtype) -> int:
    import bitsandbytes as bnb

    n = 0

    def _replace(module: torch.nn.Module, prefix: str = "") -> None:
        nonlocal n
        for name, child in list(module.named_children()):
            if isinstance(child, torch.nn.Linear) and child.weight.numel() >= MIN_PARAMS:
                new = bnb.nn.Linear4bit(
                    child.in_features,
                    child.out_features,
                    bias=child.bias is not None,
                    compute_dtype=compute_dtype,
                    quant_type="nf4",
                )
                new.weight = bnb.nn.Params4bit(
                    child.weight.data.clone(),
                    requires_grad=False,
                    quant_type="nf4",
                )
                if child.bias is not None:
                    new.bias = torch.nn.Parameter(
                        child.bias.data.clone().to(compute_dtype),
                        requires_grad=False,
                    )
                setattr(module, name, new)
                n += 1
            else:
                _replace(child, f"{prefix}.{name}" if prefix else name)

    _replace(root)
    return n


def main() -> int:
    from robometer.data.dataset_types import ProgressSample, Trajectory
    from robometer.evals.eval_server import compute_batch_outputs
    from robometer.utils.save import load_model_from_hf
    from robometer.utils.setup_utils import setup_batch_collator

    assert torch.cuda.is_available(), "need CUDA for the VRAM measurement"

    print("[quant] loading bf16 on CPU ...", flush=True)
    exp_config, tokenizer, processor, reward_model = load_model_from_hf(
        model_path="robometer/Robometer-4B",
        device="cpu",
    )
    reward_model.eval()

    print("[quant] rewriting large Linears -> NF4 ...", flush=True)
    n = quantize_nf4_in_place(reward_model, compute_dtype=torch.bfloat16)
    print(f"[quant] rewrote {n} Linear modules to NF4", flush=True)

    torch.cuda.reset_peak_memory_stats()
    print("[quant] moving to CUDA (packs nf4) ...", flush=True)
    reward_model.to("cuda")
    torch.cuda.synchronize()
    resident = torch.cuda.memory_allocated() / 1e9
    print(f"[quant] NF4 weights resident on CUDA: {resident:.2f} GB", flush=True)

    # one forward to confirm correctness post-quant
    batch_collator = setup_batch_collator(processor, tokenizer, exp_config, is_eval=True)
    T = 8
    frames = np.random.randint(0, 255, (T, 224, 224, 3), dtype=np.uint8)
    traj = Trajectory(
        frames=frames,
        frames_shape=tuple(frames.shape),
        task="pick up the cube",
        id="0",
        metadata={"subsequence_length": T},
        video_embeddings=None,
    )
    batch = batch_collator([ProgressSample(trajectory=traj, sample_type="progress")])
    progress_inputs = batch["progress_inputs"]
    for k, v in progress_inputs.items():
        if hasattr(v, "to"):
            progress_inputs[k] = v.to("cuda")

    print("[quant] running forward (discrete mode) ...", flush=True)
    with torch.no_grad():
        results = compute_batch_outputs(
            reward_model,
            tokenizer,
            progress_inputs,
            sample_type="progress",
            is_discrete_mode=True,
            num_bins=100,
        )
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() / 1e9

    prog = np.asarray(
        results["progress_pred"][0]
        if isinstance(results["progress_pred"], list)
        else results["progress_pred"],
        dtype=np.float32,
    )
    succ = results.get("outputs_success", {}).get("success_probs")
    succ = (
        np.asarray(succ[0] if isinstance(succ, list) and succ else succ, dtype=np.float32)
        if succ is not None
        else np.array([])
    )

    print("\n=== NF4 RESULT ===")
    print(f"  modules quantized: {n}")
    print(f"  NF4 resident VRAM: {resident:.2f} GB")
    print(f"  peak VRAM (incl. 8-frame forward activations): {peak:.2f} GB")
    print(f"  progress_pred: shape={prog.shape} range=[{prog.min():.4f},{prog.max():.4f}]")
    if succ.size:
        print(f"  success_probs: shape={succ.shape} range=[{succ.min():.4f},{succ.max():.4f}]")
    print(f"  GPU total 8.0 GB -> headroom after peak: {8.0 - peak:.2f} GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
