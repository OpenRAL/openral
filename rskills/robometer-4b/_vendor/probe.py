"""Phase 0 load+forward probe for robometer/Robometer-4B (ADR-0057 gating spike).

Mirrors robometer's scripts/example_inference_local.py but builds a tiny synthetic
clip programmatically (no example_videos needed, since we pip-installed the package
rather than cloning). Goal: confirm the model loads with inference-only deps and
print the exact output field names / shapes / value ranges for progress + success.

Run inside the isolated env:
    /tmp/robometer-env/bin/python rskills/robometer-4b/_vendor/probe.py
Optional: --weights <dir> to probe a quantized checkpoint, --device cuda.
"""

from __future__ import annotations

import argparse
import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="robometer/Robometer-4B")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--hw", type=int, default=224)
    ap.add_argument("--task", default="pick up the cube and place it in the bowl")
    ap.add_argument("--discrete", action="store_true")
    ap.add_argument("--num-bins", type=int, default=100)
    args = ap.parse_args()

    from robometer.data.dataset_types import ProgressSample, Trajectory
    from robometer.evals.eval_server import compute_batch_outputs
    from robometer.utils.save import load_model_from_hf
    from robometer.utils.setup_utils import setup_batch_collator

    print(f"[probe] loading {args.model_path} on {args.device} ...", flush=True)
    exp_config, tokenizer, processor, reward_model = load_model_from_hf(
        model_path=args.model_path,
        device=args.device,
    )
    print(f"[probe] loaded model class = {type(reward_model).__name__}", flush=True)
    print(f"[probe] exp_config type = {type(exp_config).__name__}", flush=True)

    batch_collator = setup_batch_collator(processor, tokenizer, exp_config, is_eval=True)

    T, H, W, C = args.frames, args.hw, args.hw, 3
    video_frames = np.random.randint(0, 255, (T, H, W, C), dtype=np.uint8)
    traj = Trajectory(
        frames=video_frames,
        frames_shape=tuple(video_frames.shape),
        task=args.task,
        id="0",
        metadata={"subsequence_length": T},
        video_embeddings=None,
    )
    progress_sample = ProgressSample(trajectory=traj, sample_type="progress")
    batch = batch_collator([progress_sample])
    # The collator returns a wrapper dict; the model inputs live under "progress_inputs".
    progress_inputs = batch["progress_inputs"]
    for key, value in progress_inputs.items():
        if hasattr(value, "to"):
            progress_inputs[key] = value.to(args.device)
    reward_model.eval()

    print(f"[probe] progress_inputs keys = {sorted(progress_inputs.keys())}", flush=True)
    print("[probe] running compute_batch_outputs ...", flush=True)
    results = compute_batch_outputs(
        reward_model,
        tokenizer,
        progress_inputs,
        sample_type="progress",
        is_discrete_mode=args.discrete,
        num_bins=args.num_bins,
    )

    print("\n=== RESULT KEYS ===")
    print(sorted(results.keys()))

    # success lives nested under outputs_success
    outputs_success = results.get("outputs_success")
    if isinstance(outputs_success, dict):
        print(f"[probe] outputs_success keys = {sorted(outputs_success.keys())}")
        for k, v in outputs_success.items():
            try:
                arr = np.asarray(v[0] if (isinstance(v, list) and v) else v, dtype=np.float32)
                print(f"  outputs_success[{k}]: shape={arr.shape} "
                      f"min={arr.min():.4f} max={arr.max():.4f}" if arr.size else f"  outputs_success[{k}]: empty")
            except Exception as exc:  # noqa: BLE001
                print(f"  outputs_success[{k}]: <{type(v).__name__}> ({exc})")

    def describe(name: str) -> None:
        val = results.get(name)
        if val is None:
            print(f"  {name}: <absent>")
            return
        arr = np.asarray(val[0] if (isinstance(val, list) and val) else val, dtype=np.float32)
        if arr.size:
            print(f"  {name}: shape={arr.shape} min={arr.min():.4f} max={arr.max():.4f} "
                  f"first5={np.round(arr.flatten()[:5], 4).tolist()}")
        else:
            print(f"  {name}: empty")

    print("\n=== PROGRESS / SUCCESS ===")
    for key in ("progress_pred", "success_probs", "success_pred", "preference", "pref_logits"):
        describe(key)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
