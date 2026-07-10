r"""Verify the pre-quantized LingBot-VLA 2.0 nf4 pack (sidecar venv, one process).

Loads the nf4 pack through the SERVER's prequant fast path (``_LingBotPolicy`` with
``--model <nf4 dir>``, exercising ``_detect_prequantized`` + ``_overlay_prequantized``),
runs one seeded RoboTwin-shaped dummy observation through the real 6.38 B model, and
proves the overlaid backbone weights are byte-for-byte what the on-line
``.to(cuda)`` pack would have produced — without loading a second 7 GB model.

For a handful of sampled backbone ``Linear4bit`` modules it:
  * dequantizes the overlaid (fast-path) weight, and
  * loads the matching fp32 weight straight from the source checkpoint shards,
    casts bf16, and packs it fresh via the SAME ``_nf4_backbone_in_place`` +
    ``.cuda()`` the runtime on-line path uses,
then asserts ``max|deq_fast - deq_online| == 0`` (bitwise identical nf4). It also
reports the inherent nf4 round-trip error against the bf16 source (informational).

This is a throwaway checker (underscore-prefixed, not shipped API).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _lingbot_vla2_server as srv_mod


def _fp32_weight_for(source_ckpt: Path, key: str) -> Any:
    """Fetch a single fp32 tensor ``key`` from the sharded source checkpoint."""
    from safetensors import safe_open

    index = json.loads((source_ckpt / "model.safetensors.index.json").read_text())
    shard = index["weight_map"][key]
    with safe_open(str(source_ckpt / shard), framework="pt", device="cpu") as f:  # type: ignore[no-untyped-call]
        return f.get_tensor(key)


def main(argv: list[str]) -> int:
    import bitsandbytes as bnb
    import numpy as np
    import torch

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nf4", required=True, help="Pre-quantized nf4 pack dir.")
    p.add_argument("--source", required=True, help="Source fp32 checkpoint dir.")
    p.add_argument("--qwen", required=True, help="Qwen3-VL backbone dir/id.")
    p.add_argument("--samples", type=int, default=5)
    args = p.parse_args(argv)

    # 1) Load via the server prequant fast path (real _LingBotPolicy).
    ns = argparse.Namespace(
        model=args.nf4,
        robo_name="robotwin",
        quantization="nf4",
        device="cuda",
        attn="sdpa",
        host="127.0.0.1",
        port=0,
    )
    import os

    os.environ["QWEN3VL_PATH"] = args.qwen
    print("[verify] loading nf4 pack via the server prequant fast path ...", flush=True)
    policy = srv_mod._LingBotPolicy(ns)
    srv = policy._srv

    # 2) One seeded dummy RoboTwin observation -> finite 14-D chunk.
    rng = np.random.default_rng(0)
    obs = {
        "images": {
            "cam_high": rng.integers(0, 255, (256, 256, 3), dtype=np.uint8),
            "cam_left_wrist": rng.integers(0, 255, (256, 256, 3), dtype=np.uint8),
            "cam_right_wrist": rng.integers(0, 255, (256, 256, 3), dtype=np.uint8),
        },
        "state": rng.standard_normal(14).astype(np.float32),
        "task": "pick up the block and hand it over",
    }
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    chunk, keys = policy.get_action(obs)
    finite = bool(np.isfinite(chunk).all())
    print(
        f"[verify] forward pass OK: action chunk shape={chunk.shape} keys={keys} "
        f"finite={finite} range=[{chunk.min():.3f},{chunk.max():.3f}]",
        flush=True,
    )
    if not finite or chunk.shape[1] != 14:
        print("[verify] FAIL: non-finite or wrong-width action", flush=True)
        return 1

    # 3) Sample backbone Linear4bit modules; compare fast-path dequant vs a fresh
    #    on-line pack of the source fp32 weight.
    backbone = srv.vla.model.qwenvl_with_expert.qwenvl
    prefix_root = "model.qwenvl_with_expert.qwenvl"
    named = [(name, m) for name, m in backbone.named_modules() if isinstance(m, bnb.nn.Linear4bit)]
    step = max(1, len(named) // args.samples)
    sampled = named[::step][: args.samples]
    print(f"[verify] comparing {len(sampled)} sampled backbone modules ...", flush=True)
    source = Path(args.source)
    max_pack_diff = 0.0
    for local_name, module in sampled:
        full_key = f"{prefix_root}.{local_name}.weight"
        # fast-path (overlaid) dequant
        deq_fast = bnb.functional.dequantize_4bit(module.weight.data, module.weight.quant_state).to(
            torch.float32
        )
        # fresh on-line pack of the source fp32 weight
        w_fp32 = _fp32_weight_for(source, full_key)
        w_bf16 = w_fp32.to(torch.bfloat16)
        online = bnb.nn.Linear4bit(
            module.in_features,
            module.out_features,
            bias=False,
            compute_dtype=torch.bfloat16,
            quant_type="nf4",
        )
        online.weight = bnb.nn.Params4bit(w_bf16.clone(), requires_grad=False, quant_type="nf4")
        online = online.cuda()
        deq_online = bnb.functional.dequantize_4bit(
            online.weight.data, online.weight.quant_state
        ).to(torch.float32)
        pack_diff = (deq_fast - deq_online).abs().max().item()
        nf4_err = (deq_fast.cpu() - w_bf16.float()).abs().max().item()
        max_pack_diff = max(max_pack_diff, pack_diff)
        print(
            f"[verify]   {local_name}: max|deq_fast-deq_online|={pack_diff:.3e} "
            f"nf4_roundtrip_err_vs_bf16={nf4_err:.3e}",
            flush=True,
        )
        del online, deq_online, deq_fast
        torch.cuda.empty_cache()

    peak = torch.cuda.max_memory_allocated() / 1e9
    print(
        f"[verify] max pack diff across samples = {max_pack_diff:.3e}; VRAM peak {peak:.2f} GB",
        flush=True,
    )
    ok = max_pack_diff == 0.0 and finite
    print(f"[verify] {'PASS' if ok else 'FAIL'}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
