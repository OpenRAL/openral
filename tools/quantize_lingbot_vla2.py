r"""Pre-quantize LingBot-VLA 2.0 to an NF4 pack for the OpenRAL rSkill.

The LingBot-VLA 2.0 sidecar (``tools/_lingbot_vla2_server.py``) normally NF4-
quantizes the Qwen3-VL backbone *at load time*: it reads the 25.5 GB fp32
checkpoint, casts bf16, rewrites the backbone ``Linear`` layers to
``bnb.nn.Linear4bit`` shells, and lets bitsandbytes pack them to nf4 on the first
``.to(cuda)`` (~30 s). This script does that pack **once, ahead of time** and
serialises the result, so downstream deploys download ~7 GB instead of 25.5 GB
and skip the per-boot conversion.

It is the LingBot-specific counterpart of ``tools/quantize_rskill.py`` (which
handles first-class lerobot / transformers policies). LingBot cannot use that
tool: its inference graph only exists inside the upstream ``lingbotvla`` package,
reconstructed exactly as the sidecar server does (config from the repo's
``configs/vla/robotwin/robotwin.yaml``, a ``lerobot`` import stub, and the
flash-attention fallback patch). So this script **runs in the sidecar venv**
(torch 2.8 / transformers 4.57.3 / bitsandbytes) and imports the server module's
helpers, guaranteeing the produced pack is byte-for-byte what the runtime
``_nf4_backbone_in_place`` would have built.

Output layout (the house nf4 pack format ``_detect_prequantized`` /
``_overlay_prequantized`` consume; identical shape to ``quantize_rskill.py``)::

    <out>/model.safetensors            packed nf4 backbone + bf16 remainder
    <out>/quantization_metadata.json   scheme=nf4 sentinel + provenance
    <out>/config.json, tokenizer*, …   the source repo's small sidecars

Two constraints shape the implementation, both because a co-resident job may be
running on the same 8 GB GPU / 62 GB host:

* **GPU-frugal.** The 8 GB bf16 backbone will not fit alongside a sim, so we never
  move the whole model to CUDA. Each backbone ``Linear4bit`` is streamed to CUDA
  on its own (bitsandbytes packs it, ~<1 GB transient), then moved straight back
  to CPU keeping the packed uint8 + quant_state. Peak GPU stays a few hundred MB.
* **RAM-frugal.** The upstream ``load_model_weights`` merges all six fp32 shards
  into one 25.5 GB dict *and* holds the 25.5 GB model — a ~51 GB peak. We replace
  it with a shard-streaming load (one ~5 GB shard resident at a time), so the peak
  is the fp32 model (~25.5 GB) + one shard (~5 GB) ≈ 30 GB.

Usage (in the sidecar venv)::

    /path/to/.sidecar_venv/bin/python tools/quantize_lingbot_vla2.py \
        --ckpt   /path/to/lingbot-vla-v2-6b        \
        --out    /path/to/lingbot-vla-v2-6b-nf4    \
        --qwen   /path/to/Qwen3-VL-4B-Instruct

The upload to the Hub is a separate step (``hf upload`` — see the rSkill README /
the model-card publish flow); this script only writes the local pack.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys
import time
import types
from pathlib import Path
from typing import Any

# The server module lives beside this script and carries the exact model
# reconstruction + nf4 rewrite the runtime uses. Import it so the produced pack
# matches the runtime shells one-to-one.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _lingbot_vla2_server as srv_mod

# Files copied verbatim from the source checkpoint into the nf4 pack (everything
# small the loader / tokenizer / processor might read). The fp32 weight shards and
# their index are deliberately excluded — the nf4 model.safetensors is the only
# weight file the pack ships.
_COPY_SUFFIXES = (".json", ".md", ".txt")
_SKIP_WEIGHTS = {"model.safetensors", "model.safetensors.index.json"}


def _frugal_load_model_weights(self: Any, path: str, strict: bool = True) -> None:
    """Shard-streaming replacement for the upstream ``load_model_weights``.

    The upstream method merges every ``*.safetensors`` shard into one dict before a
    single ``load_state_dict`` — a ~25.5 GB transient on top of the ~25.5 GB model.
    This loads one shard at a time (``strict=False`` per shard), releasing each
    before the next, then verifies full coverage so the ``strict=True`` contract
    the caller expects still holds.
    """
    import torch  # noqa: F401  # reason: keep the sidecar-venv import local
    from safetensors import safe_open

    files = sorted(glob.glob(os.path.join(path, "*.safetensors")))
    if not files:
        raise RuntimeError(f"no *.safetensors under {path}")
    seen: set[str] = set()
    for shard_path in files:
        shard: dict[str, Any] = {}
        with safe_open(shard_path, framework="pt", device="cpu") as f:  # type: ignore[no-untyped-call]
            for key in f.keys():  # noqa: SIM118  # reason: safe_open is not a dict; no __contains__
                shard[key] = f.get_tensor(key)
                seen.add(key)
        self.vla.load_state_dict(shard, strict=False)
        del shard
    model_keys = set(self.vla.state_dict().keys())
    missing = model_keys - seen
    if strict and missing:
        raise RuntimeError(
            f"frugal load left {len(missing)} model keys unfilled, e.g. "
            f"{sorted(missing)[:8]} — the source checkpoint is incomplete."
        )


def _build_bf16_model(ckpt_dir: str, qwen: str, attn: str) -> Any:
    """Reconstruct the LingBot server, frugal-load fp32 weights, cast to bf16.

    Mirrors ``_LingBotPolicy.__init__`` up to (but not including) the device move:
    same repo resolution, cli-yaml reconstruction, lerobot stub, attn fallback, and
    ``LingbotVLAv2Server`` attribute wiring — the only change is the frugal weight
    loader. Returns the server instance with ``srv.vla.model`` bf16 on CPU.
    """
    import torch

    repo = srv_mod._resolve_repo()
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    os.chdir(repo)
    os.environ["QWEN3VL_PATH"] = qwen
    srv_mod._write_cli_yaml(repo, ckpt_dir, qwen)
    srv_mod._install_lerobot_stub()

    from deploy.lingbot_vla_v2_policy import LingbotVLAv2Server
    from lingbotvla.models.vla.lingbot_vla.qwen3vl_in_vla import apply_lingbot_qwen3_vl_patch

    srv_mod._install_attn_fallback(target=attn)

    srv = LingbotVLAv2Server.__new__(LingbotVLAv2Server)
    srv.adaptive_ensemble_alpha = 0.1
    srv.action_ensemble_horizon = 8
    srv.use_length = -1
    srv.chunk_ret = True
    srv.robot_norm_path = None
    srv.task_description = None
    srv.use_compile = False
    # Swap in the RAM-frugal shard-streaming loader before load_vla calls it.
    srv.load_model_weights = types.MethodType(_frugal_load_model_weights, srv)
    apply_lingbot_qwen3_vl_patch()

    print(f"[quantize] building graph + loading fp32 weights from {ckpt_dir} ...", flush=True)
    t0 = time.perf_counter()
    srv.vla = srv.load_vla(ckpt_dir)
    srv_mod._coerce_attn_config(getattr(srv.vla, "config", None), attn)
    srv.vla.model = srv.vla.model.to(torch.bfloat16)
    print(f"[quantize] build + load + cast bf16: {time.perf_counter() - t0:.1f} s", flush=True)
    return srv


def _stream_pack_backbone(srv: Any, *, min_params: int) -> int:
    """NF4-rewrite the Qwen3-VL backbone and pack each Linear GPU-frugally.

    Uses the server's ``_nf4_backbone_in_place`` (so the shell set is identical to
    the runtime) to build ``Linear4bit`` shells, then streams each to CUDA (bnb
    packs to nf4) and back to CPU (packed uint8 + quant_state preserved). Peak GPU
    stays a few hundred MB regardless of the 8 GB backbone size.
    """
    import bitsandbytes as bnb
    import torch

    backbone = srv.vla.model.qwenvl_with_expert.qwenvl
    srv_mod._nf4_backbone_in_place(backbone, torch=torch, min_params=min_params)

    linears = [m for _, m in backbone.named_modules() if isinstance(m, bnb.nn.Linear4bit)]
    print(
        f"[quantize] packing {len(linears)} backbone Linear4bit modules (streamed) ...", flush=True
    )
    t0 = time.perf_counter()
    for i, module in enumerate(linears):
        module.cuda()  # bitsandbytes packs bf16 -> nf4 here
        module.cpu()  # packed uint8 + quant_state move back to CPU
        torch.cuda.empty_cache()
        if (i + 1) % 50 == 0 or i + 1 == len(linears):
            peak = torch.cuda.max_memory_allocated() / 1e6
            print(
                f"[quantize]   {i + 1}/{len(linears)} packed (GPU peak {peak:.0f} MB)", flush=True
            )
    print(f"[quantize] pack: {time.perf_counter() - t0:.1f} s", flush=True)
    return len(linears)


def _save_pack(
    srv: Any,
    out_dir: Path,
    *,
    ckpt_dir: str,
    source_repo: str,
    source_revision: str,
    min_params: int,
    n_packed: int,
) -> None:
    """Write ``model.safetensors`` + ``quantization_metadata.json`` + sidecars."""
    import bitsandbytes as bnb
    import torch
    from safetensors.torch import save_file

    out_dir.mkdir(parents=True, exist_ok=True)

    state = srv.vla.state_dict()
    cleaned: dict[str, Any] = {}
    dropped: list[str] = []
    for key, value in state.items():
        if isinstance(value, torch.Tensor):
            cleaned[key] = value.contiguous().cpu()
        else:
            dropped.append(key)
    if dropped:
        print(
            f"[quantize] dropped {len(dropped)} non-tensor state entries: {dropped[:5]}", flush=True
        )

    out_path = out_dir / "model.safetensors"
    print(f"[quantize] writing {out_path} ({len(cleaned)} tensors) ...", flush=True)
    t0 = time.perf_counter()
    save_file(cleaned, str(out_path))
    size_gb = out_path.stat().st_size / 1e9
    print(f"[quantize] write: {time.perf_counter() - t0:.1f} s ({size_gb:.2f} GB)", flush=True)

    meta = {
        "source_repo": source_repo,
        "source_revision": source_revision,
        "policy_class": "lingbotvla:LingbotVLAv2Server",
        "quantization": {
            "scheme": "nf4",
            "backend": "bitsandbytes",
            "backend_version": bnb.__version__,
            "torch_version": torch.__version__,
            "compute_dtype": "bfloat16",
            "min_params_to_quantize": min_params,
            "quantized_modules": n_packed,
            "target_submodule": "model.qwenvl_with_expert.qwenvl",
            "rule": (
                f"Qwen3-VL backbone (model.qwenvl_with_expert.qwenvl) Linear layers with "
                f">={min_params:_} weight elements rewritten to bnb.nn.Linear4bit (nf4, "
                "double/nested quant, bf16 compute); the sparse-MoE action expert, align "
                "heads, and action MLPs stay bf16."
            ),
            "runtime_status": (
                "loader-backed: tools/_lingbot_vla2_server.py detects "
                "quantization_metadata.json (scheme=nf4) and overlays the packed weights "
                "via _overlay_prequantized (CUDA-only)."
            ),
        },
        "dropped_state_entries": dropped,
    }
    (out_dir / "quantization_metadata.json").write_text(json.dumps(meta, indent=2))

    # Copy the source repo's small sidecars (config/tokenizer/processor/README),
    # never the fp32 weight shards.
    copied = 0
    for src in sorted(Path(ckpt_dir).iterdir()):
        if not src.is_file() or src.name in _SKIP_WEIGHTS:
            continue
        if src.name.startswith("model-") and src.suffix == ".safetensors":
            continue
        if src.suffix in _COPY_SUFFIXES or src.name == ".gitattributes":
            shutil.copy(src, out_dir / src.name)
            copied += 1
    print(f"[quantize] copied {copied} source sidecar files into {out_dir}", flush=True)


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--ckpt", required=True, help="Local fp32 LingBot-VLA 2.0 checkpoint dir.")
    p.add_argument("--out", required=True, help="Output dir for the nf4 pack.")
    p.add_argument(
        "--qwen",
        default=None,
        help="Qwen3-VL backbone path/id (default: OPENRAL_QWEN3VL_PATH env or the HF id).",
    )
    p.add_argument("--attn", choices=("sdpa", "eager"), default="sdpa")
    p.add_argument(
        "--min-params",
        type=int,
        default=2_000_000,
        help="Per-Linear weight-element threshold (default 2M, matches the sidecar server).",
    )
    p.add_argument(
        "--source-repo",
        default="robbyant/lingbot-vla-v2-6b",
        help="Upstream repo recorded in quantization_metadata.json provenance.",
    )
    p.add_argument(
        "--source-revision",
        default="11c703bf6a5c1f45b3b69168482da11fdbba53d7",
        help="Upstream revision recorded in quantization_metadata.json provenance.",
    )
    args = p.parse_args(argv)

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    import torch

    if not torch.cuda.is_available():
        print("ERROR: nf4 packing needs a CUDA GPU (bitsandbytes has no CPU 4-bit kernel).")
        return 1

    ckpt_dir = str(Path(args.ckpt).expanduser().resolve())
    qwen = args.qwen or srv_mod._resolve_qwen()

    srv = _build_bf16_model(ckpt_dir, qwen, args.attn)
    n_packed = _stream_pack_backbone(srv, min_params=args.min_params)
    _save_pack(
        srv,
        Path(args.out).expanduser().resolve(),
        ckpt_dir=ckpt_dir,
        source_repo=args.source_repo,
        source_revision=args.source_revision,
        min_params=args.min_params,
        n_packed=n_packed,
    )
    print(
        f"[quantize] done. {n_packed} nf4 modules; pack at {args.out}. "
        f"GPU peak {torch.cuda.max_memory_allocated() / 1e6:.0f} MB.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
