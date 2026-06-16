"""Shared NF4 quantize + pre-quantized meta-load helpers for the Robometer
reward sidecar (ADR-0057). Runs inside the isolated sidecar venv, so it CANNOT
import ``openral_sim._quantization`` — it re-implements the same NF4 rule
(``nn.Linear`` with ``numel >= MIN_PARAMS`` -> ``bitsandbytes`` ``Linear4bit``
nf4/bf16) plus a pre-quantized direct-load path.

Used by both:
  * ``tools/build_robometer_nf4_checkpoint.py`` — produces the publishable 3.32 GB
    pre-quantized checkpoint (and folds the non-persistent rotary buffers in).
  * ``tools/_robometer_server.py`` — loads that checkpoint directly as 4-bit (no
    bf16 materialization, no requantize) via the meta device.

Determinism (CLAUDE.md §8 reproducibility): the reward ramp must be byte-stable
across process launches. Without pinning, ``scaled_dot_product_attention`` picks
flash/mem-efficient kernels by process-warmup state (a warmed vs cold process
drifts ~0.006), so we force the math SDP kernel + deterministic algorithms. With
this pinned, the meta pre-quantized load is byte-identical to the bf16+quantize
reference (verified: same-process ``max|Δ| = 0`` and cross-process equality).
"""

from __future__ import annotations

import os

MIN_PARAMS = 4_000_000  # openral_sim._quantization.DEFAULT_MIN_PARAMS_TO_QUANTIZE

# bitsandbytes Params4bit packed-stat suffixes written alongside each .weight.
BNB_META_SUFFIXES = (
    ".absmax",
    ".quant_map",
    ".nested_absmax",
    ".nested_quant_map",
    ".quant_state.bitsandbytes__nf4",
    ".quant_state.bitsandbytes__fp4",
)


def set_cublas_workspace_env() -> None:
    """Pin the cuBLAS workspace BEFORE CUDA initialises (required for determinism)."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def apply_determinism() -> None:
    """Force byte-stable numerics. Call after importing torch, before any forward."""
    import torch

    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    # Math SDP is process-state independent; flash/mem-efficient are not.
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)


def quantize_nf4_in_place(root: object, compute_dtype: object) -> int:
    """Rewrite large ``nn.Linear`` -> ``bnb.nn.Linear4bit`` (the repo's NF4 rule).

    Returns the number of modules replaced. Quantization happens lazily on the
    first ``.to("cuda")``; this only swaps the module + clones the weight.
    """
    import bitsandbytes as bnb
    import torch

    n = 0

    def _replace(module: torch.nn.Module) -> None:
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
                    child.weight.data.clone(), requires_grad=False, quant_type="nf4"
                )
                if child.bias is not None:
                    new.bias = torch.nn.Parameter(
                        child.bias.data.clone().to(compute_dtype), requires_grad=False
                    )
                setattr(module, name, new)
                n += 1
            else:
                _replace(child)

    _replace(root)
    return n


def install_linear4bit_shells(root: object, compute_dtype: object) -> int:
    """Replace large ``nn.Linear`` with EMPTY ``Linear4bit`` shells (no packing).

    Used on a meta-device skeleton so :func:`install_prequantized` can drop in the
    saved packed weights. Same selection rule as :func:`quantize_nf4_in_place`.
    """
    import bitsandbytes as bnb
    import torch

    n = 0

    def _replace(module: torch.nn.Module) -> None:
        nonlocal n
        for name, child in list(module.named_children()):
            if isinstance(child, torch.nn.Linear) and child.weight.numel() >= MIN_PARAMS:
                setattr(
                    module,
                    name,
                    bnb.nn.Linear4bit(
                        child.in_features,
                        child.out_features,
                        bias=child.bias is not None,
                        compute_dtype=compute_dtype,
                        quant_type="nf4",
                    ),
                )
                n += 1
            else:
                _replace(child)

    _replace(root)
    return n


def install_prequantized(policy: object, state: dict, device: str) -> set:
    """Drop saved packed NF4 weights into ``Linear4bit`` shells via from_prequantized.

    Returns the set of consumed checkpoint keys (so the caller can ``load_state_dict``
    the remaining dense tensors). Bit-identical to the in-place quantization that
    produced the checkpoint (verified dequant ``max|Δ| = 0``).
    """
    import bitsandbytes as bnb
    import torch

    consumed: set[str] = set()
    for prefix, module in policy.named_modules():
        if not isinstance(module, bnb.nn.Linear4bit):
            continue
        wkey = f"{prefix}.weight"
        if wkey not in state:
            continue
        stats = {}
        for suf in BNB_META_SUFFIXES:
            full = f"{wkey}{suf}"
            if full in state:
                stats[suf.lstrip(".")] = state[full]
                consumed.add(full)
        consumed.add(wkey)
        module.weight = bnb.nn.Params4bit.from_prequantized(
            data=state[wkey], quantized_stats=stats, requires_grad=False, device=device
        )
        bkey = f"{prefix}.bias"
        if module.bias is not None and bkey in state:
            module.bias = torch.nn.Parameter(state[bkey].to(device), requires_grad=False)
            consumed.add(bkey)
    return consumed


def assign_meta_buffers(model: object, state: dict, device: str) -> int:
    """Assign any still-meta buffer (non-persistent rotary inv_freq) from ``state``.

    ``load_state_dict`` SKIPS non-persistent buffers (reports them ``unexpected``
    and never assigns), so they must be set by hand. The checkpoint folds them in;
    we copy by dotted name. Rebinds the dangling ``original_inv_freq`` attribute too.
    Returns the number of buffers assigned. Raises if a meta buffer is absent from
    the checkpoint (would otherwise leave a meta tensor in the forward path).
    """
    assigned = 0
    for bname, buf in list(model.named_buffers()):
        if not buf.is_meta:
            continue
        if bname not in state:
            raise RuntimeError(f"meta buffer {bname!r} missing from checkpoint")
        parent = model.get_submodule(bname.rsplit(".", 1)[0]) if "." in bname else model
        parent.register_buffer(bname.rsplit(".", 1)[-1], state[bname].to(device), persistent=False)
        assigned += 1
    for _nm, mod in model.named_modules():
        ifb = getattr(mod, "inv_freq", None)
        if ifb is not None and hasattr(mod, "original_inv_freq"):
            mod.original_inv_freq = ifb
    return assigned


def is_prequantized_checkpoint(path: str) -> bool:
    """True if ``path`` holds a model.safetensors with bnb NF4 packed keys."""
    import pathlib

    st = pathlib.Path(path) / "model.safetensors"
    if not st.exists():
        return False
    from safetensors import safe_open

    with safe_open(str(st), framework="pt") as f:
        for k in f.keys():  # noqa: SIM118 — safetensors handle, not a dict (no __contains__/__iter__)
            if k.endswith(".quant_state.bitsandbytes__nf4"):
                return True
    return False
