"""Determinism and meta-load helpers for the Robometer reward scorer.

The NF4 code itself (the ``Linear4bit`` shells and the ``from_prequantized``
install) is OpenRAL Pro's (ADR-0123); ``_robometer_scorer.py`` and
``build_robometer_nf4_checkpoint.py`` reach it through
``openral_rskill.quantization_gate.pro_quantization``.

Determinism (CLAUDE.md §1.8): the reward ramp must be byte-stable across process
launches. Unpinned, ``scaled_dot_product_attention`` picks flash/mem-efficient
kernels by process-warmup state (warmed vs cold drifts ~0.006), so the math SDP
kernel + deterministic algorithms are forced. With that pinned, the meta
pre-quantized load equals the bf16+quantize reference bit-for-bit (same-process
``max|Δ| = 0``, cross-process equal).
"""

from __future__ import annotations

import os


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
