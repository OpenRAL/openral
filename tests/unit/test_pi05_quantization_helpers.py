"""Unit tests for ``openral_sim._quantization`` Linear rewrites.

Exercises both ``quantize_nf4_in_place`` and ``quantize_int8_in_place``
against a real ``torch.nn.Module`` whose Linears straddle the 4M-param
selection threshold. The test verifies that large Linears get swapped
for the matching ``bitsandbytes`` Linear class and small ones are left
untouched -- the same contract :func:`openral_sim.policies.pi05._build_pi05`
relies on before its first ``.to(<cuda>)`` call.

Skipped when ``torch`` or ``bitsandbytes`` isn't importable (CLAUDE.md
§1.11 -- no mocks; real component or typed skip).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
bnb = pytest.importorskip("bitsandbytes")

from openral_sim._quantization import (  # noqa: E402
    DEFAULT_MIN_PARAMS_TO_QUANTIZE,
    quantize_int8_in_place,
    quantize_nf4_in_place,
)


def _build_module() -> torch.nn.Module:
    """Build a 2-Linear toy module: one above the 4M threshold, one below.

    Returns a module exposing:

    * ``big`` (2048 × 2048 = 4_194_304 params) -- crosses
      :data:`DEFAULT_MIN_PARAMS_TO_QUANTIZE`. Must be rewritten.
    * ``small`` (64 × 64 = 4_096 params) -- well below. Must stay
      ``torch.nn.Linear``.
    """

    class Tiny(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.big = torch.nn.Linear(2048, 2048, bias=True)
            self.small = torch.nn.Linear(64, 64, bias=True)

    return Tiny()


def test_quantize_nf4_swaps_only_large_linears() -> None:
    module = _build_module()
    assert module.big.weight.numel() >= DEFAULT_MIN_PARAMS_TO_QUANTIZE
    assert module.small.weight.numel() < DEFAULT_MIN_PARAMS_TO_QUANTIZE

    quantize_nf4_in_place(module, torch=torch, compute_dtype=torch.bfloat16)

    assert isinstance(module.big, bnb.nn.Linear4bit)
    assert isinstance(module.big.weight, bnb.nn.Params4bit)
    assert isinstance(module.small, torch.nn.Linear)


def test_quantize_int8_swaps_only_large_linears() -> None:
    module = _build_module()

    quantize_int8_in_place(module, torch=torch, compute_dtype=torch.bfloat16)

    assert isinstance(module.big, bnb.nn.Linear8bitLt)
    assert isinstance(module.big.weight, bnb.nn.Int8Params)
    # has_fp16_weights=False is what makes bnb actually pack to int8 on
    # the next .to(cuda); a stray True here would silently keep the
    # weights in fp16 and waste the memory savings.
    assert module.big.weight.has_fp16_weights is False
    assert isinstance(module.small, torch.nn.Linear)


def test_quantize_int8_respects_min_params_override() -> None:
    module = _build_module()

    # Lower the threshold so even the 4K-param ``small`` Linear is
    # rewritten -- the same lever ``tools/quantize_rskill.py`` exposes
    # via ``--min-params`` and the only realistic way to test the
    # tail of the selection rule on a CPU-friendly toy module.
    quantize_int8_in_place(
        module,
        torch=torch,
        compute_dtype=torch.bfloat16,
        min_params=1_000,
    )

    assert isinstance(module.big, bnb.nn.Linear8bitLt)
    assert isinstance(module.small, bnb.nn.Linear8bitLt)


# ── new_modules_on_meta path ───────────────────────────────────────────────


def test_quantize_nf4_meta_path_builds_real_params4bit() -> None:
    """``new_modules_on_meta=True`` still produces real Params4bit data.

    The flag wraps the bnb constructor call in ``init_empty_weights`` so
    its bf16 placeholder allocation lands on the meta device, but the
    subsequent ``new.weight = Params4bit(child.weight.data.clone(), ...)``
    must still carry the *real* source weights into the new module —
    otherwise the upcoming ``.to(<cuda>)`` would pack a meta tensor and
    crash (or worse, silently propagate garbage).
    """
    pytest.importorskip("accelerate")
    module = _build_module()
    src_big = module.big.weight.data.clone()  # snapshot real bf16 data

    quantize_nf4_in_place(
        module,
        torch=torch,
        compute_dtype=torch.bfloat16,
        new_modules_on_meta=True,
    )

    assert isinstance(module.big, bnb.nn.Linear4bit)
    assert isinstance(module.big.weight, bnb.nn.Params4bit)
    # Real data carried through — not meta, not zeros.
    assert module.big.weight.device.type != "meta"
    assert torch.equal(module.big.weight.data, src_big.to(module.big.weight.dtype))


def test_quantize_int8_meta_path_builds_real_int8params() -> None:
    """Same contract for the int8 sibling.

    The int8 path has no prequant fast-path — the only reason to set
    ``new_modules_on_meta=True`` is to skip the bnb constructor's bf16
    placeholder allocation, which is ~14 GiB across a 3.4 B-param graph.
    Real source data still has to land in the new Int8Params.
    """
    pytest.importorskip("accelerate")
    module = _build_module()
    src_big = module.big.weight.data.clone()

    quantize_int8_in_place(
        module,
        torch=torch,
        compute_dtype=torch.bfloat16,
        new_modules_on_meta=True,
    )

    assert isinstance(module.big, bnb.nn.Linear8bitLt)
    assert isinstance(module.big.weight, bnb.nn.Int8Params)
    assert module.big.weight.has_fp16_weights is False
    assert module.big.weight.device.type != "meta"
    assert torch.equal(module.big.weight.data, src_big.to(module.big.weight.dtype))
