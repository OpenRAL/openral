"""The nf4 prequant overlay loads a pack completely or refuses it.

The fast meta-init path has no other weight source, so a pack that leaves a
slot unfilled (or carries a key naming none) must raise, not run on reset
values. Uses real bitsandbytes ``Linear4bit`` packing on CUDA, round-tripped
through safetensors the way ``tools/quantize_rskill.py`` writes a pack.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("bitsandbytes") is None
    or importlib.util.find_spec("accelerate") is None,
    reason="needs bitsandbytes + accelerate: just sync --group sim",
)


def _torch() -> Any:
    import torch

    if not torch.cuda.is_available():
        pytest.skip("nf4 packing needs a CUDA device")
    return torch


def _net(torch: Any) -> Any:
    torch.manual_seed(0)
    return torch.nn.Sequential(
        torch.nn.Linear(256, 256), torch.nn.LayerNorm(256), torch.nn.Linear(256, 8)
    ).to(torch.bfloat16)


def _pack(torch: Any, tmp_path: Path) -> tuple[dict[str, Any], Any]:
    """Quantize a real net, write its state dict as a pack, read it back."""
    from openral_sim._quantization import quantize_nf4_in_place
    from safetensors.torch import load_file, save_file

    src = _net(torch)
    quantize_nf4_in_place(src, torch=torch, compute_dtype=torch.bfloat16, min_params=1)
    src = src.to("cuda")
    path = tmp_path / "model.safetensors"
    save_file({k: v.contiguous().cpu() for k, v in src.state_dict().items()}, str(path))
    return load_file(str(path), device="cpu"), src


def _shell(torch: Any) -> Any:
    """The fast meta-init shell: meta graph, Linear4bit on meta, to_empty."""
    from accelerate import init_empty_weights  # type: ignore[import-untyped,unused-ignore]
    from openral_sim._quantization import quantize_nf4_in_place

    with init_empty_weights():
        shell = _net(torch)
    quantize_nf4_in_place(
        shell, torch=torch, compute_dtype=torch.bfloat16, min_params=1, new_modules_on_meta=True
    )
    shell.to_empty(device="cpu")
    return shell


def test_a_complete_pack_reproduces_the_source(tmp_path: Path) -> None:
    from openral_sim._quantization import overlay_prequantized_state

    torch = _torch()
    state, src = _pack(torch, tmp_path)
    shell = _shell(torch)
    overlay_prequantized_state(
        shell, state, device="cuda", torch=torch, family="test", source="pack"
    )
    shell = shell.to("cuda")
    x = torch.randn(4, 256, dtype=torch.bfloat16, device="cuda")
    with torch.no_grad():
        assert torch.equal(shell(x), src(x))


@pytest.mark.parametrize("drop", ["1.weight", "2.weight", "0.bias"])
def test_a_pack_missing_a_slot_is_refused(tmp_path: Path, drop: str) -> None:
    from openral_core.exceptions import ROSConfigError
    from openral_sim._quantization import overlay_prequantized_state

    torch = _torch()
    state, _ = _pack(torch, tmp_path)
    del state[drop]
    with pytest.raises(ROSConfigError, match="1 missing"):
        overlay_prequantized_state(
            _shell(torch), state, device="cuda", torch=torch, family="test", source="pack"
        )


def test_a_pack_key_naming_no_slot_is_refused(tmp_path: Path) -> None:
    from openral_core.exceptions import ROSConfigError
    from openral_sim._quantization import overlay_prequantized_state

    torch = _torch()
    state, _ = _pack(torch, tmp_path)
    state["model.1.weight"] = state.pop("1.weight")
    with pytest.raises(ROSConfigError, match="1 unexpected"):
        overlay_prequantized_state(
            _shell(torch), state, device="cuda", torch=torch, family="test", source="pack"
        )
