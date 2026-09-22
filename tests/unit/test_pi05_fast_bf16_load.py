"""The bf16 fast meta-init path: buffers rebuilt, weights streamed straight onto the device.

Real ``torch.nn.Module``s and a real safetensors file (CLAUDE.md §1.11); no
lerobot policy is instantiated. The 3.6 B π0.5 run these helpers exist for is
measured on the rig, not here.
"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
safetensors_torch = pytest.importorskip("safetensors.torch")

from openral_sim._quantization import resolve_weights_file  # noqa: E402
from openral_sim.policies.pi05 import (  # noqa: E402
    _init_rope_and_position_buffers,
    _stream_bf16_state_to_device,
)


class _Rope(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.rope_theta = 10000.0
        self.register_buffer("inv_freq", torch.full((64,), float("nan")))


class _Vision(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("position_ids", torch.full((1, 16), -1, dtype=torch.int64))


class _Tiny(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.rope = _Rope()
        self.vision = _Vision()
        self.proj = torch.nn.Linear(8, 4)
        self.head = torch.nn.Linear(4, 2)


def test_rope_and_position_buffers_are_rebuilt() -> None:
    m = _Tiny()
    with torch.no_grad():
        _init_rope_and_position_buffers(m, torch=torch)
    assert torch.equal(m.vision.position_ids, torch.arange(16).unsqueeze(0))
    expected = 1.0 / (10000.0 ** (torch.arange(0, 128, 2).float() / 128))
    assert torch.allclose(m.rope.inv_freq, expected)


def _write_state(tmp_path: Path, model: torch.nn.Module) -> Path:
    snapshot = tmp_path / "snapshots" / "abc"
    snapshot.mkdir(parents=True)
    state = {k: v.detach().clone().float() for k, v in model.state_dict().items()}
    state.pop("head.bias")  # a key the file lacks -> reported missing
    state["not.in.model"] = torch.zeros(3)  # -> reported unexpected
    safetensors_torch.save_file(state, str(snapshot / "model.safetensors"))
    return snapshot


def test_resolve_weights_file_prefers_a_directory_holding_the_file(tmp_path: Path) -> None:
    snapshot = _write_state(tmp_path, _Tiny())
    assert resolve_weights_file(str(snapshot)) == str(snapshot / "model.safetensors")


def _stream_and_check(device: str, tmp_path: Path) -> None:
    source = _Tiny()
    with torch.no_grad():
        for p in source.parameters():
            p.copy_(torch.randn_like(p))
    snapshot = _write_state(tmp_path, source)

    target = _Tiny().to(torch.bfloat16).to(device)
    with torch.no_grad():
        for p in target.parameters():
            p.zero_()
    _stream_bf16_state_to_device(target, str(snapshot), device=device, torch=torch)

    # fp32 file -> bf16 parameters on the target device, matching to bf16 precision.
    assert target.proj.weight.dtype == torch.bfloat16
    assert str(target.proj.weight.device).startswith(device.split(":", maxsplit=1)[0])
    assert torch.allclose(
        target.proj.weight.float().cpu(), source.proj.weight.to(torch.bfloat16).float(), atol=0
    )
    assert torch.allclose(
        target.head.weight.float().cpu(), source.head.weight.to(torch.bfloat16).float()
    )
    # The key the file lacked was left alone (still zero), not corrupted.
    assert torch.count_nonzero(target.head.bias) == 0


def test_streamed_weights_land_in_place_with_the_targets_dtype(tmp_path: Path) -> None:
    _stream_and_check("cpu", tmp_path)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_streamed_weights_land_on_cuda_without_a_cpu_copy(tmp_path: Path) -> None:
    _stream_and_check("cuda:0", tmp_path)
