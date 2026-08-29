"""`install_linear4bit_shells` must leave its shells on the meta device.

The shells exist only so `install_prequantized` has somewhere to drop the
checkpoint's packed NF4 tensors — every one of them is overwritten. Built on
the default device instead, each costs host RAM proportional to the **dense**
layer, which is the materialization NF4 exists to avoid: loading
Robometer-4B peaked at 14.3 GB RSS and was OOM-killed before a single packed
weight was read.

The end-to-end proof is `test_reward_monitor_client.py::test_e2e_score_clip_in_process`,
but that needs a local GPU and the Robometer deps, so it skips on CI. This test
needs neither and pins the property directly. `openral_sim._quantization` guards
its own `Linear4bit` constructor the same way, via `accelerate.init_empty_weights`.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))

from _robometer_quant import MIN_PARAMS, install_linear4bit_shells

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("bitsandbytes") is None,
    reason="bitsandbytes not installed in this env",
)


class _Tree(nn.Module):
    """One Linear above the NF4 size rule and one below it."""

    def __init__(self) -> None:
        super().__init__()
        # 2048*2048 = 4.19M >= MIN_PARAMS (4M): quantized.
        self.big = nn.Linear(2048, 2048, bias=False)
        # 8*8 = 64: left alone.
        self.small = nn.Linear(8, 8, bias=False)


def test_shells_are_installed_on_meta_not_host_ram() -> None:
    """The regression. A real-storage shell is what OOM-killed the 4B load."""
    import bitsandbytes as bnb

    tree = _Tree()
    assert tree.big.weight.numel() >= MIN_PARAMS
    replaced = install_linear4bit_shells(tree, torch.bfloat16)

    assert replaced == 1, "only the Linear above MIN_PARAMS should be replaced"
    assert isinstance(tree.big, bnb.nn.Linear4bit)
    assert tree.big.weight.is_meta, (
        "Linear4bit shell allocated real storage — on a 4B model this is "
        "gigabytes of host RAM for a tensor install_prequantized overwrites"
    )


def test_small_linears_are_untouched_and_keep_real_weights() -> None:
    """The size rule still selects, and a skipped Linear is not moved to meta."""
    tree = _Tree()
    install_linear4bit_shells(tree, torch.bfloat16)
    assert isinstance(tree.small, nn.Linear)
    assert not tree.small.weight.is_meta
