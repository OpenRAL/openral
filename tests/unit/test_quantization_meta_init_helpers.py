"""Tests for the shared meta-init helpers in :mod:`openral_sim._quantization`.

``targeted_reset_parameters`` + ``tie_transformers_weights`` were promoted out
of the π0.5 adapter so π0.5 / MolmoAct2 / future meta-init families share one
copy (CLAUDE.md §1.13). These lock the model-agnostic contract: skip
``reset_parameters`` only for modules whose own params are fully covered by the
upcoming prequant state load, and never crash on a module that has no
``reset_parameters`` / a raising ``tie_weights``.
"""

from __future__ import annotations

import torch
from openral_sim._quantization import targeted_reset_parameters, tie_transformers_weights
from torch import nn


class _Tree(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.covered = nn.Linear(4, 4)
        self.uncovered = nn.Linear(4, 4)


class TestTargetedResetParameters:
    def test_skips_covered_resets_uncovered(self) -> None:
        """A module whose own params are all covered is skipped; others reset."""
        tree = _Tree()
        # Zero both weights so a reset (kaiming) is observable as a change.
        with torch.no_grad():
            tree.covered.weight.zero_()
            tree.uncovered.weight.zero_()
        covered = {"covered.weight", "covered.bias"}
        targeted_reset_parameters(tree, covered_keys=covered)
        # covered.weight stayed zero (reset skipped); uncovered got re-inited.
        assert torch.count_nonzero(tree.covered.weight) == 0
        assert torch.count_nonzero(tree.uncovered.weight) > 0

    def test_none_covered_resets_everything(self) -> None:
        """covered_keys=None falls back to the unconditional reset."""
        tree = _Tree()
        with torch.no_grad():
            tree.covered.weight.zero_()
        targeted_reset_parameters(tree, covered_keys=None)
        assert torch.count_nonzero(tree.covered.weight) > 0

    def test_module_without_reset_is_noop(self) -> None:
        """A module lacking reset_parameters must not raise."""

        class _NoReset(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.act = nn.ReLU()

        targeted_reset_parameters(_NoReset(), covered_keys=set())  # no exception


class TestTieTransformersWeights:
    def test_calls_tie_weights_and_survives_raisers(self) -> None:
        """tie_weights is invoked where present; a raising one is non-fatal."""
        calls: list[str] = []

        class _Tie(nn.Module):
            def tie_weights(self) -> None:
                calls.append("tied")

        class _Raises(nn.Module):
            def tie_weights(self) -> None:
                raise RuntimeError("embed_tokens is not an nn.Module")

        root = nn.Module()
        root.good = _Tie()  # type: ignore[assignment]
        root.bad = _Raises()  # type: ignore[assignment]
        tie_transformers_weights(root)  # must not raise
        assert calls == ["tied"]
