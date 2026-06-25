"""Tests for `openral_cli.main._parse_rskill_cli_arg`.

Verifies that the CLI accepts bare rSkill references and stores them
unchanged on `VLASpec.weights_uri`. Uses the in-tree
``rskills/diffusion-pusht`` manifest as the resolution target — no network,
no GPU.
"""

from __future__ import annotations

import pytest
import typer
from openral_cli.main import _parse_rskill_cli_arg


def test_bare_name_resolves() -> None:
    """`--rskill diffusion-pusht` resolves to the manifest and stores the bare name."""
    spec = _parse_rskill_cli_arg("diffusion-pusht")
    assert spec.weights_uri == "diffusion-pusht"
    assert spec.id  # model_family from manifest


def test_bare_path_resolves() -> None:
    spec = _parse_rskill_cli_arg("rskills/diffusion-pusht")
    assert spec.weights_uri == "rskills/diffusion-pusht"


def test_hf_scheme_rejected() -> None:
    """Raw ``hf://`` is never accepted — weights must come from an rSkill manifest."""
    with pytest.raises(typer.BadParameter, match="hf://"):
        _parse_rskill_cli_arg("hf://openral/diffusion-pusht")


def test_local_scheme_rejected() -> None:
    """``local://`` is rejected — pass a bare path instead."""
    with pytest.raises(typer.BadParameter, match="local://"):
        _parse_rskill_cli_arg("local://rskills/diffusion-pusht")


def test_empty_rejected() -> None:
    with pytest.raises(typer.BadParameter):
        _parse_rskill_cli_arg("")


def test_non_vla_rskill_rejected() -> None:
    """A detector/reward skill has no `model_family`, so it cannot be a VLASpec."""
    with pytest.raises(typer.BadParameter, match="no model_family"):
        _parse_rskill_cli_arg("rskills/omdet-turbo-indoor")
