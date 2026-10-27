"""ADR-0123: sub-16-bit weights need OpenRAL Pro; open runs fp32, fp16 and bf16.

Every case loads a real in-tree rSkill manifest (CLAUDE.md §1.11). The refusals
are only observable on an open-only install, so the module skips where the Pro
quantization module is installed (there the same loads must succeed, which the
quantized skills' own sim tests cover).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from openral_core import VLASpec
from openral_core.exceptions import ROSConfigError
from openral_rskill.loader import discover_intree_rskills, load_rskill_manifest
from openral_rskill.quantization_gate import (
    is_sub_16_bit,
    pro_quantization_installed,
    require_pro_for_checkpoint,
    require_pro_for_manifest,
    sidecar_quantization_args,
)
from openral_sim._quantization import QUANTIZATION_DTYPE_ENV, resolve_quant_plan

pytestmark = pytest.mark.skipif(
    pro_quantization_installed(), reason="OpenRAL Pro quantization is installed here"
)

_QUANTIZED = sorted(
    name
    for name, manifest in discover_intree_rskills()
    if is_sub_16_bit(getattr(manifest.quantization, "dtype", None))
)


def _spec(skill: str) -> VLASpec:
    manifest = load_rskill_manifest(f"rskills/{skill}")
    return VLASpec(
        id=manifest.model_family or "x",
        weights_uri=f"rskills/{skill}",
        device="auto",
        extra=dict(manifest.policy_extras),
    )


def test_the_twenty_quantized_skills_are_found() -> None:
    """The catalogue still lists the quantized skills; ADR-0123 keeps them in open."""
    assert len(_QUANTIZED) == 20, _QUANTIZED


@pytest.mark.parametrize("skill", _QUANTIZED)
def test_a_quantized_skill_refuses_to_load_without_pro(skill: str) -> None:
    """Each sub-16-bit skill raises a typed error naming OpenRAL Pro, never runs wider."""
    manifest = load_rskill_manifest(f"rskills/{skill}")
    with pytest.raises(ROSConfigError, match="OpenRAL Pro"):
        require_pro_for_manifest(manifest)
    with pytest.raises(ROSConfigError, match="OpenRAL Pro"):
        resolve_quant_plan(_spec(skill), manifest, default="nf4")


@pytest.mark.parametrize("skill", ["smolvla-libero", "act-aloha", "xvla-libero"])
def test_a_full_or_half_precision_skill_loads_in_open(skill: str) -> None:
    """bf16 / fp32 skills resolve exactly as before, with no Pro installed."""
    manifest = load_rskill_manifest(f"rskills/{skill}")
    require_pro_for_manifest(manifest)
    assert not resolve_quant_plan(_spec(skill), manifest).quantize


def test_the_env_override_cannot_quantize_an_open_skill(monkeypatch: pytest.MonkeyPatch) -> None:
    """``OPENRAL_QUANTIZATION_DTYPE=int8`` on a bf16 skill is refused, not honoured."""
    monkeypatch.setenv(QUANTIZATION_DTYPE_ENV, "int8")
    manifest = load_rskill_manifest("rskills/smolvla-libero")
    with pytest.raises(ROSConfigError, match="'int8'"):
        resolve_quant_plan(_spec("smolvla-libero"), manifest)


def test_an_upstream_quantized_checkpoint_is_refused(tmp_path: Path) -> None:
    """A checkpoint carrying a ``quantization_config`` needs Pro, whatever its manifest says."""
    # The block `tools/quantize_rskill.py` stamps on an NF4 pack, as transformers writes it.
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_vl",
                "quantization_config": {
                    "quant_method": "bitsandbytes",
                    "load_in_4bit": True,
                    "bnb_4bit_quant_type": "nf4",
                },
            }
        )
    )
    with pytest.raises(ROSConfigError, match="bitsandbytes"):
        require_pro_for_checkpoint(str(tmp_path), what="test load")


def test_an_unquantized_checkpoint_passes(tmp_path: Path) -> None:
    """A plain config, or no config at all, passes the checkpoint check."""
    require_pro_for_checkpoint(str(tmp_path), what="no config")
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen3_vl"}))
    require_pro_for_checkpoint(f"local://{tmp_path}", what="plain config")


@pytest.mark.parametrize("token", ["nf4", "int8", "prequantized_nf4"])
def test_a_sidecar_launcher_cannot_quantize_without_pro(token: str) -> None:
    """The launcher asks Pro for --quantization-config; without Pro it refuses up front."""
    with pytest.raises(ROSConfigError, match="OpenRAL Pro"):
        sidecar_quantization_args(token, what="the XR-1 sidecar")


@pytest.mark.parametrize("token", ["none", "bf16", "fp16", None])
def test_a_16_or_32_bit_sidecar_launch_needs_no_pro(token: str | None) -> None:
    """Open sidecars run fp16 / bf16 / fp32 with no config and no Pro."""
    assert sidecar_quantization_args(token, what="the XR-1 sidecar") == []


def test_a_manifest_declaring_bf16_over_a_quantized_checkpoint_is_refused(
    tmp_path: Path,
) -> None:
    """A manifest's declared dtype cannot vouch for the checkpoint it points at.

    The reward monitors, scene VLM and detector gate on ``require_pro_for_manifest``
    alone, so it must also read the selected checkpoint's own ``config.json``.
    """
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_vl",
                "quantization_config": {"quant_method": "bitsandbytes", "load_in_4bit": True},
            }
        )
    )
    manifest = load_rskill_manifest("rskills/topreward-qwen3vl-4b-nf4")
    assert manifest.quantization is not None
    manifest = manifest.model_copy(
        update={
            "weights_uri": f"local://{tmp_path}",
            "quantization": manifest.quantization.model_copy(update={"dtype": "bf16"}),
        }
    )
    with pytest.raises(ROSConfigError, match="bitsandbytes"):
        require_pro_for_manifest(manifest)


def test_the_sim_plan_checks_the_manifest_checkpoint_not_just_the_package(
    tmp_path: Path,
) -> None:
    """``spec.weights_uri`` names the rSkill package; the adapter loads the manifest's checkpoint.

    A bf16 package with no ``config.json`` of its own must not wave through a
    manifest ``weights_uri`` that is quantized upstream.
    """
    (tmp_path / "config.json").write_text(
        json.dumps({"quantization_config": {"quant_method": "bitsandbytes", "load_in_4bit": True}})
    )
    manifest = load_rskill_manifest("rskills/smolvla-libero").model_copy(
        update={"weights_uri": f"local://{tmp_path}"}
    )
    with pytest.raises(ROSConfigError, match="bitsandbytes"):
        resolve_quant_plan(_spec("smolvla-libero"), manifest)
