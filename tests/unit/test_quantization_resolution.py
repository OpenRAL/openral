"""Unified quantization resolution — one env var, one manifest home.

Every case loads a REAL in-tree rSkill manifest (CLAUDE.md §1.11): no mocks,
no synthetic manifests. The regression table is the point of the file — the
unification must not change which dtype any shipped skill loads at.
"""

from __future__ import annotations

import pytest
from openral_core import VLASpec
from openral_rskill.loader import load_rskill_manifest
from openral_sim._quantization import (
    QUANTIZATION_DTYPE_ENV,
    canonical_quant_token,
    quantization_extra,
    resolve_quant_plan,
)


def _spec(skill: str) -> VLASpec:
    """Build the VLASpec exactly as ``_load_or_build_env`` does."""
    manifest = load_rskill_manifest(f"rskills/{skill}")
    return VLASpec(
        id=manifest.model_family or "x",
        weights_uri=f"rskills/{skill}",
        device="auto",
        extra=dict(manifest.policy_extras),
    )


# (skill, manifest_dtype_is_storage, adapter default, dtype that must load)
#
# The expected column is what each family loaded BEFORE the unification, so a
# regression here means a shipped checkpoint silently changed precision.
_SHIPPED: list[tuple[str, bool, str | None, str]] = [
    # In-process families: the manifest pins the runtime dtype.
    ("pi05-libero-int8", False, None, "int8"),
    ("molmoact2-libero-nf4", False, None, "nf4"),
    ("openvla-oft-simpler-widowx-nf4", False, None, "nf4"),
    # GR00T publishes bf16 weights and NF4-packs them at load, so the declared
    # dtype is storage and the adapter default decides.
    ("gr00t-n17-libero", True, "nf4", "nf4"),
    ("gr00t-n17-so101-fruit", True, "nf4", "nf4"),
    ("gr00t-n17-b1k-turning-on-radio", True, "nf4", "nf4"),
    # RLDX declares int4, which normalises onto the sidecar's `nf4` token.
    ("rldx1-ft-libero-nf4", True, "nf4", "nf4"),
    ("rldx1-ft-gr1-nf4", True, "nf4", "nf4"),
]


@pytest.mark.parametrize(("skill", "storage", "default", "expected"), _SHIPPED)
def test_shipped_skill_dtype_is_unchanged(
    skill: str, storage: bool, default: str | None, expected: str
) -> None:
    """Each in-tree skill still resolves to the dtype it loaded at before."""
    manifest = load_rskill_manifest(f"rskills/{skill}")
    plan = resolve_quant_plan(
        _spec(skill), manifest, default=default, manifest_dtype_is_storage=storage
    )
    assert plan.dtype == expected
    assert plan.quantize is (expected in {"nf4", "int8"})


def test_env_override_reaches_every_family() -> None:
    """One env var turns packing off for an in-process and a sidecar family."""
    import os

    os.environ[QUANTIZATION_DTYPE_ENV] = "bf16"
    try:
        in_process = resolve_quant_plan(
            _spec("pi05-libero-int8"), load_rskill_manifest("rskills/pi05-libero-int8")
        )
        sidecar = resolve_quant_plan(
            _spec("gr00t-n17-libero"),
            load_rskill_manifest("rskills/gr00t-n17-libero"),
            default="nf4",
            manifest_dtype_is_storage=True,
        )
    finally:
        del os.environ[QUANTIZATION_DTYPE_ENV]

    for plan in (in_process, sidecar):
        assert plan.dtype == "bf16"
        assert plan.quantize is False
        assert plan.source == "env"


def test_env_beats_spec_extra_beats_manifest() -> None:
    """Precedence is env > spec.extra > manifest > adapter default."""
    import os

    manifest = load_rskill_manifest("rskills/pi05-libero-int8")
    spec_bf16 = VLASpec(
        id="pi05", weights_uri="rskills/pi05-libero-int8", device="auto", extra={"dtype": "bf16"}
    )
    assert resolve_quant_plan(spec_bf16, manifest).source == "spec_extra"

    os.environ[QUANTIZATION_DTYPE_ENV] = "fp32"
    try:
        plan = resolve_quant_plan(spec_bf16, manifest)
    finally:
        del os.environ[QUANTIZATION_DTYPE_ENV]
    assert (plan.dtype, plan.source) == ("fp32", "env")

    assert resolve_quant_plan(_spec("pi05-libero-int8"), manifest).source == "manifest"
    bare = VLASpec(id="x", weights_uri="r", device="auto", extra={})
    assert resolve_quant_plan(bare, None, default="nf4").source == "default"

    unset = resolve_quant_plan(bare, None)
    assert (unset.dtype, unset.quantize, unset.source) == (None, False, "unset")


def test_int4_normalises_to_the_token_sidecars_accept() -> None:
    """`tools/behavior_groot_sidecar.py` argparse only accepts none/nf4/int8."""
    manifest = load_rskill_manifest("rskills/rldx1-ft-libero-nf4")
    assert manifest.quantization.dtype.value == "int4"
    assert resolve_quant_plan(_spec("rldx1-ft-libero-nf4"), manifest).dtype == "nf4"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("int4", "nf4"),
        ("4bit", "nf4"),
        ("NF4", "nf4"),
        ("BF16", "bf16"),
        ("none", "none"),
        ("", "none"),
    ],
)
def test_canonical_tokens(raw: str, expected: str) -> None:
    assert canonical_quant_token(raw) == expected


def test_unset_is_distinct_from_none() -> None:
    """`None` means nobody declared one; `"none"` means do not quantize."""
    assert canonical_quant_token(None) is None
    assert canonical_quant_token("none") == "none"


def test_packing_knobs_live_on_the_manifest() -> None:
    """quantize_scope / nf4_min_params read from quantization.extra."""
    fruit = load_rskill_manifest("rskills/gr00t-n17-so101-fruit")
    assert quantization_extra(fruit)["quantize_scope"] == "model"

    b1k = load_rskill_manifest("rskills/gr00t-n17-b1k-turning-on-radio")
    assert quantization_extra(b1k) == {"quantize_scope": "model", "nf4_min_params": 1000000}

    # A skill that declares none gets an empty map, not an error.
    assert quantization_extra(load_rskill_manifest("rskills/gr00t-n17-libero")) == {}
    assert quantization_extra(None) == {}
