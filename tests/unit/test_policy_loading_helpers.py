"""Unit tests for ``openral_sim.policies._policy_loading`` shared helpers.

These exercise the manifest-resolution helper that replaced the
per-adapter ``_load_manifest_for_spec`` copies in pi05 / rldx / smolvla,
plus the dtype-resolution helpers that moved out of ``pi05.py`` into
``openral_sim._quantization``. Per CLAUDE.md §1.11 / §5.4 every fixture
is a real :class:`openral_core.RSkillManifest` loaded from the canonical
on-disk YAMLs under ``rskills/``; nothing is mocked.

The dtype tests stand in for the old in-pi05 unit coverage that used
to live alongside ``_manifest_dtype`` / ``_torch_dtype_for`` /
``_default_dtype``. ``torch`` is imported via :func:`pytest.importorskip`
so a bare ``openral-sim`` install without the ``sim`` group still
collects (the helpers themselves are torch-free; only this test's
assertions need a real ``torch.dtype`` to compare against).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from openral_core import VLASpec
from openral_rskill.loader import load_rskill_manifest
from openral_sim._quantization import (
    default_dtype_for_device,
    manifest_dtype,
    normalise_manifest_dtype,
    torch_dtype_for,
)
from openral_sim.policies._policy_loading import load_manifest_for_spec

_REPO_ROOT = Path(__file__).parent.parent.parent
_RSKILLS_DIR = _REPO_ROOT / "rskills"

# Canonical fixtures: each has a `quantization.dtype` so the dtype tests
# can assert against the real on-disk enum value. The pi05 / rldx
# manifests use ``int4``; smolvla-libero ships unquantized ``bf16``. These
# are NOT toy manifests — they are the manifests ``openral sim run`` consumes.
_PI05_LIBERO = _RSKILLS_DIR / "pi05-libero-nf4"  # quantization.dtype: int4
_BF16_MANIFEST = _RSKILLS_DIR / "smolvla-libero"  # quantization.dtype: bf16
_RLDX_LIBERO = _RSKILLS_DIR / "rldx1-ft-libero-nf4"  # quantization.dtype: int4


# ── load_manifest_for_spec ────────────────────────────────────────────────────


class TestLoadManifestForSpec:
    """``load_manifest_for_spec`` replaces the per-adapter copies."""

    def test_bare_path_resolves_to_real_manifest(self) -> None:
        spec = VLASpec(id="pi05", weights_uri=str(_PI05_LIBERO))
        manifest = load_manifest_for_spec(spec)
        assert manifest is not None
        # The on-disk pi05-libero-nf4 manifest pins these — assert against
        # them to catch a silent regression in the path-handling layer.
        assert manifest.model_family == "pi05"
        assert manifest.quantization is not None
        assert manifest.quantization.dtype.value == "int4"

    def test_hf_uri_returns_none(self) -> None:
        """Bare ``hf://`` URIs must yield ``None`` so adapters can fall through."""
        spec = VLASpec(id="pi05", weights_uri="hf://lerobot/smolvla_base")
        assert load_manifest_for_spec(spec) is None

    def test_empty_weights_uri_returns_none(self) -> None:
        """A spec without a weights URI is tolerated; adapters decide whether to raise."""
        spec = VLASpec(id="pi05", weights_uri="")
        assert load_manifest_for_spec(spec) is None

    def test_works_on_any_object_with_weights_uri(self) -> None:
        """Duck-typing: anything with a ``weights_uri`` attribute is fair game.

        The adapters historically passed a ``VLASpec``; the helper now
        also lands on raw dataclasses produced by the eval-layer YAML
        loader. We pin that affordance so a future refactor doesn't
        regress it silently.
        """

        class _SpecLike:
            weights_uri = str(_RLDX_LIBERO)

        manifest = load_manifest_for_spec(_SpecLike())
        assert manifest is not None
        assert manifest.model_family == "rldx"


# ── Manifest-driven dtype resolution ──────────────────────────────────────────


class TestNormaliseManifestDtype:
    """Pull ``manifest.quantization.dtype`` as a plain string."""

    def test_int4_manifest(self) -> None:
        manifest = load_rskill_manifest(str(_PI05_LIBERO))
        # ``QuantizationDtype.INT4`` enum value should land as ``"int4"``.
        assert normalise_manifest_dtype(manifest) == "int4"

    def test_bf16_manifest(self) -> None:
        manifest = load_rskill_manifest(str(_BF16_MANIFEST))
        assert normalise_manifest_dtype(manifest) == "bf16"

    def test_none_for_missing_quantization(self) -> None:
        """A manifest-shaped object without ``quantization`` lands as ``None``."""

        class _NoQuant:
            quantization = None

        assert normalise_manifest_dtype(_NoQuant()) is None


class TestManifestDtype:
    """The top-level resolver that adapters call."""

    def test_spec_extra_overrides_manifest(self) -> None:
        """Per-run override beats the manifest's pinned dtype."""
        manifest = load_rskill_manifest(str(_PI05_LIBERO))  # pinned int4
        spec = VLASpec(
            id="pi05",
            weights_uri=str(_PI05_LIBERO),
            extra={"dtype": "bf16"},
        )
        assert manifest_dtype(spec, manifest=manifest) == "bf16"

    def test_manifest_used_when_no_extra(self) -> None:
        manifest = load_rskill_manifest(str(_PI05_LIBERO))
        spec = VLASpec(id="pi05", weights_uri=str(_PI05_LIBERO))
        assert manifest_dtype(spec, manifest=manifest) == "int4"

    def test_none_when_neither_source_has_dtype(self) -> None:
        spec = VLASpec(id="pi05", weights_uri="")
        assert manifest_dtype(spec, manifest=None) is None

    def test_rldx_libero_resolves_to_int4(self) -> None:
        """Cross-adapter sanity: rldx1-ft-libero-nf4 pins int4 too."""
        manifest = load_rskill_manifest(str(_RLDX_LIBERO))
        spec = VLASpec(id="rldx", weights_uri=str(_RLDX_LIBERO))
        assert manifest_dtype(spec, manifest=manifest) == "int4"


# ── torch dtype mapping ───────────────────────────────────────────────────────


class TestTorchDtypeFor:
    """Map a manifest dtype string to a real ``torch.dtype``."""

    @pytest.fixture(scope="class")
    def torch_mod(self) -> object:
        return pytest.importorskip("torch")

    def test_bf16_strings_map_to_bfloat16(self, torch_mod: object) -> None:
        for s in ("bf16", "bfloat16", "BF16"):
            assert torch_dtype_for(torch_mod, s, "cpu") is torch_mod.bfloat16  # type: ignore[attr-defined]

    def test_fp16_strings_map_to_float16(self, torch_mod: object) -> None:
        for s in ("fp16", "float16", "half"):
            assert torch_dtype_for(torch_mod, s, "cpu") is torch_mod.float16  # type: ignore[attr-defined]

    def test_fp32_strings_map_to_float32(self, torch_mod: object) -> None:
        for s in ("fp32", "float32"):
            assert torch_dtype_for(torch_mod, s, "cpu") is torch_mod.float32  # type: ignore[attr-defined]

    def test_none_picks_cuda_aware_default(self, torch_mod: object) -> None:
        """Unknown dtype + ``cuda`` device → bf16; cpu → fp32."""
        assert torch_dtype_for(torch_mod, None, "cuda") is torch_mod.bfloat16  # type: ignore[attr-defined]
        assert torch_dtype_for(torch_mod, None, "cpu") is torch_mod.float32  # type: ignore[attr-defined]

    def test_unrecognised_dtype_falls_through_to_default(self, torch_mod: object) -> None:
        """``"nf4"`` / ``"int8"`` are quantization schemes, not torch dtypes.

        They must fall through to the device-aware default so the
        adapter's *compute* dtype lands correctly even when the headline
        dtype is one bnb consumes directly.
        """
        assert torch_dtype_for(torch_mod, "nf4", "cuda") is torch_mod.bfloat16  # type: ignore[attr-defined]
        assert torch_dtype_for(torch_mod, "int8", "cpu") is torch_mod.float32  # type: ignore[attr-defined]


# ── Default dtype per device ──────────────────────────────────────────────────


class TestDefaultDtypeForDevice:
    def test_cuda_defaults_to_nf4(self) -> None:
        assert default_dtype_for_device("cuda") == "nf4"
        assert default_dtype_for_device("cuda:0") == "nf4"

    def test_cpu_defaults_to_fp32(self) -> None:
        assert default_dtype_for_device("cpu") == "fp32"

    def test_mps_falls_back_to_fp32(self) -> None:
        """Anything that doesn't start with ``"cuda"`` should pick fp32."""
        assert default_dtype_for_device("mps") == "fp32"
