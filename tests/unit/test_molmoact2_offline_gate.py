"""Tests for the MolmoAct2 ``_hf_offline_if_cached`` offline gate.

Regression for the cold-``norm_stats.json`` first-run bug: the gate probed
``config.json`` as a proxy for "everything is cached" and flipped
``HF_HUB_OFFLINE`` on, but ``predict_action`` fetches ``norm_stats.json``
*inside* that block. On a first run the model load warms ``config.json`` but
never ``norm_stats.json``, so the offline flag blocked the lazy norm-stats
download with a ``LocalEntryNotFoundError`` surfaced as "normalization stats
file is missing". The gate must probe the file the inner block actually reads.
"""

from __future__ import annotations

import huggingface_hub
import huggingface_hub.constants as hc
import pytest
from openral_core import VLASpec
from openral_core.exceptions import ROSConfigError
from openral_rskill.loader import rSkill
from openral_sim.policies.molmoact2 import (
    _enable_expandable_segments,
    _hf_offline_if_cached,
    _require_remote_code_ack,
    _resolve_max_crops,
    _split_repo_revision,
)


class TestEnableExpandableSegments:
    """MolmoAct2 NF4 is ~6 GiB resident and peaks ~7.63 GiB; on an 8 GiB card the
    first forward's ~1.5 GiB embedding cat OOMs without the CUDA expandable-
    segments allocator. The adapter enables it before the first CUDA allocation.
    Verified on an RTX 4070: OOM without, peak 7.63 GiB fit with (see PR)."""

    _VAR = "PYTORCH_CUDA_ALLOC_CONF"

    def test_sets_when_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(self._VAR, raising=False)
        _enable_expandable_segments()
        import os

        assert os.environ[self._VAR] == "expandable_segments:True"

    def test_noop_when_already_expandable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(self._VAR, "expandable_segments:True,max_split_size_mb:128")
        _enable_expandable_segments()
        import os

        # left exactly as the operator set it
        assert os.environ[self._VAR] == "expandable_segments:True,max_split_size_mb:128"

    def test_respects_unrelated_preset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An operator's existing conf without expandable_segments is NOT overwritten."""
        monkeypatch.setenv(self._VAR, "max_split_size_mb:64")
        _enable_expandable_segments()
        import os

        assert os.environ[self._VAR] == "max_split_size_mb:64"


class TestRequireRemoteCodeAck:
    """trust_remote_code executes custom code from the repo; it must be refused
    unless the operator explicitly acknowledges it (security audit 2026-06, C3)."""

    def test_refused_without_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OPENRAL_ALLOW_REMOTE_CODE", raising=False)
        with pytest.raises(ROSConfigError, match="remote-code-execution"):
            _require_remote_code_ack("allenai/MolmoAct2-LIBERO", "abc123")

    def test_allowed_with_env_pinned(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENRAL_ALLOW_REMOTE_CODE", "1")
        _require_remote_code_ack("allenai/MolmoAct2-LIBERO", "abc123")  # must not raise

    def test_allowed_with_env_unpinned(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENRAL_ALLOW_REMOTE_CODE", "1")
        _require_remote_code_ack("allenai/MolmoAct2-LIBERO", None)  # must not raise

    def test_non_one_value_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENRAL_ALLOW_REMOTE_CODE", "true")
        with pytest.raises(ROSConfigError):
            _require_remote_code_ack("allenai/MolmoAct2-LIBERO", "abc123")


class TestSplitRepoRevision:
    """A ``@<sha>`` pin must be split into the ``revision`` kwarg, not left on
    the repo id where HF silently drops it (security audit 2026-06, H5)."""

    def test_no_pin_returns_none_revision(self) -> None:
        assert _split_repo_revision("allenai/MolmoAct2-LIBERO") == (
            "allenai/MolmoAct2-LIBERO",
            None,
        )

    def test_pinned_sha_is_split_out(self) -> None:
        assert _split_repo_revision("allenai/MolmoAct2-LIBERO@abc123") == (
            "allenai/MolmoAct2-LIBERO",
            "abc123",
        )

    def test_pinned_branch_is_split_out(self) -> None:
        assert _split_repo_revision("allenai/MolmoAct2-LIBERO@main") == (
            "allenai/MolmoAct2-LIBERO",
            "main",
        )

    def test_trailing_at_yields_none_revision(self) -> None:
        assert _split_repo_revision("allenai/MolmoAct2-LIBERO@") == (
            "allenai/MolmoAct2-LIBERO",
            None,
        )


class TestHfOfflineIfCached:
    def test_offline_flipped_when_probe_file_cached(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A cached probe file → HF_HUB_OFFLINE True inside, restored on exit."""
        monkeypatch.setattr(
            huggingface_hub, "try_to_load_from_cache", lambda _repo, _f: "/cache/config.json"
        )
        monkeypatch.setattr(hc, "HF_HUB_OFFLINE", False)
        with _hf_offline_if_cached("allenai/MolmoAct2-LIBERO"):
            assert hc.HF_HUB_OFFLINE is True
        assert hc.HF_HUB_OFFLINE is False

    def test_stays_online_when_probe_file_absent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An uncached probe file → stays online so the inner block can download."""
        monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", lambda _repo, _f: None)
        monkeypatch.setattr(hc, "HF_HUB_OFFLINE", False)
        with _hf_offline_if_cached("allenai/MolmoAct2-LIBERO"):
            assert hc.HF_HUB_OFFLINE is False

    def test_probe_file_gates_on_the_requested_file(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The bug: config.json cached but norm_stats.json not → must stay online.

        Probing ``norm_stats.json`` (what predict_action fetches) returns None
        even though ``config.json`` is warm, so the gate keeps the inner block
        online to download the norm stats instead of failing offline.
        """
        cache = {"config.json": "/cache/config.json"}
        monkeypatch.setattr(
            huggingface_hub,
            "try_to_load_from_cache",
            lambda _repo, filename: cache.get(filename),
        )
        monkeypatch.setattr(hc, "HF_HUB_OFFLINE", False)
        # config.json gate would flip offline...
        with _hf_offline_if_cached("allenai/MolmoAct2-LIBERO", probe_file="config.json"):
            assert hc.HF_HUB_OFFLINE is True
        # ...but the norm_stats.json gate (predict path) must stay online.
        with _hf_offline_if_cached("allenai/MolmoAct2-LIBERO", probe_file="norm_stats.json"):
            assert hc.HF_HUB_OFFLINE is False


class TestResolveMaxCrops:
    """``_resolve_max_crops`` precedence: vla.extra → env → manifest → None.

    The NF4 weights are a fixed ~3.5 GiB; what overflows an 8 GiB card on the
    SO-101 checkpoint is the image processor's multi-crop activations (default
    8 crops). The SO-101 rSkill pins ``image_max_crops: 4`` in its manifest so
    a fresh rollout fits without any ``vla.extra`` override — this guards that
    the manifest pin actually reaches the loader.
    """

    def _spec(self, weights_uri: str, **extra: object) -> VLASpec:
        return VLASpec(id="molmoact2", weights_uri=weights_uri, extra=dict(extra))

    def test_so101_manifest_pins_four(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The real SO-101 fixture ships image_max_crops=4 and the loader honors it."""
        monkeypatch.delenv("OPENRAL_MOLMOACT2_MAX_CROPS", raising=False)
        pkg = rSkill.from_yaml("rskills/molmoact2-so101-nf4/rskill.yaml")
        assert pkg.manifest.image_preprocessing is not None
        assert pkg.manifest.image_preprocessing.image_max_crops == 4
        spec = self._spec("rskills/molmoact2-so101-nf4")
        assert _resolve_max_crops(spec, pkg.manifest) == 4

    def test_libero_manifest_keeps_checkpoint_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The LIBERO fixture does NOT pin crops (verified at the default 8) → None."""
        monkeypatch.delenv("OPENRAL_MOLMOACT2_MAX_CROPS", raising=False)
        pkg = rSkill.from_yaml("rskills/molmoact2-libero-nf4/rskill.yaml")
        spec = self._spec("rskills/molmoact2-libero-nf4")
        assert _resolve_max_crops(spec, pkg.manifest) is None

    def test_vla_extra_overrides_manifest(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An explicit ``vla.extra.image_max_crops`` wins over the manifest pin."""
        monkeypatch.delenv("OPENRAL_MOLMOACT2_MAX_CROPS", raising=False)
        pkg = rSkill.from_yaml("rskills/molmoact2-so101-nf4/rskill.yaml")
        spec = self._spec("rskills/molmoact2-so101-nf4", image_max_crops=8)
        assert _resolve_max_crops(spec, pkg.manifest) == 8

    def test_env_overrides_manifest(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The env knob wins over the manifest pin (but loses to vla.extra)."""
        monkeypatch.setenv("OPENRAL_MOLMOACT2_MAX_CROPS", "2")
        pkg = rSkill.from_yaml("rskills/molmoact2-so101-nf4/rskill.yaml")
        spec = self._spec("rskills/molmoact2-so101-nf4")
        assert _resolve_max_crops(spec, pkg.manifest) == 2

    def test_invalid_crops_raises(self) -> None:
        """A non-positive crop count is rejected with a typed ROSConfigError."""
        spec = self._spec("rskills/molmoact2-so101-nf4", image_max_crops=0)
        with pytest.raises(ROSConfigError, match="image_max_crops must be >= 1"):
            _resolve_max_crops(spec, None)

    def test_none_when_no_pin_and_no_manifest(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """No extra, no env, no manifest → None (adapter keeps the checkpoint default)."""
        monkeypatch.delenv("OPENRAL_MOLMOACT2_MAX_CROPS", raising=False)
        spec = self._spec("rskills/molmoact2-so101-nf4")
        assert _resolve_max_crops(spec, None) is None
