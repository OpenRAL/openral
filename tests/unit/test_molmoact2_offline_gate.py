"""Tests for the MolmoAct2 ``_hf_cached_first`` cache-first loader.

Regression for two cold-cache bugs in the old proxy-file offline gate, which
flipped ``HF_HUB_OFFLINE`` on when one probe file was cached:

- ``predict_action`` was gated on ``config.json`` but fetched
  ``norm_stats.json``; the model load warms the former, never the latter, so
  the first run failed with "normalization stats file is missing".
- the weights ``from_pretrained`` was gated on ``config.json``, which the
  config load just before it warms, so a cold-cache bf16 load (Jetson Thor,
  2026-11-08) could not download its shards: "does not appear to have a file
  named pytorch_model.bin or model.safetensors".

``_hf_cached_first`` tries the load itself offline and retries online on a
miss, so no probe file can disagree with what the load reads.
"""

from __future__ import annotations

from pathlib import Path

import huggingface_hub
import huggingface_hub.constants as hc
import pytest
from openral_core import VLASpec
from openral_core.exceptions import ROSConfigError
from openral_rskill import split_repo_revision
from openral_rskill.loader import rSkill
from openral_sim.policies.molmoact2 import (
    _hf_cached_first,
    _norm_stats_path,
    _resolve_max_crops,
)


class TestSplitRepoRevision:
    """A ``@<sha>`` pin must be split into the ``revision`` kwarg, not left on
    the repo id where HF silently drops it (security audit 2026-06, H5)."""

    def test_no_pin_returns_none_revision(self) -> None:
        assert split_repo_revision("allenai/MolmoAct2-LIBERO") == (
            "allenai/MolmoAct2-LIBERO",
            None,
        )

    def test_pinned_sha_is_split_out(self) -> None:
        assert split_repo_revision("allenai/MolmoAct2-LIBERO@abc123") == (
            "allenai/MolmoAct2-LIBERO",
            "abc123",
        )

    def test_pinned_branch_is_split_out(self) -> None:
        assert split_repo_revision("allenai/MolmoAct2-LIBERO@main") == (
            "allenai/MolmoAct2-LIBERO",
            "main",
        )

    def test_local_path_is_never_split(self, tmp_path: Path) -> None:
        """A local checkpoint dir may contain ``@``; it is not a repo@rev pin."""
        ckpt = tmp_path / "rldx@v2"
        ckpt.mkdir()
        assert split_repo_revision(str(ckpt)) == (str(ckpt), None)

    def test_trailing_at_yields_none_revision(self) -> None:
        assert split_repo_revision("allenai/MolmoAct2-LIBERO@") == (
            "allenai/MolmoAct2-LIBERO",
            None,
        )


class TestHfCachedFirst:
    def test_warm_cache_loads_once_offline(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A load that the cache satisfies runs once, offline, and the flag is restored."""
        monkeypatch.setattr(hc, "HF_HUB_OFFLINE", False)
        seen: list[bool] = []

        def load() -> str:
            seen.append(hc.HF_HUB_OFFLINE)
            return "loaded"

        assert _hf_cached_first(load, repo_id="allenai/MolmoAct2-LIBERO") == "loaded"
        assert seen == [True]
        assert hc.HF_HUB_OFFLINE is False

    def test_cache_miss_retries_online(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An offline miss (``OSError``) reruns the load with the Hub reachable."""
        monkeypatch.setattr(hc, "HF_HUB_OFFLINE", False)
        seen: list[bool] = []

        def load() -> str:
            seen.append(hc.HF_HUB_OFFLINE)
            if hc.HF_HUB_OFFLINE:
                raise huggingface_hub.errors.LocalEntryNotFoundError("not cached")
            return "downloaded"

        assert _hf_cached_first(load, repo_id="allenai/MolmoAct2-LIBERO") == "downloaded"
        assert seen == [True, False]
        assert hc.HF_HUB_OFFLINE is False

    def test_online_failure_propagates(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A load that fails online too raises the online error, flag restored."""
        monkeypatch.setattr(hc, "HF_HUB_OFFLINE", False)

        def load() -> str:
            raise OSError("offline" if hc.HF_HUB_OFFLINE else "online")

        with pytest.raises(OSError, match="online"):
            _hf_cached_first(load, repo_id="allenai/MolmoAct2-LIBERO")
        assert hc.HF_HUB_OFFLINE is False

    def test_non_cache_error_is_not_retried(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Only a missing-file ``OSError`` falls back; other errors surface at once."""
        monkeypatch.setattr(hc, "HF_HUB_OFFLINE", False)
        calls: list[bool] = []

        def load() -> str:
            calls.append(hc.HF_HUB_OFFLINE)
            raise ValueError("bad config")

        with pytest.raises(ValueError, match="bad config"):
            _hf_cached_first(load, repo_id="allenai/MolmoAct2-LIBERO")
        assert calls == [True]
        assert hc.HF_HUB_OFFLINE is False

    def test_config_warm_weights_cold_from_pretrained_downloads(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The Thor repro on a real (tiny) checkpoint: config cached, weights not.

        A plain offline ``from_pretrained`` raises the exact "does not appear to
        have a file named ... model.safetensors" error; the cache-first load
        must instead download the weights and return the model.
        """
        transformers = pytest.importorskip("transformers")
        monkeypatch.setattr(hc, "HF_HUB_OFFLINE", False)
        repo = "hf-internal-testing/tiny-random-bert"
        try:
            transformers.AutoConfig.from_pretrained(repo, cache_dir=tmp_path)
        except OSError as exc:
            pytest.skip(f"Hub unreachable: {exc}")
        monkeypatch.setattr(hc, "HF_HUB_OFFLINE", True)
        with pytest.raises(OSError, match=r"model\.safetensors"):
            transformers.AutoModel.from_pretrained(repo, cache_dir=tmp_path)
        monkeypatch.setattr(hc, "HF_HUB_OFFLINE", False)
        model = _hf_cached_first(
            lambda: transformers.AutoModel.from_pretrained(repo, cache_dir=tmp_path),
            repo_id=repo,
        )
        assert model.config.model_type == "bert"
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


@pytest.mark.parametrize("skill", ["molmoact2-libero-nf4", "molmoact2-so101-nf4"])
def test_norm_stats_resolve_at_the_pinned_source_revision(skill: str) -> None:
    """``_get_robot_stats`` reads ``config._name_or_path``; it gets the pinned snapshot dir.

    A bare repo id there makes MolmoAct2 ``hf_hub_download`` the stats at HEAD.
    """
    pkg = rSkill.from_yaml(f"rskills/{skill}/rskill.yaml")
    assert pkg.manifest.source_repo is not None
    repo, revision = split_repo_revision(pkg.manifest.source_repo.removeprefix("hf://"))
    assert revision is not None and len(revision) == 40
    try:
        path = _norm_stats_path(repo, revision)
    except (OSError, huggingface_hub.errors.HfHubHTTPError) as exc:
        pytest.skip(f"Hub unreachable and norm_stats.json not cached: {exc}")
    assert path.endswith(f"/snapshots/{revision}/norm_stats.json")
