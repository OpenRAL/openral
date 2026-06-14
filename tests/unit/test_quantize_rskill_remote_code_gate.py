"""Tests for the quantize_rskill ``--trust-remote-code`` org allowlist (C3).

``--trust-remote-code`` executes custom code from the source repo (RCE). The
quantize tool hard-blocks repos outside the trusted-org allowlist unless an
operator acknowledges the risk (security audit 2026-06, C3 follow-up).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

# Load tools/quantize_rskill.py as a module (tools/ is not an installed package).
_spec = importlib.util.spec_from_file_location(
    "quantize_rskill", REPO_ROOT / "tools" / "quantize_rskill.py"
)
assert _spec is not None and _spec.loader is not None
quantize_rskill = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = quantize_rskill
_spec.loader.exec_module(quantize_rskill)

_require = quantize_rskill._require_trusted_remote_code


class TestStampQuantizationConfigLoaderGate:
    """The bnb ``quantization_config`` stamp must only touch transformers configs.

    ``_copy_source_config`` brings a lerobot policy's ``config.json`` into the
    staging dir, but lerobot config classes (e.g. ``PI05Config``) raise
    ``DecodingError: fields quantization_config are not valid`` on the unknown
    field — so quantizing a lerobot policy (pi05 / smolvla) and stamping its
    config produces a bundle the rSkill loader cannot open. The lerobot runtime
    keys off ``quantization_metadata.json`` instead. Regression for that.
    """

    def _config(self, tmp_path: Path) -> Path:
        cfg = tmp_path / "config.json"
        cfg.write_text(json.dumps({"type": "pi05", "n_obs_steps": 1}))
        return cfg

    def test_lerobot_loader_does_not_stamp(self, tmp_path: Path) -> None:
        cfg = self._config(tmp_path)
        quantize_rskill._stamp_quantization_config(tmp_path, scheme="nf4", loader="lerobot")
        assert "quantization_config" not in json.loads(cfg.read_text())

    def test_transformers_loader_stamps(self, tmp_path: Path) -> None:
        cfg = self._config(tmp_path)
        quantize_rskill._stamp_quantization_config(tmp_path, scheme="nf4", loader="transformers")
        assert "quantization_config" in json.loads(cfg.read_text())


class TestRequireTrustedRemoteCode:
    def test_default_trusted_org_passes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OPENRAL_TRUSTED_REMOTE_CODE_ORGS", raising=False)
        monkeypatch.delenv("OPENRAL_ALLOW_REMOTE_CODE", raising=False)
        _require("OpenRAL/molmoact2-quantized")  # must not raise (case-insensitive)

    def test_untrusted_org_blocked(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OPENRAL_TRUSTED_REMOTE_CODE_ORGS", raising=False)
        monkeypatch.delenv("OPENRAL_ALLOW_REMOTE_CODE", raising=False)
        with pytest.raises(ValueError, match="remote-code-execution"):
            _require("allenai/MolmoAct2-LIBERO")

    def test_env_allowlist_extends_trust(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENRAL_TRUSTED_REMOTE_CODE_ORGS", "allenai, someorg")
        monkeypatch.delenv("OPENRAL_ALLOW_REMOTE_CODE", raising=False)
        _require("allenai/MolmoAct2-LIBERO")  # now trusted

    def test_override_env_allows_with_warning(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.delenv("OPENRAL_TRUSTED_REMOTE_CODE_ORGS", raising=False)
        monkeypatch.setenv("OPENRAL_ALLOW_REMOTE_CODE", "1")
        _require("allenai/MolmoAct2-LIBERO")  # must not raise
        assert "WARNING" in capsys.readouterr().out

    def test_hf_prefix_and_revision_stripped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OPENRAL_TRUSTED_REMOTE_CODE_ORGS", raising=False)
        monkeypatch.delenv("OPENRAL_ALLOW_REMOTE_CODE", raising=False)
        with pytest.raises(ValueError, match="allenai"):
            _require("hf://allenai/MolmoAct2-LIBERO@abc123")

    def test_local_path_exempt(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OPENRAL_TRUSTED_REMOTE_CODE_ORGS", raising=False)
        monkeypatch.delenv("OPENRAL_ALLOW_REMOTE_CODE", raising=False)
        _require(str(tmp_path))  # local checkpoint — exempt, must not raise


class TestStampQuantizationConfig:
    """The uploaded mirror's config.json must carry a bnb quantization_config so
    the Hub auto-tags it (regression: the SO-101 nf4 repo shipped with only a
    stray ``8-bit`` tag and no ``nf4`` / ``4-bit`` because the tool copied the
    bf16 source config.json verbatim and never stamped the scheme)."""

    def test_nf4_config_shape(self) -> None:
        cfg = quantize_rskill._bnb_quantization_config("nf4")
        assert cfg["quant_method"] == "bitsandbytes"
        assert cfg["load_in_4bit"] is True
        assert cfg["load_in_8bit"] is False
        assert cfg["bnb_4bit_quant_type"] == "nf4"
        assert cfg["bnb_4bit_compute_dtype"] == "bfloat16"
        assert cfg["bnb_4bit_use_double_quant"] is True

    def test_int8_config_shape(self) -> None:
        cfg = quantize_rskill._bnb_quantization_config("int8")
        assert cfg["quant_method"] == "bitsandbytes"
        assert cfg["load_in_8bit"] is True
        assert cfg["load_in_4bit"] is False

    def test_unknown_scheme_raises(self) -> None:
        with pytest.raises(NotImplementedError):
            quantize_rskill._bnb_quantization_config("gptq")

    def test_stamp_injects_into_config(self, tmp_path: Path) -> None:
        (tmp_path / "config.json").write_text(json.dumps({"model_type": "molmoact2"}))
        quantize_rskill._stamp_quantization_config(tmp_path, scheme="nf4", loader="transformers")
        config = json.loads((tmp_path / "config.json").read_text())
        # Preserves the existing keys and adds the bnb block.
        assert config["model_type"] == "molmoact2"
        assert config["quantization_config"]["bnb_4bit_quant_type"] == "nf4"

    def test_stamp_overwrites_existing_block(self, tmp_path: Path) -> None:
        (tmp_path / "config.json").write_text(
            json.dumps({"quantization_config": {"quant_method": "stale"}})
        )
        quantize_rskill._stamp_quantization_config(tmp_path, scheme="nf4", loader="transformers")
        config = json.loads((tmp_path / "config.json").read_text())
        assert config["quantization_config"]["quant_method"] == "bitsandbytes"

    def test_stamp_noop_without_config(self, tmp_path: Path) -> None:
        # lerobot checkpoints carry no top-level config.json — must not crash.
        quantize_rskill._stamp_quantization_config(tmp_path, scheme="nf4", loader="transformers")
        assert not (tmp_path / "config.json").exists()
