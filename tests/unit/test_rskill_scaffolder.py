"""Unit tests for ``ral skill new`` and the underlying ``scaffold_rskill`` helper.

No mocks: the scaffolder is local I/O against the real on-disk
``rskills/template/`` directory and the real ``RSkillManifest`` /
``rSkill`` loaders. Each test writes into ``tmp_path`` for isolation.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from openral_cli._rskill_scaffolder import scaffold_rskill
from openral_cli.main import app
from openral_core.exceptions import ROSConfigError
from openral_core.schemas import RSkillLicensePosture
from openral_rskill.loader import rSkill
from typer.testing import CliRunner

runner = CliRunner()


# ── scaffold_rskill (helper) ───────────────────────────────────────────────────


class TestScaffoldSkill:
    def test_minimal_succeeds_and_loads(self, tmp_path: Path) -> None:
        """Defaults produce a manifest that round-trips through rSkill.from_yaml."""
        out = tmp_path / "pi05-pick-cube"
        scaffold_rskill(
            "pi05-pick-cube",
            out_dir=out,
            owner="your-org",
            license_=RSkillLicensePosture.APACHE_2_0,
            embodiment_tag="franka_panda",
        )
        pkg = rSkill.from_yaml(out / "rskill.yaml")
        assert pkg.manifest.name == "your-org/rskill-pi05-pick-cube"
        assert pkg.manifest.license == RSkillLicensePosture.APACHE_2_0
        assert pkg.manifest.embodiment_tags == ["franka_panda"]
        # Files copied from the template directory:
        assert (out / "README.md").is_file()
        assert (out / "eval").is_dir()

    def test_rewrites_name_license_embodiment(self, tmp_path: Path) -> None:
        """Non-default arguments land in the generated manifest."""
        out = tmp_path / "pi05-pick-cube"
        scaffold_rskill(
            "pi05-pick-cube",
            out_dir=out,
            owner="foo",
            license_=RSkillLicensePosture.MIT,
            embodiment_tag="aloha",
        )
        pkg = rSkill.from_yaml(out / "rskill.yaml")
        assert pkg.manifest.name == "foo/rskill-pi05-pick-cube"
        assert pkg.manifest.license == RSkillLicensePosture.MIT
        assert pkg.manifest.embodiment_tags == ["aloha"]
        # README sentinels rewritten too:
        readme = (out / "README.md").read_text(encoding="utf-8")
        assert "TEMPLATE_ORG" not in readme
        assert "TEMPLATE_ID" not in readme
        assert "rskill-pi05-pick-cube" in readme

    def test_weights_uri_rewritten(self, tmp_path: Path) -> None:
        """`weights_uri` and `source_repo` follow the owner / id rewrite."""
        out = tmp_path / "act-grasp"
        scaffold_rskill(
            "act-grasp",
            out_dir=out,
            owner="bar",
            license_=RSkillLicensePosture.APACHE_2_0,
            embodiment_tag="so100_follower",
        )
        raw = yaml.safe_load((out / "rskill.yaml").read_text(encoding="utf-8"))
        assert raw["weights_uri"] == "hf://bar/act-grasp"
        assert raw["source_repo"] == "hf://bar/act-grasp"

    def test_refuses_existing_dir(self, tmp_path: Path) -> None:
        """A second scaffold to the same path must refuse without --overwrite."""
        out = tmp_path / "pi05-once"
        scaffold_rskill(
            "pi05-once",
            out_dir=out,
            owner="your-org",
            license_=RSkillLicensePosture.APACHE_2_0,
            embodiment_tag="franka_panda",
        )
        with pytest.raises(ROSConfigError, match="refusing to overwrite"):
            scaffold_rskill(
                "pi05-once",
                out_dir=out,
                owner="your-org",
                license_=RSkillLicensePosture.APACHE_2_0,
                embodiment_tag="franka_panda",
            )

    def test_overwrite_replaces_existing_dir(self, tmp_path: Path) -> None:
        """`overwrite=True` replaces an existing scaffold."""
        out = tmp_path / "pi05-twice"
        scaffold_rskill(
            "pi05-twice",
            out_dir=out,
            owner="alice",
            license_=RSkillLicensePosture.APACHE_2_0,
            embodiment_tag="franka_panda",
        )
        scaffold_rskill(
            "pi05-twice",
            out_dir=out,
            owner="bob",
            license_=RSkillLicensePosture.MIT,
            embodiment_tag="aloha",
            overwrite=True,
        )
        pkg = rSkill.from_yaml(out / "rskill.yaml")
        assert pkg.manifest.name == "bob/rskill-pi05-twice"
        assert pkg.manifest.license == RSkillLicensePosture.MIT


# ── ral skill new (Typer command) ──────────────────────────────────────────────


class TestBhSkillNew:
    def test_yes_skips_prompts_and_scaffolds(self, tmp_path: Path) -> None:
        """`--yes` runs to completion with no prompts in the captured output."""
        out = tmp_path / "pi05-yes"
        result = runner.invoke(
            app,
            ["rskill", "new", "pi05-yes", "--yes", "--out-dir", str(out)],
        )
        assert result.exit_code == 0, result.output
        assert "Scaffolded" in result.output
        # No prompts in the captured stdout:
        assert "HF Hub owner" not in result.output
        assert "License posture" not in result.output
        pkg = rSkill.from_yaml(out / "rskill.yaml")
        assert pkg.manifest.name == "your-org/rskill-pi05-yes"

    def test_explicit_flags_skip_prompts(self, tmp_path: Path) -> None:
        """When owner / license / embodiment / family are all passed, no prompt fires."""
        out = tmp_path / "pi05-flags"
        result = runner.invoke(
            app,
            [
                "rskill",
                "new",
                "pi05-flags",
                "--owner",
                "foo",
                "--license",
                "mit",
                "--embodiment-tag",
                "aloha",
                "--family",
                "pi05",
                "--out-dir",
                str(out),
            ],
        )
        assert result.exit_code == 0, result.output
        assert "HF Hub owner" not in result.output
        assert "Policy family" not in result.output
        pkg = rSkill.from_yaml(out / "rskill.yaml")
        assert pkg.manifest.name == "foo/rskill-pi05-flags"
        assert pkg.manifest.license == RSkillLicensePosture.MIT
        assert pkg.manifest.embodiment_tags == ["aloha"]

    def test_interactive_prompts_drive_values(self, tmp_path: Path) -> None:
        """`CliRunner(input=...)` drives every interactive prompt, including family."""
        out = tmp_path / "pi05-interactive"
        result = runner.invoke(
            app,
            ["rskill", "new", "pi05-interactive", "--out-dir", str(out)],
            # owner, license, embodiment, family
            input="foo\nmit\naloha\nact\n",
        )
        assert result.exit_code == 0, result.output
        # All four prompts must have fired:
        assert "HF Hub owner" in result.output
        assert "License posture" in result.output
        assert "Embodiment tag" in result.output
        assert "Policy family" in result.output
        pkg = rSkill.from_yaml(out / "rskill.yaml")
        assert pkg.manifest.name == "foo/rskill-pi05-interactive"
        assert pkg.manifest.license == RSkillLicensePosture.MIT
        assert pkg.manifest.embodiment_tags == ["aloha"]
        # Family prompt landed: model_family flipped to act.
        assert pkg.manifest.model_family == "act"

    def test_interactive_family_empty_keeps_template_baseline(self, tmp_path: Path) -> None:
        """Pressing Enter at the family prompt leaves the pi0.5-shaped template baseline."""
        out = tmp_path / "pi05-empty-family"
        result = runner.invoke(
            app,
            ["rskill", "new", "pi05-empty-family", "--out-dir", str(out)],
            # owner, license, embodiment, family (empty)
            input="foo\napache-2.0\nfranka_panda\n\n",
        )
        assert result.exit_code == 0, result.output
        assert "Policy family" in result.output
        pkg = rSkill.from_yaml(out / "rskill.yaml")
        assert pkg.manifest.model_family == "pi05"  # template baseline preserved

    def test_family_flag_overrides_template(self, tmp_path: Path) -> None:
        """`--family act` flips the manifest baseline to ACT-shaped defaults."""
        out = tmp_path / "act-family"
        result = runner.invoke(
            app,
            [
                "rskill",
                "new",
                "act-family",
                "--yes",
                "--family",
                "act",
                "--out-dir",
                str(out),
            ],
        )
        assert result.exit_code == 0, result.output
        pkg = rSkill.from_yaml(out / "rskill.yaml")
        assert pkg.manifest.model_family == "act"
        assert pkg.manifest.chunk_size == 100
        assert pkg.manifest.quantization.dtype.value == "fp32"
        # ACT family drops pi0.5-specific optional blocks.
        assert pkg.manifest.min_vram_gb is None
        assert pkg.manifest.n_action_steps is None

    def test_from_hf_auto_fills_manifest(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`--from-hf` introspects a recorded config.json and pre-fills the manifest.

        Network is replaced at the HF Hub boundary with a recorded
        Deepkar/libero-test-act ``config.json`` (CLAUDE.md §1.11: real
        components everywhere except the network boundary, which is
        explicitly stubbed). Asserts the four high-value fields the
        scaffolder is responsible for landing.
        """
        import json

        recorded_config = {
            "type": "act",
            "chunk_size": 100,
            "input_features": {
                "observation.images.image": {"type": "VISUAL", "shape": [3, 256, 256]},
                "observation.images.image2": {"type": "VISUAL", "shape": [3, 256, 256]},
                "observation.state": {"type": "STATE", "shape": [8]},
            },
            "output_features": {"action": {"type": "ACTION", "shape": [7]}},
        }
        recorded_dir = tmp_path / "hf-cache"
        recorded_dir.mkdir()
        recorded_path = recorded_dir / "config.json"
        recorded_path.write_text(json.dumps(recorded_config))

        def _fake_hf_hub_download(repo_id: str, filename: str, **kwargs: object) -> str:
            assert repo_id == "Deepkar/libero-test-act"
            assert filename == "config.json"
            return str(recorded_path)

        from openral_cli import _rskill_intel

        monkeypatch.setattr(
            "huggingface_hub.hf_hub_download",
            _fake_hf_hub_download,
        )
        # Re-import path safety — ``_rskill_intel`` imports ``hf_hub_download``
        # lazily inside ``_fetch_hf_json`` so the monkeypatch above is enough.
        assert _rskill_intel.RSKILL_FAMILIES  # sanity: module imported.

        out = tmp_path / "act-from-hf"
        result = runner.invoke(
            app,
            [
                "rskill",
                "new",
                "act-from-hf",
                "--yes",
                "--from-hf",
                "Deepkar/libero-test-act",
                "--out-dir",
                str(out),
            ],
        )
        assert result.exit_code == 0, result.output
        assert "Auto-detected" in result.output
        pkg = rSkill.from_yaml(out / "rskill.yaml")
        assert pkg.manifest.model_family == "act"
        assert pkg.manifest.chunk_size == 100
        assert pkg.manifest.weights_uri == "hf://Deepkar/libero-test-act"
        # Image min_w/min_h reflect the recorded 256x256 contract, not the
        # 224x224 template default.
        assert pkg.manifest.sensors_required[0].min_width == 256
        assert pkg.manifest.sensors_required[0].min_height == 256
        # Aliases rewrite camera<N> → checkpoint feature names.
        assert pkg.manifest.image_preprocessing is not None
        assert pkg.manifest.image_preprocessing.aliases == {
            "camera1": "image",
            "camera2": "image2",
        }
        # State dim pulled off observation.state.shape.
        assert pkg.manifest.state_contract is not None
        assert pkg.manifest.state_contract.dim == 8

    def test_invalid_family_exits_nonzero(self, tmp_path: Path) -> None:
        """`--family fictional` exits non-zero and lists the supported families."""
        out = tmp_path / "bad-family"
        result = runner.invoke(
            app,
            [
                "rskill",
                "new",
                "bad-family",
                "--yes",
                "--family",
                "fictional",
                "--out-dir",
                str(out),
            ],
        )
        assert result.exit_code != 0
        assert "fictional" in result.output
        assert "act" in result.output
        assert not out.exists()

    def test_invalid_license_exits_nonzero(self, tmp_path: Path) -> None:
        """`--license fictional` exits non-zero with a useful error."""
        out = tmp_path / "pi05-bad-license"
        result = runner.invoke(
            app,
            [
                "rskill",
                "new",
                "pi05-bad-license",
                "--yes",
                "--license",
                "fictional",
                "--out-dir",
                str(out),
            ],
        )
        assert result.exit_code != 0
        assert "fictional" in result.output
        assert "apache-2.0" in result.output  # valid values listed
        assert not out.exists()

    def test_invalid_embodiment_exits_nonzero(self, tmp_path: Path) -> None:
        """`--embodiment-tag fictional` exits non-zero with a useful error."""
        out = tmp_path / "pi05-bad-embodiment"
        result = runner.invoke(
            app,
            [
                "rskill",
                "new",
                "pi05-bad-embodiment",
                "--yes",
                "--embodiment-tag",
                "fictional",
                "--out-dir",
                str(out),
            ],
        )
        assert result.exit_code != 0
        assert "fictional" in result.output
        assert "franka_panda" in result.output  # valid values listed
        assert not out.exists()

    def test_existing_dir_without_overwrite_errors(self, tmp_path: Path) -> None:
        """Second `ral skill new` to the same out-dir exits non-zero."""
        out = tmp_path / "pi05-once"
        runner.invoke(app, ["rskill", "new", "pi05-once", "--yes", "--out-dir", str(out)])
        result = runner.invoke(app, ["rskill", "new", "pi05-once", "--yes", "--out-dir", str(out)])
        assert result.exit_code != 0
        assert "refusing to overwrite" in result.output
