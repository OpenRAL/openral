"""Unit tests for ``ral skill install`` and ``ral skill list`` CLI commands.

All HF Hub I/O and the rSkill loader are mocked.  The local registry is
isolated per test via tmp_path.

Coverage
--------
- ``ral skill list``          — empty registry → informational message, exit 0
- ``ral skill list``          — populated registry → table with correct rows
- ``ral skill list --json``   — emits valid JSON array
- ``ral skill install``       — happy path (Apache license, no confirmation needed)
- ``ral skill install``       — ROSConfigError surfaces as non-zero exit
- ``ral skill install``       — proprietary license + --yes skips prompt
- ``ral skill install``       — no HUB_ID argument → non-zero exit (typer)
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from openral_cli.main import app
from openral_core.exceptions import ROSConfigError
from openral_core.schemas import (
    ActuatorRequirement,
    ControlMode,
    ControlModeSemantics,
    RSkillAction,
    RSkillLatencyBudget,
    RSkillLicensePosture,
    RSkillManifest,
    RSkillProcessors,
    RSkillRuntime,
)
from openral_rskill.loader import InstalledRSkillEntry, rSkill
from typer.testing import CliRunner

runner = CliRunner()


# ── Fixtures ───────────────────────────────────────────────────────────────────


def _make_manifest(
    name: str = "test/rskill-alpha",
    license_: RSkillLicensePosture = RSkillLicensePosture.APACHE_2_0,
) -> RSkillManifest:
    return RSkillManifest(
        name=name,
        version="0.2.0",
        license=license_,
        role="s1",
        kind="vla",
        model_family="smolvla",
        embodiment_tags=["so100_follower"],
        runtime=RSkillRuntime.PYTORCH,
        weights_uri="hf://test/rskill-alpha",
        chunk_size=16,
        latency_budget=RSkillLatencyBudget(per_chunk_ms=500.0),
        actuators_required=[
            ActuatorRequirement(
                kind=ControlMode.JOINT_POSITION,
                control_mode_semantics=ControlModeSemantics(mode="absolute"),
            )
        ],
        processors=RSkillProcessors(
            preprocessor_uri="hf://test/rskill-alpha/policy_preprocessor.json",
            postprocessor_uri="hf://test/rskill-alpha/policy_postprocessor.json",
        ),
        description="Test rSkill fixture for the ral skill install / list CLI suite.",
        actions=[RSkillAction.GENERALIST],
    )


def _make_entry(
    repo_id: str = "test/rskill-alpha",
    installed_at: str = "2026-01-01T00:00:00+00:00",
) -> InstalledRSkillEntry:
    return InstalledRSkillEntry(
        repo_id=repo_id,
        version="0.2.0",
        revision=None,
        local_dir=f"/tmp/skills/{repo_id.replace('/', '-')}",
        manifest_path=f"/tmp/skills/{repo_id.replace('/', '-')}/rskill.yaml",
        license="apache-2.0",
        role="s1",
        kind="vla",
        embodiment_tags=["so100_follower"],
        installed_at=installed_at,
    )


# ── ral skill list ─────────────────────────────────────────────────────────────


class TestSkillList:
    """Tests for ``openral rskill list`` — unified in-tree + HF-Hub-installed table."""

    def test_intree_only_lists_repo_rskills(self, tmp_path: Path) -> None:
        """With an empty installed registry, the listing still shows in-tree rSkills."""
        reg = tmp_path / "rskills.json"
        with patch("openral_rskill.loader.DEFAULT_REGISTRY_PATH", reg):
            result = runner.invoke(app, ["rskill", "list"])
        assert result.exit_code == 0
        assert "in-tree" in result.output

    def test_installed_entries_appear_under_installed_source(self, tmp_path: Path) -> None:
        """Hub-installed entries appear with the ``installed`` source tag."""
        reg = tmp_path / "rskills.json"
        rSkill._register(_make_entry("test/rskill-alpha"), reg)
        with patch("openral_rskill.loader.DEFAULT_REGISTRY_PATH", reg):
            result = runner.invoke(app, ["rskill", "list", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        installed = [row for row in data if row.get("source") == "installed"]
        assert installed, "no installed rskill in JSON output"
        assert any(row["repo_id"] == "test/rskill-alpha" for row in installed)

    def test_json_output_is_valid(self, tmp_path: Path) -> None:
        """``--json`` flag must emit a valid JSON array with both sources tagged."""
        reg = tmp_path / "rskills.json"
        rSkill._register(_make_entry("test/rskill-alpha"), reg)
        with patch("openral_rskill.loader.DEFAULT_REGISTRY_PATH", reg):
            result = runner.invoke(app, ["rskill", "list", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert isinstance(data, list)
        sources = {row["source"] for row in data}
        assert "installed" in sources
        # In-tree rSkills are auto-discovered from rskills/.
        assert "in-tree" in sources

    def test_corrupt_registry_exits_nonzero(self, tmp_path: Path) -> None:
        """Corrupt registry JSON must cause non-zero exit with error message."""
        reg = tmp_path / "rskills.json"
        reg.write_text("{{NOT JSON", encoding="utf-8")
        with patch("openral_rskill.loader.DEFAULT_REGISTRY_PATH", reg):
            result = runner.invoke(app, ["rskill", "list"])
        assert result.exit_code != 0

    def test_multiple_installed_entries_all_shown(self, tmp_path: Path) -> None:
        """All installed entries must appear in the table output."""
        reg = tmp_path / "rskills.json"
        rSkill._register(_make_entry("test/rskill-a"), reg)
        rSkill._register(
            _make_entry("test/rskill-b", installed_at="2026-06-01T00:00:00+00:00"), reg
        )
        with patch("openral_rskill.loader.DEFAULT_REGISTRY_PATH", reg):
            result = runner.invoke(app, ["rskill", "list", "--json"])
        data = json.loads(result.output)
        repo_ids = {row["repo_id"] for row in data if row.get("source") == "installed"}
        assert "test/rskill-a" in repo_ids
        assert "test/rskill-b" in repo_ids


# ── ral skill install ──────────────────────────────────────────────────────────


class TestSkillInstall:
    """Tests for ``ral skill install``."""

    def _run_install(
        self,
        hub_id: str,
        *extra_args: str,
        manifest: RSkillManifest | None = None,
        manifest_path: str = "/tmp/rskill.yaml",
        local_dir: str = "/tmp/skill_cache",
        install_error: Exception | None = None,
        tmp_path: Path | None = None,
    ) -> CliRunner.Result:
        """Invoke ``ral skill install`` with all HF Hub + loader calls mocked."""
        if manifest is None:
            manifest = _make_manifest()
        mock_hf_download = MagicMock(return_value=manifest_path)
        mock_from_yaml = MagicMock(return_value=manifest)
        if install_error:
            mock_from_pretrained = MagicMock(side_effect=install_error)
        else:
            mock_from_pretrained = MagicMock(
                return_value=rSkill(manifest=manifest, local_dir=Path(local_dir))
            )

        with (
            patch("huggingface_hub.hf_hub_download", mock_hf_download),
            patch("openral_core.schemas.RSkillManifest.from_yaml", mock_from_yaml),
            patch("openral_rskill.loader.rSkill.from_pretrained", mock_from_pretrained),
        ):
            args = ["rskill", "install", hub_id, *extra_args]
            return runner.invoke(app, args, catch_exceptions=False)

    def test_happy_path_apache_exits_zero(self) -> None:
        """Apache-2.0 skill install must exit 0 without prompting."""
        result = self._run_install("test/rskill-alpha")
        assert result.exit_code == 0
        assert "Installed" in result.output

    def test_ros_config_error_exits_nonzero(self) -> None:
        """ROSConfigError from loader must print error and exit non-zero."""
        result = self._run_install(
            "test/rskill-groot",
            install_error=ROSConfigError("non-commercial"),
        )
        assert result.exit_code != 0
        assert "Install failed" in result.output

    def test_proprietary_with_yes_skips_prompt(self) -> None:
        """--yes must bypass the confirmation prompt for proprietary licenses."""
        manifest = _make_manifest(license_=RSkillLicensePosture.PROPRIETARY)
        result = self._run_install("test/rskill-helix", "--yes", manifest=manifest)
        assert result.exit_code == 0
        assert "Installed" in result.output

    def test_revision_flag_passed(self) -> None:
        """--revision <sha> must be forwarded to from_pretrained."""
        manifest = _make_manifest()
        mock_hf_download = MagicMock(return_value="/tmp/rskill.yaml")
        mock_from_yaml = MagicMock(return_value=manifest)
        mock_from_pretrained = MagicMock(
            return_value=rSkill(manifest=manifest, local_dir=Path("/tmp/skill"))
        )
        with (
            patch("huggingface_hub.hf_hub_download", mock_hf_download),
            patch("openral_core.schemas.RSkillManifest.from_yaml", mock_from_yaml),
            patch("openral_rskill.loader.rSkill.from_pretrained", mock_from_pretrained),
        ):
            result = runner.invoke(
                app,
                ["rskill", "install", "test/rskill-alpha", "--revision", "deadbeef"],
                catch_exceptions=False,
            )
        assert result.exit_code == 0
        _, kwargs = mock_from_pretrained.call_args
        assert kwargs.get("revision") == "deadbeef"

    def test_no_revision_prints_tip(self) -> None:
        """Installing without --revision must print a pinning tip."""
        result = self._run_install("test/rskill-alpha")
        assert result.exit_code == 0
        assert "Pin a revision" in result.output

    def test_manifest_fetch_error_exits_nonzero(self) -> None:
        """Network error fetching rskill.yaml must exit non-zero."""
        mock_hf_download = MagicMock(side_effect=RuntimeError("connection refused"))
        with patch("huggingface_hub.hf_hub_download", mock_hf_download):
            result = runner.invoke(app, ["rskill", "install", "test/rskill-alpha"])
        assert result.exit_code != 0
        assert "Failed to fetch manifest" in result.output
