"""Unit tests for ``openral dataset push`` (ADR-0019 PR5).

Per CLAUDE.md §1.11 — no mocks of openral_* types. The HF Hub API is the
**network boundary**; the upload path is exercised via ``--dry-run``
(which never reaches the network) so we don't need a fake
``HfApi`` for the consent-gate tests. The single ``test_actual_upload``
case lives in a follow-up HIL job behind ``[needs-hf-token]``.

What this file covers:

* Path validation — missing ``meta/info.json``, malformed JSON.
* repo_id resolution — CLI override, info.json fallback, error on missing.
* Consent gating — interactive match / mismatch, ``--yes`` flag,
  ``OPENRAL_DATASET_CONSENT=1`` env override, non-TTY rejection.
* ``--dry-run`` short-circuit (no token resolution needed).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from openral_cli.main import app
from typer.testing import CliRunner


def _write_info_json(
    root: Path,
    *,
    repo_id: str | None = "openral/dataset-test",
    total_episodes: int = 2,
    total_frames: int = 6,
    cameras: tuple[str, ...] = ("observation.images.camera1",),
    license_str: str = "CC-BY-4.0",
) -> Path:
    """Write a minimal LeRobotDataset v3-shaped meta/info.json under ``root``.

    Matches the schema `LeRobotDatasetSink` writes on finalize so the
    consent prompt's field-extraction path runs against the same
    JSON shape it sees in production.
    """
    meta_dir = root / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    features = {key: {"dtype": "video", "shape": [32, 32, 3]} for key in cameras}
    info = {
        "codebase_version": "3.0",
        "fps": 30,
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "features": features,
        "metadata": {
            "license": license_str,
            "robot_name": "so100_follower",
            "dataset_success_rate": 0.5,
        },
    }
    if repo_id is not None:
        info["metadata"]["repo_id"] = repo_id  # type: ignore[index]
    info_path = meta_dir / "info.json"
    info_path.write_text(json.dumps(info, indent=2))
    return info_path


# ── Path / info.json validation ──────────────────────────────────────────────


def test_push_rejects_missing_info_json(tmp_path: Path) -> None:
    """A directory without meta/info.json fails with a clear config error."""
    root = tmp_path / "not_a_dataset"
    root.mkdir()
    runner = CliRunner()
    result = runner.invoke(app, ["dataset", "push", str(root), "--dry-run"])
    assert result.exit_code == 1, result.output
    # Rich wraps long error messages onto multiple lines; assert on the
    # tokens individually rather than the whole sentence.
    output_collapsed = " ".join(result.output.split())
    assert "does not look like a LeRobotDataset v3 root" in output_collapsed


def test_push_rejects_malformed_info_json(tmp_path: Path) -> None:
    """A meta/info.json with invalid JSON fails with a clear config error."""
    root = tmp_path / "ds"
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text("{not json")
    runner = CliRunner()
    result = runner.invoke(app, ["dataset", "push", str(root), "--dry-run"])
    assert result.exit_code == 1, result.output
    assert "not valid JSON" in result.output


# ── repo_id resolution ───────────────────────────────────────────────────────


def test_push_repo_id_falls_back_to_info_json(tmp_path: Path) -> None:
    """When info.json carries metadata.repo_id, no --repo-id flag is required."""
    root = tmp_path / "ds"
    _write_info_json(root, repo_id="openral/dataset-test")
    runner = CliRunner()
    result = runner.invoke(
        app,
        ["dataset", "push", str(root), "--dry-run", "--yes"],
    )
    assert result.exit_code == 0, result.output
    assert "openral/dataset-test" in result.output


def test_push_repo_id_cli_override_wins(tmp_path: Path) -> None:
    """--repo-id overrides info.json's metadata.repo_id."""
    root = tmp_path / "ds"
    _write_info_json(root, repo_id="openral/dataset-test")
    runner = CliRunner()
    result = runner.invoke(
        app,
        ["dataset", "push", str(root), "--dry-run", "--yes", "--repo-id", "myorg/override"],
    )
    assert result.exit_code == 0, result.output
    assert "myorg/override" in result.output


def test_push_rejects_missing_repo_id(tmp_path: Path) -> None:
    """Missing repo_id (both in info.json and on CLI) is a config error."""
    root = tmp_path / "ds"
    _write_info_json(root, repo_id=None)
    runner = CliRunner()
    result = runner.invoke(app, ["dataset", "push", str(root), "--dry-run", "--yes"])
    assert result.exit_code == 1, result.output
    assert "no repo_id found" in result.output


@pytest.mark.parametrize("bad_repo_id", ["nounderscore", "/leadingslash", "trailingslash/", ""])
def test_push_rejects_invalid_repo_id(tmp_path: Path, bad_repo_id: str) -> None:
    """Invalid repo_id formats are caught before the consent gate."""
    root = tmp_path / "ds"
    _write_info_json(root)
    runner = CliRunner()
    result = runner.invoke(
        app,
        ["dataset", "push", str(root), "--dry-run", "--yes", "--repo-id", bad_repo_id],
    )
    assert result.exit_code == 1, result.output


# ── Consent gating ───────────────────────────────────────────────────────────


def test_push_yes_flag_skips_consent_prompt(tmp_path: Path) -> None:
    """--yes bypasses the typer.confirm prompt entirely."""
    root = tmp_path / "ds"
    _write_info_json(root)
    runner = CliRunner()
    # No stdin input — would block on prompt without --yes.
    result = runner.invoke(app, ["dataset", "push", str(root), "--dry-run", "--yes"], input="")
    assert result.exit_code == 0, result.output


def test_push_env_override_skips_consent_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """OPENRAL_DATASET_CONSENT=1 bypasses the prompt (matches CI / scripted runs)."""
    root = tmp_path / "ds"
    _write_info_json(root)
    monkeypatch.setenv("OPENRAL_DATASET_CONSENT", "1")
    runner = CliRunner()
    result = runner.invoke(app, ["dataset", "push", str(root), "--dry-run"], input="")
    assert result.exit_code == 0, result.output


def test_push_consent_env_other_values_do_not_bypass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only OPENRAL_DATASET_CONSENT=1 bypasses; any other value still prompts."""
    root = tmp_path / "ds"
    _write_info_json(root)
    monkeypatch.setenv("OPENRAL_DATASET_CONSENT", "yes")  # NOT the magic value
    runner = CliRunner()
    # Non-TTY stdin with no override → the prompt path must reject.
    result = runner.invoke(app, ["dataset", "push", str(root), "--dry-run"], input="")
    assert result.exit_code == 1, result.output
    # Either the non-interactive guard or the typed-repo-id mismatch
    # rejects; both are valid "the prompt didn't get bypassed" outcomes.


def test_push_non_interactive_stdin_requires_override(tmp_path: Path) -> None:
    """Without --yes/env override, a non-TTY stdin must NOT proceed silently."""
    root = tmp_path / "ds"
    _write_info_json(root)
    runner = CliRunner()
    # input="" leaves stdin closed → isatty() returns False → must reject.
    result = runner.invoke(app, ["dataset", "push", str(root), "--dry-run"], input="")
    assert result.exit_code == 1, result.output


def test_push_consent_prompt_discloses_pii_categories(tmp_path: Path) -> None:
    """The prompt explicitly mentions faces / biometrics / regulatory scope.

    Real CliRunner with a real Panel render — no mocks of the consent
    text. The string check protects against accidental softening of the
    disclosure (e.g. a refactor that drops the biometrics line).
    """
    root = tmp_path / "ds"
    _write_info_json(root)
    runner = CliRunner()
    # input="\n" sends an empty line — fails the repo_id match → exit 1,
    # but the prompt body still renders to stdout, which is what we
    # assert on.
    result = runner.invoke(app, ["dataset", "push", str(root), "--dry-run"], input="\n")
    assert "PERSONAL DATA" in result.output
    assert "faces" in result.output
    assert "biometric" in result.output
    # Regulatory scope. CCPA appears uppercase in the prompt body so the
    # assertion is case-sensitive.
    assert "GDPR" in result.output


# ── --dry-run gate ───────────────────────────────────────────────────────────


def test_push_dry_run_skips_token_resolution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--dry-run completes successfully without HF_TOKEN set.

    Confirms the dry-run gate short-circuits before any HF Hub I/O.
    The non-dry-run path's missing-token branch is exercised below.
    """
    root = tmp_path / "ds"
    _write_info_json(root)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_HUB_TOKEN", raising=False)
    runner = CliRunner()
    result = runner.invoke(app, ["dataset", "push", str(root), "--dry-run", "--yes"])
    assert result.exit_code == 0, result.output
    assert "dry-run OK" in result.output


def test_push_non_dry_run_without_token_fails_fast(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A non-dry-run with no token resolves the consent gate, then fails on the token."""
    root = tmp_path / "ds"
    _write_info_json(root)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_HUB_TOKEN", raising=False)
    runner = CliRunner()
    result = runner.invoke(app, ["dataset", "push", str(root), "--yes"])
    assert result.exit_code == 1, result.output
    assert "HF_TOKEN" in result.output or "HF token" in result.output
