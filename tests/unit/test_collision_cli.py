"""`openral collision lower|check` CLI behaviour (ADR-0030).

Dry-by-default (a regenerated ACM is a safety input — never silent), explicit
`--write`, and a `check` mode that exits non-zero on drift. Real panda_mobile
manifest, real lowering, no mocks (§1.11). Heavily gated: needs yourdfpy +
robot_descriptions (the `[lowering]` group).
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from typer.testing import CliRunner

pytest.importorskip("yourdfpy")
pytest.importorskip("robot_descriptions")

from openral_cli.main import app

runner = CliRunner()


_DROPPED_PAIR = "  - [panda_link1, panda_link4]\n"


def _drifted_copy(tmp_path: Path) -> Path:
    """A panda_mobile copy whose ACM drops the link1↔link4 pair (deliberate drift).

    The manifest's ``assets.srdf = file:panda_mobile.srdf`` resolves against the
    manifest's own directory (ADR-0057 ``resolve_asset``), so the SRDF must be
    copied alongside ``robot.yaml`` or the SRDF ACM path can't be lowered.
    """
    src_dir = Path("robots/panda_mobile")
    dst = tmp_path / "robot.yaml"
    text = (src_dir / "robot.yaml").read_text(encoding="utf-8")
    assert _DROPPED_PAIR in text, "fixture expects link1↔link4 in the committed ACM"
    dst.write_text(text.replace(_DROPPED_PAIR, ""), encoding="utf-8")
    shutil.copy(src_dir / "panda_mobile.srdf", tmp_path / "panda_mobile.srdf")
    return dst


def test_lower_dry_run_prints_diff_and_does_not_write(tmp_path: Path) -> None:
    dst = _drifted_copy(tmp_path)
    before = dst.read_text(encoding="utf-8")

    result = runner.invoke(app, ["collision", "lower", "--robot", str(dst), "--acm-only"])

    assert result.exit_code == 0, result.output
    assert dst.read_text(encoding="utf-8") == before, "dry run must not mutate the manifest"
    # The diff restores the dropped link1↔link4 pair (the pi05 false-E-stop culprit).
    assert "panda_link1, panda_link4" in result.output
    assert "Dry run" in result.output


def test_lower_write_applies_and_check_then_passes(tmp_path: Path) -> None:
    dst = _drifted_copy(tmp_path)

    written = runner.invoke(
        app, ["collision", "lower", "--robot", str(dst), "--acm-only", "--write"]
    )
    assert written.exit_code == 0, written.output
    assert "Wrote" in written.output

    # After --write, check must report no drift (exit 0).
    checked = runner.invoke(app, ["collision", "check", "--robot", str(dst), "--acm-only"])
    assert checked.exit_code == 0, checked.output


def test_check_fails_on_drift(tmp_path: Path) -> None:
    dst = _drifted_copy(tmp_path)
    result = runner.invoke(app, ["collision", "check", "--robot", str(dst), "--acm-only"])
    assert result.exit_code == 1, result.output
    assert "drift" in result.output.lower()


def test_mutually_exclusive_flags_rejected(tmp_path: Path) -> None:
    dst = tmp_path / "robot.yaml"
    shutil.copy("robots/panda_mobile/robot.yaml", dst)
    result = runner.invoke(
        app, ["collision", "lower", "--robot", str(dst), "--acm-only", "--geometry-only"]
    )
    assert result.exit_code == 2
    assert "mutually exclusive" in result.output
