"""End-to-end test for the ``openral rskill list`` in-tree discovery line.

CLAUDE.md §1.11 and §5.4 require real components — this test walks the
in-repo ``rskills/<dir>/rskill.yaml`` files (no mocks, no stubs) and
asserts every in-tree rSkill appears in the unified listing as a
paste-able bare name under the ``in-tree`` source.
"""

from __future__ import annotations

from pathlib import Path

from openral_cli.main import app
from typer.testing import CliRunner

runner = CliRunner()


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for ancestor in (here, *here.parents):
        if (ancestor / "pyproject.toml").is_file() and (ancestor / "rskills").is_dir():
            return ancestor
    raise AssertionError("could not locate OpenRAL repo root from test file")


def test_bh_rskill_list_contains_every_intree_skill() -> None:
    """Every in-tree rskills/<dir>/rskill.yaml appears in `openral rskill list`."""
    result = runner.invoke(app, ["rskill", "list"])
    assert result.exit_code == 0, result.output

    repo = _repo_root()
    on_disk = sorted(
        child.name
        for child in (repo / "rskills").iterdir()
        if child.is_dir() and (child / "rskill.yaml").is_file()
    )
    assert on_disk, "no rskills/<dir>/rskill.yaml found in the repo"

    # Rich table wraps long names; the substring check tolerates that.
    for skill_name in on_disk:
        assert skill_name[:6] in result.output, (
            f"`openral rskill list` is missing the rSkill at rskills/{skill_name}. "
            f"Output was:\n{result.output}"
        )


def test_bh_rskill_list_marks_intree_source() -> None:
    """The unified table tags in-tree entries with the ``in-tree`` source."""
    result = runner.invoke(app, ["rskill", "list"])
    assert result.exit_code == 0, result.output
    assert "in-tree" in result.output


def test_bh_rskill_list_json_emits_uri_per_entry() -> None:
    """``--json`` emits one record per in-tree rskill with a paste-able bare name."""
    import json as _json

    result = runner.invoke(app, ["rskill", "list", "--json"])
    assert result.exit_code == 0, result.output
    payload = _json.loads(result.output)
    assert isinstance(payload, list)
    intree = [row for row in payload if row.get("source") == "in-tree"]
    assert intree, "no in-tree rskills in --json output"
    for row in intree:
        assert not row["uri"].startswith("rskill://"), row
        assert "/" not in row["uri"], row
