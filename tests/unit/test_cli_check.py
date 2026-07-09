"""Tests for ``openral check`` — manifest graph validation.

Real fixtures only (CLAUDE.md §1.11):
- The committed ``robots/`` / ``rskills/`` / ``scenes/`` manifests are validated
  by the real ``from_yaml`` / ``model_validate`` paths.
- Failure-path tests start from a real in-tree manifest and introduce a single
  deliberate defect (a missing asset file, an unresolvable ``robot_id``).

(JSON-Schema emission for the manifests is covered by ``tools/schema_export.py``
and its CI drift check, not by this command.)
"""

from __future__ import annotations

from pathlib import Path

import yaml
from openral_cli.check import check_description_graph
from openral_cli.main import app
from typer.testing import CliRunner

REPO_ROOT = Path(__file__).resolve().parents[2]
runner = CliRunner()


# ── The real repo is clean ──────────────────────────────────────────────────────


def test_graph_real_repo_has_no_errors() -> None:
    report = check_description_graph(REPO_ROOT)
    assert report.n_robots > 0
    assert report.n_rskills > 0
    assert report.n_scenes > 0
    assert report.errors == [], [f.model_dump() for f in report.errors]
    assert report.ok


# ── Failure paths (real manifest + one deliberate defect) ───────────────────────


def test_graph_flags_dangling_asset_ref(tmp_path: Path) -> None:
    real = yaml.safe_load((REPO_ROOT / "robots" / "franka_panda" / "robot.yaml").read_text())
    real["assets"] = {"mjcf": "file:__definitely_missing__.xml"}
    robot_dir = tmp_path / "robots" / "probe"
    robot_dir.mkdir(parents=True)
    (robot_dir / "robot.yaml").write_text(yaml.safe_dump(real), encoding="utf-8")

    report = check_description_graph(tmp_path)
    assert not report.ok
    asset_errors = [f for f in report.errors if f.rule == "asset_ref"]
    assert len(asset_errors) == 1
    assert "robots/probe" in asset_errors[0].target


def test_graph_flags_unresolvable_scene_robot_id(tmp_path: Path) -> None:
    # A real benchmark scene, copied verbatim, with no matching robot dir present.
    real_scene = (REPO_ROOT / "scenes" / "benchmark" / "libero_object.yaml").read_text()
    scene_dir = tmp_path / "scenes" / "benchmark"
    scene_dir.mkdir(parents=True)
    (scene_dir / "libero_object.yaml").write_text(real_scene, encoding="utf-8")

    report = check_description_graph(tmp_path)
    assert report.n_scenes == 1
    robot_id_errors = [f for f in report.errors if f.rule == "scene_robot_id"]
    assert len(robot_id_errors) == 1
    assert "robot.yaml" in robot_id_errors[0].message


def test_graph_warns_on_unreachable_embodiment(tmp_path: Path) -> None:
    # A real VLA rSkill with embodiment tags, but no robots to satisfy them.
    real = (REPO_ROOT / "rskills" / "act-libero" / "rskill.yaml").read_text()
    rskill_dir = tmp_path / "rskills" / "act-libero"
    rskill_dir.mkdir(parents=True)
    (rskill_dir / "rskill.yaml").write_text(real, encoding="utf-8")

    report = check_description_graph(tmp_path)
    assert report.n_rskills == 1
    assert report.ok  # a warning, not an error
    reach = [f for f in report.warnings if f.rule == "embodiment_reach"]
    assert len(reach) == 1


# ── CLI wiring ──────────────────────────────────────────────────────────────────


def test_cli_check_passes_on_real_repo() -> None:
    result = runner.invoke(app, ["check", "--repo-root", str(REPO_ROOT)])
    assert result.exit_code == 0, result.stdout


def test_cli_check_strict_fails_when_warnings_present(tmp_path: Path) -> None:
    # One unreachable-embodiment warning ⇒ --strict turns it into a failure.
    real = (REPO_ROOT / "rskills" / "act-libero" / "rskill.yaml").read_text()
    rskill_dir = tmp_path / "rskills" / "act-libero"
    rskill_dir.mkdir(parents=True)
    (rskill_dir / "rskill.yaml").write_text(real, encoding="utf-8")

    ok = runner.invoke(app, ["check", "--repo-root", str(tmp_path)])
    assert ok.exit_code == 0, ok.stdout
    strict = runner.invoke(app, ["check", "--repo-root", str(tmp_path), "--strict"])
    assert strict.exit_code == 1, strict.stdout
