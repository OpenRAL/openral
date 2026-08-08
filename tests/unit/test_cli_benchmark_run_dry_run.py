"""``openral benchmark run --dry-run`` must predict what the real run does.

The dry run previously printed a plan straight from the suite YAML without
applying :func:`openral_sim.benchmark.filter_scenes_for_skill` — the
``evaluated_tasks`` filter ``run_benchmark`` applies before any rollout. A
pairing the rSkill covers for zero tasks therefore dry-ran clean and then
raised ``ROSCapabilityMismatch`` on the real invocation, and a partially
covered suite reported an episode count several times larger than what would
actually execute.

Fixtures are the real in-tree suites and rSkill manifests (CLAUDE.md §1.11).
"""

from __future__ import annotations

from openral_cli.main import app
from typer.testing import CliRunner

runner = CliRunner()


def test_benchmark_run_dry_run_fails_when_no_task_matches() -> None:
    """Zero-coverage pairing must exit non-zero instead of printing a plan.

    ``rskills/smolvla-maniskill-franka`` declares
    ``evaluated_tasks: ["maniskill3/LiftCube"]``; the ``maniskill3_panda``
    suite ships PickCube/PushCube/PullCube/StackCube/PegInsertionSide/
    PlugCharger/PickSingleYCB and no LiftCube, so nothing would run.
    """
    result = runner.invoke(
        app,
        [
            "benchmark",
            "run",
            "--suite",
            "maniskill3_panda",
            "--rskill",
            "rskills/smolvla-maniskill-franka",
            "--dry-run",
        ],
    )
    assert result.exit_code == 1, result.output
    assert "task gate" in result.output
    assert "Nothing would run" in " ".join(result.output.split())


def test_benchmark_run_dry_run_reports_partial_coverage() -> None:
    """A task-scoped rSkill must have its skipped suite tasks surfaced.

    ``rskills/act-aloha`` covers ``aloha_transfer_cube`` only; the ``aloha``
    suite carries two tasks, so one is skipped and the printed plan must
    reflect the single task that would actually run.
    """
    result = runner.invoke(
        app,
        [
            "benchmark",
            "run",
            "--suite",
            "aloha",
            "--rskill",
            "rskills/act-aloha",
            "--dry-run",
        ],
    )
    assert result.exit_code == 0, result.output
    flat = " ".join(result.output.split())
    assert "would be skipped" in flat
    assert "tasks=1" in flat


def test_benchmark_run_dry_run_full_coverage_has_no_skip_note() -> None:
    """A fully covered suite must print a clean plan with no skip note."""
    result = runner.invoke(
        app,
        [
            "benchmark",
            "run",
            "--suite",
            "libero_spatial",
            "--rskill",
            "rskills/smolvla-libero",
            "--dry-run",
        ],
    )
    assert result.exit_code == 0, result.output
    flat = " ".join(result.output.split())
    assert "would be skipped" not in flat
    assert "tasks=10" in flat
