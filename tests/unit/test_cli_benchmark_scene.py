"""CLI surface tests for ``openral benchmark scene``.

Single-scene benchmark sibling of ``openral benchmark run --suite``. The
``scene`` subcommand accepts exactly one ``BenchmarkScene`` YAML and
delegates the rollout to :func:`openral_sim.benchmark.run_benchmark_scene`,
emitting a validated :class:`RSkillEvalResult` JSON in the same shape as
the multi-task suite runner so paper-comparison reports stay uniform.

End-to-end coverage of the typer wiring only — the runtime aggregation
logic is exercised by ``tests/unit/test_run_benchmark_scene.py``.
"""

from __future__ import annotations

from pathlib import Path

from openral_cli.main import app
from typer.testing import CliRunner

_REPO_ROOT = Path(__file__).resolve().parents[2]
_BENCHMARK_SCENE = _REPO_ROOT / "scenes" / "benchmark" / "pusht.yaml"
_SIM_SCENE = _REPO_ROOT / "scenes" / "sim" / "tabletop_cube_push.yaml"
_DEPLOY_SCENE = _REPO_ROOT / "scenes" / "deploy" / "openarm_tabletop.yaml"


def test_benchmark_scene_help_lists_config_and_rskill() -> None:
    """``openral benchmark scene --help`` exposes both required flags."""
    runner = CliRunner()
    result = runner.invoke(app, ["benchmark", "scene", "--help"])
    assert result.exit_code == 0, result.output
    for flag in ("--config", "--rskill", "--out", "--device", "--n-episodes"):
        assert flag in result.output, f"{flag} missing from help"


def test_benchmark_scene_rejects_simscene_yaml() -> None:
    """A SimScene YAML (no `metadata.paper`) must be rejected with a redirect.

    ``scenes/sim/tabletop_cube_push.yaml`` carries only ``honest_scope``
    in its metadata — no ``paper`` field — so it cannot satisfy
    :class:`BenchmarkMetadata` and is the canonical "one-tier-too-thin"
    rejection case for ``benchmark scene``.
    """
    assert _SIM_SCENE.is_file(), f"missing fixture: {_SIM_SCENE}"
    runner = CliRunner()
    result = runner.invoke(
        app,
        ["benchmark", "scene", "--config", str(_SIM_SCENE), "--rskill", "placeholder"],
    )
    assert result.exit_code != 0
    combined = (result.output or "") + (str(result.exception) if result.exception else "")
    assert "BenchmarkScene" in combined, combined


def test_benchmark_scene_rejects_deployscene_yaml() -> None:
    """A DeployScene YAML (no task block) must be rejected with a redirect."""
    assert _DEPLOY_SCENE.is_file(), f"missing fixture: {_DEPLOY_SCENE}"
    runner = CliRunner()
    result = runner.invoke(
        app,
        ["benchmark", "scene", "--config", str(_DEPLOY_SCENE), "--rskill", "placeholder"],
    )
    assert result.exit_code != 0
    combined = (result.output or "") + (str(result.exception) if result.exception else "")
    assert "BenchmarkScene" in combined, combined


def test_benchmark_scene_accepts_benchmark_yaml_in_dry_run() -> None:
    """A real BenchmarkScene YAML loads cleanly under ``--dry-run``.

    ``scenes/benchmark/pusht.yaml`` is a published-protocol BenchmarkScene
    with the full ``n_episodes`` / ``seed`` / ``metadata.paper`` /
    ``metadata.honest_scope`` block. ``--dry-run`` short-circuits before
    the rollout so this test does not load a policy or hit physics.
    """
    assert _BENCHMARK_SCENE.is_file(), f"missing fixture: {_BENCHMARK_SCENE}"
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "benchmark",
            "scene",
            "--config",
            str(_BENCHMARK_SCENE),
            "--rskill",
            "placeholder",
            "--dry-run",
        ],
    )
    assert result.exit_code == 0, result.output
    flat = " ".join(result.output.split())
    assert "scene pusht" in flat
    assert "task=pusht/0" in flat
    # n_episodes from the YAML, not the protocol default.
    assert "n_episodes=50" in flat
    # robot_id added in Task 9.5 — assert it surfaces in the dry-run output.
    assert "robot=pusht_2d" in flat


def test_benchmark_scene_dry_run_honours_n_episodes_override() -> None:
    """``--n-episodes`` overrides the YAML value before dry-run reporting."""
    assert _BENCHMARK_SCENE.is_file(), f"missing fixture: {_BENCHMARK_SCENE}"
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "benchmark",
            "scene",
            "--config",
            str(_BENCHMARK_SCENE),
            "--rskill",
            "placeholder",
            "--n-episodes",
            "3",
            "--dry-run",
        ],
    )
    assert result.exit_code == 0, result.output
    flat = " ".join(result.output.split())
    assert "n_episodes=3" in flat
