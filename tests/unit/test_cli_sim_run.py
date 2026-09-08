"""Unit tests for the `openral sim` subcommand surface.

Covers: ``openral sim run`` mounted on the main ``openral`` Typer app; ``openral sim
list`` (replaced the legacy ``openral sim run --list`` flag) prints the sim
registries; ``openral sim run --help`` shows the rollout flags without ``--list``; an
e2e smoke run (``--robot pusht_2d --scene pusht --rskill placeholder``) exercises the
real pusht adapter without an HF Hub lookup via the mock-VLA-placeholder bypass
(commit ``fix(eval): allow mock VLAs to skip rSkill manifest load``); and that
importing ``openral_cli.main`` doesn't pull in torch/mujoco/gymnasium — eval adapters
defer those imports until ``_run()`` so ``openral doctor`` startup stays light.
"""

from __future__ import annotations

import subprocess
import sys

from openral_cli.main import app
from typer.testing import CliRunner

runner = CliRunner()


def test_bh_sim_list_prints_example_configs() -> None:
    """`openral sim list` prints every ``scenes/**/*.yaml`` as paste-able paths."""
    result = runner.invoke(app, ["sim", "list"])
    assert result.exit_code == 0, result.output
    # Each line is a path to an example sim config.
    assert "scenes/" in result.output
    # Spot-check known configs.
    assert "pusht.yaml" in result.output
    assert "libero_spatial.yaml" in result.output
    # rSkill URIs no longer live here — they moved to `openral rskill list`.
    assert "rskill://" not in result.output


def test_bh_sim_run_help_shows_flags() -> None:
    """`openral sim run --help` surfaces the rollout flag set, not the root `openral` help."""
    result = runner.invoke(app, ["sim", "run", "--help"])
    assert result.exit_code == 0, result.output
    for flag in ("--config", "--rskill", "--robot", "--task", "--no-view", "--dry-run"):
        assert flag in result.output, f"flag {flag!r} missing from `openral sim run --help`"
    # Legacy free-flag composition (--scene / --vla) was removed in the
    # `feat(core,sim): SceneEnvironment + openral sim run --rskill, no legacy` commit.
    assert "--scene" not in result.output
    assert "--vla " not in result.output


def test_bh_sim_run_help_omits_list_flag() -> None:
    """`openral sim run --help` must not list ``--list`` (it moved to ``openral sim list``)."""
    result = runner.invoke(app, ["sim", "run", "--help"])
    assert result.exit_code == 0, result.output
    assert "--list" not in result.output


def test_bh_sim_run_rejects_list_flag() -> None:
    """`openral sim run --list` was removed; users must call `openral sim list` instead."""
    result = runner.invoke(app, ["sim", "run", "--list"])
    assert result.exit_code != 0
    assert "no such option" in result.output.lower()


def test_bh_sim_help_lists_subcommands() -> None:
    """`openral sim --help` advertises both `run` and `list` as subcommands."""
    result = runner.invoke(app, ["sim", "--help"])
    assert result.exit_code == 0, result.output
    assert "run" in result.output
    assert "list" in result.output


def test_bh_sim_run_rejects_record_video_flag() -> None:
    """`--record-video` was removed; `--save-video` is the single video entry point."""
    result = runner.invoke(app, ["sim", "run", "--record-video"])
    # Click prints "No such option: --record-video" and exits 2 on unknown flags.
    assert result.exit_code != 0
    assert "no such option" in result.output.lower()


def test_bh_sim_run_help_omits_record_video_flag() -> None:
    """Make sure `--record-video` no longer appears in the help text."""
    result = runner.invoke(app, ["sim", "run", "--help"])
    assert result.exit_code == 0, result.output
    assert "--record-video" not in result.output


def test_bh_sim_run_dry_run_resolves_without_building_sim() -> None:
    """`--dry-run` resolves the SimScene + rSkill but does not enter SimRunner."""
    result = runner.invoke(
        app,
        [
            "sim",
            "run",
            "--config",
            "scenes/sim/tabletop_cube_push.yaml",
            "--rskill",
            "rskills/molmoact2-so101-nf4",
            "--dry-run",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "openral sim run" in result.output
    assert "dry-run: resolved config + rSkill" in result.output
    assert "max_ticks:" in result.output


def test_bh_sim_run_legacy_scene_flag_rejected() -> None:
    """The legacy ``--scene/--vla`` free-flag form was removed.

    Canonical invocation is ``openral sim run --config FILE.yaml --rskill
    rskills/<id>``; ``--scene`` now surfaces as Click's "no such option" error.
    """
    result = runner.invoke(
        app,
        [
            "sim",
            "run",
            "--scene",
            "pusht",
            "--task",
            "pusht/0",
        ],
    )
    assert result.exit_code != 0
    assert "no such option" in result.output.lower()


def test_bh_cli_import_is_light() -> None:
    """Importing `openral_cli.main` must not transitively load torch / mujoco / gym.

    Eval registry adapters defer those imports until `_run()`; a regression would make
    every `openral doctor` invocation pay ~1 GB of CUDA libraries. Runs in a subprocess
    so the parent test interpreter's pre-loaded modules don't pollute the check.
    """
    code = (
        "import sys, openral_cli.main; "
        "heavy = {m for m in ('torch', 'mujoco', 'gymnasium', 'lerobot', 'mujoco_py') "
        "  if m in sys.modules}; "
        "print(','.join(sorted(heavy)))"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
    )
    loaded = proc.stdout.strip()
    assert loaded == "", (
        f"`openral_cli.main` import pulled in heavyweight sim modules: {loaded!r}.\n"
        "Eval adapters must keep their torch / mujoco / gym imports inside "
        "`_run()` and registered factories, not at module top level."
    )


def test_bh_sim_run_dry_run_rejects_incompatible_embodiment() -> None:
    """`--dry-run` must run the embodiment gate a real rollout hits.

    Regression: the dry-run branch returned before ``SimRunner.activate``'s
    ``openral_sim.sim_runner._check_rskill_compatibility`` call, so a
    franka_panda rSkill paired with the pusht scene printed a happy plan and
    exited 0 — making ``--dry-run`` useless as a CI wiring check.
    """
    result = runner.invoke(
        app,
        [
            "sim",
            "run",
            "--config",
            "scenes/sim/pusht.yaml",
            "--rskill",
            "rskills/smolvla-libero",
            "--dry-run",
        ],
    )
    assert result.exit_code == 1, result.output
    assert "compat error" in result.output
    assert "embodiment" in result.output.lower()


def test_bh_sim_run_dry_run_accepts_scene_render_resolution_bump() -> None:
    """A scene rendering above robot.yaml's nominal size must still pass.

    ``rskills/xr1-vlabench`` requires 480x480 while
    ``robots/franka_panda/robot.yaml`` declares 256x256 cameras; the vlabench
    scene renders at 480 and ``_check_rskill_compatibility`` syncs the sensor
    intrinsics to the real render size before the gate. Guards against a
    naive gate that reads robot.yaml alone and false-negatives here.
    """
    result = runner.invoke(
        app,
        [
            "sim",
            "run",
            "--config",
            "scenes/sim/xr1_vlabench_select_fruit.yaml",
            "--rskill",
            "rskills/xr1-vlabench",
            "--dry-run",
        ],
    )
    assert result.exit_code == 0, result.output


def test_bh_sim_run_rejects_non_vla_rskill_cleanly() -> None:
    """A detector / reward / playbook rSkill has no model_family.

    It must be refused with the same message ``openral benchmark run`` gives,
    not a raw pydantic ``string_type`` error from ``VLASpec(id=None)``.
    """
    result = runner.invoke(
        app,
        [
            "sim",
            "run",
            "--config",
            "scenes/sim/libero_spatial.yaml",
            "--rskill",
            "rskills/omdet-turbo-locator",
            "--dry-run",
        ],
    )
    assert result.exit_code == 1, result.output
    assert "no model_family" in result.output
    assert "expects a VLA skill" in result.output
    assert "string_type" not in result.output
