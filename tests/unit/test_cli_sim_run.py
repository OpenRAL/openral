"""Unit tests for the `openral sim` subcommand surface.

Confirms four things:

1. ``openral sim run`` is mounted on the main ``openral`` Typer app and reachable
   via ``typer.testing.CliRunner``.
2. ``openral sim list`` (the dedicated registry-printer that replaced the
   legacy ``openral sim run --list`` flag) prints the three sim registries.
3. ``openral sim run --help`` exposes the rollout flag set without listing
   ``--list`` (which has moved to its own subcommand).
4. End-to-end smoke: ``openral sim run --robot pusht_2d --scene pusht --rskill placeholder``
   runs the real pusht adapter for a few steps without an HF Hub
   lookup (the mock-VLA-placeholder bypass introduced in commit
   ``fix(eval): allow mock VLAs to skip rSkill manifest load``).

A fourth test verifies that importing ``openral_cli.main`` does not
transitively pull in heavyweight sim dependencies (``torch``, ``mujoco``,
``gymnasium``) — the eval adapters defer those imports until ``_run()``
is called, and we don't want `openral doctor` startup to pay for them.
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
    for flag in ("--config", "--rskill", "--robot", "--task", "--no-view"):
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


def test_bh_sim_run_legacy_scene_flag_rejected() -> None:
    """The legacy ``--scene/--vla`` free-flag form was removed.

    The canonical invocation is ``openral sim run --config FILE.yaml --rskill
    rskills/<id>``. Passing ``--scene`` now surfaces as Click's
    "no such option" error.
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

    The eval registry adapters defer those imports until `_run()` is
    invoked. A regression here would make every `openral doctor` invocation
    pay for ~1 GB of CUDA libraries.

    We spawn a fresh Python subprocess so the parent test interpreter's
    pre-loaded modules don't pollute the check.
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
