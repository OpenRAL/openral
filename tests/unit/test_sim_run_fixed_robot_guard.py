"""Tests for the scene-fixed-robot guard in ``openral sim run``.

Sim backends that hard-wire a single robot (LIBERO → Franka, MetaWorld →
Sawyer, PushT → 2-D pusher, gym-aloha → bimanual, RoboCasa → PandaMobile) used
to let ``--robot`` / ``robot_id:`` silently swap robots underneath the user.
The CLI now raises a typed ``ROSConfigError`` at config-build time, before
any rollout starts.

Canonical invocation::

    openral sim run --config FILE.yaml --rskill rskills/<id> [--robot R]

CLAUDE.md §1.11: real registered scenes through the real CLI/loader, no mocks.
The guard fires at config-build time, so no LIBERO/MetaWorld/gym-aloha install
is needed to exercise it.
"""

from __future__ import annotations

from pathlib import Path

from openral_cli.main import app
from typer.testing import CliRunner

REPO_ROOT = Path(__file__).resolve().parents[2]
LIBERO_CFG = REPO_ROOT / "scenes" / "benchmarks" / "smolvla_libero_spatial.yaml"
LIBERO_RSKILL = "rskills/smolvla-libero"
METAWORLD_CFG = REPO_ROOT / "scenes" / "benchmarks" / "smolvla_metaworld_push.yaml"
METAWORLD_RSKILL = "rskills/smolvla-metaworld"
PUSHT_CFG = REPO_ROOT / "scenes" / "benchmarks" / "diffusion_pusht.yaml"
PUSHT_RSKILL = "rskills/diffusion-pusht"

runner = CliRunner()


def test_libero_rejects_explicit_robot_flag() -> None:
    """``openral sim run --robot ur5e --config <libero.yaml>`` fails the guard."""
    if not LIBERO_CFG.exists():
        return
    result = runner.invoke(
        app,
        [
            "sim",
            "run",
            "--config",
            str(LIBERO_CFG),
            "--rskill",
            LIBERO_RSKILL,
            "--robot",
            "ur5e",
            "--no-view",
        ],
    )
    assert result.exit_code != 0
    out = result.output + (result.stderr or "")
    assert "libero_spatial" in out
    assert "franka_panda" in out
    assert "hard-fixes" in out


def test_metaworld_rejects_explicit_robot_flag() -> None:
    """``openral sim run --robot franka_panda --config <metaworld.yaml>`` fails."""
    if not METAWORLD_CFG.exists():
        return
    result = runner.invoke(
        app,
        [
            "sim",
            "run",
            "--config",
            str(METAWORLD_CFG),
            "--rskill",
            METAWORLD_RSKILL,
            "--robot",
            "franka_panda",
            "--no-view",
        ],
    )
    assert result.exit_code != 0
    out = result.output + (result.stderr or "")
    assert "sawyer" in out
    assert "hard-fixes" in out


def test_pusht_rejects_explicit_robot_flag() -> None:
    """``openral sim run --robot ur5e --config <pusht.yaml>`` fails the guard."""
    if not PUSHT_CFG.exists():
        return
    result = runner.invoke(
        app,
        [
            "sim",
            "run",
            "--config",
            str(PUSHT_CFG),
            "--rskill",
            PUSHT_RSKILL,
            "--robot",
            "ur5e",
            "--no-view",
        ],
    )
    assert result.exit_code != 0
    out = result.output + (result.stderr or "")
    assert "pusht_2d" in out
    assert "hard-fixes" in out
