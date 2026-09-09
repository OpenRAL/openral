# SPDX-License-Identifier: Apache-2.0
"""Nav2's bond teardown must void a run, and must spare one that merely ended.

Issue #256. ``lifecycle_manager_navigation`` tears down the whole navigation
stack when a managed server misses its bond heartbeat. The teardown prints no
traceback and exits non-zero nowhere, so the graph stays up and inert until the
deadline expires and the harness scores the corpse as ``deadline-no-grasp`` —
the policy failing to grasp. 31 of the 89 valid runs in the 2026-09-06 ceiling
battery died that way and every one was counted against the policy.

The same "Have not received a heartbeat" line also appears late in runs that
did their work, so the discriminator is *when*. These fixtures are real deploy
logs from that battery, trimmed but keeping the banner and the first timestamp
so the elapsed anchor is the run's true start.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))

import validation_matrix as vm  # type: ignore[import-not-found]  # reason: sibling script, no stub

FIXTURES = Path(__file__).parent / "fixtures" / "validation_matrix" / "ceiling-2026-09-06"


def _lines(name: str) -> list[str]:
    return (FIXTURES / f"{name}.log").read_text(encoding="utf-8").splitlines()


def test_early_bond_loss_voids_the_run() -> None:
    """A stack torn down 55 s in is a harness failure, not a policy outcome."""
    reason = vm._nav2_bond_teardown(_lines("sink_cup_off_r07_torn_down"))
    assert "Nav2 tore down its stack" in reason
    assert "controller_server" in reason
    # And it reaches the outcome the ledger actually counts.
    assert vm.detect_launch_failure(FIXTURES, "missing", _lines("sink_cup_off_r07_torn_down"))


@pytest.mark.parametrize(
    "name",
    ["fridge_off_r10_completed", "utensil_off_r07_completed_earliest_teardown"],
)
def test_late_bond_loss_leaves_a_real_run_alone(name: str) -> None:
    """A run that did its work keeps its result, even losing a bond at 199.6 s.

    ``utensil_off_r07`` is the tightest real case in the battery: the earliest
    bond loss in any run that completed its task. If the threshold ever drifts
    past it, real completions start disappearing into ``harness-error``.
    """
    assert vm._nav2_bond_teardown(_lines(name)) == ""


def test_a_mid_run_excerpt_is_never_anchored() -> None:
    """An excerpt starts mid-run, so its first stamp is not the run's start.

    Anchoring on it would read every late bond loss as an early one. Declining
    is the safe direction: a missed teardown leaves today's behaviour, while a
    false one silently drops a real result out of the denominator.
    """
    real = _lines("fridge_off_r10_completed")
    excerpt = ["argv: ros2 launch openral_rskill_ros deploy_e2e.launch.py", *real[3:]]
    assert vm._nav2_bond_teardown(excerpt) == ""
