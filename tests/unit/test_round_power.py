# SPDX-License-Identifier: Apache-2.0
"""``tools/round_power.py`` computes Fisher and power correctly.

The tool's whole purpose is to stop a battery being run at a size that cannot
answer its own question (#217), so a wrong number here is worse than no tool:
it would licence exactly the underpowered round it exists to prevent. These
pin the arithmetic against values that can be checked independently.

The reference values come from ``scipy.stats.fisher_exact``, which the
implementation was validated against on 300 random 2x2 tables (all matched to
1e-9). scipy is deliberately NOT a workspace dependency — it is a heavy import
for a planning script — so the checked values are inlined here rather than
recomputed at test time.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

_TOOL = Path(__file__).resolve().parents[2] / "tools" / "round_power.py"


def _load() -> ModuleType:
    """Import the tool by path — ``tools/`` is scripts, not an installed package."""
    spec = importlib.util.spec_from_file_location("round_power", _TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


round_power = _load()


@pytest.mark.parametrize(
    ("a", "n1", "b", "n2", "expected"),
    [
        # The comparison #217 exists to settle: 2026-08-26 against the
        # 2026-09-04 post-#200 arm. The issue quotes p = 0.18.
        (5, 20, 1, 20, 0.1818),
        # Both post arms pooled, which the issue quotes as p = 0.036 while
        # noting that pooling two configurations is arguably not legitimate.
        (5, 20, 2, 40, 0.0355),
        # The tripping-party shift, 2026-08-26 against Path A on.
        (6, 15, 16, 17, 0.0017),
    ],
)
def test_fisher_matches_the_values_the_issue_quotes(
    a: int, n1: int, b: int, n2: int, expected: float
) -> None:
    assert round_power.fisher_exact_two_sided(a, n1, b, n2) == pytest.approx(expected, abs=5e-5)


def test_identical_arms_are_never_significant() -> None:
    """A table with the same rate on both sides must not reject."""
    for n in (10, 20, 40):
        assert round_power.fisher_exact_two_sided(n // 4, n, n // 4, n) == pytest.approx(1.0)


def test_the_twenty_run_battery_could_not_have_separated_its_own_effect() -> None:
    """The finding that motivates the tool, pinned so it cannot drift.

    20 runs per arm against 25% -> 5% is well under half power, so
    "p = 0.18, not separable" was never evidence of no effect.
    """
    assert round_power.power(0.25, 0.05, 20) == pytest.approx(0.299, abs=0.002)
    assert round_power.power(0.25, 0.05, 20) < 0.5


def test_power_rises_with_n_and_with_effect_size() -> None:
    smaller = round_power.power(0.25, 0.05, 20)
    larger = round_power.power(0.25, 0.05, 40)
    assert smaller < larger

    subtle = round_power.power(0.25, 0.15, 40)
    blatant = round_power.power(0.25, 0.05, 40)
    assert subtle < blatant


def test_required_returns_the_first_adequate_ladder_step() -> None:
    n, achieved = round_power.required(0.25, 0.05)
    assert n == 60
    assert achieved >= 0.80
    # And the step below it is genuinely inadequate, so 60 is not slack.
    assert round_power.power(0.25, 0.05, 40) < 0.80


def test_an_undetectable_effect_reports_no_ladder_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A one-point difference is not reachable, and the tool must say so.

    The ladder is shortened for the test: walking the real one to n = 800 on a
    hopeless effect costs O(800^2) Fisher evaluations and would blow the 30 s
    unit budget (CLAUDE.md §2) to re-derive an answer that is obvious by n=40.
    What is under test is the exhausted-ladder path, not the ladder's length.
    """
    monkeypatch.setattr(round_power, "_LADDER", (20, 40))
    n, achieved = round_power.required(0.25, 0.24)
    assert n is None
    assert achieved < 0.80
