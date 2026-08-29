"""The carry survey's task registry read and its qualifying threshold.

The measuring half needs a built RoboCasa env and lives in
`tests/sim/test_panda_mobile_hal_robocasa_carry.py`; this tier covers what is
decidable without one.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))

from robocasa_carry_survey import (
    DISTRACTOR_PREFIXES,
    REQUIRED_BASE_TRANSLATION_M,
    CarryMeasurement,
    SurveyError,
    read_target50,
    robocasa_root,
)


def _measurement(furthest: float) -> CarryMeasurement:
    return CarryMeasurement(
        task="DeliverStraw",
        seed=3,
        layout=51,
        style=9,
        furthest_object_m=furthest,
        furthest_object="glass_cup",
        nearest_object_m=0.50,
        nearest_object="straw",
        lang="Take a straw from the drawer in front and place it inside the glass cup.",
    )


def test_threshold_is_the_nav2_readme_criterion() -> None:
    assert REQUIRED_BASE_TRANSLATION_M == 1.0


def test_qualifying_is_strictly_above_the_threshold() -> None:
    assert not _measurement(0.99).requires_base_translation
    assert not _measurement(1.0).requires_base_translation
    assert _measurement(1.01).requires_base_translation
    # The real DeliverStraw seed-3 measurement.
    assert _measurement(3.795).requires_base_translation


def test_distractors_are_named_so_they_can_be_excluded() -> None:
    """`StoreLeftoversInBowl`'s far objects are `distr1`/`distr2` at ~5 m.

    Counting them would have made a task whose real objects all sit within 1 m
    of the base look like a cross-kitchen carry.
    """
    assert "distr1".startswith(DISTRACTOR_PREFIXES)
    assert "distr2".startswith(DISTRACTOR_PREFIXES)
    # The second spelling, used in six places. Missing it made
    # `ArrangeBreadBasket` and `PanTransfer` report `dstr_dining*` as their
    # furthest object and read as cross-kitchen carries when they are not.
    assert "dstr_dining".startswith(DISTRACTOR_PREFIXES)
    assert "dstr_dining2".startswith(DISTRACTOR_PREFIXES)
    assert not "glass_cup".startswith(DISTRACTOR_PREFIXES)
    assert not "straw".startswith(DISTRACTOR_PREFIXES)


def test_target50_is_the_set_xr1_reports_against() -> None:
    """XR-1's model card pins `task set: target50`, 50 tasks."""
    try:
        root = robocasa_root(None)
    except SurveyError as exc:
        pytest.skip(f"RoboCasa source tree not available: {exc}")
    target50 = read_target50(root)
    assert len(target50) == 50
    # The two tasks the survey found that carry across the kitchen are in it —
    # that is what makes them usable as #108's acceptance scene.
    assert "DeliverStraw" in target50
    assert "GetToastedBread" in target50
