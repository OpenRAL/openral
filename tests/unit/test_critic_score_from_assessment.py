"""Unit tests for the reward-assessment → CriticScore mapping (ADR-0064).

``critic_score_from_assessment`` is the pure seam that lets the Robometer
``reward_monitor`` publish a generic ``openral_msgs/CriticScore`` for the Tier-C
critic producer. Pure logic, no ROS/model — runs in the plain unit tier.
"""

from __future__ import annotations

import pytest
from openral_runner.backends.reward.robometer_reward import critic_score_from_assessment


def test_progress_maps_to_score() -> None:
    score, threshold = critic_score_from_assessment({"progress_now": 0.42}, threshold=0.8)
    assert score == pytest.approx(0.42)
    assert threshold == pytest.approx(0.8)


def test_score_clamped_to_unit_interval() -> None:
    assert critic_score_from_assessment({"progress_now": 1.5}, threshold=0.8)[0] == pytest.approx(
        1.0
    )
    assert critic_score_from_assessment({"progress_now": -0.2}, threshold=0.8)[0] == pytest.approx(
        0.0
    )


def test_missing_or_nonnumeric_progress_defaults_to_zero() -> None:
    assert critic_score_from_assessment({}, threshold=0.8)[0] == 0.0
    assert critic_score_from_assessment({"progress_now": None}, threshold=0.8)[0] == 0.0
    assert critic_score_from_assessment({"progress_now": "x"}, threshold=0.8)[0] == 0.0


def test_bool_progress_is_not_treated_as_numeric() -> None:
    # ``True`` would ``float()`` to 1.0; reject bools as a defensive contract so a
    # stray flag never masquerades as full progress.
    assert critic_score_from_assessment({"progress_now": True}, threshold=0.8)[0] == 0.0


def test_threshold_is_passed_through_as_float() -> None:
    _score, threshold = critic_score_from_assessment({"progress_now": 0.5}, threshold=1)
    assert isinstance(threshold, float)
    assert threshold == pytest.approx(1.0)
