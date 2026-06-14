# python/hal/tests/test_joint_name_resolution.py
"""Unified joint-name resolution (ADR-0034 §3.6): exact wins; robosuite prefix stripped."""

from __future__ import annotations

from openral_hal.sim_attached import normalized_joint_index


def test_strips_robosuite_group_prefix() -> None:
    idx = normalized_joint_index(["robot0_joint1", "robot0_joint2", "gripper0_finger_joint1"])
    assert idx["robot0_joint1"] == 0  # exact
    assert idx["joint1"] == 0  # stripped fallback
    assert idx["finger_joint1"] == 2


def test_native_names_unchanged() -> None:
    idx = normalized_joint_index(["joint1", "finger_joint1", "left_joint1", "openarm_left_joint1"])
    assert idx["joint1"] == 0
    assert idx["left_joint1"] == 2  # 'left_' has no leading digit -> not stripped
    assert idx["openarm_left_joint1"] == 3


def test_exact_wins_over_stripped() -> None:
    idx = normalized_joint_index(["joint1", "robot0_joint1"])
    assert idx["joint1"] == 0  # the native exact entry, not the stripped robot0_


def test_ambiguous_stripped_not_added() -> None:
    idx = normalized_joint_index(["robot0_joint1", "robot1_joint1"])
    assert "joint1" not in idx  # both strip to joint1 -> ambiguous, dropped
    assert idx["robot0_joint1"] == 0 and idx["robot1_joint1"] == 1
