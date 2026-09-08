"""Continuous-leg observability for the ROS-Image object detector (issue #12).

Regression: the continuous leg logged ``detect()`` exceptions at DEBUG and
dropped empty results silently, making a crashing detector (e.g. CUDA OOM
under VLA co-residency on an 8 GB card) indistinguishable on
``/openral/perception/objects`` from a quiet scene — both leave the topic
empty, the contract world-state eviction relies on.
:func:`classify_continuous_tick` maps each tick's outcome to a log level.

``openral_perception_ros`` is ament_cmake; skips cleanly if the workspace
overlay isn't sourced (ros2-test CI sources ``install/setup.bash``).
"""

from __future__ import annotations

import pytest

pytest.importorskip("openral_perception_ros")

from openral_perception_ros.ros_image_detector_node import (
    classify_continuous_tick,
    normalize_log_level,
)


def test_detect_exception_is_surfaced_at_warning() -> None:
    """A per-frame detect failure must be VISIBLE, not swallowed (CLAUDE.md §1.4).

    A crashing/OOM detector that logs at DEBUG looks identical on the perception
    topic to a quiet scene; raising it to WARNING is what makes the failure
    diagnosable in the field.
    """
    level, message = classify_continuous_tick(
        error=RuntimeError("CUDA out of memory"), detection_count=None
    )
    assert level == "warning"
    assert "CUDA out of memory" in message


def test_empty_result_logs_liveness_at_info() -> None:
    """0 detections proves the leg is alive and merely sees nothing (vs dead).

    Both ``None`` (detector returned no metadata) and ``0`` (metadata with an
    empty detection list) are the same "quiet scene" liveness signal.
    """
    for count in (0, None):
        level, message = classify_continuous_tick(error=None, detection_count=count)
        assert level == "info", f"count={count!r}"
        assert "0 detection" in message


def test_non_empty_result_is_quiet_debug() -> None:
    """The normal path stays quiet — the published metadata is itself the signal."""
    level, message = classify_continuous_tick(error=None, detection_count=3)
    assert level == "debug"
    assert "3" in message


def test_error_takes_precedence_over_count() -> None:
    """An exception is reported even if a (stale) count is also supplied."""
    level, _ = classify_continuous_tick(error=ValueError("boom"), detection_count=5)
    assert level == "warning"


# ── normalize_log_level: OPENRAL_DETECTOR_LOG_LEVEL → rclpy severity name ──────


def test_normalize_log_level_canonical_names() -> None:
    """The five severities normalise to their rclpy LoggingSeverity names."""
    assert normalize_log_level("debug") == "DEBUG"
    assert normalize_log_level("info") == "INFO"
    assert normalize_log_level("error") == "ERROR"
    assert normalize_log_level("fatal") == "FATAL"


def test_normalize_log_level_is_case_insensitive_and_trims() -> None:
    """Operators type any case / stray whitespace — accept it."""
    assert normalize_log_level("DEBUG") == "DEBUG"
    assert normalize_log_level("  Debug ") == "DEBUG"


def test_normalize_log_level_warning_aliases_warn() -> None:
    """Both 'warn' and 'warning' map to rclpy's WARN."""
    assert normalize_log_level("warn") == "WARN"
    assert normalize_log_level("warning") == "WARN"


def test_normalize_log_level_unset_or_invalid_is_none() -> None:
    """Empty / whitespace / unknown → None, so the caller leaves the level alone."""
    for value in ("", "   ", "verbose", "trace"):
        assert normalize_log_level(value) is None, value
