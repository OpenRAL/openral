"""Continuous-leg observability for the ROS-Image object detector (issue #12).

The continuous detect+publish leg in ``ros_image_detector_node`` used to swallow
its per-frame outcome silently: a ``detect()`` exception was logged at DEBUG
(invisible at the default level) and an empty result was dropped with no trace.
On ``/openral/perception/objects`` that makes a *crashing* detector (e.g. a CUDA
OOM under VLA co-residency on an 8 GB card) indistinguishable from one watching a
quiet scene — both leave the topic empty, because the real detector publishes
nothing when it sees nothing (the ADR-0035 contract the world-state eviction
relies on). :func:`classify_continuous_tick` is the pure decision that maps each
tick's outcome to the log level that surfaces it, so the leg is observable
without changing what lands on the bus. Validated here with no LifecycleNode or
executor — the node applies its own throttling and does the publish.

``openral_perception_ros`` is a colcon-built ROS package (ament_cmake); like the
other ROS-package unit tests (``test_image_convert``), skip cleanly when the
workspace overlay isn't sourced. The ros2-test CI job sources
``install/setup.bash`` and runs this for real; ``classify_continuous_tick`` is
pure Python so the guard is only about the package being importable.
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
