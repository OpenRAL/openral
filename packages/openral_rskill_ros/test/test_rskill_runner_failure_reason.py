"""Unit tests for the rskill_runner failure-reason + goal-finalize helpers.

Covers two deploy-sim robustness fixes:

1. ``_label_runtime_failure`` — torch inference errors (CUDA OOM, dtype /
   quantization mismatch) are raw ``RuntimeError``s, NOT ``ROSError`` subclasses,
   so they used to escape ``_execute_cb`` uncaught and rclpy aborted the goal with
   an EMPTY Result (the reasoner saw ``status=6 reason=''``). The label maps them
   to a typed, reasoner-legible reason.
2. ``_finalize_goal`` — a concurrent cancel / re-dispatch can move the goal out of
   EXECUTING mid-tick, so ``abort()`` / ``succeed()`` / ``canceled()`` raise
   ``RCLError: invalid transition``. The helper must swallow that (best-effort) so
   the populated Result is still returned instead of an empty-reason abort.

Both helpers live on ``RskillRunnerNode`` (import requires rclpy), so the module
is skipped without a sourced ROS 2 install.
"""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("ROS_DISTRO"),
    reason="ROS_DISTRO not set — RskillRunnerNode import requires a sourced ROS 2 install.",
)

try:
    from openral_rskill_ros.rskill_runner_node import RskillRunnerNode
except Exception:  # noqa: BLE001  # reason: availability gate (rclpy/openral_msgs absent)
    RskillRunnerNode = None  # type: ignore[assignment, misc]


# ── 1. _label_runtime_failure truth table ──────────────────────────────────


def test_label_runtime_failure_cuda_oom() -> None:
    """A CUDA OOM (by message or by exception name) → ROSGPUMemoryError."""
    assert RskillRunnerNode is not None
    msg = "CUDA out of memory. Tried to allocate 20.00 MiB."
    assert RskillRunnerNode._label_runtime_failure(RuntimeError(msg)).startswith(
        "ROSGPUMemoryError:"
    )

    class OutOfMemoryError(RuntimeError):
        pass

    assert RskillRunnerNode._label_runtime_failure(OutOfMemoryError("boom")).startswith(
        "ROSGPUMemoryError:"
    )


def test_label_runtime_failure_dtype() -> None:
    """The smolvla dtype mismatch → ROSQuantizationError."""
    assert RskillRunnerNode is not None
    msg = "mat1 and mat2 must have the same dtype, but got Float and BFloat16"
    assert RskillRunnerNode._label_runtime_failure(RuntimeError(msg)).startswith(
        "ROSQuantizationError:"
    )


def test_label_runtime_failure_generic() -> None:
    """An unrecognised error keeps its concrete type name (never empty)."""
    assert RskillRunnerNode is not None
    out = RskillRunnerNode._label_runtime_failure(ValueError("weird"))
    assert out == "ValueError: weird"


# ── 2. _finalize_goal tolerates a racing goal state ─────────────────────────


class _NoOpLogger:
    def debug(self, *a: object, **k: object) -> None: ...
    def warning(self, *a: object, **k: object) -> None: ...
    def error(self, *a: object, **k: object) -> None: ...


class _FakeSelf:
    def get_logger(self) -> _NoOpLogger:
        return _NoOpLogger()


class _RacingGoalHandle:
    """Goal handle whose transitions raise, as rclpy does on an invalid transition."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def abort(self) -> None:
        self.calls.append("abort")
        raise RuntimeError("Failed to update goal state: invalid transition from state EXECUTING")

    def succeed(self) -> None:
        self.calls.append("succeed")
        raise RuntimeError("Failed to update goal state: invalid transition from state EXECUTING")


class _HealthyGoalHandle:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def abort(self) -> None:
        self.calls.append("abort")


def test_finalize_goal_swallows_invalid_transition() -> None:
    """A racing (already-terminal) goal must not raise out of the callback."""
    assert RskillRunnerNode is not None
    gh = _RacingGoalHandle()
    # Called unbound with a minimal fake self (only needs get_logger()).
    RskillRunnerNode._finalize_goal(_FakeSelf(), gh, "abort")  # type: ignore[arg-type]
    assert gh.calls == ["abort"]  # attempted once, exception swallowed


def test_finalize_goal_applies_transition_when_healthy() -> None:
    """On a healthy goal the transition is applied normally."""
    assert RskillRunnerNode is not None
    gh = _HealthyGoalHandle()
    RskillRunnerNode._finalize_goal(_FakeSelf(), gh, "abort")  # type: ignore[arg-type]
    assert gh.calls == ["abort"]
