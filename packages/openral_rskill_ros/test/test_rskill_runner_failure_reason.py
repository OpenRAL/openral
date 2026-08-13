"""Unit tests for the rskill_runner failure-reason/-kind + goal-finalize helpers.

Also covers the ``ExecuteRskill.Result.failure_kind`` contract: the colcon-generated
uint8 constants themselves, and ``_failure_kind_for_exception`` — the map from the
CLAUDE.md §5 exception hierarchy onto them that every failure branch of
``_execute_locked`` calls.

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
    from openral_rskill_ros.rskill_runner_node import (
        RskillRunnerNode,
        _failure_kind_for_exception,
    )
except Exception:  # reason: availability gate (rclpy/openral_msgs absent)
    RskillRunnerNode = None  # type: ignore[assignment, misc]
    _failure_kind_for_exception = None  # type: ignore[assignment]


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


# ── 3. ExecuteRskill.Result.failure_kind — the generated IDL constants ──────


def test_generated_failure_kind_constants() -> None:
    """The colcon-generated Result exposes the documented uint8 constant set.

    Reads the real generated Python module (``colcon build --packages-select
    openral_msgs``), so a value edited in ``packages/msgs/action/ExecuteRskill.action``
    without updating its consumers fails here rather than on the wire.
    """
    from openral_msgs.action import ExecuteRskill

    result_cls = ExecuteRskill.Result
    expected = {
        "FAILURE_NONE": 0,
        "FAILURE_CONFIG_ERROR": 1,
        "FAILURE_CAPABILITY_MISMATCH": 2,
        "FAILURE_RUNTIME_ERROR": 3,
        "FAILURE_SAFETY_ESTOP": 4,
        "FAILURE_PERCEPTION_STALE": 5,
        "FAILURE_PLANNING_ERROR": 6,
        "FAILURE_DEADLINE_MISSED": 7,
        "FAILURE_CANCELLED": 8,
        "FAILURE_UNKNOWN": 255,
    }
    actual = {name: int(getattr(result_cls, name)) for name in expected}
    assert actual == expected
    # No stray kinds beyond the documented set.
    assert {n for n in dir(result_cls) if n.startswith("FAILURE_")} == set(expected)
    # Additive field: a default-constructed Result decodes as FAILURE_NONE, which
    # is exactly what an older producer's Result looks like on the wire.
    fresh = result_cls()
    assert fresh.failure_kind == result_cls.FAILURE_NONE
    assert fresh.failure_reason == "" and fresh.success is False


# ── 4. _failure_kind_for_exception — the §5 hierarchy → uint8 map ───────────


def test_failure_kind_for_exception_truth_table() -> None:
    """Every §5 exception the dispatch path can raise gets its own kind."""
    from openral_core.exceptions import (
        ROSCapabilityMismatch,
        ROSConfigError,
        ROSDeadlineMissed,
        ROSError,
        ROSEStopRequested,
        ROSGPUMemoryError,
        ROSObjectNotInMemory,
        ROSPerceptionStale,
        ROSPlanningError,
        ROSQuantizationError,
        ROSReasonerInvalidPlan,
        ROSRuntimeError,
    )
    from openral_msgs.action import ExecuteRskill

    assert _failure_kind_for_exception is not None
    r = ExecuteRskill.Result
    cases: list[tuple[BaseException, int]] = [
        (ROSEStopRequested("estop"), r.FAILURE_SAFETY_ESTOP),
        (ROSCapabilityMismatch("no lidar"), r.FAILURE_CAPABILITY_MISMATCH),
        (ROSConfigError("bad manifest"), r.FAILURE_CONFIG_ERROR),
        (ROSPerceptionStale("stale depth"), r.FAILURE_PERCEPTION_STALE),
        # Subclasses resolve to their parent's kind — most-specific-first ordering.
        (ROSObjectNotInMemory("no such node"), r.FAILURE_PERCEPTION_STALE),
        (ROSPlanningError("no plan"), r.FAILURE_PLANNING_ERROR),
        (ROSReasonerInvalidPlan("undecodable"), r.FAILURE_PLANNING_ERROR),
        (ROSDeadlineMissed("rtt blew the budget"), r.FAILURE_DEADLINE_MISSED),
        (ROSRuntimeError("inference failed"), r.FAILURE_RUNTIME_ERROR),
        (ROSGPUMemoryError("cuda oom"), r.FAILURE_RUNTIME_ERROR),
        (ROSQuantizationError("dtype mismatch"), r.FAILURE_RUNTIME_ERROR),
        # A typed error with no dedicated kind is still typed → RUNTIME_ERROR.
        (ROSError("bare"), r.FAILURE_RUNTIME_ERROR),
        # Anything outside the OpenRAL surface is UNKNOWN, never a typed kind.
        (ValueError("weird"), r.FAILURE_UNKNOWN),
        (RuntimeError("torch blew up"), r.FAILURE_UNKNOWN),
    ]
    for exc, expected in cases:
        assert _failure_kind_for_exception(exc) == expected, type(exc).__name__


def test_classify_runtime_failure_agrees_with_its_label() -> None:
    """The raw-exception branch derives prose and uint8 from ONE probe.

    A CUDA OOM / dtype mismatch is labelled as a ``ROSRuntimeError`` subclass, so
    it must carry ``FAILURE_RUNTIME_ERROR``; anything unrecognised keeps its
    concrete type name and carries ``FAILURE_UNKNOWN``.
    """
    from openral_msgs.action import ExecuteRskill

    assert RskillRunnerNode is not None
    r = ExecuteRskill.Result
    for exc, prefix, kind in (
        (
            RuntimeError("CUDA out of memory. Tried to allocate 20.00 MiB."),
            "ROSGPUMemoryError:",
            r.FAILURE_RUNTIME_ERROR,
        ),
        (
            RuntimeError("mat1 and mat2 must have the same dtype, but got Float and BFloat16"),
            "ROSQuantizationError:",
            r.FAILURE_RUNTIME_ERROR,
        ),
        (ValueError("weird"), "ValueError:", r.FAILURE_UNKNOWN),
    ):
        reason, failure_kind = RskillRunnerNode._classify_runtime_failure(exc)
        assert reason.startswith(prefix)
        assert failure_kind == kind
        # The thin label projection can never disagree with the pair.
        assert RskillRunnerNode._label_runtime_failure(exc) == reason
